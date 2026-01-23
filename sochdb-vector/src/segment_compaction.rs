// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Drift-Resilient Segment & Compaction Strategy (Task 9)
//!
//! Provides immutable segment management with:
//! - Quantizer error tracking and threshold-based retraining
//! - Segment lifecycle governance
//! - Atomic version transitions
//!
//! ## Philosophy
//!
//! 1. Segments are immutable once written
//! 2. Deletes accumulate in tombstone bitvec
//! 3. Compaction merges small segments, removes tombstones
//! 4. Quantizer is retrained when drift exceeds threshold
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::segment_compaction::{SegmentManager, CompactionPolicy, Segment};
//!
//! let manager = SegmentManager::new(policy);
//! manager.add_segment(segment);
//! manager.maybe_compact();
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

// ============================================================================
// Segment Metadata
// ============================================================================

/// Unique segment identifier
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SegmentId(pub u64);

impl SegmentId {
    /// Generate next ID
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Segment state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentState {
    /// Being built, not yet searchable
    Building,
    /// Active and searchable
    Active,
    /// Being compacted into new segment
    Compacting,
    /// Marked for deletion after compaction
    Tombstoned,
    /// Deleted, awaiting GC
    Deleted,
}

/// Quantizer metadata
#[derive(Debug, Clone)]
pub struct QuantizerMeta {
    /// Quantizer version
    pub version: u32,
    /// Number of training samples used
    pub n_training_samples: usize,
    /// Training error (MSE)
    pub training_error: f32,
    /// Current estimated error
    pub current_error: f32,
    /// Created timestamp
    pub created_at: SystemTime,
}

impl Default for QuantizerMeta {
    fn default() -> Self {
        Self {
            version: 1,
            n_training_samples: 0,
            training_error: 0.0,
            current_error: 0.0,
            created_at: SystemTime::now(),
        }
    }
}

/// Segment statistics for compaction decisions
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Total vectors in segment
    pub n_vectors: usize,
    /// Deleted vectors (tombstones)
    pub n_deleted: usize,
    /// Segment size in bytes
    pub size_bytes: u64,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Number of accesses
    pub access_count: u64,
    /// Quantizer metadata
    pub quantizer_meta: QuantizerMeta,
    /// Quantization error samples (for drift detection)
    pub error_samples: Vec<f32>,
}

impl SegmentStats {
    /// Create new stats
    pub fn new(n_vectors: usize, size_bytes: u64) -> Self {
        Self {
            n_vectors,
            n_deleted: 0,
            size_bytes,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            quantizer_meta: QuantizerMeta::default(),
            error_samples: Vec::new(),
        }
    }
    
    /// Get deletion ratio
    pub fn deletion_ratio(&self) -> f32 {
        if self.n_vectors == 0 {
            0.0
        } else {
            self.n_deleted as f32 / self.n_vectors as f32
        }
    }
    
    /// Get live vector count
    pub fn live_vectors(&self) -> usize {
        self.n_vectors.saturating_sub(self.n_deleted)
    }
    
    /// Record quantizer error sample
    pub fn record_error(&mut self, error: f32) {
        self.error_samples.push(error);
        // Keep only recent samples
        if self.error_samples.len() > 1000 {
            self.error_samples.remove(0);
        }
    }
    
    /// Get current estimated quantizer error
    pub fn estimated_error(&self) -> f32 {
        if self.error_samples.is_empty() {
            self.quantizer_meta.current_error
        } else {
            let sum: f32 = self.error_samples.iter().sum();
            sum / self.error_samples.len() as f32
        }
    }
    
    /// Check if quantizer needs retraining
    pub fn needs_retraining(&self, threshold: f32) -> bool {
        let current = self.estimated_error();
        let original = self.quantizer_meta.training_error;
        
        if original == 0.0 {
            false
        } else {
            (current - original) / original > threshold
        }
    }
}

// ============================================================================
// Segment
// ============================================================================

/// Immutable segment
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment ID
    pub id: SegmentId,
    /// Segment state
    pub state: SegmentState,
    /// Statistics
    pub stats: SegmentStats,
    /// Data file path
    pub data_path: String,
    /// Index file path
    pub index_path: String,
    /// Tombstone bitvec path
    pub tombstone_path: String,
    /// Segment generation (increments with each compaction)
    pub generation: u32,
}

impl Segment {
    /// Create new segment
    pub fn new(id: SegmentId, n_vectors: usize, size_bytes: u64, data_path: String) -> Self {
        Self {
            id,
            state: SegmentState::Building,
            stats: SegmentStats::new(n_vectors, size_bytes),
            data_path: data_path.clone(),
            index_path: format!("{}.idx", data_path),
            tombstone_path: format!("{}.tomb", data_path),
            generation: 1,
        }
    }
    
    /// Mark segment as active
    pub fn activate(&mut self) {
        self.state = SegmentState::Active;
    }
    
    /// Mark vector as deleted
    pub fn mark_deleted(&mut self, count: usize) {
        self.stats.n_deleted += count;
    }
    
    /// Record access
    pub fn record_access(&mut self) {
        self.stats.access_count += 1;
        self.stats.last_accessed = SystemTime::now();
    }
}

// ============================================================================
// Compaction Policy
// ============================================================================

/// Compaction trigger conditions
#[derive(Debug, Clone)]
pub struct CompactionPolicy {
    /// Minimum deletion ratio to trigger compaction
    pub deletion_ratio_threshold: f32,
    
    /// Maximum segment size before split
    pub max_segment_size: u64,
    
    /// Minimum segment size (below this, merge with others)
    pub min_segment_size: u64,
    
    /// Target segment size for new segments
    pub target_segment_size: u64,
    
    /// Maximum segments before forced compaction
    pub max_segments: usize,
    
    /// Quantizer error drift threshold for retraining
    pub quantizer_drift_threshold: f32,
    
    /// Minimum time between compactions
    pub compaction_cooldown: Duration,
    
    /// Maximum concurrent compaction threads
    pub max_compaction_threads: usize,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            deletion_ratio_threshold: 0.3,
            max_segment_size: 1024 * 1024 * 1024, // 1 GB
            min_segment_size: 64 * 1024 * 1024,   // 64 MB
            target_segment_size: 256 * 1024 * 1024, // 256 MB
            max_segments: 100,
            quantizer_drift_threshold: 0.2, // 20% error increase triggers retraining
            compaction_cooldown: Duration::from_secs(60),
            max_compaction_threads: 2,
        }
    }
}

impl CompactionPolicy {
    /// Create policy optimized for SSD
    pub fn ssd_optimized() -> Self {
        Self {
            deletion_ratio_threshold: 0.25, // More aggressive reclamation
            target_segment_size: 512 * 1024 * 1024, // Larger segments
            ..Default::default()
        }
    }
    
    /// Create policy optimized for RAM
    pub fn ram_optimized() -> Self {
        Self {
            deletion_ratio_threshold: 0.4, // Less aggressive
            target_segment_size: 64 * 1024 * 1024, // Smaller segments
            max_segments: 50, // Fewer segments for faster search
            ..Default::default()
        }
    }
}

// ============================================================================
// Compaction Decision
// ============================================================================

/// Compaction decision for a set of segments
#[derive(Debug)]
pub enum CompactionDecision {
    /// No compaction needed
    None,
    /// Merge segments into one
    Merge(Vec<SegmentId>),
    /// Split segment
    Split(SegmentId),
    /// Retrain quantizer for segments
    Retrain(Vec<SegmentId>),
    /// Full recompaction
    FullRecompact(Vec<SegmentId>),
}

/// Compaction job
#[derive(Debug)]
pub struct CompactionJob {
    /// Job ID
    pub id: u64,
    /// Decision
    pub decision: CompactionDecision,
    /// Source segments
    pub source_segments: Vec<SegmentId>,
    /// Created time
    pub created_at: Instant,
    /// Priority (lower = higher priority)
    pub priority: u32,
}

// ============================================================================
// Compaction Planner
// ============================================================================

/// Plans compaction jobs
pub struct CompactionPlanner {
    policy: CompactionPolicy,
}

impl CompactionPlanner {
    /// Create new planner
    pub fn new(policy: CompactionPolicy) -> Self {
        Self { policy }
    }
    
    /// Analyze segments and decide on compaction
    pub fn plan(&self, segments: &[&Segment]) -> Vec<CompactionDecision> {
        let mut decisions = Vec::new();
        
        // Check for high deletion ratio segments
        let high_deletion: Vec<_> = segments.iter()
            .filter(|s| s.stats.deletion_ratio() > self.policy.deletion_ratio_threshold)
            .map(|s| s.id)
            .collect();
        
        if !high_deletion.is_empty() {
            decisions.push(CompactionDecision::Merge(high_deletion));
        }
        
        // Check for small segments to merge
        let small_segments: Vec<_> = segments.iter()
            .filter(|s| s.stats.size_bytes < self.policy.min_segment_size)
            .collect();
        
        if small_segments.len() >= 2 {
            // Group small segments for merging
            let mut current_group: Vec<SegmentId> = Vec::new();
            let mut current_size = 0u64;
            
            for seg in small_segments {
                if current_size + seg.stats.size_bytes <= self.policy.target_segment_size {
                    current_group.push(seg.id);
                    current_size += seg.stats.size_bytes;
                } else {
                    if current_group.len() >= 2 {
                        decisions.push(CompactionDecision::Merge(current_group.clone()));
                    }
                    current_group.clear();
                    current_group.push(seg.id);
                    current_size = seg.stats.size_bytes;
                }
            }
            
            if current_group.len() >= 2 {
                decisions.push(CompactionDecision::Merge(current_group));
            }
        }
        
        // Check for oversized segments
        for seg in segments {
            if seg.stats.size_bytes > self.policy.max_segment_size {
                decisions.push(CompactionDecision::Split(seg.id));
            }
        }
        
        // Check for quantizer drift
        let drifted: Vec<_> = segments.iter()
            .filter(|s| s.stats.needs_retraining(self.policy.quantizer_drift_threshold))
            .map(|s| s.id)
            .collect();
        
        if !drifted.is_empty() {
            decisions.push(CompactionDecision::Retrain(drifted));
        }
        
        // Check if too many segments
        if segments.len() > self.policy.max_segments {
            // Aggressive merge of oldest/smallest segments
            let mut sorted: Vec<_> = segments.iter().collect();
            sorted.sort_by_key(|s| s.stats.live_vectors());
            
            let to_merge: Vec<_> = sorted.iter()
                .take(segments.len() / 2)
                .map(|s| s.id)
                .collect();
            
            if to_merge.len() >= 2 {
                decisions.push(CompactionDecision::FullRecompact(to_merge));
            }
        }
        
        decisions
    }
    
    /// Get policy
    pub fn policy(&self) -> &CompactionPolicy {
        &self.policy
    }
}

// ============================================================================
// Version Manager
// ============================================================================

/// Manages segment versions for atomic transitions
pub struct VersionManager {
    /// Current version
    current_version: AtomicU64,
    /// Version to segments mapping
    versions: parking_lot::RwLock<HashMap<u64, Vec<SegmentId>>>,
}

impl VersionManager {
    /// Create new version manager
    pub fn new() -> Self {
        Self {
            current_version: AtomicU64::new(1),
            versions: parking_lot::RwLock::new(HashMap::new()),
        }
    }
    
    /// Get current version
    pub fn current(&self) -> u64 {
        self.current_version.load(Ordering::SeqCst)
    }
    
    /// Create new version with segments
    pub fn create_version(&self, segments: Vec<SegmentId>) -> u64 {
        let version = self.current_version.fetch_add(1, Ordering::SeqCst) + 1;
        self.versions.write().insert(version, segments);
        version
    }
    
    /// Switch to new version atomically
    pub fn switch_to(&self, version: u64) -> bool {
        let versions = self.versions.read();
        if versions.contains_key(&version) {
            self.current_version.store(version, Ordering::SeqCst);
            true
        } else {
            false
        }
    }
    
    /// Get segments for version
    pub fn get_segments(&self, version: u64) -> Option<Vec<SegmentId>> {
        self.versions.read().get(&version).cloned()
    }
    
    /// Rollback to previous version
    pub fn rollback(&self) -> bool {
        let current = self.current_version.load(Ordering::SeqCst);
        if current > 1 {
            self.current_version.store(current - 1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }
    
    /// Clean old versions
    pub fn clean_old_versions(&self, keep_n: usize) {
        let current = self.current();
        let mut versions = self.versions.write();
        
        let to_remove: Vec<_> = versions.keys()
            .filter(|&&v| v + keep_n as u64 <= current)
            .cloned()
            .collect();
        
        for v in to_remove {
            versions.remove(&v);
        }
    }
}

impl Default for VersionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Segment Manager
// ============================================================================

/// Manages segment lifecycle
pub struct SegmentManager {
    /// All segments
    segments: parking_lot::RwLock<HashMap<SegmentId, Segment>>,
    /// Compaction planner
    planner: CompactionPlanner,
    /// Version manager
    versions: VersionManager,
    /// Last compaction time
    last_compaction: parking_lot::Mutex<Option<Instant>>,
    /// Job counter
    job_counter: AtomicU64,
}

impl SegmentManager {
    /// Create new segment manager
    pub fn new(policy: CompactionPolicy) -> Self {
        Self {
            segments: parking_lot::RwLock::new(HashMap::new()),
            planner: CompactionPlanner::new(policy),
            versions: VersionManager::new(),
            last_compaction: parking_lot::Mutex::new(None),
            job_counter: AtomicU64::new(0),
        }
    }
    
    /// Add segment
    pub fn add_segment(&self, segment: Segment) {
        let id = segment.id;
        self.segments.write().insert(id, segment);
        
        // Update version
        let current_segments: Vec<_> = self.segments.read()
            .iter()
            .filter(|(_, s)| s.state == SegmentState::Active)
            .map(|(id, _)| *id)
            .collect();
        
        self.versions.create_version(current_segments);
    }
    
    /// Get segment
    pub fn get_segment(&self, id: SegmentId) -> Option<Segment> {
        self.segments.read().get(&id).cloned()
    }
    
    /// Mark vectors as deleted in segment
    pub fn mark_deleted(&self, id: SegmentId, count: usize) {
        if let Some(segment) = self.segments.write().get_mut(&id) {
            segment.mark_deleted(count);
        }
    }
    
    /// Record quantizer error for segment
    pub fn record_quantizer_error(&self, id: SegmentId, error: f32) {
        if let Some(segment) = self.segments.write().get_mut(&id) {
            segment.stats.record_error(error);
        }
    }
    
    /// Check if compaction is needed and return jobs
    pub fn maybe_compact(&self) -> Vec<CompactionJob> {
        // Check cooldown
        let mut last = self.last_compaction.lock();
        if let Some(last_time) = *last {
            if last_time.elapsed() < self.planner.policy().compaction_cooldown {
                return Vec::new();
            }
        }
        
        // Get active segments
        let segments = self.segments.read();
        let active: Vec<_> = segments.values()
            .filter(|s| s.state == SegmentState::Active)
            .collect();
        
        let decisions = self.planner.plan(&active);
        
        if !decisions.is_empty() {
            *last = Some(Instant::now());
        }
        
        decisions.into_iter().map(|d| {
            let source_segments = match &d {
                CompactionDecision::None => Vec::new(),
                CompactionDecision::Merge(ids) => ids.clone(),
                CompactionDecision::Split(id) => vec![*id],
                CompactionDecision::Retrain(ids) => ids.clone(),
                CompactionDecision::FullRecompact(ids) => ids.clone(),
            };
            
            CompactionJob {
                id: self.job_counter.fetch_add(1, Ordering::SeqCst),
                decision: d,
                source_segments,
                created_at: Instant::now(),
                priority: 0,
            }
        }).collect()
    }
    
    /// Execute compaction job
    pub fn execute_compaction(&self, job: &CompactionJob) -> Result<Option<Segment>, CompactionError> {
        match &job.decision {
            CompactionDecision::None => Ok(None),
            
            CompactionDecision::Merge(ids) => {
                // Mark source segments as compacting
                {
                    let mut segments = self.segments.write();
                    for id in ids {
                        if let Some(seg) = segments.get_mut(id) {
                            seg.state = SegmentState::Compacting;
                        }
                    }
                }
                
                // Create merged segment (placeholder implementation)
                let merged_id = SegmentId::next();
                let segments = self.segments.read();
                
                let total_size: u64 = ids.iter()
                    .filter_map(|id| segments.get(id))
                    .map(|s| s.stats.size_bytes)
                    .sum();
                
                let total_live: usize = ids.iter()
                    .filter_map(|id| segments.get(id))
                    .map(|s| s.stats.live_vectors())
                    .sum();
                
                let max_gen = ids.iter()
                    .filter_map(|id| segments.get(id))
                    .map(|s| s.generation)
                    .max()
                    .unwrap_or(0);
                
                drop(segments);
                
                let mut merged = Segment::new(
                    merged_id,
                    total_live,
                    total_size,
                    format!("/segments/{}", merged_id.0),
                );
                merged.generation = max_gen + 1;
                merged.state = SegmentState::Active;
                
                // Mark source segments as tombstoned
                {
                    let mut segments = self.segments.write();
                    for id in ids {
                        if let Some(seg) = segments.get_mut(id) {
                            seg.state = SegmentState::Tombstoned;
                        }
                    }
                }
                
                self.add_segment(merged.clone());
                Ok(Some(merged))
            }
            
            CompactionDecision::Split(id) => {
                // Split implementation
                let segment = self.get_segment(*id)
                    .ok_or(CompactionError::SegmentNotFound(*id))?;
                
                let half_size = segment.stats.size_bytes / 2;
                let half_vectors = segment.stats.n_vectors / 2;
                
                let seg1_id = SegmentId::next();
                let seg2_id = SegmentId::next();
                
                let mut seg1 = Segment::new(
                    seg1_id,
                    half_vectors,
                    half_size,
                    format!("/segments/{}", seg1_id.0),
                );
                seg1.generation = segment.generation + 1;
                seg1.state = SegmentState::Active;
                
                let mut seg2 = Segment::new(
                    seg2_id,
                    segment.stats.n_vectors - half_vectors,
                    segment.stats.size_bytes - half_size,
                    format!("/segments/{}", seg2_id.0),
                );
                seg2.generation = segment.generation + 1;
                seg2.state = SegmentState::Active;
                
                // Mark original as tombstoned
                if let Some(seg) = self.segments.write().get_mut(id) {
                    seg.state = SegmentState::Tombstoned;
                }
                
                self.add_segment(seg1);
                self.add_segment(seg2.clone());
                
                Ok(Some(seg2))
            }
            
            CompactionDecision::Retrain(_ids) => {
                // Retraining would involve:
                // 1. Sample vectors from segments
                // 2. Train new quantizer
                // 3. Re-encode vectors
                // 4. Create new segments
                // Placeholder for now
                Ok(None)
            }
            
            CompactionDecision::FullRecompact(ids) => {
                // Full recompaction with new quantizer
                self.execute_compaction(&CompactionJob {
                    id: job.id,
                    decision: CompactionDecision::Merge(ids.clone()),
                    source_segments: ids.clone(),
                    created_at: job.created_at,
                    priority: job.priority,
                })
            }
        }
    }
    
    /// Clean tombstoned segments
    pub fn clean_tombstones(&self) -> Vec<SegmentId> {
        let mut segments = self.segments.write();
        let tombstoned: Vec<_> = segments.iter()
            .filter(|(_, s)| s.state == SegmentState::Tombstoned)
            .map(|(id, _)| *id)
            .collect();
        
        for id in &tombstoned {
            if let Some(seg) = segments.get_mut(id) {
                seg.state = SegmentState::Deleted;
            }
        }
        
        tombstoned
    }
    
    /// Get statistics
    pub fn stats(&self) -> ManagerStats {
        let segments = self.segments.read();
        
        let total_segments = segments.len();
        let active_segments = segments.values()
            .filter(|s| s.state == SegmentState::Active)
            .count();
        
        let total_vectors: usize = segments.values()
            .filter(|s| s.state == SegmentState::Active)
            .map(|s| s.stats.n_vectors)
            .sum();
        
        let total_deleted: usize = segments.values()
            .filter(|s| s.state == SegmentState::Active)
            .map(|s| s.stats.n_deleted)
            .sum();
        
        let total_size: u64 = segments.values()
            .filter(|s| s.state == SegmentState::Active)
            .map(|s| s.stats.size_bytes)
            .sum();
        
        let avg_deletion_ratio = if active_segments > 0 {
            segments.values()
                .filter(|s| s.state == SegmentState::Active)
                .map(|s| s.stats.deletion_ratio())
                .sum::<f32>() / active_segments as f32
        } else {
            0.0
        };
        
        ManagerStats {
            total_segments,
            active_segments,
            total_vectors,
            live_vectors: total_vectors - total_deleted,
            deleted_vectors: total_deleted,
            total_size_bytes: total_size,
            avg_deletion_ratio,
            current_version: self.versions.current(),
        }
    }
    
    /// Get version manager
    pub fn versions(&self) -> &VersionManager {
        &self.versions
    }
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStats {
    pub total_segments: usize,
    pub active_segments: usize,
    pub total_vectors: usize,
    pub live_vectors: usize,
    pub deleted_vectors: usize,
    pub total_size_bytes: u64,
    pub avg_deletion_ratio: f32,
    pub current_version: u64,
}

/// Compaction error
#[derive(Debug)]
pub enum CompactionError {
    SegmentNotFound(SegmentId),
    IoError(std::io::Error),
    InvalidState(String),
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SegmentNotFound(id) => write!(f, "Segment not found: {:?}", id),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::InvalidState(s) => write!(f, "Invalid state: {}", s),
        }
    }
}

impl std::error::Error for CompactionError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_segment_lifecycle() {
        let mut segment = Segment::new(
            SegmentId::next(),
            1000,
            1024 * 1024,
            "/data/segment1".to_string(),
        );
        
        assert_eq!(segment.state, SegmentState::Building);
        
        segment.activate();
        assert_eq!(segment.state, SegmentState::Active);
        
        segment.mark_deleted(100);
        assert_eq!(segment.stats.n_deleted, 100);
        assert_eq!(segment.stats.live_vectors(), 900);
        
        let ratio = segment.stats.deletion_ratio();
        assert!((ratio - 0.1).abs() < 0.001);
    }
    
    #[test]
    fn test_compaction_planner() {
        let policy = CompactionPolicy {
            deletion_ratio_threshold: 0.3,
            min_segment_size: 1024,
            max_segment_size: 1024 * 1024,
            ..Default::default()
        };
        
        let planner = CompactionPlanner::new(policy);
        
        // Create segment with high deletion
        let mut seg1 = Segment::new(SegmentId(1), 1000, 2048, "/seg1".to_string());
        seg1.state = SegmentState::Active;
        seg1.stats.n_deleted = 400; // 40% deleted
        
        // Create small segments
        let mut seg2 = Segment::new(SegmentId(2), 100, 512, "/seg2".to_string());
        seg2.state = SegmentState::Active;
        
        let mut seg3 = Segment::new(SegmentId(3), 100, 512, "/seg3".to_string());
        seg3.state = SegmentState::Active;
        
        let segments: Vec<&Segment> = vec![&seg1, &seg2, &seg3];
        let decisions = planner.plan(&segments);
        
        // Should recommend merging high-deletion and small segments
        assert!(!decisions.is_empty());
    }
    
    #[test]
    fn test_version_manager() {
        let vm = VersionManager::new();
        
        let v1 = vm.create_version(vec![SegmentId(1), SegmentId(2)]);
        let v2 = vm.create_version(vec![SegmentId(1), SegmentId(2), SegmentId(3)]);
        
        assert!(v2 > v1);
        
        vm.switch_to(v2);
        assert_eq!(vm.current(), v2);
        
        vm.rollback();
        assert_eq!(vm.current(), v2 - 1);
        
        let segments = vm.get_segments(v1).unwrap();
        assert_eq!(segments.len(), 2);
    }
    
    #[test]
    fn test_segment_manager() {
        let policy = CompactionPolicy::default();
        let manager = SegmentManager::new(policy);
        
        // Add segments
        let mut seg1 = Segment::new(SegmentId::next(), 1000, 1024 * 1024, "/seg1".to_string());
        seg1.state = SegmentState::Active;
        manager.add_segment(seg1);
        
        let mut seg2 = Segment::new(SegmentId::next(), 500, 512 * 1024, "/seg2".to_string());
        seg2.state = SegmentState::Active;
        manager.add_segment(seg2);
        
        let stats = manager.stats();
        assert_eq!(stats.active_segments, 2);
        assert_eq!(stats.total_vectors, 1500);
    }
    
    #[test]
    fn test_quantizer_drift() {
        let mut stats = SegmentStats::new(1000, 1024);
        stats.quantizer_meta.training_error = 0.1;
        
        // No drift yet
        assert!(!stats.needs_retraining(0.2));
        
        // Add error samples showing drift
        for _ in 0..100 {
            stats.record_error(0.15); // 50% higher than training
        }
        
        assert!(stats.needs_retraining(0.2));
    }
}
