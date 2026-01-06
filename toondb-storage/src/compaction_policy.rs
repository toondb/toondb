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

//! Compaction Policy for Version and Tombstone Pruning
//!
//! This module provides intelligent compaction policies that:
//! - Aggressively prune old versions beyond retention window
//! - Collect tombstones that have expired
//! - Minimize write amplification (WA) and space amplification (SA)
//! - Balance between read performance and compaction overhead
//!
//! ## Compaction Strategies
//!
//! 1. **Leveled Compaction**: Traditional RocksDB-style L0 → L1 → ... compaction
//! 2. **Universal Compaction**: Minimize WA for write-heavy workloads
//! 3. **FIFO Compaction**: Age-based TTL with minimal overhead
//! 4. **Tiered Compaction**: Hybrid approach balancing WA and SA

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;

/// Timestamp type (milliseconds since epoch)
pub type Timestamp = u64;

/// Version retention configuration
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    /// Maximum age for old versions (None = keep forever)
    pub max_version_age: Option<Duration>,
    /// Maximum number of versions to keep per key
    pub max_versions_per_key: usize,
    /// Tombstone grace period before collection
    pub tombstone_grace_period: Duration,
    /// Minimum age before a file is eligible for compaction
    pub min_file_age: Duration,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            max_version_age: Some(Duration::from_secs(7 * 24 * 60 * 60)), // 7 days
            max_versions_per_key: 10,
            tombstone_grace_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            min_file_age: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Compaction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Leveled compaction (low SA, higher WA)
    Leveled,
    /// Universal compaction (low WA, higher SA)
    Universal,
    /// FIFO compaction (TTL-based, minimal overhead)
    Fifo,
    /// Tiered compaction (balanced WA/SA)
    Tiered,
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        Self::Leveled
    }
}

/// Compaction policy configuration
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Compaction strategy
    pub strategy: CompactionStrategy,
    /// Retention configuration
    pub retention: RetentionConfig,
    /// Level 0 file limit before triggering compaction
    pub l0_compaction_trigger: usize,
    /// Maximum level 0 files before stopping writes
    pub l0_stop_writes_trigger: usize,
    /// Maximum bytes for level 1
    pub max_bytes_for_level_base: u64,
    /// Level multiplier (bytes_for_level[n+1] = multiplier * bytes_for_level[n])
    pub max_bytes_for_level_multiplier: f64,
    /// Target file size for base level
    pub target_file_size_base: u64,
    /// File size multiplier per level
    pub target_file_size_multiplier: f64,
    /// Maximum number of concurrent compactions
    pub max_concurrent_compactions: usize,
    /// Percentage of reads to sample for tombstone density
    pub tombstone_sample_rate: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            strategy: CompactionStrategy::Leveled,
            retention: RetentionConfig::default(),
            l0_compaction_trigger: 4,
            l0_stop_writes_trigger: 12,
            max_bytes_for_level_base: 256 * 1024 * 1024, // 256 MB
            max_bytes_for_level_multiplier: 10.0,
            target_file_size_base: 64 * 1024 * 1024, // 64 MB
            target_file_size_multiplier: 1.0,
            max_concurrent_compactions: 4,
            tombstone_sample_rate: 0.01,
        }
    }
}

/// File metadata for compaction decisions
#[derive(Debug, Clone)]
pub struct CompactionFile {
    /// File ID
    pub id: u64,
    /// Level (0 = memtable flush, 1+ = compacted levels)
    pub level: u32,
    /// File size in bytes
    pub size: u64,
    /// Smallest key
    pub smallest_key: Vec<u8>,
    /// Largest key
    pub largest_key: Vec<u8>,
    /// Number of entries
    pub num_entries: u64,
    /// Number of deletions (tombstones)
    pub num_deletions: u64,
    /// Number of old versions (non-latest)
    pub num_old_versions: u64,
    /// Oldest entry timestamp
    pub oldest_timestamp: Timestamp,
    /// Newest entry timestamp
    pub newest_timestamp: Timestamp,
    /// Creation time
    pub created_at: Instant,
}

impl CompactionFile {
    /// Calculate tombstone density
    pub fn tombstone_density(&self) -> f64 {
        if self.num_entries == 0 {
            0.0
        } else {
            self.num_deletions as f64 / self.num_entries as f64
        }
    }

    /// Calculate version density (ratio of old versions)
    pub fn version_density(&self) -> f64 {
        if self.num_entries == 0 {
            0.0
        } else {
            self.num_old_versions as f64 / self.num_entries as f64
        }
    }

    /// Calculate garbage ratio (tombstones + old versions)
    pub fn garbage_ratio(&self) -> f64 {
        if self.num_entries == 0 {
            0.0
        } else {
            (self.num_deletions + self.num_old_versions) as f64 / self.num_entries as f64
        }
    }

    /// Check if file overlaps with key range
    pub fn overlaps(&self, smallest: &[u8], largest: &[u8]) -> bool {
        self.smallest_key.as_slice() <= largest && self.largest_key.as_slice() >= smallest
    }
}

/// Compaction job description
#[derive(Debug, Clone)]
pub struct CompactionJob {
    /// Job ID
    pub id: u64,
    /// Input files
    pub inputs: Vec<CompactionFile>,
    /// Target level
    pub target_level: u32,
    /// Priority (higher = more urgent)
    pub priority: CompactionPriority,
    /// Estimated output size
    pub estimated_output_size: u64,
    /// Reason for compaction
    pub reason: CompactionReason,
}

/// Compaction priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompactionPriority {
    /// Low priority - background maintenance
    Low = 0,
    /// Normal priority - regular compaction
    Normal = 1,
    /// High priority - L0 file count is getting high
    High = 2,
    /// Urgent - approaching write stall
    Urgent = 3,
}

/// Reason for triggering compaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionReason {
    /// L0 file count trigger
    L0FileCount,
    /// Level size limit exceeded
    LevelSizeLimit,
    /// High tombstone density
    TombstoneDensity,
    /// High old version density
    VersionPruning,
    /// TTL expiration
    TtlExpiration,
    /// Manual trigger
    Manual,
    /// Forced flush
    ForcedFlush,
}

/// Compaction picker - selects files for compaction
pub trait CompactionPicker: Send + Sync {
    /// Pick the next compaction job
    fn pick_compaction(&self, state: &CompactionState) -> Option<CompactionJob>;

    /// Calculate level targets
    fn calculate_level_targets(&self, state: &CompactionState) -> Vec<u64>;
}

/// State of the LSM tree for compaction decisions
#[derive(Debug, Default)]
pub struct CompactionState {
    /// Files by level
    pub files_by_level: Vec<Vec<CompactionFile>>,
    /// Total size by level
    pub size_by_level: Vec<u64>,
    /// Current timestamp for age calculations
    pub current_time: Timestamp,
    /// Oldest readable snapshot
    pub oldest_snapshot: Timestamp,
}

impl CompactionState {
    /// Get total number of L0 files
    pub fn l0_file_count(&self) -> usize {
        self.files_by_level.get(0).map(|f| f.len()).unwrap_or(0)
    }

    /// Get level count
    pub fn level_count(&self) -> usize {
        self.files_by_level.len()
    }

    /// Find overlapping files in a level
    pub fn find_overlapping(&self, level: usize, smallest: &[u8], largest: &[u8]) -> Vec<&CompactionFile> {
        self.files_by_level
            .get(level)
            .map(|files| {
                files.iter().filter(|f| f.overlaps(smallest, largest)).collect()
            })
            .unwrap_or_default()
    }
}

/// Leveled compaction picker
pub struct LeveledCompactionPicker {
    config: CompactionConfig,
    job_counter: AtomicU64,
}

impl LeveledCompactionPicker {
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            config,
            job_counter: AtomicU64::new(0),
        }
    }

    /// Pick L0 compaction
    fn pick_l0_compaction(&self, state: &CompactionState) -> Option<CompactionJob> {
        let l0_files = state.files_by_level.get(0)?;
        
        if l0_files.len() < self.config.l0_compaction_trigger {
            return None;
        }

        // All L0 files go to L1
        let inputs: Vec<_> = l0_files.clone();
        if inputs.is_empty() {
            return None;
        }

        // Find overlapping L1 files
        let smallest = inputs.iter().map(|f| f.smallest_key.as_slice()).min()?;
        let largest = inputs.iter().map(|f| f.largest_key.as_slice()).max()?;

        let l1_overlapping = state.find_overlapping(1, smallest, largest);
        let mut all_inputs = inputs;
        all_inputs.extend(l1_overlapping.into_iter().cloned());

        let priority = if l0_files.len() >= self.config.l0_stop_writes_trigger {
            CompactionPriority::Urgent
        } else if l0_files.len() >= self.config.l0_compaction_trigger * 2 {
            CompactionPriority::High
        } else {
            CompactionPriority::Normal
        };

        let estimated_output_size: u64 = all_inputs.iter().map(|f| f.size).sum();

        Some(CompactionJob {
            id: self.job_counter.fetch_add(1, AtomicOrdering::SeqCst),
            inputs: all_inputs,
            target_level: 1,
            priority,
            estimated_output_size,
            reason: CompactionReason::L0FileCount,
        })
    }

    /// Pick level-to-level compaction based on size
    fn pick_level_compaction(&self, state: &CompactionState) -> Option<CompactionJob> {
        let targets = self.calculate_level_targets(state);

        // Find level that most exceeds its target
        let mut max_score = 0.0f64;
        let mut pick_level = None;

        for level in 1..state.level_count() {
            if level >= targets.len() {
                continue;
            }
            let target = targets[level];
            if target == 0 {
                continue;
            }
            let actual = state.size_by_level.get(level).copied().unwrap_or(0);
            let score = actual as f64 / target as f64;
            if score > max_score && score > 1.0 {
                max_score = score;
                pick_level = Some(level);
            }
        }

        let level = pick_level?;
        let files = state.files_by_level.get(level)?;

        // Pick file with most garbage for version/tombstone pruning
        let pick_file = files
            .iter()
            .max_by(|a, b| {
                a.garbage_ratio()
                    .partial_cmp(&b.garbage_ratio())
                    .unwrap_or(Ordering::Equal)
            })?
            .clone();

        // Find overlapping files in next level
        let next_level = level + 1;
        let overlapping = state.find_overlapping(
            next_level,
            &pick_file.smallest_key,
            &pick_file.largest_key,
        );

        let mut inputs = vec![pick_file];
        inputs.extend(overlapping.into_iter().cloned());

        let estimated_output_size: u64 = inputs.iter().map(|f| f.size).sum();

        Some(CompactionJob {
            id: self.job_counter.fetch_add(1, AtomicOrdering::SeqCst),
            inputs,
            target_level: next_level as u32,
            priority: CompactionPriority::Normal,
            estimated_output_size,
            reason: if max_score > 2.0 {
                CompactionReason::LevelSizeLimit
            } else {
                CompactionReason::VersionPruning
            },
        })
    }

    /// Pick tombstone-driven compaction
    fn pick_tombstone_compaction(&self, state: &CompactionState) -> Option<CompactionJob> {
        // Find file with highest tombstone density
        for level in 0..state.level_count() {
            if let Some(files) = state.files_by_level.get(level) {
                for file in files {
                    // Compact if tombstone density > 50%
                    if file.tombstone_density() > 0.5 {
                        let overlapping = state.find_overlapping(
                            level + 1,
                            &file.smallest_key,
                            &file.largest_key,
                        );

                        let mut inputs = vec![file.clone()];
                        inputs.extend(overlapping.into_iter().cloned());

                        return Some(CompactionJob {
                            id: self.job_counter.fetch_add(1, AtomicOrdering::SeqCst),
                            inputs,
                            target_level: (level + 1) as u32,
                            priority: CompactionPriority::Normal,
                            estimated_output_size: file.size / 2, // Estimate 50% reduction
                            reason: CompactionReason::TombstoneDensity,
                        });
                    }
                }
            }
        }

        None
    }
}

impl CompactionPicker for LeveledCompactionPicker {
    fn pick_compaction(&self, state: &CompactionState) -> Option<CompactionJob> {
        // Priority order:
        // 1. L0 compaction (avoid write stalls)
        // 2. Tombstone-driven compaction (reclaim space)
        // 3. Level size compaction

        self.pick_l0_compaction(state)
            .or_else(|| self.pick_tombstone_compaction(state))
            .or_else(|| self.pick_level_compaction(state))
    }

    fn calculate_level_targets(&self, state: &CompactionState) -> Vec<u64> {
        let mut targets = vec![0u64; state.level_count().max(7)];

        // L0 doesn't have a size target, it has a file count trigger
        targets[0] = 0;

        // L1 is the base
        targets[1] = self.config.max_bytes_for_level_base;

        // Each subsequent level is multiplied
        for level in 2..targets.len() {
            targets[level] = (targets[level - 1] as f64
                * self.config.max_bytes_for_level_multiplier) as u64;
        }

        targets
    }
}

/// Universal compaction picker (for write-heavy workloads)
pub struct UniversalCompactionPicker {
    config: CompactionConfig,
    job_counter: AtomicU64,
}

impl UniversalCompactionPicker {
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            config,
            job_counter: AtomicU64::new(0),
        }
    }
}

impl CompactionPicker for UniversalCompactionPicker {
    fn pick_compaction(&self, state: &CompactionState) -> Option<CompactionJob> {
        // Universal compaction: pick sorted runs to merge based on size ratios
        let all_files: Vec<_> = state.files_by_level.iter().flatten().cloned().collect();

        if all_files.len() < 2 {
            return None;
        }

        // Sort by creation time (oldest first for FIFO-like behavior)
        let mut sorted_files = all_files;
        sorted_files.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        // Pick adjacent runs where size ratio is acceptable
        let size_ratio_threshold = 2.0;
        let mut inputs = Vec::new();
        let mut total_size = 0u64;

        for file in sorted_files {
            if inputs.is_empty() || (total_size as f64 / file.size as f64) < size_ratio_threshold {
                total_size += file.size;
                inputs.push(file);
            } else {
                break;
            }
        }

        if inputs.len() < 2 {
            return None;
        }

        Some(CompactionJob {
            id: self.job_counter.fetch_add(1, AtomicOrdering::SeqCst),
            inputs,
            target_level: 0, // Universal keeps everything at L0
            priority: CompactionPriority::Normal,
            estimated_output_size: total_size,
            reason: CompactionReason::LevelSizeLimit,
        })
    }

    fn calculate_level_targets(&self, _state: &CompactionState) -> Vec<u64> {
        // Universal compaction doesn't use level targets
        vec![]
    }
}

/// Version pruning filter
///
/// Determines which versions can be safely garbage collected.
pub struct VersionPruner {
    /// Retention configuration
    config: RetentionConfig,
    /// Set of active snapshot timestamps
    active_snapshots: RwLock<HashSet<Timestamp>>,
    /// Current timestamp
    current_time: AtomicU64,
}

impl VersionPruner {
    pub fn new(config: RetentionConfig) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            config,
            active_snapshots: RwLock::new(HashSet::new()),
            current_time: AtomicU64::new(now),
        }
    }

    /// Register an active snapshot
    pub fn register_snapshot(&self, timestamp: Timestamp) {
        self.active_snapshots.write().insert(timestamp);
    }

    /// Unregister a snapshot
    pub fn unregister_snapshot(&self, timestamp: Timestamp) {
        self.active_snapshots.write().remove(&timestamp);
    }

    /// Get the oldest active snapshot
    pub fn oldest_snapshot(&self) -> Option<Timestamp> {
        self.active_snapshots.read().iter().min().copied()
    }

    /// Check if a version can be pruned
    ///
    /// A version can be pruned if:
    /// 1. It's not the latest version
    /// 2. It's older than max_version_age
    /// 3. No active snapshot needs it
    ///
    /// A snapshot at time T needs all versions with timestamp <= T
    /// (these are the versions that were visible at snapshot time)
    pub fn can_prune_version(
        &self,
        version_timestamp: Timestamp,
        is_latest: bool,
        version_index: usize,
    ) -> bool {
        // Never prune the latest version
        if is_latest {
            return false;
        }

        // Check if any snapshot protects this version
        // A snapshot at time T protects versions with timestamp <= T
        if let Some(oldest) = self.oldest_snapshot() {
            if version_timestamp <= oldest {
                // Version was visible to the oldest snapshot, cannot prune
                return false;
            }
        }

        // Check max versions per key
        if version_index >= self.config.max_versions_per_key {
            return true;
        }

        // Check age-based pruning
        if let Some(max_age) = self.config.max_version_age {
            let now = self.current_time.load(AtomicOrdering::Relaxed);
            let age_ms = now.saturating_sub(version_timestamp);
            let max_age_ms = max_age.as_millis() as u64;

            if age_ms > max_age_ms {
                return true;
            }
        }

        false
    }

    /// Check if a tombstone can be collected
    ///
    /// A tombstone can be collected only if:
    /// 1. It's older than the grace period
    /// 2. No active snapshot needs to see it
    pub fn can_collect_tombstone(&self, tombstone_timestamp: Timestamp) -> bool {
        let now = self.current_time.load(AtomicOrdering::Relaxed);
        let grace_period_ms = self.config.tombstone_grace_period.as_millis() as u64;

        if now.saturating_sub(tombstone_timestamp) < grace_period_ms {
            return false;
        }

        // Check if any snapshot protects this tombstone
        if let Some(oldest) = self.oldest_snapshot() {
            if tombstone_timestamp <= oldest {
                // Tombstone was visible to the oldest snapshot, cannot collect
                return false;
            }
        }

        true
    }

    /// Update current time
    pub fn update_time(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.current_time.store(now, AtomicOrdering::Relaxed);
    }
}

/// Compaction statistics
#[derive(Debug, Default, Clone)]
pub struct CompactionStats {
    /// Number of compactions completed
    pub compactions_completed: u64,
    /// Total bytes read during compaction
    pub bytes_read: u64,
    /// Total bytes written during compaction
    pub bytes_written: u64,
    /// Entries processed
    pub entries_processed: u64,
    /// Tombstones collected
    pub tombstones_collected: u64,
    /// Old versions pruned
    pub versions_pruned: u64,
    /// Write amplification factor
    pub write_amplification: f64,
    /// Space amplification factor
    pub space_amplification: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file(id: u64, level: u32, size: u64, smallest: &str, largest: &str) -> CompactionFile {
        CompactionFile {
            id,
            level,
            size,
            smallest_key: smallest.as_bytes().to_vec(),
            largest_key: largest.as_bytes().to_vec(),
            num_entries: 1000,
            num_deletions: 0,
            num_old_versions: 0,
            oldest_timestamp: 0,
            newest_timestamp: 100,
            created_at: Instant::now(),
        }
    }

    #[test]
    fn test_leveled_picker_l0() {
        let picker = LeveledCompactionPicker::new(CompactionConfig {
            l0_compaction_trigger: 4,
            ..Default::default()
        });

        let mut state = CompactionState::default();
        state.files_by_level = vec![
            vec![
                make_file(1, 0, 1000, "a", "d"),
                make_file(2, 0, 1000, "c", "f"),
                make_file(3, 0, 1000, "e", "h"),
                make_file(4, 0, 1000, "g", "j"),
            ],
            vec![make_file(10, 1, 10000, "a", "z")],
        ];

        let job = picker.pick_compaction(&state);
        assert!(job.is_some());

        let job = job.unwrap();
        assert_eq!(job.target_level, 1);
        assert!(job.inputs.len() >= 4); // All L0 files + overlapping L1
    }

    #[test]
    fn test_tombstone_density() {
        let mut file = make_file(1, 0, 1000, "a", "z");
        file.num_entries = 100;
        file.num_deletions = 60;

        assert!(file.tombstone_density() > 0.5);
    }

    #[test]
    fn test_version_pruner() {
        let config = RetentionConfig {
            max_version_age: Some(Duration::from_secs(3600)),
            max_versions_per_key: 5,
            tombstone_grace_period: Duration::from_secs(60),
            min_file_age: Duration::from_secs(60),
        };

        let pruner = VersionPruner::new(config);

        // Latest version should never be pruned
        assert!(!pruner.can_prune_version(0, true, 0));

        // Version beyond max_versions_per_key can be pruned (if no snapshots)
        assert!(pruner.can_prune_version(0, false, 10));

        // Register a snapshot at timestamp 1000
        let snapshot_ts = 1000;
        pruner.register_snapshot(snapshot_ts);

        // Version with timestamp 500 (before snapshot) is protected by the snapshot
        // because the snapshot might need to read it
        assert!(!pruner.can_prune_version(500, false, 10));

        // Version with timestamp 1500 (after snapshot) can be pruned
        // because it was written after the snapshot was taken
        assert!(pruner.can_prune_version(1500, false, 10));
    }

    #[test]
    fn test_level_targets() {
        let picker = LeveledCompactionPicker::new(CompactionConfig {
            max_bytes_for_level_base: 256 * 1024 * 1024,
            max_bytes_for_level_multiplier: 10.0,
            ..Default::default()
        });

        let state = CompactionState {
            files_by_level: vec![vec![], vec![], vec![], vec![]],
            size_by_level: vec![0, 0, 0, 0],
            current_time: 0,
            oldest_snapshot: 0,
        };

        let targets = picker.calculate_level_targets(&state);

        assert_eq!(targets[0], 0); // L0 has no size target
        assert_eq!(targets[1], 256 * 1024 * 1024);
        assert_eq!(targets[2], 2560 * 1024 * 1024); // 10x L1
    }

    #[test]
    fn test_compaction_priority() {
        assert!(CompactionPriority::Urgent > CompactionPriority::High);
        assert!(CompactionPriority::High > CompactionPriority::Normal);
        assert!(CompactionPriority::Normal > CompactionPriority::Low);
    }

    #[test]
    fn test_file_overlaps() {
        let file = make_file(1, 0, 1000, "d", "h");

        assert!(file.overlaps(b"a", b"e")); // Overlaps start
        assert!(file.overlaps(b"f", b"z")); // Overlaps end
        assert!(file.overlaps(b"e", b"g")); // Contained
        assert!(file.overlaps(b"a", b"z")); // Contains file
        assert!(!file.overlaps(b"a", b"c")); // Before
        assert!(!file.overlaps(b"i", b"z")); // After
    }
}
