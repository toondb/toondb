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

//! # SSD Rerank Executor with IO Coalescing (Task 8)
//!
//! Provides efficient hybrid RAM+SSD reranking with:
//! - IO coalescing for locality
//! - Strict IO budget enforcement
//! - Tail-latency guardrails
//!
//! ## Architecture
//!
//! 1. Map candidate IDs to disk offsets
//! 2. Sort offsets and coalesce into minimal IO ranges
//! 3. Enforce IO budget and stop early if needed
//! 4. Cache admission for hot vectors
//!
//! ## Math/Algorithm
//!
//! Coalescing: O(R log R) sort + O(R) interval merge
//! Tail guardrails: online control via capped outstanding IO ops
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::ssd_rerank::{RerankExecutor, RerankConfig, RerankRequest};
//!
//! let config = RerankConfig::default()
//!     .io_budget(100)
//!     .coalesce_threshold(4096);
//!
//! let executor = RerankExecutor::new(config, storage);
//! let results = executor.rerank(&candidates, &query);
//! ```

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cost_model::CostTracker;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for SSD rerank executor
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Maximum number of IO operations allowed
    pub max_io_ops: u32,
    
    /// Maximum bytes to read
    pub max_io_bytes: u64,
    
    /// Maximum latency before early termination
    pub max_latency: Duration,
    
    /// Threshold for coalescing adjacent reads (bytes)
    /// Reads within this gap will be merged
    pub coalesce_threshold: u64,
    
    /// Minimum number of candidates to rerank before considering early stop
    pub min_rerank_candidates: usize,
    
    /// Enable hot vector caching
    pub enable_cache: bool,
    
    /// Cache size in number of vectors
    pub cache_size: usize,
    
    /// Queue depth for io_uring (if available)
    pub io_queue_depth: u32,
    
    /// Prefetch distance for sequential reads
    pub prefetch_distance: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            max_io_ops: 100,
            max_io_bytes: 16 * 1024 * 1024, // 16 MB
            max_latency: Duration::from_millis(50),
            coalesce_threshold: 4096, // 4 KB page
            min_rerank_candidates: 10,
            enable_cache: true,
            cache_size: 10000,
            io_queue_depth: 64,
            prefetch_distance: 4,
        }
    }
}

impl RerankConfig {
    /// Set IO budget
    pub fn io_budget(mut self, max_ops: u32) -> Self {
        self.max_io_ops = max_ops;
        self
    }
    
    /// Set coalesce threshold
    pub fn coalesce_threshold(mut self, bytes: u64) -> Self {
        self.coalesce_threshold = bytes;
        self
    }
    
    /// Set max latency
    pub fn max_latency(mut self, latency: Duration) -> Self {
        self.max_latency = latency;
        self
    }
}

// ============================================================================
// IO Range
// ============================================================================

/// A coalesced IO range
#[derive(Debug, Clone)]
pub struct IoRange {
    /// Start offset in file
    pub offset: u64,
    /// Length to read
    pub length: u64,
    /// Original candidate indices that fall within this range
    pub candidate_indices: Vec<usize>,
}

impl IoRange {
    /// Create from a single candidate
    pub fn single(offset: u64, length: u64, candidate_idx: usize) -> Self {
        Self {
            offset,
            length,
            candidate_indices: vec![candidate_idx],
        }
    }
    
    /// Try to merge with another range
    pub fn try_merge(&mut self, other: &IoRange, threshold: u64) -> bool {
        let self_end = self.offset + self.length;
        let other_end = other.offset + other.length;
        
        // Check if ranges overlap or are within threshold
        if other.offset <= self_end + threshold && self.offset <= other_end + threshold {
            // Merge
            let new_start = self.offset.min(other.offset);
            let new_end = self_end.max(other_end);
            
            self.offset = new_start;
            self.length = new_end - new_start;
            self.candidate_indices.extend_from_slice(&other.candidate_indices);
            true
        } else {
            false
        }
    }
    
    /// Get end offset
    pub fn end(&self) -> u64 {
        self.offset + self.length
    }
}

// ============================================================================
// Rerank Candidate
// ============================================================================

/// A candidate for reranking
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    /// Vector ID
    pub id: u32,
    /// Proxy score (from quantized search)
    pub proxy_score: f32,
    /// Disk offset where full vector is stored
    pub disk_offset: u64,
    /// Vector size in bytes
    pub vector_size: u32,
}

impl RerankCandidate {
    /// Create new candidate
    pub fn new(id: u32, proxy_score: f32, disk_offset: u64, vector_size: u32) -> Self {
        Self {
            id,
            proxy_score,
            disk_offset,
            vector_size,
        }
    }
}

// ============================================================================
// Rerank Result
// ============================================================================

/// Result from reranking
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Vector ID
    pub id: u32,
    /// True score (from full-precision computation)
    pub true_score: f32,
    /// Whether this was from cache
    pub from_cache: bool,
}

impl PartialEq for RerankResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RerankResult {}

impl PartialOrd for RerankResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Higher score = better, so reverse ordering for min-heap
        other.true_score.partial_cmp(&self.true_score)
    }
}

impl Ord for RerankResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// Rerank Statistics
// ============================================================================

/// Statistics from rerank execution
#[derive(Debug, Clone, Default)]
pub struct RerankStats {
    /// Number of candidates requested
    pub candidates_requested: usize,
    /// Number of candidates actually reranked
    pub candidates_reranked: usize,
    /// Number of IO operations
    pub io_ops: u32,
    /// Total bytes read
    pub io_bytes: u64,
    /// Number of coalesced ranges
    pub coalesced_ranges: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Whether budget was exhausted
    pub budget_exhausted: bool,
    /// Stop reason
    pub stop_reason: String,
    /// Total rerank time
    pub duration: Duration,
}

impl RerankStats {
    /// Compute IO amplification
    pub fn io_amplification(&self) -> f32 {
        if self.candidates_reranked == 0 {
            0.0
        } else {
            self.io_bytes as f32 / (self.candidates_reranked as f32 * 4.0 * 768.0) // Assume 768-dim f32
        }
    }
    
    /// Compute cache hit ratio
    pub fn cache_hit_ratio(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

// ============================================================================
// IO Coalescer
// ============================================================================

/// Coalesces random reads into sequential ranges
pub struct IoCoalescer {
    threshold: u64,
}

impl IoCoalescer {
    /// Create new coalescer
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }
    
    /// Coalesce candidates into IO ranges
    ///
    /// Algorithm:
    /// 1. Sort candidates by disk offset: O(n log n)
    /// 2. Merge adjacent ranges within threshold: O(n)
    /// 3. Return coalesced ranges
    pub fn coalesce(&self, candidates: &[RerankCandidate]) -> Vec<IoRange> {
        if candidates.is_empty() {
            return Vec::new();
        }
        
        // Sort by offset
        let mut indexed: Vec<(usize, &RerankCandidate)> = candidates.iter().enumerate().collect();
        indexed.sort_by_key(|(_, c)| c.disk_offset);
        
        let mut ranges: Vec<IoRange> = Vec::with_capacity(candidates.len());
        
        // Create initial range from first candidate
        let (first_idx, first) = indexed[0];
        let mut current = IoRange::single(first.disk_offset, first.vector_size as u64, first_idx);
        
        // Try to merge subsequent candidates
        for (idx, candidate) in indexed.iter().skip(1) {
            let new_range = IoRange::single(
                candidate.disk_offset,
                candidate.vector_size as u64,
                *idx,
            );
            
            if !current.try_merge(&new_range, self.threshold) {
                ranges.push(current);
                current = new_range;
            }
        }
        
        ranges.push(current);
        ranges
    }
    
    /// Compute coalescing statistics
    pub fn coalesce_stats(&self, candidates: &[RerankCandidate]) -> CoalesceStats {
        let ranges = self.coalesce(candidates);
        
        let total_raw_bytes: u64 = candidates.iter()
            .map(|c| c.vector_size as u64)
            .sum();
        
        let total_coalesced_bytes: u64 = ranges.iter()
            .map(|r| r.length)
            .sum();
        
        CoalesceStats {
            n_candidates: candidates.len(),
            n_ranges: ranges.len(),
            raw_bytes: total_raw_bytes,
            coalesced_bytes: total_coalesced_bytes,
            reduction_ratio: total_coalesced_bytes as f32 / total_raw_bytes.max(1) as f32,
        }
    }
}

/// Coalescing statistics
#[derive(Debug, Clone)]
pub struct CoalesceStats {
    pub n_candidates: usize,
    pub n_ranges: usize,
    pub raw_bytes: u64,
    pub coalesced_bytes: u64,
    pub reduction_ratio: f32,
}

// ============================================================================
// Vector Cache
// ============================================================================

/// Simple LRU cache for hot vectors
pub struct VectorCache {
    /// Cached vectors: id -> (vector, access_count)
    cache: parking_lot::RwLock<std::collections::HashMap<u32, (Vec<f32>, u64)>>,
    /// Maximum entries
    max_size: usize,
    /// Access counter for LRU
    access_counter: AtomicU64,
}

impl VectorCache {
    /// Create new cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: parking_lot::RwLock::new(std::collections::HashMap::with_capacity(max_size)),
            max_size,
            access_counter: AtomicU64::new(0),
        }
    }
    
    /// Get vector if cached
    pub fn get(&self, id: u32) -> Option<Vec<f32>> {
        let mut cache = self.cache.write();
        if let Some((vec, access)) = cache.get_mut(&id) {
            *access = self.access_counter.fetch_add(1, Ordering::Relaxed);
            Some(vec.clone())
        } else {
            None
        }
    }
    
    /// Insert vector into cache
    pub fn insert(&self, id: u32, vector: Vec<f32>) {
        let mut cache = self.cache.write();
        
        // Evict if full
        if cache.len() >= self.max_size {
            // Find LRU entry
            let lru_id = cache.iter()
                .min_by_key(|(_, (_, access))| *access)
                .map(|(id, _)| *id);
            
            if let Some(lru_id) = lru_id {
                cache.remove(&lru_id);
            }
        }
        
        let access = self.access_counter.fetch_add(1, Ordering::Relaxed);
        cache.insert(id, (vector, access));
    }
    
    /// Check if vector is cached
    pub fn contains(&self, id: u32) -> bool {
        self.cache.read().contains_key(&id)
    }
    
    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }
    
    /// Clear cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

// ============================================================================
// Rerank Executor
// ============================================================================

/// Distance function type
pub type DistanceFn = dyn Fn(&[f32], &[f32]) -> f32 + Send + Sync;

/// Storage reader type
pub type StorageReader = dyn Fn(u64, u64) -> std::io::Result<Vec<u8>> + Send + Sync;

/// Executes SSD-based reranking with IO coalescing
pub struct RerankExecutor {
    config: RerankConfig,
    coalescer: IoCoalescer,
    cache: Option<VectorCache>,
    /// Distance function (returns similarity, higher is better)
    distance_fn: Box<DistanceFn>,
    /// Storage reader function
    reader: Box<StorageReader>,
    /// Vector dimension
    dim: usize,
}

impl RerankExecutor {
    /// Create new executor with custom storage reader
    pub fn new<D, R>(config: RerankConfig, distance_fn: D, reader: R, dim: usize) -> Self
    where
        D: Fn(&[f32], &[f32]) -> f32 + Send + Sync + 'static,
        R: Fn(u64, u64) -> std::io::Result<Vec<u8>> + Send + Sync + 'static,
    {
        let cache = if config.enable_cache {
            Some(VectorCache::new(config.cache_size))
        } else {
            None
        };
        
        Self {
            coalescer: IoCoalescer::new(config.coalesce_threshold),
            cache,
            config,
            distance_fn: Box::new(distance_fn),
            reader: Box::new(reader),
            dim,
        }
    }
    
    /// Rerank candidates against query
    pub fn rerank(
        &self,
        candidates: &[RerankCandidate],
        query: &[f32],
        k: usize,
    ) -> (Vec<RerankResult>, RerankStats) {
        self.rerank_with_tracker(candidates, query, k, None)
    }
    
    /// Rerank with cost tracking
    pub fn rerank_with_tracker(
        &self,
        candidates: &[RerankCandidate],
        query: &[f32],
        k: usize,
        cost_tracker: Option<&CostTracker>,
    ) -> (Vec<RerankResult>, RerankStats) {
        let start = Instant::now();
        let mut stats = RerankStats {
            candidates_requested: candidates.len(),
            ..Default::default()
        };
        
        // Separate cached and uncached candidates
        let (cached_ids, uncached): (Vec<_>, Vec<_>) = candidates.iter().enumerate()
            .partition(|(_, c)| {
                self.cache.as_ref().map(|cache| cache.contains(c.id)).unwrap_or(false)
            });
        
        // Process cached candidates first
        let mut results: BinaryHeap<RerankResult> = BinaryHeap::new();
        
        for (idx, candidate) in cached_ids {
            if let Some(ref cache) = self.cache {
                if let Some(vector) = cache.get(candidate.id) {
                    let score = (self.distance_fn)(query, &vector);
                    results.push(RerankResult {
                        id: candidate.id,
                        true_score: score,
                        from_cache: true,
                    });
                    stats.cache_hits += 1;
                    stats.candidates_reranked += 1;
                }
            }
        }
        
        // Coalesce uncached candidates
        let uncached_candidates: Vec<_> = uncached.iter().map(|(_, c)| (*c).clone()).collect();
        let ranges = self.coalescer.coalesce(&uncached_candidates);
        stats.coalesced_ranges = ranges.len();
        
        // Budget tracking
        let mut io_ops = 0u32;
        let mut io_bytes = 0u64;
        
        // Process IO ranges
        'outer: for range in &ranges {
            // Check budgets
            if io_ops >= self.config.max_io_ops {
                stats.budget_exhausted = true;
                stats.stop_reason = "io_ops_exceeded".to_string();
                break;
            }
            
            if io_bytes + range.length > self.config.max_io_bytes {
                stats.budget_exhausted = true;
                stats.stop_reason = "io_bytes_exceeded".to_string();
                break;
            }
            
            if start.elapsed() > self.config.max_latency {
                stats.budget_exhausted = true;
                stats.stop_reason = "latency_exceeded".to_string();
                break;
            }
            
            // Track in cost tracker if provided
            if let Some(tracker) = cost_tracker {
                if !tracker.add_ssd_sequential_bytes(range.length) {
                    stats.budget_exhausted = true;
                    stats.stop_reason = "cost_budget_exhausted".to_string();
                    break;
                }
            }
            
            // Read from storage
            let data = match (self.reader)(range.offset, range.length) {
                Ok(data) => data,
                Err(_) => continue, // Skip failed reads
            };
            
            io_ops += 1;
            io_bytes += range.length;
            
            // Parse vectors from range
            for &candidate_idx in &range.candidate_indices {
                let candidate = &uncached_candidates[candidate_idx];
                
                // Calculate offset within the range
                let offset_in_range = candidate.disk_offset - range.offset;
                let start_byte = offset_in_range as usize;
                let end_byte = start_byte + candidate.vector_size as usize;
                
                if end_byte > data.len() {
                    continue; // Invalid range
                }
                
                // Parse f32 vector
                let vector_bytes = &data[start_byte..end_byte];
                let vector: Vec<f32> = vector_bytes
                    .chunks(4)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                        f32::from_le_bytes(arr)
                    })
                    .collect();
                
                // Compute true score
                let score = (self.distance_fn)(query, &vector);
                
                // Add to results
                results.push(RerankResult {
                    id: candidate.id,
                    true_score: score,
                    from_cache: false,
                });
                
                // Cache the vector
                if let Some(ref cache) = self.cache {
                    cache.insert(candidate.id, vector);
                }
                
                stats.cache_misses += 1;
                stats.candidates_reranked += 1;
                
                // Check if we can stop early (have enough good candidates)
                if results.len() >= k * 2 && stats.candidates_reranked >= self.config.min_rerank_candidates {
                    // Early termination heuristic
                }
            }
        }
        
        stats.io_ops = io_ops;
        stats.io_bytes = io_bytes;
        stats.duration = start.elapsed();
        
        if stats.stop_reason.is_empty() {
            stats.stop_reason = "complete".to_string();
        }
        
        // Extract top-k
        let mut top_k: Vec<RerankResult> = Vec::with_capacity(k);
        while top_k.len() < k && !results.is_empty() {
            if let Some(result) = results.pop() {
                top_k.push(result);
            }
        }
        
        // Sort by score descending
        top_k.sort_by(|a, b| b.true_score.partial_cmp(&a.true_score).unwrap());
        
        (top_k, stats)
    }
    
    /// Get configuration
    pub fn config(&self) -> &RerankConfig {
        &self.config
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<usize> {
        self.cache.as_ref().map(|c| c.len())
    }
}

// ============================================================================
// Mock Storage for Testing
// ============================================================================

/// Mock storage for testing
pub struct MockStorage {
    data: Vec<u8>,
}

impl MockStorage {
    /// Create mock storage with random vectors
    pub fn new(n_vectors: usize, dim: usize) -> Self {
        let mut data = Vec::with_capacity(n_vectors * dim * 4);
        
        for i in 0..n_vectors {
            for j in 0..dim {
                let val = ((i + j) as f32 / (n_vectors + dim) as f32);
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        
        Self { data }
    }
    
    /// Create reader function
    pub fn reader(&self) -> impl Fn(u64, u64) -> std::io::Result<Vec<u8>> + '_ {
        move |offset, length| {
            let start = offset as usize;
            let end = (start + length as usize).min(self.data.len());
            Ok(self.data[start..end].to_vec())
        }
    }
    
    /// Get offset for a vector
    pub fn offset(&self, id: u32, dim: usize) -> u64 {
        (id as usize * dim * 4) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_io_coalescing() {
        let coalescer = IoCoalescer::new(1024);
        
        let candidates = vec![
            RerankCandidate::new(0, 0.9, 0, 3072),      // 0-3072
            RerankCandidate::new(1, 0.8, 3072, 3072),   // 3072-6144 (adjacent)
            RerankCandidate::new(2, 0.7, 10000, 3072),  // 10000-13072 (gap)
            RerankCandidate::new(3, 0.6, 10500, 3072),  // Overlaps with #2
        ];
        
        let ranges = coalescer.coalesce(&candidates);
        
        // Should have 2 ranges: 0-6144 and 10000-13572
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].offset, 0);
        assert!(ranges[0].length >= 6144);
    }
    
    #[test]
    fn test_vector_cache() {
        let cache = VectorCache::new(3);
        
        cache.insert(1, vec![1.0, 2.0, 3.0]);
        cache.insert(2, vec![4.0, 5.0, 6.0]);
        cache.insert(3, vec![7.0, 8.0, 9.0]);
        
        assert!(cache.contains(1));
        assert!(cache.contains(2));
        assert!(cache.contains(3));
        
        // Access 1 and 2 to make them more recent
        cache.get(1);
        cache.get(2);
        
        // Insert 4, should evict 3 (least recently used)
        cache.insert(4, vec![10.0, 11.0, 12.0]);
        
        assert!(cache.contains(1));
        assert!(cache.contains(2));
        assert!(!cache.contains(3)); // Evicted
        assert!(cache.contains(4));
    }
    
    #[test]
    fn test_rerank_executor() {
        let dim = 4;
        let storage = MockStorage::new(100, dim);
        
        let config = RerankConfig::default();
        
        let distance_fn = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };
        
        let data_clone = storage.data.clone();
        let reader = move |offset: u64, length: u64| -> std::io::Result<Vec<u8>> {
            let start = offset as usize;
            let end = (start + length as usize).min(data_clone.len());
            Ok(data_clone[start..end].to_vec())
        };
        
        let executor = RerankExecutor::new(config, distance_fn, reader, dim);
        
        let candidates: Vec<RerankCandidate> = (0..10)
            .map(|i| RerankCandidate::new(
                i,
                0.9 - i as f32 * 0.01,
                storage.offset(i, dim),
                (dim * 4) as u32,
            ))
            .collect();
        
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let (results, stats) = executor.rerank(&candidates, &query, 5);
        
        assert!(results.len() <= 5);
        assert!(stats.candidates_reranked > 0);
        assert!(stats.io_ops > 0);
    }
    
    #[test]
    fn test_coalesce_stats() {
        let coalescer = IoCoalescer::new(100);
        
        // Create candidates with good locality
        let candidates: Vec<RerankCandidate> = (0..10)
            .map(|i| RerankCandidate::new(i, 0.9, i as u64 * 50, 50))
            .collect();
        
        let stats = coalescer.coalesce_stats(&candidates);
        
        assert_eq!(stats.n_candidates, 10);
        assert!(stats.n_ranges < 10); // Should be coalesced
        assert!(stats.reduction_ratio >= 1.0); // May be slightly larger due to gaps
    }
}
