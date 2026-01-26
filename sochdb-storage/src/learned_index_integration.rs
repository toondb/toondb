// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Learned Index Integration (Task 5)
//!
//! Integrates LearnedSparseIndex for O(1) expected point lookups:
//! - Accelerates row lookups by key in LSCS
//! - Falls back to binary search for outliers
//! - Provides adaptive index selection based on data distribution
//!
//! ## Lookup Flow
//!
//! ```text
//! get(key)
//!   │
//!   ▼
//! ┌─────────────────────┐
//! │ LearnedIndex.lookup │
//! └──────────┬──────────┘
//!            │
//!      ┌─────┴─────┐
//!      ▼           ▼
//!   Exact       Range[lo,hi]
//!     │             │
//!     ▼             ▼
//!   O(1)      BinarySearch
//!   fetch       O(log ε)
//! ```
//!
//! ## Index Selection
//!
//! The system automatically chooses between:
//! - **LearnedIndex**: For sequential/timestamp keys (O(1))
//! - **B-Tree**: For random/UUID keys (O(log N))
//! - **Hash**: For exact-match only (O(1))

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use sochdb_core::learned_index::{LearnedSparseIndex, LookupResult};

/// Index type based on key characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// Learned index for sequential/monotonic keys
    Learned,
    /// B-Tree for random keys
    BTree,
    /// Hash for exact match only
    Hash,
    /// No index (scan)
    None,
}

/// Key distribution statistics for index selection
#[derive(Debug, Clone, Default)]
pub struct KeyStats {
    /// Number of keys analyzed
    pub count: usize,
    /// Minimum key value
    pub min_key: u64,
    /// Maximum key value
    pub max_key: u64,
    /// Is monotonically increasing
    pub is_monotonic: bool,
    /// Density: count / (max - min + 1)
    pub density: f64,
    /// Estimated entropy (randomness)
    pub entropy: f64,
}

impl KeyStats {
    /// Analyze keys to determine statistics
    pub fn analyze(keys: &[u64]) -> Self {
        if keys.is_empty() {
            return Self::default();
        }

        let min_key = *keys.iter().min().unwrap();
        let max_key = *keys.iter().max().unwrap();
        let count = keys.len();

        // Check monotonicity
        let is_monotonic = keys.windows(2).all(|w| w[0] <= w[1]);

        // Calculate density
        let range = (max_key - min_key + 1) as f64;
        let density = count as f64 / range;

        // Estimate entropy from gaps
        let entropy = Self::estimate_entropy(keys);

        Self {
            count,
            min_key,
            max_key,
            is_monotonic,
            density,
            entropy,
        }
    }

    /// Estimate entropy from key gaps
    fn estimate_entropy(keys: &[u64]) -> f64 {
        if keys.len() < 2 {
            return 0.0;
        }

        // Calculate gap distribution
        let gaps: Vec<u64> = keys.windows(2).map(|w| w[1] - w[0]).collect();

        if gaps.is_empty() {
            return 0.0;
        }

        let mean_gap = gaps.iter().sum::<u64>() as f64 / gaps.len() as f64;
        if mean_gap == 0.0 {
            return 0.0;
        }

        // Calculate coefficient of variation as entropy proxy
        let variance = gaps
            .iter()
            .map(|&g| {
                let diff = g as f64 - mean_gap;
                diff * diff
            })
            .sum::<f64>()
            / gaps.len() as f64;

        (variance.sqrt() / mean_gap).min(1.0)
    }

    /// Recommend index type based on statistics
    pub fn recommend_index_type(&self) -> IndexType {
        if self.count == 0 {
            return IndexType::None;
        }

        // High density + monotonic = learned index
        if self.is_monotonic && self.density > 0.5 {
            return IndexType::Learned;
        }

        // Low entropy (regular gaps) = learned index
        if self.entropy < 0.3 {
            return IndexType::Learned;
        }

        // Random keys = B-Tree
        if self.entropy > 0.7 {
            return IndexType::BTree;
        }

        // Default to learned for moderate cases
        IndexType::Learned
    }
}

/// Hybrid index that combines learned index with fallback
pub struct HybridIndex {
    /// Learned sparse index
    learned: LearnedSparseIndex,
    /// Sorted keys for fallback binary search
    keys: Vec<u64>,
    /// Key to position mapping for fallback
    key_map: BTreeMap<u64, usize>,
    /// Index type in use
    index_type: IndexType,
    /// Statistics
    stats: KeyStats,
}

impl HybridIndex {
    /// Build hybrid index from sorted keys
    pub fn build(keys: &[u64]) -> Self {
        let stats = KeyStats::analyze(keys);
        let index_type = stats.recommend_index_type();

        let learned = if index_type == IndexType::Learned {
            LearnedSparseIndex::build(keys)
        } else {
            LearnedSparseIndex::empty()
        };

        let key_map: BTreeMap<u64, usize> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();

        Self {
            learned,
            keys: keys.to_vec(),
            key_map,
            index_type,
            stats,
        }
    }

    /// Lookup key position
    ///
    /// Returns the exact position if found, or None.
    pub fn lookup(&self, key: u64) -> Option<usize> {
        match self.index_type {
            IndexType::Learned => self.lookup_learned(key),
            IndexType::BTree | IndexType::Hash => self.key_map.get(&key).copied(),
            IndexType::None => self.binary_search(key),
        }
    }

    /// Lookup using learned index
    fn lookup_learned(&self, key: u64) -> Option<usize> {
        match self.learned.lookup(key) {
            LookupResult::Exact(pos) => Some(pos),
            LookupResult::Range { low, high } => {
                // Binary search within the predicted range
                self.binary_search_range(key, low, high)
            }
            LookupResult::NotFound => None,
        }
    }

    /// Binary search within a range
    fn binary_search_range(&self, key: u64, low: usize, high: usize) -> Option<usize> {
        if low > high || high >= self.keys.len() {
            return None;
        }

        let slice = &self.keys[low..=high];
        match slice.binary_search(&key) {
            Ok(pos) => Some(low + pos),
            Err(_) => None,
        }
    }

    /// Full binary search
    fn binary_search(&self, key: u64) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }

    /// Range lookup: find all positions in [start, end]
    pub fn range_lookup(&self, start: u64, end: u64) -> Vec<usize> {
        // Find start position
        let start_pos = match self.keys.binary_search(&start) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        // Find end position
        let end_pos = match self.keys.binary_search(&end) {
            Ok(pos) => pos + 1,
            Err(pos) => pos,
        };

        (start_pos..end_pos.min(self.keys.len())).collect()
    }

    /// Get index statistics
    pub fn statistics(&self) -> &KeyStats {
        &self.stats
    }

    /// Get index type
    pub fn index_type(&self) -> IndexType {
        self.index_type
    }

    /// Check if learned index is efficient for this data
    pub fn is_efficient(&self) -> bool {
        self.learned.is_efficient()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.learned.memory_bytes()
            + self.keys.len() * std::mem::size_of::<u64>()
            + self.key_map.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<usize>())
    }
}

/// Index manager for multiple tables/columns
pub struct IndexManager {
    /// Indexes by table and column
    indexes: BTreeMap<(String, String), Arc<HybridIndex>>,
}

impl IndexManager {
    /// Create a new index manager
    pub fn new() -> Self {
        Self {
            indexes: BTreeMap::new(),
        }
    }

    /// Build or update index for a column
    pub fn build_index(&mut self, table: &str, column: &str, keys: &[u64]) {
        let index = HybridIndex::build(keys);
        self.indexes
            .insert((table.to_string(), column.to_string()), Arc::new(index));
    }

    /// Get index for a column
    pub fn get_index(&self, table: &str, column: &str) -> Option<Arc<HybridIndex>> {
        self.indexes
            .get(&(table.to_string(), column.to_string()))
            .cloned()
    }

    /// Remove index
    pub fn drop_index(&mut self, table: &str, column: &str) -> bool {
        self.indexes
            .remove(&(table.to_string(), column.to_string()))
            .is_some()
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<(&str, &str, IndexType)> {
        self.indexes
            .iter()
            .map(|((table, column), index)| (table.as_str(), column.as_str(), index.index_type()))
            .collect()
    }
}

impl Default for IndexManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Point lookup executor using learned index
pub struct PointLookupExecutor<'a, V> {
    /// Index to use
    index: &'a HybridIndex,
    /// Data array to fetch from
    data: &'a [V],
}

impl<'a, V> PointLookupExecutor<'a, V> {
    /// Create a new executor
    pub fn new(index: &'a HybridIndex, data: &'a [V]) -> Self {
        Self { index, data }
    }

    /// Execute point lookup
    pub fn execute(&self, key: u64) -> Option<&V> {
        self.index.lookup(key).and_then(|pos| self.data.get(pos))
    }

    /// Execute batch lookup
    pub fn execute_batch(&self, keys: &[u64]) -> Vec<Option<&V>> {
        keys.iter().map(|&k| self.execute(k)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_keys() {
        let keys: Vec<u64> = (1..=1000).collect();
        let stats = KeyStats::analyze(&keys);

        assert!(stats.is_monotonic);
        assert!(stats.density > 0.99);
        assert_eq!(stats.recommend_index_type(), IndexType::Learned);
    }

    #[test]
    fn test_timestamp_keys() {
        // Simulate timestamp keys (microseconds, ~1ms apart)
        let base = 1700000000000000u64; // ~2023
        let keys: Vec<u64> = (0..1000).map(|i| base + i * 1000).collect();

        let stats = KeyStats::analyze(&keys);
        assert!(stats.is_monotonic);
        assert!(stats.entropy < 0.1); // Regular gaps
        assert_eq!(stats.recommend_index_type(), IndexType::Learned);
    }

    #[test]
    fn test_hybrid_index_lookup() {
        let keys: Vec<u64> = (0..1000).map(|i| i * 10).collect();
        let index = HybridIndex::build(&keys);

        // Exact match
        assert_eq!(index.lookup(500), Some(50));
        assert_eq!(index.lookup(990), Some(99));

        // Not found
        assert_eq!(index.lookup(5), None);
        assert_eq!(index.lookup(10000), None);
    }

    #[test]
    fn test_range_lookup() {
        let keys: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let index = HybridIndex::build(&keys);

        // Range [100, 300] should include positions for 100, 110, ..., 300
        let positions = index.range_lookup(100, 300);
        assert_eq!(positions.len(), 21); // 100, 110, ..., 300 = 21 values
        assert_eq!(positions[0], 10); // Position of key 100
    }

    #[test]
    fn test_point_lookup_executor() {
        let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        let values = vec!["a", "b", "c", "d", "e"];
        let index = HybridIndex::build(&keys);

        let executor = PointLookupExecutor::new(&index, &values);

        assert_eq!(executor.execute(20), Some(&"b"));
        assert_eq!(executor.execute(50), Some(&"e"));
        assert_eq!(executor.execute(25), None);
    }

    #[test]
    fn test_index_manager() {
        let mut manager = IndexManager::new();

        let keys: Vec<u64> = (0..100).collect();
        manager.build_index("users", "id", &keys);

        let index = manager.get_index("users", "id").unwrap();
        assert_eq!(index.lookup(50), Some(50));

        let indexes = manager.list_indexes();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0], ("users", "id", IndexType::Learned));
    }
}

// =============================================================================
// Task 4: Piecewise Learned Index Enhancement
// =============================================================================

/// A segment in the piecewise linear index
#[derive(Debug, Clone)]
pub struct LinearSegment {
    /// Start key of this segment
    pub start_key: u64,
    /// End key of this segment (exclusive)
    pub end_key: u64,
    /// Slope: position increase per key increase
    pub slope: f64,
    /// Intercept: position at start_key
    pub intercept: f64,
    /// Maximum error in this segment
    pub max_error: usize,
}

impl LinearSegment {
    /// Predict position for a key within this segment
    pub fn predict(&self, key: u64) -> usize {
        if key < self.start_key {
            return 0;
        }
        let delta = (key - self.start_key) as f64;
        (self.intercept + delta * self.slope).round() as usize
    }

    /// Get search bounds accounting for error
    pub fn bounds(&self, key: u64, data_len: usize) -> (usize, usize) {
        let pred = self.predict(key);
        let low = pred.saturating_sub(self.max_error);
        let high = (pred + self.max_error).min(data_len.saturating_sub(1));
        (low, high)
    }
}

/// Piecewise Linear Index using dynamic programming for optimal segmentation
///
/// ## Algorithm
/// Uses dynamic programming to find the optimal set of linear segments
/// that minimize total error while keeping segments count bounded.
///
/// Cost function: `sum(segment_errors) + lambda * num_segments`
///
/// ## Performance
/// - Construction: O(n²) with DP, O(n) with greedy
/// - Lookup: O(log S + log ε) where S = segment count, ε = max error
#[derive(Debug)]
#[allow(dead_code)]
pub struct PiecewiseLearnedIndex {
    /// Sorted segments by start_key
    segments: Vec<LinearSegment>,
    /// Target maximum error per segment
    max_error_bound: usize,
    /// Total number of keys indexed
    total_keys: usize,
    /// Construction statistics
    stats: PiecewiseStats,
}

/// Statistics about the piecewise index
#[derive(Debug, Clone, Default)]
pub struct PiecewiseStats {
    /// Number of segments
    pub segment_count: usize,
    /// Average error across segments
    pub avg_error: f64,
    /// Maximum error across all segments
    pub max_error: usize,
    /// Compression ratio (keys / segments)
    pub compression_ratio: f64,
}

impl PiecewiseLearnedIndex {
    /// Build piecewise index with specified error bound
    ///
    /// Uses greedy algorithm for efficiency (O(n) instead of O(n²) DP)
    pub fn build(keys: &[u64], max_error: usize) -> Self {
        if keys.is_empty() {
            return Self {
                segments: Vec::new(),
                max_error_bound: max_error,
                total_keys: 0,
                stats: PiecewiseStats::default(),
            };
        }

        let segments = Self::build_greedy(keys, max_error);
        let stats = Self::compute_stats(&segments, keys.len());

        Self {
            segments,
            max_error_bound: max_error,
            total_keys: keys.len(),
            stats,
        }
    }

    /// Build using dynamic programming for optimal segmentation
    ///
    /// Slower O(n²) but produces optimal segments minimizing total error.
    pub fn build_optimal(keys: &[u64], max_segments: usize) -> Self {
        if keys.is_empty() {
            return Self {
                segments: Vec::new(),
                max_error_bound: 0,
                total_keys: 0,
                stats: PiecewiseStats::default(),
            };
        }

        let segments = Self::build_dp(keys, max_segments);
        let max_error = segments.iter().map(|s| s.max_error).max().unwrap_or(0);
        let stats = Self::compute_stats(&segments, keys.len());

        Self {
            segments,
            max_error_bound: max_error,
            total_keys: keys.len(),
            stats,
        }
    }

    /// Greedy segmentation algorithm
    fn build_greedy(keys: &[u64], max_error: usize) -> Vec<LinearSegment> {
        let mut segments = Vec::new();
        let mut start_idx = 0;

        while start_idx < keys.len() {
            // Find the longest segment starting at start_idx with error <= max_error
            let (end_idx, segment) = Self::find_longest_segment(keys, start_idx, max_error);
            segments.push(segment);
            start_idx = end_idx + 1;
        }

        segments
    }

    /// Find longest segment with bounded error
    fn find_longest_segment(
        keys: &[u64],
        start_idx: usize,
        max_error: usize,
    ) -> (usize, LinearSegment) {
        let start_key = keys[start_idx];
        let mut end_idx = start_idx;
        let mut best_slope = 0.0;
        let mut best_intercept = start_idx as f64;
        let mut best_error = 0;

        // Extend segment as far as possible
        for i in (start_idx + 1)..keys.len() {
            let (slope, intercept, error) = Self::fit_segment(keys, start_idx, i);
            if error <= max_error {
                end_idx = i;
                best_slope = slope;
                best_intercept = intercept;
                best_error = error;
            } else {
                break;
            }
        }

        let end_key = if end_idx + 1 < keys.len() {
            keys[end_idx + 1]
        } else {
            keys[end_idx].saturating_add(1)
        };

        (
            end_idx,
            LinearSegment {
                start_key,
                end_key,
                slope: best_slope,
                intercept: best_intercept,
                max_error: best_error,
            },
        )
    }

    /// Fit a linear segment and compute max error
    fn fit_segment(keys: &[u64], start: usize, end: usize) -> (f64, f64, usize) {
        if start == end {
            return (0.0, start as f64, 0);
        }

        let n = (end - start + 1) as f64;
        let _start_key = keys[start] as f64;

        // Simple linear regression
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in start..=end {
            let x = (keys[i] - keys[start]) as f64;
            let y = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let slope = if sum_xx * n - sum_x * sum_x != 0.0 {
            (sum_xy * n - sum_x * sum_y) / (sum_xx * n - sum_x * sum_x)
        } else {
            0.0
        };

        let intercept = (sum_y - slope * sum_x) / n;

        // Compute max error
        let mut max_error = 0usize;
        for i in start..=end {
            let x = (keys[i] - keys[start]) as f64;
            let predicted = (intercept + slope * x).round() as isize;
            let actual = i as isize;
            let error = (predicted - actual).unsigned_abs();
            max_error = max_error.max(error);
        }

        (slope, intercept, max_error)
    }

    /// Dynamic programming for optimal segmentation
    fn build_dp(keys: &[u64], max_segments: usize) -> Vec<LinearSegment> {
        let n = keys.len();
        if n == 0 || max_segments == 0 {
            return Vec::new();
        }

        // dp[i][k] = (min_cost, best_prev_idx) for first i keys with k segments
        let mut dp: Vec<Vec<(f64, usize)>> =
            vec![vec![(f64::INFINITY, 0); max_segments + 1]; n + 1];
        dp[0][0] = (0.0, 0);

        // Precompute segment costs
        let segment_cost = |start: usize, end: usize| -> f64 {
            let (_, _, error) = Self::fit_segment(keys, start, end);
            error as f64
        };

        // Fill DP table
        for i in 1..=n {
            for k in 1..=max_segments.min(i) {
                for j in 0..i {
                    if dp[j][k - 1].0 < f64::INFINITY {
                        let cost = dp[j][k - 1].0 + segment_cost(j, i - 1);
                        if cost < dp[i][k].0 {
                            dp[i][k] = (cost, j);
                        }
                    }
                }
            }
        }

        // Find best number of segments
        let mut best_k = 1;
        let mut best_cost = f64::INFINITY;
        for (k, dp_entry) in dp[n].iter().enumerate().take(max_segments + 1).skip(1) {
            // Cost function: segment_error + lambda * num_segments
            let lambda = 10.0; // Penalty for additional segments
            let cost = dp_entry.0 + lambda * k as f64;
            if cost < best_cost {
                best_cost = cost;
                best_k = k;
            }
        }

        // Backtrack to get segments
        let mut segments = Vec::new();
        let mut i = n;
        let mut k = best_k;

        while k > 0 && i > 0 {
            let j = dp[i][k].1;
            let (slope, intercept, max_error) = Self::fit_segment(keys, j, i - 1);

            let end_key = if i < n {
                keys[i]
            } else {
                keys[i - 1].saturating_add(1)
            };

            segments.push(LinearSegment {
                start_key: keys[j],
                end_key,
                slope,
                intercept,
                max_error,
            });

            i = j;
            k -= 1;
        }

        segments.reverse();
        segments
    }

    /// Compute statistics about the index
    fn compute_stats(segments: &[LinearSegment], total_keys: usize) -> PiecewiseStats {
        if segments.is_empty() {
            return PiecewiseStats::default();
        }

        let segment_count = segments.len();
        let total_error: usize = segments.iter().map(|s| s.max_error).sum();
        let max_error = segments.iter().map(|s| s.max_error).max().unwrap_or(0);
        let avg_error = total_error as f64 / segment_count as f64;
        let compression_ratio = total_keys as f64 / segment_count as f64;

        PiecewiseStats {
            segment_count,
            avg_error,
            max_error,
            compression_ratio,
        }
    }

    /// Look up key position
    pub fn lookup(&self, key: u64, data_len: usize) -> Option<(usize, usize)> {
        if self.segments.is_empty() {
            return None;
        }

        // Binary search for the correct segment
        let segment_idx = self.find_segment(key)?;
        let segment = &self.segments[segment_idx];

        Some(segment.bounds(key, data_len))
    }

    /// Find segment containing key
    fn find_segment(&self, key: u64) -> Option<usize> {
        if self.segments.is_empty() {
            return None;
        }

        // Binary search
        let idx = self.segments.partition_point(|s| s.end_key <= key);
        if idx > 0 && idx <= self.segments.len() {
            let seg = &self.segments[idx - 1];
            if key >= seg.start_key && key < seg.end_key {
                return Some(idx - 1);
            }
        }

        if idx < self.segments.len() {
            let seg = &self.segments[idx];
            if key >= seg.start_key && key < seg.end_key {
                return Some(idx);
            }
        }

        None
    }

    /// Get statistics
    pub fn statistics(&self) -> &PiecewiseStats {
        &self.stats
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.segments.len() * std::mem::size_of::<LinearSegment>()
    }

    /// Number of segments
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod piecewise_tests {
    use super::*;

    #[test]
    fn test_piecewise_sequential() {
        let keys: Vec<u64> = (0..1000).collect();
        let index = PiecewiseLearnedIndex::build(&keys, 2);

        // Should need very few segments for sequential data
        assert!(index.segment_count() <= 5);
        assert!(index.stats.avg_error <= 2.0);

        // Lookup should work
        let (low, high) = index.lookup(500, 1000).unwrap();
        assert!(low <= 500 && 500 <= high);
    }

    #[test]
    fn test_piecewise_timestamp() {
        // Simulate timestamps with ~1ms intervals + jitter
        let base = 1700000000000000u64;
        let keys: Vec<u64> = (0..1000).map(|i| base + i * 1000 + (i % 10)).collect();

        let index = PiecewiseLearnedIndex::build(&keys, 5);

        // Should handle slight jitter
        assert!(index.stats.max_error <= 5);

        let (low, high) = index.lookup(base + 500000, 1000).unwrap();
        assert!(high - low <= 10); // Tight bounds
    }

    #[test]
    fn test_piecewise_optimal() {
        let keys: Vec<u64> = (0..100).map(|i| i * i).collect(); // Quadratic

        let index = PiecewiseLearnedIndex::build_optimal(&keys, 10);

        // Should have at least one segment
        assert!(index.segment_count() >= 1);

        // All lookups should work
        for i in 0..100 {
            let (low, high) = index.lookup(i * i, 100).unwrap();
            assert!(
                low <= i as usize && i as usize <= high,
                "Key {} (i={}): expected bounds to contain {}, got ({}, {})",
                i * i,
                i,
                i,
                low,
                high
            );
        }
    }

    #[test]
    fn test_piecewise_memory() {
        let keys: Vec<u64> = (0..10000).collect();
        let index = PiecewiseLearnedIndex::build(&keys, 10);

        // Should use much less memory than storing all keys
        let key_memory = keys.len() * std::mem::size_of::<u64>();
        assert!(index.memory_bytes() < key_memory / 10);

        println!(
            "Compression: {} keys -> {} segments ({:.1}x)",
            keys.len(),
            index.segment_count(),
            index.stats.compression_ratio
        );
    }
}

// =============================================================================
// Task 9: Recursive Model Index (RMI) with Delta Updates
// =============================================================================

/// Two-level Recursive Model Index for O(1) expected lookups
///
/// ## Architecture
/// ```text
/// Level 1 (Root):   [Linear Model] → routes to leaf model
/// Level 2 (Leaves): [PLM 0] [PLM 1] ... [PLM N] → position predictions
/// ```
///
/// ## Performance
/// - Space: O(M × params) where M = number of models
/// - Lookup: O(1) average, O(log ε) for binary search in error range
#[derive(Debug)]
pub struct RecursiveModelIndex {
    /// Root model: maps normalized key → leaf model index
    root_slope: f64,
    root_intercept: f64,
    /// Leaf models (piecewise linear within each bucket)
    leaves: Vec<PiecewiseLearnedIndex>,
    /// Min key for normalization
    min_key: u64,
    /// Max key for normalization
    max_key: u64,
    /// Key range as f64
    key_range: f64,
    /// Total keys
    num_keys: usize,
    /// Global max error
    max_error: usize,
}

impl RecursiveModelIndex {
    /// Build a 2-level RMI
    ///
    /// # Arguments
    /// * `keys` - Sorted keys
    /// * `num_leaves` - Number of leaf models (typically √N)
    /// * `leaf_max_error` - Max error per leaf segment
    pub fn build(keys: &[u64], num_leaves: usize, leaf_max_error: usize) -> Self {
        let n = keys.len();
        if n == 0 {
            return Self {
                root_slope: 0.0,
                root_intercept: 0.0,
                leaves: Vec::new(),
                min_key: 0,
                max_key: 0,
                key_range: 0.0,
                num_keys: 0,
                max_error: 0,
            };
        }

        let min_key = keys[0];
        let max_key = keys[n - 1];
        let key_range = if max_key == min_key {
            1.0
        } else {
            (max_key - min_key) as f64
        };
        let num_leaves = num_leaves.min(n).max(1);

        // Fit root model: normalized_key → bucket_index
        let bucket_size = n.div_ceil(num_leaves);
        let root_slope = num_leaves as f64; // Maps [0,1] to [0, num_leaves]
        let root_intercept = 0.0;

        // Build leaf models for each bucket
        let mut leaves = Vec::with_capacity(num_leaves);
        let mut global_max_error = 0usize;

        for bucket_idx in 0..num_leaves {
            let start = bucket_idx * bucket_size;
            let end = ((bucket_idx + 1) * bucket_size).min(n);

            if start < n {
                let bucket_keys: Vec<u64> = keys[start..end].to_vec();
                let leaf = PiecewiseLearnedIndex::build(&bucket_keys, leaf_max_error);
                global_max_error = global_max_error.max(leaf.stats.max_error);
                leaves.push(leaf);
            }
        }

        Self {
            root_slope,
            root_intercept,
            leaves,
            min_key,
            max_key,
            key_range,
            num_keys: n,
            max_error: global_max_error,
        }
    }

    /// Look up position bounds for a key
    pub fn lookup(&self, key: u64, data_len: usize) -> Option<(usize, usize)> {
        if self.num_keys == 0 || key < self.min_key || key > self.max_key {
            return None;
        }

        // Normalize key to [0, 1]
        let normalized = (key - self.min_key) as f64 / self.key_range;

        // Route to leaf
        let leaf_idx_f = self.root_slope * normalized + self.root_intercept;
        let leaf_idx = (leaf_idx_f as usize).min(self.leaves.len().saturating_sub(1));

        // Query leaf model
        if let Some(leaf) = self.leaves.get(leaf_idx) {
            // Leaf returns relative position within bucket
            if let Some((rel_low, rel_high)) = leaf.lookup(key, data_len) {
                // Convert to absolute position
                let bucket_size = self.num_keys.div_ceil(self.leaves.len());
                let bucket_start = leaf_idx * bucket_size;
                let abs_low = bucket_start + rel_low;
                let abs_high = (bucket_start + rel_high).min(data_len.saturating_sub(1));
                return Some((abs_low, abs_high));
            }
        }

        // Fallback: return full range
        Some((0, data_len.saturating_sub(1)))
    }

    /// Get space usage in bytes
    pub fn size_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let leaves: usize = self.leaves.iter().map(|l| l.memory_bytes()).sum();
        base + leaves
    }

    /// Get number of leaf models
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Get total segment count across all leaves
    pub fn total_segments(&self) -> usize {
        self.leaves.iter().map(|l| l.segment_count()).sum()
    }
}

/// Delta index for online updates without full rebuild
///
/// Maintains pending inserts/deletes in a B-tree structure,
/// merging with static learned index during lookup.
#[derive(Debug)]
pub struct DeltaIndex {
    /// Inserted keys (key → tombstone flag)
    entries: BTreeMap<u64, bool>,
    /// Insert count since last rebuild
    insert_count: usize,
    /// Delete count since last rebuild
    delete_count: usize,
    /// Rebuild threshold as fraction of static size
    rebuild_threshold: f64,
}

impl DeltaIndex {
    /// Create new delta index
    pub fn new(rebuild_threshold: f64) -> Self {
        Self {
            entries: BTreeMap::new(),
            insert_count: 0,
            delete_count: 0,
            rebuild_threshold,
        }
    }

    /// Insert a key
    pub fn insert(&mut self, key: u64) {
        if let Some(deleted) = self.entries.get_mut(&key) {
            // Resurrect deleted key
            if *deleted {
                *deleted = false;
                self.delete_count = self.delete_count.saturating_sub(1);
            }
        } else {
            self.entries.insert(key, false);
            self.insert_count += 1;
        }
    }

    /// Delete a key (tombstone)
    pub fn delete(&mut self, key: u64) {
        self.entries.insert(key, true);
        self.delete_count += 1;
    }

    /// Check if key is in delta
    pub fn contains(&self, key: u64) -> Option<bool> {
        self.entries.get(&key).copied()
    }

    /// Check if rebuild is needed
    pub fn needs_rebuild(&self, static_size: usize) -> bool {
        if static_size == 0 {
            return self.entries.len() > 100;
        }
        let delta_size = self.insert_count + self.delete_count;
        (delta_size as f64 / static_size as f64) > self.rebuild_threshold
    }

    /// Get all live keys (for rebuild)
    pub fn live_keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries
            .iter()
            .filter(|(_, deleted)| !**deleted)
            .map(|(k, _)| *k)
    }

    /// Get all deleted keys
    pub fn deleted_keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries
            .iter()
            .filter(|(_, deleted)| **deleted)
            .map(|(k, _)| *k)
    }

    /// Clear after rebuild
    pub fn clear(&mut self) {
        self.entries.clear();
        self.insert_count = 0;
        self.delete_count = 0;
    }

    /// Size of delta
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is delta empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Hybrid RMI with delta updates and B-tree fallback
#[derive(Debug)]
pub struct HybridRMI {
    /// Static RMI structure
    rmi: RecursiveModelIndex,
    /// Delta index for updates
    delta: DeltaIndex,
    /// Sorted keys for binary search
    keys: Vec<u64>,
    /// B-tree fallback for high-error keys
    btree_fallback: BTreeMap<u64, usize>,
    /// Stats
    stats: HybridRMIStats,
}

/// Statistics for HybridRMI
#[derive(Debug, Clone, Default)]
pub struct HybridRMIStats {
    /// Lookups via RMI
    pub rmi_lookups: u64,
    /// Lookups via B-tree fallback
    pub btree_lookups: u64,
    /// Lookups in delta
    pub delta_lookups: u64,
    /// Number of rebuilds
    pub rebuilds: u64,
}

impl HybridRMI {
    /// Build hybrid RMI
    pub fn build(
        keys: Vec<u64>,
        num_leaves: usize,
        leaf_max_error: usize,
        rebuild_threshold: f64,
    ) -> Self {
        let rmi = RecursiveModelIndex::build(&keys, num_leaves, leaf_max_error);

        // Find overflow keys for B-tree fallback
        let _overflow_threshold = leaf_max_error * 3;
        let mut btree_fallback = BTreeMap::new();

        for (pos, &key) in keys.iter().enumerate() {
            if let Some((low, high)) = rmi.lookup(key, keys.len())
                && (pos < low || pos > high)
            {
                btree_fallback.insert(key, pos);
            }
        }

        Self {
            rmi,
            delta: DeltaIndex::new(rebuild_threshold),
            keys,
            btree_fallback,
            stats: HybridRMIStats::default(),
        }
    }

    /// Look up a key
    pub fn lookup(&mut self, key: u64) -> Option<usize> {
        // 1. Check delta first
        if let Some(deleted) = self.delta.contains(key) {
            self.stats.delta_lookups += 1;
            if deleted {
                return None; // Deleted
            }
            // Key in delta but not deleted - need to find position
            // This is a recent insert, position unknown
            return None;
        }

        // 2. Check B-tree fallback
        if let Some(&pos) = self.btree_fallback.get(&key) {
            self.stats.btree_lookups += 1;
            return Some(pos);
        }

        // 3. Use RMI
        self.stats.rmi_lookups += 1;
        if let Some((low, high)) = self.rmi.lookup(key, self.keys.len()) {
            // Binary search in predicted range
            let range = &self.keys[low..=high.min(self.keys.len().saturating_sub(1))];
            if let Ok(idx) = range.binary_search(&key) {
                return Some(low + idx);
            }
        }

        // 4. Full binary search fallback
        self.keys.binary_search(&key).ok()
    }

    /// Insert a key (goes to delta)
    pub fn insert(&mut self, key: u64) {
        self.delta.insert(key);
        if self.delta.needs_rebuild(self.keys.len()) {
            self.rebuild();
        }
    }

    /// Delete a key
    pub fn delete(&mut self, key: u64) {
        self.delta.delete(key);
    }

    /// Rebuild the index
    pub fn rebuild(&mut self) {
        // Merge delta into keys
        let deleted: HashSet<u64> = self.delta.deleted_keys().collect();

        let mut new_keys: Vec<u64> = self
            .keys
            .iter()
            .filter(|&k| !deleted.contains(k))
            .copied()
            .collect();

        new_keys.extend(self.delta.live_keys());
        new_keys.sort_unstable();
        new_keys.dedup();

        // Rebuild RMI
        let n = new_keys.len();
        let num_leaves = ((n as f64).sqrt().ceil() as usize).max(1);
        let leaf_max_error = self.rmi.max_error.max(10);

        let new_rmi = RecursiveModelIndex::build(&new_keys, num_leaves, leaf_max_error);

        // Rebuild B-tree fallback
        let mut new_btree = BTreeMap::new();
        for (pos, &key) in new_keys.iter().enumerate() {
            if let Some((low, high)) = new_rmi.lookup(key, new_keys.len())
                && (pos < low || pos > high)
            {
                new_btree.insert(key, pos);
            }
        }

        self.rmi = new_rmi;
        self.keys = new_keys;
        self.btree_fallback = new_btree;
        self.delta.clear();
        self.stats.rebuilds += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> &HybridRMIStats {
        &self.stats
    }

    /// Get number of keys
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Space usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.rmi.size_bytes()
            + self.keys.len() * std::mem::size_of::<u64>()
            + self.delta.len() * std::mem::size_of::<(u64, bool)>()
            + self.btree_fallback.len()
                * (std::mem::size_of::<u64>() + std::mem::size_of::<usize>())
    }
}

#[cfg(test)]
mod rmi_tests {
    use super::*;

    #[test]
    fn test_rmi_build() {
        let keys: Vec<u64> = (0..10000).collect();
        let rmi = RecursiveModelIndex::build(&keys, 100, 10);

        assert_eq!(rmi.num_leaves(), 100);
        assert!(rmi.max_error <= 10);

        // Test lookups
        for i in (0..10000).step_by(100) {
            if let Some((low, high)) = rmi.lookup(i, 10000) {
                assert!(
                    low <= i as usize && high >= i as usize,
                    "Key {}: bounds ({}, {}) don't contain position",
                    i,
                    low,
                    high
                );
            }
        }
    }

    #[test]
    fn test_delta_index() {
        let mut delta = DeltaIndex::new(0.1);

        delta.insert(100);
        delta.insert(200);
        delta.delete(150);

        assert_eq!(delta.contains(100), Some(false));
        assert_eq!(delta.contains(150), Some(true)); // deleted
        assert_eq!(delta.contains(300), None);

        // Resurrect deleted key
        delta.insert(150);
        assert_eq!(delta.contains(150), Some(false));
    }

    #[test]
    fn test_hybrid_rmi_lookup() {
        let keys: Vec<u64> = (0..1000).step_by(2).collect();
        let mut rmi = HybridRMI::build(keys, 10, 5, 0.1);

        // Find existing key
        assert_eq!(rmi.lookup(100), Some(50));
        assert_eq!(rmi.lookup(500), Some(250));

        // Non-existing
        assert!(rmi.lookup(101).is_none());
    }

    #[test]
    fn test_hybrid_rmi_updates() {
        let keys: Vec<u64> = (0..100).collect();
        let mut rmi = HybridRMI::build(keys, 5, 5, 0.5);

        // Delete a key - next lookup should return None
        rmi.delete(50);
        assert!(rmi.lookup(50).is_none());

        // Insert a new key (goes to delta, won't be in RMI yet)
        rmi.insert(200);

        // Verify delta has something
        assert!(!rmi.delta.is_empty());
    }

    #[test]
    fn test_hybrid_rmi_rebuild() {
        let keys: Vec<u64> = (0..100).collect();
        let mut rmi = HybridRMI::build(keys, 5, 5, 0.05);

        let initial_len = rmi.len();

        // Insert enough to trigger rebuild
        for i in 100..115 {
            rmi.insert(i);
        }

        // Rebuild should have happened
        assert!(rmi.stats().rebuilds > 0);
        // Length should have increased
        assert!(rmi.len() > initial_len);
    }

    #[test]
    fn test_rmi_space_efficiency() {
        let n = 100_000usize;
        let keys: Vec<u64> = (0..n as u64).collect();
        let rmi = RecursiveModelIndex::build(&keys, 100, 50);

        let rmi_size = rmi.size_bytes();
        let raw_size = n * std::mem::size_of::<u64>();

        println!(
            "RMI size: {} bytes, Raw keys: {} bytes, Ratio: {:.2}x",
            rmi_size,
            raw_size,
            raw_size as f64 / rmi_size as f64
        );

        // RMI structure should be much smaller than raw keys
        assert!(
            rmi_size < raw_size / 10,
            "RMI size {} should be < 10% of raw size {}",
            rmi_size,
            raw_size
        );
    }
}
