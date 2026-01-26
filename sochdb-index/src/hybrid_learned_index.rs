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

//! Hybrid Learned Index with B-Tree Fallback
//!
//! This module implements a PGM-Index style learned index that:
//! - Uses piecewise linear models for O(1) prediction
//! - Falls back to B-tree for out-of-distribution keys and updates
//! - Adapts model parameters based on error statistics
//! - Provides 32x space reduction vs traditional B-trees
//!
//! ## Algorithm Overview
//!
//! For sorted keys K, we build a piecewise linear approximation:
//! - Each segment covers a range of keys with error ≤ ε
//! - Position prediction: pos = slope × key + intercept
//! - Binary search in [pos - ε, pos + ε] for final answer
//!
//! ## Space Complexity
//!
//! Traditional B-tree (fanout 128):
//! - Nodes = N/127, Space per node = 2KB
//! - Total = N/127 × 2KB = 16N bytes
//!
//! Learned Index (ε = 64):
//! - Segments = N/64, Space per segment = 32 bytes
//! - Total = N/64 × 32 = 0.5N bytes
//! - **32× space reduction**

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Key type (for simplicity, using u64)
pub type Key = u64;

/// Position type (array index)
pub type Position = u64;

/// A single linear segment of the model
#[derive(Debug, Clone)]
pub struct LinearSegment {
    /// First key in segment
    pub start_key: Key,
    /// Slope of linear model
    pub slope: f64,
    /// Intercept of linear model
    pub intercept: f64,
    /// Maximum error in this segment
    pub max_error: usize,
}

impl LinearSegment {
    /// Predict position for a key
    #[inline]
    pub fn predict(&self, key: Key) -> usize {
        let pos = self.slope * (key as f64) + self.intercept;
        pos.max(0.0) as usize
    }

    /// Get search bounds [lo, hi] for a key
    #[inline]
    pub fn search_bounds(&self, key: Key, data_len: usize) -> (usize, usize) {
        let predicted = self.predict(key);
        let lo = predicted.saturating_sub(self.max_error);
        let hi = (predicted + self.max_error).min(data_len.saturating_sub(1));
        (lo, hi)
    }
}

/// Statistics for adaptive model updates
#[derive(Debug, Default)]
pub struct IndexStats {
    /// Total lookups
    pub lookups: AtomicU64,
    /// Lookups that hit the learned model
    pub model_hits: AtomicU64,
    /// Lookups that used B-tree fallback
    pub btree_hits: AtomicU64,
    /// Total prediction error
    pub total_error: AtomicU64,
    /// Maximum observed error
    pub max_error: AtomicU64,
    /// Number of rebuilds
    pub rebuilds: AtomicU64,
}

/// Configuration for the hybrid index
#[derive(Debug, Clone)]
pub struct HybridIndexConfig {
    /// Maximum error per segment
    pub epsilon: usize,
    /// Threshold for switching to B-tree fallback
    pub error_threshold: usize,
    /// Number of inserts before merging fallback
    pub merge_threshold: usize,
    /// Minimum keys to justify learning
    pub min_keys_for_model: usize,
}

impl Default for HybridIndexConfig {
    fn default() -> Self {
        Self {
            epsilon: 64,
            error_threshold: 128,
            merge_threshold: 10000,
            min_keys_for_model: 100,
        }
    }
}

/// Hybrid Learned Index with B-Tree Fallback
pub struct HybridLearnedIndex {
    /// Piecewise linear model
    segments: Vec<LinearSegment>,
    /// Segment index for O(log S) segment lookup
    segment_index: BTreeMap<Key, usize>,
    /// Fallback B-tree for recent inserts
    btree_fallback: BTreeMap<Key, Position>,
    /// Configuration
    config: HybridIndexConfig,
    /// Statistics
    stats: IndexStats,
    /// Data length (for bounds checking)
    data_len: usize,
}

impl Default for HybridLearnedIndex {
    fn default() -> Self {
        Self::new(HybridIndexConfig::default())
    }
}

impl HybridLearnedIndex {
    /// Create a new empty hybrid index
    pub fn new(config: HybridIndexConfig) -> Self {
        Self {
            segments: Vec::new(),
            segment_index: BTreeMap::new(),
            btree_fallback: BTreeMap::new(),
            config,
            stats: IndexStats::default(),
            data_len: 0,
        }
    }

    /// Build index from sorted keys
    pub fn build(keys: &[Key], config: HybridIndexConfig) -> Self {
        if keys.len() < config.min_keys_for_model {
            // Too few keys - use B-tree only
            let btree: BTreeMap<Key, Position> = keys
                .iter()
                .enumerate()
                .map(|(i, &k)| (k, i as Position))
                .collect();

            return Self {
                segments: Vec::new(),
                segment_index: BTreeMap::new(),
                btree_fallback: btree,
                config,
                stats: IndexStats::default(),
                data_len: keys.len(),
            };
        }

        let segments = Self::build_segments(keys, config.epsilon);
        let segment_index = segments
            .iter()
            .enumerate()
            .map(|(idx, seg)| (seg.start_key, idx))
            .collect();

        Self {
            segments,
            segment_index,
            btree_fallback: BTreeMap::new(),
            config,
            stats: IndexStats::default(),
            data_len: keys.len(),
        }
    }

    /// Build optimal piecewise linear segments
    fn build_segments(keys: &[Key], epsilon: usize) -> Vec<LinearSegment> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut i = 0;

        while i < keys.len() {
            // Find longest segment starting at i with error ≤ epsilon
            let (segment, end) = Self::optimal_segment(&keys[i..], i, epsilon);
            segments.push(segment);
            i += end;
        }

        segments
    }

    /// Find optimal segment using simple greedy algorithm
    /// (Full PGM uses convex hull, this is a simpler approximation)
    fn optimal_segment(keys: &[Key], start_pos: usize, epsilon: usize) -> (LinearSegment, usize) {
        if keys.len() == 1 {
            return (
                LinearSegment {
                    start_key: keys[0],
                    slope: 1.0,
                    intercept: start_pos as f64,
                    max_error: 0,
                },
                1,
            );
        }

        // Try progressively longer segments
        let mut best_end = 1;
        let mut best_segment = LinearSegment {
            start_key: keys[0],
            slope: 1.0,
            intercept: start_pos as f64,
            max_error: 0,
        };

        for end in 2..=keys.len() {
            // Fit linear model to keys[0..end]
            let segment = Self::fit_linear(&keys[..end], start_pos);

            // Check max error
            let max_error = Self::compute_max_error(&keys[..end], start_pos, &segment);

            if max_error <= epsilon {
                best_end = end;
                best_segment = LinearSegment {
                    start_key: keys[0],
                    slope: segment.slope,
                    intercept: segment.intercept,
                    max_error,
                };
            } else {
                break;
            }
        }

        (best_segment, best_end)
    }

    /// Fit a linear model using least squares
    fn fit_linear(keys: &[Key], start_pos: usize) -> LinearSegment {
        let n = keys.len() as f64;

        // Simple linear regression
        let sum_x: f64 = keys.iter().map(|&k| k as f64).sum();
        let sum_y: f64 = (start_pos..start_pos + keys.len()).map(|i| i as f64).sum();
        let sum_xx: f64 = keys.iter().map(|&k| (k as f64) * (k as f64)).sum();
        let sum_xy: f64 = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k as f64) * ((start_pos + i) as f64))
            .sum();

        let denom = n * sum_xx - sum_x * sum_x;

        let (slope, intercept) = if denom.abs() < 1e-10 {
            // Degenerate case - all same key
            (0.0, start_pos as f64)
        } else {
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n;
            (slope, intercept)
        };

        LinearSegment {
            start_key: keys[0],
            slope,
            intercept,
            max_error: 0,
        }
    }

    /// Compute maximum error for a segment
    fn compute_max_error(keys: &[Key], start_pos: usize, segment: &LinearSegment) -> usize {
        let mut max_error = 0usize;

        for (i, &key) in keys.iter().enumerate() {
            let actual_pos = start_pos + i;
            let predicted = segment.predict(key);
            let error = (actual_pos as isize - predicted as isize).unsigned_abs();
            max_error = max_error.max(error);
        }

        max_error
    }

    /// Look up a key, returns position if found
    pub fn lookup(&self, key: Key, data: &[Key]) -> Option<usize> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);

        // Check B-tree fallback first (recent inserts)
        if let Some(&pos) = self.btree_fallback.get(&key) {
            self.stats.btree_hits.fetch_add(1, Ordering::Relaxed);
            return Some(pos as usize);
        }

        // No segments - fall back to binary search
        if self.segments.is_empty() {
            return data.binary_search(&key).ok();
        }

        // Find segment for this key
        let segment_idx = self.find_segment(key)?;
        let segment = &self.segments[segment_idx];

        // Get search bounds
        let (lo, hi) = segment.search_bounds(key, data.len());

        // Binary search within bounds
        let result = if lo <= hi && hi < data.len() {
            data[lo..=hi].binary_search(&key).ok().map(|i| i + lo)
        } else {
            // Bounds check failed, fall back to full search
            data.binary_search(&key).ok()
        };

        if let Some(pos) = result {
            self.stats.model_hits.fetch_add(1, Ordering::Relaxed);

            // Track error
            let predicted = segment.predict(key);
            let error = (pos as isize - predicted as isize).unsigned_abs() as u64;
            self.stats.total_error.fetch_add(error, Ordering::Relaxed);
            self.stats.max_error.fetch_max(error, Ordering::Relaxed);
        }

        result
    }

    /// Find the segment for a key
    fn find_segment(&self, key: Key) -> Option<usize> {
        // Find largest segment start_key ≤ key
        self.segment_index
            .range(..=key)
            .next_back()
            .map(|(_, &idx)| idx)
    }

    /// Insert a new key-position pair
    pub fn insert(&mut self, key: Key, position: Position) {
        self.btree_fallback.insert(key, position);
        self.data_len = self.data_len.max(position as usize + 1);

        // Check if we need to merge fallback into learned model
        if self.btree_fallback.len() >= self.config.merge_threshold {
            self.merge_fallback();
        }
    }

    /// Merge B-tree fallback into the learned model
    fn merge_fallback(&mut self) {
        if self.btree_fallback.is_empty() {
            return;
        }

        self.stats.rebuilds.fetch_add(1, Ordering::Relaxed);

        // Collect all keys (from model + fallback)
        let mut all_keys: Vec<Key> = Vec::with_capacity(self.data_len + self.btree_fallback.len());

        // This is a simplified merge - in production you'd iterate through
        // the original data structure too
        all_keys.extend(self.btree_fallback.keys());
        all_keys.sort();

        // Rebuild segments
        self.segments = Self::build_segments(&all_keys, self.config.epsilon);
        self.segment_index = self
            .segments
            .iter()
            .enumerate()
            .map(|(idx, seg)| (seg.start_key, idx))
            .collect();

        // Clear fallback
        self.btree_fallback.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Get number of segments
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get size of fallback B-tree
    pub fn fallback_size(&self) -> usize {
        self.btree_fallback.len()
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let segment_size = self.segments.len() * std::mem::size_of::<LinearSegment>();
        let index_size =
            self.segment_index.len() * (std::mem::size_of::<Key>() + std::mem::size_of::<usize>());
        let btree_size = self.btree_fallback.len()
            * (std::mem::size_of::<Key>() + std::mem::size_of::<Position>());
        segment_size + index_size + btree_size
    }

    /// Estimate B-tree equivalent memory usage
    pub fn btree_equivalent_memory(&self) -> usize {
        // B-tree node size estimate: 64 entries × 16 bytes = 1KB
        // Number of nodes ≈ data_len / 64
        let nodes = self.data_len.div_ceil(64);
        nodes * 1024
    }

    /// Get space savings ratio
    pub fn space_savings_ratio(&self) -> f64 {
        let learned = self.memory_usage();
        let btree = self.btree_equivalent_memory();
        if learned == 0 {
            1.0
        } else {
            btree as f64 / learned as f64
        }
    }

    /// Get average error from stats
    pub fn average_error(&self) -> f64 {
        let lookups = self.stats.model_hits.load(Ordering::Relaxed);
        if lookups == 0 {
            0.0
        } else {
            self.stats.total_error.load(Ordering::Relaxed) as f64 / lookups as f64
        }
    }
}

/// Builder for hybrid learned index
pub struct HybridIndexBuilder {
    keys: Vec<Key>,
    config: HybridIndexConfig,
}

impl Default for HybridIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridIndexBuilder {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            config: HybridIndexConfig::default(),
        }
    }

    pub fn with_config(mut self, config: HybridIndexConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_key(&mut self, key: Key) -> &mut Self {
        self.keys.push(key);
        self
    }

    pub fn add_keys(&mut self, keys: &[Key]) -> &mut Self {
        self.keys.extend_from_slice(keys);
        self
    }

    pub fn build(mut self) -> HybridLearnedIndex {
        self.keys.sort();
        HybridLearnedIndex::build(&self.keys, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_segment_predict() {
        let segment = LinearSegment {
            start_key: 0,
            slope: 1.0,
            intercept: 0.0,
            max_error: 0,
        };

        assert_eq!(segment.predict(0), 0);
        assert_eq!(segment.predict(100), 100);
        assert_eq!(segment.predict(1000), 1000);
    }

    #[test]
    fn test_linear_segment_bounds() {
        let segment = LinearSegment {
            start_key: 0,
            slope: 1.0,
            intercept: 0.0,
            max_error: 10,
        };

        let (lo, hi) = segment.search_bounds(50, 100);
        assert_eq!(lo, 40);
        assert_eq!(hi, 60);

        // Test bounds at edges
        let (lo, hi) = segment.search_bounds(5, 100);
        assert_eq!(lo, 0);
        assert_eq!(hi, 15);
    }

    #[test]
    fn test_build_empty() {
        let index = HybridLearnedIndex::build(&[], HybridIndexConfig::default());
        assert_eq!(index.segment_count(), 0);
    }

    #[test]
    fn test_build_single() {
        let keys = vec![42];
        let index = HybridLearnedIndex::build(&keys, HybridIndexConfig::default());
        // Few keys go into B-tree
        assert!(index.fallback_size() > 0 || index.segment_count() > 0);
    }

    #[test]
    fn test_build_sequential() {
        let keys: Vec<Key> = (0..1000).collect();
        let config = HybridIndexConfig {
            epsilon: 1,
            min_keys_for_model: 10,
            ..Default::default()
        };
        let index = HybridLearnedIndex::build(&keys, config);

        // Sequential keys should need few segments (ideally 1 with ε=1)
        assert!(index.segment_count() <= 10);

        // Look up some keys
        for &key in &[0, 100, 500, 999] {
            let pos = index.lookup(key, &keys);
            assert_eq!(pos, Some(key as usize));
        }
    }

    #[test]
    fn test_build_random() {
        let mut keys: Vec<Key> = (0..1000).map(|i| i * 7 + 13).collect();
        keys.sort();

        let config = HybridIndexConfig {
            epsilon: 64,
            min_keys_for_model: 10,
            ..Default::default()
        };
        let index = HybridLearnedIndex::build(&keys, config);

        // Should have reasonable number of segments
        assert!(index.segment_count() > 0);
        assert!(index.segment_count() < keys.len() / 10);

        // All keys should be findable
        for (i, &key) in keys.iter().enumerate() {
            let pos = index.lookup(key, &keys);
            assert_eq!(pos, Some(i), "Failed to find key {} at position {}", key, i);
        }
    }

    #[test]
    fn test_lookup_missing() {
        let keys: Vec<Key> = (0..100).map(|i| i * 2).collect(); // Even numbers
        let index = HybridLearnedIndex::build(&keys, HybridIndexConfig::default());

        // Odd numbers should not be found
        for i in (1..199).step_by(2) {
            assert_eq!(index.lookup(i, &keys), None);
        }
    }

    #[test]
    fn test_insert_and_lookup() {
        let keys: Vec<Key> = (0..100).collect();
        let mut index = HybridLearnedIndex::build(&keys, HybridIndexConfig::default());

        // Insert new key
        index.insert(1000, 100);

        // Lookup should find it in fallback
        assert!(index.fallback_size() > 0);
    }

    #[test]
    fn test_merge_threshold() {
        let keys: Vec<Key> = (0..100).collect();
        let config = HybridIndexConfig {
            merge_threshold: 50,
            min_keys_for_model: 10,
            ..Default::default()
        };
        let mut index = HybridLearnedIndex::build(&keys, config);

        // Insert enough to trigger merge
        for i in 200..260 {
            index.insert(i, i - 100);
        }

        // Should have triggered a rebuild
        assert!(index.stats().rebuilds.load(Ordering::Relaxed) > 0 || index.fallback_size() < 60);
    }

    #[test]
    fn test_stats() {
        let keys: Vec<Key> = (0..1000).collect();
        let config = HybridIndexConfig {
            epsilon: 64,
            min_keys_for_model: 10,
            ..Default::default()
        };
        let index = HybridLearnedIndex::build(&keys, config);

        // Do some lookups
        for &key in &[0, 100, 500, 999] {
            index.lookup(key, &keys);
        }

        let stats = index.stats();
        assert_eq!(stats.lookups.load(Ordering::Relaxed), 4);
        assert!(stats.model_hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_memory_savings() {
        let keys: Vec<Key> = (0..100000).collect();
        let config = HybridIndexConfig {
            epsilon: 64,
            min_keys_for_model: 10,
            ..Default::default()
        };
        let index = HybridLearnedIndex::build(&keys, config);

        let ratio = index.space_savings_ratio();
        println!("Space savings ratio: {:.1}x", ratio);

        // Should have significant space savings for sequential data
        assert!(ratio > 1.0, "Expected space savings, got ratio {}", ratio);
    }

    #[test]
    fn test_builder() {
        let mut builder = HybridIndexBuilder::new().with_config(HybridIndexConfig {
            epsilon: 32,
            min_keys_for_model: 5,
            ..Default::default()
        });

        builder.add_keys(&[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);

        let index = builder.build();

        let data = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        assert_eq!(index.lookup(7, &data), Some(3));
        assert_eq!(index.lookup(19, &data), Some(9));
    }

    #[test]
    fn test_average_error() {
        let keys: Vec<Key> = (0..1000).collect();
        let config = HybridIndexConfig {
            epsilon: 1,
            min_keys_for_model: 10,
            ..Default::default()
        };
        let index = HybridLearnedIndex::build(&keys, config);

        // Do lookups
        for key in 0..1000 {
            index.lookup(key, &keys);
        }

        let avg_error = index.average_error();
        println!("Average error: {:.2}", avg_error);

        // For sequential data with ε=1, error should be very low
        assert!(avg_error < 2.0);
    }

    #[test]
    fn test_fit_linear() {
        let keys: Vec<Key> = (0..10).collect();
        let segment = HybridLearnedIndex::fit_linear(&keys, 0);

        // Perfect linear fit for sequential keys
        assert!((segment.slope - 1.0).abs() < 0.01);
        assert!(segment.intercept.abs() < 0.01);
    }

    #[test]
    fn test_sparse_keys() {
        // Very sparse keys
        let keys: Vec<Key> = vec![0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        let config = HybridIndexConfig {
            epsilon: 1,
            min_keys_for_model: 5,
            ..Default::default()
        };
        let index = HybridLearnedIndex::build(&keys, config);

        // Should still find all keys
        for (i, &key) in keys.iter().enumerate() {
            let pos = index.lookup(key, &keys);
            assert_eq!(pos, Some(i));
        }
    }
}
