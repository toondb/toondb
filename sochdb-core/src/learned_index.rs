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

//! Learned Sparse Index (LSI)
//!
//! A novel index structure that uses linear regression to predict key positions
//! instead of tree-based lookups. For well-distributed data (timestamps, sequential IDs),
//! this provides O(1) expected lookups instead of O(log N).
//!
//! ## Mathematical Foundation
//!
//! Given sorted keys k₁ < k₂ < ... < kₙ, the CDF maps keys to positions:
//! F(k) = |{kᵢ : kᵢ ≤ k}| / N
//!
//! For many real distributions, F can be approximated by a simple linear model:
//! F̂(k) = slope * k + intercept
//!
//! With error bound ε: |F(k) - F̂(k)| ≤ ε
//! We only search 2ε positions instead of the entire index.
//!
//! ## Complexity Analysis
//!
//! | Operation | B-Tree      | Learned Index           |
//! |-----------|-------------|-------------------------|
//! | Lookup    | O(log N)    | O(1) expected, O(ε) worst |
//! | Insert    | O(log N)    | O(1) amortized + rebuild |
//! | Space     | O(N)        | O(1) + O(outliers)      |

use serde::{Deserialize, Serialize};

/// Result of a learned index lookup
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LookupResult {
    /// Exact position found (from correction table)
    Exact(usize),
    /// Range of positions to search [low, high]
    Range { low: usize, high: usize },
    /// Key is out of bounds
    NotFound,
}

/// Learned Sparse Index using linear regression
///
/// Now with key normalization for numerical stability with large u64 keys.
/// Keys are normalized to [0, n-1] before regression, avoiding precision
/// loss with values near u64::MAX.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedSparseIndex {
    /// Linear model: position ≈ slope * normalized_key + intercept
    slope: f64,
    intercept: f64,
    /// Maximum error of the model
    max_error: usize,
    /// Correction table for outliers (sparse), sorted by key
    corrections: Vec<(u64, usize)>,
    /// Minimum key in the index (for normalization)
    min_key: u64,
    /// Maximum key in the index (for normalization)
    max_key: u64,
    /// Precomputed key range as f64 (max_key - min_key)
    key_range: f64,
    /// Total number of keys
    num_keys: usize,
    /// Threshold for storing corrections (keys with error > this are stored)
    correction_threshold: usize,
}

impl LearnedSparseIndex {
    /// Default threshold for storing corrections
    const DEFAULT_CORRECTION_THRESHOLD: usize = 64;

    /// Create an empty index
    pub fn empty() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            max_error: 0,
            corrections: Vec::new(),
            min_key: 0,
            max_key: 0,
            key_range: 0.0,
            num_keys: 0,
            correction_threshold: Self::DEFAULT_CORRECTION_THRESHOLD,
        }
    }

    /// Normalize a key to range [0, n-1] for numerical stability
    ///
    /// Given keys k_min < k_max, transforms key k to:
    /// k_normalized = ((k - k_min) / (k_max - k_min)) * (n - 1)
    #[inline]
    fn normalize_key(&self, key: u64) -> f64 {
        if self.key_range == 0.0 {
            return 0.0;
        }
        // Use u128 arithmetic to avoid overflow when subtracting
        let offset = (key as u128).saturating_sub(self.min_key as u128) as f64;
        (offset / self.key_range) * (self.num_keys - 1) as f64
    }

    /// Build index from sorted keys
    ///
    /// Keys MUST be sorted in ascending order. This is not validated for performance.
    pub fn build(keys: &[u64]) -> Self {
        Self::build_with_threshold(keys, Self::DEFAULT_CORRECTION_THRESHOLD)
    }

    /// Build index with custom correction threshold
    ///
    /// Lower threshold = more corrections stored = faster lookups but more memory
    /// Higher threshold = fewer corrections = slower lookups but less memory
    pub fn build_with_threshold(keys: &[u64], correction_threshold: usize) -> Self {
        let n = keys.len();
        if n == 0 {
            return Self::empty();
        }

        if n == 1 {
            return Self {
                slope: 0.0,
                intercept: 0.0,
                max_error: 0,
                corrections: Vec::new(),
                min_key: keys[0],
                max_key: keys[0],
                key_range: 0.0,
                num_keys: 1,
                correction_threshold,
            };
        }

        let min_key = keys[0];
        let max_key = keys[n - 1];
        // Use u128 to avoid overflow when computing range
        let key_range = (max_key as u128 - min_key as u128) as f64;

        // Fit linear regression on NORMALIZED keys for numerical stability
        let (slope, intercept) = Self::linear_regression_normalized(keys, min_key, key_range, n);

        // Calculate errors and find outliers
        let mut max_error = 0usize;
        let mut corrections = Vec::new();

        for (actual_pos, &key) in keys.iter().enumerate() {
            // Use normalized key for prediction
            let normalized = if key_range == 0.0 {
                0.0
            } else {
                let offset = (key as u128 - min_key as u128) as f64;
                (offset / key_range) * (n - 1) as f64
            };
            let predicted = slope * normalized + intercept;
            let predicted_pos = predicted.round() as isize;
            let error = (actual_pos as isize - predicted_pos).unsigned_abs();

            if error > max_error {
                max_error = error;
            }

            // Store correction for large errors (outliers)
            if error > correction_threshold {
                corrections.push((key, actual_pos));
            }
        }

        Self {
            slope,
            intercept,
            max_error,
            corrections,
            min_key,
            max_key,
            key_range,
            num_keys: n,
            correction_threshold,
        }
    }

    /// Lookup: O(1) expected, O(ε) worst case
    pub fn lookup(&self, key: u64) -> LookupResult {
        if self.num_keys == 0 {
            return LookupResult::NotFound;
        }

        // Bounds check
        if key < self.min_key || key > self.max_key {
            return LookupResult::NotFound;
        }

        // Check corrections first (outliers get O(1) exact lookup)
        // Binary search in the sorted corrections vector
        if let Ok(idx) = self.corrections.binary_search_by_key(&key, |&(k, _)| k) {
            return LookupResult::Exact(self.corrections[idx].1);
        }

        // Predict position using normalized key for numerical stability
        let normalized = self.normalize_key(key);
        let predicted = self.slope * normalized + self.intercept;
        let predicted_pos = predicted.round() as isize;

        // Calculate search range based on max error
        let low = (predicted_pos - self.max_error as isize).max(0) as usize;
        let high =
            (predicted_pos + self.max_error as isize).min(self.num_keys as isize - 1) as usize;

        LookupResult::Range { low, high }
    }

    /// Lookup with a custom error margin (tighter range if you know data is regular)
    pub fn lookup_with_error(&self, key: u64, max_error: usize) -> LookupResult {
        if self.num_keys == 0 {
            return LookupResult::NotFound;
        }

        if key < self.min_key || key > self.max_key {
            return LookupResult::NotFound;
        }

        if let Ok(idx) = self.corrections.binary_search_by_key(&key, |&(k, _)| k) {
            return LookupResult::Exact(self.corrections[idx].1);
        }

        // Use normalized key for prediction
        let normalized = self.normalize_key(key);
        let predicted = self.slope * normalized + self.intercept;
        let predicted_pos = predicted.round() as isize;

        let low = (predicted_pos - max_error as isize).max(0) as usize;
        let high = (predicted_pos + max_error as isize).min(self.num_keys as isize - 1) as usize;

        LookupResult::Range { low, high }
    }

    /// Get index statistics
    pub fn stats(&self) -> LearnedIndexStats {
        LearnedIndexStats {
            num_keys: self.num_keys,
            max_error: self.max_error,
            num_corrections: self.corrections.len(),
            slope: self.slope,
            intercept: self.intercept,
            correction_ratio: if self.num_keys > 0 {
                self.corrections.len() as f64 / self.num_keys as f64
            } else {
                0.0
            },
        }
    }

    /// Returns true if this index is well-suited for learned indexing
    /// (low error, few corrections)
    pub fn is_efficient(&self) -> bool {
        // Efficient if:
        // - Max error is small (under 128 positions to search)
        // - Correction ratio is low (under 5% of keys are outliers)
        let low_error = self.max_error <= 128;
        let low_corrections =
            self.num_keys == 0 || (self.corrections.len() as f64 / self.num_keys as f64) < 0.05;
        low_error && low_corrections
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.corrections.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<usize>())
    }

    /// Linear regression on NORMALIZED keys for numerical stability
    ///
    /// Keys are normalized to [0, n-1] before regression to avoid precision loss
    /// with large u64 values (near u64::MAX, squaring would overflow f64 precision).
    fn linear_regression_normalized(
        keys: &[u64],
        min_key: u64,
        key_range: f64,
        n: usize,
    ) -> (f64, f64) {
        let n_f64 = n as f64;

        // Using numerically stable algorithm with normalized keys
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_xx: f64 = 0.0;

        for (i, &key) in keys.iter().enumerate() {
            // Normalize key to [0, n-1]
            let x = if key_range == 0.0 {
                0.0
            } else {
                let offset = (key as u128 - min_key as u128) as f64;
                (offset / key_range) * (n - 1) as f64
            };
            let y = i as f64;

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denominator = n_f64 * sum_xx - sum_x * sum_x;

        // Handle degenerate case (all keys are the same)
        if denominator.abs() < f64::EPSILON {
            return (0.0, sum_y / n_f64);
        }

        let slope = (n_f64 * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n_f64;

        (slope, intercept)
    }

    /// Legacy linear regression (kept for compatibility but deprecated)
    #[allow(dead_code)]
    fn linear_regression(keys: &[u64]) -> (f64, f64) {
        let n = keys.len() as f64;

        // Using numerically stable algorithm
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_xx: f64 = 0.0;

        for (i, &key) in keys.iter().enumerate() {
            let x = key as f64;
            let y = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denominator = n * sum_xx - sum_x * sum_x;

        // Handle degenerate case (all keys are the same)
        if denominator.abs() < f64::EPSILON {
            return (0.0, sum_y / n);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Incrementally add a key (rebuilds if model quality degrades)
    /// Returns true if a rebuild was triggered
    pub fn insert(&mut self, key: u64, position: usize, keys: &[u64]) -> bool {
        // Predict where this key should be using normalized key
        let normalized = self.normalize_key(key);
        let predicted = self.slope * normalized + self.intercept;
        let predicted_pos = predicted.round() as isize;
        let error = (position as isize - predicted_pos).unsigned_abs();

        // Update bounds
        self.min_key = self.min_key.min(key);
        self.max_key = self.max_key.max(key);
        // Recalculate key_range
        self.key_range = (self.max_key as u128 - self.min_key as u128) as f64;
        self.num_keys += 1;

        if error > self.max_error {
            self.max_error = error;
        }

        if error > self.correction_threshold {
            // Insert into sorted corrections vector
            match self.corrections.binary_search_by_key(&key, |&(k, _)| k) {
                Ok(idx) => self.corrections[idx] = (key, position),
                Err(idx) => self.corrections.insert(idx, (key, position)),
            }
        }

        // Rebuild if model is degrading too much
        if self.corrections.len() > self.num_keys / 10 {
            *self = Self::build_with_threshold(keys, self.correction_threshold);
            return true;
        }

        false
    }
}

/// Statistics about the learned index
#[derive(Debug, Clone)]
pub struct LearnedIndexStats {
    /// Number of keys in the index
    pub num_keys: usize,
    /// Maximum prediction error
    pub max_error: usize,
    /// Number of keys with corrections stored
    pub num_corrections: usize,
    /// Linear model slope
    pub slope: f64,
    /// Linear model intercept
    pub intercept: f64,
    /// Ratio of keys that need corrections
    pub correction_ratio: f64,
}

/// Piecewise Learned Index for non-linear distributions
///
/// Divides the key space into segments, each with its own linear model
#[derive(Debug, Clone)]
pub struct PiecewiseLearnedIndex {
    /// Segment boundaries (sorted)
    boundaries: Vec<u64>,
    /// Index for each segment
    segments: Vec<LearnedSparseIndex>,
}

impl PiecewiseLearnedIndex {
    /// Build piecewise index with automatic segmentation
    pub fn build(keys: &[u64], max_segments: usize) -> Self {
        if keys.is_empty() || max_segments == 0 {
            return Self {
                boundaries: vec![],
                segments: vec![],
            };
        }

        // Simple even segmentation (could use dynamic programming for optimal)
        let segment_size = keys.len().div_ceil(max_segments);
        let mut boundaries = Vec::with_capacity(max_segments);
        let mut segments = Vec::with_capacity(max_segments);

        for chunk in keys.chunks(segment_size) {
            if !chunk.is_empty() {
                boundaries.push(chunk[0]);
                segments.push(LearnedSparseIndex::build(chunk));
            }
        }

        Self {
            boundaries,
            segments,
        }
    }

    /// Find segment containing key
    fn find_segment(&self, key: u64) -> Option<usize> {
        if self.boundaries.is_empty() {
            return None;
        }

        // Binary search for segment
        match self.boundaries.binary_search(&key) {
            Ok(i) => Some(i),
            Err(i) => {
                if i == 0 {
                    None
                } else {
                    Some(i - 1)
                }
            }
        }
    }

    /// Lookup with piecewise model
    pub fn lookup(&self, key: u64) -> LookupResult {
        match self.find_segment(key) {
            Some(seg_idx) => self.segments[seg_idx].lookup(key),
            None => LookupResult::NotFound,
        }
    }

    /// Get aggregate statistics
    pub fn stats(&self) -> PiecewiseStats {
        let segment_stats: Vec<_> = self.segments.iter().map(|s| s.stats()).collect();
        let total_keys: usize = segment_stats.iter().map(|s| s.num_keys).sum();
        let max_error = segment_stats.iter().map(|s| s.max_error).max().unwrap_or(0);
        let total_corrections: usize = segment_stats.iter().map(|s| s.num_corrections).sum();

        PiecewiseStats {
            num_segments: self.segments.len(),
            total_keys,
            max_error,
            total_corrections,
            avg_segment_size: if self.segments.is_empty() {
                0.0
            } else {
                total_keys as f64 / self.segments.len() as f64
            },
        }
    }
}

/// Statistics for piecewise learned index
#[derive(Debug, Clone)]
pub struct PiecewiseStats {
    pub num_segments: usize,
    pub total_keys: usize,
    pub max_error: usize,
    pub total_corrections: usize,
    pub avg_segment_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = LearnedSparseIndex::build(&[]);
        assert_eq!(index.lookup(42), LookupResult::NotFound);
        assert_eq!(index.stats().num_keys, 0);
    }

    #[test]
    fn test_single_key() {
        let index = LearnedSparseIndex::build(&[100]);
        assert!(matches!(
            index.lookup(100),
            LookupResult::Range { low: 0, high: 0 }
        ));
        assert_eq!(index.lookup(50), LookupResult::NotFound);
        assert_eq!(index.lookup(150), LookupResult::NotFound);
    }

    #[test]
    fn test_sequential_keys() {
        // Perfect linear distribution - should have very low error
        let keys: Vec<u64> = (0..1000).collect();
        let index = LearnedSparseIndex::build(&keys);

        let stats = index.stats();
        assert!(
            stats.max_error <= 1,
            "Sequential keys should have near-zero error"
        );
        assert!(
            stats.num_corrections == 0,
            "No corrections needed for linear data"
        );

        // Lookup should give exact or near-exact positions
        if let LookupResult::Range { low, high } = index.lookup(500) {
            assert!(low <= 500 && high >= 500, "Key 500 should be in range");
            assert!(high - low <= 2, "Range should be very tight");
        }
    }

    #[test]
    fn test_timestamp_like_keys() {
        // Timestamps with gaps (realistic workload)
        let mut keys: Vec<u64> = Vec::new();
        let mut ts: u64 = 1704067200; // Jan 1, 2024
        for _ in 0..10000 {
            keys.push(ts);
            ts += 1 + (ts % 10); // Variable gaps 1-10
        }

        let index = LearnedSparseIndex::build(&keys);

        // Should still be efficient for roughly linear data
        assert!(
            index.is_efficient(),
            "Timestamp data should be efficiently indexable"
        );

        // Verify lookups
        for &key in keys.iter().take(100) {
            let result = index.lookup(key);
            assert!(
                !matches!(result, LookupResult::NotFound),
                "Existing key should be found"
            );
        }
    }

    #[test]
    fn test_sparse_keys() {
        // Very sparse keys (high gaps)
        let keys: Vec<u64> = vec![1, 100, 10000, 1000000, 100000000];
        let index = LearnedSparseIndex::build(&keys);

        // All keys should be findable
        for (i, &key) in keys.iter().enumerate() {
            match index.lookup(key) {
                LookupResult::Exact(pos) => assert_eq!(pos, i),
                LookupResult::Range { low, high } => {
                    assert!(
                        low <= i && i <= high,
                        "Key {} should be in range [{}, {}]",
                        key,
                        low,
                        high
                    );
                }
                LookupResult::NotFound => panic!("Key {} should be found", key),
            }
        }
    }

    #[test]
    fn test_out_of_bounds() {
        let keys: Vec<u64> = (100..200).collect();
        let index = LearnedSparseIndex::build(&keys);

        assert_eq!(index.lookup(50), LookupResult::NotFound);
        assert_eq!(index.lookup(250), LookupResult::NotFound);
    }

    #[test]
    fn test_piecewise_index() {
        // Create non-linear data that benefits from segmentation
        let mut keys: Vec<u64> = Vec::new();

        // Three distinct clusters
        keys.extend(0..1000); // Dense cluster 1
        keys.extend((100000..101000).step_by(10)); // Sparse cluster 2
        keys.extend(1000000..1001000); // Dense cluster 3

        let piecewise = PiecewiseLearnedIndex::build(&keys, 3);
        let stats = piecewise.stats();

        assert_eq!(stats.num_segments, 3);

        // Verify lookups across clusters
        assert!(!matches!(piecewise.lookup(500), LookupResult::NotFound));
        assert!(!matches!(piecewise.lookup(100500), LookupResult::NotFound));
        assert!(!matches!(piecewise.lookup(1000500), LookupResult::NotFound));
    }

    #[test]
    fn test_memory_efficiency() {
        // Compare memory to theoretical B-tree
        let keys: Vec<u64> = (0..100000).collect();
        let index = LearnedSparseIndex::build(&keys);

        let lsi_bytes = index.memory_bytes();
        let btree_bytes = keys.len() * std::mem::size_of::<u64>(); // Minimum B-tree overhead

        // LSI should use significantly less memory for linear data
        assert!(
            lsi_bytes < btree_bytes,
            "LSI ({} bytes) should use less memory than keys alone ({} bytes)",
            lsi_bytes,
            btree_bytes
        );
    }

    #[test]
    fn test_correction_threshold() {
        // Create data with some outliers
        let mut keys: Vec<u64> = (0..100).map(|x| x * 10).collect();
        keys.push(5000); // Outlier
        keys.sort();

        // Low threshold = more corrections
        let low_thresh = LearnedSparseIndex::build_with_threshold(&keys, 10);

        // High threshold = fewer corrections
        let high_thresh = LearnedSparseIndex::build_with_threshold(&keys, 1000);

        assert!(
            low_thresh.stats().num_corrections >= high_thresh.stats().num_corrections,
            "Lower threshold should produce more or equal corrections"
        );
    }

    // ========================================================================
    // Task 4: Key Normalization Tests for Numerical Stability
    // ========================================================================

    #[test]
    fn test_large_key_normalization() {
        // Test with keys near u64::MAX - this would overflow without normalization
        let base = u64::MAX - 1000;
        let keys: Vec<u64> = (0..100).map(|i| base + i * 10).collect();

        let index = LearnedSparseIndex::build(&keys);

        // Should have reasonable error even with huge keys
        assert!(
            index.max_error < 10,
            "Error should be small for linear data"
        );

        // Lookups should work correctly
        for (i, &key) in keys.iter().enumerate() {
            let result = index.lookup(key);
            match result {
                LookupResult::Range { low, high } => {
                    assert!(
                        low <= i && i <= high,
                        "Key {} at position {} should be in range [{}, {}]",
                        key,
                        i,
                        low,
                        high
                    );
                }
                LookupResult::Exact(pos) => {
                    assert_eq!(pos, i, "Exact position should match");
                }
                LookupResult::NotFound => {
                    panic!("Key {} should be found", key);
                }
            }
        }
    }

    #[test]
    fn test_full_range_keys() {
        // Test keys spanning from 0 to near MAX
        let keys: Vec<u64> = vec![
            0,
            1_000_000,
            1_000_000_000,
            1_000_000_000_000,
            1_000_000_000_000_000,
            u64::MAX / 2,
            u64::MAX - 1000,
            u64::MAX - 100,
            u64::MAX - 10,
            u64::MAX - 1,
        ];

        let index = LearnedSparseIndex::build(&keys);

        // All keys should be findable
        for (i, &key) in keys.iter().enumerate() {
            let result = index.lookup(key);
            match result {
                LookupResult::Range { low, high } => {
                    assert!(
                        low <= i && i <= high,
                        "Key {} at position {} should be in range [{}, {}]",
                        key,
                        i,
                        low,
                        high
                    );
                }
                LookupResult::Exact(pos) => {
                    assert_eq!(pos, i, "Exact position should match");
                }
                LookupResult::NotFound => {
                    panic!("Key {} should be found", key);
                }
            }
        }
    }

    #[test]
    fn test_timestamp_keys() {
        // Simulate realistic timestamp keys (microseconds since epoch)
        // Current time is around 1.7e15 microseconds
        let base_ts: u64 = 1_700_000_000_000_000;
        let keys: Vec<u64> = (0..1000).map(|i| base_ts + i * 1000).collect();

        let index = LearnedSparseIndex::build(&keys);

        // Should have very low error for sequential timestamps
        assert!(
            index.max_error <= 1,
            "Error for sequential timestamps should be ≤ 1, got {}",
            index.max_error
        );

        // Verify efficiency
        assert!(
            index.is_efficient(),
            "Sequential timestamp data should be efficient"
        );
    }

    #[test]
    fn test_normalization_precision() {
        // Test that normalization maintains precision
        let index = LearnedSparseIndex {
            slope: 1.0,
            intercept: 0.0,
            max_error: 0,
            corrections: Vec::new(),
            min_key: 0,
            max_key: 99,
            key_range: 99.0,
            num_keys: 100,
            correction_threshold: 64,
        };

        // Key 0 should normalize to 0.0
        assert!((index.normalize_key(0) - 0.0).abs() < f64::EPSILON);

        // Key max should normalize to n-1 = 99.0
        assert!((index.normalize_key(99) - 99.0).abs() < f64::EPSILON);

        // Middle key should normalize to middle
        assert!((index.normalize_key(49) - 49.0).abs() < 0.5);
    }
}
