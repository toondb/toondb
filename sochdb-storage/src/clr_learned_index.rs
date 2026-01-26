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

//! Compact Linear Regression (CLR) Learned Index (Task 3)
//!
//! ## Problem
//!
//! Binary search over sorted runs costs O(log N) comparisons per lookup.
//! For sorted data with predictable distributions (timestamps, sequential IDs),
//! this is wasteful - we can predict positions directly.
//!
//! ## Solution
//!
//! During memtable flush, fit a simple linear regression to the sorted keys:
//!
//! ```text
//! k̂ = floor(slope × key)
//! ```
//!
//! Store only (slope: f64, ε: u16) per sorted run.
//! On lookup: binary search only within [k̂ - ε, k̂ + ε] instead of [0, N).
//!
//! ## Mathematical Analysis
//!
//! ### Linear Model
//!
//! For sorted keys k₀ < k₁ < ... < kₙ₋₁, we fit:
//!   position ≈ slope × normalized_key + intercept
//!
//! Where normalized_key = (key - min_key) / (max_key - min_key)
//!
//! ### Error Bounds
//!
//! After fitting, compute maximum prediction error:
//!   ε = max_i |predicted_i - actual_i|
//!
//! On lookup, search within [predicted - ε, predicted + ε].
//!
//! ### Complexity
//!
//! | Operation     | Binary Search | CLR Index       |
//! |---------------|---------------|-----------------|
//! | Lookup        | O(log N)      | O(log 2ε)       |
//! | Space per run | 0             | 24 bytes        |
//! | Build time    | 0             | O(N)            |
//!
//! For well-distributed data, ε is typically small (< 100),
//! so O(log 2ε) << O(log N) when N > 10,000.
//!
//! ## Example
//!
//! For 100K entries with timestamps:
//! - Binary search: ~17 comparisons
//! - CLR with ε=50: ~7 comparisons (2.4× faster)

use serde::{Deserialize, Serialize};
use std::hash::Hash;

/// Result of a CLR index lookup
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClrLookupResult {
    /// Exact position found
    Exact(usize),
    /// Range of positions to search [low, high]
    Range { low: usize, high: usize },
    /// Key is out of bounds
    OutOfBounds,
}

/// Compact Linear Regression index for a sorted run
///
/// Uses only 24 bytes of metadata per sorted run:
/// - slope: f64 (8 bytes)
/// - intercept: f64 (8 bytes)
/// - max_error: u16 (2 bytes)
/// - min_key_hash: u32 (4 bytes for bounds check)
/// - max_key_hash: u32 (4 bytes for bounds check)
/// 
/// Note: We use key hashes for bounds checking to avoid storing
/// the actual keys (which could be large).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClrIndex {
    /// Slope of the linear model: position ≈ slope × normalized_key
    slope: f64,
    /// Intercept of the linear model
    intercept: f64,
    /// Maximum prediction error (epsilon)
    max_error: u16,
    /// Number of entries in the run
    num_entries: usize,
    /// Minimum key (for normalization)
    min_key: u64,
    /// Maximum key (for normalization)
    max_key: u64,
}

impl ClrIndex {
    /// Create a CLR index from sorted keys
    ///
    /// Keys MUST be sorted. This is not validated for performance.
    ///
    /// ## Arguments
    /// * `keys` - Sorted keys to index
    /// * `key_to_u64` - Function to convert keys to u64 for regression
    pub fn build<K, F>(keys: &[K], key_to_u64: F) -> Self
    where
        K: Clone,
        F: Fn(&K) -> u64,
    {
        let n = keys.len();
        
        if n == 0 {
            return Self {
                slope: 0.0,
                intercept: 0.0,
                max_error: 0,
                num_entries: 0,
                min_key: 0,
                max_key: 0,
            };
        }

        if n == 1 {
            let k = key_to_u64(&keys[0]);
            return Self {
                slope: 0.0,
                intercept: 0.0,
                max_error: 0,
                num_entries: 1,
                min_key: k,
                max_key: k,
            };
        }

        // Extract u64 keys
        let u64_keys: Vec<u64> = keys.iter().map(&key_to_u64).collect();
        let min_key = u64_keys[0];
        let max_key = u64_keys[n - 1];

        // Normalize keys to [0, 1] for numerical stability
        let key_range = (max_key as f64) - (min_key as f64);
        
        if key_range == 0.0 {
            // All keys are equal - degenerate case
            return Self {
                slope: 0.0,
                intercept: 0.0,
                max_error: (n / 2) as u16,
                num_entries: n,
                min_key,
                max_key,
            };
        }

        // Linear regression using normalized keys
        // position_i = slope × normalized_key_i + intercept
        // where normalized_key = (key - min_key) / key_range
        let (slope, intercept) = Self::fit_linear_regression(&u64_keys, min_key, key_range, n);

        // Compute maximum error
        let max_error = Self::compute_max_error(&u64_keys, min_key, key_range, slope, intercept);

        Self {
            slope,
            intercept,
            max_error: max_error.min(u16::MAX as usize) as u16,
            num_entries: n,
            min_key,
            max_key,
        }
    }

    /// Build from sorted u64 keys directly
    pub fn build_from_u64(keys: &[u64]) -> Self {
        Self::build(keys, |k| *k)
    }

    /// Build from sorted byte keys (uses hash for u64 conversion)
    pub fn build_from_bytes(keys: &[Vec<u8>]) -> Self {
        Self::build(keys, |k| Self::bytes_to_u64(k))
    }

    /// Convert bytes to u64 for indexing
    /// Uses first 8 bytes as big-endian u64 for ordering
    fn bytes_to_u64(bytes: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        let len = bytes.len().min(8);
        buf[..len].copy_from_slice(&bytes[..len]);
        u64::from_be_bytes(buf)
    }

    /// Fit linear regression: position = slope × normalized_key + intercept
    fn fit_linear_regression(
        keys: &[u64],
        min_key: u64,
        key_range: f64,
        n: usize,
    ) -> (f64, f64) {
        // Using Ordinary Least Squares (OLS):
        // slope = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
        // intercept = ȳ - slope × x̄
        
        // For normalized keys in [0, 1], we optimize:
        // slope = Σ(x_i × i) / Σ(x_i²) - when intercept is 0
        
        // Simple approach: fit to endpoints
        // position_0 = 0, position_{n-1} = n - 1
        // slope = (n - 1) / 1.0 = n - 1 (for normalized range [0, 1])
        
        // More accurate: full OLS
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;
        
        for (i, &key) in keys.iter().enumerate() {
            let x = ((key as f64) - (min_key as f64)) / key_range;
            let y = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        let n_f64 = n as f64;
        let mean_x = sum_x / n_f64;
        let mean_y = sum_y / n_f64;
        
        let numerator = sum_xy - n_f64 * mean_x * mean_y;
        let denominator = sum_xx - n_f64 * mean_x * mean_x;
        
        if denominator.abs() < 1e-10 {
            // Degenerate case: all keys have same normalized value
            return (0.0, mean_y);
        }
        
        let slope = numerator / denominator;
        let intercept = mean_y - slope * mean_x;
        
        (slope, intercept)
    }

    /// Compute maximum prediction error
    fn compute_max_error(
        keys: &[u64],
        min_key: u64,
        key_range: f64,
        slope: f64,
        intercept: f64,
    ) -> usize {
        let mut max_error = 0usize;
        
        for (i, &key) in keys.iter().enumerate() {
            let normalized = ((key as f64) - (min_key as f64)) / key_range;
            let predicted = slope * normalized + intercept;
            let predicted_pos = predicted.round() as isize;
            let actual_pos = i as isize;
            let error = (predicted_pos - actual_pos).unsigned_abs();
            max_error = max_error.max(error);
        }
        
        max_error
    }

    /// Predict the position of a key
    ///
    /// Returns a range [low, high] to search within.
    pub fn predict(&self, key: u64) -> ClrLookupResult {
        if self.num_entries == 0 {
            return ClrLookupResult::OutOfBounds;
        }

        if key < self.min_key {
            return ClrLookupResult::OutOfBounds;
        }

        if key > self.max_key {
            return ClrLookupResult::OutOfBounds;
        }

        let key_range = (self.max_key as f64) - (self.min_key as f64);
        
        if key_range == 0.0 {
            // All keys are equal
            return ClrLookupResult::Range {
                low: 0,
                high: self.num_entries.saturating_sub(1),
            };
        }

        let normalized = ((key as f64) - (self.min_key as f64)) / key_range;
        let predicted = self.slope * normalized + self.intercept;
        let predicted_pos = predicted.round() as isize;

        let error = self.max_error as isize;
        let low = (predicted_pos - error).max(0) as usize;
        let high = (predicted_pos + error).min(self.num_entries as isize - 1) as usize;

        if low == high {
            ClrLookupResult::Exact(low)
        } else {
            ClrLookupResult::Range { low, high }
        }
    }

    /// Predict position from bytes key
    pub fn predict_bytes(&self, key: &[u8]) -> ClrLookupResult {
        self.predict(Self::bytes_to_u64(key))
    }

    /// Get the maximum error (epsilon)
    pub fn max_error(&self) -> usize {
        self.max_error as usize
    }

    /// Get the number of entries
    pub fn num_entries(&self) -> usize {
        self.num_entries
    }

    /// Check if the model is useful (error is small enough)
    ///
    /// If max_error >= log2(num_entries), binary search is faster.
    /// Also, CLR overhead isn't worth it for very small runs.
    pub fn is_useful(&self) -> bool {
        // Minimum size threshold - CLR overhead isn't worth it for small runs
        const MIN_USEFUL_SIZE: usize = 64;
        
        if self.num_entries < MIN_USEFUL_SIZE {
            return false;
        }
        
        let log2_n = (self.num_entries as f64).log2().ceil() as usize;
        let search_range = 2 * (self.max_error as usize) + 1;
        let log2_range = (search_range as f64).log2().ceil() as usize;
        
        // CLR is useful if searching the error range is faster than binary search
        log2_range < log2_n
    }

    /// Get compression ratio: how much the search space is reduced
    pub fn compression_ratio(&self) -> f64 {
        if self.num_entries == 0 || self.max_error as usize >= self.num_entries {
            return 1.0;
        }
        
        let search_range = 2 * (self.max_error as usize) + 1;
        self.num_entries as f64 / search_range as f64
    }

    /// Memory size in bytes
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// A sorted run with an optional CLR index attached
#[derive(Debug)]
pub struct IndexedSortedRun<K, V> {
    /// The sorted entries
    entries: Vec<(K, V)>,
    /// Optional CLR index (only built if beneficial)
    clr_index: Option<ClrIndex>,
    /// Key to u64 conversion function (for byte keys)
    /// We store min/max for range checks
    #[allow(dead_code)]
    min_key: Option<K>,
    #[allow(dead_code)]
    max_key: Option<K>,
}

impl<K: Ord + Clone + Hash, V: Clone> IndexedSortedRun<K, V> {
    /// Create from sorted entries with CLR index
    pub fn from_sorted_with_index<F>(entries: Vec<(K, V)>, key_to_u64: F) -> Self
    where
        F: Fn(&K) -> u64,
    {
        if entries.is_empty() {
            return Self {
                entries,
                clr_index: None,
                min_key: None,
                max_key: None,
            };
        }

        let min_key = Some(entries.first().unwrap().0.clone());
        let max_key = Some(entries.last().unwrap().0.clone());

        // Build CLR index
        let keys: Vec<&K> = entries.iter().map(|(k, _)| k).collect();
        let clr = ClrIndex::build(&keys, |k| key_to_u64(*k));

        // Only keep index if it's beneficial
        let clr_index = if clr.is_useful() { Some(clr) } else { None };

        Self {
            entries,
            clr_index,
            min_key,
            max_key,
        }
    }

    /// Create without index (for comparison)
    pub fn from_sorted_no_index(entries: Vec<(K, V)>) -> Self {
        let min_key = entries.first().map(|(k, _)| k.clone());
        let max_key = entries.last().map(|(k, _)| k.clone());
        
        Self {
            entries,
            clr_index: None,
            min_key,
            max_key,
        }
    }

    /// Lookup using CLR index (if available) or binary search
    pub fn get<F>(&self, key: &K, key_to_u64: F) -> Option<&V>
    where
        F: Fn(&K) -> u64,
    {
        if self.entries.is_empty() {
            return None;
        }

        if let Some(ref clr) = self.clr_index {
            // Use CLR-guided search
            let key_u64 = key_to_u64(key);
            match clr.predict(key_u64) {
                ClrLookupResult::Exact(pos) => {
                    if pos < self.entries.len() && &self.entries[pos].0 == key {
                        return Some(&self.entries[pos].1);
                    }
                    // Fall back to binary search in range
                    self.binary_search_range(key, pos, pos)
                }
                ClrLookupResult::Range { low, high } => {
                    self.binary_search_range(key, low, high)
                }
                ClrLookupResult::OutOfBounds => None,
            }
        } else {
            // Fall back to standard binary search
            self.binary_search(key)
        }
    }

    /// Binary search within a range
    fn binary_search_range(&self, key: &K, low: usize, high: usize) -> Option<&V> {
        let low = low.min(self.entries.len().saturating_sub(1));
        let high = (high + 1).min(self.entries.len());
        
        if low >= high {
            // Single element
            if low < self.entries.len() && &self.entries[low].0 == key {
                return Some(&self.entries[low].1);
            }
            return None;
        }

        let slice = &self.entries[low..high];
        slice
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|idx| &self.entries[low + idx].1)
    }

    /// Standard binary search
    fn binary_search(&self, key: &K) -> Option<&V> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if CLR index is attached
    pub fn has_clr_index(&self) -> bool {
        self.clr_index.is_some()
    }

    /// Get CLR index stats
    pub fn clr_stats(&self) -> Option<ClrStats> {
        self.clr_index.as_ref().map(|clr| ClrStats {
            max_error: clr.max_error(),
            num_entries: clr.num_entries(),
            compression_ratio: clr.compression_ratio(),
            size_bytes: clr.size_bytes(),
        })
    }

    /// Iterate entries
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }
}

/// Statistics for CLR index
#[derive(Debug, Clone)]
pub struct ClrStats {
    /// Maximum prediction error
    pub max_error: usize,
    /// Number of entries indexed
    pub num_entries: usize,
    /// Search space compression ratio
    pub compression_ratio: f64,
    /// Index size in bytes
    pub size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clr_empty() {
        let keys: Vec<u64> = vec![];
        let clr = ClrIndex::build_from_u64(&keys);
        assert_eq!(clr.num_entries(), 0);
        assert!(!clr.is_useful());
    }

    #[test]
    fn test_clr_single() {
        let keys = vec![42u64];
        let clr = ClrIndex::build_from_u64(&keys);
        assert_eq!(clr.num_entries(), 1);
        assert!(!clr.is_useful());
    }

    #[test]
    fn test_clr_sequential() {
        // Sequential keys should have very low error
        let keys: Vec<u64> = (0..1000).collect();
        let clr = ClrIndex::build_from_u64(&keys);
        
        assert_eq!(clr.num_entries(), 1000);
        assert_eq!(clr.max_error(), 0); // Perfect fit for sequential data
        assert!(clr.is_useful());
        
        // Test predictions
        for i in 0..1000u64 {
            match clr.predict(i) {
                ClrLookupResult::Exact(pos) => assert_eq!(pos, i as usize),
                ClrLookupResult::Range { low, high } => {
                    assert!(low <= i as usize && i as usize <= high);
                }
                ClrLookupResult::OutOfBounds => panic!("Should not be out of bounds"),
            }
        }
    }

    #[test]
    fn test_clr_timestamps() {
        // Simulate timestamps with some variance
        let base: u64 = 1700000000000;
        let keys: Vec<u64> = (0..10000)
            .map(|i| base + i * 100 + (i % 7) as u64) // ~100ms apart with jitter
            .collect();
        
        let clr = ClrIndex::build_from_u64(&keys);
        
        assert_eq!(clr.num_entries(), 10000);
        assert!(clr.is_useful());
        
        // Max error should be small due to near-linear distribution
        assert!(clr.max_error() < 10);
        
        // Compression ratio should be high
        assert!(clr.compression_ratio() > 100.0);
    }

    #[test]
    fn test_clr_non_uniform() {
        // Non-uniform distribution: clusters with gaps
        let mut keys: Vec<u64> = Vec::new();
        // Cluster 1: 0-99
        keys.extend(0..100);
        // Gap: 100-999
        // Cluster 2: 1000-1099
        keys.extend(1000..1100);
        // Gap: 1100-9999
        // Cluster 3: 10000-10099
        keys.extend(10000..10100);
        
        let clr = ClrIndex::build_from_u64(&keys);
        
        assert_eq!(clr.num_entries(), 300);
        // Non-uniform data will have higher error
        assert!(clr.max_error() > 10);
        
        // Should still find all keys
        for &key in &keys {
            let result = clr.predict(key);
            assert!(!matches!(result, ClrLookupResult::OutOfBounds));
        }
    }

    #[test]
    fn test_clr_out_of_bounds() {
        let keys: Vec<u64> = (100..200).collect();
        let clr = ClrIndex::build_from_u64(&keys);
        
        // Below range
        assert!(matches!(clr.predict(50), ClrLookupResult::OutOfBounds));
        
        // Above range
        assert!(matches!(clr.predict(250), ClrLookupResult::OutOfBounds));
        
        // In range
        assert!(!matches!(clr.predict(150), ClrLookupResult::OutOfBounds));
    }

    #[test]
    fn test_indexed_sorted_run() {
        let entries: Vec<(u64, String)> = (0..1000)
            .map(|i| (i, format!("value_{}", i)))
            .collect();
        
        let run = IndexedSortedRun::from_sorted_with_index(entries, |k| *k);
        
        assert_eq!(run.len(), 1000);
        assert!(run.has_clr_index());
        
        // Lookup should work
        for i in 0..1000u64 {
            let result = run.get(&i, |k| *k);
            assert_eq!(result, Some(&format!("value_{}", i)));
        }
        
        // Missing keys
        assert_eq!(run.get(&2000, |k| *k), None);
    }

    #[test]
    fn test_indexed_sorted_run_bytes() {
        // Test with byte keys - need enough entries for CLR to be useful
        let entries: Vec<(Vec<u8>, u32)> = (0..500u32)
            .map(|i| (i.to_be_bytes().to_vec(), i))
            .collect();
        
        let run = IndexedSortedRun::from_sorted_with_index(entries, |k| {
            ClrIndex::bytes_to_u64(k)
        });
        
        // 500 entries >= MIN_USEFUL_SIZE (64), so CLR should be enabled
        assert!(run.has_clr_index());
        
        // Lookup
        for i in 0..500u32 {
            let key = i.to_be_bytes().to_vec();
            let result = run.get(&key, |k| ClrIndex::bytes_to_u64(k));
            assert_eq!(result, Some(&i));
        }
    }

    #[test]
    fn test_clr_usefulness() {
        // Very small runs shouldn't use CLR
        let small: Vec<u64> = (0..5).collect();
        let clr_small = ClrIndex::build_from_u64(&small);
        assert!(!clr_small.is_useful());
        
        // Large runs with good distribution should use CLR
        let large: Vec<u64> = (0..10000).collect();
        let clr_large = ClrIndex::build_from_u64(&large);
        assert!(clr_large.is_useful());
    }

    #[test]
    fn test_clr_compression_ratio() {
        let keys: Vec<u64> = (0..10000).collect();
        let clr = ClrIndex::build_from_u64(&keys);
        
        // Sequential data should have very high compression
        assert!(clr.compression_ratio() > 1000.0);
    }

    #[test]
    fn test_clr_size() {
        let clr = ClrIndex::build_from_u64(&[1, 2, 3, 4, 5]);
        
        // CLR index should be compact (~40 bytes)
        assert!(clr.size_bytes() < 100);
    }
}
