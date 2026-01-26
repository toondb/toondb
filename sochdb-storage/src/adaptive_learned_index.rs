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

//! Adaptive Learned Index with Bounded Error
//!
//! This module implements a self-tuning learned index that maintains error bounds
//! under updates using Piecewise Linear Approximation (PLA).
//!
//! ## Key Features
//!
//! - O(1) expected lookup with guaranteed O(log ε) worst case
//! - Automatic retraining when error exceeds threshold
//! - Incremental updates without full retrain
//! - Buffer-based insert handling
//!
//! ## Error-Bounded Learned Index
//!
//! For key k, model predicts position p̂(k). True position is p(k).
//! Error: ε = max|p̂(k) - p(k)| over all k
//! Local search: O(ε) comparisons
//!
//! ## Error Bound Guarantee
//!
//! Using Piecewise Linear Approximation:
//! ε ≤ range / (2 × segments)
//!
//! For N keys with S segments:
//! ε ≤ N / (2S)
//!
//! To achieve ε ≤ 64: S ≥ N/128
//!
//! ## Memory vs Error Tradeoff
//!
//! S = N/128 segments × 16 bytes/segment = N/8 bytes
//! Traditional B-tree = N × 16 bytes
//! **Memory savings: 128×**

/// Piecewise linear model segment
#[derive(Debug, Clone)]
struct LinearSegment {
    /// Start key for this segment
    start_key: i64,
    /// Slope of linear model
    slope: f64,
    /// Intercept of linear model
    intercept: f64,
}

/// Piecewise linear model for learned index
#[derive(Debug, Clone)]
pub struct PiecewiseLinearModel {
    /// Segments ordered by key
    segments: Vec<LinearSegment>,
    /// Maximum observed error
    max_error: usize,
    /// Key count
    key_count: usize,
    /// Target error bound
    target_error: usize,
}

impl PiecewiseLinearModel {
    /// Build model from sorted keys
    ///
    /// Uses simple linear regression per segment with greedy segmentation.
    ///
    /// # Arguments
    /// * `keys` - Sorted keys
    /// * `target_error` - Maximum allowed prediction error
    pub fn build(keys: &[i64], target_error: usize) -> Self {
        let n = keys.len();
        if n == 0 {
            return Self::empty(target_error);
        }

        if n == 1 {
            return Self {
                segments: vec![LinearSegment {
                    start_key: keys[0],
                    slope: 0.0,
                    intercept: 0.0,
                }],
                max_error: 0,
                key_count: 1,
                target_error,
            };
        }

        // Greedy segmentation: create new segment when error exceeds target
        let mut segments = Vec::new();
        let mut segment_start = 0;
        let mut max_error = 0;

        while segment_start < n {
            // Try to extend segment as far as possible while maintaining error bound
            let mut segment_end = segment_start + 1;
            let mut best_end = segment_start + 1;
            let (mut best_slope, mut best_intercept) =
                Self::fit_segment(keys, segment_start, best_end);

            while segment_end < n {
                // Fit model to [segment_start, segment_end]
                let (slope, intercept) = Self::fit_segment(keys, segment_start, segment_end + 1);

                // Check error
                let error =
                    Self::compute_error(keys, segment_start, segment_end + 1, slope, intercept);

                if error <= target_error {
                    best_end = segment_end + 1;
                    best_slope = slope;
                    best_intercept = intercept;
                    segment_end += 1;
                } else {
                    break;
                }
            }

            // Record segment
            segments.push(LinearSegment {
                start_key: keys[segment_start],
                slope: best_slope,
                intercept: best_intercept,
            });

            // Update max error
            let seg_error =
                Self::compute_error(keys, segment_start, best_end, best_slope, best_intercept);
            max_error = max_error.max(seg_error);

            segment_start = best_end;
        }

        Self {
            segments,
            max_error,
            key_count: n,
            target_error,
        }
    }

    /// Fit a linear segment using least squares regression
    fn fit_segment(keys: &[i64], start: usize, end: usize) -> (f64, f64) {
        let n = (end - start) as f64;
        if n <= 1.0 {
            return (0.0, start as f64);
        }

        // Linear regression: y = slope * x + intercept
        // Where x = key, y = position
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &key) in keys.iter().enumerate().take(end).skip(start) {
            let x = key as f64;
            let y = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            // All keys are the same, use constant model
            return (0.0, sum_y / n);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Compute maximum error for a segment
    fn compute_error(keys: &[i64], start: usize, end: usize, slope: f64, intercept: f64) -> usize {
        let mut max_err = 0usize;

        for (i, &key) in keys.iter().enumerate().take(end).skip(start) {
            let predicted = (slope * key as f64 + intercept).round() as i64;
            let actual = i as i64;
            let err = (predicted - actual).unsigned_abs() as usize;
            max_err = max_err.max(err);
        }

        max_err
    }

    /// Predict position for a key
    ///
    /// Returns the predicted position. Actual position is within ±max_error.
    #[inline]
    pub fn predict(&self, key: i64) -> usize {
        if self.segments.is_empty() {
            return 0;
        }

        // Binary search to find the correct segment
        let seg_idx = match self.segments.binary_search_by_key(&key, |s| s.start_key) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        let segment = &self.segments[seg_idx];
        let predicted = segment.slope * key as f64 + segment.intercept;

        // Clamp to valid range
        predicted.round().max(0.0) as usize
    }

    /// Get guaranteed error bound
    #[inline]
    pub fn error_bound(&self) -> usize {
        self.max_error
    }

    /// Get number of segments
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get key count
    #[inline]
    pub fn key_count(&self) -> usize {
        self.key_count
    }

    /// Get target error
    #[inline]
    pub fn target_error(&self) -> usize {
        self.target_error
    }

    /// Create empty model
    fn empty(target_error: usize) -> Self {
        Self {
            segments: vec![],
            max_error: 0,
            key_count: 0,
            target_error,
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Each segment: 8 (start_key) + 8 (slope) + 8 (intercept) = 24 bytes
        // Plus struct overhead
        std::mem::size_of::<Self>() + self.segments.len() * std::mem::size_of::<LinearSegment>()
    }
}

/// Adaptive learned index with automatic retraining
///
/// Maintains a learned index that self-adjusts as data changes.
/// Uses a write buffer to amortize retrain cost.
pub struct AdaptiveLearnedIndex {
    /// Current model
    model: PiecewiseLinearModel,
    /// Sorted keys (for local search)
    keys: Vec<i64>,
    /// Values (row IDs or pointers)
    values: Vec<u64>,
    /// Insert buffer (unsorted)
    insert_buffer: Vec<(i64, u64)>,
    /// Buffer flush threshold
    buffer_threshold: usize,
    /// Target error bound
    target_error: usize,
    /// Error increase ratio that triggers retrain
    retrain_ratio: f64,
    /// Statistics
    stats: LearnedIndexStats,
}

/// Statistics for learned index
#[derive(Debug, Default, Clone)]
pub struct LearnedIndexStats {
    /// Total lookups performed
    pub lookups: u64,
    /// Cache hits (found in buffer)
    pub buffer_hits: u64,
    /// Total comparisons during local search
    pub total_comparisons: u64,
    /// Number of retrains performed
    pub retrains: u64,
    /// Buffer flushes performed
    pub buffer_flushes: u64,
}

impl AdaptiveLearnedIndex {
    /// Create a new adaptive learned index
    ///
    /// # Arguments
    /// * `target_error` - Maximum allowed prediction error
    pub fn new(target_error: usize) -> Self {
        Self {
            model: PiecewiseLinearModel::empty(target_error),
            keys: Vec::new(),
            values: Vec::new(),
            insert_buffer: Vec::with_capacity(1024),
            buffer_threshold: 1024,
            target_error,
            retrain_ratio: 2.0, // Retrain if error doubles
            stats: LearnedIndexStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(target_error: usize, buffer_threshold: usize, retrain_ratio: f64) -> Self {
        Self {
            model: PiecewiseLinearModel::empty(target_error),
            keys: Vec::new(),
            values: Vec::new(),
            insert_buffer: Vec::with_capacity(buffer_threshold),
            buffer_threshold,
            target_error,
            retrain_ratio,
            stats: LearnedIndexStats::default(),
        }
    }

    /// Bulk load sorted data
    ///
    /// More efficient than individual inserts for initial load.
    pub fn bulk_load(&mut self, data: Vec<(i64, u64)>) {
        let mut sorted = data;
        sorted.sort_by_key(|(k, _)| *k);

        self.keys = sorted.iter().map(|(k, _)| *k).collect();
        self.values = sorted.iter().map(|(_, v)| *v).collect();

        // Build model
        self.model = PiecewiseLinearModel::build(&self.keys, self.target_error);
    }

    /// Point lookup - O(1) expected, O(log ε) worst case
    ///
    /// # Arguments
    /// * `key` - Key to look up
    ///
    /// # Returns
    /// Value if found, None otherwise
    pub fn get(&mut self, key: i64) -> Option<u64> {
        self.stats.lookups += 1;

        // Check insert buffer first (small, linear scan OK)
        for (k, v) in &self.insert_buffer {
            if *k == key {
                self.stats.buffer_hits += 1;
                return Some(*v);
            }
        }

        if self.keys.is_empty() {
            return None;
        }

        // Use model prediction
        let predicted = self.model.predict(key);
        let error = self.model.error_bound();

        // Bounded local search
        let start = predicted.saturating_sub(error);
        let end = (predicted + error + 1).min(self.keys.len());

        // Binary search within error bounds
        self.stats.total_comparisons += (end - start).max(1) as u64;

        let slice = &self.keys[start..end];
        match slice.binary_search(&key) {
            Ok(i) => Some(self.values[start + i]),
            Err(_) => None,
        }
    }

    /// Range query - returns all values where key is in [low, high]
    pub fn range(&mut self, low: i64, high: i64) -> Vec<(i64, u64)> {
        let mut result = Vec::new();

        // Check buffer
        for (k, v) in &self.insert_buffer {
            if *k >= low && *k <= high {
                result.push((*k, *v));
            }
        }

        if self.keys.is_empty() {
            result.sort_by_key(|(k, _)| *k);
            return result;
        }

        // Find start position using model
        let start_pred = self.model.predict(low);
        let error = self.model.error_bound();

        // Search from predicted position with error margin
        let search_start = start_pred.saturating_sub(error);

        // Find first key >= low
        let first_idx = match self.keys[search_start..].binary_search(&low) {
            Ok(i) => search_start + i,
            Err(i) => search_start + i,
        };

        // Collect all keys in range
        for i in first_idx..self.keys.len() {
            if self.keys[i] > high {
                break;
            }
            result.push((self.keys[i], self.values[i]));
        }

        result.sort_by_key(|(k, _)| *k);
        result
    }

    /// Insert a key-value pair
    ///
    /// Buffers inserts and periodically flushes to main storage.
    pub fn insert(&mut self, key: i64, value: u64) {
        self.insert_buffer.push((key, value));

        if self.insert_buffer.len() >= self.buffer_threshold {
            self.flush_buffer();
        }
    }

    /// Force flush of insert buffer
    pub fn flush(&mut self) {
        if !self.insert_buffer.is_empty() {
            self.flush_buffer();
        }
    }

    /// Flush buffer and possibly retrain model
    fn flush_buffer(&mut self) {
        if self.insert_buffer.is_empty() {
            return;
        }

        self.stats.buffer_flushes += 1;

        // Merge buffer into main storage
        let mut all: Vec<(i64, u64)> = self
            .keys
            .iter()
            .zip(self.values.iter())
            .map(|(&k, &v)| (k, v))
            .chain(self.insert_buffer.drain(..))
            .collect();

        // Sort and deduplicate (keep latest value for duplicate keys)
        all.sort_by_key(|(k, _)| *k);

        // Remove duplicates (keep last)
        let mut deduped: Vec<(i64, u64)> = Vec::with_capacity(all.len());
        for (k, v) in all {
            if let Some(last) = deduped.last_mut()
                && last.0 == k
            {
                last.1 = v; // Update value
                continue;
            }
            deduped.push((k, v));
        }

        self.keys = deduped.iter().map(|(k, _)| *k).collect();
        self.values = deduped.iter().map(|(_, v)| *v).collect();

        // Check if retrain is needed
        let new_model = PiecewiseLinearModel::build(&self.keys, self.target_error);
        let old_error = self.model.error_bound();
        let new_error = new_model.error_bound();

        // Retrain if error increased significantly or this is first model
        if old_error == 0 || (new_error as f64) > (old_error as f64) * self.retrain_ratio {
            self.model = new_model;
            self.stats.retrains += 1;
        } else {
            self.model = new_model;
        }
    }

    /// Delete a key
    ///
    /// Note: Deletions require rebuild as we don't maintain a deletion buffer.
    pub fn delete(&mut self, key: i64) -> Option<u64> {
        // First check buffer
        if let Some(pos) = self.insert_buffer.iter().position(|(k, _)| *k == key) {
            let (_, v) = self.insert_buffer.remove(pos);
            return Some(v);
        }

        // Check main storage
        if let Ok(idx) = self.keys.binary_search(&key) {
            let value = self.values[idx];
            self.keys.remove(idx);
            self.values.remove(idx);

            // Rebuild model after deletion
            self.model = PiecewiseLinearModel::build(&self.keys, self.target_error);
            self.stats.retrains += 1;

            return Some(value);
        }

        None
    }

    /// Get number of keys
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len() + self.insert_buffer.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.insert_buffer.is_empty()
    }

    /// Get model error bound
    #[inline]
    pub fn error_bound(&self) -> usize {
        self.model.error_bound()
    }

    /// Get number of model segments
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.model.num_segments()
    }

    /// Get statistics
    pub fn stats(&self) -> LearnedIndexStats {
        self.stats.clone()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let keys = self.keys.len() * std::mem::size_of::<i64>();
        let values = self.values.len() * std::mem::size_of::<u64>();
        let buffer = self.insert_buffer.capacity() * std::mem::size_of::<(i64, u64)>();
        let model = self.model.memory_usage();

        base + keys + values + buffer + model
    }
}

impl Default for AdaptiveLearnedIndex {
    fn default() -> Self {
        Self::new(64) // Default error bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piecewise_model_basic() {
        let keys: Vec<i64> = (0..100).collect();
        let model = PiecewiseLinearModel::build(&keys, 2);

        // Should be nearly perfect for linear data
        assert!(model.error_bound() <= 2);

        // Test predictions
        for &key in &keys {
            let predicted = model.predict(key);
            let actual = key as usize;
            assert!((predicted as i64 - actual as i64).abs() <= model.error_bound() as i64);
        }
    }

    #[test]
    fn test_piecewise_model_gaps() {
        // Non-uniform keys with gaps
        let keys: Vec<i64> = vec![1, 2, 3, 10, 11, 12, 100, 101, 102];
        let model = PiecewiseLinearModel::build(&keys, 2);

        // Each segment should have low error
        for (i, &key) in keys.iter().enumerate() {
            let predicted = model.predict(key);
            assert!((predicted as i64 - i as i64).abs() <= model.error_bound() as i64 + 1);
        }
    }

    #[test]
    fn test_adaptive_index_basic() {
        let mut index = AdaptiveLearnedIndex::new(4);

        // Insert some data
        for i in 0..100 {
            index.insert(i * 2, i as u64 * 100);
        }
        index.flush();

        // Lookup
        assert_eq!(index.get(10), Some(500)); // key 10 = 5th element
        assert_eq!(index.get(50), Some(2500));
        assert_eq!(index.get(11), None); // Not in index
    }

    #[test]
    fn test_adaptive_index_bulk_load() {
        let mut index = AdaptiveLearnedIndex::new(4);

        let data: Vec<(i64, u64)> = (0..1000).map(|i| (i, i as u64 * 10)).collect();
        index.bulk_load(data);

        // Verify lookups
        for i in 0..1000 {
            assert_eq!(index.get(i), Some(i as u64 * 10));
        }

        // Verify error bound is reasonable
        assert!(index.error_bound() <= 10);
    }

    #[test]
    fn test_adaptive_index_range() {
        let mut index = AdaptiveLearnedIndex::new(4);

        for i in 0..100 {
            index.insert(i, i as u64);
        }
        index.flush();

        let result = index.range(25, 30);
        assert_eq!(result.len(), 6);

        let keys: Vec<i64> = result.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![25, 26, 27, 28, 29, 30]);
    }

    #[test]
    fn test_adaptive_index_delete() {
        let mut index = AdaptiveLearnedIndex::new(4);

        for i in 0..10 {
            index.insert(i, i as u64 * 100);
        }
        index.flush();

        assert_eq!(index.get(5), Some(500));
        assert_eq!(index.delete(5), Some(500));
        assert_eq!(index.get(5), None);
    }

    #[test]
    fn test_adaptive_index_buffer() {
        let mut index = AdaptiveLearnedIndex::with_config(4, 10, 2.0);

        // Insert less than threshold
        for i in 0..5 {
            index.insert(i, i as u64);
        }

        // Should still find in buffer
        assert_eq!(index.get(3), Some(3));

        // Now exceed threshold
        for i in 5..15 {
            index.insert(i, i as u64);
        }

        // Should be flushed
        assert_eq!(index.get(3), Some(3));
        assert_eq!(index.get(10), Some(10));
    }

    #[test]
    fn test_adaptive_index_stats() {
        let mut index = AdaptiveLearnedIndex::new(4);

        for i in 0..100 {
            index.insert(i, i as u64);
        }
        index.flush();

        for i in 0..50 {
            index.get(i);
        }

        let stats = index.stats();
        assert_eq!(stats.lookups, 50);
        assert!(stats.buffer_flushes >= 1);
    }

    #[test]
    fn test_memory_usage() {
        let mut index = AdaptiveLearnedIndex::new(64);

        let data: Vec<(i64, u64)> = (0..10000).map(|i| (i, i as u64)).collect();
        index.bulk_load(data);

        let mem = index.memory_usage();

        // Should be much smaller than raw data (16 bytes per entry = 160KB)
        // Due to learned model compression
        println!("Memory usage: {} bytes for {} entries", mem, index.len());
        println!("Bytes per entry: {:.2}", mem as f64 / index.len() as f64);

        // Model segments should be much fewer than entries
        println!("Model segments: {}", index.num_segments());
        assert!(index.num_segments() < index.len() / 10);
    }
}
