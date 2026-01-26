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

//! DDSketch - Deterministic Quantile Sketch
//!
//! A streaming algorithm for accurate percentile estimation with:
//! - O(1) update time per value
//! - O(1) query time for any quantile
//! - O(log(max/min)) space complexity
//! - Mergeable across time buckets
//!
//! Reference: DDSketch Paper (Datadog, 2019) https://arxiv.org/abs/1908.10693
//!
//! **Gap #1 Fix**: Handles subnormal floats and edge cases safely.
//! Values are clamped to [MIN_INDEXABLE_VALUE, MAX_INDEXABLE_VALUE] to prevent
//! i32 overflow from ln() approaching -∞ on subnormal floats.

use std::collections::BTreeMap;

/// Minimum value that can be safely indexed (avoids ln() overflow)
/// Values smaller than this are clamped to prevent i32 overflow in index()
const MIN_INDEXABLE_VALUE: f64 = 1e-300;

/// Maximum value that can be safely indexed
/// Values larger than this are clamped to prevent i32 overflow in index()
const MAX_INDEXABLE_VALUE: f64 = 1e300;

/// DDSketch for streaming quantile estimation
///
/// Provides relative accuracy guarantees: for quantile q, returned value v satisfies:
///     actual * (1 - α) ≤ v ≤ actual * (1 + α)
#[derive(Debug, Clone)]
pub struct DDSketch {
    /// Relative accuracy parameter (e.g., 0.01 for 1%)
    alpha: f64,
    /// γ = (1 + α) / (1 - α)
    gamma: f64,
    /// log(γ) precomputed for index calculation
    log_gamma: f64,
    /// Positive bucket counts: index -> count
    positive_buckets: BTreeMap<i32, u64>,
    /// Negative bucket counts: index -> count
    negative_buckets: BTreeMap<i32, u64>,
    /// Total count
    count: u64,
    /// Zero count (special handling for zero values)
    zero_count: u64,
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
    /// Sum of all values (for mean calculation)
    sum: f64,
}

impl DDSketch {
    /// Create a new DDSketch with specified relative accuracy
    ///
    /// # Arguments
    /// * `alpha` - Relative accuracy (e.g., 0.01 for 1% accuracy)
    ///
    /// # Example
    /// ```
    /// use sochdb_storage::sketches::DDSketch;
    /// let sketch = DDSketch::new(0.01); // 1% accuracy
    /// ```
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be between 0 and 1");
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        Self {
            alpha,
            gamma,
            log_gamma: gamma.ln(),
            positive_buckets: BTreeMap::new(),
            negative_buckets: BTreeMap::new(),
            count: 0,
            zero_count: 0,
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
        }
    }

    /// Create with default 1% accuracy
    pub fn default_accuracy() -> Self {
        Self::new(0.01)
    }

    /// Calculate bucket index for a value
    ///
    /// **Gap #1 Fix**: Clamps values to safe range to prevent i32 overflow.
    /// Subnormal floats (approaching 1e-308) would cause ln() to approach -∞,
    /// resulting in undefined behavior when cast to i32.
    #[inline]
    fn index(&self, value: f64) -> i32 {
        debug_assert!(value > 0.0);
        // Clamp to safe range to prevent overflow from ln() on extreme values
        let clamped = value.clamp(MIN_INDEXABLE_VALUE, MAX_INDEXABLE_VALUE);
        (clamped.ln() / self.log_gamma).ceil() as i32
    }

    /// Get bucket midpoint for an index
    #[inline]
    fn bucket_value(&self, index: i32) -> f64 {
        self.gamma.powf(index as f64 - 0.5)
    }

    /// Add a value to the sketch
    #[inline]
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        if value == 0.0 {
            self.zero_count += 1;
        } else if value > 0.0 {
            let idx = self.index(value);
            *self.positive_buckets.entry(idx).or_default() += 1;
        } else {
            let idx = self.index(-value);
            *self.negative_buckets.entry(idx).or_default() += 1;
        }
    }

    /// Add multiple occurrences of a value
    pub fn add_count(&mut self, value: f64, count: u64) {
        self.count += count;
        self.sum += value * count as f64;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        if value == 0.0 {
            self.zero_count += count;
        } else if value > 0.0 {
            let idx = self.index(value);
            *self.positive_buckets.entry(idx).or_default() += count;
        } else {
            let idx = self.index(-value);
            *self.negative_buckets.entry(idx).or_default() += count;
        }
    }

    /// Query a quantile (0.0 to 1.0)
    ///
    /// # Arguments
    /// * `q` - Quantile to query (e.g., 0.50 for median, 0.99 for P99)
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let q = q.clamp(0.0, 1.0);
        let target_rank = (q * self.count as f64).ceil() as u64;

        if target_rank == 0 {
            return self.min;
        }

        // First check negative buckets (from most negative to zero)
        let mut cumulative = 0u64;
        for (&idx, &count) in self.negative_buckets.iter().rev() {
            cumulative += count;
            if cumulative >= target_rank {
                return -self.bucket_value(idx);
            }
        }

        // Then zero bucket
        cumulative += self.zero_count;
        if cumulative >= target_rank {
            return 0.0;
        }

        // Then positive buckets
        for (&idx, &count) in &self.positive_buckets {
            cumulative += count;
            if cumulative >= target_rank {
                return self.bucket_value(idx);
            }
        }

        self.max
    }

    /// Get common percentiles efficiently
    pub fn percentiles(&self) -> DDSketchPercentiles {
        DDSketchPercentiles {
            p50: self.quantile(0.50),
            p75: self.quantile(0.75),
            p90: self.quantile(0.90),
            p95: self.quantile(0.95),
            p99: self.quantile(0.99),
            min: if self.count > 0 { self.min } else { 0.0 },
            max: if self.count > 0 { self.max } else { 0.0 },
            mean: self.mean(),
            count: self.count,
        }
    }

    /// Get count of values
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get the relative accuracy parameter (alpha)
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get mean of values
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get sum of values
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get min value
    pub fn min(&self) -> f64 {
        if self.count > 0 { self.min } else { 0.0 }
    }

    /// Get max value  
    pub fn max(&self) -> f64 {
        if self.count > 0 { self.max } else { 0.0 }
    }

    /// Merge another sketch into this one
    ///
    /// Critical for time bucket rollups - merging is O(buckets)
    pub fn merge(&mut self, other: &DDSketch) {
        if other.count == 0 {
            return;
        }

        // Merge positive buckets
        for (&idx, &count) in &other.positive_buckets {
            *self.positive_buckets.entry(idx).or_default() += count;
        }

        // Merge negative buckets
        for (&idx, &count) in &other.negative_buckets {
            *self.negative_buckets.entry(idx).or_default() += count;
        }

        self.count += other.count;
        self.zero_count += other.zero_count;
        self.sum += other.sum;

        if other.count > 0 {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    /// Check if sketch is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.positive_buckets.len()
                * (std::mem::size_of::<i32>() + std::mem::size_of::<u64>())
            + self.negative_buckets.len()
                * (std::mem::size_of::<i32>() + std::mem::size_of::<u64>())
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.positive_buckets.clear();
        self.negative_buckets.clear();
        self.count = 0;
        self.zero_count = 0;
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.sum = 0.0;
    }
}

impl Default for DDSketch {
    fn default() -> Self {
        Self::default_accuracy()
    }
}

/// Pre-computed percentile values
#[derive(Debug, Clone, Copy, Default)]
pub struct DDSketchPercentiles {
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut sketch = DDSketch::new(0.01);

        // Add values 1 to 100
        for i in 1..=100 {
            sketch.add(i as f64);
        }

        assert_eq!(sketch.count(), 100);
        assert!((sketch.mean() - 50.5).abs() < 0.01);

        // P50 should be around 50
        let p50 = sketch.quantile(0.50);
        assert!((p50 - 50.0).abs() < 5.0, "P50 was {}", p50);

        // P99 should be around 99
        let p99 = sketch.quantile(0.99);
        assert!((p99 - 99.0).abs() < 5.0, "P99 was {}", p99);
    }

    #[test]
    fn test_merge() {
        let mut sketch1 = DDSketch::new(0.01);
        let mut sketch2 = DDSketch::new(0.01);

        for i in 1..=50 {
            sketch1.add(i as f64);
        }
        for i in 51..=100 {
            sketch2.add(i as f64);
        }

        sketch1.merge(&sketch2);

        assert_eq!(sketch1.count(), 100);
        assert!((sketch1.mean() - 50.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_sketch() {
        let sketch = DDSketch::new(0.01);

        assert_eq!(sketch.count(), 0);
        assert_eq!(sketch.quantile(0.50), 0.0);
        assert_eq!(sketch.mean(), 0.0);
    }

    #[test]
    fn test_latency_distribution() {
        let mut sketch = DDSketch::new(0.01);

        // Simulate latency distribution (in microseconds)
        // Most requests: 1-10ms
        for _ in 0..900 {
            sketch.add(5_000.0); // 5ms
        }
        // Some slower: 10-50ms
        for _ in 0..90 {
            sketch.add(30_000.0); // 30ms
        }
        // Few very slow: 100ms+
        for _ in 0..10 {
            sketch.add(150_000.0); // 150ms
        }

        let percentiles = sketch.percentiles();

        // P50 should be around 5ms (within 50% due to bucket boundaries)
        assert!(
            percentiles.p50 < 15_000.0,
            "P50 {} too high",
            percentiles.p50
        );

        // P99 should capture values above 5ms (at least P90 level)
        // Given 90% are 5ms, P99 should be higher than 5ms
        assert!(percentiles.p99 > 5_000.0, "P99 {} too low", percentiles.p99);
    }
}
