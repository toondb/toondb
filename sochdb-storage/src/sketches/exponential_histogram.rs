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

//! Exponential Histogram
//!
//! A histogram with exponentially-spaced bucket boundaries for natural
//! latency distributions. Used by OpenTelemetry Metrics for efficient
//! aggregation with O(1) merge operations.
//!
//! Reference: OpenTelemetry Exponential Histogram
//! https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram

/// Exponential histogram with configurable scale
///
/// Bucket boundaries follow exponential growth:
///     boundary[i] = base^i  where base = 2^(2^(-scale))
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ExponentialHistogram {
    /// Scale parameter (higher = more precision, more buckets)
    /// scale=0: base=2, buckets at [1,2), [2,4), [4,8), ...
    /// scale=3: base≈1.09, ~8 buckets per doubling
    scale: i32,
    /// Base = 2^(2^(-scale))
    base: f64,
    /// Log of base for index calculation
    log_base: f64,
    /// Positive bucket counts (index 0 = [1, base), index 1 = [base, base²), ...)
    positive_buckets: Vec<u64>,
    /// Negative bucket counts (mirrored)
    negative_buckets: Vec<u64>,
    /// Offset for positive buckets (allows representing smaller values)
    positive_offset: i32,
    /// Offset for negative buckets
    negative_offset: i32,
    /// Zero bucket count
    zero_count: u64,
    /// Zero bucket threshold (values in [-threshold, threshold] go to zero bucket)
    zero_threshold: f64,
    /// Total count
    count: u64,
    /// Sum of all values (for mean calculation)
    sum: f64,
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
}

impl ExponentialHistogram {
    /// Create a new exponential histogram
    ///
    /// # Arguments
    /// * `scale` - Scale parameter (-10 to 20 typical)
    ///   - scale=0: base=2.0, coarse buckets
    ///   - scale=3: base≈1.09, ~8 buckets per doubling (good default)
    ///   - scale=5: base≈1.02, ~32 buckets per doubling (high precision)
    pub fn new(scale: i32) -> Self {
        let base = 2.0_f64.powf(2.0_f64.powi(-scale));
        Self {
            scale,
            base,
            log_base: base.ln(),
            positive_buckets: Vec::new(),
            negative_buckets: Vec::new(),
            positive_offset: 0,
            negative_offset: 0,
            zero_count: 0,
            zero_threshold: 0.0,
            count: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Create with default scale (3 = good balance of precision and memory)
    pub fn default_scale() -> Self {
        Self::new(3)
    }

    /// Calculate bucket index for a positive value
    #[inline]
    fn bucket_index(&self, value: f64) -> i32 {
        debug_assert!(value > 0.0);
        // index = floor(log_base(value)) = floor(log2(value) × 2^scale)
        (value.log2() * (1 << self.scale) as f64).floor() as i32
    }

    /// Get the lower boundary of a bucket
    #[inline]
    fn bucket_lower_bound(&self, index: i32) -> f64 {
        self.base.powi(index)
    }

    /// Record a value
    pub fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        if value.abs() <= self.zero_threshold {
            self.zero_count += 1;
        } else if value > 0.0 {
            let idx = self.bucket_index(value);
            self.increment_positive(idx);
        } else {
            let idx = self.bucket_index(-value);
            self.increment_negative(idx);
        }
    }

    /// Increment positive bucket at index
    fn increment_positive(&mut self, idx: i32) {
        if self.positive_buckets.is_empty() {
            self.positive_offset = idx;
            self.positive_buckets.push(1);
        } else {
            let relative_idx = idx - self.positive_offset;
            if relative_idx < 0 {
                // Need to prepend buckets
                let prepend_count = (-relative_idx) as usize;
                let mut new_buckets = vec![0; prepend_count];
                new_buckets.append(&mut self.positive_buckets);
                self.positive_buckets = new_buckets;
                self.positive_offset = idx;
                self.positive_buckets[0] = 1;
            } else if relative_idx as usize >= self.positive_buckets.len() {
                // Need to append buckets
                self.positive_buckets.resize(relative_idx as usize + 1, 0);
                self.positive_buckets[relative_idx as usize] = 1;
            } else {
                self.positive_buckets[relative_idx as usize] += 1;
            }
        }
    }

    /// Increment negative bucket at index
    fn increment_negative(&mut self, idx: i32) {
        if self.negative_buckets.is_empty() {
            self.negative_offset = idx;
            self.negative_buckets.push(1);
        } else {
            let relative_idx = idx - self.negative_offset;
            if relative_idx < 0 {
                let prepend_count = (-relative_idx) as usize;
                let mut new_buckets = vec![0; prepend_count];
                new_buckets.append(&mut self.negative_buckets);
                self.negative_buckets = new_buckets;
                self.negative_offset = idx;
                self.negative_buckets[0] = 1;
            } else if relative_idx as usize >= self.negative_buckets.len() {
                self.negative_buckets.resize(relative_idx as usize + 1, 0);
                self.negative_buckets[relative_idx as usize] = 1;
            } else {
                self.negative_buckets[relative_idx as usize] += 1;
            }
        }
    }

    /// Merge another histogram into this one
    ///
    /// O(buckets) merge - critical for rollups
    pub fn merge(&mut self, other: &ExponentialHistogram) {
        if other.count == 0 {
            return;
        }

        // Must have same scale (or implement scale downgrade)
        assert_eq!(self.scale, other.scale, "Scale mismatch in merge");

        // Merge positive buckets
        for (i, &count) in other.positive_buckets.iter().enumerate() {
            if count > 0 {
                let idx = other.positive_offset + i as i32;
                self.add_positive_count(idx, count);
            }
        }

        // Merge negative buckets
        for (i, &count) in other.negative_buckets.iter().enumerate() {
            if count > 0 {
                let idx = other.negative_offset + i as i32;
                self.add_negative_count(idx, count);
            }
        }

        self.zero_count += other.zero_count;
        self.count += other.count;
        self.sum += other.sum;

        if other.count > 0 {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    /// Add count to positive bucket
    fn add_positive_count(&mut self, idx: i32, count: u64) {
        if self.positive_buckets.is_empty() {
            self.positive_offset = idx;
            self.positive_buckets.push(count);
        } else {
            let relative_idx = idx - self.positive_offset;
            if relative_idx < 0 {
                let prepend_count = (-relative_idx) as usize;
                let mut new_buckets = vec![0; prepend_count];
                new_buckets.append(&mut self.positive_buckets);
                self.positive_buckets = new_buckets;
                self.positive_offset = idx;
                self.positive_buckets[0] += count;
            } else if relative_idx as usize >= self.positive_buckets.len() {
                self.positive_buckets.resize(relative_idx as usize + 1, 0);
                self.positive_buckets[relative_idx as usize] += count;
            } else {
                self.positive_buckets[relative_idx as usize] += count;
            }
        }
    }

    /// Add count to negative bucket
    fn add_negative_count(&mut self, idx: i32, count: u64) {
        if self.negative_buckets.is_empty() {
            self.negative_offset = idx;
            self.negative_buckets.push(count);
        } else {
            let relative_idx = idx - self.negative_offset;
            if relative_idx < 0 {
                let prepend_count = (-relative_idx) as usize;
                let mut new_buckets = vec![0; prepend_count];
                new_buckets.append(&mut self.negative_buckets);
                self.negative_buckets = new_buckets;
                self.negative_offset = idx;
                self.negative_buckets[0] += count;
            } else if relative_idx as usize >= self.negative_buckets.len() {
                self.negative_buckets.resize(relative_idx as usize + 1, 0);
                self.negative_buckets[relative_idx as usize] += count;
            } else {
                self.negative_buckets[relative_idx as usize] += count;
            }
        }
    }

    /// Get count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get sum
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get min
    pub fn min(&self) -> f64 {
        if self.count > 0 { self.min } else { 0.0 }
    }

    /// Get max
    pub fn max(&self) -> f64 {
        if self.count > 0 { self.max } else { 0.0 }
    }

    /// Estimate a quantile
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let q = q.clamp(0.0, 1.0);
        let target_rank = (q * self.count as f64).ceil() as u64;
        let mut cumulative = 0u64;

        // Check negative buckets first (from most negative)
        for (i, &count) in self.negative_buckets.iter().enumerate().rev() {
            if count > 0 {
                cumulative += count;
                if cumulative >= target_rank {
                    let idx = self.negative_offset + i as i32;
                    return -self.bucket_lower_bound(idx);
                }
            }
        }

        // Check zero bucket
        cumulative += self.zero_count;
        if cumulative >= target_rank {
            return 0.0;
        }

        // Check positive buckets
        for (i, &count) in self.positive_buckets.iter().enumerate() {
            if count > 0 {
                cumulative += count;
                if cumulative >= target_rank {
                    let idx = self.positive_offset + i as i32;
                    // Return geometric mean of bucket (proper interpolation for log-spaced buckets)
                    // Geometric mean: sqrt(lower * upper) = lower * sqrt(base)
                    // This provides accurate quantile estimates for exponential histograms
                    let lower = self.bucket_lower_bound(idx);
                    let upper = self.bucket_lower_bound(idx + 1);
                    return (lower * upper).sqrt();
                }
            }
        }

        self.max
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.positive_buckets.clear();
        self.negative_buckets.clear();
        self.positive_offset = 0;
        self.negative_offset = 0;
        self.zero_count = 0;
        self.count = 0;
        self.sum = 0.0;
        self.min = f64::MAX;
        self.max = f64::MIN;
    }

    /// Get number of buckets used
    pub fn bucket_count(&self) -> usize {
        self.positive_buckets.len() + self.negative_buckets.len()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.positive_buckets.len() * std::mem::size_of::<u64>()
            + self.negative_buckets.len() * std::mem::size_of::<u64>()
    }
}

impl Default for ExponentialHistogram {
    fn default() -> Self {
        Self::default_scale()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_recording() {
        let mut hist = ExponentialHistogram::new(3);

        for i in 1..=100 {
            hist.record(i as f64);
        }

        assert_eq!(hist.count(), 100);
        assert!((hist.mean() - 50.5).abs() < 0.01);
        assert_eq!(hist.min(), 1.0);
        assert_eq!(hist.max(), 100.0);
    }

    #[test]
    fn test_merge() {
        let mut hist1 = ExponentialHistogram::new(3);
        let mut hist2 = ExponentialHistogram::new(3);

        for i in 1..=50 {
            hist1.record(i as f64);
        }
        for i in 51..=100 {
            hist2.record(i as f64);
        }

        hist1.merge(&hist2);

        assert_eq!(hist1.count(), 100);
        assert!((hist1.mean() - 50.5).abs() < 0.01);
    }

    #[test]
    fn test_quantiles() {
        let mut hist = ExponentialHistogram::new(3);

        // Uniform distribution 1-100
        for i in 1..=100 {
            hist.record(i as f64);
        }

        let p50 = hist.quantile(0.50);
        let p99 = hist.quantile(0.99);

        // Should be approximately correct (histogram bins introduce some error)
        // Exponential histograms have logarithmic buckets, so precision varies
        assert!(p50 > 30.0 && p50 < 70.0, "P50 was {}", p50);
        assert!(p99 > 80.0, "P99 was {}", p99);
    }

    #[test]
    fn test_latency_distribution() {
        let mut hist = ExponentialHistogram::new(3);

        // Simulate typical latency distribution
        for _ in 0..900 {
            hist.record(5.0); // 5ms - most requests
        }
        for _ in 0..90 {
            hist.record(50.0); // 50ms - some slower
        }
        for _ in 0..10 {
            hist.record(500.0); // 500ms - tail latency
        }

        assert_eq!(hist.count(), 1000);

        let p50 = hist.quantile(0.50);
        let p99 = hist.quantile(0.99);

        // P50 should be around 5ms (most data is there)
        assert!(p50 < 20.0, "P50 was {}", p50);

        // P99 should be above 5ms (captures slower requests)
        assert!(p99 > 5.0, "P99 was {}", p99);
    }
}
