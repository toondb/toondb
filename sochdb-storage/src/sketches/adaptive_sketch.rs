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

//! Adaptive Sketch - Memory-Efficient Latency Tracking
//!
//! Starts as a lightweight sparse buffer for low-traffic buckets,
//! automatically upgrades to DDSketch when sample count exceeds threshold.
//!
//! ## Memory Characteristics
//! - Sparse mode: ~50 bytes for low-traffic buckets (1-2 spans)
//! - Dense mode: ~200 bytes for high-traffic buckets (DDSketch)
//!
//! ## Why This Matters
//! The "Long Tail" problem: millions of unique (minute, project, model) combinations
//! where most have only 1-2 spans. Without adaptation, each would consume 17KB.

use crate::sketches::DDSketch;

/// Threshold for upgrading from Sparse to Dense
const SPARSE_LIMIT: usize = 128;

/// Adaptive sketch that starts lightweight and upgrades when needed
#[derive(Debug, Clone)]
pub enum AdaptiveSketch {
    /// For low-traffic buckets: store raw samples (minimal memory)
    Sparse(SparseBuffer),
    /// For high-traffic buckets: use DDSketch (bounded memory)
    Dense(DDSketch),
}

/// Lightweight buffer for low-traffic scenarios
#[derive(Debug, Clone)]
pub struct SparseBuffer {
    samples: Vec<f64>,
    min: f64,
    max: f64,
    sum: f64,
}

impl SparseBuffer {
    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(16), // Start small
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
        }
    }

    #[inline]
    pub fn insert(&mut self, value: f64) {
        self.samples.push(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
    }

    pub fn count(&self) -> usize {
        self.samples.len()
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f64
        }
    }

    /// Calculate percentile by sorting (only called when querying, not on insert)
    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get all percentiles at once (more efficient)
    pub fn percentiles(&self) -> SketchPercentiles {
        if self.samples.is_empty() {
            return SketchPercentiles::default();
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        SketchPercentiles {
            p50: sorted[(0.50 * (len - 1) as f64).round() as usize],
            p90: sorted[(0.90 * (len - 1) as f64).round() as usize],
            p95: sorted[(0.95 * (len - 1) as f64).round() as usize],
            p99: sorted[(0.99 * (len - 1) as f64).round() as usize],
        }
    }

    /// Convert to DDSketch (used during upgrade)
    pub fn to_ddsketch(&self) -> DDSketch {
        let mut sketch = DDSketch::default_accuracy();
        for &value in &self.samples {
            sketch.add(value);
        }
        sketch
    }

    /// Drain samples for upgrade
    pub fn drain(&mut self) -> impl Iterator<Item = f64> + '_ {
        self.samples.drain(..)
    }

    /// Merge another buffer
    pub fn merge(&mut self, other: &SparseBuffer) {
        self.samples.extend_from_slice(&other.samples);
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum += other.sum;
    }
}

impl Default for SparseBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Percentile results
#[derive(Debug, Clone, Default)]
pub struct SketchPercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl AdaptiveSketch {
    /// Create a new adaptive sketch (starts sparse)
    pub fn new() -> Self {
        AdaptiveSketch::Sparse(SparseBuffer::new())
    }

    /// Create with custom sparse capacity
    pub fn with_capacity(capacity: usize) -> Self {
        AdaptiveSketch::Sparse(SparseBuffer::with_capacity(capacity))
    }

    /// Insert a value, automatically upgrading if needed
    #[inline]
    pub fn insert(&mut self, value: f64) {
        match self {
            AdaptiveSketch::Sparse(buffer) => {
                buffer.insert(value);

                // Upgrade to Dense if exceeding limit
                if buffer.count() >= SPARSE_LIMIT {
                    let sketch = buffer.to_ddsketch();
                    *self = AdaptiveSketch::Dense(sketch);
                }
            }
            AdaptiveSketch::Dense(sketch) => {
                sketch.add(value);
            }
        }
    }

    /// Get sample count
    pub fn count(&self) -> u64 {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.count() as u64,
            AdaptiveSketch::Dense(sketch) => sketch.count(),
        }
    }

    /// Get minimum value
    pub fn min(&self) -> f64 {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.min(),
            AdaptiveSketch::Dense(sketch) => sketch.min(),
        }
    }

    /// Get maximum value
    pub fn max(&self) -> f64 {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.max(),
            AdaptiveSketch::Dense(sketch) => sketch.max(),
        }
    }

    /// Get sum
    pub fn sum(&self) -> f64 {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.sum(),
            AdaptiveSketch::Dense(sketch) => sketch.sum(),
        }
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.mean(),
            AdaptiveSketch::Dense(sketch) => sketch.mean(),
        }
    }

    /// Get percentile
    pub fn percentile(&self, p: f64) -> f64 {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.percentile(p),
            AdaptiveSketch::Dense(sketch) => sketch.quantile(p / 100.0),
        }
    }

    /// Get all standard percentiles
    pub fn percentiles(&self) -> SketchPercentiles {
        match self {
            AdaptiveSketch::Sparse(buffer) => buffer.percentiles(),
            AdaptiveSketch::Dense(sketch) => {
                let p = sketch.percentiles();
                SketchPercentiles {
                    p50: p.p50,
                    p90: p.p90,
                    p95: p.p95,
                    p99: p.p99,
                }
            }
        }
    }

    /// Check if in dense mode
    pub fn is_dense(&self) -> bool {
        matches!(self, AdaptiveSketch::Dense(_))
    }

    /// Force upgrade to dense mode
    pub fn upgrade_to_dense(&mut self) {
        if let AdaptiveSketch::Sparse(buffer) = self {
            let sketch = buffer.to_ddsketch();
            *self = AdaptiveSketch::Dense(sketch);
        }
    }

    /// Merge another sketch
    pub fn merge(&mut self, other: &AdaptiveSketch) {
        // Handle each combination of sparse/dense
        match (&*self, other) {
            (AdaptiveSketch::Sparse(_), AdaptiveSketch::Sparse(b)) => {
                if let AdaptiveSketch::Sparse(a) = self {
                    a.merge(b);
                    // Check if we need to upgrade after merge
                    if a.count() >= SPARSE_LIMIT {
                        let sketch = a.to_ddsketch();
                        *self = AdaptiveSketch::Dense(sketch);
                    }
                }
            }
            (AdaptiveSketch::Dense(_), AdaptiveSketch::Dense(b)) => {
                if let AdaptiveSketch::Dense(a) = self {
                    a.merge(b);
                }
            }
            (AdaptiveSketch::Sparse(_), AdaptiveSketch::Dense(dense)) => {
                // Upgrade sparse to dense, then merge
                if let AdaptiveSketch::Sparse(sparse) = self {
                    let mut sketch = sparse.to_ddsketch();
                    sketch.merge(dense);
                    *self = AdaptiveSketch::Dense(sketch);
                }
            }
            (AdaptiveSketch::Dense(_), AdaptiveSketch::Sparse(sparse)) => {
                // Convert sparse to dense and merge
                if let AdaptiveSketch::Dense(dense) = self {
                    let other_sketch = sparse.to_ddsketch();
                    dense.merge(&other_sketch);
                }
            }
        }
    }

    /// Approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            AdaptiveSketch::Sparse(buffer) => {
                std::mem::size_of::<SparseBuffer>() + buffer.samples.capacity() * 8
            }
            AdaptiveSketch::Dense(_) => {
                // DDSketch uses ~200 bytes for default config
                200
            }
        }
    }
}

impl Default for AdaptiveSketch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_to_dense_upgrade() {
        let mut sketch = AdaptiveSketch::new();

        // Should start sparse
        assert!(!sketch.is_dense());

        // Insert values up to limit
        for i in 0..SPARSE_LIMIT {
            sketch.insert(i as f64);
            if i < SPARSE_LIMIT - 1 {
                assert!(!sketch.is_dense(), "Should still be sparse at {}", i);
            }
        }

        // Should now be dense
        assert!(sketch.is_dense());
        assert_eq!(sketch.count(), SPARSE_LIMIT as u64);
    }

    #[test]
    fn test_percentiles_sparse() {
        let mut sketch = AdaptiveSketch::new();

        for i in 1..=100 {
            sketch.insert(i as f64);
        }

        let p = sketch.percentiles();
        assert!(p.p50 >= 49.0 && p.p50 <= 51.0);
        assert!(p.p99 >= 98.0);
    }

    #[test]
    fn test_percentiles_dense() {
        let mut sketch = AdaptiveSketch::new();

        // Force upgrade
        for i in 1..=200 {
            sketch.insert(i as f64);
        }

        assert!(sketch.is_dense());

        let p = sketch.percentiles();
        assert!(p.p50 >= 90.0 && p.p50 <= 110.0);
    }

    #[test]
    fn test_merge_sparse_sparse() {
        let mut a = AdaptiveSketch::new();
        let mut b = AdaptiveSketch::new();

        for i in 1..=50 {
            a.insert(i as f64);
        }
        for i in 51..=100 {
            b.insert(i as f64);
        }

        a.merge(&b);
        assert_eq!(a.count(), 100);
    }

    #[test]
    fn test_merge_upgrades_when_needed() {
        let mut a = AdaptiveSketch::new();
        let mut b = AdaptiveSketch::new();

        // Each has 70 samples (under limit)
        for i in 1..=70 {
            a.insert(i as f64);
        }
        for i in 71..=140 {
            b.insert(i as f64);
        }

        assert!(!a.is_dense());
        assert!(!b.is_dense());

        // Merge should upgrade because 140 > 128
        a.merge(&b);
        assert!(a.is_dense());
        assert_eq!(a.count(), 140);
    }

    #[test]
    fn test_memory_usage() {
        let sparse = AdaptiveSketch::new();
        let mut dense = AdaptiveSketch::new();
        dense.upgrade_to_dense();

        // Sparse should use less memory
        assert!(sparse.memory_usage() < dense.memory_usage());
    }
}
