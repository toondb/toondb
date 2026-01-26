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

//! Count-Min Sketch - Frequency Estimation
//!
//! A probabilistic data structure for estimating element frequencies with:
//! - O(1) update time per element
//! - O(w × d) space where w = width, d = depth
//! - Approximate frequency with error bounds
//!
//! Reference: Count-Min Sketch (Cormode & Muthukrishnan, 2005)
//! https://theory.stanford.edu/~tim/s15/l/l2.pdf

use std::hash::Hash;

/// Count-Min Sketch for frequency estimation
///
/// Error bounds:
///     P(estimate > true_count + εN) < δ
///     Where: w = ⌈e/ε⌉, d = ⌈ln(1/δ)⌉
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    /// Width of each row
    width: usize,
    /// Number of rows (depth)
    depth: usize,
    /// Counter matrix [depth][width]
    counters: Vec<Vec<u64>>,
    /// Seeds for hash functions
    seeds: Vec<u64>,
    /// Total count of all additions
    total: u64,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch
    ///
    /// # Arguments
    /// * `epsilon` - Relative error (e.g., 0.001 for 0.1% error)
    /// * `delta` - Failure probability (e.g., 0.01 for 99% confidence)
    pub fn new(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;

        // Generate random seeds
        let mut seeds = Vec::with_capacity(depth);
        let mut seed = 0x517cc1b727220a95u64; // Initial seed
        for _ in 0..depth {
            seed = seed.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(1);
            seeds.push(seed);
        }

        Self {
            width,
            depth,
            counters: vec![vec![0; width]; depth],
            seeds,
            total: 0,
        }
    }

    /// Create with default parameters (0.1% error, 99% confidence)
    pub fn default_params() -> Self {
        Self::new(0.001, 0.01)
    }

    /// Create with specified dimensions
    pub fn with_dimensions(width: usize, depth: usize) -> Self {
        let mut seeds = Vec::with_capacity(depth);
        let mut seed = 0x517cc1b727220a95u64;
        for _ in 0..depth {
            seed = seed.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(1);
            seeds.push(seed);
        }

        Self {
            width,
            depth,
            counters: vec![vec![0; width]; depth],
            seeds,
            total: 0,
        }
    }

    /// Hash an item with a specific seed
    #[inline]
    fn hash_with_seed<T: Hash>(&self, item: &T, seed: u64) -> usize {
        // Use seed to create different hash functions
        let state = ahash::RandomState::with_seeds(seed, seed, seed, seed);
        let hash = state.hash_one(item);
        (hash as usize) % self.width
    }

    /// Add an item with count 1
    #[inline]
    pub fn add<T: Hash>(&mut self, item: &T) {
        self.add_count(item, 1);
    }

    /// Add an item with specified count
    pub fn add_count<T: Hash>(&mut self, item: &T, count: u64) {
        for (i, &seed) in self.seeds.iter().enumerate() {
            let j = self.hash_with_seed(item, seed);
            self.counters[i][j] = self.counters[i][j].saturating_add(count);
        }
        self.total = self.total.saturating_add(count);
    }

    /// Estimate frequency of an item
    pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
        self.seeds
            .iter()
            .enumerate()
            .map(|(i, &seed)| {
                let j = self.hash_with_seed(item, seed);
                self.counters[i][j]
            })
            .min()
            .unwrap_or(0)
    }

    /// Merge another sketch into this one
    ///
    /// Useful for aggregating across time buckets
    pub fn merge(&mut self, other: &CountMinSketch) {
        assert_eq!(self.width, other.width, "Width mismatch");
        assert_eq!(self.depth, other.depth, "Depth mismatch");

        for i in 0..self.depth {
            for j in 0..self.width {
                self.counters[i][j] = self.counters[i][j].saturating_add(other.counters[i][j]);
            }
        }
        self.total = self.total.saturating_add(other.total);
    }

    /// Get total count
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Get width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Clear all counters
    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0);
        }
        self.total = 0;
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.depth * self.width * std::mem::size_of::<u64>()
            + self.seeds.len() * std::mem::size_of::<u64>()
    }
}

impl Default for CountMinSketch {
    fn default() -> Self {
        Self::default_params()
    }
}

/// Top-K tracker using Space-Saving algorithm
///
/// Maintains approximate top-K frequent items with guaranteed accuracy:
/// True top-K items are included if their count > N/k
#[derive(Debug, Clone)]
pub struct TopKTracker<K: Hash + Eq + Clone> {
    /// Maximum items to track
    capacity: usize,
    /// Items with (count, error)
    items: std::collections::HashMap<K, (u64, u64)>,
}

impl<K: Hash + Eq + Clone> TopKTracker<K> {
    /// Create a new top-K tracker
    pub fn new(k: usize) -> Self {
        Self {
            capacity: k,
            items: std::collections::HashMap::with_capacity(k),
        }
    }

    /// Add an item observation
    pub fn add(&mut self, item: K) {
        self.add_count(item, 1);
    }

    /// Add multiple observations of an item
    pub fn add_count(&mut self, item: K, count: u64) {
        if let Some((c, _)) = self.items.get_mut(&item) {
            *c += count;
        } else if self.items.len() < self.capacity {
            self.items.insert(item, (count, 0));
        } else {
            // Find minimum count item
            let min_entry = self
                .items
                .iter()
                .min_by_key(|(_, (c, _))| *c)
                .map(|(k, v)| (k.clone(), *v));

            if let Some((min_key, (min_count, _))) = min_entry {
                self.items.remove(&min_key);
                self.items.insert(item, (min_count + count, min_count));
            }
        }
    }

    /// Get top-K items sorted by count (descending)
    pub fn top_k(&self) -> Vec<(K, u64)> {
        let mut items: Vec<_> = self
            .items
            .iter()
            .map(|(k, (c, _))| (k.clone(), *c))
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items
    }

    /// Get count for a specific item
    pub fn get(&self, item: &K) -> Option<u64> {
        self.items.get(item).map(|(c, _)| *c)
    }

    /// Get count and error bound for an item
    pub fn get_with_error(&self, item: &K) -> Option<(u64, u64)> {
        self.items.get(item).copied()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get number of tracked items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Merge another tracker into this one
    pub fn merge(&mut self, other: &TopKTracker<K>) {
        for (item, (count, _)) in &other.items {
            self.add_count(item.clone(), *count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_counting() {
        let mut cms = CountMinSketch::new(0.01, 0.01);

        // Add items
        for _ in 0..100 {
            cms.add(&"hello");
        }
        for _ in 0..50 {
            cms.add(&"world");
        }

        assert!(cms.estimate(&"hello") >= 100);
        assert!(cms.estimate(&"world") >= 50);
        assert_eq!(cms.estimate(&"unknown"), 0);
    }

    #[test]
    fn test_merge() {
        let mut cms1 = CountMinSketch::with_dimensions(1000, 5);
        let mut cms2 = CountMinSketch::with_dimensions(1000, 5);

        for _ in 0..50 {
            cms1.add(&"hello");
        }
        for _ in 0..50 {
            cms2.add(&"hello");
        }

        cms1.merge(&cms2);

        assert!(cms1.estimate(&"hello") >= 100);
    }

    #[test]
    fn test_top_k() {
        let mut tracker = TopKTracker::new(3);

        // Add items with different frequencies
        for _ in 0..100 {
            tracker.add("a");
        }
        for _ in 0..50 {
            tracker.add("b");
        }
        for _ in 0..25 {
            tracker.add("c");
        }
        // When we add "d", it will evict "c" (min) and get count = 25 + 10 = 35
        for _ in 0..10 {
            tracker.add("d");
        }

        let top = tracker.top_k();

        // Should have top 3
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "a");
        assert_eq!(top[1].0, "b");
        // The third item is "d" (35) not "c" (25) due to Space-Saving algorithm
        assert_eq!(top[2].0, "d");
    }

    #[test]
    fn test_top_k_merge() {
        let mut tracker1 = TopKTracker::new(5);
        let mut tracker2 = TopKTracker::new(5);

        for _ in 0..50 {
            tracker1.add("a");
        }
        for _ in 0..50 {
            tracker2.add("a");
        }

        tracker1.merge(&tracker2);

        assert!(tracker1.get(&"a").unwrap() >= 100);
    }
}
