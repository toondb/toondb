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

//! Prefetch-Optimized Sequential Scan
//!
//! This module provides software prefetching for memory-bandwidth-limited performance.
//! Without prefetching, CPU stalls waiting for data from memory.
//!
//! ## Problem Analysis
//!
//! Memory latency: ~70ns for cache miss
//! Memory bandwidth: ~50 GB/s
//!
//! For sequential scan at 50 GB/s, reading 8-byte values:
//! - Values per second: 50GB / 8B = 6.25 billion
//! - Without prefetch: 1 / 70ns = 14.3M values/sec
//! - **Gap: 437×**
//!
//! ## Prefetch Distance Calculation
//!
//! Let:
//! - L = memory latency (70ns)
//! - T = time per iteration (processing time)
//! - B = bytes per prefetch (cache line = 64B)
//!
//! Optimal prefetch distance: D = ⌈L/T⌉
//!
//! For T = 2ns (simple comparison):
//! D = ⌈70/2⌉ = 35 iterations
//!
//! Since cache line = 64 bytes = 8 i64s:
//! D_cache_lines = ⌈35/8⌉ = 5 cache lines = 320 bytes ahead
//!
//! ## Performance
//!
//! Expected: 20-40× improvement for full table scans

/// Cache line size in bytes
pub const CACHE_LINE_SIZE: usize = 64;

/// Number of i64 values per cache line
pub const I64_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 8;

/// Default prefetch distance (cache lines ahead)
pub const DEFAULT_PREFETCH_DISTANCE: usize = 8;

/// Prefetch hint types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchHint {
    /// T0: Prefetch to all cache levels (L1, L2, L3)
    /// Use for data that will be accessed multiple times
    T0,
    /// T1: Prefetch to L2 and higher
    /// Use for data that will be accessed once or twice
    T1,
    /// T2: Prefetch to L3 only
    /// Use for streaming data
    T2,
    /// NTA: Non-temporal access
    /// Use for data that won't be reused
    NonTemporal,
}

/// Issue a software prefetch
///
/// # Safety
/// This function is safe as prefetch instructions are hints and
/// don't cause faults for invalid addresses.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn prefetch(ptr: *const u8, hint: PrefetchHint) {
    unsafe {
        use std::arch::x86_64::*;
        match hint {
            PrefetchHint::T0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            PrefetchHint::T1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            PrefetchHint::T2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            PrefetchHint::NonTemporal => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn prefetch(_ptr: *const u8, _hint: PrefetchHint) {
    // No-op on non-x86 platforms
}

/// Prefetch-optimized column scanner for i64 values
pub struct PrefetchScanner<'a> {
    data: &'a [i64],
    prefetch_distance: usize,
    hint: PrefetchHint,
}

impl<'a> PrefetchScanner<'a> {
    /// Create scanner with automatic prefetch distance
    ///
    /// Tunes prefetch distance based on data size
    pub fn new(data: &'a [i64]) -> Self {
        // Tune prefetch distance based on data size
        let prefetch_distance = if data.len() > 1_000_000 {
            64 // Large scan: aggressive prefetch
        } else if data.len() > 10_000 {
            32 // Medium scan
        } else {
            16 // Small scan
        };

        Self {
            data,
            prefetch_distance,
            hint: PrefetchHint::T0,
        }
    }

    /// Create with custom prefetch distance
    pub fn with_distance(data: &'a [i64], prefetch_distance: usize) -> Self {
        Self {
            data,
            prefetch_distance,
            hint: PrefetchHint::T0,
        }
    }

    /// Set prefetch hint type
    pub fn with_hint(mut self, hint: PrefetchHint) -> Self {
        self.hint = hint;
        self
    }

    /// Sum all values with prefetching
    ///
    /// This is the core operation demonstrating prefetch benefits.
    #[inline(never)] // Prevent over-inlining
    pub fn sum(&self) -> i64 {
        let len = self.data.len();
        let ptr = self.data.as_ptr();

        // Initial prefetches to prime the cache
        let initial_prefetch =
            (self.prefetch_distance / I64_PER_CACHE_LINE).min(len / I64_PER_CACHE_LINE);
        for i in 0..initial_prefetch {
            prefetch(
                unsafe { ptr.add(i * I64_PER_CACHE_LINE) as *const u8 },
                self.hint,
            );
        }

        let mut total: i64 = 0;
        let mut i = 0;

        while i < len {
            // Prefetch ahead
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            // Process current cache line (8 values at a time for i64)
            let end = (i + I64_PER_CACHE_LINE).min(len);
            while i < end {
                total = total.wrapping_add(unsafe { *ptr.add(i) });
                i += 1;
            }
        }

        total
    }

    /// Sum with predicate filtering and prefetching
    pub fn sum_where<F>(&self, predicate: F) -> i64
    where
        F: Fn(i64) -> bool,
    {
        let len = self.data.len();
        let ptr = self.data.as_ptr();

        let mut total: i64 = 0;
        let mut i = 0;

        while i < len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            let value = unsafe { *ptr.add(i) };
            if predicate(value) {
                total = total.wrapping_add(value);
            }
            i += 1;
        }

        total
    }

    /// Count values matching predicate with prefetching
    pub fn count_where<F>(&self, predicate: F) -> usize
    where
        F: Fn(i64) -> bool,
    {
        let len = self.data.len();
        let ptr = self.data.as_ptr();

        let mut count = 0usize;
        let mut i = 0;

        while i < len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            let value = unsafe { *ptr.add(i) };
            if predicate(value) {
                count += 1;
            }
            i += 1;
        }

        count
    }

    /// Filter values with predicate, return qualifying indices
    pub fn filter<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(i64) -> bool,
    {
        let len = self.data.len();
        let ptr = self.data.as_ptr();

        // Estimate 10% selectivity for initial allocation
        let mut result = Vec::with_capacity(len / 10);
        let mut i = 0;

        while i < len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            let value = unsafe { *ptr.add(i) };
            if predicate(value) {
                result.push(i);
            }
            i += 1;
        }

        result
    }

    /// Find minimum value with prefetching
    pub fn min(&self) -> Option<i64> {
        if self.data.is_empty() {
            return None;
        }

        let len = self.data.len();
        let ptr = self.data.as_ptr();

        let mut min_val = unsafe { *ptr };
        let mut i = 1;

        while i < len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            let value = unsafe { *ptr.add(i) };
            if value < min_val {
                min_val = value;
            }
            i += 1;
        }

        Some(min_val)
    }

    /// Find maximum value with prefetching
    pub fn max(&self) -> Option<i64> {
        if self.data.is_empty() {
            return None;
        }

        let len = self.data.len();
        let ptr = self.data.as_ptr();

        let mut max_val = unsafe { *ptr };
        let mut i = 1;

        while i < len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            let value = unsafe { *ptr.add(i) };
            if value > max_val {
                max_val = value;
            }
            i += 1;
        }

        Some(max_val)
    }
}

/// Batch prefetcher for non-sequential access patterns
///
/// When access is not sequential (e.g., following index pointers),
/// batch prefetch requests to hide latency.
pub struct ScatterPrefetcher {
    pending: Vec<*const u8>,
    batch_size: usize,
    hint: PrefetchHint,
}

impl ScatterPrefetcher {
    /// Create a new scatter prefetcher
    ///
    /// # Arguments
    /// * `batch_size` - Number of addresses to queue before issuing prefetches
    pub fn new(batch_size: usize) -> Self {
        Self {
            pending: Vec::with_capacity(batch_size),
            batch_size,
            hint: PrefetchHint::T0,
        }
    }

    /// Set prefetch hint
    pub fn with_hint(mut self, hint: PrefetchHint) -> Self {
        self.hint = hint;
        self
    }

    /// Queue address for prefetch
    ///
    /// When batch is full, prefetches are automatically issued.
    #[inline]
    pub fn queue(&mut self, addr: *const u8) {
        self.pending.push(addr);
        if self.pending.len() >= self.batch_size {
            self.flush();
        }
    }

    /// Issue all pending prefetches
    #[inline]
    pub fn flush(&mut self) {
        for &addr in &self.pending {
            prefetch(addr, self.hint);
        }
        self.pending.clear();
    }

    /// Get number of pending prefetches
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl Drop for ScatterPrefetcher {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Generic prefetch scanner for any type
pub struct GenericPrefetchScanner<'a, T> {
    data: &'a [T],
    prefetch_distance: usize,
    hint: PrefetchHint,
}

impl<'a, T> GenericPrefetchScanner<'a, T> {
    /// Create a new generic scanner
    pub fn new(data: &'a [T]) -> Self {
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = CACHE_LINE_SIZE / element_size.max(1);

        // Default distance: 8 cache lines worth of elements
        let prefetch_distance = 8 * elements_per_cache_line;

        Self {
            data,
            prefetch_distance,
            hint: PrefetchHint::T0,
        }
    }

    /// Set prefetch distance in elements
    pub fn with_distance(mut self, distance: usize) -> Self {
        self.prefetch_distance = distance;
        self
    }

    /// Set prefetch hint
    pub fn with_hint(mut self, hint: PrefetchHint) -> Self {
        self.hint = hint;
        self
    }

    /// Iterate with prefetching
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(usize, &T),
    {
        let len = self.data.len();
        let ptr = self.data.as_ptr();

        for i in 0..len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            f(i, &self.data[i]);
        }
    }

    /// Map with prefetching
    pub fn map<F, R>(&self, mut f: F) -> Vec<R>
    where
        F: FnMut(&T) -> R,
    {
        let len = self.data.len();
        let ptr = self.data.as_ptr();
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let prefetch_idx = i + self.prefetch_distance;
            if prefetch_idx < len {
                prefetch(unsafe { ptr.add(prefetch_idx) as *const u8 }, self.hint);
            }

            result.push(f(&self.data[i]));
        }

        result
    }
}

/// Prefetch-aware aggregation operations
pub mod aggregates {
    use super::*;

    /// Compute sum with automatic prefetching
    pub fn sum_i64(data: &[i64]) -> i64 {
        PrefetchScanner::new(data).sum()
    }

    /// Compute sum where predicate is true
    pub fn sum_where_i64<F>(data: &[i64], predicate: F) -> i64
    where
        F: Fn(i64) -> bool,
    {
        PrefetchScanner::new(data).sum_where(predicate)
    }

    /// Count values matching predicate
    pub fn count_where_i64<F>(data: &[i64], predicate: F) -> usize
    where
        F: Fn(i64) -> bool,
    {
        PrefetchScanner::new(data).count_where(predicate)
    }

    /// Find minimum value
    pub fn min_i64(data: &[i64]) -> Option<i64> {
        PrefetchScanner::new(data).min()
    }

    /// Find maximum value
    pub fn max_i64(data: &[i64]) -> Option<i64> {
        PrefetchScanner::new(data).max()
    }

    /// Compute average
    pub fn avg_i64(data: &[i64]) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        let sum = PrefetchScanner::new(data).sum();
        Some(sum as f64 / data.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_scanner_sum() {
        let data: Vec<i64> = (1..=1000).collect();
        let scanner = PrefetchScanner::new(&data);
        assert_eq!(scanner.sum(), 500500);
    }

    #[test]
    fn test_prefetch_scanner_sum_where() {
        let data: Vec<i64> = (1..=100).collect();
        let scanner = PrefetchScanner::new(&data);

        // Sum of even numbers: 2 + 4 + ... + 100 = 2550
        let sum = scanner.sum_where(|x| x % 2 == 0);
        assert_eq!(sum, 2550);
    }

    #[test]
    fn test_prefetch_scanner_count_where() {
        let data: Vec<i64> = (1..=100).collect();
        let scanner = PrefetchScanner::new(&data);

        let count = scanner.count_where(|x| x > 50);
        assert_eq!(count, 50);
    }

    #[test]
    fn test_prefetch_scanner_filter() {
        let data: Vec<i64> = vec![1, 5, 3, 8, 2, 9, 4];
        let scanner = PrefetchScanner::new(&data);

        let indices = scanner.filter(|x| x > 4);
        assert_eq!(indices, vec![1, 3, 5]); // indices of 5, 8, 9
    }

    #[test]
    fn test_prefetch_scanner_min_max() {
        let data: Vec<i64> = vec![5, 2, 8, 1, 9, 3];
        let scanner = PrefetchScanner::new(&data);

        assert_eq!(scanner.min(), Some(1));
        assert_eq!(scanner.max(), Some(9));
    }

    #[test]
    fn test_prefetch_scanner_empty() {
        let data: Vec<i64> = vec![];
        let scanner = PrefetchScanner::new(&data);

        assert_eq!(scanner.sum(), 0);
        assert_eq!(scanner.min(), None);
        assert_eq!(scanner.max(), None);
    }

    #[test]
    fn test_scatter_prefetcher() {
        let mut prefetcher = ScatterPrefetcher::new(4);

        let data = [1i64, 2, 3, 4, 5];

        prefetcher.queue(data.as_ptr() as *const u8);
        assert_eq!(prefetcher.pending_count(), 1);

        prefetcher.queue(data.as_ptr().wrapping_add(1) as *const u8);
        prefetcher.queue(data.as_ptr().wrapping_add(2) as *const u8);
        prefetcher.queue(data.as_ptr().wrapping_add(3) as *const u8);

        // Should auto-flush at batch_size
        assert_eq!(prefetcher.pending_count(), 0);
    }

    #[test]
    fn test_generic_scanner() {
        #[derive(Debug)]
        struct Point {
            x: i32,
            y: i32,
        }

        let points = vec![
            Point { x: 1, y: 2 },
            Point { x: 3, y: 4 },
            Point { x: 5, y: 6 },
        ];

        let scanner = GenericPrefetchScanner::new(&points);
        let sums: Vec<i32> = scanner.map(|p| p.x + p.y);

        assert_eq!(sums, vec![3, 7, 11]);
    }

    #[test]
    fn test_aggregates() {
        let data: Vec<i64> = (1..=100).collect();

        assert_eq!(aggregates::sum_i64(&data), 5050);
        assert_eq!(aggregates::min_i64(&data), Some(1));
        assert_eq!(aggregates::max_i64(&data), Some(100));
        assert_eq!(aggregates::avg_i64(&data), Some(50.5));
        assert_eq!(aggregates::count_where_i64(&data, |x| x > 90), 10);
    }

    #[test]
    fn test_custom_distance() {
        let data: Vec<i64> = (1..=100).collect();
        let scanner = PrefetchScanner::with_distance(&data, 100);
        assert_eq!(scanner.sum(), 5050);
    }

    #[test]
    fn test_hint_types() {
        let data: Vec<i64> = (1..=100).collect();

        let scanner = PrefetchScanner::new(&data).with_hint(PrefetchHint::NonTemporal);
        assert_eq!(scanner.sum(), 5050);
    }
}
