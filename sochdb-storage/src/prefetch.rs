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

//! Memory Prefetching for Range Scans
//!
//! Implements proactive prefetching to eliminate page fault latency during
//! sequential scans of memory-mapped data.
//!
//! ## jj.md Task 7: Prefetching for Range Scans
//!
//! Goals:
//! - Eliminate page fault latency during scans (~50μs per fault)
//! - 2-3x faster cold range scans
//! - Better utilization of I/O bandwidth
//!
//! ## Implementation
//!
//! Uses platform-specific prefetching:
//! - Unix: `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)`
//! - x86_64: `_mm_prefetch` intrinsic for CPU cache prefetching
//!
//! For 1M edge scan with 4KB pages (32 edges/page):
//! - Without prefetch: ~31,250 page faults × 50μs = 1.56 seconds stalled
//! - With prefetch: Near-zero stall time

/// Edge size in bytes (fixed format)
pub const EDGE_SIZE: usize = 128;

/// Number of edges to prefetch ahead during iteration
/// 16 edges = 2KB = half a page, good for L1 cache
pub const PREFETCH_DISTANCE: usize = 16;

/// Advise the OS about memory access patterns for a range scan.
///
/// This should be called before starting a sequential scan over memory-mapped
/// data to enable read-ahead and sequential prefetching.
///
/// # Arguments
/// * `data` - The memory-mapped region that will be scanned
/// * `start_offset` - Starting offset within the region
/// * `length` - Expected length of the scan
///
/// # Platform Support
/// - Unix: Uses `madvise` with `MADV_SEQUENTIAL` and `MADV_WILLNEED`
/// - Other platforms: No-op (relies on OS default behavior)
#[cfg(unix)]
pub fn advise_sequential(data: &[u8], start_offset: usize, length: usize) {
    use std::cmp::min;

    // Validate bounds
    let start = min(start_offset, data.len());
    let end = min(start + length, data.len());

    if end <= start {
        return;
    }

    // Align to page boundaries (typically 4KB)
    let page_size = page_size();
    let aligned_start = (start / page_size) * page_size;
    let aligned_length = (end - aligned_start).div_ceil(page_size) * page_size;

    unsafe {
        let ptr = data.as_ptr().add(aligned_start) as *mut libc::c_void;

        // MADV_SEQUENTIAL: Expect sequential access, enable aggressive read-ahead
        // MADV_WILLNEED: Bring pages into memory proactively
        let advice = libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED;

        let result = libc::madvise(ptr, aligned_length, advice);
        if result != 0 {
            // Non-fatal: log but continue
            #[cfg(debug_assertions)]
            eprintln!(
                "madvise failed: {} (continuing without prefetch)",
                std::io::Error::last_os_error()
            );
        }
    }
}

#[cfg(not(unix))]
pub fn advise_sequential(_data: &[u8], _start_offset: usize, _length: usize) {
    // No-op on non-Unix platforms
}

/// Advise the OS that a memory region will be accessed randomly.
///
/// This disables sequential read-ahead for point lookups.
#[cfg(unix)]
pub fn advise_random(data: &[u8]) {
    unsafe {
        let ptr = data.as_ptr() as *mut libc::c_void;
        let result = libc::madvise(ptr, data.len(), libc::MADV_RANDOM);
        if result != 0 {
            #[cfg(debug_assertions)]
            eprintln!("madvise RANDOM failed: {}", std::io::Error::last_os_error());
        }
    }
}

#[cfg(not(unix))]
pub fn advise_random(_data: &[u8]) {
    // No-op on non-Unix platforms
}

/// Advise the OS that a memory region is no longer needed.
///
/// Allows the OS to free pages if memory pressure is high.
#[cfg(unix)]
pub fn advise_dontneed(data: &[u8], start_offset: usize, length: usize) {
    use std::cmp::min;

    let start = min(start_offset, data.len());
    let end = min(start + length, data.len());

    if end <= start {
        return;
    }

    let page_size = page_size();
    let aligned_start = (start / page_size) * page_size;
    let aligned_length = (end - aligned_start).div_ceil(page_size) * page_size;

    unsafe {
        let ptr = data.as_ptr().add(aligned_start) as *mut libc::c_void;
        // MADV_DONTNEED: We're done with this memory
        libc::madvise(ptr, aligned_length, libc::MADV_DONTNEED);
    }
}

#[cfg(not(unix))]
pub fn advise_dontneed(_data: &[u8], _start_offset: usize, _length: usize) {
    // No-op on non-Unix platforms
}

/// Prefetch data into CPU cache during iteration.
///
/// Uses x86_64 `_mm_prefetch` intrinsic when available for low-latency
/// cache-to-register prefetching.
///
/// # Arguments
/// * `data` - The memory region
/// * `current_offset` - Current read position
/// * `prefetch_distance` - Number of bytes ahead to prefetch
///
/// # Safety
/// Safe to call with any offset - bounds are checked internally.
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
pub fn prefetch_ahead(data: &[u8], current_offset: usize, prefetch_distance: usize) {
    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

    let target_offset = current_offset + prefetch_distance;
    if target_offset < data.len() {
        unsafe {
            _mm_prefetch(
                data.as_ptr().add(target_offset) as *const i8,
                _MM_HINT_T0, // Prefetch to L1 cache
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn prefetch_ahead(data: &[u8], current_offset: usize, prefetch_distance: usize) {
    // aarch64 prefetch is currently unstable in Rust
    // Fall back to no-op until stabilized (issue #117217)
    // The OS-level madvise still provides significant benefits
    let _ = (data, current_offset, prefetch_distance);
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    target_arch = "aarch64"
)))]
pub fn prefetch_ahead(_data: &[u8], _current_offset: usize, _prefetch_distance: usize) {
    // No-op on unsupported platforms
}

/// Get the system page size.
#[cfg(unix)]
fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[cfg(not(unix))]
fn page_size() -> usize {
    4096 // Common default
}

/// Iterator wrapper that adds prefetching to an existing iterator.
///
/// Wraps any byte-slice based iterator and adds proactive prefetching
/// to reduce cache misses and page faults.
pub struct PrefetchingIterator<'a, I> {
    inner: I,
    data: &'a [u8],
    current_offset: usize,
    prefetch_distance_bytes: usize,
}

impl<'a, I> PrefetchingIterator<'a, I> {
    /// Create a new prefetching iterator.
    ///
    /// # Arguments
    /// * `inner` - The underlying iterator
    /// * `data` - The memory-mapped data being iterated
    /// * `prefetch_distance` - Number of items to prefetch ahead
    /// * `item_size` - Size of each item in bytes
    pub fn new(inner: I, data: &'a [u8], prefetch_distance: usize, item_size: usize) -> Self {
        Self {
            inner,
            data,
            current_offset: 0,
            prefetch_distance_bytes: prefetch_distance * item_size,
        }
    }

    /// Set the current offset (for resuming iteration)
    pub fn set_offset(&mut self, offset: usize) {
        self.current_offset = offset;
    }
}

impl<'a, I, T> Iterator for PrefetchingIterator<'a, I>
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // Prefetch ahead before reading current item
        prefetch_ahead(self.data, self.current_offset, self.prefetch_distance_bytes);

        match self.inner.next() {
            Some(item) => {
                self.current_offset += EDGE_SIZE; // Advance by one edge
                Some(item)
            }
            None => None,
        }
    }
}

/// Statistics for prefetch operations (useful for debugging/tuning)
#[derive(Debug, Default, Clone)]
pub struct PrefetchStats {
    /// Number of madvise calls made
    pub madvise_calls: u64,
    /// Number of prefetch intrinsic calls
    pub prefetch_calls: u64,
    /// Total bytes advised for prefetching
    pub bytes_advised: u64,
}

impl PrefetchStats {
    /// Create a new stats tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an madvise call
    pub fn record_madvise(&mut self, bytes: usize) {
        self.madvise_calls += 1;
        self.bytes_advised += bytes as u64;
    }

    /// Record a prefetch intrinsic call
    pub fn record_prefetch(&mut self) {
        self.prefetch_calls += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_size() {
        let ps = page_size();
        assert!(ps >= 4096, "Page size too small: {}", ps);
        assert!(ps.is_power_of_two(), "Page size not power of 2: {}", ps);
    }

    #[test]
    fn test_advise_sequential_bounds() {
        let data = vec![0u8; 4096 * 10]; // 10 pages

        // Normal case
        advise_sequential(&data, 0, data.len());

        // Partial range
        advise_sequential(&data, 4096, 8192);

        // Beyond bounds (should handle gracefully)
        advise_sequential(&data, data.len() + 1000, 1000);

        // Zero length
        advise_sequential(&data, 0, 0);
    }

    #[test]
    fn test_prefetch_ahead_bounds() {
        let data = vec![0u8; 1024];

        // Normal case
        prefetch_ahead(&data, 0, 512);

        // Near end
        prefetch_ahead(&data, 900, 200);

        // Beyond bounds (should handle gracefully)
        prefetch_ahead(&data, 1000, 500);
    }

    #[test]
    fn test_prefetching_iterator() {
        let data = vec![0u8; 128 * 100]; // 100 edges worth of data
        let items: Vec<i32> = (0..100).collect();

        let iter = PrefetchingIterator::new(items.into_iter(), &data, PREFETCH_DISTANCE, EDGE_SIZE);

        let collected: Vec<i32> = iter.collect();
        assert_eq!(collected.len(), 100);
        assert_eq!(collected[0], 0);
        assert_eq!(collected[99], 99);
    }

    #[test]
    fn test_prefetch_stats() {
        let mut stats = PrefetchStats::new();

        stats.record_madvise(4096);
        stats.record_madvise(8192);
        stats.record_prefetch();
        stats.record_prefetch();
        stats.record_prefetch();

        assert_eq!(stats.madvise_calls, 2);
        assert_eq!(stats.prefetch_calls, 3);
        assert_eq!(stats.bytes_advised, 4096 + 8192);
    }
}
