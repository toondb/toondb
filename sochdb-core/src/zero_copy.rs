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

//! Zero-Copy Iterators - mmap-based Scans
//!
//! This module provides memory-mapped file access for efficient,
//! zero-copy iteration over large data files.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Zero-Copy Iterator                           │
//! │                                                                 │
//! │  Application                    OS/Kernel                       │
//! │  ┌────────────┐                 ┌────────────────────┐         │
//! │  │  Iterator  │                 │   Page Cache       │         │
//! │  │            │                 │  ┌──────────────┐  │         │
//! │  │  &[u8] ────┼─────────────────┼─▶│ Data Pages  │  │         │
//! │  │            │    mmap         │  └──────────────┘  │         │
//! │  └────────────┘                 │         ↑         │         │
//! │        │                        │         │         │         │
//! │        │ No copy!               │    ┌────┴────┐    │         │
//! │        ▼                        │    │  Disk   │    │         │
//! │  Process data                   │    └─────────┘    │         │
//! │  in-place                       └────────────────────┘         │
//! │                                                                 │
//! │  Benefits:                                                      │
//! │  • No buffer copies                                             │
//! │  • OS manages page faults                                       │
//! │  • Efficient for sequential scans                               │
//! │  • Read-ahead by OS                                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Safety
//!
//! Memory-mapped regions can become invalid if the underlying file is
//! modified. This implementation provides:
//! - Read-only mappings by default
//! - Length validation
//! - Guard types for safe access

use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

/// Memory-mapped region
pub struct MmapRegion {
    /// Pointer to mapped memory
    ptr: *const u8,
    /// Length of mapped region
    len: usize,
    /// File descriptor (for cleanup)
    #[cfg(unix)]
    _fd: i32,
}

// Safety: MmapRegion is Send/Sync because we only allow read-only access
// and the underlying mapping is immutable
unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    /// Create a new memory-mapped region for a file
    #[cfg(unix)]
    pub fn new(file: &File) -> io::Result<Self> {
        use std::ptr;

        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        if len == 0 {
            return Ok(Self {
                ptr: ptr::null(),
                len: 0,
                _fd: file.as_raw_fd(),
            });
        }

        // SAFETY: We're creating a read-only mapping of a valid file
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            ptr: ptr as *const u8,
            len,
            _fd: file.as_raw_fd(),
        })
    }

    /// Create with advisory read-ahead hint
    #[cfg(unix)]
    pub fn new_with_readahead(file: &File) -> io::Result<Self> {
        let region = Self::new(file)?;

        if region.len > 0 {
            // Advise kernel for sequential access
            unsafe {
                libc::madvise(
                    region.ptr as *mut libc::c_void,
                    region.len,
                    libc::MADV_SEQUENTIAL,
                );
            }
        }

        Ok(region)
    }

    /// Fallback for non-Unix systems (reads entire file into memory)
    #[cfg(not(unix))]
    pub fn new(file: &File) -> io::Result<Self> {
        use std::io::Read;
        let mut file = file;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let len = buffer.len();
        let ptr = Box::into_raw(buffer.into_boxed_slice()) as *const u8;

        Ok(Self { ptr, len })
    }

    #[cfg(not(unix))]
    pub fn new_with_readahead(file: &File) -> io::Result<Self> {
        Self::new(file)
    }

    /// Get the length of the mapped region
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice of the mapped region
    ///
    /// # Safety
    /// The returned slice is only valid as long as the MmapRegion exists
    /// and the underlying file is not modified.
    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for len bytes, read-only, and properly aligned
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get a subslice of the mapped region
    pub fn slice(&self, range: Range<usize>) -> Option<&[u8]> {
        if range.end > self.len {
            return None;
        }
        Some(&self.as_slice()[range])
    }

    /// Prefetch a range of data
    #[cfg(unix)]
    pub fn prefetch(&self, range: Range<usize>) {
        if range.start >= self.len || self.ptr.is_null() {
            return;
        }

        let end = range.end.min(self.len);
        let ptr = unsafe { self.ptr.add(range.start) };
        let len = end - range.start;

        unsafe {
            libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_WILLNEED);
        }
    }

    #[cfg(not(unix))]
    pub fn prefetch(&self, _range: Range<usize>) {
        // No-op on non-Unix
    }
}

impl Drop for MmapRegion {
    #[cfg(unix)]
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }

    #[cfg(not(unix))]
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            // Reconstruct and drop the boxed slice
            unsafe {
                let slice = std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len);
                drop(Box::from_raw(slice));
            }
        }
    }
}

/// Zero-copy iterator over chunks of mapped memory
pub struct ZeroCopyIterator<'a> {
    /// The mapped region
    data: &'a [u8],
    /// Current position
    pos: usize,
    /// Chunk size
    chunk_size: usize,
    /// Statistics
    stats: Arc<IteratorStats>,
}

impl<'a> ZeroCopyIterator<'a> {
    /// Create new iterator over mapped region
    pub fn new(data: &'a [u8], chunk_size: usize) -> Self {
        Self {
            data,
            pos: 0,
            chunk_size,
            stats: Arc::new(IteratorStats::default()),
        }
    }

    /// Create with shared statistics
    pub fn with_stats(data: &'a [u8], chunk_size: usize, stats: Arc<IteratorStats>) -> Self {
        Self {
            data,
            pos: 0,
            chunk_size,
            stats,
        }
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Seek to position
    pub fn seek(&mut self, pos: usize) -> bool {
        if pos <= self.data.len() {
            self.pos = pos;
            true
        } else {
            false
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &IteratorStats {
        &self.stats
    }
}

impl<'a> Iterator for ZeroCopyIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let start = self.pos;
        let end = (start + self.chunk_size).min(self.data.len());
        self.pos = end;

        self.stats.chunks_read.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_read
            .fetch_add((end - start) as u64, Ordering::Relaxed);

        Some(&self.data[start..end])
    }
}

/// Iterator statistics
#[derive(Debug, Default)]
pub struct IteratorStats {
    pub chunks_read: AtomicU64,
    pub bytes_read: AtomicU64,
    pub seeks: AtomicU64,
}

impl IteratorStats {
    pub fn snapshot(&self) -> IteratorStatsSnapshot {
        IteratorStatsSnapshot {
            chunks_read: self.chunks_read.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            seeks: self.seeks.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IteratorStatsSnapshot {
    pub chunks_read: u64,
    pub bytes_read: u64,
    pub seeks: u64,
}

/// Block-aware iterator that parses block headers
pub struct BlockIterator<'a> {
    /// Raw data iterator
    inner: ZeroCopyIterator<'a>,
    /// Current block index
    block_index: usize,
}

impl<'a> BlockIterator<'a> {
    pub fn new(data: &'a [u8], block_size: usize) -> Self {
        Self {
            inner: ZeroCopyIterator::new(data, block_size),
            block_index: 0,
        }
    }

    /// Get current block index
    pub fn block_index(&self) -> usize {
        self.block_index
    }

    /// Skip to specific block
    pub fn skip_to_block(&mut self, index: usize) -> bool {
        let pos = index * self.inner.chunk_size;
        if self.inner.seek(pos) {
            self.block_index = index;
            true
        } else {
            false
        }
    }
}

impl<'a> Iterator for BlockIterator<'a> {
    type Item = (usize, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let block = self.inner.next()?;
        let index = self.block_index;
        self.block_index += 1;
        Some((index, block))
    }
}

/// Scanned region with optional filtering
pub struct FilteredScan<'a, F>
where
    F: Fn(&[u8]) -> bool,
{
    inner: ZeroCopyIterator<'a>,
    predicate: F,
}

impl<'a, F> FilteredScan<'a, F>
where
    F: Fn(&[u8]) -> bool,
{
    pub fn new(data: &'a [u8], chunk_size: usize, predicate: F) -> Self {
        Self {
            inner: ZeroCopyIterator::new(data, chunk_size),
            predicate,
        }
    }
}

impl<'a, F> Iterator for FilteredScan<'a, F>
where
    F: Fn(&[u8]) -> bool,
{
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let chunk = self.inner.next()?;
            if (self.predicate)(chunk) {
                return Some(chunk);
            }
        }
    }
}

/// Parallel scan configuration
#[derive(Debug, Clone)]
pub struct ParallelScanConfig {
    /// Number of parallel readers
    pub num_readers: usize,
    /// Chunk size per reader
    pub chunk_size: usize,
    /// Prefetch distance (in chunks)
    pub prefetch_distance: usize,
}

impl Default for ParallelScanConfig {
    fn default() -> Self {
        Self {
            num_readers: 4,
            chunk_size: 64 * 1024, // 64KB
            prefetch_distance: 2,
        }
    }
}

/// Range scanner for parallel processing
pub struct RangeScanner {
    /// Total length
    total_len: usize,
    /// Range size
    range_size: usize,
    /// Current range index
    current: usize,
    /// Total ranges
    total_ranges: usize,
}

impl RangeScanner {
    /// Create scanner that divides data into N ranges
    pub fn new(total_len: usize, num_ranges: usize) -> Self {
        let range_size = total_len.div_ceil(num_ranges.max(1));
        let total_ranges = if total_len > 0 {
            total_len.div_ceil(range_size)
        } else {
            0
        };

        Self {
            total_len,
            range_size,
            current: 0,
            total_ranges,
        }
    }

    /// Get range for a specific index
    pub fn range(&self, index: usize) -> Option<Range<usize>> {
        if index >= self.total_ranges {
            return None;
        }

        let start = index * self.range_size;
        let end = ((index + 1) * self.range_size).min(self.total_len);

        Some(start..end)
    }

    /// Get total number of ranges
    pub fn total_ranges(&self) -> usize {
        self.total_ranges
    }
}

impl Iterator for RangeScanner {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let range = self.range(self.current)?;
        self.current += 1;
        Some(range)
    }
}

/// Open a file for zero-copy scanning
pub fn open_for_scan(path: impl AsRef<Path>) -> io::Result<MmapRegion> {
    let file = File::open(path)?;
    MmapRegion::new_with_readahead(&file)
}

/// Create an iterator over a file
pub fn scan_file(path: impl AsRef<Path>, chunk_size: usize) -> io::Result<(MmapRegion, usize)> {
    let region = open_for_scan(path)?;
    Ok((region, chunk_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(data: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_mmap_region_basic() {
        let data = b"Hello, World! This is test data for mmap.";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();

        assert_eq!(region.len(), data.len());
        assert_eq!(region.as_slice(), data);
    }

    #[test]
    fn test_mmap_empty_file() {
        let file = create_test_file(b"");

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();

        assert!(region.is_empty());
        assert_eq!(region.as_slice(), &[] as &[u8]);
    }

    #[test]
    fn test_mmap_slice() {
        let data = b"0123456789ABCDEF";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();

        assert_eq!(region.slice(0..4), Some(&b"0123"[..]));
        assert_eq!(region.slice(4..8), Some(&b"4567"[..]));
        assert_eq!(region.slice(0..100), None);
    }

    #[test]
    fn test_zero_copy_iterator() {
        let data = b"AAAABBBBCCCCDDDD";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let iter = ZeroCopyIterator::new(region.as_slice(), 4);

        let chunks: Vec<_> = iter.collect();

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], b"AAAA");
        assert_eq!(chunks[1], b"BBBB");
        assert_eq!(chunks[2], b"CCCC");
        assert_eq!(chunks[3], b"DDDD");
    }

    #[test]
    fn test_iterator_uneven_chunks() {
        let data = b"AAABBBCC";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let iter = ZeroCopyIterator::new(region.as_slice(), 3);

        let chunks: Vec<_> = iter.collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"AAA");
        assert_eq!(chunks[1], b"BBB");
        assert_eq!(chunks[2], b"CC");
    }

    #[test]
    fn test_iterator_stats() {
        let data = b"AAAABBBBCCCC";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let iter = ZeroCopyIterator::new(region.as_slice(), 4);
        let stats = Arc::clone(&iter.stats);

        let _chunks: Vec<_> = iter.collect();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.chunks_read, 3);
        assert_eq!(snapshot.bytes_read, 12);
    }

    #[test]
    fn test_iterator_seek() {
        let data = b"AAAABBBBCCCCDDDD";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let mut iter = ZeroCopyIterator::new(region.as_slice(), 4);

        assert!(iter.seek(8));
        assert_eq!(iter.next(), Some(&b"CCCC"[..]));

        assert!(!iter.seek(100));
    }

    #[test]
    fn test_block_iterator() {
        let data = b"BLK1BLK2BLK3";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let iter = BlockIterator::new(region.as_slice(), 4);

        let blocks: Vec<_> = iter.collect();

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], (0, &b"BLK1"[..]));
        assert_eq!(blocks[1], (1, &b"BLK2"[..]));
        assert_eq!(blocks[2], (2, &b"BLK3"[..]));
    }

    #[test]
    fn test_block_iterator_skip() {
        let data = b"BLK1BLK2BLK3BLK4";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let mut iter = BlockIterator::new(region.as_slice(), 4);

        assert!(iter.skip_to_block(2));
        assert_eq!(iter.next(), Some((2, &b"BLK3"[..])));
    }

    #[test]
    fn test_filtered_scan() {
        let data = b"ABCDXXXXYYYY1234";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();

        // Filter for chunks that don't contain 'X' or 'Y'
        let scan = FilteredScan::new(region.as_slice(), 4, |chunk| {
            !chunk.contains(&b'X') && !chunk.contains(&b'Y')
        });

        let matching: Vec<_> = scan.collect();

        assert_eq!(matching.len(), 2);
        assert_eq!(matching[0], b"ABCD");
        assert_eq!(matching[1], b"1234");
    }

    #[test]
    fn test_range_scanner() {
        let scanner = RangeScanner::new(100, 4);

        assert_eq!(scanner.total_ranges(), 4);

        let ranges: Vec<_> = scanner.collect();

        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], 0..25);
        assert_eq!(ranges[1], 25..50);
        assert_eq!(ranges[2], 50..75);
        assert_eq!(ranges[3], 75..100);
    }

    #[test]
    fn test_range_scanner_uneven() {
        let scanner = RangeScanner::new(10, 3);

        let ranges: Vec<_> = scanner.collect();

        // 10 / 3 = 4 (ceiling), so ranges are 0..4, 4..8, 8..10
        assert_eq!(ranges.len(), 3);
        assert!(ranges.last().unwrap().end == 10);
    }

    #[test]
    fn test_range_scanner_empty() {
        let scanner = RangeScanner::new(0, 4);

        assert_eq!(scanner.total_ranges(), 0);
        let ranges: Vec<_> = scanner.collect();
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_parallel_scan_config() {
        let config = ParallelScanConfig::default();

        assert!(config.num_readers > 0);
        assert!(config.chunk_size > 0);
    }

    #[test]
    fn test_remaining_bytes() {
        let data = b"AAAABBBBCCCC";
        let file = create_test_file(data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();
        let mut iter = ZeroCopyIterator::new(region.as_slice(), 4);

        assert_eq!(iter.remaining(), 12);
        iter.next();
        assert_eq!(iter.remaining(), 8);
        iter.next();
        assert_eq!(iter.remaining(), 4);
        iter.next();
        assert_eq!(iter.remaining(), 0);
    }

    #[test]
    fn test_mmap_with_readahead() {
        let data = b"Test data for readahead mmap";
        let file = create_test_file(data);

        let region = MmapRegion::new_with_readahead(&File::open(file.path()).unwrap()).unwrap();

        assert_eq!(region.len(), data.len());
        assert_eq!(region.as_slice(), data);
    }

    #[test]
    fn test_prefetch() {
        let data = vec![0u8; 1024 * 1024]; // 1MB
        let file = create_test_file(&data);

        let region = MmapRegion::new(&File::open(file.path()).unwrap()).unwrap();

        // Prefetch should not crash
        region.prefetch(0..65536);
        region.prefetch(65536..131072);
    }
}
