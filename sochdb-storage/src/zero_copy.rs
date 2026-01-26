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

//! Zero-Copy SSTable Iterator
//!
//! Provides zero-copy access to edges in memory-mapped SSTables,
//! avoiding the overhead of copying 128 bytes per edge during iteration.
//!
//! ## jj.md Task 6: Zero-Copy Iterator
//!
//! Goals:
//! - 3-5x faster range scans
//! - Reduced memory bandwidth (128 bytes saved per edge)
//! - Better CPU cache utilization
//!
//! ## Implementation
//!
//! Uses `EdgeRef<'a>` to provide lazy field access directly from mmap'd memory:
//! - No allocation per edge
//! - No copy of 128 bytes
//! - Fields parsed on-demand
//!
//! ## Memory Bandwidth Savings
//!
//! ```text
//! Before: read 128B (mmap) + write 128B (copy) = 256B per edge
//! After:  read 128B (mmap) only = 128B per edge
//! Improvement: 2x memory bandwidth, 3-5x throughput
//! ```

use byteorder::{ByteOrder, LittleEndian};
use std::cmp::Ordering;

/// Size of an edge record in bytes
pub const EDGE_SIZE: usize = 128;

/// A zero-copy reference to an edge stored in memory-mapped data.
///
/// This struct provides lazy field access directly from the underlying
/// byte slice, avoiding the overhead of deserializing into an owned struct.
///
/// # Lifetime
///
/// The lifetime `'a` is tied to the underlying memory-mapped region.
/// The `EdgeRef` is only valid as long as the mmap is valid.
#[derive(Clone, Copy)]
pub struct EdgeRef<'a> {
    /// The raw 128-byte edge data
    bytes: &'a [u8; EDGE_SIZE],
}

impl<'a> EdgeRef<'a> {
    /// Create a new EdgeRef from a byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure the slice is exactly 128 bytes and contains
    /// a valid edge representation.
    #[inline]
    pub fn new(bytes: &'a [u8; EDGE_SIZE]) -> Self {
        Self { bytes }
    }

    /// Try to create an EdgeRef from a slice (with length check).
    #[inline]
    pub fn try_from_slice(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() >= EDGE_SIZE {
            let arr: &[u8; EDGE_SIZE] = bytes[..EDGE_SIZE].try_into().ok()?;
            Some(Self { bytes: arr })
        } else {
            None
        }
    }

    /// Get the edge ID (u128 at offset 0).
    #[inline]
    pub fn edge_id(&self) -> u128 {
        LittleEndian::read_u128(&self.bytes[0..16])
    }

    /// Get the timestamp in microseconds (u64 at offset 16).
    #[inline]
    pub fn timestamp_us(&self) -> u64 {
        LittleEndian::read_u64(&self.bytes[16..24])
    }

    /// Get the tenant ID (u64 at offset 24).
    #[inline]
    pub fn tenant_id(&self) -> u64 {
        LittleEndian::read_u64(&self.bytes[24..32])
    }

    /// Get the project ID (u16 at offset 32).
    #[inline]
    pub fn project_id(&self) -> u16 {
        LittleEndian::read_u16(&self.bytes[32..34])
    }

    /// Get the source node ID (u128 at offset 34).
    #[inline]
    pub fn source_node_id(&self) -> u128 {
        LittleEndian::read_u128(&self.bytes[34..50])
    }

    /// Get the target node ID (u128 at offset 50).
    #[inline]
    pub fn target_node_id(&self) -> u128 {
        LittleEndian::read_u128(&self.bytes[50..66])
    }

    /// Get the edge type (u8 at offset 66).
    #[inline]
    pub fn edge_type(&self) -> u8 {
        self.bytes[66]
    }

    /// Get the flags byte (u8 at offset 67).
    #[inline]
    pub fn flags(&self) -> u8 {
        self.bytes[67]
    }

    /// Check if this edge is a tombstone (deleted).
    #[inline]
    pub fn is_deleted(&self) -> bool {
        // Assuming bit 0 of flags indicates deletion
        (self.flags() & 0x01) != 0
    }

    /// Get the payload reference (hash or inline data at offset 68).
    #[inline]
    pub fn payload_ref(&self) -> &[u8] {
        &self.bytes[68..100]
    }

    /// Get the checksum (u32 at offset 124).
    #[inline]
    pub fn checksum(&self) -> u32 {
        LittleEndian::read_u32(&self.bytes[124..128])
    }

    /// Get the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; EDGE_SIZE] {
        self.bytes
    }

    /// Verify the edge checksum.
    pub fn verify_checksum(&self) -> bool {
        // Compute checksum of first 124 bytes
        let data = &self.bytes[..124];
        let computed = crate::block_checksum::crc32c(data);
        computed == self.checksum()
    }
}

impl std::fmt::Debug for EdgeRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeRef")
            .field("edge_id", &self.edge_id())
            .field("timestamp_us", &self.timestamp_us())
            .field("tenant_id", &self.tenant_id())
            .field("is_deleted", &self.is_deleted())
            .finish()
    }
}

impl PartialEq for EdgeRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.edge_id() == other.edge_id()
    }
}

impl Eq for EdgeRef<'_> {}

impl PartialOrd for EdgeRef<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeRef<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary sort by timestamp, secondary by edge_id
        match self.timestamp_us().cmp(&other.timestamp_us()) {
            Ordering::Equal => self.edge_id().cmp(&other.edge_id()),
            ord => ord,
        }
    }
}

/// Zero-copy iterator over edges in a memory-mapped region.
///
/// This iterator yields `EdgeRef` values that point directly into the
/// underlying mmap, avoiding any copying.
pub struct ZeroCopyIterator<'a> {
    /// The memory-mapped data
    data: &'a [u8],
    /// Current offset in the data
    offset: usize,
    /// End offset (exclusive)
    end: usize,
}

impl<'a> ZeroCopyIterator<'a> {
    /// Create a new zero-copy iterator.
    ///
    /// # Arguments
    /// * `data` - The memory-mapped data
    /// * `start` - Starting offset (must be aligned to EDGE_SIZE)
    /// * `end` - Ending offset (exclusive)
    pub fn new(data: &'a [u8], start: usize, end: usize) -> Self {
        Self {
            data,
            offset: start,
            end: end.min(data.len()),
        }
    }

    /// Create an iterator over the entire data region.
    pub fn all(data: &'a [u8]) -> Self {
        Self::new(data, 0, data.len())
    }

    /// Get the current position.
    pub fn position(&self) -> usize {
        self.offset
    }

    /// Get the remaining byte count.
    pub fn remaining(&self) -> usize {
        self.end.saturating_sub(self.offset)
    }

    /// Get the remaining edge count.
    pub fn remaining_edges(&self) -> usize {
        self.remaining() / EDGE_SIZE
    }

    /// Skip to a specific offset.
    pub fn seek(&mut self, offset: usize) {
        self.offset = offset.min(self.end);
    }

    /// Skip n edges.
    pub fn skip_edges(&mut self, n: usize) {
        self.offset = (self.offset + n * EDGE_SIZE).min(self.end);
    }
}

impl<'a> Iterator for ZeroCopyIterator<'a> {
    type Item = EdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + EDGE_SIZE > self.end {
            return None;
        }

        let bytes: &[u8; EDGE_SIZE] = self.data[self.offset..self.offset + EDGE_SIZE]
            .try_into()
            .ok()?;

        self.offset += EDGE_SIZE;
        Some(EdgeRef::new(bytes))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_edges();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ZeroCopyIterator<'_> {}

/// Zero-copy iterator with prefetching for better cache performance.
///
/// This iterator uses CPU prefetch instructions to bring future edges
/// into cache before they're accessed.
pub struct PrefetchingZeroCopyIterator<'a> {
    inner: ZeroCopyIterator<'a>,
    prefetch_distance: usize,
}

impl<'a> PrefetchingZeroCopyIterator<'a> {
    /// Create a new prefetching iterator.
    ///
    /// # Arguments
    /// * `data` - The memory-mapped data
    /// * `start` - Starting offset
    /// * `end` - Ending offset
    /// * `prefetch_distance` - Number of edges to prefetch ahead (default: 16)
    pub fn new(data: &'a [u8], start: usize, end: usize, prefetch_distance: usize) -> Self {
        Self {
            inner: ZeroCopyIterator::new(data, start, end),
            prefetch_distance,
        }
    }

    /// Create with default prefetch distance (16 edges = 2KB).
    pub fn with_default_prefetch(data: &'a [u8], start: usize, end: usize) -> Self {
        Self::new(data, start, end, 16)
    }
}

impl<'a> Iterator for PrefetchingZeroCopyIterator<'a> {
    type Item = EdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Prefetch ahead
        let prefetch_offset = self.inner.offset + self.prefetch_distance * EDGE_SIZE;
        if prefetch_offset < self.inner.end {
            crate::prefetch::prefetch_ahead(
                self.inner.data,
                self.inner.offset,
                self.prefetch_distance * EDGE_SIZE,
            );
        }

        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for PrefetchingZeroCopyIterator<'_> {}

/// Range filter for zero-copy iteration.
///
/// Wraps an iterator and filters edges by timestamp range.
pub struct TimestampRangeFilter<'a, I> {
    inner: I,
    start_ts: u64,
    end_ts: u64,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, I: Iterator<Item = EdgeRef<'a>>> TimestampRangeFilter<'a, I> {
    /// Create a new range filter.
    pub fn new(inner: I, start_ts: u64, end_ts: u64) -> Self {
        Self {
            inner,
            start_ts,
            end_ts,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, I: Iterator<Item = EdgeRef<'a>>> Iterator for TimestampRangeFilter<'a, I> {
    type Item = EdgeRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let edge = self.inner.next()?;
            let ts = edge.timestamp_us();

            if ts > self.end_ts {
                // Past the end - stop iteration
                return None;
            }

            if ts >= self.start_ts {
                return Some(edge);
            }

            // Before start - continue to next
        }
    }
}

/// Extension trait for adding zero-copy iteration methods.
pub trait ZeroCopyExt<'a> {
    /// Get a zero-copy iterator over the data.
    fn iter_zero_copy(&'a self) -> ZeroCopyIterator<'a>;

    /// Get a zero-copy iterator with prefetching.
    fn iter_zero_copy_prefetching(&'a self) -> PrefetchingZeroCopyIterator<'a>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_edge(edge_id: u128, timestamp_us: u64) -> [u8; EDGE_SIZE] {
        let mut bytes = [0u8; EDGE_SIZE];

        // Write edge_id at offset 0
        LittleEndian::write_u128(&mut bytes[0..16], edge_id);
        // Write timestamp at offset 16
        LittleEndian::write_u64(&mut bytes[16..24], timestamp_us);
        // Write tenant_id at offset 24
        LittleEndian::write_u64(&mut bytes[24..32], 1);
        // Write project_id at offset 32
        LittleEndian::write_u16(&mut bytes[32..34], 1);

        // Compute and write checksum
        let checksum = crate::block_checksum::crc32c(&bytes[..124]);
        LittleEndian::write_u32(&mut bytes[124..128], checksum);

        bytes
    }

    #[test]
    fn test_edge_ref_basic() {
        let edge_bytes = create_test_edge(12345, 1000000);
        let edge_ref = EdgeRef::new(&edge_bytes);

        assert_eq!(edge_ref.edge_id(), 12345);
        assert_eq!(edge_ref.timestamp_us(), 1000000);
        assert_eq!(edge_ref.tenant_id(), 1);
        assert_eq!(edge_ref.project_id(), 1);
        assert!(!edge_ref.is_deleted());
    }

    #[test]
    fn test_edge_ref_checksum() {
        let edge_bytes = create_test_edge(42, 500000);
        let edge_ref = EdgeRef::new(&edge_bytes);

        assert!(edge_ref.verify_checksum());

        // Corrupt the data
        let mut corrupted = edge_bytes;
        corrupted[10] ^= 0xFF;
        let corrupted_ref = EdgeRef::new(&corrupted);
        assert!(!corrupted_ref.verify_checksum());
    }

    #[test]
    fn test_zero_copy_iterator() {
        // Create multiple edges
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend_from_slice(&create_test_edge(i as u128, i as u64 * 1000));
        }

        let iter = ZeroCopyIterator::all(&data);
        assert_eq!(iter.remaining_edges(), 10);

        let edges: Vec<_> = iter.collect();
        assert_eq!(edges.len(), 10);

        for (i, edge) in edges.iter().enumerate() {
            assert_eq!(edge.edge_id(), i as u128);
            assert_eq!(edge.timestamp_us(), i as u64 * 1000);
        }
    }

    #[test]
    fn test_zero_copy_iterator_range() {
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend_from_slice(&create_test_edge(i as u128, i as u64 * 1000));
        }

        // Iterate over edges 3-7 (inclusive)
        let start = 3 * EDGE_SIZE;
        let end = 8 * EDGE_SIZE;
        let iter = ZeroCopyIterator::new(&data, start, end);

        let edges: Vec<_> = iter.collect();
        assert_eq!(edges.len(), 5);
        assert_eq!(edges[0].edge_id(), 3);
        assert_eq!(edges[4].edge_id(), 7);
    }

    #[test]
    fn test_timestamp_range_filter() {
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend_from_slice(&create_test_edge(i as u128, i as u64 * 1000));
        }

        let iter = ZeroCopyIterator::all(&data);
        let filtered = TimestampRangeFilter::new(iter, 3000, 6000);

        let edges: Vec<_> = filtered.collect();
        assert_eq!(edges.len(), 4); // timestamps 3000, 4000, 5000, 6000
        assert_eq!(edges[0].timestamp_us(), 3000);
        assert_eq!(edges[3].timestamp_us(), 6000);
    }

    #[test]
    fn test_edge_ref_ordering() {
        let edge1_bytes = create_test_edge(1, 1000);
        let edge2_bytes = create_test_edge(2, 1000);
        let edge3_bytes = create_test_edge(1, 2000);

        let edge1 = EdgeRef::new(&edge1_bytes);
        let edge2 = EdgeRef::new(&edge2_bytes);
        let edge3 = EdgeRef::new(&edge3_bytes);

        // Same timestamp, different edge_id
        assert!(edge1 < edge2);

        // Different timestamp
        assert!(edge1 < edge3);
        assert!(edge2 < edge3);
    }

    #[test]
    fn test_seek_and_skip() {
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend_from_slice(&create_test_edge(i as u128, i as u64 * 1000));
        }

        let mut iter = ZeroCopyIterator::all(&data);

        // Skip first 3 edges
        iter.skip_edges(3);
        assert_eq!(iter.remaining_edges(), 7);

        let edge = iter.next().unwrap();
        assert_eq!(edge.edge_id(), 3);

        // Seek to edge 7
        iter.seek(7 * EDGE_SIZE);
        let edge = iter.next().unwrap();
        assert_eq!(edge.edge_id(), 7);
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut data = Vec::new();
        for i in 0..5 {
            data.extend_from_slice(&create_test_edge(i as u128, i as u64 * 1000));
        }

        let iter = ZeroCopyIterator::all(&data);
        assert_eq!(iter.len(), 5);

        let mut iter = ZeroCopyIterator::all(&data);
        iter.next();
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn test_prefetching_iterator() {
        let mut data = Vec::new();
        for i in 0..100 {
            data.extend_from_slice(&create_test_edge(i as u128, i as u64 * 1000));
        }

        let iter = PrefetchingZeroCopyIterator::with_default_prefetch(&data, 0, data.len());
        let edges: Vec<_> = iter.collect();

        assert_eq!(edges.len(), 100);
    }
}
