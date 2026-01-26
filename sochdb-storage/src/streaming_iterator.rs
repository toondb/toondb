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

//! Streaming Iterator Architecture for Scans
//!
//! This module implements memory-efficient streaming iterators that:
//! - Use O(k) memory for LIMIT k queries regardless of table size
//! - Enable zero-copy SSTable reads via mmap
//! - Support merge iteration over multiple sorted sources (LSM-tree style)
//! - Provide backpressure to prevent memory exhaustion
//!
//! ## Memory Complexity
//!
//! Current scan: M = O(N) where N = rows in scan range
//! Streaming:    M = O(k + S × log(S)) where k = LIMIT, S = number of sources
//!
//! For 100M rows with LIMIT 10:
//! - Current: 10GB allocation (100M × 100 bytes)
//! - Streaming: ~10KB (10 rows + heap overhead)

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A key-value entry with timestamp for MVCC
#[derive(Debug, Clone)]
pub struct Entry<'a> {
    pub key: Cow<'a, [u8]>,
    pub value: Cow<'a, [u8]>,
    pub timestamp: u64,
    pub is_tombstone: bool,
}

impl<'a> Entry<'a> {
    /// Create a new entry
    pub fn new(
        key: impl Into<Cow<'a, [u8]>>,
        value: impl Into<Cow<'a, [u8]>>,
        timestamp: u64,
    ) -> Self {
        let value = value.into();
        let is_tombstone = value.is_empty();
        Self {
            key: key.into(),
            value,
            timestamp,
            is_tombstone,
        }
    }

    /// Create a tombstone entry
    pub fn tombstone(key: impl Into<Cow<'a, [u8]>>, timestamp: u64) -> Self {
        Self {
            key: key.into(),
            value: Cow::Borrowed(&[]),
            timestamp,
            is_tombstone: true,
        }
    }

    /// Convert to owned version
    pub fn into_owned(self) -> Entry<'static> {
        Entry {
            key: Cow::Owned(self.key.into_owned()),
            value: Cow::Owned(self.value.into_owned()),
            timestamp: self.timestamp,
            is_tombstone: self.is_tombstone,
        }
    }
}

/// Trait for a streaming source of entries
pub trait EntryIterator<'a>: Send {
    /// Peek at the next entry without consuming it
    fn peek(&self) -> Option<&Entry<'a>>;

    /// Advance to the next entry
    fn advance(&mut self);

    /// Check if the iterator is exhausted
    fn is_exhausted(&self) -> bool;

    /// Get the source priority (lower = higher priority for same key)
    fn source_priority(&self) -> u8;
}

/// Peekable wrapper around an entry iterator
struct PeekableSource<'a> {
    source: Box<dyn EntryIterator<'a> + 'a>,
    priority: u8,
}

impl<'a> PeekableSource<'a> {
    fn new(source: Box<dyn EntryIterator<'a> + 'a>) -> Self {
        let priority = source.source_priority();
        Self { source, priority }
    }
}

impl<'a> PartialEq for PeekableSource<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self.source.peek(), other.source.peek()) {
            (Some(a), Some(b)) => a.key == b.key && self.priority == other.priority,
            (None, None) => true,
            _ => false,
        }
    }
}

impl<'a> Eq for PeekableSource<'a> {}

impl<'a> PartialOrd for PeekableSource<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for PeekableSource<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.source.peek(), other.source.peek()) {
            (Some(a), Some(b)) => {
                // Min-heap: reverse ordering
                // First by key (ascending), then by priority (ascending = higher priority)
                match b.key.cmp(&a.key) {
                    Ordering::Equal => other.priority.cmp(&self.priority),
                    ord => ord,
                }
            }
            (Some(_), None) => Ordering::Greater, // None goes last
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        }
    }
}

/// Merge iterator over multiple sorted sources (LSM-tree style)
pub struct MergeIterator<'a> {
    /// Min-heap of source iterators
    heap: BinaryHeap<PeekableSource<'a>>,
    /// Current key for deduplication
    current_key: Option<Vec<u8>>,
    /// Statistics
    stats: ScanStats,
}

impl<'a> MergeIterator<'a> {
    /// Create a new merge iterator from multiple sources
    pub fn new(sources: Vec<Box<dyn EntryIterator<'a> + 'a>>) -> Self {
        let mut heap = BinaryHeap::with_capacity(sources.len());

        for source in sources {
            let peekable = PeekableSource::new(source);
            if !peekable.source.is_exhausted() {
                heap.push(peekable);
            }
        }

        Self {
            heap,
            current_key: None,
            stats: ScanStats::default(),
        }
    }

    /// Get scan statistics
    pub fn stats(&self) -> &ScanStats {
        &self.stats
    }
}

impl<'a> Iterator for MergeIterator<'a> {
    type Item = Entry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Pop the source with smallest key
            let mut source = self.heap.pop()?;

            // Get the entry (must exist since we only push non-exhausted sources)
            let entry = source.source.peek()?.clone();

            // Advance this source
            source.source.advance();

            // Re-insert if not exhausted
            if !source.source.is_exhausted() {
                self.heap.push(source);
            }

            self.stats.entries_scanned += 1;

            // Skip duplicate keys (keep only newest version)
            if let Some(ref current) = self.current_key
                && current.as_slice() == entry.key.as_ref()
            {
                self.stats.duplicates_skipped += 1;
                continue;
            }

            self.current_key = Some(entry.key.to_vec());

            // Skip tombstones
            if entry.is_tombstone {
                self.stats.tombstones_skipped += 1;
                continue;
            }

            self.stats.entries_returned += 1;
            return Some(entry);
        }
    }
}

/// Scan statistics
#[derive(Debug, Default, Clone)]
pub struct ScanStats {
    pub entries_scanned: u64,
    pub entries_returned: u64,
    pub duplicates_skipped: u64,
    pub tombstones_skipped: u64,
}

/// Vector-based entry iterator for testing and simple cases
pub struct VecIterator<'a> {
    entries: Vec<Entry<'a>>,
    position: usize,
    priority: u8,
}

impl<'a> VecIterator<'a> {
    pub fn new(entries: Vec<Entry<'a>>, priority: u8) -> Self {
        Self {
            entries,
            position: 0,
            priority,
        }
    }
}

impl<'a> EntryIterator<'a> for VecIterator<'a> {
    fn peek(&self) -> Option<&Entry<'a>> {
        self.entries.get(self.position)
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn is_exhausted(&self) -> bool {
        self.position >= self.entries.len()
    }

    fn source_priority(&self) -> u8 {
        self.priority
    }
}

/// Range-bounded iterator wrapper
#[allow(dead_code)]
pub struct RangeIterator<'a, I: EntryIterator<'a>> {
    inner: I,
    start_key: Vec<u8>,
    end_key: Vec<u8>,
    started: bool,
    ended: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, I: EntryIterator<'a>> RangeIterator<'a, I> {
    pub fn new(inner: I, start_key: Vec<u8>, end_key: Vec<u8>) -> Self {
        Self {
            inner,
            start_key,
            end_key,
            started: false,
            ended: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, I: EntryIterator<'a> + Send> EntryIterator<'a> for RangeIterator<'a, I> {
    fn peek(&self) -> Option<&Entry<'a>> {
        if self.ended {
            return None;
        }

        let entry = self.inner.peek()?;

        // Check end bound
        if entry.key.as_ref() > self.end_key.as_slice() {
            return None;
        }

        Some(entry)
    }

    fn advance(&mut self) {
        if self.ended {
            return;
        }

        self.inner.advance();

        // Check if we've passed the end
        if let Some(entry) = self.inner.peek()
            && entry.key.as_ref() > self.end_key.as_slice()
        {
            self.ended = true;
        }
    }

    fn is_exhausted(&self) -> bool {
        self.ended || self.inner.is_exhausted()
    }

    fn source_priority(&self) -> u8 {
        self.inner.source_priority()
    }
}

/// Limit iterator wrapper
pub struct LimitIterator<'a, I: Iterator<Item = Entry<'a>>> {
    inner: I,
    limit: usize,
    count: usize,
}

impl<'a, I: Iterator<Item = Entry<'a>>> LimitIterator<'a, I> {
    pub fn new(inner: I, limit: usize) -> Self {
        Self {
            inner,
            limit,
            count: 0,
        }
    }
}

impl<'a, I: Iterator<Item = Entry<'a>>> Iterator for LimitIterator<'a, I> {
    type Item = Entry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.limit {
            return None;
        }
        self.count += 1;
        self.inner.next()
    }
}

/// Filter iterator for predicate pushdown
pub struct FilterIterator<'a, I, F>
where
    I: Iterator<Item = Entry<'a>>,
    F: Fn(&Entry<'a>) -> bool,
{
    inner: I,
    predicate: F,
}

impl<'a, I, F> FilterIterator<'a, I, F>
where
    I: Iterator<Item = Entry<'a>>,
    F: Fn(&Entry<'a>) -> bool,
{
    pub fn new(inner: I, predicate: F) -> Self {
        Self { inner, predicate }
    }
}

impl<'a, I, F> Iterator for FilterIterator<'a, I, F>
where
    I: Iterator<Item = Entry<'a>>,
    F: Fn(&Entry<'a>) -> bool,
{
    type Item = Entry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = self.inner.next()?;
            if (self.predicate)(&entry) {
                return Some(entry);
            }
        }
    }
}

/// Memtable iterator (in-memory sorted data)
pub struct MemtableIterator<'a> {
    entries: Vec<Entry<'a>>,
    position: usize,
}

impl<'a> MemtableIterator<'a> {
    pub fn new(entries: Vec<Entry<'a>>) -> Self {
        Self {
            entries,
            position: 0,
        }
    }

    pub fn from_btree<K, V>(tree: &'a std::collections::BTreeMap<K, V>, timestamp: u64) -> Self
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let entries: Vec<_> = tree
            .iter()
            .map(|(k, v)| {
                Entry::new(
                    Cow::Borrowed(k.as_ref()),
                    Cow::Borrowed(v.as_ref()),
                    timestamp,
                )
            })
            .collect();

        Self::new(entries)
    }
}

impl<'a> EntryIterator<'a> for MemtableIterator<'a> {
    fn peek(&self) -> Option<&Entry<'a>> {
        self.entries.get(self.position)
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn is_exhausted(&self) -> bool {
        self.position >= self.entries.len()
    }

    fn source_priority(&self) -> u8 {
        0 // Memtable has highest priority (newest data)
    }
}

/// SSTable iterator (simulated - real impl would use mmap)
pub struct SstIterator<'a> {
    /// Block data (simulating mmap region)
    data: &'a [u8],
    /// Current position in data
    position: usize,
    /// Cached current entry
    current: Option<Entry<'a>>,
    /// End key for range bound
    end_key: Option<Vec<u8>>,
    /// SSTable level (higher = lower priority)
    level: u8,
}

impl<'a> SstIterator<'a> {
    pub fn new(data: &'a [u8], level: u8) -> Self {
        let mut iter = Self {
            data,
            position: 0,
            current: None,
            end_key: None,
            level,
        };
        iter.read_next();
        iter
    }

    pub fn with_end_key(mut self, end_key: Vec<u8>) -> Self {
        self.end_key = Some(end_key);
        self
    }

    fn read_next(&mut self) {
        if self.position >= self.data.len() {
            self.current = None;
            return;
        }

        // Simple format: [key_len: u16][value_len: u16][timestamp: u64][key][value]
        if self.position + 12 > self.data.len() {
            self.current = None;
            return;
        }

        let key_len =
            u16::from_le_bytes([self.data[self.position], self.data[self.position + 1]]) as usize;
        let value_len =
            u16::from_le_bytes([self.data[self.position + 2], self.data[self.position + 3]])
                as usize;
        let timestamp = u64::from_le_bytes([
            self.data[self.position + 4],
            self.data[self.position + 5],
            self.data[self.position + 6],
            self.data[self.position + 7],
            self.data[self.position + 8],
            self.data[self.position + 9],
            self.data[self.position + 10],
            self.data[self.position + 11],
        ]);

        let key_start = self.position + 12;
        let key_end = key_start + key_len;
        let value_end = key_end + value_len;

        if value_end > self.data.len() {
            self.current = None;
            return;
        }

        let key = &self.data[key_start..key_end];
        let value = &self.data[key_end..value_end];

        // Check end bound
        if let Some(ref end_key) = self.end_key
            && key > end_key.as_slice()
        {
            self.current = None;
            return;
        }

        self.current = Some(Entry::new(
            Cow::Borrowed(key),
            Cow::Borrowed(value),
            timestamp,
        ));
        self.position = value_end;
    }
}

impl<'a> EntryIterator<'a> for SstIterator<'a> {
    fn peek(&self) -> Option<&Entry<'a>> {
        self.current.as_ref()
    }

    fn advance(&mut self) {
        self.read_next();
    }

    fn is_exhausted(&self) -> bool {
        self.current.is_none()
    }

    fn source_priority(&self) -> u8 {
        // Lower level = higher priority (L0 > L1 > L2...)
        // Memtable is priority 0, so SST levels start at 1
        self.level + 1
    }
}

/// Builder for SSTable data (for testing)
pub struct SstBuilder {
    data: Vec<u8>,
}

impl Default for SstBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SstBuilder {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn add(&mut self, key: &[u8], value: &[u8], timestamp: u64) -> &mut Self {
        self.data
            .extend_from_slice(&(key.len() as u16).to_le_bytes());
        self.data
            .extend_from_slice(&(value.len() as u16).to_le_bytes());
        self.data.extend_from_slice(&timestamp.to_le_bytes());
        self.data.extend_from_slice(key);
        self.data.extend_from_slice(value);
        self
    }

    pub fn build(self) -> Vec<u8> {
        self.data
    }
}

/// Extension trait for iterators
pub trait IteratorExt<'a>: Iterator<Item = Entry<'a>> + Sized {
    fn limit(self, n: usize) -> LimitIterator<'a, Self> {
        LimitIterator::new(self, n)
    }

    fn filter_entries<F>(self, predicate: F) -> FilterIterator<'a, Self, F>
    where
        F: Fn(&Entry<'a>) -> bool,
    {
        FilterIterator::new(self, predicate)
    }
}

impl<'a, I: Iterator<Item = Entry<'a>>> IteratorExt<'a> for I {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(key: &[u8], value: &[u8], ts: u64) -> Entry<'static> {
        Entry {
            key: Cow::Owned(key.to_vec()),
            value: Cow::Owned(value.to_vec()),
            timestamp: ts,
            is_tombstone: value.is_empty(),
        }
    }

    #[test]
    fn test_vec_iterator() {
        let entries = vec![
            make_entry(b"a", b"1", 100),
            make_entry(b"b", b"2", 100),
            make_entry(b"c", b"3", 100),
        ];

        let iter = VecIterator::new(entries, 0);
        assert!(!iter.is_exhausted());
        assert_eq!(iter.peek().unwrap().key.as_ref(), b"a");
    }

    #[test]
    fn test_merge_iterator_single_source() {
        let entries = vec![
            make_entry(b"a", b"1", 100),
            make_entry(b"b", b"2", 100),
            make_entry(b"c", b"3", 100),
        ];

        let source: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(entries, 0));
        let mut merge = MergeIterator::new(vec![source]);

        let result: Vec<_> = merge.by_ref().collect();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].key.as_ref(), b"a");
        assert_eq!(result[1].key.as_ref(), b"b");
        assert_eq!(result[2].key.as_ref(), b"c");
    }

    #[test]
    fn test_merge_iterator_multiple_sources() {
        let source1: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"a", b"1", 100),
                make_entry(b"c", b"3", 100),
                make_entry(b"e", b"5", 100),
            ],
            0,
        ));

        let source2: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"b", b"2", 100),
                make_entry(b"d", b"4", 100),
                make_entry(b"f", b"6", 100),
            ],
            1,
        ));

        let mut merge = MergeIterator::new(vec![source1, source2]);
        let result: Vec<_> = merge.by_ref().collect();

        assert_eq!(result.len(), 6);
        assert_eq!(result[0].key.as_ref(), b"a");
        assert_eq!(result[1].key.as_ref(), b"b");
        assert_eq!(result[2].key.as_ref(), b"c");
        assert_eq!(result[3].key.as_ref(), b"d");
        assert_eq!(result[4].key.as_ref(), b"e");
        assert_eq!(result[5].key.as_ref(), b"f");
    }

    #[test]
    fn test_merge_iterator_deduplication() {
        // Source 0 has higher priority (memtable)
        let source1: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"a", b"new_value", 200), // Newer version
            ],
            0,
        ));

        // Source 1 has lower priority (SSTable)
        let source2: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"a", b"old_value", 100), // Older version
                make_entry(b"b", b"2", 100),
            ],
            1,
        ));

        let mut merge = MergeIterator::new(vec![source1, source2]);
        let result: Vec<_> = merge.by_ref().collect();

        // Should have 2 entries, with 'a' having the new value
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].key.as_ref(), b"a");
        assert_eq!(result[0].value.as_ref(), b"new_value");
        assert_eq!(result[1].key.as_ref(), b"b");

        // Check stats
        assert_eq!(merge.stats().duplicates_skipped, 1);
    }

    #[test]
    fn test_merge_iterator_tombstones() {
        let source1: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                Entry::tombstone(Cow::Owned(b"a".to_vec()), 200), // Delete
            ],
            0,
        ));

        let source2: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"a", b"old_value", 100),
                make_entry(b"b", b"2", 100),
            ],
            1,
        ));

        let mut merge = MergeIterator::new(vec![source1, source2]);
        let result: Vec<_> = merge.by_ref().collect();

        // 'a' should be skipped (tombstone), only 'b' remains
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].key.as_ref(), b"b");

        assert_eq!(merge.stats().tombstones_skipped, 1);
    }

    #[test]
    fn test_limit_iterator() {
        let entries = vec![
            make_entry(b"a", b"1", 100),
            make_entry(b"b", b"2", 100),
            make_entry(b"c", b"3", 100),
            make_entry(b"d", b"4", 100),
            make_entry(b"e", b"5", 100),
        ];

        let source: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(entries, 0));
        let merge = MergeIterator::new(vec![source]);

        let result: Vec<_> = merge.limit(3).collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].key.as_ref(), b"a");
        assert_eq!(result[2].key.as_ref(), b"c");
    }

    #[test]
    fn test_filter_iterator() {
        let entries = vec![
            make_entry(b"a", b"1", 100),
            make_entry(b"b", b"2", 100),
            make_entry(b"c", b"3", 100),
            make_entry(b"d", b"4", 100),
        ];

        let source: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(entries, 0));
        let merge = MergeIterator::new(vec![source]);

        // Filter to only keys < "c"
        let result: Vec<_> = merge
            .filter_entries(|e| e.key.as_ref() < b"c".as_slice())
            .collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].key.as_ref(), b"a");
        assert_eq!(result[1].key.as_ref(), b"b");
    }

    #[test]
    fn test_sst_builder_and_iterator() {
        let mut builder = SstBuilder::new();
        builder
            .add(b"apple", b"red", 100)
            .add(b"banana", b"yellow", 100)
            .add(b"cherry", b"red", 100);

        let data = builder.build();
        let iter = SstIterator::new(&data, 0);

        assert!(!iter.is_exhausted());
        assert_eq!(iter.peek().unwrap().key.as_ref(), b"apple");
    }

    #[test]
    fn test_sst_iterator_full() {
        let mut builder = SstBuilder::new();
        builder
            .add(b"a", b"1", 100)
            .add(b"b", b"2", 200)
            .add(b"c", b"3", 300);

        let data = builder.build();
        let mut iter = SstIterator::new(&data, 0);

        let mut results = Vec::new();
        while !iter.is_exhausted() {
            results.push(iter.peek().unwrap().clone());
            iter.advance();
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key.as_ref(), b"a");
        assert_eq!(results[0].timestamp, 100);
        assert_eq!(results[1].key.as_ref(), b"b");
        assert_eq!(results[1].timestamp, 200);
        assert_eq!(results[2].key.as_ref(), b"c");
        assert_eq!(results[2].timestamp, 300);
    }

    #[test]
    fn test_sst_with_merge() {
        let mut builder1 = SstBuilder::new();
        builder1.add(b"a", b"1", 100).add(b"c", b"3", 100);
        let data1 = builder1.build();

        let mut builder2 = SstBuilder::new();
        builder2.add(b"b", b"2", 100).add(b"d", b"4", 100);
        let data2 = builder2.build();

        // Need to use 'static lifetime for the test
        // In real usage, data would be mmap'd and outlive the iterator
        let data1_static: &'static [u8] = Box::leak(data1.into_boxed_slice());
        let data2_static: &'static [u8] = Box::leak(data2.into_boxed_slice());

        let source1: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(SstIterator::new(data1_static, 0));
        let source2: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(SstIterator::new(data2_static, 1));

        let mut merge = MergeIterator::new(vec![source1, source2]);
        let result: Vec<_> = merge.by_ref().collect();

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].key.as_ref(), b"a");
        assert_eq!(result[1].key.as_ref(), b"b");
        assert_eq!(result[2].key.as_ref(), b"c");
        assert_eq!(result[3].key.as_ref(), b"d");
    }

    #[test]
    fn test_memory_efficiency() {
        // This test demonstrates O(k) memory for LIMIT k
        // Create "large" sources (simulated)
        let entries1: Vec<Entry<'static>> = (0..100)
            .map(|i| make_entry(format!("key{:05}", i * 2).as_bytes(), b"value", 100))
            .collect();
        let entries2: Vec<Entry<'static>> = (0..100)
            .map(|i| make_entry(format!("key{:05}", i * 2 + 1).as_bytes(), b"value", 100))
            .collect();

        let source1: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(entries1, 0));
        let source2: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(entries2, 1));

        let merge = MergeIterator::new(vec![source1, source2]);

        // Only take 10 - should use O(10) output memory, not O(200)
        let result: Vec<_> = merge.limit(10).collect();

        assert_eq!(result.len(), 10);
        assert_eq!(result[0].key.as_ref(), b"key00000");
        assert_eq!(result[9].key.as_ref(), b"key00009");
    }

    #[test]
    fn test_empty_sources() {
        let sources: Vec<Box<dyn EntryIterator<'static> + 'static>> = vec![
            Box::new(VecIterator::new(vec![], 0)),
            Box::new(VecIterator::new(vec![], 1)),
        ];

        let mut merge = MergeIterator::new(sources);
        let result: Vec<_> = merge.by_ref().collect();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_scan_stats() {
        let source1: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"a", b"1", 200),
                make_entry(b"b", b"", 200), // tombstone
            ],
            0,
        ));

        let source2: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![
                make_entry(b"a", b"old", 100), // duplicate
                make_entry(b"c", b"3", 100),
            ],
            1,
        ));

        let mut merge = MergeIterator::new(vec![source1, source2]);
        let _: Vec<_> = merge.by_ref().collect();

        let stats = merge.stats();
        assert_eq!(stats.entries_scanned, 4);
        assert_eq!(stats.entries_returned, 2); // a, c
        assert_eq!(stats.duplicates_skipped, 1); // old 'a'
        assert_eq!(stats.tombstones_skipped, 1); // 'b'
    }

    #[test]
    fn test_priority_ordering() {
        // Same key in multiple sources - highest priority wins
        let source1: Box<dyn EntryIterator<'static> + 'static> = Box::new(VecIterator::new(
            vec![make_entry(b"key", b"memtable", 300)],
            0,
        )); // Highest priority

        let source2: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(vec![make_entry(b"key", b"l0", 200)], 1));

        let source3: Box<dyn EntryIterator<'static> + 'static> =
            Box::new(VecIterator::new(vec![make_entry(b"key", b"l1", 100)], 2)); // Lowest priority

        let mut merge = MergeIterator::new(vec![source1, source2, source3]);
        let result: Vec<_> = merge.by_ref().collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value.as_ref(), b"memtable");
    }
}
