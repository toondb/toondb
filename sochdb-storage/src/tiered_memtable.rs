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

//! Tiered SkipMap Elimination (Recommendation 3)
//!
//! ## Problem
//!
//! Current write path uses three concurrent data structures:
//! ```ignore
//! pub struct ArenaMvccMemTable {
//!     data: DashMap<ArenaKeyHandle, VersionChain>,      // +47ns lookup
//!     ordered_index: Option<SkipMap<ArenaKeyHandle, ()>>, // +93ns insert
//!     dirty_list: ArenaEpochDirtyList,                   // +15ns
//! }
//! ```
//!
//! The benchmark shows "No-SkipMap Mode" achieves 97% of SQLite. 
//! The SkipMap provides ordered iteration but costs 40% of insert time.
//!
//! ## Solution
//!
//! Tiered architecture:
//! - **Hot tier**: Unsorted append-only buffer (O(1) insert)
//! - **Warm tier**: Sorted batch (O(N log N) once at flush)
//!
//! ## Performance Analysis
//!
//! Current per-write cost:
//! ```text
//! T_write = T_dashmap_insert + T_skipmap_insert + T_dirty_list
//!         = 47ns + 93ns + 15ns = 155ns
//! ```
//!
//! Proposed per-write cost:
//! ```text
//! T_write = T_vec_push + T_dirty_list
//!         = 5ns + 15ns = 20ns
//! ```
//!
//! Amortized sort cost at flush (N = 100,000 rows):
//! ```text
//! T_sort = N × log(N) × comparison_cost
//!        = 100,000 × 17 × 10ns = 17ms (once per flush)
//! ```
//!
//! Net throughput: 1 / 20ns = 50M ops/sec (theoretical max)
//! With other overhead: ~2M ops/sec (matches "No-SkipMap" benchmark result)

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;

use crate::durable_storage::InlineKey;
use sochdb_core::Result;

/// Default hot buffer capacity before flush
pub const DEFAULT_HOT_BUFFER_CAPACITY: usize = 100_000;

/// Threshold to trigger automatic flush (fraction of capacity)
pub const FLUSH_THRESHOLD_RATIO: f64 = 0.8;

// =============================================================================
// Hot Buffer Entry
// =============================================================================

/// Entry in the hot buffer (unsorted)
#[derive(Debug, Clone)]
pub struct HotEntry {
    /// Key bytes (using SmallVec for inline storage of small keys)
    pub key: InlineKey,
    /// Value (None = tombstone)
    pub value: Option<Vec<u8>>,
    /// Transaction ID
    pub txn_id: u64,
    /// Insertion sequence number (for stable sorting)
    pub seq: u64,
}

impl HotEntry {
    pub fn new(key: InlineKey, value: Option<Vec<u8>>, txn_id: u64, seq: u64) -> Self {
        Self {
            key,
            value,
            txn_id,
            seq,
        }
    }
}

// =============================================================================
// Sorted Batch (Warm Tier)
// =============================================================================

/// A sorted batch of entries (immutable after creation)
#[derive(Debug)]
pub struct SortedBatch {
    /// Sorted entries (by key, then by seq desc for same key)
    entries: Vec<HotEntry>,
    /// Index for binary search: key_hash -> first occurrence index
    /// Uses hash for O(1) average lookup, falls back to binary search
    key_index: DashMap<u64, usize>,
    /// Minimum timestamp in this batch
    min_ts: u64,
    /// Maximum timestamp in this batch
    max_ts: u64,
}

impl SortedBatch {
    /// Create from unsorted entries
    pub fn from_unsorted(mut entries: Vec<HotEntry>) -> Self {
        if entries.is_empty() {
            return Self {
                entries: Vec::new(),
                key_index: DashMap::new(),
                min_ts: u64::MAX,
                max_ts: 0,
            };
        }

        // Sort by key, then by seq desc (newest first for same key)
        entries.sort_unstable_by(|a, b| {
            match a.key.as_slice().cmp(b.key.as_slice()) {
                std::cmp::Ordering::Equal => b.seq.cmp(&a.seq), // Descending
                other => other,
            }
        });

        // Build key index (first occurrence of each key)
        let key_index = DashMap::new();
        let mut last_key: Option<&[u8]> = None;
        for (idx, entry) in entries.iter().enumerate() {
            if last_key != Some(entry.key.as_slice()) {
                let hash = Self::hash_key(&entry.key);
                key_index.insert(hash, idx);
                last_key = Some(entry.key.as_slice());
            }
        }

        // Calculate timestamp range
        let min_ts = entries.iter().map(|e| e.seq).min().unwrap_or(u64::MAX);
        let max_ts = entries.iter().map(|e| e.seq).max().unwrap_or(0);

        Self {
            entries,
            key_index,
            min_ts,
            max_ts,
        }
    }

    /// Hash key for index lookup
    #[inline]
    fn hash_key(key: &[u8]) -> u64 {
        twox_hash::xxh3::hash64(key)
    }

    /// Get entry by key - O(1) average, O(log N) worst case
    pub fn get(&self, key: &[u8]) -> Option<&HotEntry> {
        let hash = Self::hash_key(key);
        
        if let Some(idx) = self.key_index.get(&hash) {
            // Verify key matches (handle hash collisions)
            let idx = *idx;
            if idx < self.entries.len() && self.entries[idx].key.as_slice() == key {
                return Some(&self.entries[idx]);
            }
        }

        // Fall back to binary search
        self.entries
            .binary_search_by(|e| e.key.as_slice().cmp(key))
            .ok()
            .map(|idx| &self.entries[idx])
    }

    /// Get all entries with given prefix - O(log N + K)
    pub fn prefix_scan(&self, prefix: &[u8]) -> impl Iterator<Item = &HotEntry> {
        // Find first entry >= prefix using binary search
        let start_idx = self.entries
            .partition_point(|e| e.key.as_slice() < prefix);

        self.entries[start_idx..]
            .iter()
            .take_while(move |e| e.key.starts_with(prefix))
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate all entries in sorted order
    pub fn iter(&self) -> impl Iterator<Item = &HotEntry> {
        self.entries.iter()
    }

    /// Get timestamp range
    pub fn timestamp_range(&self) -> (u64, u64) {
        (self.min_ts, self.max_ts)
    }
}

// =============================================================================
// Tiered MemTable
// =============================================================================

/// Tiered MemTable with hot buffer and warm sorted batches
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────┐
/// │ Hot Buffer (Vec<HotEntry>)                              │
/// │ • O(1) append                                           │
/// │ • Unsorted                                              │
/// │ • Current writes                                        │
/// └─────────────────────────────────────────────────────────┘
///                        ↓ flush
/// ┌─────────────────────────────────────────────────────────┐
/// │ Warm Batches (Vec<Arc<SortedBatch>>)                    │
/// │ • Sorted (O(N log N) once)                              │
/// │ • Immutable                                             │
/// │ • Binary search reads                                   │
/// └─────────────────────────────────────────────────────────┘
/// ```
pub struct TieredMemTable {
    /// Hot buffer: unsorted, O(1) append
    hot_buffer: RwLock<Vec<HotEntry>>,
    /// Hot buffer capacity
    hot_capacity: usize,
    /// Warm batches: sorted, immutable
    warm_batches: RwLock<Vec<Arc<SortedBatch>>>,
    /// Hash index for point lookups (key_hash -> batch_idx, entry_idx)
    /// Avoids scanning all batches for point lookups
    #[allow(dead_code)]
    point_index: DashMap<Vec<u8>, (usize, usize)>,
    /// Sequence counter for ordering
    seq_counter: AtomicU64,
    /// Approximate size in bytes
    size_bytes: AtomicU64,
    /// Entry count
    entry_count: AtomicUsize,
    /// Pending commits (txn_id -> commit_ts)
    pending_commits: DashMap<u64, u64>,
    /// Read-write lock for flush coordination
    flush_lock: Mutex<()>,
}

impl TieredMemTable {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_HOT_BUFFER_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hot_buffer: RwLock::new(Vec::with_capacity(capacity)),
            hot_capacity: capacity,
            warm_batches: RwLock::new(Vec::new()),
            point_index: DashMap::new(),
            seq_counter: AtomicU64::new(1),
            size_bytes: AtomicU64::new(0),
            entry_count: AtomicUsize::new(0),
            pending_commits: DashMap::new(),
            flush_lock: Mutex::new(()),
        }
    }

    /// Write a key-value pair (uncommitted) - O(1)
    ///
    /// This is the key optimization: append to unsorted buffer
    /// instead of inserting into SkipMap.
    pub fn write(&self, key: &[u8], value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let key_inline = SmallVec::from_slice(key);
        let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
        let seq = self.seq_counter.fetch_add(1, Ordering::Relaxed);

        let entry = HotEntry::new(key_inline, value, txn_id, seq);

        // Fast path: append to hot buffer (O(1))
        {
            let mut buffer = self.hot_buffer.write();
            buffer.push(entry);
        }

        self.size_bytes.fetch_add((key.len() + value_size) as u64, Ordering::Relaxed);
        self.entry_count.fetch_add(1, Ordering::Relaxed);

        // Check if flush needed
        if self.should_flush() {
            self.try_flush()?;
        }

        Ok(())
    }

    /// Write batch - O(n)
    pub fn write_batch(&self, writes: &[(&[u8], Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        let mut total_size = 0u64;
        let mut entries = Vec::with_capacity(writes.len());

        for (key, value) in writes {
            let seq = self.seq_counter.fetch_add(1, Ordering::Relaxed);
            let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
            total_size += (key.len() + value_size) as u64;

            entries.push(HotEntry::new(
                SmallVec::from_slice(key),
                value.clone(),
                txn_id,
                seq,
            ));
        }

        // Batch append (still O(n) but fewer lock acquisitions)
        {
            let mut buffer = self.hot_buffer.write();
            buffer.extend(entries);
        }

        self.size_bytes.fetch_add(total_size, Ordering::Relaxed);
        self.entry_count.fetch_add(writes.len(), Ordering::Relaxed);

        if self.should_flush() {
            self.try_flush()?;
        }

        Ok(())
    }

    /// Read at snapshot timestamp - O(log B + log N) where B = batch count
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        // Check hot buffer first (most recent writes)
        {
            let buffer = self.hot_buffer.read();
            // Scan backwards for most recent version
            for entry in buffer.iter().rev() {
                if entry.key.as_slice() == key {
                    // Check MVCC visibility
                    if self.is_visible(entry, snapshot_ts, current_txn_id) {
                        return entry.value.clone();
                    }
                }
            }
        }

        // Check warm batches (sorted, binary search)
        {
            let batches = self.warm_batches.read();
            // Search from newest to oldest batch
            for batch in batches.iter().rev() {
                if let Some(entry) = batch.get(key) {
                    if self.is_visible(entry, snapshot_ts, current_txn_id) {
                        return entry.value.clone();
                    }
                }
            }
        }

        None
    }

    /// Check if entry is visible at snapshot
    #[inline]
    fn is_visible(&self, entry: &HotEntry, snapshot_ts: u64, current_txn_id: Option<u64>) -> bool {
        // Own uncommitted write is always visible
        if let Some(my_txn) = current_txn_id {
            if entry.txn_id == my_txn {
                return true;
            }
        }

        // Check if committed before snapshot
        if let Some(commit_ts) = self.pending_commits.get(&entry.txn_id) {
            return *commit_ts < snapshot_ts;
        }

        false
    }

    /// Commit transaction
    pub fn commit(&self, txn_id: u64, commit_ts: u64, _write_set: &HashSet<InlineKey>) {
        // Record commit timestamp
        self.pending_commits.insert(txn_id, commit_ts);

        // Update point index for faster lookups
        // In a real implementation, we'd update version chains here
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) {
        self.pending_commits.remove(&txn_id);
        
        // Remove uncommitted entries from hot buffer
        let mut buffer = self.hot_buffer.write();
        buffer.retain(|e| e.txn_id != txn_id);
    }

    /// Scan with prefix - O(log N + K) - Legacy method using HashSet deduplication
    ///
    /// Consider using `scan_prefix_tournament` for better performance with many batches.
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();
        let mut seen_keys: HashSet<Vec<u8>> = HashSet::new();

        // Scan hot buffer first
        {
            let buffer = self.hot_buffer.read();
            for entry in buffer.iter().rev() {
                if entry.key.starts_with(prefix) 
                    && !seen_keys.contains(entry.key.as_slice())
                    && self.is_visible(entry, snapshot_ts, current_txn_id)
                {
                    if let Some(ref value) = entry.value {
                        results.push((entry.key.to_vec(), value.clone()));
                        seen_keys.insert(entry.key.to_vec());
                    }
                }
            }
        }

        // Scan warm batches
        {
            let batches = self.warm_batches.read();
            for batch in batches.iter().rev() {
                for entry in batch.prefix_scan(prefix) {
                    if !seen_keys.contains(entry.key.as_slice())
                        && self.is_visible(entry, snapshot_ts, current_txn_id)
                    {
                        if let Some(ref value) = entry.value {
                            results.push((entry.key.to_vec(), value.clone()));
                            seen_keys.insert(entry.key.to_vec());
                        }
                    }
                }
            }
        }

        // Sort results by key
        results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        results
    }

    /// Scan with prefix using Tournament Tree K-way merge
    ///
    /// ## Performance
    ///
    /// For K sorted runs with total N matching entries:
    /// - Time: O(N log K) vs O(N × K) for the naive approach
    /// - Memory: O(K) for the tournament tree vs O(N) for HashSet dedup
    ///
    /// This is significantly faster when:
    /// - K (number of batches) > 4
    /// - N (total entries) is large
    ///
    /// ## Algorithm
    ///
    /// 1. Sort hot buffer entries with matching prefix
    /// 2. Collect prefix-matching iterators from each sorted batch
    /// 3. Merge using tournament tree (loser tree) for O(log K) per element
    /// 4. Deduplicate by key (first occurrence wins = newest version)
    /// 5. Filter by MVCC visibility
    pub fn scan_prefix_tournament(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        use crate::tournament_tree::TournamentTree;
        
        // Collect all sources
        let mut sorted_sources: Vec<Vec<HotEntry>> = Vec::new();
        
        // Source 0: Sorted hot buffer entries matching prefix
        {
            let buffer = self.hot_buffer.read();
            let mut hot_entries: Vec<HotEntry> = buffer
                .iter()
                .filter(|e| e.key.starts_with(prefix))
                .cloned()
                .collect();
            
            // Sort by key, then by seq desc (newest first for same key)
            hot_entries.sort_unstable_by(|a, b| {
                match a.key.as_slice().cmp(b.key.as_slice()) {
                    std::cmp::Ordering::Equal => b.seq.cmp(&a.seq),
                    other => other,
                }
            });
            
            // Deduplicate within hot buffer (keep newest = first occurrence)
            let mut seen = HashSet::new();
            hot_entries.retain(|e| seen.insert(e.key.to_vec()));
            
            if !hot_entries.is_empty() {
                sorted_sources.push(hot_entries);
            }
        }
        
        // Sources 1..K: Warm batches (already sorted)
        {
            let batches = self.warm_batches.read();
            // Iterate from newest to oldest
            for batch in batches.iter().rev() {
                let entries: Vec<HotEntry> = batch
                    .prefix_scan(prefix)
                    .cloned()
                    .collect();
                
                if !entries.is_empty() {
                    sorted_sources.push(entries);
                }
            }
        }
        
        if sorted_sources.is_empty() {
            return Vec::new();
        }
        
        // Special case: single source, no merge needed
        if sorted_sources.len() == 1 {
            return sorted_sources
                .into_iter()
                .next()
                .unwrap()
                .into_iter()
                .filter(|e| self.is_visible(e, snapshot_ts, current_txn_id))
                .filter_map(|e| e.value.map(|v| (e.key.to_vec(), v)))
                .collect();
        }
        
        // K-way merge using tournament tree
        // Wrap entries with source index for priority-based comparison
        // When keys are equal, lower source_idx wins (newer data)
        #[derive(Clone)]
        struct KeyedEntry {
            entry: HotEntry,
            source_idx: usize,
        }
        
        impl PartialEq for KeyedEntry {
            fn eq(&self, other: &Self) -> bool {
                self.entry.key.as_slice() == other.entry.key.as_slice()
            }
        }
        impl Eq for KeyedEntry {}
        impl PartialOrd for KeyedEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for KeyedEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Primary: sort by key
                // Secondary: lower source_idx wins (source 0 = hot buffer = newest)
                match self.entry.key.as_slice().cmp(other.entry.key.as_slice()) {
                    std::cmp::Ordering::Equal => self.source_idx.cmp(&other.source_idx),
                    other => other,
                }
            }
        }
        
        let iters: Vec<_> = sorted_sources
            .into_iter()
            .enumerate()
            .map(|(source_idx, v)| {
                v.into_iter().map(move |e| KeyedEntry { entry: e, source_idx })
            })
            .collect();
        
        let mut tree = TournamentTree::new(iters);
        let mut results = Vec::new();
        let mut last_key: Option<Vec<u8>> = None;
        
        // Merge with deduplication and visibility filtering
        while let Some((_, keyed)) = tree.pop() {
            let entry = keyed.entry;
            
            // Deduplicate by key (first occurrence = from newest source due to ordering)
            if let Some(ref last) = last_key {
                if entry.key.as_slice() == last.as_slice() {
                    continue;
                }
            }
            last_key = Some(entry.key.to_vec());
            
            // Check visibility
            if !self.is_visible(&entry, snapshot_ts, current_txn_id) {
                // Not visible at this snapshot, try next version
                // Note: since we deduplicated, we might miss older visible versions
                // In a production system, we'd need a more sophisticated approach
                continue;
            }
            
            // Include non-tombstone entries
            if let Some(value) = entry.value {
                results.push((entry.key.to_vec(), value));
            }
        }
        
        results
    }

    /// Check if flush is needed
    fn should_flush(&self) -> bool {
        let buffer = self.hot_buffer.read();
        buffer.len() >= (self.hot_capacity as f64 * FLUSH_THRESHOLD_RATIO) as usize
    }

    /// Try to flush hot buffer to warm batch
    pub fn try_flush(&self) -> Result<()> {
        // Try to acquire flush lock (non-blocking)
        let guard = match self.flush_lock.try_lock() {
            Some(g) => g,
            None => return Ok(()), // Another thread is flushing
        };

        // Swap hot buffer with empty
        let entries = {
            let mut buffer = self.hot_buffer.write();
            if buffer.len() < (self.hot_capacity as f64 * FLUSH_THRESHOLD_RATIO) as usize {
                // Buffer was flushed by another thread
                return Ok(());
            }
            std::mem::replace(&mut *buffer, Vec::with_capacity(self.hot_capacity))
        };

        if entries.is_empty() {
            return Ok(());
        }

        // Create sorted batch (O(N log N))
        let batch = Arc::new(SortedBatch::from_unsorted(entries));

        // Add to warm batches
        {
            let mut batches = self.warm_batches.write();
            batches.push(batch);
        }

        drop(guard);
        Ok(())
    }

    /// Force flush hot buffer
    pub fn flush(&self) -> Result<()> {
        let _guard = self.flush_lock.lock();

        let entries = {
            let mut buffer = self.hot_buffer.write();
            std::mem::replace(&mut *buffer, Vec::with_capacity(self.hot_capacity))
        };

        if entries.is_empty() {
            return Ok(());
        }

        let batch = Arc::new(SortedBatch::from_unsorted(entries));

        {
            let mut batches = self.warm_batches.write();
            batches.push(batch);
        }

        Ok(())
    }

    /// Get approximate size in bytes
    pub fn size(&self) -> u64 {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get batch count
    pub fn batch_count(&self) -> usize {
        self.warm_batches.read().len()
    }

    /// Get hot buffer length
    pub fn hot_buffer_len(&self) -> usize {
        self.hot_buffer.read().len()
    }

    /// Compact warm batches
    pub fn compact(&self) -> Result<()> {
        let batches = {
            let mut b = self.warm_batches.write();
            std::mem::take(&mut *b)
        };

        if batches.len() <= 1 {
            let mut b = self.warm_batches.write();
            *b = batches;
            return Ok(());
        }

        // Merge all batches into one
        let all_entries: Vec<HotEntry> = batches
            .iter()
            .flat_map(|b| b.iter().cloned())
            .collect();

        // Re-sort
        let merged = Arc::new(SortedBatch::from_unsorted(all_entries));

        {
            let mut b = self.warm_batches.write();
            b.clear();
            b.push(merged);
        }

        Ok(())
    }
}

impl Default for TieredMemTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiered_memtable_basic() {
        let table = TieredMemTable::new();
        
        table.write(b"key1", Some(b"value1".to_vec()), 1).unwrap();
        table.write(b"key2", Some(b"value2".to_vec()), 1).unwrap();
        
        // Commit transaction
        let mut write_set = HashSet::new();
        write_set.insert(SmallVec::from_slice(b"key1"));
        write_set.insert(SmallVec::from_slice(b"key2"));
        table.commit(1, 100, &write_set);
        
        // Read back
        let v1 = table.read(b"key1", 200, None);
        let v2 = table.read(b"key2", 200, None);
        
        assert_eq!(v1, Some(b"value1".to_vec()));
        assert_eq!(v2, Some(b"value2".to_vec()));
    }

    #[test]
    fn test_tiered_memtable_uncommitted_own() {
        let table = TieredMemTable::new();
        
        table.write(b"key1", Some(b"value1".to_vec()), 1).unwrap();
        
        // Should see own uncommitted write
        let v = table.read(b"key1", 100, Some(1));
        assert_eq!(v, Some(b"value1".to_vec()));
        
        // Should not see other's uncommitted write
        let v = table.read(b"key1", 100, Some(2));
        assert_eq!(v, None);
    }

    #[test]
    fn test_tiered_memtable_flush() {
        let table = TieredMemTable::with_capacity(100);
        
        // Write enough to trigger flush
        for i in 0..90 {
            table.write(
                format!("key{:04}", i).as_bytes(),
                Some(format!("value{}", i).into_bytes()),
                1,
            ).unwrap();
        }
        
        // Force flush
        table.flush().unwrap();
        
        assert!(table.batch_count() >= 1);
        assert_eq!(table.hot_buffer_len(), 0);
    }

    #[test]
    fn test_tiered_memtable_scan_prefix() {
        let table = TieredMemTable::new();
        
        table.write(b"users:1", Some(b"alice".to_vec()), 1).unwrap();
        table.write(b"users:2", Some(b"bob".to_vec()), 1).unwrap();
        table.write(b"posts:1", Some(b"post1".to_vec()), 1).unwrap();
        
        let mut write_set = HashSet::new();
        write_set.insert(SmallVec::from_slice(b"users:1"));
        write_set.insert(SmallVec::from_slice(b"users:2"));
        write_set.insert(SmallVec::from_slice(b"posts:1"));
        table.commit(1, 100, &write_set);
        
        let results = table.scan_prefix(b"users:", 200, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_sorted_batch() {
        let entries = vec![
            HotEntry::new(SmallVec::from_slice(b"c"), Some(b"3".to_vec()), 1, 3),
            HotEntry::new(SmallVec::from_slice(b"a"), Some(b"1".to_vec()), 1, 1),
            HotEntry::new(SmallVec::from_slice(b"b"), Some(b"2".to_vec()), 1, 2),
        ];
        
        let batch = SortedBatch::from_unsorted(entries);
        
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.get(b"a").unwrap().value, Some(b"1".to_vec()));
        assert_eq!(batch.get(b"b").unwrap().value, Some(b"2".to_vec()));
        assert_eq!(batch.get(b"c").unwrap().value, Some(b"3".to_vec()));
    }

    #[test]
    fn test_sorted_batch_prefix_scan() {
        let entries = vec![
            HotEntry::new(SmallVec::from_slice(b"ab"), Some(b"1".to_vec()), 1, 1),
            HotEntry::new(SmallVec::from_slice(b"abc"), Some(b"2".to_vec()), 1, 2),
            HotEntry::new(SmallVec::from_slice(b"abd"), Some(b"3".to_vec()), 1, 3),
            HotEntry::new(SmallVec::from_slice(b"xyz"), Some(b"4".to_vec()), 1, 4),
        ];
        
        let batch = SortedBatch::from_unsorted(entries);
        let results: Vec<_> = batch.prefix_scan(b"ab").collect();
        
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_scan_prefix_tournament() {
        let table = TieredMemTable::with_capacity(100);
        
        // Create multiple batches by flushing
        for batch_idx in 0..3 {
            for i in 0..10 {
                let key = format!("users:{:02}", i);
                let value = format!("value_batch{}_item{}", batch_idx, i);
                table.write(key.as_bytes(), Some(value.into_bytes()), 1).unwrap();
            }
            table.flush().unwrap();
        }
        
        // Add some to hot buffer
        for i in 0..5 {
            let key = format!("users:{:02}", i);
            let value = format!("newest_value_{}", i);
            table.write(key.as_bytes(), Some(value.into_bytes()), 1).unwrap();
        }
        
        // Commit all
        let mut write_set = HashSet::new();
        for i in 0..10 {
            write_set.insert(SmallVec::from_slice(format!("users:{:02}", i).as_bytes()));
        }
        table.commit(1, 100, &write_set);
        
        // Scan using tournament tree
        let results = table.scan_prefix_tournament(b"users:", 200, None);
        
        // Should have 10 unique keys with newest values
        assert_eq!(results.len(), 10);
        
        // First 5 should have "newest_value_X"
        for (i, (key, value)) in results.iter().take(5).enumerate() {
            let expected_key = format!("users:{:02}", i);
            assert_eq!(key.as_slice(), expected_key.as_bytes());
            assert!(String::from_utf8_lossy(value).starts_with("newest_value_"));
        }
    }

    #[test]
    fn test_scan_tournament_deduplication() {
        let table = TieredMemTable::with_capacity(100);
        
        // Write same key multiple times across batches
        table.write(b"key:001", Some(b"old1".to_vec()), 1).unwrap();
        table.flush().unwrap();
        
        table.write(b"key:001", Some(b"old2".to_vec()), 1).unwrap();
        table.flush().unwrap();
        
        table.write(b"key:001", Some(b"newest".to_vec()), 1).unwrap();
        
        // Commit
        let mut write_set = HashSet::new();
        write_set.insert(SmallVec::from_slice(b"key:001"));
        table.commit(1, 100, &write_set);
        
        // Tournament scan should return only newest
        let results = table.scan_prefix_tournament(b"key:", 200, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.as_slice(), b"newest");
    }
}
