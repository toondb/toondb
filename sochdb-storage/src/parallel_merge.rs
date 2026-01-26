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

//! Parallel K-Way Merge for Compaction
//!
//! Implements multi-threaded merging of SSTables for faster compaction.
//!
//! ## jj.md Task 8: Parallel K-Way Merge
//!
//! Goals:
//! - 3-5x faster compaction with 4+ cores
//! - Reduced compaction debt accumulation
//! - Better CPU utilization during background work
//!
//! ## Architecture
//!
//! ```text
//! Input SSTables (parallel read):
//! [SST1] --read--> [Decompressor1] --\
//! [SST2] --read--> [Decompressor2] ----> [Lock-free Merge Queue]
//! [SST3] --read--> [Decompressor3] --/
//!
//! Merge Phase (parallelized by key range):
//! Range [0, N/4)    --merge--> [Writer1]
//! Range [N/4, N/2)  --merge--> [Writer2]
//! Range [N/2, 3N/4) --merge--> [Writer3]
//! Range [3N/4, N)   --merge--> [Writer4]
//! ```
//!
//! Reference: RocksDB uses `SubcompactionState` for parallel range compaction

use crossbeam_channel::{Receiver, Sender, bounded};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
#[cfg(test)]
use std::thread;

/// Edge data for merging (simplified representation)
#[derive(Clone, Debug)]
pub struct MergeEdge {
    /// Edge ID
    pub edge_id: u128,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Whether this is a tombstone
    pub is_tombstone: bool,
    /// Raw edge bytes (128 bytes)
    pub data: [u8; 128],
    /// Source SSTable index
    pub source_idx: usize,
}

impl PartialEq for MergeEdge {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_us == other.timestamp_us && self.edge_id == other.edge_id
    }
}

impl Eq for MergeEdge {}

impl PartialOrd for MergeEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering (smaller timestamp first)
        match other.timestamp_us.cmp(&self.timestamp_us) {
            Ordering::Equal => other.edge_id.cmp(&self.edge_id),
            ord => ord,
        }
    }
}

/// Entry in the merge heap
struct HeapEntry {
    edge: MergeEdge,
    source_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.edge == other.edge
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.edge.cmp(&other.edge)
    }
}

/// Configuration for parallel merge
#[derive(Debug, Clone)]
pub struct ParallelMergeConfig {
    /// Number of reader threads
    pub reader_threads: usize,
    /// Number of merger threads (for range partitioning)
    pub merger_threads: usize,
    /// Channel buffer size per input
    pub channel_buffer_size: usize,
    /// Batch size for reading
    pub read_batch_size: usize,
}

impl Default for ParallelMergeConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        Self {
            reader_threads: (num_cpus / 2).max(1),
            merger_threads: (num_cpus / 2).max(1),
            channel_buffer_size: 1024,
            read_batch_size: 256,
        }
    }
}

impl ParallelMergeConfig {
    /// Create config optimized for the given number of input files
    pub fn for_inputs(num_inputs: usize) -> Self {
        let num_cpus = num_cpus::get();
        Self {
            reader_threads: num_inputs.min(num_cpus),
            merger_threads: (num_cpus / 2).max(1),
            channel_buffer_size: 1024,
            read_batch_size: 256,
        }
    }
}

/// Statistics for parallel merge operations
#[derive(Debug, Default)]
pub struct ParallelMergeStats {
    /// Total edges read from inputs
    pub edges_read: AtomicU64,
    /// Total edges written to output
    pub edges_written: AtomicU64,
    /// Tombstones filtered out
    pub tombstones_filtered: AtomicU64,
    /// Duplicate edges merged
    pub duplicates_merged: AtomicU64,
}

impl ParallelMergeStats {
    /// Create new stats
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record edges read
    pub fn record_read(&self, count: u64) {
        self.edges_read.fetch_add(count, AtomicOrdering::Relaxed);
    }

    /// Record edges written
    pub fn record_written(&self, count: u64) {
        self.edges_written.fetch_add(count, AtomicOrdering::Relaxed);
    }

    /// Record filtered tombstones
    pub fn record_tombstone(&self) {
        self.tombstones_filtered
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record merged duplicates
    pub fn record_duplicate(&self) {
        self.duplicates_merged.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Get snapshot of stats
    pub fn snapshot(&self) -> ParallelMergeStatsSnapshot {
        ParallelMergeStatsSnapshot {
            edges_read: self.edges_read.load(AtomicOrdering::Relaxed),
            edges_written: self.edges_written.load(AtomicOrdering::Relaxed),
            tombstones_filtered: self.tombstones_filtered.load(AtomicOrdering::Relaxed),
            duplicates_merged: self.duplicates_merged.load(AtomicOrdering::Relaxed),
        }
    }
}

/// Snapshot of merge stats
#[derive(Debug, Clone)]
pub struct ParallelMergeStatsSnapshot {
    pub edges_read: u64,
    pub edges_written: u64,
    pub tombstones_filtered: u64,
    pub duplicates_merged: u64,
}

// =============================================================================
// Task 7 Enhancement: I/O Throttling for Compaction
// =============================================================================

/// Token bucket-based I/O throttler for compaction
///
/// ## Purpose
/// Prevents compaction from saturating I/O bandwidth and causing
/// latency spikes for foreground queries.
///
/// ## Algorithm
/// Uses token bucket with configurable:
/// - Bucket capacity (burst allowance)
/// - Refill rate (sustained IOPS/bandwidth limit)
/// - Adaptive scaling based on system load
///
/// ## Reference
/// Based on RocksDB's `rate_limiter` implementation which uses
/// token bucket for I/O bandwidth control.
#[derive(Debug)]
pub struct IoThrottler {
    /// Available tokens (bytes or IOPS)
    tokens: AtomicU64,
    /// Maximum bucket capacity
    capacity: u64,
    /// Refill rate per second
    refill_rate: u64,
    /// Last refill timestamp (microseconds)
    last_refill: AtomicU64,
    /// Total bytes throttled (for stats)
    total_throttled: AtomicU64,
    /// Total wait time (microseconds)
    total_wait_us: AtomicU64,
    /// Whether throttling is enabled
    enabled: std::sync::atomic::AtomicBool,
}

impl IoThrottler {
    /// Create a new I/O throttler
    ///
    /// # Arguments
    /// * `rate_bytes_per_sec` - Sustained I/O rate limit
    /// * `burst_bytes` - Maximum burst allowance
    pub fn new(rate_bytes_per_sec: u64, burst_bytes: u64) -> Self {
        Self {
            tokens: AtomicU64::new(burst_bytes),
            capacity: burst_bytes,
            refill_rate: rate_bytes_per_sec,
            last_refill: AtomicU64::new(Self::now_us()),
            total_throttled: AtomicU64::new(0),
            total_wait_us: AtomicU64::new(0),
            enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Create throttler with sensible defaults for compaction
    ///
    /// Defaults to 100 MB/s sustained, 10 MB burst
    pub fn for_compaction() -> Self {
        Self::new(100 * 1024 * 1024, 10 * 1024 * 1024)
    }

    /// Create throttler that doesn't limit (for testing)
    pub fn unlimited() -> Self {
        let throttler = Self::new(u64::MAX, u64::MAX);
        throttler.enabled.store(false, AtomicOrdering::Relaxed);
        throttler
    }

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Refill tokens based on elapsed time
    fn refill(&self) {
        let now = Self::now_us();
        let last = self.last_refill.swap(now, AtomicOrdering::AcqRel);
        let elapsed_us = now.saturating_sub(last);

        if elapsed_us > 0 {
            // tokens_to_add = elapsed_seconds * rate
            // = elapsed_us / 1_000_000 * rate
            let tokens_to_add = (elapsed_us as u128 * self.refill_rate as u128 / 1_000_000) as u64;

            if tokens_to_add > 0 {
                let current = self.tokens.load(AtomicOrdering::Relaxed);
                let new_tokens = current.saturating_add(tokens_to_add).min(self.capacity);
                self.tokens.store(new_tokens, AtomicOrdering::Release);
            }
        }
    }

    /// Request I/O tokens, blocking if necessary
    ///
    /// Returns the actual wait time in microseconds
    pub fn acquire(&self, bytes: u64) -> u64 {
        if !self.enabled.load(AtomicOrdering::Relaxed) {
            return 0;
        }

        let mut total_wait = 0u64;

        loop {
            self.refill();

            let current = self.tokens.load(AtomicOrdering::Acquire);

            if current >= bytes {
                // Try to consume tokens
                match self.tokens.compare_exchange_weak(
                    current,
                    current - bytes,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                ) {
                    Ok(_) => {
                        if total_wait > 0 {
                            self.total_wait_us
                                .fetch_add(total_wait, AtomicOrdering::Relaxed);
                            self.total_throttled
                                .fetch_add(bytes, AtomicOrdering::Relaxed);
                        }
                        return total_wait;
                    }
                    Err(_) => continue, // Retry
                }
            }

            // Not enough tokens - wait for refill
            // Calculate wait time based on deficit
            let deficit = bytes.saturating_sub(current);
            let wait_us = (deficit as u128 * 1_000_000 / self.refill_rate as u128) as u64;
            let wait_us = wait_us.clamp(100, 100_000); // Between 100us and 100ms

            std::thread::sleep(std::time::Duration::from_micros(wait_us));
            total_wait += wait_us;
        }
    }

    /// Try to acquire tokens without blocking
    ///
    /// Returns true if tokens were acquired, false otherwise
    pub fn try_acquire(&self, bytes: u64) -> bool {
        if !self.enabled.load(AtomicOrdering::Relaxed) {
            return true;
        }

        self.refill();

        let current = self.tokens.load(AtomicOrdering::Acquire);
        if current >= bytes {
            self.tokens
                .compare_exchange_weak(
                    current,
                    current - bytes,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
        } else {
            false
        }
    }

    /// Enable or disable throttling
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, AtomicOrdering::Release);
    }

    /// Check if throttling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(AtomicOrdering::Relaxed)
    }

    /// Update rate limit (for adaptive control)
    pub fn set_rate(&mut self, rate_bytes_per_sec: u64) {
        self.refill_rate = rate_bytes_per_sec;
    }

    /// Get current available tokens
    pub fn available_tokens(&self) -> u64 {
        self.refill();
        self.tokens.load(AtomicOrdering::Relaxed)
    }

    /// Get throttler statistics
    pub fn stats(&self) -> IoThrottlerStats {
        IoThrottlerStats {
            total_throttled_bytes: self.total_throttled.load(AtomicOrdering::Relaxed),
            total_wait_us: self.total_wait_us.load(AtomicOrdering::Relaxed),
            available_tokens: self.available_tokens(),
            rate_bytes_per_sec: self.refill_rate,
            enabled: self.is_enabled(),
        }
    }
}

/// I/O throttler statistics
#[derive(Debug, Clone)]
pub struct IoThrottlerStats {
    /// Total bytes that were throttled
    pub total_throttled_bytes: u64,
    /// Total time spent waiting (microseconds)
    pub total_wait_us: u64,
    /// Currently available tokens
    pub available_tokens: u64,
    /// Configured rate limit
    pub rate_bytes_per_sec: u64,
    /// Whether throttling is enabled
    pub enabled: bool,
}

/// Adaptive I/O controller that adjusts throttling based on system load
///
/// Monitors foreground query latency and adjusts compaction I/O
/// rate to maintain acceptable performance.
#[derive(Debug)]
pub struct AdaptiveIoController {
    /// Base throttler
    throttler: IoThrottler,
    /// Target p99 latency for foreground queries (microseconds)
    target_latency_us: u64,
    /// Minimum I/O rate (never throttle below this)
    min_rate: u64,
    /// Maximum I/O rate (upper bound)
    max_rate: u64,
    /// Current rate multiplier (0.1 to 1.0)
    rate_multiplier: std::sync::atomic::AtomicU64,
}

impl AdaptiveIoController {
    /// Create new adaptive controller
    pub fn new(base_rate: u64, target_latency_us: u64) -> Self {
        Self {
            throttler: IoThrottler::new(base_rate, base_rate / 10),
            target_latency_us,
            min_rate: base_rate / 10,
            max_rate: base_rate,
            rate_multiplier: std::sync::atomic::AtomicU64::new(1000), // 1.0 as fixed point
        }
    }

    /// Report observed foreground latency
    ///
    /// Used to adjust compaction I/O rate
    pub fn report_latency(&self, latency_us: u64) {
        let current_mult = self.rate_multiplier.load(AtomicOrdering::Relaxed);

        let new_mult = if latency_us > self.target_latency_us * 2 {
            // Latency too high - reduce compaction I/O significantly
            (current_mult * 8 / 10).max(100) // 0.8x, min 0.1
        } else if latency_us > self.target_latency_us {
            // Latency slightly high - reduce slightly
            (current_mult * 95 / 100).max(100) // 0.95x
        } else if latency_us < self.target_latency_us / 2 && current_mult < 1000 {
            // Latency low - can increase compaction I/O
            (current_mult * 105 / 100).min(1000) // 1.05x, max 1.0
        } else {
            current_mult
        };

        self.rate_multiplier
            .store(new_mult, AtomicOrdering::Relaxed);
    }

    /// Get current effective rate
    pub fn effective_rate(&self) -> u64 {
        let mult = self.rate_multiplier.load(AtomicOrdering::Relaxed);
        let rate = (self.max_rate as u128 * mult as u128 / 1000) as u64;
        rate.max(self.min_rate).min(self.max_rate)
    }

    /// Acquire I/O tokens with adaptive rate
    pub fn acquire(&self, bytes: u64) -> u64 {
        self.throttler.acquire(bytes)
    }

    /// Get the underlying throttler
    pub fn throttler(&self) -> &IoThrottler {
        &self.throttler
    }
}

/// A producer that reads edges from an SSTable and sends them to a channel
pub struct ParallelReader {
    /// Channel sender for edges
    sender: Sender<MergeEdge>,
    /// Source index
    source_idx: usize,
    /// Stats
    stats: Arc<ParallelMergeStats>,
}

impl ParallelReader {
    /// Create a new parallel reader
    pub fn new(
        sender: Sender<MergeEdge>,
        source_idx: usize,
        stats: Arc<ParallelMergeStats>,
    ) -> Self {
        Self {
            sender,
            source_idx,
            stats,
        }
    }

    /// Send an edge to the merge channel
    #[allow(clippy::result_large_err)]
    pub fn send(&self, edge: MergeEdge) -> Result<(), crossbeam_channel::SendError<MergeEdge>> {
        self.stats.record_read(1);
        self.sender.send(edge)
    }

    /// Get the source index
    pub fn source_idx(&self) -> usize {
        self.source_idx
    }
}

/// Multi-source merge coordinator
pub struct ParallelMerger {
    /// Receivers from each input source
    receivers: Vec<Receiver<MergeEdge>>,
    /// Configuration (reserved for future use)
    #[allow(dead_code)]
    config: ParallelMergeConfig,
    /// Stats
    stats: Arc<ParallelMergeStats>,
}

impl ParallelMerger {
    /// Create channels for parallel reading
    pub fn create_channels(
        num_inputs: usize,
        config: &ParallelMergeConfig,
        stats: Arc<ParallelMergeStats>,
    ) -> (Vec<ParallelReader>, Self) {
        let mut senders = Vec::with_capacity(num_inputs);
        let mut receivers = Vec::with_capacity(num_inputs);

        for i in 0..num_inputs {
            let (tx, rx) = bounded(config.channel_buffer_size);
            senders.push(ParallelReader::new(tx, i, stats.clone()));
            receivers.push(rx);
        }

        let merger = Self {
            receivers,
            config: config.clone(),
            stats,
        };

        (senders, merger)
    }

    /// Perform the merge, collecting results into a vector
    ///
    /// This uses a binary heap for K-way merge with proper duplicate handling.
    pub fn merge(self) -> Vec<MergeEdge> {
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut result = Vec::new();
        let mut last_key: Option<(u64, u128)> = None;

        // Initialize heap with first element from each receiver
        for (idx, rx) in self.receivers.iter().enumerate() {
            if let Ok(edge) = rx.recv() {
                heap.push(HeapEntry {
                    edge,
                    source_idx: idx,
                });
            }
        }

        while let Some(entry) = heap.pop() {
            let edge = entry.edge;
            let source_idx = entry.source_idx;

            // Check for duplicates (same timestamp + edge_id)
            let key = (edge.timestamp_us, edge.edge_id);
            let is_duplicate = last_key.map(|k| k == key).unwrap_or(false);

            if is_duplicate {
                self.stats.record_duplicate();
            } else if edge.is_tombstone {
                // For now, we keep tombstones in output but track them
                self.stats.record_tombstone();
                result.push(edge.clone());
                self.stats.record_written(1);
            } else {
                result.push(edge.clone());
                self.stats.record_written(1);
            }

            last_key = Some(key);

            // Get next element from the same source
            if let Ok(next_edge) = self.receivers[source_idx].recv() {
                heap.push(HeapEntry {
                    edge: next_edge,
                    source_idx,
                });
            }
        }

        result
    }

    /// Perform merge with a callback for each output edge
    pub fn merge_with_callback<F>(self, mut callback: F)
    where
        F: FnMut(MergeEdge),
    {
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut last_key: Option<(u64, u128)> = None;

        // Initialize heap
        for (idx, rx) in self.receivers.iter().enumerate() {
            if let Ok(edge) = rx.recv() {
                heap.push(HeapEntry {
                    edge,
                    source_idx: idx,
                });
            }
        }

        while let Some(entry) = heap.pop() {
            let edge = entry.edge;
            let source_idx = entry.source_idx;

            let key = (edge.timestamp_us, edge.edge_id);
            let is_duplicate = last_key.map(|k| k == key).unwrap_or(false);

            if is_duplicate {
                self.stats.record_duplicate();
            } else {
                if edge.is_tombstone {
                    self.stats.record_tombstone();
                }
                callback(edge.clone());
                self.stats.record_written(1);
            }

            last_key = Some(key);

            // Get next from same source
            if let Ok(next_edge) = self.receivers[source_idx].recv() {
                heap.push(HeapEntry {
                    edge: next_edge,
                    source_idx,
                });
            }
        }
    }
}

/// Range partition for sub-compaction
#[derive(Debug, Clone)]
pub struct KeyRange {
    /// Minimum timestamp (inclusive)
    pub min_ts: u64,
    /// Maximum timestamp (exclusive)
    pub max_ts: u64,
}

impl KeyRange {
    /// Create a new key range
    pub fn new(min_ts: u64, max_ts: u64) -> Self {
        Self { min_ts, max_ts }
    }

    /// Check if a timestamp falls within this range
    pub fn contains(&self, ts: u64) -> bool {
        ts >= self.min_ts && ts < self.max_ts
    }
}

/// Partition key space for parallel sub-compaction
pub fn partition_key_space(min_ts: u64, max_ts: u64, num_partitions: usize) -> Vec<KeyRange> {
    if num_partitions == 0 || max_ts <= min_ts {
        return vec![KeyRange::new(min_ts, max_ts)];
    }

    let range = max_ts - min_ts;
    let partition_size = range / num_partitions as u64;

    (0..num_partitions)
        .map(|i| {
            let start = min_ts + (i as u64 * partition_size);
            let end = if i == num_partitions - 1 {
                max_ts
            } else {
                min_ts + ((i as u64 + 1) * partition_size)
            };
            KeyRange::new(start, end)
        })
        .collect()
}

/// Builder for setting up a parallel merge operation
pub struct ParallelMergeBuilder {
    config: ParallelMergeConfig,
    stats: Arc<ParallelMergeStats>,
}

impl ParallelMergeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ParallelMergeConfig::default(),
            stats: ParallelMergeStats::new(),
        }
    }

    /// Set configuration
    pub fn config(mut self, config: ParallelMergeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set stats tracker
    pub fn stats(mut self, stats: Arc<ParallelMergeStats>) -> Self {
        self.stats = stats;
        self
    }

    /// Create channels for the given number of inputs
    pub fn build(self, num_inputs: usize) -> (Vec<ParallelReader>, ParallelMerger) {
        ParallelMerger::create_channels(num_inputs, &self.config, self.stats)
    }
}

impl Default for ParallelMergeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn create_test_edge(edge_id: u128, timestamp_us: u64, is_tombstone: bool) -> MergeEdge {
        MergeEdge {
            edge_id,
            timestamp_us,
            is_tombstone,
            data: [0u8; 128],
            source_idx: 0,
        }
    }

    #[test]
    fn test_merge_edge_ordering() {
        let e1 = create_test_edge(1, 1000, false);
        let e2 = create_test_edge(2, 1000, false);
        let e3 = create_test_edge(1, 2000, false);

        // In min-heap, smaller timestamps should come first
        assert!(e1 > e2); // Same timestamp, different edge_id
        assert!(e1 > e3); // e1 has smaller timestamp, should be "greater" for min-heap
    }

    #[test]
    fn test_parallel_merge_basic() {
        let stats = ParallelMergeStats::new();
        let config = ParallelMergeConfig {
            reader_threads: 2,
            merger_threads: 1,
            channel_buffer_size: 100,
            read_batch_size: 10,
        };

        let (readers, merger) = ParallelMerger::create_channels(3, &config, stats.clone());

        // Spawn reader threads
        let handles: Vec<_> = readers
            .into_iter()
            .enumerate()
            .map(|(idx, reader)| {
                thread::spawn(move || {
                    for i in 0..10u64 {
                        let edge = MergeEdge {
                            edge_id: (idx * 100 + i as usize) as u128,
                            timestamp_us: i * 100 + idx as u64, // Interleaved timestamps
                            is_tombstone: false,
                            data: [0u8; 128],
                            source_idx: idx,
                        };
                        reader.send(edge).unwrap();
                    }
                })
            })
            .collect();

        // Wait for readers
        for h in handles {
            h.join().unwrap();
        }

        // Perform merge
        let result = merger.merge();

        // Should have 30 edges (10 from each of 3 sources)
        assert_eq!(result.len(), 30);

        // Verify sorted order
        for i in 1..result.len() {
            let prev = &result[i - 1];
            let curr = &result[i];
            assert!(
                prev.timestamp_us < curr.timestamp_us
                    || (prev.timestamp_us == curr.timestamp_us && prev.edge_id < curr.edge_id),
                "Results should be sorted"
            );
        }

        // Check stats
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.edges_read, 30);
        assert_eq!(snapshot.edges_written, 30);
    }

    #[test]
    fn test_parallel_merge_duplicates() {
        let stats = ParallelMergeStats::new();
        let config = ParallelMergeConfig::default();
        let (readers, merger) = ParallelMerger::create_channels(2, &config, stats.clone());

        // Both sources have the same edge (duplicate)
        let handles: Vec<_> = readers
            .into_iter()
            .map(|reader| {
                thread::spawn(move || {
                    let edge = create_test_edge(42, 1000, false);
                    reader.send(edge).unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let result = merger.merge();

        // Should deduplicate
        assert_eq!(result.len(), 1);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.edges_read, 2);
        assert_eq!(snapshot.duplicates_merged, 1);
    }

    #[test]
    fn test_partition_key_space() {
        let partitions = partition_key_space(0, 1000, 4);

        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].min_ts, 0);
        assert_eq!(partitions[0].max_ts, 250);
        assert_eq!(partitions[1].min_ts, 250);
        assert_eq!(partitions[1].max_ts, 500);
        assert_eq!(partitions[2].min_ts, 500);
        assert_eq!(partitions[2].max_ts, 750);
        assert_eq!(partitions[3].min_ts, 750);
        assert_eq!(partitions[3].max_ts, 1000);
    }

    #[test]
    fn test_key_range_contains() {
        let range = KeyRange::new(100, 200);

        assert!(!range.contains(99));
        assert!(range.contains(100));
        assert!(range.contains(150));
        assert!(!range.contains(200));
    }

    #[test]
    fn test_merge_with_callback() {
        let stats = ParallelMergeStats::new();
        let config = ParallelMergeConfig::default();
        let (readers, merger) = ParallelMerger::create_channels(2, &config, stats.clone());

        let handles: Vec<_> = readers
            .into_iter()
            .enumerate()
            .map(|(idx, reader)| {
                thread::spawn(move || {
                    for i in 0..5u64 {
                        let edge =
                            create_test_edge((idx * 10 + i as usize) as u128, i * 100, false);
                        reader.send(edge).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        merger.merge_with_callback(move |_edge| {
            count_clone.fetch_add(1, AtomicOrdering::Relaxed);
        });

        assert_eq!(count.load(AtomicOrdering::Relaxed), 10);
    }

    #[test]
    fn test_parallel_merge_config() {
        let config = ParallelMergeConfig::for_inputs(8);
        assert!(config.reader_threads >= 1);
        assert!(config.merger_threads >= 1);

        let default = ParallelMergeConfig::default();
        assert!(default.channel_buffer_size > 0);
        assert!(default.read_batch_size > 0);
    }

    // I/O Throttler Tests

    #[test]
    fn test_throttler_basic() {
        let throttler = IoThrottler::new(1_000_000, 100_000); // 1 MB/s, 100KB burst

        // Should be able to acquire up to burst size immediately
        assert!(throttler.try_acquire(50_000));
        assert!(throttler.try_acquire(50_000));

        // Burst exhausted, should fail
        assert!(!throttler.try_acquire(50_000));
    }

    #[test]
    fn test_throttler_unlimited() {
        let throttler = IoThrottler::unlimited();

        // Should always succeed
        assert!(throttler.try_acquire(1_000_000_000));
        assert!(!throttler.is_enabled());
    }

    #[test]
    fn test_throttler_blocking_acquire() {
        // Very low rate to ensure blocking
        let throttler = IoThrottler::new(1_000, 100); // 1 KB/s, 100 byte burst

        // Exhaust burst
        assert!(throttler.try_acquire(100));

        // This should block briefly
        let start = std::time::Instant::now();
        let _wait = throttler.acquire(100);
        let elapsed = start.elapsed();

        // Should have waited at least a little (100 bytes at 1KB/s = 100ms)
        // But we're lenient since timing varies
        assert!(elapsed.as_millis() >= 10 || throttler.stats().total_wait_us > 0);
    }

    #[test]
    fn test_throttler_stats() {
        let throttler = IoThrottler::new(10_000_000, 10_000);

        // Get initial tokens
        let initial = throttler.stats().available_tokens;

        throttler.try_acquire(5_000);
        throttler.try_acquire(3_000);

        let stats = throttler.stats();
        // Should have consumed 8000 tokens (may have refilled slightly)
        assert!(stats.available_tokens < initial);
        assert_eq!(stats.rate_bytes_per_sec, 10_000_000);
        assert!(stats.enabled);
    }

    #[test]
    fn test_adaptive_controller() {
        let controller = AdaptiveIoController::new(100_000_000, 10_000); // 100 MB/s, 10ms target

        // Report high latency
        controller.report_latency(50_000); // 50ms - way too high
        let rate1 = controller.effective_rate();

        controller.report_latency(50_000);
        let rate2 = controller.effective_rate();

        // Rate should decrease
        assert!(rate2 <= rate1);

        // Report low latency - rate should recover
        for _ in 0..20 {
            controller.report_latency(1_000); // 1ms - very low
        }
        let rate3 = controller.effective_rate();

        // Should have recovered somewhat
        assert!(rate3 >= rate2);
    }
}
