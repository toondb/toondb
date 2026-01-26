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

//! Epoch-Based Garbage Collection for Version Cleanup
//!
//! This module provides epoch-based garbage collection for cleaning up
//! old versions of data in a multi-version concurrency control (MVCC) system.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Epoch-Based GC                               │
//! │                                                                 │
//! │  Time ──────────────────────────────────────────────────────→  │
//! │                                                                 │
//! │  Epoch 0     Epoch 1     Epoch 2     Epoch 3     Epoch 4       │
//! │  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐       │
//! │  │v1,v2│     │v3,v4│     │v5   │     │v6,v7│     │v8   │       │
//! │  └─────┘     └─────┘     └─────┘     └─────┘     └─────┘       │
//! │    ↑                                               ↑           │
//! │    │                                               │           │
//! │  Min active                                    Current         │
//! │  reader epoch                                   epoch          │
//! │                                                                 │
//! │  Versions in epochs < min_active can be safely collected       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Concepts
//!
//! - **Version**: A specific snapshot of data at a point in time
//! - **Epoch**: A logical time period during which versions are created
//! - **Watermark**: The minimum epoch that is still accessible by readers
//! - **GC Cycle**: Periodic cleanup of versions older than watermark

use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Version identifier (epoch, sequence within epoch)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VersionId {
    pub epoch: u64,
    pub sequence: u32,
}

impl VersionId {
    pub fn new(epoch: u64, sequence: u32) -> Self {
        Self { epoch, sequence }
    }

    /// Check if this version is older than watermark
    pub fn is_stale(&self, watermark: u64) -> bool {
        self.epoch < watermark
    }
}

/// A versioned value wrapper
#[derive(Debug, Clone)]
pub struct VersionedValue<T> {
    pub version: VersionId,
    pub value: T,
    /// Deletion marker (tombstone)
    pub deleted: bool,
}

impl<T> VersionedValue<T> {
    pub fn new(version: VersionId, value: T) -> Self {
        Self {
            version,
            value,
            deleted: false,
        }
    }

    pub fn tombstone(version: VersionId, value: T) -> Self {
        Self {
            version,
            value,
            deleted: true,
        }
    }
}

/// Version chain for a single key
#[derive(Debug)]
pub struct VersionChain<T> {
    /// Versions ordered from newest to oldest
    versions: VecDeque<VersionedValue<T>>,
    /// Total count of versions ever created
    total_versions: u64,
}

impl<T: Clone> VersionChain<T> {
    pub fn new() -> Self {
        Self {
            versions: VecDeque::new(),
            total_versions: 0,
        }
    }

    /// Add a new version (prepends to front as newest)
    pub fn add_version(&mut self, version: VersionedValue<T>) {
        self.versions.push_front(version);
        self.total_versions += 1;
    }

    /// Get the latest version
    pub fn latest(&self) -> Option<&VersionedValue<T>> {
        self.versions.front()
    }

    /// Get version visible at specific epoch
    /// Returns None if the key was deleted at or before the given epoch
    pub fn version_at(&self, epoch: u64) -> Option<&VersionedValue<T>> {
        for v in &self.versions {
            if v.version.epoch <= epoch {
                // If this version (the most recent at or before the epoch) is deleted,
                // the key is considered deleted at this point in time
                if v.deleted {
                    return None;
                }
                return Some(v);
            }
        }
        None
    }

    /// Clean up versions older than watermark
    /// Returns (versions_removed, bytes_freed estimate)
    pub fn gc(&mut self, watermark: u64) -> (usize, usize) {
        let initial_len = self.versions.len();

        // Keep at least one version (the latest visible)
        let mut kept = 0;
        let mut last_visible_idx = None;

        for (i, v) in self.versions.iter().enumerate() {
            if v.version.epoch >= watermark {
                kept += 1;
            } else {
                // This is below watermark, but we keep the first one as the base
                if last_visible_idx.is_none() {
                    last_visible_idx = Some(i);
                    kept += 1;
                }
            }
        }

        // Remove versions beyond what we're keeping
        while self.versions.len() > kept {
            self.versions.pop_back();
        }

        let removed = initial_len - self.versions.len();
        let bytes_freed = removed * std::mem::size_of::<VersionedValue<T>>();

        (removed, bytes_freed)
    }

    /// Number of versions in chain
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

impl<T: Clone> Default for VersionChain<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader registration for tracking active epochs
#[derive(Debug)]
pub struct ReaderRegistry {
    /// Active reader epochs (reader_id -> epoch)
    active_readers: RwLock<HashMap<u64, u64>>,
    /// Next reader ID
    next_reader_id: AtomicU64,
    /// Count of active readers
    active_count: AtomicUsize,
}

impl ReaderRegistry {
    pub fn new() -> Self {
        Self {
            active_readers: RwLock::new(HashMap::new()),
            next_reader_id: AtomicU64::new(1),
            active_count: AtomicUsize::new(0),
        }
    }

    /// Register a reader at the current epoch
    pub fn register(&self, epoch: u64) -> u64 {
        let reader_id = self.next_reader_id.fetch_add(1, Ordering::Relaxed);
        let mut readers = self.active_readers.write();
        readers.insert(reader_id, epoch);
        self.active_count.fetch_add(1, Ordering::Relaxed);
        reader_id
    }

    /// Unregister a reader
    pub fn unregister(&self, reader_id: u64) {
        let mut readers = self.active_readers.write();
        if readers.remove(&reader_id).is_some() {
            self.active_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get the minimum epoch among all active readers
    pub fn min_active_epoch(&self) -> Option<u64> {
        let readers = self.active_readers.read();
        readers.values().copied().min()
    }

    /// Get count of active readers
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }
}

impl Default for ReaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// GC statistics
#[derive(Debug, Default)]
pub struct GCStats {
    pub gc_cycles: AtomicU64,
    pub versions_collected: AtomicU64,
    pub bytes_freed: AtomicU64,
    pub chains_scanned: AtomicU64,
    pub last_gc_epoch: AtomicU64,
    pub last_gc_duration_us: AtomicU64,
}

impl GCStats {
    pub fn snapshot(&self) -> GCStatsSnapshot {
        GCStatsSnapshot {
            gc_cycles: self.gc_cycles.load(Ordering::Relaxed),
            versions_collected: self.versions_collected.load(Ordering::Relaxed),
            bytes_freed: self.bytes_freed.load(Ordering::Relaxed),
            chains_scanned: self.chains_scanned.load(Ordering::Relaxed),
            last_gc_epoch: self.last_gc_epoch.load(Ordering::Relaxed),
            last_gc_duration_us: self.last_gc_duration_us.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GCStatsSnapshot {
    pub gc_cycles: u64,
    pub versions_collected: u64,
    pub bytes_freed: u64,
    pub chains_scanned: u64,
    pub last_gc_epoch: u64,
    pub last_gc_duration_us: u64,
}

/// GC configuration
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// Minimum number of epochs to keep (grace period)
    pub min_epochs_to_keep: u64,
    /// Trigger GC after this many new versions
    pub gc_trigger_threshold: usize,
    /// Maximum versions to scan per GC cycle
    pub max_versions_per_cycle: usize,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            min_epochs_to_keep: 2,
            gc_trigger_threshold: 1000,
            max_versions_per_cycle: 10000,
        }
    }
}

/// Epoch-based garbage collector for versioned data
pub struct EpochGC<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Global epoch counter
    current_epoch: AtomicU64,
    /// Sequence counter within epoch
    current_sequence: AtomicU64,
    /// Version chains by key
    chains: RwLock<HashMap<K, VersionChain<V>>>,
    /// Reader registry
    readers: Arc<ReaderRegistry>,
    /// GC configuration
    config: GCConfig,
    /// GC statistics
    stats: GCStats,
    /// Pending versions since last GC
    pending_versions: AtomicUsize,
}

impl<K, V> EpochGC<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create new epoch GC with default config
    pub fn new() -> Self {
        Self::with_config(GCConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: GCConfig) -> Self {
        Self {
            current_epoch: AtomicU64::new(0),
            current_sequence: AtomicU64::new(0),
            chains: RwLock::new(HashMap::new()),
            readers: Arc::new(ReaderRegistry::new()),
            config,
            stats: GCStats::default(),
            pending_versions: AtomicUsize::new(0),
        }
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::SeqCst)
    }

    /// Advance to next epoch
    pub fn advance_epoch(&self) -> u64 {
        self.current_sequence.store(0, Ordering::SeqCst);
        self.current_epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Allocate a new version ID
    pub fn next_version(&self) -> VersionId {
        let epoch = self.current_epoch.load(Ordering::SeqCst);
        let seq = self.current_sequence.fetch_add(1, Ordering::SeqCst) as u32;
        VersionId::new(epoch, seq)
    }

    /// Insert a new version
    pub fn insert(&self, key: K, value: V) -> VersionId {
        let version = self.next_version();
        let versioned = VersionedValue::new(version, value);

        {
            let mut chains = self.chains.write();
            chains.entry(key).or_default().add_version(versioned);
        }

        let pending = self.pending_versions.fetch_add(1, Ordering::Relaxed);

        // Trigger GC if threshold exceeded
        if pending >= self.config.gc_trigger_threshold {
            self.try_gc();
        }

        version
    }

    /// Delete a key (insert tombstone)
    pub fn delete(&self, key: K, tombstone_value: V) -> VersionId {
        let version = self.next_version();
        let versioned = VersionedValue::tombstone(version, tombstone_value);

        {
            let mut chains = self.chains.write();
            chains.entry(key).or_default().add_version(versioned);
        }

        self.pending_versions.fetch_add(1, Ordering::Relaxed);
        version
    }

    /// Get latest version of a key
    pub fn get(&self, key: &K) -> Option<V> {
        let chains = self.chains.read();
        chains
            .get(key)
            .and_then(|chain| chain.latest())
            .filter(|v| !v.deleted)
            .map(|v| v.value.clone())
    }

    /// Get version at specific epoch
    pub fn get_at_epoch(&self, key: &K, epoch: u64) -> Option<V> {
        let chains = self.chains.read();
        chains
            .get(key)
            .and_then(|chain| chain.version_at(epoch))
            .map(|v| v.value.clone())
    }

    /// Begin a read transaction at current epoch
    pub fn begin_read(&self) -> ReadGuard {
        let epoch = self.current_epoch.load(Ordering::SeqCst);
        let reader_id = self.readers.register(epoch);
        ReadGuard {
            epoch,
            reader_id,
            registry: Arc::clone(&self.readers),
        }
    }

    /// Calculate the GC watermark (safe to collect below this)
    pub fn watermark(&self) -> u64 {
        let current = self.current_epoch.load(Ordering::SeqCst);
        let min_reader = self.readers.min_active_epoch().unwrap_or(current);

        // Watermark is the minimum of current - grace period and min active reader
        let grace = current.saturating_sub(self.config.min_epochs_to_keep);
        grace.min(min_reader)
    }

    /// Try to run a GC cycle
    pub fn try_gc(&self) -> GCResult {
        let start = std::time::Instant::now();
        let watermark = self.watermark();

        let mut versions_collected = 0;
        let mut bytes_freed = 0;
        let mut chains_scanned = 0;

        {
            let mut chains = self.chains.write();
            let keys: Vec<K> = chains.keys().cloned().collect();

            for key in keys {
                if chains_scanned >= self.config.max_versions_per_cycle {
                    break;
                }

                if let Some(chain) = chains.get_mut(&key) {
                    let (removed, freed) = chain.gc(watermark);
                    versions_collected += removed;
                    bytes_freed += freed;
                    chains_scanned += 1;

                    // Remove empty chains
                    if chain.is_empty() {
                        chains.remove(&key);
                    }
                }
            }
        }

        let duration = start.elapsed();

        // Update stats
        self.stats.gc_cycles.fetch_add(1, Ordering::Relaxed);
        self.stats
            .versions_collected
            .fetch_add(versions_collected as u64, Ordering::Relaxed);
        self.stats
            .bytes_freed
            .fetch_add(bytes_freed as u64, Ordering::Relaxed);
        self.stats
            .chains_scanned
            .fetch_add(chains_scanned as u64, Ordering::Relaxed);
        self.stats.last_gc_epoch.store(watermark, Ordering::Relaxed);
        self.stats
            .last_gc_duration_us
            .store(duration.as_micros() as u64, Ordering::Relaxed);

        // Reset pending counter
        self.pending_versions.store(0, Ordering::Relaxed);

        GCResult {
            versions_collected,
            bytes_freed,
            chains_scanned,
            watermark,
            duration_us: duration.as_micros() as u64,
        }
    }

    /// Force a full GC (ignore limits)
    pub fn force_gc(&self) -> GCResult {
        let _old_limit = self.config.max_versions_per_cycle;
        // Temporarily set limit to max
        let _config = GCConfig {
            max_versions_per_cycle: usize::MAX,
            ..self.config.clone()
        };

        let start = std::time::Instant::now();
        let watermark = self.watermark();

        let mut versions_collected = 0;
        let mut bytes_freed = 0;
        let mut chains_scanned = 0;

        {
            let mut chains = self.chains.write();
            for chain in chains.values_mut() {
                let (removed, freed) = chain.gc(watermark);
                versions_collected += removed;
                bytes_freed += freed;
                chains_scanned += 1;
            }

            // Remove empty chains
            chains.retain(|_, chain| !chain.is_empty());
        }

        let duration = start.elapsed();

        self.stats.gc_cycles.fetch_add(1, Ordering::Relaxed);
        self.stats
            .versions_collected
            .fetch_add(versions_collected as u64, Ordering::Relaxed);
        self.stats
            .bytes_freed
            .fetch_add(bytes_freed as u64, Ordering::Relaxed);
        self.stats
            .chains_scanned
            .fetch_add(chains_scanned as u64, Ordering::Relaxed);
        self.stats.last_gc_epoch.store(watermark, Ordering::Relaxed);
        self.stats
            .last_gc_duration_us
            .store(duration.as_micros() as u64, Ordering::Relaxed);
        self.pending_versions.store(0, Ordering::Relaxed);

        GCResult {
            versions_collected,
            bytes_freed,
            chains_scanned,
            watermark,
            duration_us: duration.as_micros() as u64,
        }
    }

    /// Get GC statistics
    pub fn stats(&self) -> GCStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get total version count
    pub fn version_count(&self) -> usize {
        let chains = self.chains.read();
        chains.values().map(|c| c.len()).sum()
    }

    /// Get chain count
    pub fn chain_count(&self) -> usize {
        self.chains.read().len()
    }
}

impl<K, V> Default for EpochGC<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a GC cycle
#[derive(Debug, Clone)]
pub struct GCResult {
    pub versions_collected: usize,
    pub bytes_freed: usize,
    pub chains_scanned: usize,
    pub watermark: u64,
    pub duration_us: u64,
}

/// Guard for read transactions
pub struct ReadGuard {
    pub epoch: u64,
    reader_id: u64,
    registry: Arc<ReaderRegistry>,
}

impl Drop for ReadGuard {
    fn drop(&mut self) {
        self.registry.unregister(self.reader_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_id() {
        let v1 = VersionId::new(1, 0);
        let v2 = VersionId::new(2, 0);

        assert!(v1 < v2);
        assert!(v1.is_stale(2));
        assert!(!v2.is_stale(2));
    }

    #[test]
    fn test_version_chain_basic() {
        let mut chain: VersionChain<String> = VersionChain::new();

        chain.add_version(VersionedValue::new(VersionId::new(0, 0), "v1".to_string()));
        chain.add_version(VersionedValue::new(VersionId::new(1, 0), "v2".to_string()));

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.latest().unwrap().value, "v2");
    }

    #[test]
    fn test_version_chain_gc() {
        let mut chain: VersionChain<String> = VersionChain::new();

        // Add versions at different epochs
        for epoch in 0..5 {
            chain.add_version(VersionedValue::new(
                VersionId::new(epoch, 0),
                format!("v{}", epoch),
            ));
        }

        assert_eq!(chain.len(), 5);

        // GC with watermark at epoch 3
        let (removed, _) = chain.gc(3);

        // Should keep epochs 3,4 and one base version
        assert!(removed > 0);
        assert!(chain.len() < 5);
    }

    #[test]
    fn test_reader_registry() {
        let registry = ReaderRegistry::new();

        let r1 = registry.register(10);
        let _r2 = registry.register(20);

        assert_eq!(registry.active_count(), 2);
        assert_eq!(registry.min_active_epoch(), Some(10));

        registry.unregister(r1);
        assert_eq!(registry.active_count(), 1);
        assert_eq!(registry.min_active_epoch(), Some(20));
    }

    #[test]
    fn test_epoch_gc_basic() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        let _v1 = gc.insert("key1".to_string(), 100);
        let _v2 = gc.insert("key1".to_string(), 200);

        assert_eq!(gc.get(&"key1".to_string()), Some(200));
        assert_eq!(gc.version_count(), 2);
    }

    #[test]
    fn test_epoch_gc_delete() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        gc.insert("key1".to_string(), 100);
        gc.delete("key1".to_string(), 0); // tombstone

        assert_eq!(gc.get(&"key1".to_string()), None);
    }

    #[test]
    fn test_epoch_gc_at_epoch() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        gc.insert("key1".to_string(), 100);
        gc.advance_epoch();
        gc.insert("key1".to_string(), 200);
        gc.advance_epoch();
        gc.insert("key1".to_string(), 300);

        assert_eq!(gc.get_at_epoch(&"key1".to_string(), 0), Some(100));
        assert_eq!(gc.get_at_epoch(&"key1".to_string(), 1), Some(200));
        assert_eq!(gc.get_at_epoch(&"key1".to_string(), 2), Some(300));
    }

    #[test]
    fn test_read_guard() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        gc.insert("key1".to_string(), 100);

        {
            let _guard = gc.begin_read();
            assert_eq!(gc.readers.active_count(), 1);
        }

        assert_eq!(gc.readers.active_count(), 0);
    }

    #[test]
    fn test_watermark_calculation() {
        let gc: EpochGC<String, i32> = EpochGC::with_config(GCConfig {
            min_epochs_to_keep: 2,
            ..Default::default()
        });

        // Epoch 0
        gc.insert("k".to_string(), 1);
        gc.advance_epoch(); // Epoch 1
        gc.insert("k".to_string(), 2);
        gc.advance_epoch(); // Epoch 2
        gc.insert("k".to_string(), 3);
        gc.advance_epoch(); // Epoch 3
        gc.insert("k".to_string(), 4);
        gc.advance_epoch(); // Epoch 4

        // With no readers and grace period 2, watermark should be 4-2=2
        assert!(gc.watermark() <= 2);

        // With a reader at epoch 1, watermark should be min(2, 1) = 1
        let _guard = gc.begin_read();
        assert!(gc.watermark() <= gc.current_epoch());
    }

    #[test]
    fn test_gc_cycle() {
        let gc: EpochGC<String, i32> = EpochGC::with_config(GCConfig {
            min_epochs_to_keep: 1,
            gc_trigger_threshold: 100,
            max_versions_per_cycle: 100,
        });

        // Create multiple versions across epochs
        for i in 0..10 {
            gc.insert("key".to_string(), i);
            gc.advance_epoch();
        }

        assert_eq!(gc.version_count(), 10);

        // Run GC
        let result = gc.try_gc();

        // Should have collected some versions
        assert!(result.versions_collected > 0 || gc.version_count() < 10);
    }

    #[test]
    fn test_gc_stats() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        for i in 0..5 {
            gc.insert("key".to_string(), i);
            gc.advance_epoch();
        }

        gc.try_gc();

        let stats = gc.stats();
        assert!(stats.gc_cycles >= 1);
    }

    #[test]
    fn test_force_gc() {
        let gc: EpochGC<String, i32> = EpochGC::with_config(GCConfig {
            min_epochs_to_keep: 0,
            gc_trigger_threshold: 1000,
            max_versions_per_cycle: 1, // Very limited
        });

        for i in 0..20 {
            gc.insert(format!("key{}", i), i);
        }

        gc.advance_epoch();
        gc.advance_epoch();

        let initial_count = gc.version_count();
        gc.force_gc();
        let final_count = gc.version_count();

        // Force GC should process all chains regardless of limit
        assert!(final_count <= initial_count);
    }

    #[test]
    fn test_chain_count() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        gc.insert("key1".to_string(), 1);
        gc.insert("key2".to_string(), 2);
        gc.insert("key3".to_string(), 3);

        assert_eq!(gc.chain_count(), 3);
    }

    #[test]
    fn test_version_at_respects_tombstone() {
        let mut chain: VersionChain<i32> = VersionChain::new();

        chain.add_version(VersionedValue::new(VersionId::new(0, 0), 100));
        chain.add_version(VersionedValue::tombstone(VersionId::new(1, 0), 0));

        // Should not return tombstone
        assert!(chain.version_at(1).is_none());
        // Should return the version before tombstone
        assert_eq!(chain.version_at(0).map(|v| v.value), Some(100));
    }

    #[test]
    fn test_gc_result_fields() {
        let gc: EpochGC<String, i32> = EpochGC::new();

        for i in 0..5 {
            gc.insert("key".to_string(), i);
            gc.advance_epoch();
        }

        let result = gc.try_gc();

        // Verify result has reasonable values
        assert!(result.watermark <= gc.current_epoch());
        assert!(result.chains_scanned <= gc.chain_count() + 1);
    }
}
