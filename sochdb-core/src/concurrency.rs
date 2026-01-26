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

//! Hierarchical Lock Architecture with Epoch-Based Reclamation
//!
//! This module implements a production-quality concurrency control system:
//! - Intent locks (IS, IX, S, X) for table-level coordination
//! - Sharded row-level locks (256 shards) for minimal contention
//! - Epoch-based reclamation for safe lock-free reads
//! - Optimistic concurrency control for HNSW updates
//!
//! ## Lock Compatibility Matrix
//!
//! ```text
//!         IS   IX   S    X
//! IS      ✓    ✓    ✓    ✗
//! IX      ✓    ✓    ✗    ✗
//! S       ✓    ✗    ✓    ✗
//! X       ✗    ✗    ✗    ✗
//! ```
//!
//! ## Sharded Lock Table
//!
//! 256 shards reduce contention by distributing locks across independent mutexes.
//! With N concurrent writers on M rows:
//! - Per-shard arrival rate: λ' = λ/256
//! - Average wait time: W' ≈ W/256

use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Table identifier
pub type TableId = u64;

/// Row identifier
pub type RowId = u128;

/// Transaction ID for lock ownership
pub type TxnId = u64;

/// Intent lock modes for table-level locks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentLock {
    /// Intent Shared - transaction intends to read some rows
    IntentShared,
    /// Intent Exclusive - transaction intends to write some rows
    IntentExclusive,
    /// Shared - table-level read lock (e.g., for full table scan)
    Shared,
    /// Exclusive - table-level write lock (e.g., for DDL)
    Exclusive,
}

impl IntentLock {
    /// Check if this lock is compatible with another lock
    pub fn is_compatible(&self, other: &IntentLock) -> bool {
        use IntentLock::*;
        matches!(
            (self, other),
            (IntentShared, IntentShared)
                | (IntentShared, IntentExclusive)
                | (IntentShared, Shared)
                | (IntentExclusive, IntentShared)
                | (IntentExclusive, IntentExclusive)
                | (Shared, IntentShared)
                | (Shared, Shared)
        )
    }
}

/// Row-level lock modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Shared (read) lock
    Shared,
    /// Exclusive (write) lock
    Exclusive,
}

impl LockMode {
    pub fn is_compatible(&self, other: &LockMode) -> bool {
        matches!((self, other), (LockMode::Shared, LockMode::Shared))
    }
}

/// Lock held on a table
#[derive(Debug)]
struct TableLockEntry {
    mode: IntentLock,
    holders: Vec<TxnId>,
    #[allow(dead_code)]
    waiters: Vec<(TxnId, IntentLock)>,
}

impl TableLockEntry {
    fn new(mode: IntentLock, txn_id: TxnId) -> Self {
        Self {
            mode,
            holders: vec![txn_id],
            waiters: Vec::new(),
        }
    }
}

/// Lock held on a row
#[derive(Debug)]
struct RowLockEntry {
    mode: LockMode,
    holders: Vec<TxnId>,
}

impl RowLockEntry {
    fn new(mode: LockMode, txn_id: TxnId) -> Self {
        Self {
            mode,
            holders: vec![txn_id],
        }
    }
}

/// Sharded lock table for row-level locks
/// 256 shards reduce contention by ~256x for uniform key distribution
pub struct ShardedLockTable {
    shards: [Mutex<HashMap<RowId, RowLockEntry>>; 256],
    stats: LockTableStats,
}

impl Default for ShardedLockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardedLockTable {
    /// Create a new sharded lock table
    pub fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            stats: LockTableStats::default(),
        }
    }

    /// Get shard index for a row
    #[inline]
    fn shard_index(&self, row_id: RowId) -> usize {
        // Use upper bits for better distribution
        ((row_id >> 64) as usize ^ (row_id as usize)) % 256
    }

    /// Try to acquire a lock on a row
    pub fn try_lock(&self, row_id: RowId, mode: LockMode, txn_id: TxnId) -> LockResult {
        let shard_idx = self.shard_index(row_id);
        let mut shard = self.shards[shard_idx].lock();

        if let Some(entry) = shard.get_mut(&row_id) {
            // Check if already held by this transaction
            if entry.holders.contains(&txn_id) {
                // Upgrade if needed
                if entry.mode == LockMode::Shared && mode == LockMode::Exclusive {
                    if entry.holders.len() == 1 {
                        entry.mode = LockMode::Exclusive;
                        self.stats.upgrades.fetch_add(1, Ordering::Relaxed);
                        return LockResult::Acquired;
                    } else {
                        self.stats.conflicts.fetch_add(1, Ordering::Relaxed);
                        return LockResult::WouldBlock;
                    }
                }
                return LockResult::AlreadyHeld;
            }

            // Check compatibility
            if entry.mode.is_compatible(&mode) {
                entry.holders.push(txn_id);
                self.stats.shared_acquired.fetch_add(1, Ordering::Relaxed);
                return LockResult::Acquired;
            }

            self.stats.conflicts.fetch_add(1, Ordering::Relaxed);
            return LockResult::WouldBlock;
        }

        // No existing lock - acquire
        shard.insert(row_id, RowLockEntry::new(mode, txn_id));
        match mode {
            LockMode::Shared => self.stats.shared_acquired.fetch_add(1, Ordering::Relaxed),
            LockMode::Exclusive => self
                .stats
                .exclusive_acquired
                .fetch_add(1, Ordering::Relaxed),
        };
        LockResult::Acquired
    }

    /// Release a lock on a row
    pub fn unlock(&self, row_id: RowId, txn_id: TxnId) -> bool {
        let shard_idx = self.shard_index(row_id);
        let mut shard = self.shards[shard_idx].lock();

        if let Some(entry) = shard.get_mut(&row_id)
            && let Some(pos) = entry.holders.iter().position(|&id| id == txn_id)
        {
            entry.holders.remove(pos);
            self.stats.released.fetch_add(1, Ordering::Relaxed);

            if entry.holders.is_empty() {
                shard.remove(&row_id);
            }
            return true;
        }
        false
    }

    /// Release all locks held by a transaction
    pub fn unlock_all(&self, txn_id: TxnId) -> usize {
        let mut count = 0;
        for shard in &self.shards {
            let mut shard_guard = shard.lock();
            let to_remove: Vec<RowId> = shard_guard
                .iter()
                .filter(|(_, entry)| entry.holders.contains(&txn_id))
                .map(|(&row_id, _)| row_id)
                .collect();

            for row_id in to_remove {
                if let Some(entry) = shard_guard.get_mut(&row_id)
                    && let Some(pos) = entry.holders.iter().position(|&id| id == txn_id)
                {
                    entry.holders.remove(pos);
                    count += 1;

                    if entry.holders.is_empty() {
                        shard_guard.remove(&row_id);
                    }
                }
            }
        }
        self.stats
            .released
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Get statistics
    pub fn stats(&self) -> &LockTableStats {
        &self.stats
    }
}

/// Lock table statistics
#[derive(Debug, Default)]
pub struct LockTableStats {
    pub shared_acquired: AtomicU64,
    pub exclusive_acquired: AtomicU64,
    pub upgrades: AtomicU64,
    pub conflicts: AtomicU64,
    pub released: AtomicU64,
}

/// Result of a lock attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockResult {
    /// Lock was acquired
    Acquired,
    /// Lock is already held by this transaction
    AlreadyHeld,
    /// Lock would block (conflict with existing lock)
    WouldBlock,
    /// Deadlock detected
    Deadlock,
}

/// Hierarchical lock manager with table and row-level locks
pub struct LockManager {
    /// Table-level intent locks
    table_locks: DashMap<TableId, TableLockEntry>,
    /// Row-level sharded locks (per table)
    row_locks: DashMap<TableId, Arc<ShardedLockTable>>,
    /// Epoch-based reclamation for safe memory access
    epoch: AtomicU64,
    /// Statistics
    stats: LockManagerStats,
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LockManager {
    /// Create a new lock manager
    pub fn new() -> Self {
        Self {
            table_locks: DashMap::new(),
            row_locks: DashMap::new(),
            epoch: AtomicU64::new(0),
            stats: LockManagerStats::default(),
        }
    }

    /// Acquire an intent lock on a table
    pub fn lock_table(&self, table_id: TableId, mode: IntentLock, txn_id: TxnId) -> LockResult {
        use dashmap::mapref::entry::Entry;

        match self.table_locks.entry(table_id) {
            Entry::Vacant(vacant) => {
                vacant.insert(TableLockEntry::new(mode, txn_id));
                self.stats
                    .table_locks_acquired
                    .fetch_add(1, Ordering::Relaxed);
                LockResult::Acquired
            }
            Entry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();

                // Check if already held by this transaction
                if entry.holders.contains(&txn_id) {
                    return LockResult::AlreadyHeld;
                }

                // Check compatibility
                if entry.mode.is_compatible(&mode) {
                    entry.holders.push(txn_id);
                    self.stats
                        .table_locks_acquired
                        .fetch_add(1, Ordering::Relaxed);
                    return LockResult::Acquired;
                }

                self.stats.table_conflicts.fetch_add(1, Ordering::Relaxed);
                LockResult::WouldBlock
            }
        }
    }

    /// Release a table-level lock
    pub fn unlock_table(&self, table_id: TableId, txn_id: TxnId) -> bool {
        if let Some(mut entry) = self.table_locks.get_mut(&table_id)
            && let Some(pos) = entry.holders.iter().position(|&id| id == txn_id)
        {
            entry.holders.remove(pos);
            self.stats
                .table_locks_released
                .fetch_add(1, Ordering::Relaxed);

            if entry.holders.is_empty() {
                drop(entry);
                self.table_locks.remove(&table_id);
            }
            return true;
        }
        false
    }

    /// Get or create row lock table for a table
    fn get_row_lock_table(&self, table_id: TableId) -> Arc<ShardedLockTable> {
        self.row_locks
            .entry(table_id)
            .or_insert_with(|| Arc::new(ShardedLockTable::new()))
            .clone()
    }

    /// Acquire a row-level lock
    pub fn lock_row(
        &self,
        table_id: TableId,
        row_id: RowId,
        mode: LockMode,
        txn_id: TxnId,
    ) -> LockResult {
        // First acquire intent lock on table
        let intent_mode = match mode {
            LockMode::Shared => IntentLock::IntentShared,
            LockMode::Exclusive => IntentLock::IntentExclusive,
        };

        match self.lock_table(table_id, intent_mode, txn_id) {
            LockResult::Acquired | LockResult::AlreadyHeld => {}
            result => return result,
        }

        // Then acquire row lock
        let row_locks = self.get_row_lock_table(table_id);
        row_locks.try_lock(row_id, mode, txn_id)
    }

    /// Release a row-level lock
    pub fn unlock_row(&self, table_id: TableId, row_id: RowId, txn_id: TxnId) -> bool {
        if let Some(row_locks) = self.row_locks.get(&table_id) {
            return row_locks.unlock(row_id, txn_id);
        }
        false
    }

    /// Release all locks held by a transaction
    pub fn release_all(&self, txn_id: TxnId) -> usize {
        let mut count = 0;

        // Release row locks
        for entry in self.row_locks.iter() {
            count += entry.value().unlock_all(txn_id);
        }

        // Release table locks
        let table_ids: Vec<TableId> = self
            .table_locks
            .iter()
            .filter(|e| e.value().holders.contains(&txn_id))
            .map(|e| *e.key())
            .collect();

        for table_id in table_ids {
            if self.unlock_table(table_id, txn_id) {
                count += 1;
            }
        }

        count
    }

    /// Enter a new epoch (for epoch-based reclamation)
    pub fn enter_epoch(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::AcqRel)
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Get statistics
    pub fn stats(&self) -> &LockManagerStats {
        &self.stats
    }
}

/// Lock manager statistics
#[derive(Debug, Default)]
pub struct LockManagerStats {
    pub table_locks_acquired: AtomicU64,
    pub table_locks_released: AtomicU64,
    pub table_conflicts: AtomicU64,
}

/// Optimistic concurrency control for HNSW nodes
/// Uses version counters instead of locks for wait-free reads
pub struct OptimisticVersion {
    /// Version counter (even = stable, odd = being modified)
    version: AtomicU64,
}

impl Default for OptimisticVersion {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimisticVersion {
    pub fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
        }
    }

    /// Read the version (for optimistic read)
    #[inline]
    pub fn read_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Check if version is stable (not being modified)
    #[inline]
    pub fn is_stable(&self, version: u64) -> bool {
        version & 1 == 0
    }

    /// Validate that version hasn't changed
    #[inline]
    pub fn validate(&self, read_version: u64) -> bool {
        // Ensure we see all writes before version check
        std::sync::atomic::fence(Ordering::Acquire);
        self.version.load(Ordering::Relaxed) == read_version
    }

    /// Try to begin a write (returns None if concurrent write in progress)
    pub fn try_write_begin(&self) -> Option<WriteGuard<'_>> {
        let current = self.version.load(Ordering::Acquire);

        // Check if stable
        if !self.is_stable(current) {
            return None;
        }

        // Try to CAS to odd (writing) state
        match self.version.compare_exchange(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => Some(WriteGuard {
                version: &self.version,
                start_version: current,
            }),
            Err(_) => None,
        }
    }
}

/// Guard that commits version on drop
pub struct WriteGuard<'a> {
    version: &'a AtomicU64,
    start_version: u64,
}

impl<'a> WriteGuard<'a> {
    /// Commit the write (increment version to even)
    pub fn commit(self) {
        self.version
            .store(self.start_version + 2, Ordering::Release);
        std::mem::forget(self); // Don't run drop
    }

    /// Abort the write (restore original version)
    pub fn abort(self) {
        self.version.store(self.start_version, Ordering::Release);
        std::mem::forget(self);
    }
}

impl<'a> Drop for WriteGuard<'a> {
    fn drop(&mut self) {
        // If dropped without commit/abort, abort the write
        self.version.store(self.start_version, Ordering::Release);
    }
}

/// Epoch-based reclamation for safe memory access
pub struct EpochGuard {
    manager: Arc<EpochManager>,
    epoch: u64,
}

impl Drop for EpochGuard {
    fn drop(&mut self) {
        self.manager.leave_epoch(self.epoch);
    }
}

/// Epoch manager for safe memory reclamation
pub struct EpochManager {
    /// Current global epoch
    global_epoch: AtomicU64,
    /// Number of threads in each epoch
    epoch_counts: [AtomicUsize; 4],
    /// Retired items pending reclamation
    retired: Mutex<Vec<(u64, Box<dyn Send>)>>,
}

impl Default for EpochManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochManager {
    pub fn new() -> Self {
        Self {
            global_epoch: AtomicU64::new(0),
            epoch_counts: std::array::from_fn(|_| AtomicUsize::new(0)),
            retired: Mutex::new(Vec::new()),
        }
    }

    /// Pin the current epoch (enter critical section)
    pub fn pin(self: &Arc<Self>) -> EpochGuard {
        let epoch = self.global_epoch.load(Ordering::Acquire);
        self.epoch_counts[(epoch % 4) as usize].fetch_add(1, Ordering::AcqRel);

        EpochGuard {
            manager: self.clone(),
            epoch,
        }
    }

    /// Leave an epoch
    fn leave_epoch(&self, epoch: u64) {
        self.epoch_counts[(epoch % 4) as usize].fetch_sub(1, Ordering::AcqRel);
    }

    /// Advance the global epoch (called periodically)
    pub fn advance(&self) {
        let current = self.global_epoch.load(Ordering::Acquire);
        let old_epoch = (current + 2) % 4; // Two epochs ago

        // Check if old epoch is empty
        if self.epoch_counts[old_epoch as usize].load(Ordering::Acquire) == 0 {
            self.global_epoch.fetch_add(1, Ordering::AcqRel);
            self.reclaim(current.saturating_sub(2));
        }
    }

    /// Retire an object for later reclamation
    pub fn retire<T: Send + 'static>(&self, item: T) {
        let epoch = self.global_epoch.load(Ordering::Acquire);
        let mut retired = self.retired.lock();
        retired.push((epoch, Box::new(item)));
    }

    /// Reclaim objects from old epochs
    fn reclaim(&self, safe_epoch: u64) {
        let mut retired = self.retired.lock();
        retired.retain(|(epoch, _)| *epoch > safe_epoch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_intent_lock_compatibility() {
        use IntentLock::*;

        // IS compatible with IS, IX, S
        assert!(IntentShared.is_compatible(&IntentShared));
        assert!(IntentShared.is_compatible(&IntentExclusive));
        assert!(IntentShared.is_compatible(&Shared));
        assert!(!IntentShared.is_compatible(&Exclusive));

        // IX compatible with IS, IX
        assert!(IntentExclusive.is_compatible(&IntentShared));
        assert!(IntentExclusive.is_compatible(&IntentExclusive));
        assert!(!IntentExclusive.is_compatible(&Shared));
        assert!(!IntentExclusive.is_compatible(&Exclusive));

        // S compatible with IS, S
        assert!(Shared.is_compatible(&IntentShared));
        assert!(!Shared.is_compatible(&IntentExclusive));
        assert!(Shared.is_compatible(&Shared));
        assert!(!Shared.is_compatible(&Exclusive));

        // X compatible with nothing
        assert!(!Exclusive.is_compatible(&IntentShared));
        assert!(!Exclusive.is_compatible(&IntentExclusive));
        assert!(!Exclusive.is_compatible(&Shared));
        assert!(!Exclusive.is_compatible(&Exclusive));
    }

    #[test]
    fn test_sharded_lock_table_basic() {
        let table = ShardedLockTable::new();

        // Acquire exclusive lock
        assert_eq!(
            table.try_lock(1, LockMode::Exclusive, 100),
            LockResult::Acquired
        );

        // Can't acquire another exclusive on same row
        assert_eq!(
            table.try_lock(1, LockMode::Exclusive, 200),
            LockResult::WouldBlock
        );

        // Can't acquire shared on exclusive
        assert_eq!(
            table.try_lock(1, LockMode::Shared, 200),
            LockResult::WouldBlock
        );

        // Can acquire on different row
        assert_eq!(
            table.try_lock(2, LockMode::Exclusive, 200),
            LockResult::Acquired
        );

        // Release and reacquire
        assert!(table.unlock(1, 100));
        assert_eq!(
            table.try_lock(1, LockMode::Exclusive, 200),
            LockResult::Acquired
        );
    }

    #[test]
    fn test_sharded_lock_table_shared() {
        let table = ShardedLockTable::new();

        // Multiple shared locks on same row
        assert_eq!(
            table.try_lock(1, LockMode::Shared, 100),
            LockResult::Acquired
        );
        assert_eq!(
            table.try_lock(1, LockMode::Shared, 200),
            LockResult::Acquired
        );
        assert_eq!(
            table.try_lock(1, LockMode::Shared, 300),
            LockResult::Acquired
        );

        // Can't acquire exclusive while shared held
        assert_eq!(
            table.try_lock(1, LockMode::Exclusive, 400),
            LockResult::WouldBlock
        );

        // Release all shared, then exclusive works
        assert!(table.unlock(1, 100));
        assert!(table.unlock(1, 200));
        assert!(table.unlock(1, 300));
        assert_eq!(
            table.try_lock(1, LockMode::Exclusive, 400),
            LockResult::Acquired
        );
    }

    #[test]
    fn test_sharded_lock_upgrade() {
        let table = ShardedLockTable::new();

        // Acquire shared
        assert_eq!(
            table.try_lock(1, LockMode::Shared, 100),
            LockResult::Acquired
        );

        // Try to upgrade - should succeed since we're only holder
        assert_eq!(
            table.try_lock(1, LockMode::Exclusive, 100),
            LockResult::Acquired
        );

        // Now we hold exclusive
        assert_eq!(
            table.try_lock(1, LockMode::Shared, 200),
            LockResult::WouldBlock
        );
    }

    #[test]
    fn test_lock_manager_hierarchical() {
        let manager = LockManager::new();

        // Acquire row lock (should auto-acquire intent lock)
        assert_eq!(
            manager.lock_row(1, 100, LockMode::Exclusive, 1000),
            LockResult::Acquired
        );

        // Another transaction can read different row
        assert_eq!(
            manager.lock_row(1, 200, LockMode::Shared, 2000),
            LockResult::Acquired
        );

        // Can't acquire exclusive on locked row
        assert_eq!(
            manager.lock_row(1, 100, LockMode::Exclusive, 2000),
            LockResult::WouldBlock
        );

        // Release all
        let released = manager.release_all(1000);
        assert!(released >= 1);
    }

    #[test]
    fn test_optimistic_version() {
        let version = OptimisticVersion::new();

        // Read should see stable version (0)
        let v = version.read_version();
        assert!(version.is_stable(v));
        assert!(version.validate(v));

        // Write should increment
        {
            let guard = version.try_write_begin().unwrap();
            let v_during = version.read_version();
            assert!(!version.is_stable(v_during)); // Odd = writing
            guard.commit();
        }

        // After commit, version is 2 (even = stable)
        let v2 = version.read_version();
        assert!(version.is_stable(v2));
        assert_eq!(v2, 2);
    }

    #[test]
    fn test_optimistic_concurrent() {
        let version = Arc::new(OptimisticVersion::new());

        // Start a write
        let guard = version.try_write_begin().unwrap();

        // Concurrent write attempt should fail
        let version2 = version.clone();
        let result = version2.try_write_begin();
        assert!(result.is_none());

        // Finish write
        guard.commit();

        // Now another write should succeed
        let guard2 = version.try_write_begin().unwrap();
        guard2.commit();
    }

    #[test]
    fn test_epoch_manager() {
        let manager = Arc::new(EpochManager::new());

        // Pin epoch
        let guard1 = manager.pin();
        assert_eq!(guard1.epoch, 0);

        // Retire something
        manager.retire(vec![1, 2, 3]);

        // Advance epoch
        manager.advance();

        // New pin gets new epoch
        let guard2 = manager.pin();
        assert!(guard2.epoch >= guard1.epoch);

        // Drop guards
        drop(guard1);
        drop(guard2);

        // Can advance more
        manager.advance();
        manager.advance();
    }

    #[test]
    fn test_sharded_distribution() {
        let table = ShardedLockTable::new();

        // Lock many rows and verify distribution
        for i in 0..1000u128 {
            assert_eq!(table.try_lock(i, LockMode::Shared, 1), LockResult::Acquired);
        }

        // Count locks per shard (should be somewhat distributed)
        let mut non_empty_shards = 0;
        for shard in &table.shards {
            if !shard.lock().is_empty() {
                non_empty_shards += 1;
            }
        }

        // With 1000 locks across 256 shards, should use many shards
        assert!(
            non_empty_shards > 100,
            "Expected better distribution: {} shards used",
            non_empty_shards
        );
    }

    #[test]
    fn test_unlock_all() {
        let table = ShardedLockTable::new();

        // Lock many rows as txn 100
        for i in 0..50u128 {
            table.try_lock(i, LockMode::Exclusive, 100);
        }

        // Lock some as txn 200
        for i in 50..100u128 {
            table.try_lock(i, LockMode::Exclusive, 200);
        }

        // Unlock all for txn 100
        let released = table.unlock_all(100);
        assert_eq!(released, 50);

        // Txn 200's locks should still be held
        assert_eq!(
            table.try_lock(50, LockMode::Exclusive, 300),
            LockResult::WouldBlock
        );

        // Txn 100's former locks should be available
        assert_eq!(
            table.try_lock(0, LockMode::Exclusive, 300),
            LockResult::Acquired
        );
    }

    #[test]
    fn test_concurrent_locks() {
        let table = Arc::new(ShardedLockTable::new());
        let mut handles = vec![];

        // Spawn threads that each lock different rows
        for txn_id in 0..16u64 {
            let table = table.clone();
            handles.push(thread::spawn(move || {
                let start = txn_id as u128 * 100;
                for i in 0..100 {
                    let result = table.try_lock(start + i, LockMode::Exclusive, txn_id);
                    assert_eq!(result, LockResult::Acquired);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify stats
        assert_eq!(
            table.stats().exclusive_acquired.load(Ordering::Relaxed),
            1600
        );
    }
}
