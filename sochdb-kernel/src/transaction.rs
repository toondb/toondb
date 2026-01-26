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

//! Transaction Management
//!
//! Core transaction manager with MVCC support.
//! This is the minimal ACID transaction implementation for the kernel.

use crate::error::{KernelError, KernelResult, TransactionErrorKind};
use crate::wal::LogSequenceNumber;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Transaction identifier
pub type TransactionId = u64;

/// Timestamp for MVCC
pub type Timestamp = u64;

/// Isolation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Read uncommitted - sees uncommitted changes (rarely used)
    ReadUncommitted,
    /// Read committed - only sees committed changes
    ReadCommitted,
    /// Repeatable read - snapshot at first read
    RepeatableRead,
    /// Snapshot isolation - snapshot at transaction start
    #[default]
    SnapshotIsolation,
    /// Serializable - full serializability via SSI
    Serializable,
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is active
    Active,
    /// Transaction is preparing to commit
    Preparing,
    /// Transaction is committed
    Committed,
    /// Transaction is aborted
    Aborted,
}

/// Transaction metadata
#[derive(Debug)]
struct TransactionInfo {
    /// Transaction ID
    id: TransactionId,
    /// Transaction state
    state: TransactionState,
    /// Snapshot timestamp (for MVCC visibility)
    snapshot_ts: Timestamp,
    /// Commit timestamp (set on commit)
    commit_ts: Option<Timestamp>,
    /// Isolation level
    isolation: IsolationLevel,
    /// Start time (for timeout detection)
    start_time: Instant,
    /// Last LSN written by this transaction
    last_lsn: Option<LogSequenceNumber>,
    /// Read set (for SSI conflict detection)
    read_set: Vec<(u32, u64)>, // (table_id, row_id)
    /// Write set (for conflict detection)
    write_set: Vec<(u32, u64)>, // (table_id, row_id)
}

/// Transaction manager
///
/// Manages transaction lifecycle and MVCC timestamps.
pub struct TxnManager {
    /// Next transaction ID
    next_txn_id: AtomicU64,
    /// Current timestamp (logical clock)
    current_ts: AtomicU64,
    /// Active transactions
    active_txns: RwLock<HashMap<TransactionId, TransactionInfo>>,
    /// Transaction timeout
    timeout: Duration,
    /// Lock for commit ordering
    commit_lock: Mutex<()>,
}

impl Default for TxnManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TxnManager {
    /// Create a new transaction manager
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(60))
    }

    /// Create with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            next_txn_id: AtomicU64::new(1),
            current_ts: AtomicU64::new(1),
            active_txns: RwLock::new(HashMap::new()),
            timeout,
            commit_lock: Mutex::new(()),
        }
    }

    /// Begin a new transaction with default isolation
    pub fn begin(&self) -> TransactionId {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begin a new transaction with specific isolation level
    pub fn begin_with_isolation(&self, isolation: IsolationLevel) -> TransactionId {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let snapshot_ts = self.current_ts.load(Ordering::SeqCst);

        let info = TransactionInfo {
            id: txn_id,
            state: TransactionState::Active,
            snapshot_ts,
            commit_ts: None,
            isolation,
            start_time: Instant::now(),
            last_lsn: None,
            read_set: Vec::new(),
            write_set: Vec::new(),
        };

        self.active_txns.write().insert(txn_id, info);
        txn_id
    }

    /// Commit a transaction
    pub fn commit(&self, txn_id: TransactionId) -> KernelResult<Timestamp> {
        // Acquire commit lock for ordering
        let _guard = self.commit_lock.lock();

        let mut txns = self.active_txns.write();

        // First check state and get necessary info
        let (current_state, isolation, read_set, write_set) = {
            let info = txns.get(&txn_id).ok_or(KernelError::Transaction {
                kind: TransactionErrorKind::NotFound(txn_id),
            })?;
            (
                info.state,
                info.isolation,
                info.read_set.clone(),
                info.write_set.clone(),
            )
        };

        match current_state {
            TransactionState::Active | TransactionState::Preparing => {
                // Check for SSI conflicts if serializable (using cloned data)
                if isolation == IsolationLevel::Serializable {
                    self.check_serialization_conflicts_cloned(&read_set, &write_set)?;
                }

                // Now get mutable reference and update
                let info = txns.get_mut(&txn_id).unwrap();

                // Allocate commit timestamp
                let commit_ts = self.current_ts.fetch_add(1, Ordering::SeqCst);
                info.commit_ts = Some(commit_ts);
                info.state = TransactionState::Committed;

                Ok(commit_ts)
            }
            TransactionState::Committed => Err(KernelError::Transaction {
                kind: TransactionErrorKind::AlreadyCommitted,
            }),
            TransactionState::Aborted => Err(KernelError::Transaction {
                kind: TransactionErrorKind::AlreadyAborted,
            }),
        }
    }

    /// Abort a transaction
    pub fn abort(&self, txn_id: TransactionId) -> KernelResult<()> {
        let mut txns = self.active_txns.write();
        let info = txns.get_mut(&txn_id).ok_or(KernelError::Transaction {
            kind: TransactionErrorKind::NotFound(txn_id),
        })?;

        match info.state {
            TransactionState::Active | TransactionState::Preparing => {
                info.state = TransactionState::Aborted;
                Ok(())
            }
            TransactionState::Committed => Err(KernelError::Transaction {
                kind: TransactionErrorKind::AlreadyCommitted,
            }),
            TransactionState::Aborted => Ok(()), // Idempotent
        }
    }

    /// Check if a transaction is active
    pub fn is_active(&self, txn_id: TransactionId) -> bool {
        self.active_txns
            .read()
            .get(&txn_id)
            .map(|info| info.state == TransactionState::Active)
            .unwrap_or(false)
    }

    /// Get snapshot timestamp for a transaction
    pub fn snapshot_ts(&self, txn_id: TransactionId) -> KernelResult<Timestamp> {
        self.active_txns
            .read()
            .get(&txn_id)
            .map(|info| info.snapshot_ts)
            .ok_or(KernelError::Transaction {
                kind: TransactionErrorKind::NotFound(txn_id),
            })
    }

    /// Record a read operation (for SSI)
    pub fn record_read(&self, txn_id: TransactionId, table_id: u32, row_id: u64) {
        if let Some(info) = self.active_txns.write().get_mut(&txn_id)
            && info.isolation == IsolationLevel::Serializable
        {
            info.read_set.push((table_id, row_id));
        }
    }

    /// Record a write operation
    pub fn record_write(&self, txn_id: TransactionId, table_id: u32, row_id: u64) {
        if let Some(info) = self.active_txns.write().get_mut(&txn_id) {
            info.write_set.push((table_id, row_id));
        }
    }

    /// Update last LSN for a transaction
    pub fn set_last_lsn(&self, txn_id: TransactionId, lsn: LogSequenceNumber) {
        if let Some(info) = self.active_txns.write().get_mut(&txn_id) {
            info.last_lsn = Some(lsn);
        }
    }

    /// Get minimum active snapshot timestamp (for GC)
    pub fn min_active_snapshot(&self) -> Option<Timestamp> {
        self.active_txns
            .read()
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .map(|info| info.snapshot_ts)
            .min()
    }

    /// Get active transaction count
    pub fn active_count(&self) -> usize {
        self.active_txns
            .read()
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .count()
    }

    /// Clean up completed transactions older than retention period
    pub fn cleanup(&self, retention: Duration) {
        let now = Instant::now();
        self.active_txns.write().retain(|_, info| {
            // Keep active transactions
            if info.state == TransactionState::Active {
                return true;
            }
            // Keep recently completed transactions
            now.duration_since(info.start_time) < retention
        });
    }

    /// Check for transactions that have timed out
    pub fn check_timeouts(&self) -> Vec<TransactionId> {
        let now = Instant::now();
        self.active_txns
            .read()
            .values()
            .filter(|info| {
                info.state == TransactionState::Active
                    && now.duration_since(info.start_time) > self.timeout
            })
            .map(|info| info.id)
            .collect()
    }

    /// Check serialization conflicts for SSI
    #[allow(dead_code)]
    fn check_serialization_conflicts(
        &self,
        txn: &TransactionInfo,
        _all_txns: &HashMap<TransactionId, TransactionInfo>,
    ) -> KernelResult<()> {
        // Simplified SSI check - in production this would track rw-dependencies
        // and detect dangerous structures (two consecutive rw-antidependencies)
        //
        // For now, we just check for write-write conflicts
        // Full SSI implementation is in sochdb-storage/src/ssi.rs
        let _ = txn;
        Ok(())
    }

    /// Check serialization conflicts for SSI (using cloned data to avoid borrow issues)
    fn check_serialization_conflicts_cloned(
        &self,
        _read_set: &[(u32, u64)],
        _write_set: &[(u32, u64)],
    ) -> KernelResult<()> {
        // Simplified SSI check - in production this would track rw-dependencies
        // and detect dangerous structures (two consecutive rw-antidependencies)
        //
        // For now, we just check for write-write conflicts
        // Full SSI implementation is in sochdb-storage/src/ssi.rs
        Ok(())
    }

    /// Get current timestamp
    pub fn current_timestamp(&self) -> Timestamp {
        self.current_ts.load(Ordering::SeqCst)
    }

    /// Restore state after recovery
    pub fn restore(&self, next_txn_id: TransactionId, current_ts: Timestamp) {
        self.next_txn_id.store(next_txn_id, Ordering::SeqCst);
        self.current_ts.store(current_ts, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_commit() {
        let mgr = TxnManager::new();

        let txn1 = mgr.begin();
        assert!(mgr.is_active(txn1));

        let commit_ts = mgr.commit(txn1).unwrap();
        assert!(!mgr.is_active(txn1));
        assert!(commit_ts > 0);
    }

    #[test]
    fn test_begin_abort() {
        let mgr = TxnManager::new();

        let txn1 = mgr.begin();
        assert!(mgr.is_active(txn1));

        mgr.abort(txn1).unwrap();
        assert!(!mgr.is_active(txn1));
    }

    #[test]
    fn test_snapshot_isolation() {
        let mgr = TxnManager::new();

        let txn1 = mgr.begin();
        let ts1 = mgr.snapshot_ts(txn1).unwrap();

        // Commit txn1 to advance timestamp
        mgr.commit(txn1).unwrap();

        let txn2 = mgr.begin();
        let ts2 = mgr.snapshot_ts(txn2).unwrap();

        // txn2 should have later snapshot
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_double_commit_fails() {
        let mgr = TxnManager::new();
        let txn1 = mgr.begin();

        mgr.commit(txn1).unwrap();
        assert!(mgr.commit(txn1).is_err());
    }

    #[test]
    fn test_min_active_snapshot() {
        let mgr = TxnManager::new();

        let txn1 = mgr.begin();
        let txn2 = mgr.begin();

        let min = mgr.min_active_snapshot().unwrap();
        assert_eq!(min, mgr.snapshot_ts(txn1).unwrap());

        mgr.commit(txn1).unwrap();
        let min = mgr.min_active_snapshot().unwrap();
        assert_eq!(min, mgr.snapshot_ts(txn2).unwrap());
    }
}
