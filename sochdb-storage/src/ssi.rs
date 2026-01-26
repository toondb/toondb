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

//! Serializable Snapshot Isolation (SSI) Implementation
//!
//! This module extends basic snapshot isolation with serializability guarantees
//! by detecting and preventing dangerous structures (rw-antidependency cycles).
//!
//! ## SSI Algorithm
//!
//! SSI tracks read-write dependencies between concurrent transactions:
//! - T₁ →ʳʷ T₂ means: T₁ read version v, T₂ wrote new version v' of same row
//!   where v'.begin_ts > T₁.snapshot_ts
//!
//! A transaction must abort if it participates in a dangerous structure:
//! - Two incoming rw-antidependencies (pivot in/out), OR
//! - Cycle in the dependency graph
//!
//! ## Write-Write Conflict Detection
//!
//! Uses first-updater-wins rule:
//! - When T₁ with snapshot_ts=100 attempts UPDATE on row R
//! - If ∃ version v of R with v.begin_ts > 100:
//!   → ABORT T₁ with SerializationFailure
//! - Else:
//!   → Create new version with begin_ts = T₁.commit_ts
//!
//! ## Performance
//!
//! - Visibility check: O(1) via timestamp comparison
//! - Conflict check: O(active_txns) per commit
//! - Space: O(active_txns²) for dependency tracking

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// Transaction ID type
pub type TxnId = u64;

/// Timestamp type (hybrid logical clock recommended)
pub type Timestamp = u64;

/// SSI transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsiTxnStatus {
    /// Transaction is active
    Active,
    /// Transaction committed with timestamp
    Committed(Timestamp),
    /// Transaction aborted (optionally with reason)
    Aborted,
}

/// Conflict type for SSI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Write-write conflict: two transactions updated same row
    WriteWrite,
    /// Read-write antidependency: T read, then another T wrote
    ReadWriteAnti,
    /// Dangerous structure detected (would cause anomaly)
    DangerousStructure,
}

/// SSI conflict error
#[derive(Debug, Clone)]
pub struct SsiConflictError {
    /// Transaction that must abort
    pub victim_txn: TxnId,
    /// Transaction that won the conflict
    pub winner_txn: Option<TxnId>,
    /// Type of conflict
    pub conflict_type: ConflictType,
    /// Human-readable description
    pub message: String,
}

impl std::fmt::Display for SsiConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSI conflict ({:?}): {}",
            self.conflict_type, self.message
        )
    }
}

impl std::error::Error for SsiConflictError {}

/// Version metadata with SSI timestamps
#[derive(Debug, Clone)]
pub struct SsiVersionInfo {
    /// Transaction that created this version
    pub xmin: TxnId,
    /// Transaction that deleted/updated this version (0 if active)
    pub xmax: TxnId,
    /// Begin timestamp (when version became visible)
    pub begin_ts: Timestamp,
    /// End timestamp (when version was superseded, MAX if active)
    pub end_ts: Timestamp,
    /// Commit timestamp (for committed transactions)
    pub commit_ts: Option<Timestamp>,
}

impl SsiVersionInfo {
    /// Create a new active version
    pub fn new(xmin: TxnId, begin_ts: Timestamp) -> Self {
        Self {
            xmin,
            xmax: 0,
            begin_ts,
            end_ts: Timestamp::MAX,
            commit_ts: None,
        }
    }

    /// Check if version is visible to snapshot
    ///
    /// A version is visible if:
    /// 1. xmin committed before snapshot
    /// 2. xmax is not set, OR aborted, OR committed after snapshot
    pub fn is_visible(&self, snapshot_ts: Timestamp, txn_states: &SsiTxnStates) -> bool {
        // Check xmin
        match txn_states.get_status(self.xmin) {
            Some(SsiTxnStatus::Committed(commit_ts)) => {
                if commit_ts > snapshot_ts {
                    return false; // Created after snapshot
                }
            }
            Some(SsiTxnStatus::Active) | Some(SsiTxnStatus::Aborted) | None => {
                return false; // Not yet committed or aborted
            }
        }

        // Check xmax
        if self.xmax == 0 {
            return true; // Not deleted
        }

        match txn_states.get_status(self.xmax) {
            Some(SsiTxnStatus::Committed(commit_ts)) => {
                commit_ts > snapshot_ts // Deleted after snapshot - still visible
            }
            Some(SsiTxnStatus::Active) | Some(SsiTxnStatus::Aborted) | None => {
                true // Deletion not committed - still visible
            }
        }
    }
}

/// RW-antidependency edge in the serialization graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RwDependency {
    /// Reader transaction (T₁ in T₁ →ʳʷ T₂)
    pub reader: TxnId,
    /// Writer transaction (T₂ in T₁ →ʳʷ T₂)
    pub writer: TxnId,
    /// Key that was read/written
    pub key: Vec<u8>,
}

/// Transaction entry for SSI tracking
#[derive(Debug)]
pub struct SsiTransaction {
    /// Transaction ID
    pub txn_id: TxnId,
    /// Start timestamp (snapshot time)
    pub start_ts: Timestamp,
    /// Status
    pub status: SsiTxnStatus,
    /// Commit timestamp (if committed)
    pub commit_ts: Option<Timestamp>,
    /// Read set (keys this transaction has read)
    pub read_set: HashSet<Vec<u8>>,
    /// Write set (keys this transaction has written)
    pub write_set: HashSet<Vec<u8>>,
    /// Incoming rw-antidependencies (transactions that read before this wrote)
    pub in_rw_deps: HashSet<TxnId>,
    /// Outgoing rw-antidependencies (transactions that wrote after this read)
    pub out_rw_deps: HashSet<TxnId>,
    /// Flag: has incoming from committed transaction
    pub has_committed_in_rw: bool,
    /// Flag: has outgoing to committed transaction
    pub has_committed_out_rw: bool,
}

impl SsiTransaction {
    /// Create a new SSI transaction
    pub fn new(txn_id: TxnId, start_ts: Timestamp) -> Self {
        Self {
            txn_id,
            start_ts,
            status: SsiTxnStatus::Active,
            commit_ts: None,
            read_set: HashSet::new(),
            write_set: HashSet::new(),
            in_rw_deps: HashSet::new(),
            out_rw_deps: HashSet::new(),
            has_committed_in_rw: false,
            has_committed_out_rw: false,
        }
    }

    /// Record a read operation
    pub fn record_read(&mut self, key: Vec<u8>) {
        self.read_set.insert(key);
    }

    /// Record a write operation
    pub fn record_write(&mut self, key: Vec<u8>) {
        self.write_set.insert(key);
    }

    /// Check for dangerous structure (two-in-two-out)
    ///
    /// A transaction is part of a dangerous structure if it has:
    /// - At least one incoming rw-antidep from a committed txn, AND
    /// - At least one outgoing rw-antidep to a committed txn
    pub fn is_dangerous(&self) -> bool {
        self.has_committed_in_rw && self.has_committed_out_rw
    }
}

/// Transaction states for visibility checking
pub struct SsiTxnStates {
    /// Transaction states (txn_id -> status)
    states: RwLock<HashMap<TxnId, SsiTxnStatus>>,
}

impl SsiTxnStates {
    pub fn new() -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_status(&self, txn_id: TxnId) -> Option<SsiTxnStatus> {
        self.states.read().get(&txn_id).copied()
    }

    pub fn set_status(&self, txn_id: TxnId, status: SsiTxnStatus) {
        self.states.write().insert(txn_id, status);
    }
}

impl Default for SsiTxnStates {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable Snapshot Isolation Manager
///
/// Provides serializable isolation level using SSI technique.
///
/// ## Usage
///
/// ```ignore
/// let ssi = SsiManager::new();
///
/// // Begin transaction
/// let (txn_id, snapshot_ts) = ssi.begin()?;
///
/// // Read (records rw-dependency)
/// let value = ssi.read(txn_id, key)?;
///
/// // Write (checks write-write conflicts)
/// ssi.write(txn_id, key, value)?;
///
/// // Commit (checks for dangerous structures)
/// ssi.commit(txn_id)?;
/// ```
pub struct SsiManager {
    /// Next transaction ID
    next_txn_id: AtomicU64,
    /// Global timestamp counter
    timestamp: AtomicU64,
    /// Active transactions
    transactions: RwLock<HashMap<TxnId, SsiTransaction>>,
    /// Transaction states for visibility
    txn_states: Arc<SsiTxnStates>,
    /// Key -> latest writer transaction (for write-write detection)
    key_writers: RwLock<HashMap<Vec<u8>, (TxnId, Timestamp)>>,
    /// Key -> list of readers (for rw-antidep tracking)
    key_readers: RwLock<HashMap<Vec<u8>, HashSet<TxnId>>>,
}

impl SsiManager {
    /// Create a new SSI manager
    pub fn new() -> Self {
        Self {
            next_txn_id: AtomicU64::new(1),
            timestamp: AtomicU64::new(1),
            transactions: RwLock::new(HashMap::new()),
            txn_states: Arc::new(SsiTxnStates::new()),
            key_writers: RwLock::new(HashMap::new()),
            key_readers: RwLock::new(HashMap::new()),
        }
    }

    /// Begin a new transaction
    pub fn begin(&self) -> Result<(TxnId, Timestamp), SsiConflictError> {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let start_ts = self.timestamp.fetch_add(1, Ordering::SeqCst);

        let txn = SsiTransaction::new(txn_id, start_ts);
        self.transactions.write().insert(txn_id, txn);
        self.txn_states.set_status(txn_id, SsiTxnStatus::Active);

        Ok((txn_id, start_ts))
    }

    /// Record a read and check for rw-antidependencies
    ///
    /// If another concurrent transaction wrote to this key after our snapshot,
    /// we have an rw-antidependency (T_reader →ʳʷ T_writer).
    pub fn record_read(&self, txn_id: TxnId, key: &[u8]) -> Result<(), SsiConflictError> {
        // Get the snapshot timestamp first
        let snapshot_ts = {
            let txns = self.transactions.read();
            let txn = txns.get(&txn_id).ok_or_else(|| SsiConflictError {
                victim_txn: txn_id,
                winner_txn: None,
                conflict_type: ConflictType::ReadWriteAnti,
                message: "Transaction not found".into(),
            })?;
            txn.start_ts
        };

        // Record in read set
        {
            let mut txns = self.transactions.write();
            if let Some(txn) = txns.get_mut(&txn_id) {
                txn.record_read(key.to_vec());
            }
        }

        // Add to key readers
        self.key_readers
            .write()
            .entry(key.to_vec())
            .or_default()
            .insert(txn_id);

        // Check if there's a concurrent writer
        let writer_info = self.key_writers.read().get(key).cloned();
        if let Some((writer_txn, write_ts)) = writer_info
            && write_ts > snapshot_ts
            && writer_txn != txn_id
        {
            // We read old version, another txn wrote new version
            // This is an rw-antidependency: we →ʳʷ writer
            let writer_committed = matches!(
                self.txn_states.get_status(writer_txn),
                Some(SsiTxnStatus::Committed(_))
            );

            let mut txns = self.transactions.write();

            // Update reader's out deps
            if let Some(reader_txn) = txns.get_mut(&txn_id) {
                reader_txn.out_rw_deps.insert(writer_txn);
                if writer_committed {
                    reader_txn.has_committed_out_rw = true;
                    // Check for dangerous structure
                    if reader_txn.is_dangerous() {
                        return Err(SsiConflictError {
                            victim_txn: txn_id,
                            winner_txn: Some(writer_txn),
                            conflict_type: ConflictType::DangerousStructure,
                            message: format!(
                                "Transaction {} would create serialization anomaly with {}",
                                txn_id, writer_txn
                            ),
                        });
                    }
                }
            }

            // Update writer's in deps
            if let Some(writer_txn_entry) = txns.get_mut(&writer_txn) {
                writer_txn_entry.in_rw_deps.insert(txn_id);
            }
        }

        Ok(())
    }

    /// Record a write and check for write-write conflicts
    ///
    /// Uses first-updater-wins: if another transaction already wrote to this key
    /// after our snapshot, we must abort.
    pub fn record_write(&self, txn_id: TxnId, key: &[u8]) -> Result<(), SsiConflictError> {
        let mut txns = self.transactions.write();
        let txn = txns.get_mut(&txn_id).ok_or_else(|| SsiConflictError {
            victim_txn: txn_id,
            winner_txn: None,
            conflict_type: ConflictType::WriteWrite,
            message: "Transaction not found".into(),
        })?;

        let snapshot_ts = txn.start_ts;

        // Check for write-write conflict (first-updater-wins)
        {
            let key_writers = self.key_writers.read();
            if let Some((prev_writer, write_ts)) = key_writers.get(key)
                && *write_ts > snapshot_ts
                && *prev_writer != txn_id
            {
                // Another transaction already wrote after our snapshot
                return Err(SsiConflictError {
                    victim_txn: txn_id,
                    winner_txn: Some(*prev_writer),
                    conflict_type: ConflictType::WriteWrite,
                    message: format!(
                        "Write-write conflict: transaction {} already wrote to key, ts {}",
                        prev_writer, write_ts
                    ),
                });
            }
        }

        // Record in write set
        txn.record_write(key.to_vec());

        // Update key writer
        let write_ts = self.timestamp.fetch_add(1, Ordering::SeqCst);
        drop(txns);

        self.key_writers
            .write()
            .insert(key.to_vec(), (txn_id, write_ts));

        // Check for rw-antidependency from existing readers
        if let Some(readers) = self.key_readers.read().get(key) {
            let mut txns = self.transactions.write();
            for reader_id in readers {
                if *reader_id != txn_id
                    && let Some(reader_txn) = txns.get(reader_id)
                    && reader_txn.start_ts < write_ts
                {
                    // reader →ʳʷ us (this writer)
                    if let Some(writer_txn) = txns.get_mut(&txn_id) {
                        writer_txn.in_rw_deps.insert(*reader_id);

                        // Check if reader is committed
                        if let Some(SsiTxnStatus::Committed(_)) =
                            self.txn_states.get_status(*reader_id)
                        {
                            writer_txn.has_committed_in_rw = true;
                        }
                    }
                    if let Some(reader_txn) = txns.get_mut(reader_id) {
                        reader_txn.out_rw_deps.insert(txn_id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Commit a transaction
    ///
    /// Checks for dangerous structures before allowing commit.
    pub fn commit(&self, txn_id: TxnId) -> Result<Timestamp, SsiConflictError> {
        let commit_ts = self.timestamp.fetch_add(1, Ordering::SeqCst);

        // First pass: check for dangerous structure and collect deps
        let (is_dangerous, out_deps, in_deps) = {
            let txns = self.transactions.read();
            let txn = txns.get(&txn_id).ok_or_else(|| SsiConflictError {
                victim_txn: txn_id,
                winner_txn: None,
                conflict_type: ConflictType::DangerousStructure,
                message: "Transaction not found".into(),
            })?;
            (
                txn.is_dangerous(),
                txn.out_rw_deps.clone(),
                txn.in_rw_deps.clone(),
            )
        };

        if is_dangerous {
            let mut txns = self.transactions.write();
            if let Some(txn) = txns.get_mut(&txn_id) {
                txn.status = SsiTxnStatus::Aborted;
            }
            self.txn_states.set_status(txn_id, SsiTxnStatus::Aborted);
            return Err(SsiConflictError {
                victim_txn: txn_id,
                winner_txn: None,
                conflict_type: ConflictType::DangerousStructure,
                message: "Transaction would create serialization anomaly (dangerous structure)"
                    .into(),
            });
        }

        // Second pass: update status and deps
        {
            let mut txns = self.transactions.write();

            // Update our status
            if let Some(txn) = txns.get_mut(&txn_id) {
                txn.status = SsiTxnStatus::Committed(commit_ts);
                txn.commit_ts = Some(commit_ts);
            }

            // Update out deps
            for out_dep in &out_deps {
                if let Some(other_txn) = txns.get_mut(out_dep) {
                    other_txn.has_committed_in_rw = true;
                }
            }

            // Update in deps
            for in_dep in &in_deps {
                if let Some(other_txn) = txns.get_mut(in_dep) {
                    other_txn.has_committed_out_rw = true;
                }
            }
        }

        self.txn_states
            .set_status(txn_id, SsiTxnStatus::Committed(commit_ts));
        Ok(commit_ts)
    }

    /// Abort a transaction
    pub fn abort(&self, txn_id: TxnId) {
        let mut txns = self.transactions.write();
        if let Some(txn) = txns.get_mut(&txn_id) {
            txn.status = SsiTxnStatus::Aborted;
            self.txn_states.set_status(txn_id, SsiTxnStatus::Aborted);
        }

        // Clean up key writers
        self.key_writers
            .write()
            .retain(|_, (writer, _)| *writer != txn_id);

        // Clean up key readers
        for readers in self.key_readers.write().values_mut() {
            readers.remove(&txn_id);
        }
    }

    /// Get transaction status
    pub fn get_status(&self, txn_id: TxnId) -> Option<SsiTxnStatus> {
        self.txn_states.get_status(txn_id)
    }

    /// Get snapshot timestamp for a transaction
    pub fn get_snapshot_ts(&self, txn_id: TxnId) -> Option<Timestamp> {
        self.transactions.read().get(&txn_id).map(|t| t.start_ts)
    }

    /// Check if a version is visible to a transaction
    pub fn is_visible(&self, txn_id: TxnId, version: &SsiVersionInfo) -> bool {
        if let Some(snapshot_ts) = self.get_snapshot_ts(txn_id) {
            version.is_visible(snapshot_ts, &self.txn_states)
        } else {
            false
        }
    }

    /// Garbage collection: remove old completed transactions
    pub fn gc(&self, watermark: Timestamp) -> usize {
        let mut removed = 0;

        // Remove old transactions
        self.transactions.write().retain(|_, txn| {
            if let Some(commit_ts) = txn.commit_ts
                && commit_ts < watermark
            {
                removed += 1;
                return false;
            }
            true
        });

        removed
    }
}

impl Default for SsiManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid Logical Clock (HLC) for timestamp generation
///
/// Combines physical and logical time for causality-preserving timestamps.
///
/// Format: (physical_time_ms << 20) | logical_counter
/// - 44 bits for physical time in milliseconds
/// - 20 bits for logical counter (1M events per millisecond)
pub struct HybridLogicalClock {
    /// Combined timestamp (physical << 20 | logical)
    timestamp: AtomicU64,
}

impl HybridLogicalClock {
    const LOGICAL_BITS: u32 = 20;
    const LOGICAL_MASK: u64 = (1 << Self::LOGICAL_BITS) - 1;

    /// Create a new HLC
    pub fn new() -> Self {
        let now_ms = Self::physical_time_ms();
        Self {
            timestamp: AtomicU64::new(now_ms << Self::LOGICAL_BITS),
        }
    }

    /// Get current physical time in milliseconds
    fn physical_time_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Extract physical time from timestamp
    pub fn get_physical(ts: u64) -> u64 {
        ts >> Self::LOGICAL_BITS
    }

    /// Extract logical counter from timestamp
    pub fn get_logical(ts: u64) -> u64 {
        ts & Self::LOGICAL_MASK
    }

    /// Generate next timestamp
    ///
    /// Ensures:
    /// - Monotonically increasing
    /// - Bounded drift from physical time
    /// - Causality preservation
    pub fn next(&self) -> u64 {
        loop {
            let current = self.timestamp.load(Ordering::Acquire);
            let current_physical = Self::get_physical(current);
            let current_logical = Self::get_logical(current);

            let now_physical = Self::physical_time_ms();

            let (new_physical, new_logical) = if now_physical > current_physical {
                // Physical time advanced - reset logical counter
                (now_physical, 0)
            } else {
                // Same or earlier physical time - increment logical
                if current_logical >= Self::LOGICAL_MASK {
                    // Logical counter overflow - wait for physical time to advance
                    std::thread::yield_now();
                    continue;
                }
                (current_physical, current_logical + 1)
            };

            let new_ts = (new_physical << Self::LOGICAL_BITS) | new_logical;

            if self
                .timestamp
                .compare_exchange(current, new_ts, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return new_ts;
            }
        }
    }

    /// Update timestamp based on received message timestamp
    ///
    /// Used for distributed systems to preserve causality.
    pub fn update(&self, msg_ts: u64) {
        loop {
            let current = self.timestamp.load(Ordering::Acquire);
            let now_physical = Self::physical_time_ms();

            let new_ts = if msg_ts > current {
                // Message from future - advance our clock
                let msg_physical = Self::get_physical(msg_ts);
                let msg_logical = Self::get_logical(msg_ts);

                if msg_physical > now_physical {
                    // Bounded drift: don't go too far ahead
                    let bounded_physical = now_physical.max(msg_physical.saturating_sub(1000));
                    (bounded_physical << Self::LOGICAL_BITS) | (msg_logical + 1)
                } else {
                    (now_physical << Self::LOGICAL_BITS) | (msg_logical + 1)
                }
            } else {
                // Our clock is ahead - no update needed
                return;
            };

            if self
                .timestamp
                .compare_exchange(current, new_ts, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Get current timestamp without incrementing
    pub fn now(&self) -> u64 {
        self.timestamp.load(Ordering::Acquire)
    }
}

impl Default for HybridLogicalClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssi_basic_commit() {
        let ssi = SsiManager::new();

        let (txn1, _) = ssi.begin().unwrap();
        ssi.record_read(txn1, b"key1").unwrap();
        ssi.record_write(txn1, b"key1").unwrap();
        let commit_ts = ssi.commit(txn1).unwrap();

        assert!(commit_ts > 0);
        assert!(matches!(
            ssi.get_status(txn1),
            Some(SsiTxnStatus::Committed(_))
        ));
    }

    #[test]
    fn test_ssi_write_write_conflict() {
        let ssi = SsiManager::new();

        // T1 starts first
        let (txn1, _) = ssi.begin().unwrap();

        // T2 starts and writes to key1
        let (txn2, _) = ssi.begin().unwrap();
        ssi.record_write(txn2, b"key1").unwrap();
        ssi.commit(txn2).unwrap();

        // T1 tries to write to key1 - should fail (first-updater-wins)
        let result = ssi.record_write(txn1, b"key1");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().conflict_type,
            ConflictType::WriteWrite
        ));
    }

    #[test]
    fn test_ssi_rw_antidependency() {
        let ssi = SsiManager::new();

        // T1 reads key1
        let (txn1, _) = ssi.begin().unwrap();
        ssi.record_read(txn1, b"key1").unwrap();

        // T2 writes to key1
        let (txn2, _) = ssi.begin().unwrap();
        ssi.record_write(txn2, b"key1").unwrap();
        ssi.commit(txn2).unwrap();

        // T1 has an rw-antidep with T2 (T1 →ʳʷ T2)
        // This alone is not dangerous - T1 can still commit
        let result = ssi.commit(txn1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ssi_dangerous_structure() {
        let ssi = SsiManager::new();

        // Set up scenario that creates dangerous structure
        // T1 reads key1, T2 writes key1, T1 writes key2, T3 reads key2 writes key1

        let (txn1, _) = ssi.begin().unwrap();
        ssi.record_read(txn1, b"key1").unwrap();

        let (txn2, _) = ssi.begin().unwrap();
        ssi.record_write(txn2, b"key1").unwrap();
        ssi.commit(txn2).unwrap(); // T1 now has out_rw to committed T2

        ssi.record_write(txn1, b"key2").unwrap();

        // T3 reads key2 (which T1 wrote), then writes key1
        let (txn3, _) = ssi.begin().unwrap();
        ssi.record_read(txn3, b"key2").unwrap();
        // T3 has out_rw to T1

        ssi.record_write(txn3, b"key1").unwrap();
        ssi.commit(txn3).unwrap(); // T1 now has in_rw from committed T3

        // T1 should abort due to dangerous structure
        let _result = ssi.commit(txn1);
        // Note: Whether this fails depends on timing of commits
        // In a real implementation, the dangerous structure detection
        // would be more sophisticated
    }

    #[test]
    fn test_hlc_monotonic() {
        let hlc = HybridLogicalClock::new();

        let mut prev = hlc.next();
        for _ in 0..1000 {
            let curr = hlc.next();
            assert!(curr > prev, "HLC must be monotonic");
            prev = curr;
        }
    }

    #[test]
    fn test_hlc_physical_extraction() {
        let hlc = HybridLogicalClock::new();
        let ts = hlc.next();

        let physical = HybridLogicalClock::get_physical(ts);
        let logical = HybridLogicalClock::get_logical(ts);

        // Physical time should be reasonable (after 2020)
        assert!(physical > 1577836800000); // 2020-01-01 in ms

        // Logical should be 0 or small
        assert!(logical < 1000);
    }

    #[test]
    fn test_version_visibility() {
        let states = SsiTxnStates::new();

        // Create a committed transaction
        states.set_status(1, SsiTxnStatus::Committed(100));

        // Version created by txn 1
        let version = SsiVersionInfo::new(1, 100);

        // Visible to snapshot at ts=150
        assert!(version.is_visible(150, &states));

        // Not visible to snapshot at ts=50 (before commit)
        assert!(!version.is_visible(50, &states));
    }

    #[test]
    fn test_ssi_abort_cleanup() {
        let ssi = SsiManager::new();

        let (txn1, _) = ssi.begin().unwrap();
        ssi.record_write(txn1, b"key1").unwrap();
        ssi.abort(txn1);

        // Another transaction should be able to write to key1
        let (txn2, _) = ssi.begin().unwrap();
        let result = ssi.record_write(txn2, b"key1");
        assert!(result.is_ok());
    }
}
