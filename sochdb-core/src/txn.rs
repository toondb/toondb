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

//! Transaction Manager for ACID Transactions
//!
//! Provides ACID guarantees using WAL-based transaction management:
//! - Atomicity: All writes in a transaction succeed or fail together
//! - Consistency: Transactions move database from valid state to valid state
//! - Isolation: MVCC snapshot isolation for concurrent transactions
//! - Durability: Committed transactions survive crashes via WAL

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Transaction ID - monotonically increasing
pub type TxnId = u64;

/// Transaction states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

/// WAL record types for ACID transactions (ARIES-style)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalRecordType {
    /// Data write within transaction
    Data = 0x01,
    /// Transaction begin marker
    TxnBegin = 0x10,
    /// Transaction commit marker
    TxnCommit = 0x11,
    /// Transaction abort marker
    TxnAbort = 0x12,
    /// Checkpoint for recovery optimization
    Checkpoint = 0x20,
    /// Schema change (DDL)
    SchemaChange = 0x30,
    /// Compensation Log Record (CLR) for ARIES undo operations
    CompensationLogRecord = 0x40,
    /// End of checkpoint (contains active transactions and dirty pages)
    CheckpointEnd = 0x21,
    /// Page update with before/after images
    PageUpdate = 0x02,
}

impl TryFrom<u8> for WalRecordType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(WalRecordType::Data),
            0x10 => Ok(WalRecordType::TxnBegin),
            0x11 => Ok(WalRecordType::TxnCommit),
            0x12 => Ok(WalRecordType::TxnAbort),
            0x20 => Ok(WalRecordType::Checkpoint),
            0x21 => Ok(WalRecordType::CheckpointEnd),
            0x30 => Ok(WalRecordType::SchemaChange),
            0x40 => Ok(WalRecordType::CompensationLogRecord),
            0x02 => Ok(WalRecordType::PageUpdate),
            _ => Err(()),
        }
    }
}

/// Log Sequence Number (LSN) for ARIES recovery
///
/// LSN ordering guarantee: If LSN(A) < LSN(B), then A happened before B in the WAL.
/// This is critical for:
/// - Redo: Only redo operations where page_lsn < record_lsn
/// - Undo: Process undo in reverse LSN order
pub type Lsn = u64;

/// Page ID for tracking dirty pages
pub type PageId = u64;

/// ARIES transaction table entry for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AriesTransactionEntry {
    /// Transaction ID
    pub txn_id: TxnId,
    /// Transaction state during recovery
    pub state: TxnState,
    /// LSN of last log record for this transaction
    pub last_lsn: Lsn,
    /// LSN to undo next (for rollback)
    pub undo_next_lsn: Option<Lsn>,
}

/// ARIES dirty page table entry for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AriesDirtyPageEntry {
    /// Page ID
    pub page_id: PageId,
    /// Recovery LSN - first LSN that might have dirtied this page
    pub rec_lsn: Lsn,
}

/// Checkpoint data for ARIES recovery
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AriesCheckpointData {
    /// Active transactions at checkpoint time
    pub active_transactions: Vec<AriesTransactionEntry>,
    /// Dirty pages at checkpoint time
    pub dirty_pages: Vec<AriesDirtyPageEntry>,
    /// LSN where checkpoint started
    pub begin_checkpoint_lsn: Lsn,
}

/// A write operation buffered in a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxnWrite {
    /// Key being written
    pub key: Vec<u8>,
    /// Value being written (None for delete)
    pub value: Option<Vec<u8>>,
    /// Table/collection this write belongs to
    pub table: String,
}

/// A read operation recorded for conflict detection
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TxnRead {
    pub key: Vec<u8>,
    pub table: String,
}

/// WAL entry with ARIES transaction support
///
/// Extends standard WAL entries with ARIES-specific fields:
/// - LSN: Log Sequence Number for ordering and idempotent recovery
/// - prev_lsn: Previous LSN for this transaction (undo chain)
/// - undo_info: Before-image for undo operations
/// - page_id: Page affected by this operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxnWalEntry {
    /// Type of this WAL record
    pub record_type: WalRecordType,
    /// Transaction ID
    pub txn_id: TxnId,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Optional key for data records
    pub key: Option<Vec<u8>>,
    /// Optional value for data records (after-image)
    pub value: Option<Vec<u8>>,
    /// Optional table name
    pub table: Option<String>,
    /// CRC32 checksum
    pub checksum: u32,
    /// ARIES: Log Sequence Number (assigned when appended to WAL)
    #[serde(default)]
    pub lsn: Lsn,
    /// ARIES: Previous LSN in this transaction's chain (for undo)
    #[serde(default)]
    pub prev_lsn: Option<Lsn>,
    /// ARIES: Page ID affected by this record
    #[serde(default)]
    pub page_id: Option<PageId>,
    /// ARIES: Before-image for undo (original value before update)
    #[serde(default)]
    pub undo_info: Option<Vec<u8>>,
    /// ARIES: For CLRs, the next LSN to undo (skips compensated operations)
    #[serde(default)]
    pub undo_next_lsn: Option<Lsn>,
}

impl TxnWalEntry {
    pub fn new_begin(txn_id: TxnId, timestamp_us: u64) -> Self {
        Self {
            record_type: WalRecordType::TxnBegin,
            txn_id,
            timestamp_us,
            key: None,
            value: None,
            table: None,
            checksum: 0,
            lsn: 0,
            prev_lsn: None,
            page_id: None,
            undo_info: None,
            undo_next_lsn: None,
        }
    }

    pub fn new_commit(txn_id: TxnId, timestamp_us: u64) -> Self {
        Self {
            record_type: WalRecordType::TxnCommit,
            txn_id,
            timestamp_us,
            key: None,
            value: None,
            table: None,
            checksum: 0,
            lsn: 0,
            prev_lsn: None,
            page_id: None,
            undo_info: None,
            undo_next_lsn: None,
        }
    }

    pub fn new_abort(txn_id: TxnId, timestamp_us: u64) -> Self {
        Self {
            record_type: WalRecordType::TxnAbort,
            txn_id,
            timestamp_us,
            key: None,
            value: None,
            table: None,
            checksum: 0,
            lsn: 0,
            prev_lsn: None,
            page_id: None,
            undo_info: None,
            undo_next_lsn: None,
        }
    }

    pub fn new_data(
        txn_id: TxnId,
        timestamp_us: u64,
        table: String,
        key: Vec<u8>,
        value: Option<Vec<u8>>,
    ) -> Self {
        Self {
            record_type: WalRecordType::Data,
            txn_id,
            timestamp_us,
            key: Some(key),
            value,
            table: Some(table),
            checksum: 0,
            lsn: 0,
            prev_lsn: None,
            page_id: None,
            undo_info: None,
            undo_next_lsn: None,
        }
    }

    /// Create a new ARIES-style data record with before-image for undo
    #[allow(clippy::too_many_arguments)]
    pub fn new_aries_data(
        txn_id: TxnId,
        timestamp_us: u64,
        table: String,
        key: Vec<u8>,
        value: Option<Vec<u8>>,
        page_id: PageId,
        prev_lsn: Option<Lsn>,
        undo_info: Option<Vec<u8>>,
    ) -> Self {
        Self {
            record_type: WalRecordType::Data,
            txn_id,
            timestamp_us,
            key: Some(key),
            value,
            table: Some(table),
            checksum: 0,
            lsn: 0, // Assigned when appended to WAL
            prev_lsn,
            page_id: Some(page_id),
            undo_info,
            undo_next_lsn: None,
        }
    }

    /// Create a Compensation Log Record (CLR) for ARIES undo
    ///
    /// CLRs are redo-only records that describe undo operations.
    /// They include undo_next_lsn which points to the next record to undo,
    /// skipping the compensated operation.
    #[allow(clippy::too_many_arguments)]
    pub fn new_clr(
        txn_id: TxnId,
        timestamp_us: u64,
        table: String,
        key: Vec<u8>,
        value: Option<Vec<u8>>,
        page_id: PageId,
        prev_lsn: Lsn,
        undo_next_lsn: Lsn,
    ) -> Self {
        Self {
            record_type: WalRecordType::CompensationLogRecord,
            txn_id,
            timestamp_us,
            key: Some(key),
            value,
            table: Some(table),
            checksum: 0,
            lsn: 0,
            prev_lsn: Some(prev_lsn),
            page_id: Some(page_id),
            undo_info: None, // CLRs don't need undo info (redo-only)
            undo_next_lsn: Some(undo_next_lsn),
        }
    }

    /// Create a checkpoint end record with recovery data
    pub fn new_checkpoint_end(
        timestamp_us: u64,
        checkpoint_data: AriesCheckpointData,
    ) -> Result<Self, String> {
        let data = bincode::serialize(&checkpoint_data)
            .map_err(|e| format!("Failed to serialize checkpoint data: {}", e))?;
        Ok(Self {
            record_type: WalRecordType::CheckpointEnd,
            txn_id: 0,
            timestamp_us,
            key: None,
            value: Some(data),
            table: None,
            checksum: 0,
            lsn: 0,
            prev_lsn: None,
            page_id: None,
            undo_info: None,
            undo_next_lsn: None,
        })
    }

    /// Extract checkpoint data from a CheckpointEnd record
    pub fn get_checkpoint_data(&self) -> Option<AriesCheckpointData> {
        if self.record_type != WalRecordType::CheckpointEnd {
            return None;
        }
        self.value
            .as_ref()
            .and_then(|data| bincode::deserialize(data).ok())
    }

    /// Calculate and set checksum
    pub fn compute_checksum(&mut self) {
        let data = self.serialize_for_checksum();
        self.checksum = crc32fast::hash(&data);
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        let data = self.serialize_for_checksum();
        crc32fast::hash(&data) == self.checksum
    }

    fn serialize_for_checksum(&self) -> Vec<u8> {
        // Serialize without checksum field
        let mut buf = Vec::new();
        buf.push(self.record_type as u8);
        buf.extend(&self.txn_id.to_le_bytes());
        buf.extend(&self.timestamp_us.to_le_bytes());
        if let Some(ref key) = self.key {
            buf.extend(&(key.len() as u32).to_le_bytes());
            buf.extend(key);
        } else {
            buf.extend(&0u32.to_le_bytes());
        }
        if let Some(ref value) = self.value {
            buf.extend(&(value.len() as u32).to_le_bytes());
            buf.extend(value);
        } else {
            buf.extend(&0u32.to_le_bytes());
        }
        if let Some(ref table) = self.table {
            buf.extend(&(table.len() as u32).to_le_bytes());
            buf.extend(table.as_bytes());
        } else {
            buf.extend(&0u32.to_le_bytes());
        }
        buf
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.serialize_for_checksum();
        buf.extend(&self.checksum.to_le_bytes());
        buf
    }

    /// Deserialize from bytes with proper error propagation
    ///
    /// Returns an error if:
    /// - Data is too short (minimum 21 bytes)
    /// - Record type is invalid
    /// - Data is truncated mid-field
    /// - UTF-8 encoding is invalid for table name
    /// - Checksum validation fails
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        // Fixed header: 1 (type) + 8 (txn_id) + 8 (timestamp) + 4 (checksum minimum) = 21
        if data.len() < 21 {
            return Err(format!(
                "WAL entry too short: {} bytes, need at least 21",
                data.len()
            ));
        }

        let record_type = WalRecordType::try_from(data[0])
            .map_err(|_| format!("Invalid WAL record type: {}", data[0]))?;

        let txn_id = u64::from_le_bytes(
            data[1..9]
                .try_into()
                .map_err(|_| "Failed to parse txn_id: slice too short")?,
        );
        let timestamp_us = u64::from_le_bytes(
            data[9..17]
                .try_into()
                .map_err(|_| "Failed to parse timestamp: slice too short")?,
        );

        let mut offset = 17;

        // Parse key with bounds checking
        if offset + 4 > data.len() {
            return Err(format!(
                "WAL entry truncated at key_len: offset {} + 4 > {}",
                offset,
                data.len()
            ));
        }
        let key_len = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "Failed to parse key_len")?,
        ) as usize;
        offset += 4;

        if offset + key_len > data.len() {
            return Err(format!(
                "WAL entry truncated at key: need {} bytes at offset {}, have {}",
                key_len,
                offset,
                data.len()
            ));
        }
        let key = if key_len > 0 {
            Some(data[offset..offset + key_len].to_vec())
        } else {
            None
        };
        offset += key_len;

        // Parse value with bounds checking
        if offset + 4 > data.len() {
            return Err(format!(
                "WAL entry truncated at value_len: offset {} + 4 > {}",
                offset,
                data.len()
            ));
        }
        let value_len = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "Failed to parse value_len")?,
        ) as usize;
        offset += 4;

        if offset + value_len > data.len() {
            return Err(format!(
                "WAL entry truncated at value: need {} bytes at offset {}, have {}",
                value_len,
                offset,
                data.len()
            ));
        }
        let value = if value_len > 0 {
            Some(data[offset..offset + value_len].to_vec())
        } else {
            None
        };
        offset += value_len;

        // Parse table name with bounds checking
        if offset + 4 > data.len() {
            return Err(format!(
                "WAL entry truncated at table_len: offset {} + 4 > {}",
                offset,
                data.len()
            ));
        }
        let table_len = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "Failed to parse table_len")?,
        ) as usize;
        offset += 4;

        if offset + table_len > data.len() {
            return Err(format!(
                "WAL entry truncated at table: need {} bytes at offset {}, have {}",
                table_len,
                offset,
                data.len()
            ));
        }
        let table = if table_len > 0 {
            Some(
                String::from_utf8(data[offset..offset + table_len].to_vec())
                    .map_err(|e| format!("Invalid UTF-8 in table name: {}", e))?,
            )
        } else {
            None
        };
        offset += table_len;

        // Parse checksum with bounds checking
        if offset + 4 > data.len() {
            return Err(format!(
                "WAL entry truncated at checksum: offset {} + 4 > {}",
                offset,
                data.len()
            ));
        }
        let checksum = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "Failed to parse checksum")?,
        );

        let entry = Self {
            record_type,
            txn_id,
            timestamp_us,
            key,
            value,
            table,
            checksum,
            // ARIES fields default to zero/None for backward compatibility
            lsn: 0,
            prev_lsn: None,
            page_id: None,
            undo_info: None,
            undo_next_lsn: None,
        };

        // Verify checksum to detect corruption
        if !entry.verify_checksum() {
            return Err(format!(
                "WAL entry checksum mismatch for txn_id {}: expected valid checksum, got {}",
                entry.txn_id, entry.checksum
            ));
        }

        Ok(entry)
    }
}

/// Transaction handle for the user
#[derive(Debug)]
pub struct Transaction {
    /// Unique transaction ID
    pub id: TxnId,
    /// Transaction state
    pub state: TxnState,
    /// Start timestamp for MVCC
    pub start_ts: u64,
    /// Commit timestamp (set on commit)
    pub commit_ts: Option<u64>,
    /// Buffered writes
    pub writes: Vec<TxnWrite>,
    /// Read set for conflict detection
    pub read_set: HashSet<TxnRead>,
    /// Isolation level
    pub isolation: IsolationLevel,
}

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Read committed - see committed changes
    ReadCommitted,
    /// Snapshot isolation - consistent point-in-time view
    #[default]
    SnapshotIsolation,
    /// Serializable - strongest isolation
    Serializable,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(id: TxnId, start_ts: u64, isolation: IsolationLevel) -> Self {
        Self {
            id,
            state: TxnState::Active,
            start_ts,
            commit_ts: None,
            writes: Vec::new(),
            read_set: HashSet::new(),
            isolation,
        }
    }

    /// Buffer a write operation
    pub fn put(&mut self, table: &str, key: Vec<u8>, value: Vec<u8>) {
        self.writes.push(TxnWrite {
            key,
            value: Some(value),
            table: table.to_string(),
        });
    }

    /// Buffer a delete operation
    pub fn delete(&mut self, table: &str, key: Vec<u8>) {
        self.writes.push(TxnWrite {
            key,
            value: None,
            table: table.to_string(),
        });
    }

    /// Record a read for conflict detection
    pub fn record_read(&mut self, table: &str, key: Vec<u8>) {
        self.read_set.insert(TxnRead {
            key,
            table: table.to_string(),
        });
    }

    /// Check for read-your-writes
    pub fn get_local(&self, table: &str, key: &[u8]) -> Option<&TxnWrite> {
        self.writes
            .iter()
            .rev()
            .find(|w| w.table == table && w.key == key)
    }

    /// Check if transaction has any writes
    pub fn is_read_only(&self) -> bool {
        self.writes.is_empty()
    }
}

/// Transaction Manager stats
#[derive(Debug, Clone, Default)]
pub struct TxnStats {
    pub active_count: u64,
    pub committed_count: u64,
    pub aborted_count: u64,
    pub conflict_aborts: u64,
}

/// Transaction Manager (in-memory, no WAL durability)
///
/// Manages transaction lifecycle and provides ACID guarantees for in-memory
/// operations. This implementation does NOT include WAL integration.
/// 
/// For production workloads requiring durability, use [`sochdb_storage::MvccTransactionManager`]
/// which includes:
/// - Write-ahead logging for crash recovery  
/// - Serializable Snapshot Isolation (SSI)
/// - Group commit for high throughput
/// - Event-driven async architecture
pub struct TransactionManager {
    /// Next transaction ID
    next_txn_id: AtomicU64,
    /// Current timestamp counter
    timestamp_counter: AtomicU64,
    /// Committed transaction watermark
    committed_watermark: AtomicU64,
    /// Statistics
    stats: parking_lot::RwLock<TxnStats>,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            next_txn_id: AtomicU64::new(1),
            timestamp_counter: AtomicU64::new(1),
            committed_watermark: AtomicU64::new(0),
            stats: parking_lot::RwLock::new(TxnStats::default()),
        }
    }

    /// Begin a new transaction
    pub fn begin(&self) -> Transaction {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begin a transaction with specific isolation level
    pub fn begin_with_isolation(&self, isolation: IsolationLevel) -> Transaction {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let start_ts = self.timestamp_counter.fetch_add(1, Ordering::SeqCst);

        {
            let mut stats = self.stats.write();
            stats.active_count += 1;
        }

        Transaction::new(txn_id, start_ts, isolation)
    }

    /// Get commit timestamp
    pub fn get_commit_ts(&self) -> u64 {
        self.timestamp_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Mark transaction as committed
    pub fn mark_committed(&self, txn: &mut Transaction) {
        txn.state = TxnState::Committed;
        txn.commit_ts = Some(self.get_commit_ts());

        let mut stats = self.stats.write();
        stats.active_count = stats.active_count.saturating_sub(1);
        stats.committed_count += 1;
    }

    /// Mark transaction as aborted
    pub fn mark_aborted(&self, txn: &mut Transaction) {
        txn.state = TxnState::Aborted;

        let mut stats = self.stats.write();
        stats.active_count = stats.active_count.saturating_sub(1);
        stats.aborted_count += 1;
    }

    /// Mark transaction as aborted due to conflict
    pub fn mark_conflict_abort(&self, txn: &mut Transaction) {
        self.mark_aborted(txn);

        let mut stats = self.stats.write();
        stats.conflict_aborts += 1;
    }

    /// Get the oldest active transaction timestamp
    pub fn oldest_active_ts(&self) -> u64 {
        self.committed_watermark.load(Ordering::SeqCst)
    }

    /// Update the committed watermark
    pub fn advance_watermark(&self, new_watermark: u64) {
        self.committed_watermark
            .fetch_max(new_watermark, Ordering::SeqCst);
    }

    /// Get current stats
    pub fn stats(&self) -> TxnStats {
        self.stats.read().clone()
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let mgr = TransactionManager::new();

        let mut txn = mgr.begin();
        assert_eq!(txn.state, TxnState::Active);
        assert!(txn.is_read_only());

        txn.put("users", vec![1], vec![2, 3, 4]);
        assert!(!txn.is_read_only());

        mgr.mark_committed(&mut txn);
        assert_eq!(txn.state, TxnState::Committed);
        assert!(txn.commit_ts.is_some());
    }

    #[test]
    fn test_read_your_writes() {
        let mgr = TransactionManager::new();
        let mut txn = mgr.begin();

        txn.put("users", vec![1], vec![10, 20]);
        txn.put("users", vec![1], vec![30, 40]); // Overwrite

        let local = txn.get_local("users", &[1]);
        assert!(local.is_some());
        assert_eq!(local.unwrap().value, Some(vec![30, 40]));
    }

    #[test]
    fn test_wal_entry_serialization() {
        let mut entry = TxnWalEntry::new_data(
            42,
            1234567890,
            "users".to_string(),
            vec![1, 2, 3],
            Some(vec![4, 5, 6]),
        );
        entry.compute_checksum();

        let bytes = entry.to_bytes();
        let parsed = TxnWalEntry::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.txn_id, 42);
        assert_eq!(parsed.timestamp_us, 1234567890);
        assert_eq!(parsed.table, Some("users".to_string()));
        assert_eq!(parsed.key, Some(vec![1, 2, 3]));
        assert_eq!(parsed.value, Some(vec![4, 5, 6]));
        assert!(parsed.verify_checksum());
    }

    #[test]
    fn test_transaction_stats() {
        let mgr = TransactionManager::new();

        let mut txn1 = mgr.begin();
        let mut txn2 = mgr.begin();

        assert_eq!(mgr.stats().active_count, 2);

        mgr.mark_committed(&mut txn1);
        assert_eq!(mgr.stats().committed_count, 1);

        mgr.mark_aborted(&mut txn2);
        assert_eq!(mgr.stats().aborted_count, 1);
        assert_eq!(mgr.stats().active_count, 0);
    }

    #[test]
    fn test_wal_entry_error_too_short() {
        // Less than minimum 21 bytes
        let short_data = vec![0u8; 10];
        let result = TxnWalEntry::from_bytes(&short_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_wal_entry_error_invalid_record_type() {
        // Create data with invalid record type (255)
        let mut data = vec![0u8; 30];
        data[0] = 255; // Invalid record type
        let result = TxnWalEntry::from_bytes(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid WAL record type"));
    }

    #[test]
    fn test_wal_entry_error_truncated_key() {
        // Create entry claiming 1000 byte key but data too short
        let mut entry =
            TxnWalEntry::new_data(1, 100, "test".to_string(), vec![1, 2], Some(vec![3, 4]));
        entry.compute_checksum();
        let mut bytes = entry.to_bytes();

        // Corrupt key_len to claim huge key
        let huge_len: u32 = 10000;
        bytes[17..21].copy_from_slice(&huge_len.to_le_bytes());

        let result = TxnWalEntry::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("truncated at key"));
    }

    #[test]
    fn test_wal_entry_error_corrupted_checksum() {
        let mut entry = TxnWalEntry::new_data(
            42,
            1234567890,
            "users".to_string(),
            vec![1, 2, 3],
            Some(vec![4, 5, 6]),
        );
        entry.compute_checksum();

        let mut bytes = entry.to_bytes();
        // Corrupt the checksum (last 4 bytes)
        let len = bytes.len();
        bytes[len - 1] ^= 0xFF; // Flip bits

        let result = TxnWalEntry::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("checksum mismatch"));
    }

    #[test]
    fn test_wal_entry_error_invalid_utf8_table() {
        let mut entry = TxnWalEntry::new_data(1, 100, "test".to_string(), vec![1], Some(vec![2]));
        entry.compute_checksum();
        let mut bytes = entry.to_bytes();

        // Find table offset and corrupt UTF-8
        // Header: 1 + 8 + 8 = 17, key_len: 4, key: 1, value_len: 4, value: 1, table_len: 4, table: 4
        let table_start = 17 + 4 + 1 + 4 + 1 + 4;
        bytes[table_start] = 0xFF; // Invalid UTF-8 byte

        let result = TxnWalEntry::from_bytes(&bytes);
        // Either checksum or UTF-8 error
        assert!(result.is_err());
    }
}
