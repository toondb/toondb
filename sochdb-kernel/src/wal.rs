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

//! Write-Ahead Logging (WAL)
//!
//! Minimal WAL implementation for the kernel.
//! Provides durability guarantees for transactions.

use crate::error::{KernelError, KernelResult, WalErrorKind};
use crate::kernel_api::PageId;
use crate::transaction::TransactionId;
use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Log Sequence Number - unique identifier for WAL records
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LogSequenceNumber(pub u64);

impl LogSequenceNumber {
    /// Invalid/null LSN (max value as sentinel)
    pub const INVALID: Self = Self(u64::MAX);

    /// Create a new LSN
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the raw value
    pub fn value(&self) -> u64 {
        self.0
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.0 != u64::MAX
    }
}

impl std::fmt::Display for LogSequenceNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LSN({})", self.0)
    }
}

/// WAL record types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalRecordType {
    /// Transaction begin
    Begin = 1,
    /// Transaction commit
    Commit = 2,
    /// Transaction abort
    Abort = 3,
    /// Data update (with undo info)
    Update = 4,
    /// Data insert
    Insert = 5,
    /// Data delete
    Delete = 6,
    /// Compensation log record (for rollback)
    Clr = 7,
    /// Checkpoint begin
    CheckpointBegin = 8,
    /// Checkpoint end
    CheckpointEnd = 9,
    /// Page allocation
    AllocPage = 10,
    /// Page deallocation
    FreePage = 11,
}

impl TryFrom<u8> for WalRecordType {
    type Error = KernelError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Begin),
            2 => Ok(Self::Commit),
            3 => Ok(Self::Abort),
            4 => Ok(Self::Update),
            5 => Ok(Self::Insert),
            6 => Ok(Self::Delete),
            7 => Ok(Self::Clr),
            8 => Ok(Self::CheckpointBegin),
            9 => Ok(Self::CheckpointEnd),
            10 => Ok(Self::AllocPage),
            11 => Ok(Self::FreePage),
            _ => Err(KernelError::Wal {
                kind: WalErrorKind::Corrupted,
            }),
        }
    }
}

/// WAL record
#[derive(Debug, Clone)]
pub struct WalRecord {
    /// Log sequence number
    pub lsn: LogSequenceNumber,
    /// Previous LSN for this transaction (for undo chain)
    pub prev_lsn: LogSequenceNumber,
    /// Transaction ID
    pub txn_id: TransactionId,
    /// Record type
    pub record_type: WalRecordType,
    /// Page ID (for page-level operations)
    pub page_id: Option<PageId>,
    /// Redo data
    pub redo_data: Bytes,
    /// Undo data (for compensation)
    pub undo_data: Bytes,
    /// Checksum
    pub checksum: u32,
}

impl WalRecord {
    /// Record header size: lsn(8) + prev_lsn(8) + txn_id(8) + type(1) + page_id(8) + redo_len(4) + undo_len(4) + checksum(4)
    const HEADER_SIZE: usize = 45;

    /// Create a new WAL record
    pub fn new(
        lsn: LogSequenceNumber,
        prev_lsn: LogSequenceNumber,
        txn_id: TransactionId,
        record_type: WalRecordType,
        page_id: Option<PageId>,
        redo_data: Bytes,
        undo_data: Bytes,
    ) -> Self {
        let mut record = Self {
            lsn,
            prev_lsn,
            txn_id,
            record_type,
            page_id,
            redo_data,
            undo_data,
            checksum: 0,
        };
        record.checksum = record.compute_checksum();
        record
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(
            Self::HEADER_SIZE + self.redo_data.len() + self.undo_data.len(),
        );

        buf.put_u64_le(self.lsn.0);
        buf.put_u64_le(self.prev_lsn.0);
        buf.put_u64_le(self.txn_id);
        buf.put_u8(self.record_type as u8);
        buf.put_u64_le(self.page_id.unwrap_or(0));
        buf.put_u32_le(self.redo_data.len() as u32);
        buf.put_u32_le(self.undo_data.len() as u32);
        buf.put_slice(&self.redo_data);
        buf.put_slice(&self.undo_data);
        buf.put_u32_le(self.checksum);

        buf.freeze()
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> KernelResult<Self> {
        if data.len() < Self::HEADER_SIZE {
            return Err(KernelError::Wal {
                kind: WalErrorKind::Corrupted,
            });
        }

        let lsn = LogSequenceNumber(u64::from_le_bytes(data[0..8].try_into().unwrap()));
        let prev_lsn = LogSequenceNumber(u64::from_le_bytes(data[8..16].try_into().unwrap()));
        let txn_id = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let record_type = WalRecordType::try_from(data[24])?;
        let page_id_raw = u64::from_le_bytes(data[25..33].try_into().unwrap());
        let page_id = if page_id_raw == 0 {
            None
        } else {
            Some(page_id_raw)
        };
        let redo_len = u32::from_le_bytes(data[33..37].try_into().unwrap()) as usize;
        let undo_len = u32::from_le_bytes(data[37..41].try_into().unwrap()) as usize;

        let expected_len = Self::HEADER_SIZE + redo_len + undo_len;
        if data.len() < expected_len {
            return Err(KernelError::Wal {
                kind: WalErrorKind::Corrupted,
            });
        }

        let redo_start = 41;
        let redo_data = Bytes::copy_from_slice(&data[redo_start..redo_start + redo_len]);
        let undo_start = redo_start + redo_len;
        let undo_data = Bytes::copy_from_slice(&data[undo_start..undo_start + undo_len]);
        let checksum_start = undo_start + undo_len;
        let checksum =
            u32::from_le_bytes(data[checksum_start..checksum_start + 4].try_into().unwrap());

        let record = Self {
            lsn,
            prev_lsn,
            txn_id,
            record_type,
            page_id,
            redo_data,
            undo_data,
            checksum,
        };

        // Verify checksum
        let computed = record.compute_checksum();
        if computed != checksum {
            return Err(KernelError::Wal {
                kind: WalErrorKind::ChecksumMismatch {
                    expected: checksum,
                    actual: computed,
                },
            });
        }

        Ok(record)
    }

    /// Compute checksum for the record
    fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.lsn.0.to_le_bytes());
        hasher.update(&self.prev_lsn.0.to_le_bytes());
        hasher.update(&self.txn_id.to_le_bytes());
        hasher.update(&[self.record_type as u8]);
        hasher.update(&self.page_id.unwrap_or(0).to_le_bytes());
        hasher.update(&self.redo_data);
        hasher.update(&self.undo_data);
        hasher.finalize()
    }

    /// Get serialized size
    pub fn size(&self) -> usize {
        Self::HEADER_SIZE + self.redo_data.len() + self.undo_data.len()
    }
}

/// WAL Manager
///
/// Manages write-ahead log for durability.
pub struct WalManager {
    /// WAL file path
    path: PathBuf,
    /// WAL file handle
    file: Mutex<File>,
    /// Next LSN to allocate
    next_lsn: AtomicU64,
    /// Durable LSN (everything up to this is fsynced)
    durable_lsn: AtomicU64,
    /// Per-transaction last LSN (for undo chain)
    txn_last_lsn: RwLock<HashMap<TransactionId, LogSequenceNumber>>,
    /// Last checkpoint LSN
    checkpoint_lsn: AtomicU64,
    /// Buffer for batching writes
    write_buffer: Mutex<BytesMut>,
    /// Buffer threshold for auto-flush (bytes)
    buffer_threshold: usize,
}

impl WalManager {
    /// Default buffer threshold: 64KB
    const DEFAULT_BUFFER_THRESHOLD: usize = 64 * 1024;

    /// Open or create a WAL file
    pub fn open(path: impl AsRef<Path>) -> KernelResult<Self> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let file_len = file.metadata()?.len();
        // LSN starts at 0 for empty file, otherwise at end of file for new writes
        let next_lsn = file_len;

        Ok(Self {
            path,
            file: Mutex::new(file),
            next_lsn: AtomicU64::new(next_lsn),
            durable_lsn: AtomicU64::new(if file_len > 0 { file_len } else { 0 }),
            txn_last_lsn: RwLock::new(HashMap::new()),
            checkpoint_lsn: AtomicU64::new(0),
            write_buffer: Mutex::new(BytesMut::with_capacity(Self::DEFAULT_BUFFER_THRESHOLD)),
            buffer_threshold: Self::DEFAULT_BUFFER_THRESHOLD,
        })
    }

    /// Append a record to the WAL
    ///
    /// Returns the LSN of the appended record.
    pub fn append(&self, record: &mut WalRecord) -> KernelResult<LogSequenceNumber> {
        // Allocate LSN
        let lsn = LogSequenceNumber(
            self.next_lsn
                .fetch_add(record.size() as u64, Ordering::SeqCst),
        );
        record.lsn = lsn;

        // Set prev_lsn from transaction's last LSN
        if let Some(&prev) = self.txn_last_lsn.read().get(&record.txn_id) {
            record.prev_lsn = prev;
        }

        // Update checksum with final LSN
        record.checksum = record.compute_checksum();

        // Serialize
        let data = record.serialize();

        // Buffer the write
        let mut buffer = self.write_buffer.lock();
        buffer.extend_from_slice(&data);

        // Update transaction's last LSN
        self.txn_last_lsn.write().insert(record.txn_id, lsn);

        // Auto-flush if buffer exceeds threshold
        if buffer.len() >= self.buffer_threshold {
            drop(buffer);
            self.flush()?;
        }

        Ok(lsn)
    }

    /// Flush buffered writes to disk
    pub fn flush(&self) -> KernelResult<()> {
        let mut buffer = self.write_buffer.lock();
        if buffer.is_empty() {
            return Ok(());
        }

        let data = buffer.split().freeze();
        let mut file = self.file.lock();

        // Seek to end and write
        file.seek(SeekFrom::End(0))?;
        file.write_all(&data)?;

        Ok(())
    }

    /// Sync WAL to durable storage (fsync)
    pub fn sync(&self) -> KernelResult<LogSequenceNumber> {
        // First flush any buffered writes
        self.flush()?;

        // Then fsync
        let file = self.file.lock();
        file.sync_all()?;

        // Update durable LSN
        let current_lsn = self.next_lsn.load(Ordering::SeqCst);
        self.durable_lsn.store(current_lsn, Ordering::SeqCst);

        Ok(LogSequenceNumber(current_lsn))
    }

    /// Get the current durable LSN
    pub fn durable_lsn(&self) -> LogSequenceNumber {
        LogSequenceNumber(self.durable_lsn.load(Ordering::SeqCst))
    }

    /// Get the next LSN that will be allocated
    pub fn next_lsn(&self) -> LogSequenceNumber {
        LogSequenceNumber(self.next_lsn.load(Ordering::SeqCst))
    }

    /// Log a transaction begin
    pub fn log_begin(&self, txn_id: TransactionId) -> KernelResult<LogSequenceNumber> {
        let mut record = WalRecord::new(
            LogSequenceNumber::INVALID,
            LogSequenceNumber::INVALID,
            txn_id,
            WalRecordType::Begin,
            None,
            Bytes::new(),
            Bytes::new(),
        );
        self.append(&mut record)
    }

    /// Log a transaction commit
    pub fn log_commit(&self, txn_id: TransactionId) -> KernelResult<LogSequenceNumber> {
        let prev_lsn = self
            .txn_last_lsn
            .read()
            .get(&txn_id)
            .copied()
            .unwrap_or(LogSequenceNumber::INVALID);
        let mut record = WalRecord::new(
            LogSequenceNumber::INVALID,
            prev_lsn,
            txn_id,
            WalRecordType::Commit,
            None,
            Bytes::new(),
            Bytes::new(),
        );
        let lsn = self.append(&mut record)?;

        // Sync on commit for durability
        self.sync()?;

        // Clean up transaction state
        self.txn_last_lsn.write().remove(&txn_id);

        Ok(lsn)
    }

    /// Log a transaction abort
    pub fn log_abort(&self, txn_id: TransactionId) -> KernelResult<LogSequenceNumber> {
        let prev_lsn = self
            .txn_last_lsn
            .read()
            .get(&txn_id)
            .copied()
            .unwrap_or(LogSequenceNumber::INVALID);
        let mut record = WalRecord::new(
            LogSequenceNumber::INVALID,
            prev_lsn,
            txn_id,
            WalRecordType::Abort,
            None,
            Bytes::new(),
            Bytes::new(),
        );
        let lsn = self.append(&mut record)?;

        // Clean up transaction state
        self.txn_last_lsn.write().remove(&txn_id);

        Ok(lsn)
    }

    /// Log an update operation
    pub fn log_update(
        &self,
        txn_id: TransactionId,
        page_id: PageId,
        redo_data: Bytes,
        undo_data: Bytes,
    ) -> KernelResult<LogSequenceNumber> {
        let prev_lsn = self
            .txn_last_lsn
            .read()
            .get(&txn_id)
            .copied()
            .unwrap_or(LogSequenceNumber::INVALID);
        let mut record = WalRecord::new(
            LogSequenceNumber::INVALID,
            prev_lsn,
            txn_id,
            WalRecordType::Update,
            Some(page_id),
            redo_data,
            undo_data,
        );
        self.append(&mut record)
    }

    /// Log a checkpoint begin
    pub fn log_checkpoint_begin(&self) -> KernelResult<LogSequenceNumber> {
        let mut record = WalRecord::new(
            LogSequenceNumber::INVALID,
            LogSequenceNumber::INVALID,
            0, // System transaction
            WalRecordType::CheckpointBegin,
            None,
            Bytes::new(),
            Bytes::new(),
        );
        self.append(&mut record)
    }

    /// Log a checkpoint end with active transactions
    pub fn log_checkpoint_end(
        &self,
        active_txns: &[TransactionId],
    ) -> KernelResult<LogSequenceNumber> {
        // Serialize active transaction list
        let mut redo_data = BytesMut::with_capacity(active_txns.len() * 8);
        for &txn_id in active_txns {
            redo_data.put_u64_le(txn_id);
        }

        let mut record = WalRecord::new(
            LogSequenceNumber::INVALID,
            LogSequenceNumber::INVALID,
            0, // System transaction
            WalRecordType::CheckpointEnd,
            None,
            redo_data.freeze(),
            Bytes::new(),
        );
        let lsn = self.append(&mut record)?;

        // Sync checkpoint
        self.sync()?;

        // Update checkpoint LSN
        self.checkpoint_lsn.store(lsn.0, Ordering::SeqCst);

        Ok(lsn)
    }

    /// Get last checkpoint LSN
    pub fn checkpoint_lsn(&self) -> Option<LogSequenceNumber> {
        let lsn = self.checkpoint_lsn.load(Ordering::SeqCst);
        if lsn == 0 {
            None
        } else {
            Some(LogSequenceNumber(lsn))
        }
    }

    /// Read all records from a given LSN
    pub fn read_from(&self, start_lsn: LogSequenceNumber) -> KernelResult<Vec<WalRecord>> {
        // Flush any pending writes first
        self.flush()?;

        let mut file = self.file.lock();
        let file_len = file.metadata()?.len();

        if start_lsn.0 >= file_len {
            return Ok(Vec::new());
        }

        file.seek(SeekFrom::Start(start_lsn.0))?;

        let mut buffer = vec![0u8; (file_len - start_lsn.0) as usize];
        file.read_exact(&mut buffer)?;

        let mut records = Vec::new();
        let mut offset = 0;

        while offset < buffer.len() {
            match WalRecord::deserialize(&buffer[offset..]) {
                Ok(record) => {
                    let size = record.size();
                    records.push(record);
                    offset += size;
                }
                Err(_) => {
                    // End of valid records (possibly torn write)
                    break;
                }
            }
        }

        Ok(records)
    }

    /// Get WAL file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Truncate WAL up to a given LSN (for space reclamation after checkpoint)
    pub fn truncate_before(&self, _lsn: LogSequenceNumber) -> KernelResult<()> {
        // In production, this would copy records after LSN to a new file
        // and rename. For simplicity, we skip this.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_record_serialize_deserialize() {
        let record = WalRecord::new(
            LogSequenceNumber(100),
            LogSequenceNumber(50),
            1,
            WalRecordType::Update,
            Some(42),
            Bytes::from_static(b"redo data"),
            Bytes::from_static(b"undo data"),
        );

        let serialized = record.serialize();
        let deserialized = WalRecord::deserialize(&serialized).unwrap();

        assert_eq!(record.lsn, deserialized.lsn);
        assert_eq!(record.prev_lsn, deserialized.prev_lsn);
        assert_eq!(record.txn_id, deserialized.txn_id);
        assert_eq!(record.record_type, deserialized.record_type);
        assert_eq!(record.page_id, deserialized.page_id);
        assert_eq!(record.redo_data, deserialized.redo_data);
        assert_eq!(record.undo_data, deserialized.undo_data);
    }

    #[test]
    fn test_wal_manager_append_sync() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let wal = WalManager::open(&wal_path).unwrap();

        // Log begin
        let lsn1 = wal.log_begin(1).unwrap();
        assert!(lsn1.is_valid());

        // Log update
        let lsn2 = wal
            .log_update(
                1,
                100,
                Bytes::from_static(b"new value"),
                Bytes::from_static(b"old value"),
            )
            .unwrap();
        assert!(lsn2 > lsn1);

        // Sync
        let durable = wal.sync().unwrap();
        assert!(durable >= lsn2);
    }

    #[test]
    fn test_wal_recovery() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write some records
        let first_lsn = {
            let wal = WalManager::open(&wal_path).unwrap();
            let lsn = wal.log_begin(1).unwrap();
            wal.log_update(1, 100, Bytes::from_static(b"data"), Bytes::new())
                .unwrap();
            wal.log_commit(1).unwrap();
            lsn
        };

        // Reopen and read
        {
            let wal = WalManager::open(&wal_path).unwrap();
            let records = wal.read_from(first_lsn).unwrap();

            assert!(
                records.len() >= 3,
                "Expected at least 3 records, got {}",
                records.len()
            );
            assert_eq!(records[0].record_type, WalRecordType::Begin);
            assert_eq!(records[1].record_type, WalRecordType::Update);
            assert_eq!(records[2].record_type, WalRecordType::Commit);
        }
    }
}
