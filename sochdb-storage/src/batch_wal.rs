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

//! Batched WAL with Vectored I/O
//!
//! This module implements batched WAL writing using vectored I/O (writev)
//! to minimize syscall overhead and improve write throughput.
//!
//! ## Problem Analysis
//!
//! Current WAL writes one entry per syscall:
//! - Syscall overhead: ~200-400 cycles
//! - Context switch potential: ~1000 cycles
//! - For 100K rows × 4 cols: 400K syscalls
//!
//! ## Solution
//!
//! Batch WAL entries into single vectored write:
//! - Single syscall per batch (up to 1000 entries)
//! - Vectored I/O (writev) eliminates intermediate copies
//!
//! ## Math
//!
//! Syscall amortization for N=100K entries, k=4 columns, B=1000 batch size:
//!
//! T_unbatched = N × k × S = 100K × 4 × 300 = 120M cycles
//! T_batched = ⌈N × k / B⌉ × S = ⌈400K / 1000⌉ × 300 = 120K cycles
//!
//! **Speedup: 1000× (syscall portion only)**
//!
//! ## Performance
//!
//! Expected throughput: 10-20× improvement for bulk inserts

use std::fs::{File, OpenOptions};
use std::io::{self, IoSlice, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::txn_wal::TxnWalEntry;
use parking_lot::Mutex;
use sochdb_core::{Result, SochDBError};

/// Batch header format:
/// [magic: u32][version: u16][entry_count: u16][total_bytes: u32][checksum: u32]
const BATCH_HEADER_SIZE: usize = 16;
const BATCH_MAGIC: u32 = 0x42415443; // "BATC"
const BATCH_VERSION: u16 = 1;

/// Default maximum batch size (number of entries)
pub const DEFAULT_MAX_BATCH_SIZE: usize = 1000;

/// Default maximum batch bytes (64KB)
pub const DEFAULT_MAX_BATCH_BYTES: usize = 64 * 1024;

/// Statistics for batched WAL writer
#[derive(Debug, Default, Clone)]
pub struct BatchedWalStats {
    /// Total entries written
    pub entries_written: u64,
    /// Total batches written
    pub batches_written: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total syncs performed
    pub syncs_performed: u64,
    /// Average batch size
    pub avg_batch_size: f64,
}

/// Batched WAL writer with vectored I/O
///
/// Accumulates WAL entries and writes them in batches using writev()
/// for optimal I/O performance.
pub struct BatchedWalWriter {
    /// File handle
    file: File,
    /// Pending entries (pre-serialized)
    pending: Vec<Vec<u8>>,
    /// Total pending bytes
    pending_bytes: usize,
    /// Maximum batch size (entries)
    max_batch_size: usize,
    /// Maximum batch bytes
    max_batch_bytes: usize,
    /// Batch header buffer (reused)
    header_buf: Vec<u8>,
    /// Statistics
    stats: BatchedWalStats,
}

impl BatchedWalWriter {
    /// Create a new batched WAL writer
    ///
    /// # Arguments
    /// * `path` - Path to WAL file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_config(path, DEFAULT_MAX_BATCH_SIZE, DEFAULT_MAX_BATCH_BYTES)
    }

    /// Create with custom configuration
    ///
    /// # Arguments
    /// * `path` - Path to WAL file
    /// * `max_batch_size` - Maximum entries per batch
    /// * `max_batch_bytes` - Maximum bytes per batch
    pub fn with_config<P: AsRef<Path>>(
        path: P,
        max_batch_size: usize,
        max_batch_bytes: usize,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
            .map_err(SochDBError::Io)?;

        Ok(Self {
            file,
            pending: Vec::with_capacity(max_batch_size),
            pending_bytes: 0,
            max_batch_size,
            max_batch_bytes,
            header_buf: vec![0u8; BATCH_HEADER_SIZE],
            stats: BatchedWalStats::default(),
        })
    }

    /// Create from an existing file handle
    pub fn from_file(file: File) -> Self {
        Self {
            file,
            pending: Vec::with_capacity(DEFAULT_MAX_BATCH_SIZE),
            pending_bytes: 0,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            max_batch_bytes: DEFAULT_MAX_BATCH_BYTES,
            header_buf: vec![0u8; BATCH_HEADER_SIZE],
            stats: BatchedWalStats::default(),
        }
    }

    /// Add entry to pending batch
    ///
    /// Entry will be serialized and added to the pending batch.
    /// Automatic flush occurs if batch limits are reached.
    pub fn append(&mut self, entry: &TxnWalEntry) -> Result<()> {
        let serialized = entry.to_bytes();
        self.pending_bytes += serialized.len();
        self.pending.push(serialized);

        // Auto-flush if limits reached
        if self.pending.len() >= self.max_batch_size || self.pending_bytes >= self.max_batch_bytes {
            self.flush()?;
        }

        Ok(())
    }

    /// Add pre-serialized entry bytes
    #[inline]
    pub fn append_bytes(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.pending_bytes += bytes.len();
        self.pending.push(bytes);

        if self.pending.len() >= self.max_batch_size || self.pending_bytes >= self.max_batch_bytes {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush pending entries with vectored I/O
    ///
    /// Returns the number of entries written.
    pub fn flush(&mut self) -> Result<usize> {
        if self.pending.is_empty() {
            return Ok(0);
        }

        let count = self.pending.len();

        // Build batch header
        self.header_buf[0..4].copy_from_slice(&BATCH_MAGIC.to_le_bytes());
        self.header_buf[4..6].copy_from_slice(&BATCH_VERSION.to_le_bytes());
        self.header_buf[6..8].copy_from_slice(&(count as u16).to_le_bytes());
        self.header_buf[8..12].copy_from_slice(&(self.pending_bytes as u32).to_le_bytes());

        // Compute checksum over header (excluding checksum field)
        let checksum = crc32fast::hash(&self.header_buf[..12]);
        self.header_buf[12..16].copy_from_slice(&checksum.to_le_bytes());

        // Build iovec array for writev()
        let mut iovecs: Vec<IoSlice> = Vec::with_capacity(1 + self.pending.len());
        iovecs.push(IoSlice::new(&self.header_buf));
        for entry in &self.pending {
            iovecs.push(IoSlice::new(entry));
        }

        // Single vectored write - NO INTERMEDIATE COPIES
        let expected = BATCH_HEADER_SIZE + self.pending_bytes;
        let written = self.file.write_vectored(&iovecs).map_err(SochDBError::Io)?;

        if written != expected {
            return Err(SochDBError::Io(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("Incomplete batch write: {} < {}", written, expected),
            )));
        }

        // Update stats
        self.stats.entries_written += count as u64;
        self.stats.batches_written += 1;
        self.stats.bytes_written += written as u64;
        self.stats.avg_batch_size =
            self.stats.entries_written as f64 / self.stats.batches_written as f64;

        // Clear pending
        self.pending.clear();
        self.pending_bytes = 0;

        Ok(count)
    }

    /// Sync to disk (fsync)
    pub fn sync(&mut self) -> Result<()> {
        // Flush any pending entries first
        if !self.pending.is_empty() {
            self.flush()?;
        }

        self.file.sync_data().map_err(SochDBError::Io)?;

        self.stats.syncs_performed += 1;
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> BatchedWalStats {
        self.stats.clone()
    }

    /// Get pending entry count
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get pending bytes
    #[inline]
    pub fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }
}

impl Drop for BatchedWalWriter {
    fn drop(&mut self) {
        // Best effort flush on drop
        let _ = self.flush();
    }
}

/// Batch entry accumulator for a single transaction
///
/// Collects all writes for a transaction and commits them as a single batch.
pub struct BatchAccumulator {
    /// Transaction ID
    txn_id: u64,
    /// Accumulated entries
    entries: Vec<TxnWalEntry>,
}

impl BatchAccumulator {
    /// Create a new batch accumulator for a transaction
    pub fn new(txn_id: u64) -> Self {
        Self {
            txn_id,
            entries: Vec::with_capacity(16),
        }
    }

    /// Add a write to the batch (does not hit WAL yet)
    pub fn write(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.entries
            .push(TxnWalEntry::data(self.txn_id, key, value));
    }

    /// Add a delete to the batch
    pub fn delete(&mut self, key: Vec<u8>) {
        // Delete is represented as a data entry with empty value
        // The storage layer interprets empty value as tombstone
        self.entries
            .push(TxnWalEntry::data(self.txn_id, key, Vec::new()));
    }

    /// Get entry count
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Commit batch - writes all entries to WAL in single batch
    ///
    /// # Arguments
    /// * `writer` - Batched WAL writer
    ///
    /// # Returns
    /// The number of entries written
    pub fn commit(mut self, writer: &mut BatchedWalWriter) -> Result<usize> {
        // Add commit marker
        self.entries.push(TxnWalEntry::txn_commit(self.txn_id));

        let count = self.entries.len();

        // Append all entries to WAL
        for entry in &self.entries {
            writer.append(entry)?;
        }

        // Force flush and sync on commit
        writer.flush()?;
        writer.sync()?;

        Ok(count)
    }

    /// Abort the batch (discard all pending writes)
    pub fn abort(self) {
        // Just drop the entries - nothing written to WAL
    }

    /// Get the transaction ID
    #[inline]
    pub fn txn_id(&self) -> u64 {
        self.txn_id
    }
}

/// Thread-safe batched WAL writer
///
/// Wraps BatchedWalWriter with a mutex for concurrent access.
pub struct ConcurrentBatchedWal {
    inner: Mutex<BatchedWalWriter>,
    /// Next transaction ID
    next_txn_id: AtomicU64,
}

impl ConcurrentBatchedWal {
    /// Create a new concurrent batched WAL
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            inner: Mutex::new(BatchedWalWriter::new(path)?),
            next_txn_id: AtomicU64::new(1),
        })
    }

    /// Begin a new transaction batch
    pub fn begin(&self) -> BatchAccumulator {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        BatchAccumulator::new(txn_id)
    }

    /// Commit a transaction batch
    pub fn commit(&self, batch: BatchAccumulator) -> Result<usize> {
        let mut writer = self.inner.lock();
        batch.commit(&mut writer)
    }

    /// Append a single entry
    pub fn append(&self, entry: &TxnWalEntry) -> Result<()> {
        self.inner.lock().append(entry)
    }

    /// Force flush
    pub fn flush(&self) -> Result<usize> {
        self.inner.lock().flush()
    }

    /// Force sync
    pub fn sync(&self) -> Result<()> {
        self.inner.lock().sync()
    }

    /// Get statistics
    pub fn stats(&self) -> BatchedWalStats {
        self.inner.lock().stats()
    }
}

/// Batch reader for recovery
///
/// Reads batched WAL entries during crash recovery.
pub struct BatchedWalReader {
    file: File,
    position: u64,
}

impl BatchedWalReader {
    /// Open a batched WAL file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(SochDBError::Io)?;

        Ok(Self { file, position: 0 })
    }

    /// Read the next batch of entries
    ///
    /// Returns None if EOF or error
    pub fn read_batch(&mut self) -> Result<Option<Vec<TxnWalEntry>>> {
        use std::io::Read;

        // Read batch header
        let mut header = [0u8; BATCH_HEADER_SIZE];
        match self.file.read_exact(&mut header) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(SochDBError::Io(e)),
        }

        // Validate magic
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != BATCH_MAGIC {
            return Err(SochDBError::Internal("Invalid batch magic".into()));
        }

        // Read batch metadata
        let _version = u16::from_le_bytes(header[4..6].try_into().unwrap());
        let entry_count = u16::from_le_bytes(header[6..8].try_into().unwrap()) as usize;
        let total_bytes = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let stored_checksum = u32::from_le_bytes(header[12..16].try_into().unwrap());

        // Validate checksum
        let computed_checksum = crc32fast::hash(&header[..12]);
        if stored_checksum != computed_checksum {
            return Err(SochDBError::Internal(
                "Batch header checksum mismatch".into(),
            ));
        }

        // Read all entry data
        let mut data = vec![0u8; total_bytes];
        self.file.read_exact(&mut data).map_err(SochDBError::Io)?;

        // Parse individual entries
        let mut entries = Vec::with_capacity(entry_count);
        let mut cursor = std::io::Cursor::new(&data);

        for _ in 0..entry_count {
            let entry = TxnWalEntry::from_reader(&mut cursor)?;
            entries.push(entry);
        }

        self.position += BATCH_HEADER_SIZE as u64 + total_bytes as u64;

        Ok(Some(entries))
    }

    /// Get current file position
    pub fn position(&self) -> u64 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_batch_write_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Write some entries
        {
            let mut writer = BatchedWalWriter::new(&path).unwrap();

            for i in 0..10 {
                let entry = TxnWalEntry::data(
                    1,
                    format!("key{}", i).into_bytes(),
                    format!("value{}", i).into_bytes(),
                );
                writer.append(&entry).unwrap();
            }

            writer.flush().unwrap();
        }

        // Read back
        {
            let mut reader = BatchedWalReader::open(&path).unwrap();
            let batch = reader.read_batch().unwrap().unwrap();

            assert_eq!(batch.len(), 10);
            for (i, entry) in batch.iter().enumerate() {
                assert_eq!(entry.key, format!("key{}", i).into_bytes());
                assert_eq!(entry.value, format!("value{}", i).into_bytes());
            }
        }
    }

    #[test]
    fn test_auto_flush_on_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut writer = BatchedWalWriter::with_config(&path, 5, 1024 * 1024).unwrap();

        // Add 4 entries - should not auto-flush
        for i in 0..4 {
            let entry = TxnWalEntry::data(1, vec![i], vec![i]);
            writer.append(&entry).unwrap();
        }
        assert_eq!(writer.pending_count(), 4);

        // Add 5th entry - should auto-flush
        let entry = TxnWalEntry::data(1, vec![4], vec![4]);
        writer.append(&entry).unwrap();
        assert_eq!(writer.pending_count(), 0); // Flushed

        let stats = writer.stats();
        assert_eq!(stats.batches_written, 1);
        assert_eq!(stats.entries_written, 5);
    }

    #[test]
    fn test_batch_accumulator() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = ConcurrentBatchedWal::new(&path).unwrap();

        // Begin transaction and add writes
        let mut batch = wal.begin();
        batch.write(b"key1".to_vec(), b"value1".to_vec());
        batch.write(b"key2".to_vec(), b"value2".to_vec());
        batch.write(b"key3".to_vec(), b"value3".to_vec());

        assert_eq!(batch.len(), 3);

        // Commit
        let count = wal.commit(batch).unwrap();
        assert_eq!(count, 4); // 3 writes + 1 commit marker

        // Verify stats
        let stats = wal.stats();
        assert_eq!(stats.entries_written, 4);
    }

    #[test]
    fn test_batch_abort() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = ConcurrentBatchedWal::new(&path).unwrap();
        let wal_stats_before = wal.stats();

        // Begin transaction
        let mut batch = wal.begin();
        batch.write(b"key1".to_vec(), b"value1".to_vec());
        batch.write(b"key2".to_vec(), b"value2".to_vec());

        // Abort
        batch.abort();

        // Verify nothing was written
        let stats = wal.stats();
        assert_eq!(stats.entries_written, wal_stats_before.entries_written);
    }
}
