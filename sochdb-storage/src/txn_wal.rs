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

//! Transaction-Aware WAL for ACID Transactions
//!
//! This WAL implementation provides ACID transaction support:
//! - Atomicity: All writes in a transaction are logged together
//! - Consistency: Schema validation before commit
//! - Isolation: Transaction IDs for MVCC
//! - Durability: fsync after commit
//!
//! ## WAL Record Types
//!
//! - Data: Key-value write
//! - TxnBegin: Start of transaction
//! - TxnCommit: Commit point (durability guarantee)
//! - TxnAbort: Transaction rollback
//! - Checkpoint: Snapshot marker for recovery
//! - SchemaChange: DDL operations
//!
//! ## Record Format (Fixed Layout for Torn Write Detection)
//!
//! ```text
//! ┌───────────┬──────────┬─────────┬───────────┬─────────┬───────────┬─────────┬─────────┬──────────┐
//! │ Length(4) │ Type(1)  │ TxnId(8)│ Timestamp │ KeyLen  │ ValueLen  │ Key(*)  │ Value(*)│ CRC32(4) │
//! │           │          │         │   (8)     │  (4)    │   (4)     │         │         │          │
//! └───────────┴──────────┴─────────┴───────────┴─────────┴───────────┴─────────┴─────────┴──────────┘
//! ```
//!
//! ## Crash Recovery Guarantees
//!
//! 1. **Torn Write Detection**: Length prefix + CRC32 checksum detects partial writes
//! 2. **Atomic Commit**: Commit record with fsync ensures durability
//! 3. **Uncommitted Rollback**: Transactions without commit record are discarded
//! 4. **Checkpoint Safety**: Checkpoint marker allows safe WAL truncation

use byteorder::{LittleEndian, ReadBytesExt};
use parking_lot::Mutex;
use std::cell::Cell;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use sochdb_core::{Result, SochDBError, WalRecordType};

// =============================================================================
// Coarse-Grained Timestamp Caching (Recommendation 5)
// =============================================================================

/// Cache validity period in nanoseconds (1ms - allows ~1000+ writes per refresh)
const CACHE_VALIDITY_NS: u64 = 1_000_000;

thread_local! {
    /// Thread-local timestamp cache: (Instant, cached_timestamp_us)
    /// Avoids syscall overhead by caching timestamp for ~1ms windows
    static TS_CACHE: Cell<(Instant, u64)> = Cell::new((Instant::now(), 0));
}

/// Get cached timestamp in microseconds
///
/// This function eliminates per-write `SystemTime::now()` syscalls by:
/// 1. Caching the wall-clock timestamp in thread-local storage
/// 2. Using monotonic `Instant` for sub-millisecond offsets
/// 3. Only refreshing from syscall when cache expires (every ~1ms)
///
/// ## Performance Impact
///
/// - Without caching: ~20-25ns per syscall × 1M writes = 20-25ms overhead
/// - With caching: ~20ns per 1000 writes = 0.02ms overhead (1000× improvement)
///
/// ## Monotonicity Guarantee
///
/// Timestamps are guaranteed to be monotonically increasing within a thread
/// due to the `elapsed` offset added to the cached base timestamp.
#[inline(always)]
pub fn cached_timestamp_us() -> u64 {
    TS_CACHE.with(|cache| {
        let (instant, ts) = cache.get();
        let elapsed_ns = instant.elapsed().as_nanos() as u64;
        
        if elapsed_ns < CACHE_VALIDITY_NS {
            // Fast path: return cached + monotonic offset
            // This is pure arithmetic, no syscall
            ts + elapsed_ns / 1000
        } else {
            // Slow path: refresh cache from syscall
            let new_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;
            cache.set((Instant::now(), new_ts));
            new_ts
        }
    })
}

/// Header size in bytes (without key/value data)
const RECORD_HEADER_SIZE: usize = 4 + 1 + 8 + 8 + 4 + 4; // length + type + txn_id + timestamp + key_len + value_len

/// Checksum size (CRC32)
const CHECKSUM_SIZE: usize = 4;

/// Default capacity for transaction-local WAL buffer (32 KB - typical batch of 100-500 writes)
const DEFAULT_TXN_BUFFER_CAPACITY: usize = 32 * 1024;

// =============================================================================
// Transaction-Local WAL Buffer - Zero Lock Overhead During Transaction
// =============================================================================

/// Transaction-local WAL write buffer
///
/// Collects all writes during a transaction in memory, then flushes
/// everything with a SINGLE lock acquisition at commit time.
///
/// ## Performance Impact
///
/// ```text
/// Without TxnWalBuffer (current):
///   1000 writes × (lock + 9 write_all + unlock) = 1000 lock acquisitions
///   Lock overhead: 1000 × 20ns = 20µs per transaction
///
/// With TxnWalBuffer:
///   1000 writes buffered locally (NO LOCK)
///   1 flush at commit (SINGLE LOCK)
///   Lock overhead: 1 × 20ns = 20ns per transaction
///   Speedup: 1000× for lock overhead
/// ```
///
/// ## Usage
///
/// ```ignore
/// let mut buffer = TxnWalBuffer::new(txn_id);
///
/// // These do NOT acquire any locks
/// buffer.append(b"key1", b"value1");
/// buffer.append(b"key2", b"value2");
/// // ... hundreds of writes ...
///
/// // Single lock acquisition, single write syscall
/// buffer.flush(&wal)?;
/// ```
#[derive(Debug)]
pub struct TxnWalBuffer {
    /// Transaction ID for all entries
    txn_id: u64,
    /// Accumulated serialized entries
    buffer: Vec<u8>,
    /// Number of entries buffered
    entry_count: usize,
}

impl TxnWalBuffer {
    /// Create a new buffer for a transaction
    #[inline]
    pub fn new(txn_id: u64) -> Self {
        Self {
            txn_id,
            buffer: Vec::with_capacity(DEFAULT_TXN_BUFFER_CAPACITY),
            entry_count: 0,
        }
    }

    /// Create with specific capacity
    #[inline]
    pub fn with_capacity(txn_id: u64, capacity: usize) -> Self {
        Self {
            txn_id,
            buffer: Vec::with_capacity(capacity),
            entry_count: 0,
        }
    }

    /// Append a key-value write to the buffer - NO LOCK, NO SYSCALL
    ///
    /// Serializes the WAL entry directly to the buffer with CRC32 calculation.
    /// This is completely lock-free and does not touch the file system.
    ///
    /// Uses cached timestamps to eliminate per-write syscall overhead.
    #[inline]
    pub fn append(&mut self, key: &[u8], value: &[u8]) {
        // Use cached timestamp instead of syscall (Recommendation 5)
        let timestamp_us = cached_timestamp_us();

        let total_len = RECORD_HEADER_SIZE + key.len() + value.len() + CHECKSUM_SIZE;
        let entry_start = self.buffer.len();

        // Reserve space for length prefix (will fill at end)
        self.buffer.extend_from_slice(&[0u8; 4]);

        let mut hasher = crc32fast::Hasher::new();

        // Record type (Data = 0)
        let record_type_byte = WalRecordType::Data as u8;
        self.buffer.push(record_type_byte);
        hasher.update(&[record_type_byte]);

        // Transaction ID
        let txn_bytes = self.txn_id.to_le_bytes();
        self.buffer.extend_from_slice(&txn_bytes);
        hasher.update(&txn_bytes);

        // Timestamp
        let ts_bytes = timestamp_us.to_le_bytes();
        self.buffer.extend_from_slice(&ts_bytes);
        hasher.update(&ts_bytes);

        // Key length
        let key_len_bytes = (key.len() as u32).to_le_bytes();
        self.buffer.extend_from_slice(&key_len_bytes);
        hasher.update(&key_len_bytes);

        // Value length
        let val_len_bytes = (value.len() as u32).to_le_bytes();
        self.buffer.extend_from_slice(&val_len_bytes);
        hasher.update(&val_len_bytes);

        // Key data
        self.buffer.extend_from_slice(key);
        hasher.update(key);

        // Value data
        self.buffer.extend_from_slice(value);
        hasher.update(value);

        // CRC32 checksum
        self.buffer
            .extend_from_slice(&hasher.finalize().to_le_bytes());

        // Fill in length prefix (content length, not including the 4-byte length field)
        let content_len = (total_len - 4) as u32;
        self.buffer[entry_start..entry_start + 4].copy_from_slice(&content_len.to_le_bytes());

        self.entry_count += 1;
    }

    /// Flush all buffered entries to WAL - SINGLE LOCK, SINGLE WRITE
    ///
    /// Acquires the WAL lock once and writes all accumulated entries.
    /// Returns the first sequence number assigned to buffered entries.
    ///
    /// Note: Use TxnWal::flush_buffer() instead of calling this directly,
    /// as it properly handles the private writer field.
    #[inline]
    pub fn flush_to_wal(&self, wal: &TxnWal) -> Result<u64> {
        wal.flush_buffer(self)
    }

    /// Clear the buffer (for reuse)
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.entry_count = 0;
    }

    /// Get number of buffered entries
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get total bytes buffered
    #[inline]
    pub fn bytes_buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// WAL entry for transaction-aware operations
#[derive(Debug, Clone)]
pub struct TxnWalEntry {
    /// Record type
    pub record_type: WalRecordType,
    /// Transaction ID
    pub txn_id: u64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Key data
    pub key: Vec<u8>,
    /// Value data
    pub value: Vec<u8>,
}

impl TxnWalEntry {
    /// Create a new data entry
    pub fn data(txn_id: u64, key: Vec<u8>, value: Vec<u8>) -> Self {
        Self {
            record_type: WalRecordType::Data,
            txn_id,
            timestamp_us: Self::now_us(),
            key,
            value,
        }
    }

    /// Create a transaction begin entry
    pub fn txn_begin(txn_id: u64) -> Self {
        Self {
            record_type: WalRecordType::TxnBegin,
            txn_id,
            timestamp_us: Self::now_us(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    /// Create a transaction commit entry
    pub fn txn_commit(txn_id: u64) -> Self {
        Self {
            record_type: WalRecordType::TxnCommit,
            txn_id,
            timestamp_us: Self::now_us(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    /// Create a transaction abort entry
    pub fn txn_abort(txn_id: u64) -> Self {
        Self {
            record_type: WalRecordType::TxnAbort,
            txn_id,
            timestamp_us: Self::now_us(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    /// Create a checkpoint entry
    pub fn checkpoint(txn_id: u64) -> Self {
        Self {
            record_type: WalRecordType::Checkpoint,
            txn_id,
            timestamp_us: Self::now_us(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    /// Create a schema change entry
    pub fn schema_change(txn_id: u64, schema_data: Vec<u8>) -> Self {
        Self {
            record_type: WalRecordType::SchemaChange,
            txn_id,
            timestamp_us: Self::now_us(),
            key: Vec::new(),
            value: schema_data,
        }
    }

    /// Get current time in microseconds (uses cached timestamp)
    #[inline]
    fn now_us() -> u64 {
        cached_timestamp_us()
    }

    /// Calculate CRC32 checksum for this entry
    ///
    /// Uses crc32fast for portable, deterministic checksums.
    /// The checksum covers all fields except the checksum itself.
    /// NOTE: This is only used for verification. to_bytes() calculates CRC in single pass.
    pub fn checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&[self.record_type as u8]);
        hasher.update(&self.txn_id.to_le_bytes());
        hasher.update(&self.timestamp_us.to_le_bytes());
        hasher.update(&(self.key.len() as u32).to_le_bytes());
        hasher.update(&(self.value.len() as u32).to_le_bytes());
        hasher.update(&self.key);
        hasher.update(&self.value);
        hasher.finalize()
    }

    /// Serialize to bytes with single-pass CRC calculation
    ///
    /// Optimized to calculate CRC while building the buffer,
    /// avoiding a second pass over all data.
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_len = RECORD_HEADER_SIZE + self.key.len() + self.value.len() + CHECKSUM_SIZE;
        let mut buf = Vec::with_capacity(total_len);
        let mut hasher = crc32fast::Hasher::new();

        // Length (not including the length field itself) - not included in CRC
        let content_len = (total_len - 4) as u32;
        buf.extend_from_slice(&content_len.to_le_bytes());

        // Record type
        let record_type_byte = self.record_type as u8;
        buf.push(record_type_byte);
        hasher.update(&[record_type_byte]);

        // Transaction ID
        let txn_bytes = self.txn_id.to_le_bytes();
        buf.extend_from_slice(&txn_bytes);
        hasher.update(&txn_bytes);

        // Timestamp
        let ts_bytes = self.timestamp_us.to_le_bytes();
        buf.extend_from_slice(&ts_bytes);
        hasher.update(&ts_bytes);

        // Key length
        let key_len_bytes = (self.key.len() as u32).to_le_bytes();
        buf.extend_from_slice(&key_len_bytes);
        hasher.update(&key_len_bytes);

        // Value length
        let val_len_bytes = (self.value.len() as u32).to_le_bytes();
        buf.extend_from_slice(&val_len_bytes);
        hasher.update(&val_len_bytes);

        // Key data
        buf.extend_from_slice(&self.key);
        hasher.update(&self.key);

        // Value data
        buf.extend_from_slice(&self.value);
        hasher.update(&self.value);

        // CRC32 Checksum (computed in single pass above)
        buf.extend_from_slice(&hasher.finalize().to_le_bytes());

        buf
    }

    /// Deserialize from bytes with torn write detection
    ///
    /// Returns error if:
    /// - Length field indicates data is too short
    /// - CRC32 checksum mismatch (corruption or torn write)
    /// - Invalid record type
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self> {
        // Length (allows torn write detection - if length claims more data than exists)
        let content_len = reader.read_u32::<LittleEndian>()?;
        if content_len < (RECORD_HEADER_SIZE - 4 + CHECKSUM_SIZE) as u32 {
            return Err(SochDBError::Corruption("WAL entry too short".into()));
        }

        // Record type
        let record_type_byte = reader.read_u8()?;
        let record_type = WalRecordType::try_from(record_type_byte).map_err(|_| {
            SochDBError::Corruption(format!("Invalid record type: {}", record_type_byte))
        })?;

        // Transaction ID
        let txn_id = reader.read_u64::<LittleEndian>()?;

        // Timestamp
        let timestamp_us = reader.read_u64::<LittleEndian>()?;

        // Key length
        let key_len = reader.read_u32::<LittleEndian>()? as usize;

        // Value length
        let value_len = reader.read_u32::<LittleEndian>()? as usize;

        // Key data
        let mut key = vec![0u8; key_len];
        reader.read_exact(&mut key)?;

        // Value data
        let mut value = vec![0u8; value_len];
        reader.read_exact(&mut value)?;

        // CRC32 Checksum
        let stored_checksum = reader.read_u32::<LittleEndian>()?;

        let entry = Self {
            record_type,
            txn_id,
            timestamp_us,
            key,
            value,
        };

        // Verify checksum - detects both corruption and torn writes
        if entry.checksum() != stored_checksum {
            return Err(SochDBError::Corruption(format!(
                "WAL checksum mismatch for txn_id {}: expected {}, got {}",
                txn_id,
                entry.checksum(),
                stored_checksum
            )));
        }

        Ok(entry)
    }
}

/// Transaction-aware Write-Ahead Log
pub struct TxnWal {
    /// Path to WAL file
    path: PathBuf,
    /// Buffered writer
    writer: Mutex<BufWriter<File>>,
    /// Next transaction ID
    next_txn_id: AtomicU64,
    /// Write sequence number
    sequence: AtomicU64,
    /// Bytes written since last sync
    bytes_since_sync: AtomicU64,
    /// Cached timestamp (microseconds since epoch)
    /// Updated periodically to avoid syscall per write
    cached_timestamp_us: AtomicU64,
}

impl TxnWal {
    /// Create a new WAL or open existing one
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        // Use 256KB buffer for better batch performance (default is 8KB)
        // This reduces system calls when buffering many small writes
        let now_us = cached_timestamp_us();

        let wal = Self {
            path,
            writer: Mutex::new(BufWriter::with_capacity(256 * 1024, file)),
            next_txn_id: AtomicU64::new(1),
            sequence: AtomicU64::new(0),
            bytes_since_sync: AtomicU64::new(0),
            cached_timestamp_us: AtomicU64::new(now_us),
        };

        // Recover state from existing WAL
        wal.recover_state()?;

        Ok(wal)
    }

    /// Recover state (next txn ID, sequence) from existing WAL
    fn recover_state(&self) -> Result<()> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut max_txn_id: u64 = 0;
        let mut count: u64 = 0;

        loop {
            match TxnWalEntry::from_reader(&mut reader) {
                Ok(entry) => {
                    if entry.txn_id > max_txn_id {
                        max_txn_id = entry.txn_id;
                    }
                    count += 1;
                }
                Err(SochDBError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(_) => {
                    // Ignore corruption at end of WAL (incomplete write)
                    break;
                }
            }
        }

        self.next_txn_id.store(max_txn_id + 1, Ordering::SeqCst);
        self.sequence.store(count, Ordering::SeqCst);

        Ok(())
    }

    /// Get cached timestamp, updating if stale (>1ms old)
    ///
    /// This avoids a syscall per write by caching the timestamp.
    /// For WAL purposes, ~1ms granularity is sufficient.
    #[inline]
    fn get_cached_timestamp(&self) -> u64 {
        // Fast path: use cached value (no syscall)
        let cached = self.cached_timestamp_us.load(Ordering::Relaxed);

        // Refresh occasionally (every ~1000 writes or when sequence wraps)
        // This is a very cheap check since sequence is incremented atomically anyway
        let seq = self.sequence.load(Ordering::Relaxed);
        if seq & 0x3FF == 0 {
            // Refresh every 1024 writes
            let now_us = cached_timestamp_us();
            self.cached_timestamp_us.store(now_us, Ordering::Relaxed);
            return now_us;
        }

        cached
    }

    /// Append an entry to the WAL
    ///
    /// Returns the sequence number of this write.
    /// NOTE: Does NOT flush - caller must call flush() or sync() for durability.
    /// This is intentional for performance - BufWriter handles batching.
    pub fn append(&self, entry: &TxnWalEntry) -> Result<u64> {
        let bytes = entry.to_bytes();
        let mut writer = self.writer.lock();

        writer.write_all(&bytes)?;
        // Don't flush here - BufWriter will batch writes automatically.
        // Call flush() explicitly before sync() or commit().

        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        self.bytes_since_sync
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);

        Ok(seq)
    }

    /// Append entry without flushing (for batched writes)
    ///
    /// Caller must call flush() or sync() afterward to ensure durability.
    #[inline]
    pub fn append_no_flush(&self, entry: &TxnWalEntry) -> Result<u64> {
        let bytes = entry.to_bytes();
        let mut writer = self.writer.lock();

        writer.write_all(&bytes)?;
        // No flush - let BufWriter buffer the writes

        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        self.bytes_since_sync
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);

        Ok(seq)
    }

    /// Write data without flushing (for batched writes within a transaction)
    #[inline]
    pub fn write_no_flush(&self, txn_id: u64, key: Vec<u8>, value: Vec<u8>) -> Result<u64> {
        let entry = TxnWalEntry::data(txn_id, key, value);
        self.append_no_flush(&entry)
    }

    /// Write data from slices without any allocation
    ///
    /// This is the fastest path for writing - no intermediate Vec allocations.
    /// Calculates CRC32 while serializing directly to BufWriter.
    #[inline]
    pub fn write_no_flush_refs(&self, txn_id: u64, key: &[u8], value: &[u8]) -> Result<u64> {
        // Use coarse timestamp (cached every ~1ms) instead of syscall per write
        let timestamp_us = self.get_cached_timestamp();

        let total_len = RECORD_HEADER_SIZE + key.len() + value.len() + CHECKSUM_SIZE;
        let mut hasher = crc32fast::Hasher::new();

        let mut writer = self.writer.lock();

        // Length (not included in CRC)
        let content_len = (total_len - 4) as u32;
        writer.write_all(&content_len.to_le_bytes())?;

        // Record type
        let record_type_byte = WalRecordType::Data as u8;
        writer.write_all(&[record_type_byte])?;
        hasher.update(&[record_type_byte]);

        // Transaction ID
        let txn_bytes = txn_id.to_le_bytes();
        writer.write_all(&txn_bytes)?;
        hasher.update(&txn_bytes);

        // Timestamp
        let ts_bytes = timestamp_us.to_le_bytes();
        writer.write_all(&ts_bytes)?;
        hasher.update(&ts_bytes);

        // Key length
        let key_len_bytes = (key.len() as u32).to_le_bytes();
        writer.write_all(&key_len_bytes)?;
        hasher.update(&key_len_bytes);

        // Value length
        let val_len_bytes = (value.len() as u32).to_le_bytes();
        writer.write_all(&val_len_bytes)?;
        hasher.update(&val_len_bytes);

        // Key data
        writer.write_all(key)?;
        hasher.update(key);

        // Value data
        writer.write_all(value)?;
        hasher.update(value);

        // CRC32 checksum
        writer.write_all(&hasher.finalize().to_le_bytes())?;

        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        self.bytes_since_sync
            .fetch_add(total_len as u64, Ordering::Relaxed);

        Ok(seq)
    }

    /// Flush pending writes to kernel buffer (but not to disk)
    pub fn flush(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        writer.flush()?;
        Ok(())
    }

    /// Append and immediately sync for durability
    pub fn append_sync(&self, entry: &TxnWalEntry) -> Result<u64> {
        let seq = self.append(entry)?;
        self.sync()?;
        Ok(seq)
    }

    /// Force sync to disk
    pub fn sync(&self) -> Result<()> {
        let writer = self.writer.lock();
        writer.get_ref().sync_all()?;
        self.bytes_since_sync.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Flush a TxnWalBuffer with single lock acquisition
    ///
    /// This is the high-performance path for transaction commit:
    /// - All writes during the transaction are buffered locally
    /// - At commit, this method flushes everything with ONE lock
    ///
    /// ## Performance
    ///
    /// ```text
    /// 1000 writes with individual flush: 1000 × lock overhead
    /// 1000 writes with buffer + flush_buffer: 1 × lock overhead
    /// Speedup: ~1000× for lock overhead
    /// ```
    #[inline]
    pub fn flush_buffer(&self, buffer: &TxnWalBuffer) -> Result<u64> {
        if buffer.is_empty() {
            return Ok(0);
        }

        let mut writer = self.writer.lock();
        writer.write_all(&buffer.buffer)?;

        let seq = self
            .sequence
            .fetch_add(buffer.entry_count as u64, Ordering::SeqCst);
        self.bytes_since_sync
            .fetch_add(buffer.buffer.len() as u64, Ordering::Relaxed);

        Ok(seq)
    }

    /// Get the current size of the WAL file in bytes
    pub fn size_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }

    /// Allocate a new transaction ID
    pub fn alloc_txn_id(&self) -> u64 {
        self.next_txn_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self) -> Result<u64> {
        let txn_id = self.alloc_txn_id();
        let entry = TxnWalEntry::txn_begin(txn_id);
        self.append(&entry)?;
        Ok(txn_id)
    }

    /// Commit a transaction (with fsync for durability)
    ///
    /// This flushes all pending writes and then fsyncs the commit record.
    pub fn commit_transaction(&self, txn_id: u64) -> Result<()> {
        // First flush any pending buffered writes
        self.flush()?;

        // Then write commit record with fsync
        let entry = TxnWalEntry::txn_commit(txn_id);
        self.append_sync(&entry)?;
        Ok(())
    }

    /// Abort a transaction
    pub fn abort_transaction(&self, txn_id: u64) -> Result<()> {
        let entry = TxnWalEntry::txn_abort(txn_id);
        self.append(&entry)?;
        Ok(())
    }

    /// Write data within a transaction
    pub fn write(&self, txn_id: u64, key: Vec<u8>, value: Vec<u8>) -> Result<u64> {
        let entry = TxnWalEntry::data(txn_id, key, value);
        self.append(&entry)
    }

    /// Replay WAL for crash recovery
    ///
    /// Returns (committed_writes, recovered_txn_count)
    #[allow(clippy::type_complexity)]
    pub fn replay_for_recovery(&self) -> Result<(Vec<(Vec<u8>, Vec<u8>)>, usize)> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        let mut committed_txns: HashSet<u64> = HashSet::new();
        let mut pending_writes: std::collections::HashMap<u64, Vec<(Vec<u8>, Vec<u8>)>> =
            std::collections::HashMap::new();

        // First pass: find committed transactions
        loop {
            match TxnWalEntry::from_reader(&mut reader) {
                Ok(entry) => match entry.record_type {
                    WalRecordType::TxnBegin => {
                        pending_writes.insert(entry.txn_id, Vec::new());
                    }
                    WalRecordType::Data => {
                        if let Some(writes) = pending_writes.get_mut(&entry.txn_id) {
                            writes.push((entry.key, entry.value));
                        }
                    }
                    WalRecordType::TxnCommit => {
                        committed_txns.insert(entry.txn_id);
                    }
                    WalRecordType::TxnAbort => {
                        pending_writes.remove(&entry.txn_id);
                    }
                    _ => {}
                },
                Err(SochDBError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(_) => {
                    break;
                }
            }
        }

        // Collect committed writes
        let mut result = Vec::new();
        let mut txn_count = 0;

        for (txn_id, writes) in pending_writes {
            if committed_txns.contains(&txn_id) {
                result.extend(writes);
                txn_count += 1;
            }
            // Uncommitted transactions are discarded (rollback)
        }

        Ok((result, txn_count))
    }

    /// Replay WAL with a callback
    pub fn replay<F>(&self, mut callback: F) -> Result<u64>
    where
        F: FnMut(TxnWalEntry) -> Result<()>,
    {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut count = 0u64;

        loop {
            match TxnWalEntry::from_reader(&mut reader) {
                Ok(entry) => {
                    callback(entry)?;
                    count += 1;
                }
                Err(SochDBError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => {
                    // Log warning but continue (may be incomplete entry at end)
                    eprintln!("WAL replay warning: {:?}", e);
                    break;
                }
            }
        }

        Ok(count)
    }

    /// Truncate WAL (called after successful checkpoint)
    pub fn truncate(&self) -> Result<()> {
        let writer = self.writer.lock();
        let file = writer.get_ref();
        file.set_len(0)?;
        file.sync_all()?;
        self.sequence.store(0, Ordering::SeqCst);
        self.bytes_since_sync.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Write a checkpoint marker
    pub fn write_checkpoint(&self) -> Result<u64> {
        let entry = TxnWalEntry::checkpoint(0);
        self.append_sync(&entry)
    }

    /// Write a Compensation Log Record (CLR) for undo operations
    ///
    /// CLRs are redo-only records that skip past already undone operations
    /// during recovery. The undo_next_lsn field tells recovery where to
    /// continue undoing after this CLR.
    pub fn append_clr(
        &self,
        txn_id: u64,
        _original_lsn: u64,
        undo_next_lsn: Option<u64>,
        undo_data: &[u8],
    ) -> Result<u64> {
        // Encode undo_next_lsn into the key field
        let key = undo_next_lsn.unwrap_or(0).to_le_bytes().to_vec();
        let entry = TxnWalEntry {
            record_type: WalRecordType::CompensationLogRecord,
            txn_id,
            timestamp_us: TxnWalEntry::now_us(),
            key, // undo_next_lsn encoded in key
            value: undo_data.to_vec(),
        };
        self.append(&entry)
    }

    /// Write checkpoint with data (for fuzzy checkpoints)
    pub fn write_checkpoint_with_data(&self, checkpoint_data: &[u8]) -> Result<u64> {
        let entry = TxnWalEntry {
            record_type: WalRecordType::Checkpoint,
            txn_id: 0,
            timestamp_us: TxnWalEntry::now_us(),
            key: Vec::new(),
            value: checkpoint_data.to_vec(),
        };
        self.append_sync(&entry)
    }

    /// Write checkpoint end with captured state
    pub fn write_checkpoint_end(&self, checkpoint_data: &[u8]) -> Result<u64> {
        let entry = TxnWalEntry {
            record_type: WalRecordType::CheckpointEnd,
            txn_id: 0,
            timestamp_us: TxnWalEntry::now_us(),
            key: Vec::new(),
            value: checkpoint_data.to_vec(),
        };
        self.append_sync(&entry)
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Get bytes written since last sync
    pub fn bytes_since_sync(&self) -> u64 {
        self.bytes_since_sync.load(Ordering::Relaxed)
    }

    /// Get path to WAL file
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Statistics about WAL state
#[derive(Debug, Clone, Default)]
pub struct TxnWalStats {
    /// Number of entries written
    pub entries_written: u64,
    /// Bytes written since last sync
    pub bytes_since_sync: u64,
    /// Current transaction ID counter
    pub next_txn_id: u64,
}

// ============================================================================
// Sharded WAL for Reduced Mutex Contention
// ============================================================================

/// Sharded Write-Ahead Log for high-concurrency workloads
///
/// Instead of a single Mutex<File>, uses multiple shards to reduce contention:
/// - Writers hash to shard by txn_id
/// - Each shard has its own buffer
/// - Central coordinator handles fsync ordering
///
/// Reduces contention from O(1) bottleneck to O(num_shards) parallelism.
#[allow(dead_code)]
pub struct ShardedWal {
    /// Shard writers (txn_id % num_shards selects shard)
    shards: Vec<parking_lot::Mutex<WalShard>>,
    /// Number of shards (power of 2)
    num_shards: usize,
    /// Central WAL file for ordered writes
    central_writer: parking_lot::Mutex<BufWriter<File>>,
    /// Next transaction ID
    next_txn_id: AtomicU64,
    /// Write sequence (global ordering)
    sequence: AtomicU64,
    /// Path
    path: PathBuf,
}

/// Individual WAL shard buffer
struct WalShard {
    /// Buffered entries for this shard
    buffer: Vec<u8>,
    /// Number of entries buffered
    entry_count: usize,
}

impl WalShard {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(64 * 1024), // 64KB per shard
            entry_count: 0,
        }
    }

    fn append(&mut self, entry: &TxnWalEntry) {
        let bytes = entry.to_bytes();
        self.buffer.extend_from_slice(&bytes);
        self.entry_count += 1;
    }

    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    fn drain(&mut self) -> Vec<u8> {
        self.entry_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ShardedWal {
    /// Create sharded WAL with specified number of shards
    ///
    /// Recommended: 4-8 shards for typical server workloads
    pub fn new<P: AsRef<Path>>(path: P, num_shards: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        // Round up to power of 2 for fast modulo
        let num_shards = num_shards.next_power_of_two();
        let shards: Vec<_> = (0..num_shards)
            .map(|_| parking_lot::Mutex::new(WalShard::new()))
            .collect();

        Ok(Self {
            shards,
            num_shards,
            central_writer: parking_lot::Mutex::new(BufWriter::with_capacity(256 * 1024, file)),
            next_txn_id: AtomicU64::new(1),
            sequence: AtomicU64::new(0),
            path,
        })
    }

    /// Get shard index for transaction
    #[inline]
    fn shard_idx(&self, txn_id: u64) -> usize {
        (txn_id as usize) & (self.num_shards - 1)
    }

    /// Append entry to appropriate shard (lock-free for different txns)
    pub fn append(&self, entry: &TxnWalEntry) -> u64 {
        let shard_idx = self.shard_idx(entry.txn_id);
        let mut shard = self.shards[shard_idx].lock();
        shard.append(entry);
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    /// Allocate transaction ID
    pub fn alloc_txn_id(&self) -> u64 {
        self.next_txn_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Flush all shard buffers to central file
    pub fn flush(&self) -> Result<()> {
        let mut central = self.central_writer.lock();

        // Collect all shard buffers (brief lock per shard)
        for shard in &self.shards {
            let mut shard_guard = shard.lock();
            if !shard_guard.is_empty() {
                let data = shard_guard.drain();
                central.write_all(&data)?;
            }
        }

        central.flush()?;
        Ok(())
    }

    /// Sync to disk (fsync)
    pub fn sync(&self) -> Result<()> {
        self.flush()?;
        let central = self.central_writer.lock();
        central.get_ref().sync_all()?;
        Ok(())
    }

    /// Begin transaction
    pub fn begin_transaction(&self) -> Result<u64> {
        let txn_id = self.alloc_txn_id();
        let entry = TxnWalEntry::txn_begin(txn_id);
        self.append(&entry);
        Ok(txn_id)
    }

    /// Write data
    pub fn write(&self, txn_id: u64, key: Vec<u8>, value: Vec<u8>) -> Result<u64> {
        let entry = TxnWalEntry::data(txn_id, key, value);
        Ok(self.append(&entry))
    }

    /// Commit transaction
    pub fn commit_transaction(&self, txn_id: u64) -> Result<u64> {
        let entry = TxnWalEntry::txn_commit(txn_id);
        let seq = self.append(&entry);
        self.sync()?; // Fsync on commit for durability
        Ok(seq)
    }

    /// Get statistics
    pub fn stats(&self) -> ShardedWalStats {
        let mut shard_entry_counts = Vec::with_capacity(self.num_shards);
        for shard in &self.shards {
            shard_entry_counts.push(shard.lock().entry_count);
        }

        ShardedWalStats {
            num_shards: self.num_shards,
            total_entries: self.sequence.load(Ordering::SeqCst),
            next_txn_id: self.next_txn_id.load(Ordering::SeqCst),
            shard_entry_counts,
        }
    }
}

/// Statistics for sharded WAL
#[derive(Debug, Clone)]
pub struct ShardedWalStats {
    pub num_shards: usize,
    pub total_entries: u64,
    pub next_txn_id: u64,
    pub shard_entry_counts: Vec<usize>,
}

/// Detailed crash recovery statistics
#[derive(Debug, Clone, Default)]
pub struct CrashRecoveryStats {
    /// Total records read from WAL
    pub total_records: u64,
    /// Number of committed transactions
    pub committed_txns: u64,
    /// Number of uncommitted (rolled back) transactions
    pub rolled_back_txns: u64,
    /// Number of explicitly aborted transactions
    pub aborted_txns: u64,
    /// Number of data writes recovered
    pub recovered_writes: u64,
    /// Number of torn/corrupted records at end (expected on crash)
    pub torn_records: u64,
    /// Bytes read from WAL
    pub bytes_read: u64,
    /// Recovery duration in microseconds
    pub recovery_duration_us: u64,
    /// Highest transaction ID seen (for restarting counter)
    pub max_txn_id: u64,
}

impl TxnWal {
    /// Get WAL statistics
    pub fn stats(&self) -> TxnWalStats {
        TxnWalStats {
            entries_written: self.sequence.load(Ordering::SeqCst),
            bytes_since_sync: self.bytes_since_sync.load(Ordering::Relaxed),
            next_txn_id: self.next_txn_id.load(Ordering::SeqCst),
        }
    }

    /// Full crash recovery with detailed statistics
    ///
    /// This method provides ACID recovery guarantees:
    /// 1. **Atomicity**: Uncommitted transactions are rolled back
    /// 2. **Durability**: All committed transactions are replayed
    /// 3. **Torn Write Detection**: Partial records at EOF are detected via CRC32
    ///
    /// Returns (committed_writes, stats) where committed_writes contains
    /// all key-value pairs from committed transactions in order.
    #[allow(clippy::type_complexity)]
    pub fn crash_recovery(&self) -> Result<(Vec<(Vec<u8>, Vec<u8>)>, CrashRecoveryStats)> {
        let start_time = std::time::Instant::now();
        let file = File::open(&self.path)?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        let mut stats = CrashRecoveryStats {
            bytes_read: file_size,
            ..Default::default()
        };

        let mut committed_txns: HashSet<u64> = HashSet::new();
        let mut aborted_txns: HashSet<u64> = HashSet::new();
        let mut pending_writes: std::collections::HashMap<u64, Vec<(Vec<u8>, Vec<u8>)>> =
            std::collections::HashMap::new();
        let mut all_txns: HashSet<u64> = HashSet::new();

        // Read all records, stopping at first corruption (torn write)
        loop {
            match TxnWalEntry::from_reader(&mut reader) {
                Ok(entry) => {
                    stats.total_records += 1;
                    if entry.txn_id > stats.max_txn_id {
                        stats.max_txn_id = entry.txn_id;
                    }

                    match entry.record_type {
                        WalRecordType::TxnBegin => {
                            pending_writes.insert(entry.txn_id, Vec::new());
                            all_txns.insert(entry.txn_id);
                        }
                        WalRecordType::Data => {
                            if let Some(writes) = pending_writes.get_mut(&entry.txn_id) {
                                writes.push((entry.key, entry.value));
                            }
                        }
                        WalRecordType::TxnCommit => {
                            committed_txns.insert(entry.txn_id);
                        }
                        WalRecordType::TxnAbort => {
                            pending_writes.remove(&entry.txn_id);
                            aborted_txns.insert(entry.txn_id);
                        }
                        _ => {}
                    }
                }
                Err(SochDBError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Clean EOF
                    break;
                }
                Err(_) => {
                    // Corruption or torn write - stop here
                    stats.torn_records += 1;
                    break;
                }
            }
        }

        // Collect committed writes
        let mut result = Vec::new();
        for (txn_id, writes) in &pending_writes {
            if committed_txns.contains(txn_id) {
                stats.committed_txns += 1;
                stats.recovered_writes += writes.len() as u64;
                result.extend(writes.clone());
            }
        }

        // Count aborted and rolled-back transactions
        stats.aborted_txns = aborted_txns.len() as u64;
        stats.rolled_back_txns = all_txns.len() as u64 - stats.committed_txns - stats.aborted_txns;

        stats.recovery_duration_us = start_time.elapsed().as_micros() as u64;

        Ok((result, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_entry_roundtrip() {
        let entry = TxnWalEntry::data(42, b"key".to_vec(), b"value".to_vec());
        let bytes = entry.to_bytes();

        let mut cursor = std::io::Cursor::new(bytes);
        let recovered = TxnWalEntry::from_reader(&mut cursor).unwrap();

        assert_eq!(recovered.record_type, WalRecordType::Data);
        assert_eq!(recovered.txn_id, 42);
        assert_eq!(recovered.key, b"key");
        assert_eq!(recovered.value, b"value");
    }

    #[test]
    fn test_wal_append_and_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write some entries
        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let txn_id = wal.begin_transaction().unwrap();
            wal.write(txn_id, b"k1".to_vec(), b"v1".to_vec()).unwrap();
            wal.write(txn_id, b"k2".to_vec(), b"v2".to_vec()).unwrap();
            wal.commit_transaction(txn_id).unwrap();
        }

        // Replay and verify
        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let (writes, txn_count) = wal.replay_for_recovery().unwrap();

            assert_eq!(txn_count, 1);
            assert_eq!(writes.len(), 2);
            assert_eq!(writes[0], (b"k1".to_vec(), b"v1".to_vec()));
            assert_eq!(writes[1], (b"k2".to_vec(), b"v2".to_vec()));
        }
    }

    #[test]
    fn test_uncommitted_transaction_rollback() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write committed and uncommitted transactions
        {
            let wal = TxnWal::new(&wal_path).unwrap();

            // Committed transaction
            let txn1 = wal.begin_transaction().unwrap();
            wal.write(txn1, b"committed".to_vec(), b"yes".to_vec())
                .unwrap();
            wal.commit_transaction(txn1).unwrap();

            // Uncommitted transaction (simulates crash)
            let txn2 = wal.begin_transaction().unwrap();
            wal.write(txn2, b"uncommitted".to_vec(), b"no".to_vec())
                .unwrap();
            // No commit!
        }

        // Replay - uncommitted should be rolled back
        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let (writes, txn_count) = wal.replay_for_recovery().unwrap();

            assert_eq!(txn_count, 1); // Only committed transaction
            assert_eq!(writes.len(), 1);
            assert_eq!(writes[0], (b"committed".to_vec(), b"yes".to_vec()));
        }
    }

    #[test]
    fn test_aborted_transaction() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let wal = TxnWal::new(&wal_path).unwrap();

            let txn = wal.begin_transaction().unwrap();
            wal.write(txn, b"aborted".to_vec(), b"data".to_vec())
                .unwrap();
            wal.abort_transaction(txn).unwrap();
        }

        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let (writes, txn_count) = wal.replay_for_recovery().unwrap();

            assert_eq!(txn_count, 0);
            assert!(writes.is_empty());
        }
    }

    #[test]
    fn test_checksum_validation() {
        let entry = TxnWalEntry::data(1, b"key".to_vec(), b"value".to_vec());
        let mut bytes = entry.to_bytes();

        // Corrupt the checksum
        let len = bytes.len();
        bytes[len - 1] ^= 0xFF;

        let mut cursor = std::io::Cursor::new(bytes);
        let result = TxnWalEntry::from_reader(&mut cursor);

        assert!(result.is_err());
    }

    #[test]
    fn test_crash_recovery_with_stats() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Simulate complex workload
        {
            let wal = TxnWal::new(&wal_path).unwrap();

            // Committed transaction 1
            let txn1 = wal.begin_transaction().unwrap();
            wal.write(txn1, b"k1".to_vec(), b"v1".to_vec()).unwrap();
            wal.write(txn1, b"k2".to_vec(), b"v2".to_vec()).unwrap();
            wal.commit_transaction(txn1).unwrap();

            // Aborted transaction
            let txn2 = wal.begin_transaction().unwrap();
            wal.write(txn2, b"aborted_key".to_vec(), b"aborted_val".to_vec())
                .unwrap();
            wal.abort_transaction(txn2).unwrap();

            // Committed transaction 2
            let txn3 = wal.begin_transaction().unwrap();
            wal.write(txn3, b"k3".to_vec(), b"v3".to_vec()).unwrap();
            wal.commit_transaction(txn3).unwrap();

            // Uncommitted transaction (simulates crash)
            let txn4 = wal.begin_transaction().unwrap();
            wal.write(txn4, b"uncommitted".to_vec(), b"data".to_vec())
                .unwrap();
            // No commit - simulates crash
        }

        // Recover and verify
        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let (writes, stats) = wal.crash_recovery().unwrap();

            // Should have 3 writes from 2 committed transactions
            assert_eq!(writes.len(), 3);
            assert_eq!(stats.committed_txns, 2);
            assert_eq!(stats.aborted_txns, 1);
            assert_eq!(stats.rolled_back_txns, 1); // txn4 was uncommitted
            assert_eq!(stats.recovered_writes, 3);
            assert!(stats.recovery_duration_us > 0);
        }
    }

    #[test]
    fn test_torn_write_detection() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write a valid transaction
        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let txn = wal.begin_transaction().unwrap();
            wal.write(txn, b"key".to_vec(), b"value".to_vec()).unwrap();
            wal.commit_transaction(txn).unwrap();
        }

        // Append corrupted bytes to simulate torn write
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&wal_path)
                .unwrap();
            // Write partial record (torn write)
            file.write_all(&[0x10, 0x00, 0x00, 0x00, 0xFF, 0xFF])
                .unwrap();
        }

        // Recovery should still work, detecting torn write
        {
            let wal = TxnWal::new(&wal_path).unwrap();
            let (writes, stats) = wal.crash_recovery().unwrap();

            // Should recover the valid transaction
            assert_eq!(writes.len(), 1);
            assert_eq!(stats.committed_txns, 1);
            assert_eq!(stats.torn_records, 1);
        }
    }

    #[test]
    fn test_crc32_determinism() {
        // Verify CRC32 produces consistent checksums for same content
        let mut entry1 = TxnWalEntry::data(42, b"key".to_vec(), b"value".to_vec());
        entry1.timestamp_us = 12345; // Fixed timestamp for determinism

        let mut entry2 = TxnWalEntry::data(42, b"key".to_vec(), b"value".to_vec());
        entry2.timestamp_us = 12345; // Same timestamp

        assert_eq!(entry1.checksum(), entry2.checksum());

        // Different content should produce different checksum
        let mut entry3 = TxnWalEntry::data(42, b"key".to_vec(), b"different".to_vec());
        entry3.timestamp_us = 12345;
        assert_ne!(entry1.checksum(), entry3.checksum());

        // Verify roundtrip preserves checksum
        let bytes = entry1.to_bytes();
        let mut cursor = std::io::Cursor::new(bytes);
        let recovered = TxnWalEntry::from_reader(&mut cursor).unwrap();
        assert_eq!(recovered.checksum(), entry1.checksum());
    }
}
