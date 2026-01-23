// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # WAL-Disciplined Atomic Multi-Index Writes (Task 14)
//!
//! Provides write-ahead logging for atomic multi-index operations.
//! Every multi-index write (embedding + blob + graph edges) must:
//! 1. Write intent to WAL
//! 2. Apply operations
//! 3. Write commit to WAL
//!
//! ## Design
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                        Write Path                               │
//! ├────────────────────────────────────────────────────────────────┤
//! │  1. Create Intent Record with LSN                              │
//! │  2. WAL.append(INTENT, ops...)                                 │
//! │  3. For each op:                                               │
//! │     - Apply to appropriate index                               │
//! │     - Track in intent                                          │
//! │  4. WAL.append(COMMIT, intent_id)                              │
//! │  5. Mark intent complete                                       │
//! └────────────────────────────────────────────────────────────────┘
//!
//! ┌────────────────────────────────────────────────────────────────┐
//! │                      Recovery Path                              │
//! ├────────────────────────────────────────────────────────────────┤
//! │  1. Scan WAL from last checkpoint                              │
//! │  2. Collect uncommitted intents                                │
//! │  3. For each uncommitted intent:                               │
//! │     - Replay forward (redo all ops)                            │
//! │     - Or roll back (undo applied ops)                          │
//! │  4. Write recovery complete marker                             │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Invariants
//!
//! - All writes are idempotent (can be replayed safely)
//! - LSN ordering ensures recovery correctness
//! - Commit records are atomic (single fsync)

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

// ============================================================================
// Log Sequence Number
// ============================================================================

/// Log Sequence Number - unique, monotonically increasing identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Lsn(pub u64);

impl Lsn {
    pub const ZERO: Lsn = Lsn(0);
    
    pub fn next(&self) -> Lsn {
        Lsn(self.0 + 1)
    }
}

// ============================================================================
// WAL Record Types
// ============================================================================

/// Type of WAL record
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WalRecordType {
    /// Intent: start of atomic operation
    Intent = 1,
    /// Single operation within intent
    Operation = 2,
    /// Commit: all operations completed
    Commit = 3,
    /// Abort: intent should be rolled back
    Abort = 4,
    /// Checkpoint: consistent point for recovery
    Checkpoint = 5,
}

/// A WAL record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// Log sequence number
    pub lsn: Lsn,
    /// Record type
    pub record_type: WalRecordType,
    /// Intent ID (for grouping related records)
    pub intent_id: u64,
    /// Record payload
    pub payload: WalPayload,
    /// Timestamp
    pub timestamp: u64,
    /// CRC32 checksum for integrity
    pub checksum: u32,
}

/// WAL record payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalPayload {
    /// Intent start
    IntentStart {
        memory_id: String,
        op_count: usize,
    },
    /// Operation details
    Operation {
        op_index: usize,
        op_type: String,
        key: Vec<u8>,
        value: Option<Vec<u8>>,
    },
    /// Commit marker
    Commit,
    /// Abort marker with reason
    Abort {
        reason: String,
    },
    /// Checkpoint data
    Checkpoint {
        last_committed_lsn: Lsn,
        intent_count: usize,
    },
}

impl WalRecord {
    /// Create new record
    pub fn new(lsn: Lsn, record_type: WalRecordType, intent_id: u64, payload: WalPayload) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let mut record = Self {
            lsn,
            record_type,
            intent_id,
            payload,
            timestamp,
            checksum: 0,
        };
        
        record.checksum = record.compute_checksum();
        record
    }
    
    /// Compute CRC32 checksum
    fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.lsn.0.to_le_bytes());
        hasher.update(&[self.record_type as u8]);
        hasher.update(&self.intent_id.to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        // Add payload hash
        if let Ok(payload_bytes) = bincode::serialize(&self.payload) {
            hasher.update(&payload_bytes);
        }
        hasher.finalize()
    }
    
    /// Verify checksum
    pub fn verify(&self) -> bool {
        let expected = {
            let mut temp = self.clone();
            temp.checksum = 0;
            temp.compute_checksum()
        };
        self.checksum == expected
    }
    
    /// Serialize to bytes
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

// ============================================================================
// WAL Writer
// ============================================================================

/// Configuration for WAL
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory for WAL files
    pub dir: PathBuf,
    /// Maximum WAL file size before rotation
    pub max_file_size: u64,
    /// Sync mode
    pub sync_mode: SyncMode,
    /// Buffer size for writes
    pub buffer_size: usize,
}

/// Sync mode for WAL writes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Sync every record (safest, slowest)
    EveryRecord,
    /// Sync on commit only
    OnCommit,
    /// Periodic sync (batched)
    Periodic,
    /// No sync (fastest, unsafe)
    None,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("./wal"),
            max_file_size: 64 * 1024 * 1024, // 64 MB
            sync_mode: SyncMode::OnCommit,
            buffer_size: 64 * 1024, // 64 KB
        }
    }
}

/// Write-Ahead Log writer
pub struct WalWriter {
    config: WalConfig,
    current_file: Mutex<Option<BufWriter<File>>>,
    current_lsn: AtomicU64,
    current_file_size: AtomicU64,
    current_file_id: AtomicU64,
}

impl WalWriter {
    /// Create new WAL writer
    pub fn new(config: WalConfig) -> io::Result<Self> {
        std::fs::create_dir_all(&config.dir)?;
        
        let writer = Self {
            config,
            current_file: Mutex::new(None),
            current_lsn: AtomicU64::new(1),
            current_file_size: AtomicU64::new(0),
            current_file_id: AtomicU64::new(1),
        };
        
        writer.rotate_if_needed()?;
        Ok(writer)
    }
    
    /// Get current LSN
    pub fn current_lsn(&self) -> Lsn {
        Lsn(self.current_lsn.load(Ordering::SeqCst))
    }
    
    /// Allocate next LSN
    fn next_lsn(&self) -> Lsn {
        Lsn(self.current_lsn.fetch_add(1, Ordering::SeqCst))
    }
    
    /// Rotate WAL file if needed
    fn rotate_if_needed(&self) -> io::Result<()> {
        let size = self.current_file_size.load(Ordering::Relaxed);
        
        if size >= self.config.max_file_size || self.current_file.lock().is_none() {
            let file_id = self.current_file_id.fetch_add(1, Ordering::SeqCst);
            let path = self.config.dir.join(format!("wal_{:016x}.log", file_id));
            
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(&path)?;
            
            let mut writer = BufWriter::with_capacity(self.config.buffer_size, file);
            
            // Write header
            let header = b"SOCHWAL1"; // Magic + version
            writer.write_all(header)?;
            
            *self.current_file.lock() = Some(writer);
            self.current_file_size.store(8, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// Append a record to WAL
    pub fn append(&self, record: WalRecord) -> io::Result<Lsn> {
        self.rotate_if_needed()?;
        
        let bytes = record.to_bytes()?;
        let record_len = bytes.len() as u32;
        
        let mut file = self.current_file.lock();
        if let Some(ref mut writer) = *file {
            // Write length prefix
            writer.write_all(&record_len.to_le_bytes())?;
            // Write record
            writer.write_all(&bytes)?;
            
            // Sync based on mode
            match self.config.sync_mode {
                SyncMode::EveryRecord => writer.flush()?,
                SyncMode::OnCommit if record.record_type == WalRecordType::Commit => {
                    writer.flush()?;
                    writer.get_ref().sync_all()?;
                }
                _ => {}
            }
            
            self.current_file_size.fetch_add(4 + bytes.len() as u64, Ordering::Relaxed);
        }
        
        Ok(record.lsn)
    }
    
    /// Write intent record
    pub fn write_intent(&self, intent_id: u64, memory_id: &str, op_count: usize) -> io::Result<Lsn> {
        let lsn = self.next_lsn();
        let record = WalRecord::new(
            lsn,
            WalRecordType::Intent,
            intent_id,
            WalPayload::IntentStart {
                memory_id: memory_id.to_string(),
                op_count,
            },
        );
        self.append(record)
    }
    
    /// Write operation record
    pub fn write_operation(
        &self,
        intent_id: u64,
        op_index: usize,
        op_type: &str,
        key: &[u8],
        value: Option<&[u8]>,
    ) -> io::Result<Lsn> {
        let lsn = self.next_lsn();
        let record = WalRecord::new(
            lsn,
            WalRecordType::Operation,
            intent_id,
            WalPayload::Operation {
                op_index,
                op_type: op_type.to_string(),
                key: key.to_vec(),
                value: value.map(|v| v.to_vec()),
            },
        );
        self.append(record)
    }
    
    /// Write commit record
    pub fn write_commit(&self, intent_id: u64) -> io::Result<Lsn> {
        let lsn = self.next_lsn();
        let record = WalRecord::new(
            lsn,
            WalRecordType::Commit,
            intent_id,
            WalPayload::Commit,
        );
        self.append(record)
    }
    
    /// Write abort record
    pub fn write_abort(&self, intent_id: u64, reason: &str) -> io::Result<Lsn> {
        let lsn = self.next_lsn();
        let record = WalRecord::new(
            lsn,
            WalRecordType::Abort,
            intent_id,
            WalPayload::Abort {
                reason: reason.to_string(),
            },
        );
        self.append(record)
    }
    
    /// Write checkpoint record
    pub fn write_checkpoint(&self, last_committed_lsn: Lsn, intent_count: usize) -> io::Result<Lsn> {
        let lsn = self.next_lsn();
        let record = WalRecord::new(
            lsn,
            WalRecordType::Checkpoint,
            0,
            WalPayload::Checkpoint {
                last_committed_lsn,
                intent_count,
            },
        );
        self.append(record)?;
        
        // Force sync on checkpoint
        if let Some(ref mut writer) = *self.current_file.lock() {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        
        Ok(lsn)
    }
    
    /// Sync all pending writes
    pub fn sync(&self) -> io::Result<()> {
        if let Some(ref mut writer) = *self.current_file.lock() {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        Ok(())
    }
}

// ============================================================================
// WAL Reader
// ============================================================================

/// WAL reader for recovery
pub struct WalReader {
    dir: PathBuf,
}

impl WalReader {
    /// Create new reader
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }
    
    /// List WAL files in order
    pub fn list_files(&self) -> io::Result<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map(|e| e == "log").unwrap_or(false)
                    && p.file_name()
                        .map(|n| n.to_string_lossy().starts_with("wal_"))
                        .unwrap_or(false)
            })
            .collect();
        
        files.sort();
        Ok(files)
    }
    
    /// Read all records from a file
    pub fn read_file(&self, path: &Path) -> io::Result<Vec<WalRecord>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Verify header
        let mut header = [0u8; 8];
        reader.read_exact(&mut header)?;
        
        if &header != b"SOCHWAL1" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid WAL header",
            ));
        }
        
        let mut records = Vec::new();
        
        loop {
            // Read length prefix
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            
            let len = u32::from_le_bytes(len_buf) as usize;
            
            // Read record
            let mut record_buf = vec![0u8; len];
            reader.read_exact(&mut record_buf)?;
            
            let record = WalRecord::from_bytes(&record_buf)?;
            
            if !record.verify() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Checksum mismatch at LSN {:?}", record.lsn),
                ));
            }
            
            records.push(record);
        }
        
        Ok(records)
    }
    
    /// Read all records from all files
    pub fn read_all(&self) -> io::Result<Vec<WalRecord>> {
        let mut all_records = Vec::new();
        
        for path in self.list_files()? {
            let records = self.read_file(&path)?;
            all_records.extend(records);
        }
        
        // Sort by LSN
        all_records.sort_by_key(|r| r.lsn);
        
        Ok(all_records)
    }
    
    /// Read records since a checkpoint
    pub fn read_since(&self, checkpoint_lsn: Lsn) -> io::Result<Vec<WalRecord>> {
        let all = self.read_all()?;
        Ok(all.into_iter().filter(|r| r.lsn > checkpoint_lsn).collect())
    }
}

// ============================================================================
// Atomic Writer with WAL
// ============================================================================

/// Atomic multi-index writer with WAL discipline
pub struct WalAtomicWriter {
    wal: Arc<WalWriter>,
    next_intent_id: AtomicU64,
    /// Pending intents (not yet committed)
    pending: RwLock<HashMap<u64, PendingIntent>>,
}

/// A pending intent being processed
#[derive(Debug)]
struct PendingIntent {
    intent_id: u64,
    memory_id: String,
    start_lsn: Lsn,
    ops_completed: usize,
    total_ops: usize,
}

impl WalAtomicWriter {
    /// Create new atomic writer
    pub fn new(wal: Arc<WalWriter>) -> Self {
        Self {
            wal,
            next_intent_id: AtomicU64::new(1),
            pending: RwLock::new(HashMap::new()),
        }
    }
    
    /// Start an atomic operation
    pub fn begin(&self, memory_id: &str, op_count: usize) -> io::Result<u64> {
        let intent_id = self.next_intent_id.fetch_add(1, Ordering::SeqCst);
        
        // Write intent to WAL
        let lsn = self.wal.write_intent(intent_id, memory_id, op_count)?;
        
        // Track pending intent
        self.pending.write().insert(intent_id, PendingIntent {
            intent_id,
            memory_id: memory_id.to_string(),
            start_lsn: lsn,
            ops_completed: 0,
            total_ops: op_count,
        });
        
        Ok(intent_id)
    }
    
    /// Record an operation
    pub fn record_op(
        &self,
        intent_id: u64,
        op_type: &str,
        key: &[u8],
        value: Option<&[u8]>,
    ) -> io::Result<Lsn> {
        let op_index = {
            let mut pending = self.pending.write();
            let intent = pending.get_mut(&intent_id)
                .ok_or_else(|| io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Intent {} not found", intent_id),
                ))?;
            
            let idx = intent.ops_completed;
            intent.ops_completed += 1;
            idx
        };
        
        self.wal.write_operation(intent_id, op_index, op_type, key, value)
    }
    
    /// Commit an atomic operation
    pub fn commit(&self, intent_id: u64) -> io::Result<Lsn> {
        // Write commit to WAL
        let lsn = self.wal.write_commit(intent_id)?;
        
        // Remove from pending
        self.pending.write().remove(&intent_id);
        
        Ok(lsn)
    }
    
    /// Abort an atomic operation
    pub fn abort(&self, intent_id: u64, reason: &str) -> io::Result<Lsn> {
        // Write abort to WAL
        let lsn = self.wal.write_abort(intent_id, reason)?;
        
        // Remove from pending
        self.pending.write().remove(&intent_id);
        
        Ok(lsn)
    }
    
    /// Get pending intent count
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_wal_write_read() -> io::Result<()> {
        let tmp = TempDir::new()?;
        let config = WalConfig {
            dir: tmp.path().to_path_buf(),
            sync_mode: SyncMode::None,
            ..Default::default()
        };
        
        // Write some records
        let writer = WalWriter::new(config)?;
        
        let lsn1 = writer.write_intent(1, "memory1", 2)?;
        let lsn2 = writer.write_operation(1, 0, "PUT", b"key1", Some(b"value1"))?;
        let lsn3 = writer.write_operation(1, 1, "PUT", b"key2", Some(b"value2"))?;
        let lsn4 = writer.write_commit(1)?;
        
        writer.sync()?;
        
        // Read back
        let reader = WalReader::new(tmp.path());
        let records = reader.read_all()?;
        
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].lsn, lsn1);
        assert_eq!(records[3].lsn, lsn4);
        assert_eq!(records[3].record_type, WalRecordType::Commit);
        
        Ok(())
    }
    
    #[test]
    fn test_atomic_writer() -> io::Result<()> {
        let tmp = TempDir::new()?;
        let config = WalConfig {
            dir: tmp.path().to_path_buf(),
            sync_mode: SyncMode::None,
            ..Default::default()
        };
        
        let wal = Arc::new(WalWriter::new(config)?);
        let writer = WalAtomicWriter::new(wal.clone());
        
        // Start atomic operation
        let intent_id = writer.begin("test_memory", 2)?;
        assert_eq!(writer.pending_count(), 1);
        
        // Record operations
        writer.record_op(intent_id, "PUT", b"key1", Some(b"value1"))?;
        writer.record_op(intent_id, "PUT", b"key2", Some(b"value2"))?;
        
        // Commit
        writer.commit(intent_id)?;
        assert_eq!(writer.pending_count(), 0);
        
        Ok(())
    }
}
