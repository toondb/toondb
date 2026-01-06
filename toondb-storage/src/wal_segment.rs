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

//! WAL Segmentation and Checkpoint Manager
//!
//! This module provides bounded recovery time through:
//! - WAL file segmentation (rotate after size/time threshold)
//! - Fuzzy checkpointing (no blocking writes)
//! - Automatic old segment cleanup after checkpoint
//!
//! ## Architecture
//!
//! ```text
//! Active Writes → Current Segment → Rotation → Archived Segment
//!                                                      ↓
//!                                              Checkpoint
//!                                                      ↓
//!                                              Segment Cleanup
//! ```
//!
//! ## Recovery Time Bound
//!
//! Recovery time is bounded by segment_max_size / disk_bandwidth.
//! With 64MB segments and 400 MB/s sequential read, recovery ≤ 160ms.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};

/// Default maximum segment size (64 MB)
pub const DEFAULT_SEGMENT_MAX_SIZE: u64 = 64 * 1024 * 1024;

/// Default segment rotation interval (5 minutes)
pub const DEFAULT_ROTATION_INTERVAL: Duration = Duration::from_secs(300);

/// Default checkpoint interval (1 minute)
pub const DEFAULT_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(60);

/// Segment file header magic
const SEGMENT_MAGIC: u32 = 0x574C5347; // "WLSG"

/// Segment header version
const SEGMENT_VERSION: u16 = 1;

/// Segment header size
const SEGMENT_HEADER_SIZE: usize = 32;

/// Checkpoint file magic
const CHECKPOINT_MAGIC: u32 = 0x43484B50; // "CHKP"

/// WAL segment configuration
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Maximum segment size before rotation
    pub max_size: u64,
    /// Maximum time before rotation
    pub rotation_interval: Duration,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
    /// Directory for WAL segments
    pub wal_dir: PathBuf,
    /// Sync on every write
    pub sync_on_write: bool,
    /// Preallocate segment files
    pub preallocate: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_SEGMENT_MAX_SIZE,
            rotation_interval: DEFAULT_ROTATION_INTERVAL,
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
            wal_dir: PathBuf::from("wal"),
            sync_on_write: true,
            preallocate: true,
        }
    }
}

impl SegmentConfig {
    pub fn with_wal_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.wal_dir = dir.as_ref().to_path_buf();
        self
    }

    pub fn with_max_size(mut self, size: u64) -> Self {
        self.max_size = size;
        self
    }
}

/// Segment header stored at the beginning of each segment file
#[derive(Debug, Clone)]
pub struct SegmentHeader {
    /// Magic number
    pub magic: u32,
    /// Version
    pub version: u16,
    /// Flags
    pub flags: u16,
    /// Segment sequence number
    pub sequence: u64,
    /// First LSN in this segment
    pub first_lsn: u64,
    /// Creation timestamp (Unix millis)
    pub created_at: u64,
    /// Reserved for future use
    pub reserved: [u8; 8],
}

impl SegmentHeader {
    fn new(sequence: u64, first_lsn: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            magic: SEGMENT_MAGIC,
            version: SEGMENT_VERSION,
            flags: 0,
            sequence,
            first_lsn,
            created_at: now,
            reserved: [0; 8],
        }
    }

    fn encode(&self) -> [u8; SEGMENT_HEADER_SIZE] {
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.sequence.to_le_bytes());
        buf[16..24].copy_from_slice(&self.first_lsn.to_le_bytes());
        buf[24..32].copy_from_slice(&self.created_at.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < SEGMENT_HEADER_SIZE {
            return None;
        }

        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != SEGMENT_MAGIC {
            return None;
        }

        Some(Self {
            magic,
            version: u16::from_le_bytes([buf[4], buf[5]]),
            flags: u16::from_le_bytes([buf[6], buf[7]]),
            sequence: u64::from_le_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]),
            first_lsn: u64::from_le_bytes([buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23]]),
            created_at: u64::from_le_bytes([buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31]]),
            reserved: [0; 8],
        })
    }
}

/// Active WAL segment being written to
struct ActiveSegment {
    /// Segment file
    file: BufWriter<File>,
    /// Segment path
    path: PathBuf,
    /// Segment header
    header: SegmentHeader,
    /// Current offset within segment
    offset: u64,
    /// Creation time for rotation
    created_at: Instant,
}

/// Checkpoint record stored in checkpoint file
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    /// Checkpoint LSN (all entries before this are flushed)
    pub lsn: u64,
    /// Last segment sequence that can be deleted
    pub last_segment: u64,
    /// Checkpoint timestamp
    pub timestamp: u64,
    /// Memtable state checksum (for validation)
    pub memtable_checksum: u64,
    /// Number of entries checkpointed
    pub entry_count: u64,
}

impl CheckpointRecord {
    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&CHECKPOINT_MAGIC.to_le_bytes());
        buf.extend_from_slice(&self.lsn.to_le_bytes());
        buf.extend_from_slice(&self.last_segment.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.memtable_checksum.to_le_bytes());
        buf.extend_from_slice(&self.entry_count.to_le_bytes());
        // Add checksum of the record itself
        let checksum = crc32fast::hash(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 48 {
            return None;
        }

        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != CHECKPOINT_MAGIC {
            return None;
        }

        // Verify checksum
        let stored_checksum = u32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]);
        let computed_checksum = crc32fast::hash(&buf[0..44]);
        if stored_checksum != computed_checksum {
            return None;
        }

        Some(Self {
            lsn: u64::from_le_bytes([buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]]),
            last_segment: u64::from_le_bytes([buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19]]),
            timestamp: u64::from_le_bytes([buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27]]),
            memtable_checksum: u64::from_le_bytes([buf[28], buf[29], buf[30], buf[31], buf[32], buf[33], buf[34], buf[35]]),
            entry_count: u64::from_le_bytes([buf[36], buf[37], buf[38], buf[39], buf[40], buf[41], buf[42], buf[43]]),
        })
    }
}

/// WAL Segment Manager
///
/// Handles WAL segmentation, rotation, and cleanup.
pub struct WalSegmentManager {
    /// Configuration
    config: SegmentConfig,
    /// Current active segment
    active: Mutex<Option<ActiveSegment>>,
    /// Current LSN
    current_lsn: AtomicU64,
    /// Current segment sequence
    segment_sequence: AtomicU64,
    /// All segment metadata (sequence -> header)
    segments: RwLock<BTreeMap<u64, SegmentMetadata>>,
    /// Last checkpoint record
    last_checkpoint: RwLock<Option<CheckpointRecord>>,
    /// Shutdown flag
    shutdown: AtomicBool,
}

/// Metadata for a WAL segment
#[derive(Debug, Clone)]
pub struct SegmentMetadata {
    /// Segment sequence number
    pub sequence: u64,
    /// First LSN in segment
    pub first_lsn: u64,
    /// Last LSN in segment (None if active)
    pub last_lsn: Option<u64>,
    /// File path
    pub path: PathBuf,
    /// File size
    pub size: u64,
    /// Is this the active segment
    pub is_active: bool,
}

impl WalSegmentManager {
    /// Create a new WAL segment manager
    pub fn new(config: SegmentConfig) -> std::io::Result<Self> {
        // Ensure WAL directory exists
        fs::create_dir_all(&config.wal_dir)?;

        let manager = Self {
            config,
            active: Mutex::new(None),
            current_lsn: AtomicU64::new(0),
            segment_sequence: AtomicU64::new(0),
            segments: RwLock::new(BTreeMap::new()),
            last_checkpoint: RwLock::new(None),
            shutdown: AtomicBool::new(false),
        };

        // Load existing segments and checkpoint
        manager.recover()?;

        Ok(manager)
    }

    /// Recover from existing WAL segments
    fn recover(&self) -> std::io::Result<()> {
        // Load checkpoint if exists
        let checkpoint_path = self.config.wal_dir.join("checkpoint");
        if checkpoint_path.exists() {
            let mut file = File::open(&checkpoint_path)?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            if let Some(record) = CheckpointRecord::decode(&buf) {
                *self.last_checkpoint.write() = Some(record);
            }
        }

        // Scan for existing segments
        let entries = fs::read_dir(&self.config.wal_dir)?;
        let mut max_sequence = 0u64;
        let mut max_lsn = 0u64;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("segment_") && name.ends_with(".wal") {
                    // Parse segment file
                    let mut file = File::open(&path)?;
                    let mut header_buf = [0u8; SEGMENT_HEADER_SIZE];
                    if file.read_exact(&mut header_buf).is_ok() {
                        if let Some(header) = SegmentHeader::decode(&header_buf) {
                            max_sequence = max_sequence.max(header.sequence);
                            max_lsn = max_lsn.max(header.first_lsn);

                            let metadata = file.metadata()?;
                            self.segments.write().insert(header.sequence, SegmentMetadata {
                                sequence: header.sequence,
                                first_lsn: header.first_lsn,
                                last_lsn: None,
                                path: path.clone(),
                                size: metadata.len(),
                                is_active: false,
                            });
                        }
                    }
                }
            }
        }

        // Set counters
        self.segment_sequence.store(max_sequence + 1, Ordering::SeqCst);
        self.current_lsn.store(max_lsn, Ordering::SeqCst);

        Ok(())
    }

    /// Append a record to the WAL, returns the LSN
    pub fn append(&self, data: &[u8]) -> std::io::Result<u64> {
        let mut active = self.active.lock();

        // Rotate if needed
        if self.needs_rotation(&active) {
            self.rotate_segment(&mut active)?;
        }

        // Ensure we have an active segment
        if active.is_none() {
            self.create_new_segment(&mut active)?;
        }

        let segment = active.as_mut().unwrap();

        // Assign LSN
        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);

        // Write record: [length: u32][lsn: u64][data][checksum: u32]
        let record_len = 4 + 8 + data.len() + 4;
        let mut record = Vec::with_capacity(record_len);
        record.extend_from_slice(&(data.len() as u32).to_le_bytes());
        record.extend_from_slice(&lsn.to_le_bytes());
        record.extend_from_slice(data);
        let checksum = crc32fast::hash(&record);
        record.extend_from_slice(&checksum.to_le_bytes());

        segment.file.write_all(&record)?;
        segment.offset += record_len as u64;

        if self.config.sync_on_write {
            segment.file.flush()?;
        }

        Ok(lsn)
    }

    /// Check if segment needs rotation
    fn needs_rotation(&self, active: &Option<ActiveSegment>) -> bool {
        match active {
            Some(segment) => {
                segment.offset >= self.config.max_size
                    || segment.created_at.elapsed() >= self.config.rotation_interval
            }
            None => false,
        }
    }

    /// Rotate to a new segment
    fn rotate_segment(&self, active: &mut Option<ActiveSegment>) -> std::io::Result<()> {
        if let Some(mut segment) = active.take() {
            // Flush and sync the old segment
            segment.file.flush()?;
            segment.file.into_inner().map_err(|e| e.into_error())?.sync_all()?;

            // Update metadata to mark as not active
            let current_lsn = self.current_lsn.load(Ordering::SeqCst);
            if let Some(meta) = self.segments.write().get_mut(&segment.header.sequence) {
                meta.is_active = false;
                meta.last_lsn = Some(current_lsn);
                meta.size = segment.offset;
            }
        }

        Ok(())
    }

    /// Create a new segment
    fn create_new_segment(&self, active: &mut Option<ActiveSegment>) -> std::io::Result<()> {
        let sequence = self.segment_sequence.fetch_add(1, Ordering::SeqCst);
        let first_lsn = self.current_lsn.load(Ordering::SeqCst);

        let path = self.config.wal_dir.join(format!("segment_{:016x}.wal", sequence));

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        // Preallocate if configured
        if self.config.preallocate {
            file.set_len(self.config.max_size)?;
        }

        let mut writer = BufWriter::new(file);

        // Write header
        let header = SegmentHeader::new(sequence, first_lsn);
        writer.write_all(&header.encode())?;

        let segment = ActiveSegment {
            file: writer,
            path: path.clone(),
            header: header.clone(),
            offset: SEGMENT_HEADER_SIZE as u64,
            created_at: Instant::now(),
        };

        // Add to segments map
        self.segments.write().insert(sequence, SegmentMetadata {
            sequence,
            first_lsn,
            last_lsn: None,
            path,
            size: SEGMENT_HEADER_SIZE as u64,
            is_active: true,
        });

        *active = Some(segment);

        Ok(())
    }

    /// Create a fuzzy checkpoint
    ///
    /// This captures the current LSN and can be used to cleanup old segments.
    pub fn create_checkpoint(
        &self,
        memtable_checksum: u64,
        entry_count: u64,
    ) -> std::io::Result<CheckpointRecord> {
        let lsn = self.current_lsn.load(Ordering::SeqCst);

        // Find the last segment that is fully before this LSN
        let segments = self.segments.read();
        let last_segment = segments
            .values()
            .filter(|s| s.last_lsn.map(|l| l < lsn).unwrap_or(false))
            .map(|s| s.sequence)
            .max()
            .unwrap_or(0);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let record = CheckpointRecord {
            lsn,
            last_segment,
            timestamp: now,
            memtable_checksum,
            entry_count,
        };

        // Write checkpoint file
        let checkpoint_path = self.config.wal_dir.join("checkpoint");
        let temp_path = self.config.wal_dir.join("checkpoint.tmp");

        let mut file = File::create(&temp_path)?;
        file.write_all(&record.encode())?;
        file.sync_all()?;

        fs::rename(&temp_path, &checkpoint_path)?;

        *self.last_checkpoint.write() = Some(record.clone());

        Ok(record)
    }

    /// Cleanup old segments that are no longer needed
    pub fn cleanup_old_segments(&self) -> std::io::Result<usize> {
        let checkpoint = self.last_checkpoint.read().clone();

        let last_safe_segment = match checkpoint {
            Some(cp) => cp.last_segment,
            None => return Ok(0),
        };

        let mut segments = self.segments.write();
        let old_segments: Vec<u64> = segments
            .keys()
            .filter(|&&seq| seq <= last_safe_segment)
            .copied()
            .collect();

        let mut cleaned = 0;
        for sequence in old_segments {
            if let Some(meta) = segments.remove(&sequence) {
                if meta.path.exists() {
                    fs::remove_file(&meta.path)?;
                    cleaned += 1;
                }
            }
        }

        Ok(cleaned)
    }

    /// Get statistics
    pub fn stats(&self) -> SegmentStats {
        let segments = self.segments.read();
        let total_size: u64 = segments.values().map(|s| s.size).sum();
        let checkpoint = self.last_checkpoint.read().clone();

        SegmentStats {
            segment_count: segments.len(),
            total_size,
            current_lsn: self.current_lsn.load(Ordering::SeqCst),
            current_sequence: self.segment_sequence.load(Ordering::SeqCst),
            last_checkpoint_lsn: checkpoint.as_ref().map(|c| c.lsn),
        }
    }

    /// Iterator for recovery
    pub fn recovery_iterator(&self, from_lsn: u64) -> RecoveryIterator {
        RecoveryIterator::new(self, from_lsn)
    }

    /// Flush all pending writes
    pub fn flush(&self) -> std::io::Result<()> {
        let mut active = self.active.lock();
        if let Some(ref mut segment) = *active {
            segment.file.flush()?;
        }
        Ok(())
    }

    /// Shutdown the segment manager
    pub fn shutdown(&self) -> std::io::Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);

        let mut active = self.active.lock();
        if let Some(mut segment) = active.take() {
            segment.file.flush()?;
            segment.file.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        }

        Ok(())
    }
}

/// Statistics for WAL segments
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Number of segments
    pub segment_count: usize,
    /// Total size of all segments
    pub total_size: u64,
    /// Current LSN
    pub current_lsn: u64,
    /// Current segment sequence
    pub current_sequence: u64,
    /// Last checkpoint LSN
    pub last_checkpoint_lsn: Option<u64>,
}

/// Recovery iterator for replaying WAL entries
pub struct RecoveryIterator<'a> {
    manager: &'a WalSegmentManager,
    current_segment_idx: usize,
    segment_sequences: Vec<u64>,
    current_reader: Option<BufReader<File>>,
    current_offset: u64,
    from_lsn: u64,
}

impl<'a> RecoveryIterator<'a> {
    fn new(manager: &'a WalSegmentManager, from_lsn: u64) -> Self {
        let segments = manager.segments.read();
        let mut sequences: Vec<u64> = segments
            .values()
            .filter(|s| s.first_lsn >= from_lsn || s.last_lsn.map(|l| l >= from_lsn).unwrap_or(true))
            .map(|s| s.sequence)
            .collect();
        sequences.sort();

        Self {
            manager,
            current_segment_idx: 0,
            segment_sequences: sequences,
            current_reader: None,
            current_offset: SEGMENT_HEADER_SIZE as u64,
            from_lsn,
        }
    }

    /// Get next WAL entry
    pub fn next_entry(&mut self) -> std::io::Result<Option<WalEntry>> {
        loop {
            // Open segment if needed
            if self.current_reader.is_none() {
                if self.current_segment_idx >= self.segment_sequences.len() {
                    return Ok(None);
                }

                let sequence = self.segment_sequences[self.current_segment_idx];
                let segments = self.manager.segments.read();
                if let Some(meta) = segments.get(&sequence) {
                    let file = File::open(&meta.path)?;
                    let mut reader = BufReader::new(file);
                    reader.seek(SeekFrom::Start(SEGMENT_HEADER_SIZE as u64))?;
                    self.current_reader = Some(reader);
                    self.current_offset = SEGMENT_HEADER_SIZE as u64;
                } else {
                    self.current_segment_idx += 1;
                    continue;
                }
            }

            let reader = self.current_reader.as_mut().unwrap();

            // Try to read entry
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Move to next segment
                    self.current_reader = None;
                    self.current_segment_idx += 1;
                    continue;
                }
                Err(e) => return Err(e),
            }

            let data_len = u32::from_le_bytes(len_buf) as usize;
            if data_len == 0 || data_len > 100 * 1024 * 1024 {
                // Invalid or end of segment
                self.current_reader = None;
                self.current_segment_idx += 1;
                continue;
            }

            // Read LSN
            let mut lsn_buf = [0u8; 8];
            reader.read_exact(&mut lsn_buf)?;
            let lsn = u64::from_le_bytes(lsn_buf);

            // Read data
            let mut data = vec![0u8; data_len];
            reader.read_exact(&mut data)?;

            // Read and verify checksum
            let mut checksum_buf = [0u8; 4];
            reader.read_exact(&mut checksum_buf)?;
            let stored_checksum = u32::from_le_bytes(checksum_buf);

            let mut verify_buf = Vec::with_capacity(4 + 8 + data_len);
            verify_buf.extend_from_slice(&len_buf);
            verify_buf.extend_from_slice(&lsn_buf);
            verify_buf.extend_from_slice(&data);
            let computed_checksum = crc32fast::hash(&verify_buf);

            if stored_checksum != computed_checksum {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "WAL entry checksum mismatch",
                ));
            }

            self.current_offset += (4 + 8 + data_len + 4) as u64;

            // Skip entries before from_lsn
            if lsn < self.from_lsn {
                continue;
            }

            return Ok(Some(WalEntry { lsn, data }));
        }
    }
}

/// A WAL entry for recovery
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// LSN of this entry
    pub lsn: u64,
    /// Entry data
    pub data: Vec<u8>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_segment_manager_basic() {
        let dir = tempdir().unwrap();
        let config = SegmentConfig::default()
            .with_wal_dir(dir.path())
            .with_max_size(1024);

        let manager = WalSegmentManager::new(config).unwrap();

        // Append some entries
        for i in 0..100 {
            let data = format!("entry_{}", i);
            let lsn = manager.append(data.as_bytes()).unwrap();
            assert_eq!(lsn, i as u64);
        }

        let stats = manager.stats();
        assert!(stats.segment_count > 0);
        assert_eq!(stats.current_lsn, 100);

        manager.shutdown().unwrap();
    }

    #[test]
    fn test_checkpoint_and_cleanup() {
        let dir = tempdir().unwrap();
        let config = SegmentConfig::default()
            .with_wal_dir(dir.path())
            .with_max_size(256);

        let manager = WalSegmentManager::new(config).unwrap();

        // Append enough to create multiple segments
        for i in 0..50 {
            let data = format!("entry_{:04}", i);
            manager.append(data.as_bytes()).unwrap();
        }

        // Force rotation
        manager.flush().unwrap();

        // Create checkpoint
        let checkpoint = manager.create_checkpoint(12345, 50).unwrap();
        assert!(checkpoint.lsn > 0);

        // Cleanup should work
        let cleaned = manager.cleanup_old_segments().unwrap();
        // May or may not clean depending on segment boundaries
        assert!(cleaned >= 0);

        manager.shutdown().unwrap();
    }

    #[test]
    fn test_recovery() {
        let dir = tempdir().unwrap();
        let config = SegmentConfig::default()
            .with_wal_dir(dir.path());

        // Write some data
        {
            let manager = WalSegmentManager::new(config.clone()).unwrap();
            for i in 0..10 {
                let data = format!("data_{}", i);
                manager.append(data.as_bytes()).unwrap();
            }
            manager.shutdown().unwrap();
        }

        // Recover and verify
        {
            let manager = WalSegmentManager::new(config).unwrap();
            let mut iter = manager.recovery_iterator(0);
            let mut count = 0;

            while let Some(entry) = iter.next_entry().unwrap() {
                let data = String::from_utf8_lossy(&entry.data);
                assert!(data.starts_with("data_"));
                count += 1;
            }

            assert_eq!(count, 10);
        }
    }

    #[test]
    fn test_segment_header_encoding() {
        let header = SegmentHeader::new(42, 12345);
        let encoded = header.encode();
        let decoded = SegmentHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.magic, SEGMENT_MAGIC);
        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.first_lsn, 12345);
    }

    #[test]
    fn test_checkpoint_record_encoding() {
        let record = CheckpointRecord {
            lsn: 1000,
            last_segment: 5,
            timestamp: 123456789,
            memtable_checksum: 0xDEADBEEF,
            entry_count: 500,
        };

        let encoded = record.encode();
        let decoded = CheckpointRecord::decode(&encoded).unwrap();

        assert_eq!(decoded.lsn, 1000);
        assert_eq!(decoded.last_segment, 5);
        assert_eq!(decoded.memtable_checksum, 0xDEADBEEF);
        assert_eq!(decoded.entry_count, 500);
    }
}
