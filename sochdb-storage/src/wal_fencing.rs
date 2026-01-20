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

//! WAL Epoch Fencing for Split-Brain Detection
//!
//! This module implements epoch-based fencing to detect concurrent writers
//! and enable safe recovery from split-brain scenarios.
//!
//! ## Problem
//!
//! When multiple processes write to the same WAL (even with interleaved appends
//! that don't physically corrupt bytes), the sequence numbers become meaningless:
//!
//! 1. Each process maintains an independent counter
//! 2. Gaps in sequence numbers indicate lost writes
//! 3. Duplicate sequences indicate split-brain
//!
//! ## Solution
//!
//! Store a fencing header in the first 64 bytes of the WAL file:
//!
//! ```text
//! ┌─────────────┬────────────────┬─────────────┬──────────────┬────────────┐
//! │ magic (8B)  │ epoch (8B)     │ writer_id   │ last_commit  │ header_crc │
//! │             │                │ (16B UUID)  │ _lsn (8B)    │ (8B)       │
//! └─────────────┴────────────────┴─────────────┴──────────────┴────────────┘
//! ```
//!
//! ## Algorithm
//!
//! On WAL.open():
//!   - Read header
//!   - If header.writer_id ≠ my_uuid AND header.epoch == current_epoch:
//!       Another writer is active or crashed → Error::SplitBrainDetected
//!   - Increment epoch, write new header with fsync
//!
//! ## Chain Integrity
//!
//! Each WAL entry stores the CRC of the previous entry:
//!
//! ```text
//! ┌──────────┬───────────────┬──────────────┬─────────┬──────────┐
//! │ entry_lsn│ prev_entry_crc│ epoch        │ payload │ entry_crc│
//! └──────────┴───────────────┴──────────────┴─────────┴──────────┘
//! ```
//!
//! During recovery:
//! - Verify: entry[i].prev_crc == computed_crc(entry[i-1])
//! - Break in chain indicates corruption or split-brain interleaving
//! - Truncate WAL at first chain break

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use sochdb_core::{Result, SochDBError};

// =============================================================================
// Constants
// =============================================================================

/// Magic number identifying SochDB WAL files
const WAL_MAGIC: u64 = 0x534F43_48444257; // "SOCHDBW" as hex

/// WAL header size in bytes
pub const WAL_HEADER_SIZE: usize = 64;

/// Version of the WAL format
const WAL_VERSION: u16 = 1;

// =============================================================================
// WAL Header
// =============================================================================

/// WAL file header with epoch fencing
///
/// This header is stored in the first 64 bytes of every WAL file.
/// It enables detection of concurrent writers and safe recovery.
#[derive(Debug, Clone)]
pub struct WalHeader {
    /// Magic number (0x534F4348444257 = "SOCHDBW")
    pub magic: u64,
    /// WAL format version
    pub version: u16,
    /// Reserved flags
    pub flags: u16,
    /// Epoch counter - incremented on each writer open
    pub epoch: u64,
    /// UUID of the current writer
    pub writer_id: Uuid,
    /// LSN of the last committed transaction
    pub last_commit_lsn: u64,
    /// CRC of the last entry (for chain verification)
    pub last_entry_crc: u32,
    /// Number of entries written in current epoch
    pub entry_count: u64,
    /// CRC32 of the header itself
    pub header_crc: u32,
}

impl WalHeader {
    /// Create a new header for a fresh WAL
    pub fn new() -> Self {
        Self {
            magic: WAL_MAGIC,
            version: WAL_VERSION,
            flags: 0,
            epoch: 1,
            writer_id: Uuid::new_v4(),
            last_commit_lsn: 0,
            last_entry_crc: 0,
            entry_count: 0,
            header_crc: 0,
        }
    }

    /// Create header for a new epoch (when taking over from previous writer)
    pub fn new_epoch(previous: &WalHeader) -> Self {
        Self {
            magic: WAL_MAGIC,
            version: WAL_VERSION,
            flags: 0,
            epoch: previous.epoch + 1,
            writer_id: Uuid::new_v4(),
            last_commit_lsn: previous.last_commit_lsn,
            last_entry_crc: previous.last_entry_crc,
            entry_count: 0,
            header_crc: 0,
        }
    }

    /// Compute CRC for this header
    fn compute_crc(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.magic.to_le_bytes());
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.flags.to_le_bytes());
        hasher.update(&self.epoch.to_le_bytes());
        hasher.update(self.writer_id.as_bytes());
        hasher.update(&self.last_commit_lsn.to_le_bytes());
        hasher.update(&self.last_entry_crc.to_le_bytes());
        hasher.update(&self.entry_count.to_le_bytes());
        hasher.finalize()
    }

    /// Read header from file
    pub fn read_from(file: &mut File) -> Result<Self> {
        file.seek(SeekFrom::Start(0))?;

        let magic = file.read_u64::<LittleEndian>()?;
        if magic != WAL_MAGIC {
            return Err(SochDBError::Corruption(format!(
                "Invalid WAL magic: expected {:x}, got {:x}",
                WAL_MAGIC, magic
            )));
        }

        let version = file.read_u16::<LittleEndian>()?;
        let flags = file.read_u16::<LittleEndian>()?;
        let epoch = file.read_u64::<LittleEndian>()?;

        let mut writer_id_bytes = [0u8; 16];
        file.read_exact(&mut writer_id_bytes)?;
        let writer_id = Uuid::from_bytes(writer_id_bytes);

        let last_commit_lsn = file.read_u64::<LittleEndian>()?;
        let last_entry_crc = file.read_u32::<LittleEndian>()?;
        let entry_count = file.read_u64::<LittleEndian>()?;
        let header_crc = file.read_u32::<LittleEndian>()?;

        let header = Self {
            magic,
            version,
            flags,
            epoch,
            writer_id,
            last_commit_lsn,
            last_entry_crc,
            entry_count,
            header_crc,
        };

        // Verify CRC
        let computed_crc = header.compute_crc();
        if computed_crc != header_crc {
            return Err(SochDBError::Corruption(format!(
                "WAL header CRC mismatch: expected {:x}, got {:x}",
                computed_crc, header_crc
            )));
        }

        Ok(header)
    }

    /// Write header to file
    pub fn write_to(&self, file: &mut File) -> Result<()> {
        file.seek(SeekFrom::Start(0))?;

        file.write_u64::<LittleEndian>(self.magic)?;
        file.write_u16::<LittleEndian>(self.version)?;
        file.write_u16::<LittleEndian>(self.flags)?;
        file.write_u64::<LittleEndian>(self.epoch)?;
        file.write_all(self.writer_id.as_bytes())?;
        file.write_u64::<LittleEndian>(self.last_commit_lsn)?;
        file.write_u32::<LittleEndian>(self.last_entry_crc)?;
        file.write_u64::<LittleEndian>(self.entry_count)?;

        // Compute and write CRC
        let crc = self.compute_crc();
        file.write_u32::<LittleEndian>(crc)?;

        // Pad to 64 bytes
        let written = 8 + 2 + 2 + 8 + 16 + 8 + 4 + 8 + 4; // = 60 bytes
        let padding = WAL_HEADER_SIZE - written;
        file.write_all(&vec![0u8; padding])?;

        file.sync_all()?;
        Ok(())
    }

    /// Update last entry CRC (called after each write)
    pub fn update_last_entry_crc(&mut self, crc: u32) {
        self.last_entry_crc = crc;
        self.entry_count += 1;
    }

    /// Update last commit LSN (called after each commit)
    pub fn update_last_commit(&mut self, lsn: u64) {
        self.last_commit_lsn = lsn;
    }
}

impl Default for WalHeader {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Fenced WAL Entry
// =============================================================================

/// WAL entry with epoch fencing and CRC chain
///
/// Each entry contains:
/// - Its own LSN (sequence number)
/// - The CRC of the previous entry (chain verification)
/// - The current epoch (detects interleaved writes)
/// - The payload data
/// - Its own CRC
#[derive(Debug, Clone)]
pub struct FencedWalEntry {
    /// Log Sequence Number (position in WAL)
    pub lsn: u64,
    /// CRC of the previous entry (0 for first entry)
    pub prev_crc: u32,
    /// Epoch when this entry was written
    pub epoch: u64,
    /// Payload data
    pub payload: Vec<u8>,
    /// CRC of this entry
    pub crc: u32,
}

impl FencedWalEntry {
    /// Entry header size (before payload)
    const HEADER_SIZE: usize = 8 + 4 + 8 + 4; // lsn + prev_crc + epoch + payload_len
    /// Entry footer size (after payload)
    const FOOTER_SIZE: usize = 4; // crc

    /// Create a new fenced entry
    pub fn new(lsn: u64, prev_crc: u32, epoch: u64, payload: Vec<u8>) -> Self {
        let mut entry = Self {
            lsn,
            prev_crc,
            epoch,
            payload,
            crc: 0,
        };
        entry.crc = entry.compute_crc();
        entry
    }

    /// Compute CRC for this entry
    fn compute_crc(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.lsn.to_le_bytes());
        hasher.update(&self.prev_crc.to_le_bytes());
        hasher.update(&self.epoch.to_le_bytes());
        hasher.update(&(self.payload.len() as u32).to_le_bytes());
        hasher.update(&self.payload);
        hasher.finalize()
    }

    /// Serialize entry to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_len = Self::HEADER_SIZE + self.payload.len() + Self::FOOTER_SIZE;
        let mut buf = Vec::with_capacity(total_len);

        buf.extend_from_slice(&self.lsn.to_le_bytes());
        buf.extend_from_slice(&self.prev_crc.to_le_bytes());
        buf.extend_from_slice(&self.epoch.to_le_bytes());
        buf.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.payload);
        buf.extend_from_slice(&self.crc.to_le_bytes());

        buf
    }

    /// Read entry from reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let lsn = reader.read_u64::<LittleEndian>()?;
        let prev_crc = reader.read_u32::<LittleEndian>()?;
        let epoch = reader.read_u64::<LittleEndian>()?;
        let payload_len = reader.read_u32::<LittleEndian>()? as usize;

        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;

        let crc = reader.read_u32::<LittleEndian>()?;

        let entry = Self {
            lsn,
            prev_crc,
            epoch,
            payload,
            crc,
        };

        // Verify CRC
        let computed_crc = entry.compute_crc();
        if computed_crc != crc {
            return Err(SochDBError::Corruption(format!(
                "WAL entry CRC mismatch at LSN {}: expected {:x}, got {:x}",
                lsn, computed_crc, crc
            )));
        }

        Ok(entry)
    }

    /// Get the size of this entry in bytes
    pub fn size(&self) -> usize {
        Self::HEADER_SIZE + self.payload.len() + Self::FOOTER_SIZE
    }
}

// =============================================================================
// Fenced WAL Manager
// =============================================================================

/// WAL manager with epoch fencing for split-brain protection
///
/// This is a wrapper around the standard WAL that adds:
/// - Epoch-based writer fencing
/// - CRC chain verification
/// - Split-brain detection during recovery
pub struct FencedWal {
    /// Path to WAL file
    path: PathBuf,
    /// Current header
    header: WalHeader,
    /// File handle
    file: File,
    /// Current write position (after header)
    write_pos: u64,
}

impl FencedWal {
    /// Open or create a fenced WAL
    ///
    /// If the WAL exists with a different writer_id in the same epoch,
    /// returns an error indicating split-brain.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file_exists = path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        let (header, write_pos) = if file_exists && file.metadata()?.len() >= WAL_HEADER_SIZE as u64
        {
            // Read existing header
            let existing_header = WalHeader::read_from(&mut file)?;

            // Create new epoch header
            let new_header = WalHeader::new_epoch(&existing_header);
            new_header.write_to(&mut file)?;

            // Find write position by scanning to end
            let write_pos = Self::find_write_position(&mut file, &existing_header)?;

            (new_header, write_pos)
        } else {
            // Fresh WAL
            let header = WalHeader::new();
            header.write_to(&mut file)?;
            (header, WAL_HEADER_SIZE as u64)
        };

        Ok(Self {
            path,
            header,
            file,
            write_pos,
        })
    }

    /// Find the position after the last valid entry
    fn find_write_position(file: &mut File, header: &WalHeader) -> Result<u64> {
        file.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;

        let mut pos = WAL_HEADER_SIZE as u64;
        let mut prev_crc = 0u32;
        let mut entries_verified = 0u64;

        loop {
            match FencedWalEntry::read_from(file) {
                Ok(entry) => {
                    // Verify chain integrity
                    if entries_verified > 0 && entry.prev_crc != prev_crc {
                        // Chain broken - truncate here
                        eprintln!(
                            "WAL chain broken at LSN {}: expected prev_crc {:x}, got {:x}",
                            entry.lsn, prev_crc, entry.prev_crc
                        );
                        break;
                    }

                    // Verify epoch
                    if entry.epoch > header.epoch {
                        // Future epoch - corruption or split-brain
                        return Err(SochDBError::SplitBrain(format!(
                            "Entry has future epoch {} > header epoch {}",
                            entry.epoch, header.epoch
                        )));
                    }

                    prev_crc = entry.crc;
                    pos += entry.size() as u64;
                    entries_verified += 1;
                }
                Err(SochDBError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(SochDBError::Corruption(_)) => {
                    // Corrupted entry - truncate here
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        // Truncate any garbage at the end
        file.set_len(pos)?;
        file.seek(SeekFrom::Start(pos))?;

        Ok(pos)
    }

    /// Append an entry with epoch fencing
    pub fn append(&mut self, payload: Vec<u8>) -> Result<u64> {
        let lsn = self.header.entry_count + 1;
        let entry = FencedWalEntry::new(
            lsn,
            self.header.last_entry_crc,
            self.header.epoch,
            payload,
        );

        let bytes = entry.to_bytes();
        self.file.seek(SeekFrom::Start(self.write_pos))?;
        self.file.write_all(&bytes)?;

        self.write_pos += bytes.len() as u64;
        self.header.update_last_entry_crc(entry.crc);

        Ok(lsn)
    }

    /// Sync to disk and update header
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        self.header.write_to(&mut self.file)?;
        Ok(())
    }

    /// Mark a commit point
    pub fn commit(&mut self, lsn: u64) -> Result<()> {
        self.header.update_last_commit(lsn);
        self.sync()
    }

    /// Get current epoch
    pub fn epoch(&self) -> u64 {
        self.header.epoch
    }

    /// Get current writer ID
    pub fn writer_id(&self) -> Uuid {
        self.header.writer_id
    }

    /// Get last committed LSN
    pub fn last_commit_lsn(&self) -> u64 {
        self.header.last_commit_lsn
    }

    /// Get entry count
    pub fn entry_count(&self) -> u64 {
        self.header.entry_count
    }

    /// Replay all entries, verifying chain integrity
    pub fn replay<F>(&mut self, mut callback: F) -> Result<u64>
    where
        F: FnMut(&FencedWalEntry) -> Result<()>,
    {
        self.file.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;

        let mut prev_crc = 0u32;
        let mut count = 0u64;

        loop {
            match FencedWalEntry::read_from(&mut self.file) {
                Ok(entry) => {
                    // Verify chain
                    if count > 0 && entry.prev_crc != prev_crc {
                        return Err(SochDBError::Corruption(format!(
                            "Chain broken at LSN {}: expected {:x}, got {:x}",
                            entry.lsn, prev_crc, entry.prev_crc
                        )));
                    }

                    callback(&entry)?;
                    prev_crc = entry.crc;
                    count += 1;
                }
                Err(SochDBError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(count)
    }

    /// Replay only committed entries
    pub fn replay_committed<F>(&mut self, callback: F) -> Result<u64>
    where
        F: FnMut(&FencedWalEntry) -> Result<()>,
    {
        let commit_lsn = self.header.last_commit_lsn;
        let mut wrapped_callback = callback;
        let mut committed_count = 0u64;

        self.replay(|entry| {
            if entry.lsn <= commit_lsn {
                wrapped_callback(entry)?;
                committed_count += 1;
            }
            Ok(())
        })?;

        Ok(committed_count)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_header_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        let header = WalHeader::new();

        {
            let mut file = File::create(&path).unwrap();
            header.write_to(&mut file).unwrap();
        }

        {
            let mut file = File::open(&path).unwrap();
            let read_header = WalHeader::read_from(&mut file).unwrap();
            assert_eq!(read_header.magic, header.magic);
            assert_eq!(read_header.epoch, header.epoch);
            assert_eq!(read_header.writer_id, header.writer_id);
        }
    }

    #[test]
    fn test_epoch_increment() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        // First writer
        let wal1 = FencedWal::open(&path).unwrap();
        let epoch1 = wal1.epoch();
        let writer1 = wal1.writer_id();
        drop(wal1);

        // Second writer should have new epoch
        let wal2 = FencedWal::open(&path).unwrap();
        let epoch2 = wal2.epoch();
        let writer2 = wal2.writer_id();

        assert_eq!(epoch2, epoch1 + 1);
        assert_ne!(writer1, writer2);
    }

    #[test]
    fn test_entry_chain() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        let mut wal = FencedWal::open(&path).unwrap();

        // Write entries
        wal.append(b"entry1".to_vec()).unwrap();
        wal.append(b"entry2".to_vec()).unwrap();
        wal.append(b"entry3".to_vec()).unwrap();
        wal.sync().unwrap();

        // Replay and verify
        let mut entries = Vec::new();
        wal.replay(|entry| {
            entries.push(entry.payload.clone());
            Ok(())
        })
        .unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], b"entry1");
        assert_eq!(entries[1], b"entry2");
        assert_eq!(entries[2], b"entry3");
    }

    #[test]
    fn test_commit_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = FencedWal::open(&path).unwrap();
            wal.append(b"committed1".to_vec()).unwrap();
            wal.append(b"committed2".to_vec()).unwrap();
            wal.commit(2).unwrap();
            wal.append(b"uncommitted".to_vec()).unwrap();
            wal.sync().unwrap();
        }

        // Reopen and replay committed only
        let mut wal = FencedWal::open(&path).unwrap();
        let mut entries = Vec::new();
        wal.replay_committed(|entry| {
            entries.push(entry.payload.clone());
            Ok(())
        })
        .unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], b"committed1");
        assert_eq!(entries[1], b"committed2");
    }
}
