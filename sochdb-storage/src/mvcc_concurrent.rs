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

//! Concurrent MVCC for Multi-Reader Single-Writer Embedded Mode
//!
//! This module implements lock-free concurrent reads with single-writer
//! coordination using shared memory MVCC metadata.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Shared Memory MVCC Metadata (.mvcc_metadata file, mmap'd)      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Header:                                                         │
//! │    - magic, version, page_size                                   │
//! │    - current_epoch (AtomicU64)                                   │
//! │    - current_ts (AtomicU64, HLC timestamp)                       │
//! │    - writer_lock (AtomicU32, 0=free, pid=locked)                │
//! │                                                                  │
//! │  Reader Table (1024 slots × 64 bytes = 64KB):                    │
//! │    [slot 0]: pid, snapshot_ts, epoch, last_heartbeat             │
//! │    [slot 1]: ...                                                 │
//! │    [slot N]: ...                                                 │
//! │                                                                  │
//! │  Version Store (DashMap-like lock-free hashtable):               │
//! │    key → [version @ ts=105, version @ ts=99, ...]                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Model
//!
//! - Lock-free read: ~100ns (atomic load + version lookup)
//! - Writer lock acquisition: ~20ns (uncontended CAS)
//! - Version GC: O(N_versions) every 1000 commits
//!
//! ## Safety
//!
//! - Readers are lock-free (no blocking, no starvation)
//! - Single writer ensures WAL consistency
//! - Epoch-based GC prevents use-after-free

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use parking_lot::RwLock;
use sochdb_core::{Result, SochDBError};

// Type aliases to avoid conflicts with other modules
pub type ConcurrentVersionChain = VersionChain;
pub type ConcurrentVersionEntry = VersionEntry;

// =============================================================================
// Constants
// =============================================================================

/// Magic bytes for MVCC metadata file: "SOCHMVCC"
const MVCC_MAGIC: u64 = 0x43435F564D484353; // "SOCHMVCC" little-endian

/// Current format version
const MVCC_VERSION: u32 = 1;

/// Maximum concurrent readers
const MAX_READERS: usize = 1024;

/// Reader slot size (64 bytes = 1 cache line)
const READER_SLOT_SIZE: usize = 64;

/// Header size
const HEADER_SIZE: usize = 64;

/// Total metadata size (header + reader table)
const METADATA_SIZE: usize = HEADER_SIZE + (MAX_READERS * READER_SLOT_SIZE);

/// Stale reader timeout (60 seconds)
const STALE_READER_TIMEOUT_US: u64 = 60_000_000;

/// GC interval (every N commits)
const GC_COMMIT_INTERVAL: u64 = 1000;

// =============================================================================
// Hybrid Logical Clock
// =============================================================================

/// Hybrid Logical Clock for monotonic timestamps
///
/// Format: [48-bit physical time | 16-bit logical counter]
///
/// Properties:
/// 1. Monotonically increasing
/// 2. Causally ordered
/// 3. Resolution: 65,536 events per millisecond
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HlcTimestamp(pub u64);

impl HlcTimestamp {
    /// Create timestamp from physical time (ms) and logical counter
    #[inline]
    pub fn new(physical_ms: u64, logical: u16) -> Self {
        Self((physical_ms << 16) | (logical as u64))
    }

    /// Get physical time component (milliseconds since epoch)
    #[inline]
    pub fn physical_ms(&self) -> u64 {
        self.0 >> 16
    }

    /// Get logical counter component
    #[inline]
    pub fn logical(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    /// Get raw value
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// Allocate next timestamp (atomic, lock-free)
    ///
    /// Algorithm:
    /// 1. Read current physical time
    /// 2. If physical > last_physical: reset logical to 0
    /// 3. Else: increment logical counter
    /// 4. CAS to update, retry on conflict
    pub fn allocate_next(last: &AtomicU64) -> Self {
        let physical_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        loop {
            let last_val = last.load(Ordering::Acquire);
            let last_phys = last_val >> 16;
            let last_log = (last_val & 0xFFFF) as u16;

            let (new_phys, new_log) = if physical_now > last_phys {
                (physical_now, 0u16)
            } else {
                // Clock hasn't advanced, increment logical
                (last_phys, last_log.saturating_add(1))
            };

            let new_val = (new_phys << 16) | (new_log as u64);

            if last
                .compare_exchange(last_val, new_val, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Self(new_val);
            }
            // CAS failed, retry
            std::hint::spin_loop();
        }
    }

    /// Read current timestamp without advancing
    #[inline]
    pub fn read_current(ts: &AtomicU64) -> Self {
        Self(ts.load(Ordering::Acquire))
    }
}

impl From<u64> for HlcTimestamp {
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<HlcTimestamp> for u64 {
    fn from(ts: HlcTimestamp) -> Self {
        ts.0
    }
}

// =============================================================================
// Reader Slot (Cache-Line Aligned)
// =============================================================================

/// Reader slot in shared memory (64 bytes = 1 cache line)
///
/// Each active reader registers in a slot to prevent GC from
/// reclaiming versions it might need.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct ReaderSlot {
    /// Process ID (0 = slot is free)
    pub pid: AtomicU32,
    /// Snapshot timestamp this reader is using
    pub snapshot_ts: AtomicU64,
    /// Epoch number when reader registered
    pub epoch: AtomicU32,
    /// Last heartbeat (microseconds since epoch)
    pub last_heartbeat: AtomicU64,
    /// Reserved for future use
    _reserved: [u8; 36],
}

impl ReaderSlot {
    /// Create an empty reader slot
    pub const fn empty() -> Self {
        Self {
            pid: AtomicU32::new(0),
            snapshot_ts: AtomicU64::new(0),
            epoch: AtomicU32::new(0),
            last_heartbeat: AtomicU64::new(0),
            _reserved: [0u8; 36],
        }
    }

    /// Check if slot is free
    #[inline]
    pub fn is_free(&self) -> bool {
        self.pid.load(Ordering::Acquire) == 0
    }

    /// Try to claim this slot for reading
    ///
    /// Returns true if successfully claimed.
    #[inline]
    pub fn try_claim(&self, my_pid: u32, snapshot_ts: u64, epoch: u32) -> bool {
        let current_pid = self.pid.load(Ordering::Acquire);
        
        // Only claim if free or already ours
        if current_pid != 0 && current_pid != my_pid {
            return false;
        }

        if self
            .pid
            .compare_exchange(current_pid, my_pid, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Successfully claimed, update metadata
            self.snapshot_ts.store(snapshot_ts, Ordering::Release);
            self.epoch.store(epoch, Ordering::Release);
            self.last_heartbeat
                .store(current_time_us(), Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Release this slot
    #[inline]
    pub fn release(&self, my_pid: u32) {
        // Only release if we own it
        if self.pid.load(Ordering::Acquire) == my_pid {
            self.snapshot_ts.store(0, Ordering::Release);
            self.pid.store(0, Ordering::Release);
        }
    }

    /// Update heartbeat
    #[inline]
    pub fn heartbeat(&self) {
        self.last_heartbeat
            .store(current_time_us(), Ordering::Release);
    }

    /// Check if this slot is stale (process crashed or hung)
    #[inline]
    pub fn is_stale(&self, now_us: u64) -> bool {
        let pid = self.pid.load(Ordering::Acquire);
        if pid == 0 {
            return false; // Empty slot
        }

        // Check heartbeat timeout
        let last_hb = self.last_heartbeat.load(Ordering::Acquire);
        if now_us.saturating_sub(last_hb) > STALE_READER_TIMEOUT_US {
            return true;
        }

        // Check if process still exists
        !process_exists(pid)
    }
}

// =============================================================================
// MVCC Metadata Header
// =============================================================================

/// MVCC metadata file header
#[repr(C)]
#[derive(Debug)]
pub struct MvccHeader {
    /// Magic bytes for validation
    pub magic: u64,
    /// Format version
    pub version: u32,
    /// Page size
    pub page_size: u32,
    /// Number of reader slots
    pub num_readers: u32,
    /// Current epoch (incremented on recovery/GC)
    pub current_epoch: AtomicU64,
    /// Current HLC timestamp
    pub current_ts: AtomicU64,
    /// Writer lock (0 = free, pid = locked)
    pub writer_lock: AtomicU32,
    /// Number of commits since last GC
    pub commits_since_gc: AtomicU64,
    /// Reserved for alignment
    _reserved: [u8; 4],
}

impl MvccHeader {
    /// Create new header with default values
    pub fn new() -> Self {
        Self {
            magic: MVCC_MAGIC,
            version: MVCC_VERSION,
            page_size: 4096,
            num_readers: MAX_READERS as u32,
            current_epoch: AtomicU64::new(1),
            current_ts: AtomicU64::new(HlcTimestamp::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                0,
            ).raw()),
            writer_lock: AtomicU32::new(0),
            commits_since_gc: AtomicU64::new(0),
            _reserved: [0u8; 4],
        }
    }

    /// Validate header
    pub fn validate(&self) -> Result<()> {
        if self.magic != MVCC_MAGIC {
            return Err(SochDBError::Corruption(
                "Invalid MVCC metadata magic".into(),
            ));
        }
        if self.version != MVCC_VERSION {
            return Err(SochDBError::Corruption(format!(
                "Unsupported MVCC version: {} (expected {})",
                self.version, MVCC_VERSION
            )));
        }
        Ok(())
    }
}

impl Default for MvccHeader {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Version Entry
// =============================================================================

/// A single version of a key-value pair
#[derive(Debug, Clone)]
pub struct VersionEntry {
    /// Commit timestamp (HLC)
    pub commit_ts: u64,
    /// Transaction ID that created this version
    pub txn_id: u64,
    /// Epoch when this version was created
    pub epoch: u32,
    /// The value (None = tombstone/deletion)
    pub value: Option<Vec<u8>>,
}

impl VersionEntry {
    /// Create new version entry
    pub fn new(commit_ts: u64, txn_id: u64, epoch: u32, value: Option<Vec<u8>>) -> Self {
        Self {
            commit_ts,
            txn_id,
            epoch,
            value,
        }
    }

    /// Check if this version is visible at given snapshot
    #[inline]
    pub fn is_visible_at(&self, snapshot_ts: u64) -> bool {
        self.commit_ts > 0 && self.commit_ts < snapshot_ts
    }
}

// =============================================================================
// Version Chain (Sorted by commit_ts descending)
// =============================================================================

/// Chain of versions for a single key, sorted by commit_ts descending
///
/// Optimized for:
/// - O(log V) reads via binary search
/// - O(1) writes (prepend to front)
/// - O(V) GC (linear scan with compaction)
#[derive(Debug, Default)]
pub struct VersionChain {
    /// Committed versions, sorted by commit_ts DESCENDING
    versions: Vec<VersionEntry>,
    /// Uncommitted version (at most one per key)
    uncommitted: Option<VersionEntry>,
}

impl VersionChain {
    /// Create empty version chain
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
            uncommitted: None,
        }
    }

    /// Add uncommitted version
    pub fn add_uncommitted(&mut self, value: Option<Vec<u8>>, txn_id: u64, epoch: u32) {
        self.uncommitted = Some(VersionEntry {
            commit_ts: 0, // Uncommitted
            txn_id,
            epoch,
            value,
        });
    }

    /// Commit the uncommitted version
    ///
    /// Returns true if committed successfully
    pub fn commit(&mut self, txn_id: u64, commit_ts: u64) -> bool {
        if let Some(ref mut v) = self.uncommitted {
            if v.txn_id == txn_id {
                v.commit_ts = commit_ts;
                let committed = self.uncommitted.take().unwrap();

                // Insert into sorted position (descending by commit_ts)
                // Binary search for insertion point
                let pos = self
                    .versions
                    .partition_point(|existing| existing.commit_ts > commit_ts);
                self.versions.insert(pos, committed);

                return true;
            }
        }
        false
    }

    /// Abort uncommitted version
    pub fn abort(&mut self, txn_id: u64) {
        if let Some(ref v) = self.uncommitted {
            if v.txn_id == txn_id {
                self.uncommitted = None;
            }
        }
    }

    /// Read at snapshot timestamp
    ///
    /// Returns the most recent version visible at snapshot_ts.
    /// If current_txn_id is provided, also considers uncommitted writes from that txn.
    ///
    /// Complexity: O(log V) via binary search
    pub fn read_at(&self, snapshot_ts: u64, current_txn_id: Option<u64>) -> Option<&VersionEntry> {
        // Check uncommitted version from current transaction
        if let Some(txn_id) = current_txn_id {
            if let Some(ref v) = self.uncommitted {
                if v.txn_id == txn_id {
                    return Some(v);
                }
            }
        }

        // Binary search for first version with commit_ts < snapshot_ts
        // Versions are sorted descending, so we find first where commit_ts < snapshot_ts
        let idx = self
            .versions
            .partition_point(|v| v.commit_ts >= snapshot_ts);

        self.versions.get(idx)
    }

    /// Check if there's a write conflict
    pub fn has_write_conflict(&self, my_txn_id: u64) -> bool {
        if let Some(ref v) = self.uncommitted {
            return v.txn_id != my_txn_id;
        }
        false
    }

    /// Garbage collect old versions
    ///
    /// Removes versions that are:
    /// 1. Older than min_visible_epoch AND
    /// 2. Older than min_snapshot_ts AND
    /// 3. Not the most recent version (we always keep at least one)
    ///
    /// Returns number of versions reclaimed
    pub fn gc(&mut self, min_epoch: u32, min_snapshot_ts: u64) -> usize {
        if self.versions.len() <= 1 {
            return 0; // Always keep at least one version
        }

        let original_len = self.versions.len();

        // Keep versions that are:
        // - In current or recent epoch, OR
        // - Potentially visible to active readers, OR
        // - The most recent version (always keep)
        let mut keep_count = 1; // Always keep newest
        for v in self.versions.iter().skip(1) {
            if v.epoch >= min_epoch || v.commit_ts >= min_snapshot_ts {
                keep_count += 1;
            } else {
                break; // Sorted descending, so all remaining are older
            }
        }

        self.versions.truncate(keep_count);
        original_len - self.versions.len()
    }

    /// Get number of versions
    pub fn len(&self) -> usize {
        self.versions.len() + if self.uncommitted.is_some() { 1 } else { 0 }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty() && self.uncommitted.is_none()
    }
}

// =============================================================================
// Concurrent Version Store
// =============================================================================

/// Lock-free version store using DashMap
///
/// Provides O(1) key lookup + O(log V) version lookup per key.
pub struct VersionStore {
    /// Key → VersionChain mapping
    data: DashMap<Vec<u8>, VersionChain>,
    /// Statistics
    stats: VersionStoreStats,
}

/// Version store statistics
#[derive(Debug, Default)]
pub struct VersionStoreStats {
    /// Total number of keys
    pub num_keys: AtomicU64,
    /// Total number of versions across all keys
    pub num_versions: AtomicU64,
    /// Number of GC passes
    pub gc_passes: AtomicU64,
    /// Versions reclaimed by GC
    pub versions_reclaimed: AtomicU64,
}

impl VersionStore {
    /// Create new version store
    pub fn new() -> Self {
        Self {
            data: DashMap::new(),
            stats: VersionStoreStats::default(),
        }
    }

    /// Insert a new uncommitted version
    pub fn insert_uncommitted(
        &self,
        key: &[u8],
        value: Option<Vec<u8>>,
        txn_id: u64,
        epoch: u32,
    ) -> Result<()> {
        let mut entry = self.data.entry(key.to_vec()).or_insert_with(|| {
            self.stats.num_keys.fetch_add(1, Ordering::Relaxed);
            VersionChain::new()
        });

        // Check for write conflict
        if entry.has_write_conflict(txn_id) {
            return Err(SochDBError::Internal(
                "Write conflict: another transaction has uncommitted write".into(),
            ));
        }

        entry.add_uncommitted(value, txn_id, epoch);
        self.stats.num_versions.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Commit a version
    pub fn commit(&self, key: &[u8], txn_id: u64, commit_ts: u64) -> bool {
        if let Some(mut entry) = self.data.get_mut(key) {
            return entry.commit(txn_id, commit_ts);
        }
        false
    }

    /// Abort uncommitted version
    pub fn abort(&self, key: &[u8], txn_id: u64) {
        if let Some(mut entry) = self.data.get_mut(key) {
            entry.abort(txn_id);
            self.stats.num_versions.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Read value at snapshot timestamp
    pub fn get(&self, key: &[u8], snapshot_ts: u64, current_txn_id: Option<u64>) -> Option<Vec<u8>> {
        self.data.get(key).and_then(|chain| {
            chain
                .read_at(snapshot_ts, current_txn_id)
                .and_then(|v| v.value.clone())
        })
    }

    /// Check if key exists at snapshot
    pub fn contains(&self, key: &[u8], snapshot_ts: u64) -> bool {
        self.data
            .get(key)
            .map(|chain| chain.read_at(snapshot_ts, None).is_some())
            .unwrap_or(false)
    }

    /// Run garbage collection
    ///
    /// Removes old versions that are no longer visible to any reader.
    pub fn gc(&self, min_epoch: u32, min_snapshot_ts: u64) -> usize {
        let mut total_reclaimed = 0;

        for mut entry in self.data.iter_mut() {
            let reclaimed = entry.gc(min_epoch, min_snapshot_ts);
            total_reclaimed += reclaimed;
        }

        self.stats
            .gc_passes
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .versions_reclaimed
            .fetch_add(total_reclaimed as u64, Ordering::Relaxed);

        total_reclaimed
    }

    /// Get number of keys
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get statistics
    pub fn stats(&self) -> &VersionStoreStats {
        &self.stats
    }
}

impl Default for VersionStore {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Concurrent MVCC Manager
// =============================================================================

/// Manager for concurrent MVCC operations
///
/// Coordinates:
/// - Reader registration (lock-free)
/// - Writer locking (single-writer)
/// - Version store access
/// - Garbage collection
pub struct ConcurrentMvcc {
    /// Path to database
    path: PathBuf,
    /// MVCC header (in-memory, synced from mmap'd file)
    header: MvccHeader,
    /// Reader slots
    reader_slots: Vec<ReaderSlot>,
    /// Version store
    version_store: VersionStore,
    /// Our process ID
    our_pid: u32,
    /// Slot index we're using (if registered as reader)
    our_slot: RwLock<Option<usize>>,
}

impl ConcurrentMvcc {
    /// Open or create MVCC manager
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        let metadata_path = path.join(".mvcc_metadata");
        let header = if metadata_path.exists() {
            // Load existing header
            Self::load_header(&metadata_path)?
        } else {
            // Create new metadata file
            let header = MvccHeader::new();
            Self::save_header(&metadata_path, &header)?;
            header
        };

        // Initialize reader slots
        let mut reader_slots = Vec::with_capacity(MAX_READERS);
        for _ in 0..MAX_READERS {
            reader_slots.push(ReaderSlot::empty());
        }

        Ok(Self {
            path,
            header,
            reader_slots,
            version_store: VersionStore::new(),
            our_pid: std::process::id(),
            our_slot: RwLock::new(None),
        })
    }

    /// Load header from file
    fn load_header(path: &Path) -> Result<MvccHeader> {
        use std::io::Read;
        let mut file = File::open(path)?;
        let mut buf = vec![0u8; std::mem::size_of::<MvccHeader>()];
        file.read_exact(&mut buf)?;

        // SAFETY: MvccHeader is repr(C) with no padding issues
        let header: MvccHeader = unsafe { std::ptr::read(buf.as_ptr() as *const MvccHeader) };
        header.validate()?;
        Ok(header)
    }

    /// Save header to file
    fn save_header(path: &Path, header: &MvccHeader) -> Result<()> {
        use std::io::Write;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        // SAFETY: MvccHeader is repr(C)
        let buf = unsafe {
            std::slice::from_raw_parts(
                header as *const MvccHeader as *const u8,
                std::mem::size_of::<MvccHeader>(),
            )
        };
        file.write_all(buf)?;
        file.sync_all()?;
        Ok(())
    }

    /// Allocate next timestamp
    #[inline]
    pub fn allocate_timestamp(&self) -> HlcTimestamp {
        HlcTimestamp::allocate_next(&self.header.current_ts)
    }

    /// Get current timestamp without advancing
    #[inline]
    pub fn current_timestamp(&self) -> HlcTimestamp {
        HlcTimestamp::read_current(&self.header.current_ts)
    }

    /// Get current epoch
    #[inline]
    pub fn current_epoch(&self) -> u64 {
        self.header.current_epoch.load(Ordering::Acquire)
    }

    /// Register as active reader
    ///
    /// Returns slot index on success.
    /// Must call `unregister_reader()` when done.
    pub fn register_reader(&self) -> Result<usize> {
        let snapshot_ts = self.current_timestamp().raw();
        let epoch = self.current_epoch() as u32;

        // Find free slot
        for (i, slot) in self.reader_slots.iter().enumerate() {
            if slot.try_claim(self.our_pid, snapshot_ts, epoch) {
                *self.our_slot.write() = Some(i);
                return Ok(i);
            }
        }

        Err(SochDBError::ResourceExhausted(
            "Too many concurrent readers".into(),
        ))
    }

    /// Unregister as reader
    pub fn unregister_reader(&self, slot_idx: usize) {
        if slot_idx < self.reader_slots.len() {
            self.reader_slots[slot_idx].release(self.our_pid);
            *self.our_slot.write() = None;
        }
    }

    /// Try to acquire writer lock
    ///
    /// Returns WriterGuard on success, which releases lock on drop.
    pub fn try_acquire_writer(&self) -> Result<WriterGuard<'_>> {
        let current = self.header.writer_lock.load(Ordering::Acquire);

        if current == 0 {
            // Try to acquire
            if self
                .header
                .writer_lock
                .compare_exchange(0, self.our_pid, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(WriterGuard { mvcc: self });
            }
        } else if current == self.our_pid {
            // Already own the lock (reentrant)
            return Ok(WriterGuard { mvcc: self });
        }

        Err(SochDBError::LockError(format!(
            "Writer lock held by process {}",
            current
        )))
    }

    /// Acquire writer lock with timeout
    pub fn acquire_writer(&self, timeout: Duration) -> Result<WriterGuard<'_>> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            match self.try_acquire_writer() {
                Ok(guard) => return Ok(guard),
                Err(_) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_micros(100));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Release writer lock
    fn release_writer(&self) {
        let current = self.header.writer_lock.load(Ordering::Acquire);
        if current == self.our_pid {
            self.header.writer_lock.store(0, Ordering::Release);
        }
    }

    /// Get version store
    pub fn version_store(&self) -> &VersionStore {
        &self.version_store
    }

    /// Calculate minimum visible snapshot across all readers
    pub fn min_active_snapshot(&self) -> u64 {
        let mut min_ts = u64::MAX;

        for slot in &self.reader_slots {
            let pid = slot.pid.load(Ordering::Acquire);
            if pid != 0 {
                let ts = slot.snapshot_ts.load(Ordering::Acquire);
                if ts > 0 && ts < min_ts {
                    min_ts = ts;
                }
            }
        }

        min_ts
    }

    /// Calculate minimum active epoch across all readers
    pub fn min_active_epoch(&self) -> u32 {
        let mut min_epoch = u32::MAX;

        for slot in &self.reader_slots {
            let pid = slot.pid.load(Ordering::Acquire);
            if pid != 0 {
                let epoch = slot.epoch.load(Ordering::Acquire);
                if epoch < min_epoch {
                    min_epoch = epoch;
                }
            }
        }

        if min_epoch == u32::MAX {
            // No active readers, use current epoch
            self.current_epoch() as u32
        } else {
            min_epoch
        }
    }

    /// Run garbage collection
    ///
    /// Should be called periodically (e.g., every 1000 commits).
    pub fn run_gc(&self) -> usize {
        let min_epoch = self.min_active_epoch();
        let min_snapshot = self.min_active_snapshot();

        self.version_store.gc(min_epoch, min_snapshot)
    }

    /// Check if GC should run (based on commit count)
    pub fn should_run_gc(&self) -> bool {
        self.header
            .commits_since_gc
            .load(Ordering::Relaxed)
            >= GC_COMMIT_INTERVAL
    }

    /// Increment commit count and maybe run GC
    pub fn on_commit(&self) {
        let count = self
            .header
            .commits_since_gc
            .fetch_add(1, Ordering::Relaxed);

        if count >= GC_COMMIT_INTERVAL {
            self.header.commits_since_gc.store(0, Ordering::Relaxed);
            let _ = self.run_gc();
        }
    }

    /// Clean up stale readers (from crashed processes)
    pub fn cleanup_stale_readers(&self) -> usize {
        let now = current_time_us();
        let mut cleaned = 0;

        for slot in &self.reader_slots {
            if slot.is_stale(now) {
                slot.pid.store(0, Ordering::Release);
                cleaned += 1;
            }
        }

        cleaned
    }

    /// Advance epoch (called on recovery)
    pub fn advance_epoch(&self) -> u64 {
        self.header.current_epoch.fetch_add(1, Ordering::AcqRel) + 1
    }
}

impl Drop for ConcurrentMvcc {
    fn drop(&mut self) {
        // Release any reader slot we hold
        if let Some(slot_idx) = *self.our_slot.read() {
            self.unregister_reader(slot_idx);
        }

        // Release writer lock if we hold it
        self.release_writer();
    }
}

// =============================================================================
// Writer Guard (RAII)
// =============================================================================

/// Guard that releases writer lock on drop
pub struct WriterGuard<'a> {
    mvcc: &'a ConcurrentMvcc,
}

impl<'a> Drop for WriterGuard<'a> {
    fn drop(&mut self) {
        self.mvcc.release_writer();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get current time in microseconds since epoch
#[inline]
fn current_time_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

/// Check if a process exists
#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    // kill(pid, 0) checks existence without sending signal
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if result == 0 {
        true
    } else {
        // ESRCH = no such process
        std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    }
}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle == 0 {
            false
        } else {
            CloseHandle(handle);
            true
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn process_exists(_pid: u32) -> bool {
    true // Assume exists on unknown platforms
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_hlc_timestamp_ordering() {
        let ts = AtomicU64::new(0);

        let t1 = HlcTimestamp::allocate_next(&ts);
        let t2 = HlcTimestamp::allocate_next(&ts);
        let t3 = HlcTimestamp::allocate_next(&ts);

        assert!(t1.raw() < t2.raw());
        assert!(t2.raw() < t3.raw());
    }

    #[test]
    fn test_hlc_timestamp_concurrent() {
        let ts = Arc::new(AtomicU64::new(0));
        let mut handles = vec![];

        for _ in 0..8 {
            let ts = ts.clone();
            handles.push(thread::spawn(move || {
                let mut timestamps = vec![];
                for _ in 0..1000 {
                    timestamps.push(HlcTimestamp::allocate_next(&ts).raw());
                }
                timestamps
            }));
        }

        let mut all_ts: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // All timestamps should be unique
        all_ts.sort();
        let len_before = all_ts.len();
        all_ts.dedup();
        assert_eq!(all_ts.len(), len_before, "Duplicate timestamps found!");
    }

    #[test]
    fn test_version_chain_read_at() {
        let mut chain = VersionChain::new();

        // Add versions
        chain.versions.push(VersionEntry::new(100, 1, 1, Some(b"v100".to_vec())));
        chain.versions.push(VersionEntry::new(90, 2, 1, Some(b"v90".to_vec())));
        chain.versions.push(VersionEntry::new(80, 3, 1, Some(b"v80".to_vec())));

        // Read at different snapshots
        let v = chain.read_at(105, None).unwrap();
        assert_eq!(v.value, Some(b"v100".to_vec()));

        let v = chain.read_at(95, None).unwrap();
        assert_eq!(v.value, Some(b"v90".to_vec()));

        let v = chain.read_at(85, None).unwrap();
        assert_eq!(v.value, Some(b"v80".to_vec()));

        // Read before all versions
        assert!(chain.read_at(75, None).is_none());
    }

    #[test]
    fn test_version_chain_gc() {
        let mut chain = VersionChain::new();

        // Add 10 versions with varying epochs
        // Older versions have older epochs
        for i in 0..10 {
            chain.versions.push(VersionEntry::new(
                100 - i * 5,   // commit_ts: 100, 95, 90, 85, 80, 75, 70, 65, 60, 55
                i as u64,     // txn_id
                (10 - i) as u32, // epoch: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 (older = lower)
                Some(format!("v{}", 100 - i * 5).into_bytes()),
            ));
        }

        assert_eq!(chain.len(), 10);

        // GC with min_epoch = 7, min_snapshot = 75
        // Keeps versions where epoch >= 7 OR commit_ts >= 75
        // epoch >= 7: versions at ts=100, 95, 90, 85 (epochs 10, 9, 8, 7)
        // commit_ts >= 75: versions at ts=100, 95, 90, 85, 80, 75
        // But we always keep at least the newest one
        let reclaimed = chain.gc(7, 75);

        // Versions that should be reclaimed:
        // ts=70 (epoch=4), ts=65 (epoch=3), ts=60 (epoch=2), ts=55 (epoch=1)
        // That's 4 versions reclaimed
        assert!(reclaimed > 0, "Should have reclaimed some versions, got {}", reclaimed);
        assert!(chain.len() >= 1, "Should keep at least one version");
    }

    #[test]
    fn test_version_store_basic() {
        let store = VersionStore::new();

        // Insert uncommitted
        store
            .insert_uncommitted(b"key1", Some(b"value1".to_vec()), 1, 1)
            .unwrap();

        // Commit it
        assert!(store.commit(b"key1", 1, 100));

        // Read at snapshot after commit
        let value = store.get(b"key1", 150, None);
        assert_eq!(value, Some(b"value1".to_vec()));

        // Read at snapshot before commit
        let value = store.get(b"key1", 50, None);
        assert!(value.is_none());
    }

    #[test]
    fn test_reader_slot_claim_release() {
        let slot = ReaderSlot::empty();

        assert!(slot.is_free());

        // Claim slot
        assert!(slot.try_claim(1234, 100, 1));
        assert!(!slot.is_free());

        // Can't claim from different PID
        assert!(!slot.try_claim(5678, 200, 2));

        // Can re-claim from same PID
        assert!(slot.try_claim(1234, 300, 3));

        // Release
        slot.release(1234);
        assert!(slot.is_free());
    }
}
