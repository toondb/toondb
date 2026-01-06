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

//! SuperVersion Metadata + Copy-on-Write Version Set
//!
//! This module implements RocksDB-style SuperVersion metadata management for
//! near lock-free reads. The key insight is that read paths only need a consistent
//! snapshot of metadata (memtable, immutable memtables, SSTable levels), not
//! exclusive access to the underlying data structures.
//!
//! ## Problem Analysis
//!
//! Previous implementation suffered from lock contention:
//! - Foreground reads must traverse memtable (RwLock) → immutables → levels
//! - Each level switch may acquire additional locks
//! - Compaction/flush modifies metadata while readers hold references
//! - Lock-order complexity leads to deadlock potential
//!
//! ## Solution: SuperVersion + ArcSwap
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       VersionSet                                 │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │ current: ArcSwap<SuperVersion>  ◄─── Atomic swap (O(1))     ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! │                            │                                     │
//! │                            ▼                                     │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │                    SuperVersion                              ││
//! │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  ││
//! │  │  │ memtable    │  │ immutables   │  │ level_files       │  ││
//! │  │  │ Arc<...>    │  │ Arc<Vec<..>> │  │ Arc<Vec<Level>>   │  ││
//! │  │  └─────────────┘  └──────────────┘  └───────────────────┘  ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Read Path (Lock-Free)
//!
//! 1. Load current SuperVersion via atomic: `let sv = version_set.get();` // O(1)
//! 2. Search memtable → immutables → levels using sv references
//! 3. Release sv when done (Arc drop)
//!
//! No locks acquired. Readers are completely decoupled from writers.
//!
//! ## Write Path (Copy-on-Write)
//!
//! 1. Clone current SuperVersion's inner Arcs
//! 2. Modify the new copy (e.g., add new immutable memtable)
//! 3. Atomically swap new SuperVersion into place
//! 4. Old SuperVersion kept alive by existing readers (Arc)
//!
//! ## Complexity Analysis
//!
//! | Operation          | Old (with locks)     | New (SuperVersion)        |
//! |--------------------|----------------------|---------------------------|
//! | Read acquire       | O(1) but contended   | O(1) atomic load          |
//! | Read release       | O(1) unlock          | O(1) Arc decrement        |
//! | Metadata update    | O(1) + lock wait     | O(changed_metadata) clone |
//! | Concurrent reads   | Serialized on RwLock | Truly parallel            |
//!
//! For 8 threads, expected speedup: ~6-8x on read-heavy workloads.

use arc_swap::{ArcSwap, Guard};
use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// =============================================================================
// File Metadata
// =============================================================================

/// Metadata for an SSTable file
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// Unique file number
    pub file_number: u64,
    /// File size in bytes
    pub file_size: u64,
    /// Smallest key in file (for range queries)
    pub smallest_key: Vec<u8>,
    /// Largest key in file (for range queries)
    pub largest_key: Vec<u8>,
    /// Number of entries in file
    pub num_entries: u64,
    /// Minimum sequence number in file
    pub min_seqno: u64,
    /// Maximum sequence number in file
    pub max_seqno: u64,
    /// Path to the file
    pub path: PathBuf,
    /// Bloom filter (if loaded)
    pub bloom_filter: Option<Arc<BloomFilterHandle>>,
    /// Whether file is being compacted
    pub being_compacted: bool,
}

impl FileMetadata {
    /// Check if key might be in this file using bloom filter
    #[inline]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        match &self.bloom_filter {
            Some(bf) => bf.may_contain(key),
            None => true, // No filter = must check file
        }
    }

    /// Check if a key range overlaps with this file
    #[inline]
    pub fn overlaps_range(&self, start: &[u8], end: &[u8]) -> bool {
        if end.is_empty() {
            // Unbounded end - overlaps if smallest_key <= end conceptually
            self.smallest_key.as_slice() >= start || self.largest_key.as_slice() >= start
        } else {
            // Standard range overlap check
            self.smallest_key.as_slice() <= end && self.largest_key.as_slice() >= start
        }
    }
}

/// Handle to a bloom filter (may be memory-mapped or in-memory)
#[derive(Debug)]
pub struct BloomFilterHandle {
    /// The bloom filter bits
    bits: Vec<u64>,
    /// Number of hash functions
    num_hashes: u32,
}

impl BloomFilterHandle {
    /// Create a new bloom filter handle
    pub fn new(bits: Vec<u64>, num_hashes: u32) -> Self {
        Self { bits, num_hashes }
    }

    /// Check if key may be present
    #[inline]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        if self.bits.is_empty() {
            return true;
        }
        let num_bits = self.bits.len() * 64;
        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if self.bits[word_idx] & (1 << bit_pos) == 0 {
                return false;
            }
        }
        true
    }

    #[inline]
    fn hash1(key: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    #[inline]
    fn hash2(key: &[u8]) -> u64 {
        twox_hash::xxh3::hash64(key)
    }
}

// =============================================================================
// Level Metadata
// =============================================================================

/// Metadata for a single level in the LSM tree
#[derive(Debug, Clone)]
pub struct LevelMetadata {
    /// Level number (0 = unsorted, 1+ = sorted)
    pub level: u32,
    /// Files in this level (sorted by smallest_key for L1+)
    pub files: Vec<Arc<FileMetadata>>,
    /// Total size of files in this level
    pub total_size: u64,
    /// Target size for this level (for compaction decisions)
    pub target_size: u64,
    /// Compaction score (size / target_size)
    pub compaction_score: f64,
}

impl LevelMetadata {
    /// Create a new empty level
    pub fn new(level: u32, target_size: u64) -> Self {
        Self {
            level,
            files: Vec::new(),
            total_size: 0,
            target_size,
            compaction_score: 0.0,
        }
    }

    /// Find files that may contain a key using binary search (for L1+)
    pub fn find_files_for_key(&self, key: &[u8]) -> Vec<&Arc<FileMetadata>> {
        if self.level == 0 {
            // L0 is unsorted - must check all files
            self.files
                .iter()
                .filter(|f| {
                    f.smallest_key.as_slice() <= key && f.largest_key.as_slice() >= key
                })
                .collect()
        } else {
            // L1+ is sorted - binary search for overlapping files
            let idx = self.files.partition_point(|f| f.largest_key.as_slice() < key);
            if idx < self.files.len() && self.files[idx].smallest_key.as_slice() <= key {
                vec![&self.files[idx]]
            } else {
                vec![]
            }
        }
    }

    /// Find files that overlap with a key range
    pub fn find_files_for_range(&self, start: &[u8], end: &[u8]) -> Vec<&Arc<FileMetadata>> {
        if self.level == 0 {
            // L0 is unsorted - check all files
            self.files
                .iter()
                .filter(|f| f.overlaps_range(start, end))
                .collect()
        } else {
            // L1+ is sorted - find range using binary search
            let start_idx = self.files.partition_point(|f| f.largest_key.as_slice() < start);
            let end_idx = if end.is_empty() {
                self.files.len()
            } else {
                self.files.partition_point(|f| f.smallest_key.as_slice() <= end)
            };
            self.files[start_idx..end_idx].iter().collect()
        }
    }

    /// Recalculate compaction score
    pub fn update_compaction_score(&mut self) {
        if self.target_size > 0 {
            self.compaction_score = self.total_size as f64 / self.target_size as f64;
        } else {
            self.compaction_score = 0.0;
        }
    }
}

// =============================================================================
// Immutable MemTable Reference
// =============================================================================

/// Reference to an immutable memtable (sealed, pending flush)
#[derive(Debug, Clone)]
pub struct ImmutableMemTableRef {
    /// Unique ID for this immutable memtable
    pub id: u64,
    /// Sequence number when sealed
    pub seal_seqno: u64,
    /// Approximate size in bytes
    pub size_bytes: u64,
    /// Reference to the actual memtable data
    /// This is opaque - the actual type is provided by the storage layer
    pub data: Arc<dyn ImmutableMemTable>,
}

/// Trait for immutable memtable operations
pub trait ImmutableMemTable: Send + Sync + std::fmt::Debug {
    /// Get a value at a specific sequence number
    fn get(&self, key: &[u8], seqno: u64) -> Option<Option<Vec<u8>>>;
    
    /// Iterate over all entries
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Option<Vec<u8>>, u64)> + '_>;
    
    /// Get approximate size
    fn size(&self) -> u64;
}

// =============================================================================
// SuperVersion - The Core Abstraction
// =============================================================================

/// SuperVersion: A consistent snapshot of all storage metadata
///
/// This is the key abstraction for lock-free reads. A SuperVersion contains
/// Arc references to:
/// - Current mutable memtable
/// - List of immutable memtables (pending flush)
/// - All SSTable levels and their file metadata
///
/// Readers acquire a SuperVersion (O(1) atomic load), use it for their entire
/// read operation, then release it (O(1) Arc drop). No locks required.
///
/// Writers create a new SuperVersion with modified metadata and atomically
/// swap it in. Old versions remain valid until all readers release them.
#[derive(Debug, Clone)]
pub struct SuperVersion {
    /// Version number (monotonically increasing)
    pub version_number: u64,
    
    /// Current mutable memtable reference
    /// Note: The memtable itself may have internal synchronization,
    /// but the *reference* is immutable within a SuperVersion
    pub memtable_version: u64,
    
    /// Immutable memtables (oldest first)
    pub immutable_memtables: Arc<Vec<ImmutableMemTableRef>>,
    
    /// SSTable levels (L0, L1, L2, ...)
    pub levels: Arc<Vec<LevelMetadata>>,
    
    /// Minimum sequence number safe to garbage collect
    /// Versions with seqno < this can be pruned during compaction
    pub min_snapshot_seqno: u64,
    
    /// Current log (WAL) number for crash recovery
    pub log_number: u64,
    
    /// Prev log number (for two-log protocol during flush)
    pub prev_log_number: u64,
    
    /// Next file number to allocate
    pub next_file_number: u64,
    
    /// Manifest file number
    pub manifest_file_number: u64,
}

impl SuperVersion {
    /// Create a new empty SuperVersion
    pub fn new() -> Self {
        Self {
            version_number: 1,
            memtable_version: 1,
            immutable_memtables: Arc::new(Vec::new()),
            levels: Arc::new(Vec::new()),
            min_snapshot_seqno: 0,
            log_number: 1,
            prev_log_number: 0,
            next_file_number: 2,
            manifest_file_number: 1,
        }
    }

    /// Create a new SuperVersion with updated immutable memtables
    pub fn with_new_immutable(&self, imm: ImmutableMemTableRef) -> Self {
        let mut new_imms = (*self.immutable_memtables).clone();
        new_imms.push(imm);
        
        Self {
            version_number: self.version_number + 1,
            memtable_version: self.memtable_version + 1,
            immutable_memtables: Arc::new(new_imms),
            levels: Arc::clone(&self.levels),
            min_snapshot_seqno: self.min_snapshot_seqno,
            log_number: self.log_number,
            prev_log_number: self.prev_log_number,
            next_file_number: self.next_file_number,
            manifest_file_number: self.manifest_file_number,
        }
    }

    /// Create a new SuperVersion after flushing immutable memtables
    pub fn with_flushed_memtables(
        &self,
        flushed_ids: &[u64],
        new_files: Vec<(u32, Arc<FileMetadata>)>, // (level, file)
    ) -> Self {
        // Remove flushed immutable memtables
        let new_imms: Vec<_> = self.immutable_memtables
            .iter()
            .filter(|imm| !flushed_ids.contains(&imm.id))
            .cloned()
            .collect();
        
        // Add new files to levels
        let mut new_levels = (*self.levels).clone();
        for (level, file) in new_files {
            // Ensure level exists
            while new_levels.len() <= level as usize {
                let target_size = self.level_target_size(new_levels.len() as u32);
                new_levels.push(LevelMetadata::new(new_levels.len() as u32, target_size));
            }
            
            let lm = &mut new_levels[level as usize];
            lm.total_size += file.file_size;
            lm.files.push(file);
            lm.update_compaction_score();
        }
        
        Self {
            version_number: self.version_number + 1,
            memtable_version: self.memtable_version,
            immutable_memtables: Arc::new(new_imms),
            levels: Arc::new(new_levels),
            min_snapshot_seqno: self.min_snapshot_seqno,
            log_number: self.log_number,
            prev_log_number: self.prev_log_number,
            next_file_number: self.next_file_number,
            manifest_file_number: self.manifest_file_number,
        }
    }

    /// Create a new SuperVersion after compaction
    pub fn with_compaction_result(
        &self,
        input_files: &[(u32, u64)], // (level, file_number)
        output_files: Vec<(u32, Arc<FileMetadata>)>,
    ) -> Self {
        let mut new_levels = (*self.levels).clone();
        
        // Remove input files
        for (level, file_num) in input_files {
            if let Some(lm) = new_levels.get_mut(*level as usize) {
                if let Some(pos) = lm.files.iter().position(|f| f.file_number == *file_num) {
                    let removed = lm.files.remove(pos);
                    lm.total_size -= removed.file_size;
                }
            }
        }
        
        // Add output files
        for (level, file) in output_files {
            while new_levels.len() <= level as usize {
                let target_size = self.level_target_size(new_levels.len() as u32);
                new_levels.push(LevelMetadata::new(new_levels.len() as u32, target_size));
            }
            
            let lm = &mut new_levels[level as usize];
            lm.total_size += file.file_size;
            
            // Insert in sorted order for L1+
            if level > 0 {
                let pos = lm.files.partition_point(|f| f.smallest_key < file.smallest_key);
                lm.files.insert(pos, file);
            } else {
                lm.files.push(file);
            }
            lm.update_compaction_score();
        }
        
        Self {
            version_number: self.version_number + 1,
            memtable_version: self.memtable_version,
            immutable_memtables: Arc::clone(&self.immutable_memtables),
            levels: Arc::new(new_levels),
            min_snapshot_seqno: self.min_snapshot_seqno,
            log_number: self.log_number,
            prev_log_number: self.prev_log_number,
            next_file_number: self.next_file_number,
            manifest_file_number: self.manifest_file_number,
        }
    }

    /// Calculate target size for a level
    fn level_target_size(&self, level: u32) -> u64 {
        // Level 0: 64MB, Level 1: 256MB, each subsequent level 10x
        match level {
            0 => 64 * 1024 * 1024,
            1 => 256 * 1024 * 1024,
            _ => 256 * 1024 * 1024 * 10u64.pow(level - 1),
        }
    }

    /// Get total number of files across all levels
    pub fn total_file_count(&self) -> usize {
        self.levels.iter().map(|l| l.files.len()).sum()
    }

    /// Get level with highest compaction score
    pub fn pick_compaction_level(&self) -> Option<u32> {
        self.levels
            .iter()
            .filter(|l| l.compaction_score > 1.0)
            .max_by(|a, b| a.compaction_score.partial_cmp(&b.compaction_score).unwrap())
            .map(|l| l.level)
    }
}

impl Default for SuperVersion {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// VersionSet - The Top-Level Manager
// =============================================================================

/// VersionSet manages the current SuperVersion and provides atomic updates
///
/// This is the entry point for all metadata access. Readers call `get()` to
/// acquire the current SuperVersion, writers call `install()` to atomically
/// swap in a new version.
///
/// ## Thread Safety
///
/// - `get()`: Lock-free (atomic load)
/// - `install()`: Serialized through internal mutex (writers must coordinate)
///
/// Multiple readers can proceed in parallel with zero synchronization.
/// Writers are serialized to ensure consistent version progression.
pub struct VersionSet {
    /// Current SuperVersion (atomically swappable)
    current: ArcSwap<SuperVersion>,
    
    /// Version number counter
    version_counter: AtomicU64,
    
    /// Write serialization lock
    /// Only one writer can modify the version at a time
    write_lock: Mutex<()>,
    
    /// Snapshot registry - tracks active snapshots for GC
    snapshots: Mutex<BTreeMap<u64, u64>>, // seqno -> ref_count
    
    /// Database directory
    db_path: PathBuf,
}

impl VersionSet {
    /// Create a new VersionSet
    pub fn new(db_path: PathBuf) -> Self {
        Self {
            current: ArcSwap::from_pointee(SuperVersion::new()),
            version_counter: AtomicU64::new(1),
            write_lock: Mutex::new(()),
            snapshots: Mutex::new(BTreeMap::new()),
            db_path,
        }
    }

    /// Get current SuperVersion (lock-free)
    ///
    /// This is the hot path for reads. Returns a Guard that holds an Arc
    /// to the current SuperVersion. The Guard can be dereferenced to access
    /// the SuperVersion.
    ///
    /// ## Performance
    ///
    /// - Time: O(1) atomic load
    /// - No locks acquired
    /// - No memory allocation
    #[inline]
    pub fn get(&self) -> Guard<Arc<SuperVersion>> {
        self.current.load()
    }

    /// Get current SuperVersion as owned Arc
    ///
    /// Use this when you need to hold the SuperVersion across await points
    /// or store it for later use.
    #[inline]
    pub fn get_arc(&self) -> Arc<SuperVersion> {
        self.current.load_full()
    }

    /// Install a new SuperVersion (serialized)
    ///
    /// This atomically swaps the new version into place. The old version
    /// remains valid for any readers that acquired it before the swap.
    ///
    /// ## Safety
    ///
    /// Only one writer should call this at a time. Use `with_write_lock()`
    /// to serialize concurrent updates.
    pub fn install(&self, new_version: SuperVersion) {
        let _guard = self.write_lock.lock();
        self.current.store(Arc::new(new_version));
        self.version_counter.fetch_add(1, Ordering::SeqCst);
    }

    /// Execute a function with exclusive write access
    ///
    /// Use this to ensure atomic read-modify-write operations on the
    /// version set. The function receives the current SuperVersion and
    /// should return the new SuperVersion to install.
    pub fn with_write_lock<F>(&self, f: F) -> SuperVersion
    where
        F: FnOnce(&SuperVersion) -> SuperVersion,
    {
        let _guard = self.write_lock.lock();
        let current = self.current.load();
        let new_version = f(&current);
        self.current.store(Arc::new(new_version.clone()));
        self.version_counter.fetch_add(1, Ordering::SeqCst);
        new_version
    }

    /// Register a snapshot at the given sequence number
    ///
    /// Returns the snapshot sequence number for later release.
    pub fn register_snapshot(&self, seqno: u64) -> u64 {
        let mut snapshots = self.snapshots.lock();
        *snapshots.entry(seqno).or_insert(0) += 1;
        seqno
    }

    /// Release a snapshot
    pub fn release_snapshot(&self, seqno: u64) {
        let mut snapshots = self.snapshots.lock();
        if let Some(count) = snapshots.get_mut(&seqno) {
            *count -= 1;
            if *count == 0 {
                snapshots.remove(&seqno);
            }
        }
    }

    /// Get minimum sequence number that must be preserved
    ///
    /// Returns the oldest snapshot sequence number, or the current version's
    /// min_snapshot_seqno if no snapshots are active.
    pub fn min_preserved_seqno(&self) -> u64 {
        let snapshots = self.snapshots.lock();
        snapshots.keys().next().copied().unwrap_or_else(|| {
            self.current.load().min_snapshot_seqno
        })
    }

    /// Update min_snapshot_seqno based on active snapshots
    pub fn update_min_snapshot_seqno(&self) {
        let min_seqno = self.min_preserved_seqno();
        self.with_write_lock(|current| {
            SuperVersion {
                min_snapshot_seqno: min_seqno,
                version_number: current.version_number + 1,
                ..current.clone()
            }
        });
    }

    /// Get database path
    pub fn db_path(&self) -> &PathBuf {
        &self.db_path
    }

    /// Get current version number
    pub fn version_number(&self) -> u64 {
        self.version_counter.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for VersionSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let current = self.current.load();
        f.debug_struct("VersionSet")
            .field("version_number", &current.version_number)
            .field("num_immutables", &current.immutable_memtables.len())
            .field("num_levels", &current.levels.len())
            .field("total_files", &current.total_file_count())
            .field("db_path", &self.db_path)
            .finish()
    }
}

// =============================================================================
// SuperVersionHandle - RAII Guard for SuperVersion
// =============================================================================

/// RAII handle for SuperVersion access
///
/// This provides a convenient way to access a SuperVersion with automatic
/// cleanup. The handle can be cloned to share access across threads.
pub struct SuperVersionHandle {
    version: Arc<SuperVersion>,
    version_set: Arc<VersionSet>,
    snapshot_seqno: Option<u64>,
}

impl SuperVersionHandle {
    /// Create a new handle from a VersionSet
    pub fn new(version_set: Arc<VersionSet>) -> Self {
        let version = version_set.get_arc();
        Self {
            version,
            version_set,
            snapshot_seqno: None,
        }
    }

    /// Create a handle with a registered snapshot
    pub fn with_snapshot(version_set: Arc<VersionSet>, seqno: u64) -> Self {
        let registered_seqno = version_set.register_snapshot(seqno);
        let version = version_set.get_arc();
        Self {
            version,
            version_set,
            snapshot_seqno: Some(registered_seqno),
        }
    }

    /// Get the SuperVersion
    #[inline]
    pub fn version(&self) -> &SuperVersion {
        &self.version
    }

    /// Get the snapshot sequence number (if any)
    pub fn snapshot_seqno(&self) -> Option<u64> {
        self.snapshot_seqno
    }
}

impl Drop for SuperVersionHandle {
    fn drop(&mut self) {
        if let Some(seqno) = self.snapshot_seqno {
            self.version_set.release_snapshot(seqno);
        }
    }
}

impl Clone for SuperVersionHandle {
    fn clone(&self) -> Self {
        // If we have a snapshot, register another reference
        if let Some(seqno) = self.snapshot_seqno {
            self.version_set.register_snapshot(seqno);
        }
        Self {
            version: Arc::clone(&self.version),
            version_set: Arc::clone(&self.version_set),
            snapshot_seqno: self.snapshot_seqno,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_superversion_creation() {
        let sv = SuperVersion::new();
        assert_eq!(sv.version_number, 1);
        assert!(sv.immutable_memtables.is_empty());
        assert!(sv.levels.is_empty());
    }

    #[test]
    fn test_version_set_get() {
        let vs = VersionSet::new(PathBuf::from("/tmp/test"));
        let sv = vs.get();
        assert_eq!(sv.version_number, 1);
    }

    #[test]
    fn test_version_set_install() {
        let vs = VersionSet::new(PathBuf::from("/tmp/test"));
        
        let new_sv = SuperVersion {
            version_number: 2,
            ..SuperVersion::new()
        };
        vs.install(new_sv);
        
        let sv = vs.get();
        assert_eq!(sv.version_number, 2);
    }

    #[test]
    fn test_concurrent_reads() {
        let vs = Arc::new(VersionSet::new(PathBuf::from("/tmp/test")));
        let mut handles = vec![];

        for _ in 0..10 {
            let vs_clone = Arc::clone(&vs);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let sv = vs_clone.get();
                    assert!(sv.version_number >= 1);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_snapshot_registry() {
        let vs = VersionSet::new(PathBuf::from("/tmp/test"));
        
        // Register snapshots
        vs.register_snapshot(100);
        vs.register_snapshot(200);
        vs.register_snapshot(100); // Duplicate
        
        assert_eq!(vs.min_preserved_seqno(), 100);
        
        // Release one reference to 100
        vs.release_snapshot(100);
        assert_eq!(vs.min_preserved_seqno(), 100); // Still have one ref
        
        // Release second reference
        vs.release_snapshot(100);
        assert_eq!(vs.min_preserved_seqno(), 200);
        
        // Release 200
        vs.release_snapshot(200);
    }

    #[test]
    fn test_level_binary_search() {
        let mut level = LevelMetadata::new(1, 256 * 1024 * 1024);
        
        // Add files in sorted order
        for i in 0..10 {
            let file = Arc::new(FileMetadata {
                file_number: i as u64,
                file_size: 1024,
                smallest_key: format!("{:02}", i * 10).into_bytes(),
                largest_key: format!("{:02}", i * 10 + 9).into_bytes(),
                num_entries: 100,
                min_seqno: 1,
                max_seqno: 100,
                path: PathBuf::from(format!("/tmp/{}.sst", i)),
                bloom_filter: None,
                being_compacted: false,
            });
            level.files.push(file);
            level.total_size += 1024;
        }
        
        // Test point lookup
        let files = level.find_files_for_key(b"25");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_number, 2);
        
        // Test range lookup
        let files = level.find_files_for_range(b"15", b"35");
        assert_eq!(files.len(), 3); // Files 1, 2, 3
    }
}
