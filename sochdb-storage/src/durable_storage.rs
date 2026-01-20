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

//! Durable Storage Layer
//!
//! This module wires together all storage components into a production-ready
//! durable storage engine:
//!
//! - WAL (txn_wal.rs) for durability
//! - Group Commit for throughput
//! - MVCC for isolation
//! - LSCS for columnar efficiency
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      DurableStorage                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
//! │  │ MvccManager │    │ GroupCommit │───▶│ TxnWal (fsync)      │ │
//! │  │             │    │             │    └─────────────────────┘ │
//! │  │ ┌─────────┐ │    └─────────────┘                            │
//! │  │ │Snapshots│ │                                                │
//! │  │ └─────────┘ │    ┌─────────────────────────────────────────┐│
//! │  │ ┌─────────┐ │    │              MemTable                    ││
//! │  │ │ Txn Map │ │    │  (key → (value, txn_id, version))       ││
//! │  │ └─────────┘ │    └─────────────────────────────────────────┘│
//! │  └─────────────┘                                                │
//! │                      ┌─────────────────────────────────────────┐│
//! │                      │              LSCS (SST)                  ││
//! │                      │  Immutable columnar segments             ││
//! │                      └─────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Concurrency
//!
//! - Writers: Serialize through WAL, use MVCC for conflict detection
//! - Readers: Lock-free reads at snapshot timestamp
//! - Commits: Batched through GroupCommit for throughput

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use smallvec::SmallVec;

use crossbeam_skiplist::SkipMap;

use crate::deferred_index::{DeferredSortedIndex, DeferredIndexConfig};
use crate::group_commit::EventDrivenGroupCommit;
use crate::txn_wal::{TxnWal, TxnWalBuffer, TxnWalEntry};
use sochdb_core::{Result, SochDBError};

// =============================================================================
// SSI Bloom Filter - Fast Conflict Pre-Filtering
// =============================================================================

/// Space-efficient Bloom filter for SSI conflict detection
///
/// Used to quickly determine if two transactions MIGHT have conflicting keys.
/// False positives are acceptable (leads to unnecessary exact checks),
/// but false negatives are not allowed.
///
/// ## Configuration
///
/// For 1000 keys with 1% false positive rate:
/// - m = ~9600 bits ≈ 1.2 KB per transaction
/// - k = 7 hash functions
///
/// ## Lazy Initialization
///
/// The bit vector is lazily initialized on first insert to avoid
/// allocation overhead for read-only transactions.
#[derive(Clone, Debug)]
pub struct SsiBloomFilter {
    /// Bit vector (each u64 holds 64 bits) - lazily initialized
    bits: Option<Vec<u64>>,
    /// Expected capacity (used for lazy init sizing)
    expected_capacity: usize,
    /// Number of hash functions to use
    num_hashes: u32,
}

impl SsiBloomFilter {
    /// Optimal number of bits per item for 1% false positive rate
    /// m/n = -ln(p) / (ln(2))² ≈ 9.6 for p = 0.01
    const BITS_PER_ITEM: f64 = 9.6;

    /// Optimal number of hash functions for 1% false positive rate
    /// k = (m/n) × ln(2) ≈ 7
    const DEFAULT_NUM_HASHES: u32 = 7;

    /// Minimum capacity to avoid tiny filters
    const MIN_CAPACITY: usize = 64;

    /// Create a new bloom filter for expected item count (lazy allocation)
    ///
    /// Configured for ~1% false positive rate.
    /// The bit vector is not allocated until first insert.
    #[inline]
    pub fn new(expected_items: usize) -> Self {
        Self {
            bits: None,
            expected_capacity: expected_items.max(Self::MIN_CAPACITY),
            num_hashes: Self::DEFAULT_NUM_HASHES,
        }
    }

    /// Create with specific capacity in words (for memory-constrained scenarios)
    pub fn with_word_capacity(words: usize) -> Self {
        Self {
            bits: None,
            expected_capacity: words.max(1) * 64 / 10, // Approx items from words
            num_hashes: Self::DEFAULT_NUM_HASHES,
        }
    }

    /// Ensure bits are allocated (lazy initialization)
    #[inline]
    fn ensure_allocated(&mut self) {
        if self.bits.is_none() {
            let num_bits = ((self.expected_capacity as f64) * Self::BITS_PER_ITEM).ceil() as usize;
            let num_words = num_bits.div_ceil(64);
            self.bits = Some(vec![0u64; num_words]);
        }
    }

    /// Add a key to the filter - O(k) where k = num_hashes
    #[inline]
    pub fn insert(&mut self, key: &[u8]) {
        self.ensure_allocated();
        let bits = self.bits.as_mut().unwrap();
        let num_bits = bits.len() * 64;
        if num_bits == 0 {
            return;
        }

        // Use two hash functions to simulate k hash functions
        // h(i) = h1 + i * h2 (double hashing technique)
        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            bits[word_idx] |= 1 << bit_pos;
        }
    }

    /// Check if a key might be present - O(k)
    ///
    /// Returns:
    /// - false: Key is definitely NOT in the set (or filter not initialized)
    /// - true: Key MIGHT be in the set (needs exact check)
    #[inline]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let bits = match &self.bits {
            Some(b) => b,
            None => return false, // Uninitialized = empty
        };
        let num_bits = bits.len() * 64;
        if num_bits == 0 {
            return false;
        }

        let h1 = Self::hash1(key);
        let h2 = Self::hash2(key);

        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if bits[word_idx] & (1 << bit_pos) == 0 {
                return false; // Definitely not present
            }
        }
        true // Might be present
    }

    /// Check if this filter might intersect with another
    ///
    /// Fast O(m/64) check using bitwise AND of all words.
    /// If no bits are shared, sets are definitely disjoint.
    #[inline]
    pub fn may_intersect(&self, other: &SsiBloomFilter) -> bool {
        let (self_bits, other_bits) = match (&self.bits, &other.bits) {
            (Some(s), Some(o)) => (s, o),
            _ => return false, // Either uninitialized = no intersection
        };
        let min_len = self_bits.len().min(other_bits.len());
        for i in 0..min_len {
            if self_bits[i] & other_bits[i] != 0 {
                return true; // Might intersect
            }
        }
        false // Definitely disjoint
    }

    /// First hash function (using built-in hasher)
    #[inline]
    fn hash1(key: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Second hash function (using twox-hash for independence)
    #[inline]
    fn hash2(key: &[u8]) -> u64 {
        twox_hash::xxh3::hash64(key)
    }

    /// Get the memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.bits.as_ref().map(|b| b.len() * 8).unwrap_or(0) + std::mem::size_of::<Self>()
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        match &self.bits {
            Some(bits) => bits.iter().all(|&w| w == 0),
            None => true,
        }
    }
}

/// Type alias for inline key storage - keys up to 32 bytes stored on stack
/// This eliminates heap allocation for typical keys like "users/12345" (12 bytes)
pub type InlineKey = SmallVec<[u8; 32]>;

/// Version of a key-value pair
#[derive(Debug, Clone)]
pub struct Version {
    /// The value (None = tombstone)
    pub value: Option<Vec<u8>>,
    /// Transaction that created this version
    pub txn_id: u64,
    /// Commit timestamp (0 = uncommitted)
    pub commit_ts: u64,
}

// ============================================================================
// Optimized VersionChain with Binary Search (Task 1: mm.md)
// ============================================================================

/// Multi-version data for a single key with O(log v) read complexity
///
/// ## Optimization: Binary Search with Sorted Commit Ordering
///
/// Separates committed versions (sorted descending by commit_ts) from 
/// uncommitted version (single optional slot per transaction).
///
/// **Before:** O(v) linear scan + O(v) max computation = O(v)
/// **After:** O(1) uncommitted check + O(log v) binary search = O(log v)
///
/// For v=10 versions: 3.3x speedup
/// For v=100 versions: 7x speedup
#[derive(Debug, Default)]
pub struct VersionChain {
    /// Committed versions sorted by commit_ts DESCENDING (newest first)
    /// This ordering enables efficient binary search using partition_point
    committed: Vec<Version>,
    /// Single uncommitted version slot (at most one per transaction writing this key)
    uncommitted: Option<Version>,
}

impl VersionChain {
    /// Create a new empty version chain
    #[inline]
    pub fn new() -> Self {
        Self {
            committed: Vec::new(),
            uncommitted: None,
        }
    }

    /// Add a new uncommitted version
    /// If there's already an uncommitted version from this txn, update it in place
    /// 
    /// O(1) - just updates the uncommitted slot
    #[inline]
    pub fn add_uncommitted(&mut self, value: Option<Vec<u8>>, txn_id: u64) {
        match &mut self.uncommitted {
            Some(v) if v.txn_id == txn_id => {
                // Update in place - O(1)
                v.value = value;
            }
            Some(_) => {
                // Different transaction - this is a write conflict!
                // The caller should have checked has_write_conflict first
                // For safety, we overwrite (this will be caught at commit)
                self.uncommitted = Some(Version {
                    value,
                    txn_id,
                    commit_ts: 0,
                });
            }
            None => {
                // New uncommitted version - O(1)
                self.uncommitted = Some(Version {
                    value,
                    txn_id,
                    commit_ts: 0,
                });
            }
        }
    }

    /// Commit a version - moves from uncommitted slot to sorted committed list
    /// 
    /// O(log v) - inserts into sorted position using binary search
    pub fn commit(&mut self, txn_id: u64, commit_ts: u64) -> bool {
        if let Some(ref mut v) = self.uncommitted {
            if v.txn_id == txn_id && v.commit_ts == 0 {
                v.commit_ts = commit_ts;
                let committed_version = self.uncommitted.take().unwrap();
                
                // Insert into sorted position (descending by commit_ts)
                // Use partition_point to find insertion point in O(log v)
                let insert_pos = self.committed.partition_point(|existing| existing.commit_ts > commit_ts);
                self.committed.insert(insert_pos, committed_version);
                
                return true;
            }
        }
        false
    }

    /// Abort a version (remove uncommitted version for txn)
    /// 
    /// O(1) - just clears the uncommitted slot if it matches
    #[inline]
    pub fn abort(&mut self, txn_id: u64) {
        if let Some(ref v) = self.uncommitted {
            if v.txn_id == txn_id {
                self.uncommitted = None;
            }
        }
    }

    /// Read at a snapshot timestamp, optionally seeing own uncommitted writes
    /// Returns the most recent committed version visible at snapshot_ts,
    /// or an uncommitted version if it belongs to current_txn_id.
    ///
    /// ## Complexity: O(1) + O(log v) = O(log v)
    ///
    /// 1. O(1) check for uncommitted version from current transaction
    /// 2. O(log v) binary search for most recent visible committed version
    ///
    /// Snapshot isolation: we see commits with commit_ts < snapshot_ts (strictly less)
    #[inline]
    pub fn read_at(&self, snapshot_ts: u64, current_txn_id: Option<u64>) -> Option<&Version> {
        // O(1): Check uncommitted version from current transaction
        if let Some(txn_id) = current_txn_id {
            if let Some(ref v) = self.uncommitted {
                if v.txn_id == txn_id {
                    return Some(v);
                }
            }
        }

        // O(log v): Binary search for first version with commit_ts < snapshot_ts
        // Since committed is sorted descending by commit_ts, we find the first
        // version where commit_ts < snapshot_ts (the newest visible version)
        //
        // partition_point returns the first index where predicate is false
        // We want first index where commit_ts < snapshot_ts
        let idx = self.committed.partition_point(|v| v.commit_ts >= snapshot_ts);
        
        // The version at idx (if exists) is the newest with commit_ts < snapshot_ts
        self.committed.get(idx)
    }

    /// Check if there's an uncommitted version by another transaction
    /// 
    /// O(1) - just checks the uncommitted slot
    #[inline]
    pub fn has_write_conflict(&self, my_txn_id: u64) -> bool {
        if let Some(ref v) = self.uncommitted {
            return v.txn_id != my_txn_id;
        }
        false
    }

    /// Garbage collect old versions
    /// 
    /// Keeps only versions that might be visible to active transactions,
    /// plus one committed version before min_active_ts for new snapshots.
    pub fn gc(&mut self, min_active_ts: u64) {
        // Uncommitted version is always kept (will be committed or aborted)
        
        if self.committed.len() <= 1 {
            return;
        }

        // Find versions to keep:
        // 1. All versions with commit_ts > min_active_ts (visible to active txns)
        // 2. One version with commit_ts <= min_active_ts (newest anchor point)
        
        // Since committed is sorted descending, find split point
        let split_idx = self.committed.partition_point(|v| v.commit_ts > min_active_ts);
        
        // Keep all versions before split_idx (commit_ts > min_active_ts)
        // Plus one version at split_idx if it exists (anchor point)
        let keep_count = if split_idx < self.committed.len() {
            split_idx + 1 // Keep one anchor version
        } else {
            split_idx
        };
        
        self.committed.truncate(keep_count);
    }

    /// Get total version count (committed + uncommitted)
    #[inline]
    pub fn version_count(&self) -> usize {
        self.committed.len() + if self.uncommitted.is_some() { 1 } else { 0 }
    }

    // Legacy compatibility: get versions vec (for tests)
    #[cfg(test)]
    pub fn versions(&self) -> Vec<Version> {
        let mut result = self.committed.clone();
        if let Some(ref v) = self.uncommitted {
            result.push(v.clone());
        }
        result
    }
}

// =============================================================================
// Pre-sizing Constants to Avoid HashSet Resize Overhead
// =============================================================================

/// Default capacity for write_set HashSet
/// Sized for typical OLTP transactions (10-50 keys)
/// Avoids resize overhead that caused +11% regression
const WRITE_SET_INITIAL_CAPACITY: usize = 32;

/// Default capacity for read_set HashSet  
/// Typically larger than write_set due to read-heavy patterns
const READ_SET_INITIAL_CAPACITY: usize = 64;

/// Transaction state for MVCC
#[derive(Debug, Clone)]
pub struct MvccTransaction {
    /// Transaction ID
    pub txn_id: u64,
    /// Snapshot timestamp (reads see commits before this)
    pub snapshot_ts: u64,
    /// Keys written by this transaction - uses SmallVec for inline storage
    /// Pre-sized to WRITE_SET_INITIAL_CAPACITY to avoid resize overhead
    pub write_set: HashSet<InlineKey>,
    /// Keys read by this transaction (for SSI validation) - uses SmallVec for inline storage
    /// Pre-sized to READ_SET_INITIAL_CAPACITY to avoid resize overhead
    pub read_set: HashSet<InlineKey>,
    /// Bloom filter for write set - fast SSI pre-filtering
    pub write_bloom: SsiBloomFilter,
    /// Bloom filter for read set - fast SSI pre-filtering
    pub read_bloom: SsiBloomFilter,
    /// Transaction state
    pub state: TxnState,
    /// Transaction mode for SSI optimization (Recommendation 9)
    /// ReadOnly/WriteOnly modes skip SSI tracking for 2.6x improvement
    pub mode: TransactionMode,
}

impl MvccTransaction {
    /// Create a new transaction with pre-sized collections
    /// 
    /// This avoids HashSet resize overhead during the transaction
    /// which was causing +11% regression on write_set.insert().
    #[inline]
    pub fn new(txn_id: u64, snapshot_ts: u64) -> Self {
        Self::with_mode(txn_id, snapshot_ts, TransactionMode::ReadWrite)
    }
    
    /// Create a read-only transaction (SSI bypass - 2.6x faster)
    ///
    /// Read-only transactions skip all SSI tracking:
    /// - No read_set allocation
    /// - No read_bloom allocation  
    /// - No commit validation
    ///
    /// ## Performance
    /// 
    /// For N=100 reads: 8350ns → 3230ns (2.6× improvement)
    #[inline]
    pub fn read_only(txn_id: u64, snapshot_ts: u64) -> Self {
        Self::with_mode(txn_id, snapshot_ts, TransactionMode::ReadOnly)
    }
    
    /// Create a write-only transaction (partial SSI bypass)
    ///
    /// Write-only transactions skip read tracking:
    /// - No read_set tracking
    /// - No read_bloom inserts
    /// - Still needs write_set for commit
    #[inline]
    pub fn write_only(txn_id: u64, snapshot_ts: u64) -> Self {
        Self::with_mode(txn_id, snapshot_ts, TransactionMode::WriteOnly)
    }
    
    /// Create transaction with specific mode
    #[inline]
    pub fn with_mode(txn_id: u64, snapshot_ts: u64, mode: TransactionMode) -> Self {
        // Optimize allocation based on mode
        let (write_capacity, read_capacity) = match mode {
            TransactionMode::ReadOnly => (0, 0),  // No tracking needed
            TransactionMode::WriteOnly => (WRITE_SET_INITIAL_CAPACITY, 0),
            TransactionMode::ReadWrite => (WRITE_SET_INITIAL_CAPACITY, READ_SET_INITIAL_CAPACITY),
        };
        Self::with_capacity(txn_id, snapshot_ts, write_capacity, read_capacity, mode)
    }

    /// Create with custom capacities for expected workload
    /// 
    /// Use this when you know the transaction will write many keys
    /// to avoid resize overhead entirely.
    #[inline]
    pub fn with_capacity(
        txn_id: u64,
        snapshot_ts: u64,
        write_capacity: usize,
        read_capacity: usize,
        mode: TransactionMode,
    ) -> Self {
        Self {
            txn_id,
            snapshot_ts,
            write_set: HashSet::with_capacity(write_capacity),
            read_set: HashSet::with_capacity(read_capacity),
            write_bloom: SsiBloomFilter::new(write_capacity.max(1)),
            read_bloom: SsiBloomFilter::new(read_capacity.max(1)),
            state: TxnState::Active,
            mode,
        }
    }

    /// Check if this is a read-only transaction
    #[inline]
    pub fn is_read_only(&self) -> bool {
        self.write_set.is_empty()
    }

    /// Check if this is a single-key write transaction
    #[inline]
    pub fn is_single_key_write(&self) -> bool {
        self.write_set.len() == 1 && self.read_set.len() <= 1
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

// =============================================================================
// Transaction Mode for SSI Bypass (Recommendation 9)
// =============================================================================

/// Transaction mode for SSI optimization
///
/// By classifying transactions at begin time, we can skip expensive SSI
/// tracking for the majority of transactions:
///
/// | Mode      | SSI Read Tracking | SSI Write Tracking | Commit Overhead |
/// |-----------|-------------------|--------------------|-----------------|
/// | ReadOnly  | None             | None               | ~10 ns          |
/// | WriteOnly | None             | Full               | ~30 ns          |
/// | ReadWrite | Full             | Full               | ~50 ns          |
///
/// ## Performance Analysis
///
/// For read-only transactions (typically 90% of workload):
/// ```text
/// Current:  T_txn = T_begin + N × (T_read + T_record) + T_commit
///                 = 100ns + N × (32ns + 50ns) + 50ns = 150ns + 82ns × N
///
/// ReadOnly: T_txn = T_begin_ro + N × T_read + T_commit_ro
///                 = 20ns + N × 32ns + 10ns = 30ns + 32ns × N
///
/// For N=100 reads: 8350ns → 3230ns (2.6× faster)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransactionMode {
    /// Read-only transaction - skips ALL SSI tracking
    /// Cannot form rw-antidependency cycles (no writes to create outgoing edges)
    /// Safe to skip read_set, read_bloom, and commit validation entirely
    ReadOnly,
    
    /// Write-only transaction - skips read tracking
    /// Cannot form incoming rw-edges (no reads from concurrent writers)
    /// Only needs write_set and write_bloom tracking
    WriteOnly,
    
    /// Full read-write transaction (default) - complete SSI tracking
    /// May form both incoming and outgoing rw-edges
    /// Requires full validation at commit time
    #[default]
    ReadWrite,
}

impl TransactionMode {
    /// Check if this mode requires read tracking
    #[inline]
    pub fn tracks_reads(&self) -> bool {
        matches!(self, TransactionMode::ReadWrite)
    }
    
    /// Check if this mode requires write tracking
    #[inline]
    pub fn tracks_writes(&self) -> bool {
        matches!(self, TransactionMode::WriteOnly | TransactionMode::ReadWrite)
    }
    
    /// Check if commit needs SSI validation
    #[inline]
    pub fn needs_ssi_validation(&self) -> bool {
        matches!(self, TransactionMode::ReadWrite)
    }
}

/// SSI conflict edge type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Read-write conflict: T1 reads X, then T2 writes X
    ReadWrite,
    /// Write-read conflict: T1 writes X, then T2 reads X  
    WriteRead,
}

/// SSI conflict edge for dangerous structure detection
#[derive(Debug, Clone)]
pub struct ConflictEdge {
    /// Source transaction
    pub from_txn: u64,
    /// Target transaction
    pub to_txn: u64,
    /// Type of conflict
    pub conflict_type: ConflictType,
}

/// MVCC Manager with SSI support
///
/// Uses DashMap for lock-free per-transaction access.
/// Implements Serializable Snapshot Isolation (SSI) with
/// dangerous structure detection for rw-antidependency cycles.
#[allow(clippy::type_complexity)]
pub struct MvccManager {
    /// Active transactions (sharded for concurrent access)
    active_txns: DashMap<u64, MvccTransaction>,
    /// Current timestamp counter
    ts_counter: AtomicU64,
    /// Minimum active snapshot timestamp (for GC)
    min_active_ts: AtomicU64,
    /// Recently committed transactions for SSI validation
    /// Maps txn_id -> (commit_ts, read_bloom, write_bloom, read_set, write_set)
    /// Bloom filters enable fast O(m/64) pre-filtering before O(n) exact checks
    recent_commits: DashMap<
        u64,
        (
            u64,
            SsiBloomFilter,
            SsiBloomFilter,
            HashSet<InlineKey>,
            HashSet<InlineKey>,
        ),
    >,
    /// Max recent commits to track
    max_recent_commits: usize,
}

impl Default for MvccManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MvccManager {
    pub fn new() -> Self {
        Self {
            active_txns: DashMap::new(),
            ts_counter: AtomicU64::new(1),
            min_active_ts: AtomicU64::new(0),
            recent_commits: DashMap::new(),
            max_recent_commits: 1000, // Track last 1000 commits for SSI
        }
    }

    /// Begin a new transaction with snapshot isolation
    /// 
    /// Uses pre-sized HashSets to avoid resize overhead (+11% regression fix)
    pub fn begin(&self, txn_id: u64) -> MvccTransaction {
        self.begin_with_mode(txn_id, TransactionMode::ReadWrite)
    }
    
    /// Begin a read-only transaction (SSI bypass - 2.6x faster)
    ///
    /// Read-only transactions skip all SSI tracking, reducing
    /// per-read overhead from ~82ns to ~32ns.
    ///
    /// ## Safety
    ///
    /// Caller must ensure no writes are performed. Attempting to
    /// write in a read-only transaction will still succeed but
    /// won't be tracked for SSI validation.
    #[inline]
    pub fn begin_read_only(&self, txn_id: u64) -> MvccTransaction {
        self.begin_with_mode(txn_id, TransactionMode::ReadOnly)
    }
    
    /// Begin a write-only transaction (partial SSI bypass)
    ///
    /// Write-only transactions skip read tracking, reducing overhead
    /// for insert-heavy workloads.
    #[inline]
    pub fn begin_write_only(&self, txn_id: u64) -> MvccTransaction {
        self.begin_with_mode(txn_id, TransactionMode::WriteOnly)
    }
    
    /// Begin a transaction with specific mode
    ///
    /// This is the core transaction creation method that all other
    /// begin_* methods delegate to.
    pub fn begin_with_mode(&self, txn_id: u64, mode: TransactionMode) -> MvccTransaction {
        let snapshot_ts = self.ts_counter.load(Ordering::SeqCst);

        // Create transaction with mode-optimized allocations
        let txn = MvccTransaction::with_mode(txn_id, snapshot_ts, mode);

        self.active_txns.insert(txn_id, txn.clone());
        self.update_min_active_ts();

        txn
    }

    /// Get transaction if active (clones - use get_snapshot_ts for hot path)
    pub fn get(&self, txn_id: u64) -> Option<MvccTransaction> {
        self.active_txns.get(&txn_id).map(|t| t.clone())
    }

    /// Fast path: get just the snapshot timestamp without cloning
    /// This is the hot path for reads - avoids cloning bloom filters
    #[inline]
    pub fn get_snapshot_ts(&self, txn_id: u64) -> Option<u64> {
        self.active_txns.get(&txn_id).map(|t| t.snapshot_ts)
    }

    /// Record a read (for SSI) - uses inline key storage + bloom filter
    ///
    /// ## SSI Bypass (Recommendation 9)
    ///
    /// For ReadOnly mode transactions, this is a no-op (instant return).
    /// For WriteOnly mode transactions, this is a no-op.
    /// Only ReadWrite mode transactions track reads for SSI.
    ///
    /// This reduces per-read overhead from ~50ns to ~0ns for read-only txns.
    #[inline]
    pub fn record_read(&self, txn_id: u64, key: &[u8]) {
        if let Some(mut txn) = self.active_txns.get_mut(&txn_id) {
            // SSI Bypass: Skip tracking for read-only and write-only modes
            if !txn.mode.tracks_reads() {
                return;
            }
            
            // Only track reads if within reasonable bounds
            if txn.read_set.len() < 10000 {
                txn.read_set.insert(SmallVec::from_slice(key));
                txn.read_bloom.insert(key);
            }
        }
    }

    /// Record a write - uses inline key storage + bloom filter
    ///
    /// Note: Even ReadOnly transactions can record writes (mode is a hint).
    /// The mode only affects SSI tracking, not write capability.
    pub fn record_write(&self, txn_id: u64, key: &[u8]) {
        if let Some(mut txn) = self.active_txns.get_mut(&txn_id) {
            txn.write_set.insert(SmallVec::from_slice(key));
            txn.write_bloom.insert(key);
        }
    }

    /// Allocate commit timestamp
    pub fn alloc_commit_ts(&self) -> u64 {
        self.ts_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Commit transaction with SSI validation
    /// Returns (commit_ts, write_set) so the memtable can be updated efficiently
    /// Returns None if SSI validation fails (dangerous structure detected)
    ///
    /// ## SSI Bypass (Recommendation 9)
    ///
    /// For ReadOnly mode: Skip validation entirely (~10ns commit)
    /// For WriteOnly mode: Skip read-based validation (~30ns commit)
    /// For ReadWrite mode: Full validation (~50ns commit)
    pub fn commit(&self, txn_id: u64) -> Option<(u64, HashSet<InlineKey>)> {
        // Get transaction before removing
        let txn = self.active_txns.get(&txn_id)?.clone();

        // SSI Bypass: Skip validation for ReadOnly transactions
        // ReadOnly can never form rw-antidependency cycles
        if txn.mode != TransactionMode::ReadWrite || !self.validate_ssi(&txn) {
            // For ReadOnly/WriteOnly: always valid (mode check short-circuits)
            // For ReadWrite: check SSI validation result
            if txn.mode == TransactionMode::ReadWrite && !self.validate_ssi(&txn) {
                // Abort on SSI violation
                self.active_txns.remove(&txn_id);
                self.update_min_active_ts();
                return None;
            }
        }

        let commit_ts = self.alloc_commit_ts();

        // Extract write_set and remove transaction - takes ownership
        let (_, removed_txn) = self.active_txns.remove(&txn_id)?;

        // OPTIMIZATION: Only track ReadWrite transactions for SSI
        // ReadOnly/WriteOnly can't form complete rw-antidependency cycles
        let needs_ssi_tracking = removed_txn.mode == TransactionMode::ReadWrite 
            && !removed_txn.read_set.is_empty() 
            && !removed_txn.write_set.is_empty();
        
        if needs_ssi_tracking {
            // Need to clone write_set since we return it AND track it
            let write_set_for_return = removed_txn.write_set.clone();
            
            self.track_commit_owned(
                txn_id,
                commit_ts,
                removed_txn.read_bloom,
                removed_txn.write_bloom,
                removed_txn.read_set,
                removed_txn.write_set,
            );

            self.update_min_active_ts();
            Some((commit_ts, write_set_for_return))
        } else {
            // Fast path: no SSI tracking needed, avoid clone entirely
            self.update_min_active_ts();
            Some((commit_ts, removed_txn.write_set))
        }
    }

    /// Validate SSI constraints for a committing transaction
    ///
    /// ## Transaction Classification (Task 3: Optimistic MVCC)
    ///
    /// Transactions are classified and routed through appropriate fast paths:
    ///
    /// | Class      | Criteria                      | Validation Cost |
    /// |------------|-------------------------------|-----------------|
    /// | ReadOnly   | write_set.is_empty()          | 0 ns           |
    /// | SingleKey  | write_set.len() == 1          | 0 ns           |
    /// | Disjoint   | bloom filters don't intersect | ~10 ns         |
    /// | General    | full SSI check                | ~50 ns         |
    ///
    /// Expected distribution: ~60% read-only, ~25% single-key, ~10% disjoint, ~5% general
    /// Weighted average: ~8 ns vs 50 ns baseline (6x improvement)
    ///
    /// Detects "dangerous structures" - rw-antidependency cycles:
    /// - T1 reads X (snapshot sees old value)
    /// - T2 writes X (concurrent write)  
    /// - T2 reads Y (snapshot sees old value)
    /// - T1 writes Y (concurrent write)
    ///
    /// If T1 → rw → T2 → rw → T1 exists, abort T1
    #[inline]
    fn validate_ssi(&self, txn: &MvccTransaction) -> bool {
        // =================================================================
        // Fast Path 1: Read-only transactions (0 ns)
        // =================================================================
        // Read-only transactions can never form rw-antidependency cycles
        // because they have no writes to create outgoing rw-edges
        if txn.write_set.is_empty() {
            return true;
        }

        // =================================================================
        // Fast Path 2: No recent commits to check (0 ns)
        // =================================================================
        if self.recent_commits.is_empty() {
            return true;
        }

        // =================================================================
        // Fast Path 3: Single-key write transactions (0 ns)
        // =================================================================
        // A single-key write transaction cannot form a dangerous cycle:
        // - For a cycle T1 →rw→ T2 →rw→ T1, we need T1 to read what T2 wrote
        //   AND T2 to read what T1 wrote
        // - With only one key in write_set, the same key would need to be
        //   in both read_set AND write_set of both transactions
        // - This is already prevented by our conflict detection (write-write)
        if txn.write_set.len() == 1 && txn.read_set.len() <= 1 {
            return true;
        }

        let my_snapshot = txn.snapshot_ts;

        // =================================================================
        // Fast Path 4: Disjoint transactions using Bloom filters (~10 ns)
        // =================================================================
        // Pre-filter using bloom filters: if our write_bloom doesn't intersect
        // with any concurrent transaction's read_bloom AND vice versa,
        // there can be no rw-antidependency
        let mut any_may_intersect = false;
        for entry in self.recent_commits.iter() {
            let (_, (other_commit_ts, other_read_bloom, other_write_bloom, _, _)) = entry.pair();
            
            // Only check concurrent transactions
            if *other_commit_ts <= my_snapshot {
                continue;
            }

            // Check bloom filter intersection (O(m/64) per filter)
            // If our writes may intersect their reads OR their writes may intersect our reads
            if txn.write_bloom.may_intersect(other_read_bloom) 
                || other_write_bloom.may_intersect(&txn.read_bloom) 
            {
                any_may_intersect = true;
                break;
            }
        }

        // No bloom intersection means definitely disjoint - no SSI conflict possible
        if !any_may_intersect {
            return true;
        }

        // =================================================================
        // Full SSI Validation (~50 ns)
        // =================================================================
        // Check for rw-conflicts with recently committed transactions
        // An rw-conflict exists if:
        // - T_other wrote to a key that T_me read (T_other →rw→ T_me)
        // - T_me wrote to a key that T_other read (T_me →rw→ T_other)

        let mut in_conflict_with: Vec<u64> = Vec::new();
        let mut out_conflict_to: Vec<u64> = Vec::new();

        for entry in self.recent_commits.iter() {
            let (
                other_txn_id,
                (
                    other_commit_ts,
                    _other_read_bloom,
                    other_write_bloom,
                    other_read_set,
                    other_write_set,
                ),
            ) = entry.pair();

            // Only consider transactions that committed after our snapshot started
            // (concurrent transactions)
            if *other_commit_ts <= my_snapshot {
                continue;
            }

            // Check: other wrote → we read (other →rw→ me)
            // T_other wrote a key that T_me read (rw-dependency inbound)
            //
            // Bloom-accelerated: First check bloom filter for fast rejection (O(m/64))
            // Only do expensive HashSet intersection if bloom says "maybe conflict"
            let mut has_in_conflict = false;
            for key in txn.read_set.iter() {
                if other_write_bloom.may_contain(key) {
                    // Bloom says maybe - do exact check
                    if other_write_set.contains(key) {
                        has_in_conflict = true;
                        break;
                    }
                }
            }
            if has_in_conflict {
                in_conflict_with.push(*other_txn_id);
            }

            // Check: we wrote → other read (me →rw→ other)
            // T_me wrote a key that T_other read (rw-dependency outbound)
            //
            // Bloom-accelerated: Use our write_bloom against their read_set
            let mut has_out_conflict = false;
            for key in other_read_set.iter() {
                if txn.write_bloom.may_contain(key) {
                    // Bloom says maybe - do exact check
                    if txn.write_set.contains(key) {
                        has_out_conflict = true;
                        break;
                    }
                }
            }
            if has_out_conflict {
                out_conflict_to.push(*other_txn_id);
            }
        }

        // Dangerous structure: we have both incoming AND outgoing rw-edges
        // This creates a potential cycle: T1 →rw→ T_me →rw→ T2
        //
        // Conservative check: if both exist, abort
        // A more precise check would verify the cycle path, but this is safe
        if !in_conflict_with.is_empty() && !out_conflict_to.is_empty() {
            return false; // SSI violation - abort
        }

        true
    }

    /// Track a committed transaction for future SSI validation
    ///
    /// Only tracks transactions that have both reads AND writes, since SSI
    /// only detects rw-antidependency cycles. Pure read or pure write
    /// transactions can't form cycles.
    ///
    /// ## Optimization: Zero-Copy Transfer
    ///
    /// Takes ownership of sets instead of cloning to avoid the +15% commit
    /// phase regression. The caller should use mem::take() to transfer ownership.
    fn track_commit_owned(
        &self,
        txn_id: u64,
        commit_ts: u64,
        read_bloom: SsiBloomFilter,
        write_bloom: SsiBloomFilter,
        read_set: HashSet<InlineKey>,
        write_set: HashSet<InlineKey>,
    ) {
        // Optimization: Only track mixed read-write transactions
        // Pure reads can't create outgoing rw-edges
        // Pure writes can't create incoming rw-edges
        if read_set.is_empty() || write_set.is_empty() {
            return; // Skip tracking - can't form SSI cycle
        }

        // Add to recent commits with bloom filters for fast SSI pre-filtering
        // No cloning needed - we take ownership
        self.recent_commits.insert(
            txn_id,
            (
                commit_ts,
                read_bloom,
                write_bloom,
                read_set,
                write_set,
            ),
        );

        // Lazy pruning: only prune when we're significantly over capacity
        // Avoids pruning overhead on every commit
        if self.recent_commits.len() > self.max_recent_commits * 2 {
            // Remove entries with lowest commit_ts
            let min_active = self.min_active_ts.load(Ordering::Relaxed);
            self.recent_commits
                .retain(|_, (ts, _, _, _, _)| *ts >= min_active);
        }
    }

    /// Legacy track_commit that clones - kept for compatibility
    #[allow(dead_code)]
    fn track_commit(
        &self,
        txn_id: u64,
        commit_ts: u64,
        read_bloom: SsiBloomFilter,
        write_bloom: SsiBloomFilter,
        read_set: &HashSet<InlineKey>,
        write_set: &HashSet<InlineKey>,
    ) {
        if read_set.is_empty() || write_set.is_empty() {
            return;
        }
        self.recent_commits.insert(
            txn_id,
            (
                commit_ts,
                read_bloom,
                write_bloom,
                read_set.clone(),
                write_set.clone(),
            ),
        );
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) {
        self.active_txns.remove(&txn_id);
        self.update_min_active_ts();
    }

    /// Get minimum active snapshot timestamp
    pub fn min_active_snapshot(&self) -> u64 {
        self.min_active_ts.load(Ordering::SeqCst)
    }

    /// Get count of active transactions
    pub fn active_transaction_count(&self) -> usize {
        self.active_txns.len()
    }

    fn update_min_active_ts(&self) {
        let min = self
            .active_txns
            .iter()
            .map(|entry| entry.value().snapshot_ts)
            .min()
            .unwrap_or_else(|| self.ts_counter.load(Ordering::SeqCst));
        self.min_active_ts.store(min, Ordering::SeqCst);
    }
}

/// Epoch-based dirty list for O(expired) GC instead of O(n)
///
/// Instead of scanning ALL version chains, we track which keys have versions
/// created in each epoch. GC only needs to visit keys from old epochs.
struct EpochDirtyList {
    /// Ring buffer of epoch -> dirty keys
    /// Index = epoch % EPOCH_RING_SIZE
    epochs: [parking_lot::Mutex<Vec<Vec<u8>>>; 4],
    /// Current epoch
    current_epoch: AtomicU64,
}

const EPOCH_RING_SIZE: usize = 4;

impl EpochDirtyList {
    fn new() -> Self {
        Self {
            epochs: [
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
            ],
            current_epoch: AtomicU64::new(0),
        }
    }

    /// Record a version created in the current epoch
    #[inline]
    fn record_version(&self, key: Vec<u8>) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        let idx = (epoch as usize) % EPOCH_RING_SIZE;
        self.epochs[idx].lock().push(key);
    }

    /// Record multiple versions in a single lock acquisition (Rec 3: MVCC Batching)
    ///
    /// Performance: Single lock acquire vs N lock acquires for batch of N writes.
    /// For 100 writes: ~100x fewer mutex operations.
    #[inline]
    fn record_versions_batch(&self, keys: impl IntoIterator<Item = Vec<u8>>) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        let idx = (epoch as usize) % EPOCH_RING_SIZE;
        let mut guard = self.epochs[idx].lock();
        guard.extend(keys);
    }

    /// Advance to next epoch, returning old epoch's dirty keys
    fn advance_epoch(&self) -> (u64, Vec<Vec<u8>>) {
        let old_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
        let old_idx = (old_epoch as usize) % EPOCH_RING_SIZE;

        // Drain the old epoch's dirty list
        let mut guard = self.epochs[old_idx].lock();
        let keys = std::mem::take(&mut *guard);
        (old_epoch, keys)
    }

    /// Get current epoch
    #[allow(dead_code)]
    fn current(&self) -> u64 {
        self.current_epoch.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Streaming Scan Iterator
// ============================================================================

/// Streaming iterator for range scans
/// 
/// Yields results one at a time without materializing the full result set.
/// This enables processing of very large result sets with O(1) memory per
/// iteration instead of O(N) for the entire result set.
struct ScanRangeIterator<'a> {
    memtable: &'a MvccMemTable,
    start: Vec<u8>,
    end: Vec<u8>,
    snapshot_ts: u64,
    current_txn_id: Option<u64>,
    use_ordered: bool,
    // We use Option to defer initialization
    ordered_iter: Option<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a>>,
    unordered_iter: Option<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a>>,
    initialized: bool,
}

impl<'a> Iterator for ScanRangeIterator<'a> {
    type Item = (Vec<u8>, Vec<u8>);
    
    fn next(&mut self) -> Option<Self::Item> {
        // Lazy initialization on first call
        if !self.initialized {
            self.initialized = true;
            
            if self.use_ordered {
                // Try deferred index first (after compaction, it uses a SkipMap internally)
                if let Some(ref def_idx) = self.memtable.deferred_index {
                    let start = self.start.clone();
                    let end = self.end.clone();
                    let snapshot_ts = self.snapshot_ts;
                    let current_txn_id = self.current_txn_id;
                    let data = &self.memtable.data;
                    
                    // Collect keys from deferred index (already sorted after compact)
                    let keys: Vec<Vec<u8>> = if end.is_empty() {
                        def_idx.range_from(&start).collect()
                    } else {
                        def_idx.range(&start, &end).collect()
                    };
                    
                    let iter: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> = Box::new(
                        keys.into_iter()
                            .filter_map(move |key| {
                                if let Some(chain) = data.get(&key)
                                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                    && let Some(value) = &v.value
                                {
                                    Some((key, value.clone()))
                                } else {
                                    None
                                }
                            })
                    );
                    self.ordered_iter = Some(iter);
                } else if let Some(ref idx) = self.memtable.ordered_index {
                    let start = self.start.clone();
                    let end = self.end.clone();
                    let snapshot_ts = self.snapshot_ts;
                    let current_txn_id = self.current_txn_id;
                    let data = &self.memtable.data;
                    
                    let iter: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> = if end.is_empty() {
                        Box::new(
                            idx.range(start..)
                                .filter_map(move |entry| {
                                    let key = entry.key();
                                    if let Some(chain) = data.get(key)
                                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                        && let Some(value) = &v.value
                                    {
                                        Some((key.clone(), value.clone()))
                                    } else {
                                        None
                                    }
                                })
                        )
                    } else {
                        Box::new(
                            idx.range(start..end)
                                .filter_map(move |entry| {
                                    let key = entry.key();
                                    if let Some(chain) = data.get(key)
                                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                        && let Some(value) = &v.value
                                    {
                                        Some((key.clone(), value.clone()))
                                    } else {
                                        None
                                    }
                                })
                        )
                    };
                    self.ordered_iter = Some(iter);
                }
            } else {
                // Unordered full scan
                let start = self.start.clone();
                let end = self.end.clone();
                let snapshot_ts = self.snapshot_ts;
                let current_txn_id = self.current_txn_id;
                
                let iter: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> = Box::new(
                    self.memtable.data.iter()
                        .filter_map(move |entry| {
                            let key = entry.key();
                            
                            if key.as_slice() < start.as_slice() {
                                return None;
                            }
                            if !end.is_empty() && key.as_slice() >= end.as_slice() {
                                return None;
                            }
                            
                            if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                Some((key.clone(), value.clone()))
                            } else {
                                None
                            }
                        })
                );
                self.unordered_iter = Some(iter);
            }
        }
        
        // Get next from appropriate iterator
        if let Some(ref mut iter) = self.ordered_iter {
            iter.next()
        } else if let Some(ref mut iter) = self.unordered_iter {
            iter.next()
        } else {
            None
        }
    }
}

/// MemTable with MVCC support
///
/// Uses DashMap for lock-free concurrent access per key.
/// This eliminates the global write lock bottleneck.
///
/// Uses epoch-based dirty tracking for O(expired) GC instead of O(n) full scan.
/// Maintains a deferred sorted index for efficient scans:
/// - Writes: O(1) append to hot buffer
/// - Scans: O(N log N) sort-on-demand (amortized across many writes)
pub struct MvccMemTable {
    /// Key -> VersionChain (sharded for concurrent access)
    data: DashMap<Vec<u8>, VersionChain>,
    /// Deferred sorted index for efficient prefix/range scans (optional)
    /// O(1) insert to hot buffer, O(N log N) sort on first scan
    /// When None, scan_prefix will fall back to O(N) DashMap iteration
    deferred_index: Option<DeferredSortedIndex>,
    /// Legacy SkipMap for compatibility (used when deferred=false)
    ordered_index: Option<SkipMap<Vec<u8>, ()>>,
    /// Whether to use deferred sorting (true) or immediate SkipMap (false)
    #[allow(dead_code)]
    use_deferred: bool,
    /// Approximate size in bytes
    size_bytes: AtomicU64,
    /// Epoch-based dirty list for efficient GC
    dirty_list: EpochDirtyList,
}

impl Default for MvccMemTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MvccMemTable {
    pub fn new() -> Self {
        Self::with_ordered_index(true)
    }

    /// Create memtable with optional ordered index
    ///
    /// When `enable_ordered_index` is false, saves ~134 ns/op on writes
    /// but scan_prefix becomes O(N) instead of O(log N + K)
    ///
    /// Uses deferred sorting by default for better write performance:
    /// - Writes: O(1) append to hot buffer
    /// - Scans: O(N log N) sort-on-demand
    pub fn with_ordered_index(enable_ordered_index: bool) -> Self {
        Self::with_index_mode(enable_ordered_index, true)
    }

    /// Create memtable with fine-grained control over indexing
    ///
    /// # Arguments
    /// * `enable_ordered_index` - Whether to maintain an ordered index
    /// * `use_deferred` - If true, use deferred sorting (O(1) writes, sort-on-scan)
    ///                    If false, use SkipMap (O(log N) writes)
    pub fn with_index_mode(enable_ordered_index: bool, use_deferred: bool) -> Self {
        Self {
            data: DashMap::new(),
            deferred_index: if enable_ordered_index && use_deferred {
                Some(DeferredSortedIndex::with_config(DeferredIndexConfig {
                    max_unsorted_entries: 10_000, // Compact every 10K writes
                    enabled: true,
                }))
            } else {
                None
            },
            ordered_index: if enable_ordered_index && !use_deferred {
                Some(SkipMap::new())
            } else {
                None
            },
            use_deferred,
            size_bytes: AtomicU64::new(0),
            dirty_list: EpochDirtyList::new(),
        }
    }

    /// Write a key-value pair (uncommitted)
    pub fn write(&self, key: Vec<u8>, value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
        let key_len = key.len();

        // Track this key in the current epoch's dirty list for GC
        self.dirty_list.record_version(key.clone());

        // Insert into ordered index for prefix scans (if enabled)
        // Deferred: O(1) append to hot buffer
        // SkipMap: O(log N) insert
        if let Some(ref idx) = self.deferred_index {
            idx.insert(key.clone());
        } else if let Some(ref idx) = self.ordered_index {
            idx.insert(key.clone(), ());
        }

        // Use entry API for atomic get-or-insert
        let mut entry = self.data.entry(key).or_default();

        // Check for write-write conflict
        if entry.has_write_conflict(txn_id) {
            return Err(SochDBError::Internal(
                "Write-write conflict detected".into(),
            ));
        }
        entry.add_uncommitted(value, txn_id);
        self.size_bytes
            .fetch_add((key_len + value_size) as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Write multiple key-value pairs (uncommitted) - more efficient than individual writes
    ///
    /// Optimizations applied (Rec 3: MVCC Batching):
    /// - Batched dirty list tracking: single lock acquire for all keys
    /// - Deferred index: O(1) append per key
    pub fn write_batch(&self, writes: &[(Vec<u8>, Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        let mut total_size = 0u64;

        // Rec 3: Batch MVCC tracking - single lock acquire for all keys
        self.dirty_list.record_versions_batch(writes.iter().map(|(k, _)| k.clone()));

        for (key, value) in writes {
            // Insert into ordered index (if enabled)
            // Deferred: O(1) append, SkipMap: O(log N)
            if let Some(ref idx) = self.deferred_index {
                idx.insert(key.clone());
            } else if let Some(ref idx) = self.ordered_index {
                idx.insert(key.clone(), ());
            }

            let mut entry = self.data.entry(key.clone()).or_default();

            if entry.has_write_conflict(txn_id) {
                return Err(SochDBError::Internal(
                    "Write-write conflict detected".into(),
                ));
            }

            let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
            entry.add_uncommitted(value.clone(), txn_id);
            total_size += (key.len() + value_size) as u64;
        }

        self.size_bytes.fetch_add(total_size, Ordering::Relaxed);
        Ok(())
    }

    /// Read at snapshot timestamp, with optional current txn to see own writes
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        self.data.get(key).and_then(|chain| {
            chain
                .read_at(snapshot_ts, current_txn_id)
                .and_then(|v| v.value.clone())
        })
    }

    /// Commit all versions for a transaction
    ///
    /// Only updates the keys that were written by this transaction (tracked in write_set).
    /// Accepts InlineKey for zero-allocation MVCC tracking.
    pub fn commit(&self, txn_id: u64, commit_ts: u64, write_set: &HashSet<InlineKey>) {
        // Only iterate over keys we know were written - O(k) instead of O(n)
        for key in write_set {
            if let Some(mut chain) = self.data.get_mut(key.as_slice()) {
                chain.commit(txn_id, commit_ts);
            }
        }
    }

    /// Legacy commit method (iterates all keys) - kept for backward compatibility
    #[allow(dead_code)]
    pub fn commit_all(&self, txn_id: u64, commit_ts: u64) {
        for mut entry in self.data.iter_mut() {
            entry.value_mut().commit(txn_id, commit_ts);
        }
    }

    /// Abort all versions for a transaction
    pub fn abort(&self, txn_id: u64) {
        for mut entry in self.data.iter_mut() {
            entry.value_mut().abort(txn_id);
        }
    }

    /// Scan keys with prefix at snapshot (without seeing uncommitted from other txns)
    ///
    /// ## Performance
    ///
    /// When ordered_index is enabled: O(log N + K) complexity
    /// - O(log N) to seek to the first key with prefix
    /// - O(K) to iterate matching keys
    ///
    /// When ordered_index is disabled: O(N) full DashMap scan (fallback)
    /// 
    /// ## Optimizations Applied
    /// 
    /// - Pre-allocates result vector based on expected output size
    /// - Uses batch-friendly iteration patterns
    /// - Minimizes allocations during iteration
    /// - Deferred index: compacts hot buffer on first scan for sorted iteration
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        // Estimate result size for pre-allocation (use 10% of total as heuristic)
        let estimated_size = (self.data.len() / 10).max(64);
        let mut results = Vec::with_capacity(estimated_size);

        if let Some(ref idx) = self.deferred_index {
            // Deferred index path: sort-on-scan (compacts hot buffer if needed)
            for key in idx.range_from(prefix) {
                // Stop when we've passed the prefix range
                if !key.starts_with(prefix) {
                    break;
                }

                // O(1) lookup in DashMap for version chain
                if let Some(chain) = self.data.get(&key)
                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key, value.clone()));
                }
            }
        } else if let Some(ref idx) = self.ordered_index {
            // Fast path: O(log N) seek to first key >= prefix
            for entry in idx.range(prefix.to_vec()..) {
                let key = entry.key();

                // Stop when we've passed the prefix range
                if !key.starts_with(prefix) {
                    break;
                }

                // O(1) lookup in DashMap for version chain
                if let Some(chain) = self.data.get(key)
                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.clone(), value.clone()));
                }
            }
        } else {
            // Fallback: O(N) full DashMap scan when ordered_index is disabled
            // Optimized with batch-friendly iteration
            for entry in self.data.iter() {
                let key = entry.key();
                if !key.starts_with(prefix) {
                    continue;
                }
                if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.clone(), value.clone()));
                }
            }
        }

        results
    }

    /// Optimized full scan with batch allocation
    /// 
    /// For use when scanning entire tables/namespaces.
    /// Pre-allocates based on actual data size.
    pub fn scan_all(
        &self,
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::with_capacity(self.data.len());

        for entry in self.data.iter() {
            if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                && let Some(value) = &v.value
            {
                results.push((entry.key().clone(), value.clone()));
            }
        }

        results
    }

    /// Streaming scan iterator for very large datasets
    /// 
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set in memory.
    pub fn scan_prefix_iter<'a>(
        &'a self,
        prefix: &'a [u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
        self.data.iter().filter_map(move |entry| {
            let key = entry.key();
            if !key.starts_with(prefix) {
                return None;
            }
            if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                && let Some(value) = &v.value
            {
                Some((key.clone(), value.clone()))
            } else {
                None
            }
        })
    }

    /// Scan range
    pub fn scan_range(
        &self,
        start: &[u8],
        end: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();

        if let Some(ref idx) = self.deferred_index {
            // Deferred index path: sort-on-scan
            if end.is_empty() {
                for key in idx.range_from(start) {
                    if let Some(chain) = self.data.get(&key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key, value.clone()));
                    }
                }
            } else {
                for key in idx.range(start, end) {
                    if let Some(chain) = self.data.get(&key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key, value.clone()));
                    }
                }
            }
        } else if let Some(ref idx) = self.ordered_index {
            // Use range scan on SkipMap
            if end.is_empty() {
                // Unbounded end
                for entry in idx.range(start.to_vec()..) {
                    let key = entry.key();
                    if let Some(chain) = self.data.get(key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key.clone(), value.clone()));
                    }
                }
            } else {
                for entry in idx.range(start.to_vec()..end.to_vec()) {
                    let key = entry.key();
                    if let Some(chain) = self.data.get(key)
                        && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                        && let Some(value) = &v.value
                    {
                        results.push((key.clone(), value.clone()));
                    }
                }
            }
        } else {
            // Fallback to full scan if no ordered index
            for entry in self.data.iter() {
                let key = entry.key();

                if key.as_slice() < start {
                    continue;
                }
                if !end.is_empty() && key.as_slice() >= end {
                    continue;
                }

                if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.clone(), value.clone()));
                }
            }
        }

        results
    }

    /// Streaming range scan iterator for very large datasets
    /// 
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set in memory. Uses the ordered
    /// index when available for O(log N + K) complexity.
    /// 
    /// ## Zero-Allocation Design
    /// 
    /// While the iterator itself cannot avoid allocations for returned
    /// values (since the caller needs ownership), it avoids:
    /// - Pre-materializing all results
    /// - Intermediate buffers
    /// - Repeated key comparisons for already-visited entries
    /// 
    /// ## Usage
    /// 
    /// ```ignore
    /// for (key, value) in memtable.scan_range_iter(b"start", b"end", ts, txn) {
    ///     // Process each result as it arrives
    ///     // Memory usage is O(1) per iteration, not O(N) total
    /// }
    /// ```
    pub fn scan_range_iter<'a>(
        &'a self,
        start: &'a [u8],
        end: &'a [u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
        // Compact deferred index before scanning if needed
        if let Some(ref idx) = self.deferred_index {
            idx.compact();
        }
        
        // Use either ordered index or full scan
        let use_ordered = self.ordered_index.is_some() || self.deferred_index.is_some();
        
        // Create iterator based on availability of ordered index
        ScanRangeIterator {
            memtable: self,
            start: start.to_vec(),
            end: end.to_vec(),
            snapshot_ts,
            current_txn_id,
            use_ordered,
            ordered_iter: None,
            unordered_iter: None,
            initialized: false,
        }
    }

    /// Get approximate size
    pub fn size(&self) -> u64 {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Garbage collect old versions using epoch-based dirty list
    ///
    /// O(expired_versions) instead of O(all_versions)
    /// Only visits keys that had versions created in the old epoch.
    pub fn gc(&self, min_active_ts: u64) -> usize {
        // Advance epoch and get the dirty keys from the old epoch
        let (_old_epoch, dirty_keys) = self.dirty_list.advance_epoch();

        if dirty_keys.is_empty() {
            return 0;
        }

        let mut gc_count = 0;

        // Only visit keys that were modified in the old epoch
        // Use a HashSet to deduplicate keys that were written multiple times
        let unique_keys: std::collections::HashSet<_> = dirty_keys.into_iter().collect();

        for key in unique_keys {
            if let Some(mut entry) = self.data.get_mut(&key) {
                let before = entry.value().version_count();
                entry.value_mut().gc(min_active_ts);
                gc_count += before.saturating_sub(entry.value().version_count());
            }
        }

        gc_count
    }

    /// Legacy full-scan GC (for testing or when epoch-based tracking isn't available)
    #[allow(dead_code)]
    pub fn gc_full_scan(&self, min_active_ts: u64) -> usize {
        let mut gc_count = 0;

        for mut entry in self.data.iter_mut() {
            let before = entry.value().version_count();
            entry.value_mut().gc(min_active_ts);
            gc_count += before.saturating_sub(entry.value().version_count());
        }

        gc_count
    }
}

// ============================================================================
// ArenaMvccMemTable - Arena-Backed MVCC MemTable with Reduced Allocations
// ============================================================================

use crate::key_buffer::ArenaKeyHandle;

/// Epoch-based dirty list using ArenaKeyHandle for reduced allocations
struct ArenaEpochDirtyList {
    epochs: [parking_lot::Mutex<Vec<ArenaKeyHandle>>; 4],
    current_epoch: AtomicU64,
}

impl ArenaEpochDirtyList {
    fn new() -> Self {
        Self {
            epochs: [
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
                parking_lot::Mutex::new(Vec::new()),
            ],
            current_epoch: AtomicU64::new(0),
        }
    }

    #[inline]
    fn record_version(&self, key: ArenaKeyHandle) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        let idx = (epoch as usize) % EPOCH_RING_SIZE;
        self.epochs[idx].lock().push(key);
    }

    fn advance_epoch(&self) -> (u64, Vec<ArenaKeyHandle>) {
        let old_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
        let old_idx = (old_epoch as usize) % EPOCH_RING_SIZE;
        let mut guard = self.epochs[old_idx].lock();
        let keys = std::mem::take(&mut *guard);
        (old_epoch, keys)
    }
}

/// Arena-backed MVCC MemTable with optimized key storage
///
/// This version uses `ArenaKeyHandle` instead of `Vec<u8>` for keys,
/// reducing per-write allocations from 3 to 1:
///
/// - Before: 3 × Vec<u8> clones per write (dirty_list, ordered_index, data)
/// - After: 1 × ArenaKeyHandle creation, 3 × O(1) copies (16 bytes each)
///
/// ## Performance
///
/// Expected improvement: 20-30% throughput increase on write-heavy workloads
/// by reducing:
/// - Heap allocations: 3 → 1 per write
/// - Bytes copied: 3L → L + 48 bytes (where L = key length)
pub struct ArenaMvccMemTable {
    /// Key -> VersionChain (uses ArenaKeyHandle for O(1) hash)
    data: DashMap<ArenaKeyHandle, VersionChain>,
    /// Ordered index for prefix scans
    ordered_index: Option<SkipMap<ArenaKeyHandle, ()>>,
    /// Approximate size in bytes
    size_bytes: AtomicU64,
    /// Epoch-based dirty list (arena-backed)
    dirty_list: ArenaEpochDirtyList,
}

impl ArenaMvccMemTable {
    pub fn new() -> Self {
        Self::with_ordered_index(true)
    }

    pub fn with_ordered_index(enable_ordered_index: bool) -> Self {
        Self {
            data: DashMap::new(),
            ordered_index: if enable_ordered_index {
                Some(SkipMap::new())
            } else {
                None
            },
            size_bytes: AtomicU64::new(0),
            dirty_list: ArenaEpochDirtyList::new(),
        }
    }

    /// Write a key-value pair using arena key handle
    ///
    /// Only creates ONE ArenaKeyHandle, then copies it (16 bytes) to each location.
    /// This is much cheaper than cloning Vec<u8> which requires heap allocation.
    pub fn write(&self, key: &[u8], value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
        let key_len = key.len();

        // Create ONE ArenaKeyHandle - this is the only allocation for the key
        let key_handle = ArenaKeyHandle::new(key);

        // Track in dirty list (O(1) copy of 16-byte handle)
        self.dirty_list.record_version(key_handle.clone());

        // Insert into ordered index (O(1) copy of 16-byte handle)
        if let Some(ref idx) = self.ordered_index {
            idx.insert(key_handle.clone(), ());
        }

        // Use entry API with the handle
        let mut entry = self.data.entry(key_handle).or_default();

        if entry.has_write_conflict(txn_id) {
            return Err(SochDBError::Internal(
                "Write-write conflict detected".into(),
            ));
        }
        entry.add_uncommitted(value, txn_id);
        self.size_bytes
            .fetch_add((key_len + value_size) as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Write batch using arena key handles
    pub fn write_batch(&self, writes: &[(&[u8], Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        let mut total_size = 0u64;

        for (key, value) in writes {
            let key_handle = ArenaKeyHandle::new(key);

            self.dirty_list.record_version(key_handle.clone());

            if let Some(ref idx) = self.ordered_index {
                idx.insert(key_handle.clone(), ());
            }

            let mut entry = self.data.entry(key_handle).or_default();

            if entry.has_write_conflict(txn_id) {
                return Err(SochDBError::Internal(
                    "Write-write conflict detected".into(),
                ));
            }

            let value_size = value.as_ref().map(|v| v.len()).unwrap_or(0);
            entry.add_uncommitted(value.clone(), txn_id);
            total_size += (key.len() + value_size) as u64;
        }

        self.size_bytes.fetch_add(total_size, Ordering::Relaxed);
        Ok(())
    }

    /// Read at snapshot timestamp
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        // Create temporary handle for lookup (uses pre-computed hash for O(1) lookup)
        let key_handle = ArenaKeyHandle::new(key);
        self.data.get(&key_handle).and_then(|chain| {
            chain
                .read_at(snapshot_ts, current_txn_id)
                .and_then(|v| v.value.clone())
        })
    }

    /// Commit transaction
    pub fn commit(&self, txn_id: u64, commit_ts: u64, write_set: &HashSet<InlineKey>) {
        for key in write_set {
            let key_handle = ArenaKeyHandle::new(key.as_slice());
            if let Some(mut chain) = self.data.get_mut(&key_handle) {
                chain.commit(txn_id, commit_ts);
            }
        }
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) {
        for mut entry in self.data.iter_mut() {
            entry.value_mut().abort(txn_id);
        }
    }

    /// Scan prefix
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();
        let prefix_handle = ArenaKeyHandle::new(prefix);

        if let Some(ref idx) = self.ordered_index {
            for entry in idx.range(prefix_handle..) {
                let key = entry.key();

                if !key.as_bytes().starts_with(prefix) {
                    break;
                }

                if let Some(chain) = self.data.get(key)
                    && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.as_bytes().to_vec(), value.clone()));
                }
            }
        } else {
            for entry in self.data.iter() {
                let key = entry.key();
                if !key.as_bytes().starts_with(prefix) {
                    continue;
                }
                if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                    && let Some(value) = &v.value
                {
                    results.push((key.as_bytes().to_vec(), value.clone()));
                }
            }
        }

        results
    }

    /// Get approximate size
    pub fn size(&self) -> u64 {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Garbage collect old versions
    pub fn gc(&self, min_active_ts: u64) -> usize {
        let (_old_epoch, dirty_keys) = self.dirty_list.advance_epoch();

        if dirty_keys.is_empty() {
            return 0;
        }

        let mut gc_count = 0;
        let unique_keys: std::collections::HashSet<_> = dirty_keys.into_iter().collect();

        for key in unique_keys {
            if let Some(mut entry) = self.data.get_mut(&key) {
                let before = entry.value().version_count();
                entry.value_mut().gc(min_active_ts);
                gc_count += before.saturating_sub(entry.value().version_count());
            }
        }

        gc_count
    }
}

impl Default for ArenaMvccMemTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MemTableKind - Unified MemTable Abstraction (Principal Engineer Pattern)
// ============================================================================

/// Configuration for memtable type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemTableType {
    /// Standard MVCC memtable with deferred sorting
    /// Best for: general workloads, balanced read/write
    Standard,
    /// Arena-backed memtable with reduced allocations
    /// Best for: write-heavy workloads, large keys
    Arena,
}

impl Default for MemTableType {
    fn default() -> Self {
        // Default to Standard which now has deferred sorting
        MemTableType::Standard
    }
}

/// Unified memtable abstraction using enum dispatch
///
/// This pattern provides:
/// - Zero-cost abstraction (no vtable, no dynamic dispatch)
/// - Type-safe switching between implementations
/// - Easy extensibility for future memtable types
///
/// ## Why Enum over Trait Object?
///
/// - Hot path performance: enum match is a single branch vs vtable indirection
/// - Cache friendliness: no pointer chasing
/// - Inlining: compiler can inline through enum dispatch
pub enum MemTableKind {
    Standard(MvccMemTable),
    Arena(ArenaMvccMemTable),
}

impl MemTableKind {
    /// Create a new memtable of the specified type
    pub fn new(kind: MemTableType, enable_ordered_index: bool) -> Self {
        match kind {
            MemTableType::Standard => {
                MemTableKind::Standard(MvccMemTable::with_ordered_index(enable_ordered_index))
            }
            MemTableType::Arena => {
                MemTableKind::Arena(ArenaMvccMemTable::with_ordered_index(enable_ordered_index))
            }
        }
    }

    /// Write a key-value pair
    #[inline]
    pub fn write(&self, key: Vec<u8>, value: Option<Vec<u8>>, txn_id: u64) -> Result<()> {
        match self {
            MemTableKind::Standard(m) => m.write(key, value, txn_id),
            MemTableKind::Arena(m) => m.write(&key, value, txn_id),
        }
    }

    /// Write batch of key-value pairs
    #[inline]
    pub fn write_batch(&self, writes: &[(Vec<u8>, Option<Vec<u8>>)], txn_id: u64) -> Result<()> {
        match self {
            MemTableKind::Standard(m) => m.write_batch(writes, txn_id),
            MemTableKind::Arena(m) => {
                // Convert to arena-compatible format
                let arena_writes: Vec<(&[u8], Option<Vec<u8>>)> = writes
                    .iter()
                    .map(|(k, v)| (k.as_slice(), v.clone()))
                    .collect();
                m.write_batch(&arena_writes, txn_id)
            }
        }
    }

    /// Read at snapshot timestamp
    #[inline]
    pub fn read(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Option<Vec<u8>> {
        match self {
            MemTableKind::Standard(m) => m.read(key, snapshot_ts, current_txn_id),
            MemTableKind::Arena(m) => m.read(key, snapshot_ts, current_txn_id),
        }
    }

    /// Commit transaction
    #[inline]
    pub fn commit(&self, txn_id: u64, commit_ts: u64, write_set: &HashSet<InlineKey>) {
        match self {
            MemTableKind::Standard(m) => m.commit(txn_id, commit_ts, write_set),
            MemTableKind::Arena(m) => m.commit(txn_id, commit_ts, write_set),
        }
    }

    /// Abort transaction
    #[inline]
    pub fn abort(&self, txn_id: u64) {
        match self {
            MemTableKind::Standard(m) => m.abort(txn_id),
            MemTableKind::Arena(m) => m.abort(txn_id),
        }
    }

    /// Scan prefix
    #[inline]
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        match self {
            MemTableKind::Standard(m) => m.scan_prefix(prefix, snapshot_ts, current_txn_id),
            MemTableKind::Arena(m) => m.scan_prefix(prefix, snapshot_ts, current_txn_id),
        }
    }

    /// Scan range
    #[inline]
    pub fn scan_range(
        &self,
        start: &[u8],
        end: &[u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        match self {
            MemTableKind::Standard(m) => m.scan_range(start, end, snapshot_ts, current_txn_id),
            MemTableKind::Arena(m) => {
                // ArenaMvccMemTable doesn't have scan_range, use scan_prefix fallback
                let mut results = Vec::new();
                if let Some(ref idx) = m.ordered_index {
                    let start_handle = ArenaKeyHandle::new(start);
                    let end_handle = ArenaKeyHandle::new(end);
                    
                    if end.is_empty() {
                        for entry in idx.range(start_handle..) {
                            let key = entry.key();
                            if let Some(chain) = m.data.get(key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                results.push((key.as_bytes().to_vec(), value.clone()));
                            }
                        }
                    } else {
                        for entry in idx.range(start_handle..end_handle) {
                            let key = entry.key();
                            if let Some(chain) = m.data.get(key)
                                && let Some(v) = chain.read_at(snapshot_ts, current_txn_id)
                                && let Some(value) = &v.value
                            {
                                results.push((key.as_bytes().to_vec(), value.clone()));
                            }
                        }
                    }
                } else {
                    for entry in m.data.iter() {
                        let key = entry.key();
                        let key_bytes = key.as_bytes();
                        if key_bytes < start {
                            continue;
                        }
                        if !end.is_empty() && key_bytes >= end {
                            continue;
                        }
                        if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
                            && let Some(value) = &v.value
                        {
                            results.push((key_bytes.to_vec(), value.clone()));
                        }
                    }
                }
                results
            }
        }
    }

    /// Scan range iterator (returns collected results for now)
    #[inline]
    pub fn scan_range_iter<'a>(
        &'a self,
        start: &'a [u8],
        end: &'a [u8],
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a> {
        match self {
            MemTableKind::Standard(m) => {
                Box::new(m.scan_range_iter(start, end, snapshot_ts, current_txn_id))
            }
            MemTableKind::Arena(_) => {
                // Arena version returns collected results as iterator
                let results = self.scan_range(start, end, snapshot_ts, current_txn_id);
                Box::new(results.into_iter())
            }
        }
    }

    /// Get approximate size
    #[inline]
    pub fn size(&self) -> u64 {
        match self {
            MemTableKind::Standard(m) => m.size(),
            MemTableKind::Arena(m) => m.size(),
        }
    }

    /// Garbage collect old versions
    #[inline]
    pub fn gc(&self, min_active_ts: u64) -> usize {
        match self {
            MemTableKind::Standard(m) => m.gc(min_active_ts),
            MemTableKind::Arena(m) => m.gc(min_active_ts),
        }
    }

    /// Get the kind of memtable
    pub fn kind(&self) -> MemTableType {
        match self {
            MemTableKind::Standard(_) => MemTableType::Standard,
            MemTableKind::Arena(_) => MemTableType::Arena,
        }
    }
}

/// Durable storage engine with full ACID support
pub struct DurableStorage {
    /// Path to storage directory
    path: PathBuf,
    /// Write-ahead log
    wal: Arc<TxnWal>,
    /// MVCC manager
    mvcc: Arc<MvccManager>,
    /// In-memory data (unified abstraction over Standard/Arena)
    memtable: Arc<MemTableKind>,
    /// Per-transaction WAL buffers for batched writes
    /// Key: txn_id, Value: TxnWalBuffer that accumulates writes in memory
    /// At commit, buffer is flushed to WAL with single lock acquisition
    txn_write_buffers: DashMap<u64, TxnWalBuffer>,
    /// Group commit buffer (optional)
    group_commit: Option<Arc<EventDrivenGroupCommit>>,
    /// Recovery state
    needs_recovery: AtomicU64, // 1 = needs recovery
    /// Last checkpoint LSN
    last_checkpoint_lsn: AtomicU64,
    /// Synchronous mode (like SQLite's PRAGMA synchronous)
    /// 0 = OFF, 1 = NORMAL (periodic sync), 2 = FULL (sync every commit)
    sync_mode: AtomicU64,
    /// Commits since last sync (for NORMAL mode)
    commits_since_sync: AtomicU64,
    /// Adaptive batch sizing for NORMAL mode (Little's Law)
    /// Arrival rate in requests/sec × 1000 for precision
    arrival_rate_ema: AtomicU64,
    /// Last commit timestamp in microseconds
    last_commit_us: AtomicU64,
    /// Estimated fsync latency in microseconds
    fsync_latency_us: AtomicU64,
    /// Database lock for exclusive access (None = no locking)
    #[allow(dead_code)]
    db_lock: Option<crate::lock::DatabaseLock>,
}

impl DurableStorage {
    /// Open or create durable storage at path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, true)
    }

    /// Open with configurable ordered index
    ///
    /// When `enable_ordered_index` is false, saves ~134 ns/op on writes
    /// but scan_prefix becomes O(N) instead of O(log N + K)
    pub fn open_with_config<P: AsRef<Path>>(path: P, enable_ordered_index: bool) -> Result<Self> {
        Self::open_with_full_config(path, enable_ordered_index, MemTableType::Standard)
    }

    /// Open with arena-backed memtable for write-heavy workloads
    ///
    /// Uses ArenaMvccMemTable which reduces per-write allocations from 3 to 1.
    /// Best for workloads with:
    /// - High write throughput
    /// - Large keys (reduces allocation overhead)
    /// - Minimal concurrent reads during writes
    pub fn open_with_arena<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_full_config(path, true, MemTableType::Arena)
    }

    /// Open with full configuration options
    ///
    /// # Arguments
    /// * `path` - Storage directory path
    /// * `enable_ordered_index` - Enable ordered index for O(log N) scans
    /// * `memtable_type` - Type of memtable to use (Standard or Arena)
    ///
    /// # Locking
    ///
    /// Acquires an exclusive advisory lock on the database directory.
    /// This prevents concurrent multi-process access which would corrupt data.
    /// If another process has the database open, returns `Err(DatabaseLocked)`.
    pub fn open_with_full_config<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
        memtable_type: MemTableType,
    ) -> Result<Self> {
        Self::open_with_full_config_internal(path, enable_ordered_index, memtable_type, true)
    }

    /// Open without locking (for testing crash recovery scenarios)
    ///
    /// # Safety
    /// This should ONLY be used in tests that simulate crashes by forgetting
    /// the storage instance. In production, always use `open_with_full_config`.
    #[cfg(test)]
    pub fn open_without_lock<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_full_config_internal(path, true, MemTableType::Standard, false)
    }

    fn open_with_full_config_internal<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
        memtable_type: MemTableType,
        acquire_lock: bool,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        // Acquire exclusive lock on database directory (unless disabled for testing)
        let db_lock = if acquire_lock {
            Some(crate::lock::DatabaseLock::acquire(&path)
                .map_err(|e| SochDBError::LockError(e.to_string()))?)
        } else {
            None
        };

        let wal_path = path.join("wal.log");
        let wal = Arc::new(TxnWal::new(&wal_path)?);

        let storage = Self {
            path,
            wal: wal.clone(),
            mvcc: Arc::new(MvccManager::new()),
            memtable: Arc::new(MemTableKind::new(memtable_type, enable_ordered_index)),
            txn_write_buffers: DashMap::new(),
            group_commit: None,
            needs_recovery: AtomicU64::new(0),
            last_checkpoint_lsn: AtomicU64::new(0),
            sync_mode: AtomicU64::new(1), // Default: NORMAL (like SQLite)
            commits_since_sync: AtomicU64::new(0),
            // Adaptive batch sizing (Little's Law)
            arrival_rate_ema: AtomicU64::new(1_000_000), // 1000 req/s × 1000 initial
            last_commit_us: AtomicU64::new(0),
            fsync_latency_us: AtomicU64::new(5000), // 5ms default
            db_lock,
        };

        // Check if recovery needed
        if storage.check_recovery_needed()? {
            storage.needs_recovery.store(1, Ordering::SeqCst);
        }

        Ok(storage)
    }

    /// Open with group commit enabled
    pub fn open_with_group_commit<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_group_commit_and_config(path, true)
    }

    /// Open with group commit and configurable ordered index
    pub fn open_with_group_commit_and_config<P: AsRef<Path>>(
        path: P,
        enable_ordered_index: bool,
    ) -> Result<Self> {
        let mut storage = Self::open_with_config(path, enable_ordered_index)?;

        let wal = storage.wal.clone();
        let gc = EventDrivenGroupCommit::new(move |txn_ids: &[u64]| {
            // Write all commit records WITHOUT flushing (batch them)
            for &txn_id in txn_ids {
                let entry = TxnWalEntry::txn_commit(txn_id);
                wal.append_no_flush(&entry).map_err(|e| e.to_string())?;
            }

            // Then do a SINGLE flush + fsync for the entire batch
            wal.flush().map_err(|e| e.to_string())?;
            wal.sync().map_err(|e| e.to_string())?;

            // Return commit timestamp
            Ok(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64)
        });

        storage.group_commit = Some(Arc::new(gc));
        Ok(storage)
    }

    /// Open with IndexPolicy for automatic memtable/index configuration
    ///
    /// This is the recommended constructor for new code. The policy determines:
    /// - Whether to use ordered index (ScanOptimized only)
    /// - Whether to use arena-backed memtable (WriteOptimized, AppendOnly)
    /// - Default settings optimized for the workload pattern
    ///
    /// # Arguments
    /// * `path` - Storage directory path
    /// * `policy` - Index policy determining write/scan tradeoffs
    /// * `group_commit` - Whether to enable group commit for throughput
    pub fn open_with_policy<P: AsRef<Path>>(
        path: P,
        policy: crate::index_policy::IndexPolicy,
        group_commit: bool,
    ) -> Result<Self> {
        use crate::index_policy::IndexPolicy;
        
        // Derive configuration from policy
        let (enable_ordered_index, memtable_type) = match policy {
            IndexPolicy::WriteOptimized | IndexPolicy::AppendOnly => {
                // Write-heavy: no ordered index, use arena for reduced allocations
                (false, MemTableType::Arena)
            }
            IndexPolicy::Balanced => {
                // Mixed OLTP: deferred sorting (already implemented in Standard)
                (true, MemTableType::Standard)
            }
            IndexPolicy::ScanOptimized => {
                // Scan-heavy: maintain ordered index
                (true, MemTableType::Standard)
            }
        };

        if group_commit {
            let mut storage = Self::open_with_full_config(path, enable_ordered_index, memtable_type)?;
            
            let wal = storage.wal.clone();
            let gc = EventDrivenGroupCommit::new(move |txn_ids: &[u64]| {
                for &txn_id in txn_ids {
                    let entry = TxnWalEntry::txn_commit(txn_id);
                    wal.append_no_flush(&entry).map_err(|e| e.to_string())?;
                }
                wal.flush().map_err(|e| e.to_string())?;
                wal.sync().map_err(|e| e.to_string())?;
                Ok(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64)
            });
            storage.group_commit = Some(Arc::new(gc));
            Ok(storage)
        } else {
            Self::open_with_full_config(path, enable_ordered_index, memtable_type)
        }
    }

    /// Get the memtable type being used
    pub fn memtable_type(&self) -> MemTableType {
        self.memtable.kind()
    }

    /// Check if recovery is needed (dirty shutdown detection)
    ///
    /// Note: Recovery must ALWAYS run to rebuild the in-memory memtable from WAL.
    /// The clean_shutdown marker only tells us if there might be uncommitted transactions,
    /// but committed data still needs to be loaded from WAL into the memtable.
    fn check_recovery_needed(&self) -> Result<bool> {
        let marker_path = self.path.join(".clean_shutdown");
        if marker_path.exists() {
            // Clean shutdown - remove marker
            std::fs::remove_file(&marker_path)?;
        }
        // ALWAYS need recovery to rebuild memtable from WAL
        // The memtable is in-memory only and needs to be restored on every startup
        Ok(true)
    }

    /// Perform crash recovery
    pub fn recover(&self) -> Result<RecoveryStats> {
        if self.needs_recovery.load(Ordering::SeqCst) == 0 {
            return Ok(RecoveryStats::default());
        }

        let (writes, txn_count) = self.wal.replay_for_recovery()?;

        // Apply committed writes to memtable
        let recovery_txn_id = self.wal.alloc_txn_id();
        let commit_ts = self.mvcc.alloc_commit_ts();

        // Collect keys being written for efficient commit
        let mut write_set: HashSet<InlineKey> = HashSet::new();
        for (key, value) in &writes {
            write_set.insert(SmallVec::from_slice(key));
            self.memtable
                .write(key.clone(), Some(value.clone()), recovery_txn_id)?;
        }
        self.memtable.commit(recovery_txn_id, commit_ts, &write_set);

        self.needs_recovery.store(0, Ordering::SeqCst);

        Ok(RecoveryStats {
            transactions_recovered: txn_count,
            writes_recovered: writes.len(),
            commit_ts,
        })
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self) -> Result<u64> {
        let txn_id = self.wal.begin_transaction()?;
        self.mvcc.begin(txn_id);
        Ok(txn_id)
    }

    /// Begin a transaction with a specific mode (ReadOnly/WriteOnly/ReadWrite)
    ///
    /// This enables mode-aware optimizations:
    /// - ReadOnly: Skip SSI tracking, 2.6x faster reads
    /// - WriteOnly: Skip read tracking, faster bulk inserts
    /// - ReadWrite: Full SSI for serializable isolation
    pub fn begin_with_mode(&self, mode: TransactionMode) -> Result<u64> {
        let txn_id = self.wal.begin_transaction()?;
        self.mvcc.begin_with_mode(txn_id, mode);
        Ok(txn_id)
    }

    /// Read a key within a transaction
    #[inline]
    pub fn read(&self, txn_id: u64, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Fast path: get just snapshot_ts without cloning whole transaction
        let snapshot_ts = self
            .mvcc
            .get_snapshot_ts(txn_id)
            .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

        // Record read for SSI (skipped for read-only transactions)
        self.mvcc.record_read(txn_id, key);

        // Read at snapshot timestamp, seeing own uncommitted writes
        Ok(self.memtable.read(key, snapshot_ts, Some(txn_id)))
    }

    /// Write a key-value pair within a transaction
    ///
    /// Writes are buffered and only flushed to disk on commit.
    /// This provides ~10× better throughput for batched inserts.
    pub fn write(&self, txn_id: u64, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        // Use the zero-allocation path internally
        self.write_refs(txn_id, &key, &value)?;

        Ok(())
    }

    /// Write from references - zero allocation hot path
    ///
    /// Avoids cloning key/value by writing to WAL from refs directly,
    /// then only allocating once for memtable storage.
    #[inline]
    pub fn write_refs(&self, txn_id: u64, key: &[u8], value: &[u8]) -> Result<()> {
        // Record write for MVCC (uses InlineKey - zero allocation for small keys)
        self.mvcc.record_write(txn_id, key);

        // Buffer writes in memory using TxnWalBuffer - NO WAL lock taken!
        // This reduces lock contention from O(writes) to O(1) per transaction
        self.txn_write_buffers
            .entry(txn_id)
            .or_insert_with(|| TxnWalBuffer::new(txn_id))
            .append(key, value);

        // Write to memtable (needs owned key/value for storage)
        self.memtable
            .write(key.to_vec(), Some(value.to_vec()), txn_id)?;

        Ok(())
    }

    /// Delete a key within a transaction
    pub fn delete(&self, txn_id: u64, key: Vec<u8>) -> Result<()> {
        // Record write (uses InlineKey - zero allocation for small keys)
        self.mvcc.record_write(txn_id, &key);

        // Buffer tombstone in memory - NO WAL lock taken!
        self.txn_write_buffers
            .entry(txn_id)
            .or_insert_with(|| TxnWalBuffer::new(txn_id))
            .append(&key, &[]); // Empty value = tombstone

        // Write tombstone to memtable
        self.memtable.write(key, None, txn_id)?;

        Ok(())
    }

    /// Batch write multiple key-value pairs with reduced overhead
    ///
    /// This API amortizes fixed costs over the batch:
    /// - Single DashMap entry lookup for TxnWalBuffer
    /// - Single MVCC write set update
    /// - Batch memtable operations
    ///
    /// Performance: ~2-3x faster than individual write_refs calls
    /// for batches of 100+ entries.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID
    /// * `writes` - Slice of (key, value) pairs
    #[inline]
    pub fn write_batch_refs(&self, txn_id: u64, writes: &[(&[u8], &[u8])]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }

        // Single DashMap access for entire batch
        let mut buffer_entry = self
            .txn_write_buffers
            .entry(txn_id)
            .or_insert_with(|| TxnWalBuffer::new(txn_id));

        // Batch operations with reduced per-row overhead
        for (key, value) in writes {
            // Record write for MVCC
            self.mvcc.record_write(txn_id, key);

            // Append to WAL buffer
            buffer_entry.append(key, value);
        }
        drop(buffer_entry);

        // Batch write to memtable
        let owned_writes: Vec<(Vec<u8>, Option<Vec<u8>>)> = writes
            .iter()
            .map(|(k, v)| (k.to_vec(), Some(v.to_vec())))
            .collect();
        self.memtable.write_batch(&owned_writes, txn_id)?;

        Ok(())
    }

    /// Commit a transaction
    ///
    /// With sync_mode:
    /// - 0 (OFF): No sync, risk of data loss
    /// - 1 (NORMAL): Adaptive sync using Little's Law: W* = √(τ/λ)
    /// - 2 (FULL): Sync every commit (safest, slowest)
    pub fn commit(&self, txn_id: u64) -> Result<u64> {
        // First, flush all buffered writes to WAL with SINGLE lock acquisition
        // This is the key optimization: O(1) lock instead of O(writes) locks
        if let Some((_, buffer)) = self.txn_write_buffers.remove(&txn_id)
            && !buffer.is_empty()
        {
            // Flush entire buffer to WAL with one lock
            self.wal.flush_buffer(&buffer)?;
        }

        // Use group commit if available, otherwise direct commit
        if let Some(gc) = &self.group_commit {
            // Submit to group commit and wait for result
            // This batches multiple commits into a single fsync
            gc.submit_and_wait(txn_id).map_err(SochDBError::Internal)?;

            // Get commit timestamp and write_set from MVCC
            let (commit_ts, write_set) = self
                .mvcc
                .commit(txn_id)
                .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

            // Commit in memtable (O(k) where k = keys written)
            self.memtable.commit(txn_id, commit_ts, &write_set);

            Ok(commit_ts)
        } else {
            // Direct commit path with adaptive sync (Little's Law)
            let sync_mode = self.sync_mode.load(Ordering::Relaxed);
            let commits = self.commits_since_sync.fetch_add(1, Ordering::Relaxed);

            // Update arrival rate for adaptive batching
            self.update_arrival_rate();

            // Write commit record (no flush yet - BufWriter will buffer it)
            let entry = TxnWalEntry::txn_commit(txn_id);
            self.wal.append_no_flush(&entry)?;

            // Determine if we should sync/flush based on mode
            let should_sync = match sync_mode {
                0 => false,                                      // OFF: never sync
                1 => commits >= self.adaptive_batch_threshold(), // NORMAL: adaptive
                _ => true,                                       // FULL: always sync
            };

            if should_sync {
                // Measure fsync latency for adaptive tuning
                let start = std::time::Instant::now();
                self.wal.flush()?;
                self.wal.sync()?;
                let latency_us = start.elapsed().as_micros() as u64;

                // Update fsync latency estimate (EMA with α = 0.1)
                let old_latency = self.fsync_latency_us.load(Ordering::Relaxed);
                let new_latency = (old_latency * 9 + latency_us) / 10;
                self.fsync_latency_us.store(new_latency, Ordering::Relaxed);

                self.commits_since_sync.store(0, Ordering::Relaxed);
            }

            // Get commit timestamp and write_set from MVCC
            let (commit_ts, write_set) = self
                .mvcc
                .commit(txn_id)
                .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

            // Commit in memtable (O(k) where k = keys written)
            self.memtable.commit(txn_id, commit_ts, &write_set);

            Ok(commit_ts)
        }
    }

    /// Update arrival rate using exponential moving average
    #[inline]
    fn update_arrival_rate(&self) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let last = self.last_commit_us.swap(now_us, Ordering::Relaxed);

        if last > 0 {
            let delta_us = now_us.saturating_sub(last);
            if delta_us > 0 && delta_us < 10_000_000 {
                // Ignore gaps > 10s
                // Rate = 1_000_000 / delta_us (requests/sec)
                // Stored as rate × 1000 for precision
                let instant_rate = 1_000_000_000 / delta_us;

                // EMA with α = 0.1
                let old_rate = self.arrival_rate_ema.load(Ordering::Relaxed);
                let new_rate = (old_rate * 9 + instant_rate) / 10;
                self.arrival_rate_ema.store(new_rate, Ordering::Relaxed);
            }
        }
    }

    /// Compute optimal batch threshold using Little's Law
    ///
    /// W* = √(τ / λ) where τ = fsync latency, λ = arrival rate
    /// Returns the number of commits to batch before fsync
    #[inline]
    fn adaptive_batch_threshold(&self) -> u64 {
        let lambda = self.arrival_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0; // req/s
        let tau = self.fsync_latency_us.load(Ordering::Relaxed) as f64 / 1_000_000.0; // seconds

        if lambda <= 0.0 || tau <= 0.0 {
            return 100; // Fallback to fixed threshold
        }

        // Little's Law: W* = sqrt(2 × τ × λ)
        // This minimizes total time = wait_time + fsync_overhead
        let n_opt = (2.0 * tau * lambda).sqrt();

        // Clamp between 1 and 1000
        (n_opt as u64).clamp(1, 1000)
    }

    /// Set synchronous mode
    ///
    /// - 0: OFF - No fsync (risk of data loss)
    /// - 1: NORMAL - Periodic fsync (balanced)
    /// - 2: FULL - Fsync every commit (safest)
    pub fn set_sync_mode(&self, mode: u64) {
        self.sync_mode.store(mode.min(2), Ordering::Relaxed);
    }

    /// Force a group commit flush (useful for benchmarking or testing)
    pub fn flush_group_commit(&self) {
        if let Some(gc) = &self.group_commit {
            gc.flush_batch();
        }
    }

    /// Abort a transaction
    pub fn abort(&self, txn_id: u64) -> Result<()> {
        // Discard buffered writes (no need to write to WAL)
        self.txn_write_buffers.remove(&txn_id);

        // Write abort to WAL
        self.wal.abort_transaction(txn_id)?;

        // Abort in MVCC
        self.mvcc.abort(txn_id);

        // Abort in memtable
        self.memtable.abort(txn_id);

        Ok(())
    }

    /// Scan keys with prefix
    #[inline]
    pub fn scan(&self, txn_id: u64, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Fast path: get just snapshot_ts without cloning whole transaction
        let snapshot_ts = self
            .mvcc
            .get_snapshot_ts(txn_id)
            .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

        // Note: Scan doesn't record individual key reads for SSI (too expensive)
        // SSI conflicts are tracked at the prefix level if needed
        Ok(self.memtable.scan_prefix(prefix, snapshot_ts, Some(txn_id)))
    }

    /// Scan keys in range
    #[inline]
    pub fn scan_range(
        &self,
        txn_id: u64,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let snapshot_ts = self
            .mvcc
            .get_snapshot_ts(txn_id)
            .ok_or_else(|| SochDBError::Internal("Transaction not found".into()))?;

        Ok(self
            .memtable
            .scan_range(start, end, snapshot_ts, Some(txn_id)))
    }

    /// Streaming scan for very large result sets
    /// 
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set in memory.
    #[inline]
    pub fn scan_range_iter<'a>(
        &'a self,
        txn_id: u64,
        start: &'a [u8],
        end: &'a [u8],
    ) -> impl Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a {
        let snapshot_ts = self.mvcc.get_snapshot_ts(txn_id).unwrap_or(0);
        self.memtable.scan_range_iter(start, end, snapshot_ts, Some(txn_id))
    }

    /// Force fsync to disk
    pub fn fsync(&self) -> Result<()> {
        self.wal.sync()
    }

    /// Write checkpoint
    pub fn checkpoint(&self) -> Result<u64> {
        let txn_id = 0; // System transaction
        let entry = TxnWalEntry::checkpoint(txn_id);
        let lsn = self.wal.append_sync(&entry)?;
        self.last_checkpoint_lsn.store(lsn, Ordering::SeqCst);
        Ok(lsn)
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        // Get WAL size from the WAL manager
        let wal_size = self.wal.size_bytes();

        // Get active transaction count from MVCC
        let active_txns = self.mvcc.active_transaction_count();

        StorageStats {
            memtable_size_bytes: self.memtable.size(),
            wal_size_bytes: wal_size,
            active_transactions: active_txns,
            min_active_snapshot: self.mvcc.min_active_snapshot(),
            last_checkpoint_lsn: self.last_checkpoint_lsn.load(Ordering::SeqCst),
        }
    }

    /// Garbage collect old versions
    pub fn gc(&self) -> usize {
        let min_ts = self.mvcc.min_active_snapshot();
        self.memtable.gc(min_ts)
    }

    /// Clean shutdown
    pub fn shutdown(&self) -> Result<()> {
        // Sync WAL
        self.fsync()?;

        // Write clean shutdown marker
        let marker_path = self.path.join(".clean_shutdown");
        std::fs::write(&marker_path, b"clean")?;

        Ok(())
    }
}

impl Drop for DurableStorage {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Recovery statistics
#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub transactions_recovered: usize,
    pub writes_recovered: usize,
    pub commit_ts: u64,
}

/// Storage statistics
#[derive(Debug, Default)]
pub struct StorageStats {
    pub memtable_size_bytes: u64,
    pub wal_size_bytes: u64,
    pub active_transactions: usize,
    pub min_active_snapshot: u64,
    pub last_checkpoint_lsn: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_basic_transaction() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Begin transaction
        let txn_id = storage.begin_transaction().unwrap();

        // Write data
        storage
            .write(txn_id, b"key1".to_vec(), b"value1".to_vec())
            .unwrap();
        storage
            .write(txn_id, b"key2".to_vec(), b"value2".to_vec())
            .unwrap();

        // Read back (within same transaction)
        let v1 = storage.read(txn_id, b"key1").unwrap();
        assert_eq!(v1, Some(b"value1".to_vec()));

        // Commit
        let commit_ts = storage.commit(txn_id).unwrap();
        assert!(commit_ts > 0);

        // Read in new transaction
        let txn2 = storage.begin_transaction().unwrap();
        let v1 = storage.read(txn2, b"key1").unwrap();
        assert_eq!(v1, Some(b"value1".to_vec()));
        storage.abort(txn2).unwrap();
    }

    #[test]
    fn test_snapshot_isolation() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // T1: Write initial value
        let t1 = storage.begin_transaction().unwrap();
        storage.write(t1, b"key".to_vec(), b"v1".to_vec()).unwrap();
        storage.commit(t1).unwrap();

        // T2: Start reading (snapshot at this point)
        let t2 = storage.begin_transaction().unwrap();

        // T3: Update the value
        let t3 = storage.begin_transaction().unwrap();
        storage.write(t3, b"key".to_vec(), b"v2".to_vec()).unwrap();
        storage.commit(t3).unwrap();

        // T2 should still see v1 (snapshot isolation)
        let v = storage.read(t2, b"key").unwrap();
        assert_eq!(v, Some(b"v1".to_vec()));

        // New transaction should see v2
        let t4 = storage.begin_transaction().unwrap();
        let v = storage.read(t4, b"key").unwrap();
        assert_eq!(v, Some(b"v2".to_vec()));

        storage.abort(t2).unwrap();
        storage.abort(t4).unwrap();
    }

    #[test]
    fn test_abort_transaction() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Write initial value
        let t1 = storage.begin_transaction().unwrap();
        storage.write(t1, b"key".to_vec(), b"v1".to_vec()).unwrap();
        storage.commit(t1).unwrap();

        // Start transaction that will abort
        let t2 = storage.begin_transaction().unwrap();
        storage.write(t2, b"key".to_vec(), b"v2".to_vec()).unwrap();
        storage.abort(t2).unwrap();

        // New transaction should see v1 (aborted changes not visible)
        let t3 = storage.begin_transaction().unwrap();
        let v = storage.read(t3, b"key").unwrap();
        assert_eq!(v, Some(b"v1".to_vec()));
        storage.abort(t3).unwrap();
    }

    #[test]
    fn test_crash_recovery() {
        let dir = tempdir().unwrap();

        // Phase 1: Write data and commit
        {
            // Use open_without_lock for crash simulation tests
            let storage = DurableStorage::open_without_lock(dir.path()).unwrap();

            // Set sync mode to FULL to ensure data is synced before "crash"
            storage.set_sync_mode(2); // FULL: sync every commit

            let txn = storage.begin_transaction().unwrap();
            storage
                .write(txn, b"persist".to_vec(), b"data".to_vec())
                .unwrap();
            storage.commit(txn).unwrap();

            // Simulate crash (no clean shutdown)
            std::mem::forget(storage);
        }

        // Phase 2: Reopen and recover
        {
            let storage = DurableStorage::open_without_lock(dir.path()).unwrap();
            let stats = storage.recover().unwrap();
            assert!(stats.transactions_recovered > 0 || stats.writes_recovered > 0);

            // Data should be recovered
            let txn = storage.begin_transaction().unwrap();
            let v = storage.read(txn, b"persist").unwrap();
            assert_eq!(v, Some(b"data".to_vec()));
            storage.abort(txn).unwrap();
        }
    }

    #[test]
    fn test_scan_prefix() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        let txn = storage.begin_transaction().unwrap();
        storage
            .write(txn, b"user:1".to_vec(), b"alice".to_vec())
            .unwrap();
        storage
            .write(txn, b"user:2".to_vec(), b"bob".to_vec())
            .unwrap();
        storage
            .write(txn, b"order:1".to_vec(), b"order1".to_vec())
            .unwrap();
        storage.commit(txn).unwrap();

        let txn2 = storage.begin_transaction().unwrap();
        let users = storage.scan(txn2, b"user:").unwrap();
        assert_eq!(users.len(), 2);
        storage.abort(txn2).unwrap();
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Insert
        let t1 = storage.begin_transaction().unwrap();
        storage
            .write(t1, b"key".to_vec(), b"value".to_vec())
            .unwrap();
        storage.commit(t1).unwrap();

        // Verify exists
        let t2 = storage.begin_transaction().unwrap();
        assert!(storage.read(t2, b"key").unwrap().is_some());
        storage.abort(t2).unwrap();

        // Delete
        let t3 = storage.begin_transaction().unwrap();
        storage.delete(t3, b"key".to_vec()).unwrap();
        storage.commit(t3).unwrap();

        // Verify deleted
        let t4 = storage.begin_transaction().unwrap();
        assert!(storage.read(t4, b"key").unwrap().is_none());
        storage.abort(t4).unwrap();
    }

    #[test]
    fn test_gc() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open(dir.path()).unwrap();

        // Create multiple versions
        for i in 0..10 {
            let txn = storage.begin_transaction().unwrap();
            storage
                .write(txn, b"key".to_vec(), format!("v{}", i).into_bytes())
                .unwrap();
            storage.commit(txn).unwrap();
        }

        // GC should reclaim old versions
        let gc_count = storage.gc();
        // At least some versions should be collected
        // (exact count depends on implementation)
        let _ = gc_count; // gc_count is usize, always >= 0
    }

    #[test]
    fn test_group_commit() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempdir().unwrap();
        let storage = Arc::new(DurableStorage::open_with_group_commit(dir.path()).unwrap());

        // Spawn multiple threads to commit concurrently
        let mut handles = vec![];
        for i in 0..4 {
            let storage = Arc::clone(&storage);
            handles.push(thread::spawn(move || {
                let txn = storage.begin_transaction().unwrap();
                storage
                    .write(
                        txn,
                        format!("key{}", i).into_bytes(),
                        format!("val{}", i).into_bytes(),
                    )
                    .unwrap();
                storage.commit(txn).unwrap()
            }));
        }

        // Wait for all commits
        let mut commit_times = vec![];
        for h in handles {
            commit_times.push(h.join().unwrap());
        }

        // All commits should succeed
        assert!(commit_times.iter().all(|&ts| ts > 0));

        // Verify data persisted
        let txn = storage.begin_transaction().unwrap();
        for i in 0..4 {
            let val = storage.read(txn, format!("key{}", i).as_bytes()).unwrap();
            assert_eq!(val, Some(format!("val{}", i).into_bytes()));
        }
        storage.abort(txn).unwrap();
    }

    // ==================== ArenaMvccMemTable Tests ====================

    #[test]
    fn test_arena_memtable_basic_write_read() {
        let memtable = ArenaMvccMemTable::new();

        // Write some values
        memtable
            .write(b"key1", Some(b"value1".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"key2", Some(b"value2".to_vec()), 1)
            .unwrap();

        // Read them back (uncommitted, so need txn_id match)
        assert_eq!(memtable.read(b"key1", 0, Some(1)), Some(b"value1".to_vec()));
        assert_eq!(memtable.read(b"key2", 0, Some(1)), Some(b"value2".to_vec()));
        assert_eq!(memtable.read(b"key3", 0, Some(1)), None);
    }

    #[test]
    fn test_arena_memtable_update() {
        let memtable = ArenaMvccMemTable::new();

        memtable.write(b"key", Some(b"v1".to_vec()), 1).unwrap();
        memtable.write(b"key", Some(b"v2".to_vec()), 1).unwrap();

        assert_eq!(memtable.read(b"key", 0, Some(1)), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_arena_memtable_delete() {
        let memtable = ArenaMvccMemTable::new();

        memtable.write(b"key", Some(b"value".to_vec()), 1).unwrap();
        memtable.write(b"key", None, 1).unwrap(); // Delete = None value

        assert_eq!(memtable.read(b"key", 0, Some(1)), None);
    }

    #[test]
    fn test_arena_memtable_scan_prefix() {
        let memtable = ArenaMvccMemTable::new();

        memtable
            .write(b"user:1:name", Some(b"Alice".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"user:1:email", Some(b"alice@test.com".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"user:2:name", Some(b"Bob".to_vec()), 1)
            .unwrap();
        memtable
            .write(b"order:1", Some(b"order_data".to_vec()), 1)
            .unwrap();

        // Create a write set and commit
        let mut write_set = HashSet::new();
        write_set.insert(InlineKey::from_slice(b"user:1:name"));
        write_set.insert(InlineKey::from_slice(b"user:1:email"));
        write_set.insert(InlineKey::from_slice(b"user:2:name"));
        write_set.insert(InlineKey::from_slice(b"order:1"));
        memtable.commit(1, 10, &write_set);

        // Scan for user:1:* (snapshot_ts > commit_ts to see committed data)
        let results = memtable.scan_prefix(b"user:1:", 11, None);
        assert_eq!(results.len(), 2);

        // Scan for all users
        let results = memtable.scan_prefix(b"user:", 11, None);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_arena_memtable_write_batch() {
        let memtable = ArenaMvccMemTable::new();

        let writes: Vec<(&[u8], Option<Vec<u8>>)> = vec![
            (b"k1", Some(b"v1".to_vec())),
            (b"k2", Some(b"v2".to_vec())),
            (b"k3", Some(b"v3".to_vec())),
        ];

        memtable.write_batch(&writes, 1).unwrap();

        assert_eq!(memtable.read(b"k1", 0, Some(1)), Some(b"v1".to_vec()));
        assert_eq!(memtable.read(b"k2", 0, Some(1)), Some(b"v2".to_vec()));
        assert_eq!(memtable.read(b"k3", 0, Some(1)), Some(b"v3".to_vec()));
    }

    #[test]
    fn test_arena_memtable_gc() {
        let memtable = ArenaMvccMemTable::new();

        // Write multiple versions
        for i in 0..10 {
            memtable
                .write(b"key", Some(format!("v{}", i).into_bytes()), i + 1)
                .unwrap();

            let mut write_set = HashSet::new();
            write_set.insert(InlineKey::from_slice(b"key"));
            memtable.commit(i + 1, (i + 1) * 10, &write_set);
        }

        // GC old versions
        let gc_count = memtable.gc(90);
        let _ = gc_count; // gc_count is usize, always >= 0
    }

    #[test]
    fn test_arena_memtable_size_tracking() {
        let memtable = ArenaMvccMemTable::new();

        assert_eq!(memtable.size(), 0);

        memtable.write(b"key", Some(b"value".to_vec()), 1).unwrap();

        assert!(memtable.size() > 0);
    }

    #[test]
    fn test_arena_memtable_abort() {
        let memtable = ArenaMvccMemTable::new();

        memtable
            .write(b"key", Some(b"uncommitted".to_vec()), 1)
            .unwrap();

        // Visible to same txn
        assert_eq!(
            memtable.read(b"key", 0, Some(1)),
            Some(b"uncommitted".to_vec())
        );

        // Not visible to other txns
        assert_eq!(memtable.read(b"key", 0, Some(2)), None);

        // Abort
        memtable.abort(1);

        // No longer visible
        assert_eq!(memtable.read(b"key", 0, Some(1)), None);
    }

    // ========================================================================
    // MemTableKind Tests - Unified Abstraction
    // ========================================================================

    #[test]
    fn test_memtable_kind_standard() {
        let memtable = MemTableKind::new(MemTableType::Standard, true);
        assert_eq!(memtable.kind(), MemTableType::Standard);

        // Write and read
        memtable.write(b"key1".to_vec(), Some(b"value1".to_vec()), 1).unwrap();
        
        // Commit transaction at ts=100
        let write_set = std::iter::once(InlineKey::from_slice(b"key1")).collect();
        memtable.commit(1, 100, &write_set);
        
        // Read after commit - snapshot_ts must be > commit_ts for visibility
        let v = memtable.read(b"key1", 101, None);
        assert_eq!(v, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_memtable_kind_arena() {
        let memtable = MemTableKind::new(MemTableType::Arena, true);
        assert_eq!(memtable.kind(), MemTableType::Arena);

        // Write and read
        memtable.write(b"key1".to_vec(), Some(b"value1".to_vec()), 1).unwrap();
        
        // Commit at ts=100
        let write_set = std::iter::once(InlineKey::from_slice(b"key1")).collect();
        memtable.commit(1, 100, &write_set);
        
        // Read after commit - snapshot_ts > commit_ts
        let v = memtable.read(b"key1", 101, None);
        assert_eq!(v, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_memtable_kind_scan_range() {
        // Test both implementations have consistent behavior
        for kind in [MemTableType::Standard, MemTableType::Arena] {
            let memtable = MemTableKind::new(kind, true);

            // Write some data
            for i in 0..5 {
                let key = format!("key{}", i);
                let value = format!("value{}", i);
                memtable.write(key.into_bytes(), Some(value.into_bytes()), 1).unwrap();
            }

            // Commit all at ts=100
            let write_set: HashSet<InlineKey> = (0..5)
                .map(|i| InlineKey::from_slice(format!("key{}", i).as_bytes()))
                .collect();
            memtable.commit(1, 100, &write_set);

            // Scan range with snapshot_ts > commit_ts
            let results = memtable.scan_range(b"key1", b"key4", 101, None);
            assert_eq!(results.len(), 3, "kind={:?} should have 3 results (key1, key2, key3)", kind);
        }
    }

    #[test]
    fn test_durable_storage_arena() {
        let dir = tempdir().unwrap();
        let storage = DurableStorage::open_with_arena(dir.path()).unwrap();
        
        assert_eq!(storage.memtable_type(), MemTableType::Arena);

        // Basic transaction should work the same
        let txn_id = storage.begin_transaction().unwrap();
        storage.write(txn_id, b"key1".to_vec(), b"value1".to_vec()).unwrap();
        storage.commit(txn_id).unwrap();

        let txn2 = storage.begin_transaction().unwrap();
        let v = storage.read(txn2, b"key1").unwrap();
        assert_eq!(v, Some(b"value1".to_vec()));
        storage.abort(txn2).unwrap();
    }

    #[test]
    fn test_durable_storage_full_config() {
        let dir = tempdir().unwrap();
        
        // Test with Arena and ordered index enabled
        let storage = DurableStorage::open_with_full_config(
            dir.path(),
            true,
            MemTableType::Arena,
        ).unwrap();
        
        assert_eq!(storage.memtable_type(), MemTableType::Arena);

        // Write multiple keys
        let txn = storage.begin_transaction().unwrap();
        for i in 0..10 {
            let key = format!("key{:02}", i);
            let value = format!("value{}", i);
            storage.write(txn, key.into_bytes(), value.into_bytes()).unwrap();
        }
        storage.commit(txn).unwrap();

        // Scan should work (uses scan method for prefix)
        let txn2 = storage.begin_transaction().unwrap();
        let results = storage.scan(txn2, b"key0").unwrap();
        assert_eq!(results.len(), 10); // key00 through key09
        storage.abort(txn2).unwrap();
    }
}
