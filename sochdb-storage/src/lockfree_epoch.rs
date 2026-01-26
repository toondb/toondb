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

//! Lock-Free Epoch Tracking via Slot Array (Task 3)
//!
//! Replaces mutex-synchronized vectors with a lock-free slot array using
//! atomic operations, achieving O(1) constant-time recording with zero contention.
//!
//! ## Problem
//!
//! The current `EpochDirtyList` uses mutex-protected vectors:
//! ```text
//! epochs: [parking_lot::Mutex<Vec<Vec<u8>>>; 4]
//! ```
//!
//! This design has three fundamental problems:
//! 1. Lock contention under high concurrency
//! 2. Allocation amplification from vector growth
//! 3. False sharing from adjacent mutex cache lines
//!
//! ## Solution
//!
//! Lock-free circular slot array with atomic operations:
//! - O(1) slot reservation via fetch_add (never fails, never blocks)
//! - Cache-line aligned slots to prevent false sharing
//! - Pre-allocated memory eliminates allocation overhead
//!
//! ## Memory Ordering
//!
//! | Operation | Ordering | Rationale |
//! |-----------|----------|-----------|
//! | `write_cursor.fetch_add` | Relaxed | Slot reservation doesn't require visibility ordering |
//! | `slot.epoch.store` | Release | Pairs with Acquire in drain to see slot contents |
//! | `current_epoch.load` | Acquire | Ensures we see the current epoch value |
//! | `advance_epoch` fence | SeqCst | Establishes total order for epoch transitions |

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Number of slots in the circular buffer
/// With 64 bytes per slot, 64K slots = 4MB memory
const NUM_SLOTS: usize = 65536;

/// Maximum epochs to track
const MAX_EPOCHS: usize = 8;

/// Cache-line aligned slot for lock-free recording
#[repr(align(64))]
#[derive(Debug)]
struct CacheAlignedSlot {
    /// Hash of the key (64-bit FxHash)
    key_hash: AtomicU64,
    /// Epoch when this slot was written
    epoch: AtomicU32,
    /// Padding to fill cache line
    _padding: [u8; 52],
}

impl Default for CacheAlignedSlot {
    fn default() -> Self {
        Self {
            key_hash: AtomicU64::new(0),
            epoch: AtomicU32::new(u32::MAX), // Invalid epoch marker
            _padding: [0; 52],
        }
    }
}

/// Lock-free epoch tracker using circular slot array
/// 
/// ## Performance Characteristics
/// 
/// - Record version: O(1) constant time, zero contention
/// - Advance epoch: O(1) plus O(n) drain of old epoch slots
/// - Memory: O(NUM_SLOTS) fixed = ~4MB
/// 
/// ## Thread Safety
/// 
/// All operations are lock-free and thread-safe. Multiple threads can
/// record versions concurrently without any synchronization.
pub struct LockFreeEpochTracker {
    /// Pre-allocated slot array (cache-line aligned)
    slots: Box<[CacheAlignedSlot; NUM_SLOTS]>,
    
    /// Monotonic write cursor (wraps around)
    write_cursor: AtomicU64,
    
    /// Epoch boundary markers (cursor position when epoch advanced)
    epoch_boundaries: [AtomicU64; MAX_EPOCHS],
    
    /// Current epoch index
    current_epoch: AtomicU32,
}

impl LockFreeEpochTracker {
    /// Create a new lock-free epoch tracker
    pub fn new() -> Self {
        // Initialize slots array
        let slots: Vec<CacheAlignedSlot> = (0..NUM_SLOTS)
            .map(|_| CacheAlignedSlot::default())
            .collect();
        
        let slots_array: Box<[CacheAlignedSlot; NUM_SLOTS]> = slots
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to create slots array"));
        
        Self {
            slots: slots_array,
            write_cursor: AtomicU64::new(0),
            epoch_boundaries: std::array::from_fn(|_| AtomicU64::new(0)),
            current_epoch: AtomicU32::new(0),
        }
    }
    
    /// Record a version created in the current epoch (lock-free)
    /// 
    /// This operation is O(1) and never blocks. Multiple threads can
    /// call this concurrently with zero contention.
    /// 
    /// # Arguments
    /// * `key_hash` - Pre-computed 64-bit hash of the key
    #[inline]
    pub fn record_version_hash(&self, key_hash: u64) {
        let epoch = self.current_epoch.load(Ordering::Acquire);
        
        // Reserve slot with atomic fetch_add (never fails, never blocks)
        let slot_idx = self.write_cursor.fetch_add(1, Ordering::Relaxed) as usize % NUM_SLOTS;
        
        // Write to reserved slot - we own this slot exclusively
        self.slots[slot_idx].key_hash.store(key_hash, Ordering::Relaxed);
        self.slots[slot_idx].epoch.store(epoch, Ordering::Release);
    }
    
    /// Record a version using key bytes (computes hash internally)
    #[inline]
    pub fn record_version(&self, key: &[u8]) {
        let hash = Self::hash_key(key);
        self.record_version_hash(hash);
    }
    
    /// Advance to next epoch, returning drain iterator for old epoch
    /// 
    /// This establishes a memory fence ensuring all prior writes are visible
    /// before the drain begins.
    pub fn advance_epoch(&self) -> EpochDrain<'_> {
        // 1. Record boundary before incrementing
        let old_cursor = self.write_cursor.load(Ordering::Acquire);
        let old_epoch = self.current_epoch.fetch_add(1, Ordering::AcqRel);
        
        // 2. Store boundary marker
        let boundary_idx = (old_epoch as usize) % MAX_EPOCHS;
        self.epoch_boundaries[boundary_idx].store(old_cursor, Ordering::Release);
        
        // 3. Fence ensures all prior writes are visible before drain
        std::sync::atomic::fence(Ordering::SeqCst);
        
        // 4. Calculate start position (previous epoch's boundary)
        let prev_boundary_idx = (old_epoch.wrapping_sub(1) as usize) % MAX_EPOCHS;
        let start = if old_epoch == 0 {
            0
        } else {
            self.epoch_boundaries[prev_boundary_idx].load(Ordering::Acquire)
        };
        
        EpochDrain {
            tracker: self,
            epoch: old_epoch,
            current: start,
            end: old_cursor,
        }
    }
    
    /// Get current epoch
    #[inline]
    pub fn current(&self) -> u32 {
        self.current_epoch.load(Ordering::Relaxed)
    }
    
    /// Get total recorded count (may wrap)
    #[inline]
    pub fn total_recorded(&self) -> u64 {
        self.write_cursor.load(Ordering::Relaxed)
    }
    
    /// Compute FxHash for a key (fast non-cryptographic hash)
    #[inline]
    fn hash_key(key: &[u8]) -> u64 {
        // FxHash implementation (Rust's default fast hasher)
        const K: u64 = 0x517cc1b727220a95;
        let mut hash: u64 = 0;
        
        // Process 8 bytes at a time
        let chunks = key.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            hash = hash.rotate_left(5) ^ word;
            hash = hash.wrapping_mul(K);
        }
        
        // Handle remaining bytes
        if !remainder.is_empty() {
            let mut last_word = 0u64;
            for (i, &byte) in remainder.iter().enumerate() {
                last_word |= (byte as u64) << (i * 8);
            }
            hash = hash.rotate_left(5) ^ last_word;
            hash = hash.wrapping_mul(K);
        }
        
        hash
    }
}

impl Default for LockFreeEpochTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over dirty key hashes from a specific epoch
/// 
/// This is returned by `advance_epoch()` and yields all key hashes
/// that were recorded during the old epoch.
pub struct EpochDrain<'a> {
    tracker: &'a LockFreeEpochTracker,
    epoch: u32,
    current: u64,
    end: u64,
}

impl<'a> Iterator for EpochDrain<'a> {
    type Item = u64;
    
    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.end {
            let slot_idx = (self.current as usize) % NUM_SLOTS;
            self.current += 1;
            
            // Check if slot belongs to our epoch
            let slot_epoch = self.tracker.slots[slot_idx].epoch.load(Ordering::Acquire);
            if slot_epoch == self.epoch {
                let hash = self.tracker.slots[slot_idx].key_hash.load(Ordering::Relaxed);
                return Some(hash);
            }
            // Slot may have been overwritten by newer epoch - skip
        }
        None
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.end - self.current) as usize;
        (0, Some(remaining))
    }
}

impl EpochDrain<'_> {
    /// Get the epoch being drained
    pub fn epoch(&self) -> u32 {
        self.epoch
    }
    
    /// Get approximate number of slots to drain
    pub fn remaining_estimate(&self) -> u64 {
        self.end.saturating_sub(self.current)
    }
}

// ============================================================================
// Hybrid Epoch Tracker with Bloom Filter
// ============================================================================

/// Hybrid tracker with Bloom filter for fast negative lookups
/// 
/// Uses a parallel Bloom filter to enable O(1) "definitely not dirty" checks,
/// reducing false positives from hash collisions.
pub struct HybridEpochTracker {
    /// Lock-free slot tracker
    slots: LockFreeEpochTracker,
    
    /// Bloom filter for fast negative checks (128 KB)
    bloom: AtomicBloomFilter,
    
    /// Current epoch's Bloom filter
    epoch_bloom: [AtomicBloomFilter; MAX_EPOCHS],
}

impl HybridEpochTracker {
    /// Bloom filter size in bits (2^20 = 1M bits = 128 KB)
    const BLOOM_BITS: usize = 1 << 20;
    
    /// Number of hash functions (optimal for ~100K insertions)
    #[allow(dead_code)]
    const BLOOM_K: usize = 3;
    
    /// Create a new hybrid epoch tracker
    pub fn new() -> Self {
        Self {
            slots: LockFreeEpochTracker::new(),
            bloom: AtomicBloomFilter::new(Self::BLOOM_BITS),
            epoch_bloom: std::array::from_fn(|_| AtomicBloomFilter::new(Self::BLOOM_BITS)),
        }
    }
    
    /// Record a version in current epoch
    #[inline]
    pub fn record_version(&self, key: &[u8]) {
        let hash = LockFreeEpochTracker::hash_key(key);
        self.slots.record_version_hash(hash);
        self.bloom.insert(hash);
        
        let epoch = self.slots.current() as usize % MAX_EPOCHS;
        self.epoch_bloom[epoch].insert(hash);
    }
    
    /// Fast check if key might be dirty
    /// 
    /// Returns `false` if key is definitely NOT dirty (no false negatives).
    /// Returns `true` if key might be dirty (possible false positives).
    #[inline]
    pub fn might_be_dirty(&self, key: &[u8]) -> bool {
        let hash = LockFreeEpochTracker::hash_key(key);
        self.bloom.may_contain(hash)
    }
    
    /// Check if key might be dirty in a specific epoch
    #[inline]
    pub fn might_be_dirty_in_epoch(&self, key: &[u8], epoch: u32) -> bool {
        let hash = LockFreeEpochTracker::hash_key(key);
        let epoch_idx = (epoch as usize) % MAX_EPOCHS;
        self.epoch_bloom[epoch_idx].may_contain(hash)
    }
    
    /// Advance epoch and clear old Bloom filter
    pub fn advance_epoch(&self) -> EpochDrain<'_> {
        let drain = self.slots.advance_epoch();
        
        // Clear the old epoch's Bloom filter for reuse
        let old_epoch_idx = (drain.epoch() as usize) % MAX_EPOCHS;
        self.epoch_bloom[old_epoch_idx].clear();
        
        drain
    }
    
    /// Get current epoch
    #[inline]
    pub fn current(&self) -> u32 {
        self.slots.current()
    }
}

impl Default for HybridEpochTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Atomic Bloom Filter
// ============================================================================

/// Lock-free Bloom filter using atomic bit operations
pub struct AtomicBloomFilter {
    /// Bit array (64 bits per AtomicU64)
    bits: Box<[AtomicU64]>,
    /// Number of bits
    num_bits: usize,
}

impl AtomicBloomFilter {
    /// Create new Bloom filter with specified number of bits
    pub fn new(num_bits: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        let bits: Vec<AtomicU64> = (0..num_words)
            .map(|_| AtomicU64::new(0))
            .collect();
        
        Self {
            bits: bits.into_boxed_slice(),
            num_bits,
        }
    }
    
    /// Insert hash into Bloom filter
    #[inline]
    pub fn insert(&self, hash: u64) {
        // Use 3 hash functions derived from single hash
        let h1 = hash;
        let h2 = hash.rotate_left(21);
        let h3 = hash.rotate_left(42);
        
        self.set_bit((h1 as usize) % self.num_bits);
        self.set_bit((h2 as usize) % self.num_bits);
        self.set_bit((h3 as usize) % self.num_bits);
    }
    
    /// Check if hash may be present (false positives possible)
    #[inline]
    pub fn may_contain(&self, hash: u64) -> bool {
        let h1 = hash;
        let h2 = hash.rotate_left(21);
        let h3 = hash.rotate_left(42);
        
        self.get_bit((h1 as usize) % self.num_bits)
            && self.get_bit((h2 as usize) % self.num_bits)
            && self.get_bit((h3 as usize) % self.num_bits)
    }
    
    /// Clear all bits
    pub fn clear(&self) {
        for word in self.bits.iter() {
            word.store(0, Ordering::Relaxed);
        }
    }
    
    #[inline]
    fn set_bit(&self, bit: usize) {
        let word_idx = bit / 64;
        let bit_idx = bit % 64;
        self.bits[word_idx].fetch_or(1 << bit_idx, Ordering::Relaxed);
    }
    
    #[inline]
    fn get_bit(&self, bit: usize) -> bool {
        let word_idx = bit / 64;
        let bit_idx = bit % 64;
        (self.bits[word_idx].load(Ordering::Relaxed) & (1 << bit_idx)) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    
    #[test]
    fn test_lockfree_epoch_basic() {
        let tracker = LockFreeEpochTracker::new();
        
        // Record some versions
        tracker.record_version(b"key1");
        tracker.record_version(b"key2");
        tracker.record_version(b"key3");
        
        assert_eq!(tracker.total_recorded(), 3);
        assert_eq!(tracker.current(), 0);
        
        // Advance epoch
        let drain = tracker.advance_epoch();
        assert_eq!(drain.epoch(), 0);
        
        let hashes: Vec<u64> = drain.collect();
        assert_eq!(hashes.len(), 3);
        assert_eq!(tracker.current(), 1);
    }
    
    #[test]
    fn test_lockfree_epoch_concurrent() {
        let tracker = Arc::new(LockFreeEpochTracker::new());
        let num_threads = 8;
        let ops_per_thread = 10000;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let key = format!("thread{}:key{}", t, i);
                        tracker.record_version(key.as_bytes());
                    }
                })
            })
            .collect();
        
        for h in handles {
            h.join().unwrap();
        }
        
        assert_eq!(tracker.total_recorded(), (num_threads * ops_per_thread) as u64);
    }
    
    #[test]
    fn test_bloom_filter() {
        let bloom = AtomicBloomFilter::new(1 << 16);
        
        // Insert some hashes
        for i in 0..1000u64 {
            bloom.insert(i * 12345);
        }
        
        // Check insertions
        for i in 0..1000u64 {
            assert!(bloom.may_contain(i * 12345));
        }
        
        // Check false positive rate (should be low)
        let mut false_positives = 0;
        for i in 1000..2000u64 {
            if bloom.may_contain(i * 12345 + 1) {
                false_positives += 1;
            }
        }
        
        // False positive rate should be < 10%
        assert!(false_positives < 100, "False positive rate too high: {}", false_positives);
    }
    
    #[test]
    fn test_hybrid_tracker() {
        let tracker = HybridEpochTracker::new();
        
        // Record versions
        tracker.record_version(b"users/1");
        tracker.record_version(b"users/2");
        tracker.record_version(b"orders/1");
        
        // Check Bloom filter
        assert!(tracker.might_be_dirty(b"users/1"));
        assert!(tracker.might_be_dirty(b"users/2"));
        assert!(tracker.might_be_dirty(b"orders/1"));
        
        // Non-existent key should likely not be in Bloom
        // (might have false positive)
        let non_existent_count = (0..100)
            .filter(|i| tracker.might_be_dirty(format!("nonexistent/{}", i).as_bytes()))
            .count();
        
        // Should have very few false positives
        assert!(non_existent_count < 10);
    }
    
    #[test]
    fn test_epoch_boundary_handling() {
        let tracker = LockFreeEpochTracker::new();
        
        // Record in epoch 0
        tracker.record_version(b"epoch0_key1");
        tracker.record_version(b"epoch0_key2");
        
        // Advance to epoch 1
        let drain0 = tracker.advance_epoch();
        let epoch0_hashes: Vec<_> = drain0.collect();
        assert_eq!(epoch0_hashes.len(), 2);
        
        // Record in epoch 1
        tracker.record_version(b"epoch1_key1");
        tracker.record_version(b"epoch1_key2");
        tracker.record_version(b"epoch1_key3");
        
        // Advance to epoch 2
        let drain1 = tracker.advance_epoch();
        let epoch1_hashes: Vec<_> = drain1.collect();
        assert_eq!(epoch1_hashes.len(), 3);
        
        assert_eq!(tracker.current(), 2);
    }
    
    #[test]
    fn test_hash_consistency() {
        // Verify hash function produces consistent results
        let key = b"test_key_for_hashing";
        let hash1 = LockFreeEpochTracker::hash_key(key);
        let hash2 = LockFreeEpochTracker::hash_key(key);
        assert_eq!(hash1, hash2);
        
        // Different keys should (usually) have different hashes
        let key2 = b"different_key";
        let hash3 = LockFreeEpochTracker::hash_key(key2);
        assert_ne!(hash1, hash3);
    }
}
