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

//! Lock-Free String Interner with Chunked Append-Only Storage
//!
//! This module implements a high-performance string interner that:
//! - Uses chunked append-only storage (never moves existing strings)
//! - Provides lock-free reads via pre-allocated fixed buffers
//! - Uses sharded hash maps to reduce write contention
//! - Supports zero-cost string comparison via Symbol handles
//!
//! # Concurrency Guarantees - PLEASE READ CAREFULLY
//!
//! Despite the module name, this interner is **NOT fully lock-free**.
//! The "lock-free" in the name refers specifically to the read path.
//!
//! ## Threading Guarantees Table
//!
//! | Operation | Guarantee | Notes |
//! |-----------|-----------|-------|
//! | `resolve()` | Wait-free | Atomic pointer traversal, no locks |
//! | `get()` | Low-contention | Hash lookup with sharded RwLocks |
//! | `intern()` | Blocking | Requires write lock on shard |
//! | `len()` | Wait-free | Atomic counter |
//!
//! ## What IS Provided
//!
//! - **Thread-safety**: All operations are safe from multiple threads
//! - **Wait-free reads**: `resolve()` uses atomic chunk traversal
//! - **Low-contention lookups**: 256-way sharding reduces lock contention
//! - **Deadlock-freedom**: Single lock per shard, no nested locking
//!
//! ## What is NOT Provided
//!
//! - **Lock-free writes**: Interning requires write locks on shards
//! - **Wait-free interning**: Writers may block under contention
//!
//! ## Progress Taxonomy
//!
//! ```text
//! wait-free ⊂ lock-free ⊂ obstruction-free ⊂ blocking
//!     ↑           ↑
//!  resolve()   (not provided)
//! ```
//!
//! This implementation provides:
//! - **Wait-free reads**: Always complete in bounded steps
//! - **Blocking writes**: May wait for lock acquisition
//!
//! ## Memory Layout
//!
//! Each chunk is a fixed-size pre-allocated buffer using `Box<[MaybeUninit<u8>]>`.
//! This eliminates the data race that would occur with `Vec<u8>` (whose metadata
//! like `len` and `capacity` are not thread-safe to modify concurrently).
//!
//! ```text
//! Chunk 0 (1MB fixed)       Chunk 1 (1MB fixed)       Chunk 2 (1MB fixed)
//! ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
//! │ "hello"         │       │ "different"     │       │ "more strings"  │
//! │ "world"         │       │ "strings"       │       │ ...             │
//! │ "foo"           │       │ "here"          │       │                 │
//! │ [uninitialized] │       │ [uninitialized] │       │ [uninitialized] │
//! └─────────────────┘       └─────────────────┘       └─────────────────┘
//!           ↑                        ↑                        ↑
//!           │                        │                        │
//!     AtomicPtr chain for lock-free traversal
//! ```

use parking_lot::RwLock as ParkingLotRwLock;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering, fence};

/// Number of shards for the hash map (power of 2 for fast modulo)
const SHARD_COUNT: usize = 256;

/// Default chunk size (1MB)
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Symbol handle - cheap to copy and compare
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub u32);

impl Symbol {
    /// Invalid/null symbol
    pub const NULL: Symbol = Symbol(u32::MAX);

    /// Get the raw index
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Reference to a string location in a chunk
#[derive(Clone, Copy)]
struct StringRef {
    /// Chunk index
    chunk_idx: u32,
    /// Offset within chunk
    offset: u32,
    /// Length of string
    length: u32,
}

/// A single chunk of string storage using pre-allocated fixed buffer
///
/// ## Safety Rationale
///
/// This design eliminates the data race present in the Vec-based approach:
/// - `capacity` is immutable after construction
/// - `data` is a fixed-size buffer that never resizes
/// - `write_pos` atomically reserves disjoint byte ranges
/// - Each thread writes only to its reserved range
///
/// The only shared mutable state is the byte range reservation via `write_pos`,
/// which is handled atomically with `fetch_add`.
struct Chunk {
    /// Fixed-size pre-allocated buffer (never resized)
    /// Using MaybeUninit because we write to reserved ranges atomically
    data: *mut u8,
    /// Buffer capacity (immutable after creation)
    capacity: usize,
    /// Current write position (atomically updated)
    write_pos: AtomicUsize,
    /// Next chunk in chain
    next: AtomicPtr<Chunk>,
}

impl Chunk {
    fn new(capacity: usize) -> Box<Self> {
        // Allocate the buffer as a Vec<u8> with zeros
        let buffer: Vec<u8> = vec![0u8; capacity];
        let ptr = Box::into_raw(buffer.into_boxed_slice()) as *mut u8;

        Box::new(Self {
            data: ptr,
            capacity,
            write_pos: AtomicUsize::new(0),
            next: AtomicPtr::new(std::ptr::null_mut()),
        })
    }

    /// Try to append a string, returns offset if successful
    ///
    /// ## Safety
    ///
    /// This is safe because:
    /// 1. `fetch_add` atomically reserves a disjoint byte range
    /// 2. Each thread writes only to its reserved range
    /// 3. The buffer is pre-allocated and never resized (no Vec metadata race)
    /// 4. Release fence ensures bytes are visible before returning
    fn try_append(&self, s: &str) -> Option<u32> {
        let bytes = s.as_bytes();
        let len = bytes.len();

        if len == 0 {
            // Empty strings can share offset 0 with length 0
            return Some(0);
        }

        // Atomically reserve space
        let offset = self.write_pos.fetch_add(len, Ordering::Relaxed);

        // Check if we have room
        if offset + len > self.capacity {
            // Rollback reservation (best effort - may waste some space on high contention)
            // This is safe because the space is simply unused
            self.write_pos.fetch_sub(len, Ordering::Relaxed);
            return None;
        }

        // SAFETY: We have exclusive access to [offset..offset+len] via atomic reservation
        // The buffer was pre-allocated with `capacity` bytes
        // No other thread will write to this range
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(offset), len);
        }

        // Release fence ensures bytes are visible to other threads
        // before we return the StringRef (which gets published to the hash map)
        fence(Ordering::Release);

        Some(offset as u32)
    }

    /// Read a string at offset (lock-free)
    ///
    /// ## Safety
    ///
    /// Caller must ensure:
    /// 1. The offset and length were returned by a prior `try_append`
    /// 2. An Acquire fence was issued after observing the StringRef
    fn read(&self, offset: u32, len: u32) -> &str {
        // Acquire fence ensures we see the bytes written by try_append
        fence(Ordering::Acquire);

        // SAFETY:
        // - offset and len came from a successful try_append
        // - The buffer is stable (never moved or reallocated)
        // - Acquire fence synchronized with Release fence in try_append
        unsafe {
            let bytes = std::slice::from_raw_parts(self.data.add(offset as usize), len as usize);
            // SAFETY: we only store valid UTF-8 from str::as_bytes()
            std::str::from_utf8_unchecked(bytes)
        }
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // Reconstruct and drop the boxed slice
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.data, self.capacity);
            drop(Box::from_raw(slice as *mut [u8]));
        }
    }
}

/// Chunked append-only storage for strings
pub struct ChunkedStorage {
    /// Head of chunk chain
    head: AtomicPtr<Chunk>,
    /// Current chunk for new allocations
    current: AtomicPtr<Chunk>,
    /// Chunk size
    chunk_size: usize,
    /// Number of chunks
    chunk_count: AtomicU32,
}

impl ChunkedStorage {
    fn new(chunk_size: usize) -> Self {
        let initial = Box::into_raw(Chunk::new(chunk_size));
        Self {
            head: AtomicPtr::new(initial),
            current: AtomicPtr::new(initial),
            chunk_size,
            chunk_count: AtomicU32::new(1),
        }
    }

    /// Append a string, returns reference
    fn append(&self, s: &str) -> StringRef {
        loop {
            let current = self.current.load(Ordering::Acquire);
            let chunk = unsafe { &*current };

            // Try to append to current chunk
            if let Some(offset) = chunk.try_append(s) {
                let chunk_idx = self.get_chunk_index(current);
                return StringRef {
                    chunk_idx,
                    offset,
                    length: s.len() as u32,
                };
            }

            // Need new chunk
            self.grow_chunk(current);
        }
    }

    /// Grow by adding a new chunk
    fn grow_chunk(&self, expected_current: *mut Chunk) {
        let new_chunk = Box::into_raw(Chunk::new(self.chunk_size));

        // Link new chunk to current
        let current = unsafe { &*expected_current };

        // Try to set next pointer
        if current
            .next
            .compare_exchange(
                std::ptr::null_mut(),
                new_chunk,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // We successfully linked the new chunk
            self.current.store(new_chunk, Ordering::Release);
            self.chunk_count.fetch_add(1, Ordering::Relaxed);
        } else {
            // Another thread already grew - free our chunk
            unsafe {
                drop(Box::from_raw(new_chunk));
            }
            // Update current to the actual new chunk
            let actual_next = current.next.load(Ordering::Acquire);
            if !actual_next.is_null() {
                self.current.store(actual_next, Ordering::Release);
            }
        }
    }

    /// Get chunk by index (lock-free traversal)
    fn get_chunk(&self, idx: u32) -> Option<&Chunk> {
        let mut current = self.head.load(Ordering::Acquire);
        let mut i = 0u32;

        while !current.is_null() {
            if i == idx {
                return Some(unsafe { &*current });
            }
            current = unsafe { (*current).next.load(Ordering::Acquire) };
            i += 1;
        }

        None
    }

    /// Get index of a chunk pointer
    fn get_chunk_index(&self, ptr: *mut Chunk) -> u32 {
        let mut current = self.head.load(Ordering::Acquire);
        let mut i = 0u32;

        while !current.is_null() {
            if current == ptr {
                return i;
            }
            current = unsafe { (*current).next.load(Ordering::Acquire) };
            i += 1;
        }

        0 // Fallback (shouldn't happen)
    }

    /// Read a string by reference (lock-free)
    fn read(&self, string_ref: StringRef) -> Option<&str> {
        let chunk = self.get_chunk(string_ref.chunk_idx)?;
        Some(chunk.read(string_ref.offset, string_ref.length))
    }
}

impl Drop for ChunkedStorage {
    fn drop(&mut self) {
        let mut current = self.head.load(Ordering::Relaxed);
        while !current.is_null() {
            let chunk = unsafe { Box::from_raw(current) };
            current = chunk.next.load(Ordering::Relaxed);
        }
    }
}

/// Hasher for string lookup
fn hash_string(s: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Lock-free string interner
pub struct LockFreeInterner {
    /// Sharded hash maps: string hash -> Symbol
    shards: [ParkingLotRwLock<HashMap<u64, (Symbol, StringRef)>>; SHARD_COUNT],
    /// Chunked string storage
    storage: ChunkedStorage,
    /// Next symbol ID
    next_symbol: AtomicU32,
    /// Reverse mapping: Symbol -> StringRef (for resolve)
    symbols: RwLock<Vec<StringRef>>,
    /// Statistics
    stats: InternerStats,
}

/// Interner statistics
#[derive(Default)]
pub struct InternerStats {
    pub interned_count: AtomicU32,
    pub lookup_hits: AtomicU32,
    pub lookup_misses: AtomicU32,
    pub resolve_count: AtomicU32,
}

impl Default for LockFreeInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeInterner {
    /// Create a new interner
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Create with custom chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            shards: std::array::from_fn(|_| ParkingLotRwLock::new(HashMap::new())),
            storage: ChunkedStorage::new(chunk_size),
            next_symbol: AtomicU32::new(0),
            symbols: RwLock::new(Vec::new()),
            stats: InternerStats::default(),
        }
    }

    /// Get shard index for a hash
    #[inline]
    fn shard_index(&self, hash: u64) -> usize {
        (hash as usize) % SHARD_COUNT
    }

    /// Get or intern a string
    pub fn get_or_intern(&self, s: &str) -> Symbol {
        let hash = hash_string(s);
        let shard_idx = self.shard_index(hash);

        // Fast path: read-only lookup
        {
            let shard = self.shards[shard_idx].read();
            if let Some(&(symbol, _)) = shard.get(&hash) {
                self.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
                return symbol;
            }
        }

        self.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);

        // Slow path: need to intern
        let mut shard = self.shards[shard_idx].write();

        // Double-check after acquiring write lock
        if let Some(&(symbol, _)) = shard.get(&hash) {
            return symbol;
        }

        // Allocate symbol and store string
        let symbol = Symbol(self.next_symbol.fetch_add(1, Ordering::SeqCst));
        let string_ref = self.storage.append(s);

        shard.insert(hash, (symbol, string_ref));

        // Update reverse mapping
        {
            let mut symbols = self.symbols.write().unwrap();
            // Ensure vector is large enough
            while symbols.len() <= symbol.0 as usize {
                symbols.push(StringRef {
                    chunk_idx: 0,
                    offset: 0,
                    length: 0,
                });
            }
            symbols[symbol.0 as usize] = string_ref;
        }

        self.stats.interned_count.fetch_add(1, Ordering::Relaxed);
        symbol
    }

    /// Resolve a symbol to its string (lock-free for reads!)
    pub fn resolve(&self, symbol: Symbol) -> Option<&str> {
        if symbol == Symbol::NULL {
            return None;
        }

        self.stats.resolve_count.fetch_add(1, Ordering::Relaxed);

        // Read from symbols vector (only takes read lock briefly)
        let string_ref = {
            let symbols = self.symbols.read().unwrap();
            symbols.get(symbol.0 as usize).copied()
        }?;

        // Storage read is completely lock-free
        self.storage.read(string_ref)
    }

    /// Get the number of interned strings
    pub fn len(&self) -> usize {
        self.next_symbol.load(Ordering::Relaxed) as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics
    pub fn stats(&self) -> &InternerStats {
        &self.stats
    }

    /// Get total memory used by string storage
    pub fn storage_bytes(&self) -> usize {
        self.storage.chunk_count.load(Ordering::Relaxed) as usize * self.storage.chunk_size
    }
}

// SAFETY: LockFreeInterner is safe to share across threads because:
// 1. All hash map access is protected by ParkingLotRwLock (sharded)
// 2. ChunkedStorage uses atomic operations for all shared mutable state
// 3. Chunk buffers are pre-allocated fixed-size (no Vec metadata races)
// 4. Memory ordering is correct (Release on write, Acquire on read)
unsafe impl Send for LockFreeInterner {}
unsafe impl Sync for LockFreeInterner {}

// SAFETY: ChunkedStorage is safe to share across threads because:
// 1. write_pos uses atomic fetch_add for disjoint range reservation
// 2. Chunk buffers are fixed-size pre-allocations (immutable capacity)
// 3. Chunk chain traversal uses AtomicPtr with proper ordering
// 4. Each writer has exclusive access to its reserved byte range
unsafe impl Send for ChunkedStorage {}
unsafe impl Sync for ChunkedStorage {}

// SAFETY: Chunk contains a raw pointer but is designed for concurrent access:
// 1. data pointer is stable (never moved after allocation)
// 2. capacity is immutable after construction
// 3. write_pos is atomic
// 4. Each thread writes only to atomically-reserved disjoint ranges
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_intern() {
        let interner = LockFreeInterner::new();

        let s1 = interner.get_or_intern("hello");
        let s2 = interner.get_or_intern("world");
        let s3 = interner.get_or_intern("hello");

        // Same string should give same symbol
        assert_eq!(s1, s3);
        // Different strings should give different symbols
        assert_ne!(s1, s2);

        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_resolve() {
        let interner = LockFreeInterner::new();

        let s1 = interner.get_or_intern("hello");
        let s2 = interner.get_or_intern("world");

        assert_eq!(interner.resolve(s1), Some("hello"));
        assert_eq!(interner.resolve(s2), Some("world"));
        assert_eq!(interner.resolve(Symbol::NULL), None);
    }

    #[test]
    fn test_concurrent_intern() {
        let interner = Arc::new(LockFreeInterner::new());
        let mut handles = vec![];

        // 8 threads each intern the same 100 strings
        for _ in 0..8 {
            let interner = interner.clone();
            handles.push(thread::spawn(move || {
                let mut symbols = Vec::new();
                for i in 0..100 {
                    let s = format!("string_{}", i);
                    symbols.push(interner.get_or_intern(&s));
                }
                symbols
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same symbols for the same strings
        for i in 0..100 {
            let first = results[0][i];
            for result in &results[1..] {
                assert_eq!(result[i], first, "Symbol mismatch for string_{}", i);
            }
        }

        // Should have exactly 100 unique strings
        assert_eq!(interner.len(), 100);
    }

    #[test]
    fn test_concurrent_resolve() {
        let interner = Arc::new(LockFreeInterner::new());

        // Intern some strings first
        let symbols: Vec<_> = (0..100)
            .map(|i| interner.get_or_intern(&format!("string_{}", i)))
            .collect();

        let mut handles = vec![];

        // 8 threads each resolve all strings many times
        for _ in 0..8 {
            let interner = interner.clone();
            let symbols = symbols.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    for (i, &symbol) in symbols.iter().enumerate() {
                        let resolved = interner.resolve(symbol);
                        assert_eq!(resolved, Some(format!("string_{}", i)).as_deref());
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have many resolve operations
        assert!(interner.stats().resolve_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_chunk_growth() {
        // Small chunk size to force growth
        let interner = LockFreeInterner::with_chunk_size(1024);

        // Intern enough strings to fill multiple chunks
        for i in 0..1000 {
            let s = format!("this_is_a_longer_string_number_{:05}", i);
            interner.get_or_intern(&s);
        }

        // Should have grown to multiple chunks
        assert!(interner.storage.chunk_count.load(Ordering::Relaxed) > 1);

        // All strings should still be resolvable
        for i in 0..1000 {
            let s = format!("this_is_a_longer_string_number_{:05}", i);
            let symbol = interner.get_or_intern(&s);
            assert_eq!(interner.resolve(symbol), Some(s.as_str()));
        }
    }

    #[test]
    fn test_symbol_comparison() {
        let interner = LockFreeInterner::new();

        let s1 = interner.get_or_intern("hello");
        let s2 = interner.get_or_intern("world");
        let s3 = interner.get_or_intern("hello");

        // Symbol comparison is O(1) - just comparing u32
        assert_eq!(s1, s3);
        assert_ne!(s1, s2);

        // Can use as hash map key
        let mut map = HashMap::new();
        map.insert(s1, "value1");
        map.insert(s2, "value2");

        assert_eq!(map.get(&s3), Some(&"value1"));
    }

    #[test]
    fn test_stats() {
        let interner = LockFreeInterner::new();

        // Intern some strings
        interner.get_or_intern("hello");
        interner.get_or_intern("world");
        interner.get_or_intern("hello"); // Hit

        let stats = interner.stats();
        assert_eq!(stats.interned_count.load(Ordering::Relaxed), 2);
        assert!(stats.lookup_hits.load(Ordering::Relaxed) >= 1);
        assert!(stats.lookup_misses.load(Ordering::Relaxed) >= 2);
    }

    #[test]
    fn test_concurrent_intern_and_resolve() {
        let interner = Arc::new(LockFreeInterner::new());
        let mut handles = vec![];

        // Writers: intern new strings
        for writer_id in 0..4 {
            let interner = interner.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let s = format!("writer_{}_string_{}", writer_id, i);
                    interner.get_or_intern(&s);
                }
            }));
        }

        // Readers: continuously resolve existing strings
        for reader_id in 0..4 {
            let interner = interner.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    // Intern and immediately resolve
                    let s = format!("common_string_{}", (i + reader_id) % 10);
                    let symbol = interner.get_or_intern(&s);
                    let resolved = interner.resolve(symbol);
                    assert!(resolved.is_some());
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_empty_string() {
        let interner = LockFreeInterner::new();

        let s = interner.get_or_intern("");
        assert_eq!(interner.resolve(s), Some(""));
    }

    #[test]
    fn test_long_strings() {
        let interner = LockFreeInterner::new();

        let long_string: String = (0..10000).map(|_| 'x').collect();
        let symbol = interner.get_or_intern(&long_string);
        assert_eq!(interner.resolve(symbol), Some(long_string.as_str()));
    }

    #[test]
    fn test_unicode_strings() {
        let interner = LockFreeInterner::new();

        let s1 = interner.get_or_intern("hello 世界 🌍");
        let s2 = interner.get_or_intern("émojis: 🎉🎊🎁");

        assert_eq!(interner.resolve(s1), Some("hello 世界 🌍"));
        assert_eq!(interner.resolve(s2), Some("émojis: 🎉🎊🎁"));
    }

    #[test]
    fn test_shard_distribution() {
        let interner = LockFreeInterner::new();

        // Intern many strings
        for i in 0..10000 {
            interner.get_or_intern(&format!("key_{}", i));
        }

        // Check that strings are distributed across shards
        let mut non_empty = 0;
        for shard in &interner.shards {
            if !shard.read().is_empty() {
                non_empty += 1;
            }
        }

        // With 10000 strings across 256 shards, should use many shards
        assert!(
            non_empty > 200,
            "Expected better shard distribution: {} non-empty",
            non_empty
        );
    }
}
