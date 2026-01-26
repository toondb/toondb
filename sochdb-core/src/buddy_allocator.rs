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

//! Buddy Allocator for SochDB Memory Management
//!
//! # DEPRECATION NOTICE
//!
//! **This module is deprecated and will be removed in a future release.**
//!
//! ## Recommendation: Use jemalloc Instead
//!
//! Enable the `jemalloc` feature in your `Cargo.toml`:
//!
//! ```toml
//! sochdb-core = { version = "...", features = ["jemalloc"] }
//! ```
//!
//! ### Why jemalloc is preferred:
//!
//! 1. **Production-proven**: Used by Firefox, Facebook, Redis, RocksDB
//! 2. **Thread-local caching**: Eliminates lock contention on hot paths
//! 3. **Better fragmentation handling**: Size-class based allocation
//! 4. **Automatic memory return**: Returns memory to OS via `madvise`
//! 5. **No maintenance burden**: Well-tested, battle-hardened code
//!
//! ### Issues with this custom allocator:
//!
//! 1. **Virtual address tracking only**: Does not manage real memory
//! 2. **Lock contention**: RwLock on every allocation
//! 3. **No memory reclamation**: Never returns memory to OS
//! 4. **Internal fragmentation**: Buddy allocation wastes ~50% for small allocs
//!
//! ## Migration Path
//!
//! If you are using `BuddyAllocator` or `SlabAllocator` directly:
//!
//! ```rust,ignore
//! // Before (deprecated):
//! let allocator = BuddyAllocator::new(64 * 1024 * 1024)?;
//! let addr = allocator.allocate(1024)?;
//!
//! // After (recommended):
//! // Simply enable the jemalloc feature and use standard allocation:
//! let buffer: Vec<u8> = Vec::with_capacity(1024);
//! ```
//!
//! ## Original Documentation
//!
//! Implements a buddy system allocator for efficient power-of-2 memory block allocation.
//! Key features:
//! - O(1) allocation and deallocation for available blocks
//! - Automatic block splitting and merging
//! - Memory coalescing to reduce fragmentation
//! - Thread-safe with fine-grained locking
//! - Support for multiple memory pools

use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Minimum block size (16 bytes)
pub const MIN_BLOCK_SIZE: usize = 16;
/// Maximum block size (1 GB)
pub const MAX_BLOCK_SIZE: usize = 1 << 30;
/// Default pool size (64 MB)
pub const DEFAULT_POOL_SIZE: usize = 64 * 1024 * 1024;

/// Error types for buddy allocator operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuddyError {
    /// Requested size is too large
    SizeTooLarge(usize),
    /// Requested size is zero
    ZeroSize,
    /// No memory available
    OutOfMemory,
    /// Invalid address
    InvalidAddress(usize),
    /// Double free detected
    DoubleFree(usize),
    /// Block not found
    BlockNotFound(usize),
    /// Pool exhausted
    PoolExhausted,
}

/// Block header stored at the beginning of each block
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockHeader {
    /// Magic number for validation
    magic: u32,
    /// Order (log2 of size)
    order: u8,
    /// Flags (allocated, etc.)
    flags: u8,
    /// Padding
    _padding: [u8; 2],
    /// Size of the block
    size: u32,
}

#[allow(dead_code)]
const BLOCK_MAGIC: u32 = 0xB0DD_1E5A;
#[allow(dead_code)]
const FLAG_ALLOCATED: u8 = 0x01;
#[allow(dead_code)]
const FLAG_SPLIT: u8 = 0x02;

#[allow(dead_code)]
impl BlockHeader {
    fn new(order: u8, size: u32) -> Self {
        Self {
            magic: BLOCK_MAGIC,
            order,
            flags: 0,
            _padding: [0; 2],
            size,
        }
    }

    fn is_valid(&self) -> bool {
        self.magic == BLOCK_MAGIC
    }

    fn is_allocated(&self) -> bool {
        self.flags & FLAG_ALLOCATED != 0
    }

    fn set_allocated(&mut self, allocated: bool) {
        if allocated {
            self.flags |= FLAG_ALLOCATED;
        } else {
            self.flags &= !FLAG_ALLOCATED;
        }
    }

    fn is_split(&self) -> bool {
        self.flags & FLAG_SPLIT != 0
    }

    fn set_split(&mut self, split: bool) {
        if split {
            self.flags |= FLAG_SPLIT;
        } else {
            self.flags &= !FLAG_SPLIT;
        }
    }
}

/// A free block in the buddy allocator
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    /// Address of the block
    addr: usize,
    /// Next free block in the list (0 = end)
    next: usize,
}

/// Statistics for the buddy allocator
#[derive(Debug, Default)]
pub struct BuddyStats {
    /// Total allocations
    pub allocations: AtomicU64,
    /// Total deallocations
    pub deallocations: AtomicU64,
    /// Current allocated bytes
    pub allocated_bytes: AtomicUsize,
    /// Peak allocated bytes
    pub peak_allocated_bytes: AtomicUsize,
    /// Number of splits
    pub splits: AtomicU64,
    /// Number of merges
    pub merges: AtomicU64,
    /// Failed allocations
    pub failed_allocations: AtomicU64,
}

impl BuddyStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_allocation(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let old = self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
        let new = old + size;

        // Update peak if necessary
        let mut current_peak = self.peak_allocated_bytes.load(Ordering::Relaxed);
        while new > current_peak {
            match self.peak_allocated_bytes.compare_exchange_weak(
                current_peak,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => current_peak = p,
            }
        }
    }

    pub fn record_deallocation(&self, size: usize) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    pub fn record_split(&self) {
        self.splits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_merge(&self) {
        self.merges.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_failed_allocation(&self) {
        self.failed_allocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// Calculate the order (log2) for a given size
#[inline]
pub fn size_to_order(size: usize) -> u8 {
    if size <= MIN_BLOCK_SIZE {
        return 4; // 2^4 = 16
    }

    // Round up to next power of 2
    let bits = (size - 1).leading_zeros();
    (usize::BITS - bits) as u8
}

/// Calculate the size for a given order
#[inline]
pub fn order_to_size(order: u8) -> usize {
    1 << order
}

/// Calculate buddy address for a given address and order
#[inline]
pub fn buddy_addr(addr: usize, order: u8) -> usize {
    addr ^ (1 << order)
}

/// A single memory pool managed by the buddy allocator
pub struct MemoryPool {
    /// Base address of the pool
    base: usize,
    /// Total size of the pool
    size: usize,
    /// Maximum order (log2 of pool size)
    max_order: u8,
    /// Minimum order (log2 of minimum block size)
    min_order: u8,
    /// Free lists for each order (index = order - min_order)
    free_lists: Vec<Mutex<Vec<usize>>>,
    /// Map of allocated blocks: address -> order
    allocated: RwLock<HashMap<usize, u8>>,
    /// Statistics
    stats: BuddyStats,
}

impl MemoryPool {
    /// Create a new memory pool with the given base address and size
    pub fn new(base: usize, size: usize) -> Result<Self, BuddyError> {
        if size == 0 {
            return Err(BuddyError::ZeroSize);
        }

        // Ensure size is power of 2
        if !size.is_power_of_two() {
            return Err(BuddyError::SizeTooLarge(size));
        }

        let max_order = size_to_order(size);
        let min_order = size_to_order(MIN_BLOCK_SIZE);

        let num_orders = (max_order - min_order + 1) as usize;
        let mut free_lists = Vec::with_capacity(num_orders);

        for _ in 0..num_orders {
            free_lists.push(Mutex::new(Vec::new()));
        }

        // Initialize with one block of maximum size
        free_lists[num_orders - 1].lock().push(base);

        Ok(Self {
            base,
            size,
            max_order,
            min_order,
            free_lists,
            allocated: RwLock::new(HashMap::new()),
            stats: BuddyStats::new(),
        })
    }

    /// Get the index into free_lists for a given order
    fn list_index(&self, order: u8) -> usize {
        (order - self.min_order) as usize
    }

    /// Allocate a block of the given size
    pub fn allocate(&self, size: usize) -> Result<usize, BuddyError> {
        if size == 0 {
            return Err(BuddyError::ZeroSize);
        }

        let required_order = size_to_order(size);

        if required_order > self.max_order {
            self.stats.record_failed_allocation();
            return Err(BuddyError::SizeTooLarge(size));
        }

        // Find the smallest available block that fits
        let mut found_order = None;
        for order in required_order..=self.max_order {
            let idx = self.list_index(order);
            let list = self.free_lists[idx].lock();
            if !list.is_empty() {
                found_order = Some(order);
                break;
            }
        }

        let available_order = match found_order {
            Some(o) => o,
            None => {
                self.stats.record_failed_allocation();
                return Err(BuddyError::OutOfMemory);
            }
        };

        // Pop a block from the found level
        let addr = {
            let idx = self.list_index(available_order);
            self.free_lists[idx]
                .lock()
                .pop()
                .ok_or(BuddyError::OutOfMemory)?
        };

        // Split down to the required size
        let final_addr = self.split_block(addr, available_order, required_order);

        // Mark as allocated
        self.allocated.write().insert(final_addr, required_order);
        self.stats.record_allocation(order_to_size(required_order));

        Ok(final_addr)
    }

    /// Split a block from current_order down to target_order
    fn split_block(&self, addr: usize, current_order: u8, target_order: u8) -> usize {
        let mut current_addr = addr;
        let mut order = current_order;

        while order > target_order {
            order -= 1;
            self.stats.record_split();

            // Calculate buddy address
            let buddy = buddy_addr(current_addr, order);

            // Add buddy to free list
            let idx = self.list_index(order);
            self.free_lists[idx].lock().push(buddy);

            // Keep the lower address
            if buddy < current_addr {
                current_addr = buddy;
            }
        }

        current_addr
    }

    /// Free a previously allocated block
    pub fn deallocate(&self, addr: usize) -> Result<(), BuddyError> {
        // Get the order of this block
        let order = {
            let mut allocated = self.allocated.write();
            allocated
                .remove(&addr)
                .ok_or(BuddyError::InvalidAddress(addr))?
        };

        self.stats.record_deallocation(order_to_size(order));

        // Try to merge with buddy
        self.merge_block(addr, order);

        Ok(())
    }

    /// Merge a freed block with its buddy if possible
    fn merge_block(&self, addr: usize, order: u8) {
        let mut current_addr = addr;
        let mut current_order = order;

        while current_order < self.max_order {
            let buddy = buddy_addr(current_addr, current_order);

            // Check if buddy is in the free list
            let idx = self.list_index(current_order);
            let mut list = self.free_lists[idx].lock();

            if let Some(pos) = list.iter().position(|&a| a == buddy) {
                // Remove buddy from free list
                list.swap_remove(pos);
                drop(list);

                self.stats.record_merge();

                // Use lower address as the new block
                current_addr = current_addr.min(buddy);
                current_order += 1;
            } else {
                // Buddy not free, add current block to free list
                list.push(current_addr);
                return;
            }
        }

        // Reached max order, add to top-level free list
        let idx = self.list_index(current_order);
        self.free_lists[idx].lock().push(current_addr);
    }

    /// Check if an address is within this pool
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.base + self.size
    }

    /// Get pool statistics
    pub fn stats(&self) -> &BuddyStats {
        &self.stats
    }

    /// Get the number of free blocks at each order
    pub fn free_block_counts(&self) -> Vec<(u8, usize)> {
        let mut counts = Vec::new();
        for order in self.min_order..=self.max_order {
            let idx = self.list_index(order);
            let count = self.free_lists[idx].lock().len();
            counts.push((order, count));
        }
        counts
    }

    /// Get total free bytes
    pub fn free_bytes(&self) -> usize {
        let mut total = 0;
        for order in self.min_order..=self.max_order {
            let idx = self.list_index(order);
            let count = self.free_lists[idx].lock().len();
            total += count * order_to_size(order);
        }
        total
    }
}

/// Multi-pool buddy allocator
pub struct BuddyAllocator {
    /// Memory pools
    pools: RwLock<Vec<MemoryPool>>,
    /// Default pool size for new pools
    default_pool_size: usize,
    /// Next pool base address (for virtual addressing)
    next_base: AtomicUsize,
    /// Global statistics
    stats: BuddyStats,
}

impl BuddyAllocator {
    /// Create a new buddy allocator with default settings
    pub fn new() -> Self {
        Self::with_pool_size(DEFAULT_POOL_SIZE)
    }

    /// Create a new buddy allocator with the specified default pool size
    pub fn with_pool_size(pool_size: usize) -> Self {
        // Round up to power of 2
        let pool_size = pool_size.next_power_of_two();

        Self {
            pools: RwLock::new(Vec::new()),
            default_pool_size: pool_size,
            next_base: AtomicUsize::new(0x1000), // Start after null page
            stats: BuddyStats::new(),
        }
    }

    /// Add a new memory pool
    pub fn add_pool(&self, size: usize) -> Result<usize, BuddyError> {
        let size = size.next_power_of_two();
        let base = self.next_base.fetch_add(size, Ordering::SeqCst);

        let pool = MemoryPool::new(base, size)?;
        let pool_id = {
            let mut pools = self.pools.write();
            let id = pools.len();
            pools.push(pool);
            id
        };

        Ok(pool_id)
    }

    /// Ensure at least one pool exists
    fn ensure_pool(&self) -> Result<(), BuddyError> {
        let pools = self.pools.read();
        if pools.is_empty() {
            drop(pools);
            self.add_pool(self.default_pool_size)?;
        }
        Ok(())
    }

    /// Allocate memory
    pub fn allocate(&self, size: usize) -> Result<usize, BuddyError> {
        if size == 0 {
            return Err(BuddyError::ZeroSize);
        }

        self.ensure_pool()?;

        // Try each pool
        let pools = self.pools.read();
        for pool in pools.iter() {
            if let Ok(addr) = pool.allocate(size) {
                self.stats
                    .record_allocation(order_to_size(size_to_order(size)));
                return Ok(addr);
            }
        }

        // No pool had space, try to add a new one
        drop(pools);

        let required_size = size.next_power_of_two().max(self.default_pool_size);
        self.add_pool(required_size)?;

        let pools = self.pools.read();
        if let Some(pool) = pools.last()
            && let Ok(addr) = pool.allocate(size)
        {
            self.stats
                .record_allocation(order_to_size(size_to_order(size)));
            return Ok(addr);
        }

        self.stats.record_failed_allocation();
        Err(BuddyError::OutOfMemory)
    }

    /// Deallocate memory
    pub fn deallocate(&self, addr: usize) -> Result<(), BuddyError> {
        let pools = self.pools.read();

        for pool in pools.iter() {
            // Check if this pool has this address allocated
            let has_addr = pool.allocated.read().contains_key(&addr);
            if has_addr {
                let order = {
                    let allocated = pool.allocated.read();
                    allocated.get(&addr).copied()
                };

                if let Some(order) = order {
                    self.stats.record_deallocation(order_to_size(order));
                }

                return pool.deallocate(addr);
            }
        }

        Err(BuddyError::InvalidAddress(addr))
    }

    /// Get global statistics
    pub fn stats(&self) -> &BuddyStats {
        &self.stats
    }

    /// Get total free bytes across all pools
    pub fn total_free_bytes(&self) -> usize {
        let pools = self.pools.read();
        pools.iter().map(|p| p.free_bytes()).sum()
    }

    /// Get number of pools
    pub fn pool_count(&self) -> usize {
        self.pools.read().len()
    }
}

impl Default for BuddyAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed buddy allocator for allocating objects of a specific type
pub struct TypedBuddyAllocator<T> {
    allocator: BuddyAllocator,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TypedBuddyAllocator<T> {
    /// Create a new typed allocator
    pub fn new() -> Self {
        Self {
            allocator: BuddyAllocator::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Allocate space for one T
    pub fn allocate_one(&self) -> Result<usize, BuddyError> {
        self.allocator.allocate(std::mem::size_of::<T>())
    }

    /// Allocate space for N elements
    pub fn allocate_array(&self, count: usize) -> Result<usize, BuddyError> {
        self.allocator.allocate(std::mem::size_of::<T>() * count)
    }

    /// Deallocate
    pub fn deallocate(&self, addr: usize) -> Result<(), BuddyError> {
        self.allocator.deallocate(addr)
    }
}

impl<T> Default for TypedBuddyAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Slab allocator built on top of buddy allocator for fixed-size objects
pub struct SlabAllocator {
    /// Object size
    object_size: usize,
    /// Objects per slab
    objects_per_slab: usize,
    /// Underlying buddy allocator
    buddy: BuddyAllocator,
    /// Free object lists per slab: slab_base -> list of free offsets
    slabs: RwLock<HashMap<usize, Vec<usize>>>,
    /// Allocated objects: addr -> slab_base
    allocated: RwLock<HashMap<usize, usize>>,
    /// Statistics
    stats: BuddyStats,
}

impl SlabAllocator {
    /// Create a new slab allocator for objects of the given size
    pub fn new(object_size: usize) -> Self {
        // Round up object size to minimum alignment
        let object_size = object_size.max(8).next_power_of_two();

        // Calculate objects per slab (aim for ~4KB slabs)
        let slab_size = 4096usize;
        let objects_per_slab = slab_size / object_size;

        Self {
            object_size,
            objects_per_slab: objects_per_slab.max(1),
            buddy: BuddyAllocator::new(),
            slabs: RwLock::new(HashMap::new()),
            allocated: RwLock::new(HashMap::new()),
            stats: BuddyStats::new(),
        }
    }

    /// Allocate one object
    pub fn allocate(&self) -> Result<usize, BuddyError> {
        // Try to find a slab with free objects
        {
            let mut slabs = self.slabs.write();
            for (base, free_list) in slabs.iter_mut() {
                if let Some(offset) = free_list.pop() {
                    let addr = base + offset;
                    self.allocated.write().insert(addr, *base);
                    self.stats.record_allocation(self.object_size);
                    return Ok(addr);
                }
            }
        }

        // No free objects, allocate a new slab
        let slab_size = self.object_size * self.objects_per_slab;
        let slab_base = self.buddy.allocate(slab_size)?;

        // Initialize free list for new slab (skip first object, return it)
        let mut free_list = Vec::with_capacity(self.objects_per_slab - 1);
        for i in 1..self.objects_per_slab {
            free_list.push(i * self.object_size);
        }

        self.slabs.write().insert(slab_base, free_list);
        self.allocated.write().insert(slab_base, slab_base);
        self.stats.record_allocation(self.object_size);

        Ok(slab_base)
    }

    /// Deallocate an object
    pub fn deallocate(&self, addr: usize) -> Result<(), BuddyError> {
        let slab_base = self
            .allocated
            .write()
            .remove(&addr)
            .ok_or(BuddyError::InvalidAddress(addr))?;

        let offset = addr - slab_base;
        self.slabs
            .write()
            .get_mut(&slab_base)
            .ok_or(BuddyError::BlockNotFound(slab_base))?
            .push(offset);

        self.stats.record_deallocation(self.object_size);

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> &BuddyStats {
        &self.stats
    }

    /// Get object size
    pub fn object_size(&self) -> usize {
        self.object_size
    }
}

/// Arena allocator using buddy allocator for backing storage
pub struct BuddyArena {
    /// Underlying buddy allocator
    buddy: BuddyAllocator,
    /// Current arena block
    current_block: Mutex<Option<ArenaBlock>>,
    /// Block size
    block_size: usize,
    /// All allocated block addresses for cleanup
    blocks: RwLock<Vec<usize>>,
}

struct ArenaBlock {
    base: usize,
    offset: usize,
    size: usize,
}

impl BuddyArena {
    /// Create a new arena with the specified block size
    pub fn new(block_size: usize) -> Self {
        let block_size = block_size.next_power_of_two();

        Self {
            buddy: BuddyAllocator::new(),
            current_block: Mutex::new(None),
            block_size,
            blocks: RwLock::new(Vec::new()),
        }
    }

    /// Allocate from the arena (bump allocation)
    pub fn allocate(&self, size: usize, align: usize) -> Result<usize, BuddyError> {
        if size == 0 {
            return Err(BuddyError::ZeroSize);
        }

        let mut current = self.current_block.lock();

        // Try to allocate from current block
        if let Some(ref mut block) = *current {
            let aligned_offset = (block.offset + align - 1) & !(align - 1);
            if aligned_offset + size <= block.size {
                block.offset = aligned_offset + size;
                return Ok(block.base + aligned_offset);
            }
        }

        // Need a new block
        let new_size = size.max(self.block_size).next_power_of_two();
        let base = self.buddy.allocate(new_size)?;

        self.blocks.write().push(base);

        let aligned_offset = (align - 1) & !(align - 1);
        *current = Some(ArenaBlock {
            base,
            offset: aligned_offset + size,
            size: new_size,
        });

        Ok(base + aligned_offset)
    }

    /// Allocate a value and return its address
    pub fn allocate_val<T>(&self, _val: T) -> Result<usize, BuddyError> {
        self.allocate(std::mem::size_of::<T>(), std::mem::align_of::<T>())
    }

    /// Reset the arena (free all blocks)
    pub fn reset(&self) {
        let mut current = self.current_block.lock();
        *current = None;

        let blocks = std::mem::take(&mut *self.blocks.write());
        for block in blocks {
            let _ = self.buddy.deallocate(block);
        }
    }

    /// Get total allocated size
    pub fn allocated_size(&self) -> usize {
        self.blocks.read().len() * self.block_size
    }
}

impl Drop for BuddyArena {
    fn drop(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_to_order() {
        assert_eq!(size_to_order(1), 4); // min = 16
        assert_eq!(size_to_order(16), 4); // 2^4 = 16
        assert_eq!(size_to_order(17), 5); // needs 2^5 = 32
        assert_eq!(size_to_order(32), 5);
        assert_eq!(size_to_order(64), 6);
        assert_eq!(size_to_order(1024), 10);
        assert_eq!(size_to_order(1025), 11); // needs 2048
    }

    #[test]
    fn test_order_to_size() {
        assert_eq!(order_to_size(4), 16);
        assert_eq!(order_to_size(5), 32);
        assert_eq!(order_to_size(10), 1024);
        assert_eq!(order_to_size(20), 1 << 20);
    }

    #[test]
    fn test_buddy_addr() {
        // For order 4 (size 16):
        // addr 0 -> buddy 16, addr 16 -> buddy 0
        assert_eq!(buddy_addr(0, 4), 16);
        assert_eq!(buddy_addr(16, 4), 0);

        // For order 5 (size 32):
        assert_eq!(buddy_addr(0, 5), 32);
        assert_eq!(buddy_addr(32, 5), 0);
        assert_eq!(buddy_addr(64, 5), 96);
        assert_eq!(buddy_addr(96, 5), 64);
    }

    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new(0, 1024).unwrap();

        // Allocate 16 bytes (min size)
        let addr1 = pool.allocate(16).unwrap();
        assert!(pool.contains(addr1));

        // Allocate another 16 bytes
        let addr2 = pool.allocate(16).unwrap();
        assert_ne!(addr1, addr2);

        // Free both
        pool.deallocate(addr1).unwrap();
        pool.deallocate(addr2).unwrap();
    }

    #[test]
    fn test_memory_pool_splitting() {
        let pool = MemoryPool::new(0, 256).unwrap();

        // Pool starts with one 256-byte block
        // Allocating 16 bytes should split multiple times
        let addr = pool.allocate(16).unwrap();

        // Check that splits occurred
        assert!(pool.stats().splits.load(Ordering::Relaxed) > 0);

        pool.deallocate(addr).unwrap();
    }

    #[test]
    fn test_memory_pool_merging() {
        let pool = MemoryPool::new(0, 256).unwrap();

        // Allocate two adjacent 16-byte blocks
        let addr1 = pool.allocate(16).unwrap();
        let addr2 = pool.allocate(16).unwrap();

        // Free both - should trigger merging
        pool.deallocate(addr1).unwrap();
        pool.deallocate(addr2).unwrap();

        // Check that merges occurred
        assert!(pool.stats().merges.load(Ordering::Relaxed) > 0);

        // Should be able to allocate the full pool again
        let addr3 = pool.allocate(256).unwrap();
        assert_eq!(addr3, 0);
    }

    #[test]
    fn test_memory_pool_out_of_memory() {
        let pool = MemoryPool::new(0, 256).unwrap();

        // Fill up the pool
        let mut addrs = Vec::new();
        for _ in 0..16 {
            addrs.push(pool.allocate(16).unwrap());
        }

        // Should fail now
        assert!(matches!(pool.allocate(16), Err(BuddyError::OutOfMemory)));

        // Free one and try again
        pool.deallocate(addrs.pop().unwrap()).unwrap();
        assert!(pool.allocate(16).is_ok());
    }

    #[test]
    fn test_buddy_allocator_basic() {
        let alloc = BuddyAllocator::new();

        let addr1 = alloc.allocate(100).unwrap();
        let addr2 = alloc.allocate(200).unwrap();

        assert_ne!(addr1, addr2);

        alloc.deallocate(addr1).unwrap();
        alloc.deallocate(addr2).unwrap();
    }

    #[test]
    fn test_buddy_allocator_auto_pool() {
        let alloc = BuddyAllocator::with_pool_size(1024);

        assert_eq!(alloc.pool_count(), 0);

        let _ = alloc.allocate(16).unwrap();

        assert_eq!(alloc.pool_count(), 1);
    }

    #[test]
    fn test_buddy_allocator_multiple_pools() {
        let alloc = BuddyAllocator::with_pool_size(256);

        // Allocate more than one pool can hold
        let mut addrs = Vec::new();
        for _ in 0..32 {
            addrs.push(alloc.allocate(16).unwrap());
        }

        // Should have multiple pools now
        assert!(alloc.pool_count() > 1);

        // Free all
        for addr in addrs {
            alloc.deallocate(addr).unwrap();
        }
    }

    #[test]
    fn test_typed_allocator() {
        #[repr(C)]
        struct MyStruct {
            a: u64,
            b: u32,
            c: u16,
        }

        let alloc = TypedBuddyAllocator::<MyStruct>::new();

        let addr1 = alloc.allocate_one().unwrap();
        let addr2 = alloc.allocate_array(10).unwrap();

        assert_ne!(addr1, addr2);

        alloc.deallocate(addr1).unwrap();
        alloc.deallocate(addr2).unwrap();
    }

    #[test]
    fn test_slab_allocator() {
        let slab = SlabAllocator::new(32);

        // Allocate several objects
        let mut addrs = Vec::new();
        for _ in 0..10 {
            addrs.push(slab.allocate().unwrap());
        }

        // All addresses should be unique
        for i in 0..addrs.len() {
            for j in (i + 1)..addrs.len() {
                assert_ne!(addrs[i], addrs[j]);
            }
        }

        // Free all
        for addr in addrs {
            slab.deallocate(addr).unwrap();
        }
    }

    #[test]
    fn test_slab_reuse() {
        let slab = SlabAllocator::new(64);

        let addr1 = slab.allocate().unwrap();
        slab.deallocate(addr1).unwrap();

        let addr2 = slab.allocate().unwrap();

        // Should reuse the same address
        assert_eq!(addr1, addr2);
    }

    #[test]
    fn test_buddy_arena() {
        let arena = BuddyArena::new(4096);

        // Allocate several items
        let addr1 = arena.allocate(100, 8).unwrap();
        let addr2 = arena.allocate(200, 16).unwrap();
        let addr3 = arena.allocate(50, 4).unwrap();

        // All unique addresses
        assert_ne!(addr1, addr2);
        assert_ne!(addr2, addr3);
        assert_ne!(addr1, addr3);

        // Check alignment
        assert_eq!(addr2 % 16, 0);

        // Reset
        arena.reset();

        // Can allocate again
        let _ = arena.allocate(100, 8).unwrap();
    }

    #[test]
    fn test_arena_large_allocation() {
        let arena = BuddyArena::new(256);

        // Allocate more than block size
        let result = arena.allocate(1024, 8);
        // Should succeed
        assert!(result.is_ok(), "Allocation failed: {:?}", result);
        let addr = result.unwrap();
        // Arena uses bump allocation from buddy allocator pool
        // The buddy allocator starts at base 0x1000
        println!("Arena allocated address: {:#x}", addr);
    }

    #[test]
    fn test_stats_tracking() {
        let pool = MemoryPool::new(0, 1024).unwrap();

        let addr = pool.allocate(64).unwrap();

        assert!(pool.stats().allocations.load(Ordering::Relaxed) > 0);
        assert!(pool.stats().allocated_bytes.load(Ordering::Relaxed) > 0);

        pool.deallocate(addr).unwrap();

        assert!(pool.stats().deallocations.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_free_bytes_tracking() {
        let pool = MemoryPool::new(0, 1024).unwrap();

        let initial_free = pool.free_bytes();
        assert_eq!(initial_free, 1024);

        let addr = pool.allocate(64).unwrap();

        let after_alloc = pool.free_bytes();
        assert!(after_alloc < initial_free);

        pool.deallocate(addr).unwrap();

        let after_free = pool.free_bytes();
        assert_eq!(after_free, initial_free);
    }

    #[test]
    fn test_double_free() {
        let pool = MemoryPool::new(0, 1024).unwrap();

        let addr = pool.allocate(64).unwrap();
        pool.deallocate(addr).unwrap();

        // Double free should fail
        assert!(matches!(
            pool.deallocate(addr),
            Err(BuddyError::InvalidAddress(_))
        ));
    }

    #[test]
    fn test_invalid_address() {
        let pool = MemoryPool::new(0, 1024).unwrap();

        // Try to free an address that was never allocated
        assert!(matches!(
            pool.deallocate(999),
            Err(BuddyError::InvalidAddress(_))
        ));
    }

    #[test]
    fn test_concurrent_allocations() {
        use std::sync::Arc;
        use std::thread;

        let alloc = Arc::new(BuddyAllocator::new());
        let mut handles = Vec::new();

        // Use fewer threads and allocations to reduce complexity
        for _ in 0..2 {
            let alloc = Arc::clone(&alloc);
            handles.push(thread::spawn(move || {
                let mut addrs = Vec::new();
                for _ in 0..10 {
                    if let Ok(addr) = alloc.allocate(32) {
                        addrs.push(addr);
                    }
                }
                // Add a small delay between alloc and dealloc phases
                std::thread::sleep(std::time::Duration::from_millis(1));
                for addr in addrs {
                    let _ = alloc.deallocate(addr);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
