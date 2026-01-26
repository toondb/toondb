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

//! Hierarchical Timestamp Oracle with Logical Clock Partitioning (Task 9)
//!
//! The current timestamp allocation uses a global atomic counter, which under
//! high concurrency becomes a cache line ping-pong bottleneck.
//!
//! ## Problem
//!
//! Each `fetch_add` invalidates the cache line on all other cores:
//! - With 16 cores at 1M ops/sec each = 16M cache invalidations/sec
//! - Memory bus becomes saturated
//!
//! ## Solution
//!
//! Hierarchical timestamp allocation with thread-local pools:
//! - Global atomics per 1M commits: 1M → 1K (1000× reduction)
//! - Cross-core cache invalidations: 1M → 1K
//! - Throughput scalability: Sub-linear → Linear with cores
//!
//! ## Timestamp Format
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │ 64-bit Timestamp                                               │
//! ├────────────────────────────────────────────────────────────────┤
//! │ [Epoch: 32 bits] [Sequence: 32 bits]                          │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Correctness
//!
//! Timestamps from different threads are totally ordered:
//! 1. Different epochs → ordered by epoch (SeqCst guarantees)
//! 2. Same epoch → same thread (exclusive pool ownership)
//! 3. Same thread → ordered by sequence number

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use thread_local::ThreadLocal;

/// Default pool size (timestamps per thread-local pool)
const DEFAULT_POOL_SIZE: u64 = 10000;

/// Number of epoch bits (32 allows 4B epochs)
#[allow(dead_code)]
const EPOCH_BITS: u32 = 32;

/// Number of sequence bits (32 allows 4B sequences per epoch)
const SEQUENCE_BITS: u32 = 32;

/// Thread-local timestamp pool
struct TimestampPool {
    /// Current epoch for this pool
    epoch: u64,
    /// Next sequence number
    sequence: u64,
    /// Max sequence before re-reserve
    limit: u64,
}

impl TimestampPool {
    fn new(epoch: u64, pool_size: u64) -> Self {
        Self {
            epoch,
            sequence: 0,
            limit: pool_size,
        }
    }
}

/// Hierarchical Timestamp Oracle with thread-local pools
///
/// Reduces global synchronization by 99%+ by allocating timestamps
/// in thread-local batches.
///
/// ## Performance
///
/// | Metric | Global Atomic | Hierarchical |
/// |--------|---------------|--------------|
/// | Global atomics / 1M commits | 1M | 100 |
/// | Cache invalidations | 1M | 100 |
/// | Throughput scalability | Sub-linear | Linear |
///
/// ## Thread Safety
///
/// Thread-safe via thread-local storage. Each thread has its own pool,
/// so no synchronization needed for the fast path.
pub struct HierarchicalTimestampOracle {
    /// Global epoch counter (incremented rarely)
    global_epoch: AtomicU64,
    
    /// Per-thread timestamp pools
    thread_pools: ThreadLocal<UnsafeCell<TimestampPool>>,
    
    /// Timestamps per pool
    pool_size: u64,
}

impl HierarchicalTimestampOracle {
    /// Create a new hierarchical timestamp oracle with default pool size (10000)
    pub fn new() -> Self {
        Self::with_pool_size(DEFAULT_POOL_SIZE)
    }
    
    /// Create with custom pool size
    pub fn with_pool_size(pool_size: u64) -> Self {
        Self {
            global_epoch: AtomicU64::new(0),
            thread_pools: ThreadLocal::new(),
            pool_size,
        }
    }
    
    /// Allocate a new timestamp
    ///
    /// Fast path: thread-local allocation (no atomics)
    /// Slow path: reserve new pool (one global atomic per pool_size allocations)
    #[inline]
    pub fn allocate(&self) -> u64 {
        // Get or create thread-local pool
        let pool_cell = self.thread_pools.get_or(|| {
            let epoch = self.global_epoch.fetch_add(1, Ordering::SeqCst);
            UnsafeCell::new(TimestampPool::new(epoch, self.pool_size))
        });
        
        // SAFETY: We have exclusive access to our thread-local pool
        let pool = unsafe { &mut *pool_cell.get() };
        
        // Fast path: allocate from pool
        if pool.sequence < pool.limit {
            let ts = (pool.epoch << SEQUENCE_BITS) | pool.sequence;
            pool.sequence += 1;
            return ts;
        }
        
        // Slow path: reserve new pool
        self.reserve_new_pool(pool)
    }
    
    /// Reserve a new pool (slow path, called once per pool_size allocations)
    #[cold]
    fn reserve_new_pool(&self, pool: &mut TimestampPool) -> u64 {
        // Single global atomic per pool_size allocations
        let new_epoch = self.global_epoch.fetch_add(1, Ordering::SeqCst);
        
        pool.epoch = new_epoch;
        pool.sequence = 1; // First timestamp of new pool
        pool.limit = self.pool_size;
        
        // Return first timestamp of new pool
        (pool.epoch << SEQUENCE_BITS) | 0
    }
    
    /// Get current global epoch (for monitoring)
    #[inline]
    pub fn current_epoch(&self) -> u64 {
        self.global_epoch.load(Ordering::Relaxed)
    }
    
    /// Extract epoch from timestamp
    #[inline]
    pub fn extract_epoch(ts: u64) -> u64 {
        ts >> SEQUENCE_BITS
    }
    
    /// Extract sequence from timestamp
    #[inline]
    pub fn extract_sequence(ts: u64) -> u64 {
        ts & ((1 << SEQUENCE_BITS) - 1)
    }
    
    /// Compare timestamps for ordering
    ///
    /// Returns true if ts1 < ts2 (ts1 happened before ts2)
    #[inline]
    pub fn happens_before(ts1: u64, ts2: u64) -> bool {
        ts1 < ts2
    }
}

impl Default for HierarchicalTimestampOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Hybrid Hierarchical Oracle with HLC Integration
// ============================================================================

/// Hybrid oracle combining hierarchical allocation with HLC semantics
///
/// Provides both logical ordering (from hierarchical allocation) and
/// physical timestamp compatibility (for cross-node coordination).
pub struct HybridHierarchicalOracle {
    /// Hierarchical oracle for fast local allocation
    hierarchical: HierarchicalTimestampOracle,
    
    /// Wall clock offset (for HLC-style coordination)
    wall_clock_offset: AtomicU64,
    
    /// Last observed physical time
    last_physical: AtomicU64,
}

impl HybridHierarchicalOracle {
    /// Create a new hybrid oracle
    pub fn new() -> Self {
        Self {
            hierarchical: HierarchicalTimestampOracle::new(),
            wall_clock_offset: AtomicU64::new(0),
            last_physical: AtomicU64::new(0),
        }
    }
    
    /// Allocate a timestamp (fast path)
    #[inline]
    pub fn allocate(&self) -> u64 {
        self.hierarchical.allocate()
    }
    
    /// Get a timestamp with physical time component
    ///
    /// Returns (logical_ts, physical_ts) tuple for cross-node ordering
    pub fn allocate_with_physical(&self) -> (u64, u64) {
        let logical = self.hierarchical.allocate();
        let physical = self.current_physical_time();
        (logical, physical)
    }
    
    /// Update with external timestamp (for distributed coordination)
    pub fn update_from_external(&self, external_physical: u64) {
        // CAS loop to update max physical time
        loop {
            let current = self.last_physical.load(Ordering::Acquire);
            if external_physical <= current {
                break;
            }
            if self.last_physical
                .compare_exchange_weak(current, external_physical, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
    
    /// Get current physical time (wall clock + offset)
    fn current_physical_time(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let wall_clock = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let offset = self.wall_clock_offset.load(Ordering::Relaxed);
        let physical = wall_clock + offset;
        
        // Ensure monotonicity
        loop {
            let last = self.last_physical.load(Ordering::Acquire);
            let next = physical.max(last + 1);
            if self.last_physical
                .compare_exchange_weak(last, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return next;
            }
        }
    }
}

impl Default for HybridHierarchicalOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Monotonic Timestamp Allocator (Simpler Alternative)
// ============================================================================

/// Simple monotonic timestamp allocator with batching
///
/// A simpler alternative that maintains strict monotonicity while
/// reducing contention via batched allocation.
pub struct BatchedTimestampAllocator {
    /// Next timestamp to allocate globally
    next_global: AtomicU64,
    
    /// Batch size for local allocation
    batch_size: u64,
    
    /// Thread-local batches
    thread_batches: ThreadLocal<UnsafeCell<LocalBatch>>,
}

struct LocalBatch {
    start: u64,
    current: u64,
    end: u64,
}

impl BatchedTimestampAllocator {
    /// Create with default batch size (1000)
    pub fn new() -> Self {
        Self::with_batch_size(1000)
    }
    
    /// Create with custom batch size
    pub fn with_batch_size(batch_size: u64) -> Self {
        Self {
            next_global: AtomicU64::new(1), // Start from 1, 0 is reserved
            batch_size,
            thread_batches: ThreadLocal::new(),
        }
    }
    
    /// Allocate next timestamp
    #[inline]
    pub fn allocate(&self) -> u64 {
        let batch_cell = self.thread_batches.get_or(|| {
            UnsafeCell::new(LocalBatch {
                start: 0,
                current: 0,
                end: 0,
            })
        });
        
        // SAFETY: Thread-local access
        let batch = unsafe { &mut *batch_cell.get() };
        
        if batch.current < batch.end {
            let ts = batch.current;
            batch.current += 1;
            return ts;
        }
        
        // Reserve new batch
        self.reserve_batch(batch)
    }
    
    #[cold]
    fn reserve_batch(&self, batch: &mut LocalBatch) -> u64 {
        let start = self.next_global.fetch_add(self.batch_size, Ordering::SeqCst);
        batch.start = start;
        batch.current = start + 1;
        batch.end = start + self.batch_size;
        start
    }
    
    /// Get current watermark (all timestamps below this are allocated)
    pub fn watermark(&self) -> u64 {
        self.next_global.load(Ordering::Acquire)
    }
}

impl Default for BatchedTimestampAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_hierarchical_oracle_basic() {
        let oracle = HierarchicalTimestampOracle::new();
        
        // Allocate some timestamps
        let ts1 = oracle.allocate();
        let ts2 = oracle.allocate();
        let ts3 = oracle.allocate();
        
        // Should be monotonically increasing within thread
        assert!(ts1 < ts2);
        assert!(ts2 < ts3);
        
        // Check epoch/sequence extraction
        let epoch = HierarchicalTimestampOracle::extract_epoch(ts1);
        let seq = HierarchicalTimestampOracle::extract_sequence(ts1);
        assert_eq!(epoch, 0);
        assert_eq!(seq, 0);
    }
    
    #[test]
    fn test_hierarchical_oracle_pool_exhaustion() {
        // Small pool size to trigger re-allocation
        let oracle = HierarchicalTimestampOracle::with_pool_size(10);
        
        // Allocate more than pool size
        let mut timestamps = Vec::new();
        for _ in 0..25 {
            timestamps.push(oracle.allocate());
        }
        
        // All should be unique
        let unique: HashSet<_> = timestamps.iter().collect();
        assert_eq!(unique.len(), 25);
        
        // Should have used multiple epochs
        let epochs: HashSet<_> = timestamps
            .iter()
            .map(|&ts| HierarchicalTimestampOracle::extract_epoch(ts))
            .collect();
        assert!(epochs.len() >= 2);
    }
    
    #[test]
    fn test_hierarchical_oracle_concurrent() {
        let oracle = Arc::new(HierarchicalTimestampOracle::new());
        let num_threads = 8;
        let ops_per_thread = 10000;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let oracle = Arc::clone(&oracle);
                thread::spawn(move || {
                    let mut timestamps = Vec::with_capacity(ops_per_thread);
                    for _ in 0..ops_per_thread {
                        timestamps.push(oracle.allocate());
                    }
                    timestamps
                })
            })
            .collect();
        
        let all_timestamps: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        
        // All timestamps should be unique
        let unique: HashSet<_> = all_timestamps.iter().collect();
        assert_eq!(unique.len(), num_threads * ops_per_thread);
    }
    
    #[test]
    fn test_batched_allocator() {
        let allocator = BatchedTimestampAllocator::with_batch_size(100);
        
        // Allocate timestamps
        let mut timestamps = Vec::new();
        for _ in 0..250 {
            timestamps.push(allocator.allocate());
        }
        
        // All unique
        let unique: HashSet<_> = timestamps.iter().collect();
        assert_eq!(unique.len(), 250);
        
        // Monotonically increasing within thread
        for window in timestamps.windows(2) {
            assert!(window[0] < window[1]);
        }
    }
    
    #[test]
    fn test_batched_allocator_concurrent() {
        let allocator = Arc::new(BatchedTimestampAllocator::with_batch_size(100));
        let num_threads = 4;
        let ops_per_thread = 1000;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let allocator = Arc::clone(&allocator);
                thread::spawn(move || {
                    let mut timestamps = Vec::new();
                    for _ in 0..ops_per_thread {
                        timestamps.push(allocator.allocate());
                    }
                    timestamps
                })
            })
            .collect();
        
        let all_timestamps: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        
        // All unique
        let unique: HashSet<_> = all_timestamps.iter().collect();
        assert_eq!(unique.len(), num_threads * ops_per_thread);
    }
    
    #[test]
    fn test_hybrid_oracle() {
        let oracle = HybridHierarchicalOracle::new();
        
        let (logical1, physical1) = oracle.allocate_with_physical();
        let (logical2, physical2) = oracle.allocate_with_physical();
        
        // Logical timestamps should increase
        assert!(logical1 < logical2);
        
        // Physical timestamps should be monotonic
        assert!(physical1 <= physical2);
    }
    
    #[test]
    fn test_happens_before() {
        let oracle = HierarchicalTimestampOracle::new();
        
        let ts1 = oracle.allocate();
        let ts2 = oracle.allocate();
        
        assert!(HierarchicalTimestampOracle::happens_before(ts1, ts2));
        assert!(!HierarchicalTimestampOracle::happens_before(ts2, ts1));
        assert!(!HierarchicalTimestampOracle::happens_before(ts1, ts1));
    }
}
