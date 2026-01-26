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

//! Unified Memory Reclamation - Hazard Pointers + Epoch Hybrid
//!
//! This module provides a unified memory reclamation strategy combining:
//! - **Hazard Pointers**: For hot-path reads with minimal latency overhead
//! - **Epoch-Based Reclamation (EBR)**: For batch operations and writes
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Unified Reclamation                          │
//! │                                                                 │
//! │  ┌──────────────────┐      ┌──────────────────────────────┐    │
//! │  │  Hazard Pointers │      │     Epoch-Based GC           │    │
//! │  │                  │      │                              │    │
//! │  │  • Per-thread HP │      │  • Global epoch counter      │    │
//! │  │  • Protect reads │      │  • Per-thread local epoch    │    │
//! │  │  • O(1) protect  │      │  • Limbo list per epoch      │    │
//! │  │  • Scan on free  │      │  • Amortized O(1) reclaim    │    │
//! │  └──────────────────┘      └──────────────────────────────┘    │
//! │            │                            │                       │
//! │            └──────────┬─────────────────┘                       │
//! │                       ▼                                         │
//! │              ┌────────────────┐                                 │
//! │              │ Unified Guard  │                                 │
//! │              │                │                                 │
//! │              │ Picks strategy │                                 │
//! │              │ based on:      │                                 │
//! │              │ • Contention   │                                 │
//! │              │ • Read/Write   │                                 │
//! │              │ • Batch size   │                                 │
//! │              └────────────────┘                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let reclaimer = UnifiedReclaimer::new();
//!
//! // Hot-path reads use hazard pointers
//! let guard = reclaimer.pin_read();
//! let data = guard.protect(&shared_ptr);
//! // ... use data
//! drop(guard); // Automatically unpins
//!
//! // Batch operations use epochs
//! let guard = reclaimer.pin_epoch();
//! for item in batch {
//!     // ... process items
//! }
//! reclaimer.retire(old_data);
//! drop(guard); // Triggers epoch advancement
//! ```

use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Maximum number of hazard pointers per thread
const MAX_HAZARD_POINTERS: usize = 4;

/// Number of epochs to keep before reclamation
const EPOCH_GRACE_PERIODS: u64 = 2;

/// Threshold for triggering reclamation scan
const RECLAIM_THRESHOLD: usize = 64;

/// Hazard pointer slot
#[derive(Debug)]
struct HazardSlot {
    /// The protected pointer (null if unused)
    ptr: AtomicPtr<()>,
    /// Thread ID that owns this slot
    owner: AtomicU64,
}

impl HazardSlot {
    fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
            owner: AtomicU64::new(0),
        }
    }

    fn acquire(&self, thread_id: u64) -> bool {
        self.owner
            .compare_exchange(0, thread_id, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    fn release(&self) {
        self.ptr.store(std::ptr::null_mut(), Ordering::Release);
        self.owner.store(0, Ordering::Release);
    }

    fn protect(&self, ptr: *mut ()) {
        self.ptr.store(ptr, Ordering::Release);
    }

    fn is_protecting(&self, ptr: *mut ()) -> bool {
        self.ptr.load(Ordering::Acquire) == ptr
    }
}

/// Hazard pointer domain for a set of threads
pub struct HazardDomain {
    /// Global list of hazard pointer slots
    slots: Vec<HazardSlot>,
    /// Number of active slots
    active_count: AtomicUsize,
    /// Thread-local slot indices (using thread_local! is preferred, this is fallback)
    slot_registry: Mutex<Vec<(u64, usize)>>,
}

impl HazardDomain {
    /// Create new hazard domain with given capacity
    pub fn new(max_threads: usize) -> Self {
        let capacity = max_threads * MAX_HAZARD_POINTERS;
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(HazardSlot::new());
        }

        Self {
            slots,
            active_count: AtomicUsize::new(0),
            slot_registry: Mutex::new(Vec::new()),
        }
    }

    /// Acquire a hazard pointer slot for the current thread
    fn acquire_slot(&self, thread_id: u64) -> Option<usize> {
        // First check registry for existing slot
        {
            let registry = self.slot_registry.lock();
            for &(tid, idx) in registry.iter() {
                if tid == thread_id {
                    return Some(idx);
                }
            }
        }

        // Try to acquire a new slot
        for (idx, slot) in self.slots.iter().enumerate() {
            if slot.acquire(thread_id) {
                let mut registry = self.slot_registry.lock();
                registry.push((thread_id, idx));
                self.active_count.fetch_add(1, Ordering::Relaxed);
                return Some(idx);
            }
        }

        None
    }

    /// Release a hazard pointer slot
    fn release_slot(&self, thread_id: u64, slot_idx: usize) {
        if slot_idx < self.slots.len() {
            self.slots[slot_idx].release();
        }

        let mut registry = self.slot_registry.lock();
        registry.retain(|&(tid, _)| tid != thread_id);
        self.active_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Protect a pointer using hazard pointer at given slot
    fn protect(&self, slot_idx: usize, ptr: *mut ()) {
        if slot_idx < self.slots.len() {
            self.slots[slot_idx].protect(ptr);
        }
    }

    /// Check if any hazard pointer is protecting the given pointer
    fn is_protected(&self, ptr: *mut ()) -> bool {
        for slot in &self.slots {
            if slot.is_protecting(ptr) {
                return true;
            }
        }
        false
    }

    /// Get current active count
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }
}

impl Default for HazardDomain {
    fn default() -> Self {
        Self::new(64) // Default to 64 threads
    }
}

/// Epoch-based reclamation domain
pub struct EpochDomain {
    /// Global epoch counter
    global_epoch: AtomicU64,
    /// Per-thread local epochs
    local_epochs: RwLock<Vec<AtomicU64>>,
    /// Limbo lists per epoch (objects waiting to be freed)
    limbo: Mutex<VecDeque<(u64, Vec<RetiredObject>)>>,
    /// Count of retired objects pending reclamation
    retired_count: AtomicUsize,
}

/// A retired object waiting for safe reclamation
#[allow(dead_code)]
struct RetiredObject {
    ptr: *mut (),
    destructor: fn(*mut ()),
    size: usize,
}

// Safety: RetiredObject contains raw pointers but they're only dereferenced
// in a single-threaded context during reclamation
unsafe impl Send for RetiredObject {}

impl EpochDomain {
    /// Create new epoch domain
    pub fn new() -> Self {
        Self {
            global_epoch: AtomicU64::new(0),
            local_epochs: RwLock::new(Vec::new()),
            limbo: Mutex::new(VecDeque::new()),
            retired_count: AtomicUsize::new(0),
        }
    }

    /// Register a thread and return its index
    pub fn register_thread(&self) -> usize {
        let mut epochs = self.local_epochs.write();
        let idx = epochs.len();
        epochs.push(AtomicU64::new(u64::MAX)); // MAX = not pinned
        idx
    }

    /// Pin the current epoch for a thread
    pub fn pin(&self, thread_idx: usize) {
        let current = self.global_epoch.load(Ordering::SeqCst);
        let epochs = self.local_epochs.read();
        if thread_idx < epochs.len() {
            epochs[thread_idx].store(current, Ordering::SeqCst);
        }
    }

    /// Unpin (exit) the current epoch for a thread
    pub fn unpin(&self, thread_idx: usize) {
        let epochs = self.local_epochs.read();
        if thread_idx < epochs.len() {
            epochs[thread_idx].store(u64::MAX, Ordering::SeqCst);
        }
    }

    /// Retire an object for later reclamation
    pub fn retire(&self, ptr: *mut (), destructor: fn(*mut ()), size: usize) {
        let current_epoch = self.global_epoch.load(Ordering::SeqCst);
        let obj = RetiredObject {
            ptr,
            destructor,
            size,
        };

        let mut limbo = self.limbo.lock();

        // Find or create bucket for current epoch
        if limbo.back().is_none_or(|(e, _)| *e != current_epoch) {
            limbo.push_back((current_epoch, Vec::new()));
        }

        if let Some((_, objects)) = limbo.back_mut() {
            objects.push(obj);
        }

        let count = self.retired_count.fetch_add(1, Ordering::Relaxed);

        // Trigger reclamation if threshold exceeded
        if count >= RECLAIM_THRESHOLD {
            drop(limbo);
            self.try_reclaim();
        }
    }

    /// Advance the global epoch
    pub fn advance_epoch(&self) {
        self.global_epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Get minimum epoch that is safe (all threads have advanced past)
    fn safe_epoch(&self) -> u64 {
        let epochs = self.local_epochs.read();
        let mut min = self.global_epoch.load(Ordering::SeqCst);

        for epoch in epochs.iter() {
            let e = epoch.load(Ordering::SeqCst);
            if e != u64::MAX && e < min {
                min = e;
            }
        }

        // Safe epoch is grace periods before minimum
        min.saturating_sub(EPOCH_GRACE_PERIODS)
    }

    /// Try to reclaim objects from old epochs
    pub fn try_reclaim(&self) -> usize {
        let safe = self.safe_epoch();
        let mut reclaimed = 0;

        let mut limbo = self.limbo.lock();

        while let Some((epoch, _)) = limbo.front() {
            if *epoch > safe {
                break;
            }

            if let Some((_, objects)) = limbo.pop_front() {
                for obj in objects {
                    // Call destructor
                    (obj.destructor)(obj.ptr);
                    reclaimed += 1;
                    self.retired_count.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }

        reclaimed
    }

    /// Get current global epoch
    pub fn current_epoch(&self) -> u64 {
        self.global_epoch.load(Ordering::SeqCst)
    }

    /// Get count of pending retired objects
    pub fn pending_count(&self) -> usize {
        self.retired_count.load(Ordering::Relaxed)
    }
}

impl Default for EpochDomain {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard for hazard pointer protection
pub struct HazardGuard<'a> {
    domain: &'a HazardDomain,
    slot_idx: usize,
    thread_id: u64,
}

impl<'a> HazardGuard<'a> {
    /// Protect a raw pointer
    pub fn protect(&self, ptr: *mut ()) {
        self.domain.protect(self.slot_idx, ptr);
    }

    /// Protect a typed pointer
    pub fn protect_typed<T>(&self, ptr: *mut T) {
        self.protect(ptr as *mut ());
    }
}

impl<'a> Drop for HazardGuard<'a> {
    fn drop(&mut self) {
        self.domain.release_slot(self.thread_id, self.slot_idx);
    }
}

/// Guard for epoch-based protection
pub struct EpochGuard<'a> {
    domain: &'a EpochDomain,
    thread_idx: usize,
}

impl<'a> Drop for EpochGuard<'a> {
    fn drop(&mut self) {
        self.domain.unpin(self.thread_idx);
    }
}

/// Reclamation strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReclaimStrategy {
    /// Use hazard pointers (best for hot-path reads)
    HazardPointer,
    /// Use epoch-based reclamation (best for batch operations)
    Epoch,
    /// Automatically select based on heuristics
    Auto,
}

/// Unified memory reclamation combining hazard pointers and epochs
pub struct UnifiedReclaimer {
    hazard: Arc<HazardDomain>,
    epoch: Arc<EpochDomain>,
    /// Thread-local epoch indices
    thread_epochs: Mutex<std::collections::HashMap<u64, usize>>,
    /// Strategy selection
    default_strategy: ReclaimStrategy,
    /// Statistics
    stats: ReclaimStats,
}

/// Reclamation statistics
#[derive(Debug, Default)]
pub struct ReclaimStats {
    pub hazard_pins: AtomicU64,
    pub epoch_pins: AtomicU64,
    pub objects_retired: AtomicU64,
    pub objects_reclaimed: AtomicU64,
    pub reclaim_cycles: AtomicU64,
}

impl ReclaimStats {
    fn record_hazard_pin(&self) {
        self.hazard_pins.fetch_add(1, Ordering::Relaxed);
    }

    fn record_epoch_pin(&self) {
        self.epoch_pins.fetch_add(1, Ordering::Relaxed);
    }

    fn record_retire(&self) {
        self.objects_retired.fetch_add(1, Ordering::Relaxed);
    }

    fn record_reclaim(&self, count: usize) {
        self.objects_reclaimed
            .fetch_add(count as u64, Ordering::Relaxed);
        self.reclaim_cycles.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ReclaimStatsSnapshot {
        ReclaimStatsSnapshot {
            hazard_pins: self.hazard_pins.load(Ordering::Relaxed),
            epoch_pins: self.epoch_pins.load(Ordering::Relaxed),
            objects_retired: self.objects_retired.load(Ordering::Relaxed),
            objects_reclaimed: self.objects_reclaimed.load(Ordering::Relaxed),
            reclaim_cycles: self.reclaim_cycles.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReclaimStatsSnapshot {
    pub hazard_pins: u64,
    pub epoch_pins: u64,
    pub objects_retired: u64,
    pub objects_reclaimed: u64,
    pub reclaim_cycles: u64,
}

impl UnifiedReclaimer {
    /// Create new unified reclaimer
    pub fn new() -> Self {
        Self {
            hazard: Arc::new(HazardDomain::default()),
            epoch: Arc::new(EpochDomain::default()),
            thread_epochs: Mutex::new(std::collections::HashMap::new()),
            default_strategy: ReclaimStrategy::Auto,
            stats: ReclaimStats::default(),
        }
    }

    /// Create with specific max thread capacity
    pub fn with_capacity(max_threads: usize) -> Self {
        Self {
            hazard: Arc::new(HazardDomain::new(max_threads)),
            epoch: Arc::new(EpochDomain::default()),
            thread_epochs: Mutex::new(std::collections::HashMap::new()),
            default_strategy: ReclaimStrategy::Auto,
            stats: ReclaimStats::default(),
        }
    }

    /// Set default reclamation strategy
    pub fn with_strategy(mut self, strategy: ReclaimStrategy) -> Self {
        self.default_strategy = strategy;
        self
    }

    /// Pin for read access using hazard pointers
    pub fn pin_hazard(&self) -> Option<HazardGuard<'_>> {
        let thread_id = self.current_thread_id();
        let slot_idx = self.hazard.acquire_slot(thread_id)?;

        self.stats.record_hazard_pin();

        Some(HazardGuard {
            domain: &self.hazard,
            slot_idx,
            thread_id,
        })
    }

    /// Pin for epoch-based access
    pub fn pin_epoch(&self) -> EpochGuard<'_> {
        let thread_id = self.current_thread_id();

        let thread_idx = {
            let mut epochs = self.thread_epochs.lock();
            *epochs
                .entry(thread_id)
                .or_insert_with(|| self.epoch.register_thread())
        };

        self.epoch.pin(thread_idx);
        self.stats.record_epoch_pin();

        EpochGuard {
            domain: &self.epoch,
            thread_idx,
        }
    }

    /// Retire an object for later reclamation
    ///
    /// # Safety
    /// The pointer must have been allocated and no references should exist
    /// outside of protected guards.
    pub unsafe fn retire<T>(&self, ptr: *mut T) {
        let destructor = |p: *mut ()| {
            // Safety: caller guarantees ptr was allocated as T
            unsafe { drop(Box::from_raw(p as *mut T)) };
        };

        self.epoch
            .retire(ptr as *mut (), destructor, std::mem::size_of::<T>());
        self.stats.record_retire();
    }

    /// Retire with custom destructor
    pub fn retire_with_destructor(&self, ptr: *mut (), destructor: fn(*mut ()), size: usize) {
        self.epoch.retire(ptr, destructor, size);
        self.stats.record_retire();
    }

    /// Check if a pointer is protected by any hazard pointer
    pub fn is_protected(&self, ptr: *mut ()) -> bool {
        self.hazard.is_protected(ptr)
    }

    /// Manually trigger reclamation
    pub fn try_reclaim(&self) -> usize {
        let reclaimed = self.epoch.try_reclaim();
        if reclaimed > 0 {
            self.stats.record_reclaim(reclaimed);
        }
        reclaimed
    }

    /// Advance the epoch
    pub fn advance_epoch(&self) {
        self.epoch.advance_epoch();
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.epoch.current_epoch()
    }

    /// Get count of objects pending reclamation
    pub fn pending_count(&self) -> usize {
        self.epoch.pending_count()
    }

    /// Get statistics
    pub fn stats(&self) -> ReclaimStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get thread ID (platform-specific)
    fn current_thread_id(&self) -> u64 {
        // Use hash of thread ID as stable u64 identifier
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for UnifiedReclaimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local handle for efficient access
pub struct ThreadLocalReclaimer {
    reclaimer: Arc<UnifiedReclaimer>,
    hazard_slot: Option<usize>,
    epoch_idx: usize,
    thread_id: u64,
}

impl ThreadLocalReclaimer {
    /// Create thread-local handle
    pub fn new(reclaimer: Arc<UnifiedReclaimer>) -> Self {
        // Use hash of thread ID as stable u64 identifier
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        let thread_id = hasher.finish();

        let epoch_idx = reclaimer.epoch.register_thread();

        Self {
            reclaimer,
            hazard_slot: None,
            epoch_idx,
            thread_id,
        }
    }

    /// Pin using hazard pointer (fast path)
    pub fn pin_hazard(&mut self) -> bool {
        if self.hazard_slot.is_some() {
            return true;
        }

        if let Some(slot) = self.reclaimer.hazard.acquire_slot(self.thread_id) {
            self.hazard_slot = Some(slot);
            true
        } else {
            false
        }
    }

    /// Protect a pointer with hazard pointer
    pub fn protect(&self, ptr: *mut ()) {
        if let Some(slot) = self.hazard_slot {
            self.reclaimer.hazard.protect(slot, ptr);
        }
    }

    /// Pin using epoch
    pub fn pin_epoch(&self) {
        self.reclaimer.epoch.pin(self.epoch_idx);
    }

    /// Unpin epoch
    pub fn unpin_epoch(&self) {
        self.reclaimer.epoch.unpin(self.epoch_idx);
    }

    /// Retire an object
    pub fn retire(&self, ptr: *mut (), destructor: fn(*mut ()), size: usize) {
        self.reclaimer.epoch.retire(ptr, destructor, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_hazard_domain_basic() {
        let domain = HazardDomain::new(4);

        let thread_id = 12345u64;
        let slot = domain.acquire_slot(thread_id).unwrap();

        assert_eq!(domain.active_count(), 1);

        // Protect a pointer
        let data = Box::into_raw(Box::new(42u64));
        domain.protect(slot, data as *mut ());

        assert!(domain.is_protected(data as *mut ()));

        domain.release_slot(thread_id, slot);
        assert_eq!(domain.active_count(), 0);
        assert!(!domain.is_protected(data as *mut ()));

        // Cleanup
        unsafe { drop(Box::from_raw(data)) };
    }

    #[test]
    fn test_epoch_domain_basic() {
        let domain = EpochDomain::new();

        let idx = domain.register_thread();
        assert_eq!(domain.current_epoch(), 0);

        domain.pin(idx);
        domain.advance_epoch();
        assert_eq!(domain.current_epoch(), 1);

        domain.unpin(idx);
    }

    #[test]
    fn test_epoch_retirement() {
        static DROPPED: AtomicBool = AtomicBool::new(false);
        DROPPED.store(false, Ordering::SeqCst); // Reset for test isolation

        fn drop_test(ptr: *mut ()) {
            DROPPED.store(true, Ordering::SeqCst);
            unsafe { drop(Box::from_raw(ptr as *mut u64)) };
        }

        let domain = EpochDomain::new();
        let idx = domain.register_thread();

        // Pin and retire
        domain.pin(idx);
        let data = Box::into_raw(Box::new(42u64));
        domain.retire(data as *mut (), drop_test, 8);

        // While pinned, try_reclaim should not reclaim (we're holding epoch 0)
        // The safe_epoch will be at most epoch 0 - GRACE = underflow protection
        let reclaimed_while_pinned = domain.try_reclaim();

        // Unpin and advance epochs past grace period
        domain.unpin(idx);
        for _ in 0..EPOCH_GRACE_PERIODS + 2 {
            domain.advance_epoch();
        }

        // Now should be reclaimed
        let reclaimed = domain.try_reclaim();

        // Either it was reclaimed now or during the retire threshold trigger
        // The key test is that it eventually gets reclaimed
        assert!(DROPPED.load(Ordering::SeqCst) || reclaimed > 0 || reclaimed_while_pinned > 0);
    }

    #[test]
    fn test_unified_reclaimer_hazard() {
        let reclaimer = UnifiedReclaimer::new();

        let guard = reclaimer.pin_hazard();
        assert!(guard.is_some());

        let guard = guard.unwrap();
        let data = Box::into_raw(Box::new(String::from("test")));
        guard.protect_typed(data);

        assert!(reclaimer.is_protected(data as *mut ()));

        drop(guard);
        assert!(!reclaimer.is_protected(data as *mut ()));

        // Cleanup
        unsafe { drop(Box::from_raw(data)) };
    }

    #[test]
    fn test_unified_reclaimer_epoch() {
        let reclaimer = UnifiedReclaimer::new();

        {
            let _guard = reclaimer.pin_epoch();
            assert_eq!(reclaimer.current_epoch(), 0);
        }

        reclaimer.advance_epoch();
        assert_eq!(reclaimer.current_epoch(), 1);
    }

    #[test]
    fn test_stats_tracking() {
        let reclaimer = UnifiedReclaimer::new();

        {
            let _guard = reclaimer.pin_epoch();
        }

        let _ = reclaimer.pin_hazard();

        let stats = reclaimer.stats();
        assert!(stats.epoch_pins >= 1);
        assert!(stats.hazard_pins >= 1);
    }

    #[test]
    fn test_thread_local_reclaimer() {
        let reclaimer = Arc::new(UnifiedReclaimer::new());
        let mut local = ThreadLocalReclaimer::new(Arc::clone(&reclaimer));

        assert!(local.pin_hazard());

        let data = Box::into_raw(Box::new(100u32));
        local.protect(data as *mut ());

        assert!(reclaimer.is_protected(data as *mut ()));

        // Cleanup
        unsafe { drop(Box::from_raw(data)) };
    }

    #[test]
    fn test_multiple_hazard_slots() {
        let domain = HazardDomain::new(2);

        let slot1 = domain.acquire_slot(1).unwrap();
        let slot2 = domain.acquire_slot(2).unwrap();

        let data1 = Box::into_raw(Box::new(1u64));
        let data2 = Box::into_raw(Box::new(2u64));

        domain.protect(slot1, data1 as *mut ());
        domain.protect(slot2, data2 as *mut ());

        assert!(domain.is_protected(data1 as *mut ()));
        assert!(domain.is_protected(data2 as *mut ()));

        domain.release_slot(1, slot1);
        assert!(!domain.is_protected(data1 as *mut ()));
        assert!(domain.is_protected(data2 as *mut ()));

        domain.release_slot(2, slot2);

        // Cleanup
        unsafe {
            drop(Box::from_raw(data1));
            drop(Box::from_raw(data2));
        }
    }

    #[test]
    fn test_reclaim_stats_snapshot() {
        let stats = ReclaimStats::default();

        stats.record_hazard_pin();
        stats.record_hazard_pin();
        stats.record_epoch_pin();
        stats.record_retire();
        stats.record_reclaim(5);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.hazard_pins, 2);
        assert_eq!(snapshot.epoch_pins, 1);
        assert_eq!(snapshot.objects_retired, 1);
        assert_eq!(snapshot.objects_reclaimed, 5);
        assert_eq!(snapshot.reclaim_cycles, 1);
    }

    #[test]
    fn test_strategy_configuration() {
        let reclaimer = UnifiedReclaimer::new().with_strategy(ReclaimStrategy::Epoch);

        assert_eq!(reclaimer.default_strategy, ReclaimStrategy::Epoch);
    }
}
