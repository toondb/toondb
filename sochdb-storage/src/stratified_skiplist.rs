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

//! Stratified SkipList with Deferred LSM Promotion (Task 2)
//!
//! This module provides a probabilistic hot-key buffer with CAS-based insertion
//! and batched promotion to the underlying LSM tree.
//!
//! ## Problem
//!
//! Every insert traverses the full memtable skiplist, even for frequently updated
//! keys that will be overwritten soon. Hot keys cause repeated work.
//!
//! ## Solution
//!
//! Two-level structure:
//! 1. **Hot Buffer:** Small, CAS-based skiplist for recent writes
//! 2. **Cold Storage:** Full memtable (promoted in batches)
//!
//! Hot keys are absorbed by the buffer; only final values get promoted.
//!
//! ## Performance
//!
//! | Metric | Before | After |
//! |--------|--------|-------|
//! | Hot key update | O(log N) | O(log H) where H << N |
//! | Promotion overhead | Per-op | Batched (amortized) |
//! | Lock contention | High | CAS-based (lock-free) |

use std::cmp::Ordering;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Maximum skiplist height
const MAX_HEIGHT: usize = 16;

/// Probability for height selection (1/4)
const P: u32 = 4;

/// Default hot buffer capacity before promotion
const DEFAULT_HOT_CAPACITY: usize = 4096;

// ============================================================================
// Skip Node (Lock-Free Tower)
// ============================================================================

/// Tower of forward pointers for a skip node
#[repr(C)]
struct Tower<K, V> {
    /// Height of this tower (1..=MAX_HEIGHT)
    height: usize,
    /// Forward pointers at each level
    next: [AtomicPtr<SkipNode<K, V>>; MAX_HEIGHT],
}

impl<K, V> Tower<K, V> {
    /// Create a new tower with the given height
    fn new(height: usize) -> Self {
        let mut next: [AtomicPtr<SkipNode<K, V>>; MAX_HEIGHT] =
            std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut()));
        
        // Initialize all pointers to null
        for ptr in next.iter_mut().take(height) {
            *ptr = AtomicPtr::new(ptr::null_mut());
        }
        
        Self { height, next }
    }
    
    /// Get the forward pointer at a level
    #[inline]
    fn get(&self, level: usize) -> *mut SkipNode<K, V> {
        self.next[level].load(AtomicOrdering::Acquire)
    }
    
    /// Set the forward pointer at a level
    #[inline]
    fn set(&self, level: usize, node: *mut SkipNode<K, V>) {
        self.next[level].store(node, AtomicOrdering::Release);
    }
    
    /// CAS the forward pointer at a level
    #[inline]
    fn cas(
        &self,
        level: usize,
        expected: *mut SkipNode<K, V>,
        new: *mut SkipNode<K, V>,
    ) -> Result<*mut SkipNode<K, V>, *mut SkipNode<K, V>> {
        self.next[level]
            .compare_exchange(expected, new, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
    }
}

/// A node in the skip list
#[repr(C)]
struct SkipNode<K, V> {
    /// The key
    key: K,
    /// The value (can be updated atomically)
    value: AtomicPtr<V>,
    /// Version counter for optimistic concurrency
    version: AtomicU64,
    /// Tower of forward pointers
    tower: Tower<K, V>,
}

impl<K, V> SkipNode<K, V> {
    /// Create a new skip node
    fn new(key: K, value: V, height: usize) -> *mut Self {
        let value_ptr = Box::into_raw(Box::new(value));
        let node = Box::new(Self {
            key,
            value: AtomicPtr::new(value_ptr),
            version: AtomicU64::new(1),
            tower: Tower::new(height),
        });
        Box::into_raw(node)
    }
    
    /// Get the current value
    #[inline]
    unsafe fn get_value(&self) -> &V {
        unsafe { &*self.value.load(AtomicOrdering::Acquire) }
    }
    
    /// Update the value (CAS-based)
    #[inline]
    unsafe fn update_value(&self, new_value: V) -> V {
        let new_ptr = Box::into_raw(Box::new(new_value));
        let old_ptr = self.value.swap(new_ptr, AtomicOrdering::AcqRel);
        self.version.fetch_add(1, AtomicOrdering::Release);
        unsafe { *Box::from_raw(old_ptr) }
    }
}

// ============================================================================
// Stratified SkipList
// ============================================================================

/// Thread-local random state for height selection
fn random_height() -> usize {
    let mut height = 1;
    // Use a simple PRNG based on pointer address and time
    let mut state = (&height as *const _ as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        );
    
    while height < MAX_HEIGHT {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        if (state >> 17) % (P as u64) != 0 {
            break;
        }
        height += 1;
    }
    height
}

/// Stratified skip list with lock-free operations
pub struct StratifiedSkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Head sentinel node
    head: *mut SkipNode<K, V>,
    /// Current maximum height
    max_height: AtomicUsize,
    /// Number of elements
    len: AtomicUsize,
    /// Capacity before promotion
    capacity: usize,
    /// Promotion callback
    promoter: Option<Arc<dyn Fn(Vec<(K, V)>) + Send + Sync>>,
}

impl<K, V> StratifiedSkipList<K, V>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    /// Create a new stratified skip list
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_HOT_CAPACITY)
    }
    
    /// Create with custom capacity
    pub fn with_capacity(capacity: usize) -> Self {
        // Create head sentinel with default values
        let head = SkipNode::new(K::default(), V::default(), MAX_HEIGHT);
        
        Self {
            head,
            max_height: AtomicUsize::new(1),
            len: AtomicUsize::new(0),
            capacity,
            promoter: None,
        }
    }
    
    /// Set the promotion callback
    pub fn set_promoter<F>(&mut self, promoter: F)
    where
        F: Fn(Vec<(K, V)>) + Send + Sync + 'static,
    {
        self.promoter = Some(Arc::new(promoter));
    }
    
    /// Insert or update a key-value pair (CAS-based)
    ///
    /// Returns the old value if the key existed.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        // Check if we need to promote before inserting
        if self.len() >= self.capacity {
            self.try_promote();
        }
        
        let height = random_height();
        let mut prev = [ptr::null_mut::<SkipNode<K, V>>(); MAX_HEIGHT];
        let mut next = [ptr::null_mut::<SkipNode<K, V>>(); MAX_HEIGHT];
        
        loop {
            // Find position
            self.find_position(&key, &mut prev, &mut next);
            
            // Check if key exists
            if !next[0].is_null() {
                let existing = unsafe { &*next[0] };
                if existing.key == key {
                    // Update existing value
                    let old = unsafe { existing.update_value(value.clone()) };
                    return Some(old);
                }
            }
            
            // Create new node
            let new_node = SkipNode::new(key.clone(), value.clone(), height);
            
            // Link at level 0 first
            unsafe {
                (*new_node).tower.set(0, next[0]);
            }
            
            // CAS at level 0
            let prev_ptr = if prev[0].is_null() { self.head } else { prev[0] };
            match unsafe { (*prev_ptr).tower.cas(0, next[0], new_node) } {
                Ok(_) => {
                    // Successfully inserted at level 0
                    // Now link higher levels
                    for level in 1..height {
                        loop {
                            unsafe {
                                (*new_node).tower.set(level, next[level]);
                            }
                            
                            let prev_at_level = if prev[level].is_null() { self.head } else { prev[level] };
                            if unsafe { (*prev_at_level).tower.cas(level, next[level], new_node) }.is_ok() {
                                break;
                            }
                            
                            // Re-find position at this level
                            self.find_position(&key, &mut prev, &mut next);
                        }
                    }
                    
                    // Update max height
                    loop {
                        let current_max = self.max_height.load(AtomicOrdering::Relaxed);
                        if height <= current_max {
                            break;
                        }
                        if self.max_height
                            .compare_exchange_weak(current_max, height, AtomicOrdering::Release, AtomicOrdering::Relaxed)
                            .is_ok()
                        {
                            break;
                        }
                    }
                    
                    self.len.fetch_add(1, AtomicOrdering::Release);
                    return None;
                }
                Err(_) => {
                    // CAS failed, someone else inserted
                    // Free the node we created and retry
                    unsafe {
                        let value_ptr = (*new_node).value.load(AtomicOrdering::Relaxed);
                        drop(Box::from_raw(value_ptr));
                        drop(Box::from_raw(new_node));
                    }
                    continue;
                }
            }
        }
    }
    
    /// Find the position for a key
    fn find_position(
        &self,
        key: &K,
        prev: &mut [*mut SkipNode<K, V>; MAX_HEIGHT],
        next: &mut [*mut SkipNode<K, V>; MAX_HEIGHT],
    ) {
        let max_height = self.max_height.load(AtomicOrdering::Acquire);
        let mut current = self.head;
        
        for level in (0..max_height).rev() {
            loop {
                let next_node = unsafe { (*current).tower.get(level) };
                if next_node.is_null() {
                    break;
                }
                
                let next_key = unsafe { &(*next_node).key };
                match next_key.cmp(key) {
                    Ordering::Less => {
                        current = next_node;
                    }
                    Ordering::Equal | Ordering::Greater => {
                        break;
                    }
                }
            }
            
            prev[level] = if current == self.head { ptr::null_mut() } else { current };
            next[level] = unsafe { (*current).tower.get(level) };
        }
    }
    
    /// Get a value by key
    pub fn get(&self, key: &K) -> Option<V> {
        let max_height = self.max_height.load(AtomicOrdering::Acquire);
        let mut current = self.head;
        
        for level in (0..max_height).rev() {
            loop {
                let next_node = unsafe { (*current).tower.get(level) };
                if next_node.is_null() {
                    break;
                }
                
                let next_key = unsafe { &(*next_node).key };
                match next_key.cmp(key) {
                    Ordering::Less => {
                        current = next_node;
                    }
                    Ordering::Equal => {
                        let value = unsafe { (*next_node).get_value().clone() };
                        return Some(value);
                    }
                    Ordering::Greater => {
                        break;
                    }
                }
            }
        }
        
        None
    }
    
    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(AtomicOrdering::Acquire)
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Try to promote entries to the underlying storage
    fn try_promote(&self) {
        if let Some(ref promoter) = self.promoter {
            let entries = self.drain();
            if !entries.is_empty() {
                promoter(entries);
            }
        }
    }
    
    /// Drain all entries for promotion
    pub fn drain(&self) -> Vec<(K, V)> {
        let mut entries = Vec::with_capacity(self.len());
        let mut current = unsafe { (*self.head).tower.get(0) };
        
        while !current.is_null() {
            let node = unsafe { &*current };
            let key = node.key.clone();
            let value = unsafe { node.get_value().clone() };
            entries.push((key, value));
            current = node.tower.get(0);
        }
        
        // Note: In production, we'd use proper marking for deletion
        // For now, just reset the count (simplified)
        self.len.store(0, AtomicOrdering::Release);
        
        entries
    }
    
    /// Iterate over all entries (read-only)
    pub fn iter(&self) -> impl Iterator<Item = (K, V)> + '_ {
        SkipListIter {
            current: unsafe { (*self.head).tower.get(0) },
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K, V> Default for StratifiedSkipList<K, V>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for StratifiedSkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    fn drop(&mut self) {
        let mut current = unsafe { (*self.head).tower.get(0) };
        
        while !current.is_null() {
            let next = unsafe { (*current).tower.get(0) };
            unsafe {
                let value_ptr = (*current).value.load(AtomicOrdering::Relaxed);
                drop(Box::from_raw(value_ptr));
                drop(Box::from_raw(current));
            }
            current = next;
        }
        
        // Free head
        unsafe {
            let value_ptr = (*self.head).value.load(AtomicOrdering::Relaxed);
            drop(Box::from_raw(value_ptr));
            drop(Box::from_raw(self.head));
        }
    }
}

// Safety: StratifiedSkipList uses atomic operations for all mutations
unsafe impl<K: Send + Sync + Ord + Clone, V: Send + Sync + Clone> Send for StratifiedSkipList<K, V> {}
unsafe impl<K: Send + Sync + Ord + Clone, V: Send + Sync + Clone> Sync for StratifiedSkipList<K, V> {}

/// Iterator over skip list entries
struct SkipListIter<'a, K, V> {
    current: *mut SkipNode<K, V>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, K: Clone, V: Clone> Iterator for SkipListIter<'a, K, V> {
    type Item = (K, V);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            return None;
        }
        
        let node = unsafe { &*self.current };
        let key = node.key.clone();
        let value = unsafe { node.get_value().clone() };
        self.current = node.tower.get(0);
        
        Some((key, value))
    }
}

// ============================================================================
// Batch Promoter
// ============================================================================

/// Statistics for promotion
#[derive(Debug, Clone, Default)]
pub struct PromotionStats {
    /// Number of promotions
    pub promotion_count: u64,
    /// Total entries promoted
    pub entries_promoted: u64,
    /// Average batch size
    pub avg_batch_size: f64,
}

/// Batch promoter for deferred LSM promotion
pub struct BatchPromoter<K, V>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    /// Hot buffer (stratified skip list)
    hot_buffer: StratifiedSkipList<K, V>,
    /// Pending promotion batches
    #[allow(dead_code)]
    pending: std::sync::Mutex<Vec<Vec<(K, V)>>>,
    /// Statistics
    stats: std::sync::Mutex<PromotionStats>,
    /// Background promoter thread handle
    _background: Option<std::thread::JoinHandle<()>>,
}

impl<K, V> BatchPromoter<K, V>
where
    K: Ord + Clone + Default + Send + Sync + 'static,
    V: Clone + Default + Send + Sync + 'static,
{
    /// Create a new batch promoter
    pub fn new(hot_capacity: usize) -> Arc<Self> {
        let promoter = Arc::new(Self {
            hot_buffer: StratifiedSkipList::with_capacity(hot_capacity),
            pending: std::sync::Mutex::new(Vec::new()),
            stats: std::sync::Mutex::new(PromotionStats::default()),
            _background: None,
        });
        
        promoter
    }
    
    /// Insert a hot key
    pub fn insert_hot(&self, key: K, value: V) -> Option<V> {
        self.hot_buffer.insert(key, value)
    }
    
    /// Get a value (checks hot buffer first)
    pub fn get(&self, key: &K) -> Option<V> {
        self.hot_buffer.get(key)
    }
    
    /// Force promotion of all hot entries
    pub fn force_promote(&self) -> Vec<(K, V)> {
        let entries = self.hot_buffer.drain();
        
        if !entries.is_empty() {
            let mut stats = self.stats.lock().unwrap();
            stats.promotion_count += 1;
            stats.entries_promoted += entries.len() as u64;
            stats.avg_batch_size = stats.entries_promoted as f64 / stats.promotion_count as f64;
        }
        
        entries
    }
    
    /// Get statistics
    pub fn stats(&self) -> PromotionStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Get hot buffer size
    pub fn hot_size(&self) -> usize {
        self.hot_buffer.len()
    }
}

// ============================================================================
// Deferred Index (combines hot buffer with cold storage)
// ============================================================================

/// Deferred index with stratified storage
pub struct DeferredIndex<K, V, Cold>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    /// Hot tier (stratified skip list)
    hot: StratifiedSkipList<K, V>,
    /// Cold tier (underlying storage)
    cold: Cold,
    /// Promotion threshold
    promotion_threshold: usize,
    /// Insert count since last promotion
    insert_count: AtomicUsize,
}

impl<K, V, Cold> DeferredIndex<K, V, Cold>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
    Cold: ColdStorage<K, V>,
{
    /// Create a new deferred index
    pub fn new(cold: Cold, promotion_threshold: usize) -> Self {
        Self {
            hot: StratifiedSkipList::with_capacity(promotion_threshold),
            cold,
            promotion_threshold,
            insert_count: AtomicUsize::new(0),
        }
    }
    
    /// Insert a key-value pair
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let count = self.insert_count.fetch_add(1, AtomicOrdering::Relaxed);
        
        // Check if we need to promote
        if count >= self.promotion_threshold {
            self.promote();
        }
        
        self.hot.insert(key, value)
    }
    
    /// Get a value (hot tier first, then cold)
    pub fn get(&self, key: &K) -> Option<V> {
        // Check hot tier first
        if let Some(value) = self.hot.get(key) {
            return Some(value);
        }
        
        // Fall back to cold tier
        self.cold.get(key)
    }
    
    /// Promote hot entries to cold storage
    pub fn promote(&self) {
        let entries = self.hot.drain();
        
        if !entries.is_empty() {
            self.cold.insert_batch(entries);
            self.insert_count.store(0, AtomicOrdering::Release);
        }
    }
    
    /// Get hot tier size
    pub fn hot_size(&self) -> usize {
        self.hot.len()
    }
}

/// Trait for cold storage backends
pub trait ColdStorage<K, V>: Send + Sync {
    /// Get a value
    fn get(&self, key: &K) -> Option<V>;
    
    /// Insert a batch of entries
    fn insert_batch(&self, entries: Vec<(K, V)>);
}

// ============================================================================
// Simple In-Memory Cold Storage (for testing)
// ============================================================================

/// Simple hashmap-based cold storage
pub struct HashMapCold<K, V> {
    data: parking_lot::RwLock<std::collections::HashMap<K, V>>,
}

impl<K, V> HashMapCold<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    /// Create new cold storage
    pub fn new() -> Self {
        Self {
            data: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

impl<K, V> Default for HashMapCold<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> ColdStorage<K, V> for HashMapCold<K, V>
where
    K: Eq + std::hash::Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    fn get(&self, key: &K) -> Option<V> {
        self.data.read().get(key).cloned()
    }
    
    fn insert_batch(&self, entries: Vec<(K, V)>) {
        let mut data = self.data.write();
        for (k, v) in entries {
            data.insert(k, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_skiplist_basic() {
        let list: StratifiedSkipList<i32, String> = StratifiedSkipList::new();
        
        assert!(list.insert(1, "one".to_string()).is_none());
        assert!(list.insert(2, "two".to_string()).is_none());
        assert!(list.insert(3, "three".to_string()).is_none());
        
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(&1), Some("one".to_string()));
        assert_eq!(list.get(&2), Some("two".to_string()));
        assert_eq!(list.get(&3), Some("three".to_string()));
        assert_eq!(list.get(&4), None);
    }
    
    #[test]
    fn test_skiplist_update() {
        let list: StratifiedSkipList<i32, String> = StratifiedSkipList::new();
        
        assert!(list.insert(1, "one".to_string()).is_none());
        assert_eq!(list.insert(1, "ONE".to_string()), Some("one".to_string()));
        
        assert_eq!(list.len(), 1);
        assert_eq!(list.get(&1), Some("ONE".to_string()));
    }
    
    #[test]
    fn test_skiplist_concurrent() {
        let list = Arc::new(StratifiedSkipList::<i32, i32>::with_capacity(100000));
        let mut handles = vec![];
        
        for t in 0..4 {
            let list_clone = list.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let key = t * 1000 + i;
                    list_clone.insert(key, key * 2);
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(list.len(), 4000);
        
        // Verify some values
        assert_eq!(list.get(&0), Some(0));
        assert_eq!(list.get(&1000), Some(2000));
        assert_eq!(list.get(&2000), Some(4000));
    }
    
    #[test]
    fn test_skiplist_drain() {
        let list: StratifiedSkipList<i32, i32> = StratifiedSkipList::new();
        
        for i in 0..100 {
            list.insert(i, i * 2);
        }
        
        let entries = list.drain();
        assert_eq!(entries.len(), 100);
        
        // Should be sorted
        for (i, (k, v)) in entries.iter().enumerate() {
            assert_eq!(*k, i as i32);
            assert_eq!(*v, (i * 2) as i32);
        }
    }
    
    #[test]
    fn test_batch_promoter() {
        let promoter = BatchPromoter::<i32, i32>::new(100);
        
        for i in 0..50 {
            promoter.insert_hot(i, i * 2);
        }
        
        assert_eq!(promoter.hot_size(), 50);
        assert_eq!(promoter.get(&10), Some(20));
        
        let promoted = promoter.force_promote();
        assert_eq!(promoted.len(), 50);
        
        let stats = promoter.stats();
        assert_eq!(stats.promotion_count, 1);
        assert_eq!(stats.entries_promoted, 50);
    }
    
    #[test]
    fn test_deferred_index() {
        let cold = HashMapCold::<i32, i32>::new();
        let index = DeferredIndex::new(cold, 10);
        
        // Insert hot keys
        for i in 0..5 {
            index.insert(i, i * 10);
        }
        
        // Get from hot tier
        assert_eq!(index.get(&3), Some(30));
        
        // Insert more to trigger promotion
        for i in 5..15 {
            index.insert(i, i * 10);
        }
        
        // After promotion, should still find values
        assert_eq!(index.get(&0), Some(0)); // Was promoted
        assert_eq!(index.get(&12), Some(120)); // Was promoted or still hot
    }
    
    #[test]
    fn test_hot_key_absorption() {
        let list: StratifiedSkipList<String, i32> = StratifiedSkipList::new();
        
        // Simulate hot key pattern
        for _ in 0..100 {
            list.insert("hot_key".to_string(), 42);
        }
        
        // Should only have one entry (absorbed)
        assert_eq!(list.len(), 1);
        assert_eq!(list.get(&"hot_key".to_string()), Some(42));
    }
}
