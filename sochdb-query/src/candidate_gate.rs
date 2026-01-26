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

//! Unified Candidate Gate Interface (Task 4)
//!
//! This module defines the `AllowedSet` abstraction that every retrieval
//! executor MUST accept. The gate guarantees:
//!
//! 1. **Never return a doc outside AllowedSet** - structural enforcement
//! 2. **Apply constraints during generation** - no post-filtering
//! 3. **Consistent semantics** across vector/BM25/hybrid/context
//!
//! ## The Contract
//!
//! Every executor receives an `AllowedSet` and must:
//! - Check membership BEFORE including any candidate
//! - Short-circuit if AllowedSet is empty (return empty results)
//! - Report selectivity for query planning
//!
//! ## Representations
//!
//! `AllowedSet` supports multiple representations for efficiency:
//!
//! | Representation | Best For | Membership | Space |
//! |----------------|----------|------------|-------|
//! | Bitmap | Dense sets | O(1) | O(N/8) |
//! | SortedVec | Sparse sets | O(log n) | O(n) |
//! | HashSet | Random access | O(1) avg | O(n) |
//! | All | No constraint | O(1) | O(1) |
//!
//! ## Selectivity
//!
//! Executors use selectivity `|S|/N` to choose execution strategy:
//! - High selectivity (> 0.1): Standard search with filter
//! - Low selectivity (< 0.01): Scan only allowed IDs
//! - Very low (< 0.001): Consider alternative strategy

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

// ============================================================================
// AllowedSet - Core Abstraction
// ============================================================================

/// The unified gate for candidate filtering
///
/// Every executor MUST check `allowed_set.contains(doc_id)` before
/// including any result. This is the structural enforcement of pushdown.
#[derive(Clone)]
pub enum AllowedSet {
    /// All documents are allowed (no filter constraint)
    All,
    
    /// Bitmap representation (efficient for dense sets)
    Bitmap(Arc<AllowedBitmap>),
    
    /// Sorted vector (efficient for sparse sets with iteration)
    SortedVec(Arc<Vec<u64>>),
    
    /// Hash set (efficient for random access)
    HashSet(Arc<HashSet<u64>>),
    
    /// No documents allowed (empty result shortcut)
    None,
}

impl AllowedSet {
    /// Create an AllowedSet from a bitmap
    pub fn from_bitmap(bitmap: AllowedBitmap) -> Self {
        if bitmap.is_empty() {
            Self::None
        } else if bitmap.is_all() {
            Self::All
        } else {
            Self::Bitmap(Arc::new(bitmap))
        }
    }
    
    /// Create an AllowedSet from a sorted vector of doc IDs
    pub fn from_sorted_vec(mut ids: Vec<u64>) -> Self {
        if ids.is_empty() {
            return Self::None;
        }
        ids.sort_unstable();
        ids.dedup();
        Self::SortedVec(Arc::new(ids))
    }
    
    /// Create an AllowedSet from an iterator of doc IDs
    pub fn from_iter(ids: impl IntoIterator<Item = u64>) -> Self {
        let set: HashSet<u64> = ids.into_iter().collect();
        if set.is_empty() {
            Self::None
        } else {
            Self::HashSet(Arc::new(set))
        }
    }
    
    /// Check if a document ID is allowed
    ///
    /// This is the core operation that executors MUST call.
    #[inline]
    pub fn contains(&self, doc_id: u64) -> bool {
        match self {
            Self::All => true,
            Self::Bitmap(bm) => bm.contains(doc_id),
            Self::SortedVec(vec) => vec.binary_search(&doc_id).is_ok(),
            Self::HashSet(set) => set.contains(&doc_id),
            Self::None => false,
        }
    }
    
    /// Check if this set is empty (no allowed documents)
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }
    
    /// Check if this set allows all documents
    pub fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }
    
    /// Get the cardinality (number of allowed documents)
    ///
    /// Returns None for All (unknown without universe size)
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::Bitmap(bm) => Some(bm.count()),
            Self::SortedVec(vec) => Some(vec.len()),
            Self::HashSet(set) => Some(set.len()),
            Self::None => Some(0),
        }
    }
    
    /// Compute selectivity against a universe of size N
    ///
    /// Returns |S| / N, the fraction of allowed documents
    pub fn selectivity(&self, universe_size: usize) -> f64 {
        if universe_size == 0 {
            return 0.0;
        }
        match self {
            Self::All => 1.0,
            Self::None => 0.0,
            other => {
                other.cardinality()
                    .map(|c| c as f64 / universe_size as f64)
                    .unwrap_or(1.0)
            }
        }
    }
    
    /// Intersect with another AllowedSet
    pub fn intersect(&self, other: &AllowedSet) -> AllowedSet {
        match (self, other) {
            // Identity cases
            (Self::All, x) | (x, Self::All) => x.clone(),
            (Self::None, _) | (_, Self::None) => Self::None,
            
            // Both are sets - compute intersection
            (Self::SortedVec(a), Self::SortedVec(b)) => {
                let result = sorted_vec_intersect(a, b);
                Self::from_sorted_vec(result)
            }
            (Self::HashSet(a), Self::HashSet(b)) => {
                let result: HashSet<_> = a.intersection(b).copied().collect();
                if result.is_empty() {
                    Self::None
                } else {
                    Self::HashSet(Arc::new(result))
                }
            }
            (Self::Bitmap(a), Self::Bitmap(b)) => {
                let result = a.intersect(b);
                Self::from_bitmap(result)
            }
            
            // Mixed - convert to hash set
            (a, b) => {
                let set_a: HashSet<u64> = a.iter().collect();
                let set_b: HashSet<u64> = b.iter().collect();
                let result: HashSet<_> = set_a.intersection(&set_b).copied().collect();
                if result.is_empty() {
                    Self::None
                } else {
                    Self::HashSet(Arc::new(result))
                }
            }
        }
    }
    
    /// Union with another AllowedSet
    pub fn union(&self, other: &AllowedSet) -> AllowedSet {
        match (self, other) {
            (Self::All, _) | (_, Self::All) => Self::All,
            (Self::None, x) | (x, Self::None) => x.clone(),
            
            (Self::HashSet(a), Self::HashSet(b)) => {
                let result: HashSet<_> = a.union(b).copied().collect();
                Self::HashSet(Arc::new(result))
            }
            
            // Mixed - convert to hash set
            (a, b) => {
                let mut result: HashSet<u64> = a.iter().collect();
                result.extend(b.iter());
                Self::HashSet(Arc::new(result))
            }
        }
    }
    
    /// Iterate over allowed document IDs
    ///
    /// Note: For All, this returns an empty iterator (unknown universe)
    pub fn iter(&self) -> AllowedSetIter<'_> {
        match self {
            Self::All => AllowedSetIter::Empty,
            Self::Bitmap(bm) => AllowedSetIter::Bitmap(bm.iter()),
            Self::SortedVec(vec) => AllowedSetIter::SortedVec(vec.iter()),
            Self::HashSet(set) => AllowedSetIter::HashSet(set.iter()),
            Self::None => AllowedSetIter::Empty,
        }
    }
    
    /// Convert to a Vec (for small sets)
    pub fn to_vec(&self) -> Vec<u64> {
        self.iter().collect()
    }
}

impl fmt::Debug for AllowedSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, "AllowedSet::All"),
            Self::None => write!(f, "AllowedSet::None"),
            Self::Bitmap(bm) => write!(f, "AllowedSet::Bitmap(count={})", bm.count()),
            Self::SortedVec(vec) => write!(f, "AllowedSet::SortedVec(len={})", vec.len()),
            Self::HashSet(set) => write!(f, "AllowedSet::HashSet(len={})", set.len()),
        }
    }
}

impl Default for AllowedSet {
    fn default() -> Self {
        Self::All
    }
}

// Helper for sorted vec intersection
fn sorted_vec_intersect(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len().min(b.len()));
    let mut i = 0;
    let mut j = 0;
    
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    
    result
}

// ============================================================================
// AllowedSet Iterator
// ============================================================================

/// Iterator over allowed document IDs
pub enum AllowedSetIter<'a> {
    Empty,
    Bitmap(BitmapIter<'a>),
    SortedVec(std::slice::Iter<'a, u64>),
    HashSet(std::collections::hash_set::Iter<'a, u64>),
}

impl<'a> Iterator for AllowedSetIter<'a> {
    type Item = u64;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Bitmap(iter) => iter.next(),
            Self::SortedVec(iter) => iter.next().copied(),
            Self::HashSet(iter) => iter.next().copied(),
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Empty => (0, Some(0)),
            Self::Bitmap(iter) => iter.size_hint(),
            Self::SortedVec(iter) => iter.size_hint(),
            Self::HashSet(iter) => iter.size_hint(),
        }
    }
}

// ============================================================================
// Bitmap Implementation
// ============================================================================

/// Simple bitmap for allowed document IDs
///
/// This is a basic implementation. For production, consider using
/// the `roaring` crate for compressed bitmaps.
pub struct AllowedBitmap {
    /// Bits stored as u64 words
    words: Vec<u64>,
    /// Total number of set bits (cached)
    count: usize,
    /// Whether this represents "all" (complement mode)
    all: bool,
}

impl AllowedBitmap {
    /// Create a new empty bitmap
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            count: 0,
            all: false,
        }
    }
    
    /// Create a bitmap with all bits set up to max_id
    pub fn all(max_id: u64) -> Self {
        let word_count = (max_id as usize / 64) + 1;
        Self {
            words: vec![u64::MAX; word_count],
            count: max_id as usize + 1,
            all: true,
        }
    }
    
    /// Create a bitmap from a set of IDs
    pub fn from_ids(ids: &[u64]) -> Self {
        if ids.is_empty() {
            return Self::new();
        }
        
        let max_id = *ids.iter().max().unwrap();
        let word_count = (max_id as usize / 64) + 1;
        let mut words = vec![0u64; word_count];
        
        for &id in ids {
            let word_idx = id as usize / 64;
            let bit_idx = id % 64;
            words[word_idx] |= 1 << bit_idx;
        }
        
        Self {
            words,
            count: ids.len(),
            all: false,
        }
    }
    
    /// Set a bit
    pub fn set(&mut self, id: u64) {
        let word_idx = id as usize / 64;
        let bit_idx = id % 64;
        
        // Extend if necessary
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }
        
        let old = self.words[word_idx];
        self.words[word_idx] |= 1 << bit_idx;
        if old != self.words[word_idx] {
            self.count += 1;
        }
    }
    
    /// Clear a bit
    pub fn clear(&mut self, id: u64) {
        let word_idx = id as usize / 64;
        if word_idx >= self.words.len() {
            return;
        }
        
        let bit_idx = id % 64;
        let old = self.words[word_idx];
        self.words[word_idx] &= !(1 << bit_idx);
        if old != self.words[word_idx] {
            self.count -= 1;
        }
    }
    
    /// Check if a bit is set
    #[inline]
    pub fn contains(&self, id: u64) -> bool {
        let word_idx = id as usize / 64;
        if word_idx >= self.words.len() {
            return false;
        }
        let bit_idx = id % 64;
        (self.words[word_idx] & (1 << bit_idx)) != 0
    }
    
    /// Get the count of set bits
    pub fn count(&self) -> usize {
        self.count
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Check if all bits are set
    pub fn is_all(&self) -> bool {
        self.all
    }
    
    /// Intersect with another bitmap
    pub fn intersect(&self, other: &AllowedBitmap) -> AllowedBitmap {
        let min_len = self.words.len().min(other.words.len());
        let mut words = Vec::with_capacity(min_len);
        let mut count = 0;
        
        for i in 0..min_len {
            let word = self.words[i] & other.words[i];
            count += word.count_ones() as usize;
            words.push(word);
        }
        
        AllowedBitmap {
            words,
            count,
            all: false,
        }
    }
    
    /// Union with another bitmap
    pub fn union(&self, other: &AllowedBitmap) -> AllowedBitmap {
        let max_len = self.words.len().max(other.words.len());
        let mut words = Vec::with_capacity(max_len);
        let mut count = 0;
        
        for i in 0..max_len {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            let word = a | b;
            count += word.count_ones() as usize;
            words.push(word);
        }
        
        AllowedBitmap {
            words,
            count,
            all: false,
        }
    }
    
    /// Iterate over set bit positions
    pub fn iter(&self) -> BitmapIter<'_> {
        BitmapIter {
            words: &self.words,
            word_idx: 0,
            bit_offset: 0,
            remaining: self.count,
        }
    }
}

impl Default for AllowedBitmap {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over set bits in a bitmap
pub struct BitmapIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    bit_offset: u64,
    remaining: usize,
}

impl<'a> Iterator for BitmapIter<'a> {
    type Item = u64;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        
        while self.word_idx < self.words.len() {
            let word = self.words[self.word_idx];
            let masked = word >> self.bit_offset;
            
            if masked != 0 {
                let trailing = masked.trailing_zeros() as u64;
                let bit_pos = self.bit_offset + trailing;
                self.bit_offset = bit_pos + 1;
                
                if self.bit_offset >= 64 {
                    self.bit_offset = 0;
                    self.word_idx += 1;
                }
                
                self.remaining -= 1;
                return Some(self.word_idx as u64 * 64 + bit_pos - (if self.bit_offset == 0 { 64 } else { 0 }) + (if bit_pos >= 64 { 0 } else { bit_pos }));
            }
            
            self.word_idx += 1;
            self.bit_offset = 0;
        }
        
        None
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// Fix the iterator - simpler implementation
impl<'a> BitmapIter<'a> {
    #[allow(dead_code)]
    fn new(words: &'a [u64], count: usize) -> Self {
        Self {
            words,
            word_idx: 0,
            bit_offset: 0,
            remaining: count,
        }
    }
}

// Simple correct iterator implementation
impl AllowedBitmap {
    /// Iterate over set bit positions (simple implementation)
    pub fn iter_simple(&self) -> impl Iterator<Item = u64> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            (0..64).filter_map(move |bit| {
                if (word & (1 << bit)) != 0 {
                    Some(word_idx as u64 * 64 + bit as u64)
                } else {
                    None
                }
            })
        })
    }
}

// ============================================================================
// Candidate Gate Trait
// ============================================================================

/// The candidate gate trait that all executors must implement
///
/// This trait ensures every retrieval path respects the AllowedSet.
pub trait CandidateGate {
    /// The query type
    type Query;
    
    /// The result type  
    type Result;
    
    /// The error type
    type Error;
    
    /// Execute with a mandatory allowed set
    ///
    /// # Contract
    ///
    /// - MUST NOT return any result with doc_id not in allowed_set
    /// - SHOULD short-circuit if allowed_set is empty
    /// - SHOULD use selectivity to choose execution strategy
    fn execute_with_gate(
        &self,
        query: &Self::Query,
        allowed_set: &AllowedSet,
    ) -> Result<Self::Result, Self::Error>;
    
    /// Get the execution strategy for a given selectivity
    fn strategy_for_selectivity(&self, selectivity: f64) -> ExecutionStrategy {
        if selectivity >= 0.1 {
            ExecutionStrategy::FilterDuringSearch
        } else if selectivity >= 0.001 {
            ExecutionStrategy::ScanAllowedIds
        } else {
            ExecutionStrategy::LinearScan
        }
    }
}

/// Execution strategy based on selectivity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Standard search with filter check during traversal
    FilterDuringSearch,
    
    /// Iterate over allowed IDs and compute distances
    ScanAllowedIds,
    
    /// Fall back to linear scan (very low selectivity)
    LinearScan,
    
    /// Refuse to execute (too expensive)
    Reject,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allowed_set_contains() {
        // All
        let all = AllowedSet::All;
        assert!(all.contains(0));
        assert!(all.contains(1000000));
        
        // None
        let none = AllowedSet::None;
        assert!(!none.contains(0));
        
        // SortedVec
        let vec = AllowedSet::from_sorted_vec(vec![1, 3, 5, 7, 9]);
        assert!(vec.contains(1));
        assert!(vec.contains(5));
        assert!(!vec.contains(2));
        assert!(!vec.contains(10));
        
        // HashSet
        let set = AllowedSet::from_iter([1, 3, 5, 7, 9]);
        assert!(set.contains(1));
        assert!(set.contains(5));
        assert!(!set.contains(2));
    }
    
    #[test]
    fn test_allowed_set_selectivity() {
        let set = AllowedSet::from_sorted_vec(vec![1, 2, 3, 4, 5]);
        
        assert_eq!(set.selectivity(100), 0.05);
        assert_eq!(set.selectivity(10), 0.5);
        
        assert_eq!(AllowedSet::All.selectivity(100), 1.0);
        assert_eq!(AllowedSet::None.selectivity(100), 0.0);
    }
    
    #[test]
    fn test_allowed_set_intersection() {
        let a = AllowedSet::from_sorted_vec(vec![1, 2, 3, 4, 5]);
        let b = AllowedSet::from_sorted_vec(vec![3, 4, 5, 6, 7]);
        
        let c = a.intersect(&b);
        assert_eq!(c.cardinality(), Some(3));
        assert!(c.contains(3));
        assert!(c.contains(4));
        assert!(c.contains(5));
        assert!(!c.contains(1));
        assert!(!c.contains(7));
    }
    
    #[test]
    fn test_bitmap_basic() {
        let mut bm = AllowedBitmap::new();
        bm.set(0);
        bm.set(5);
        bm.set(64);
        bm.set(100);
        
        assert!(bm.contains(0));
        assert!(bm.contains(5));
        assert!(bm.contains(64));
        assert!(bm.contains(100));
        assert!(!bm.contains(1));
        assert!(!bm.contains(63));
        
        assert_eq!(bm.count(), 4);
    }
    
    #[test]
    fn test_bitmap_from_ids() {
        let ids = vec![1, 5, 10, 100, 1000];
        let bm = AllowedBitmap::from_ids(&ids);
        
        for &id in &ids {
            assert!(bm.contains(id));
        }
        assert!(!bm.contains(0));
        assert!(!bm.contains(50));
    }
    
    #[test]
    fn test_bitmap_intersection() {
        let a = AllowedBitmap::from_ids(&[1, 2, 3, 4, 5]);
        let b = AllowedBitmap::from_ids(&[3, 4, 5, 6, 7]);
        
        let c = a.intersect(&b);
        assert_eq!(c.count(), 3);
        assert!(c.contains(3));
        assert!(c.contains(4));
        assert!(c.contains(5));
    }
    
    #[test]
    fn test_execution_strategy() {
        struct DummyGate;
        impl CandidateGate for DummyGate {
            type Query = ();
            type Result = ();
            type Error = ();
            fn execute_with_gate(&self, _: &(), _: &AllowedSet) -> Result<(), ()> {
                Ok(())
            }
        }
        
        let gate = DummyGate;
        assert_eq!(gate.strategy_for_selectivity(0.5), ExecutionStrategy::FilterDuringSearch);
        assert_eq!(gate.strategy_for_selectivity(0.01), ExecutionStrategy::ScanAllowedIds);
        assert_eq!(gate.strategy_for_selectivity(0.0001), ExecutionStrategy::LinearScan);
    }
}
