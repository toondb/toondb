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

//! # Cardinality-Aware Filter Indexing (Task 7)
//!
//! Implements adaptive filter representation based on attribute cardinality,
//! density, and query frequency.
//!
//! ## Representations
//!
//! 1. **Roaring Bitmap**: Best for low-cardinality (tenant, status, category)
//! 2. **Postings List**: Best for high-cardinality sparse (user_id, doc_id)
//! 3. **Hashed Set**: Best for very high cardinality with prefix queries
//!
//! ## Math/Algorithm
//!
//! Cost-based planning: choose representation minimizing
//! Bytes(filter_eval) + Bytes(extra_scan_due_to_postfilter)
//!
//! Postings win for sparse/high-cardinality, bitmaps for dense/low-cardinality.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::filter_indexing::{FilterIndex, FilterPolicy, AttributeStats};
//!
//! let stats = AttributeStats::new("tenant_id")
//!     .cardinality(100)
//!     .density(0.8);
//!
//! let policy = FilterPolicy::auto_select(&stats, n_vectors);
//! let index = FilterIndex::new(policy);
//! ```

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;

// ============================================================================
// Filter Representation
// ============================================================================

/// Filter representation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterRepresentation {
    /// Roaring bitmap - best for low cardinality, high density
    /// Memory: ~O(N/8) for dense, O(N_set) for sparse
    RoaringBitmap,
    
    /// Sorted postings list - best for high cardinality, sparse
    /// Memory: O(N_set * 4) for u32 IDs
    PostingsList,
    
    /// Hash set - best for very high cardinality, point queries
    /// Memory: O(N_set * 8) with some overhead
    HashedSet,
    
    /// Inverted postings - best for multi-value attributes (tags)
    /// Memory: O(N_values * avg_posting_size)
    InvertedPostings,
}

impl FilterRepresentation {
    /// Estimate memory bytes for this representation
    pub fn estimated_bytes(&self, n_vectors: usize, cardinality: usize, density: f32) -> usize {
        let n_set = (n_vectors as f32 * density) as usize;
        
        match self {
            Self::RoaringBitmap => {
                // Roaring uses ~16 bytes per run for RLE, or raw bitmap for dense
                if density > 0.5 {
                    n_vectors / 8 // Bitmap mode
                } else {
                    n_set * 2 // Array mode (16-bit within containers)
                }
            }
            Self::PostingsList => {
                n_set * 4 // u32 per ID
            }
            Self::HashedSet => {
                n_set * 12 // Hash overhead
            }
            Self::InvertedPostings => {
                cardinality * 16 + n_set * 4 // Overhead per value + postings
            }
        }
    }
    
    /// Estimate query cost (relative units)
    pub fn estimated_query_cost(&self, n_vectors: usize, cardinality: usize, density: f32, selectivity: f32) -> f32 {
        let n_set = (n_vectors as f32 * density) as usize;
        let expected_result = (n_set as f32 * selectivity) as usize;
        
        match self {
            Self::RoaringBitmap => {
                // Bitmap intersection is fast
                (n_vectors / 64) as f32 // Operations on 64-bit words
            }
            Self::PostingsList => {
                // Binary search + scan
                (expected_result as f32).log2() + expected_result as f32
            }
            Self::HashedSet => {
                // O(1) lookup per ID
                expected_result as f32
            }
            Self::InvertedPostings => {
                // Depends on number of matching values
                cardinality as f32 * 0.01 + expected_result as f32
            }
        }
    }
}

// ============================================================================
// Attribute Statistics
// ============================================================================

/// Statistics about a filterable attribute
#[derive(Debug, Clone)]
pub struct AttributeStats {
    /// Attribute name
    pub name: String,
    
    /// Number of distinct values
    pub cardinality: usize,
    
    /// Fraction of vectors that have this attribute
    pub density: f32,
    
    /// Average selectivity of queries on this attribute
    pub avg_selectivity: f32,
    
    /// Query frequency (queries per second)
    pub query_frequency: f32,
    
    /// Is multi-valued (like tags)
    pub is_multi_valued: bool,
    
    /// Value distribution (optional: value -> count)
    pub value_distribution: Option<HashMap<String, usize>>,
}

impl AttributeStats {
    /// Create new stats for an attribute
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            cardinality: 1,
            density: 1.0,
            avg_selectivity: 0.5,
            query_frequency: 1.0,
            is_multi_valued: false,
            value_distribution: None,
        }
    }
    
    /// Set cardinality
    pub fn cardinality(mut self, c: usize) -> Self {
        self.cardinality = c;
        self
    }
    
    /// Set density
    pub fn density(mut self, d: f32) -> Self {
        self.density = d;
        self
    }
    
    /// Set average selectivity
    pub fn selectivity(mut self, s: f32) -> Self {
        self.avg_selectivity = s;
        self
    }
    
    /// Set query frequency
    pub fn frequency(mut self, f: f32) -> Self {
        self.query_frequency = f;
        self
    }
    
    /// Set multi-valued
    pub fn multi_valued(mut self, m: bool) -> Self {
        self.is_multi_valued = m;
        self
    }
    
    /// Compute cardinality ratio (cardinality / n_vectors)
    pub fn cardinality_ratio(&self, n_vectors: usize) -> f32 {
        self.cardinality as f32 / n_vectors.max(1) as f32
    }
    
    /// Is this a low-cardinality attribute?
    pub fn is_low_cardinality(&self, n_vectors: usize) -> bool {
        self.cardinality_ratio(n_vectors) < 0.01 || self.cardinality < 1000
    }
    
    /// Is this a high-cardinality attribute?
    pub fn is_high_cardinality(&self, n_vectors: usize) -> bool {
        self.cardinality_ratio(n_vectors) > 0.1 && self.cardinality > 10000
    }
}

// ============================================================================
// Filter Policy
// ============================================================================

/// Policy for filter indexing
#[derive(Debug, Clone)]
pub struct FilterPolicy {
    /// Representation to use
    pub representation: FilterRepresentation,
    
    /// Build index per-list (for IVF) vs global
    pub per_list: bool,
    
    /// Cache filter results
    pub cache_results: bool,
    
    /// Threshold for switching to post-filter
    pub postfilter_threshold: f32,
}

impl FilterPolicy {
    /// Auto-select best policy based on statistics
    pub fn auto_select(stats: &AttributeStats, n_vectors: usize) -> Self {
        let repr = if stats.is_multi_valued {
            FilterRepresentation::InvertedPostings
        } else if stats.is_low_cardinality(n_vectors) && stats.density > 0.5 {
            // Low cardinality + dense = bitmap
            FilterRepresentation::RoaringBitmap
        } else if stats.is_high_cardinality(n_vectors) && stats.density < 0.1 {
            // High cardinality + sparse = postings
            FilterRepresentation::PostingsList
        } else if stats.cardinality > 100000 {
            // Very high cardinality = hash set
            FilterRepresentation::HashedSet
        } else {
            // Default to bitmap
            FilterRepresentation::RoaringBitmap
        };
        
        Self {
            representation: repr,
            per_list: stats.avg_selectivity < 0.1, // Per-list for selective filters
            cache_results: stats.query_frequency > 10.0,
            postfilter_threshold: 0.8, // Switch to post-filter above 80% selectivity
        }
    }
    
    /// Create bitmap policy
    pub fn bitmap() -> Self {
        Self {
            representation: FilterRepresentation::RoaringBitmap,
            per_list: false,
            cache_results: true,
            postfilter_threshold: 0.8,
        }
    }
    
    /// Create postings policy
    pub fn postings() -> Self {
        Self {
            representation: FilterRepresentation::PostingsList,
            per_list: true,
            cache_results: false,
            postfilter_threshold: 0.9,
        }
    }
}

// ============================================================================
// Filter Index Implementations
// ============================================================================

/// Simple bitmap filter (simplified Roaring-like)
#[derive(Debug, Clone)]
pub struct BitmapFilter {
    /// Bitmap words
    words: Vec<u64>,
    /// Number of bits
    n_bits: usize,
}

impl BitmapFilter {
    /// Create empty bitmap
    pub fn new(n_bits: usize) -> Self {
        let n_words = (n_bits + 63) / 64;
        Self {
            words: vec![0; n_words],
            n_bits,
        }
    }
    
    /// Set a bit
    pub fn set(&mut self, idx: u32) {
        if (idx as usize) < self.n_bits {
            let word = idx as usize / 64;
            let bit = idx as usize % 64;
            self.words[word] |= 1 << bit;
        }
    }
    
    /// Check if bit is set
    pub fn contains(&self, idx: u32) -> bool {
        if (idx as usize) >= self.n_bits {
            return false;
        }
        let word = idx as usize / 64;
        let bit = idx as usize % 64;
        (self.words[word] & (1 << bit)) != 0
    }
    
    /// AND with another bitmap
    pub fn and(&self, other: &BitmapFilter) -> BitmapFilter {
        let n_words = self.words.len().min(other.words.len());
        let mut result = BitmapFilter::new(self.n_bits.min(other.n_bits));
        for i in 0..n_words {
            result.words[i] = self.words[i] & other.words[i];
        }
        result
    }
    
    /// OR with another bitmap
    pub fn or(&self, other: &BitmapFilter) -> BitmapFilter {
        let n_words = self.words.len().max(other.words.len());
        let mut result = BitmapFilter::new(self.n_bits.max(other.n_bits));
        for i in 0..self.words.len() {
            result.words[i] |= self.words[i];
        }
        for i in 0..other.words.len() {
            result.words[i] |= other.words[i];
        }
        result
    }
    
    /// Count set bits
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
    
    /// Iterate over set bits
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            (0..64).filter_map(move |bit| {
                if (word & (1 << bit)) != 0 {
                    Some((word_idx * 64 + bit) as u32)
                } else {
                    None
                }
            })
        }).filter(move |&idx| (idx as usize) < self.n_bits)
    }
    
    /// Memory footprint
    pub fn memory_bytes(&self) -> usize {
        self.words.len() * 8
    }
}

/// Postings list filter
#[derive(Debug, Clone)]
pub struct PostingsFilter {
    /// Sorted vector IDs
    ids: Vec<u32>,
}

impl PostingsFilter {
    /// Create empty postings list
    pub fn new() -> Self {
        Self { ids: Vec::new() }
    }
    
    /// Create from sorted IDs
    pub fn from_ids(ids: Vec<u32>) -> Self {
        let mut ids = ids;
        ids.sort_unstable();
        ids.dedup();
        Self { ids }
    }
    
    /// Add an ID
    pub fn add(&mut self, id: u32) {
        match self.ids.binary_search(&id) {
            Ok(_) => {} // Already present
            Err(pos) => self.ids.insert(pos, id),
        }
    }
    
    /// Check if ID is present
    pub fn contains(&self, id: u32) -> bool {
        self.ids.binary_search(&id).is_ok()
    }
    
    /// Intersect with another postings list
    pub fn intersect(&self, other: &PostingsFilter) -> PostingsFilter {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        while i < self.ids.len() && j < other.ids.len() {
            match self.ids[i].cmp(&other.ids[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(self.ids[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        
        PostingsFilter { ids: result }
    }
    
    /// Union with another postings list
    pub fn union(&self, other: &PostingsFilter) -> PostingsFilter {
        let mut result = Vec::with_capacity(self.ids.len() + other.ids.len());
        let mut i = 0;
        let mut j = 0;
        
        while i < self.ids.len() && j < other.ids.len() {
            match self.ids[i].cmp(&other.ids[j]) {
                std::cmp::Ordering::Less => {
                    result.push(self.ids[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(other.ids[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(self.ids[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        
        result.extend_from_slice(&self.ids[i..]);
        result.extend_from_slice(&other.ids[j..]);
        
        PostingsFilter { ids: result }
    }
    
    /// Get count
    pub fn count(&self) -> usize {
        self.ids.len()
    }
    
    /// Iterate over IDs
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.ids.iter().copied()
    }
    
    /// Memory footprint
    pub fn memory_bytes(&self) -> usize {
        self.ids.len() * 4
    }
}

impl Default for PostingsFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unified Filter Index
// ============================================================================

/// Unified filter index with adaptive representation
pub struct FilterIndex {
    /// Attribute name
    attribute: String,
    /// Policy
    policy: FilterPolicy,
    /// Inverted index: value -> filter
    inverted: HashMap<String, FilterData>,
    /// Number of vectors
    n_vectors: usize,
}

/// Filter data can be bitmap or postings
enum FilterData {
    Bitmap(BitmapFilter),
    Postings(PostingsFilter),
}

impl FilterIndex {
    /// Create new filter index
    pub fn new(attribute: &str, n_vectors: usize, policy: FilterPolicy) -> Self {
        Self {
            attribute: attribute.to_string(),
            policy,
            inverted: HashMap::new(),
            n_vectors,
        }
    }
    
    /// Add a value for a vector
    pub fn add(&mut self, vector_id: u32, value: &str) {
        let filter = self.inverted.entry(value.to_string()).or_insert_with(|| {
            match self.policy.representation {
                FilterRepresentation::RoaringBitmap => {
                    FilterData::Bitmap(BitmapFilter::new(self.n_vectors))
                }
                _ => FilterData::Postings(PostingsFilter::new()),
            }
        });
        
        match filter {
            FilterData::Bitmap(b) => b.set(vector_id),
            FilterData::Postings(p) => p.add(vector_id),
        }
    }
    
    /// Query for a value, returns matching vector IDs
    pub fn query(&self, value: &str) -> Option<Vec<u32>> {
        self.inverted.get(value).map(|filter| match filter {
            FilterData::Bitmap(b) => b.iter().collect(),
            FilterData::Postings(p) => p.iter().collect(),
        })
    }
    
    /// Query with selectivity estimate
    pub fn query_with_stats(&self, value: &str) -> (Option<Vec<u32>>, f32) {
        match self.inverted.get(value) {
            Some(filter) => {
                let ids: Vec<u32> = match filter {
                    FilterData::Bitmap(b) => b.iter().collect(),
                    FilterData::Postings(p) => p.iter().collect(),
                };
                let selectivity = ids.len() as f32 / self.n_vectors.max(1) as f32;
                (Some(ids), selectivity)
            }
            None => (None, 0.0),
        }
    }
    
    /// Check if a specific vector matches a value
    pub fn contains(&self, vector_id: u32, value: &str) -> bool {
        match self.inverted.get(value) {
            Some(FilterData::Bitmap(b)) => b.contains(vector_id),
            Some(FilterData::Postings(p)) => p.contains(vector_id),
            None => false,
        }
    }
    
    /// Get memory footprint
    pub fn memory_bytes(&self) -> usize {
        self.inverted.values().map(|f| match f {
            FilterData::Bitmap(b) => b.memory_bytes(),
            FilterData::Postings(p) => p.memory_bytes(),
        }).sum()
    }
    
    /// Get statistics
    pub fn stats(&self) -> FilterIndexStats {
        let n_values = self.inverted.len();
        let total_entries: usize = self.inverted.values().map(|f| match f {
            FilterData::Bitmap(b) => b.count(),
            FilterData::Postings(p) => p.count(),
        }).sum();
        
        FilterIndexStats {
            attribute: self.attribute.clone(),
            n_values,
            total_entries,
            memory_bytes: self.memory_bytes(),
            representation: self.policy.representation,
        }
    }
}

/// Statistics about a filter index
#[derive(Debug, Clone)]
pub struct FilterIndexStats {
    pub attribute: String,
    pub n_values: usize,
    pub total_entries: usize,
    pub memory_bytes: usize,
    pub representation: FilterRepresentation,
}

// ============================================================================
// Filter Index Manager
// ============================================================================

/// Manages multiple filter indexes
pub struct FilterIndexManager {
    indexes: HashMap<String, FilterIndex>,
    n_vectors: usize,
}

impl FilterIndexManager {
    /// Create new manager
    pub fn new(n_vectors: usize) -> Self {
        Self {
            indexes: HashMap::new(),
            n_vectors,
        }
    }
    
    /// Add or get an index for an attribute
    pub fn get_or_create(&mut self, attribute: &str, stats: &AttributeStats) -> &mut FilterIndex {
        let n_vectors = self.n_vectors;
        self.indexes.entry(attribute.to_string()).or_insert_with(|| {
            let policy = FilterPolicy::auto_select(stats, n_vectors);
            FilterIndex::new(attribute, n_vectors, policy)
        })
    }
    
    /// Add a value to an attribute's index
    pub fn add(&mut self, attribute: &str, vector_id: u32, value: &str) {
        if let Some(index) = self.indexes.get_mut(attribute) {
            index.add(vector_id, value);
        }
    }
    
    /// Query an attribute
    pub fn query(&self, attribute: &str, value: &str) -> Option<Vec<u32>> {
        self.indexes.get(attribute).and_then(|idx| idx.query(value))
    }
    
    /// Total memory footprint
    pub fn memory_bytes(&self) -> usize {
        self.indexes.values().map(|idx| idx.memory_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitmap_filter() {
        let mut bitmap = BitmapFilter::new(1000);
        
        bitmap.set(0);
        bitmap.set(10);
        bitmap.set(100);
        bitmap.set(999);
        
        assert!(bitmap.contains(0));
        assert!(bitmap.contains(10));
        assert!(!bitmap.contains(1));
        assert!(!bitmap.contains(500));
        
        assert_eq!(bitmap.count(), 4);
    }
    
    #[test]
    fn test_bitmap_intersection() {
        let mut a = BitmapFilter::new(100);
        let mut b = BitmapFilter::new(100);
        
        a.set(1);
        a.set(2);
        a.set(3);
        
        b.set(2);
        b.set(3);
        b.set(4);
        
        let c = a.and(&b);
        
        assert!(!c.contains(1));
        assert!(c.contains(2));
        assert!(c.contains(3));
        assert!(!c.contains(4));
    }
    
    #[test]
    fn test_postings_filter() {
        let mut postings = PostingsFilter::new();
        
        postings.add(5);
        postings.add(10);
        postings.add(3);
        postings.add(5); // Duplicate
        
        assert!(postings.contains(3));
        assert!(postings.contains(5));
        assert!(postings.contains(10));
        assert!(!postings.contains(7));
        
        assert_eq!(postings.count(), 3);
    }
    
    #[test]
    fn test_postings_intersection() {
        let a = PostingsFilter::from_ids(vec![1, 2, 3, 5, 7]);
        let b = PostingsFilter::from_ids(vec![2, 3, 6, 7, 8]);
        
        let c = a.intersect(&b);
        
        assert_eq!(c.count(), 3);
        assert!(c.contains(2));
        assert!(c.contains(3));
        assert!(c.contains(7));
    }
    
    #[test]
    fn test_policy_selection() {
        // Low cardinality, high density -> bitmap
        let stats1 = AttributeStats::new("status")
            .cardinality(5)
            .density(0.9);
        let policy1 = FilterPolicy::auto_select(&stats1, 1_000_000);
        assert_eq!(policy1.representation, FilterRepresentation::RoaringBitmap);
        
        // High cardinality, low density -> postings
        let stats2 = AttributeStats::new("user_id")
            .cardinality(500_000)
            .density(0.001);
        let policy2 = FilterPolicy::auto_select(&stats2, 1_000_000);
        assert_eq!(policy2.representation, FilterRepresentation::PostingsList);
    }
    
    #[test]
    fn test_filter_index() {
        let policy = FilterPolicy::bitmap();
        let mut index = FilterIndex::new("category", 1000, policy);
        
        index.add(0, "electronics");
        index.add(1, "electronics");
        index.add(2, "clothing");
        index.add(3, "electronics");
        
        let electronics = index.query("electronics").unwrap();
        assert_eq!(electronics.len(), 3);
        assert!(electronics.contains(&0));
        assert!(electronics.contains(&1));
        assert!(electronics.contains(&3));
        
        let clothing = index.query("clothing").unwrap();
        assert_eq!(clothing.len(), 1);
    }
}
