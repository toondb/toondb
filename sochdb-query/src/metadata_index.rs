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

//! Metadata Index Primitives (Task 3)
//!
//! This module provides fast set algebra over document IDs for filter pushdown.
//! Instead of post-filtering huge candidate sets, we compute `AllowedSet` once
//! and reuse it across all retrieval paths.
//!
//! ## Index Types
//!
//! | Field Type | Index Structure | Best For |
//! |------------|-----------------|----------|
//! | Equality (namespace, source) | Value → Bitmap | Exact match, IN |
//! | Time range (timestamp) | Sorted (t, doc_id) array | Range queries |
//! | Doc ID set | Direct bitmap/sorted vec | Explicit ID lists |
//!
//! ## Complexity Analysis
//!
//! - Bitmap intersection: O(min(|A|, |B|)) with roaring-style containers
//! - Membership check: O(1) for bitmap
//! - Range lookup: O(log n) to locate endpoints + O(k) for k results
//!
//! ## Memory Efficiency
//!
//! For the local-first use case (typical corpus 10K-1M docs):
//! - Bitmap: ~125KB for 1M docs (1 bit per doc)
//! - Sorted array: 8 bytes per doc in result set
//!
//! ## Usage
//!
//! ```ignore
//! // Build indexes during ingestion
//! let mut idx = MetadataIndex::new();
//! idx.add_equality("namespace", "production", doc_id);
//! idx.add_timestamp("created_at", 1703980800, doc_id);
//!
//! // Query time: compute AllowedSet from FilterIR
//! let allowed = idx.evaluate(&filter_ir)?;
//! ```

use std::collections::{BTreeMap, HashMap};
use std::sync::RwLock;

use crate::candidate_gate::AllowedSet;
use crate::filter_ir::{FilterAtom, FilterIR, FilterValue};

// ============================================================================
// Posting Set - Doc IDs for a single value
// ============================================================================

/// A set of document IDs for a single indexed value
#[derive(Debug, Clone)]
pub struct PostingSet {
    /// Document IDs (sorted)
    doc_ids: Vec<u64>,
}

impl PostingSet {
    /// Create a new empty posting set
    pub fn new() -> Self {
        Self { doc_ids: Vec::new() }
    }
    
    /// Create from a vec of doc IDs
    pub fn from_vec(mut ids: Vec<u64>) -> Self {
        ids.sort_unstable();
        ids.dedup();
        Self { doc_ids: ids }
    }
    
    /// Add a document ID
    pub fn add(&mut self, doc_id: u64) {
        // Maintain sorted order
        match self.doc_ids.binary_search(&doc_id) {
            Ok(_) => {} // Already present
            Err(pos) => self.doc_ids.insert(pos, doc_id),
        }
    }
    
    /// Remove a document ID
    pub fn remove(&mut self, doc_id: u64) {
        if let Ok(pos) = self.doc_ids.binary_search(&doc_id) {
            self.doc_ids.remove(pos);
        }
    }
    
    /// Check if contains a doc ID
    pub fn contains(&self, doc_id: u64) -> bool {
        self.doc_ids.binary_search(&doc_id).is_ok()
    }
    
    /// Get count
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }
    
    /// Convert to AllowedSet
    pub fn to_allowed_set(&self) -> AllowedSet {
        if self.doc_ids.is_empty() {
            AllowedSet::None
        } else {
            AllowedSet::from_sorted_vec(self.doc_ids.clone())
        }
    }
    
    /// Intersect with another posting set
    pub fn intersect(&self, other: &PostingSet) -> PostingSet {
        let mut result = Vec::with_capacity(self.doc_ids.len().min(other.doc_ids.len()));
        let mut i = 0;
        let mut j = 0;
        
        while i < self.doc_ids.len() && j < other.doc_ids.len() {
            match self.doc_ids[i].cmp(&other.doc_ids[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(self.doc_ids[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        
        PostingSet { doc_ids: result }
    }
    
    /// Union with another posting set
    pub fn union(&self, other: &PostingSet) -> PostingSet {
        let mut result = Vec::with_capacity(self.doc_ids.len() + other.doc_ids.len());
        let mut i = 0;
        let mut j = 0;
        
        while i < self.doc_ids.len() && j < other.doc_ids.len() {
            match self.doc_ids[i].cmp(&other.doc_ids[j]) {
                std::cmp::Ordering::Less => {
                    result.push(self.doc_ids[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(other.doc_ids[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(self.doc_ids[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        
        result.extend_from_slice(&self.doc_ids[i..]);
        result.extend_from_slice(&other.doc_ids[j..]);
        
        PostingSet { doc_ids: result }
    }
    
    /// Get iterator
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.doc_ids.iter().copied()
    }
}

impl Default for PostingSet {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Equality Index - Value → Posting Set
// ============================================================================

/// Index for equality lookups (namespace, source, project_id, etc.)
#[derive(Debug, Default)]
pub struct EqualityIndex {
    /// Map from string value to posting set
    string_postings: HashMap<String, PostingSet>,
    /// Map from integer value to posting set
    int_postings: HashMap<i64, PostingSet>,
    /// Map from uint value to posting set
    uint_postings: HashMap<u64, PostingSet>,
}

impl EqualityIndex {
    /// Create a new equality index
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a document with a string value
    pub fn add_string(&mut self, value: &str, doc_id: u64) {
        self.string_postings
            .entry(value.to_string())
            .or_default()
            .add(doc_id);
    }
    
    /// Add a document with an integer value
    pub fn add_int(&mut self, value: i64, doc_id: u64) {
        self.int_postings
            .entry(value)
            .or_default()
            .add(doc_id);
    }
    
    /// Add a document with an unsigned integer value
    pub fn add_uint(&mut self, value: u64, doc_id: u64) {
        self.uint_postings
            .entry(value)
            .or_default()
            .add(doc_id);
    }
    
    /// Remove a document with a string value
    pub fn remove_string(&mut self, value: &str, doc_id: u64) {
        if let Some(posting) = self.string_postings.get_mut(value) {
            posting.remove(doc_id);
            if posting.is_empty() {
                self.string_postings.remove(value);
            }
        }
    }
    
    /// Lookup documents with a string value
    pub fn lookup_string(&self, value: &str) -> AllowedSet {
        self.string_postings
            .get(value)
            .map(|p| p.to_allowed_set())
            .unwrap_or(AllowedSet::None)
    }
    
    /// Lookup documents with an integer value
    pub fn lookup_int(&self, value: i64) -> AllowedSet {
        self.int_postings
            .get(&value)
            .map(|p| p.to_allowed_set())
            .unwrap_or(AllowedSet::None)
    }
    
    /// Lookup documents with a uint value
    pub fn lookup_uint(&self, value: u64) -> AllowedSet {
        self.uint_postings
            .get(&value)
            .map(|p| p.to_allowed_set())
            .unwrap_or(AllowedSet::None)
    }
    
    /// Lookup documents in a set of string values (OR)
    pub fn lookup_string_in(&self, values: &[String]) -> AllowedSet {
        let sets: Vec<_> = values.iter()
            .filter_map(|v| self.string_postings.get(v))
            .collect();
        
        if sets.is_empty() {
            return AllowedSet::None;
        }
        
        // Union all posting sets
        let mut result = sets[0].clone();
        for set in &sets[1..] {
            result = result.union(set);
        }
        
        result.to_allowed_set()
    }
    
    /// Lookup documents in a set of uint values (OR)
    pub fn lookup_uint_in(&self, values: &[u64]) -> AllowedSet {
        let sets: Vec<_> = values.iter()
            .filter_map(|v| self.uint_postings.get(v))
            .collect();
        
        if sets.is_empty() {
            return AllowedSet::None;
        }
        
        let mut result = sets[0].clone();
        for set in &sets[1..] {
            result = result.union(set);
        }
        
        result.to_allowed_set()
    }
    
    /// Get all unique values for this field
    pub fn string_values(&self) -> impl Iterator<Item = &str> {
        self.string_postings.keys().map(|s| s.as_str())
    }
    
    /// Get statistics
    pub fn stats(&self) -> EqualityIndexStats {
        EqualityIndexStats {
            unique_string_values: self.string_postings.len(),
            unique_int_values: self.int_postings.len(),
            unique_uint_values: self.uint_postings.len(),
            total_postings: self.string_postings.values().map(|p| p.len()).sum::<usize>()
                + self.int_postings.values().map(|p| p.len()).sum::<usize>()
                + self.uint_postings.values().map(|p| p.len()).sum::<usize>(),
        }
    }
}

/// Statistics for an equality index
#[derive(Debug, Clone)]
pub struct EqualityIndexStats {
    pub unique_string_values: usize,
    pub unique_int_values: usize,
    pub unique_uint_values: usize,
    pub total_postings: usize,
}

// ============================================================================
// Range Index - Sorted (value, doc_id) for range queries
// ============================================================================

/// Index for range queries (timestamp, score, etc.)
#[derive(Debug, Default)]
pub struct RangeIndex {
    /// Sorted entries: (value, doc_id)
    /// Using BTreeMap for efficient range queries
    entries: BTreeMap<i64, PostingSet>,
    /// Total document count
    doc_count: usize,
}

impl RangeIndex {
    /// Create a new range index
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a document with a timestamp/value
    pub fn add(&mut self, value: i64, doc_id: u64) {
        self.entries.entry(value).or_default().add(doc_id);
        self.doc_count += 1;
    }
    
    /// Add a document with an unsigned value
    pub fn add_uint(&mut self, value: u64, doc_id: u64) {
        self.add(value as i64, doc_id);
    }
    
    /// Remove a document
    pub fn remove(&mut self, value: i64, doc_id: u64) {
        if let Some(posting) = self.entries.get_mut(&value) {
            posting.remove(doc_id);
            if posting.is_empty() {
                self.entries.remove(&value);
            }
            self.doc_count -= 1;
        }
    }
    
    /// Query a range [min, max] (inclusive)
    pub fn range_query(
        &self,
        min: Option<i64>,
        max: Option<i64>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> AllowedSet {
        use std::ops::Bound;
        
        let start = match min {
            Some(v) if min_inclusive => Bound::Included(v),
            Some(v) => Bound::Excluded(v),
            None => Bound::Unbounded,
        };
        
        let end = match max {
            Some(v) if max_inclusive => Bound::Included(v),
            Some(v) => Bound::Excluded(v),
            None => Bound::Unbounded,
        };
        
        // Collect all doc IDs in range
        let mut result = PostingSet::new();
        for (_, posting) in self.entries.range((start, end)) {
            result = result.union(posting);
        }
        
        result.to_allowed_set()
    }
    
    /// Query for values greater than a threshold
    pub fn greater_than(&self, value: i64, inclusive: bool) -> AllowedSet {
        self.range_query(Some(value), None, inclusive, true)
    }
    
    /// Query for values less than a threshold
    pub fn less_than(&self, value: i64, inclusive: bool) -> AllowedSet {
        self.range_query(None, Some(value), true, inclusive)
    }
    
    /// Get statistics
    pub fn stats(&self) -> RangeIndexStats {
        let values: Vec<_> = self.entries.keys().collect();
        RangeIndexStats {
            unique_values: self.entries.len(),
            total_docs: self.doc_count,
            min_value: values.first().copied().copied(),
            max_value: values.last().copied().copied(),
        }
    }
}

/// Statistics for a range index
#[derive(Debug, Clone)]
pub struct RangeIndexStats {
    pub unique_values: usize,
    pub total_docs: usize,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
}

// ============================================================================
// Composite Metadata Index
// ============================================================================

/// Composite metadata index supporting multiple field types
#[derive(Debug, Default)]
pub struct MetadataIndex {
    /// Equality indexes by field name
    equality_indexes: HashMap<String, EqualityIndex>,
    /// Range indexes by field name
    range_indexes: HashMap<String, RangeIndex>,
    /// Total document count
    doc_count: usize,
}

impl MetadataIndex {
    /// Create a new metadata index
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an equality field value
    pub fn add_equality(&mut self, field: &str, value: &FilterValue, doc_id: u64) {
        let index = self.equality_indexes
            .entry(field.to_string())
            .or_default();
        
        match value {
            FilterValue::String(s) => index.add_string(s, doc_id),
            FilterValue::Int64(i) => index.add_int(*i, doc_id),
            FilterValue::Uint64(u) => index.add_uint(*u, doc_id),
            _ => {} // Ignore other types
        }
    }
    
    /// Add a string equality field
    pub fn add_string(&mut self, field: &str, value: &str, doc_id: u64) {
        self.equality_indexes
            .entry(field.to_string())
            .or_default()
            .add_string(value, doc_id);
    }
    
    /// Add a range field value (for time, scores, etc.)
    pub fn add_range(&mut self, field: &str, value: i64, doc_id: u64) {
        self.range_indexes
            .entry(field.to_string())
            .or_default()
            .add(value, doc_id);
    }
    
    /// Add a timestamp (convenience for u64 timestamps)
    pub fn add_timestamp(&mut self, field: &str, timestamp: u64, doc_id: u64) {
        self.add_range(field, timestamp as i64, doc_id);
    }
    
    /// Update document count
    pub fn set_doc_count(&mut self, count: usize) {
        self.doc_count = count;
    }
    
    /// Increment document count
    pub fn inc_doc_count(&mut self) {
        self.doc_count += 1;
    }
    
    /// Get document count
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }
    
    /// Evaluate a filter atom
    pub fn evaluate_atom(&self, atom: &FilterAtom) -> AllowedSet {
        match atom {
            FilterAtom::Eq { field, value } => {
                if let Some(index) = self.equality_indexes.get(field) {
                    match value {
                        FilterValue::String(s) => index.lookup_string(s),
                        FilterValue::Int64(i) => index.lookup_int(*i),
                        FilterValue::Uint64(u) => index.lookup_uint(*u),
                        _ => AllowedSet::All, // Unknown type, don't filter
                    }
                } else {
                    AllowedSet::All // No index, can't filter
                }
            }
            
            FilterAtom::In { field, values } => {
                if let Some(index) = self.equality_indexes.get(field) {
                    // Check if all values are strings
                    let strings: Vec<String> = values.iter()
                        .filter_map(|v| match v {
                            FilterValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                    
                    if strings.len() == values.len() {
                        return index.lookup_string_in(&strings);
                    }
                    
                    // Check if all values are uints
                    let uints: Vec<u64> = values.iter()
                        .filter_map(|v| match v {
                            FilterValue::Uint64(u) => Some(*u),
                            _ => None,
                        })
                        .collect();
                    
                    if uints.len() == values.len() {
                        return index.lookup_uint_in(&uints);
                    }
                }
                AllowedSet::All // Can't evaluate with index
            }
            
            FilterAtom::Range { field, min, max, min_inclusive, max_inclusive } => {
                if let Some(index) = self.range_indexes.get(field) {
                    let min_val = min.as_ref().and_then(|v| match v {
                        FilterValue::Int64(i) => Some(*i),
                        FilterValue::Uint64(u) => Some(*u as i64),
                        _ => None,
                    });
                    let max_val = max.as_ref().and_then(|v| match v {
                        FilterValue::Int64(i) => Some(*i),
                        FilterValue::Uint64(u) => Some(*u as i64),
                        _ => None,
                    });
                    
                    index.range_query(min_val, max_val, *min_inclusive, *max_inclusive)
                } else {
                    AllowedSet::All
                }
            }
            
            FilterAtom::True => AllowedSet::All,
            FilterAtom::False => AllowedSet::None,
            
            // Other atoms fall through to All (post-filter if needed)
            _ => AllowedSet::All,
        }
    }
    
    /// Evaluate a complete filter IR
    ///
    /// This is the main entry point for computing AllowedSet from FilterIR.
    pub fn evaluate(&self, filter: &FilterIR) -> AllowedSet {
        if filter.is_all() {
            return AllowedSet::All;
        }
        if filter.is_none() {
            return AllowedSet::None;
        }
        
        // Start with All, intersect with each clause
        let mut result = AllowedSet::All;
        
        for clause in &filter.clauses {
            // Evaluate disjunction: OR of atoms
            let clause_result = self.evaluate_disjunction(clause);
            
            // Intersect with running result (AND of clauses)
            result = result.intersect(&clause_result);
            
            // Short-circuit if empty
            if result.is_empty() {
                return AllowedSet::None;
            }
        }
        
        result
    }
    
    /// Evaluate a disjunction (OR of atoms)
    fn evaluate_disjunction(&self, clause: &crate::filter_ir::Disjunction) -> AllowedSet {
        if clause.atoms.len() == 1 {
            return self.evaluate_atom(&clause.atoms[0]);
        }
        
        // Union all atom results
        let mut result = AllowedSet::None;
        for atom in &clause.atoms {
            let atom_result = self.evaluate_atom(atom);
            result = result.union(&atom_result);
            
            // Short-circuit if All
            if result.is_all() {
                return AllowedSet::All;
            }
        }
        
        result
    }
    
    /// Get selectivity estimate for a filter
    pub fn estimate_selectivity(&self, filter: &FilterIR) -> f64 {
        if self.doc_count == 0 {
            return 1.0;
        }
        
        let allowed = self.evaluate(filter);
        allowed.selectivity(self.doc_count)
    }
}

// ============================================================================
// Thread-Safe Wrapper
// ============================================================================

/// Thread-safe metadata index
pub struct ConcurrentMetadataIndex {
    inner: RwLock<MetadataIndex>,
}

impl ConcurrentMetadataIndex {
    /// Create a new concurrent index
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(MetadataIndex::new()),
        }
    }
    
    /// Add a string field
    pub fn add_string(&self, field: &str, value: &str, doc_id: u64) {
        self.inner.write().unwrap().add_string(field, value, doc_id);
    }
    
    /// Add a timestamp
    pub fn add_timestamp(&self, field: &str, timestamp: u64, doc_id: u64) {
        self.inner.write().unwrap().add_timestamp(field, timestamp, doc_id);
    }
    
    /// Evaluate a filter
    pub fn evaluate(&self, filter: &FilterIR) -> AllowedSet {
        self.inner.read().unwrap().evaluate(filter)
    }
    
    /// Update document count
    pub fn set_doc_count(&self, count: usize) {
        self.inner.write().unwrap().set_doc_count(count);
    }
}

impl Default for ConcurrentMetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_ir::FilterBuilder;
    
    #[test]
    fn test_posting_set_basic() {
        let mut ps = PostingSet::new();
        ps.add(1);
        ps.add(5);
        ps.add(3);
        
        assert!(ps.contains(1));
        assert!(ps.contains(3));
        assert!(ps.contains(5));
        assert!(!ps.contains(2));
        assert_eq!(ps.len(), 3);
    }
    
    #[test]
    fn test_posting_set_intersection() {
        let a = PostingSet::from_vec(vec![1, 2, 3, 4, 5]);
        let b = PostingSet::from_vec(vec![3, 4, 5, 6, 7]);
        
        let c = a.intersect(&b);
        assert_eq!(c.len(), 3);
        assert!(c.contains(3));
        assert!(c.contains(4));
        assert!(c.contains(5));
    }
    
    #[test]
    fn test_equality_index() {
        let mut idx = EqualityIndex::new();
        idx.add_string("production", 1);
        idx.add_string("production", 2);
        idx.add_string("staging", 3);
        
        let result = idx.lookup_string("production");
        assert_eq!(result.cardinality(), Some(2));
        
        let result2 = idx.lookup_string("staging");
        assert_eq!(result2.cardinality(), Some(1));
        
        let result3 = idx.lookup_string("dev");
        assert!(result3.is_empty());
    }
    
    #[test]
    fn test_range_index() {
        let mut idx = RangeIndex::new();
        idx.add(100, 1);
        idx.add(200, 2);
        idx.add(300, 3);
        idx.add(400, 4);
        idx.add(500, 5);
        
        // Range [200, 400]
        let result = idx.range_query(Some(200), Some(400), true, true);
        assert_eq!(result.cardinality(), Some(3));
        
        // Greater than 300
        let result2 = idx.greater_than(300, false);
        assert_eq!(result2.cardinality(), Some(2));
        
        // Less than 300
        let result3 = idx.less_than(300, true);
        assert_eq!(result3.cardinality(), Some(3));
    }
    
    #[test]
    fn test_metadata_index_evaluation() {
        let mut idx = MetadataIndex::new();
        
        // Add documents
        for i in 0..10 {
            idx.add_string("namespace", "production", i);
            idx.add_timestamp("created_at", 1000 + i * 100, i);
        }
        for i in 10..20 {
            idx.add_string("namespace", "staging", i);
            idx.add_timestamp("created_at", 1000 + i * 100, i);
        }
        idx.set_doc_count(20);
        
        // Filter: namespace = production
        let filter = FilterBuilder::new()
            .namespace("production")
            .build();
        
        let result = idx.evaluate(&filter);
        assert_eq!(result.cardinality(), Some(10));
        
        // Filter: namespace = production AND created_at >= 1500
        let filter2 = FilterBuilder::new()
            .namespace("production")
            .gte("created_at", 1500i64)
            .build();
        
        let result2 = idx.evaluate(&filter2);
        // Docs 5-9 have timestamps 1500-1900
        assert_eq!(result2.cardinality(), Some(5));
    }
    
    #[test]
    fn test_selectivity_estimate() {
        let mut idx = MetadataIndex::new();
        
        for i in 0..100 {
            let ns = if i % 10 == 0 { "rare" } else { "common" };
            idx.add_string("namespace", ns, i);
        }
        idx.set_doc_count(100);
        
        let common_filter = FilterBuilder::new().namespace("common").build();
        let rare_filter = FilterBuilder::new().namespace("rare").build();
        
        let common_selectivity = idx.estimate_selectivity(&common_filter);
        let rare_selectivity = idx.estimate_selectivity(&rare_filter);
        
        assert!(common_selectivity > rare_selectivity);
        assert!((common_selectivity - 0.9).abs() < 0.01);
        assert!((rare_selectivity - 0.1).abs() < 0.01);
    }
}
