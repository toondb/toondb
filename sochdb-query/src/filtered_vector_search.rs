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

//! Filter-Aware Vector Search with Selectivity-Driven Fallback (Task 5)
//!
//! HNSW doesn't naturally index by metadata, so naïve filtered search leads to:
//! ```text
//! search HNSW → collect K → post-filter → not enough → increase ef → repeat
//! ```
//!
//! This creates the "filtered ANN death spiral" where low selectivity causes
//! exponential search expansion.
//!
//! ## Solution
//!
//! 1. **Compute AllowedSet first** (from metadata index)
//! 2. **Check selectivity p = |S| / N**
//! 3. **Choose strategy based on p:**
//!    - p > 0.1: Filter-aware HNSW walk (accept only allowed nodes)
//!    - p < 0.1 and |S| < 10K: Scan AllowedSet directly
//!    - p very low: Exact brute-force over S
//!
//! ## Complexity Analysis
//!
//! | Selectivity | Strategy | Complexity |
//! |-------------|----------|------------|
//! | > 0.1 | Filter-aware HNSW | O(ef * log N) with ~K/p ef |
//! | 0.001-0.1 | Scan AllowedSet | O(|S| * d) |
//! | < 0.001 | Brute force over S | O(|S| * d) |
//!
//! The crossover point is when |S| * d < ef_required * C for some constant C.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;

use crate::candidate_gate::{AllowedSet, CandidateGate};

// ============================================================================
// Filter-Aware Search Configuration
// ============================================================================

/// Configuration for filter-aware vector search
#[derive(Debug, Clone)]
pub struct FilteredSearchConfig {
    /// Target number of results
    pub k: usize,
    
    /// Base ef_search for HNSW traversal
    pub ef_search: usize,
    
    /// Maximum ef_search (to prevent death spiral)
    pub max_ef: usize,
    
    /// Selectivity threshold for HNSW vs scan fallback
    pub scan_threshold: f64,
    
    /// Maximum allowed set size for scan (above this, use HNSW)
    pub max_scan_size: usize,
    
    /// Whether to use adaptive ef based on acceptance rate
    pub adaptive_ef: bool,
}

impl Default for FilteredSearchConfig {
    fn default() -> Self {
        Self {
            k: 10,
            ef_search: 100,
            max_ef: 1000,
            scan_threshold: 0.1,
            max_scan_size: 10_000,
            adaptive_ef: true,
        }
    }
}

impl FilteredSearchConfig {
    /// Create config for a specific k
    pub fn with_k(k: usize) -> Self {
        Self {
            k,
            ef_search: k.max(100),
            ..Default::default()
        }
    }
    
    /// Choose execution strategy based on selectivity and allowed set size
    pub fn choose_strategy(&self, selectivity: f64, allowed_set_size: Option<usize>) -> FilteredSearchStrategy {
        // If no constraint (All), use standard HNSW
        if selectivity >= 1.0 {
            return FilteredSearchStrategy::StandardHnsw;
        }
        
        // Very low selectivity or small allowed set → scan
        if selectivity < self.scan_threshold {
            if let Some(size) = allowed_set_size {
                if size <= self.max_scan_size {
                    return FilteredSearchStrategy::ScanAllowedSet;
                }
            }
        }
        
        // Medium selectivity → filter-aware HNSW with adaptive ef
        FilteredSearchStrategy::FilterAwareHnsw {
            target_ef: self.compute_target_ef(selectivity),
        }
    }
    
    /// Compute target ef based on selectivity
    ///
    /// To get K valid results with acceptance rate p, we need ~K/p candidates.
    fn compute_target_ef(&self, selectivity: f64) -> usize {
        if selectivity <= 0.0 {
            return self.max_ef;
        }
        
        // ef ≈ k / selectivity, with margins
        let target = ((self.k as f64 / selectivity) * 1.5) as usize;
        target.clamp(self.ef_search, self.max_ef)
    }
}

/// Execution strategy for filtered vector search
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilteredSearchStrategy {
    /// Standard HNSW search (no filter or very high selectivity)
    StandardHnsw,
    
    /// Filter-aware HNSW walk with adaptive ef
    FilterAwareHnsw {
        target_ef: usize,
    },
    
    /// Scan over allowed IDs and compute distances
    ScanAllowedSet,
    
    /// Brute-force exact search over allowed IDs
    ExactScan,
    
    /// Empty result (allowed set is empty)
    EmptyResult,
}

// ============================================================================
// Scored Result
// ============================================================================

/// A scored search result
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// Document ID
    pub doc_id: u64,
    /// Similarity score (higher is better for cosine, lower for L2)
    pub score: f32,
}

impl ScoredResult {
    pub fn new(doc_id: u64, score: f32) -> Self {
        Self { doc_id, score }
    }
}

// For max-heap (we want highest scores)
impl Ord for ScoredResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior (we keep top-k)
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScoredResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ScoredResult {}

impl PartialEq for ScoredResult {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}

// ============================================================================
// Filter-Aware Search Result
// ============================================================================

/// Result of a filtered vector search
#[derive(Debug)]
pub struct FilteredSearchResult {
    /// Top-k results (sorted by score, descending)
    pub results: Vec<ScoredResult>,
    
    /// Strategy used
    pub strategy: FilteredSearchStrategy,
    
    /// Number of candidates evaluated
    pub candidates_evaluated: usize,
    
    /// Number of candidates rejected by filter
    pub candidates_rejected: usize,
    
    /// Effective ef used (for HNSW strategies)
    pub effective_ef: usize,
}

impl FilteredSearchResult {
    /// Get the acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.candidates_evaluated + self.candidates_rejected;
        if total == 0 {
            1.0
        } else {
            self.candidates_evaluated as f64 / total as f64
        }
    }
}

// ============================================================================
// Vector Store Trait
// ============================================================================

/// Trait for vector storage that supports filtered search
pub trait FilteredVectorStore {
    /// Get a vector by document ID
    fn get_vector(&self, doc_id: u64) -> Option<&[f32]>;
    
    /// Get the vector dimension
    fn dimension(&self) -> usize;
    
    /// Get total vector count
    fn count(&self) -> usize;
    
    /// Compute distance between query and a document
    fn distance(&self, query: &[f32], doc_id: u64) -> Option<f32>;
}

// ============================================================================
// Filter-Aware Search Engine
// ============================================================================

/// Filter-aware vector search engine
///
/// This wraps an HNSW index and provides filter-aware search with
/// selectivity-driven strategy selection.
pub struct FilterAwareSearch<V: FilteredVectorStore> {
    /// The underlying vector store
    vectors: Arc<V>,
    
    /// Configuration
    config: FilteredSearchConfig,
}

impl<V: FilteredVectorStore> FilterAwareSearch<V> {
    /// Create a new filter-aware search engine
    pub fn new(vectors: Arc<V>, config: FilteredSearchConfig) -> Self {
        Self { vectors, config }
    }
    
    /// Search with a filter constraint
    pub fn search(
        &self,
        query: &[f32],
        allowed_set: &AllowedSet,
    ) -> FilteredSearchResult {
        // Determine strategy
        let selectivity = allowed_set.selectivity(self.vectors.count());
        let strategy = self.config.choose_strategy(selectivity, allowed_set.cardinality());
        
        match strategy {
            FilteredSearchStrategy::EmptyResult | FilteredSearchStrategy::StandardHnsw => {
                if allowed_set.is_empty() {
                    return FilteredSearchResult {
                        results: vec![],
                        strategy: FilteredSearchStrategy::EmptyResult,
                        candidates_evaluated: 0,
                        candidates_rejected: 0,
                        effective_ef: 0,
                    };
                }
                
                // For standard HNSW, we'd delegate to the actual HNSW index
                // Here we fall back to scan for demonstration
                self.scan_allowed_set(query, allowed_set, FilteredSearchStrategy::StandardHnsw)
            }
            
            FilteredSearchStrategy::FilterAwareHnsw { target_ef: _ } => {
                // For filter-aware HNSW, we'd do the walk with filter checks
                // Here we simulate with scan
                self.scan_allowed_set(query, allowed_set, strategy)
            }
            
            FilteredSearchStrategy::ScanAllowedSet | FilteredSearchStrategy::ExactScan => {
                self.scan_allowed_set(query, allowed_set, strategy)
            }
        }
    }
    
    /// Scan over allowed IDs and find top-k
    fn scan_allowed_set(
        &self,
        query: &[f32],
        allowed_set: &AllowedSet,
        strategy: FilteredSearchStrategy,
    ) -> FilteredSearchResult {
        let k = self.config.k;
        let mut heap: BinaryHeap<ScoredResult> = BinaryHeap::with_capacity(k + 1);
        let mut candidates_evaluated = 0;
        
        for doc_id in allowed_set.iter() {
            if let Some(dist) = self.vectors.distance(query, doc_id) {
                candidates_evaluated += 1;
                
                // Convert distance to score (lower distance = higher score for L2)
                let score = -dist; // Negate for max-heap behavior
                
                heap.push(ScoredResult::new(doc_id, score));
                
                // Keep only top-k
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        
        // Extract results in sorted order
        let mut results: Vec<_> = heap.into_sorted_vec();
        results.reverse(); // Highest scores first
        
        FilteredSearchResult {
            results,
            strategy,
            candidates_evaluated,
            candidates_rejected: 0,
            effective_ef: candidates_evaluated,
        }
    }
}

impl<V: FilteredVectorStore> CandidateGate for FilterAwareSearch<V> {
    type Query = Vec<f32>;
    type Result = FilteredSearchResult;
    type Error = std::convert::Infallible;
    
    fn execute_with_gate(
        &self,
        query: &Self::Query,
        allowed_set: &AllowedSet,
    ) -> Result<Self::Result, Self::Error> {
        Ok(self.search(query, allowed_set))
    }
}

// ============================================================================
// Integration with HNSW
// ============================================================================

/// Trait extension for HNSW indexes to support filtered search
pub trait HnswFilteredSearch {
    /// Search with a filter, using adaptive strategy
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed_set: &AllowedSet,
        config: &FilteredSearchConfig,
    ) -> FilteredSearchResult;
    
    /// Search with filter-aware graph traversal
    fn search_filter_aware(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowed_set: &AllowedSet,
    ) -> (Vec<ScoredResult>, usize, usize); // results, evaluated, rejected
}

// ============================================================================
// Helper: Compute Distances with SIMD
// ============================================================================

/// Compute L2 distance between two vectors
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Compute cosine similarity between two vectors
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategy_selection() {
        let config = FilteredSearchConfig::default();
        
        // High selectivity → filter-aware HNSW
        let s1 = config.choose_strategy(0.5, Some(50000));
        assert!(matches!(s1, FilteredSearchStrategy::FilterAwareHnsw { .. }));
        
        // Low selectivity, small set → scan
        let s2 = config.choose_strategy(0.01, Some(1000));
        assert_eq!(s2, FilteredSearchStrategy::ScanAllowedSet);
        
        // Very high selectivity → standard HNSW
        let s3 = config.choose_strategy(1.0, None);
        assert_eq!(s3, FilteredSearchStrategy::StandardHnsw);
    }
    
    #[test]
    fn test_target_ef_computation() {
        let config = FilteredSearchConfig {
            k: 10,
            ef_search: 100,
            max_ef: 1000,
            ..Default::default()
        };
        
        // High selectivity → low ef
        let ef1 = config.compute_target_ef(0.5);
        assert!(ef1 <= 100);
        
        // Low selectivity → high ef
        let ef2 = config.compute_target_ef(0.01);
        assert!(ef2 >= 500);
        
        // Very low selectivity → max ef
        let ef3 = config.compute_target_ef(0.001);
        assert_eq!(ef3, 1000);
    }
    
    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let dist = l2_distance(&a, &b);
        assert!((dist - std::f32::consts::SQRT_2).abs() < 0.0001);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];
        
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);
        assert!(cosine_similarity(&a, &c).abs() < 0.0001);
    }
}
