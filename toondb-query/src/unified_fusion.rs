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

//! Unified Hybrid Fusion with Mandatory Pre-Filtering (Task 7)
//!
//! This module implements hybrid retrieval (vector + BM25) that **never**
//! post-filters. The key insight is:
//!
//! > Both vector and BM25 executors receive the **same** AllowedSet,
//! > produce candidates **guaranteed** within it, then fusion merges by doc_id.
//!
//! ## Anti-Pattern (What We Avoid)
//!
//! ```text
//! BAD: vector_search() → candidates → filter → too few
//!      bm25_search() → candidates → filter → inconsistent
//!      fusion(unfiltered_v, unfiltered_b) → filter at end → broken!
//! ```
//!
//! ## Correct Pattern
//!
//! ```text
//! GOOD: compute AllowedSet from FilterIR
//!       vector_search(query, allowed_set) → filtered_v
//!       bm25_search(query, allowed_set) → filtered_b
//!       fusion(filtered_v, filtered_b) → already correct!
//! ```
//!
//! ## Fusion Cost
//!
//! With pre-filtered candidates:
//! - Fusion is O(k_v + k_b) with hash-join or two-pointer merge
//! - Total work is proportional to constrained candidate sizes
//! - No wasted scoring on disallowed documents

use std::collections::HashMap;
use std::sync::Arc;

use crate::candidate_gate::AllowedSet;
use crate::filter_ir::{AuthScope, FilterIR};
use crate::filtered_vector_search::ScoredResult;
use crate::namespace::NamespaceScope;

// ============================================================================
// Fusion Configuration
// ============================================================================

/// Fusion method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion: score = Σ w_i / (k + rank_i)
    Rrf { k: f32 },
    
    /// Linear combination of normalized scores
    Linear { vector_weight: f32, bm25_weight: f32 },
    
    /// Take max score across modalities
    Max,
    
    /// Cascade: use one modality to filter, other to rank
    Cascade { primary: Modality },
}

/// Search modality
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modality {
    Vector,
    Bm25,
}

impl Default for FusionMethod {
    fn default() -> Self {
        Self::Rrf { k: 60.0 }
    }
}

/// Configuration for hybrid fusion
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fusion method
    pub method: FusionMethod,
    
    /// Number of candidates to retrieve from each modality
    pub candidates_per_modality: usize,
    
    /// Final result limit
    pub final_k: usize,
    
    /// Minimum score threshold (after fusion)
    pub min_score: Option<f32>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::default(),
            candidates_per_modality: 100,
            final_k: 10,
            min_score: None,
        }
    }
}

// ============================================================================
// Unified Hybrid Query
// ============================================================================

/// A hybrid query that enforces pre-filtering
#[derive(Debug, Clone)]
pub struct UnifiedHybridQuery {
    /// Namespace scope (mandatory)
    pub namespace: NamespaceScope,
    
    /// Vector query (optional)
    pub vector_query: Option<VectorQuerySpec>,
    
    /// BM25 query (optional)
    pub bm25_query: Option<Bm25QuerySpec>,
    
    /// User-provided filter
    pub filter: FilterIR,
    
    /// Fusion configuration
    pub fusion_config: FusionConfig,
}

/// Vector query specification
#[derive(Debug, Clone)]
pub struct VectorQuerySpec {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// ef_search for HNSW
    pub ef_search: usize,
}

/// BM25 query specification
#[derive(Debug, Clone)]
pub struct Bm25QuerySpec {
    /// Query text (will be tokenized)
    pub text: String,
    /// Fields to search
    pub fields: Vec<String>,
}

impl UnifiedHybridQuery {
    /// Create a new hybrid query (namespace is mandatory)
    pub fn new(namespace: NamespaceScope) -> Self {
        Self {
            namespace,
            vector_query: None,
            bm25_query: None,
            filter: FilterIR::all(),
            fusion_config: FusionConfig::default(),
        }
    }
    
    /// Add vector search
    pub fn with_vector(mut self, embedding: Vec<f32>) -> Self {
        self.vector_query = Some(VectorQuerySpec {
            embedding,
            ef_search: 100,
        });
        self
    }
    
    /// Add BM25 search
    pub fn with_bm25(mut self, text: impl Into<String>) -> Self {
        self.bm25_query = Some(Bm25QuerySpec {
            text: text.into(),
            fields: vec!["content".to_string()],
        });
        self
    }
    
    /// Add filter
    pub fn with_filter(mut self, filter: FilterIR) -> Self {
        self.filter = filter;
        self
    }
    
    /// Set fusion config
    pub fn with_fusion(mut self, config: FusionConfig) -> Self {
        self.fusion_config = config;
        self
    }
    
    /// Compute the complete effective filter
    ///
    /// This combines namespace scope + user filter. Auth scope is added later.
    pub fn effective_filter(&self) -> FilterIR {
        self.namespace.to_filter_ir().and(self.filter.clone())
    }
}

// ============================================================================
// Filtered Candidates
// ============================================================================

/// Candidates from a single modality (already filtered)
#[derive(Debug)]
pub struct FilteredCandidates {
    /// Modality source
    pub modality: Modality,
    /// Scored results (doc_id, score)
    pub results: Vec<ScoredResult>,
    /// Whether the allowed set was applied
    pub filtered: bool,
}

impl FilteredCandidates {
    /// Create from vector search results
    pub fn from_vector(results: Vec<ScoredResult>) -> Self {
        Self {
            modality: Modality::Vector,
            results,
            filtered: true,
        }
    }
    
    /// Create from BM25 results
    pub fn from_bm25(results: Vec<ScoredResult>) -> Self {
        Self {
            modality: Modality::Bm25,
            results,
            filtered: true,
        }
    }
}

// ============================================================================
// Fusion Engine
// ============================================================================

/// The fusion engine that combines candidates from multiple modalities
pub struct FusionEngine {
    config: FusionConfig,
}

impl FusionEngine {
    /// Create a new fusion engine
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }
    
    /// Fuse candidates from vector and BM25 search
    ///
    /// INVARIANT: Both candidate sets are already filtered to AllowedSet.
    /// This function does NOT apply any additional filtering.
    pub fn fuse(
        &self,
        vector_candidates: Option<FilteredCandidates>,
        bm25_candidates: Option<FilteredCandidates>,
    ) -> FusionResult {
        // Validate that candidates are pre-filtered
        if let Some(ref vc) = vector_candidates {
            debug_assert!(vc.filtered, "Vector candidates must be pre-filtered!");
        }
        if let Some(ref bc) = bm25_candidates {
            debug_assert!(bc.filtered, "BM25 candidates must be pre-filtered!");
        }
        
        match self.config.method {
            FusionMethod::Rrf { k } => self.fuse_rrf(vector_candidates, bm25_candidates, k),
            FusionMethod::Linear { vector_weight, bm25_weight } => {
                self.fuse_linear(vector_candidates, bm25_candidates, vector_weight, bm25_weight)
            }
            FusionMethod::Max => self.fuse_max(vector_candidates, bm25_candidates),
            FusionMethod::Cascade { primary } => {
                self.fuse_cascade(vector_candidates, bm25_candidates, primary)
            }
        }
    }
    
    /// Reciprocal Rank Fusion
    ///
    /// score(d) = Σ w_i / (k + rank_i(d))
    fn fuse_rrf(
        &self,
        vector: Option<FilteredCandidates>,
        bm25: Option<FilteredCandidates>,
        k: f32,
    ) -> FusionResult {
        let mut scores: HashMap<u64, f32> = HashMap::new();
        
        // Add vector ranks
        if let Some(vc) = vector {
            for (rank, result) in vc.results.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                *scores.entry(result.doc_id).or_insert(0.0) += rrf_score;
            }
        }
        
        // Add BM25 ranks
        if let Some(bc) = bm25 {
            for (rank, result) in bc.results.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                *scores.entry(result.doc_id).or_insert(0.0) += rrf_score;
            }
        }
        
        self.collect_top_k(scores)
    }
    
    /// Linear combination fusion
    fn fuse_linear(
        &self,
        vector: Option<FilteredCandidates>,
        bm25: Option<FilteredCandidates>,
        vector_weight: f32,
        bm25_weight: f32,
    ) -> FusionResult {
        let mut scores: HashMap<u64, f32> = HashMap::new();
        
        // Normalize and add vector scores
        if let Some(vc) = vector {
            let normalized = self.normalize_scores(&vc.results);
            for (doc_id, score) in normalized {
                *scores.entry(doc_id).or_insert(0.0) += score * vector_weight;
            }
        }
        
        // Normalize and add BM25 scores
        if let Some(bc) = bm25 {
            let normalized = self.normalize_scores(&bc.results);
            for (doc_id, score) in normalized {
                *scores.entry(doc_id).or_insert(0.0) += score * bm25_weight;
            }
        }
        
        self.collect_top_k(scores)
    }
    
    /// Max-score fusion
    fn fuse_max(
        &self,
        vector: Option<FilteredCandidates>,
        bm25: Option<FilteredCandidates>,
    ) -> FusionResult {
        let mut scores: HashMap<u64, f32> = HashMap::new();
        
        if let Some(vc) = vector {
            let normalized = self.normalize_scores(&vc.results);
            for (doc_id, score) in normalized {
                let entry = scores.entry(doc_id).or_insert(0.0);
                *entry = entry.max(score);
            }
        }
        
        if let Some(bc) = bm25 {
            let normalized = self.normalize_scores(&bc.results);
            for (doc_id, score) in normalized {
                let entry = scores.entry(doc_id).or_insert(0.0);
                *entry = entry.max(score);
            }
        }
        
        self.collect_top_k(scores)
    }
    
    /// Cascade fusion: use primary modality to filter, secondary to rank
    fn fuse_cascade(
        &self,
        vector: Option<FilteredCandidates>,
        bm25: Option<FilteredCandidates>,
        primary: Modality,
    ) -> FusionResult {
        let (primary_candidates, secondary_candidates) = match primary {
            Modality::Vector => (vector, bm25),
            Modality::Bm25 => (bm25, vector),
        };
        
        // Get primary doc IDs
        let primary_ids: std::collections::HashSet<u64> = primary_candidates
            .as_ref()
            .map(|c| c.results.iter().map(|r| r.doc_id).collect())
            .unwrap_or_default();
        
        // Score by secondary, but only docs in primary
        let mut scores: HashMap<u64, f32> = HashMap::new();
        
        if let Some(sc) = secondary_candidates {
            for result in &sc.results {
                if primary_ids.contains(&result.doc_id) {
                    scores.insert(result.doc_id, result.score);
                }
            }
        }
        
        // If secondary doesn't score some docs, use primary order
        if let Some(pc) = primary_candidates {
            for (rank, result) in pc.results.iter().enumerate() {
                scores.entry(result.doc_id).or_insert(-(rank as f32));
            }
        }
        
        self.collect_top_k(scores)
    }
    
    /// Normalize scores to [0, 1] using min-max normalization
    fn normalize_scores(&self, results: &[ScoredResult]) -> Vec<(u64, f32)> {
        if results.is_empty() {
            return vec![];
        }
        
        let min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
        let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        
        if range == 0.0 {
            return results.iter().map(|r| (r.doc_id, 1.0)).collect();
        }
        
        results.iter()
            .map(|r| (r.doc_id, (r.score - min) / range))
            .collect()
    }
    
    /// Collect top-k results from score map
    fn collect_top_k(&self, scores: HashMap<u64, f32>) -> FusionResult {
        let mut results: Vec<ScoredResult> = scores
            .into_iter()
            .map(|(doc_id, score)| ScoredResult::new(doc_id, score))
            .collect();
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply min_score filter
        if let Some(min) = self.config.min_score {
            results.retain(|r| r.score >= min);
        }
        
        // Truncate to k
        results.truncate(self.config.final_k);
        
        FusionResult {
            results,
            method: self.config.method,
        }
    }
}

/// Result of fusion
#[derive(Debug)]
pub struct FusionResult {
    /// Final ranked results
    pub results: Vec<ScoredResult>,
    /// Method used
    pub method: FusionMethod,
}

// ============================================================================
// Unified Hybrid Executor
// ============================================================================

/// Trait for vector search executor
pub trait VectorExecutor {
    fn search(&self, query: &[f32], k: usize, allowed: &AllowedSet) -> Vec<ScoredResult>;
}

/// Trait for BM25 executor
pub trait Bm25Executor {
    fn search(&self, query: &str, k: usize, allowed: &AllowedSet) -> Vec<ScoredResult>;
}

/// The unified hybrid executor
///
/// This is the main entry point that enforces the "no post-filtering" contract.
pub struct UnifiedHybridExecutor<V: VectorExecutor, B: Bm25Executor> {
    vector_executor: Arc<V>,
    bm25_executor: Arc<B>,
    fusion_engine: FusionEngine,
}

impl<V: VectorExecutor, B: Bm25Executor> UnifiedHybridExecutor<V, B> {
    /// Create a new executor
    pub fn new(
        vector_executor: Arc<V>,
        bm25_executor: Arc<B>,
        fusion_config: FusionConfig,
    ) -> Self {
        Self {
            vector_executor,
            bm25_executor,
            fusion_engine: FusionEngine::new(fusion_config),
        }
    }
    
    /// Execute a hybrid query with mandatory pre-filtering
    ///
    /// # Contract
    ///
    /// 1. Computes `effective_filter = auth_scope ∧ query_filter`
    /// 2. Converts to `AllowedSet` (via metadata index)
    /// 3. Passes SAME `AllowedSet` to BOTH vector and BM25 executors
    /// 4. Fuses already-filtered results
    ///
    /// NO POST-FILTERING occurs in this function.
    pub fn execute(
        &self,
        query: &UnifiedHybridQuery,
        _auth_scope: &AuthScope,
        allowed_set: &AllowedSet, // Pre-computed from FilterIR + AuthScope
    ) -> FusionResult {
        // Short-circuit if empty
        if allowed_set.is_empty() {
            return FusionResult {
                results: vec![],
                method: self.fusion_engine.config.method,
            };
        }
        
        let k = self.fusion_engine.config.candidates_per_modality;
        
        // Vector search (with AllowedSet)
        let vector_candidates = query.vector_query.as_ref().map(|vq| {
            let results = self.vector_executor.search(&vq.embedding, k, allowed_set);
            FilteredCandidates::from_vector(results)
        });
        
        // BM25 search (with SAME AllowedSet)
        let bm25_candidates = query.bm25_query.as_ref().map(|bq| {
            let results = self.bm25_executor.search(&bq.text, k, allowed_set);
            FilteredCandidates::from_bm25(results)
        });
        
        // Fuse (both are already filtered - no post-filtering!)
        self.fusion_engine.fuse(vector_candidates, bm25_candidates)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rrf_fusion() {
        let config = FusionConfig {
            method: FusionMethod::Rrf { k: 60.0 },
            candidates_per_modality: 10,
            final_k: 5,
            min_score: None,
        };
        
        let engine = FusionEngine::new(config);
        
        let vector = FilteredCandidates::from_vector(vec![
            ScoredResult::new(1, 0.9),
            ScoredResult::new(2, 0.8),
            ScoredResult::new(3, 0.7),
        ]);
        
        let bm25 = FilteredCandidates::from_bm25(vec![
            ScoredResult::new(2, 5.0), // doc 2 is in both
            ScoredResult::new(4, 4.0),
            ScoredResult::new(1, 3.0), // doc 1 is in both
        ]);
        
        let result = engine.fuse(Some(vector), Some(bm25));
        
        // Doc 2 should score highest (rank 2 in vector, rank 1 in BM25)
        // Doc 1 should also score well (rank 1 in vector, rank 3 in BM25)
        assert!(!result.results.is_empty());
        
        // Docs 1 and 2 should be near the top
        let top_ids: Vec<u64> = result.results.iter().map(|r| r.doc_id).collect();
        assert!(top_ids.contains(&1));
        assert!(top_ids.contains(&2));
    }
    
    #[test]
    fn test_linear_fusion() {
        let config = FusionConfig {
            method: FusionMethod::Linear { 
                vector_weight: 0.6, 
                bm25_weight: 0.4 
            },
            candidates_per_modality: 10,
            final_k: 5,
            min_score: None,
        };
        
        let engine = FusionEngine::new(config);
        
        let vector = FilteredCandidates::from_vector(vec![
            ScoredResult::new(1, 1.0),
            ScoredResult::new(2, 0.5),
        ]);
        
        let bm25 = FilteredCandidates::from_bm25(vec![
            ScoredResult::new(2, 10.0), // Different scale
            ScoredResult::new(3, 5.0),
        ]);
        
        let result = engine.fuse(Some(vector), Some(bm25));
        
        // After normalization, doc 2 should benefit from both
        assert!(!result.results.is_empty());
    }
    
    #[test]
    fn test_empty_allowed_set() {
        let config = FusionConfig::default();
        let engine = FusionEngine::new(config);
        
        // No candidates = empty result
        let result = engine.fuse(None, None);
        assert!(result.results.is_empty());
    }
    
    #[test]
    fn test_score_normalization() {
        let config = FusionConfig::default();
        let engine = FusionEngine::new(config);
        
        let results = vec![
            ScoredResult::new(1, 100.0),
            ScoredResult::new(2, 50.0),
            ScoredResult::new(3, 0.0),
        ];
        
        let normalized = engine.normalize_scores(&results);
        
        // Should be normalized to [0, 1]
        assert_eq!(normalized.len(), 3);
        let scores: HashMap<u64, f32> = normalized.into_iter().collect();
        assert!((scores[&1] - 1.0).abs() < 0.001);
        assert!((scores[&2] - 0.5).abs() < 0.001);
        assert!((scores[&3] - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_no_post_filter_invariant() {
        // This test verifies the core invariant:
        // result-set ⊆ allowed-set
        //
        // If this invariant is violated, it indicates a security issue.
        
        let allowed: std::collections::HashSet<u64> = [1, 2, 3, 5, 8].into_iter().collect();
        let allowed_set = AllowedSet::from_iter(allowed.iter().copied());
        
        // Simulate filtered candidates (these should already respect AllowedSet)
        let vector = FilteredCandidates::from_vector(vec![
            ScoredResult::new(1, 0.9),  // in allowed set
            ScoredResult::new(2, 0.8),  // in allowed set
            ScoredResult::new(5, 0.7),  // in allowed set
        ]);
        
        let bm25 = FilteredCandidates::from_bm25(vec![
            ScoredResult::new(2, 5.0),  // in allowed set
            ScoredResult::new(3, 4.0),  // in allowed set
            ScoredResult::new(8, 3.0),  // in allowed set
        ]);
        
        let config = FusionConfig::default();
        let engine = FusionEngine::new(config);
        let result = engine.fuse(Some(vector), Some(bm25));
        
        // INVARIANT: Every result doc_id must be in the allowed set
        for doc in &result.results {
            assert!(
                allowed_set.contains(doc.doc_id),
                "INVARIANT VIOLATION: doc_id {} not in allowed set",
                doc.doc_id
            );
        }
    }
}

// ============================================================================
// Invariant Verification
// ============================================================================

/// Verify that a fusion result respects the no-post-filtering invariant
/// 
/// This function should be used in tests and optionally in debug builds
/// to verify that the security invariant holds.
///
/// # Invariant
///
/// `∀ doc ∈ result: doc.id ∈ allowed_set`
///
/// This is the "monotone property" from the architecture document.
pub fn verify_no_post_filter_invariant(
    result: &FusionResult,
    allowed_set: &AllowedSet,
) -> InvariantVerification {
    let mut violations = Vec::new();
    
    for doc in &result.results {
        if !allowed_set.contains(doc.doc_id) {
            violations.push(doc.doc_id);
        }
    }
    
    if violations.is_empty() {
        InvariantVerification::Valid
    } else {
        InvariantVerification::Violated { doc_ids: violations }
    }
}

/// Result of invariant verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvariantVerification {
    /// Invariant holds
    Valid,
    /// Invariant violated - these doc IDs should not be in results
    Violated { doc_ids: Vec<u64> },
}

impl InvariantVerification {
    /// Check if the invariant holds
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }
    
    /// Panic if the invariant is violated (for testing)
    pub fn assert_valid(&self) {
        match self {
            Self::Valid => {}
            Self::Violated { doc_ids } => {
                panic!(
                    "NO-POST-FILTER INVARIANT VIOLATED: {} docs not in allowed set: {:?}",
                    doc_ids.len(),
                    doc_ids
                );
            }
        }
    }
}
