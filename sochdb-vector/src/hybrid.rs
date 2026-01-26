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

//! Hybrid Search with RRF Fusion (Task 4)
//!
//! This module combines vector similarity search (ANN) with lexical search (BM25)
//! using Reciprocal Rank Fusion (RRF) for score combination.
//!
//! ## RRF Algorithm
//!
//! ```text
//! RRF_score(d) = Σ weight_i / (k + rank_i(d))
//! ```
//!
//! Where:
//! - `k` is typically 60 (robust default)
//! - `rank_i(d)` is the rank of document d in result list i (1-indexed)
//! - `weight_i` is the weight for result list i
//!
//! ## Pipeline
//!
//! ```text
//!                    Query
//!                      │
//!           ┌──────────┴──────────┐
//!           │                     │
//!           ▼                     ▼
//!    ┌─────────────┐       ┌─────────────┐
//!    │   Vector    │       │   Lexical   │
//!    │   Search    │       │   Search    │
//!    │   (HNSW)    │       │   (BM25)    │
//!    └──────┬──────┘       └──────┬──────┘
//!           │                     │
//!           │  [(id, score), ...]│  [(id, score), ...]
//!           │                     │
//!           └──────────┬──────────┘
//!                      │
//!                      ▼
//!               ┌─────────────┐
//!               │  RRF Fusion │
//!               └──────┬──────┘
//!                      │
//!                      ▼
//!               ┌─────────────┐
//!               │   Filter    │
//!               │  (optional) │
//!               └──────┬──────┘
//!                      │
//!                      ▼
//!               ┌─────────────┐
//!               │   Top-K     │
//!               └─────────────┘
//! ```

use std::collections::HashMap;

// ============================================================================
// Types
// ============================================================================

/// Document ID type
pub type DocId = u64;

/// Search result with score
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document ID
    pub doc_id: DocId,
    
    /// Combined score (from fusion)
    pub score: f32,
    
    /// Component scores for debugging
    pub component_scores: Option<ComponentScores>,
}

/// Individual component scores
#[derive(Debug, Clone)]
pub struct ComponentScores {
    /// Vector similarity score
    pub vector_score: Option<f32>,
    
    /// Vector rank (1-indexed)
    pub vector_rank: Option<usize>,
    
    /// Lexical (BM25) score
    pub lexical_score: Option<f32>,
    
    /// Lexical rank (1-indexed)
    pub lexical_rank: Option<usize>,
}

// ============================================================================
// RRF Configuration
// ============================================================================

/// Configuration for Reciprocal Rank Fusion
#[derive(Debug, Clone, Copy)]
pub struct RRFConfig {
    /// RRF k parameter (typically 60)
    /// Higher values give more weight to lower-ranked results
    pub k: f32,
    
    /// Weight for vector search results
    pub vector_weight: f32,
    
    /// Weight for lexical search results
    pub lexical_weight: f32,
}

impl Default for RRFConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            vector_weight: 1.0,
            lexical_weight: 1.0,
        }
    }
}

impl RRFConfig {
    /// Create with custom weights
    pub fn with_weights(vector_weight: f32, lexical_weight: f32) -> Self {
        Self {
            k: 60.0,
            vector_weight,
            lexical_weight,
        }
    }
    
    /// Emphasize vector search (semantic)
    pub fn semantic_focused() -> Self {
        Self {
            k: 60.0,
            vector_weight: 0.7,
            lexical_weight: 0.3,
        }
    }
    
    /// Emphasize lexical search (keyword)
    pub fn keyword_focused() -> Self {
        Self {
            k: 60.0,
            vector_weight: 0.3,
            lexical_weight: 0.7,
        }
    }
    
    /// Balanced (equal weights)
    pub fn balanced() -> Self {
        Self::default()
    }
}

// ============================================================================
// RRF Fusion
// ============================================================================

/// Reciprocal Rank Fusion combiner
pub struct RRFFusion {
    config: RRFConfig,
}

impl RRFFusion {
    /// Create a new RRF fusion combiner
    pub fn new(config: RRFConfig) -> Self {
        Self { config }
    }
    
    /// Fuse vector and lexical search results
    ///
    /// # Arguments
    /// * `vector_results` - Results from vector search, sorted by score descending
    /// * `lexical_results` - Results from lexical search, sorted by score descending
    /// * `limit` - Maximum number of results to return
    /// * `keep_details` - Whether to include component scores
    ///
    /// # Returns
    /// Fused results sorted by combined score descending
    pub fn fuse(
        &self,
        vector_results: &[(DocId, f32)],
        lexical_results: &[(DocId, f32)],
        limit: usize,
        keep_details: bool,
    ) -> Vec<SearchResult> {
        let k = self.config.k;
        
        // Build rank maps (1-indexed ranks)
        let mut doc_scores: HashMap<DocId, FusionState> = HashMap::new();
        
        // Add vector results
        for (rank, &(doc_id, score)) in vector_results.iter().enumerate() {
            let rrf_score = self.config.vector_weight / (k + (rank + 1) as f32);
            
            let state = doc_scores.entry(doc_id).or_default();
            state.rrf_score += rrf_score;
            state.vector_score = Some(score);
            state.vector_rank = Some(rank + 1);
        }
        
        // Add lexical results
        for (rank, &(doc_id, score)) in lexical_results.iter().enumerate() {
            let rrf_score = self.config.lexical_weight / (k + (rank + 1) as f32);
            
            let state = doc_scores.entry(doc_id).or_default();
            state.rrf_score += rrf_score;
            state.lexical_score = Some(score);
            state.lexical_rank = Some(rank + 1);
        }
        
        // Convert to results and sort
        let mut results: Vec<SearchResult> = doc_scores
            .into_iter()
            .map(|(doc_id, state)| {
                SearchResult {
                    doc_id,
                    score: state.rrf_score,
                    component_scores: if keep_details {
                        Some(ComponentScores {
                            vector_score: state.vector_score,
                            vector_rank: state.vector_rank,
                            lexical_score: state.lexical_score,
                            lexical_rank: state.lexical_rank,
                        })
                    } else {
                        None
                    },
                }
            })
            .collect();
        
        // Sort by RRF score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Limit
        results.truncate(limit);
        
        results
    }
    
    /// Fuse multiple result lists with custom weights
    pub fn fuse_multi(
        &self,
        result_lists: &[(&[(DocId, f32)], f32)], // (results, weight)
        limit: usize,
    ) -> Vec<SearchResult> {
        let k = self.config.k;
        let mut doc_scores: HashMap<DocId, f32> = HashMap::new();
        
        for (results, weight) in result_lists {
            for (rank, &(doc_id, _score)) in results.iter().enumerate() {
                let rrf_score = *weight / (k + (rank + 1) as f32);
                *doc_scores.entry(doc_id).or_default() += rrf_score;
            }
        }
        
        let mut results: Vec<SearchResult> = doc_scores
            .into_iter()
            .map(|(doc_id, score)| SearchResult {
                doc_id,
                score,
                component_scores: None,
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);
        
        results
    }
}

/// Internal state for fusion
#[derive(Default)]
struct FusionState {
    rrf_score: f32,
    vector_score: Option<f32>,
    vector_rank: Option<usize>,
    lexical_score: Option<f32>,
    lexical_rank: Option<usize>,
}

impl Default for RRFFusion {
    fn default() -> Self {
        Self::new(RRFConfig::default())
    }
}

// ============================================================================
// Hybrid Search Engine
// ============================================================================

/// A combined vector + lexical search engine
pub struct HybridSearchEngine<V, L> {
    /// Vector search backend
    vector_search: V,
    
    /// Lexical search backend
    lexical_search: L,
    
    /// RRF fusion config
    fusion_config: RRFConfig,
    
    /// Over-fetch factor for better fusion results
    overfetch_factor: f32,
}

/// Trait for vector search backends
pub trait VectorSearchBackend {
    /// Search for similar vectors
    fn search(&self, query: &[f32], k: usize) -> Vec<(DocId, f32)>;
}

/// Trait for lexical search backends
pub trait LexicalSearchBackend {
    /// Search by text query
    fn search(&self, query: &str, k: usize) -> Vec<(DocId, f32)>;
}

impl<V, L> HybridSearchEngine<V, L>
where
    V: VectorSearchBackend,
    L: LexicalSearchBackend,
{
    /// Create a new hybrid search engine
    pub fn new(vector_search: V, lexical_search: L) -> Self {
        Self {
            vector_search,
            lexical_search,
            fusion_config: RRFConfig::default(),
            overfetch_factor: 2.0,
        }
    }
    
    /// Set fusion configuration
    pub fn with_fusion_config(mut self, config: RRFConfig) -> Self {
        self.fusion_config = config;
        self
    }
    
    /// Set over-fetch factor
    pub fn with_overfetch(mut self, factor: f32) -> Self {
        self.overfetch_factor = factor.max(1.0);
        self
    }
    
    /// Perform hybrid search
    pub fn search(
        &self,
        vector_query: Option<&[f32]>,
        text_query: Option<&str>,
        limit: usize,
    ) -> Vec<SearchResult> {
        let fetch_k = (limit as f32 * self.overfetch_factor) as usize;
        
        // Get vector results
        let vector_results = match vector_query {
            Some(q) => self.vector_search.search(q, fetch_k),
            None => Vec::new(),
        };
        
        // Get lexical results
        let lexical_results = match text_query {
            Some(q) => self.lexical_search.search(q, fetch_k),
            None => Vec::new(),
        };
        
        // If only one type of search, return directly
        if vector_results.is_empty() {
            return lexical_results
                .into_iter()
                .take(limit)
                .map(|(doc_id, score)| SearchResult {
                    doc_id,
                    score,
                    component_scores: None,
                })
                .collect();
        }
        
        if lexical_results.is_empty() {
            return vector_results
                .into_iter()
                .take(limit)
                .map(|(doc_id, score)| SearchResult {
                    doc_id,
                    score,
                    component_scores: None,
                })
                .collect();
        }
        
        // Fuse results
        let fusion = RRFFusion::new(self.fusion_config);
        fusion.fuse(&vector_results, &lexical_results, limit, false)
    }
    
    /// Perform hybrid search with detailed scores
    pub fn search_detailed(
        &self,
        vector_query: Option<&[f32]>,
        text_query: Option<&str>,
        limit: usize,
    ) -> Vec<SearchResult> {
        let fetch_k = (limit as f32 * self.overfetch_factor) as usize;
        
        let vector_results = vector_query
            .map(|q| self.vector_search.search(q, fetch_k))
            .unwrap_or_default();
        
        let lexical_results = text_query
            .map(|q| self.lexical_search.search(q, fetch_k))
            .unwrap_or_default();
        
        let fusion = RRFFusion::new(self.fusion_config);
        fusion.fuse(&vector_results, &lexical_results, limit, true)
    }
}

// ============================================================================
// Filter Integration
// ============================================================================

/// Post-filter results by metadata predicate
pub fn filter_results<F>(results: Vec<SearchResult>, predicate: F, limit: usize) -> Vec<SearchResult>
where
    F: Fn(DocId) -> bool,
{
    results
        .into_iter()
        .filter(|r| predicate(r.doc_id))
        .take(limit)
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rrf_fusion_basic() {
        let fusion = RRFFusion::default();
        
        let vector_results = vec![
            (1, 0.95),
            (2, 0.90),
            (3, 0.85),
        ];
        
        let lexical_results = vec![
            (2, 5.0),  // Shared with vector
            (4, 4.5),
            (3, 4.0),  // Shared with vector
        ];
        
        let results = fusion.fuse(&vector_results, &lexical_results, 10, false);
        
        // Doc 2 appears in both lists - should rank high
        assert!(!results.is_empty());
        
        // Check that scores are computed
        for r in &results {
            assert!(r.score > 0.0);
        }
    }
    
    #[test]
    fn test_rrf_fusion_with_details() {
        let fusion = RRFFusion::default();
        
        let vector_results = vec![(1, 0.9), (2, 0.8)];
        let lexical_results = vec![(2, 5.0), (3, 4.0)];
        
        let results = fusion.fuse(&vector_results, &lexical_results, 10, true);
        
        // Find doc 2 (appears in both)
        let doc2 = results.iter().find(|r| r.doc_id == 2).unwrap();
        let scores = doc2.component_scores.as_ref().unwrap();
        
        assert_eq!(scores.vector_rank, Some(2)); // Rank 2 in vector results
        assert_eq!(scores.lexical_rank, Some(1)); // Rank 1 in lexical results
        assert_eq!(scores.vector_score, Some(0.8));
        assert_eq!(scores.lexical_score, Some(5.0));
    }
    
    #[test]
    fn test_rrf_ranking() {
        let fusion = RRFFusion::default();
        
        // Doc 1: rank 1 in vector, not in lexical
        // Doc 2: rank 2 in vector, rank 1 in lexical
        // Doc 2 should win because it appears in both
        let vector_results = vec![(1, 0.95), (2, 0.90)];
        let lexical_results = vec![(2, 5.0)];
        
        let results = fusion.fuse(&vector_results, &lexical_results, 10, false);
        
        assert_eq!(results[0].doc_id, 2); // Doc 2 should be first
    }
    
    #[test]
    fn test_rrf_weights() {
        // Heavy lexical weight
        let config = RRFConfig::keyword_focused();
        let fusion = RRFFusion::new(config);
        
        // Doc 1: rank 1 in vector only
        // Doc 2: rank 1 in lexical only
        let vector_results = vec![(1, 0.95)];
        let lexical_results = vec![(2, 5.0)];
        
        let results = fusion.fuse(&vector_results, &lexical_results, 10, false);
        
        // Doc 2 should win with keyword-focused config
        assert_eq!(results[0].doc_id, 2);
    }
    
    #[test]
    fn test_fuse_multi() {
        let fusion = RRFFusion::default();
        
        let list1: Vec<(DocId, f32)> = vec![(1, 0.9), (2, 0.8)];
        let list2: Vec<(DocId, f32)> = vec![(2, 0.9), (3, 0.8)];
        let list3: Vec<(DocId, f32)> = vec![(3, 0.9), (1, 0.8)];
        
        let results = fusion.fuse_multi(
            &[
                (&list1, 1.0),
                (&list2, 1.0),
                (&list3, 1.0),
            ],
            10,
        );
        
        // All docs should appear
        let doc_ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&2));
        assert!(doc_ids.contains(&3));
    }
    
    #[test]
    fn test_filter_results() {
        let results = vec![
            SearchResult { doc_id: 1, score: 0.9, component_scores: None },
            SearchResult { doc_id: 2, score: 0.8, component_scores: None },
            SearchResult { doc_id: 3, score: 0.7, component_scores: None },
            SearchResult { doc_id: 4, score: 0.6, component_scores: None },
        ];
        
        // Filter to only even doc IDs
        let filtered = filter_results(results, |id| id % 2 == 0, 10);
        
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].doc_id, 2);
        assert_eq!(filtered[1].doc_id, 4);
    }
}
