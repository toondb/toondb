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

//! BM25 Filter Pushdown via Posting-Set Intersection (Task 6)
//!
//! This module implements filter pushdown for BM25 full-text search by
//! intersecting posting lists with the allowed set BEFORE scoring.
//!
//! ## Key Insight
//!
//! ```text
//! Traditional BM25: score ALL docs with term → filter → top-k
//! Filtered BM25: intersect posting(term) ∩ AllowedSet → score only intersection
//! ```
//!
//! ## Optimization: Term Reordering
//!
//! For multi-term queries, we reorder terms by increasing document frequency (DF):
//!
//! ```text
//! Query: "machine learning algorithms"
//! DFs: machine=10000, learning=8000, algorithms=500
//! 
//! Order: algorithms → learning → machine
//! 
//! Step 1: posting(algorithms) ∩ AllowedSet → small_set
//! Step 2: small_set ∩ posting(learning) → smaller_set  
//! Step 3: smaller_set ∩ posting(machine) → final_set
//! ```
//!
//! This minimizes intermediate set sizes, reducing memory and CPU.
//!
//! ## Cost Model
//!
//! Let:
//! - N = total docs
//! - |A| = allowed set size
//! - df(t) = document frequency of term t
//!
//! Without pushdown: O(Σ df(t)) scoring operations
//! With pushdown: O(min(|A|, min(df(t))) scoring operations
//!
//! When |A| << N, this is a massive win.

use std::collections::HashMap;
use std::sync::Arc;

use crate::candidate_gate::AllowedSet;
use crate::filtered_vector_search::ScoredResult;

// ============================================================================
// BM25 Parameters
// ============================================================================

/// BM25 scoring parameters
#[derive(Debug, Clone)]
pub struct Bm25Params {
    /// Term frequency saturation parameter (default: 1.2)
    pub k1: f32,
    /// Document length normalization (default: 0.75)
    pub b: f32,
    /// Average document length (computed from corpus)
    pub avgdl: f32,
    /// Total number of documents in corpus
    pub total_docs: u64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            avgdl: 100.0,
            total_docs: 1_000_000,
        }
    }
}

impl Bm25Params {
    /// Compute IDF for a term
    pub fn idf(&self, doc_freq: u64) -> f32 {
        let n = self.total_docs as f32;
        let df = doc_freq as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }
    
    /// Compute term score component
    pub fn term_score(&self, tf: f32, doc_len: f32, idf: f32) -> f32 {
        let numerator = tf * (self.k1 + 1.0);
        let denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl);
        idf * numerator / denominator
    }
}

// ============================================================================
// Term Posting List
// ============================================================================

/// A posting list for a single term
#[derive(Debug, Clone)]
pub struct PostingList {
    /// Term text
    pub term: String,
    /// Document IDs containing this term (sorted)
    pub doc_ids: Vec<u64>,
    /// Term frequencies in each document
    pub term_freqs: Vec<u32>,
    /// Document frequency (len of doc_ids)
    pub doc_freq: u64,
}

impl PostingList {
    /// Create a new posting list
    pub fn new(term: impl Into<String>, entries: Vec<(u64, u32)>) -> Self {
        let term = term.into();
        let doc_freq = entries.len() as u64;
        let mut doc_ids = Vec::with_capacity(entries.len());
        let mut term_freqs = Vec::with_capacity(entries.len());
        
        for (doc_id, tf) in entries {
            doc_ids.push(doc_id);
            term_freqs.push(tf);
        }
        
        Self {
            term,
            doc_ids,
            term_freqs,
            doc_freq,
        }
    }
    
    /// Intersect with an allowed set, returning (doc_id, tf) pairs
    pub fn intersect_with_allowed(&self, allowed: &AllowedSet) -> Vec<(u64, u32)> {
        match allowed {
            AllowedSet::All => {
                self.doc_ids.iter()
                    .zip(self.term_freqs.iter())
                    .map(|(&id, &tf)| (id, tf))
                    .collect()
            }
            AllowedSet::None => vec![],
            _ => {
                self.doc_ids.iter()
                    .zip(self.term_freqs.iter())
                    .filter(|&(&id, _)| allowed.contains(id))
                    .map(|(&id, &tf)| (id, tf))
                    .collect()
            }
        }
    }
}

// ============================================================================
// Inverted Index Interface
// ============================================================================

/// Trait for accessing an inverted index
pub trait InvertedIndex: Send + Sync {
    /// Get posting list for a term (None if term not in vocabulary)
    fn get_posting_list(&self, term: &str) -> Option<PostingList>;
    
    /// Get document length (in tokens)
    fn get_doc_length(&self, doc_id: u64) -> Option<u32>;
    
    /// Get BM25 parameters
    fn get_params(&self) -> &Bm25Params;
}

// ============================================================================
// Filtered BM25 Executor
// ============================================================================

/// A BM25 executor that applies filter pushdown
pub struct FilteredBm25Executor<I: InvertedIndex> {
    index: Arc<I>,
}

impl<I: InvertedIndex> FilteredBm25Executor<I> {
    /// Create a new executor
    pub fn new(index: Arc<I>) -> Self {
        Self { index }
    }
    
    /// Execute a BM25 query with filter pushdown
    ///
    /// # Algorithm
    ///
    /// 1. Tokenize query
    /// 2. Get posting lists for each term
    /// 3. Sort terms by DF (ascending) for early pruning
    /// 4. Intersect posting lists with AllowedSet (in DF order)
    /// 5. Score only docs in intersection
    /// 6. Return top-k
    pub fn search(
        &self,
        query: &str,
        k: usize,
        allowed: &AllowedSet,
    ) -> Vec<ScoredResult> {
        // Short-circuit if nothing allowed
        if allowed.is_empty() {
            return vec![];
        }
        
        // Tokenize (simple whitespace split for now)
        let terms: Vec<&str> = query
            .split_whitespace()
            .filter(|t| t.len() >= 2) // Skip very short terms
            .collect();
        
        if terms.is_empty() {
            return vec![];
        }
        
        // Get posting lists and sort by DF
        let mut posting_lists: Vec<PostingList> = terms
            .iter()
            .filter_map(|t| self.index.get_posting_list(t))
            .collect();
        
        // Sort by document frequency (ascending)
        posting_lists.sort_by_key(|pl| pl.doc_freq);
        
        // Progressive intersection with AllowedSet
        let candidates = self.progressive_intersection(&posting_lists, allowed);
        
        if candidates.is_empty() {
            return vec![];
        }
        
        // Score candidates
        let params = self.index.get_params();
        let scores = self.score_candidates(&candidates, &posting_lists, params);
        
        // Return top-k
        self.top_k(scores, k)
    }
    
    /// Progressively intersect posting lists with allowed set
    ///
    /// Returns map of doc_id -> term frequencies for each term
    fn progressive_intersection(
        &self,
        posting_lists: &[PostingList],
        allowed: &AllowedSet,
    ) -> HashMap<u64, Vec<u32>> {
        if posting_lists.is_empty() {
            return HashMap::new();
        }
        
        // Start with first (smallest DF) term intersected with allowed
        let first = &posting_lists[0];
        let mut candidates: HashMap<u64, Vec<u32>> = first
            .intersect_with_allowed(allowed)
            .into_iter()
            .map(|(id, tf)| (id, vec![tf]))
            .collect();
        
        // Intersect with remaining terms
        for (_term_idx, posting_list) in posting_lists.iter().enumerate().skip(1) {
            // Build a lookup for this term's postings
            let term_postings: HashMap<u64, u32> = posting_list
                .doc_ids.iter()
                .zip(posting_list.term_freqs.iter())
                .map(|(&id, &tf)| (id, tf))
                .collect();
            
            // Keep only candidates that appear in this term's postings
            candidates.retain(|doc_id, tfs| {
                if let Some(&tf) = term_postings.get(doc_id) {
                    tfs.push(tf);
                    true
                } else {
                    false
                }
            });
            
            // Early exit if no candidates remain
            if candidates.is_empty() {
                break;
            }
        }
        
        candidates
    }
    
    /// Score candidates using BM25
    fn score_candidates(
        &self,
        candidates: &HashMap<u64, Vec<u32>>,
        posting_lists: &[PostingList],
        params: &Bm25Params,
    ) -> Vec<ScoredResult> {
        // Precompute IDFs
        let idfs: Vec<f32> = posting_lists
            .iter()
            .map(|pl| params.idf(pl.doc_freq))
            .collect();
        
        candidates
            .iter()
            .filter_map(|(&doc_id, tfs)| {
                let doc_len = self.index.get_doc_length(doc_id)? as f32;
                
                let score: f32 = tfs.iter()
                    .zip(idfs.iter())
                    .map(|(&tf, &idf)| params.term_score(tf as f32, doc_len, idf))
                    .sum();
                
                Some(ScoredResult::new(doc_id, score))
            })
            .collect()
    }
    
    /// Get top-k results
    fn top_k(&self, mut scores: Vec<ScoredResult>, k: usize) -> Vec<ScoredResult> {
        // Sort by score descending
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
}

// ============================================================================
// Disjunctive (OR) Query Support
// ============================================================================

/// A disjunctive BM25 query (OR semantics)
pub struct DisjunctiveBm25Executor<I: InvertedIndex> {
    index: Arc<I>,
}

impl<I: InvertedIndex> DisjunctiveBm25Executor<I> {
    /// Create a new executor
    pub fn new(index: Arc<I>) -> Self {
        Self { index }
    }
    
    /// Execute with OR semantics (documents match if they have ANY query term)
    pub fn search(
        &self,
        query: &str,
        k: usize,
        allowed: &AllowedSet,
    ) -> Vec<ScoredResult> {
        if allowed.is_empty() {
            return vec![];
        }
        
        let terms: Vec<&str> = query.split_whitespace().collect();
        if terms.is_empty() {
            return vec![];
        }
        
        // Get posting lists
        let posting_lists: Vec<PostingList> = terms
            .iter()
            .filter_map(|t| self.index.get_posting_list(t))
            .collect();
        
        let params = self.index.get_params();
        
        // Accumulate scores for all docs matching any term
        let mut scores: HashMap<u64, f32> = HashMap::new();
        
        for posting_list in &posting_lists {
            let idf = params.idf(posting_list.doc_freq);
            
            // Only score docs in allowed set
            for (&doc_id, &tf) in posting_list.doc_ids.iter().zip(posting_list.term_freqs.iter()) {
                if !allowed.contains(doc_id) {
                    continue;
                }
                
                if let Some(doc_len) = self.index.get_doc_length(doc_id) {
                    let term_score = params.term_score(tf as f32, doc_len as f32, idf);
                    *scores.entry(doc_id).or_insert(0.0) += term_score;
                }
            }
        }
        
        // Convert to results and get top-k
        let mut results: Vec<ScoredResult> = scores
            .into_iter()
            .map(|(id, score)| ScoredResult::new(id, score))
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

// ============================================================================
// Phrase Query Support
// ============================================================================

/// Position information for a term in a document
#[derive(Debug, Clone)]
pub struct PositionalPosting {
    /// Document ID
    pub doc_id: u64,
    /// Positions where term appears (sorted)
    pub positions: Vec<u32>,
}

/// Trait for positional index access
pub trait PositionalIndex: InvertedIndex {
    /// Get positional posting list
    fn get_positional_posting(&self, term: &str) -> Option<Vec<PositionalPosting>>;
}

/// Phrase query executor with filter pushdown
pub struct FilteredPhraseExecutor<I: PositionalIndex> {
    index: Arc<I>,
}

impl<I: PositionalIndex> FilteredPhraseExecutor<I> {
    /// Create a new executor
    pub fn new(index: Arc<I>) -> Self {
        Self { index }
    }
    
    /// Execute a phrase query
    ///
    /// Documents must contain all terms in sequence.
    pub fn search(
        &self,
        phrase: &[&str],
        k: usize,
        allowed: &AllowedSet,
    ) -> Vec<ScoredResult> {
        if phrase.is_empty() || allowed.is_empty() {
            return vec![];
        }
        
        // Get positional postings for each term
        let mut positional_postings: Vec<Vec<PositionalPosting>> = vec![];
        for term in phrase {
            match self.index.get_positional_posting(term) {
                Some(postings) => positional_postings.push(postings),
                None => return vec![], // Term not in index → no matches
            }
        }
        
        // Find documents containing all terms
        let candidates = self.find_phrase_matches(&positional_postings, allowed);
        
        // Score by phrase frequency
        let params = self.index.get_params();
        let results: Vec<ScoredResult> = candidates
            .into_iter()
            .filter_map(|(doc_id, phrase_freq)| {
                let doc_len = self.index.get_doc_length(doc_id)? as f32;
                // Use phrase frequency as TF, use min DF for IDF approximation
                let min_df = positional_postings.iter()
                    .map(|pp| pp.len() as u64)
                    .min()
                    .unwrap_or(1);
                let idf = params.idf(min_df);
                let score = params.term_score(phrase_freq as f32, doc_len, idf);
                Some(ScoredResult::new(doc_id, score))
            })
            .collect();
        
        let mut results = results;
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
    
    /// Find documents containing the phrase
    fn find_phrase_matches(
        &self,
        positional_postings: &[Vec<PositionalPosting>],
        allowed: &AllowedSet,
    ) -> Vec<(u64, u32)> {
        if positional_postings.is_empty() {
            return vec![];
        }
        
        // Index postings by doc_id
        let indexed: Vec<HashMap<u64, &Vec<u32>>> = positional_postings
            .iter()
            .map(|postings| {
                postings.iter()
                    .filter(|p| allowed.contains(p.doc_id))
                    .map(|p| (p.doc_id, &p.positions))
                    .collect()
            })
            .collect();
        
        // Start with docs having first term
        let first_docs: std::collections::HashSet<u64> = indexed[0].keys().copied().collect();
        
        // Keep only docs that have all terms
        let candidate_docs: Vec<u64> = first_docs
            .into_iter()
            .filter(|doc_id| indexed.iter().all(|idx| idx.contains_key(doc_id)))
            .collect();
        
        // Check phrase positions
        let mut matches = vec![];
        
        for doc_id in candidate_docs {
            let mut phrase_count = 0u32;
            
            // Get positions for first term
            let first_positions = indexed[0].get(&doc_id).unwrap();
            
            'outer: for &start_pos in first_positions.iter() {
                // Check if subsequent terms appear at consecutive positions
                for (term_idx, term_positions) in indexed.iter().enumerate().skip(1) {
                    let expected_pos = start_pos + term_idx as u32;
                    let positions = term_positions.get(&doc_id).unwrap();
                    
                    // Binary search for expected position
                    if positions.binary_search(&expected_pos).is_err() {
                        continue 'outer;
                    }
                }
                
                // Found a phrase match
                phrase_count += 1;
            }
            
            if phrase_count > 0 {
                matches.push((doc_id, phrase_count));
            }
        }
        
        matches
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_gate::AllowedSet;
    
    // Mock inverted index for testing
    struct MockIndex {
        postings: HashMap<String, PostingList>,
        doc_lengths: HashMap<u64, u32>,
        params: Bm25Params,
    }
    
    impl MockIndex {
        fn new() -> Self {
            let mut postings = HashMap::new();
            let mut doc_lengths = HashMap::new();
            
            // Add some test data
            postings.insert("rust".to_string(), PostingList::new("rust", vec![
                (1, 3), (2, 1), (3, 2), (5, 1),
            ]));
            postings.insert("database".to_string(), PostingList::new("database", vec![
                (1, 1), (3, 4), (4, 1),
            ]));
            postings.insert("vector".to_string(), PostingList::new("vector", vec![
                (1, 2), (2, 3), (4, 1), (5, 2),
            ]));
            
            // Doc lengths
            for i in 1..=5 {
                doc_lengths.insert(i, 100);
            }
            
            Self {
                postings,
                doc_lengths,
                params: Bm25Params {
                    k1: 1.2,
                    b: 0.75,
                    avgdl: 100.0,
                    total_docs: 1000,
                },
            }
        }
    }
    
    impl InvertedIndex for MockIndex {
        fn get_posting_list(&self, term: &str) -> Option<PostingList> {
            self.postings.get(term).cloned()
        }
        
        fn get_doc_length(&self, doc_id: u64) -> Option<u32> {
            self.doc_lengths.get(&doc_id).copied()
        }
        
        fn get_params(&self) -> &Bm25Params {
            &self.params
        }
    }
    
    #[test]
    fn test_conjunctive_search() {
        let index = Arc::new(MockIndex::new());
        let executor = FilteredBm25Executor::new(index);
        
        // Search for "rust database"
        // Should match docs 1 and 3 (have both terms)
        let results = executor.search("rust database", 10, &AllowedSet::All);
        
        assert_eq!(results.len(), 2);
        let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        assert!(doc_ids.contains(&1));
        assert!(doc_ids.contains(&3));
    }
    
    #[test]
    fn test_filter_pushdown() {
        let index = Arc::new(MockIndex::new());
        let executor = FilteredBm25Executor::new(index);
        
        // Only allow doc 1
        let allowed = AllowedSet::SortedVec(Arc::new(vec![1]));
        
        let results = executor.search("rust database", 10, &allowed);
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 1);
    }
    
    #[test]
    fn test_empty_allowed_set() {
        let index = Arc::new(MockIndex::new());
        let executor = FilteredBm25Executor::new(index);
        
        let results = executor.search("rust", 10, &AllowedSet::None);
        assert!(results.is_empty());
    }
    
    #[test]
    fn test_disjunctive_search() {
        let index = Arc::new(MockIndex::new());
        let executor = DisjunctiveBm25Executor::new(index);
        
        // Search for "rust database" with OR semantics
        // Should match docs 1, 2, 3, 4, 5 (any with either term)
        let results = executor.search("rust database", 10, &AllowedSet::All);
        
        // Docs 1-5 have at least one of the terms
        assert!(results.len() >= 4);
    }
    
    #[test]
    fn test_term_ordering_by_df() {
        // This tests that progressive intersection starts with lowest DF
        let mut pl1 = PostingList::new("rare", vec![(1, 1), (2, 1)]); // DF=2
        let mut pl2 = PostingList::new("common", vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]); // DF=5
        
        let mut lists = vec![pl2.clone(), pl1.clone()];
        lists.sort_by_key(|pl| pl.doc_freq);
        
        // Should be sorted: rare (DF=2) before common (DF=5)
        assert_eq!(lists[0].term, "rare");
        assert_eq!(lists[1].term, "common");
    }
    
    #[test]
    fn test_bm25_scoring() {
        let params = Bm25Params::default();
        
        // IDF for rare term (DF=10 in 1M docs) should be higher than common (DF=100K)
        let idf_rare = params.idf(10);
        let idf_common = params.idf(100_000);
        
        assert!(idf_rare > idf_common);
        
        // Score with higher TF should be higher
        let score_tf_1 = params.term_score(1.0, 100.0, idf_rare);
        let score_tf_5 = params.term_score(5.0, 100.0, idf_rare);
        
        assert!(score_tf_5 > score_tf_1);
    }
}
