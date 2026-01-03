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

//! BM25 Scoring for Lexical Search (Task 4)
//!
//! This module implements BM25 (Best Matching 25) scoring for keyword search.
//! BM25 is the standard ranking function for lexical retrieval, balancing:
//! - Term frequency (TF): How often a term appears in a document
//! - Inverse document frequency (IDF): How rare a term is across all documents
//! - Document length normalization: Penalizing very long documents
//!
//! ## BM25 Formula
//!
//! ```text
//! score(q, d) = Σ IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * |d|/avgdl))
//! ```
//!
//! Where:
//! - `TF(t,d)` = term frequency of term t in document d
//! - `IDF(t)` = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
//! - `N` = total number of documents
//! - `df(t)` = number of documents containing term t
//! - `|d|` = length of document d
//! - `avgdl` = average document length
//! - `k1` = term frequency saturation parameter (typically 1.2)
//! - `b` = length normalization parameter (typically 0.75)

use std::collections::HashMap;

// ============================================================================
// BM25 Configuration
// ============================================================================

/// BM25 scoring parameters
#[derive(Debug, Clone, Copy)]
pub struct BM25Config {
    /// Term frequency saturation parameter (k1)
    /// Higher values give more weight to term frequency
    /// Typical range: 1.2 - 2.0
    pub k1: f32,
    
    /// Length normalization parameter (b)
    /// 0.0 = no length normalization
    /// 1.0 = full length normalization
    /// Typical value: 0.75
    pub b: f32,
    
    /// Minimum IDF to filter out very common terms
    pub min_idf: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }
}

impl BM25Config {
    /// Lucene-style BM25 parameters
    pub fn lucene() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }
    
    /// Elasticsearch-style parameters
    pub fn elasticsearch() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }
    
    /// Parameters optimized for short queries
    pub fn short_queries() -> Self {
        Self {
            k1: 1.5,
            b: 0.5, // Less length normalization
            min_idf: 0.0,
        }
    }
}

// ============================================================================
// BM25 Scorer
// ============================================================================

/// BM25 scorer for a document collection
pub struct BM25Scorer {
    /// Configuration
    config: BM25Config,
    
    /// Total number of documents
    num_docs: usize,
    
    /// Average document length (in tokens)
    avg_doc_len: f32,
    
    /// Document frequency for each term
    doc_freqs: HashMap<String, usize>,
    
    /// Precomputed IDF scores for terms
    idf_cache: HashMap<String, f32>,
}

impl BM25Scorer {
    /// Create a new BM25 scorer
    pub fn new(config: BM25Config) -> Self {
        Self {
            config,
            num_docs: 0,
            avg_doc_len: 0.0,
            doc_freqs: HashMap::new(),
            idf_cache: HashMap::new(),
        }
    }
    
    /// Build the scorer from a collection of documents
    pub fn build<I, D, T>(documents: I, config: BM25Config) -> Self
    where
        I: IntoIterator<Item = D>,
        D: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut scorer = Self::new(config);
        let mut total_len = 0usize;
        let mut num_docs = 0usize;
        let mut doc_freqs: HashMap<String, usize> = HashMap::new();
        
        for doc in documents {
            num_docs += 1;
            let mut seen_terms: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut doc_len = 0usize;
            
            for token in doc {
                let term = token.as_ref().to_lowercase();
                if !term.is_empty() {
                    seen_terms.insert(term);
                    doc_len += 1;
                }
            }
            
            total_len += doc_len;
            
            for term in seen_terms {
                *doc_freqs.entry(term).or_insert(0) += 1;
            }
        }
        
        scorer.num_docs = num_docs;
        scorer.avg_doc_len = if num_docs > 0 { total_len as f32 / num_docs as f32 } else { 0.0 };
        scorer.doc_freqs = doc_freqs;
        
        // Precompute IDF scores
        scorer.precompute_idf();
        
        scorer
    }
    
    /// Precompute IDF scores for all terms
    fn precompute_idf(&mut self) {
        for (term, &df) in &self.doc_freqs {
            let idf = self.compute_idf(df, self.num_docs);
            if idf >= self.config.min_idf {
                self.idf_cache.insert(term.clone(), idf);
            }
        }
    }
    
    /// Compute IDF for a term
    #[inline]
    fn compute_idf(&self, df: usize, n: usize) -> f32 {
        // IDF with Robertson-Sparck Jones formula + 1 to avoid negative values
        let n = n as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }
    
    /// Get IDF for a term
    pub fn idf(&self, term: &str) -> f32 {
        self.idf_cache
            .get(&term.to_lowercase())
            .copied()
            .unwrap_or_else(|| {
                // Unknown term - use maximum IDF
                ((self.num_docs as f32 + 0.5) / 0.5 + 1.0).ln()
            })
    }
    
    /// Score a document for a query
    pub fn score<I, T>(&self, query_terms: I, doc_terms: &[T], doc_len: usize) -> f32
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str> + std::hash::Hash + Eq,
    {
        // Build term frequency map for document
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for term in doc_terms {
            *tf.entry(term.as_ref()).or_insert(0) += 1;
        }
        
        let k1 = self.config.k1;
        let b = self.config.b;
        let dl = doc_len as f32;
        let avgdl = self.avg_doc_len;
        
        let mut score = 0.0f32;
        
        for query_term in query_terms {
            let term = query_term.as_ref().to_lowercase();
            let term_str = term.as_str();
            
            // Get TF for this term in the document
            let term_tf = *tf.get(term_str).unwrap_or(&0) as f32;
            if term_tf == 0.0 {
                continue;
            }
            
            // Get IDF
            let idf = self.idf(&term);
            
            // BM25 scoring formula
            let numerator = term_tf * (k1 + 1.0);
            let denominator = term_tf + k1 * (1.0 - b + b * dl / avgdl);
            
            score += idf * numerator / denominator;
        }
        
        score
    }
    
    /// Score a document given precomputed term frequencies
    #[inline]
    pub fn score_with_tf(&self, query_terms: &[String], doc_tf: &HashMap<String, usize>, doc_len: usize) -> f32 {
        let k1 = self.config.k1;
        let b = self.config.b;
        let dl = doc_len as f32;
        let avgdl = self.avg_doc_len;
        
        let mut score = 0.0f32;
        
        for term in query_terms {
            let term_tf = *doc_tf.get(term).unwrap_or(&0) as f32;
            if term_tf == 0.0 {
                continue;
            }
            
            let idf = self.idf(term);
            let numerator = term_tf * (k1 + 1.0);
            let denominator = term_tf + k1 * (1.0 - b + b * dl / avgdl);
            
            score += idf * numerator / denominator;
        }
        
        score
    }
    
    /// Update stats when adding a document
    pub fn add_document<I, T>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut doc_len = 0usize;
        
        for token in tokens {
            let term = token.as_ref().to_lowercase();
            if !term.is_empty() {
                seen.insert(term);
                doc_len += 1;
            }
        }
        
        // Update average document length
        let total_len = self.avg_doc_len * self.num_docs as f32;
        self.num_docs += 1;
        self.avg_doc_len = (total_len + doc_len as f32) / self.num_docs as f32;
        
        // Update document frequencies
        let num_docs = self.num_docs;
        let min_idf = self.config.min_idf;
        
        for term in seen {
            let df = self.doc_freqs.entry(term.clone()).or_insert(0);
            *df += 1;
            let df_val = *df;
            
            // Update IDF cache (compute inline to avoid borrow issues)
            let idf = (((num_docs - df_val + 1) as f32) / (df_val as f32 + 0.5)).ln();
            if idf >= min_idf {
                self.idf_cache.insert(term, idf);
            } else {
                self.idf_cache.remove(&term);
            }
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> BM25Stats {
        BM25Stats {
            num_docs: self.num_docs,
            avg_doc_len: self.avg_doc_len,
            vocab_size: self.doc_freqs.len(),
        }
    }
}

/// BM25 scorer statistics
#[derive(Debug, Clone)]
pub struct BM25Stats {
    pub num_docs: usize,
    pub avg_doc_len: f32,
    pub vocab_size: usize,
}

// ============================================================================
// Simple Tokenizer
// ============================================================================

/// Simple whitespace + lowercase tokenizer
///
/// For MVP, this is sufficient. Can be replaced with more sophisticated
/// tokenizers (stemming, stopwords, etc.) later.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Tokenize with minimal normalization
pub fn tokenize_minimal(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty() && s.len() > 1) // Filter single chars
        .collect()
}

/// Tokenize query (keeps original for exact matching, adds lowercase)
pub fn tokenize_query(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for part in text.split_whitespace() {
        let lower = part.to_lowercase();
        tokens.push(lower);
    }
    tokens
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bm25_basic() {
        let docs = vec![
            vec!["hello", "world"],
            vec!["hello", "there"],
            vec!["goodbye", "world"],
        ];
        
        let scorer = BM25Scorer::build(
            docs.iter().map(|d| d.iter()),
            BM25Config::default(),
        );
        
        assert_eq!(scorer.num_docs, 3);
        assert!((scorer.avg_doc_len - 2.0).abs() < 0.001);
    }
    
    #[test]
    fn test_bm25_idf() {
        let docs = vec![
            vec!["common", "common", "rare"],
            vec!["common", "other"],
            vec!["common", "another"],
        ];
        
        let scorer = BM25Scorer::build(
            docs.iter().map(|d| d.iter()),
            BM25Config::default(),
        );
        
        // "common" appears in all 3 docs, "rare" in only 1
        let idf_common = scorer.idf("common");
        let idf_rare = scorer.idf("rare");
        
        // Rare terms should have higher IDF
        assert!(idf_rare > idf_common);
    }
    
    #[test]
    fn test_bm25_scoring() {
        let docs = vec![
            vec!["the", "quick", "brown", "fox"],
            vec!["the", "lazy", "dog"],
            vec!["quick", "quick", "quick"], // High TF for "quick"
        ];
        
        let scorer = BM25Scorer::build(
            docs.iter().map(|d| d.iter()),
            BM25Config::default(),
        );
        
        // Score doc 3 for "quick"
        let score = scorer.score(
            vec!["quick"],
            &["quick", "quick", "quick"],
            3,
        );
        
        assert!(score > 0.0);
        
        // Score doc 1 for "quick"
        let score1 = scorer.score(
            vec!["quick"],
            &["the", "quick", "brown", "fox"],
            4,
        );
        
        // Doc 3 should score higher (more occurrences of "quick")
        assert!(score > score1);
    }
    
    #[test]
    fn test_tokenize() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize(text);
        
        assert_eq!(tokens, vec!["hello,", "world!", "this", "is", "a", "test."]);
    }
    
    #[test]
    fn test_tokenize_minimal() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize_minimal(text);
        
        // Single chars and punctuation filtered
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // Single char
    }
    
    #[test]
    fn test_add_document() {
        let mut scorer = BM25Scorer::new(BM25Config::default());
        
        scorer.add_document(vec!["hello", "world"]);
        assert_eq!(scorer.num_docs, 1);
        
        scorer.add_document(vec!["hello", "there", "friend"]);
        assert_eq!(scorer.num_docs, 2);
        
        // Average should be (2 + 3) / 2 = 2.5
        assert!((scorer.avg_doc_len - 2.5).abs() < 0.001);
    }
}
