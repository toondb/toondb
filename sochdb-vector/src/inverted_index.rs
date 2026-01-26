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

//! Inverted Index for Lexical Search (Task 4)
//!
//! This module implements an inverted index for BM25-based lexical search.
//!
//! ## Structure
//!
//! ```text
//! Term → PostingList
//! ┌────────────────────────────────────────────────────────────────┐
//! │  "hello" → [(doc_1, tf=2, positions=[0,5]), (doc_3, tf=1, ...)]│
//! │  "world" → [(doc_1, tf=1, positions=[1]), (doc_2, tf=3, ...)] │
//! │  ...                                                           │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Query Execution
//!
//! 1. Tokenize query
//! 2. Look up posting lists for each query term
//! 3. Score documents using BM25
//! 4. Return top-K results

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use crate::bm25::{BM25Config, BM25Scorer, tokenize_minimal};

// ============================================================================
// Types
// ============================================================================

/// Document ID type
pub type DocId = u64;

/// Term position in document
pub type Position = u32;

/// Term frequency
pub type TermFreq = u32;

// ============================================================================
// Posting List
// ============================================================================

/// A posting for a single document
#[derive(Debug, Clone)]
pub struct Posting {
    /// Document ID
    pub doc_id: DocId,
    
    /// Term frequency in this document
    pub term_freq: TermFreq,
    
    /// Positions of the term in the document (optional)
    pub positions: Option<Vec<Position>>,
}

impl Posting {
    /// Create a new posting
    pub fn new(doc_id: DocId, term_freq: TermFreq) -> Self {
        Self {
            doc_id,
            term_freq,
            positions: None,
        }
    }
    
    /// Create with positions
    pub fn with_positions(doc_id: DocId, positions: Vec<Position>) -> Self {
        Self {
            doc_id,
            term_freq: positions.len() as TermFreq,
            positions: Some(positions),
        }
    }
}

/// A posting list for a term
#[derive(Debug, Clone, Default)]
pub struct PostingList {
    /// Postings sorted by doc_id for efficient merge
    postings: Vec<Posting>,
}

impl PostingList {
    /// Create a new empty posting list
    pub fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }
    
    /// Add a posting (maintains sorted order)
    pub fn add(&mut self, posting: Posting) {
        match self.postings.binary_search_by_key(&posting.doc_id, |p| p.doc_id) {
            Ok(idx) => {
                // Update existing
                self.postings[idx] = posting;
            }
            Err(idx) => {
                // Insert at correct position
                self.postings.insert(idx, posting);
            }
        }
    }
    
    /// Get posting for a document
    pub fn get(&self, doc_id: DocId) -> Option<&Posting> {
        self.postings
            .binary_search_by_key(&doc_id, |p| p.doc_id)
            .ok()
            .map(|idx| &self.postings[idx])
    }
    
    /// Number of documents containing this term
    pub fn doc_freq(&self) -> usize {
        self.postings.len()
    }
    
    /// Iterate over postings
    pub fn iter(&self) -> impl Iterator<Item = &Posting> {
        self.postings.iter()
    }
    
    /// Get all document IDs
    pub fn doc_ids(&self) -> Vec<DocId> {
        self.postings.iter().map(|p| p.doc_id).collect()
    }
}

// ============================================================================
// Document Info
// ============================================================================

/// Information about an indexed document
#[derive(Debug, Clone)]
pub struct DocumentInfo {
    /// Document length (in tokens)
    pub length: u32,
    
    /// Term frequencies
    pub term_freqs: HashMap<String, TermFreq>,
}

// ============================================================================
// Inverted Index
// ============================================================================

/// Inverted index for lexical search
pub struct InvertedIndex {
    /// Term to posting list mapping
    index: RwLock<HashMap<String, PostingList>>,
    
    /// Document info (for BM25 scoring)
    docs: RwLock<HashMap<DocId, DocumentInfo>>,
    
    /// BM25 scorer
    scorer: RwLock<BM25Scorer>,
    
    /// Next document ID
    next_doc_id: RwLock<DocId>,
    
    /// Whether to store positions
    store_positions: bool,
}

impl InvertedIndex {
    /// Create a new inverted index
    pub fn new(config: BM25Config) -> Self {
        Self {
            index: RwLock::new(HashMap::new()),
            docs: RwLock::new(HashMap::new()),
            scorer: RwLock::new(BM25Scorer::new(config)),
            next_doc_id: RwLock::new(0),
            store_positions: false,
        }
    }
    
    /// Enable position storage (for phrase queries)
    pub fn with_positions(mut self) -> Self {
        self.store_positions = true;
        self
    }
    
    /// Index a document
    ///
    /// Returns the assigned document ID.
    pub fn add_document(&self, text: &str) -> DocId {
        let tokens = tokenize_minimal(text);
        self.add_document_tokens(&tokens)
    }
    
    /// Index a document with specific ID
    pub fn add_document_with_id(&self, doc_id: DocId, text: &str) {
        let tokens = tokenize_minimal(text);
        self.add_document_tokens_with_id(doc_id, &tokens);
    }
    
    /// Index a document from tokens
    pub fn add_document_tokens(&self, tokens: &[String]) -> DocId {
        let doc_id = {
            let mut next = self.next_doc_id.write();
            let id = *next;
            *next += 1;
            id
        };
        
        self.add_document_tokens_with_id(doc_id, tokens);
        doc_id
    }
    
    /// Index a document from tokens with specific ID
    pub fn add_document_tokens_with_id(&self, doc_id: DocId, tokens: &[String]) {
        // Build term frequencies and positions
        let mut term_freqs: HashMap<String, TermFreq> = HashMap::new();
        let mut term_positions: HashMap<String, Vec<Position>> = HashMap::new();
        
        for (pos, token) in tokens.iter().enumerate() {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
            if self.store_positions {
                term_positions
                    .entry(token.clone())
                    .or_default()
                    .push(pos as Position);
            }
        }
        
        // Update index
        {
            let mut index = self.index.write();
            for (term, tf) in &term_freqs {
                let posting = if self.store_positions {
                    Posting::with_positions(
                        doc_id,
                        term_positions.get(term).cloned().unwrap_or_default(),
                    )
                } else {
                    Posting::new(doc_id, *tf)
                };
                
                index
                    .entry(term.clone())
                    .or_default()
                    .add(posting);
            }
        }
        
        // Update document info
        {
            let mut docs = self.docs.write();
            docs.insert(doc_id, DocumentInfo {
                length: tokens.len() as u32,
                term_freqs,
            });
        }
        
        // Update BM25 scorer
        {
            let mut scorer = self.scorer.write();
            scorer.add_document(tokens.iter().map(|s| s.as_str()));
        }
    }
    
    /// Remove a document from the index
    pub fn remove_document(&self, doc_id: DocId) -> bool {
        let doc_info = {
            let mut docs = self.docs.write();
            docs.remove(&doc_id)
        };
        
        if let Some(info) = doc_info {
            let mut index = self.index.write();
            for term in info.term_freqs.keys() {
                if let Some(posting_list) = index.get_mut(term) {
                    posting_list.postings.retain(|p| p.doc_id != doc_id);
                }
            }
            true
        } else {
            false
        }
    }
    
    /// Search the index
    ///
    /// Returns document IDs with scores, sorted by score descending.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(DocId, f32)> {
        let query_tokens = tokenize_minimal(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }
        
        self.search_tokens(&query_tokens, limit)
    }
    
    /// Search with pre-tokenized query
    pub fn search_tokens(&self, query_tokens: &[String], limit: usize) -> Vec<(DocId, f32)> {
        if query_tokens.is_empty() {
            return Vec::new();
        }
        
        let index = self.index.read();
        let docs = self.docs.read();
        let scorer = self.scorer.read();
        
        // Collect candidate documents (union of posting lists)
        let mut candidates: HashSet<DocId> = HashSet::new();
        for token in query_tokens {
            if let Some(posting_list) = index.get(token) {
                for posting in posting_list.iter() {
                    candidates.insert(posting.doc_id);
                }
            }
        }
        
        // Score candidates
        let mut results: Vec<(DocId, f32)> = candidates
            .into_iter()
            .filter_map(|doc_id| {
                let doc_info = docs.get(&doc_id)?;
                let score = scorer.score_with_tf(
                    query_tokens,
                    &doc_info.term_freqs.iter()
                        .map(|(k, &v)| (k.clone(), v as usize))
                        .collect(),
                    doc_info.length as usize,
                );
                if score > 0.0 {
                    Some((doc_id, score))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Limit results
        results.truncate(limit);
        
        results
    }
    
    /// Get posting list for a term
    pub fn get_posting_list(&self, term: &str) -> Option<PostingList> {
        self.index.read().get(&term.to_lowercase()).cloned()
    }
    
    /// Get document count
    pub fn num_documents(&self) -> usize {
        self.docs.read().len()
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.index.read().len()
    }
    
    /// Get document info
    pub fn get_document_info(&self, doc_id: DocId) -> Option<DocumentInfo> {
        self.docs.read().get(&doc_id).cloned()
    }
    
    /// Check if a document exists
    pub fn has_document(&self, doc_id: DocId) -> bool {
        self.docs.read().contains_key(&doc_id)
    }
}

// ============================================================================
// Inverted Index Builder
// ============================================================================

/// Builder for batch index construction
pub struct InvertedIndexBuilder {
    config: BM25Config,
    store_positions: bool,
}

impl InvertedIndexBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: BM25Config::default(),
            store_positions: false,
        }
    }
    
    /// Set BM25 configuration
    pub fn with_config(mut self, config: BM25Config) -> Self {
        self.config = config;
        self
    }
    
    /// Enable position storage
    pub fn with_positions(mut self) -> Self {
        self.store_positions = true;
        self
    }
    
    /// Build index from documents
    pub fn build<I>(self, documents: I) -> InvertedIndex
    where
        I: IntoIterator<Item = (DocId, String)>,
    {
        let index = if self.store_positions {
            InvertedIndex::new(self.config).with_positions()
        } else {
            InvertedIndex::new(self.config)
        };
        
        for (doc_id, text) in documents {
            index.add_document_with_id(doc_id, &text);
        }
        
        index
    }
}

impl Default for InvertedIndexBuilder {
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
    
    #[test]
    fn test_posting_list() {
        let mut list = PostingList::new();
        
        list.add(Posting::new(1, 2));
        list.add(Posting::new(3, 1));
        list.add(Posting::new(2, 3));
        
        assert_eq!(list.doc_freq(), 3);
        
        // Should be sorted by doc_id
        let ids = list.doc_ids();
        assert_eq!(ids, vec![1, 2, 3]);
        
        // Get specific posting
        let p = list.get(2).unwrap();
        assert_eq!(p.term_freq, 3);
    }
    
    #[test]
    fn test_add_document() {
        let index = InvertedIndex::new(BM25Config::default());
        
        let doc1 = index.add_document("hello world");
        let doc2 = index.add_document("hello there");
        
        assert_eq!(doc1, 0);
        assert_eq!(doc2, 1);
        assert_eq!(index.num_documents(), 2);
        
        // Check posting list
        let hello_list = index.get_posting_list("hello").unwrap();
        assert_eq!(hello_list.doc_freq(), 2);
    }
    
    #[test]
    fn test_search() {
        let index = InvertedIndex::new(BM25Config::default());
        
        index.add_document("the quick brown fox jumps over the lazy dog");
        index.add_document("quick quick quick fox"); // High TF for "quick"
        index.add_document("lazy lazy lazy dog");    // High TF for "lazy"
        
        // Search for "quick"
        let results = index.search("quick", 10);
        assert!(!results.is_empty());
        
        // Doc with highest TF for "quick" should score highest
        assert_eq!(results[0].0, 1); // doc_id 1 has "quick quick quick"
    }
    
    #[test]
    fn test_search_multi_term() {
        let index = InvertedIndex::new(BM25Config::default());
        
        index.add_document("apple banana cherry");
        index.add_document("apple banana");
        index.add_document("apple");
        
        // Multi-term query
        let results = index.search("apple banana cherry", 10);
        
        // Doc with most terms should score highest
        assert_eq!(results[0].0, 0);
    }
    
    #[test]
    fn test_remove_document() {
        let index = InvertedIndex::new(BM25Config::default());
        
        let doc1 = index.add_document("hello world");
        let doc2 = index.add_document("hello there");
        
        assert!(index.has_document(doc1));
        assert!(index.remove_document(doc1));
        assert!(!index.has_document(doc1));
        
        // "hello" should still exist (in doc2)
        let hello_list = index.get_posting_list("hello").unwrap();
        assert_eq!(hello_list.doc_freq(), 1);
        
        // "world" should be gone
        let world_list = index.get_posting_list("world").unwrap();
        assert_eq!(world_list.doc_freq(), 0);
    }
    
    #[test]
    fn test_builder() {
        let documents = vec![
            (0, "hello world".to_string()),
            (1, "hello there".to_string()),
            (2, "goodbye world".to_string()),
        ];
        
        let index = InvertedIndexBuilder::new()
            .with_config(BM25Config::lucene())
            .build(documents);
        
        assert_eq!(index.num_documents(), 3);
        assert!(index.vocab_size() > 0);
    }
    
    #[test]
    fn test_positions() {
        let index = InvertedIndex::new(BM25Config::default()).with_positions();
        
        let doc_id = index.add_document("hello world hello");
        
        let hello_list = index.get_posting_list("hello").unwrap();
        let posting = hello_list.get(doc_id).unwrap();
        
        assert_eq!(posting.positions, Some(vec![0, 2]));
    }
}
