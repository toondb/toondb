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

//! Multi-Vector Documents with Stable Aggregation Semantics (Task 5)
//!
//! This module enables documents to have multiple vectors (e.g., for chunks/paragraphs)
//! with deterministic aggregation during search.
//!
//! ## Design
//!
//! ```text
//! Document (doc_id=123)
//! ├── Chunk 0 → Vector 0 (internal_id=1000)
//! ├── Chunk 1 → Vector 1 (internal_id=1001)
//! ├── Chunk 2 → Vector 2 (internal_id=1002)
//! └── Chunk 3 → Vector 3 (internal_id=1003)
//!
//! Search: query → [1001, 1003, 1002] (internal IDs with scores)
//!        → Aggregate by doc_id → doc_123: max(score(1001), score(1003), score(1002))
//! ```
//!
//! ## Aggregation Methods
//!
//! - **Max**: Use the best-matching chunk's score (ColBERT-like late interaction)
//! - **Mean**: Average all chunk scores (good for comprehensive coverage)
//! - **First**: Use the first chunk's score (for ordered content)
//!
//! ## API
//!
//! ```ignore
//! // Insert multi-vector document
//! collection.insert_multi(
//!     doc_id="doc_123",
//!     vectors=[v1, v2, v3, v4],
//!     metadata={...},
//! )
//!
//! // Search with aggregation
//! collection.search(
//!     query,
//!     aggregate="max",  // max|mean|first
//! )
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

// ============================================================================
// Types
// ============================================================================

/// Document ID (user-provided, stable identifier)
pub type DocId = String;

/// Internal vector ID (storage-assigned)
pub type InternalId = u64;

/// Chunk/part index within a document
pub type ChunkIndex = u32;

// ============================================================================
// Aggregation
// ============================================================================

/// Aggregation method for multi-vector documents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationMethod {
    /// Use the maximum score across all chunks (recommended for most use cases)
    /// This is equivalent to "did any chunk match well?"
    #[default]
    Max,
    
    /// Use the average score across all chunks
    /// Good for measuring overall document relevance
    Mean,
    
    /// Use the score of the first chunk
    /// Good for documents where order matters (e.g., abstracts first)
    First,
    
    /// Use the score of the last chunk
    Last,
    
    /// Sum of all chunk scores (for sparse-like behavior)
    Sum,
}

impl AggregationMethod {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "max" => Some(Self::Max),
            "mean" | "avg" | "average" => Some(Self::Mean),
            "first" => Some(Self::First),
            "last" => Some(Self::Last),
            "sum" => Some(Self::Sum),
            _ => None,
        }
    }
}

/// Aggregate scores for a document
#[derive(Debug, Clone)]
pub struct DocumentScore {
    /// Document ID
    pub doc_id: DocId,
    
    /// Aggregated score
    pub score: f32,
    
    /// Best matching chunk index (for max aggregation)
    pub best_chunk: Option<ChunkIndex>,
    
    /// Number of chunks that matched
    pub matched_chunks: usize,
    
    /// All chunk scores (optional, for debugging)
    pub chunk_scores: Option<Vec<(ChunkIndex, f32)>>,
}

impl DocumentScore {
    /// Create from chunk scores with aggregation
    pub fn aggregate(
        doc_id: DocId,
        chunk_scores: Vec<(ChunkIndex, f32)>,
        method: AggregationMethod,
        keep_details: bool,
    ) -> Self {
        if chunk_scores.is_empty() {
            return Self {
                doc_id,
                score: 0.0,
                best_chunk: None,
                matched_chunks: 0,
                chunk_scores: if keep_details { Some(Vec::new()) } else { None },
            };
        }
        
        let matched_chunks = chunk_scores.len();
        
        let (score, best_chunk) = match method {
            AggregationMethod::Max => {
                let (idx, &(chunk, score)) = chunk_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                (score, Some(chunk))
            }
            AggregationMethod::Mean => {
                let sum: f32 = chunk_scores.iter().map(|(_, s)| s).sum();
                (sum / chunk_scores.len() as f32, None)
            }
            AggregationMethod::First => {
                let (chunk, score) = chunk_scores
                    .iter()
                    .min_by_key(|(idx, _)| *idx)
                    .copied()
                    .unwrap();
                (score, Some(chunk))
            }
            AggregationMethod::Last => {
                let (chunk, score) = chunk_scores
                    .iter()
                    .max_by_key(|(idx, _)| *idx)
                    .copied()
                    .unwrap();
                (score, Some(chunk))
            }
            AggregationMethod::Sum => {
                let sum: f32 = chunk_scores.iter().map(|(_, s)| s).sum();
                (sum, None)
            }
        };
        
        Self {
            doc_id,
            score,
            best_chunk,
            matched_chunks,
            chunk_scores: if keep_details { Some(chunk_scores) } else { None },
        }
    }
}

// ============================================================================
// Multi-Vector Index Mapping
// ============================================================================

/// Mapping from internal vector IDs to document IDs and chunk indices
#[derive(Debug, Clone)]
pub struct MultiVectorMapping {
    /// Map from internal ID to (doc_id, chunk_index)
    internal_to_doc: HashMap<InternalId, (DocId, ChunkIndex)>,
    
    /// Map from doc_id to list of internal IDs (ordered by chunk index)
    doc_to_internal: HashMap<DocId, Vec<InternalId>>,
    
    /// Next internal ID to assign
    next_internal_id: InternalId,
}

impl MultiVectorMapping {
    /// Create a new empty mapping
    pub fn new() -> Self {
        Self {
            internal_to_doc: HashMap::new(),
            doc_to_internal: HashMap::new(),
            next_internal_id: 0,
        }
    }
    
    /// Insert a multi-vector document, returning the internal IDs
    pub fn insert_document(&mut self, doc_id: DocId, num_chunks: usize) -> Vec<InternalId> {
        // Remove existing if present
        self.remove_document(&doc_id);
        
        let mut internal_ids = Vec::with_capacity(num_chunks);
        
        for chunk_idx in 0..num_chunks {
            let internal_id = self.next_internal_id;
            self.next_internal_id += 1;
            
            self.internal_to_doc.insert(internal_id, (doc_id.clone(), chunk_idx as ChunkIndex));
            internal_ids.push(internal_id);
        }
        
        self.doc_to_internal.insert(doc_id, internal_ids.clone());
        
        internal_ids
    }
    
    /// Remove a document and its vectors
    pub fn remove_document(&mut self, doc_id: &str) -> Option<Vec<InternalId>> {
        if let Some(internal_ids) = self.doc_to_internal.remove(doc_id) {
            for id in &internal_ids {
                self.internal_to_doc.remove(id);
            }
            Some(internal_ids)
        } else {
            None
        }
    }
    
    /// Lookup document ID and chunk index for an internal ID
    #[inline]
    pub fn get_doc(&self, internal_id: InternalId) -> Option<(&DocId, ChunkIndex)> {
        self.internal_to_doc.get(&internal_id).map(|(d, c)| (d, *c))
    }
    
    /// Get all internal IDs for a document
    pub fn get_internal_ids(&self, doc_id: &str) -> Option<&[InternalId]> {
        self.doc_to_internal.get(doc_id).map(|v| v.as_slice())
    }
    
    /// Check if a document exists
    pub fn has_document(&self, doc_id: &str) -> bool {
        self.doc_to_internal.contains_key(doc_id)
    }
    
    /// Get the number of documents
    pub fn num_documents(&self) -> usize {
        self.doc_to_internal.len()
    }
    
    /// Get the total number of vectors
    pub fn num_vectors(&self) -> usize {
        self.internal_to_doc.len()
    }
}

impl Default for MultiVectorMapping {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-Vector Aggregator
// ============================================================================

/// Aggregates search results from vector level to document level
pub struct MultiVectorAggregator {
    /// Mapping from internal IDs to documents
    mapping: Arc<RwLock<MultiVectorMapping>>,
    
    /// Default aggregation method
    default_method: AggregationMethod,
}

impl MultiVectorAggregator {
    /// Create a new aggregator
    pub fn new(mapping: Arc<RwLock<MultiVectorMapping>>) -> Self {
        Self {
            mapping,
            default_method: AggregationMethod::Max,
        }
    }
    
    /// Set the default aggregation method
    pub fn with_default_method(mut self, method: AggregationMethod) -> Self {
        self.default_method = method;
        self
    }
    
    /// Aggregate vector search results to document results
    ///
    /// Input: Vec<(internal_id, score)> from vector search
    /// Output: Vec<DocumentScore> sorted by aggregated score
    pub fn aggregate(
        &self,
        vector_results: &[(InternalId, f32)],
        method: Option<AggregationMethod>,
        limit: usize,
    ) -> Vec<DocumentScore> {
        let method = method.unwrap_or(self.default_method);
        let mapping = self.mapping.read();
        
        // Group by document
        let mut doc_chunks: HashMap<&DocId, Vec<(ChunkIndex, f32)>> = HashMap::new();
        
        for &(internal_id, score) in vector_results {
            if let Some((doc_id, chunk_idx)) = mapping.get_doc(internal_id) {
                doc_chunks
                    .entry(doc_id)
                    .or_default()
                    .push((chunk_idx, score));
            }
        }
        
        // Aggregate each document
        let mut results: Vec<DocumentScore> = doc_chunks
            .into_iter()
            .map(|(doc_id, chunks)| {
                DocumentScore::aggregate(doc_id.clone(), chunks, method, false)
            })
            .collect();
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Limit
        results.truncate(limit);
        
        results
    }
    
    /// Aggregate with detailed chunk information
    pub fn aggregate_detailed(
        &self,
        vector_results: &[(InternalId, f32)],
        method: Option<AggregationMethod>,
        limit: usize,
    ) -> Vec<DocumentScore> {
        let method = method.unwrap_or(self.default_method);
        let mapping = self.mapping.read();
        
        // Group by document
        let mut doc_chunks: HashMap<&DocId, Vec<(ChunkIndex, f32)>> = HashMap::new();
        
        for &(internal_id, score) in vector_results {
            if let Some((doc_id, chunk_idx)) = mapping.get_doc(internal_id) {
                doc_chunks
                    .entry(doc_id)
                    .or_default()
                    .push((chunk_idx, score));
            }
        }
        
        // Aggregate each document with details
        let mut results: Vec<DocumentScore> = doc_chunks
            .into_iter()
            .map(|(doc_id, chunks)| {
                DocumentScore::aggregate(doc_id.clone(), chunks, method, true)
            })
            .collect();
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Limit
        results.truncate(limit);
        
        results
    }
}

// ============================================================================
// Multi-Vector Collection (High-Level API)
// ============================================================================

/// Configuration for multi-vector storage
#[derive(Debug, Clone)]
pub struct MultiVectorConfig {
    /// Maximum chunks per document
    pub max_chunks_per_doc: usize,
    
    /// Default aggregation method
    pub default_aggregation: AggregationMethod,
    
    /// Over-fetch factor for ensuring enough unique documents
    pub overfetch_factor: f32,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            max_chunks_per_doc: 1000,
            default_aggregation: AggregationMethod::Max,
            overfetch_factor: 2.0,
        }
    }
}

/// Multi-vector document for insertion
#[derive(Debug, Clone)]
pub struct MultiVectorDocument {
    /// Document ID (stable, user-provided)
    pub id: DocId,
    
    /// Vectors for each chunk
    pub vectors: Vec<Vec<f32>>,
    
    /// Optional: text content for each chunk (for hybrid search)
    pub chunks_text: Option<Vec<String>>,
    
    /// Document-level metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MultiVectorDocument {
    /// Create a new multi-vector document
    pub fn new(id: impl Into<DocId>, vectors: Vec<Vec<f32>>) -> Self {
        Self {
            id: id.into(),
            vectors,
            chunks_text: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Add chunk text content
    pub fn with_text(mut self, chunks: Vec<String>) -> Self {
        self.chunks_text = Some(chunks);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Number of chunks
    pub fn num_chunks(&self) -> usize {
        self.vectors.len()
    }
    
    /// Validate the document
    pub fn validate(&self, expected_dim: usize) -> Result<(), MultiVectorError> {
        if self.vectors.is_empty() {
            return Err(MultiVectorError::NoVectors);
        }
        
        for (i, v) in self.vectors.iter().enumerate() {
            if v.len() != expected_dim {
                return Err(MultiVectorError::DimensionMismatch {
                    chunk: i,
                    expected: expected_dim,
                    actual: v.len(),
                });
            }
        }
        
        if let Some(ref texts) = self.chunks_text {
            if texts.len() != self.vectors.len() {
                return Err(MultiVectorError::ChunkCountMismatch {
                    vectors: self.vectors.len(),
                    texts: texts.len(),
                });
            }
        }
        
        Ok(())
    }
}

/// Errors for multi-vector operations
#[derive(Debug, thiserror::Error)]
pub enum MultiVectorError {
    #[error("document must have at least one vector")]
    NoVectors,
    
    #[error("dimension mismatch in chunk {chunk}: expected {expected}, got {actual}")]
    DimensionMismatch {
        chunk: usize,
        expected: usize,
        actual: usize,
    },
    
    #[error("chunk count mismatch: {vectors} vectors but {texts} texts")]
    ChunkCountMismatch { vectors: usize, texts: usize },
    
    #[error("too many chunks: {count} exceeds limit of {limit}")]
    TooManyChunks { count: usize, limit: usize },
    
    #[error("document not found: {0}")]
    NotFound(DocId),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aggregation_max() {
        let chunks = vec![
            (0, 0.5),
            (1, 0.9),
            (2, 0.3),
        ];
        
        let result = DocumentScore::aggregate(
            "doc1".to_string(),
            chunks,
            AggregationMethod::Max,
            false,
        );
        
        assert_eq!(result.score, 0.9);
        assert_eq!(result.best_chunk, Some(1));
        assert_eq!(result.matched_chunks, 3);
    }
    
    #[test]
    fn test_aggregation_mean() {
        let chunks = vec![
            (0, 0.6),
            (1, 0.9),
            (2, 0.3),
        ];
        
        let result = DocumentScore::aggregate(
            "doc1".to_string(),
            chunks,
            AggregationMethod::Mean,
            false,
        );
        
        assert!((result.score - 0.6).abs() < 0.001); // (0.6 + 0.9 + 0.3) / 3 = 0.6
    }
    
    #[test]
    fn test_aggregation_first() {
        let chunks = vec![
            (2, 0.3),
            (0, 0.5),
            (1, 0.9),
        ];
        
        let result = DocumentScore::aggregate(
            "doc1".to_string(),
            chunks,
            AggregationMethod::First,
            false,
        );
        
        assert_eq!(result.score, 0.5); // Chunk 0 has score 0.5
        assert_eq!(result.best_chunk, Some(0));
    }
    
    #[test]
    fn test_mapping_insert() {
        let mut mapping = MultiVectorMapping::new();
        
        let ids = mapping.insert_document("doc1".to_string(), 3);
        assert_eq!(ids.len(), 3);
        
        // Check reverse lookup
        for (i, &id) in ids.iter().enumerate() {
            let (doc_id, chunk) = mapping.get_doc(id).unwrap();
            assert_eq!(doc_id, "doc1");
            assert_eq!(chunk as usize, i);
        }
    }
    
    #[test]
    fn test_mapping_remove() {
        let mut mapping = MultiVectorMapping::new();
        
        let ids = mapping.insert_document("doc1".to_string(), 3);
        
        let removed = mapping.remove_document("doc1").unwrap();
        assert_eq!(removed, ids);
        
        // Should not be found
        assert!(mapping.get_doc(ids[0]).is_none());
        assert!(!mapping.has_document("doc1"));
    }
    
    #[test]
    fn test_aggregator() {
        let mapping = Arc::new(RwLock::new(MultiVectorMapping::new()));
        
        // Insert two documents
        {
            let mut m = mapping.write();
            m.insert_document("doc1".to_string(), 3); // IDs 0, 1, 2
            m.insert_document("doc2".to_string(), 2); // IDs 3, 4
        }
        
        let aggregator = MultiVectorAggregator::new(mapping);
        
        // Simulate search results
        let vector_results = vec![
            (1, 0.95), // doc1, chunk 1
            (3, 0.90), // doc2, chunk 0
            (0, 0.85), // doc1, chunk 0
            (4, 0.80), // doc2, chunk 1
        ];
        
        let doc_results = aggregator.aggregate(&vector_results, Some(AggregationMethod::Max), 10);
        
        assert_eq!(doc_results.len(), 2);
        assert_eq!(doc_results[0].doc_id, "doc1");
        assert_eq!(doc_results[0].score, 0.95);
        assert_eq!(doc_results[1].doc_id, "doc2");
        assert_eq!(doc_results[1].score, 0.90);
    }
    
    #[test]
    fn test_multi_vector_document() {
        let doc = MultiVectorDocument::new("doc1", vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ])
        .with_text(vec!["chunk 1".to_string(), "chunk 2".to_string()])
        .with_metadata("author", serde_json::json!("Alice"));
        
        assert_eq!(doc.num_chunks(), 2);
        assert!(doc.validate(3).is_ok());
        assert!(doc.validate(4).is_err()); // Wrong dimension
    }
}
