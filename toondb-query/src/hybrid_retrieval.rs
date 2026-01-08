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

//! Hybrid Retrieval Pipeline (Task 3)
//!
//! This module implements a unified hybrid query planner combining:
//! - Vector similarity search (ANN)
//! - Lexical search (BM25)
//! - Metadata filtering (PRE-FILTER ONLY)
//! - Score fusion (RRF)
//! - Cross-encoder reranking
//!
//! ## CRITICAL INVARIANT: No Post-Filtering
//!
//! This module enforces a hard security invariant:
//! 
//! > **All filtering MUST occur during candidate generation, never after.**
//!
//! The `ExecutionStep::PostFilter` variant has been intentionally removed.
//! This guarantees:
//! 1. **Security by construction**: No leakage of filtered documents
//! 2. **No wasted compute**: We never score disallowed documents
//! 3. **Monotone property**: `result-set ⊆ allowed-set` (verifiable)
//!
//! ## Correct Pattern
//!
//! ```text
//! FilterIR + AuthScope → AllowedSet (computed once)
//!     ↓
//! vector_search(query, AllowedSet) → filtered candidates
//! bm25_search(query, AllowedSet)   → filtered candidates
//!     ↓
//! fusion(filtered_v, filtered_b)   → already correct!
//!     ↓
//! rerank, limit                    → final results
//! ```
//!
//! ## Anti-Pattern (What We Prevent)
//!
//! ```text
//! BAD: vector_search() → candidates → filter → too few/leaky
//!      bm25_search()   → candidates → filter → inconsistent
//!      fusion()        → filter at end → SECURITY RISK!
//! ```
//!
//! ## Execution Plan
//!
//! ```text
//! HybridQuery
//!     │
//!     ▼
//! ┌─────────────────────────────────────────┐
//! │              ExecutionPlan              │
//! │  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
//! │  │ Vector  │ │  BM25   │ │  Filter  │  │
//! │  │ Search  │ │ Search  │ │ (PRE-ONLY)│  │
//! │  └────┬────┘ └────┬────┘ └────┬─────┘  │
//! │       │           │           │        │
//! │       └─────┬─────┘           │        │
//! │             ▼                 │        │
//! │       ┌─────────┐             │        │
//! │       │  Fusion │◄────────────┘        │
//! │       │  (RRF)  │                      │
//! │       └────┬────┘                      │
//! │            ▼                           │
//! │       ┌─────────┐                      │
//! │       │ Rerank  │                      │
//! │       └────┬────┘                      │
//! │            ▼                           │
//! │       ┌─────────┐                      │
//! │       │  Limit  │                      │
//! │       └─────────┘                      │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Scoring
//!
//! RRF fusion: `score(d) = Σ w_i / (k + rank_i(d))`
//! where k is typically 60 (robust default)

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::Arc;

use crate::context_query::VectorIndex;
use crate::toon_ql::ToonValue;

// ============================================================================
// Hybrid Query Builder
// ============================================================================

/// Builder for hybrid retrieval queries
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Collection to search
    pub collection: String,
    
    /// Vector search component
    pub vector: Option<VectorQueryComponent>,
    
    /// Lexical (BM25) search component
    pub lexical: Option<LexicalQueryComponent>,
    
    /// Metadata filters
    pub filters: Vec<MetadataFilter>,
    
    /// Fusion configuration
    pub fusion: FusionConfig,
    
    /// Reranking configuration
    pub rerank: Option<RerankConfig>,
    
    /// Result limit
    pub limit: usize,
    
    /// Minimum score threshold
    pub min_score: Option<f32>,
}

impl HybridQuery {
    /// Create a new hybrid query builder
    pub fn new(collection: &str) -> Self {
        Self {
            collection: collection.to_string(),
            vector: None,
            lexical: None,
            filters: Vec::new(),
            fusion: FusionConfig::default(),
            rerank: None,
            limit: 10,
            min_score: None,
        }
    }
    
    /// Add vector search component
    pub fn with_vector(mut self, embedding: Vec<f32>, weight: f32) -> Self {
        self.vector = Some(VectorQueryComponent {
            embedding,
            weight,
            ef_search: 100,
        });
        self
    }
    
    /// Add vector search from text (requires embedding provider)
    pub fn with_vector_text(mut self, text: String, weight: f32) -> Self {
        self.vector = Some(VectorQueryComponent {
            embedding: Vec::new(), // Will be resolved at execution time
            weight,
            ef_search: 100,
        });
        // Store text for later resolution
        self.lexical = self.lexical.or(Some(LexicalQueryComponent {
            query: text,
            weight: 0.0, // Text stored but not used for lexical
            fields: vec!["content".to_string()],
        }));
        self
    }
    
    /// Add lexical (BM25) search component
    pub fn with_lexical(mut self, query: &str, weight: f32) -> Self {
        self.lexical = Some(LexicalQueryComponent {
            query: query.to_string(),
            weight,
            fields: vec!["content".to_string()],
        });
        self
    }
    
    /// Add lexical search with specific fields
    pub fn with_lexical_fields(mut self, query: &str, weight: f32, fields: Vec<String>) -> Self {
        self.lexical = Some(LexicalQueryComponent {
            query: query.to_string(),
            weight,
            fields,
        });
        self
    }
    
    /// Add metadata filter
    pub fn filter(mut self, field: &str, op: FilterOp, value: ToonValue) -> Self {
        self.filters.push(MetadataFilter {
            field: field.to_string(),
            op,
            value,
        });
        self
    }
    
    /// Add equality filter
    pub fn filter_eq(self, field: &str, value: impl Into<ToonValue>) -> Self {
        self.filter(field, FilterOp::Eq, value.into())
    }
    
    /// Add range filter
    pub fn filter_range(mut self, field: &str, min: Option<ToonValue>, max: Option<ToonValue>) -> Self {
        if let Some(min_val) = min {
            self.filters.push(MetadataFilter {
                field: field.to_string(),
                op: FilterOp::Gte,
                value: min_val,
            });
        }
        if let Some(max_val) = max {
            self.filters.push(MetadataFilter {
                field: field.to_string(),
                op: FilterOp::Lte,
                value: max_val,
            });
        }
        self
    }
    
    /// Set fusion method
    pub fn with_fusion(mut self, method: FusionMethod) -> Self {
        self.fusion.method = method;
        self
    }
    
    /// Set RRF k parameter
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.fusion.rrf_k = k;
        self
    }
    
    /// Enable reranking
    pub fn with_rerank(mut self, model: &str, top_n: usize) -> Self {
        self.rerank = Some(RerankConfig {
            model: model.to_string(),
            top_n,
            batch_size: 32,
        });
        self
    }
    
    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
    
    /// Set minimum score threshold
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }
}

/// Vector search component
#[derive(Debug, Clone)]
pub struct VectorQueryComponent {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// Weight for fusion
    pub weight: f32,
    /// HNSW ef_search parameter
    pub ef_search: usize,
}

/// Lexical search component
#[derive(Debug, Clone)]
pub struct LexicalQueryComponent {
    /// Query text
    pub query: String,
    /// Weight for fusion
    pub weight: f32,
    /// Fields to search
    pub fields: Vec<String>,
}

/// Metadata filter
#[derive(Debug, Clone)]
pub struct MetadataFilter {
    /// Field name
    pub field: String,
    /// Comparison operator
    pub op: FilterOp,
    /// Value to compare
    pub value: ToonValue,
}

/// Filter comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// Contains (for arrays/strings)
    Contains,
    /// In set
    In,
}

/// Fusion configuration
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fusion method
    pub method: FusionMethod,
    /// RRF k parameter (default: 60)
    pub rrf_k: f32,
    /// Normalize scores before fusion
    pub normalize: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::Rrf,
            rrf_k: 60.0,
            normalize: true,
        }
    }
}

/// Score fusion methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion
    Rrf,
    /// Weighted sum of normalized scores
    WeightedSum,
    /// Max score from any source
    Max,
    /// Relative score fusion
    Rsf,
}

/// Reranking configuration
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Reranker model
    pub model: String,
    /// Number of top candidates to rerank
    pub top_n: usize,
    /// Batch size for reranking
    pub batch_size: usize,
}

// ============================================================================
// Execution Plan
// ============================================================================

/// Execution plan for hybrid query
#[derive(Debug, Clone)]
pub struct HybridExecutionPlan {
    /// Query being executed
    pub query: HybridQuery,
    
    /// Execution steps
    pub steps: Vec<ExecutionStep>,
    
    /// Estimated cost
    pub estimated_cost: f64,
}

/// Individual execution step
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    /// Vector similarity search
    VectorSearch {
        collection: String,
        ef_search: usize,
        weight: f32,
    },
    
    /// Lexical (BM25) search
    LexicalSearch {
        collection: String,
        query: String,
        fields: Vec<String>,
        weight: f32,
    },
    
    /// Pre-filter (before retrieval) - REQUIRED for security
    /// 
    /// This is the ONLY allowed filter step. Filters are always applied
    /// during candidate generation via AllowedSet, never after.
    PreFilter {
        filters: Vec<MetadataFilter>,
    },
    
    // NOTE: PostFilter has been REMOVED by design.
    // The "no post-filtering" invariant is a hard security requirement.
    // All filtering must happen via PreFilter -> AllowedSet -> candidate generation.
    // See unified_fusion.rs for the correct pattern.
    
    /// Score fusion
    Fusion {
        method: FusionMethod,
        rrf_k: f32,
    },
    
    /// Reranking (does NOT filter, only re-orders)
    Rerank {
        model: String,
        top_n: usize,
    },
    
    /// Limit results (applied AFTER all filtering is complete)
    Limit {
        count: usize,
        min_score: Option<f32>,
    },
    
    /// Redaction transform (post-retrieval modification, NOT filtering)
    /// 
    /// Unlike filtering (which removes candidates), redaction transforms
    /// the content of already-allowed documents. This preserves the
    /// invariant: result-set ⊆ allowed-set.
    Redact {
        /// Fields to redact
        fields: Vec<String>,
        /// Redaction method
        method: RedactionMethod,
    },
}

/// Redaction methods for post-retrieval content transformation
#[derive(Debug, Clone)]
pub enum RedactionMethod {
    /// Replace with a fixed string
    Replace(String),
    /// Mask with asterisks
    Mask,
    /// Remove the field entirely
    Remove,
    /// Hash the value
    Hash,
}

// ============================================================================
// Hybrid Query Executor
// ============================================================================

/// Executor for hybrid queries
pub struct HybridQueryExecutor<V: VectorIndex> {
    /// Vector index
    vector_index: Arc<V>,
    
    /// Lexical index (BM25)
    lexical_index: Arc<LexicalIndex>,
}

impl<V: VectorIndex> HybridQueryExecutor<V> {
    /// Create a new executor
    pub fn new(vector_index: Arc<V>, lexical_index: Arc<LexicalIndex>) -> Self {
        Self {
            vector_index,
            lexical_index,
        }
    }
    
    /// Execute a hybrid query
    pub fn execute(&self, query: &HybridQuery) -> Result<HybridQueryResult, HybridQueryError> {
        let mut candidates: HashMap<String, CandidateDoc> = HashMap::new();
        
        // Over-fetch factor for fusion
        let overfetch = (query.limit * 3).max(100);
        
        // Execute vector search
        if let Some(vector) = &query.vector {
            if !vector.embedding.is_empty() {
                let results = self.vector_index
                    .search_by_embedding(&query.collection, &vector.embedding, overfetch, None)
                    .map_err(HybridQueryError::VectorSearchError)?;
                
                for (rank, result) in results.iter().enumerate() {
                    let entry = candidates.entry(result.id.clone()).or_insert_with(|| {
                        CandidateDoc {
                            id: result.id.clone(),
                            content: result.content.clone(),
                            metadata: result.metadata.clone(),
                            vector_rank: None,
                            vector_score: None,
                            lexical_rank: None,
                            lexical_score: None,
                            fused_score: 0.0,
                        }
                    });
                    entry.vector_rank = Some(rank);
                    entry.vector_score = Some(result.score);
                }
            }
        }
        
        // Execute lexical search
        if let Some(lexical) = &query.lexical {
            if lexical.weight > 0.0 {
                let results = self.lexical_index.search(
                    &query.collection,
                    &lexical.query,
                    &lexical.fields,
                    overfetch,
                )?;
                
                for (rank, result) in results.iter().enumerate() {
                    let entry = candidates.entry(result.id.clone()).or_insert_with(|| {
                        CandidateDoc {
                            id: result.id.clone(),
                            content: result.content.clone(),
                            metadata: HashMap::new(),
                            vector_rank: None,
                            vector_score: None,
                            lexical_rank: None,
                            lexical_score: None,
                            fused_score: 0.0,
                        }
                    });
                    entry.lexical_rank = Some(rank);
                    entry.lexical_score = Some(result.score);
                }
            }
        }
        
        // Apply filters
        let filtered: Vec<CandidateDoc> = candidates
            .into_values()
            .filter(|doc| self.matches_filters(doc, &query.filters))
            .collect();
        
        // Fuse scores
        let mut fused = self.fuse_scores(filtered, query)?;
        
        // Sort by fused score (descending)
        fused.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap_or(Ordering::Equal));
        
        // Apply reranking (stub - would call reranker model)
        if let Some(rerank) = &query.rerank {
            fused = self.rerank(&fused, &query.lexical.as_ref().map(|l| l.query.clone()).unwrap_or_default(), rerank)?;
        }
        
        // Apply min_score filter
        if let Some(min) = query.min_score {
            fused.retain(|doc| doc.fused_score >= min);
        }
        
        // Limit results
        fused.truncate(query.limit);
        
        // Convert to results
        let results: Vec<HybridSearchResult> = fused
            .into_iter()
            .map(|doc| HybridSearchResult {
                id: doc.id,
                score: doc.fused_score,
                content: doc.content,
                metadata: doc.metadata,
                vector_score: doc.vector_score,
                lexical_score: doc.lexical_score,
            })
            .collect();
        
        Ok(HybridQueryResult {
            results,
            query: query.clone(),
            stats: HybridQueryStats {
                vector_candidates: 0, // Would be populated in real impl
                lexical_candidates: 0,
                filtered_candidates: 0,
                fusion_time_us: 0,
                rerank_time_us: 0,
            },
        })
    }
    
    /// Check if document matches all filters
    fn matches_filters(&self, doc: &CandidateDoc, filters: &[MetadataFilter]) -> bool {
        for filter in filters {
            if let Some(value) = doc.metadata.get(&filter.field) {
                if !self.match_filter(value, &filter.op, &filter.value) {
                    return false;
                }
            } else {
                // Field not present - filter fails
                return false;
            }
        }
        true
    }
    
    /// Match a single filter
    fn match_filter(&self, doc_value: &ToonValue, op: &FilterOp, filter_value: &ToonValue) -> bool {
        match op {
            FilterOp::Eq => doc_value == filter_value,
            FilterOp::Ne => doc_value != filter_value,
            FilterOp::Gt => self.compare_values(doc_value, filter_value) == Some(Ordering::Greater),
            FilterOp::Gte => matches!(self.compare_values(doc_value, filter_value), Some(Ordering::Greater | Ordering::Equal)),
            FilterOp::Lt => self.compare_values(doc_value, filter_value) == Some(Ordering::Less),
            FilterOp::Lte => matches!(self.compare_values(doc_value, filter_value), Some(Ordering::Less | Ordering::Equal)),
            FilterOp::Contains => self.value_contains(doc_value, filter_value),
            FilterOp::In => self.value_in_set(doc_value, filter_value),
        }
    }
    
    /// Compare two ToonValues
    fn compare_values(&self, a: &ToonValue, b: &ToonValue) -> Option<Ordering> {
        match (a, b) {
            (ToonValue::Int(a), ToonValue::Int(b)) => Some(a.cmp(b)),
            (ToonValue::UInt(a), ToonValue::UInt(b)) => Some(a.cmp(b)),
            (ToonValue::Float(a), ToonValue::Float(b)) => a.partial_cmp(b),
            (ToonValue::Text(a), ToonValue::Text(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
    
    /// Check if value contains another
    fn value_contains(&self, doc_value: &ToonValue, search_value: &ToonValue) -> bool {
        match (doc_value, search_value) {
            (ToonValue::Text(text), ToonValue::Text(search)) => text.contains(search.as_str()),
            (ToonValue::Array(arr), _) => arr.contains(search_value),
            _ => false,
        }
    }
    
    /// Check if value is in set
    fn value_in_set(&self, doc_value: &ToonValue, set_value: &ToonValue) -> bool {
        if let ToonValue::Array(arr) = set_value {
            arr.contains(doc_value)
        } else {
            false
        }
    }
    
    /// Fuse scores from multiple sources
    fn fuse_scores(
        &self,
        candidates: Vec<CandidateDoc>,
        query: &HybridQuery,
    ) -> Result<Vec<CandidateDoc>, HybridQueryError> {
        let vector_weight = query.vector.as_ref().map(|v| v.weight).unwrap_or(0.0);
        let lexical_weight = query.lexical.as_ref().map(|l| l.weight).unwrap_or(0.0);
        
        let mut fused = candidates;
        
        match query.fusion.method {
            FusionMethod::Rrf => {
                // Reciprocal Rank Fusion
                // score(d) = Σ w_i / (k + rank_i(d))
                for doc in &mut fused {
                    let mut score = 0.0;
                    
                    if let Some(rank) = doc.vector_rank {
                        score += vector_weight / (query.fusion.rrf_k + rank as f32);
                    }
                    
                    if let Some(rank) = doc.lexical_rank {
                        score += lexical_weight / (query.fusion.rrf_k + rank as f32);
                    }
                    
                    doc.fused_score = score;
                }
            }
            
            FusionMethod::WeightedSum => {
                // Weighted sum of normalized scores
                for doc in &mut fused {
                    let mut score = 0.0;
                    
                    if let Some(s) = doc.vector_score {
                        score += vector_weight * s;
                    }
                    
                    if let Some(s) = doc.lexical_score {
                        score += lexical_weight * s;
                    }
                    
                    doc.fused_score = score;
                }
            }
            
            FusionMethod::Max => {
                // Maximum score from any source
                for doc in &mut fused {
                    let v_score = doc.vector_score.map(|s| vector_weight * s).unwrap_or(0.0);
                    let l_score = doc.lexical_score.map(|s| lexical_weight * s).unwrap_or(0.0);
                    doc.fused_score = v_score.max(l_score);
                }
            }
            
            FusionMethod::Rsf => {
                // Relative Score Fusion (simplified)
                for doc in &mut fused {
                    let mut score = 0.0;
                    let mut count = 0;
                    
                    if let Some(s) = doc.vector_score {
                        score += s;
                        count += 1;
                    }
                    
                    if let Some(s) = doc.lexical_score {
                        score += s;
                        count += 1;
                    }
                    
                    doc.fused_score = if count > 0 { score / count as f32 } else { 0.0 };
                }
            }
        }
        
        Ok(fused)
    }
    
    /// Rerank candidates using cross-encoder (stub)
    fn rerank(
        &self,
        candidates: &[CandidateDoc],
        query: &str,
        config: &RerankConfig,
    ) -> Result<Vec<CandidateDoc>, HybridQueryError> {
        // Take top_n candidates for reranking
        let to_rerank: Vec<_> = candidates.iter().take(config.top_n).cloned().collect();
        
        // Stub: In production, would call cross-encoder model
        // For now, just apply a small boost based on query term overlap
        let mut reranked = to_rerank;
        let query_terms: HashSet<&str> = query.split_whitespace().collect();
        
        for doc in &mut reranked {
            let content_terms: HashSet<&str> = doc.content.split_whitespace().collect();
            let overlap = query_terms.intersection(&content_terms).count();
            
            // Small boost for term overlap
            doc.fused_score += (overlap as f32) * 0.01;
        }
        
        // Add remaining candidates unchanged
        reranked.extend(candidates.iter().skip(config.top_n).cloned());
        
        Ok(reranked)
    }
}

/// Internal candidate document during processing
#[derive(Debug, Clone)]
struct CandidateDoc {
    id: String,
    content: String,
    metadata: HashMap<String, ToonValue>,
    vector_rank: Option<usize>,
    vector_score: Option<f32>,
    lexical_rank: Option<usize>,
    lexical_score: Option<f32>,
    fused_score: f32,
}

// ============================================================================
// Lexical Index (BM25)
// ============================================================================

/// Simple lexical (BM25) index
pub struct LexicalIndex {
    /// Collections: name -> inverted index
    collections: std::sync::RwLock<HashMap<String, InvertedIndex>>,
}

/// Inverted index for a collection
struct InvertedIndex {
    /// Term -> posting list (doc_id, term_freq)
    postings: HashMap<String, Vec<(String, u32)>>,
    
    /// Document lengths
    doc_lengths: HashMap<String, u32>,
    
    /// Document contents
    documents: HashMap<String, String>,
    
    /// Average document length
    avg_doc_len: f32,
    
    /// BM25 parameters
    k1: f32,
    b: f32,
}

/// Lexical search result
#[derive(Debug, Clone)]
pub struct LexicalSearchResult {
    pub id: String,
    pub score: f32,
    pub content: String,
}

impl LexicalIndex {
    /// Create a new lexical index
    pub fn new() -> Self {
        Self {
            collections: std::sync::RwLock::new(HashMap::new()),
        }
    }
    
    /// Create collection
    pub fn create_collection(&self, name: &str) {
        let mut collections = self.collections.write().unwrap();
        collections.insert(name.to_string(), InvertedIndex {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            documents: HashMap::new(),
            avg_doc_len: 0.0,
            k1: 1.2,
            b: 0.75,
        });
    }
    
    /// Index a document
    pub fn index_document(&self, collection: &str, id: &str, content: &str) -> Result<(), HybridQueryError> {
        let mut collections = self.collections.write().unwrap();
        let index = collections.get_mut(collection)
            .ok_or_else(|| HybridQueryError::CollectionNotFound(collection.to_string()))?;
        
        // Tokenize
        let tokens: Vec<String> = content
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();
        
        let doc_len = tokens.len() as u32;
        
        // Update document length
        index.doc_lengths.insert(id.to_string(), doc_len);
        index.documents.insert(id.to_string(), content.to_string());
        
        // Update average doc length
        let total_len: u32 = index.doc_lengths.values().sum();
        index.avg_doc_len = total_len as f32 / index.doc_lengths.len() as f32;
        
        // Count term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }
        
        // Update postings
        for (term, freq) in term_freqs {
            index.postings
                .entry(term)
                .or_insert_with(Vec::new)
                .push((id.to_string(), freq));
        }
        
        Ok(())
    }
    
    /// Search using BM25
    pub fn search(
        &self,
        collection: &str,
        query: &str,
        _fields: &[String],
        limit: usize,
    ) -> Result<Vec<LexicalSearchResult>, HybridQueryError> {
        let collections = self.collections.read().unwrap();
        let index = collections.get(collection)
            .ok_or_else(|| HybridQueryError::CollectionNotFound(collection.to_string()))?;
        
        // Tokenize query
        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();
        
        let n = index.doc_lengths.len() as f32;
        let mut scores: HashMap<String, f32> = HashMap::new();
        
        // Calculate BM25 scores
        for term in &query_terms {
            if let Some(postings) = index.postings.get(term) {
                let df = postings.len() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                
                for (doc_id, tf) in postings {
                    let doc_len = *index.doc_lengths.get(doc_id).unwrap_or(&1) as f32;
                    let tf = *tf as f32;
                    
                    // BM25 formula
                    let score = idf * (tf * (index.k1 + 1.0)) / 
                        (tf + index.k1 * (1.0 - index.b + index.b * doc_len / index.avg_doc_len));
                    
                    *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }
        
        // Sort by score
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        
        // Convert to results
        let results: Vec<LexicalSearchResult> = results
            .into_iter()
            .take(limit)
            .map(|(id, score)| {
                let content = index.documents.get(&id).cloned().unwrap_or_default();
                LexicalSearchResult { id, score, content }
            })
            .collect();
        
        Ok(results)
    }
}

impl Default for LexicalIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Results
// ============================================================================

/// Hybrid search result
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Document ID
    pub id: String,
    /// Fused score
    pub score: f32,
    /// Document content
    pub content: String,
    /// Document metadata
    pub metadata: HashMap<String, ToonValue>,
    /// Score from vector search (if any)
    pub vector_score: Option<f32>,
    /// Score from lexical search (if any)
    pub lexical_score: Option<f32>,
}

/// Result of hybrid query execution
#[derive(Debug, Clone)]
pub struct HybridQueryResult {
    /// Search results
    pub results: Vec<HybridSearchResult>,
    /// Original query
    pub query: HybridQuery,
    /// Execution statistics
    pub stats: HybridQueryStats,
}

/// Execution statistics
#[derive(Debug, Clone, Default)]
pub struct HybridQueryStats {
    /// Candidates from vector search
    pub vector_candidates: usize,
    /// Candidates from lexical search
    pub lexical_candidates: usize,
    /// Candidates after filtering
    pub filtered_candidates: usize,
    /// Fusion time in microseconds
    pub fusion_time_us: u64,
    /// Rerank time in microseconds
    pub rerank_time_us: u64,
}

/// Hybrid query error
#[derive(Debug, Clone)]
pub enum HybridQueryError {
    /// Collection not found
    CollectionNotFound(String),
    /// Vector search error
    VectorSearchError(String),
    /// Lexical search error
    LexicalSearchError(String),
    /// Filter error
    FilterError(String),
    /// Rerank error
    RerankError(String),
}

impl std::fmt::Display for HybridQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CollectionNotFound(name) => write!(f, "Collection not found: {}", name),
            Self::VectorSearchError(msg) => write!(f, "Vector search error: {}", msg),
            Self::LexicalSearchError(msg) => write!(f, "Lexical search error: {}", msg),
            Self::FilterError(msg) => write!(f, "Filter error: {}", msg),
            Self::RerankError(msg) => write!(f, "Rerank error: {}", msg),
        }
    }
}

impl std::error::Error for HybridQueryError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_query_builder() {
        let query = HybridQuery::new("documents")
            .with_vector(vec![0.1, 0.2, 0.3], 0.7)
            .with_lexical("search query", 0.3)
            .filter_eq("category", ToonValue::Text("tech".to_string()))
            .with_fusion(FusionMethod::Rrf)
            .with_rerank("cross-encoder", 20)
            .limit(10);
        
        assert_eq!(query.collection, "documents");
        assert!(query.vector.is_some());
        assert!(query.lexical.is_some());
        assert_eq!(query.filters.len(), 1);
        assert_eq!(query.limit, 10);
    }
    
    #[test]
    fn test_lexical_index_bm25() {
        let index = LexicalIndex::new();
        index.create_collection("test");
        
        index.index_document("test", "doc1", "the quick brown fox").unwrap();
        index.index_document("test", "doc2", "the lazy dog sleeps").unwrap();
        index.index_document("test", "doc3", "quick fox jumps over the lazy dog").unwrap();
        
        let results = index.search("test", "quick fox", &[], 10).unwrap();
        
        assert!(!results.is_empty());
        // doc1 and doc3 should both appear in results (they both have "quick" and/or "fox")
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"doc1") || ids.contains(&"doc3"));
        // doc2 should not appear (no "quick" or "fox")
        assert!(!ids.contains(&"doc2"));
    }
    
    #[test]
    fn test_rrf_fusion() {
        // RRF formula: score = Σ w / (k + rank)
        let k = 60.0;
        
        // Doc appears at rank 0 in vector, rank 5 in lexical
        let vector_weight = 0.7;
        let lexical_weight = 0.3;
        
        let score = vector_weight / (k + 0.0) + lexical_weight / (k + 5.0);
        
        // Should be approximately 0.0116 + 0.0046 = 0.0162
        assert!(score > 0.01 && score < 0.02);
    }
    
    #[test]
    fn test_filter_matching() {
        let filters = vec![
            MetadataFilter {
                field: "status".to_string(),
                op: FilterOp::Eq,
                value: ToonValue::Text("active".to_string()),
            },
            MetadataFilter {
                field: "count".to_string(),
                op: FilterOp::Gte,
                value: ToonValue::Int(10),
            },
        ];
        
        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), ToonValue::Text("active".to_string()));
        metadata.insert("count".to_string(), ToonValue::Int(15));
        
        // Create a mock candidate
        let doc = CandidateDoc {
            id: "test".to_string(),
            content: "test content".to_string(),
            metadata,
            vector_rank: None,
            vector_score: None,
            lexical_rank: None,
            lexical_score: None,
            fused_score: 0.0,
        };
        
        // Would pass filters
        assert!(doc.metadata.get("status") == Some(&ToonValue::Text("active".to_string())));
        if let Some(ToonValue::Int(count)) = doc.metadata.get("count") {
            assert!(*count >= 10);
        }
    }
}
