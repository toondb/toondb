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

//! Automatic Embedding Generation (Task 2)
//!
//! This module provides colocated embedding resolution for text-to-vector conversion.
//! It enables first-class text search by automatically generating embeddings.
//!
//! ## Design
//!
//! ```text
//! search_text(collection, text, k)
//!     │
//!     ▼
//! ┌─────────────────┐
//! │ EmbeddingProvider │
//! │  ├─ LRU Cache    │
//! │  └─ ONNX Runtime │
//! └─────────────────┘
//!     │
//!     ▼
//! search_by_embedding(collection, embedding, k)
//! ```
//!
//! ## Providers
//!
//! - `LocalProvider`: Uses FastEmbed/ONNX for offline embedding
//! - `CachedProvider`: LRU cache wrapper for any provider
//! - `MockProvider`: For testing
//!
//! ## Complexity
//!
//! - Embedding generation: O(n) where n = text length (transformer inference)
//! - Cache lookup: O(1) expected (hash-based LRU)
//! - Batch embedding: O(k) compute with ~O(1) ONNX session overhead

use std::sync::Arc;
use moka::sync::Cache;

// ============================================================================
// Embedding Provider Trait
// ============================================================================

/// Error type for embedding operations
#[derive(Debug, Clone)]
pub enum EmbeddingError {
    /// Model not loaded or unavailable
    ModelNotAvailable(String),
    /// Text too long for model
    TextTooLong { max_length: usize, actual: usize },
    /// Dimension mismatch
    DimensionMismatch { expected: usize, actual: usize },
    /// Provider error
    ProviderError(String),
    /// Cache error
    CacheError(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotAvailable(model) => write!(f, "Embedding model not available: {}", model),
            Self::TextTooLong { max_length, actual } => {
                write!(f, "Text too long: {} > {} max", actual, max_length)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::ProviderError(msg) => write!(f, "Provider error: {}", msg),
            Self::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Result type for embedding operations
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Embedding provider trait
pub trait EmbeddingProvider: Send + Sync {
    /// Get the model name
    fn model_name(&self) -> &str;
    
    /// Get the embedding dimension
    fn dimension(&self) -> usize;
    
    /// Maximum text length (in characters or tokens)
    fn max_length(&self) -> usize;
    
    /// Generate embedding for a single text
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>>;
    
    /// Generate embeddings for multiple texts (batch)
    fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        // Default implementation: sequential embedding
        texts.iter().map(|t| self.embed(t)).collect()
    }
    
    /// Normalize an embedding vector (L2 normalization)
    fn normalize(&self, embedding: &mut [f32]) {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }
}

// ============================================================================
// Embedding Configuration
// ============================================================================

/// Configuration for embedding providers
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Model identifier (e.g., "all-MiniLM-L6-v2")
    pub model: String,
    
    /// Model path (for local ONNX models)
    pub model_path: Option<String>,
    
    /// Embedding dimension
    pub dimension: usize,
    
    /// Maximum text length
    pub max_length: usize,
    
    /// Whether to normalize embeddings
    pub normalize: bool,
    
    /// Batch size for embedding generation
    pub batch_size: usize,
    
    /// Cache size (number of embeddings to cache)
    pub cache_size: usize,
    
    /// Cache TTL in seconds (0 = no expiry)
    pub cache_ttl_secs: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "all-MiniLM-L6-v2".to_string(),
            model_path: None,
            dimension: 384, // MiniLM dimension
            max_length: 512,
            normalize: true,
            batch_size: 32,
            cache_size: 10_000,
            cache_ttl_secs: 3600, // 1 hour
        }
    }
}

impl EmbeddingConfig {
    /// Create config for sentence-transformers models
    pub fn sentence_transformer(model: &str) -> Self {
        let dimension = match model {
            "all-MiniLM-L6-v2" => 384,
            "all-MiniLM-L12-v2" => 384,
            "all-mpnet-base-v2" => 768,
            "paraphrase-MiniLM-L6-v2" => 384,
            "multi-qa-MiniLM-L6-cos-v1" => 384,
            _ => 384, // Default
        };
        
        Self {
            model: model.to_string(),
            dimension,
            ..Default::default()
        }
    }
    
    /// Create config for OpenAI-compatible models
    pub fn openai(model: &str) -> Self {
        let dimension = match model {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => 1536,
        };
        
        Self {
            model: model.to_string(),
            dimension,
            max_length: 8192,
            ..Default::default()
        }
    }
}

// ============================================================================
// Mock Embedding Provider (for testing)
// ============================================================================

/// Mock embedding provider for testing
pub struct MockEmbeddingProvider {
    config: EmbeddingConfig,
    /// Deterministic embeddings based on text hash
    use_hash: bool,
}

impl MockEmbeddingProvider {
    /// Create a new mock provider
    pub fn new(dimension: usize) -> Self {
        Self {
            config: EmbeddingConfig {
                model: "mock".to_string(),
                dimension,
                ..Default::default()
            },
            use_hash: true,
        }
    }
    
    /// Create with custom config
    pub fn with_config(config: EmbeddingConfig) -> Self {
        Self {
            config,
            use_hash: true,
        }
    }
    
    /// Generate a deterministic embedding from text
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut embedding = Vec::with_capacity(self.config.dimension);
        
        // Generate pseudo-random values based on text hash
        for i in 0..self.config.dimension {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Convert to f32 in range [-1, 1]
            let value = ((hash as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32;
            embedding.push(value);
        }
        
        embedding
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.config.model
    }
    
    fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    fn max_length(&self) -> usize {
        self.config.max_length
    }
    
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        if text.len() > self.config.max_length {
            return Err(EmbeddingError::TextTooLong {
                max_length: self.config.max_length,
                actual: text.len(),
            });
        }
        
        let mut embedding = if self.use_hash {
            self.hash_embed(text)
        } else {
            vec![0.0; self.config.dimension]
        };
        
        if self.config.normalize {
            self.normalize(&mut embedding);
        }
        
        Ok(embedding)
    }
}

// ============================================================================
// Cached Embedding Provider
// ============================================================================

/// LRU-cached embedding provider wrapper
pub struct CachedEmbeddingProvider<P: EmbeddingProvider> {
    /// Inner provider
    inner: P,
    
    /// LRU cache: text hash -> embedding
    cache: Cache<u64, Vec<f32>>,
    
    /// Cache statistics
    stats: Arc<CacheStats>,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: std::sync::atomic::AtomicUsize,
    /// Number of cache misses
    pub misses: std::sync::atomic::AtomicUsize,
    /// Number of embeddings cached
    pub size: std::sync::atomic::AtomicUsize,
}

impl CacheStats {
    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

impl<P: EmbeddingProvider> CachedEmbeddingProvider<P> {
    /// Create a new cached provider
    pub fn new(inner: P, cache_size: usize) -> Self {
        Self {
            inner,
            cache: Cache::new(cache_size as u64),
            stats: Arc::new(CacheStats::default()),
        }
    }
    
    /// Create with TTL
    pub fn with_ttl(inner: P, cache_size: usize, ttl_secs: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(cache_size as u64)
            .time_to_live(std::time::Duration::from_secs(ttl_secs))
            .build();
        
        Self {
            inner,
            cache,
            stats: Arc::new(CacheStats::default()),
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> &Arc<CacheStats> {
        &self.stats
    }
    
    /// Compute hash for cache key
    fn text_hash(text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
}

impl<P: EmbeddingProvider> EmbeddingProvider for CachedEmbeddingProvider<P> {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
    
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    
    fn max_length(&self) -> usize {
        self.inner.max_length()
    }
    
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        let hash = Self::text_hash(text);
        
        // Check cache
        if let Some(cached) = self.cache.get(&hash) {
            self.stats.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(cached);
        }
        
        self.stats.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Generate embedding
        let embedding = self.inner.embed(text)?;
        
        // Cache result
        self.cache.insert(hash, embedding.clone());
        self.stats.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(embedding)
    }
    
    fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached: Vec<(usize, &str)> = Vec::new();
        
        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            let hash = Self::text_hash(text);
            if let Some(cached) = self.cache.get(&hash) {
                self.stats.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                results.push((i, cached));
            } else {
                self.stats.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                uncached.push((i, *text));
            }
        }
        
        // Generate embeddings for uncached texts
        if !uncached.is_empty() {
            let uncached_texts: Vec<&str> = uncached.iter().map(|(_, t)| *t).collect();
            let embeddings = self.inner.embed_batch(&uncached_texts)?;
            
            for ((i, text), embedding) in uncached.iter().zip(embeddings.into_iter()) {
                let hash = Self::text_hash(text);
                self.cache.insert(hash, embedding.clone());
                self.stats.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                results.push((*i, embedding));
            }
        }
        
        // Sort by original index
        results.sort_by_key(|(i, _)| *i);
        Ok(results.into_iter().map(|(_, e)| e).collect())
    }
}

// ============================================================================
// Local ONNX Provider (Stub)
// ============================================================================

/// Local ONNX-based embedding provider
/// 
/// This is a stub implementation. In production, this would use:
/// - ort (ONNX Runtime) for model inference
/// - fastembed-rs for pre-packaged models
/// - tokenizers for text preprocessing
#[derive(Debug)]
pub struct LocalOnnxProvider {
    config: EmbeddingConfig,
    /// Model weights (placeholder)
    #[allow(dead_code)]
    model_loaded: bool,
}

impl LocalOnnxProvider {
    /// Create a new local ONNX provider
    pub fn new(config: EmbeddingConfig) -> EmbeddingResult<Self> {
        // In production: Load ONNX model from path
        Ok(Self {
            config,
            model_loaded: false,
        })
    }
    
    /// Load a pre-trained model by name
    pub fn load_pretrained(model_name: &str) -> EmbeddingResult<Self> {
        let config = EmbeddingConfig::sentence_transformer(model_name);
        Self::new(config)
    }
}

impl EmbeddingProvider for LocalOnnxProvider {
    fn model_name(&self) -> &str {
        &self.config.model
    }
    
    fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    fn max_length(&self) -> usize {
        self.config.max_length
    }
    
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        // Stub: Return mock embedding
        // In production: Run ONNX inference
        let mock = MockEmbeddingProvider::with_config(self.config.clone());
        mock.embed(text)
    }
}

// ============================================================================
// Embedding-Enabled Vector Index
// ============================================================================

/// Vector index with automatic text embedding
pub struct EmbeddingVectorIndex<V, P> 
where
    V: crate::context_query::VectorIndex,
    P: EmbeddingProvider,
{
    /// Underlying vector index
    index: Arc<V>,
    
    /// Embedding provider
    provider: Arc<P>,
}

impl<V, P> EmbeddingVectorIndex<V, P>
where
    V: crate::context_query::VectorIndex,
    P: EmbeddingProvider,
{
    /// Create a new embedding-enabled vector index
    pub fn new(index: Arc<V>, provider: Arc<P>) -> Self {
        Self { index, provider }
    }
    
    /// Search by text (automatically generates embedding)
    pub fn search_text(
        &self,
        collection: &str,
        text: &str,
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<crate::context_query::VectorSearchResult>, String> {
        // Generate embedding
        let embedding = self.provider.embed(text)
            .map_err(|e| e.to_string())?;
        
        // Search by embedding
        self.index.search_by_embedding(collection, &embedding, k, min_score)
    }
    
    /// Search by embedding (pass-through)
    pub fn search_embedding(
        &self,
        collection: &str,
        embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<crate::context_query::VectorSearchResult>, String> {
        // Validate dimension
        if embedding.len() != self.provider.dimension() {
            return Err(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.provider.dimension(),
                embedding.len()
            ));
        }
        
        self.index.search_by_embedding(collection, embedding, k, min_score)
    }
    
    /// Get the embedding provider
    pub fn provider(&self) -> &Arc<P> {
        &self.provider
    }
    
    /// Get the underlying index
    pub fn index(&self) -> &Arc<V> {
        &self.index
    }
}

impl<V, P> crate::context_query::VectorIndex for EmbeddingVectorIndex<V, P>
where
    V: crate::context_query::VectorIndex,
    P: EmbeddingProvider,
{
    fn search_by_embedding(
        &self,
        collection: &str,
        embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<crate::context_query::VectorSearchResult>, String> {
        self.search_embedding(collection, embedding, k, min_score)
    }
    
    fn search_by_text(
        &self,
        collection: &str,
        text: &str,
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<crate::context_query::VectorSearchResult>, String> {
        self.search_text(collection, text, k, min_score)
    }
    
    fn stats(&self, collection: &str) -> Option<crate::context_query::VectorIndexStats> {
        self.index.stats(collection)
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a cached mock embedding provider for testing
pub fn create_mock_provider(dimension: usize, cache_size: usize) -> CachedEmbeddingProvider<MockEmbeddingProvider> {
    let mock = MockEmbeddingProvider::new(dimension);
    CachedEmbeddingProvider::new(mock, cache_size)
}

/// Create an embedding-enabled vector index with mock provider
pub fn create_embedding_index<V: crate::context_query::VectorIndex>(
    index: Arc<V>,
    dimension: usize,
) -> EmbeddingVectorIndex<V, CachedEmbeddingProvider<MockEmbeddingProvider>> {
    let provider = Arc::new(create_mock_provider(dimension, 10_000));
    EmbeddingVectorIndex::new(index, provider)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_embedding_deterministic() {
        let provider = MockEmbeddingProvider::new(384);
        
        let emb1 = provider.embed("hello world").unwrap();
        let emb2 = provider.embed("hello world").unwrap();
        
        assert_eq!(emb1, emb2);
        assert_eq!(emb1.len(), 384);
    }
    
    #[test]
    fn test_mock_embedding_different_texts() {
        let provider = MockEmbeddingProvider::new(384);
        
        let emb1 = provider.embed("hello").unwrap();
        let emb2 = provider.embed("world").unwrap();
        
        assert_ne!(emb1, emb2);
    }
    
    #[test]
    fn test_cached_provider() {
        let mock = MockEmbeddingProvider::new(128);
        let cached = CachedEmbeddingProvider::new(mock, 100);
        
        // First call - miss
        let _ = cached.embed("test text").unwrap();
        assert_eq!(cached.stats().hits.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(cached.stats().misses.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        // Second call - hit
        let _ = cached.embed("test text").unwrap();
        assert_eq!(cached.stats().hits.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(cached.stats().misses.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        assert!(cached.stats().hit_rate() > 0.4);
    }
    
    #[test]
    fn test_batch_embedding() {
        let mock = MockEmbeddingProvider::new(128);
        let cached = CachedEmbeddingProvider::new(mock, 100);
        
        let texts = vec!["hello", "world", "test"];
        let embeddings = cached.embed_batch(&texts).unwrap();
        
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 128);
        }
    }
    
    #[test]
    fn test_normalization() {
        let provider = MockEmbeddingProvider::new(3);
        let emb = provider.embed("test").unwrap();
        
        // Check L2 norm is approximately 1
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_text_too_long() {
        let config = EmbeddingConfig {
            max_length: 10,
            ..Default::default()
        };
        let provider = MockEmbeddingProvider::with_config(config);
        
        let result = provider.embed("this is a very long text that exceeds the limit");
        assert!(matches!(result, Err(EmbeddingError::TextTooLong { .. })));
    }
}
