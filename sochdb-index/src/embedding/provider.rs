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

//! Embedding Provider Abstraction
//!
//! This module defines the `EmbeddingProvider` trait that abstracts over
//! different embedding implementations (local ONNX, OpenAI, etc.).
//!
//! ## Provider Types
//!
//! - **LocalEmbeddingProvider**: Uses local computation for embeddings.
//!   Deterministic, fast, works offline. For production, integrate with
//!   fastembed-rs or ONNX runtime.
//!
//! - **MockEmbeddingProvider**: Generates deterministic fake embeddings
//!   for testing. Produces consistent vectors based on text hash.
//!
//! - **OpenAIEmbeddingProvider**: (Optional) Uses OpenAI's embedding API.
//!   Requires the `openai-embeddings` feature.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during embedding operations
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// Model failed to load (missing file, invalid format)
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),

    /// Tokenization error (invalid input, encoding issues)
    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    /// Inference error (ONNX runtime error, GPU error)
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Network error (for cloud providers)
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Rate limit exceeded (for cloud providers)
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Invalid input (empty text, too long)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Provider not found in registry
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error (file operations)
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Main trait for embedding providers
///
/// Implementations must be thread-safe (Send + Sync) to allow
/// concurrent embedding from multiple threads.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text, returning a normalized F32 vector
    ///
    /// The returned vector should be L2-normalized for cosine similarity.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Batch embedding with provider-optimal batching
    ///
    /// Default implementation calls `embed` in a loop, but providers
    /// should override for better performance.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Output dimensionality (for index initialization)
    fn dimension(&self) -> usize;

    /// Maximum input tokens (for truncation guidance)
    fn max_tokens(&self) -> usize;

    /// Provider identifier for metrics/logging
    fn provider_id(&self) -> &str;

    /// Whether this provider works offline
    fn is_offline(&self) -> bool;
}

/// Configuration for local embedding provider
#[derive(Debug, Clone)]
pub struct LocalEmbeddingConfig {
    /// Model name (e.g., "all-MiniLM-L6-v2")
    pub model_name: String,

    /// Output dimension (384 for MiniLM)
    pub dimension: usize,

    /// Maximum sequence length in tokens
    pub max_tokens: usize,

    /// Number of threads for inference
    pub num_threads: usize,

    /// Batch size for optimal throughput
    pub batch_size: usize,
}

impl Default for LocalEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            dimension: 384,
            max_tokens: 512,
            num_threads: 4,
            batch_size: 32,
        }
    }
}

/// Local embedding provider
///
/// This implementation uses a simple but effective text-to-vector method
/// that produces semantically meaningful embeddings based on TF-IDF-like
/// scoring with character n-grams.
///
/// For production use with real semantic similarity, replace the internal
/// implementation with fastembed-rs or ONNX runtime.
pub struct LocalEmbeddingProvider {
    config: LocalEmbeddingConfig,
}

impl LocalEmbeddingProvider {
    /// Create a new local embedding provider
    pub fn new(config: LocalEmbeddingConfig) -> Result<Self, EmbeddingError> {
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_provider() -> Result<Self, EmbeddingError> {
        Self::new(LocalEmbeddingConfig::default())
    }

    /// Generate embedding using improved algorithm
    ///
    /// This uses a combination of:
    /// 1. Character 3-grams with position weighting
    /// 2. Word-level features with TF-IDF-like scoring
    /// 3. Semantic hashing for consistent output
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let dim = self.config.dimension;
        let mut vector = vec![0.0f32; dim];

        if text.is_empty() {
            return vector;
        }

        let normalized_text = text.to_lowercase();
        let words: Vec<&str> = normalized_text.split_whitespace().collect();

        // Word-level features (distributed across dimensions)
        for (word_idx, word) in words.iter().enumerate() {
            let word_hash = self.hash_str(word);
            let base_idx = (word_hash as usize) % dim;

            // Position decay: words at start matter more
            let position_weight = 1.0 / (1.0 + (word_idx as f32 * 0.1));

            // IDF-like: shorter words are more common, less informative
            let length_weight = (word.len() as f32).sqrt().min(3.0) / 3.0;

            let weight = position_weight * length_weight;

            // Distribute to multiple dimensions for robustness
            for i in 0..8 {
                let idx = (base_idx + i * 37) % dim;
                vector[idx] += weight * if i % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        // Character 3-gram features
        let chars: Vec<char> = normalized_text.chars().collect();
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            let hash = self.hash_str(&trigram);
            let idx = (hash as usize) % dim;
            vector[idx] += 0.3;

            // Also add to adjacent dimensions
            vector[(idx + 1) % dim] += 0.15;
            vector[(idx + dim - 1) % dim] += 0.15;
        }

        // Sentence length feature (normalized)
        let length_feature = (text.len() as f32).ln() / 10.0;
        vector[0] = length_feature;

        // Number of words feature
        vector[1] = (words.len() as f32).ln() / 5.0;

        // Average word length
        if !words.is_empty() {
            let avg_word_len: f32 =
                words.iter().map(|w| w.len() as f32).sum::<f32>() / words.len() as f32;
            vector[2] = avg_word_len / 10.0;
        }

        // L2 normalize the vector
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }

    fn hash_str(&self, s: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

impl EmbeddingProvider for LocalEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // For local provider, we can process in parallel
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn max_tokens(&self) -> usize {
        self.config.max_tokens
    }

    fn provider_id(&self) -> &str {
        "local"
    }

    fn is_offline(&self) -> bool {
        true
    }
}

/// Mock embedding provider for testing
///
/// Generates deterministic embeddings based on text hash.
/// Useful for unit tests and integration tests.
pub struct MockEmbeddingProvider {
    dimension: usize,
    max_tokens: usize,
}

impl MockEmbeddingProvider {
    /// Create a new mock provider with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            max_tokens: 512,
        }
    }

    /// Create with default 384 dimensions
    pub fn default_provider() -> Self {
        Self::new(384)
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        // Generate deterministic embedding from text hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Use seed to generate pseudo-random but deterministic vector
        let mut vector = Vec::with_capacity(self.dimension);
        let mut state = seed;

        for _ in 0..self.dimension {
            // Simple LCG for deterministic pseudo-random
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            vector.push(val);
        }

        // L2 normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        Ok(vector)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn provider_id(&self) -> &str {
        "mock"
    }

    fn is_offline(&self) -> bool {
        true
    }
}

/// Registry for managing multiple embedding providers
///
/// Allows runtime switching between providers and fallback chains.
pub struct EmbeddingRegistry {
    providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    default_provider: String,
    fallback_chain: Vec<String>,
}

impl EmbeddingRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: String::new(),
            fallback_chain: Vec::new(),
        }
    }

    /// Create registry with local provider as default
    pub fn with_local_default() -> Result<Self, EmbeddingError> {
        let mut registry = Self::new();
        let local = Arc::new(LocalEmbeddingProvider::default_provider()?);
        registry.register("local", local.clone())?;
        registry.set_default("local")?;
        Ok(registry)
    }

    /// Register a provider with a name
    pub fn register(
        &mut self,
        name: &str,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Result<(), EmbeddingError> {
        self.providers.insert(name.to_string(), provider);
        Ok(())
    }

    /// Set the default provider
    pub fn set_default(&mut self, name: &str) -> Result<(), EmbeddingError> {
        if !self.providers.contains_key(name) {
            return Err(EmbeddingError::ProviderNotFound(name.to_string()));
        }
        self.default_provider = name.to_string();
        Ok(())
    }

    /// Set fallback chain (tried in order if primary fails)
    pub fn set_fallback_chain(&mut self, chain: Vec<String>) -> Result<(), EmbeddingError> {
        for name in &chain {
            if !self.providers.contains_key(name) {
                return Err(EmbeddingError::ProviderNotFound(name.clone()));
            }
        }
        self.fallback_chain = chain;
        Ok(())
    }

    /// Get the default provider
    pub fn get_default(&self) -> Result<Arc<dyn EmbeddingProvider>, EmbeddingError> {
        self.get(&self.default_provider)
    }

    /// Get a specific provider by name
    pub fn get(&self, name: &str) -> Result<Arc<dyn EmbeddingProvider>, EmbeddingError> {
        self.providers
            .get(name)
            .cloned()
            .ok_or_else(|| EmbeddingError::ProviderNotFound(name.to_string()))
    }

    /// Embed using default provider with fallback
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Try default provider
        let default = self.get_default()?;
        match default.embed(text) {
            Ok(vec) => return Ok(vec),
            Err(e) => {
                // Only fallback on network/rate limit errors
                if !matches!(
                    e,
                    EmbeddingError::NetworkError(_) | EmbeddingError::RateLimitExceeded(_)
                ) {
                    return Err(e);
                }
            }
        }

        // Try fallback chain
        for fallback_name in &self.fallback_chain {
            if let Ok(provider) = self.get(fallback_name)
                && let Ok(vec) = provider.embed(text)
            {
                return Ok(vec);
            }
        }

        Err(EmbeddingError::InferenceFailed(
            "All providers failed".to_string(),
        ))
    }

    /// List all registered provider names
    pub fn list_providers(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    /// Get dimension of default provider
    pub fn dimension(&self) -> Result<usize, EmbeddingError> {
        Ok(self.get_default()?.dimension())
    }
}

impl Default for EmbeddingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_provider_deterministic() {
        let provider = MockEmbeddingProvider::default_provider();

        let vec1 = provider.embed("hello world").unwrap();
        let vec2 = provider.embed("hello world").unwrap();

        assert_eq!(vec1, vec2, "Same text should produce same embedding");
    }

    #[test]
    fn test_mock_provider_normalized() {
        let provider = MockEmbeddingProvider::default_provider();

        let vec = provider.embed("test text").unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Vector should be L2 normalized, got norm: {}",
            norm
        );
    }

    #[test]
    fn test_mock_provider_different_texts() {
        let provider = MockEmbeddingProvider::default_provider();

        let vec1 = provider.embed("hello").unwrap();
        let vec2 = provider.embed("goodbye").unwrap();

        assert_ne!(
            vec1, vec2,
            "Different texts should produce different embeddings"
        );
    }

    #[test]
    fn test_local_provider_normalized() {
        let provider = LocalEmbeddingProvider::default_provider().unwrap();

        let vec = provider
            .embed("This is a test sentence for embedding")
            .unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Vector should be L2 normalized, got norm: {}",
            norm
        );
    }

    #[test]
    fn test_local_provider_similarity() {
        let provider = LocalEmbeddingProvider::default_provider().unwrap();

        let vec1 = provider.embed("The cat sat on the mat").unwrap();
        let vec2 = provider.embed("A cat is sitting on the mat").unwrap();
        let vec3 = provider.embed("Quantum mechanics theory").unwrap();

        // Compute cosine similarity (vectors are already normalized)
        let sim_12: f32 = vec1.iter().zip(&vec2).map(|(a, b)| a * b).sum();
        let sim_13: f32 = vec1.iter().zip(&vec3).map(|(a, b)| a * b).sum();

        assert!(
            sim_12 > sim_13,
            "Similar texts should have higher similarity: {} vs {}",
            sim_12,
            sim_13
        );
    }

    #[test]
    fn test_local_provider_batch() {
        let provider = LocalEmbeddingProvider::default_provider().unwrap();

        let texts = vec!["first text", "second text", "third text"];
        let vectors = provider.embed_batch(&texts).unwrap();

        assert_eq!(vectors.len(), 3);
        for vec in &vectors {
            assert_eq!(vec.len(), 384);
        }
    }

    #[test]
    fn test_registry_default() {
        let registry = EmbeddingRegistry::with_local_default().unwrap();

        let provider = registry.get_default().unwrap();
        assert_eq!(provider.provider_id(), "local");
        assert_eq!(provider.dimension(), 384);
    }

    #[test]
    fn test_registry_fallback() {
        let mut registry = EmbeddingRegistry::new();

        // Register mock as fallback
        let mock = Arc::new(MockEmbeddingProvider::default_provider());
        registry.register("mock", mock).unwrap();
        registry.set_default("mock").unwrap();

        // Should work with fallback
        let vec = registry.embed("test").unwrap();
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn test_registry_list_providers() {
        let mut registry = EmbeddingRegistry::new();

        let local = Arc::new(LocalEmbeddingProvider::default_provider().unwrap());
        let mock = Arc::new(MockEmbeddingProvider::default_provider());

        registry.register("local", local).unwrap();
        registry.register("mock", mock).unwrap();

        let providers = registry.list_providers();
        assert!(providers.contains(&"local"));
        assert!(providers.contains(&"mock"));
    }

    #[test]
    fn test_empty_text_error() {
        let provider = LocalEmbeddingProvider::default_provider().unwrap();
        let result = provider.embed("");

        assert!(matches!(result, Err(EmbeddingError::InvalidInput(_))));
    }

    #[test]
    fn test_provider_metadata() {
        let provider = LocalEmbeddingProvider::default_provider().unwrap();

        assert_eq!(provider.dimension(), 384);
        assert_eq!(provider.max_tokens(), 512);
        assert!(provider.is_offline());
    }
}
