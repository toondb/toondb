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

//! FastEmbed ONNX Integration
//!
//! This module provides local embedding using the `fastembed` crate with
//! ONNX runtime for offline inference.
//!
//! Supports models like:
//! - all-MiniLM-L6-v2 (384 dimensions, fast)
//! - BAAI/bge-small-en-v1.5 (384 dimensions)
//! - nomic-embed-text-v1 (768 dimensions, long context)

use crate::embedding::config::LocalModelConfig;
use crate::embedding::normalize::normalize_l2;
use crate::embedding::provider::{EmbeddingError, EmbeddingProvider};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;

/// FastEmbed-based local embedding provider
///
/// Uses ONNX runtime for efficient local inference without API calls.
pub struct FastEmbedProvider {
    /// The underlying fastembed model
    model: Arc<TextEmbedding>,
    /// Model configuration
    config: LocalModelConfig,
    /// Output dimension
    dimension: usize,
}

impl FastEmbedProvider {
    /// Create a new FastEmbed provider from configuration
    pub fn new(config: &LocalModelConfig) -> Result<Self, EmbeddingError> {
        // Map model name to fastembed enum
        let embedding_model = Self::resolve_model(&config.model)?;
        let dimension = Self::get_dimension(&embedding_model);

        // Set up init options
        let mut options = InitOptions::new(embedding_model);
        options = options.with_show_download_progress(config.auto_download);

        if let Some(ref cache_dir) = config.cache_dir {
            options = options.with_cache_dir(std::path::PathBuf::from(cache_dir));
        }

        // Create the model (this may download on first use)
        let model = TextEmbedding::try_new(options).map_err(|e| {
            EmbeddingError::ModelLoadFailed(format!("Failed to load fastembed model: {}", e))
        })?;

        Ok(Self {
            model: Arc::new(model),
            config: config.clone(),
            dimension,
        })
    }

    /// Resolve model name to fastembed enum
    fn resolve_model(name: &str) -> Result<EmbeddingModel, EmbeddingError> {
        match name.to_lowercase().as_str() {
            "all-minilm-l6-v2" | "minilm" | "minilm-l6" => Ok(EmbeddingModel::AllMiniLML6V2),
            "bge-small-en" | "bge-small" => Ok(EmbeddingModel::BGESmallENV15),
            "bge-base-en" | "bge-base" => Ok(EmbeddingModel::BGEBaseENV15),
            "bge-large-en" | "bge-large" => Ok(EmbeddingModel::BGELargeENV15),
            "nomic-embed-text" | "nomic" => Ok(EmbeddingModel::NomicEmbedTextV1),
            "gte-small" => Ok(EmbeddingModel::GTESmall),
            "gte-base" => Ok(EmbeddingModel::GTEBase),
            "gte-large" => Ok(EmbeddingModel::GTELarge),
            "multilingual-e5-small" | "e5-small" => Ok(EmbeddingModel::MultilingualE5Small),
            "multilingual-e5-base" | "e5-base" => Ok(EmbeddingModel::MultilingualE5Base),
            "multilingual-e5-large" | "e5-large" => Ok(EmbeddingModel::MultilingualE5Large),
            "paraphrase-minilm-l12" => Ok(EmbeddingModel::ParaphraseMLMiniLML12V2),
            "paraphrase-mpnet" => Ok(EmbeddingModel::ParaphraseMLMpnetBaseV2),
            _ => Err(EmbeddingError::ConfigError(format!(
                "Unknown fastembed model: {}. Supported: all-minilm-l6-v2, bge-small-en, bge-base-en, \
                 bge-large-en, nomic-embed-text, gte-small, gte-base, gte-large, \
                 multilingual-e5-small, multilingual-e5-base, multilingual-e5-large, \
                 paraphrase-minilm-l12, paraphrase-mpnet",
                name
            ))),
        }
    }

    /// Get output dimension for a model
    fn get_dimension(model: &EmbeddingModel) -> usize {
        match model {
            EmbeddingModel::AllMiniLML6V2 => 384,
            EmbeddingModel::BGESmallENV15 => 384,
            EmbeddingModel::BGEBaseENV15 => 768,
            EmbeddingModel::BGELargeENV15 => 1024,
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::GTESmall => 384,
            EmbeddingModel::GTEBase => 768,
            EmbeddingModel::GTELarge => 1024,
            EmbeddingModel::MultilingualE5Small => 384,
            EmbeddingModel::MultilingualE5Base => 768,
            EmbeddingModel::MultilingualE5Large => 1024,
            EmbeddingModel::ParaphraseMLMiniLML12V2 => 384,
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,
            _ => 384, // Default fallback
        }
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.config.model
    }
}

impl EmbeddingProvider for FastEmbedProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        let texts = vec![text];
        let embeddings = self
            .model
            .embed(texts, None)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("Embedding failed: {}", e)))?;

        if embeddings.is_empty() {
            return Err(EmbeddingError::InferenceFailed(
                "Empty embedding result".to_string(),
            ));
        }

        let mut vector = embeddings.into_iter().next().unwrap();
        normalize_l2(&mut vector);

        Ok(vector)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check for empty texts
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(EmbeddingError::InvalidInput(format!(
                    "Empty text at index {}",
                    i
                )));
            }
        }

        // Convert to owned strings for fastembed
        let owned_texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let text_refs: Vec<&str> = owned_texts.iter().map(|s| s.as_str()).collect();

        // Process in batches
        let batch_size = self.config.batch_size;
        let mut all_vectors = Vec::with_capacity(texts.len());

        for chunk in text_refs.chunks(batch_size) {
            let embeddings = self.model.embed(chunk.to_vec(), None).map_err(|e| {
                EmbeddingError::InferenceFailed(format!("Batch embedding failed: {}", e))
            })?;

            for mut vector in embeddings {
                normalize_l2(&mut vector);
                all_vectors.push(vector);
            }
        }

        Ok(all_vectors)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_tokens(&self) -> usize {
        // Most fastembed models support 512 tokens
        512
    }

    fn provider_id(&self) -> &str {
        "fastembed"
    }

    fn is_offline(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model() {
        assert!(FastEmbedProvider::resolve_model("all-minilm-l6-v2").is_ok());
        assert!(FastEmbedProvider::resolve_model("minilm").is_ok());
        assert!(FastEmbedProvider::resolve_model("bge-small-en").is_ok());
        assert!(FastEmbedProvider::resolve_model("nomic-embed-text").is_ok());
        assert!(FastEmbedProvider::resolve_model("unknown-model").is_err());
    }

    #[test]
    fn test_get_dimension() {
        assert_eq!(
            FastEmbedProvider::get_dimension(&EmbeddingModel::AllMiniLML6V2),
            384
        );
        assert_eq!(
            FastEmbedProvider::get_dimension(&EmbeddingModel::BGEBaseENV15),
            768
        );
        assert_eq!(
            FastEmbedProvider::get_dimension(&EmbeddingModel::BGELargeENV15),
            1024
        );
    }

    // Integration test - requires model download
    #[test]
    #[ignore = "requires model download (~90MB)"]
    fn test_fastembed_embedding() {
        let config = LocalModelConfig {
            model: "all-minilm-l6-v2".to_string(),
            auto_download: true,
            ..Default::default()
        };

        let provider = FastEmbedProvider::new(&config).unwrap();
        let vector = provider.embed("Hello, world!").unwrap();

        assert_eq!(vector.len(), 384);

        // Check normalization
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires model download (~90MB)"]
    fn test_fastembed_batch() {
        let config = LocalModelConfig::default();
        let provider = FastEmbedProvider::new(&config).unwrap();

        let texts = vec!["first text", "second text", "third text"];
        let vectors = provider.embed_batch(&texts).unwrap();

        assert_eq!(vectors.len(), 3);
        for vec in &vectors {
            assert_eq!(vec.len(), 384);

            // Check normalization
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    #[ignore = "requires model download (~90MB)"]
    fn test_fastembed_similarity() {
        let config = LocalModelConfig::default();
        let provider = FastEmbedProvider::new(&config).unwrap();

        let vec1 = provider.embed("The cat sat on the mat").unwrap();
        let vec2 = provider.embed("A cat is sitting on the mat").unwrap();
        let vec3 = provider.embed("Quantum mechanics theory").unwrap();

        // Compute cosine similarity (vectors are already normalized)
        let sim_12: f32 = vec1.iter().zip(&vec2).map(|(a, b)| a * b).sum();
        let sim_13: f32 = vec1.iter().zip(&vec3).map(|(a, b)| a * b).sum();

        // Similar sentences should have higher similarity
        assert!(
            sim_12 > sim_13,
            "Similar texts should have higher similarity: {} vs {}",
            sim_12,
            sim_13
        );
    }
}
