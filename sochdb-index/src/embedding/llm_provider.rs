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

//! LLM Crate Embedding Provider
//!
//! This module provides integration with the `llm` crate's EmbeddingProvider trait,
//! enabling use of various cloud and local LLM providers for embeddings.
//!
//! Supported backends via the llm crate:
//! - OpenAI (text-embedding-3-small, text-embedding-3-large)
//! - Azure OpenAI
//! - Ollama (local)
//! - Google AI (Gemini)
//! - Cohere
//! - Mistral
//! - HuggingFace
//! - XAI

use crate::embedding::config::{LlmBackend, LlmProviderConfig};
use crate::embedding::normalize::normalize_l2;
use crate::embedding::provider::{EmbeddingError, EmbeddingProvider};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Embedding provider using the llm crate
pub struct LlmEmbeddingProvider {
    /// The underlying llm provider
    inner: Arc<dyn llm::embedding::EmbeddingProvider + Send + Sync>,
    /// Configuration
    config: LlmProviderConfig,
    /// Tokio runtime for async operations
    runtime: Runtime,
    /// Cached dimension
    dimension: usize,
}

impl LlmEmbeddingProvider {
    /// Create a new LLM embedding provider from configuration
    pub fn new(config: LlmProviderConfig) -> Result<Self, EmbeddingError> {
        // Create tokio runtime for async operations
        let runtime = Runtime::new().map_err(|e| {
            EmbeddingError::ConfigError(format!("Failed to create tokio runtime: {}", e))
        })?;

        // Build the appropriate llm backend
        let inner: Arc<dyn llm::embedding::EmbeddingProvider + Send + Sync> = match config.backend {
            LlmBackend::OpenAI => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError(
                            "OpenAI API key required. Set OPENAI_API_KEY or provide api_key"
                                .to_string(),
                        )
                    })?;

                let mut builder = llm::builder().backend(llm::Backend::OpenAI);
                builder = builder.api_key(api_key);

                if let Some(ref base_url) = config.base_url {
                    builder = builder.base_url(base_url);
                }

                Arc::new(builder.build().map_err(|e| {
                    EmbeddingError::ConfigError(format!("Failed to build OpenAI client: {}", e))
                })?)
            }

            LlmBackend::AzureOpenAI => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("AZURE_OPENAI_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError("Azure OpenAI API key required".to_string())
                    })?;

                let base_url = config.base_url.clone().or_else(|| {
                        std::env::var("AZURE_OPENAI_ENDPOINT").ok()
                    }).ok_or_else(|| {
                        EmbeddingError::ConfigError(
                            "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT or provide base_url".to_string(),
                        )
                    })?;

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::AzureOpenAI)
                        .api_key(api_key)
                        .base_url(&base_url)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build Azure OpenAI client: {}",
                                e
                            ))
                        })?,
                )
            }

            LlmBackend::Ollama => {
                let base_url = config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434".to_string());

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::Ollama)
                        .base_url(&base_url)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build Ollama client: {}",
                                e
                            ))
                        })?,
                )
            }

            LlmBackend::Google => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError("Google API key required".to_string())
                    })?;

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::Google)
                        .api_key(api_key)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build Google client: {}",
                                e
                            ))
                        })?,
                )
            }

            LlmBackend::Cohere => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("COHERE_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError("Cohere API key required".to_string())
                    })?;

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::Cohere)
                        .api_key(api_key)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build Cohere client: {}",
                                e
                            ))
                        })?,
                )
            }

            LlmBackend::Mistral => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError("Mistral API key required".to_string())
                    })?;

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::Mistral)
                        .api_key(api_key)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build Mistral client: {}",
                                e
                            ))
                        })?,
                )
            }

            LlmBackend::HuggingFace => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("HUGGINGFACE_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError("HuggingFace API key required".to_string())
                    })?;

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::HuggingFace)
                        .api_key(api_key)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build HuggingFace client: {}",
                                e
                            ))
                        })?,
                )
            }

            LlmBackend::XAI => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("XAI_API_KEY").ok())
                    .ok_or_else(|| {
                        EmbeddingError::ConfigError("XAI API key required".to_string())
                    })?;

                Arc::new(
                    llm::builder()
                        .backend(llm::Backend::XAI)
                        .api_key(api_key)
                        .build()
                        .map_err(|e| {
                            EmbeddingError::ConfigError(format!(
                                "Failed to build XAI client: {}",
                                e
                            ))
                        })?,
                )
            }
        };

        // Get dimension from config or use backend default
        let dimension = config
            .dimension
            .unwrap_or_else(|| config.backend.default_dimension());

        Ok(Self {
            inner,
            config,
            runtime,
            dimension,
        })
    }

    /// Get the backend type
    pub fn backend(&self) -> LlmBackend {
        self.config.backend
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.config.model
    }
}

impl EmbeddingProvider for LlmEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        // Run async embedding in sync context
        let input = vec![text.to_string()];
        let inner = Arc::clone(&self.inner);

        let result = self
            .runtime
            .block_on(async move { inner.embed(input).await })
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        if result.is_empty() {
            return Err(EmbeddingError::InferenceFailed(
                "Empty embedding result".to_string(),
            ));
        }

        // Get first embedding and normalize
        let mut vector = result.into_iter().next().unwrap();
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

        let input: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let inner = Arc::clone(&self.inner);
        let batch_size = self.config.batch_size;

        // Process in batches to respect API limits
        let mut all_vectors = Vec::with_capacity(texts.len());

        for chunk in input.chunks(batch_size) {
            let chunk_vec: Vec<String> = chunk.to_vec();
            let inner_clone = Arc::clone(&inner);

            let batch_result = self
                .runtime
                .block_on(async move { inner_clone.embed(chunk_vec).await })
                .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

            // Normalize each vector
            for mut vector in batch_result {
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
        // Most embedding models support 8191 tokens
        8191
    }

    fn provider_id(&self) -> &str {
        match self.config.backend {
            LlmBackend::OpenAI => "openai",
            LlmBackend::AzureOpenAI => "azure-openai",
            LlmBackend::Ollama => "ollama",
            LlmBackend::Google => "google",
            LlmBackend::Cohere => "cohere",
            LlmBackend::Mistral => "mistral",
            LlmBackend::HuggingFace => "huggingface",
            LlmBackend::XAI => "xai",
        }
    }

    fn is_offline(&self) -> bool {
        // Only Ollama can work offline (when running locally)
        matches!(self.config.backend, LlmBackend::Ollama)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_properties() {
        assert_eq!(LlmBackend::OpenAI.default_dimension(), 1536);
        assert_eq!(LlmBackend::Ollama.default_dimension(), 768);
        assert!(LlmBackend::OpenAI.requires_api_key());
        assert!(!LlmBackend::Ollama.requires_api_key());
    }

    // Integration tests require API keys, so they're skipped in CI
    #[test]
    #[ignore = "requires OPENAI_API_KEY"]
    fn test_openai_embedding() {
        let config = LlmProviderConfig {
            backend: LlmBackend::OpenAI,
            model: "text-embedding-3-small".to_string(),
            ..Default::default()
        };

        let provider = LlmEmbeddingProvider::new(config).unwrap();
        let vector = provider.embed("Hello, world!").unwrap();

        assert_eq!(vector.len(), 1536);

        // Check normalization
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires Ollama running locally"]
    fn test_ollama_embedding() {
        let config = LlmProviderConfig {
            backend: LlmBackend::Ollama,
            model: "nomic-embed-text".to_string(),
            base_url: Some("http://localhost:11434".to_string()),
            ..Default::default()
        };

        let provider = LlmEmbeddingProvider::new(config).unwrap();
        let vector = provider.embed("Hello, world!").unwrap();

        assert!(!vector.is_empty());

        // Check normalization
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
