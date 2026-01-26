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

//! OpenAI Embedding Provider
//!
//! This module provides integration with OpenAI's embedding API,
//! supporting both OpenAI and Azure OpenAI endpoints.
//!
//! ## Features
//!
//! - Automatic batching (up to 2048 texts per request)
//! - Rate limiting with exponential backoff
//! - Cost tracking per session
//! - Connection pooling via reqwest
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_index::embedding::openai::{OpenAIEmbeddingProvider, OpenAIConfig};
//!
//! let config = OpenAIConfig {
//!     api_key: "sk-...".to_string(),
//!     model: OpenAIEmbeddingModel::TextEmbedding3Small,
//!     ..Default::default()
//! };
//!
//! let provider = OpenAIEmbeddingProvider::new(config)?;
//! let embeddings = provider.embed_batch(&["hello", "world"])?;
//! ```

use super::provider::{EmbeddingError, EmbeddingProvider};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// OpenAI embedding model options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIEmbeddingModel {
    /// text-embedding-3-small: 1536 dimensions, $0.02/1M tokens
    TextEmbedding3Small,
    /// text-embedding-3-large: 3072 dimensions, $0.13/1M tokens
    TextEmbedding3Large,
    /// text-embedding-ada-002: 1536 dimensions (legacy)
    TextEmbeddingAda002,
}

impl OpenAIEmbeddingModel {
    /// Get the API model name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
        }
    }

    /// Get output dimension
    pub fn dimension(&self) -> usize {
        match self {
            Self::TextEmbedding3Small => 1536,
            Self::TextEmbedding3Large => 3072,
            Self::TextEmbeddingAda002 => 1536,
        }
    }

    /// Get cost per token in dollars
    pub fn cost_per_token(&self) -> f64 {
        match self {
            Self::TextEmbedding3Small => 0.02 / 1_000_000.0,
            Self::TextEmbedding3Large => 0.13 / 1_000_000.0,
            Self::TextEmbeddingAda002 => 0.10 / 1_000_000.0,
        }
    }
}

/// Configuration for OpenAI embedding provider
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key (required)
    pub api_key: String,

    /// Model to use
    pub model: OpenAIEmbeddingModel,

    /// API endpoint (None for default OpenAI, Some for Azure)
    pub endpoint: Option<String>,

    /// Azure API version (required for Azure endpoints)
    pub azure_api_version: Option<String>,

    /// Maximum texts per API request (max 2048)
    pub batch_size: usize,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Request timeout in seconds
    pub timeout_secs: u64,

    /// Optional dimension reduction (only for text-embedding-3 models)
    pub reduce_dimensions: Option<usize>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: OpenAIEmbeddingModel::TextEmbedding3Small,
            endpoint: None,
            azure_api_version: None,
            batch_size: 100, // Conservative default
            max_retries: 3,
            timeout_secs: 30,
            reduce_dimensions: None,
        }
    }
}

impl OpenAIConfig {
    /// Create config for Azure OpenAI
    pub fn azure(endpoint: String, api_key: String, deployment_name: &str) -> Self {
        Self {
            api_key,
            endpoint: Some(format!(
                "{}/openai/deployments/{}/embeddings",
                endpoint.trim_end_matches('/'),
                deployment_name
            )),
            azure_api_version: Some("2024-02-01".to_string()),
            ..Default::default()
        }
    }

    /// Create config for standard OpenAI
    pub fn openai(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }
}

/// Request body for OpenAI embedding API
#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    encoding_format: &'static str,
}

/// Response from OpenAI embedding API
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// OpenAI API error response
#[derive(Debug, Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

/// Simple rate limiter with sliding window
struct RateLimiter {
    requests_per_minute: u32,
    tokens_per_minute: u32,
    request_times: parking_lot::Mutex<std::collections::VecDeque<Instant>>,
    token_count: AtomicU64,
    last_reset: parking_lot::Mutex<Instant>,
}

impl RateLimiter {
    fn new(requests_per_minute: u32, tokens_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            tokens_per_minute,
            request_times: parking_lot::Mutex::new(std::collections::VecDeque::new()),
            token_count: AtomicU64::new(0),
            last_reset: parking_lot::Mutex::new(Instant::now()),
        }
    }

    fn check_and_record(&self, estimated_tokens: u32) -> Result<(), EmbeddingError> {
        let now = Instant::now();
        let window = Duration::from_secs(60);

        // Check request rate
        {
            let mut times = self.request_times.lock();

            // Remove old entries
            while let Some(front) = times.front() {
                if now.duration_since(*front) > window {
                    times.pop_front();
                } else {
                    break;
                }
            }

            if times.len() >= self.requests_per_minute as usize {
                return Err(EmbeddingError::RateLimitExceeded(format!(
                    "Request rate limit: {} requests/minute",
                    self.requests_per_minute
                )));
            }

            times.push_back(now);
        }

        // Check token rate (approximate)
        {
            let mut last_reset = self.last_reset.lock();
            if now.duration_since(*last_reset) > window {
                self.token_count.store(0, Ordering::Relaxed);
                *last_reset = now;
            }
        }

        let current_tokens = self.token_count.fetch_add(estimated_tokens as u64, Ordering::Relaxed);
        if current_tokens + estimated_tokens as u64 > self.tokens_per_minute as u64 {
            return Err(EmbeddingError::RateLimitExceeded(format!(
                "Token rate limit: {} tokens/minute",
                self.tokens_per_minute
            )));
        }

        Ok(())
    }
}

/// OpenAI embedding provider
pub struct OpenAIEmbeddingProvider {
    client: reqwest::blocking::Client,
    config: OpenAIConfig,
    rate_limiter: RateLimiter,
    cost_cents: AtomicU64,
    total_tokens: AtomicU64,
}

impl OpenAIEmbeddingProvider {
    /// Create a new OpenAI embedding provider
    pub fn new(config: OpenAIConfig) -> Result<Self, EmbeddingError> {
        if config.api_key.is_empty() {
            return Err(EmbeddingError::ConfigError(
                "API key is required".to_string(),
            ));
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(5)
            .build()
            .map_err(|e| EmbeddingError::NetworkError(e.to_string()))?;

        Ok(Self {
            client,
            config,
            // OpenAI limits: 3000 RPM, 1M TPM for tier 1
            rate_limiter: RateLimiter::new(500, 150_000),
            cost_cents: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
        })
    }

    /// Get total cost in cents since creation
    pub fn cost_cents(&self) -> u64 {
        self.cost_cents.load(Ordering::Relaxed)
    }

    /// Get total tokens used since creation
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Get the API URL
    fn get_url(&self) -> String {
        if let Some(endpoint) = &self.config.endpoint {
            // Azure or custom endpoint
            if let Some(api_version) = &self.config.azure_api_version {
                format!("{}?api-version={}", endpoint, api_version)
            } else {
                endpoint.clone()
            }
        } else {
            // Standard OpenAI
            "https://api.openai.com/v1/embeddings".to_string()
        }
    }

    /// Execute request with retry logic
    fn execute_with_retry(
        &self,
        texts: &[&str],
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff: 1s, 2s, 4s, ...
                let delay = Duration::from_secs(1 << (attempt - 1));
                std::thread::sleep(delay);
            }

            match self.execute_request(texts) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    // Only retry on transient errors
                    match &e {
                        EmbeddingError::NetworkError(_)
                        | EmbeddingError::RateLimitExceeded(_) => {
                            last_error = Some(e);
                            continue;
                        }
                        _ => return Err(e),
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            EmbeddingError::NetworkError("Max retries exceeded".to_string())
        }))
    }

    /// Execute a single API request
    fn execute_request(&self, texts: &[&str]) -> Result<EmbeddingResponse, EmbeddingError> {
        // Estimate tokens (rough: 1 token ≈ 4 chars)
        let estimated_tokens: u32 = texts
            .iter()
            .map(|t| (t.len() / 4 + 1) as u32)
            .sum();

        self.rate_limiter.check_and_record(estimated_tokens)?;

        let request = EmbeddingRequest {
            model: self.config.model.as_str(),
            input: texts.to_vec(),
            dimensions: self.config.reduce_dimensions,
            encoding_format: "float",
        };

        let url = self.get_url();
        let mut req = self.client.post(&url);

        // Add authorization header
        if self.config.azure_api_version.is_some() {
            req = req.header("api-key", &self.config.api_key);
        } else {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response = req
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| EmbeddingError::NetworkError(e.to_string()))?;

        let status = response.status();

        if status.is_success() {
            let embed_response: EmbeddingResponse = response
                .json()
                .map_err(|e| EmbeddingError::InferenceFailed(format!("Parse error: {}", e)))?;

            // Track usage
            let tokens = embed_response.usage.total_tokens;
            self.total_tokens.fetch_add(tokens as u64, Ordering::Relaxed);

            let cost = (tokens as f64 * self.config.model.cost_per_token() * 100.0) as u64;
            self.cost_cents.fetch_add(cost, Ordering::Relaxed);

            Ok(embed_response)
        } else if status.as_u16() == 429 {
            Err(EmbeddingError::RateLimitExceeded(
                "API rate limit exceeded".to_string(),
            ))
        } else {
            let error: ApiError = response
                .json()
                .unwrap_or(ApiError {
                    error: ApiErrorDetail {
                        message: format!("HTTP {}", status),
                        error_type: None,
                        code: None,
                    },
                });
            Err(EmbeddingError::InferenceFailed(error.error.message))
        }
    }
}

impl EmbeddingProvider for OpenAIEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        let response = self.execute_with_retry(&[text])?;

        response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| EmbeddingError::InferenceFailed("No embedding returned".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        for text in texts {
            if text.is_empty() {
                return Err(EmbeddingError::InvalidInput("Empty text in batch".to_string()));
            }
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in texts.chunks(self.config.batch_size) {
            let response = self.execute_with_retry(chunk)?;

            // Sort by index to maintain order
            let mut data = response.data;
            data.sort_by_key(|d| d.index);

            all_embeddings.extend(data.into_iter().map(|d| d.embedding));
        }

        Ok(all_embeddings)
    }

    fn dimension(&self) -> usize {
        self.config
            .reduce_dimensions
            .unwrap_or_else(|| self.config.model.dimension())
    }

    fn max_tokens(&self) -> usize {
        8191 // OpenAI limit for embedding models
    }

    fn provider_id(&self) -> &str {
        if self.config.azure_api_version.is_some() {
            "azure-openai"
        } else {
            "openai"
        }
    }

    fn is_offline(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_properties() {
        let model = OpenAIEmbeddingModel::TextEmbedding3Small;
        assert_eq!(model.as_str(), "text-embedding-3-small");
        assert_eq!(model.dimension(), 1536);
        assert!(model.cost_per_token() > 0.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = OpenAIConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.max_retries, 3);
        assert!(config.endpoint.is_none());
    }

    #[test]
    fn test_azure_config() {
        let config = OpenAIConfig::azure(
            "https://my-resource.openai.azure.com".to_string(),
            "key".to_string(),
            "my-deployment",
        );

        assert!(config.endpoint.is_some());
        assert!(config.azure_api_version.is_some());
        assert!(config.endpoint.unwrap().contains("my-deployment"));
    }

    #[test]
    fn test_provider_creation_fails_without_key() {
        let config = OpenAIConfig::default();
        let result = OpenAIEmbeddingProvider::new(config);
        assert!(matches!(result, Err(EmbeddingError::ConfigError(_))));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10, 1000);

        // Should succeed within limits
        for _ in 0..10 {
            assert!(limiter.check_and_record(50).is_ok());
        }

        // Should fail at limit
        assert!(limiter.check_and_record(50).is_err());
    }
}
