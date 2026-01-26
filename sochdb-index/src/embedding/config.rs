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

//! Embedding Configuration
//!
//! This module defines configuration types for the embedding system,
//! including provider selection, resource limits, and trigger settings.

use serde::{Deserialize, Serialize};

/// Main embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Which provider to use
    pub provider: EmbeddingProviderType,

    /// Local model settings
    pub local: LocalModelConfig,

    /// When to generate embeddings
    pub trigger: EmbeddingTrigger,

    /// Resource limits
    pub limits: ResourceLimits,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProviderType::Local,
            local: LocalModelConfig::default(),
            trigger: EmbeddingTrigger::All,
            limits: ResourceLimits::detect_system_defaults(),
        }
    }
}

/// Embedding provider type selection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingProviderType {
    /// Local embedding using built-in algorithm or fastembed (default, offline)
    Local,

    /// Use LLM crate provider (OpenAI, Ollama, Azure, etc.)
    Llm(LlmProviderConfig),

    /// Try LLM provider, fall back to local on failure
    LlmWithLocalFallback(LlmProviderConfig),

    /// Disable embeddings entirely
    Disabled,
}

/// Configuration for LLM crate embedding providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    /// Provider backend type
    pub backend: LlmBackend,

    /// API key (for cloud providers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Model name to use for embeddings
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// API base URL (for custom endpoints)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Output dimension (some providers support dimension reduction)
    #[serde(default)]
    pub dimension: Option<usize>,

    /// Maximum batch size for API calls
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum retries on failure
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_batch_size() -> usize {
    100
}

fn default_timeout() -> u64 {
    30
}

fn default_max_retries() -> u32 {
    3
}

impl Default for LlmProviderConfig {
    fn default() -> Self {
        Self {
            backend: LlmBackend::OpenAI,
            api_key: None,
            model: default_embedding_model(),
            base_url: None,
            dimension: None,
            batch_size: default_batch_size(),
            timeout_secs: default_timeout(),
            max_retries: default_max_retries(),
        }
    }
}

/// Supported LLM backends for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmBackend {
    /// OpenAI API (text-embedding-3-small, text-embedding-3-large)
    OpenAI,

    /// Azure OpenAI Service
    AzureOpenAI,

    /// Ollama (local LLM server)
    Ollama,

    /// Google AI (Gemini)
    Google,

    /// Cohere embeddings
    Cohere,

    /// Mistral AI
    Mistral,

    /// Hugging Face Inference API
    HuggingFace,

    /// XAI (Grok)
    XAI,
}

impl LlmBackend {
    /// Get default model for this backend
    pub fn default_model(&self) -> &'static str {
        match self {
            LlmBackend::OpenAI => "text-embedding-3-small",
            LlmBackend::AzureOpenAI => "text-embedding-3-small",
            LlmBackend::Ollama => "nomic-embed-text",
            LlmBackend::Google => "text-embedding-004",
            LlmBackend::Cohere => "embed-english-v3.0",
            LlmBackend::Mistral => "mistral-embed",
            LlmBackend::HuggingFace => "sentence-transformers/all-MiniLM-L6-v2",
            LlmBackend::XAI => "embedding",
        }
    }

    /// Get default dimension for this backend
    pub fn default_dimension(&self) -> usize {
        match self {
            LlmBackend::OpenAI => 1536,
            LlmBackend::AzureOpenAI => 1536,
            LlmBackend::Ollama => 768,
            LlmBackend::Google => 768,
            LlmBackend::Cohere => 1024,
            LlmBackend::Mistral => 1024,
            LlmBackend::HuggingFace => 384,
            LlmBackend::XAI => 1536,
        }
    }

    /// Whether this backend requires an API key
    pub fn requires_api_key(&self) -> bool {
        !matches!(self, LlmBackend::Ollama)
    }
}

/// Local model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelConfig {
    /// Model name (e.g., "all-MiniLM-L6-v2")
    pub model: String,

    /// Maximum concurrent inference threads
    pub max_threads: usize,

    /// Batch size for inference
    pub batch_size: usize,

    /// Auto-download new models (for fastembed)
    pub auto_download: bool,

    /// Cache directory for models
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<String>,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            model: "all-MiniLM-L6-v2".to_string(),
            max_threads: num_cpus / 2,
            batch_size: 32,
            auto_download: true,
            cache_dir: None,
        }
    }
}

/// Resource limits for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum RAM for embedding index (MB)
    pub max_memory_mb: usize,

    /// Maximum vectors to keep in index
    pub max_vectors: usize,

    /// CPU usage limit during inference (percentage, 0-100)
    pub max_cpu_percent: u8,

    /// Maximum queue size for pending embeddings
    pub max_queue_size: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self::detect_system_defaults()
    }
}

impl ResourceLimits {
    /// Auto-detect sensible defaults based on system resources
    pub fn detect_system_defaults() -> Self {
        // Try to get system memory, default to 8GB if unavailable
        let total_ram_mb = Self::get_system_memory_mb().unwrap_or(8192);

        // Use ~6% of RAM for embedding index
        let max_memory_mb = (total_ram_mb as f64 * 0.06) as usize;

        // Each PQ vector is ~48 bytes
        let max_vectors = max_memory_mb * 1024 * 1024 / 48;

        Self {
            max_memory_mb,
            max_vectors,
            max_cpu_percent: 50, // Leave headroom for UI
            max_queue_size: 10000,
        }
    }

    /// Get system memory in MB
    fn get_system_memory_mb() -> Option<usize> {
        // Platform-specific memory detection
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let output = Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .ok()?;
            let mem_bytes: u64 = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .ok()?;
            Some((mem_bytes / (1024 * 1024)) as usize)
        }

        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().ok()?;
                        return Some(kb / 1024);
                    }
                }
            }
            None
        }

        #[cfg(target_os = "windows")]
        {
            // Windows: use simple default
            None
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            None
        }
    }
}

/// When to generate embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum EmbeddingTrigger {
    /// Embed all traces with payload
    All,

    /// Only embed traces matching filter
    Filtered {
        /// Span types to embed
        span_types: Vec<String>,
        /// Minimum token count
        min_tokens: u32,
    },

    /// Manual trigger only (user initiates embedding)
    Manual,

    /// Embed on search (lazy embedding)
    OnSearch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert!(matches!(config.provider, EmbeddingProviderType::Local));
        assert_eq!(config.local.model, "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_llm_backend_defaults() {
        assert_eq!(LlmBackend::OpenAI.default_model(), "text-embedding-3-small");
        assert_eq!(LlmBackend::OpenAI.default_dimension(), 1536);
        assert!(LlmBackend::OpenAI.requires_api_key());
        assert!(!LlmBackend::Ollama.requires_api_key());
    }

    #[test]
    fn test_config_serialization() {
        let config = EmbeddingConfig {
            provider: EmbeddingProviderType::Llm(LlmProviderConfig {
                backend: LlmBackend::Ollama,
                api_key: None,
                model: "nomic-embed-text".to_string(),
                base_url: Some("http://localhost:11434".to_string()),
                dimension: Some(768),
                batch_size: 50,
                timeout_secs: 30,
                max_retries: 3,
            }),
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let parsed: EmbeddingConfig = serde_json::from_str(&json).unwrap();

        if let EmbeddingProviderType::Llm(llm_config) = &parsed.provider {
            assert_eq!(llm_config.backend, LlmBackend::Ollama);
            assert_eq!(llm_config.model, "nomic-embed-text");
        } else {
            panic!("Expected Llm provider type");
        }
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::detect_system_defaults();
        assert!(limits.max_memory_mb > 0);
        assert!(limits.max_vectors > 0);
        assert!(limits.max_cpu_percent <= 100);
    }
}
