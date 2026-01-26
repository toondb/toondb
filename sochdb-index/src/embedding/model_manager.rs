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

//! Model Manager
//!
//! Handles model download, caching, and version management for local embeddings.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during model management
#[derive(Error, Debug)]
pub enum ModelError {
    /// Model not found
    #[error("Model not found: {0}")]
    NotFound(String),

    /// Download failed
    #[error("Download failed: {0}")]
    DownloadFailed(String),

    /// Checksum mismatch
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Model manifest describing a downloadable model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Model name (e.g., "all-MiniLM-L6-v2")
    pub name: String,

    /// Version string
    pub version: String,

    /// SHA-256 checksum of the model file
    pub checksum_sha256: String,

    /// Output embedding dimension
    pub embedding_dim: usize,

    /// Maximum sequence length in tokens
    pub max_seq_length: usize,

    /// Download URL
    pub download_url: String,

    /// File size in bytes
    pub size_bytes: u64,

    /// Description of the model
    pub description: String,
}

/// Model manager for handling local embedding models
pub struct ModelManager {
    /// Directory for storing models
    model_dir: PathBuf,

    /// Known model manifests
    manifests: HashMap<String, ModelManifest>,

    /// Currently loaded models (for caching)
    loaded: HashMap<String, LoadedModel>,
}

/// A loaded model ready for use
#[allow(dead_code)]
struct LoadedModel {
    manifest: ModelManifest,
    path: PathBuf,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Result<Self, ModelError> {
        let model_dir = Self::get_model_dir()?;

        // Create directory if it doesn't exist
        fs::create_dir_all(&model_dir)?;

        let mut manager = Self {
            model_dir,
            manifests: HashMap::new(),
            loaded: HashMap::new(),
        };

        // Register built-in model manifests
        manager.register_default_models();

        // Load cached manifests
        manager.load_cached_manifests()?;

        Ok(manager)
    }

    /// Get the platform-specific model directory
    fn get_model_dir() -> Result<PathBuf, ModelError> {
        // Linux: ~/.local/share/sochdb/models/
        // macOS: ~/Library/Application Support/sochdb/models/
        // Windows: %APPDATA%\sochdb\models\
        let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));

        Ok(data_dir.join("sochdb").join("models"))
    }

    /// Register default model manifests
    fn register_default_models(&mut self) {
        // all-MiniLM-L6-v2 - Most popular, good balance
        self.manifests.insert(
            "all-MiniLM-L6-v2".to_string(),
            ModelManifest {
                name: "all-MiniLM-L6-v2".to_string(),
                version: "1.0.0".to_string(),
                checksum_sha256: "".to_string(), // Verified by fastembed
                embedding_dim: 384,
                max_seq_length: 512,
                download_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
                    .to_string(),
                size_bytes: 90_000_000,
                description: "Fast, general-purpose embedding model (384 dim)".to_string(),
            },
        );

        // BGE Small EN
        self.manifests.insert(
            "bge-small-en-v1.5".to_string(),
            ModelManifest {
                name: "bge-small-en-v1.5".to_string(),
                version: "1.5.0".to_string(),
                checksum_sha256: "".to_string(),
                embedding_dim: 384,
                max_seq_length: 512,
                download_url: "https://huggingface.co/BAAI/bge-small-en-v1.5".to_string(),
                size_bytes: 130_000_000,
                description: "High quality English embedding model (384 dim)".to_string(),
            },
        );

        // Nomic Embed Text
        self.manifests.insert(
            "nomic-embed-text-v1".to_string(),
            ModelManifest {
                name: "nomic-embed-text-v1".to_string(),
                version: "1.0.0".to_string(),
                checksum_sha256: "".to_string(),
                embedding_dim: 768,
                max_seq_length: 8192,
                download_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1".to_string(),
                size_bytes: 550_000_000,
                description: "Long context embedding model (768 dim, 8K context)".to_string(),
            },
        );

        // GTE Small
        self.manifests.insert(
            "gte-small".to_string(),
            ModelManifest {
                name: "gte-small".to_string(),
                version: "1.0.0".to_string(),
                checksum_sha256: "".to_string(),
                embedding_dim: 384,
                max_seq_length: 512,
                download_url: "https://huggingface.co/thenlper/gte-small".to_string(),
                size_bytes: 70_000_000,
                description: "Compact, high-quality embeddings (384 dim)".to_string(),
            },
        );
    }

    /// Load cached manifests from disk
    fn load_cached_manifests(&mut self) -> Result<(), ModelError> {
        let manifest_path = self.model_dir.join("manifests.json");

        if manifest_path.exists() {
            let content = fs::read_to_string(&manifest_path)?;
            let cached: HashMap<String, ModelManifest> = serde_json::from_str(&content)
                .map_err(|e| ModelError::ParseError(e.to_string()))?;

            // Merge with built-in, preferring cached for updates
            for (name, manifest) in cached {
                self.manifests.insert(name, manifest);
            }
        }

        Ok(())
    }

    /// Save manifests to disk
    #[allow(dead_code)]
    fn save_manifests(&self) -> Result<(), ModelError> {
        let manifest_path = self.model_dir.join("manifests.json");
        let content = serde_json::to_string_pretty(&self.manifests)
            .map_err(|e| ModelError::ParseError(e.to_string()))?;
        fs::write(&manifest_path, content)?;
        Ok(())
    }

    /// Get manifest for a model
    pub fn get_manifest(&self, name: &str) -> Option<&ModelManifest> {
        self.manifests.get(name)
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<&ModelManifest> {
        self.manifests.values().collect()
    }

    /// Check if a model is downloaded
    pub fn is_downloaded(&self, name: &str) -> bool {
        if let Some(manifest) = self.manifests.get(name) {
            let model_path = self.model_dir.join(&manifest.name);
            model_path.exists()
        } else {
            false
        }
    }

    /// Get the path for a downloaded model
    pub fn get_model_path(&self, name: &str) -> Option<PathBuf> {
        if let Some(manifest) = self.manifests.get(name) {
            let model_path = self.model_dir.join(&manifest.name);
            if model_path.exists() {
                return Some(model_path);
            }
        }
        None
    }

    /// Verify checksum of a downloaded model
    pub fn verify_checksum(&self, name: &str) -> Result<bool, ModelError> {
        let manifest = self
            .manifests
            .get(name)
            .ok_or_else(|| ModelError::NotFound(name.to_string()))?;

        // Skip if no checksum defined
        if manifest.checksum_sha256.is_empty() {
            return Ok(true);
        }

        let model_path = self
            .get_model_path(name)
            .ok_or_else(|| ModelError::NotFound(format!("Model {} not downloaded", name)))?;

        // Compute SHA-256 of model file
        let content = fs::read(&model_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let result = hasher.finalize();
        let actual = format!("{:x}", result);

        if actual != manifest.checksum_sha256 {
            return Err(ModelError::ChecksumMismatch {
                expected: manifest.checksum_sha256.clone(),
                actual,
            });
        }

        Ok(true)
    }

    /// Get storage usage for all downloaded models
    pub fn get_storage_usage(&self) -> Result<u64, ModelError> {
        let mut total = 0u64;

        for name in self.manifests.keys() {
            if let Some(path) = self.get_model_path(name)
                && let Ok(metadata) = fs::metadata(&path)
            {
                total += metadata.len();
            }
        }

        Ok(total)
    }

    /// Delete a downloaded model
    pub fn delete_model(&mut self, name: &str) -> Result<(), ModelError> {
        let model_path = self
            .get_model_path(name)
            .ok_or_else(|| ModelError::NotFound(format!("Model {} not found", name)))?;

        // Remove from loaded cache
        self.loaded.remove(name);

        // Delete the model directory
        if model_path.is_dir() {
            fs::remove_dir_all(&model_path)?;
        } else {
            fs::remove_file(&model_path)?;
        }

        Ok(())
    }

    /// Get recommended model for a use case
    pub fn recommend_model(&self, use_case: ModelUseCase) -> &str {
        match use_case {
            ModelUseCase::GeneralPurpose => "all-MiniLM-L6-v2",
            ModelUseCase::HighQuality => "bge-small-en-v1.5",
            ModelUseCase::LongContext => "nomic-embed-text-v1",
            ModelUseCase::LowMemory => "gte-small",
            ModelUseCase::Multilingual => "multilingual-e5-small",
        }
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            model_dir: PathBuf::from("."),
            manifests: HashMap::new(),
            loaded: HashMap::new(),
        })
    }
}

/// Use case for model recommendation
#[derive(Debug, Clone, Copy)]
pub enum ModelUseCase {
    /// General purpose (default)
    GeneralPurpose,
    /// High quality at cost of speed
    HighQuality,
    /// Long context (8K+ tokens)
    LongContext,
    /// Minimize memory usage
    LowMemory,
    /// Multilingual support
    Multilingual,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_manager_creation() {
        let manager = ModelManager::new().unwrap();
        assert!(!manager.manifests.is_empty());
    }

    #[test]
    fn test_list_models() {
        let manager = ModelManager::new().unwrap();
        let models = manager.list_models();
        assert!(!models.is_empty());

        // Check that all-MiniLM-L6-v2 is included
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"all-MiniLM-L6-v2"));
    }

    #[test]
    fn test_get_manifest() {
        let manager = ModelManager::new().unwrap();

        let manifest = manager.get_manifest("all-MiniLM-L6-v2");
        assert!(manifest.is_some());

        let manifest = manifest.unwrap();
        assert_eq!(manifest.embedding_dim, 384);
        assert_eq!(manifest.max_seq_length, 512);
    }

    #[test]
    fn test_recommend_model() {
        let manager = ModelManager::new().unwrap();

        assert_eq!(
            manager.recommend_model(ModelUseCase::GeneralPurpose),
            "all-MiniLM-L6-v2"
        );
        assert_eq!(
            manager.recommend_model(ModelUseCase::LongContext),
            "nomic-embed-text-v1"
        );
    }

    #[test]
    fn test_model_dir() {
        let dir = ModelManager::get_model_dir().unwrap();
        assert!(dir.to_string_lossy().contains("sochdb"));
        assert!(dir.to_string_lossy().contains("models"));
    }
}
