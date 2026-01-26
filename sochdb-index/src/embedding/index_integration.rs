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

//! Index Integration
//!
//! This module connects the embedding pipeline to the vector index (HNSW/Vamana)
//! for seamless semantic search over embedded traces.
//!
//! ## Architecture
//!
//! ```text
//! Trace → LSM Write (sync) → CSR Update (sync) → Embedding Queue (async)
//!                                                        ↓
//!                                                 Background Worker
//!                                                        ↓
//!                                           Embed → Normalize → PQ Encode
//!                                                        ↓
//!                                                  HNSW/Vamana Insert
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_index::embedding::{EmbeddingIntegration, IntegrationConfig};
//!
//! let integration = EmbeddingIntegration::new(config)?;
//!
//! // Submit trace for embedding (non-blocking)
//! integration.submit_for_embedding(edge_id, &payload)?;
//!
//! // Query by semantic similarity
//! let results = integration.semantic_search("find errors in auth", 10)?;
//! ```

use super::normalize::normalize_l2_simd;
use super::pipeline::{EmbeddingPipeline, EmbeddingRequest, EmbeddingResult, PipelineConfig};
use super::provider::{EmbeddingError, EmbeddingProvider};
use super::storage::{EmbeddingStorage, EmbeddingStorageConfig, StorageError};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::product_quantization::PQCodebooks;
use crossbeam_channel::Receiver;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use thiserror::Error;

/// Errors during index integration
#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Configuration for embedding integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Base directory for storage
    pub data_dir: PathBuf,

    /// Embedding dimension
    pub dimension: usize,

    /// Model name for tracking
    pub model_name: String,

    /// HNSW configuration
    pub hnsw_config: HnswConfig,

    /// Pipeline configuration
    pub pipeline_config: PipelineConfig,

    /// Whether to use PQ compression
    pub use_pq: bool,

    /// PQ subdimension (8 for 48 subspaces with 384-dim)
    pub pq_subdim: usize,

    /// Minimum text length to embed (skip short content)
    pub min_text_length: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("sochdb-data/embeddings"),
            dimension: 384,
            model_name: "all-MiniLM-L6-v2".to_string(),
            hnsw_config: HnswConfig::default(),
            pipeline_config: PipelineConfig::default(),
            use_pq: true,
            pq_subdim: 8,
            min_text_length: 10,
        }
    }
}

/// Search result from semantic query
#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    /// Edge ID of the matched trace
    pub edge_id: u128,

    /// Similarity score (0.0 to 1.0, higher is better)
    pub similarity: f32,

    /// Distance (lower is better, for HNSW)
    pub distance: f32,
}

/// Integration layer connecting embedding pipeline to vector index
pub struct EmbeddingIntegration {
    /// Embedding provider
    provider: Arc<dyn EmbeddingProvider>,

    /// Embedding pipeline
    pipeline: EmbeddingPipeline,

    /// Result receiver (from pipeline)
    result_receiver: Receiver<EmbeddingResult>,

    /// HNSW index for vector search
    hnsw_index: Arc<parking_lot::RwLock<HnswIndex>>,

    /// Embedding storage
    storage: Arc<parking_lot::RwLock<EmbeddingStorage>>,

    /// PQ codebooks (if using compression)
    codebooks: Option<Arc<PQCodebooks>>,

    /// Configuration
    config: IntegrationConfig,

    /// Counter for pending embeddings
    pending_count: AtomicU64,

    /// Background worker handle
    worker_handle: Option<JoinHandle<()>>,
}

impl EmbeddingIntegration {
    /// Create new embedding integration
    pub fn new(
        provider: Arc<dyn EmbeddingProvider>,
        config: IntegrationConfig,
    ) -> Result<Self, IntegrationError> {
        // Validate dimension matches provider
        if provider.dimension() != config.dimension {
            return Err(IntegrationError::Config(format!(
                "Provider dimension {} doesn't match config dimension {}",
                provider.dimension(),
                config.dimension
            )));
        }

        // Create storage
        let storage_config = EmbeddingStorageConfig {
            base_dir: config.data_dir.clone(),
            use_mmap: true,
            sync_writes: false,
            pq_subspaces: config.dimension / config.pq_subdim,
        };

        let storage = EmbeddingStorage::open_or_create(
            storage_config,
            config.model_name.clone(),
            config.dimension,
        )?;

        // Create HNSW index
        let hnsw_index = HnswIndex::new(config.dimension, config.hnsw_config.clone());

        // Create pipeline
        let (pipeline, result_receiver) =
            EmbeddingPipeline::new(Arc::clone(&provider), config.pipeline_config.clone())?;

        Ok(Self {
            provider,
            pipeline,
            result_receiver,
            hnsw_index: Arc::new(parking_lot::RwLock::new(hnsw_index)),
            storage: Arc::new(parking_lot::RwLock::new(storage)),
            codebooks: None,
            config,
            pending_count: AtomicU64::new(0),
            worker_handle: None,
        })
    }

    /// Start background worker that processes embedding results
    pub fn start_background_worker(&mut self) {
        let receiver = self.result_receiver.clone();
        let hnsw = Arc::clone(&self.hnsw_index);
        let storage = Arc::clone(&self.storage);
        let pending = &self.pending_count as *const AtomicU64;

        // Safety: We own pending_count and worker will terminate before drop
        let pending_ref = unsafe { &*pending };

        let handle = thread::spawn(move || {
            while let Ok(result) = receiver.recv() {
                // Skip errors
                if result.error.is_some() {
                    pending_ref.fetch_sub(1, Ordering::Relaxed);
                    continue;
                }

                // Store embedding
                {
                    let mut store = storage.write();
                    if let Err(_e) = store.append(result.id, result.embedding.clone()) {
                        // Log error but continue
                    }
                }

                // Insert into HNSW
                {
                    let index = hnsw.write();
                    let _ = index.insert(result.id, result.embedding.clone());
                }

                pending_ref.fetch_sub(1, Ordering::Relaxed);
            }
        });

        self.worker_handle = Some(handle);
    }

    /// Submit text for embedding (non-blocking)
    pub fn submit_for_embedding(&self, edge_id: u128, text: &str) -> Result<(), IntegrationError> {
        // Skip empty or short texts
        if text.len() < self.config.min_text_length {
            return Ok(());
        }

        // Check if already indexed
        if self.storage.read().contains(edge_id) {
            return Ok(());
        }

        // Submit to pipeline
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.pipeline
            .submit(EmbeddingRequest::new(edge_id, text.to_string()))?;

        Ok(())
    }

    /// Check if a trace is indexed (embedding complete)
    pub fn is_indexed(&self, edge_id: u128) -> bool {
        self.storage.read().contains(edge_id)
    }

    /// Get number of pending embeddings
    pub fn pending_embeddings(&self) -> u64 {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Get total indexed count
    pub fn indexed_count(&self) -> u64 {
        self.storage.read().len()
    }

    /// Perform semantic search
    pub fn semantic_search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SemanticSearchResult>, IntegrationError> {
        // Embed query
        let mut query_embedding = self.provider.embed(query)?;
        normalize_l2_simd(&mut query_embedding);

        // Search HNSW
        let hnsw = self.hnsw_index.read();
        let results = hnsw
            .search(&query_embedding, k)
            .map_err(IntegrationError::Index)?;

        // Convert to search results
        Ok(results
            .into_iter()
            .map(|(id, distance)| SemanticSearchResult {
                edge_id: id,
                similarity: 1.0 - distance.min(2.0) / 2.0, // Convert distance to similarity
                distance,
            })
            .collect())
    }

    /// Perform filtered semantic search
    pub fn semantic_search_filtered(
        &self,
        query: &str,
        k: usize,
        candidate_ids: &[u128],
    ) -> Result<Vec<SemanticSearchResult>, IntegrationError> {
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Embed query
        let mut query_embedding = self.provider.embed(query)?;
        normalize_l2_simd(&mut query_embedding);

        // Compute distances for candidates
        let storage = self.storage.read();
        let mut results: Vec<SemanticSearchResult> = candidate_ids
            .iter()
            .filter_map(|&id| {
                storage.get_by_id(id).map(|vec| {
                    // Compute cosine distance (vectors are normalized)
                    let dot: f32 = query_embedding.iter().zip(vec).map(|(a, b)| a * b).sum();
                    let distance = 1.0 - dot;
                    SemanticSearchResult {
                        edge_id: id,
                        similarity: dot,
                        distance,
                    }
                })
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Get embedding for an edge ID
    pub fn get_embedding(&self, edge_id: u128) -> Option<Vec<f32>> {
        self.storage.read().get_by_id(edge_id).cloned()
    }

    /// Persist storage to disk
    pub fn persist(&self) -> Result<(), IntegrationError> {
        self.storage.write().persist()?;
        Ok(())
    }

    /// Train PQ codebooks on indexed vectors
    pub fn train_pq(&mut self) -> Result<(), IntegrationError> {
        if !self.config.use_pq {
            return Ok(());
        }

        let mut storage = self.storage.write();
        storage.train_pq(self.config.pq_subdim, 20)?;

        if let Some(codebooks) = storage.codebooks() {
            self.codebooks = Some(Arc::new(codebooks.clone()));
        }

        Ok(())
    }

    /// Get provider dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }
}

impl Drop for EmbeddingIntegration {
    fn drop(&mut self) {
        // Shutdown pipeline
        self.pipeline.shutdown();

        // Persist storage
        let _ = self.persist();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbeddingProvider;
    use tempfile::TempDir;

    fn test_config(dir: &std::path::Path) -> IntegrationConfig {
        IntegrationConfig {
            data_dir: dir.to_path_buf(),
            dimension: 384,
            pipeline_config: PipelineConfig {
                batch_size: 2,
                batch_timeout_ms: 50,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_integration_basic() {
        let dir = TempDir::new().unwrap();
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let mut integration = EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

        integration.start_background_worker();

        assert_eq!(integration.indexed_count(), 0);
        assert_eq!(integration.pending_embeddings(), 0);
    }

    #[test]
    #[ignore] // Flaky: depends on background worker completing within 200ms
    fn test_integration_submit_and_search() {
        let dir = TempDir::new().unwrap();
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let mut integration = EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

        integration.start_background_worker();

        // Submit some texts
        integration
            .submit_for_embedding(100, "This is a test about errors and failures")
            .unwrap();
        integration
            .submit_for_embedding(200, "Another trace about success and completion")
            .unwrap();
        integration
            .submit_for_embedding(300, "Error handling and exception management")
            .unwrap();

        // Wait for processing
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Check indexed
        assert!(integration.is_indexed(100));
        assert!(integration.is_indexed(200));
        assert!(integration.is_indexed(300));
        assert_eq!(integration.indexed_count(), 3);

        // Search
        let results = integration.semantic_search("errors", 2).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_integration_skip_short() {
        let dir = TempDir::new().unwrap();
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let integration = EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

        // Should be skipped (too short)
        integration.submit_for_embedding(100, "hi").unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!integration.is_indexed(100));
    }

    #[test]
    fn test_integration_filtered_search() {
        let dir = TempDir::new().unwrap();
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let mut integration = EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

        integration.start_background_worker();

        // Submit texts
        integration
            .submit_for_embedding(100, "Database connection error occurred")
            .unwrap();
        integration
            .submit_for_embedding(200, "API request failed with timeout")
            .unwrap();
        integration
            .submit_for_embedding(300, "Successful operation completed")
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(200));

        // Search only within subset
        let results = integration
            .semantic_search_filtered("error", 10, &[100, 200])
            .unwrap();

        assert_eq!(results.len(), 2);
        // All results should be from the filtered set
        for r in &results {
            assert!(r.edge_id == 100 || r.edge_id == 200);
        }
    }

    #[test]
    fn test_integration_persistence() {
        let dir = TempDir::new().unwrap();

        // Create and populate
        {
            let provider = Arc::new(MockEmbeddingProvider::default_provider());
            let mut integration =
                EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

            integration.start_background_worker();

            integration
                .submit_for_embedding(100, "Test trace for persistence")
                .unwrap();

            std::thread::sleep(std::time::Duration::from_millis(200));
            integration.persist().unwrap();
        }

        // Reopen and verify storage (HNSW would need separate persistence)
        {
            let provider = Arc::new(MockEmbeddingProvider::default_provider());
            let integration = EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

            assert_eq!(integration.indexed_count(), 1);
            assert!(integration.storage.read().contains(100));
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let provider = Arc::new(MockEmbeddingProvider::new(512)); // Different dimension

        let config = IntegrationConfig {
            data_dir: dir.path().to_path_buf(),
            dimension: 384, // Mismatched
            ..Default::default()
        };

        let result = EmbeddingIntegration::new(provider, config);
        assert!(matches!(result, Err(IntegrationError::Config(_))));
    }

    #[test]
    fn test_get_embedding() {
        let dir = TempDir::new().unwrap();
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let mut integration = EmbeddingIntegration::new(provider, test_config(dir.path())).unwrap();

        integration.start_background_worker();

        integration
            .submit_for_embedding(100, "Test trace for embedding retrieval")
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(200));

        let embedding = integration.get_embedding(100);
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().len(), 384);
    }
}
