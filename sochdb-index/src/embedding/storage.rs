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

//! Embedding Storage
//!
//! This module handles persistent storage of embeddings with efficient
//! access patterns for both search and retrieval.
//!
//! ## Storage Layout
//!
//! ```text
//! embeddings/
//! ├── pq_codes.bin    # PQ codes (48 bytes per vector, always in RAM)
//! ├── codebooks.bin   # PQ codebooks (~400 KB, always in RAM)
//! ├── vectors.bin     # Full F32 vectors (mmap'd, paged on demand)
//! ├── id_map.bin      # edge_id → vector_idx mapping
//! └── metadata.json   # Model version, count, checksum
//! ```
//!
//! ## Access Patterns
//!
//! - **Hot path (search)**: PQ codes in RAM for approximate distance
//! - **Cold path (rerank)**: Full vectors via mmap with page faults
//! - **ID lookup**: B-tree map for sparse edge_ids

use crate::product_quantization::PQCodebooks;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid storage format: {0}")]
    InvalidFormat(String),

    #[error("Storage not found: {0}")]
    NotFound(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },
}

/// Configuration for embedding storage
#[derive(Debug, Clone)]
pub struct EmbeddingStorageConfig {
    /// Base directory for storage
    pub base_dir: PathBuf,

    /// Whether to use memory mapping for vectors
    pub use_mmap: bool,

    /// Whether to persist on every write
    pub sync_writes: bool,

    /// PQ subspaces (48 for 32x compression)
    pub pq_subspaces: usize,
}

impl Default for EmbeddingStorageConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("embeddings"),
            use_mmap: true,
            sync_writes: false,
            pq_subspaces: 48,
        }
    }
}

/// Metadata for embedding storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Format version
    pub version: String,

    /// Embedding model name
    pub model_name: String,

    /// Model version/hash
    pub model_hash: String,

    /// Embedding dimension
    pub dimension: usize,

    /// Number of stored vectors
    pub count: u64,

    /// PQ subspaces
    pub pq_subspaces: usize,

    /// Centroids per subspace
    pub centroids_per_subspace: usize,

    /// SHA-256 checksum of data
    pub checksum: Option<String>,

    /// Creation timestamp
    pub created_at: String,

    /// Last modified timestamp
    pub modified_at: String,
}

impl EmbeddingMetadata {
    /// Create new metadata
    pub fn new(model_name: String, dimension: usize) -> Self {
        let now = chrono_lite_now();
        Self {
            version: "1.0".to_string(),
            model_name,
            model_hash: String::new(),
            dimension,
            count: 0,
            pq_subspaces: 48,
            centroids_per_subspace: 256,
            checksum: None,
            created_at: now.clone(),
            modified_at: now,
        }
    }
}

/// Simple timestamp function (avoids chrono dependency)
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// ID mapping entry (for binary serialization)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[allow(dead_code)]
struct IdMapEntry {
    edge_id: u128,
    vector_idx: u64,
}

/// Embedding storage manager
pub struct EmbeddingStorage {
    /// Configuration
    config: EmbeddingStorageConfig,

    /// Metadata
    metadata: EmbeddingMetadata,

    /// edge_id → vector_idx mapping
    id_map: BTreeMap<u128, u64>,

    /// Reverse mapping for iteration
    idx_to_id: Vec<u128>,

    /// Full vectors (in memory for simplicity, could use mmap)
    vectors: Vec<Vec<f32>>,

    /// PQ codebooks (trained centroids)
    codebooks: Option<PQCodebooks>,

    /// PQ codes for each vector (raw bytes, n_subspaces per vector)
    pq_codes: Option<Vec<Vec<u8>>>,

    /// Dirty flag for persistence
    dirty: bool,
}

impl EmbeddingStorage {
    /// Create new storage with configuration
    pub fn new(config: EmbeddingStorageConfig, model_name: String, dimension: usize) -> Self {
        Self {
            config,
            metadata: EmbeddingMetadata::new(model_name, dimension),
            id_map: BTreeMap::new(),
            idx_to_id: Vec::new(),
            vectors: Vec::new(),
            codebooks: None,
            pq_codes: None,
            dirty: false,
        }
    }

    /// Open existing storage or create new
    pub fn open_or_create(
        config: EmbeddingStorageConfig,
        model_name: String,
        dimension: usize,
    ) -> Result<Self, StorageError> {
        let metadata_path = config.base_dir.join("metadata.json");

        if metadata_path.exists() {
            Self::open(config)
        } else {
            fs::create_dir_all(&config.base_dir)?;
            Ok(Self::new(config, model_name, dimension))
        }
    }

    /// Open existing storage
    pub fn open(config: EmbeddingStorageConfig) -> Result<Self, StorageError> {
        let metadata_path = config.base_dir.join("metadata.json");
        let vectors_path = config.base_dir.join("vectors.bin");
        let id_map_path = config.base_dir.join("id_map.bin");

        // Load metadata
        let metadata: EmbeddingMetadata = {
            let file = File::open(&metadata_path)
                .map_err(|_| StorageError::NotFound(metadata_path.display().to_string()))?;
            serde_json::from_reader(BufReader::new(file))
                .map_err(|e| StorageError::Serialization(e.to_string()))?
        };

        // Load vectors
        let vectors = if vectors_path.exists() {
            Self::load_vectors(&vectors_path, metadata.dimension, metadata.count as usize)?
        } else {
            Vec::new()
        };

        // Load ID map
        let (id_map, idx_to_id) = if id_map_path.exists() {
            Self::load_id_map(&id_map_path)?
        } else {
            (BTreeMap::new(), Vec::new())
        };

        // Load codebooks if present
        let codebooks_path = config.base_dir.join("codebooks.bin");
        let codebooks = if codebooks_path.exists() {
            Some(Self::load_codebooks(&codebooks_path)?)
        } else {
            None
        };

        // Load PQ codes if present
        let pq_codes_path = config.base_dir.join("pq_codes.bin");
        let pq_codes = if pq_codes_path.exists() && codebooks.is_some() {
            Some(Self::load_pq_codes(
                &pq_codes_path,
                metadata.count as usize,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            metadata,
            id_map,
            idx_to_id,
            vectors,
            codebooks,
            pq_codes,
            dirty: false,
        })
    }

    /// Append a new embedding
    pub fn append(&mut self, edge_id: u128, vector: Vec<f32>) -> Result<u64, StorageError> {
        if vector.len() != self.metadata.dimension {
            return Err(StorageError::InvalidFormat(format!(
                "Vector dimension {} doesn't match storage dimension {}",
                vector.len(),
                self.metadata.dimension
            )));
        }

        let idx = self.metadata.count;

        // Store vector
        self.vectors.push(vector);

        // Update ID mappings
        self.id_map.insert(edge_id, idx);
        self.idx_to_id.push(edge_id);

        // Increment count
        self.metadata.count += 1;
        self.dirty = true;

        Ok(idx)
    }

    /// Get vector by edge ID
    pub fn get_by_id(&self, edge_id: u128) -> Option<&Vec<f32>> {
        self.id_map
            .get(&edge_id)
            .and_then(|&idx| self.vectors.get(idx as usize))
    }

    /// Get vector by index
    pub fn get_by_index(&self, idx: u64) -> Option<&Vec<f32>> {
        self.vectors.get(idx as usize)
    }

    /// Get edge ID by index
    pub fn get_id_by_index(&self, idx: u64) -> Option<u128> {
        self.idx_to_id.get(idx as usize).copied()
    }

    /// Check if edge ID exists
    pub fn contains(&self, edge_id: u128) -> bool {
        self.id_map.contains_key(&edge_id)
    }

    /// Get number of stored embeddings
    pub fn len(&self) -> u64 {
        self.metadata.count
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.metadata.count == 0
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.metadata.dimension
    }

    /// Get metadata
    pub fn metadata(&self) -> &EmbeddingMetadata {
        &self.metadata
    }

    /// Set PQ codebooks (after training)
    pub fn set_codebooks(&mut self, codebooks: PQCodebooks) {
        self.codebooks = Some(codebooks);
        self.dirty = true;
    }

    /// Get PQ codebooks
    pub fn codebooks(&self) -> Option<&PQCodebooks> {
        self.codebooks.as_ref()
    }

    /// Set PQ codes for all vectors
    pub fn set_pq_codes(&mut self, codes: Vec<Vec<u8>>) {
        self.pq_codes = Some(codes);
        self.dirty = true;
    }

    /// Get PQ codes
    pub fn pq_codes(&self) -> Option<&Vec<Vec<u8>>> {
        self.pq_codes.as_ref()
    }

    /// Get PQ code for a specific vector by index
    pub fn get_pq_code(&self, idx: u64) -> Option<&[u8]> {
        self.pq_codes
            .as_ref()
            .and_then(|codes| codes.get(idx as usize).map(|v| v.as_slice()))
    }

    /// Train PQ codebooks on stored vectors
    pub fn train_pq(&mut self, subdim: usize, n_iter: usize) -> Result<(), StorageError> {
        if self.vectors.is_empty() {
            return Err(StorageError::InvalidFormat(
                "Cannot train PQ on empty storage".to_string(),
            ));
        }

        // Convert to ndarray format
        let training_data: Vec<_> = self
            .vectors
            .iter()
            .map(|v| ndarray::Array1::from_vec(v.clone()))
            .collect();

        // Train codebooks with proper arguments
        let codebooks = PQCodebooks::train(&training_data, n_iter, subdim);

        self.codebooks = Some(codebooks);
        self.dirty = true;

        Ok(())
    }

    /// Persist storage to disk
    pub fn persist(&mut self) -> Result<(), StorageError> {
        if !self.dirty {
            return Ok(());
        }

        fs::create_dir_all(&self.config.base_dir)?;

        // Update timestamp
        self.metadata.modified_at = chrono_lite_now();

        // Save metadata
        self.save_metadata()?;

        // Save vectors
        self.save_vectors()?;

        // Save ID map
        self.save_id_map()?;

        // Save codebooks if present
        if self.codebooks.is_some() {
            self.save_codebooks()?;
        }

        // Save PQ codes if present
        if self.pq_codes.is_some() {
            self.save_pq_codes()?;
        }

        self.dirty = false;
        Ok(())
    }

    /// Iterate over all vectors
    pub fn iter(&self) -> impl Iterator<Item = (u128, &Vec<f32>)> {
        self.idx_to_id
            .iter()
            .zip(self.vectors.iter())
            .map(|(&id, vec)| (id, vec))
    }

    // Private helper methods

    fn save_metadata(&self) -> Result<(), StorageError> {
        let path = self.config.base_dir.join("metadata.json");
        let file = File::create(path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), &self.metadata)
            .map_err(|e| StorageError::Serialization(e.to_string()))
    }

    fn save_vectors(&self) -> Result<(), StorageError> {
        let path = self.config.base_dir.join("vectors.bin");
        let mut file = BufWriter::new(File::create(path)?);

        // Write dimension and count header
        file.write_all(&(self.metadata.dimension as u32).to_le_bytes())?;
        file.write_all(&self.metadata.count.to_le_bytes())?;

        // Write vectors
        for vec in &self.vectors {
            for &val in vec {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        file.flush()?;
        Ok(())
    }

    fn load_vectors(
        path: &Path,
        dimension: usize,
        count: usize,
    ) -> Result<Vec<Vec<f32>>, StorageError> {
        let mut file = BufReader::new(File::open(path)?);

        // Read header
        let mut dim_bytes = [0u8; 4];
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut dim_bytes)?;
        file.read_exact(&mut count_bytes)?;

        let stored_dim = u32::from_le_bytes(dim_bytes) as usize;
        let stored_count = u64::from_le_bytes(count_bytes) as usize;

        if stored_dim != dimension {
            return Err(StorageError::InvalidFormat(format!(
                "Dimension mismatch: {} vs {}",
                stored_dim, dimension
            )));
        }

        // Read vectors
        let mut vectors = Vec::with_capacity(count.min(stored_count));
        let mut float_bytes = [0u8; 4];

        for _ in 0..stored_count {
            let mut vec = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                file.read_exact(&mut float_bytes)?;
                vec.push(f32::from_le_bytes(float_bytes));
            }
            vectors.push(vec);
        }

        Ok(vectors)
    }

    fn save_id_map(&self) -> Result<(), StorageError> {
        let path = self.config.base_dir.join("id_map.bin");
        let mut file = BufWriter::new(File::create(path)?);

        // Write count
        file.write_all(&(self.id_map.len() as u64).to_le_bytes())?;

        // Write entries
        for (&edge_id, &vector_idx) in &self.id_map {
            file.write_all(&edge_id.to_le_bytes())?;
            file.write_all(&vector_idx.to_le_bytes())?;
        }

        file.flush()?;
        Ok(())
    }

    fn load_id_map(path: &Path) -> Result<(BTreeMap<u128, u64>, Vec<u128>), StorageError> {
        let mut file = BufReader::new(File::open(path)?);

        // Read count
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;

        let mut id_map = BTreeMap::new();
        let mut idx_to_id = vec![0u128; count];

        let mut id_bytes = [0u8; 16];
        let mut idx_bytes = [0u8; 8];

        for _ in 0..count {
            file.read_exact(&mut id_bytes)?;
            file.read_exact(&mut idx_bytes)?;

            let edge_id = u128::from_le_bytes(id_bytes);
            let vector_idx = u64::from_le_bytes(idx_bytes);

            id_map.insert(edge_id, vector_idx);
            if (vector_idx as usize) < idx_to_id.len() {
                idx_to_id[vector_idx as usize] = edge_id;
            }
        }

        Ok((id_map, idx_to_id))
    }

    fn save_codebooks(&self) -> Result<(), StorageError> {
        let path = self.config.base_dir.join("codebooks.bin");
        let file = File::create(path)?;

        if let Some(codebooks) = &self.codebooks {
            bincode::serialize_into(BufWriter::new(file), codebooks)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
        }

        Ok(())
    }

    fn load_codebooks(path: &Path) -> Result<PQCodebooks, StorageError> {
        let file = File::open(path)?;
        bincode::deserialize_from(BufReader::new(file))
            .map_err(|e| StorageError::Serialization(e.to_string()))
    }

    fn save_pq_codes(&self) -> Result<(), StorageError> {
        let path = self.config.base_dir.join("pq_codes.bin");
        let file = File::create(path)?;

        if let Some(codes) = &self.pq_codes {
            bincode::serialize_into(BufWriter::new(file), codes)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
        }

        Ok(())
    }

    fn load_pq_codes(path: &Path, count: usize) -> Result<Vec<Vec<u8>>, StorageError> {
        let file = File::open(path)?;
        let codes: Vec<Vec<u8>> = bincode::deserialize_from(BufReader::new(file))
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        if codes.len() != count {
            return Err(StorageError::InvalidFormat(format!(
                "PQ codes count mismatch: {} vs {}",
                codes.len(),
                count
            )));
        }

        Ok(codes)
    }
}

impl Drop for EmbeddingStorage {
    fn drop(&mut self) {
        if self.dirty {
            let _ = self.persist();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(dir: &Path) -> EmbeddingStorageConfig {
        EmbeddingStorageConfig {
            base_dir: dir.to_path_buf(),
            ..Default::default()
        }
    }

    #[test]
    fn test_storage_basic() {
        let dir = TempDir::new().unwrap();
        let storage = EmbeddingStorage::new(test_config(dir.path()), "test-model".to_string(), 384);

        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.dimension(), 384);
    }

    #[test]
    fn test_storage_append_and_get() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            EmbeddingStorage::new(test_config(dir.path()), "test-model".to_string(), 4);

        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![5.0, 6.0, 7.0, 8.0];

        let idx1 = storage.append(100, vec1.clone()).unwrap();
        let idx2 = storage.append(200, vec2.clone()).unwrap();

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(storage.len(), 2);

        // Get by ID
        assert_eq!(storage.get_by_id(100), Some(&vec1));
        assert_eq!(storage.get_by_id(200), Some(&vec2));
        assert_eq!(storage.get_by_id(300), None);

        // Get by index
        assert_eq!(storage.get_by_index(0), Some(&vec1));
        assert_eq!(storage.get_by_index(1), Some(&vec2));

        // ID lookup
        assert_eq!(storage.get_id_by_index(0), Some(100));
        assert_eq!(storage.get_id_by_index(1), Some(200));
    }

    #[test]
    fn test_storage_persistence() {
        let dir = TempDir::new().unwrap();

        // Create and populate
        {
            let mut storage =
                EmbeddingStorage::new(test_config(dir.path()), "test-model".to_string(), 4);

            storage.append(100, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            storage.append(200, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
            storage.persist().unwrap();
        }

        // Reopen and verify
        {
            let storage = EmbeddingStorage::open(test_config(dir.path())).unwrap();

            assert_eq!(storage.len(), 2);
            assert_eq!(storage.get_by_id(100), Some(&vec![1.0, 2.0, 3.0, 4.0]));
            assert_eq!(storage.get_by_id(200), Some(&vec![5.0, 6.0, 7.0, 8.0]));
        }
    }

    #[test]
    fn test_storage_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            EmbeddingStorage::new(test_config(dir.path()), "test-model".to_string(), 4);

        let result = storage.append(100, vec![1.0, 2.0, 3.0]); // Wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_iter() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            EmbeddingStorage::new(test_config(dir.path()), "test-model".to_string(), 2);

        storage.append(10, vec![1.0, 2.0]).unwrap();
        storage.append(20, vec![3.0, 4.0]).unwrap();
        storage.append(30, vec![5.0, 6.0]).unwrap();

        let items: Vec<_> = storage.iter().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (10, &vec![1.0, 2.0]));
        assert_eq!(items[1], (20, &vec![3.0, 4.0]));
        assert_eq!(items[2], (30, &vec![5.0, 6.0]));
    }

    #[test]
    fn test_storage_contains() {
        let dir = TempDir::new().unwrap();
        let mut storage =
            EmbeddingStorage::new(test_config(dir.path()), "test-model".to_string(), 2);

        storage.append(100, vec![1.0, 2.0]).unwrap();

        assert!(storage.contains(100));
        assert!(!storage.contains(200));
    }

    #[test]
    fn test_open_or_create() {
        let dir = TempDir::new().unwrap();

        // First call creates
        {
            let mut storage = EmbeddingStorage::open_or_create(
                test_config(dir.path()),
                "test-model".to_string(),
                4,
            )
            .unwrap();

            storage.append(100, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            storage.persist().unwrap();
        }

        // Second call opens
        {
            let storage = EmbeddingStorage::open_or_create(
                test_config(dir.path()),
                "test-model".to_string(),
                4,
            )
            .unwrap();

            assert_eq!(storage.len(), 1);
            assert!(storage.contains(100));
        }
    }

    #[test]
    fn test_metadata() {
        let dir = TempDir::new().unwrap();
        let storage = EmbeddingStorage::new(test_config(dir.path()), "my-model".to_string(), 384);

        let meta = storage.metadata();
        assert_eq!(meta.model_name, "my-model");
        assert_eq!(meta.dimension, 384);
        assert_eq!(meta.version, "1.0");
    }
}
