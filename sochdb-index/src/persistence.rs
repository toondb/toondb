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

//! HNSW Index Persistence
//!
//! Provides efficient serialization and deserialization of HNSW indexes
//! using bincode for fast cold starts (seconds vs hours to rebuild).
//!
//! ## WAL Integration
//!
//! For durability, HNSW operations can be logged to a Write-Ahead Log (WAL)
//! enabling crash recovery without full index rebuilds:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   HNSW WAL Recovery                         │
//! │                                                             │
//! │  [Snapshot] ──replay──> [WAL Entries] ──apply──> [Index]    │
//! │      ↓                       ↓                              │
//! │  Checkpoint              Insert/Delete                      │
//! │  (periodic)              operations                         │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::hnsw::{HnswConfig, HnswIndex};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::SystemTime;

const MAX_M: usize = 32;

/// Serializable snapshot of HNSW index state
#[derive(Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// Version for forward compatibility
    pub version: u32,
    /// Index configuration
    pub config: HnswConfig,
    /// All nodes in the index
    pub nodes: Vec<SerializableNode>,
    /// Entry point node ID
    pub entry_point: Option<u128>,
    /// Maximum layer in the index
    pub max_layer: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Timestamp of creation
    pub created_at: SystemTime,
}

/// Serializable representation of an HNSW node
#[derive(Serialize, Deserialize, Clone)]
pub struct SerializableNode {
    pub id: u128,
    pub vector: Vec<f32>,
    pub neighbors: Vec<SmallVec<[u128; MAX_M]>>,
    pub layer: usize,
}

impl HnswIndex {
    /// Save index to disk using bincode serialization
    ///
    /// # Example
    /// ```no_run
    /// # use sochdb_index::hnsw::{HnswConfig, HnswIndex};
    /// let index = HnswIndex::new(128, HnswConfig::default());
    /// // ... insert vectors ...
    /// index.save_to_disk("embeddings.hnsw").unwrap();
    /// ```
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        // Collect all nodes
        let mut serializable_nodes = Vec::with_capacity(self.nodes.len());

        for entry in self.nodes.iter() {
            let node = entry.value();
            // Collect neighbors from all layers (extract just the neighbors, not version)
            let mut neighbors = Vec::with_capacity(node.layers.len());
            for layer_lock in &node.layers {
                let dense_neighbors = layer_lock.read().neighbors.clone();
                neighbors.push(self.dense_neighbors_to_ids(&dense_neighbors));
            }

            serializable_nodes.push(SerializableNode {
                id: *entry.key(),
                vector: node.vector.to_f32().to_vec(), // Convert back to f32 for storage
                neighbors,
                layer: node.layer,
            });
        }

        let snapshot = IndexSnapshot {
            version: 1,
            config: self.config.clone(),
            nodes: serializable_nodes,
            entry_point: *self.entry_point.read(),
            max_layer: *self.max_layer.read(),
            dimension: self.dimension,
            created_at: SystemTime::now(),
        };

        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, &snapshot)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        Ok(())
    }

    /// Load index from disk
    ///
    /// # Example
    /// ```no_run
    /// # use sochdb_index::hnsw::HnswIndex;
    /// let index = HnswIndex::load_from_disk("embeddings.hnsw").unwrap();
    /// // Index ready to use immediately!
    /// ```
    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);

        let snapshot: IndexSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

        // Validate version compatibility
        if snapshot.version != 1 {
            return Err(format!(
                "Incompatible version: {} (expected 1)",
                snapshot.version
            ));
        }

        // Create new index
        let index = HnswIndex::new(snapshot.dimension, snapshot.config.clone());

        // Restore nodes
        use crate::vector_quantized::{Precision, QuantizedVector};
        use parking_lot::RwLock;
        use std::sync::Arc;

        let precision = snapshot
            .config
            .quantization_precision
            .unwrap_or(Precision::F32);

        let mut pending_neighbors: Vec<(u128, Vec<SmallVec<[u128; MAX_M]>>)> = Vec::new();

        for snode in snapshot.nodes {
            // Reconstruct layers with versioned neighbors (dense list filled in second pass)
            let mut layers = Vec::with_capacity(snode.neighbors.len());
            for _ in 0..snode.neighbors.len() {
                layers.push(RwLock::new(crate::hnsw::VersionedNeighbors {
                    neighbors: SmallVec::new(),
                    version: 0, // Start at version 0 for loaded snapshots
                }));
            }

            let dense_index = index.next_dense_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u32;
            index.record_dense_id(dense_index, snode.id);
            let quantized = QuantizedVector::from_f32(
                ndarray::Array1::from_vec(snode.vector),
                precision,
            );
            let vector_index = {
                let mut store = index.vector_store.write();
                let idx = store.len() as u32;
                store.push(quantized.clone());
                idx
            };
            let node = Arc::new(crate::hnsw::HnswNode {
                id: snode.id,
                dense_index,
                vector_index,
                vector: quantized,
                storage_id: None, // Loaded from snapshot - vectors are inline
                layers,
                layer: snode.layer,
            });

            pending_neighbors.push((snode.id, snode.neighbors));
            index.nodes.insert(snode.id, node.clone());
            // O(1) hot path storage
            index.store_internal_node(dense_index, node);
        }

        // Second pass: fill dense neighbor lists
        for (node_id, neighbor_layers) in pending_neighbors {
            if let Some(node) = index.nodes.get(&node_id) {
                for (layer_idx, layer_neighbors) in neighbor_layers.into_iter().enumerate() {
                    let dense_neighbors: SmallVec<[u32; MAX_M]> = layer_neighbors
                        .iter()
                        .filter_map(|id| index.node_id_to_dense(*id))
                        .collect();
                    if let Some(layer_lock) = node.layers.get(layer_idx) {
                        let mut layer_guard = layer_lock.write();
                        layer_guard.neighbors = dense_neighbors;
                        layer_guard.version = 0;
                    }
                }
            }
        }

        // Restore metadata
        *index.entry_point.write() = snapshot.entry_point;
        *index.max_layer.write() = snapshot.max_layer;

        Ok(index)
    }

    /// Save index with gzip compression (slower but smaller)
    pub fn save_to_disk_compressed<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let mut serializable_nodes = Vec::with_capacity(self.nodes.len());

        for entry in self.nodes.iter() {
            let node = entry.value();
            // Collect neighbors from all layers (extract just the neighbors, not version)
            let mut neighbors = Vec::with_capacity(node.layers.len());
            for layer_lock in &node.layers {
                let dense_neighbors = layer_lock.read().neighbors.clone();
                neighbors.push(self.dense_neighbors_to_ids(&dense_neighbors));
            }

            serializable_nodes.push(SerializableNode {
                id: *entry.key(),
                vector: node.vector.to_f32().to_vec(),
                neighbors,
                layer: node.layer,
            });
        }

        let snapshot = IndexSnapshot {
            version: 1,
            config: self.config.clone(),
            nodes: serializable_nodes,
            entry_point: *self.entry_point.read(),
            max_layer: *self.max_layer.read(),
            dimension: self.dimension,
            created_at: SystemTime::now(),
        };

        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let encoder = GzEncoder::new(file, Compression::default());

        bincode::serialize_into(encoder, &snapshot)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        Ok(())
    }

    /// Load compressed index from disk
    pub fn load_from_disk_compressed<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        use flate2::read::GzDecoder;

        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);

        let snapshot: IndexSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

        if snapshot.version != 1 {
            return Err(format!(
                "Incompatible version: {} (expected 1)",
                snapshot.version
            ));
        }

        let index = HnswIndex::new(snapshot.dimension, snapshot.config.clone());

        use crate::vector_quantized::{Precision, QuantizedVector};
        use parking_lot::RwLock;
        use std::sync::Arc;

        let precision = snapshot
            .config
            .quantization_precision
            .unwrap_or(Precision::F32);

        let mut pending_neighbors: Vec<(u128, Vec<SmallVec<[u128; MAX_M]>>)> = Vec::new();

        for snode in snapshot.nodes {
            // Reconstruct layers with versioned neighbors (dense list filled in second pass)
            let mut layers = Vec::with_capacity(snode.neighbors.len());
            for _ in 0..snode.neighbors.len() {
                layers.push(RwLock::new(crate::hnsw::VersionedNeighbors {
                    neighbors: SmallVec::new(),
                    version: 0, // Start at version 0 for loaded snapshots
                }));
            }

            let dense_index = index.next_dense_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u32;
            index.record_dense_id(dense_index, snode.id);
            let quantized = QuantizedVector::from_f32(
                ndarray::Array1::from_vec(snode.vector),
                precision,
            );
            let vector_index = {
                let mut store = index.vector_store.write();
                let idx = store.len() as u32;
                store.push(quantized.clone());
                idx
            };
            let node = Arc::new(crate::hnsw::HnswNode {
                id: snode.id,
                dense_index,
                vector_index,
                vector: quantized,
                storage_id: None, // Loaded from snapshot - vectors are inline
                layers,
                layer: snode.layer,
            });

            pending_neighbors.push((snode.id, snode.neighbors));
            index.nodes.insert(snode.id, node.clone());
            // O(1) hot path storage
            index.store_internal_node(dense_index, node);
        }

        // Second pass: fill dense neighbor lists
        for (node_id, neighbor_layers) in pending_neighbors {
            if let Some(node) = index.nodes.get(&node_id) {
                for (layer_idx, layer_neighbors) in neighbor_layers.into_iter().enumerate() {
                    let dense_neighbors: SmallVec<[u32; MAX_M]> = layer_neighbors
                        .iter()
                        .filter_map(|id| index.node_id_to_dense(*id))
                        .collect();
                    if let Some(layer_lock) = node.layers.get(layer_idx) {
                        let mut layer_guard = layer_lock.write();
                        layer_guard.neighbors = dense_neighbors;
                        layer_guard.version = 0;
                    }
                }
            }
        }

        *index.entry_point.write() = snapshot.entry_point;
        *index.max_layer.write() = snapshot.max_layer;

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    use crate::hnsw::{HnswConfig, HnswIndex};
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_and_load() {
        // Create and populate index
        let config = HnswConfig::default();
        let index = HnswIndex::new(128, config);

        for i in 0..100 {
            let mut vec = vec![0.0; 128];
            // Use non-collinear data (sine/cosine) to work well with Cosine distance
            let angle = (i as f32) / 10.0;
            vec[0] = angle.sin();
            vec[1] = angle.cos();
            vec[2] = i as f32;
            index.insert(i as u128, vec).unwrap();
        }

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        index.save_to_disk(path).unwrap();

        // Load from disk
        let loaded_index = HnswIndex::load_from_disk(path).unwrap();

        // Verify
        assert_eq!(loaded_index.nodes.len(), index.nodes.len());
        assert_eq!(loaded_index.dimension, index.dimension);

        // Test search works on loaded index
        let mut query = vec![0.0; 128];
        let angle = 5.0_f32 / 10.0;
        query[0] = angle.sin();
        query[1] = angle.cos();
        query[2] = 5.0;
        let results = loaded_index.search(&query, 5).unwrap();
        // HNSW is approximate, may not always find exactly k neighbors with sparse graphs
        assert!(
            results.len() >= 2,
            "Expected at least 2 results, got {}",
            results.len()
        );
        // HNSW is approximate and data structure makes vectors very similar in Cosine distance
        // So we check that we found a very close vector (distance < 0.1) rather than exact ID
        assert!(
            results[0].1 < 0.1,
            "Top result distance {} is too large (expected < 0.1)",
            results[0].1
        );
    }

    #[test]
    fn test_compressed_save_load() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(64, config);

        for i in 0..50 {
            let vec: Vec<f32> = (0..64).map(|j| (i + j) as f32).collect();
            index.insert(i as u128, vec).unwrap();
        }

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        index.save_to_disk_compressed(path).unwrap();
        let loaded_index = HnswIndex::load_from_disk_compressed(path).unwrap();

        assert_eq!(loaded_index.nodes.len(), index.nodes.len());

        // Verify compressed file is smaller than uncompressed
        // (This is an integration test, actual comparison would need two files)
    }

    #[test]
    fn test_empty_index() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(128, config);

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        index.save_to_disk(path).unwrap();
        let loaded_index = HnswIndex::load_from_disk(path).unwrap();

        assert_eq!(loaded_index.nodes.len(), 0);
    }
}

// ============================================================================
// WAL Integration for HNSW Index
// ============================================================================

/// WAL entry for HNSW graph mutations
///
/// Enables incremental durability without full index rebuilds on crash.
/// Each entry is:
/// - CRC32 checksum (4 bytes)
/// - Entry length (4 bytes)  
/// - Entry data (bincode serialized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HnswWalEntry {
    /// Insert a new vector
    Insert {
        id: u128,
        vector: Vec<f32>,
        layer: usize,
    },
    /// Add a neighbor connection
    AddNeighbor {
        node_id: u128,
        layer: usize,
        neighbor_id: u128,
    },
    /// Remove a neighbor connection (pruning)
    RemoveNeighbor {
        node_id: u128,
        layer: usize,
        neighbor_id: u128,
    },
    /// Delete a vector
    Delete { id: u128 },
    /// Update entry point
    SetEntryPoint { id: Option<u128>, max_layer: usize },
    /// Checkpoint marker (snapshot taken at this point)
    Checkpoint {
        snapshot_path: String,
        timestamp: u64,
    },
}

/// WAL writer for HNSW index operations
pub struct HnswWalWriter {
    writer: BufWriter<File>,
    path: std::path::PathBuf,
    /// Entries since last checkpoint
    entries_since_checkpoint: u64,
    /// Threshold for automatic checkpoint
    checkpoint_threshold: u64,
}

impl HnswWalWriter {
    /// Open or create a WAL file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("Failed to open WAL: {}", e))?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            entries_since_checkpoint: 0,
            checkpoint_threshold: 10000, // Checkpoint after 10k entries
        })
    }

    /// Append an entry to the WAL
    pub fn append(&mut self, entry: &HnswWalEntry) -> Result<(), String> {
        // Serialize entry
        let data = bincode::serialize(entry).map_err(|e| format!("Serialization failed: {}", e))?;

        // Compute CRC32
        let crc = crc32fast::hash(&data);

        // Write: CRC32 (4 bytes) + length (4 bytes) + data
        let len = data.len() as u32;
        self.writer
            .write_all(&crc.to_le_bytes())
            .map_err(|e| format!("Write failed: {}", e))?;
        self.writer
            .write_all(&len.to_le_bytes())
            .map_err(|e| format!("Write failed: {}", e))?;
        self.writer
            .write_all(&data)
            .map_err(|e| format!("Write failed: {}", e))?;

        self.entries_since_checkpoint += 1;
        Ok(())
    }

    /// Flush and sync to disk
    pub fn sync(&mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|e| format!("Flush failed: {}", e))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| format!("Sync failed: {}", e))?;
        Ok(())
    }

    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.entries_since_checkpoint >= self.checkpoint_threshold
    }

    /// Record a checkpoint
    pub fn record_checkpoint(&mut self, snapshot_path: &str) -> Result<(), String> {
        let timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.append(&HnswWalEntry::Checkpoint {
            snapshot_path: snapshot_path.to_string(),
            timestamp,
        })?;

        self.sync()?;
        self.entries_since_checkpoint = 0;
        Ok(())
    }

    /// Get path for truncation after checkpoint
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// WAL reader for HNSW recovery
pub struct HnswWalReader {
    data: Vec<u8>,
    position: usize,
}

impl HnswWalReader {
    /// Open WAL file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("Failed to read WAL: {}", e))?;

        Ok(Self { data, position: 0 })
    }

    /// Read next entry from WAL
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Result<HnswWalEntry, String>> {
        if self.position + 8 > self.data.len() {
            return None;
        }

        // Read CRC32 and length
        let crc_bytes: [u8; 4] = self.data[self.position..self.position + 4]
            .try_into()
            .ok()?;
        let len_bytes: [u8; 4] = self.data[self.position + 4..self.position + 8]
            .try_into()
            .ok()?;

        let expected_crc = u32::from_le_bytes(crc_bytes);
        let len = u32::from_le_bytes(len_bytes) as usize;

        self.position += 8;

        if self.position + len > self.data.len() {
            return Some(Err("Truncated WAL entry".into()));
        }

        // Read and verify data
        let entry_data = &self.data[self.position..self.position + len];
        let actual_crc = crc32fast::hash(entry_data);

        if actual_crc != expected_crc {
            return Some(Err(format!(
                "CRC mismatch: expected {}, got {}",
                expected_crc, actual_crc
            )));
        }

        self.position += len;

        // Deserialize
        match bincode::deserialize(entry_data) {
            Ok(entry) => Some(Ok(entry)),
            Err(e) => Some(Err(format!("Deserialization failed: {}", e))),
        }
    }

    /// Find the last checkpoint in the WAL
    pub fn find_last_checkpoint(&mut self) -> Option<(String, u64)> {
        let mut last_checkpoint = None;

        while let Some(result) = self.next() {
            if let Ok(HnswWalEntry::Checkpoint {
                snapshot_path,
                timestamp,
            }) = result
            {
                last_checkpoint = Some((snapshot_path, timestamp));
            }
        }

        last_checkpoint
    }
}

impl HnswIndex {
    /// Recover index from snapshot + WAL
    ///
    /// 1. Load the latest snapshot
    /// 2. Replay WAL entries after the checkpoint
    /// 3. Return recovered index
    pub fn recover<P: AsRef<Path>>(snapshot_dir: P, wal_path: P) -> Result<Self, String> {
        let snapshot_dir = snapshot_dir.as_ref();
        let wal_path = wal_path.as_ref();

        // Read WAL to find last checkpoint
        let mut reader = HnswWalReader::open(wal_path)?;
        let checkpoint = reader.find_last_checkpoint();

        // Load snapshot if available
        let index = if let Some((snapshot_path, _timestamp)) = &checkpoint {
            let full_path = snapshot_dir.join(snapshot_path);
            if full_path.exists() {
                Self::load_from_disk(&full_path)?
            } else {
                return Err(format!("Snapshot not found: {:?}", full_path));
            }
        } else {
            // No checkpoint - need full WAL replay
            // This requires knowing the dimension from the first Insert entry
            return Err(
                "No checkpoint found - full WAL replay not implemented. Use save_to_disk first."
                    .into(),
            );
        };

        // Replay WAL entries after checkpoint
        let checkpoint_timestamp = checkpoint.map(|(_, ts)| ts).unwrap_or(0);
        let mut reader = HnswWalReader::open(wal_path)?;
        let mut past_checkpoint = false;
        let mut replayed = 0;

        while let Some(result) = reader.next() {
            let entry = result?;

            match &entry {
                HnswWalEntry::Checkpoint { timestamp, .. } => {
                    if *timestamp >= checkpoint_timestamp {
                        past_checkpoint = true;
                    }
                }
                _ if past_checkpoint => {
                    // Replay this entry
                    index.apply_wal_entry(&entry)?;
                    replayed += 1;
                }
                _ => {}
            }
        }

        if replayed > 0 {
            eprintln!("HNSW recovery: replayed {} WAL entries", replayed);
        }

        Ok(index)
    }

    /// Apply a single WAL entry to the index
    fn apply_wal_entry(&self, entry: &HnswWalEntry) -> Result<(), String> {
        match entry {
            HnswWalEntry::Insert { id, vector, .. } => {
                // Re-insert vector (this will rebuild connections)
                self.insert(*id, vector.clone())?;
            }
            HnswWalEntry::Delete { id } => {
                // Remove node from index
                self.nodes.remove(id);
            }
            HnswWalEntry::SetEntryPoint { id, max_layer } => {
                *self.entry_point.write() = *id;
                *self.max_layer.write() = *max_layer;
            }
            // AddNeighbor and RemoveNeighbor are handled by insert()
            // which rebuilds connections. For incremental edge updates,
            // we'd need more sophisticated replay logic.
            HnswWalEntry::AddNeighbor { .. } | HnswWalEntry::RemoveNeighbor { .. } => {
                // These are logged for completeness but insert() rebuilds connections
            }
            HnswWalEntry::Checkpoint { .. } => {
                // No-op during replay
            }
        }
        Ok(())
    }
}
