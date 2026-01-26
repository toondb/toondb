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

//! Vector index for semantic search
//!
//! Simplified HNSW-like implementation for nearest neighbor search.

use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use parking_lot::RwLock;
use ndarray::Array1;

/// Vector embedding (fixed 768 dimensions for typical LLM embeddings)
pub type Embedding = Array1<f32>;

/// Vector index entry
#[derive(Clone)]
struct VectorEntry {
    edge_id: u128,
    vector: Embedding,
}

/// Distance metric
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Simple brute-force vector index for semantic search
///
/// **Performance Characteristics:**
/// - Search: O(N) linear scan over all vectors
/// - Space: O(N * D) where D is embedding dimension
///
/// This is intentionally a simple implementation for the MVP. For production
/// workloads with >100K vectors, this should be upgraded to an approximate
/// nearest neighbor (ANN) index like HNSW, which provides O(log N) search.
///
/// The API is designed to make this upgrade transparent - just swap the
/// implementation without changing callers.
pub struct VectorIndex {
    entries: RwLock<Vec<VectorEntry>>,
    metric: DistanceMetric,
    expected_dim: Option<usize>,  // Expected vector dimensionality
}

impl VectorIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            metric,
            expected_dim: None,  // Will be set on first add()
        }
    }

    /// Create a new index with a specific expected dimension
    pub fn with_dimension(metric: DistanceMetric, dim: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            metric,
            expected_dim: Some(dim),
        }
    }

    /// Add a vector for an edge
    ///
    /// Returns an error if the vector dimension doesn't match expected dimension.
    pub fn add(&self, edge_id: u128, vector: Embedding) -> Result<(), String> {
        let dim = vector.len();

        // Check dimension consistency
        if let Some(expected) = self.expected_dim {
            if dim != expected {
                return Err(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    expected, dim
                ));
            }
        }

        self.entries.write().push(VectorEntry {
            edge_id,
            vector,
        });

        Ok(())
    }

    /// Find k nearest neighbors
    ///
    /// Returns an error if query dimension doesn't match index dimension.
    pub fn search(&self, query: &Embedding, k: usize) -> Result<Vec<(u128, f32)>, String> {
        let entries = self.entries.read();

        // Check query dimension
        if let Some(expected) = self.expected_dim {
            if query.len() != expected {
                return Err(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    expected,
                    query.len()
                ));
            }
        }

        if entries.is_empty() {
            return Ok(Vec::new());
        }

        // Compute distances to all vectors
        let mut distances: Vec<_> = entries
            .iter()
            .map(|entry| {
                let dist = self.distance(query, &entry.vector);
                (entry.edge_id, dist)
            })
            .collect();

        // Sort by distance (ascending for Euclidean, descending for cosine/dot)
        match self.metric {
            DistanceMetric::Euclidean => {
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
            DistanceMetric::Cosine | DistanceMetric::DotProduct => {
                distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
        }

        // Return top k
        distances.truncate(k);
        Ok(distances)
    }

    /// Compute distance between two vectors
    fn distance(&self, a: &Embedding, b: &Embedding) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let dot = a.dot(b);
                let norm_a = a.dot(a).sqrt();
                let norm_b = b.dot(b).sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot / (norm_a * norm_b)
                }
            }
            DistanceMetric::Euclidean => {
                let diff = a - b;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::DotProduct => a.dot(b),
        }
    }

    /// Get number of indexed vectors
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear all vectors
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Save vector index to disk
    ///
    /// **Persistence Format:**
    /// ```text
    /// [magic: "CHRL_VEC" 8 bytes]
    /// [version: u32 = 1]
    /// [metric: u8] (0=Cosine, 1=Euclidean, 2=DotProduct)
    /// [dimension: u32]
    /// [count: u64]
    /// [entry_1: edge_id (16 bytes) + vector (dimension * 4 bytes)]
    /// [entry_2: ...]
    /// ...
    /// ```
    ///
    /// TODO(Task 8): Add SSTable-like format for large indexes with:
    /// - Block compression (LZ4)
    /// - Index blocks for binary search
    /// - Bloom filters for existence checks
    /// - Hot/warm/cold tiering based on access patterns
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let entries = self.entries.read();
        let mut file = BufWriter::new(File::create(path)?);
        
        // Magic header
        file.write_all(b"CHRL_VEC")?;
        
        // Version
        file.write_all(&1u32.to_le_bytes())?;
        
        // Metric
        let metric_byte = match self.metric {
            DistanceMetric::Cosine => 0u8,
            DistanceMetric::Euclidean => 1u8,
            DistanceMetric::DotProduct => 2u8,
        };
        file.write_all(&[metric_byte])?;
        
        // Dimension (use first entry or 0 if empty)
        let dimension = entries.first().map(|e| e.vector.len()).unwrap_or(0) as u32;
        file.write_all(&dimension.to_le_bytes())?;
        
        // Count
        file.write_all(&(entries.len() as u64).to_le_bytes())?;
        
        // Entries
        for entry in entries.iter() {
            // Edge ID (u128 = 16 bytes)
            file.write_all(&entry.edge_id.to_le_bytes())?;
            
            // Vector (f32 array)
            for &val in entry.vector.as_slice().unwrap() {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        
        file.flush()?;
        Ok(())
    }

    /// Load vector index from disk
    ///
    /// Returns a new VectorIndex populated with data from disk.
    /// If the file doesn't exist or is corrupt, returns an error.
    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        
        // Read and verify magic header
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != b"CHRL_VEC" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid vector index magic header"
            ));
        }
        
        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported vector index version: {}", version)
            ));
        }
        
        // Read metric
        let mut metric_byte = [0u8; 1];
        file.read_exact(&mut metric_byte)?;
        let metric = match metric_byte[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid metric byte: {}", metric_byte[0])
            )),
        };
        
        // Read dimension
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;
        
        // Read count
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
        // Read entries
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            // Edge ID
            let mut edge_id_bytes = [0u8; 16];
            file.read_exact(&mut edge_id_bytes)?;
            let edge_id = u128::from_le_bytes(edge_id_bytes);
            
            // Vector
            let mut vector_data = vec![0f32; dimension];
            for val in vector_data.iter_mut() {
                let mut f_bytes = [0u8; 4];
                file.read_exact(&mut f_bytes)?;
                *val = f32::from_le_bytes(f_bytes);
            }
            let vector = Array1::from_vec(vector_data);
            
            entries.push(VectorEntry { edge_id, vector });
        }
        
        Ok(Self {
            entries: RwLock::new(entries),
            metric,
            expected_dim: if dimension > 0 { Some(dimension) } else { None },
        })
    }

    /// Create vector index with persistence enabled
    ///
    /// If the index file exists, loads from disk. Otherwise creates a new index.
    /// Use this in SochDB::open() to enable automatic persistence.
    pub fn with_persistence<P: AsRef<Path>>(path: P, metric: DistanceMetric) -> Self {
        if path.as_ref().exists() {
            Self::load_from_disk(&path).unwrap_or_else(|e| {
                eprintln!("Warning: Failed to load vector index: {}. Starting fresh.", e);
                Self::new(metric)
            })
        } else {
            Self::new(metric)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_vector_index_cosine() {
        let index = VectorIndex::new(DistanceMetric::Cosine);

        // Add some vectors
        index.add(1, arr1(&[1.0, 0.0, 0.0])).unwrap();
        index.add(2, arr1(&[0.9, 0.1, 0.0])).unwrap(); // Similar to 1
        index.add(3, arr1(&[0.0, 1.0, 0.0])).unwrap(); // Orthogonal to 1

        // Search for vector similar to [1, 0, 0]
        let query = arr1(&[1.0, 0.0, 0.0]);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Exact match
        assert_eq!(results[1].0, 2); // Next closest
    }

    #[test]
    fn test_vector_index_euclidean() {
        let index = VectorIndex::new(DistanceMetric::Euclidean);

        index.add(1, arr1(&[0.0, 0.0])).unwrap();
        index.add(2, arr1(&[1.0, 1.0])).unwrap();
        index.add(3, arr1(&[10.0, 10.0])).unwrap();

        let query = arr1(&[0.5, 0.5]);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // Closest should be either 1 or 2
        assert!(results[0].0 == 1 || results[0].0 == 2);
    }

    #[test]
    fn test_vector_index_operations() {
        let index = VectorIndex::new(DistanceMetric::Cosine);

        assert!(index.is_empty());

        index.add(1, arr1(&[1.0, 0.0])).unwrap();
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());

        index.clear();
        assert!(index.is_empty());
    }

    #[test]
    fn test_dimension_validation() {
        let index = VectorIndex::with_dimension(DistanceMetric::Cosine, 3);

        // Add vector with correct dimension
        assert!(index.add(1, arr1(&[1.0, 0.0, 0.0])).is_ok());

        // Try to add vector with wrong dimension
        let result = index.add(2, arr1(&[1.0, 0.0]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));

        // Search with correct dimension
        let query = arr1(&[1.0, 0.0, 0.0]);
        assert!(index.search(&query, 1).is_ok());

        // Search with wrong dimension
        let wrong_query = arr1(&[1.0, 0.0]);
        let result = index.search(&wrong_query, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }
}
