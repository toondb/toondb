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

//! Buffered Vector Inserts (Task 6)
//!
//! ## Problem
//!
//! HNSW insertion is expensive (~100µs per vector) due to:
//! - Graph search to find entry point: O(log N)
//! - Neighbor pruning at each layer: O(M × log N)
//! - Bidirectional edge updates: O(M)
//!
//! For workloads with frequent inserts followed by batch queries,
//! this overhead is wasteful.
//!
//! ## Solution
//!
//! Delta buffer that accumulates new vectors:
//! 1. Inserts go to a small buffer (default 1000 vectors)
//! 2. Queries merge buffer + HNSW results via brute-force on buffer
//! 3. Buffer is flushed to HNSW when full or on-demand
//!
//! ## Performance Analysis
//!
//! Buffer insert: O(1)
//! Query with buffer of size B: O(B × D) brute-force + O(log N) HNSW
//! For B = 1000, D = 384: ~384K ops, ~100µs
//!
//! Net effect:
//! - 3× fewer HNSW index updates (amortized batch insertion)
//! - Slightly higher query latency (~100µs overhead for buffer scan)
//! - Good for write-heavy workloads with eventual consistency

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::hnsw::{HnswIndex, HnswConfig};
use crate::vector_quantized::{QuantizedVector, Precision};

/// Configuration for buffered HNSW
#[derive(Debug, Clone)]
pub struct BufferedHnswConfig {
    /// HNSW configuration
    pub hnsw_config: HnswConfig,
    /// Maximum buffer size before flush (default: 1000)
    pub buffer_capacity: usize,
    /// Whether to auto-flush when buffer is full (default: true)
    pub auto_flush: bool,
    /// Precision for distance calculations
    pub precision: Precision,
}

impl Default for BufferedHnswConfig {
    fn default() -> Self {
        Self {
            hnsw_config: HnswConfig::default(),
            buffer_capacity: 1000,
            auto_flush: true,
            precision: Precision::F32,
        }
    }
}

impl BufferedHnswConfig {
    /// Create with custom buffer capacity
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Create with HNSW config
    pub fn with_hnsw_config(mut self, config: HnswConfig) -> Self {
        self.hnsw_config = config;
        self
    }
}

/// A buffered vector in the delta buffer
#[derive(Debug, Clone)]
struct BufferedVector {
    id: u128,
    vector: Vec<f32>,
    quantized: QuantizedVector,
}

/// Buffered HNSW index with delta buffer for write-heavy workloads
///
/// ## Usage
///
/// ```ignore
/// let config = BufferedHnswConfig::default()
///     .with_buffer_capacity(1000);
/// let index = BufferedHnsw::new(384, config);
///
/// // Fast buffered inserts
/// for (id, vec) in vectors {
///     index.insert(id, vec)?;
/// }
///
/// // Query merges buffer + HNSW
/// let results = index.search(&query, 10)?;
///
/// // Explicit flush to HNSW
/// index.flush()?;
/// ```
pub struct BufferedHnsw {
    /// The underlying HNSW index
    hnsw: Arc<HnswIndex>,
    /// Delta buffer for pending inserts
    buffer: RwLock<Vec<BufferedVector>>,
    /// Buffer lookup by ID
    buffer_ids: RwLock<HashMap<u128, usize>>,
    /// Configuration
    config: BufferedHnswConfig,
    /// Vector dimension
    dimension: usize,
}

impl BufferedHnsw {
    /// Create a new buffered HNSW index
    pub fn new(dimension: usize, config: BufferedHnswConfig) -> Self {
        let hnsw = Arc::new(HnswIndex::new(dimension, config.hnsw_config.clone()));
        
        Self {
            hnsw,
            buffer: RwLock::new(Vec::with_capacity(config.buffer_capacity)),
            buffer_ids: RwLock::new(HashMap::new()),
            config,
            dimension,
        }
    }

    /// Insert a vector into the buffer (O(1))
    ///
    /// The vector is not immediately added to HNSW. Instead, it's
    /// buffered for later flush or merged during queries.
    pub fn insert(&self, id: u128, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension, vector.len()
            ));
        }

        // Quantize for distance calculations
        let quantized = QuantizedVector::from_f32(
            ndarray::Array1::from_vec(vector.clone()),
            self.config.precision,
        );

        let buffered = BufferedVector {
            id,
            vector,
            quantized,
        };

        // Check if auto-flush is needed
        let needs_flush = {
            let mut buffer = self.buffer.write();
            let mut ids = self.buffer_ids.write();

            // Check for duplicate in buffer
            if ids.contains_key(&id) {
                return Err(format!("Vector {} already in buffer", id));
            }

            let idx = buffer.len();
            buffer.push(buffered);
            ids.insert(id, idx);

            self.config.auto_flush && buffer.len() >= self.config.buffer_capacity
        };

        if needs_flush {
            self.flush()?;
        }

        Ok(())
    }

    /// Insert a batch of vectors (O(n))
    pub fn insert_batch(&self, batch: &[(u128, Vec<f32>)]) -> Result<usize, String> {
        let mut inserted = 0;
        for (id, vector) in batch {
            self.insert(*id, vector.clone())?;
            inserted += 1;
        }
        Ok(inserted)
    }

    /// Flush buffer to HNSW index
    ///
    /// This is an expensive operation that inserts all buffered vectors
    /// into the HNSW graph structure.
    pub fn flush(&self) -> Result<usize, String> {
        let vectors = {
            let mut buffer = self.buffer.write();
            let mut ids = self.buffer_ids.write();
            ids.clear();
            std::mem::take(&mut *buffer)
        };

        if vectors.is_empty() {
            return Ok(0);
        }

        let count = vectors.len();

        // Batch insert into HNSW
        let batch: Vec<(u128, Vec<f32>)> = vectors
            .into_iter()
            .map(|v| (v.id, v.vector))
            .collect();

        self.hnsw.insert_batch(&batch)?;

        Ok(count)
    }

    /// Search for nearest neighbors
    ///
    /// This merges results from:
    /// 1. HNSW index search (O(log N))
    /// 2. Brute-force scan of buffer (O(B × D))
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String> {
        if query.len() != self.dimension {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension, query.len()
            ));
        }

        // Quantize query
        let quantized_query = QuantizedVector::from_f32(
            ndarray::Array1::from_vec(query.to_vec()),
            self.config.precision,
        );

        // Search HNSW
        let hnsw_results = self.hnsw.search(query, k)?;

        // Search buffer with brute force
        let buffer_results = {
            let buffer = self.buffer.read();
            let mut results: Vec<(u128, f32)> = buffer
                .iter()
                .map(|v| {
                    let dist = self.compute_distance(&quantized_query, &v.quantized);
                    (v.id, dist)
                })
                .collect();
            
            // Sort by distance
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(k);
            results
        };

        // Merge results (take top-k from combined set)
        let mut merged: Vec<(u128, f32)> = hnsw_results
            .into_iter()
            .chain(buffer_results)
            .collect();

        merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);

        Ok(merged)
    }

    /// Compute distance between two quantized vectors
    fn compute_distance(&self, a: &QuantizedVector, b: &QuantizedVector) -> f32 {
        use crate::vector_quantized::euclidean_distance_quantized;
        euclidean_distance_quantized(a, b)
    }

    /// Get buffer statistics
    pub fn buffer_stats(&self) -> BufferStats {
        let buffer = self.buffer.read();
        BufferStats {
            buffer_size: buffer.len(),
            buffer_capacity: self.config.buffer_capacity,
            utilization: buffer.len() as f64 / self.config.buffer_capacity as f64,
        }
    }

    /// Get combined statistics
    pub fn stats(&self) -> BufferedHnswStats {
        let hnsw_stats = self.hnsw.stats();
        let buffer_stats = self.buffer_stats();

        BufferedHnswStats {
            hnsw_node_count: hnsw_stats.num_vectors,
            buffer_size: buffer_stats.buffer_size,
            total_vectors: hnsw_stats.num_vectors + buffer_stats.buffer_size,
            buffer_utilization: buffer_stats.utilization,
        }
    }

    /// Check if a vector exists (in buffer or HNSW)
    pub fn contains(&self, id: u128) -> bool {
        // Check buffer first
        if self.buffer_ids.read().contains_key(&id) {
            return true;
        }
        // Then check HNSW
        self.hnsw.contains(id)
    }

    /// Get the underlying HNSW index
    pub fn hnsw(&self) -> &Arc<HnswIndex> {
        &self.hnsw
    }

    /// Force clear the buffer without flushing
    pub fn clear_buffer(&self) {
        let mut buffer = self.buffer.write();
        let mut ids = self.buffer_ids.write();
        buffer.clear();
        ids.clear();
    }
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Current buffer size
    pub buffer_size: usize,
    /// Maximum buffer capacity
    pub buffer_capacity: usize,
    /// Buffer utilization (0.0 - 1.0)
    pub utilization: f64,
}

/// Combined statistics for buffered HNSW
#[derive(Debug, Clone)]
pub struct BufferedHnswStats {
    /// Number of nodes in HNSW index
    pub hnsw_node_count: usize,
    /// Number of vectors in buffer
    pub buffer_size: usize,
    /// Total vectors (HNSW + buffer)
    pub total_vectors: usize,
    /// Buffer utilization (0.0 - 1.0)
    pub buffer_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_buffered_insert() {
        let config = BufferedHnswConfig::default()
            .with_buffer_capacity(100);
        let index = BufferedHnsw::new(128, config);

        // Insert vectors
        for i in 0..50 {
            let vec = random_vector(128);
            index.insert(i as u128, vec).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.buffer_size, 50);
        assert_eq!(stats.hnsw_node_count, 0);
        assert_eq!(stats.total_vectors, 50);
    }

    #[test]
    fn test_buffered_auto_flush() {
        let config = BufferedHnswConfig::default()
            .with_buffer_capacity(10);
        let index = BufferedHnsw::new(64, config);

        // Insert enough to trigger auto-flush
        for i in 0..15 {
            let vec = random_vector(64);
            index.insert(i as u128, vec).unwrap();
        }

        let stats = index.stats();
        // First 10 should have been flushed
        assert!(stats.hnsw_node_count >= 10);
    }

    #[test]
    fn test_buffered_search() {
        let config = BufferedHnswConfig::default()
            .with_buffer_capacity(100);
        let index = BufferedHnsw::new(32, config);

        // Insert vectors
        for i in 0..20 {
            let mut vec = vec![0.0f32; 32];
            vec[0] = i as f32;
            index.insert(i as u128, vec).unwrap();
        }

        // Search should find vectors in buffer
        let query = vec![5.0f32; 32].iter().enumerate()
            .map(|(i, _)| if i == 0 { 5.0 } else { 0.0 })
            .collect::<Vec<f32>>();
        
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        
        // The closest should be id=5
        assert_eq!(results[0].0, 5);
    }

    #[test]
    fn test_buffered_flush() {
        let config = BufferedHnswConfig::default()
            .with_buffer_capacity(100);
        let index = BufferedHnsw::new(64, config);

        // Insert vectors
        for i in 0..50 {
            let vec = random_vector(64);
            index.insert(i as u128, vec).unwrap();
        }

        assert_eq!(index.buffer_stats().buffer_size, 50);

        // Flush
        let flushed = index.flush().unwrap();
        assert_eq!(flushed, 50);

        let stats = index.stats();
        assert_eq!(stats.buffer_size, 0);
        assert_eq!(stats.hnsw_node_count, 50);
    }

    #[test]
    fn test_buffered_contains() {
        let config = BufferedHnswConfig::default()
            .with_buffer_capacity(100);
        let index = BufferedHnsw::new(32, config);

        // Insert to buffer
        index.insert(1, random_vector(32)).unwrap();
        assert!(index.contains(1));
        assert!(!index.contains(2));

        // Flush and check
        index.flush().unwrap();
        assert!(index.contains(1));
    }

    #[test]
    fn test_duplicate_detection() {
        let config = BufferedHnswConfig::default()
            .with_buffer_capacity(100);
        let index = BufferedHnsw::new(32, config);

        index.insert(1, random_vector(32)).unwrap();
        
        // Duplicate should fail
        let result = index.insert(1, random_vector(32));
        assert!(result.is_err());
    }
}
