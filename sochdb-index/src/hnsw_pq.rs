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

//! HNSW with Product Quantization Integration
//!
//! Provides 32x memory reduction for HNSW vector indices by using
//! Product Quantization for distance estimation during search.
//!
//! ## jj.md Task 11: PQ for HNSW
//!
//! Goals:
//! - 10-32x memory reduction for vector storage
//! - Support 10M+ vectors on commodity hardware
//! - Acceptable recall loss (<5% at 10x compression)
//!
//! ## Architecture
//!
//! ```text
//! Query Vector → ADC Distance Tables → Approximate K-NN
//!                                         ↓
//!                        Re-rank top 2*K with exact distances
//!                                         ↓
//!                                     Final K-NN
//! ```
//!
//! ## Memory Comparison (768-dim vectors, 1M vectors)
//!
//! - Full F32: 768 × 4 × 1M = 3GB
//! - With PQ:  96 bytes × 1M = 96MB (32x reduction!)
//!
//! Reference: "Product Quantization for Nearest Neighbor Search" (Jégou et al., 2011)

use crate::product_quantization::{PQCodebooks, PQCodes};
use dashmap::DashMap;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Asymmetric Distance Computation (ADC) lookup tables
///
/// Precomputed distances from query subvectors to all centroids.
/// Used for O(1) distance lookups during PQ search.
pub struct ADCTable {
    /// Distance tables: [subspace][centroid_idx] -> distance
    tables: Vec<Vec<f32>>,
    /// Number of subspaces
    n_subspaces: usize,
}

impl ADCTable {
    /// Create ADC lookup table for a query vector
    ///
    /// Precomputes distances from each query subvector to all centroids.
    /// This allows O(n_subspaces) distance computation instead of O(dim).
    pub fn new(query: &Array1<f32>, codebooks: &PQCodebooks) -> Self {
        let n_subspaces = codebooks.n_subspaces;
        let subdim = codebooks.subdim;
        let mut tables = Vec::with_capacity(n_subspaces);

        for (subspace_idx, subspace_centroids) in codebooks.centroids.iter().enumerate() {
            let start = subspace_idx * subdim;
            let end = start + subdim;
            let query_sub = query.slice(ndarray::s![start..end]);

            // Compute distance from query subvector to each centroid
            let distances: Vec<f32> = subspace_centroids
                .iter()
                .map(|centroid| {
                    // L2 distance (squared)
                    query_sub
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum()
                })
                .collect();

            tables.push(distances);
        }

        Self {
            tables,
            n_subspaces,
        }
    }

    /// Compute approximate distance to a PQ-encoded vector
    ///
    /// Uses table lookups instead of full distance computation.
    /// O(n_subspaces) = O(48) for 384-dim vectors
    #[inline]
    pub fn distance(&self, codes: &PQCodes) -> f32 {
        codes
            .codes
            .iter()
            .enumerate()
            .map(|(subspace, &code)| self.tables[subspace][code as usize])
            .sum()
    }

    /// Get the number of subspaces
    pub fn n_subspaces(&self) -> usize {
        self.n_subspaces
    }
}

/// PQ-accelerated HNSW search result
#[derive(Debug, Clone)]
pub struct PQSearchResult {
    /// Vector ID
    pub id: u128,
    /// Approximate distance (from PQ)
    pub approx_distance: f32,
    /// Exact distance (if re-ranked)
    pub exact_distance: Option<f32>,
}

impl PartialEq for PQSearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.approx_distance == other.approx_distance
    }
}

impl Eq for PQSearchResult {}

impl PartialOrd for PQSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PQSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        other
            .approx_distance
            .partial_cmp(&self.approx_distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Configuration for PQ-accelerated search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQSearchConfig {
    /// Number of candidates to consider before re-ranking
    pub ef_search: usize,
    /// Factor for re-ranking (re-rank top K * rerank_factor results)
    pub rerank_factor: usize,
    /// Whether to use asymmetric distance computation
    pub use_adc: bool,
}

impl Default for PQSearchConfig {
    fn default() -> Self {
        Self {
            ef_search: 100,
            rerank_factor: 2,
            use_adc: true,
        }
    }
}

/// Storage for PQ-encoded vectors
///
/// Stores only the PQ codes (e.g., 48 bytes per vector) instead of
/// full vectors (e.g., 3KB per 768-dim vector).
pub struct PQVectorStore {
    /// PQ-encoded vectors: id -> codes
    codes: DashMap<u128, PQCodes>,
    /// Trained codebooks
    codebooks: Arc<PQCodebooks>,
    /// Number of vectors stored
    count: std::sync::atomic::AtomicU64,
}

impl PQVectorStore {
    /// Create a new PQ vector store with trained codebooks
    pub fn new(codebooks: PQCodebooks) -> Self {
        Self {
            codes: DashMap::new(),
            codebooks: Arc::new(codebooks),
            count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Train codebooks from sample vectors and create a store
    pub fn train_and_create(samples: &[Array1<f32>], n_iter: usize, subdim: usize) -> Self {
        let codebooks = PQCodebooks::train(samples, n_iter, subdim);
        Self::new(codebooks)
    }

    /// Add a vector to the store
    pub fn add(&self, id: u128, vector: &Array1<f32>) {
        let codes = self.codebooks.encode(vector);
        self.codes.insert(id, codes);
        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get PQ codes for a vector
    pub fn get(&self, id: u128) -> Option<PQCodes> {
        self.codes.get(&id).map(|r| r.clone())
    }

    /// Remove a vector from the store
    pub fn remove(&self, id: u128) -> bool {
        if self.codes.remove(&id).is_some() {
            self.count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get the codebooks
    pub fn codebooks(&self) -> &PQCodebooks {
        &self.codebooks
    }

    /// Get the number of vectors stored
    pub fn len(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        let codes_size = self.len() * self.codebooks.n_subspaces;
        let codebooks_size = self.codebooks.n_subspaces * 256 * self.codebooks.subdim * 4;
        codes_size + codebooks_size
    }

    /// Create ADC table for a query
    pub fn create_adc_table(&self, query: &Array1<f32>) -> ADCTable {
        ADCTable::new(query, &self.codebooks)
    }

    /// Search for nearest neighbors using ADC
    ///
    /// Returns candidates sorted by approximate distance.
    /// Should be followed by re-ranking with exact distances.
    pub fn search_adc(&self, query: &Array1<f32>, k: usize, ef: usize) -> Vec<PQSearchResult> {
        let adc = self.create_adc_table(query);
        let mut heap = BinaryHeap::new();

        // Score all vectors
        for entry in self.codes.iter() {
            let id = *entry.key();
            let codes = entry.value();
            let distance = adc.distance(codes);

            let result = PQSearchResult {
                id,
                approx_distance: distance,
                exact_distance: None,
            };

            if heap.len() < ef {
                heap.push(std::cmp::Reverse(result));
            } else if let Some(std::cmp::Reverse(worst)) = heap.peek()
                && distance < worst.approx_distance
            {
                heap.pop();
                heap.push(std::cmp::Reverse(PQSearchResult {
                    id,
                    approx_distance: distance,
                    exact_distance: None,
                }));
            }
        }

        // Extract top-k results
        let mut results: Vec<PQSearchResult> =
            heap.into_iter().map(|std::cmp::Reverse(r)| r).collect();

        results.sort_by(|a, b| {
            a.approx_distance
                .partial_cmp(&b.approx_distance)
                .unwrap_or(Ordering::Equal)
        });

        results.truncate(k);
        results
    }
}

/// Statistics for PQ operations
#[derive(Debug, Default, Clone)]
pub struct PQStats {
    /// Number of vectors encoded
    pub vectors_encoded: u64,
    /// Number of ADC searches performed
    pub adc_searches: u64,
    /// Number of re-rankings performed
    pub reranks: u64,
    /// Total distance computations (for comparison)
    pub distance_computations: u64,
}

impl PQStats {
    /// Record an encoding operation
    pub fn record_encode(&mut self) {
        self.vectors_encoded += 1;
    }

    /// Record an ADC search
    pub fn record_adc_search(&mut self, candidates: usize) {
        self.adc_searches += 1;
        self.distance_computations += candidates as u64;
    }

    /// Record a rerank operation
    pub fn record_rerank(&mut self, reranked: usize) {
        self.reranks += 1;
        self.distance_computations += reranked as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::Rng;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Array1<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                Array1::from_vec(v)
            })
            .collect()
    }

    #[test]
    fn test_adc_table_creation() {
        let samples = generate_random_vectors(200, 64);
        let codebooks = PQCodebooks::train(&samples, 10, 8);

        let query = generate_random_vectors(1, 64).pop().unwrap();
        let adc = ADCTable::new(&query, &codebooks);

        assert_eq!(adc.n_subspaces(), 8); // 64 / 8 = 8 subspaces
    }

    #[test]
    fn test_pq_vector_store() {
        let samples = generate_random_vectors(200, 64);
        let store = PQVectorStore::train_and_create(&samples, 10, 8);

        // Add some vectors
        for (i, v) in samples.iter().enumerate().take(10) {
            store.add(i as u128, v);
        }

        assert_eq!(store.len(), 10);

        // Get codes
        let codes = store.get(0).unwrap();
        assert_eq!(codes.codes.len(), 8); // 8 subspaces

        // Remove
        assert!(store.remove(0));
        assert_eq!(store.len(), 9);
    }

    #[test]
    fn test_adc_distance() {
        let samples = generate_random_vectors(200, 64);
        let codebooks = PQCodebooks::train(&samples, 10, 8);

        let query = generate_random_vectors(1, 64).pop().unwrap();
        let adc = ADCTable::new(&query, &codebooks);

        // Encode a vector
        let v = &samples[0];
        let codes = codebooks.encode(v);

        // ADC distance should be reasonable
        let dist = adc.distance(&codes);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_search_adc() {
        let samples = generate_random_vectors(500, 64);
        let store = PQVectorStore::train_and_create(&samples, 10, 8);

        // Add vectors
        for (i, v) in samples.iter().enumerate() {
            store.add(i as u128, v);
        }

        // Search
        let query = generate_random_vectors(1, 64).pop().unwrap();
        let results = store.search_adc(&query, 10, 50);

        assert!(results.len() <= 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].approx_distance <= results[i].approx_distance);
        }
    }

    #[test]
    fn test_memory_usage() {
        let samples = generate_random_vectors(200, 64);
        let store = PQVectorStore::train_and_create(&samples, 10, 8);

        // Add 100 vectors
        for (i, v) in samples.iter().enumerate().take(100) {
            store.add(i as u128, v);
        }

        let mem = store.memory_usage();

        // 100 vectors × 8 subspaces = 800 bytes for codes
        // 8 subspaces × 256 centroids × 8 dims × 4 bytes = 65536 bytes for codebooks
        assert!(mem > 0);
        println!("Memory usage: {} bytes", mem);
    }

    #[test]
    fn test_pq_config() {
        let config = PQSearchConfig::default();
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.rerank_factor, 2);
        assert!(config.use_adc);
    }
}
