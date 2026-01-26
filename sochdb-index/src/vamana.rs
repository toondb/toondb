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

//! Vamana Graph Index (DiskANN-style)
//!
//! A single-layer graph index optimized for:
//! - Fewer hops (tunable α parameter for long edges)
//! - Disk-friendly access patterns
//! - Better integration with LSM storage
//!
//! Uses the existing:
//! - MmapVectorStorage for full vectors on disk
//! - Product Quantization for in-memory distance estimation
//!
//! Key advantages over HNSW:
//! - Single layer = simpler, faster graph traversal
//! - RobustPrune algorithm ensures angular diversity (long edges)
//! - Backedge deltas reduce write amplification
//! - PQ codes in RAM = 32x less memory than F16

use crate::product_quantization::{DistanceTable, PQCodebooks, PQCodes};
use crate::vector_storage::VectorStorage;
use dashmap::DashMap;
use ndarray::Array1;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum out-degree (R in DiskANN paper)
const DEFAULT_MAX_DEGREE: usize = 64;
/// Maximum degree before pruning (allows backedge accumulation)
const MAX_DEGREE_BOUND: usize = 96;

/// Vamana index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VamanaConfig {
    /// Maximum out-degree for each node (R)
    pub max_degree: usize,
    /// Pruning parameter: higher = longer edges, fewer hops
    /// Typical: 1.0 (short edges), 1.2 (balanced), 1.5 (long edges)
    pub alpha: f32,
    /// Search list size during construction (L_build)
    pub build_search_list: usize,
    /// Search list size during query (L_search)
    pub query_search_list: usize,
    /// Vector dimension (e.g., 384 for MiniLM)
    pub dimension: usize,
    /// Number of PQ subspaces (e.g., 48 for 384-dim with subdim=8)
    pub pq_subspaces: usize,
    /// Dimensions per PQ subspace
    pub pq_subdim: usize,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            max_degree: DEFAULT_MAX_DEGREE,
            alpha: 1.2,
            build_search_list: 100,
            query_search_list: 50,
            dimension: 384,
            pq_subspaces: 48,
            pq_subdim: 8,
        }
    }
}

impl VamanaConfig {
    /// Create config for common embedding dimensions
    pub fn for_dimension(dim: usize) -> Self {
        let subdim = 8;
        let subspaces = dim / subdim;
        Self {
            dimension: dim,
            pq_subspaces: subspaces,
            pq_subdim: subdim,
            ..Default::default()
        }
    }
}

/// Vamana graph node
///
/// Compact representation: just neighbors (no inline vector)
/// Vectors are stored separately in VectorStorage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VamanaNode {
    /// Neighbors (node IDs)
    pub neighbors: SmallVec<[u128; DEFAULT_MAX_DEGREE]>,
}

impl VamanaNode {
    fn new() -> Self {
        Self {
            neighbors: SmallVec::new(),
        }
    }

    fn with_neighbors(neighbors: Vec<u128>) -> Self {
        Self {
            neighbors: SmallVec::from_vec(neighbors),
        }
    }
}

/// Vamana index integrated with existing storage
pub struct VamanaIndex {
    config: VamanaConfig,

    /// Graph structure: node_id → neighbors
    /// Uses DashMap for concurrent access
    graph: Arc<DashMap<u128, VamanaNode>>,

    /// Backedge deltas: accumulated backedges not yet pruned
    /// Key insight from CoreNN: don't rewrite node on every insert
    backedge_deltas: Arc<DashMap<u128, Vec<u128>>>,

    /// PQ codebooks for distance estimation
    codebooks: Arc<RwLock<Option<PQCodebooks>>>,

    /// PQ codes for all vectors (48 bytes each, in RAM)
    /// This is the KEY to scaling: 10M vectors = 480 MB
    pq_codes: Arc<DashMap<u128, PQCodes>>,

    /// Full vectors (can be None if using external storage)
    vectors: Arc<DashMap<u128, Array1<f32>>>,

    /// Optional: external vector storage (e.g., MmapVectorStorage)
    vector_storage: Option<Arc<dyn VectorStorage>>,

    /// Mapping from node ID to storage ID (for external storage)
    id_to_storage: Arc<DashMap<u128, u64>>,

    /// Medoid (entry point) - the most central node
    medoid: Arc<RwLock<Option<u128>>>,

    /// Node count
    count: AtomicU64,

    /// Consolidation counter
    consolidation_counter: AtomicU64,
}

impl VamanaIndex {
    /// Create new Vamana index with in-memory vector storage
    pub fn new(config: VamanaConfig) -> Self {
        Self {
            config,
            graph: Arc::new(DashMap::new()),
            backedge_deltas: Arc::new(DashMap::new()),
            codebooks: Arc::new(RwLock::new(None)),
            pq_codes: Arc::new(DashMap::new()),
            vectors: Arc::new(DashMap::new()),
            vector_storage: None,
            id_to_storage: Arc::new(DashMap::new()),
            medoid: Arc::new(RwLock::new(None)),
            count: AtomicU64::new(0),
            consolidation_counter: AtomicU64::new(0),
        }
    }

    /// Create new Vamana index with external vector storage
    pub fn with_storage(config: VamanaConfig, storage: Arc<dyn VectorStorage>) -> Self {
        Self {
            config,
            graph: Arc::new(DashMap::new()),
            backedge_deltas: Arc::new(DashMap::new()),
            codebooks: Arc::new(RwLock::new(None)),
            pq_codes: Arc::new(DashMap::new()),
            vectors: Arc::new(DashMap::new()),
            vector_storage: Some(storage),
            id_to_storage: Arc::new(DashMap::new()),
            medoid: Arc::new(RwLock::new(None)),
            count: AtomicU64::new(0),
            consolidation_counter: AtomicU64::new(0),
        }
    }

    /// Train PQ codebooks on initial vector sample
    /// Call this before bulk insertion for best quality
    pub fn train_codebooks(&self, sample_vectors: &[Array1<f32>]) {
        let codebooks = PQCodebooks::train(sample_vectors, 20, self.config.pq_subdim);
        *self.codebooks.write() = Some(codebooks);
    }

    /// Train codebooks from raw slices
    pub fn train_codebooks_from_slices(&self, sample_vectors: &[Vec<f32>]) {
        let arrays: Vec<Array1<f32>> = sample_vectors
            .iter()
            .map(|v| Array1::from_vec(v.clone()))
            .collect();
        self.train_codebooks(&arrays);
    }

    /// Check if codebooks are trained
    pub fn has_codebooks(&self) -> bool {
        self.codebooks.read().is_some()
    }

    /// Insert a vector into the index
    ///
    /// Algorithm:
    /// 1. Store full vector (in-memory or external storage)
    /// 2. Encode to PQ codes (kept in RAM)
    /// 3. Search for neighbors using PQ distances
    /// 4. RobustPrune to select diverse neighbors
    /// 5. Add backedges (as deltas, not rewrites!)
    pub fn insert(&self, id: u128, vector: Vec<f32>) -> Result<(), String> {
        let vector_array = Array1::from_vec(vector);
        self.insert_array(id, vector_array)
    }

    /// Insert with Array1 directly
    pub fn insert_array(&self, id: u128, vector: Array1<f32>) -> Result<(), String> {
        if vector.len() != self.config.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.len()
            ));
        }

        // 1. Store full vector
        if let Some(ref storage) = self.vector_storage {
            let storage_id = storage
                .append(&vector)
                .map_err(|e| format!("Storage error: {}", e))?;
            self.id_to_storage.insert(id, storage_id);
        } else {
            self.vectors.insert(id, vector.clone());
        }

        // 2. PQ encode for RAM
        let pq = {
            let codebooks = self.codebooks.read();
            let codebooks = codebooks
                .as_ref()
                .ok_or("Codebooks not trained. Call train_codebooks first.")?;
            codebooks.encode(&vector)
        };
        self.pq_codes.insert(id, pq);

        // 3. Get or create medoid
        let medoid_id = {
            let mut medoid_guard = self.medoid.write();
            if medoid_guard.is_none() {
                *medoid_guard = Some(id);
                self.graph.insert(id, VamanaNode::new());
                self.count.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            medoid_guard.unwrap()
        };

        // 4. Greedy search to find candidates
        let dist_table = {
            let codebooks = self.codebooks.read();
            let codebooks = codebooks.as_ref().unwrap();
            codebooks.build_distance_table(&vector)
        };

        let candidates =
            self.greedy_search_internal(&dist_table, medoid_id, self.config.build_search_list);

        // 5. RobustPrune to select neighbors
        let neighbors = self.robust_prune(id, candidates);

        // 6. Insert node with pruned neighbors
        self.graph
            .insert(id, VamanaNode::with_neighbors(neighbors.clone()));

        // 7. Add backedges as DELTAS (CoreNN's key insight!)
        for &neighbor_id in &neighbors {
            self.backedge_deltas
                .entry(neighbor_id)
                .or_default()
                .push(id);
        }

        self.count.fetch_add(1, Ordering::Relaxed);

        // 8. Periodically consolidate backedges
        let counter = self.consolidation_counter.fetch_add(1, Ordering::Relaxed);
        if counter % 1000 == 999 {
            self.consolidate_backedges();
        }

        Ok(())
    }

    /// Greedy search using PQ distance table (beam search algorithm)
    ///
    /// This implements the "beam search" algorithm used in DiskANN:
    /// 1. Start from entry point (medoid)
    /// 2. Maintain a "frontier" of best candidates to expand
    /// 3. Expand best unexpanded candidate
    /// 4. Keep track of L best candidates seen so far
    fn greedy_search_internal(
        &self,
        dist_table: &DistanceTable,
        start: u128,
        search_list_size: usize,
    ) -> Vec<(u128, f32)> {
        let mut visited = HashSet::new();

        // Results: best L candidates found so far
        let mut results: Vec<(u128, f32)> = Vec::with_capacity(search_list_size * 2);

        // Start from medoid
        if let Some(pq) = self.pq_codes.get(&start) {
            let dist = dist_table.distance(&pq);
            results.push((start, dist));
            visited.insert(start);
        } else {
            return results;
        }

        // Pointer to the next candidate to expand
        let mut expand_idx = 0;

        // Keep expanding until we've expanded all candidates or hit limit
        while expand_idx < results.len() && expand_idx < search_list_size * 3 {
            let current_id = results[expand_idx].0;
            expand_idx += 1;

            // Expand neighbors from graph
            if let Some(node) = self.graph.get(&current_id) {
                for &neighbor_id in node.neighbors.iter() {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    if let Some(pq) = self.pq_codes.get(&neighbor_id) {
                        let dist = dist_table.distance(&pq);
                        results.push((neighbor_id, dist));
                    }
                }
            }

            // Also check backedge deltas
            if let Some(deltas) = self.backedge_deltas.get(&current_id) {
                for &neighbor_id in deltas.iter() {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    if let Some(pq) = self.pq_codes.get(&neighbor_id) {
                        let dist = dist_table.distance(&pq);
                        results.push((neighbor_id, dist));
                    }
                }
            }

            // Re-sort to keep best candidates at front for expansion
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Truncate to search list size
        results.truncate(search_list_size);
        results
    }

    /// RobustPrune: Vamana's key algorithm for diverse neighbor selection
    ///
    /// Ensures angular diversity: don't pick neighbors that are too close to
    /// already-selected neighbors. This creates the "long edges" that enable
    /// O(log N) search with fewer hops.
    fn robust_prune(&self, node_id: u128, mut candidates: Vec<(u128, f32)>) -> Vec<u128> {
        // Sort candidates by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove self
        candidates.retain(|(id, _)| *id != node_id);

        let mut selected = Vec::with_capacity(self.config.max_degree);

        for (candidate_id, candidate_dist) in candidates {
            if selected.len() >= self.config.max_degree {
                break;
            }

            // Check if candidate is dominated by any already-selected neighbor
            // A candidate is dominated if: dist(selected, candidate) * alpha < dist(node, candidate)
            let dominated = selected.iter().any(|&selected_id| {
                // Get PQ codes for distance calculation
                let selected_pq = match self.pq_codes.get(&selected_id) {
                    Some(pq) => pq,
                    None => return false,
                };
                let candidate_pq = match self.pq_codes.get(&candidate_id) {
                    Some(pq) => pq,
                    None => return false,
                };

                // Approximate distance between selected and candidate
                let dist_selected_candidate =
                    self.approximate_pq_distance(&selected_pq, &candidate_pq);

                // Check domination condition with alpha
                dist_selected_candidate * self.config.alpha < candidate_dist
            });

            if !dominated {
                selected.push(candidate_id);
            }
        }

        selected
    }

    /// Approximate distance between two PQ codes
    /// Uses simple hamming-like metric as approximation
    fn approximate_pq_distance(&self, a: &PQCodes, b: &PQCodes) -> f32 {
        // Simple approximation: count matching codes and scale
        let matching: usize = a
            .codes
            .iter()
            .zip(b.codes.iter())
            .filter(|(x, y)| *x == *y)
            .count();

        let total = a.codes.len();
        // Convert to distance-like metric: more matches = smaller distance
        ((total - matching) as f32) / (total as f32) * 10.0
    }

    /// Consolidate backedge deltas into main graph
    /// Called periodically to prevent delta accumulation
    pub fn consolidate_backedges(&self) {
        let entries: Vec<_> = self
            .backedge_deltas
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();

        for (node_id, deltas) in entries {
            if deltas.is_empty() {
                continue;
            }

            if let Some(mut node) = self.graph.get_mut(&node_id) {
                // Merge deltas into neighbors
                let mut all_neighbors: HashSet<u128> = node.neighbors.iter().copied().collect();
                for delta in deltas {
                    all_neighbors.insert(delta);
                }

                // Prune if over capacity
                if all_neighbors.len() > MAX_DEGREE_BOUND {
                    // Build distance table for this node's vector
                    if let Some(vector) = self.get_full_vector(node_id) {
                        let codebooks = self.codebooks.read();
                        if let Some(ref cb) = *codebooks {
                            let dist_table = cb.build_distance_table(&vector);
                            drop(codebooks);

                            // Calculate distances to all neighbors
                            let mut neighbor_dists: Vec<(u128, f32)> = all_neighbors
                                .iter()
                                .filter_map(|&n| {
                                    self.pq_codes
                                        .get(&n)
                                        .map(|pq| (n, dist_table.distance(&pq)))
                                })
                                .collect();

                            // Sort and take top max_degree
                            neighbor_dists.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            neighbor_dists.truncate(self.config.max_degree);

                            node.neighbors = SmallVec::from_vec(
                                neighbor_dists.iter().map(|(id, _)| *id).collect(),
                            );
                        }
                    }
                } else {
                    node.neighbors = SmallVec::from_vec(all_neighbors.into_iter().collect());
                }
            }
        }

        // Clear all deltas
        self.backedge_deltas.clear();
    }

    /// Get full vector (from in-memory storage or external storage)
    fn get_full_vector(&self, id: u128) -> Option<Array1<f32>> {
        // Try in-memory first
        if let Some(vec) = self.vectors.get(&id) {
            return Some(vec.clone());
        }

        // Try external storage
        if let Some(ref storage) = self.vector_storage
            && let Some(storage_id) = self.id_to_storage.get(&id)
        {
            return storage.get(*storage_id).ok();
        }

        None
    }

    /// Search for k nearest neighbors
    ///
    /// Uses PQ distances for graph traversal
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String> {
        let query_array = Array1::from_vec(query.to_vec());

        let dist_table = {
            let codebooks = self.codebooks.read();
            let codebooks = codebooks.as_ref().ok_or("Codebooks not trained")?;
            codebooks.build_distance_table(&query_array)
        };

        let medoid = *self.medoid.read();
        let medoid_id = medoid.ok_or("Index is empty")?;

        // Search with larger list for better recall
        let search_size = self.config.query_search_list.max(k * 2);
        let candidates = self.greedy_search_internal(&dist_table, medoid_id, search_size);

        // Return top k by PQ distance
        let results: Vec<_> = candidates.into_iter().take(k).collect();
        Ok(results)
    }

    /// Search with reranking using full vectors (higher accuracy, slower)
    pub fn search_rerank(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String> {
        let query_array = Array1::from_vec(query.to_vec());

        // Get more candidates than needed for reranking
        let candidates = self.search(query, k * 2)?;

        // Rerank with full vectors using true L2 distance
        let mut reranked: Vec<_> = candidates
            .into_iter()
            .filter_map(|(id, _approx_dist)| {
                self.get_full_vector(id).map(|vec| {
                    let dist = squared_l2(&query_array, &vec);
                    (id, dist)
                })
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);

        Ok(reranked)
    }

    /// Get index statistics
    pub fn stats(&self) -> VamanaStats {
        let mut total_neighbors = 0usize;
        let mut total_deltas = 0usize;

        for entry in self.graph.iter() {
            total_neighbors += entry.value().neighbors.len();
        }

        for entry in self.backedge_deltas.iter() {
            total_deltas += entry.value().len();
        }

        let count = self.count.load(Ordering::Relaxed) as usize;

        VamanaStats {
            node_count: count,
            avg_degree: if count > 0 {
                total_neighbors as f32 / count as f32
            } else {
                0.0
            },
            pending_deltas: total_deltas,
            pq_memory_bytes: count * self.config.pq_subspaces, // 1 byte per subspace
            graph_memory_bytes: total_neighbors * 16,          // 16 bytes per u128 neighbor
            codebook_memory_bytes: self
                .codebooks
                .read()
                .as_ref()
                .map(|c| c.memory_size())
                .unwrap_or(0),
        }
    }

    /// Number of vectors in the index
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed) as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Save index to directory
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // Save codebooks
        if let Some(ref codebooks) = *self.codebooks.read() {
            codebooks.save(dir.join("codebooks.bin.gz"))?;
        }

        // Save graph
        let graph_data: Vec<(u128, VamanaNode)> = self
            .graph
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        let graph_file = std::fs::File::create(dir.join("graph.bin.gz"))?;
        let encoder = flate2::write::GzEncoder::new(graph_file, flate2::Compression::fast());
        bincode::serialize_into(encoder, &graph_data)
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Save PQ codes
        let pq_data: Vec<(u128, PQCodes)> = self
            .pq_codes
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        let pq_file = std::fs::File::create(dir.join("pq_codes.bin.gz"))?;
        let encoder = flate2::write::GzEncoder::new(pq_file, flate2::Compression::fast());
        bincode::serialize_into(encoder, &pq_data).map_err(|e| io::Error::other(e.to_string()))?;

        // Save config and metadata
        let metadata = VamanaMetadata {
            config: self.config.clone(),
            medoid: *self.medoid.read(),
            count: self.count.load(Ordering::Relaxed),
        };
        let meta_file = std::fs::File::create(dir.join("metadata.bin.gz"))?;
        let encoder = flate2::write::GzEncoder::new(meta_file, flate2::Compression::fast());
        bincode::serialize_into(encoder, &metadata).map_err(|e| io::Error::other(e.to_string()))?;

        Ok(())
    }

    /// Load index from directory
    pub fn load<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();

        // Load metadata
        let meta_file = std::fs::File::open(dir.join("metadata.bin.gz"))?;
        let decoder = flate2::read::GzDecoder::new(meta_file);
        let metadata: VamanaMetadata =
            bincode::deserialize_from(decoder).map_err(|e| io::Error::other(e.to_string()))?;

        // Load codebooks
        let codebooks = PQCodebooks::load(dir.join("codebooks.bin.gz")).ok();

        // Load graph
        let graph_file = std::fs::File::open(dir.join("graph.bin.gz"))?;
        let decoder = flate2::read::GzDecoder::new(graph_file);
        let graph_data: Vec<(u128, VamanaNode)> =
            bincode::deserialize_from(decoder).map_err(|e| io::Error::other(e.to_string()))?;

        let graph = DashMap::new();
        for (id, node) in graph_data {
            graph.insert(id, node);
        }

        // Load PQ codes
        let pq_file = std::fs::File::open(dir.join("pq_codes.bin.gz"))?;
        let decoder = flate2::read::GzDecoder::new(pq_file);
        let pq_data: Vec<(u128, PQCodes)> =
            bincode::deserialize_from(decoder).map_err(|e| io::Error::other(e.to_string()))?;

        let pq_codes = DashMap::new();
        for (id, pq) in pq_data {
            pq_codes.insert(id, pq);
        }

        Ok(Self {
            config: metadata.config,
            graph: Arc::new(graph),
            backedge_deltas: Arc::new(DashMap::new()),
            codebooks: Arc::new(RwLock::new(codebooks)),
            pq_codes: Arc::new(pq_codes),
            vectors: Arc::new(DashMap::new()),
            vector_storage: None,
            id_to_storage: Arc::new(DashMap::new()),
            medoid: Arc::new(RwLock::new(metadata.medoid)),
            count: AtomicU64::new(metadata.count),
            consolidation_counter: AtomicU64::new(0),
        })
    }
}

/// Metadata for persistence
#[derive(Serialize, Deserialize)]
struct VamanaMetadata {
    config: VamanaConfig,
    medoid: Option<u128>,
    count: u64,
}

/// Vamana index statistics
#[derive(Debug, Clone)]
pub struct VamanaStats {
    pub node_count: usize,
    pub avg_degree: f32,
    pub pending_deltas: usize,
    pub pq_memory_bytes: usize,
    pub graph_memory_bytes: usize,
    pub codebook_memory_bytes: usize,
}

impl VamanaStats {
    /// Total memory usage in megabytes
    pub fn total_memory_mb(&self) -> f64 {
        (self.pq_memory_bytes + self.graph_memory_bytes + self.codebook_memory_bytes) as f64
            / (1024.0 * 1024.0)
    }
}

/// Helper: squared L2 distance
#[inline]
fn squared_l2(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Array1<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| Array1::from_iter((0..dim).map(|_| rng.r#gen::<f32>())))
            .collect()
    }

    #[test]
    fn test_vamana_basic_insert_search() {
        let config = VamanaConfig::for_dimension(64);
        let index = VamanaIndex::new(config);

        // Generate training data
        let vectors = generate_random_vectors(100, 64);

        // Train codebooks
        index.train_codebooks(&vectors);

        // Insert vectors
        for (i, vec) in vectors.iter().enumerate() {
            index
                .insert_array(i as u128, vec.clone())
                .expect("Insert should succeed");
        }

        assert_eq!(index.len(), 100);

        // Search
        let query = vectors[0].to_vec();
        let results = index.search(&query, 5).expect("Search should succeed");

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // First result should be the query itself (or very close)
        println!(
            "Top result ID: {}, distance: {}",
            results[0].0, results[0].1
        );
    }

    #[test]
    fn test_vamana_search_quality() {
        let config = VamanaConfig::for_dimension(128);
        let index = VamanaIndex::new(config);

        let vectors = generate_random_vectors(500, 128);
        index.train_codebooks(&vectors[..100]);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert_array(i as u128, vec.clone()).unwrap();
        }

        // Search for a known vector
        let query_idx = 42;
        let query = vectors[query_idx].to_vec();
        let results = index.search(&query, 10).unwrap();

        // The query vector should be in top results
        let found = results.iter().any(|(id, _)| *id == query_idx as u128);
        println!("Query found in results: {}", found);
        println!("Results: {:?}", results.iter().take(5).collect::<Vec<_>>());
    }

    #[test]
    fn test_vamana_memory_efficiency() {
        let dim = 384;
        let config = VamanaConfig::for_dimension(dim);
        let index = VamanaIndex::new(config);

        let vectors = generate_random_vectors(1000, dim);
        index.train_codebooks(&vectors[..100]);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert_array(i as u128, vec.clone()).unwrap();
        }

        let stats = index.stats();
        println!("Stats: {:?}", stats);
        println!("Total memory: {:.2} MB", stats.total_memory_mb());

        // PQ memory should be much less than raw vector memory
        let raw_memory = 1000 * dim * 4; // bytes
        let pq_memory = stats.pq_memory_bytes;
        let compression = raw_memory as f32 / pq_memory as f32;

        println!(
            "Raw: {} bytes, PQ: {} bytes, Compression: {:.1}x",
            raw_memory, pq_memory, compression
        );
        assert!(compression > 20.0); // Should be ~32x
    }

    #[test]
    fn test_vamana_consolidation() {
        let config = VamanaConfig::for_dimension(64);
        let index = VamanaIndex::new(config);

        let vectors = generate_random_vectors(100, 64);
        index.train_codebooks(&vectors);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert_array(i as u128, vec.clone()).unwrap();
        }

        // Manually trigger consolidation
        index.consolidate_backedges();

        let stats = index.stats();
        assert_eq!(stats.pending_deltas, 0);
    }

    #[test]
    fn test_vamana_save_load() {
        let config = VamanaConfig::for_dimension(64);
        let index = VamanaIndex::new(config);

        let vectors = generate_random_vectors(50, 64);
        index.train_codebooks(&vectors);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert_array(i as u128, vec.clone()).unwrap();
        }

        // Save
        let temp_dir = tempfile::tempdir().unwrap();
        index.save(temp_dir.path()).unwrap();

        // Load
        let loaded = VamanaIndex::load(temp_dir.path()).unwrap();

        assert_eq!(loaded.len(), index.len());

        // Search should work on loaded index
        let results = loaded.search(&vectors[0].to_vec(), 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_vamana_stats() {
        let config = VamanaConfig::for_dimension(128);
        let index = VamanaIndex::new(config);

        let vectors = generate_random_vectors(200, 128);
        index.train_codebooks(&vectors[..50]);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert_array(i as u128, vec.clone()).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.node_count, 200);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.codebook_memory_bytes > 0);
    }
}
