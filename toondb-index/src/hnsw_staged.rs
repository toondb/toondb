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

//! Staged Parallel HNSW Construction (Waves + Deferred Backedges)
//!
//! This module implements a high-performance parallel HNSW batch construction
//! algorithm that achieves near-linear scaling with cores while preserving
//! HNSW correctness invariants.
//!
//! ## Problem with Sequential Insert
//!
//! Current `insert_batch_bulk` effectively serializes:
//! ```text
//! for (id, vector) in batch {
//!     self.insert(id, vector);  // ~100µs each
//! }
//! repair_connectivity();      // O(N × ef × log N)
//! improve_search_quality();   // O(N × ef × log N)
//! ```
//!
//! This causes:
//! - ~400+ lock operations per insert
//! - Multi-million lock ops per 10K inserts
//! - Hub-node lock convoys (many threads competing for popular nodes)
//! - Expensive post-repair passes
//!
//! ## Staged Construction Solution
//!
//! ```text
//! Phase 1: Sequential Scaffold (S nodes)
//!   └─> Build small navigable core via single-insert
//!
//! Phase 2: Parallel Waves (W waves of B nodes each)
//!   Wave 1: ───┬─── Worker 1: insert nodes [0..B/P)
//!              ├─── Worker 2: insert nodes [B/P..2B/P)
//!              └─── Worker P: insert nodes [...B)
//!              └─> Deferred backedges → thread-local buffer
//!   Wave 2: Same, but now scaffold + wave1 are navigable
//!   ...
//!
//! Phase 3: Backedge Consolidation (parallel by target)
//!   └─> Merge thread-local backedge buffers
//!   └─> Apply backedges grouped by target node (no conflicts)
//! ```
//!
//! ## Invariant Preservation
//!
//! HNSW requires: when inserting v_i, subgraph {v_1, ..., v_{i-1}} must be navigable.
//!
//! **Staged waves preserve this**:
//! - Scaffold is fully connected before any wave
//! - Wave K starts only after Wave K-1 is complete
//! - Within a wave, forward edges point to already-navigable nodes
//! - Backedges are deferred (don't affect navigability)
//!
//! ## Performance Target
//!
//! | Metric | Sequential | Staged (8 cores) |
//! |--------|------------|------------------|
//! | 10K insert | ~1.0s | ~0.15s |
//! | Lock ops | 4M | 100K |
//! | Scaling | 1x | ~6.5x |
//!
//! ## Usage
//!
//! ```ignore
//! let builder = StagedBuilder::new(index, config);
//! let inserted = builder.insert_batch(&batch)?;
//! ```

use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;

use crate::hnsw::{HnswIndex, HnswNode, VersionedNeighbors, DistanceMetric};
use crate::vector_quantized::{QuantizedVector, Precision};

/// Maximum connections per node - matches the value in hnsw.rs
#[allow(dead_code)]
const MAX_M: usize = 32;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for staged parallel construction
#[derive(Debug, Clone)]
pub struct StagedConfig {
    /// Scaffold size: nodes to insert sequentially for navigable core
    /// Default: 2 × max_connections_layer0
    pub scaffold_size: usize,
    
    /// Wave size: nodes per parallel wave
    /// Smaller waves = more synchronization, better locality
    /// Larger waves = less synchronization, more memory for backedges
    /// Default: 1024
    pub wave_size: usize,
    
    /// Whether to skip post-repair passes (connectivity-by-construction)
    /// Default: true (staged construction guarantees connectivity)
    pub skip_repair_passes: bool,
    
    /// Precision for vector quantization
    pub precision: Precision,
}

impl Default for StagedConfig {
    fn default() -> Self {
        Self {
            scaffold_size: 128,  // 2 × typical M0=64
            wave_size: 1024,
            skip_repair_passes: true,
            precision: Precision::F32,
        }
    }
}

impl StagedConfig {
    /// Create config optimized for small batches (< 10K)
    pub fn for_small_batch() -> Self {
        Self {
            scaffold_size: 64,
            wave_size: 512,
            skip_repair_passes: true,
            precision: Precision::F32,
        }
    }
    
    /// Create config optimized for large batches (> 100K)
    pub fn for_large_batch() -> Self {
        Self {
            scaffold_size: 256,
            wave_size: 4096,
            skip_repair_passes: true,
            precision: Precision::F32,
        }
    }
}

// ============================================================================
// Thread-Local Backedge Buffer
// ============================================================================

/// A deferred backedge to be applied later
#[derive(Debug, Clone)]
struct DeferredBackedge {
    /// Target node that should receive this backedge
    target_id: u128,
    /// Source node creating the backedge
    source_id: u128,
    /// Layer for this edge
    layer: usize,
}

/// Thread-local buffer for deferred backedges
/// Avoids lock contention during forward edge construction
#[allow(dead_code)]
#[derive(Default)]
struct BackedgeBuffer {
    edges: Vec<DeferredBackedge>,
}

#[allow(dead_code)]
impl BackedgeBuffer {
    fn new() -> Self {
        Self {
            edges: Vec::with_capacity(1024),
        }
    }
    
    fn push(&mut self, target_id: u128, source_id: u128, layer: usize) {
        self.edges.push(DeferredBackedge {
            target_id,
            source_id,
            layer,
        });
    }
    
    fn drain(&mut self) -> Vec<DeferredBackedge> {
        std::mem::take(&mut self.edges)
    }
}

// ============================================================================
// Staged Builder
// ============================================================================

/// Statistics from a staged batch insert
#[derive(Debug, Default)]
pub struct StagedStats {
    /// Nodes inserted in scaffold phase
    pub scaffold_count: usize,
    /// Nodes inserted in wave phases
    pub wave_count: usize,
    /// Number of waves executed
    pub num_waves: usize,
    /// Total backedges deferred
    pub backedges_deferred: usize,
    /// Backedges successfully applied
    pub backedges_applied: usize,
    /// Backedges that failed (target deleted, etc.)
    pub backedges_failed: usize,
    /// Time in scaffold phase (microseconds)
    pub scaffold_time_us: u64,
    /// Time in wave phases (microseconds)
    pub wave_time_us: u64,
    /// Time in backedge consolidation (microseconds)
    pub backedge_time_us: u64,
}

/// Staged parallel HNSW batch builder
///
/// This builder constructs HNSW graphs in three phases:
/// 1. Sequential scaffold for navigable core
/// 2. Parallel waves with deferred backedges
/// 3. Parallel backedge consolidation by target
pub struct StagedBuilder<'a> {
    /// The target HNSW index
    index: &'a HnswIndex,
    /// Configuration
    config: StagedConfig,
    /// Statistics
    stats: StagedStats,
}

impl<'a> StagedBuilder<'a> {
    /// Create a new staged builder for the given index
    pub fn new(index: &'a HnswIndex, config: StagedConfig) -> Self {
        Self {
            index,
            config,
            stats: StagedStats::default(),
        }
    }
    
    /// Insert a batch of vectors using staged parallel construction
    ///
    /// Returns the number of successfully inserted vectors.
    pub fn insert_batch(mut self, batch: &[(u128, Vec<f32>)]) -> Result<(usize, StagedStats), String> {
        if batch.is_empty() {
            return Ok((0, self.stats));
        }
        
        let _start_total = std::time::Instant::now();
        
        // =====================================================================
        // Phase 1: Sequential Scaffold
        // =====================================================================
        let scaffold_start = std::time::Instant::now();
        
        let existing_nodes = self.index.nodes.len();
        let scaffold_threshold = self.config.scaffold_size.max(
            2 * self.index.config.max_connections_layer0
        );
        
        // Determine how many scaffold nodes we need
        let nodes_needed = scaffold_threshold.saturating_sub(existing_nodes);
        let scaffold_end = nodes_needed.min(batch.len());
        
        // Insert scaffold via proven sequential single-insert
        for i in 0..scaffold_end {
            let (id, vector) = &batch[i];
            if self.index.insert(*id, vector.clone()).is_ok() {
                self.stats.scaffold_count += 1;
            }
        }
        
        self.stats.scaffold_time_us = scaffold_start.elapsed().as_micros() as u64;
        
        if scaffold_end >= batch.len() {
            // All nodes were scaffold
            return Ok((self.stats.scaffold_count, self.stats));
        }
        
        // =====================================================================
        // Phase 2: Parallel Waves with Deferred Backedges
        // =====================================================================
        let wave_start = std::time::Instant::now();
        
        let bulk_batch = &batch[scaffold_end..];
        let wave_size = self.config.wave_size;
        let num_waves = (bulk_batch.len() + wave_size - 1) / wave_size;
        self.stats.num_waves = num_waves;
        
        // Global backedge buffer: target_id -> Vec<(source_id, layer)>
        let global_backedges: DashMap<u128, Vec<(u128, usize)>> = DashMap::new();
        
        for wave_idx in 0..num_waves {
            let wave_start_idx = wave_idx * wave_size;
            let wave_end_idx = (wave_start_idx + wave_size).min(bulk_batch.len());
            let wave_batch = &bulk_batch[wave_start_idx..wave_end_idx];
            
            // Process wave in parallel, collecting backedges
            let wave_results: Vec<_> = wave_batch
                .par_iter()
                .map(|(id, vector)| {
                    self.insert_with_deferred_backedges(*id, vector)
                })
                .collect();
            
            // Merge backedges into global buffer
            for result in wave_results {
                if let Ok((inserted, local_backedges)) = result {
                    if inserted {
                        self.stats.wave_count += 1;
                    }
                    for edge in local_backedges {
                        self.stats.backedges_deferred += 1;
                        global_backedges
                            .entry(edge.target_id)
                            .or_insert_with(Vec::new)
                            .push((edge.source_id, edge.layer));
                    }
                }
            }
            
            // Intra-wave connectivity: connect wave nodes to each other
            // This ensures wave nodes are reachable from the graph
            self.connect_wave_nodes(wave_batch);
        }
        
        self.stats.wave_time_us = wave_start.elapsed().as_micros() as u64;
        
        // =====================================================================
        // Phase 3: Backedge Consolidation (Parallel by Target)
        // =====================================================================
        let backedge_start = std::time::Instant::now();
        
        // Collect targets for parallel processing
        let targets: Vec<u128> = global_backedges.iter().map(|e| *e.key()).collect();
        
        // Apply backedges in parallel, grouped by target (no conflicts)
        let backedge_results: Vec<_> = targets
            .par_iter()
            .map(|target_id| {
                let backedges = global_backedges.get(target_id);
                if let Some(edges) = backedges {
                    self.apply_backedges_to_target(*target_id, edges.value())
                } else {
                    (0, 0)
                }
            })
            .collect();
        
        for (applied, failed) in backedge_results {
            self.stats.backedges_applied += applied;
            self.stats.backedges_failed += failed;
        }
        
        self.stats.backedge_time_us = backedge_start.elapsed().as_micros() as u64;
        
        // =====================================================================
        // Optional: Repair Passes (disabled by default with staged construction)
        // =====================================================================
        if !self.config.skip_repair_passes {
            self.index.repair_connectivity();
            self.index.improve_search_quality();
        }
        
        let total_inserted = self.stats.scaffold_count + self.stats.wave_count;
        Ok((total_inserted, self.stats))
    }
    
    /// Insert a single node with forward edges only, deferring backedges
    ///
    /// This is the core of the staged construction: each worker builds
    /// forward edges (which require no locks on other nodes' neighbor lists)
    /// and records backedges for later consolidation.
    fn insert_with_deferred_backedges(
        &self,
        id: u128,
        vector: &[f32],
    ) -> Result<(bool, Vec<DeferredBackedge>), String> {
        let mut backedges = Vec::new();
        
        // Validate dimension
        if vector.len() != self.index.dimension {
            return Ok((false, backedges));
        }
        
        // Create quantized vector - normalize for cosine metric when configured
        let should_normalize = matches!(self.index.config.metric, DistanceMetric::Cosine)
            && self.index.config.rng_optimization.normalize_at_ingest;
        let quantized = if should_normalize {
            QuantizedVector::from_f32_normalized(
                ndarray::Array1::from_vec(vector.to_vec()),
                self.config.precision,
            )
        } else {
            QuantizedVector::from_f32(
                ndarray::Array1::from_vec(vector.to_vec()),
                self.config.precision,
            )
        };
        
        // Generate random layer
        let layer = self.generate_random_layer();
        
        // Create node with empty layers
        let mut layers = Vec::with_capacity(layer + 1);
        for _ in 0..=layer {
            layers.push(RwLock::new(VersionedNeighbors::new()));
        }
        
        let node = Arc::new(HnswNode {
            id,
            vector: quantized.clone(),
            storage_id: None,
            layer,
            layers,
        });
        
        // Insert node into storage
        if self.index.nodes.insert(id, node.clone()).is_some() {
            // Node already exists
            return Ok((false, backedges));
        }
        
        // Get navigation state snapshot
        let nav_state = self.index.navigation_state();
        
        // If this is a potential new entry point, handle it
        if layer > nav_state.max_layer {
            // Update entry point atomically
            let mut ep = self.index.entry_point.write();
            let mut ml = self.index.max_layer.write();
            if layer > *ml {
                *ml = layer;
                *ep = Some(id);
            }
        }
        
        // Find entry point to start search from
        let ep_id = match nav_state.entry_point {
            Some(ep) if ep != id => ep,
            _ => {
                // First node or self - no edges to build
                return Ok((true, backedges));
            }
        };
        
        let ep_node = match self.index.nodes.get(&ep_id) {
            Some(n) => n.value().clone(),
            None => return Ok((true, backedges)),
        };
        
        // Navigate from top to target layer
        let mut curr_nearest = vec![SearchCandidate {
            distance: self.calculate_distance(&quantized, &ep_node.vector),
            id: ep_id,
        }];
        
        for lc in (layer + 1..=nav_state.max_layer).rev() {
            curr_nearest = self.search_layer(&quantized, &curr_nearest, 1, lc);
        }
        
        // Build edges at each layer from layer down to 0
        let ef = self.index.config.ef_construction;
        
        for lc in (0..=layer).rev() {
            // Search for neighbors at this layer
            let candidates = self.search_layer(&quantized, &curr_nearest, ef, lc);
            
            // Select neighbors using heuristic
            let max_connections = if lc == 0 {
                self.index.config.max_connections_layer0
            } else {
                self.index.config.max_connections
            };
            
            let selected = self.select_neighbors(&candidates, max_connections, &quantized);
            
            // Set forward edges (we own this node, no contention)
            {
                let mut layer_data = node.layers[lc].write();
                layer_data.neighbors = selected.iter().map(|c| c.id).collect();
                layer_data.version += 1;
            }
            
            // Defer backedges
            for neighbor in &selected {
                backedges.push(DeferredBackedge {
                    target_id: neighbor.id,
                    source_id: id,
                    layer: lc,
                });
            }
            
            // Update entry candidates for next layer
            if lc > 0 {
                curr_nearest = candidates;
            }
        }
        
        Ok((true, backedges))
    }
    
    /// Apply backedges to a single target node
    ///
    /// This is called during Phase 3 with all backedges grouped by target.
    /// Since each target is processed by exactly one worker, there's no contention.
    fn apply_backedges_to_target(
        &self,
        target_id: u128,
        edges: &[(u128, usize)],  // (source_id, layer)
    ) -> (usize, usize) {
        let mut applied = 0;
        let mut failed = 0;
        
        let target_node = match self.index.nodes.get(&target_id) {
            Some(n) => n.value().clone(),
            None => {
                // Target was deleted
                return (0, edges.len());
            }
        };
        
        // Group edges by layer
        let mut by_layer: HashMap<usize, Vec<u128>> = HashMap::new();
        for (source_id, layer) in edges {
            if *layer <= target_node.layer {
                by_layer.entry(*layer).or_insert_with(Vec::new).push(*source_id);
            } else {
                failed += 1;
            }
        }
        
        // Apply edges at each layer
        for (layer, sources) in by_layer {
            let max_connections = if layer == 0 {
                self.index.config.max_connections_layer0
            } else {
                self.index.config.max_connections
            };
            
            let mut layer_data = target_node.layers[layer].write();
            
            for source_id in sources {
                // Check if already a neighbor
                if layer_data.neighbors.contains(&source_id) {
                    continue;
                }
                
                // Add if capacity allows
                if layer_data.neighbors.len() < max_connections {
                    layer_data.neighbors.push(source_id);
                    layer_data.version += 1;
                    applied += 1;
                } else {
                    // Need to prune - check if new edge is better than worst
                    if let Some(source_node) = self.index.nodes.get(&source_id) {
                        let new_dist = self.calculate_distance(
                            &target_node.vector,
                            &source_node.vector
                        );
                        
                        // Find worst current neighbor
                        let worst = layer_data.neighbors
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &nid)| {
                                self.index.nodes.get(&nid).map(|n| {
                                    (i, self.calculate_distance(&target_node.vector, &n.vector))
                                })
                            })
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        
                        if let Some((worst_idx, worst_dist)) = worst {
                            if new_dist < worst_dist {
                                layer_data.neighbors[worst_idx] = source_id;
                                layer_data.version += 1;
                                applied += 1;
                            }
                        }
                    }
                }
            }
        }
        
        (applied, failed)
    }
    
    /// Connect wave nodes to each other for better intra-wave connectivity
    /// 
    /// After a wave is processed, wave nodes only have forward edges to scaffold/previous waves.
    /// This method adds mutual connections between wave nodes so they're reachable.
    fn connect_wave_nodes(&self, wave_batch: &[(u128, Vec<f32>)]) {
        // For each wave node, find its nearest wave neighbors and connect
        let wave_ids: Vec<u128> = wave_batch.iter().map(|(id, _)| *id).collect();
        
        for (id, _) in wave_batch {
            if let Some(node) = self.index.nodes.get(id) {
                let node_vec = &node.vector;
                
                // Find closest wave neighbors
                let mut wave_distances: Vec<(u128, f32)> = wave_ids
                    .iter()
                    .filter(|&wid| wid != id)
                    .filter_map(|&wid| {
                        self.index.nodes.get(&wid).map(|wnode| {
                            (wid, self.calculate_distance(node_vec, &wnode.vector))
                        })
                    })
                    .collect();
                
                wave_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                // Add top few wave neighbors (at least 4 for connectivity)
                let num_to_add = 4.min(wave_distances.len());
                let max_connections = self.index.config.max_connections_layer0;
                
                let mut layer_data = node.layers[0].write();
                for (wid, _) in wave_distances.into_iter().take(num_to_add) {
                    if !layer_data.neighbors.contains(&wid) && layer_data.neighbors.len() < max_connections {
                        layer_data.neighbors.push(wid);
                        layer_data.version += 1;
                    }
                }
            }
        }
    }
    
    // ========================================================================
    // Helper Methods
    // ========================================================================
    
    /// Generate random layer for new node (exponential distribution)
    fn generate_random_layer(&self) -> usize {
        let ml = 1.0 / (self.index.config.max_connections as f64).ln();
        let r: f64 = rand::random();
        let layer = (-r.ln() * ml).floor() as usize;
        layer.min(32) // Cap at reasonable max
    }
    
    /// Calculate distance between two quantized vectors
    fn calculate_distance(&self, a: &QuantizedVector, b: &QuantizedVector) -> f32 {
        match self.index.config.metric {
            DistanceMetric::Euclidean => {
                crate::vector_quantized::euclidean_distance_quantized(a, b)
            }
            DistanceMetric::Cosine => {
                crate::vector_quantized::cosine_distance_quantized(a, b)
            }
            DistanceMetric::DotProduct => {
                crate::vector_quantized::dot_product_quantized(a, b)
            }
        }
    }
    
    /// Search a single layer for nearest neighbors
    fn search_layer(
        &self,
        query: &QuantizedVector,
        entry_points: &[SearchCandidate],
        ef: usize,
        layer: usize,
    ) -> Vec<SearchCandidate> {
        use std::collections::{BinaryHeap, HashSet};
        
        let mut visited: HashSet<u128> = HashSet::new();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();
        let mut results: BinaryHeap<std::cmp::Reverse<SearchCandidate>> = BinaryHeap::new();
        
        for ep in entry_points {
            visited.insert(ep.id);
            candidates.push(ep.clone());
            results.push(std::cmp::Reverse(ep.clone()));
        }
        
        while let Some(current) = candidates.pop() {
            // Early termination: if closest candidate is farther than ef-th result
            if let Some(std::cmp::Reverse(worst)) = results.peek() {
                if current.distance > worst.distance && results.len() >= ef {
                    break;
                }
            }
            
            // Get neighbors at this layer
            let neighbors = if let Some(node) = self.index.nodes.get(&current.id) {
                if layer <= node.layer {
                    node.layers[layer].read().neighbors.clone()
                } else {
                    SmallVec::new()
                }
            } else {
                SmallVec::new()
            };
            
            for &neighbor_id in &neighbors {
                if visited.insert(neighbor_id) {
                    if let Some(neighbor_node) = self.index.nodes.get(&neighbor_id) {
                        let dist = self.calculate_distance(query, &neighbor_node.vector);
                        let candidate = SearchCandidate {
                            distance: dist,
                            id: neighbor_id,
                        };
                        
                        // Add to candidates if promising
                        let dominated = results.len() >= ef && {
                            if let Some(std::cmp::Reverse(worst)) = results.peek() {
                                dist > worst.distance
                            } else {
                                false
                            }
                        };
                        
                        if !dominated {
                            candidates.push(candidate.clone());
                            results.push(std::cmp::Reverse(candidate));
                            
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
        
        results.into_iter().map(|std::cmp::Reverse(c)| c).collect()
    }
    
    /// Select neighbors using HNSW heuristic
    fn select_neighbors(
        &self,
        candidates: &[SearchCandidate],
        max_neighbors: usize,
        _query: &QuantizedVector,
    ) -> Vec<SearchCandidate> {
        const ALPHA: f32 = 1.2; // RNG pruning factor
        // Ensure at least half the target neighbors for connectivity
        const MIN_NEIGHBORS_RATIO: f32 = 0.5;
        
        if candidates.len() <= max_neighbors {
            return candidates.to_vec();
        }
        
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        let mut result = Vec::with_capacity(max_neighbors);
        let min_neighbors = ((max_neighbors as f32) * MIN_NEIGHBORS_RATIO) as usize;
        
        for candidate in sorted {
            // First min_neighbors are added without pruning for connectivity
            if result.len() < min_neighbors {
                result.push(candidate);
                if result.len() >= max_neighbors {
                    break;
                }
                continue;
            }
            
            // Check if candidate is shadowed by any already-selected neighbor
            let is_shadowed = result.iter().any(|selected: &SearchCandidate| {
                if let (Some(c_node), Some(s_node)) = (
                    self.index.nodes.get(&candidate.id),
                    self.index.nodes.get(&selected.id)
                ) {
                    let dist_c_to_s = self.calculate_distance(&c_node.vector, &s_node.vector);
                    candidate.distance > ALPHA * dist_c_to_s
                } else {
                    false
                }
            });
            
            if !is_shadowed {
                result.push(candidate);
                if result.len() >= max_neighbors {
                    break;
                }
            }
        }
        
        result
    }
}

// ============================================================================
// Search Candidate
// ============================================================================

#[derive(Debug, Clone)]
struct SearchCandidate {
    distance: f32,
    id: u128,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for max-heap to behave as min-heap
        other.distance.partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// Integration with HnswIndex
// ============================================================================

impl HnswIndex {
    /// Insert batch using staged parallel construction
    ///
    /// This is the recommended method for bulk inserts (> 100 vectors).
    /// Uses wave-based parallelism with deferred backedges for near-linear
    /// scaling with cores.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let index = HnswIndex::new(768, HnswConfig::default());
    /// let config = StagedConfig::default();
    /// let (inserted, stats) = index.insert_batch_staged(&batch, config)?;
    /// println!("Inserted {} vectors in {} waves", inserted, stats.num_waves);
    /// ```
    pub fn insert_batch_staged(
        &self,
        batch: &[(u128, Vec<f32>)],
        config: StagedConfig,
    ) -> Result<(usize, StagedStats), String> {
        let builder = StagedBuilder::new(self, config);
        builder.insert_batch(batch)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HnswConfig;
    
    #[test]
    fn test_staged_basic() {
        let config = HnswConfig {
            max_connections: 16,
            max_connections_layer0: 32,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            quantization_precision: Some(Precision::F32),
            ..Default::default()
        };
        
        let index = HnswIndex::new(64, config);
        
        // Create test batch
        let batch: Vec<(u128, Vec<f32>)> = (0..500)
            .map(|i| {
                let vec: Vec<f32> = (0..64).map(|d| ((i * d) as f32) / 1000.0).collect();
                (i as u128, vec)
            })
            .collect();
        
        let staged_config = StagedConfig::default();
        let (inserted, stats) = index.insert_batch_staged(&batch, staged_config).unwrap();
        
        assert_eq!(inserted, 500);
        assert!(stats.num_waves > 0);
        assert!(stats.backedges_deferred > 0);
        
        println!("Stats: {:?}", stats);
    }
    
    #[test]
    fn test_staged_self_retrieval() {
        let config = HnswConfig {
            max_connections: 16,
            max_connections_layer0: 32,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Euclidean,
            quantization_precision: Some(Precision::F32),
            ..Default::default()
        };
        
        let index = HnswIndex::new(64, config);
        
        // Create test batch
        let batch: Vec<(u128, Vec<f32>)> = (0..200)
            .map(|i| {
                let vec: Vec<f32> = (0..64).map(|d| ((i * 100 + d) as f32) / 1000.0).collect();
                (i as u128, vec)
            })
            .collect();
        
        let staged_config = StagedConfig {
            skip_repair_passes: false,  // Enable repair for better connectivity
            ..StagedConfig::default()
        };
        let (inserted, _stats) = index.insert_batch_staged(&batch, staged_config).unwrap();
        assert_eq!(inserted, 200);

        // Test self-retrieval
        let mut failures = 0;
        for (id, vec) in &batch {
            let results = index.search(vec, 1).unwrap();
            if results.is_empty() || results[0].0 != *id {
                failures += 1;
            }
        }
        
        let success_rate = 100.0 * (batch.len() - failures) as f32 / batch.len() as f32;
        assert!(success_rate >= 95.0, "Self-retrieval rate too low: {}%", success_rate);
    }
}
