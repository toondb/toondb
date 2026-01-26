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

//! Core HNSW implementation for WebAssembly
//!
//! This is a simplified, self-contained HNSW implementation optimized for WASM:
//! - No external crate dependencies beyond std
//! - Minimal memory footprint
//! - Compatible with JavaScript typed arrays

use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, BigUint64Array, Array};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use smallvec::SmallVec;

use crate::{IndexStats, SearchResult};

/// Maximum connections per node (M parameter)
const DEFAULT_M: usize = 16;
/// Maximum connections for layer 0 (M0 = 2 * M)
#[allow(dead_code)]
const DEFAULT_M0: usize = 32;
/// Default construction ef
const DEFAULT_EF_CONSTRUCTION: usize = 100;
/// Default search ef
const DEFAULT_EF_SEARCH: usize = 50;
/// Maximum neighbors stored inline
const MAX_INLINE_NEIGHBORS: usize = 32;

/// HNSW node
struct WasmNode {
    id: u64,
    vector: Vec<f32>,
    /// Neighbors at each layer
    layers: Vec<SmallVec<[u32; MAX_INLINE_NEIGHBORS]>>,
}

/// Priority queue entry
#[derive(Clone)]
struct Candidate {
    distance: f32,
    node_idx: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (for result collection)
#[derive(Clone)]
struct MaxCandidate {
    distance: f32,
    node_idx: u32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal ordering for max-heap
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

/// WASM-compatible HNSW Vector Index
#[wasm_bindgen]
pub struct WasmVectorIndex {
    dimension: usize,
    max_connections: usize,
    max_connections_layer0: usize,
    ef_construction: usize,
    ef_search: usize,
    level_multiplier: f32,
    
    nodes: Vec<WasmNode>,
    id_to_idx: HashMap<u64, u32>,
    entry_point: Option<u32>,
    max_layer: usize,
}

#[wasm_bindgen]
impl WasmVectorIndex {
    /// Create a new vector index
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension (e.g., 768 for BERT)
    /// * `max_connections` - M parameter (default: 16)
    /// * `ef_construction` - Construction ef (default: 100)
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, max_connections: usize, ef_construction: usize) -> Self {
        let m = if max_connections == 0 { DEFAULT_M } else { max_connections };
        let ef = if ef_construction == 0 { DEFAULT_EF_CONSTRUCTION } else { ef_construction };
        
        Self {
            dimension,
            max_connections: m,
            max_connections_layer0: m * 2,
            ef_construction: ef,
            ef_search: DEFAULT_EF_SEARCH,
            level_multiplier: 1.0 / (m as f32).ln(),
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_layer: 0,
        }
    }
    
    /// Get the number of vectors in the index
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get vector dimension
    #[wasm_bindgen(getter)]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Set search ef parameter
    #[wasm_bindgen(setter)]
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef.max(1);
    }
    
    /// Insert a single vector
    #[wasm_bindgen]
    pub fn insert(&mut self, id: u64, vector: Float32Array) -> bool {
        if vector.length() as usize != self.dimension {
            return false;
        }
        
        if self.id_to_idx.contains_key(&id) {
            return false;
        }
        
        let vec: Vec<f32> = vector.to_vec();
        self.insert_internal(id, vec)
    }
    
    /// Insert a batch of vectors
    ///
    /// # Arguments
    /// * `ids` - BigUint64Array of vector IDs
    /// * `vectors` - Float32Array of flattened vectors (row-major)
    ///
    /// # Returns
    /// Number of vectors successfully inserted
    #[wasm_bindgen(js_name = insertBatch)]
    pub fn insert_batch(&mut self, ids: BigUint64Array, vectors: Float32Array) -> u32 {
        let num_ids = ids.length() as usize;
        let num_floats = vectors.length() as usize;
        
        if num_floats != num_ids * self.dimension {
            return 0;
        }
        
        let ids_vec: Vec<u64> = ids.to_vec();
        let vectors_vec: Vec<f32> = vectors.to_vec();
        
        let mut inserted = 0u32;
        
        for (i, &id) in ids_vec.iter().enumerate() {
            if self.id_to_idx.contains_key(&id) {
                continue;
            }
            
            let start = i * self.dimension;
            let end = start + self.dimension;
            let vec = vectors_vec[start..end].to_vec();
            
            if self.insert_internal(id, vec) {
                inserted += 1;
            }
        }
        
        inserted
    }
    
    /// Search for k-nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Float32Array query vector
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    /// Array of SearchResult objects
    #[wasm_bindgen]
    pub fn search(&self, query: Float32Array, k: usize) -> Array {
        let results = Array::new();
        
        if query.length() as usize != self.dimension {
            return results;
        }
        
        if self.nodes.is_empty() {
            return results;
        }
        
        let query_vec: Vec<f32> = query.to_vec();
        let neighbors = self.search_internal(&query_vec, k);
        
        for (id, distance) in neighbors {
            let result = SearchResult::new(id, distance);
            results.push(&JsValue::from(result));
        }
        
        results
    }
    
    /// Get index statistics
    #[wasm_bindgen]
    pub fn stats(&self) -> IndexStats {
        let avg_connections = if self.nodes.is_empty() {
            0.0
        } else {
            let total: usize = self.nodes.iter()
                .map(|n| n.layers.iter().map(|l| l.len()).sum::<usize>())
                .sum();
            total as f32 / self.nodes.len() as f32
        };
        
        IndexStats {
            num_vectors: self.nodes.len() as u32,
            dimension: self.dimension as u32,
            max_layer: self.max_layer as u32,
            avg_connections,
        }
    }
    
    /// Check if a vector ID exists
    #[wasm_bindgen]
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_idx.contains_key(&id)
    }
    
    /// Clear all vectors from the index
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.id_to_idx.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }
}

// Internal methods (not exposed to WASM)
impl WasmVectorIndex {
    /// Internal insert implementation
    fn insert_internal(&mut self, id: u64, vector: Vec<f32>) -> bool {
        // Assign random layer
        let level = self.random_level();
        
        // Create node
        let node_idx = self.nodes.len() as u32;
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(SmallVec::new());
        }
        
        let node = WasmNode {
            id,
            vector: vector.clone(),
            layers,
        };
        
        self.nodes.push(node);
        self.id_to_idx.insert(id, node_idx);
        
        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_layer = level;
            return true;
        }
        
        let entry_point = self.entry_point.unwrap();
        // Clone the query vector to avoid borrow issues during modification
        let query = vector;
        
        // Navigate from top layer down to level+1
        let mut current = entry_point;
        for layer in (level + 1..=self.max_layer).rev() {
            current = self.search_layer_single(&query, current, layer);
        }
        
        // Insert at each layer from level down to 0
        for layer in (0..=level.min(self.max_layer)).rev() {
            let neighbors = self.search_layer(&query, current, self.ef_construction, layer);
            
            // Select best M neighbors
            let max_m = if layer == 0 { self.max_connections_layer0 } else { self.max_connections };
            let selected = self.select_neighbors(&neighbors, max_m);
            
            // Connect new node to neighbors
            for &neighbor_idx in &selected {
                self.nodes[node_idx as usize].layers[layer].push(neighbor_idx);
            }
            
            // Connect neighbors back to new node (bidirectional)
            for &neighbor_idx in &selected {
                let neighbor = &mut self.nodes[neighbor_idx as usize];
                if layer < neighbor.layers.len() {
                    if neighbor.layers[layer].len() < max_m {
                        neighbor.layers[layer].push(node_idx);
                    }
                }
            }
            
            if !neighbors.is_empty() {
                current = neighbors[0].1;
            }
        }
        
        // Update entry point if new node has higher layer
        if level > self.max_layer {
            self.entry_point = Some(node_idx);
            self.max_layer = level;
        }
        
        true
    }
    
    /// Internal search implementation
    fn search_internal(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() {
            return vec![];
        }
        
        let entry_point = self.entry_point.unwrap();
        
        // Navigate from top layer to layer 1
        let mut current = entry_point;
        for layer in (1..=self.max_layer).rev() {
            current = self.search_layer_single(query, current, layer);
        }
        
        // Search layer 0 with ef_search
        let candidates = self.search_layer(query, current, self.ef_search.max(k), 0);
        
        // Return top-k results
        candidates
            .into_iter()
            .take(k)
            .map(|(dist, idx)| (self.nodes[idx as usize].id, dist))
            .collect()
    }
    
    /// Search a single layer for the nearest neighbor
    fn search_layer_single(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut current = entry;
        let mut current_dist = self.distance(query, &self.nodes[entry as usize].vector);
        
        loop {
            let mut improved = false;
            
            let neighbors = &self.nodes[current as usize].layers[layer];
            for &neighbor_idx in neighbors.iter() {
                let dist = self.distance(query, &self.nodes[neighbor_idx as usize].vector);
                if dist < current_dist {
                    current = neighbor_idx;
                    current_dist = dist;
                    improved = true;
                }
            }
            
            if !improved {
                break;
            }
        }
        
        current
    }
    
    /// Search a layer with ef parameter
    fn search_layer(&self, query: &[f32], entry: u32, ef: usize, layer: usize) -> Vec<(f32, u32)> {
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();
        
        let entry_dist = self.distance(query, &self.nodes[entry as usize].vector);
        candidates.push(Candidate { distance: entry_dist, node_idx: entry });
        results.push(MaxCandidate { distance: entry_dist, node_idx: entry });
        visited[entry as usize] = true;
        
        while let Some(current) = candidates.pop() {
            // Check if we've found enough good candidates
            if let Some(worst) = results.peek() {
                if current.distance > worst.distance && results.len() >= ef {
                    break;
                }
            }
            
            // Explore neighbors
            let neighbors = &self.nodes[current.node_idx as usize].layers.get(layer);
            if let Some(neighbors) = neighbors {
                for &neighbor_idx in neighbors.iter() {
                    if !visited[neighbor_idx as usize] {
                        visited[neighbor_idx as usize] = true;
                        
                        let dist = self.distance(query, &self.nodes[neighbor_idx as usize].vector);
                        
                        let should_add = if let Some(worst) = results.peek() {
                            results.len() < ef || dist < worst.distance
                        } else {
                            true
                        };
                        
                        if should_add {
                            candidates.push(Candidate { distance: dist, node_idx: neighbor_idx });
                            results.push(MaxCandidate { distance: dist, node_idx: neighbor_idx });
                            
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
        
        // Convert to sorted vec
        let mut result_vec: Vec<(f32, u32)> = results
            .into_iter()
            .map(|c| (c.distance, c.node_idx))
            .collect();
        result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        result_vec
    }
    
    /// Select best neighbors (simple heuristic)
    fn select_neighbors(&self, candidates: &[(f32, u32)], max_m: usize) -> Vec<u32> {
        candidates
            .iter()
            .take(max_m)
            .map(|(_, idx)| *idx)
            .collect()
    }
    
    /// Compute L2 distance
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Generate random level for new node
    fn random_level(&self) -> usize {
        let r: f32 = rand::random();
        let level = (-r.ln() * self.level_multiplier).floor() as usize;
        level.min(16) // Cap at 16 layers
    }
}
