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

//! # Vector Storage Hot-Path Layout (Task 16)
//!
//! Optimized memory layout for vector storage focused on hot-path performance:
//! - Contiguous embedding storage for cache efficiency
//! - SIMD-aligned vectors
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Embedding Block (64KB aligned)                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Header (64B)      │ Padding │ Vectors (contiguous, 32B aligned) │
//! │ - magic           │         │ Vec[0]: [f32; D]                  │
//! │ - version         │         │ Vec[1]: [f32; D]                  │
//! │ - count           │         │ ...                               │
//! │ - dim             │         │ Vec[N-1]: [f32; D]                │
//! │ - checksum        │         │                                   │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Neighbor Block (64KB aligned)                 │
//! ├─────────────────────────────────────────────────────────────────┤
//!
//! ## Design Principles
//!
//! 1. **SIMD Alignment**: All vectors 32-byte aligned for AVX2
//! 2. **Cache Lines**: Hot data packed into 64-byte cache lines
//! 3. **Prefetching**: Neighbor lookups prefetch next block
//! 4. **Contiguous**: Avoid pointer chasing in hot path

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
// Constants
// ============================================================================

/// SIMD alignment for vectors (AVX2 = 32 bytes, AVX-512 = 64 bytes)
pub const SIMD_ALIGNMENT: usize = 32;

/// Cache line size
pub const CACHE_LINE_SIZE: usize = 64;

/// Block alignment (page size)
pub const BLOCK_ALIGNMENT: usize = 4096;

/// Magic number for embedding blocks
pub const EMBEDDING_MAGIC: u32 = 0x564543_01; // "VEC" + version

/// Magic number for neighbor blocks
pub const NEIGHBOR_MAGIC: u32 = 0x4E4249_01; // "NBI" + version

// ============================================================================
// Block Header
// ============================================================================

/// Header for an embedding block
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct EmbeddingBlockHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Block version
    pub version: u32,
    /// Number of vectors in block
    pub count: u32,
    /// Vector dimension
    pub dim: u32,
    /// Offset to first vector (from start of block)
    pub data_offset: u32,
    /// Stride between vectors (bytes)
    pub stride: u32,
    /// CRC32 checksum of data
    pub checksum: u32,
    /// Reserved for future use
    pub reserved: [u32; 9],
}

impl EmbeddingBlockHeader {
    /// Create new header
    pub fn new(count: u32, dim: u32) -> Self {
        let vector_size = dim as usize * size_of::<f32>();
        let stride = align_up(vector_size, SIMD_ALIGNMENT);
        let data_offset = size_of::<Self>();
        
        Self {
            magic: EMBEDDING_MAGIC,
            version: 1,
            count,
            dim,
            data_offset: data_offset as u32,
            stride: stride as u32,
            checksum: 0,
            reserved: [0; 9],
        }
    }
    
    /// Validate header
    pub fn is_valid(&self) -> bool {
        self.magic == EMBEDDING_MAGIC && self.version <= 1
    }
    
    /// Get total block size
    pub fn block_size(&self) -> usize {
        self.data_offset as usize + (self.count as usize * self.stride as usize)
    }
}

/// Header for a neighbor block
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct NeighborBlockHeader {
    /// Magic number
    pub magic: u32,
    /// Block version
    pub version: u32,
    /// Number of nodes
    pub node_count: u32,
    /// Maximum edges per node
    pub max_edges: u32,
    /// Data offset
    pub data_offset: u32,
    /// Stride between neighbor lists
    pub stride: u32,
    /// Checksum
    pub checksum: u32,
    /// Reserved
    pub reserved: [u32; 9],
}

impl NeighborBlockHeader {
    /// Create new header
    pub fn new(node_count: u32, max_edges: u32) -> Self {
        let list_size = max_edges as usize * size_of::<u32>();
        let stride = align_up(list_size, CACHE_LINE_SIZE);
        let data_offset = size_of::<Self>();
        
        Self {
            magic: NEIGHBOR_MAGIC,
            version: 1,
            node_count,
            max_edges: max_edges,
            data_offset: data_offset as u32,
            stride: stride as u32,
            checksum: 0,
            reserved: [0; 9],
        }
    }
    
    /// Validate header
    pub fn is_valid(&self) -> bool {
        self.magic == NEIGHBOR_MAGIC && self.version <= 1
    }
    
    /// Get total block size
    pub fn block_size(&self) -> usize {
        self.data_offset as usize + (self.node_count as usize * self.stride as usize)
    }
}

// ============================================================================
// Aligned Allocation
// ============================================================================

/// Align a value up to the given alignment
#[inline]
pub const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Align a value down to the given alignment
#[inline]
pub const fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}

/// Allocate aligned memory
pub fn alloc_aligned(size: usize, alignment: usize) -> Option<NonNull<u8>> {
    if size == 0 {
        return None;
    }
    
    let layout = Layout::from_size_align(size, alignment).ok()?;
    
    unsafe {
        let ptr = alloc_zeroed(layout);
        NonNull::new(ptr)
    }
}

/// Free aligned memory
pub unsafe fn free_aligned(ptr: NonNull<u8>, size: usize, alignment: usize) {
    if let Ok(layout) = Layout::from_size_align(size, alignment) {
        dealloc(ptr.as_ptr(), layout);
    }
}

// ============================================================================
// Embedding Storage
// ============================================================================

/// Contiguous, SIMD-aligned embedding storage
pub struct EmbeddingStorage {
    /// Pointer to block
    data: NonNull<u8>,
    /// Total block size
    size: usize,
    /// Header (cached)
    header: EmbeddingBlockHeader,
}

impl EmbeddingStorage {
    /// Create new storage for given capacity
    pub fn new(capacity: usize, dim: usize) -> Option<Self> {
        let header = EmbeddingBlockHeader::new(capacity as u32, dim as u32);
        let size = align_up(header.block_size(), BLOCK_ALIGNMENT);
        
        let data = alloc_aligned(size, BLOCK_ALIGNMENT)?;
        
        // Write header
        unsafe {
            let header_ptr = data.as_ptr() as *mut EmbeddingBlockHeader;
            header_ptr.write(header.clone());
        }
        
        Some(Self { data, size, header })
    }
    
    /// Get vector by index
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.header.count as usize {
            return None;
        }
        
        let offset = self.header.data_offset as usize + index * self.header.stride as usize;
        
        unsafe {
            let ptr = self.data.as_ptr().add(offset) as *const f32;
            Some(std::slice::from_raw_parts(ptr, self.header.dim as usize))
        }
    }
    
    /// Get mutable vector by index
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [f32]> {
        if index >= self.header.count as usize {
            return None;
        }
        
        let offset = self.header.data_offset as usize + index * self.header.stride as usize;
        
        unsafe {
            let ptr = self.data.as_ptr().add(offset) as *mut f32;
            Some(std::slice::from_raw_parts_mut(ptr, self.header.dim as usize))
        }
    }
    
    /// Set vector at index
    #[inline]
    pub fn set(&mut self, index: usize, vector: &[f32]) -> bool {
        if let Some(slot) = self.get_mut(index) {
            if vector.len() == slot.len() {
                slot.copy_from_slice(vector);
                return true;
            }
        }
        false
    }
    
    /// Prefetch vector for upcoming access
    #[inline]
    pub fn prefetch(&self, index: usize) {
        if index < self.header.count as usize {
            let offset = self.header.data_offset as usize + index * self.header.stride as usize;
            
            unsafe {
                let ptr = self.data.as_ptr().add(offset);
                
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<{std::arch::x86_64::_MM_HINT_T0}>(ptr as *const i8);
                }
                
                #[cfg(target_arch = "aarch64")]
                {
                    // No stable prefetch intrinsic on aarch64 yet; intentionally no-op.
                    let _ = ptr;
                }
            }
        }
    }
    
    /// Get dimension
    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.header.count as usize
    }
    
    /// Get stride (bytes between vectors)
    pub fn stride(&self) -> usize {
        self.header.stride as usize
    }
    
    /// Get raw pointer for SIMD operations
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        unsafe {
            self.data.as_ptr().add(self.header.data_offset as usize) as *const f32
        }
    }
    
    /// Get raw mutable pointer
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        unsafe {
            self.data.as_ptr().add(self.header.data_offset as usize) as *mut f32
        }
    }
}

impl Drop for EmbeddingStorage {
    fn drop(&mut self) {
        unsafe {
            free_aligned(self.data, self.size, BLOCK_ALIGNMENT);
        }
    }
}

// Safety: The storage is internally synchronized
unsafe impl Send for EmbeddingStorage {}
unsafe impl Sync for EmbeddingStorage {}

// ============================================================================
// Neighbor Storage
// ============================================================================

/// Contiguous neighbor list storage for graph traversal
pub struct NeighborStorage {
    /// Pointer to block
    data: NonNull<u8>,
    /// Total block size
    size: usize,
    /// Header (cached)
    header: NeighborBlockHeader,
    /// Edge counts per node (atomic for concurrent updates)
    edge_counts: Vec<AtomicU32>,
}

impl NeighborStorage {
    /// Create new storage
    pub fn new(node_count: usize, max_edges: usize) -> Option<Self> {
        let header = NeighborBlockHeader::new(node_count as u32, max_edges as u32);
        let size = align_up(header.block_size(), BLOCK_ALIGNMENT);
        
        let data = alloc_aligned(size, BLOCK_ALIGNMENT)?;
        
        // Write header
        unsafe {
            let header_ptr = data.as_ptr() as *mut NeighborBlockHeader;
            header_ptr.write(header.clone());
        }
        
        // Initialize edge counts
        let edge_counts: Vec<AtomicU32> = (0..node_count)
            .map(|_| AtomicU32::new(0))
            .collect();
        
        Some(Self {
            data,
            size,
            header,
            edge_counts,
        })
    }
    /// Get neighbor list for node
    #[inline]
    pub fn get_neighbors(&self, node: usize) -> Option<&[u32]> {
        if node >= self.header.node_count as usize {
            return None;
        }
        
        let offset = self.header.data_offset as usize + node * self.header.stride as usize;
        let count = self.edge_counts[node].load(Ordering::Relaxed) as usize;
        
        unsafe {
            let ptr = self.data.as_ptr().add(offset) as *const u32;
            Some(std::slice::from_raw_parts(ptr, count.min(self.header.max_edges as usize)))
        }
    }
    
    /// Get mutable neighbor list
    #[inline]
    fn get_neighbors_mut(&mut self, node: usize) -> Option<&mut [u32]> {
        if node >= self.header.node_count as usize {
            return None;
        }
        
        let offset = self.header.data_offset as usize + node * self.header.stride as usize;
        
        unsafe {
            let ptr = self.data.as_ptr().add(offset) as *mut u32;
            Some(std::slice::from_raw_parts_mut(ptr, self.header.max_edges as usize))
        }
    }
    
    /// Add neighbor to node (thread-safe)
    pub fn add_neighbor(&self, node: usize, neighbor: u32) -> bool {
        if node >= self.header.node_count as usize {
            return false;
        }
        
        let current = self.edge_counts[node].fetch_add(1, Ordering::AcqRel);
        
        if current >= self.header.max_edges {
            self.edge_counts[node].fetch_sub(1, Ordering::Release);
            return false;
        }
        
        let offset = self.header.data_offset as usize + node * self.header.stride as usize;
        
        unsafe {
            let ptr = self.data.as_ptr().add(offset) as *mut u32;
            ptr.add(current as usize).write(neighbor);
        }
        
        true
    }
    
    /// Set all neighbors for a node (replaces existing)
    pub fn set_neighbors(&mut self, node: usize, neighbors: &[u32]) -> bool {
        let max_edges = self.header.max_edges as usize;
        if let Some(slot) = self.get_neighbors_mut(node) {
            let count = neighbors.len().min(max_edges);
            slot[..count].copy_from_slice(&neighbors[..count]);
            self.edge_counts[node].store(count as u32, Ordering::Release);
            true
        } else {
            false
        }
    }
    
    /// Prefetch neighbor list for upcoming traversal
    #[inline]
    pub fn prefetch(&self, node: usize) {
        if node < self.header.node_count as usize {
            let offset = self.header.data_offset as usize + node * self.header.stride as usize;
            
            unsafe {
                let ptr = self.data.as_ptr().add(offset);
                
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<{std::arch::x86_64::_MM_HINT_T0}>(ptr as *const i8);
                }
                
                #[cfg(target_arch = "aarch64")]
                {
                    // No stable prefetch intrinsic on aarch64 yet; intentionally no-op.
                    let _ = ptr;
                }
            }
        }
    }
    
    /// Prefetch neighbors of neighbors (two-hop prefetch)
    pub fn prefetch_neighbors(&self, embeddings: &EmbeddingStorage, node: usize) {
        if let Some(neighbors) = self.get_neighbors(node) {
            // Prefetch first few neighbors' vectors
            for &neighbor in neighbors.iter().take(4) {
                embeddings.prefetch(neighbor as usize);
            }
        }
    }
    
    /// Get edge count for node
    pub fn edge_count(&self, node: usize) -> usize {
        if node < self.edge_counts.len() {
            self.edge_counts[node].load(Ordering::Relaxed) as usize
        } else {
            0
        }
    }
    
    /// Get max edges per node
    pub fn max_edges(&self) -> usize {
        self.header.max_edges as usize
    }
    
    /// Get node count
    pub fn node_count(&self) -> usize {
        self.header.node_count as usize
    }
}

impl Drop for NeighborStorage {
    fn drop(&mut self) {
        unsafe {
            free_aligned(self.data, self.size, BLOCK_ALIGNMENT);
        }
    }
}

unsafe impl Send for NeighborStorage {}
unsafe impl Sync for NeighborStorage {}

// ============================================================================
// Hot-Path Vector Store
// ============================================================================

/// Combined embedding and neighbor storage optimized for HNSW traversal
pub struct HotPathVectorStore {
    /// Embedding storage (contiguous, SIMD-aligned)
    embeddings: EmbeddingStorage,
    /// Neighbor lists per layer
    neighbors: Vec<NeighborStorage>,
    /// Entry point for search
    entry_point: AtomicU32,
    /// Number of layers
    num_layers: usize,
}

impl HotPathVectorStore {
    /// Create new store
    pub fn new(capacity: usize, dim: usize, num_layers: usize, max_edges: usize) -> Option<Self> {
        let embeddings = EmbeddingStorage::new(capacity, dim)?;
        
        let mut neighbors = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            neighbors.push(NeighborStorage::new(capacity, max_edges)?);
        }
        
        Some(Self {
            embeddings,
            neighbors,
            entry_point: AtomicU32::new(0),
            num_layers,
        })
    }
    
    /// Get embedding
    #[inline]
    pub fn get_embedding(&self, id: usize) -> Option<&[f32]> {
        self.embeddings.get(id)
    }
    
    /// Set embedding
    pub fn set_embedding(&mut self, id: usize, vector: &[f32]) -> bool {
        self.embeddings.set(id, vector)
    }
    
    /// Get neighbors at layer
    #[inline]
    pub fn get_neighbors(&self, id: usize, layer: usize) -> Option<&[u32]> {
        self.neighbors.get(layer)?.get_neighbors(id)
    }
    
    /// Add neighbor at layer
    pub fn add_neighbor(&self, id: usize, layer: usize, neighbor: u32) -> bool {
        if let Some(storage) = self.neighbors.get(layer) {
            storage.add_neighbor(id, neighbor)
        } else {
            false
        }
    }
    
    /// Prefetch for traversal (embedding + neighbors)
    #[inline]
    pub fn prefetch_node(&self, id: usize, layer: usize) {
        self.embeddings.prefetch(id);
        if let Some(neighbors) = self.neighbors.get(layer) {
            neighbors.prefetch(id);
        }
    }
    
    /// Get entry point
    pub fn entry_point(&self) -> u32 {
        self.entry_point.load(Ordering::Relaxed)
    }
    
    /// Set entry point
    pub fn set_entry_point(&self, id: u32) {
        self.entry_point.store(id, Ordering::Release);
    }
    
    /// Get dimension
    pub fn dim(&self) -> usize {
        self.embeddings.dim()
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.embeddings.capacity()
    }
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch distance computation with prefetching
pub struct BatchDistanceComputer<'a> {
    store: &'a HotPathVectorStore,
    query: &'a [f32],
}

impl<'a> BatchDistanceComputer<'a> {
    /// Create new batch computer
    pub fn new(store: &'a HotPathVectorStore, query: &'a [f32]) -> Self {
        Self { store, query }
    }
    
    /// Compute distances to batch of candidates with prefetching
    pub fn compute_batch(&self, candidates: &[u32]) -> Vec<(u32, f32)> {
        let mut results = Vec::with_capacity(candidates.len());
        
        // Prefetch ahead
        const PREFETCH_DISTANCE: usize = 4;
        
        for (i, &id) in candidates.iter().enumerate() {
            // Prefetch future candidates
            if i + PREFETCH_DISTANCE < candidates.len() {
                self.store.embeddings.prefetch(candidates[i + PREFETCH_DISTANCE] as usize);
            }
            
            // Compute distance
            if let Some(vector) = self.store.get_embedding(id as usize) {
                let dist = l2_distance(self.query, vector);
                results.push((id, dist));
            }
        }
        
        results
    }
}

/// L2 squared distance
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alignment() {
        assert_eq!(align_up(100, 32), 128);
        assert_eq!(align_up(128, 32), 128);
        assert_eq!(align_up(129, 32), 160);
        assert_eq!(align_down(100, 32), 96);
    }
    
    #[test]
    fn test_embedding_storage() {
        let mut storage = EmbeddingStorage::new(100, 128).unwrap();
        
        // Set and get vector
        let vector: Vec<f32> = (0..128).map(|i| i as f32).collect();
        assert!(storage.set(0, &vector));
        
        let retrieved = storage.get(0).unwrap();
        assert_eq!(retrieved, vector.as_slice());
        
        // Check alignment
        let ptr = storage.as_ptr();
        assert_eq!(ptr as usize % SIMD_ALIGNMENT, 0);
    }
    
    #[test]
    fn test_neighbor_storage() {
        let mut storage = NeighborStorage::new(100, 32).unwrap();
        
        // Add neighbors
        assert!(storage.add_neighbor(0, 1));
        assert!(storage.add_neighbor(0, 5));
        assert!(storage.add_neighbor(0, 10));
        
        let neighbors = storage.get_neighbors(0).unwrap();
        assert_eq!(neighbors, &[1, 5, 10]);
        
        // Set neighbors
        storage.set_neighbors(1, &[2, 4, 6, 8]);
        let neighbors = storage.get_neighbors(1).unwrap();
        assert_eq!(neighbors, &[2, 4, 6, 8]);
    }
    
    #[test]
    fn test_hot_path_store() {
        let mut store = HotPathVectorStore::new(100, 64, 3, 16).unwrap();
        
        // Set embedding
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        assert!(store.set_embedding(0, &vector));
        
        // Set entry point
        store.set_entry_point(0);
        assert_eq!(store.entry_point(), 0);
        
        // Add neighbors
        assert!(store.add_neighbor(0, 0, 1));
        assert!(store.add_neighbor(0, 0, 2));
        
        let neighbors = store.get_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, &[1, 2]);
    }
    
    #[test]
    fn test_batch_distance() {
        let mut store = HotPathVectorStore::new(10, 4, 1, 8).unwrap();
        
        // Set some vectors
        for i in 0..10 {
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
            store.set_embedding(i, &vector);
        }
        
        let query = vec![0.0, 1.0, 2.0, 3.0];
        let computer = BatchDistanceComputer::new(&store, &query);
        
        let candidates: Vec<u32> = (0..5).collect();
        let results = computer.compute_batch(&candidates);
        
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0); // First candidate
    }
}
