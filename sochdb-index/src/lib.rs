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

//! SochDB Index Layer
//!
//! Novel indexing structures for TOON-native data access.
//!
//! ## Novel Data Structures
//!
//! - **Learned Sparse Index** (`learned_index`): O(1) expected lookups using linear regression
//!   instead of B-Trees. For well-distributed data (timestamps, sequential IDs), this provides
//!   10-100× faster lookups than traditional indexes.
//!
//! - **Trie-Columnar Hybrid** (`trie_columnar`): Combines path trie traversal O(|path|) with
//!   columnar storage for efficient TOON document access. Path lookup depends on path length,
//!   NOT data size.
//!
//! ## Vector Indices
//!
//! This module provides two vector index implementations:
//!
//! - **HNSW** (`hnsw`): Hierarchical Navigable Small World graphs for ANN search.
//!   Good for general purpose vector search with low latency.
//!
//! - **Vamana** (`vamana`): DiskANN-style single-layer graph with Product Quantization.
//!   Optimized for massive scale (10M+ vectors) with 32x memory reduction.
//!
//! ## Embedding Module
//!
//! The `embedding` module provides a complete embedding pipeline:
//!
//! - **Providers**: Local ONNX inference (offline) and OpenAI API integration
//! - **Pipeline**: Batched processing with background workers
//! - **Storage**: Persistent embedding storage with PQ compression
//! - **Normalization**: SIMD-optimized L2 normalization
//!
//! ## Product Quantization
//!
//! The `product_quantization` module provides 32x compression for embeddings:
//! - 384-dim vector (1536 bytes as f32) → 48 bytes as PQ codes
//! - 10M vectors: 15 GB → 480 MB

// Novel data structures
// LearnedSparseIndex is now in sochdb-core to avoid cyclic dependencies
pub use sochdb_core::learned_index;
pub mod adaptive_ef; // Adaptive ef_construction with quality feedback (Task 9)
pub mod hybrid_learned_index;
pub mod trie_columnar; // PGM-style learned index with B-tree fallback (mm.md Task 7)

// Vector indexing
pub mod aosoa_tiles; // AoSoA vector tiles for cache-line aligned SIMD access (P0 optimization)
pub mod compression;
pub mod csr_graph; // CSR packed adjacency for cache-efficient traversal (P0 optimization)
pub mod distance_cache; // LRU distance computation cache (Task 11)
pub mod edge_delta_buffer; // Batched edge delta application (Task 7)
pub mod embedding;
pub mod hnsw;
pub mod hnsw_pq;
pub mod hnsw_staged; // Staged parallel HNSW construction with waves + deferred backedges
pub mod hot_buffer_hnsw; // Hot buffer + background flush for ultra-fast inserts
pub mod buffered_hnsw; // Buffered HNSW with delta buffer (Task 6)
pub mod internal_id; // Dense u32 ID mapping for cache-friendly traversal (P0 optimization)
pub mod johnson_lindenstrauss; // Low-dimensional projection pre-filter (Task 5)
pub mod lockfree_hnsw; // Lock-free HNSW with CAS operations (mm.md Task 5)
pub mod metrics;
pub mod node_ordering; // Locality-driven node ordering: BFS/RCM/Hilbert (P0 optimization)
pub mod optimized_search; // Optimized HNSW search with CSR + internal IDs + batched expansion
pub mod unified_search; // Unified production search path: CSR + bitmap + AoSoA + batched expansion
pub mod parallel_waves; // Wave-based parallel graph construction (Task 3)
pub mod persistence;
pub mod product_quantization;
pub mod scratch_buffers; // Thread-local scratch buffer pool (Task 10)
pub mod storage; // Contiguous node storage (Task 4)
pub mod unified_quant; // Unified quantization contract: F32/F16/BF16/I8/PQ/BPS (P0 optimization)
pub mod vamana;
pub mod vector;
pub mod vector_arena; // Arena-based vector storage (Task 3)
pub mod vector_quantized;
pub mod vector_simd;
pub mod vector_storage;

// Performance optimization modules
pub mod atomic_entry_point; // Lock-free entry point + max_layer via packed atomic CAS
pub mod prefetch_scan;
pub mod predicated_simd; // Predicated SIMD kernels with masked operations (P0 optimization)
pub mod simd_batch_distance; // SIMD batch distance with tiled processing (Task 8)
pub mod simd_scan; // SIMD-accelerated column scans (Task 5)

// Performance optimization modules (jj.txt optimizations)
pub mod simd_distance; // SIMD Distance Kernels for HNSW (AVX2/AVX-512/NEON)
pub mod contiguous_graph; // Contiguous Graph Memory Layout (flattened neighbor lists)
pub mod zero_copy_batch;

// Test modules
#[cfg(test)]
pub mod rng_optimization_tests; // Zero-Copy Batch Ingest (arena-based storage)
pub mod profiling; // End-to-end profiling (SOCHDB_PROFILING=1)

// FFI bindings for Python and other languages
pub mod ffi;

// Arrow IPC for cross-language zero-copy data exchange
pub mod arrow_ipc;

// Re-exports for novel structures (LearnedSparseIndex comes from sochdb-core)
pub use hybrid_learned_index::{
    HybridIndexBuilder, HybridIndexConfig, HybridLearnedIndex, IndexStats, LinearSegment,
};
pub use sochdb_core::learned_index::{
    LearnedIndexStats, LearnedSparseIndex, LookupResult, PiecewiseLearnedIndex,
};
pub use trie_columnar::{
    ArraySchema, ColumnData, ColumnId, ColumnRef, ColumnStore, FieldType, PathResolution, TchStats,
    TrieColumnarHybrid, TrieNode, TrieNodeType,
};

// Re-exports for vector indexing
pub use compression::{CompressionLevel, QuantizedVectorI8, StoredVector};
pub use embedding::{
    EmbeddingError, EmbeddingIntegration, EmbeddingPipeline, EmbeddingProvider, EmbeddingRegistry,
    EmbeddingRequest, EmbeddingStorage, EmbeddingStorageConfig, IntegrationConfig,
    IntegrationError, LocalEmbeddingConfig, LocalEmbeddingProvider, MockEmbeddingProvider,
    PipelineConfig, SemanticSearchResult,
};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats, MemoryStats};
pub use hnsw_staged::{StagedBuilder, StagedConfig, StagedStats};
pub use hot_buffer_hnsw::{HotBufferHnsw, HotBufferConfig, HotBufferStats};
pub use atomic_entry_point::{AtomicNavigationState, AtomicNavigationStateU128};
pub use hnsw_pq::{ADCTable, PQSearchConfig, PQSearchResult, PQVectorStore};
pub use internal_id::{IdMapper, InternalId, VisitedBitmap, INVALID_INTERNAL_ID, MAX_INTERNAL_ID};
pub use csr_graph::{CsrGraph, CsrGraphBuilder, CsrLayer, CsrGraphStats, InternalSearchCandidate};
pub use buffered_hnsw::{BufferedHnsw, BufferedHnswConfig, BufferedHnswStats, BufferStats};
pub use product_quantization::{DistanceTable, PQCodebooks, PQCodes};
pub use unified_quant::{QuantLevel, UnifiedQuantizedVector, QuantPipelineConfig, PipelineStage, StageCandidates, UnifiedScorer};
pub use node_ordering::{NodeOrderer, NodePermutation, OrderingStats, OrderingStrategy, reorder_csr_graph};
pub use aosoa_tiles::{VectorTile, TiledVectorStore, TiledStoreStats, AlignedBlock, DEFAULT_TILE_SIZE};
pub use vamana::{VamanaConfig, VamanaIndex, VamanaStats};
pub use vector::{DistanceMetric, Embedding, VectorIndex};
pub use vector_quantized::{Precision, QuantizedVector};
pub use vector_storage::{MemoryVectorStorage, MmapVectorStorage, VectorStorage};

// Re-exports for performance optimization modules
pub use prefetch_scan::{
    CACHE_LINE_SIZE, DEFAULT_PREFETCH_DISTANCE, GenericPrefetchScanner, PrefetchHint,
    PrefetchScanner, ScatterPrefetcher, aggregates as prefetch_aggregates, prefetch,
};
pub use predicated_simd::{PredicatedSimd, PredicatedSimd8, SimdMask};
pub use simd_scan::{BitVec, SimdLevel, SimdScanner, scalar as simd_scalar};
