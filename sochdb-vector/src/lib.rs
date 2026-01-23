//! Streaming Elimination Vector Search Engine
//! 
//! A silicon-sympathetic, CPU-first ANN design that avoids pointer chasing,
//! supports updates, and makes recall a controllable knob.
//!
//! # Architecture
//!
//! The engine uses two complementary views for candidate generation:
//! - **RDF (Rare-Dominant Fingerprint)**: IR-style inverted lists for precision
//! - **BPS (Block Projection Sketch)**: Dense-friendly streaming scans for recall
//!
//! Final ranking uses int8 dot products with outlier-aware correction.
//!
//! # SIMD Acceleration
//!
//! The engine uses pure Rust SIMD implementations for critical operations:
//! - BPS scans: AVX2 (32x) / AVX512 (64x) / NEON (16x) speedup
//! - int8 dot products: 8x speedup on AVX2
//! - Visibility checks: 4x speedup on AVX2 / 2x on NEON
//!
//! All SIMD code is written in Rust using `core::arch` intrinsics,
//! enabling cross-function inlining and eliminating FFI overhead.
//!
//! See [`simd`] module for the pure Rust implementations and
//! [`dispatch`] module for runtime CPU detection and kernel dispatch.

pub mod catalog;
pub mod config;
pub mod dispatch;
pub mod error;
pub mod filter;
pub mod bmi2_paths; // P1: BMI2 fast paths with PEXT/PDEP for bit packing/unpacking
pub mod lsm;
pub mod outlier_encoding; // P1: Optimized outlier encoding with bitvector/sorted list hybrid
pub mod query;
pub mod rerank;
pub mod rotation;
pub mod search_plan; // P1: Quantization-aware search planning with SLA optimization
pub mod segment;
pub mod shard_topology; // P1: Shard-first ANN routing to minimize fan-out
pub mod numa_alloc; // P1: NUMA-aware memory allocation
pub mod compaction; // P1: Per-shard compaction isolation
pub mod filter_pushdown; // P2: Filter/projection pushdown plugin API
pub mod jit_ir; // P2: JIT compilation for filter expressions
pub mod simd;
pub mod types;

// Performance optimization modules (jj.txt optimizations)
pub mod batch_segment_writer; // True Batch SegmentWriter Ingest with contiguous API
pub mod async_rotation; // Async Rotation Pipeline (background Walsh-Hadamard)
pub mod simd_hadamard; // SIMD-Accelerated Walsh-Hadamard Transform
pub mod async_lsm; // Non-Blocking LSM Sealing with WAL durability
pub mod lazy_segment; // Lazy BPS/RDF/Rerank Construction (build-on-first-query)

// Hybrid search and multi-vector modules (Task 4-6)
pub mod bm25; // BM25 scoring for lexical search
pub mod inverted_index; // Inverted index for keyword search
pub mod hybrid; // Hybrid search with RRF fusion
pub mod multi_vector; // Multi-vector documents with aggregation
pub mod tombstones; // Tombstone-based logical deletion

// Task implementations for operationalized vector search
pub mod cost_model;           // Task 1: Cost model budgets
pub mod guarantee_ladder;     // Task 2: Guarantee ladder modes
pub mod list_bounds;          // Task 3: Cosine/Dot list bounds
pub mod compressed_routing;   // Task 4: Routing cache-resident
pub mod quantization_calibration; // Task 5: Quantization error calibration
pub mod portable_simd;        // Task 6: Portable SIMD kernels
pub mod filter_indexing;      // Task 7: Cardinality-aware filter indexing
pub mod ssd_rerank;           // Task 8: SSD rerank executor
pub mod segment_compaction;   // Task 9: Drift-resilient compaction
pub mod query_telemetry;      // Task 10: Per-query telemetry
pub mod hot_path_layout;      // Task 16: Vector hot-path layout

// Re-export main types
pub use config::EngineConfig;
pub use error::{Error, Result};
pub use query::QueryEngine;
pub use search_plan::{SearchPlan, SearchPlanner, SearchSLA, OptimizationMode, PipelineStage, StageQuantLevel};
pub use shard_topology::{ShardTopology, ShardRouter, Centroid, RoutingDecision, TopologyConfig};
pub use segment::Segment;
pub use types::*;

// Re-export SIMD dispatch for cross-crate usage
pub use dispatch::{
    BpsScanDispatcher, DotI8Dispatcher, VisibilityDispatcher,
    CpuFeatures, SimdLevel,
    cpu_features, simd_level, simd_available, dispatch_info,
};

// Re-export hybrid search types
pub use bm25::{BM25Config, BM25Scorer, BM25Stats, tokenize, tokenize_minimal};
pub use inverted_index::{InvertedIndex, InvertedIndexBuilder, PostingList, Posting};
pub use hybrid::{RRFConfig, RRFFusion, HybridSearchEngine, SearchResult, ComponentScores};
pub use multi_vector::{
    AggregationMethod, DocumentScore, MultiVectorMapping, MultiVectorAggregator,
    MultiVectorDocument, MultiVectorConfig, MultiVectorError,
};
pub use tombstones::{TombstoneManager, TombstoneFilter, TombstoneError};
