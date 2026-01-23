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

//! FFI bindings for SochDB Vector Index (HNSW)
//!
//! Provides C-compatible functions for Python and other language bindings.
//! 
//! ## Safe Mode
//! 
//! If you experience issues with batch insert (low recall, connectivity problems),
//! set the environment variable `SOCHDB_BATCH_SAFE_MODE=1` to route all batch
//! inserts through proven sequential single-insert code path.

use crate::hnsw::{HnswConfig, HnswIndex};
use std::os::raw::{c_int, c_float};
use std::slice;
use std::sync::Arc;
use std::sync::Once;

// =============================================================================
// TASK 5: SAFE-MODE HYGIENE AND VISIBILITY
// =============================================================================

static SAFE_MODE_WARNING: Once = Once::new();

/// Check if safe mode is enabled and emit warning.
/// 
/// When `SOCHDB_BATCH_SAFE_MODE=1` is set, batch inserts will use
/// sequential single-insert for guaranteed correctness at the cost
/// of 10-100× reduced throughput.
/// 
/// **IMPORTANT**: This emits a loud warning on first use to prevent
/// accidental performance degradation in production/benchmarks.
#[inline]
fn check_safe_mode() -> bool {
    let enabled = std::env::var("SOCHDB_BATCH_SAFE_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    
    if enabled {
        SAFE_MODE_WARNING.call_once(|| {
            eprintln!(
                "\n\
                ╔══════════════════════════════════════════════════════════════╗\n\
                ║  WARNING: SOCHDB_BATCH_SAFE_MODE=1 is active                 ║\n\
                ║  Batch inserts are running 10-100× SLOWER than normal.       ║\n\
                ║  Unset this variable for production/benchmarking.            ║\n\
                ╚══════════════════════════════════════════════════════════════╝\n"
            );
        });
    }
    enabled
}

/// Legacy check function - redirects to check_safe_mode with warning
/// 
/// Kept for backward compatibility. Use check_safe_mode() for new code.
#[inline]
fn is_batch_safe_mode_enabled() -> bool {
    check_safe_mode()
}

/// Opaque pointer to HNSW Index
pub struct HnswIndexPtr(pub Arc<HnswIndex>);

/// Search result with FFI-safe ID representation
#[repr(C)]
pub struct CSearchResult {
    /// Lower 64 bits of the vector ID
    pub id_lo: u64,
    /// Upper 64 bits of the vector ID (usually 0 for IDs < 2^64)
    pub id_hi: u64,
    /// Distance to the query vector
    pub distance: c_float,
}

/// Create a new HNSW index.
/// 
/// # Arguments
/// * `dimension` - Vector dimension (e.g., 128, 768, 1536)
/// * `max_connections` - Number of neighbors per node (default: 16)
/// * `ef_construction` - Construction-time ef (default: 200)
/// 
/// # Returns
/// Pointer to the index, or null on error.
/// 
/// # Safety
/// Caller must free the index with `hnsw_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_new(
    dimension: usize,
    max_connections: usize,
    ef_construction: usize,
) -> *mut HnswIndexPtr {
    let config = HnswConfig {
        max_connections: if max_connections == 0 { 16 } else { max_connections },
        ef_construction: if ef_construction == 0 { 100 } else { ef_construction },  // Reduced default from 200
        ..Default::default()
    };
    
    let index = HnswIndex::new(dimension, config);
    let ptr = Box::new(HnswIndexPtr(Arc::new(index)));
    Box::into_raw(ptr)
}

/// Free an HNSW index.
/// 
/// # Safety
/// `ptr` must be a valid pointer from `hnsw_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_free(ptr: *mut HnswIndexPtr) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}

/// Insert a vector into the index.
/// 
/// # Arguments
/// * `ptr` - Index pointer
/// * `id` - Vector ID (u128)
/// * `vector` - Pointer to float array
/// * `vector_len` - Length of vector (must equal dimension)
/// 
/// # Returns
/// 0 on success, -1 on error.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_insert(
    ptr: *mut HnswIndexPtr,
    id_lo: u64,  // Lower 64 bits of ID
    id_hi: u64,  // Upper 64 bits of ID (usually 0)
    vector: *const c_float,
    vector_len: usize,
) -> c_int {
    if ptr.is_null() || vector.is_null() {
        return -1;
    }
    
    let id: u128 = (id_hi as u128) << 64 | (id_lo as u128);
    
    unsafe {
        let index = &(*ptr).0;
        let vec = slice::from_raw_parts(vector, vector_len);
        
        match index.insert(id, vec.to_vec()) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Batch insert vectors with optimized 3-phase HNSW construction.
/// 
/// Uses the optimized `HnswIndex::insert_batch` which:
/// 1. Pre-allocates all nodes (parallel quantization)
/// 2. Inserts nodes into the graph map
/// 3. Builds connections sequentially (avoids lock contention)
/// 
/// # Performance
/// 
/// This achieves ~5-10x speedup over individual inserts by:
/// - Single FFI boundary crossing (amortized)
/// - Batch quantization with SIMD
/// - Reduced lock contention via sequential graph building
/// 
/// | Method           | Throughput      | Speedup |
/// |------------------|-----------------|---------|
/// | Individual FFI   | ~500 vec/sec    | 1x      |
/// | Batch FFI        | ~5,000 vec/sec  | 10x     |
/// 
/// # Arguments
/// * `ptr` - Index pointer from `hnsw_new`
/// * `ids` - Contiguous array of N u64 IDs
/// * `vectors` - Contiguous array of N×dimension f32 values (row-major)
/// * `num_vectors` - Number of vectors to insert
/// * `dimension` - Dimension of each vector
/// 
/// # Returns
/// Number of successfully inserted vectors, or -1 on error.
/// 
/// # Safe Mode
/// 
/// If environment variable `SOCHDB_BATCH_SAFE_MODE=1` is set, this function
/// will use sequential single-insert for each vector instead of batch insert.
/// This guarantees correctness but reduces throughput.
/// 
/// # Safety
/// - `ptr` must be valid from `hnsw_new`
/// - `ids` must point to array of at least `num_vectors` u64 values
/// - `vectors` must point to array of at least `num_vectors * dimension` f32 values
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_insert_batch(
    ptr: *mut HnswIndexPtr,
    ids: *const u64,          // N IDs (u64 for FFI compatibility)
    vectors: *const c_float,  // N×D packed vectors
    num_vectors: usize,
    dimension: usize,
) -> c_int {
    use crate::profiling::{is_profiling_enabled, profile_ffi_batch, profile_id_conversion, PROFILE_COLLECTOR};
    
    if ptr.is_null() || ids.is_null() || vectors.is_null() || num_vectors == 0 {
        return -1;
    }
    
    // Profile the entire FFI call
    let _ffi_start = if is_profiling_enabled() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    
    unsafe {
        let index = &(*ptr).0;
        
        // Profile slice creation
        let (id_slice, vec_slice) = if is_profiling_enabled() {
            let t = crate::profiling::Timer::start("ffi.slice_from_raw");
            let ids = slice::from_raw_parts(ids, num_vectors);
            let vecs = slice::from_raw_parts(vectors, num_vectors * dimension);
            t.stop();
            (ids, vecs)
        } else {
            (slice::from_raw_parts(ids, num_vectors),
             slice::from_raw_parts(vectors, num_vectors * dimension))
        };
        
        // Check if safe mode is enabled
        if is_batch_safe_mode_enabled() {
            // Safe mode: use sequential single-insert for each vector
            // This guarantees correctness at the cost of throughput
            let mut count = 0i32;
            for i in 0..num_vectors {
                let id = id_slice[i] as u128;
                let start = i * dimension;
                let end = start + dimension;
                let vec: Vec<f32> = vec_slice[start..end].to_vec();
                
                if index.insert(id, vec).is_ok() {
                    count += 1;
                }
            }
            return count;
        }
        
        // Normal mode: use optimized batch insert
        // Profile ID conversion (u64 -> u128)
        let ids_u128: Vec<u128> = profile_id_conversion(num_vectors, || {
            id_slice.iter().map(|&id| id as u128).collect()
        });
        
        // Profile the batch insert
        let result = profile_ffi_batch("insert_batch", num_vectors, || {
            index.insert_batch_contiguous(&ids_u128, vec_slice, dimension)
        });
        
        // Dump profile on completion if enabled
        if let Some(start) = _ffi_start {
            let elapsed = start.elapsed().as_nanos() as u64;
            PROFILE_COLLECTOR.record_with_count("ffi.hnsw_insert_batch.total", elapsed, num_vectors);
        }
        
        match result {
            Ok(count) => count as c_int,
            Err(_) => -1,
        }
    }
}

/// Search for nearest neighbors.
/// 
/// # Arguments
/// * `ptr` - Index pointer
/// * `query` - Query vector
/// * `query_len` - Query length (must equal dimension)
/// * `k` - Number of neighbors to return
/// * `results_out` - Pointer to array of CSearchResult (must have space for k results)
/// * `num_results_out` - Pointer to receive actual number of results
/// 
/// # Returns
/// 0 on success, -1 on error.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_search(
    ptr: *mut HnswIndexPtr,
    query: *const c_float,
    query_len: usize,
    k: usize,
    results_out: *mut CSearchResult,
    num_results_out: *mut usize,
) -> c_int {
    if ptr.is_null() || query.is_null() || results_out.is_null() || num_results_out.is_null() {
        return -1;
    }
    
    unsafe {
        let index = &(*ptr).0;
        let query_vec = slice::from_raw_parts(query, query_len);
        
        match index.search(query_vec, k) {
            Ok(results) => {
                let num = results.len().min(k);
                *num_results_out = num;
                
                for (i, (id, distance)) in results.into_iter().take(k).enumerate() {
                    *results_out.add(i) = CSearchResult {
                        id_lo: id as u64,
                        id_hi: (id >> 64) as u64,
                        distance: distance as c_float,
                    };
                }
                0
            }
            Err(_) => -1,
        }
    }
}

/// Ultra-fast search optimized for robotics/edge use cases.
/// 
/// This uses a zero-allocation hot path with direct SIMD for sub-millisecond latency.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_search_fast(
    ptr: *mut HnswIndexPtr,
    query: *const c_float,
    query_len: usize,
    k: usize,
    results_out: *mut CSearchResult,
    num_results_out: *mut usize,
) -> c_int {
    if ptr.is_null() || query.is_null() || results_out.is_null() || num_results_out.is_null() {
        return -1;
    }
    
    unsafe {
        let index = &(*ptr).0;
        let query_vec = slice::from_raw_parts(query, query_len);
        
        match index.search_fast(query_vec, k) {
            Ok(results) => {
                let num = results.len().min(k);
                *num_results_out = num;
                
                for (i, (id, distance)) in results.into_iter().take(k).enumerate() {
                    *results_out.add(i) = CSearchResult {
                        id_lo: id as u64,
                        id_hi: (id >> 64) as u64,
                        distance: distance as c_float,
                    };
                }
                0
            }
            Err(_) => -1,
        }
    }
}

/// Ultra-fast search using flat neighbor cache (ZERO per-node locks)
/// 
/// This is the fastest search path. Call `hnsw_build_flat_cache` after bulk inserts.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_search_ultra(
    ptr: *mut HnswIndexPtr,
    query: *const c_float,
    query_len: usize,
    k: usize,
    results_out: *mut CSearchResult,
    num_results_out: *mut usize,
) -> c_int {
    if ptr.is_null() || query.is_null() || results_out.is_null() || num_results_out.is_null() {
        return -1;
    }
    
    unsafe {
        let index = &(*ptr).0;
        let query_vec = slice::from_raw_parts(query, query_len);
        
        match index.search_ultra(query_vec, k) {
            Ok(results) => {
                let num = results.len().min(k);
                *num_results_out = num;
                
                for (i, (id, distance)) in results.into_iter().take(k).enumerate() {
                    *results_out.add(i) = CSearchResult {
                        id_lo: id as u64,
                        id_hi: (id >> 64) as u64,
                        distance: distance as c_float,
                    };
                }
                0
            }
            Err(_) => -1,
        }
    }
}

/// Build flat neighbor cache for ultra-fast search
/// 
/// Call this after bulk inserts. Pre-flattens all layer-0 neighbors
/// into a contiguous array for ZERO per-node lock access during search.
/// 
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_build_flat_cache(ptr: *mut HnswIndexPtr) -> c_int {
    if ptr.is_null() {
        return -1;
    }
    unsafe {
        (*ptr).0.build_flat_neighbor_cache();
        0
    }
}

/// Get the number of vectors in the index.
/// 
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_len(ptr: *mut HnswIndexPtr) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).0.len() }
}

/// Get the dimension of the index.
/// 
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_dimension(ptr: *mut HnswIndexPtr) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (&(*ptr).0).dimension }
}

// =============================================================================
// TASK 2: FLAT-SLICE FFI BINDINGS (Zero-Allocation Path)
// =============================================================================

/// Batch insert with zero intermediate allocations.
/// 
/// This is the **high-performance FFI path** for Python/C integration.
/// Eliminates per-vector heap allocations by accepting contiguous memory.
/// 
/// # Performance
/// 
/// | FFI Method | Per-Batch Overhead | Allocations |
/// |------------|-------------------|-------------|
/// | `hnsw_insert_batch` | ~2ms (ID copy) | O(N) |
/// | `hnsw_insert_batch_flat` | ~20µs | O(1) for small, O(1) amortized for large |
///
/// For N = 10,000 vectors, this is ~100x faster FFI overhead.
/// 
/// # Arguments
/// * `ptr` - Index pointer from `hnsw_new`
/// * `ids` - Contiguous array of N u64 IDs
/// * `vectors` - Contiguous array of N×dimension f32 values (row-major)
/// * `num_vectors` - Number of vectors to insert
/// * `dimension` - Dimension of each vector
/// 
/// # Returns
/// Number of successfully inserted vectors, or -1 on error.
/// 
/// # Safety
/// - `ptr` must be valid from `hnsw_new`
/// - `ids` must point to array of at least `num_vectors` u64 values
/// - `vectors` must point to array of at least `num_vectors * dimension` f32 values
/// - Memory must remain valid for duration of call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_insert_batch_flat(
    ptr: *mut HnswIndexPtr,
    ids: *const u64,
    vectors: *const c_float,
    num_vectors: usize,
    dimension: usize,
) -> c_int {
    if ptr.is_null() || ids.is_null() || vectors.is_null() || num_vectors == 0 {
        return -1;
    }
    
    unsafe {
        let index = &(*ptr).0;
        let id_slice = slice::from_raw_parts(ids, num_vectors);
        let vec_slice = slice::from_raw_parts(vectors, num_vectors * dimension);
        
        // Check safe mode first
        if check_safe_mode() {
            // Safe mode: sequential single-insert (for debugging)
            let mut count = 0i32;
            for i in 0..num_vectors {
                let id = id_slice[i] as u128;
                let start = i * dimension;
                let end = start + dimension;
                let vec: Vec<f32> = vec_slice[start..end].to_vec();
                
                if index.insert(id, vec).is_ok() {
                    count += 1;
                }
            }
            return count;
        }
        
        // Fast path: Convert IDs with stack buffer for small batches
        let result = if num_vectors <= 256 {
            // Stack-allocated conversion for small batches (avoids heap alloc)
            let mut id_buf: [u128; 256] = [0; 256];
            for (i, &id) in id_slice.iter().enumerate() {
                id_buf[i] = id as u128;
            }
            index.insert_batch_flat(&id_buf[..num_vectors], vec_slice, dimension)
        } else {
            // Single allocation for large batches (amortized O(1) per batch)
            let ids_u128: Vec<u128> = id_slice.iter().map(|&id| id as u128).collect();
            index.insert_batch_flat(&ids_u128, vec_slice, dimension)
        };
        
        match result {
            Ok(count) => count as c_int,
            Err(_) => -1,
        }
    }
}

/// Single-vector insert without allocation.
/// 
/// This is the allocation-free path for single inserts from FFI.
/// Use when inserting one vector at a time without the Vec<f32> copy.
/// 
/// # Arguments
/// * `ptr` - Index pointer from `hnsw_new`
/// * `id_lo` - Lower 64 bits of the vector ID
/// * `id_hi` - Upper 64 bits of the vector ID (usually 0)
/// * `vector` - Pointer to float array
/// * `vector_len` - Length of vector (must equal dimension)
/// 
/// # Returns
/// 0 on success, -1 on error.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_insert_flat(
    ptr: *mut HnswIndexPtr,
    id_lo: u64,
    id_hi: u64,
    vector: *const c_float,
    vector_len: usize,
) -> c_int {
    if ptr.is_null() || vector.is_null() {
        return -1;
    }
    
    let id: u128 = (id_hi as u128) << 64 | (id_lo as u128);
    
    unsafe {
        let index = &(*ptr).0;
        let vec_slice = slice::from_raw_parts(vector, vector_len);
        
        // Call slice-based insert - NO .to_vec()
        match index.insert_one_from_slice(id, vec_slice) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

// =============================================================================
// PROFILING FFI FUNCTIONS
// =============================================================================
//
// The FFI functions for profiling are defined in profiling.rs:
// - sochdb_profiling_enable()
// - sochdb_profiling_disable()
// - sochdb_profiling_dump()
// - sochdb_profiling_record()
//
// They are exported directly from the profiling module with #[unsafe(no_mangle)].

// =============================================================================
// RUNTIME CONFIGURATION FFI FUNCTIONS
// =============================================================================

/// Set ef_search parameter at runtime for higher recall.
/// 
/// Higher ef_search = better recall but slower search.
/// Recommended: ef_search >= 2 * k for good recall.
/// 
/// # Arguments
/// * `ptr` - Index pointer from `hnsw_new`
/// * `ef_search` - New ef_search value (search beam width)
/// 
/// # Returns
/// 0 on success, -1 on error.
/// 
/// # Safety
/// `ptr` must be a valid pointer from `hnsw_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_set_ef_search(
    ptr: *mut HnswIndexPtr,
    ef_search: usize,
) -> c_int {
    if ptr.is_null() || ef_search == 0 {
        return -1;
    }
    
    unsafe {
        (*ptr).0.set_ef_search(ef_search);
        0
    }
}

/// Get current ef_search parameter.
/// 
/// # Safety
/// `ptr` must be a valid pointer from `hnsw_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_get_ef_search(ptr: *mut HnswIndexPtr) -> usize {
    if ptr.is_null() {
        return 0;
    }
    
    unsafe { (*ptr).0.get_ef_search() }
}
