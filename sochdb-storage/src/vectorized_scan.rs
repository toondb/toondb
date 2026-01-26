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

//! SIMD-Accelerated Vectorized Scan Engine (Recommendation 2)
//!
//! ## Problem
//!
//! Current scan implementation is row-at-a-time:
//! ```ignore
//! for entry in self.data.iter() {  // DashMap iteration - pointer chasing
//!     if let Some(v) = entry.value().read_at(snapshot_ts, current_txn_id)
//!         && let Some(value) = &v.value  // Option unwrapping
//!     {
//!         results.push((key.clone(), value.clone())); // Heap allocs
//!     }
//! }
//! ```
//!
//! This has:
//! - DashMap iterator overhead (~20ns per entry)
//! - MVCC version chain traversal per row
//! - Clone allocations in hot path
//!
//! ## Solution
//!
//! Vectorized execution: process 1024+ rows in a batch, amortizing overhead.
//!
//! ## Performance Analysis
//!
//! Vectorized execution model:
//! - **Batch size B = 1024 rows**
//! - **Per-batch overhead**: O(1) iterator setup + O(B) SIMD operations
//! - **Amortized cost**: `(setup + B×simd_op) / B ≈ simd_op` for large B
//!
//! SIMD comparison (AVX-512):
//! ```text
//! Scalar: 100 cycles × 1000 rows = 100,000 cycles
//! SIMD: 100 cycles × (1000/16) = 6,250 cycles (16x speedup)
//! ```
//!
//! Memory bandwidth calculation:
//! - Current: 300 bytes/row × 1M rows = 300 MB, random access
//! - Proposed: 60 bytes/row × 1M rows = 60 MB, sequential scan
//! - RAM bandwidth: ~50 GB/s → theoretical 833M rows/sec
//!
//! ## Expected Improvement
//!
//! 10-20x scan throughput improvement

use std::sync::atomic::{AtomicUsize, Ordering};

/// Default batch size for vectorized operations
/// 1024 rows fits comfortably in L2 cache (~256KB with 256-byte rows)
pub const DEFAULT_BATCH_SIZE: usize = 1024;

/// Minimum batch size (below this, scalar is faster)
pub const MIN_BATCH_SIZE: usize = 64;

/// Maximum batch size (above this, memory pressure increases)
pub const MAX_BATCH_SIZE: usize = 8192;

// =============================================================================
// Column Vectors for SIMD Operations
// =============================================================================

/// Typed column vector for SIMD-friendly access
/// 
/// Each variant stores contiguous values for vectorized operations.
#[derive(Debug, Clone)]
pub enum ColumnVector {
    /// Boolean column (bit-packed for SIMD)
    Bool(Vec<bool>),
    /// 64-bit signed integers (SIMD-friendly)
    Int64(Vec<i64>),
    /// 64-bit unsigned integers
    UInt64(Vec<u64>),
    /// 64-bit floats
    Float64(Vec<f64>),
    /// Variable-length strings (Arrow-style: offsets + data)
    String {
        offsets: Vec<u32>,
        data: Vec<u8>,
    },
    /// Binary data (Arrow-style: offsets + data)
    Binary {
        offsets: Vec<u32>,
        data: Vec<u8>,
    },
    /// Null bitmap for any column (packed bits)
    Null(Vec<u64>),
}

impl ColumnVector {
    /// Create empty column vector of specified type
    pub fn new_int64(capacity: usize) -> Self {
        ColumnVector::Int64(Vec::with_capacity(capacity))
    }

    pub fn new_float64(capacity: usize) -> Self {
        ColumnVector::Float64(Vec::with_capacity(capacity))
    }

    pub fn new_string(capacity: usize) -> Self {
        ColumnVector::String {
            offsets: Vec::with_capacity(capacity + 1),
            data: Vec::with_capacity(capacity * 32), // Assume avg 32 bytes per string
        }
    }

    /// Get length of vector
    pub fn len(&self) -> usize {
        match self {
            ColumnVector::Bool(v) => v.len(),
            ColumnVector::Int64(v) => v.len(),
            ColumnVector::UInt64(v) => v.len(),
            ColumnVector::Float64(v) => v.len(),
            ColumnVector::String { offsets, .. } => offsets.len().saturating_sub(1),
            ColumnVector::Binary { offsets, .. } => offsets.len().saturating_sub(1),
            ColumnVector::Null(v) => v.len() * 64,
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            ColumnVector::Bool(v) => v.len(),
            ColumnVector::Int64(v) => v.len() * 8,
            ColumnVector::UInt64(v) => v.len() * 8,
            ColumnVector::Float64(v) => v.len() * 8,
            ColumnVector::String { offsets, data } => offsets.len() * 4 + data.len(),
            ColumnVector::Binary { offsets, data } => offsets.len() * 4 + data.len(),
            ColumnVector::Null(v) => v.len() * 8,
        }
    }

    /// Sum for Int64 column (SIMD-accelerated)
    #[cfg(target_arch = "x86_64")]
    pub fn sum_i64(&self) -> Option<i64> {
        match self {
            ColumnVector::Int64(values) => {
                if values.is_empty() {
                    return Some(0);
                }
                
                // Use SIMD for large vectors
                if values.len() >= 16 {
                    Some(simd_sum_i64(values))
                } else {
                    Some(values.iter().sum())
                }
            }
            _ => None,
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn sum_i64(&self) -> Option<i64> {
        match self {
            ColumnVector::Int64(values) => Some(values.iter().sum()),
            _ => None,
        }
    }

    /// Sum for Float64 column (SIMD-accelerated)
    #[cfg(target_arch = "x86_64")]
    pub fn sum_f64(&self) -> Option<f64> {
        match self {
            ColumnVector::Float64(values) => {
                if values.is_empty() {
                    return Some(0.0);
                }
                
                if values.len() >= 8 {
                    Some(simd_sum_f64(values))
                } else {
                    Some(values.iter().sum())
                }
            }
            _ => None,
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn sum_f64(&self) -> Option<f64> {
        match self {
            ColumnVector::Float64(values) => Some(values.iter().sum()),
            _ => None,
        }
    }
}

// =============================================================================
// SIMD Implementations (x86_64 with AVX2)
// =============================================================================

/// SIMD sum for i64 values using AVX2 (4 × i64 per vector)
#[cfg(target_arch = "x86_64")]
fn simd_sum_i64(values: &[i64]) -> i64 {
    #[cfg(target_feature = "avx2")]
    {
        use std::arch::x86_64::*;
        
        unsafe {
            let mut sum = _mm256_setzero_si256();
            let chunks = values.len() / 4;
            let ptr = values.as_ptr();
            
            for i in 0..chunks {
                let v = _mm256_loadu_si256(ptr.add(i * 4) as *const __m256i);
                sum = _mm256_add_epi64(sum, v);
            }
            
            // Horizontal add
            let arr: [i64; 4] = std::mem::transmute(sum);
            let simd_total: i64 = arr.iter().sum();
            
            // Add remaining elements
            let remaining: i64 = values[chunks * 4..].iter().sum();
            simd_total + remaining
        }
    }
    
    #[cfg(not(target_feature = "avx2"))]
    {
        values.iter().sum()
    }
}

/// SIMD sum for f64 values using AVX (4 × f64 per vector)
#[cfg(target_arch = "x86_64")]
fn simd_sum_f64(values: &[f64]) -> f64 {
    #[cfg(target_feature = "avx")]
    {
        use std::arch::x86_64::*;
        
        unsafe {
            let mut sum = _mm256_setzero_pd();
            let chunks = values.len() / 4;
            let ptr = values.as_ptr();
            
            for i in 0..chunks {
                let v = _mm256_loadu_pd(ptr.add(i * 4));
                sum = _mm256_add_pd(sum, v);
            }
            
            // Horizontal add
            let arr: [f64; 4] = std::mem::transmute(sum);
            let simd_total: f64 = arr.iter().sum();
            
            // Add remaining elements
            let remaining: f64 = values[chunks * 4..].iter().sum();
            simd_total + remaining
        }
    }
    
    #[cfg(not(target_feature = "avx"))]
    {
        values.iter().sum()
    }
}

// =============================================================================
// Vectorized Batch for Processing
// =============================================================================

/// A batch of rows for vectorized processing
/// 
/// Instead of processing one row at a time, we accumulate rows into
/// batches and process them together for better cache utilization
/// and SIMD opportunities.
#[derive(Debug)]
pub struct VectorBatch {
    /// Column data in columnar format
    columns: Vec<(String, ColumnVector)>,
    /// Row count in this batch
    row_count: usize,
    /// Capacity (pre-allocated)
    capacity: usize,
    /// Selection vector (indexes of selected rows after filtering)
    selection: Option<Vec<usize>>,
}

impl VectorBatch {
    /// Create a new batch with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            columns: Vec::new(),
            row_count: 0,
            capacity,
            selection: None,
        }
    }

    /// Create batch with default size
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BATCH_SIZE)
    }

    /// Get batch capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current row count
    pub fn row_count(&self) -> usize {
        if let Some(ref sel) = self.selection {
            sel.len()
        } else {
            self.row_count
        }
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.row_count >= self.capacity
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Add a column to the batch
    pub fn add_column(&mut self, name: impl Into<String>, column: ColumnVector) {
        self.columns.push((name.into(), column));
        if self.row_count == 0 {
            self.row_count = self.columns.last().map(|(_, c)| c.len()).unwrap_or(0);
        }
    }

    /// Get column by name
    pub fn column(&self, name: &str) -> Option<&ColumnVector> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }

    /// Get column by index
    pub fn column_at(&self, idx: usize) -> Option<&ColumnVector> {
        self.columns.get(idx).map(|(_, c)| c)
    }

    /// Get column count
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Set selection vector (for filtering)
    pub fn set_selection(&mut self, selection: Vec<usize>) {
        self.selection = Some(selection);
    }

    /// Clear selection vector
    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    /// Get total memory size
    pub fn memory_size(&self) -> usize {
        self.columns.iter().map(|(_, c)| c.memory_size()).sum()
    }

    /// Reset batch for reuse
    pub fn reset(&mut self) {
        self.columns.clear();
        self.row_count = 0;
        self.selection = None;
    }
}

impl Default for VectorBatch {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Vectorized Scan Engine
// =============================================================================

/// Statistics for vectorized scan operations
#[derive(Debug, Default)]
pub struct VectorizedScanStats {
    /// Total rows scanned
    pub rows_scanned: AtomicUsize,
    /// Batches processed
    pub batches_processed: AtomicUsize,
    /// Rows passing filter
    pub rows_passed: AtomicUsize,
    /// Bytes read from storage
    pub bytes_read: AtomicUsize,
}

impl VectorizedScanStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_batch(&self, rows: usize, passed: usize, bytes: usize) {
        self.rows_scanned.fetch_add(rows, Ordering::Relaxed);
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.rows_passed.fetch_add(passed, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn rows_scanned(&self) -> usize {
        self.rows_scanned.load(Ordering::Relaxed)
    }

    pub fn batches_processed(&self) -> usize {
        self.batches_processed.load(Ordering::Relaxed)
    }
}

/// Predicate for vectorized filtering
pub trait VectorPredicate: Send + Sync {
    /// Apply predicate to a column vector, returning selection bitmap
    fn evaluate(&self, column: &ColumnVector) -> Vec<bool>;
    
    /// Get the column name this predicate operates on
    fn column_name(&self) -> &str;
}

/// Comparison predicate for i64 columns
#[derive(Debug, Clone)]
pub struct Int64Comparison {
    column_name: String,
    op: ComparisonOp,
    value: i64,
}

/// Comparison operators
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}

impl Int64Comparison {
    pub fn new(column_name: impl Into<String>, op: ComparisonOp, value: i64) -> Self {
        Self {
            column_name: column_name.into(),
            op,
            value,
        }
    }

    pub fn eq(column_name: impl Into<String>, value: i64) -> Self {
        Self::new(column_name, ComparisonOp::Equal, value)
    }

    pub fn gt(column_name: impl Into<String>, value: i64) -> Self {
        Self::new(column_name, ComparisonOp::GreaterThan, value)
    }

    pub fn lt(column_name: impl Into<String>, value: i64) -> Self {
        Self::new(column_name, ComparisonOp::LessThan, value)
    }
}

impl VectorPredicate for Int64Comparison {
    fn evaluate(&self, column: &ColumnVector) -> Vec<bool> {
        match column {
            ColumnVector::Int64(values) => {
                let cmp_value = self.value;
                match self.op {
                    ComparisonOp::Equal => values.iter().map(|&v| v == cmp_value).collect(),
                    ComparisonOp::NotEqual => values.iter().map(|&v| v != cmp_value).collect(),
                    ComparisonOp::LessThan => values.iter().map(|&v| v < cmp_value).collect(),
                    ComparisonOp::LessEqual => values.iter().map(|&v| v <= cmp_value).collect(),
                    ComparisonOp::GreaterThan => values.iter().map(|&v| v > cmp_value).collect(),
                    ComparisonOp::GreaterEqual => values.iter().map(|&v| v >= cmp_value).collect(),
                }
            }
            _ => vec![false; column.len()],
        }
    }

    fn column_name(&self) -> &str {
        &self.column_name
    }
}

/// Apply SIMD-optimized comparison for i64 values
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn simd_compare_i64_gt(values: &[i64], threshold: i64) -> Vec<bool> {
    use std::arch::x86_64::*;
    
    let mut result = vec![false; values.len()];
    let chunks = values.len() / 4;
    
    unsafe {
        let threshold_vec = _mm256_set1_epi64x(threshold);
        
        for i in 0..chunks {
            let v = _mm256_loadu_si256(values.as_ptr().add(i * 4) as *const __m256i);
            let cmp = _mm256_cmpgt_epi64(v, threshold_vec);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            
            // Each i64 occupies 8 bytes, so bits 0-7, 8-15, 16-23, 24-31
            result[i * 4] = (mask & 0xFF) != 0;
            result[i * 4 + 1] = (mask & 0xFF00) != 0;
            result[i * 4 + 2] = (mask & 0xFF0000) != 0;
            result[i * 4 + 3] = (mask & 0xFF000000) != 0;
        }
        
        // Handle remaining
        for i in (chunks * 4)..values.len() {
            result[i] = values[i] > threshold;
        }
    }
    
    result
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub fn simd_compare_i64_gt(values: &[i64], threshold: i64) -> Vec<bool> {
    values.iter().map(|&v| v > threshold).collect()
}

// =============================================================================
// Vectorized Scan Iterator
// =============================================================================

/// Configuration for vectorized scans
#[derive(Debug, Clone)]
pub struct VectorizedScanConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable prefetching
    pub prefetch_enabled: bool,
    /// Prefetch distance in rows
    pub prefetch_distance: usize,
    /// Enable SIMD acceleration
    pub simd_enabled: bool,
}

impl Default for VectorizedScanConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            prefetch_enabled: true,
            prefetch_distance: 16,
            simd_enabled: true,
        }
    }
}

impl VectorizedScanConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
        self
    }

    pub fn with_prefetch(mut self, enabled: bool) -> Self {
        self.prefetch_enabled = enabled;
        self
    }
}

// =============================================================================
// SIMD Visibility Filtering (Task 4: Zero-Copy SIMD Scans)
// =============================================================================

/// SIMD-accelerated MVCC visibility filter
///
/// ## Performance Analysis
///
/// For a snapshot timestamp `S`, a row with commit_ts `C` is visible if:
/// - `C != 0` (committed) AND `C < S` (committed before snapshot)
///
/// This can be vectorized by processing 4 timestamps at a time (AVX2):
/// ```text
/// visible[i] = (commit_ts[i] != 0) & (commit_ts[i] < snapshot_ts)
/// ```
///
/// Throughput improvement:
/// - Scalar: ~1 billion comparisons/sec
/// - AVX2 (4-way): ~4 billion comparisons/sec  
/// - AVX-512 (8-way): ~8 billion comparisons/sec
///
/// For 1M rows: scalar = 1ms, SIMD = 125-250µs (4-8x speedup)
pub struct SimdVisibilityFilter;

impl SimdVisibilityFilter {
    /// Filter a batch of commit timestamps for visibility
    ///
    /// Returns a bitmask where 1 = visible, 0 = not visible
    #[inline]
    pub fn filter_batch(commit_ts: &[u64], snapshot_ts: u64) -> Vec<bool> {
        let mut result = vec![false; commit_ts.len()];
        Self::filter_batch_into(commit_ts, snapshot_ts, &mut result);
        result
    }

    /// Filter into an existing buffer to avoid allocation
    #[inline]
    pub fn filter_batch_into(commit_ts: &[u64], snapshot_ts: u64, out: &mut [bool]) {
        assert_eq!(commit_ts.len(), out.len());

        #[cfg(target_arch = "x86_64")]
        {
            Self::filter_batch_simd_x86(commit_ts, snapshot_ts, out);
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::filter_batch_simd_neon(commit_ts, snapshot_ts, out);
            return;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::filter_batch_scalar(commit_ts, snapshot_ts, out);
        }
    }

    /// Scalar fallback implementation
    #[inline]
    fn filter_batch_scalar(commit_ts: &[u64], snapshot_ts: u64, out: &mut [bool]) {
        for (i, &ts) in commit_ts.iter().enumerate() {
            // Visible if: committed (ts != 0) AND committed before snapshot (ts < snapshot_ts)
            out[i] = ts != 0 && ts < snapshot_ts;
        }
    }

    /// x86_64 AVX2 implementation (4 u64s per iteration)
    #[cfg(target_arch = "x86_64")]
    fn filter_batch_simd_x86(commit_ts: &[u64], snapshot_ts: u64, out: &mut [bool]) {
        let n = commit_ts.len();
        if n == 0 {
            return;
        }

        // Process in chunks of 4 (AVX2 processes 4 × u64)
        let chunks = n / 4;
        let remainder = n % 4;

        // Use wide comparison on x86_64
        // Note: AVX2 doesn't have native u64 comparison, so we use signed comparison
        // which works for our use case (timestamps are positive)
        #[cfg(target_feature = "avx2")]
        unsafe {
            use std::arch::x86_64::*;

            let zero = _mm256_setzero_si256();
            let snapshot_vec = _mm256_set1_epi64x(snapshot_ts as i64);

            for chunk in 0..chunks {
                let ptr = commit_ts.as_ptr().add(chunk * 4) as *const __m256i;
                let ts_vec = _mm256_loadu_si256(ptr);

                // Check ts != 0 using PCMPEQ and inverting
                let not_zero = _mm256_xor_si256(
                    _mm256_cmpeq_epi64(ts_vec, zero),
                    _mm256_set1_epi64x(-1), // All 1s
                );

                // Check ts < snapshot using PCMPGT and inverting
                // (a < b) == !(a >= b) == !(a > b || a == b)
                let less_than = _mm256_xor_si256(
                    _mm256_or_si256(
                        _mm256_cmpgt_epi64(ts_vec, snapshot_vec),
                        _mm256_cmpeq_epi64(ts_vec, snapshot_vec),
                    ),
                    _mm256_set1_epi64x(-1),
                );

                // Combine: visible = not_zero AND less_than
                let visible = _mm256_and_si256(not_zero, less_than);

                // Extract results - each 64-bit lane is either all 1s or all 0s
                let mask: [i64; 4] = std::mem::transmute(visible);
                for j in 0..4 {
                    out[chunk * 4 + j] = mask[j] != 0;
                }
            }
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            // SSE2 fallback (2 × u64 per iteration)
            let chunks = n / 2;
            for chunk in 0..chunks {
                let base = chunk * 2;
                for j in 0..2 {
                    let ts = commit_ts[base + j];
                    out[base + j] = ts != 0 && ts < snapshot_ts;
                }
            }
        }

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            let ts = commit_ts[base + i];
            out[base + i] = ts != 0 && ts < snapshot_ts;
        }
    }

    /// ARM NEON implementation (2 u64s per iteration)
    #[cfg(target_arch = "aarch64")]
    fn filter_batch_simd_neon(commit_ts: &[u64], snapshot_ts: u64, out: &mut [bool]) {
        // NEON doesn't have all the u64 operations we need, so use scalar on ARM
        // In a production system, we'd use assembly or wait for better NEON intrinsics
        Self::filter_batch_scalar(commit_ts, snapshot_ts, out);
    }

    /// Filter with transaction ID for self-visibility
    ///
    /// A row is visible if:
    /// - (commit_ts != 0 AND commit_ts < snapshot_ts), OR
    /// - txn_id == current_txn_id (own uncommitted writes)
    #[inline]
    pub fn filter_batch_with_txn(
        commit_ts: &[u64],
        txn_ids: &[u64],
        snapshot_ts: u64,
        current_txn_id: u64,
        out: &mut [bool],
    ) {
        assert_eq!(commit_ts.len(), txn_ids.len());
        assert_eq!(commit_ts.len(), out.len());

        // First pass: standard visibility
        Self::filter_batch_into(commit_ts, snapshot_ts, out);

        // Second pass: add self-visibility
        for (i, &txn_id) in txn_ids.iter().enumerate() {
            if txn_id == current_txn_id {
                out[i] = true;
            }
        }
    }

    /// Count visible rows without allocating a result vector
    #[inline]
    pub fn count_visible(commit_ts: &[u64], snapshot_ts: u64) -> usize {
        let mut count = 0;
        for &ts in commit_ts {
            if ts != 0 && ts < snapshot_ts {
                count += 1;
            }
        }
        count
    }
}

/// Versioned slice for zero-copy access
///
/// Holds a reference to data along with visibility metadata,
/// avoiding copies during scan iteration.
#[derive(Debug, Clone)]
pub struct VersionedSlice<'a> {
    /// Key bytes (zero-copy reference)
    pub key: &'a [u8],
    /// Value bytes (zero-copy reference, None = tombstone)
    pub value: Option<&'a [u8]>,
    /// Commit timestamp (0 = uncommitted)
    pub commit_ts: u64,
    /// Transaction ID that wrote this version
    pub txn_id: u64,
}

impl<'a> VersionedSlice<'a> {
    /// Check visibility at a snapshot
    #[inline]
    pub fn is_visible(&self, snapshot_ts: u64, current_txn_id: Option<u64>) -> bool {
        // Self-visibility
        if let Some(my_txn) = current_txn_id {
            if self.txn_id == my_txn {
                return true;
            }
        }
        // Standard visibility: committed before snapshot
        self.commit_ts != 0 && self.commit_ts < snapshot_ts
    }
}

/// Streaming scan iterator with SIMD visibility filtering
///
/// ## Design
///
/// Instead of materializing all results upfront, this iterator:
/// 1. Fetches batches of entries from the underlying source
/// 2. Applies SIMD visibility filtering to the batch
/// 3. Yields visible entries one at a time
///
/// This reduces memory allocation and leverages SIMD for batch filtering.
pub struct StreamingScanIterator<'a, I>
where
    I: Iterator<Item = VersionedSlice<'a>>,
{
    /// Underlying iterator
    source: I,
    /// Current batch of entries
    batch: Vec<VersionedSlice<'a>>,
    /// Visibility mask for current batch
    visibility: Vec<bool>,
    /// Current position in batch
    pos: usize,
    /// Snapshot timestamp for visibility
    snapshot_ts: u64,
    /// Current transaction ID for self-visibility
    current_txn_id: Option<u64>,
    /// Batch size for prefetching
    batch_size: usize,
}

impl<'a, I> StreamingScanIterator<'a, I>
where
    I: Iterator<Item = VersionedSlice<'a>>,
{
    /// Create a new streaming scan iterator
    pub fn new(source: I, snapshot_ts: u64, current_txn_id: Option<u64>) -> Self {
        Self::with_batch_size(source, snapshot_ts, current_txn_id, DEFAULT_BATCH_SIZE)
    }

    /// Create with custom batch size
    pub fn with_batch_size(
        source: I,
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
        batch_size: usize,
    ) -> Self {
        Self {
            source,
            batch: Vec::with_capacity(batch_size),
            visibility: Vec::with_capacity(batch_size),
            pos: 0,
            snapshot_ts,
            current_txn_id,
            batch_size,
        }
    }

    /// Fetch next batch and compute visibility
    fn fetch_batch(&mut self) -> bool {
        self.batch.clear();
        self.visibility.clear();
        self.pos = 0;

        // Collect batch
        for entry in self.source.by_ref().take(self.batch_size) {
            self.batch.push(entry);
        }

        if self.batch.is_empty() {
            return false;
        }

        // Extract commit timestamps for SIMD filtering
        let commit_ts: Vec<u64> = self.batch.iter().map(|e| e.commit_ts).collect();
        self.visibility.resize(self.batch.len(), false);

        if let Some(txn_id) = self.current_txn_id {
            let txn_ids: Vec<u64> = self.batch.iter().map(|e| e.txn_id).collect();
            SimdVisibilityFilter::filter_batch_with_txn(
                &commit_ts,
                &txn_ids,
                self.snapshot_ts,
                txn_id,
                &mut self.visibility,
            );
        } else {
            SimdVisibilityFilter::filter_batch_into(&commit_ts, self.snapshot_ts, &mut self.visibility);
        }

        true
    }
}

impl<'a, I> Iterator for StreamingScanIterator<'a, I>
where
    I: Iterator<Item = VersionedSlice<'a>>,
{
    type Item = VersionedSlice<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Exhausted current batch?
            while self.pos >= self.batch.len() {
                if !self.fetch_batch() {
                    return None;
                }
            }

            // Find next visible entry in current batch
            while self.pos < self.batch.len() {
                let idx = self.pos;
                self.pos += 1;

                if self.visibility[idx] {
                    return Some(self.batch[idx].clone());
                }
            }
        }
    }
}

// =============================================================================
// SoA Batch + Late Materialization (80/20 Optimization)
// =============================================================================

/// Structure of Arrays (SoA) batch for optimal SIMD visibility filtering
///
/// ## Why SoA?
///
/// AoS (Array of Structures) layout:
/// ```text
/// [key, value, commit_ts, txn_id], [key, value, commit_ts, txn_id], ...
/// ```
/// - Poor cache utilization for visibility checks
/// - SIMD loads scatter data across cache lines
///
/// SoA (Structure of Arrays) layout:
/// ```text
/// commit_ts: [ts1, ts2, ts3, ts4, ...]  // Contiguous for SIMD
/// txn_ids:   [id1, id2, id3, id4, ...]  // Contiguous for SIMD
/// keys:      [k1, k2, k3, k4, ...]      // Only accessed for visible rows
/// values:    [v1, v2, v3, v4, ...]      // Late materialized
/// ```
///
/// SIMD can process 4-8 timestamps per cycle with perfect cache utilization.
///
/// ## Late Materialization
///
/// Values are NOT copied into the batch. Instead, we store offsets/handles
/// and only materialize values for rows that pass visibility filtering.
///
/// For scans where 90% of rows are filtered out, this saves ~90% of value copies.
#[derive(Debug)]
pub struct SoaBatch<'a> {
    /// Contiguous commit timestamps for SIMD visibility filtering
    pub commit_ts: Vec<u64>,
    /// Contiguous transaction IDs for self-visibility checking
    pub txn_ids: Vec<u64>,
    /// Key references (zero-copy)
    pub keys: Vec<&'a [u8]>,
    /// Value materializer handles (late binding)
    /// None = tombstone, Some = handle to materialize value
    pub value_handles: Vec<Option<ValueHandle<'a>>>,
    /// Pre-computed visibility mask (after SIMD filtering)
    pub visibility: Vec<bool>,
    /// Selection vector: indices of visible rows for late materialization
    pub selection: Vec<usize>,
}

/// Handle for late value materialization
///
/// Instead of copying values upfront, we store a handle that can
/// materialize the value on-demand when needed.
#[derive(Debug, Clone, Copy)]
pub enum ValueHandle<'a> {
    /// Direct reference (zero-copy for inmemory data)
    Direct(&'a [u8]),
    /// Offset in a data block (for disk-resident data)
    BlockOffset { block_id: u32, offset: u32, len: u32 },
    /// Deferred load from arena
    ArenaSlot { arena_id: u32, slot: u32 },
}

impl<'a> ValueHandle<'a> {
    /// Materialize the value (called only for visible rows)
    pub fn materialize(&self) -> Option<&'a [u8]> {
        match self {
            ValueHandle::Direct(data) => Some(*data),
            // For block/arena handles, would call into storage layer
            // For now, return None (would be implemented with storage context)
            ValueHandle::BlockOffset { .. } => None,
            ValueHandle::ArenaSlot { .. } => None,
        }
    }
}

impl<'a> SoaBatch<'a> {
    /// Create a new SoA batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            commit_ts: Vec::with_capacity(capacity),
            txn_ids: Vec::with_capacity(capacity),
            keys: Vec::with_capacity(capacity),
            value_handles: Vec::with_capacity(capacity),
            visibility: Vec::with_capacity(capacity),
            selection: Vec::with_capacity(capacity),
        }
    }

    /// Add an entry to the batch (SoA decomposition)
    #[inline]
    pub fn push(&mut self, key: &'a [u8], value: Option<&'a [u8]>, commit_ts: u64, txn_id: u64) {
        self.commit_ts.push(commit_ts);
        self.txn_ids.push(txn_id);
        self.keys.push(key);
        self.value_handles.push(value.map(ValueHandle::Direct));
    }

    /// Add with block handle (for disk-resident values)
    #[inline]
    pub fn push_deferred(
        &mut self,
        key: &'a [u8],
        handle: Option<ValueHandle<'a>>,
        commit_ts: u64,
        txn_id: u64,
    ) {
        self.commit_ts.push(commit_ts);
        self.txn_ids.push(txn_id);
        self.keys.push(key);
        self.value_handles.push(handle);
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.commit_ts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.commit_ts.is_empty()
    }

    /// Clear the batch for reuse
    pub fn clear(&mut self) {
        self.commit_ts.clear();
        self.txn_ids.clear();
        self.keys.clear();
        self.value_handles.clear();
        self.visibility.clear();
        self.selection.clear();
    }

    /// Apply SIMD visibility filtering and build selection vector
    ///
    /// This is the hot path - SIMD processes commit_ts array directly
    /// without touching keys/values until we know what's visible.
    pub fn filter_visibility(&mut self, snapshot_ts: u64, current_txn_id: Option<u64>) {
        let n = self.len();
        self.visibility.resize(n, false);
        self.selection.clear();

        // SIMD filter on contiguous commit_ts array
        if let Some(txn_id) = current_txn_id {
            SimdVisibilityFilter::filter_batch_with_txn(
                &self.commit_ts,
                &self.txn_ids,
                snapshot_ts,
                txn_id,
                &mut self.visibility,
            );
        } else {
            SimdVisibilityFilter::filter_batch_into(&self.commit_ts, snapshot_ts, &mut self.visibility);
        }

        // Build selection vector (indices of visible rows)
        for (i, &visible) in self.visibility.iter().enumerate() {
            if visible {
                self.selection.push(i);
            }
        }
    }

    /// Get visible row count (after filtering)
    pub fn visible_count(&self) -> usize {
        self.selection.len()
    }

    /// Iterate over visible rows with late materialization
    ///
    /// Values are only materialized for rows in the selection vector.
    pub fn iter_visible(&self) -> impl Iterator<Item = (&'a [u8], Option<&'a [u8]>)> + '_ {
        self.selection.iter().map(move |&idx| {
            let key = self.keys[idx];
            let value = self.value_handles[idx].and_then(|h| h.materialize());
            (key, value)
        })
    }

    /// Iterate visible rows with full metadata
    pub fn iter_visible_full(
        &self,
    ) -> impl Iterator<Item = (&'a [u8], Option<&'a [u8]>, u64, u64)> + '_ {
        self.selection.iter().map(move |&idx| {
            let key = self.keys[idx];
            let value = self.value_handles[idx].and_then(|h| h.materialize());
            let ts = self.commit_ts[idx];
            let txn = self.txn_ids[idx];
            (key, value, ts, txn)
        })
    }
}

/// High-performance SoA scan iterator with SIMD + late materialization
///
/// ## Performance Characteristics
///
/// | Phase              | Cache Behavior           | SIMD Usage    |
/// |--------------------|--------------------------|---------------|
/// | Load batch         | Sequential (SoA arrays)  | N/A           |
/// | Visibility filter  | L1-hot (commit_ts only)  | 4-8× speedup  |
/// | Build selection    | Sequential (visibility)  | Auto-vectorized |
/// | Materialize values | Random (visible only)    | N/A           |
///
/// For 1M rows with 10% selectivity:
/// - Old: Process 1M rows with random access = ~50ms
/// - New: SIMD filter 1M × 8 bytes = ~1ms, materialize 100K values = ~5ms
/// - **~8× speedup** from SoA + SIMD + late materialization
pub struct SoaScanIterator<'a, S>
where
    S: SoaSource<'a>,
{
    /// Source that provides SoA batches
    source: S,
    /// Current batch
    batch: SoaBatch<'a>,
    /// Position in selection vector
    pos: usize,
    /// Snapshot timestamp
    snapshot_ts: u64,
    /// Current transaction ID
    current_txn_id: Option<u64>,
    /// Batch size
    #[allow(dead_code)]
    batch_size: usize,
    /// Statistics
    stats: SoaScanStats,
}

/// Statistics for SoA scan performance monitoring
#[derive(Debug, Default, Clone)]
pub struct SoaScanStats {
    /// Total rows scanned
    pub rows_scanned: usize,
    /// Rows that passed visibility filter
    pub rows_visible: usize,
    /// Rows where values were materialized
    pub values_materialized: usize,
    /// Number of batches processed
    pub batches_processed: usize,
}

impl SoaScanStats {
    /// Get selectivity ratio
    pub fn selectivity(&self) -> f64 {
        if self.rows_scanned == 0 {
            0.0
        } else {
            self.rows_visible as f64 / self.rows_scanned as f64
        }
    }

    /// Get value materialization efficiency
    /// (1.0 = all visible rows materialized, lower = some skipped)
    pub fn materialization_efficiency(&self) -> f64 {
        if self.rows_visible == 0 {
            1.0
        } else {
            self.values_materialized as f64 / self.rows_visible as f64
        }
    }
}

/// Trait for sources that provide SoA batches
pub trait SoaSource<'a> {
    /// Fill a batch with entries from the source
    /// Returns false if source is exhausted
    fn fill_batch(&mut self, batch: &mut SoaBatch<'a>) -> bool;
}

impl<'a, S> SoaScanIterator<'a, S>
where
    S: SoaSource<'a>,
{
    /// Create new SoA scan iterator
    pub fn new(source: S, snapshot_ts: u64, current_txn_id: Option<u64>) -> Self {
        Self::with_batch_size(source, snapshot_ts, current_txn_id, DEFAULT_BATCH_SIZE)
    }

    /// Create with custom batch size
    pub fn with_batch_size(
        source: S,
        snapshot_ts: u64,
        current_txn_id: Option<u64>,
        batch_size: usize,
    ) -> Self {
        Self {
            source,
            batch: SoaBatch::with_capacity(batch_size),
            pos: 0,
            snapshot_ts,
            current_txn_id,
            batch_size,
            stats: SoaScanStats::default(),
        }
    }

    /// Fetch next batch with visibility filtering
    fn fetch_batch(&mut self) -> bool {
        self.batch.clear();
        self.pos = 0;

        // Fill batch from source (SoA format)
        if !self.source.fill_batch(&mut self.batch) {
            return false;
        }

        self.stats.rows_scanned += self.batch.len();
        self.stats.batches_processed += 1;

        // SIMD visibility filtering on contiguous arrays
        self.batch.filter_visibility(self.snapshot_ts, self.current_txn_id);
        self.stats.rows_visible += self.batch.visible_count();

        true
    }

    /// Get scan statistics
    pub fn stats(&self) -> &SoaScanStats {
        &self.stats
    }
}

impl<'a, S> Iterator for SoaScanIterator<'a, S>
where
    S: SoaSource<'a>,
{
    type Item = (&'a [u8], Option<&'a [u8]>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Need new batch?
            while self.pos >= self.batch.selection.len() {
                if !self.fetch_batch() {
                    return None;
                }
            }

            // Get next visible row (late materialization)
            let sel_idx = self.pos;
            self.pos += 1;
            let row_idx = self.batch.selection[sel_idx];

            let key = self.batch.keys[row_idx];
            let value = self.batch.value_handles[row_idx].and_then(|h| h.materialize());
            self.stats.values_materialized += 1;

            return Some((key, value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_vector_int64() {
        let mut v = ColumnVector::Int64(vec![1, 2, 3, 4, 5]);
        assert_eq!(v.len(), 5);
        assert_eq!(v.sum_i64(), Some(15));
    }

    #[test]
    fn test_column_vector_float64() {
        let v = ColumnVector::Float64(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.len(), 4);
        assert_eq!(v.sum_f64(), Some(10.0));
    }

    #[test]
    fn test_vector_batch() {
        let mut batch = VectorBatch::with_capacity(1024);
        batch.add_column("id", ColumnVector::Int64(vec![1, 2, 3]));
        batch.add_column("value", ColumnVector::Float64(vec![1.5, 2.5, 3.5]));
        
        assert_eq!(batch.row_count(), 3);
        assert_eq!(batch.column_count(), 2);
        assert!(batch.column("id").is_some());
    }

    #[test]
    fn test_int64_comparison() {
        let col = ColumnVector::Int64(vec![1, 5, 10, 15, 20]);
        let pred = Int64Comparison::gt("test", 10);
        let result = pred.evaluate(&col);
        
        assert_eq!(result, vec![false, false, false, true, true]);
    }

    #[test]
    fn test_simd_sum_i64_large() {
        // Test with enough elements to trigger SIMD path
        let values: Vec<i64> = (0..1000).collect();
        let expected: i64 = (0..1000).sum();
        
        let col = ColumnVector::Int64(values);
        assert_eq!(col.sum_i64(), Some(expected));
    }

    #[test]
    fn test_simd_compare_gt() {
        let values: Vec<i64> = vec![1, 5, 10, 15, 20, 25, 30, 35];
        let result = simd_compare_i64_gt(&values, 12);
        assert_eq!(result, vec![false, false, false, true, true, true, true, true]);
    }

    #[test]
    fn test_vectorized_scan_config() {
        let config = VectorizedScanConfig::new()
            .with_batch_size(2048)
            .with_prefetch(true);
        
        assert_eq!(config.batch_size, 2048);
        assert!(config.prefetch_enabled);
    }

    #[test]
    fn test_simd_visibility_filter_basic() {
        // commit_ts: 0 = uncommitted, others = commit time
        let commit_ts = vec![0, 10, 20, 30, 40];
        let snapshot_ts = 25;
        
        let result = SimdVisibilityFilter::filter_batch(&commit_ts, snapshot_ts);
        
        // Visible: ts != 0 AND ts < 25
        // ts=0: not visible (uncommitted)
        // ts=10: visible (10 < 25)
        // ts=20: visible (20 < 25)
        // ts=30: not visible (30 >= 25)
        // ts=40: not visible (40 >= 25)
        assert_eq!(result, vec![false, true, true, false, false]);
    }

    #[test]
    fn test_simd_visibility_filter_with_txn() {
        let commit_ts = vec![0, 10, 0, 30, 40];
        let txn_ids = vec![1, 2, 1, 4, 5];  // txn_id 1 appears twice
        let snapshot_ts = 25;
        let current_txn_id = 1;
        
        let mut result = vec![false; 5];
        SimdVisibilityFilter::filter_batch_with_txn(
            &commit_ts,
            &txn_ids,
            snapshot_ts,
            current_txn_id,
            &mut result,
        );
        
        // Visible:
        // [0]: uncommitted but own txn -> visible
        // [1]: committed at 10 < 25 -> visible
        // [2]: uncommitted but own txn -> visible
        // [3]: committed at 30 >= 25 -> not visible
        // [4]: committed at 40 >= 25 -> not visible
        assert_eq!(result, vec![true, true, true, false, false]);
    }

    #[test]
    fn test_simd_visibility_filter_large() {
        // Test with enough elements to trigger SIMD path
        let n = 1000;
        let commit_ts: Vec<u64> = (1..=n as u64).collect();
        let snapshot_ts = 500;
        
        let result = SimdVisibilityFilter::filter_batch(&commit_ts, snapshot_ts);
        
        // First 499 should be visible (1..500 < 500)
        let visible_count = result.iter().filter(|&&v| v).count();
        assert_eq!(visible_count, 499);
    }

    #[test]
    fn test_versioned_slice_visibility() {
        let slice = VersionedSlice {
            key: b"test",
            value: Some(b"value"),
            commit_ts: 100,
            txn_id: 1,
        };
        
        assert!(slice.is_visible(200, None));
        assert!(!slice.is_visible(50, None));
        assert!(slice.is_visible(50, Some(1))); // Self-visibility
    }

    #[test]
    fn test_streaming_scan_iterator() {
        let entries: Vec<VersionedSlice<'static>> = vec![
            VersionedSlice { key: b"a", value: Some(b"1"), commit_ts: 10, txn_id: 1 },
            VersionedSlice { key: b"b", value: Some(b"2"), commit_ts: 0, txn_id: 2 },  // Uncommitted
            VersionedSlice { key: b"c", value: Some(b"3"), commit_ts: 30, txn_id: 3 },  // After snapshot
            VersionedSlice { key: b"d", value: Some(b"4"), commit_ts: 15, txn_id: 4 },
        ];
        
        let iter = StreamingScanIterator::new(entries.into_iter(), 25, None);
        let visible: Vec<_> = iter.collect();
        
        // Only entries with commit_ts 10 and 15 should be visible
        assert_eq!(visible.len(), 2);
        assert_eq!(visible[0].key, b"a");
        assert_eq!(visible[1].key, b"d");
    }

    #[test]
    fn test_soa_batch_basic() {
        let mut batch = SoaBatch::with_capacity(100);
        
        batch.push(b"key1", Some(b"value1"), 10, 1);
        batch.push(b"key2", Some(b"value2"), 20, 2);
        batch.push(b"key3", None, 30, 3);  // Tombstone
        batch.push(b"key4", Some(b"value4"), 0, 4);  // Uncommitted
        
        assert_eq!(batch.len(), 4);
        assert_eq!(batch.commit_ts, vec![10, 20, 30, 0]);
        assert_eq!(batch.txn_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_soa_batch_visibility_filter() {
        let mut batch = SoaBatch::with_capacity(100);
        
        batch.push(b"k1", Some(b"v1"), 10, 1);  // Visible (10 < 25)
        batch.push(b"k2", Some(b"v2"), 0, 2);   // Not visible (uncommitted)
        batch.push(b"k3", Some(b"v3"), 20, 3);  // Visible (20 < 25)
        batch.push(b"k4", Some(b"v4"), 30, 4);  // Not visible (30 >= 25)
        batch.push(b"k5", Some(b"v5"), 0, 5);   // Not visible (uncommitted)
        
        batch.filter_visibility(25, None);
        
        assert_eq!(batch.visibility, vec![true, false, true, false, false]);
        assert_eq!(batch.selection, vec![0, 2]);  // Indices of visible rows
        assert_eq!(batch.visible_count(), 2);
    }

    #[test]
    fn test_soa_batch_self_visibility() {
        let mut batch = SoaBatch::with_capacity(100);
        
        batch.push(b"k1", Some(b"v1"), 0, 42);  // Own uncommitted -> visible
        batch.push(b"k2", Some(b"v2"), 10, 1);  // Committed -> visible
        batch.push(b"k3", Some(b"v3"), 0, 99);  // Other's uncommitted -> not visible
        
        batch.filter_visibility(25, Some(42));
        
        assert_eq!(batch.visibility, vec![true, true, false]);
        assert_eq!(batch.selection, vec![0, 1]);
    }

    #[test]
    fn test_soa_batch_late_materialization() {
        let mut batch = SoaBatch::with_capacity(100);
        
        batch.push(b"key1", Some(b"val1"), 10, 1);
        batch.push(b"key2", Some(b"val2"), 0, 2);   // Filtered out
        batch.push(b"key3", Some(b"val3"), 15, 3);
        
        batch.filter_visibility(25, None);
        
        // Iterate visible - values materialized only now
        let visible: Vec<_> = batch.iter_visible().collect();
        
        assert_eq!(visible.len(), 2);
        assert_eq!(visible[0], (b"key1".as_slice(), Some(b"val1".as_slice())));
        assert_eq!(visible[1], (b"key3".as_slice(), Some(b"val3".as_slice())));
    }

    #[test]
    fn test_soa_scan_stats() {
        let mut batch = SoaBatch::with_capacity(100);
        
        // 10 rows, 3 visible
        for i in 0..10u64 {
            let ts = if i < 3 { 10 } else { 0 };  // First 3 committed, rest uncommitted
            batch.push(b"key", Some(b"val"), ts, i);
        }
        
        batch.filter_visibility(25, None);
        
        let selectivity = batch.visible_count() as f64 / batch.len() as f64;
        assert!((selectivity - 0.3).abs() < 0.01);  // 30% selectivity
    }

    #[test]
    fn test_soa_batch_simd_large() {
        // Test with enough entries to trigger SIMD paths
        let mut batch = SoaBatch::with_capacity(2000);
        
        for i in 0..1000u64 {
            // Alternating visible/not visible
            let ts = if i % 2 == 0 { 10 } else { 50 };
            batch.push(b"k", Some(b"v"), ts, i);
        }
        
        batch.filter_visibility(25, None);
        
        // Should have 500 visible (even indices with ts=10)
        assert_eq!(batch.visible_count(), 500);
        
        // Verify selection indices are correct
        for (i, &idx) in batch.selection.iter().enumerate() {
            assert_eq!(idx, i * 2);  // 0, 2, 4, 6, ...
        }
    }
}
