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

//! # Cache-Resident Routing Layer (Task 4)
//!
//! Ensures routing fits in LLC (Last-Level Cache) to bound latency variance.
//!
//! ## Architecture
//!
//! Two-stage routing with compressed centroids:
//! 1. **Coarse stage**: FP16/int8 centroids in compressed space (fits in LLC)
//! 2. **Fine stage**: Refine top candidates (optionally in full precision)
//!
//! ## Math/Algorithm
//!
//! Cache complexity constraint: ensure routing working set W ≤ LLC_size
//!
//! For C centroids of dimension d:
//! - FP32: 4·C·d bytes (often exceeds LLC)
//! - FP16: 2·C·d bytes (50% reduction)
//! - Int8: C·d bytes (75% reduction)
//! - PQ: C·(d/m)·1 bytes (even smaller for high-dim)
//!
//! Multi-stage ranking: O(C·d_compressed + k·d_full)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::compressed_routing::{RoutingLayer, RoutingConfig, CentroidCompression};
//!
//! let config = RoutingConfig::default()
//!     .compression(CentroidCompression::Fp16)
//!     .refine_top_k(32);
//!
//! let routing = RoutingLayer::build(&centroids, config);
//! let top_lists = routing.route(&query, 16);
//! ```

use std::sync::Arc;

use crate::list_bounds::{DistanceMetric, SphericalCapMetadata};

// ============================================================================
// Centroid Compression
// ============================================================================

/// Compression method for centroids
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentroidCompression {
    /// Full FP32 precision (baseline)
    Fp32,
    /// FP16 half precision (2x compression)
    Fp16,
    /// Int8 quantization (4x compression)
    Int8,
    /// Product Quantization (high compression for high-dim)
    PQ { n_subquantizers: usize, n_bits: u8 },
    /// OPQ + PQ (optimized rotation before PQ)
    OPQ { n_subquantizers: usize, n_bits: u8 },
}

impl CentroidCompression {
    /// Compute bytes per centroid for this compression
    pub fn bytes_per_centroid(&self, dim: usize) -> usize {
        match self {
            Self::Fp32 => dim * 4,
            Self::Fp16 => dim * 2,
            Self::Int8 => dim,
            Self::PQ { n_subquantizers, n_bits } => {
                // Each subquantizer produces n_bits, pack into bytes
                (*n_subquantizers * *n_bits as usize + 7) / 8
            }
            Self::OPQ { n_subquantizers, n_bits } => {
                (*n_subquantizers * *n_bits as usize + 7) / 8
            }
        }
    }
    
    /// Check if this compression will fit in given cache size
    pub fn fits_in_cache(&self, n_centroids: usize, dim: usize, cache_bytes: usize) -> bool {
        self.bytes_per_centroid(dim) * n_centroids <= cache_bytes
    }
    
    /// Recommend compression level based on constraints
    pub fn recommend(n_centroids: usize, dim: usize, cache_bytes: usize) -> Self {
        // Try each compression level until one fits
        for compression in [
            Self::Fp32,
            Self::Fp16,
            Self::Int8,
            Self::PQ { n_subquantizers: dim / 4, n_bits: 8 },
        ] {
            if compression.fits_in_cache(n_centroids, dim, cache_bytes) {
                return compression;
            }
        }
        // Fall back to aggressive PQ
        Self::PQ { n_subquantizers: 16, n_bits: 4 }
    }
}

// ============================================================================
// Routing Configuration
// ============================================================================

/// Configuration for routing layer
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// Compression method for coarse centroids
    pub compression: CentroidCompression,
    
    /// Number of top lists to refine in second stage
    pub refine_top_k: usize,
    
    /// Use full precision for refinement
    pub full_precision_refine: bool,
    
    /// Target LLC size (for cache-awareness)
    pub target_llc_bytes: usize,
    
    /// Distance metric
    pub metric: DistanceMetric,
    
    /// Prefetch depth for sequential access
    pub prefetch_depth: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            compression: CentroidCompression::Fp16,
            refine_top_k: 64,
            full_precision_refine: true,
            target_llc_bytes: 32 * 1024 * 1024, // 32 MB LLC
            metric: DistanceMetric::Cosine,
            prefetch_depth: 4,
        }
    }
}

impl RoutingConfig {
    /// Set compression method
    pub fn compression(mut self, compression: CentroidCompression) -> Self {
        self.compression = compression;
        self
    }
    
    /// Set refinement count
    pub fn refine_top_k(mut self, k: usize) -> Self {
        self.refine_top_k = k;
        self
    }
    
    /// Set target LLC size
    pub fn target_llc(mut self, bytes: usize) -> Self {
        self.target_llc_bytes = bytes;
        self
    }
    
    /// Set distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

// ============================================================================
// Compressed Centroid Storage
// ============================================================================

/// FP16 encoded centroid storage
#[derive(Debug, Clone)]
pub struct Fp16Centroids {
    /// Packed FP16 data (2 bytes per element)
    data: Vec<u16>,
    /// Number of centroids
    n_centroids: usize,
    /// Dimension
    dim: usize,
}

impl Fp16Centroids {
    /// Build from FP32 centroids
    pub fn from_fp32(centroids: &[f32], dim: usize) -> Self {
        let n_centroids = centroids.len() / dim;
        let data: Vec<u16> = centroids.iter()
            .map(|&x| f32_to_f16(x))
            .collect();
        
        Self { data, n_centroids, dim }
    }
    
    /// Get centroid as FP32 (for refinement)
    pub fn get_fp32(&self, idx: usize) -> Vec<f32> {
        let start = idx * self.dim;
        self.data[start..start + self.dim]
            .iter()
            .map(|&x| f16_to_f32(x))
            .collect()
    }
    
    /// Compute dot products with query in FP16
    pub fn dot_products(&self, query: &[f32]) -> Vec<f32> {
        let query_f16: Vec<u16> = query.iter().map(|&x| f32_to_f16(x)).collect();
        
        (0..self.n_centroids)
            .map(|i| {
                let start = i * self.dim;
                let centroid = &self.data[start..start + self.dim];
                dot_f16(centroid, &query_f16)
            })
            .collect()
    }
    
    /// Memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 2
    }
}

/// Int8 quantized centroid storage
#[derive(Debug, Clone)]
pub struct Int8Centroids {
    /// Quantized data
    data: Vec<i8>,
    /// Scale factor per dimension
    scales: Vec<f32>,
    /// Zero point per dimension  
    zero_points: Vec<f32>,
    /// Number of centroids
    n_centroids: usize,
    /// Dimension
    dim: usize,
}

impl Int8Centroids {
    /// Build from FP32 centroids with per-dimension quantization
    pub fn from_fp32(centroids: &[f32], dim: usize) -> Self {
        let n_centroids = centroids.len() / dim;
        
        // Compute min/max per dimension
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        
        for i in 0..n_centroids {
            for j in 0..dim {
                let val = centroids[i * dim + j];
                mins[j] = mins[j].min(val);
                maxs[j] = maxs[j].max(val);
            }
        }
        
        // Compute scales and zero points
        let mut scales = Vec::with_capacity(dim);
        let mut zero_points = Vec::with_capacity(dim);
        
        for j in 0..dim {
            let range = maxs[j] - mins[j];
            let scale = if range > 1e-10 { range / 255.0 } else { 1.0 };
            scales.push(scale);
            zero_points.push(mins[j]);
        }
        
        // Quantize
        let data: Vec<i8> = centroids.iter().enumerate()
            .map(|(idx, &val)| {
                let j = idx % dim;
                let q = ((val - zero_points[j]) / scales[j]).round() as i32;
                q.clamp(-128, 127) as i8
            })
            .collect();
        
        Self { data, scales, zero_points, n_centroids, dim }
    }
    
    /// Get centroid as FP32 (dequantized)
    pub fn get_fp32(&self, idx: usize) -> Vec<f32> {
        let start = idx * self.dim;
        (0..self.dim)
            .map(|j| {
                self.data[start + j] as f32 * self.scales[j] + self.zero_points[j]
            })
            .collect()
    }
    
    /// Compute dot products with query using int8 arithmetic
    pub fn dot_products(&self, query: &[f32]) -> Vec<f32> {
        // Quantize query
        let query_i8: Vec<i8> = query.iter().enumerate()
            .map(|(j, &val)| {
                let q = ((val - self.zero_points[j]) / self.scales[j]).round() as i32;
                q.clamp(-128, 127) as i8
            })
            .collect();
        
        (0..self.n_centroids)
            .map(|i| {
                let start = i * self.dim;
                let centroid = &self.data[start..start + self.dim];
                
                // Compute dot product in int32 then convert
                let dot_i32: i32 = centroid.iter()
                    .zip(query_i8.iter())
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum();
                
                // Approximate dequantization (simplified)
                dot_i32 as f32 * self.scales[0] * self.scales[0]
            })
            .collect()
    }
    
    /// Memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }
}

// ============================================================================
// Routing Layer
// ============================================================================

/// Compressed routing layer for cache-resident operations
pub struct RoutingLayer {
    /// Compressed centroids for coarse search
    compressed: CompressedCentroids,
    
    /// Full-precision centroids for refinement (optional)
    full_precision: Option<Vec<f32>>,
    
    /// Spherical cap metadata per list
    caps: Vec<SphericalCapMetadata>,
    
    /// Configuration
    config: RoutingConfig,
    
    /// Dimension
    dim: usize,
    
    /// Number of lists
    n_lists: usize,
}

/// Enum for different compression types
enum CompressedCentroids {
    Fp32(Vec<f32>),
    Fp16(Fp16Centroids),
    Int8(Int8Centroids),
}

impl RoutingLayer {
    /// Build routing layer from FP32 centroids
    pub fn build(
        centroids: &[f32],
        dim: usize,
        config: RoutingConfig,
    ) -> Self {
        let n_lists = centroids.len() / dim;
        
        // Build compressed centroids
        let compressed = match config.compression {
            CentroidCompression::Fp32 => CompressedCentroids::Fp32(centroids.to_vec()),
            CentroidCompression::Fp16 => CompressedCentroids::Fp16(
                Fp16Centroids::from_fp32(centroids, dim)
            ),
            CentroidCompression::Int8 => CompressedCentroids::Int8(
                Int8Centroids::from_fp32(centroids, dim)
            ),
            _ => {
                // PQ/OPQ not implemented yet, fall back to FP16
                CompressedCentroids::Fp16(Fp16Centroids::from_fp32(centroids, dim))
            }
        };
        
        // Store full precision for refinement if configured
        let full_precision = if config.full_precision_refine {
            Some(centroids.to_vec())
        } else {
            None
        };
        
        // Build spherical cap metadata per list
        let caps: Vec<SphericalCapMetadata> = (0..n_lists)
            .map(|i| {
                let centroid = &centroids[i * dim..(i + 1) * dim];
                SphericalCapMetadata {
                    centroid: centroid.to_vec(),
                    theta_max: 0.0, // Will be updated when vectors are added
                    min_dot_to_centroid: 1.0,
                    max_dot_to_centroid: 1.0,
                    vector_count: 0,
                    mean_dot_to_centroid: 1.0,
                }
            })
            .collect();
        
        Self {
            compressed,
            full_precision,
            caps,
            config,
            dim,
            n_lists,
        }
    }
    
    /// Route query to top-k lists
    ///
    /// Two-stage process:
    /// 1. Coarse ranking using compressed centroids (cache-resident)
    /// 2. Refine top candidates using full precision (optional)
    pub fn route(&self, query: &[f32], n_probes: usize) -> Vec<ListCandidate> {
        let n_probes = n_probes.min(self.n_lists);
        
        // Stage 1: Coarse ranking with compressed centroids
        let mut coarse_scores = self.coarse_scores(query);
        
        // Get top-k indices for refinement
        let refine_k = self.config.refine_top_k.min(self.n_lists);
        let mut indices: Vec<usize> = (0..self.n_lists).collect();
        
        // Partial sort for top-k
        if self.config.metric.higher_is_better() {
            indices.select_nth_unstable_by(refine_k - 1, |&a, &b| {
                coarse_scores[b].partial_cmp(&coarse_scores[a]).unwrap()
            });
        } else {
            indices.select_nth_unstable_by(refine_k - 1, |&a, &b| {
                coarse_scores[a].partial_cmp(&coarse_scores[b]).unwrap()
            });
        }
        
        let top_indices = &indices[..refine_k];
        
        // Stage 2: Refine with full precision (if available)
        let refined_scores = if let Some(ref full) = self.full_precision {
            self.refine_scores(query, top_indices, full)
        } else {
            top_indices.iter().map(|&i| coarse_scores[i]).collect()
        };
        
        // Build candidates with bounds
        let mut candidates: Vec<ListCandidate> = top_indices.iter()
            .zip(refined_scores.iter())
            .map(|(&idx, &score)| {
                ListCandidate {
                    list_idx: idx as u32,
                    score,
                    bound: self.compute_bound(idx, query),
                    vector_count: self.caps[idx].vector_count,
                }
            })
            .collect();
        
        // Sort by score and take top n_probes
        if self.config.metric.higher_is_better() {
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }
        
        candidates.truncate(n_probes);
        candidates
    }
    
    /// Compute coarse scores using compressed centroids
    fn coarse_scores(&self, query: &[f32]) -> Vec<f32> {
        match &self.compressed {
            CompressedCentroids::Fp32(data) => {
                self.dot_products_fp32(query, data)
            }
            CompressedCentroids::Fp16(fp16) => {
                fp16.dot_products(query)
            }
            CompressedCentroids::Int8(int8) => {
                int8.dot_products(query)
            }
        }
    }
    
    /// Compute full-precision dot products
    fn dot_products_fp32(&self, query: &[f32], centroids: &[f32]) -> Vec<f32> {
        (0..self.n_lists)
            .map(|i| {
                let centroid = &centroids[i * self.dim..(i + 1) * self.dim];
                dot_product_f32(query, centroid)
            })
            .collect()
    }
    
    /// Refine scores for selected indices
    fn refine_scores(&self, query: &[f32], indices: &[usize], centroids: &[f32]) -> Vec<f32> {
        indices.iter()
            .map(|&i| {
                let centroid = &centroids[i * self.dim..(i + 1) * self.dim];
                dot_product_f32(query, centroid)
            })
            .collect()
    }
    
    /// Compute bound for a list
    fn compute_bound(&self, idx: usize, query: &[f32]) -> f32 {
        let cap = &self.caps[idx];
        let dot = dot_product_f32(query, &cap.centroid);
        let angle = dot.clamp(-1.0, 1.0).acos();
        let min_angle = (angle - cap.theta_max).max(0.0);
        min_angle.cos()
    }
    
    /// Update spherical cap metadata for a list
    pub fn update_cap(&mut self, list_idx: usize, cap: SphericalCapMetadata) {
        if list_idx < self.caps.len() {
            self.caps[list_idx] = cap;
        }
    }
    
    /// Get memory footprint
    pub fn memory_bytes(&self) -> usize {
        let compressed_bytes = match &self.compressed {
            CompressedCentroids::Fp32(data) => data.len() * 4,
            CompressedCentroids::Fp16(fp16) => fp16.memory_bytes(),
            CompressedCentroids::Int8(int8) => int8.memory_bytes(),
        };
        
        let full_bytes = self.full_precision.as_ref()
            .map(|v| v.len() * 4)
            .unwrap_or(0);
        
        let cap_bytes = self.caps.len() * std::mem::size_of::<SphericalCapMetadata>();
        
        compressed_bytes + full_bytes + cap_bytes
    }
    
    /// Check if routing layer fits in target cache
    pub fn fits_in_cache(&self) -> bool {
        let compressed_bytes = match &self.compressed {
            CompressedCentroids::Fp32(data) => data.len() * 4,
            CompressedCentroids::Fp16(fp16) => fp16.memory_bytes(),
            CompressedCentroids::Int8(int8) => int8.memory_bytes(),
        };
        
        compressed_bytes <= self.config.target_llc_bytes
    }
    
    /// Get routing statistics
    pub fn stats(&self) -> RoutingStats {
        RoutingStats {
            n_lists: self.n_lists,
            dim: self.dim,
            compression: format!("{:?}", self.config.compression),
            compressed_bytes: match &self.compressed {
                CompressedCentroids::Fp32(data) => data.len() * 4,
                CompressedCentroids::Fp16(fp16) => fp16.memory_bytes(),
                CompressedCentroids::Int8(int8) => int8.memory_bytes(),
            },
            total_bytes: self.memory_bytes(),
            fits_in_cache: self.fits_in_cache(),
            target_cache_bytes: self.config.target_llc_bytes,
        }
    }
}

/// Candidate list from routing
#[derive(Debug, Clone)]
pub struct ListCandidate {
    /// List index
    pub list_idx: u32,
    /// Similarity score to centroid
    pub score: f32,
    /// Upper bound on best score in this list
    pub bound: f32,
    /// Number of vectors in this list
    pub vector_count: u32,
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub n_lists: usize,
    pub dim: usize,
    pub compression: String,
    pub compressed_bytes: usize,
    pub total_bytes: usize,
    pub fits_in_cache: bool,
    pub target_cache_bytes: usize,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// FP32 to FP16 conversion (IEEE 754 half-precision)
#[inline]
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;
    
    // Handle special cases
    if exp == 0xff {
        // Inf or NaN
        return ((sign << 15) | 0x7c00 | (frac >> 13)) as u16;
    }
    if exp == 0 {
        // Zero or subnormal
        return (sign << 15) as u16;
    }
    
    // Adjust exponent for FP16 bias
    let new_exp = exp - 127 + 15;
    
    if new_exp >= 31 {
        // Overflow to infinity
        return ((sign << 15) | 0x7c00) as u16;
    }
    if new_exp <= 0 {
        // Underflow to zero
        return (sign << 15) as u16;
    }
    
    let new_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | new_frac) as u16
}

/// FP16 to FP32 conversion
#[inline]
fn f16_to_f32(x: u16) -> f32 {
    let sign = ((x >> 15) & 1) as u32;
    let exp = ((x >> 10) & 0x1f) as u32;
    let frac = (x & 0x3ff) as u32;
    
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let normalized = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -normalized } else { normalized };
    }
    if exp == 31 {
        if frac == 0 {
            return f32::from_bits((sign << 31) | 0x7f800000);
        }
        return f32::NAN;
    }
    
    let new_exp = (exp as i32 - 15 + 127) as u32;
    let new_frac = frac << 13;
    f32::from_bits((sign << 31) | (new_exp << 23) | new_frac)
}

/// FP16 dot product
#[inline]
fn dot_f16(a: &[u16], b: &[u16]) -> f32 {
    // Compute in FP32 for accuracy
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f16_to_f32(x) * f16_to_f32(y))
        .sum()
}

/// FP32 dot product
#[inline]
fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compression_bytes() {
        let dim = 768;
        
        assert_eq!(CentroidCompression::Fp32.bytes_per_centroid(dim), 3072);
        assert_eq!(CentroidCompression::Fp16.bytes_per_centroid(dim), 1536);
        assert_eq!(CentroidCompression::Int8.bytes_per_centroid(dim), 768);
    }
    
    #[test]
    fn test_compression_recommendation() {
        let cache_32mb = 32 * 1024 * 1024;
        let dim = 768;
        
        // 10k centroids at FP32 = 30MB, should fit
        let rec1 = CentroidCompression::recommend(10_000, dim, cache_32mb);
        assert!(matches!(rec1, CentroidCompression::Fp32));
        
        // 20k centroids at FP32 = 60MB, need FP16
        let rec2 = CentroidCompression::recommend(20_000, dim, cache_32mb);
        assert!(matches!(rec2, CentroidCompression::Fp16));
        
        // 50k centroids need Int8
        let rec3 = CentroidCompression::recommend(50_000, dim, cache_32mb);
        assert!(matches!(rec3, CentroidCompression::Int8));
    }
    
    #[test]
    fn test_fp16_conversion() {
        let values = [0.0, 1.0, -1.0, 0.5, 0.123, 100.0, -100.0];
        
        for &x in &values {
            let f16 = f32_to_f16(x);
            let back = f16_to_f32(f16);
            let rel_error = if x.abs() > 1e-10 {
                (x - back).abs() / x.abs()
            } else {
                (x - back).abs()
            };
            assert!(rel_error < 0.01, "FP16 roundtrip error too high: {} -> {} -> {}", x, f16, back);
        }
    }
    
    #[test]
    fn test_routing_layer() {
        let dim = 4;
        let n_centroids = 10;
        let centroids: Vec<f32> = (0..n_centroids * dim)
            .map(|i| (i as f32 / (n_centroids * dim) as f32))
            .collect();
        
        let config = RoutingConfig::default()
            .compression(CentroidCompression::Fp16)
            .refine_top_k(5);
        
        let routing = RoutingLayer::build(&centroids, dim, config);
        
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let candidates = routing.route(&query, 3);
        
        assert_eq!(candidates.len(), 3);
        assert!(routing.fits_in_cache());
    }
    
    #[test]
    fn test_int8_centroids() {
        let dim = 4;
        let centroids = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ];
        
        let int8 = Int8Centroids::from_fp32(&centroids, dim);
        
        // Check dequantization is approximate
        let recovered = int8.get_fp32(0);
        for i in 0..dim {
            let error = (recovered[i] - centroids[i]).abs();
            assert!(error < 0.1, "Int8 quantization error too high");
        }
    }
}
