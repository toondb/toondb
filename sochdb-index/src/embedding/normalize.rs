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

//! L2 Normalization for Embeddings
//!
//! Pre-normalizing vectors at insertion time speeds up search by eliminating
//! per-comparison normalization overhead.
//!
//! ## Benefits
//!
//! For normalized vectors:
//! - `cosine_similarity(a, b) = dot_product(a, b)`
//! - Saves 2× sqrt + 2× division per comparison (~40 CPU cycles)
//!
//! ## SIMD Optimization
//!
//! On x86_64 with AVX2:
//! - 8 floats processed per instruction
//! - ~3-5x faster than scalar implementation
//! - Automatic fallback to scalar on unsupported platforms

/// L2 normalize a vector in place (scalar implementation)
///
/// Transforms v ∈ ℝ^d to unit sphere S^(d-1):
/// v̂ = v / ‖v‖₂ = v / √(Σᵢ vᵢ²)
///
/// # Arguments
/// * `v` - Vector to normalize in place
///
/// # Returns
/// The original L2 norm of the vector
///
/// # Example
/// ```
/// use sochdb_index::embedding::normalize_l2;
///
/// let mut v = vec![3.0f32, 4.0];
/// let norm = normalize_l2(&mut v);
/// assert!((norm - 5.0).abs() < 1e-6);
/// assert!((v[0] - 0.6).abs() < 1e-6);
/// assert!((v[1] - 0.8).abs() < 1e-6);
/// ```
pub fn normalize_l2(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }

    // Compute squared norm
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();

    // Handle near-zero vectors
    if norm_sq < 1e-16 {
        // Zero vector - leave as is
        return 0.0;
    }

    let norm = norm_sq.sqrt();
    let inv_norm = 1.0 / norm;

    // Scale by reciprocal (faster than division)
    for x in v.iter_mut() {
        *x *= inv_norm;
    }

    norm
}

/// L2 normalize a vector in place using SIMD (when available)
///
/// Uses AVX2 on x86_64 for 8-wide parallel processing.
/// Falls back to scalar on other architectures.
///
/// # Arguments
/// * `v` - Vector to normalize in place (length should be multiple of 8 for best perf)
///
/// # Returns
/// The original L2 norm of the vector
#[cfg(target_arch = "x86_64")]
pub fn normalize_l2_simd(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }

    // Check for AVX2 support
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { normalize_l2_avx2(v) }
    } else {
        normalize_l2(v)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn normalize_l2_simd(v: &mut [f32]) -> f32 {
    normalize_l2(v)
}

/// AVX2-optimized L2 normalization
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normalize_l2_avx2(v: &mut [f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = v.len();

    if len < 8 {
        // Too small for SIMD, use scalar
        return normalize_l2(v);
    }

    // Phase 1: Compute squared norm with FMA
    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;

    for i in 0..chunks {
        // SAFETY: i * 8 < chunks * 8 <= len, so pointer is valid
        let ptr = unsafe { v.as_ptr().add(i * 8) };
        let vec = unsafe { _mm256_loadu_ps(ptr) };
        sum = _mm256_fmadd_ps(vec, vec, sum); // sum += vec * vec
    }

    // Horizontal sum of 8 floats
    let mut norm_sq = unsafe { horizontal_sum_avx2(sum) };

    // Add remainder (scalar)
    for val in v.iter().skip(chunks * 8) {
        norm_sq += val * val;
    }

    // Handle near-zero vectors
    if norm_sq < 1e-16 {
        return 0.0;
    }

    let norm = norm_sq.sqrt();

    // Phase 2: Multiply by reciprocal (faster than division)
    let inv_norm = _mm256_set1_ps(1.0 / norm);

    for i in 0..chunks {
        // SAFETY: i * 8 < chunks * 8 <= len, so pointer is valid
        let ptr = unsafe { v.as_mut_ptr().add(i * 8) };
        let vec = unsafe { _mm256_loadu_ps(ptr) };
        let normalized = _mm256_mul_ps(vec, inv_norm);
        unsafe { _mm256_storeu_ps(ptr, normalized) };
    }

    // Normalize remainder (scalar)
    let inv_norm_scalar = 1.0 / norm;
    for val in v.iter_mut().skip(chunks * 8) {
        *val *= inv_norm_scalar;
    }

    norm
}

/// Horizontal sum of 8 f32 values in __m256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // Sum upper and lower 128-bit lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);

    // Horizontal sum of 4 floats
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);

    _mm_cvtss_f32(sum32)
}

/// Compute L2 norm without normalizing
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Check if a vector is normalized (L2 norm ≈ 1)
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    let norm = l2_norm(v);
    (norm - 1.0).abs() < tolerance
}

/// Normalize a batch of vectors
pub fn normalize_batch(vectors: &mut [Vec<f32>]) {
    for v in vectors.iter_mut() {
        normalize_l2_simd(v);
    }
}

/// Cosine similarity for pre-normalized vectors (just dot product)
pub fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine distance for pre-normalized vectors
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_normalized(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_l2_basic() {
        let mut v = vec![3.0f32, 4.0];
        let norm = normalize_l2(&mut v);

        assert!((norm - 5.0).abs() < 1e-6, "Norm should be 5, got {}", norm);
        assert!(
            (v[0] - 0.6).abs() < 1e-6,
            "v[0] should be 0.6, got {}",
            v[0]
        );
        assert!(
            (v[1] - 0.8).abs() < 1e-6,
            "v[1] should be 0.8, got {}",
            v[1]
        );
    }

    #[test]
    fn test_normalize_l2_unit_vector() {
        let mut v = vec![1.0f32, 0.0, 0.0];
        let norm = normalize_l2(&mut v);

        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_l2_zero_vector() {
        let mut v = vec![0.0f32; 10];
        let norm = normalize_l2(&mut v);

        assert_eq!(norm, 0.0);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_l2_large_vector() {
        let mut v: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let original: Vec<f32> = v.clone();

        let norm = normalize_l2(&mut v);
        assert!(norm > 0.0);

        // Check normalized
        let new_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (new_norm - 1.0).abs() < 1e-5,
            "Normalized norm: {}",
            new_norm
        );

        // Check direction preserved (proportional)
        let ratio = original[100] / v[100];
        for i in 0..v.len() {
            if original[i].abs() > 1e-6 {
                let r = original[i] / v[i];
                assert!(
                    (r - ratio).abs() < 1e-4,
                    "Direction not preserved at index {}: {} vs {}",
                    i,
                    r,
                    ratio
                );
            }
        }
    }

    #[test]
    fn test_normalize_l2_simd() {
        // Test with various sizes including non-multiples of 8
        for size in [7, 8, 9, 16, 32, 63, 64, 100, 384] {
            let mut v: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let mut v_scalar = v.clone();

            let norm_simd = normalize_l2_simd(&mut v);
            let norm_scalar = normalize_l2(&mut v_scalar);

            assert!(
                (norm_simd - norm_scalar).abs() < 1e-5,
                "Size {}: SIMD norm {} != scalar norm {}",
                size,
                norm_simd,
                norm_scalar
            );

            for i in 0..size {
                assert!(
                    (v[i] - v_scalar[i]).abs() < 1e-5,
                    "Size {}, index {}: SIMD {} != scalar {}",
                    size,
                    i,
                    v[i],
                    v_scalar[i]
                );
            }
        }
    }

    #[test]
    fn test_is_normalized() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        assert!(!is_normalized(&v, 1e-5));

        normalize_l2(&mut v);
        assert!(is_normalized(&v, 1e-5));
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        let mut a = vec![1.0f32, 0.0, 0.0];
        let mut b = vec![1.0f32, 0.0, 0.0];
        normalize_l2(&mut a);
        normalize_l2(&mut b);

        let sim = cosine_similarity_normalized(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Same vectors should have sim=1");

        let mut c = vec![0.0f32, 1.0, 0.0];
        normalize_l2(&mut c);

        let sim_ortho = cosine_similarity_normalized(&a, &c);
        assert!(
            sim_ortho.abs() < 1e-6,
            "Orthogonal vectors should have sim=0"
        );
    }

    #[test]
    fn test_cosine_distance_normalized() {
        let mut a = vec![1.0f32, 0.0];
        let mut b = vec![-1.0f32, 0.0];
        normalize_l2(&mut a);
        normalize_l2(&mut b);

        let dist = cosine_distance_normalized(&a, &b);
        assert!(
            (dist - 2.0).abs() < 1e-6,
            "Opposite vectors should have distance=2"
        );
    }

    #[test]
    fn test_normalize_batch() {
        let mut vectors = vec![vec![3.0f32, 4.0], vec![1.0, 0.0], vec![1.0, 1.0, 1.0]];

        normalize_batch(&mut vectors);

        for v in &vectors {
            assert!(is_normalized(v, 1e-5));
        }
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0f32, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6);
    }
}
