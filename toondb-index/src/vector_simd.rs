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

//! SIMD-Optimized Vector Operations
//!
//! Provides SIMD-accelerated distance calculations inspired by CoreNN.
//! Achieves 2-4x speedup over scalar operations for distance metrics.

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_storeu_ps, _mm_sub_ps,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps,
    _mm256_storeu_ps, _mm256_sub_ps,
};

/// SIMD-optimized dot product for f32 vectors
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { dot_product_avx2(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse4.1"
    ))]
    {
        unsafe { dot_product_sse(a, b) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        dot_product_neon(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        target_arch = "aarch64"
    )))]
    {
        dot_product_scalar(a, b)
    }
}

/// Scalar fallback for dot product
#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// SSE4.1-optimized dot product (4-wide SIMD)
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum_vec = _mm_setzero_ps();

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let a_vec = _mm_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(idx));
        let prod = _mm_mul_ps(a_vec, b_vec);
        sum_vec = _mm_add_ps(sum_vec, prod);
    }

    // Horizontal sum of 4 elements
    let mut sum_array = [0f32; 4];
    _mm_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut result = sum_array.iter().sum::<f32>();

    // Handle remainder
    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }

    result
}

/// AVX2-optimized dot product (8-wide SIMD, 2x faster than SSE)
/// Uses FMA (Fused Multiply-Add) for 2x throughput improvement
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum_vec = _mm256_setzero_ps();

    // Process 8 elements at a time with FMA
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
        // FMA: sum_vec = a_vec * b_vec + sum_vec (single instruction!)
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // Horizontal sum of 8 elements
    let mut sum_array = [0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut result = sum_array.iter().sum::<f32>();

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

/// SIMD-optimized squared L2 norm (for Euclidean distance)
#[inline]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { l2_squared_avx2(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse4.1"
    ))]
    {
        unsafe { l2_squared_sse(a, b) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        l2_squared_neon(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        target_arch = "aarch64"
    )))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[target_feature(enable = "sse4.1")]
unsafe fn l2_squared_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum_vec = _mm_setzero_ps();

    for i in 0..chunks {
        let idx = i * 4;
        let a_vec = _mm_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm_sub_ps(a_vec, b_vec);
        let squared = _mm_mul_ps(diff, diff);
        sum_vec = _mm_add_ps(sum_vec, squared);
    }

    let mut sum_array = [0f32; 4];
    _mm_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut result = sum_array.iter().sum::<f32>();

    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        // FMA: sum_vec = diff * diff + sum_vec
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    let mut sum_array = [0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut result = sum_array.iter().sum::<f32>();

    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// SIMD-optimized L2 norm (magnitude)
#[inline]
pub fn l2_norm_f32(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { l2_norm_avx2(a) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse4.1"
    ))]
    {
        unsafe { l2_norm_sse(a) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        l2_norm_neon(a)
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        target_arch = "aarch64"
    )))]
    {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[target_feature(enable = "sse4.1")]
unsafe fn l2_norm_sse(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut sum_vec = _mm_setzero_ps();

    for i in 0..chunks {
        let idx = i * 4;
        let a_vec = _mm_loadu_ps(a.as_ptr().add(idx));
        let squared = _mm_mul_ps(a_vec, a_vec);
        sum_vec = _mm_add_ps(sum_vec, squared);
    }

    let mut sum_array = [0f32; 4];
    _mm_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut result = sum_array.iter().sum::<f32>();

    for i in (chunks * 4)..len {
        result += a[i] * a[i];
    }

    result.sqrt()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_norm_avx2(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
        // FMA: sum_vec = a_vec * a_vec + sum_vec
        sum_vec = _mm256_fmadd_ps(a_vec, a_vec, sum_vec);
    }

    let mut sum_array = [0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut result = sum_array.iter().sum::<f32>();

    for i in (chunks * 8)..len {
        result += a[i] * a[i];
    }

    result.sqrt()
}

/// SIMD-optimized cosine similarity (returns distance: 1 - similarity)
#[inline]
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_f32(a, b);
    let norm_a = l2_norm_f32(a);
    let norm_b = l2_norm_f32(b);
    1.0 - (dot / (norm_a * norm_b + 1e-8))
}

/// SIMD-optimized Euclidean distance
#[inline]
pub fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_f32(a, b).sqrt()
}

// =============================================================================
// AVX-512 Implementations (16-wide SIMD, ~2x faster than AVX2)
// =============================================================================

/// AVX-512 dot product - 16 f32 per instruction
/// Provides ~2x speedup over AVX2 for large vectors (d >= 128)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps,
    };

    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 16;

    let mut sum_vec = _mm512_setzero_ps();

    // Process 16 elements at a time with FMA
    for i in 0..chunks {
        let idx = i * 16;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(idx));
        // FMA: sum_vec = a_vec * b_vec + sum_vec
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // Horizontal sum of 16 elements (AVX-512 has dedicated instruction)
    let mut result = _mm512_reduce_add_ps(sum_vec);

    // Handle remainder
    for i in (chunks * 16)..len {
        result += a[i] * b[i];
    }

    result
}

/// AVX-512 L2 squared distance
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps, _mm512_sub_ps,
    };

    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 16;

    let mut sum_vec = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm512_sub_ps(a_vec, b_vec);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    let mut result = _mm512_reduce_add_ps(sum_vec);

    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// AVX-512 L2 norm
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_norm_avx512(a: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps,
    };

    let len = a.len();
    let chunks = len / 16;

    let mut sum_vec = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        sum_vec = _mm512_fmadd_ps(a_vec, a_vec, sum_vec);
    }

    let mut result = _mm512_reduce_add_ps(sum_vec);

    for i in (chunks * 16)..len {
        result += a[i] * a[i];
    }

    result.sqrt()
}

// =============================================================================
// ARM NEON Implementations (128-bit, 4 f32 per instruction)
// =============================================================================

/// ARM NEON dot product for Apple Silicon and ARM servers
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vfmaq_f32, vld1q_f32, vmovq_n_f32};

    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut sum_vec: float32x4_t = vmovq_n_f32(0.0);

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = vld1q_f32(a.as_ptr().add(idx));
            let b_vec = vld1q_f32(b.as_ptr().add(idx));
            // FMA: sum_vec = a_vec * b_vec + sum_vec
            sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
        }

        // Horizontal sum
        let mut result = vaddvq_f32(sum_vec);

        // Handle remainder
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }

        result
    }
}

/// ARM NEON L2 squared distance
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{
        float32x4_t, vaddvq_f32, vfmaq_f32, vld1q_f32, vmovq_n_f32, vsubq_f32,
    };

    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut sum_vec: float32x4_t = vmovq_n_f32(0.0);

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = vld1q_f32(a.as_ptr().add(idx));
            let b_vec = vld1q_f32(b.as_ptr().add(idx));
            let diff = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff, diff);
        }

        let mut result = vaddvq_f32(sum_vec);

        for i in (chunks * 4)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result
    }
}

/// ARM NEON L2 norm
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_norm_neon(a: &[f32]) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vfmaq_f32, vld1q_f32, vmovq_n_f32};

    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut sum_vec: float32x4_t = vmovq_n_f32(0.0);

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = vld1q_f32(a.as_ptr().add(idx));
            sum_vec = vfmaq_f32(sum_vec, a_vec, a_vec);
        }

        let mut result = vaddvq_f32(sum_vec);

        for i in (chunks * 4)..len {
            result += a[i] * a[i];
        }

        result.sqrt()
    }
}

/// ARM NEON cosine distance
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_neon(a, b);
    let norm_a = l2_norm_neon(a);
    let norm_b = l2_norm_neon(b);
    1.0 - (dot / (norm_a * norm_b + 1e-8))
}

/// ARM NEON Euclidean distance
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_neon(a, b).sqrt()
}

// =============================================================================
// Runtime dispatch helpers
// =============================================================================

/// Check if AVX-512 is available at runtime
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Check if AVX2 is available at runtime
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Get the best available SIMD instruction set as a string
pub fn get_simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512";
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1";
        }
        return "Scalar";
    }

    #[cfg(target_arch = "aarch64")]
    {
        return "NEON";
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "Scalar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let result = dot_product_f32(&a, &b);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = l2_squared_f32(&a, &b);
        let expected = (1.0f32 - 4.0).powi(2) + (2.0f32 - 5.0).powi(2) + (3.0f32 - 6.0).powi(2);
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm() {
        let a = vec![3.0, 4.0];
        let result = l2_norm_f32(&a);
        let expected = 5.0; // 3-4-5 triangle
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let result = cosine_distance_f32(&a, &b);
        assert!(result.abs() < 1e-5); // Same vectors = 0 distance

        let c = vec![0.0, 1.0, 0.0];
        let result2 = cosine_distance_f32(&a, &c);
        assert!((result2 - 1.0).abs() < 1e-5); // Orthogonal = 1 distance
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = euclidean_distance_f32(&a, &b);
        let expected = 5.0;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_simd_level() {
        let level = get_simd_level();
        println!("SIMD level: {}", level);
        assert!(!level.is_empty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = dot_product_neon(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-4);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let result = cosine_distance_neon(&a, &b);
        assert!(result.abs() < 1e-4);
    }
}
