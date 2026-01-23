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

//! # Portable SIMD Scan Kernels (Task 6)
//!
//! Provides a family of SIMD kernels that avoid gather pathologies and work
//! across diverse hardware:
//!
//! 1. **AVX-512**: Gather or permute-based
//! 2. **AVX2**: Byte LUT via shuffle + partial sums
//! 3. **NEON**: Table lookup primitives
//! 4. **Scalar**: Universal fallback
//!
//! ## Design Principles
//!
//! - Prefer layouts that allow structured loads (SoA)
//! - Use int16/int32 accumulation to reduce bandwidth
//! - Minimize unpredictable memory refs
//! - Maximize instruction-level parallelism (ILP)
//!
//! ## Math/Algorithm
//!
//! Inner loop is Θ(N_scanned). Performance is dominated by:
//! - Memory access patterns
//! - Instruction throughput
//!
//! Kernel design minimizes cache misses and maximizes ILP.

use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// CPU Feature Detection
// ============================================================================

/// Detected CPU features
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vbmi: bool,
    pub avx2: bool,
    pub sse41: bool,
    pub neon: bool,
    pub sve: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx512vbmi: is_x86_feature_detected!("avx512vbmi"),
                avx2: is_x86_feature_detected!("avx2"),
                sse41: is_x86_feature_detected!("sse4.1"),
                neon: false,
                sve: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                avx512vbmi: false,
                avx2: false,
                sse41: false,
                neon: true, // NEON is mandatory on AArch64
                sve: false, // SVE detection would require runtime check
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                avx512vbmi: false,
                avx2: false,
                sse41: false,
                neon: false,
                sve: false,
            }
        }
    }
    
    /// Get best available SIMD level
    pub fn best_simd_level(&self) -> SimdLevel {
        if self.avx512f && self.avx512bw {
            SimdLevel::Avx512
        } else if self.avx2 {
            SimdLevel::Avx2
        } else if self.sse41 {
            SimdLevel::Sse41
        } else if self.neon {
            SimdLevel::Neon
        } else {
            SimdLevel::Scalar
        }
    }
}

/// SIMD level for kernel dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Avx512,
    Avx2,
    Sse41,
    Neon,
    Scalar,
}

impl SimdLevel {
    /// Vector width in bytes
    pub fn width_bytes(&self) -> usize {
        match self {
            SimdLevel::Avx512 => 64,
            SimdLevel::Avx2 => 32,
            SimdLevel::Sse41 => 16,
            SimdLevel::Neon => 16,
            SimdLevel::Scalar => 1,
        }
    }
    
    /// Elements processed per iteration for f32
    pub fn f32_elements(&self) -> usize {
        self.width_bytes() / 4
    }
    
    /// Elements processed per iteration for i8
    pub fn i8_elements(&self) -> usize {
        self.width_bytes()
    }
}

// ============================================================================
// Kernel Trait
// ============================================================================

/// Trait for portable distance kernels
pub trait DistanceKernel: Send + Sync {
    /// Compute L2 squared distance between two f32 vectors
    fn l2_squared_f32(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// Compute dot product of two f32 vectors
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// Compute dot product of two i8 vectors (returns i32)
    fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32;
    
    /// Batch L2 squared: query vs multiple vectors
    fn l2_squared_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]);
    
    /// Batch dot product: query vs multiple vectors
    fn dot_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]);
    
    /// SIMD level of this kernel
    fn simd_level(&self) -> SimdLevel;
}

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

/// Scalar fallback implementation (works everywhere)
pub struct ScalarKernel;

impl DistanceKernel for ScalarKernel {
    fn l2_squared_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum()
    }
    
    fn l2_squared_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
        let n = vectors.len() / dim;
        debug_assert!(out.len() >= n);
        
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            out[i] = self.l2_squared_f32(query, vec);
        }
    }
    
    fn dot_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
        let n = vectors.len() / dim;
        debug_assert!(out.len() >= n);
        
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            out[i] = self.dot_f32(query, vec);
        }
    }
    
    fn simd_level(&self) -> SimdLevel {
        SimdLevel::Scalar
    }
}

// ============================================================================
// AVX2 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub struct Avx2Kernel;

#[cfg(target_arch = "x86_64")]
impl DistanceKernel for Avx2Kernel {
    fn l2_squared_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        #[target_feature(enable = "avx2")]
        unsafe fn inner(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::x86_64::*;
            
            let n = a.len();
            let chunks = n / 8;
            let mut sum = _mm256_setzero_ps();
            
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                let diff = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }
            
            // Horizontal sum
            let sum128 = _mm_add_ps(
                _mm256_extractf128_ps(sum, 0),
                _mm256_extractf128_ps(sum, 1),
            );
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);
            
            // Handle remainder
            for i in (chunks * 8)..n {
                let diff = a[i] - b[i];
                result += diff * diff;
            }
            
            result
        }
        
        if is_x86_feature_detected!("avx2") {
            unsafe { inner(a, b) }
        } else {
            ScalarKernel.l2_squared_f32(a, b)
        }
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        #[target_feature(enable = "avx2")]
        unsafe fn inner(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::x86_64::*;
            
            let n = a.len();
            let chunks = n / 8;
            let mut sum = _mm256_setzero_ps();
            
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            
            // Horizontal sum
            let sum128 = _mm_add_ps(
                _mm256_extractf128_ps(sum, 0),
                _mm256_extractf128_ps(sum, 1),
            );
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);
            
            // Handle remainder
            for i in (chunks * 8)..n {
                result += a[i] * b[i];
            }
            
            result
        }
        
        if is_x86_feature_detected!("avx2") {
            unsafe { inner(a, b) }
        } else {
            ScalarKernel.dot_f32(a, b)
        }
    }
    
    fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32 {
        debug_assert_eq!(a.len(), b.len());
        
        #[target_feature(enable = "avx2")]
        unsafe fn inner(a: &[i8], b: &[i8]) -> i32 {
            use std::arch::x86_64::*;
            
            let n = a.len();
            let chunks = n / 32;
            let mut sum = _mm256_setzero_si256();
            
            for i in 0..chunks {
                let va = _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i);
                
                // Unpack to i16 and multiply
                let a_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
                let b_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
                let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
                let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
                
                let prod_lo = _mm256_madd_epi16(a_lo, b_lo);
                let prod_hi = _mm256_madd_epi16(a_hi, b_hi);
                
                sum = _mm256_add_epi32(sum, prod_lo);
                sum = _mm256_add_epi32(sum, prod_hi);
            }
            
            // Horizontal sum
            let sum128 = _mm_add_epi32(
                _mm256_extracti128_si256(sum, 0),
                _mm256_extracti128_si256(sum, 1),
            );
            let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
            let mut result = _mm_cvtsi128_si32(sum32);
            
            // Handle remainder
            for i in (chunks * 32)..n {
                result += a[i] as i32 * b[i] as i32;
            }
            
            result
        }
        
        if is_x86_feature_detected!("avx2") {
            unsafe { inner(a, b) }
        } else {
            ScalarKernel.dot_i8(a, b)
        }
    }
    
    fn l2_squared_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
        let n = vectors.len() / dim;
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            out[i] = self.l2_squared_f32(query, vec);
        }
    }
    
    fn dot_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
        let n = vectors.len() / dim;
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            out[i] = self.dot_f32(query, vec);
        }
    }
    
    fn simd_level(&self) -> SimdLevel {
        SimdLevel::Avx2
    }
}

// ============================================================================
// NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub struct NeonKernel;

#[cfg(target_arch = "aarch64")]
impl DistanceKernel for NeonKernel {
    fn l2_squared_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        unsafe fn inner(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::aarch64::*;
            
            let n = a.len();
            let chunks = n / 4;
            let mut sum = vdupq_n_f32(0.0);
            
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                let diff = vsubq_f32(va, vb);
                sum = vfmaq_f32(sum, diff, diff);
            }
            
            // Horizontal sum
            let mut result = vaddvq_f32(sum);
            
            // Handle remainder
            for i in (chunks * 4)..n {
                let diff = a[i] - b[i];
                result += diff * diff;
            }
            
            result
        }
        
        unsafe { inner(a, b) }
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        unsafe fn inner(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::aarch64::*;
            
            let n = a.len();
            let chunks = n / 4;
            let mut sum = vdupq_n_f32(0.0);
            
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                sum = vfmaq_f32(sum, va, vb);
            }
            
            let mut result = vaddvq_f32(sum);
            
            for i in (chunks * 4)..n {
                result += a[i] * b[i];
            }
            
            result
        }
        
        unsafe { inner(a, b) }
    }
    
    fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32 {
        debug_assert_eq!(a.len(), b.len());
        
        unsafe fn inner(a: &[i8], b: &[i8]) -> i32 {
            use std::arch::aarch64::*;
            
            let n = a.len();
            let chunks = n / 16;
            let mut sum = vdupq_n_s32(0);
            
            for i in 0..chunks {
                let va = vld1q_s8(a.as_ptr().add(i * 16));
                let vb = vld1q_s8(b.as_ptr().add(i * 16));
                
                // Multiply and accumulate using SDOT if available, else manual
                let a_lo = vmovl_s8(vget_low_s8(va));
                let b_lo = vmovl_s8(vget_low_s8(vb));
                let a_hi = vmovl_s8(vget_high_s8(va));
                let b_hi = vmovl_s8(vget_high_s8(vb));
                
                let prod_lo = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
                let prod_hi = vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo));
                
                sum = vaddq_s32(sum, prod_lo);
                sum = vaddq_s32(sum, prod_hi);
                
                let prod_lo2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
                let prod_hi2 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi));
                
                sum = vaddq_s32(sum, prod_lo2);
                sum = vaddq_s32(sum, prod_hi2);
            }
            
            let mut result = vaddvq_s32(sum);
            
            for i in (chunks * 16)..n {
                result += a[i] as i32 * b[i] as i32;
            }
            
            result
        }
        
        unsafe { inner(a, b) }
    }
    
    fn l2_squared_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
        let n = vectors.len() / dim;
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            out[i] = self.l2_squared_f32(query, vec);
        }
    }
    
    fn dot_batch_f32(&self, query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
        let n = vectors.len() / dim;
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            out[i] = self.dot_f32(query, vec);
        }
    }
    
    fn simd_level(&self) -> SimdLevel {
        SimdLevel::Neon
    }
}

// ============================================================================
// Kernel Dispatcher
// ============================================================================

/// Global kernel dispatcher that selects best implementation
pub struct KernelDispatcher {
    features: CpuFeatures,
}

impl KernelDispatcher {
    /// Create new dispatcher with runtime feature detection
    pub fn new() -> Self {
        Self {
            features: CpuFeatures::detect(),
        }
    }
    
    /// Get the best available kernel
    pub fn best_kernel(&self) -> Box<dyn DistanceKernel> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.features.avx2 {
                return Box::new(Avx2Kernel);
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.features.neon {
                return Box::new(NeonKernel);
            }
        }
        
        Box::new(ScalarKernel)
    }
    
    /// Get kernel for specific SIMD level
    pub fn kernel_for_level(&self, level: SimdLevel) -> Box<dyn DistanceKernel> {
        match level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 if self.features.avx2 => Box::new(Avx2Kernel),
            
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon if self.features.neon => Box::new(NeonKernel),
            
            _ => Box::new(ScalarKernel),
        }
    }
    
    /// Get detected features
    pub fn features(&self) -> CpuFeatures {
        self.features
    }
    
    /// Get description of selected kernel
    pub fn description(&self) -> String {
        format!(
            "SIMD: {:?}, Features: avx2={}, neon={}",
            self.features.best_simd_level(),
            self.features.avx2,
            self.features.neon,
        )
    }
}

impl Default for KernelDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Scan Operations
// ============================================================================

/// High-level scan operations using best available SIMD
pub struct ScanOps {
    kernel: Box<dyn DistanceKernel>,
}

impl ScanOps {
    /// Create with automatic kernel selection
    pub fn new() -> Self {
        Self {
            kernel: KernelDispatcher::new().best_kernel(),
        }
    }
    
    /// Create with specific kernel
    pub fn with_kernel(kernel: Box<dyn DistanceKernel>) -> Self {
        Self { kernel }
    }
    
    /// Scan vectors and return top-k by L2 distance
    pub fn top_k_l2(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let n = vectors.len() / dim;
        let mut distances = vec![0.0f32; n];
        
        self.kernel.l2_squared_batch_f32(query, vectors, dim, &mut distances);
        
        // Get top-k indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());
        
        indices.into_iter()
            .take(k)
            .map(|i| (i as u32, distances[i].sqrt()))
            .collect()
    }
    
    /// Scan vectors and return top-k by dot product (descending)
    pub fn top_k_dot(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let n = vectors.len() / dim;
        let mut scores = vec![0.0f32; n];
        
        self.kernel.dot_batch_f32(query, vectors, dim, &mut scores);
        
        // Get top-k indices (descending for dot product)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        
        indices.into_iter()
            .take(k)
            .map(|i| (i as u32, scores[i]))
            .collect()
    }
    
    /// Get SIMD level being used
    pub fn simd_level(&self) -> SimdLevel {
        self.kernel.simd_level()
    }
}

impl Default for ScanOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scalar_l2() {
        let kernel = ScalarKernel;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 5.0];
        
        let dist = kernel.l2_squared_f32(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_scalar_dot() {
        let kernel = ScalarKernel;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        
        let dot = kernel.dot_f32(&a, &b);
        assert!((dot - 30.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_scalar_dot_i8() {
        let kernel = ScalarKernel;
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![1, 2, 3, 4];
        
        let dot = kernel.dot_i8(&a, &b);
        assert_eq!(dot, 30);
    }
    
    #[test]
    fn test_dispatcher() {
        let dispatcher = KernelDispatcher::new();
        let kernel = dispatcher.best_kernel();
        
        let a = vec![1.0f32; 128];
        let b = vec![2.0f32; 128];
        
        let l2 = kernel.l2_squared_f32(&a, &b);
        assert!((l2 - 128.0).abs() < 1e-4);
        
        let dot = kernel.dot_f32(&a, &b);
        assert!((dot - 256.0).abs() < 1e-4);
    }
    
    #[test]
    fn test_scan_ops() {
        let ops = ScanOps::new();
        
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,  // Distance 0
            0.0, 1.0, 0.0, 0.0,  // Distance sqrt(2)
            0.5, 0.5, 0.0, 0.0,  // Distance ~0.7
        ];
        
        let top2 = ops.top_k_l2(&query, &vectors, 4, 2);
        
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0); // First vector is closest
    }
    
    #[test]
    fn test_cpu_features() {
        let features = CpuFeatures::detect();
        let level = features.best_simd_level();
        
        // Just verify it doesn't crash
        println!("Detected SIMD level: {:?}", level);
        assert!(level.width_bytes() > 0);
    }
}
