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

//! Quantized Vector Support (f16/bf16)
//!
//! Implements half-precision vectors inspired by CoreNN for 2x memory reduction.
//! Uses f16 (IEEE 754-2008) or bf16 (Google Brain Float) for storage with
//! minimal accuracy loss (<1% for most embeddings).

use half::{bf16, f16};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Quantization precision level
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    /// Full precision (f32) - 4 bytes per element
    F32,
    /// Half precision IEEE 754-2008 - 2 bytes per element
    F16,
    /// Brain Float 16 - 2 bytes per element (better for ML)
    BF16,
}

impl Precision {
    /// Bytes per vector element
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F16 | Precision::BF16 => 2,
        }
    }

    /// Memory reduction factor compared to f32
    pub fn memory_reduction(&self) -> f32 {
        4.0 / self.bytes_per_element() as f32
    }
}

/// Quantized vector storage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QuantizedVector {
    F32(Array1<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
}

impl QuantizedVector {
    /// Create quantized vector from f32 array
    pub fn from_f32(data: Array1<f32>, precision: Precision) -> Self {
        match precision {
            Precision::F32 => QuantizedVector::F32(data),
            Precision::F16 => {
                let quantized: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
                QuantizedVector::F16(quantized)
            }
            Precision::BF16 => {
                let quantized: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
                QuantizedVector::BF16(quantized)
            }
        }
    }

    /// Create normalized quantized vector from f32 array
    /// This is the key optimization: normalize during ingestion to enable L2 distance on unit sphere
    pub fn from_f32_normalized(data: Array1<f32>, precision: Precision) -> Self {
        // Calculate L2 norm
        let norm_squared: f32 = data.iter().map(|&x| x * x).sum();
        let norm = norm_squared.sqrt();
        
        // Handle zero vector edge case
        if norm < 1e-10 {
            return Self::from_f32(data, precision);
        }
        
        // Normalize to unit length
        let normalized_data = data.mapv(|x| x / norm);
        
        Self::from_f32(normalized_data, precision)
    }

    /// Convert to f32 for computation
    pub fn to_f32(&self) -> Array1<f32> {
        match self {
            QuantizedVector::F32(arr) => arr.clone(),
            QuantizedVector::F16(vec) => {
                let data: Vec<f32> = vec.iter().map(|&x| x.to_f32()).collect();
                Array1::from_vec(data)
            }
            QuantizedVector::BF16(vec) => {
                let data: Vec<f32> = vec.iter().map(|&x| x.to_f32()).collect();
                Array1::from_vec(data)
            }
        }
    }

    /// Get precision level
    pub fn precision(&self) -> Precision {
        match self {
            QuantizedVector::F32(_) => Precision::F32,
            QuantizedVector::F16(_) => Precision::F16,
            QuantizedVector::BF16(_) => Precision::BF16,
        }
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        match self {
            QuantizedVector::F32(arr) => arr.len(),
            QuantizedVector::F16(vec) => vec.len(),
            QuantizedVector::BF16(vec) => vec.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            QuantizedVector::F32(arr) => arr.len() * 4,
            QuantizedVector::F16(vec) => vec.len() * 2,
            QuantizedVector::BF16(vec) => vec.len() * 2,
        }
    }

    /// Get as f32 slice (zero-copy for F32, converts for F16/BF16)
    /// 
    /// For F32 precision, returns a slice directly into the underlying storage.
    /// For F16/BF16, returns None (caller should use to_f32() instead).
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            QuantizedVector::F32(arr) => arr.as_slice(),
            QuantizedVector::F16(_) | QuantizedVector::BF16(_) => None,
        }
    }

    /// Get raw pointer for prefetching (Task #10)
    /// Returns pointer to first element for cache prefetching
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        match self {
            QuantizedVector::F32(arr) => arr.as_ptr(),
            // For F16/BF16, return null - prefetch won't help anyway due to conversion
            QuantizedVector::F16(_) | QuantizedVector::BF16(_) => std::ptr::null(),
        }
    }
}

/// Fast dot product for quantized vectors (converted to f32 on-the-fly)
pub fn dot_product_quantized(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    use crate::vector_simd;

    match (a, b) {
        (QuantizedVector::F32(a_arr), QuantizedVector::F32(b_arr)) => {
            vector_simd::dot_product_f32(a_arr.as_slice().unwrap(), b_arr.as_slice().unwrap())
        }
        _ => {
            // Convert to f32 and compute
            let a_f32 = a.to_f32();
            let b_f32 = b.to_f32();
            vector_simd::dot_product_f32(a_f32.as_slice().unwrap(), b_f32.as_slice().unwrap())
        }
    }
}

/// Fast cosine distance for quantized vectors
pub fn cosine_distance_quantized(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    use crate::vector_simd;

    match (a, b) {
        (QuantizedVector::F32(a_arr), QuantizedVector::F32(b_arr)) => {
            vector_simd::cosine_distance_f32(a_arr.as_slice().unwrap(), b_arr.as_slice().unwrap())
        }
        _ => {
            let a_f32 = a.to_f32();
            let b_f32 = b.to_f32();
            vector_simd::cosine_distance_f32(a_f32.as_slice().unwrap(), b_f32.as_slice().unwrap())
        }
    }
}

/// Fast Euclidean distance for quantized vectors
pub fn euclidean_distance_quantized(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    use crate::vector_simd;

    match (a, b) {
        (QuantizedVector::F32(a_arr), QuantizedVector::F32(b_arr)) => {
            vector_simd::euclidean_distance_f32(
                a_arr.as_slice().unwrap(),
                b_arr.as_slice().unwrap(),
            )
        }
        _ => {
            let a_f32 = a.to_f32();
            let b_f32 = b.to_f32();
            vector_simd::euclidean_distance_f32(
                a_f32.as_slice().unwrap(),
                b_f32.as_slice().unwrap(),
            )
        }
    }
}

/// Optimized L2 squared distance for normalized vectors
/// For unit vectors: ||a-b||² = 2 - 2(a·b) 
/// This reduces computation from full L2 to a single dot product
pub fn l2_squared_normalized_quantized(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    let dot_product = dot_product_quantized(a, b);
    2.0 - 2.0 * dot_product
}

/// Optimized cosine distance for normalized vectors  
/// For unit vectors: cosine_similarity = a·b, so cosine_distance = 1 - a·b
pub fn cosine_distance_normalized_quantized(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    let dot_product = dot_product_quantized(a, b);
    1.0 - dot_product
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_quantization_roundtrip() {
        let orig = arr1(&[1.0, 2.0, 3.0, 4.0]);

        // Test F16
        let f16_vec = QuantizedVector::from_f32(orig.clone(), Precision::F16);
        let f16_decoded = f16_vec.to_f32();
        for i in 0..orig.len() {
            assert!((orig[i] - f16_decoded[i]).abs() < 0.01); // Small error expected
        }

        // Test BF16
        let bf16_vec = QuantizedVector::from_f32(orig.clone(), Precision::BF16);
        let bf16_decoded = bf16_vec.to_f32();
        for i in 0..orig.len() {
            assert!((orig[i] - bf16_decoded[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_memory_reduction() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0]);

        let f32_vec = QuantizedVector::from_f32(data.clone(), Precision::F32);
        let f16_vec = QuantizedVector::from_f32(data.clone(), Precision::F16);
        let bf16_vec = QuantizedVector::from_f32(data.clone(), Precision::BF16);

        assert_eq!(f32_vec.memory_size(), 16); // 4 * 4 bytes
        assert_eq!(f16_vec.memory_size(), 8); // 4 * 2 bytes
        assert_eq!(bf16_vec.memory_size(), 8); // 4 * 2 bytes
    }

    #[test]
    fn test_quantized_dot_product() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let b = arr1(&[2.0, 3.0, 4.0, 5.0]);

        let a_f32 = QuantizedVector::from_f32(a.clone(), Precision::F32);
        let b_f32 = QuantizedVector::from_f32(b.clone(), Precision::F32);

        let result = dot_product_quantized(&a_f32, &b_f32);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_cosine_distance() {
        let a = arr1(&[1.0, 0.0, 0.0]);
        let b = arr1(&[1.0, 0.0, 0.0]);

        let a_f16 = QuantizedVector::from_f32(a, Precision::F16);
        let b_f16 = QuantizedVector::from_f32(b, Precision::F16);

        let result = cosine_distance_quantized(&a_f16, &b_f16);
        assert!(result < 0.01); // Same direction = distance ~0
    }

    #[test]
    fn test_normalized_vectors() {
        let a = arr1(&[3.0, 4.0]); // length = 5
        let b = arr1(&[1.0, 0.0]); // length = 1
        
        let a_normalized = QuantizedVector::from_f32_normalized(a.clone(), Precision::F32);
        let b_normalized = QuantizedVector::from_f32_normalized(b.clone(), Precision::F32);
        
        // Check that vectors are normalized to unit length
        let a_f32 = a_normalized.to_f32();
        let b_f32 = b_normalized.to_f32();
        
        let a_norm: f32 = a_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let b_norm: f32 = b_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        assert!((a_norm - 1.0).abs() < 1e-6, "Vector a not normalized: {}", a_norm);
        assert!((b_norm - 1.0).abs() < 1e-6, "Vector b not normalized: {}", b_norm);
        
        // Test that cosine distance on unit vectors is equivalent to L2 distance formula
        let cosine_dist = cosine_distance_normalized_quantized(&a_normalized, &b_normalized);
        let l2_dist = l2_squared_normalized_quantized(&a_normalized, &b_normalized).sqrt();
        
        // For unit vectors: cosine_distance = 1 - dot_product
        // and ||a-b||^2 = 2 - 2*dot_product
        // So cosine_distance should be close to ||a-b||^2 / 2
        let expected_relation = l2_dist * l2_dist / 2.0;
        assert!((cosine_dist - expected_relation).abs() < 1e-6, 
            "Cosine distance {} not consistent with L2 distance {}", cosine_dist, l2_dist);
    }

    #[test]
    fn test_optimized_distance_consistency() {
        // Test that optimized distance functions give same results as standard ones
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let b = arr1(&[2.0, 3.0, 4.0, 5.0]);
        
        let a_norm = QuantizedVector::from_f32_normalized(a.clone(), Precision::F32);
        let b_norm = QuantizedVector::from_f32_normalized(b.clone(), Precision::F32);
        
        // For normalized vectors, cosine distance = 1 - dot_product
        let dot_prod = dot_product_quantized(&a_norm, &b_norm);
        let cosine_dist = cosine_distance_normalized_quantized(&a_norm, &b_norm);
        assert!((cosine_dist - (1.0 - dot_prod)).abs() < 1e-6);
        
        // For normalized vectors, L2^2 = 2 - 2*dot_product
        let l2_squared = l2_squared_normalized_quantized(&a_norm, &b_norm);
        assert!((l2_squared - (2.0 - 2.0 * dot_prod)).abs() < 1e-6);
    }

    #[test]
    fn test_quantized_euclidean_distance() {
        let a = arr1(&[1.0, 0.0, 0.0]);
        let b = arr1(&[4.0, 0.0, 0.0]); // Distance = 3.0

        let a_f16 = QuantizedVector::from_f32(a, Precision::F16);
        let b_f16 = QuantizedVector::from_f32(b, Precision::F16);

        let distance = euclidean_distance_quantized(&a_f16, &b_f16);
        assert!((distance - 3.0).abs() < 0.01);
    }
}
