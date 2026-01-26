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

//! Vector Compression Module
//!
//! Provides f16 and int8 quantization for memory reduction
//! - f16: 50% memory reduction with minimal accuracy loss
//! - int8: 75% memory reduction with 1-2% recall drop

use half::f16;
use std::borrow::Cow;

/// Stored vector with compression options
#[derive(Debug, Clone)]
pub enum StoredVector {
    /// Full precision (4 bytes per dimension)
    F32(Vec<f32>),
    /// Half precision (2 bytes per dimension, 50% memory savings)
    F16(Vec<f16>),
    /// 8-bit quantized (1 byte per dimension, 75% memory savings)
    I8(QuantizedVectorI8),
}

/// 8-bit scalar quantized vector
#[derive(Debug, Clone)]
pub struct QuantizedVectorI8 {
    /// Quantized values in [-128, 127] range
    pub values: Vec<i8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Offset for dequantization
    pub offset: f32,
}

impl StoredVector {
    /// Create from f32 vector with specified precision
    pub fn from_f32(vec: Vec<f32>, precision: CompressionLevel) -> Self {
        match precision {
            CompressionLevel::F32 => StoredVector::F32(vec),
            CompressionLevel::F16 => {
                let f16_vec: Vec<f16> = vec.iter().map(|&x| f16::from_f32(x)).collect();
                StoredVector::F16(f16_vec)
            }
            CompressionLevel::I8 => StoredVector::I8(quantize_i8(&vec)),
        }
    }

    /// Get as f32 slice - zero-copy for F32, allocation for others
    pub fn as_f32(&self) -> Cow<'_, [f32]> {
        match self {
            StoredVector::F32(v) => Cow::Borrowed(v),
            StoredVector::F16(v) => {
                let f32_vec: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
                Cow::Owned(f32_vec)
            }
            StoredVector::I8(q) => Cow::Owned(dequantize_i8(q)),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        match self {
            StoredVector::F32(v) => v.len() * std::mem::size_of::<f32>(),
            StoredVector::F16(v) => v.len() * std::mem::size_of::<f16>(),
            StoredVector::I8(q) => {
                q.values.len() * std::mem::size_of::<i8>() + std::mem::size_of::<f32>() * 2
                // scale + offset
            }
        }
    }

    /// Get dimension count
    pub fn dimension(&self) -> usize {
        match self {
            StoredVector::F32(v) => v.len(),
            StoredVector::F16(v) => v.len(),
            StoredVector::I8(q) => q.values.len(),
        }
    }
}

/// Compression level options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionLevel {
    /// No compression (4 bytes/dim)
    #[default]
    F32,
    /// Half precision (2 bytes/dim, 50% savings)
    F16,
    /// 8-bit quantization (1 byte/dim, 75% savings)
    I8,
}

/// Quantize f32 vector to i8 using min-max scaling
pub fn quantize_i8(vec: &[f32]) -> QuantizedVectorI8 {
    if vec.is_empty() {
        return QuantizedVectorI8 {
            values: Vec::new(),
            scale: 1.0,
            offset: 0.0,
        };
    }

    // Find min and max values
    let mut min_val = vec[0];
    let mut max_val = vec[0];
    for &val in vec.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    // Calculate scale and offset for mapping to [-128, 127]
    let range = max_val - min_val;
    let scale = if range > 1e-8 { range / 255.0 } else { 1.0 };
    let offset = min_val;

    // Quantize
    let values: Vec<i8> = vec
        .iter()
        .map(|&x| {
            let normalized = (x - offset) / scale;
            let quantized = (normalized - 128.0).clamp(-128.0, 127.0);
            quantized as i8
        })
        .collect();

    QuantizedVectorI8 {
        values,
        scale,
        offset,
    }
}

/// Dequantize i8 vector back to f32
pub fn dequantize_i8(q: &QuantizedVectorI8) -> Vec<f32> {
    q.values
        .iter()
        .map(|&x| (x as f32 + 128.0) * q.scale + q.offset)
        .collect()
}

/// Compute Mean Squared Error between two vectors
pub fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() {
        return f32::INFINITY;
    }

    let sum: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();

    sum / original.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_vector() -> Vec<f32> {
        vec![0.1, 0.5, -0.3, 0.8, -0.2, 0.4, 0.0, 0.9, -0.5, 0.2]
    }

    #[test]
    fn test_f16_compression() {
        let original = generate_test_vector();
        let stored = StoredVector::from_f32(original.clone(), CompressionLevel::F16);

        // Check memory savings
        assert_eq!(
            stored.memory_bytes(),
            original.len() * std::mem::size_of::<f16>()
        );

        // Check reconstruction accuracy
        let reconstructed = stored.as_f32();
        let mse = compute_mse(&original, &reconstructed);
        assert!(mse < 1e-4, "F16 MSE too high: {} (expected < 0.0001)", mse);
    }

    #[test]
    fn test_i8_quantization() {
        let original = generate_test_vector();
        let stored = StoredVector::from_f32(original.clone(), CompressionLevel::I8);

        // Check memory savings
        assert!(stored.memory_bytes() < original.len() * std::mem::size_of::<f32>());

        // Check reconstruction
        let reconstructed = stored.as_f32();
        assert_eq!(reconstructed.len(), original.len());

        // MSE should be reasonable (not perfect due to quantization)
        let mse = compute_mse(&original, &reconstructed);
        assert!(mse < 0.1, "I8 MSE too high: {}", mse);
    }

    #[test]
    fn test_quantize_dequantize_i8() {
        let original = vec![0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0];
        let quantized = quantize_i8(&original);
        let reconstructed = dequantize_i8(&quantized);

        assert_eq!(original.len(), reconstructed.len());

        // Check that values are approximately preserved
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            assert!(
                error < 0.02,
                "Quantization error too large: {} vs {} (error: {})",
                orig,
                recon,
                error
            );
        }
    }

    #[test]
    fn test_compression_memory_savings() {
        let dim = 768; // OpenAI embedding size
        let vec: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();

        let f32_stored = StoredVector::from_f32(vec.clone(), CompressionLevel::F32);
        let f16_stored = StoredVector::from_f32(vec.clone(), CompressionLevel::F16);
        let i8_stored = StoredVector::from_f32(vec.clone(), CompressionLevel::I8);

        let f32_bytes = f32_stored.memory_bytes();
        let f16_bytes = f16_stored.memory_bytes();
        let i8_bytes = i8_stored.memory_bytes();

        println!("F32: {} bytes", f32_bytes);
        println!(
            "F16: {} bytes ({:.1}% of F32)",
            f16_bytes,
            f16_bytes as f64 / f32_bytes as f64 * 100.0
        );
        println!(
            "I8:  {} bytes ({:.1}% of F32)",
            i8_bytes,
            i8_bytes as f64 / f32_bytes as f64 * 100.0
        );

        // F16 should be ~50% of F32
        assert!((f16_bytes as f64 / f32_bytes as f64) < 0.52);
        assert!(f16_bytes as f64 / f32_bytes as f64 > 0.48);

        // I8 should be ~25% of F32 (plus overhead for scale/offset)
        assert!((i8_bytes as f64 / f32_bytes as f64) < 0.30);
        assert!((i8_bytes as f64 / f32_bytes as f64) > 0.20);
    }

    #[test]
    fn test_empty_vector() {
        let empty: Vec<f32> = Vec::new();
        let stored = StoredVector::from_f32(empty, CompressionLevel::I8);
        assert_eq!(stored.dimension(), 0);
        assert_eq!(stored.memory_bytes(), 8); // Just scale + offset
    }

    #[test]
    fn test_constant_vector() {
        // All same values - edge case for quantization
        let constant = vec![0.5; 100];
        let quantized = quantize_i8(&constant);
        let reconstructed = dequantize_i8(&quantized);

        for &val in reconstructed.iter() {
            assert!((val - 0.5).abs() < 0.1);
        }
    }
}
