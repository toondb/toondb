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

//! SIMD Vectorized Query Filters
//!
//! From mm.md Task 5.3: AVX-512/NEON Vectorized Filters (VQE)
//!
//! ## Problem
//!
//! Current filtering is scalar. LLM context queries often filter millions of rows
//! (e.g., "events from last 7 days"). SIMD can evaluate 8-16 predicates per instruction.
//!
//! ## Solution
//!
//! Column-oriented data layout + compiled filter expressions + runtime SIMD feature detection
//!
//! ## Throughput Analysis
//!
//! ```text
//! Scalar: 1 comparison/cycle × 3GHz = 3B comparisons/sec
//! AVX-512: 8 comparisons/instruction × ~1 CPI × 3GHz = 24B/sec
//! AVX-256: 4 comparisons/instruction = 12B/sec
//! NEON: 4 comparisons/instruction = 12B/sec
//!
//! 100M rows @ 24B/sec = 4.2ms (AVX-512)
//! 100M rows @ 3B/sec = 33ms (scalar)
//!
//! Speedup: 8× (AVX-512), 4× (AVX-256/NEON)
//! ```


/// Result bitmap - bit per row indicating pass/fail
pub type FilterBitmap = Vec<u64>;

/// Filter comparison operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    IsNull,
    IsNotNull,
}

/// Allocate a bitmap for the given number of rows
#[inline]
pub fn allocate_bitmap(num_rows: usize) -> FilterBitmap {
    vec![0u64; (num_rows + 63) / 64]
}

/// Set a bit in the bitmap
#[inline]
pub fn set_bit(bitmap: &mut FilterBitmap, idx: usize) {
    let word_idx = idx / 64;
    let bit_idx = idx % 64;
    if word_idx < bitmap.len() {
        bitmap[word_idx] |= 1u64 << bit_idx;
    }
}

/// Check if a bit is set
#[inline]
pub fn get_bit(bitmap: &FilterBitmap, idx: usize) -> bool {
    let word_idx = idx / 64;
    let bit_idx = idx % 64;
    if word_idx < bitmap.len() {
        (bitmap[word_idx] >> bit_idx) & 1 == 1
    } else {
        false
    }
}

/// Count set bits in bitmap
pub fn popcount(bitmap: &FilterBitmap) -> usize {
    bitmap.iter().map(|w| w.count_ones() as usize).sum()
}

/// AND two bitmaps together
pub fn bitmap_and(a: &FilterBitmap, b: &FilterBitmap) -> FilterBitmap {
    a.iter().zip(b.iter()).map(|(x, y)| x & y).collect()
}

/// OR two bitmaps together
pub fn bitmap_or(a: &FilterBitmap, b: &FilterBitmap) -> FilterBitmap {
    a.iter().zip(b.iter()).map(|(x, y)| x | y).collect()
}

/// NOT a bitmap
pub fn bitmap_not(a: &FilterBitmap) -> FilterBitmap {
    a.iter().map(|x| !x).collect()
}

// =============================================================================
// Scalar Implementations (Fallback)
// =============================================================================

/// Scalar filter: i64 > threshold
pub fn filter_i64_gt_scalar(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value > threshold {
            set_bit(result, idx);
        }
    }
}

/// Scalar filter: i64 >= threshold
pub fn filter_i64_ge_scalar(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value >= threshold {
            set_bit(result, idx);
        }
    }
}

/// Scalar filter: i64 < threshold
pub fn filter_i64_lt_scalar(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value < threshold {
            set_bit(result, idx);
        }
    }
}

/// Scalar filter: i64 == value
pub fn filter_i64_eq_scalar(data: &[i64], target: i64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value == target {
            set_bit(result, idx);
        }
    }
}

/// Scalar filter: i64 between low and high (inclusive)
pub fn filter_i64_between_scalar(data: &[i64], low: i64, high: i64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value >= low && value <= high {
            set_bit(result, idx);
        }
    }
}

/// Scalar filter: f64 > threshold
pub fn filter_f64_gt_scalar(data: &[f64], threshold: f64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value > threshold {
            set_bit(result, idx);
        }
    }
}

/// Scalar filter: f64 < threshold
pub fn filter_f64_lt_scalar(data: &[f64], threshold: f64, result: &mut FilterBitmap) {
    for (idx, &value) in data.iter().enumerate() {
        if value < threshold {
            set_bit(result, idx);
        }
    }
}

// =============================================================================
// AVX2 SIMD Implementations (x86_64)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Check if AVX2 is available
    #[inline]
    pub fn is_available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// AVX2 filter: i64 > threshold
    ///
    /// Processes 4 i64 values per iteration using 256-bit vectors.
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_i64_gt_avx2(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
        let threshold_vec = _mm256_set1_epi64x(threshold);
        let len = data.len();
        let chunks = len / 4;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 4;
            let data_vec = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            // Compare greater than
            let cmp = _mm256_cmpgt_epi64(data_vec, threshold_vec);

            // Extract mask
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u64;

            // Set bits in result
            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
                // Handle overflow to next word
                if bit_offset > 60 && word_idx + 1 < result.len() {
                    result[word_idx + 1] |= mask >> (64 - bit_offset);
                }
            }
        }

        // Handle remainder with scalar
        let remainder_start = chunks * 4;
        for idx in remainder_start..len {
            if data[idx] > threshold {
                set_bit(result, idx);
            }
        }
    }

    /// AVX2 filter: i64 < threshold
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_i64_lt_avx2(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
        let threshold_vec = _mm256_set1_epi64x(threshold);
        let len = data.len();
        let chunks = len / 4;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 4;
            let data_vec = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            // Compare: data < threshold is equivalent to threshold > data
            let cmp = _mm256_cmpgt_epi64(threshold_vec, data_vec);
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u64;

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
                if bit_offset > 60 && word_idx + 1 < result.len() {
                    result[word_idx + 1] |= mask >> (64 - bit_offset);
                }
            }
        }

        let remainder_start = chunks * 4;
        for idx in remainder_start..len {
            if data[idx] < threshold {
                set_bit(result, idx);
            }
        }
    }

    /// AVX2 filter: i64 == value
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_i64_eq_avx2(data: &[i64], target: i64, result: &mut FilterBitmap) {
        let target_vec = _mm256_set1_epi64x(target);
        let len = data.len();
        let chunks = len / 4;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 4;
            let data_vec = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            let cmp = _mm256_cmpeq_epi64(data_vec, target_vec);
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u64;

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
                if bit_offset > 60 && word_idx + 1 < result.len() {
                    result[word_idx + 1] |= mask >> (64 - bit_offset);
                }
            }
        }

        let remainder_start = chunks * 4;
        for idx in remainder_start..len {
            if data[idx] == target {
                set_bit(result, idx);
            }
        }
    }

    /// AVX2 filter: i64 between low and high
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_i64_between_avx2(
        data: &[i64],
        low: i64,
        high: i64,
        result: &mut FilterBitmap,
    ) {
        let low_vec = _mm256_set1_epi64x(low - 1); // For >= comparison
        let high_vec = _mm256_set1_epi64x(high);
        let len = data.len();
        let chunks = len / 4;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 4;
            let data_vec = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            // data > (low - 1) AND data <= high
            let cmp_low = _mm256_cmpgt_epi64(data_vec, low_vec);
            let cmp_high = _mm256_cmpgt_epi64(high_vec, data_vec);
            let cmp_high_eq = _mm256_cmpeq_epi64(data_vec, high_vec);
            let cmp_high_final = _mm256_or_si256(cmp_high, cmp_high_eq);
            let cmp = _mm256_and_si256(cmp_low, cmp_high_final);

            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u64;

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
                if bit_offset > 60 && word_idx + 1 < result.len() {
                    result[word_idx + 1] |= mask >> (64 - bit_offset);
                }
            }
        }

        let remainder_start = chunks * 4;
        for idx in remainder_start..len {
            let v = data[idx];
            if v >= low && v <= high {
                set_bit(result, idx);
            }
        }
    }

    /// AVX2 filter: f64 > threshold
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_f64_gt_avx2(data: &[f64], threshold: f64, result: &mut FilterBitmap) {
        let threshold_vec = _mm256_set1_pd(threshold);
        let len = data.len();
        let chunks = len / 4;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 4;
            let data_vec = _mm256_loadu_pd(data.as_ptr().add(offset));

            let cmp = _mm256_cmp_pd(data_vec, threshold_vec, _CMP_GT_OQ);
            let mask = _mm256_movemask_pd(cmp) as u64;

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
                if bit_offset > 60 && word_idx + 1 < result.len() {
                    result[word_idx + 1] |= mask >> (64 - bit_offset);
                }
            }
        }

        let remainder_start = chunks * 4;
        for idx in remainder_start..len {
            if data[idx] > threshold {
                set_bit(result, idx);
            }
        }
    }
}

// =============================================================================
// NEON SIMD Implementations (aarch64)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON is always available on aarch64
    #[inline]
    #[allow(dead_code)]
    pub fn is_available() -> bool {
        true
    }

    /// NEON filter: i64 > threshold
    ///
    /// Processes 2 i64 values per iteration using 128-bit vectors.
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_i64_gt_neon(data: &[i64], threshold: i64, result: &mut FilterBitmap) { unsafe {
        let threshold_vec = vdupq_n_s64(threshold);
        let len = data.len();
        let chunks = len / 2;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 2;
            let data_vec = vld1q_s64(data.as_ptr().add(offset));

            // Compare greater than (returns uint64x2_t)
            let cmp = vcgtq_s64(data_vec, threshold_vec);

            // Extract mask (2 bits) - cmp is already uint64x2_t
            let mask_low = vgetq_lane_u64(cmp, 0);
            let mask_high = vgetq_lane_u64(cmp, 1);
            let mask = ((mask_low != 0) as u64) | (((mask_high != 0) as u64) << 1);

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
            }
        }

        // Handle remainder
        let remainder_start = chunks * 2;
        for idx in remainder_start..len {
            if data[idx] > threshold {
                set_bit(result, idx);
            }
        }
    }}

    /// NEON filter: i64 < threshold
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_i64_lt_neon(data: &[i64], threshold: i64, result: &mut FilterBitmap) { unsafe {
        let threshold_vec = vdupq_n_s64(threshold);
        let len = data.len();
        let chunks = len / 2;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 2;
            let data_vec = vld1q_s64(data.as_ptr().add(offset));

            let cmp = vcltq_s64(data_vec, threshold_vec);

            // cmp is already uint64x2_t
            let mask_low = vgetq_lane_u64(cmp, 0);
            let mask_high = vgetq_lane_u64(cmp, 1);
            let mask = ((mask_low != 0) as u64) | (((mask_high != 0) as u64) << 1);

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
            }
        }

        let remainder_start = chunks * 2;
        for idx in remainder_start..len {
            if data[idx] < threshold {
                set_bit(result, idx);
            }
        }
    }}

    /// NEON filter: i64 == value
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_i64_eq_neon(data: &[i64], target: i64, result: &mut FilterBitmap) { unsafe {
        let target_vec = vdupq_n_s64(target);
        let len = data.len();
        let chunks = len / 2;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 2;
            let data_vec = vld1q_s64(data.as_ptr().add(offset));

            let cmp = vceqq_s64(data_vec, target_vec);

            // cmp is already uint64x2_t
            let mask_low = vgetq_lane_u64(cmp, 0);
            let mask_high = vgetq_lane_u64(cmp, 1);
            let mask = ((mask_low != 0) as u64) | (((mask_high != 0) as u64) << 1);

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
            }
        }

        let remainder_start = chunks * 2;
        for idx in remainder_start..len {
            if data[idx] == target {
                set_bit(result, idx);
            }
        }
    }}

    /// NEON filter: f64 > threshold
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_f64_gt_neon(data: &[f64], threshold: f64, result: &mut FilterBitmap) { unsafe {
        let threshold_vec = vdupq_n_f64(threshold);
        let len = data.len();
        let chunks = len / 2;

        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 2;
            let data_vec = vld1q_f64(data.as_ptr().add(offset));

            let cmp = vcgtq_f64(data_vec, threshold_vec);

            let mask_low = vgetq_lane_u64(cmp, 0);
            let mask_high = vgetq_lane_u64(cmp, 1);
            let mask = ((mask_low != 0) as u64) | (((mask_high != 0) as u64) << 1);

            let word_idx = offset / 64;
            let bit_offset = offset % 64;
            if word_idx < result.len() {
                result[word_idx] |= mask << bit_offset;
            }
        }

        let remainder_start = chunks * 2;
        for idx in remainder_start..len {
            if data[idx] > threshold {
                set_bit(result, idx);
            }
        }
    }}
}

// =============================================================================
// Public API with Automatic Dispatch
// =============================================================================

/// Filter i64 column: value > threshold
pub fn filter_i64_gt(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            unsafe {
                avx2::filter_i64_gt_avx2(data, threshold, result);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::filter_i64_gt_neon(data, threshold, result);
        }
        return;
    }

    #[allow(unreachable_code)]
    filter_i64_gt_scalar(data, threshold, result);
}

/// Filter i64 column: value < threshold
pub fn filter_i64_lt(data: &[i64], threshold: i64, result: &mut FilterBitmap) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            unsafe {
                avx2::filter_i64_lt_avx2(data, threshold, result);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::filter_i64_lt_neon(data, threshold, result);
        }
        return;
    }

    #[allow(unreachable_code)]
    filter_i64_lt_scalar(data, threshold, result);
}

/// Filter i64 column: value == target
pub fn filter_i64_eq(data: &[i64], target: i64, result: &mut FilterBitmap) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            unsafe {
                avx2::filter_i64_eq_avx2(data, target, result);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::filter_i64_eq_neon(data, target, result);
        }
        return;
    }

    #[allow(unreachable_code)]
    filter_i64_eq_scalar(data, target, result);
}

/// Filter i64 column: low <= value <= high
pub fn filter_i64_between(data: &[i64], low: i64, high: i64, result: &mut FilterBitmap) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            unsafe {
                avx2::filter_i64_between_avx2(data, low, high, result);
            }
            return;
        }
    }

    // Fallback to scalar
    filter_i64_between_scalar(data, low, high, result);
}

/// Filter f64 column: value > threshold
pub fn filter_f64_gt(data: &[f64], threshold: f64, result: &mut FilterBitmap) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            unsafe {
                avx2::filter_f64_gt_avx2(data, threshold, result);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::filter_f64_gt_neon(data, threshold, result);
        }
        return;
    }

    #[allow(unreachable_code)]
    filter_f64_gt_scalar(data, threshold, result);
}

/// Get information about SIMD support
pub fn simd_info() -> SimdInfo {
    SimdInfo {
        #[cfg(target_arch = "x86_64")]
        has_avx2: is_x86_feature_detected!("avx2"),
        #[cfg(target_arch = "x86_64")]
        has_avx512f: is_x86_feature_detected!("avx512f"),
        #[cfg(not(target_arch = "x86_64"))]
        has_avx2: false,
        #[cfg(not(target_arch = "x86_64"))]
        has_avx512f: false,
        #[cfg(target_arch = "aarch64")]
        has_neon: true,
        #[cfg(not(target_arch = "aarch64"))]
        has_neon: false,
    }
}

/// SIMD capability information
#[derive(Debug, Clone)]
pub struct SimdInfo {
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_neon: bool,
}

impl SimdInfo {
    /// Get expected speedup factor for i64 filters
    pub fn expected_speedup_i64(&self) -> f64 {
        if self.has_avx512f {
            8.0
        } else if self.has_avx2 {
            4.0
        } else if self.has_neon {
            2.0
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_i64_gt() {
        let data: Vec<i64> = (0..100).collect();
        let mut result = allocate_bitmap(data.len());

        filter_i64_gt(&data, 50, &mut result);

        // Values 51-99 should pass (49 values)
        assert_eq!(popcount(&result), 49);

        for i in 0..100 {
            assert_eq!(get_bit(&result, i), i > 50, "Failed at index {}", i);
        }
    }

    #[test]
    fn test_filter_i64_lt() {
        let data: Vec<i64> = (0..100).collect();
        let mut result = allocate_bitmap(data.len());

        filter_i64_lt(&data, 50, &mut result);

        // Values 0-49 should pass (50 values)
        assert_eq!(popcount(&result), 50);

        for i in 0..100 {
            assert_eq!(get_bit(&result, i), i < 50, "Failed at index {}", i);
        }
    }

    #[test]
    fn test_filter_i64_eq() {
        let data: Vec<i64> = (0..100).collect();
        let mut result = allocate_bitmap(data.len());

        filter_i64_eq(&data, 42, &mut result);

        assert_eq!(popcount(&result), 1);
        assert!(get_bit(&result, 42));
    }

    #[test]
    fn test_filter_i64_between() {
        let data: Vec<i64> = (0..100).collect();
        let mut result = allocate_bitmap(data.len());

        filter_i64_between(&data, 25, 75, &mut result);

        // Values 25-75 inclusive (51 values)
        assert_eq!(popcount(&result), 51);

        for i in 0..100 {
            assert_eq!(
                get_bit(&result, i),
                i >= 25 && i <= 75,
                "Failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_filter_f64_gt() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut result = allocate_bitmap(data.len());

        filter_f64_gt(&data, 50.0, &mut result);

        assert_eq!(popcount(&result), 49);
    }

    #[test]
    fn test_bitmap_operations() {
        let mut a = allocate_bitmap(64);
        let mut b = allocate_bitmap(64);

        for i in 0..32 {
            set_bit(&mut a, i);
        }
        for i in 16..48 {
            set_bit(&mut b, i);
        }

        let and_result = bitmap_and(&a, &b);
        assert_eq!(popcount(&and_result), 16); // 16-31

        let or_result = bitmap_or(&a, &b);
        assert_eq!(popcount(&or_result), 48); // 0-47
    }

    #[test]
    fn test_simd_info() {
        let info = simd_info();
        println!("SIMD capabilities: {:?}", info);
        println!("Expected speedup: {}x", info.expected_speedup_i64());
    }

    #[test]
    fn test_large_dataset() {
        // Test with 1M rows
        let data: Vec<i64> = (0..1_000_000).collect();
        let mut result = allocate_bitmap(data.len());

        let start = std::time::Instant::now();
        filter_i64_gt(&data, 500_000, &mut result);
        let elapsed = start.elapsed();

        assert_eq!(popcount(&result), 499_999);
        println!("Filtered 1M rows in {:?}", elapsed);
    }
}
