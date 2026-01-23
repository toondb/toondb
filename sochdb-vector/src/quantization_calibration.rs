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

//! # Quantization Error Calibration (Task 5)
//!
//! Calibrates quantization error per-list (or per-cluster) and uses it in stopping decisions.
//!
//! ## Problem
//!
//! With PQ/ADC, the kth score is a proxy score, not the true score.
//! Stopping on list bounds vs kth requires them to be comparable in the true metric.
//!
//! ## Solution
//!
//! Learn empirical error envelopes:
//! - Per-list quantiles for ε = ŝ - s under representative queries
//! - At query time, convert proxy thresholds into safe true-score thresholds
//!
//! ## Math/Algorithm
//!
//! PAC-style calibration:
//! - Store ε_L(1-δ) such that P(ε ≤ ε_L) ≥ 1-δ
//! - Stopping compares LB_true(list) vs UB_true(kth) using these envelopes
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::quantization_calibration::{ErrorCalibrator, ErrorEnvelope};
//!
//! // During offline training
//! let mut calibrator = ErrorCalibrator::new(n_lists);
//! calibrator.record_error(list_idx, proxy_score, true_score);
//! let envelopes = calibrator.finalize();
//!
//! // At query time
//! let proxy_kth = 0.85;
//! let safe_threshold = envelopes.safe_true_threshold(list_idx, proxy_kth, 0.99);
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ============================================================================
// Error Sample
// ============================================================================

/// A single error sample: ε = proxy - true
#[derive(Debug, Clone, Copy)]
pub struct ErrorSample {
    /// Proxy score (from quantized representation)
    pub proxy: f32,
    /// True score (from full-precision computation)
    pub true_score: f32,
    /// Error: proxy - true
    pub error: f32,
}

impl ErrorSample {
    /// Create from proxy and true scores
    pub fn new(proxy: f32, true_score: f32) -> Self {
        Self {
            proxy,
            true_score,
            error: proxy - true_score,
        }
    }
}

// ============================================================================
// Error Envelope
// ============================================================================

/// Pre-computed error envelope for a list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    /// List index
    pub list_idx: u32,
    
    /// Quantiles of error distribution
    /// Key: quantile (e.g., 0.95, 0.99, 0.999)
    /// Value: error bound at that quantile
    pub quantiles: HashMap<u32, f32>, // Use u32 for serialization (quantile * 10000)
    
    /// Mean error
    pub mean_error: f32,
    
    /// Standard deviation of error
    pub std_error: f32,
    
    /// Maximum observed error
    pub max_error: f32,
    
    /// Minimum observed error  
    pub min_error: f32,
    
    /// Number of samples used for calibration
    pub sample_count: u32,
}

impl ErrorEnvelope {
    /// Get error bound for a quantile
    /// 
    /// Returns ε such that P(error ≤ ε) ≥ quantile
    pub fn error_at_quantile(&self, quantile: f32) -> f32 {
        let key = (quantile * 10000.0).round() as u32;
        
        // Direct lookup
        if let Some(&error) = self.quantiles.get(&key) {
            return error;
        }
        
        // Interpolate between available quantiles
        let mut below_key = 0u32;
        let mut above_key = 10000u32;
        let mut below_val = self.min_error;
        let mut above_val = self.max_error;
        
        for (&k, &v) in &self.quantiles {
            if k < key && k > below_key {
                below_key = k;
                below_val = v;
            }
            if k > key && k < above_key {
                above_key = k;
                above_val = v;
            }
        }
        
        // Linear interpolation
        if above_key > below_key {
            let t = (key - below_key) as f32 / (above_key - below_key) as f32;
            below_val + t * (above_val - below_val)
        } else {
            self.max_error
        }
    }
    
    /// Convert proxy threshold to safe true-score threshold
    ///
    /// For similarity (higher is better):
    /// true_score ≥ proxy - error_bound
    ///
    /// Returns threshold such that P(true ≥ threshold | proxy = p) ≥ confidence
    pub fn safe_true_threshold(&self, proxy: f32, confidence: f32) -> f32 {
        let error_bound = self.error_at_quantile(confidence);
        proxy - error_bound
    }
    
    /// Convert true threshold to safe proxy threshold
    ///
    /// For filtering candidates before rerank:
    /// proxy ≥ true + error_bound (conservative)
    pub fn safe_proxy_threshold(&self, true_threshold: f32, confidence: f32) -> f32 {
        let error_bound = self.error_at_quantile(confidence);
        true_threshold + error_bound
    }
    
    /// Check if proxy score definitely beats true threshold
    pub fn definitely_beats(&self, proxy: f32, true_threshold: f32) -> bool {
        // Use max error for deterministic guarantee
        proxy - self.max_error > true_threshold
    }
    
    /// Check if proxy score might beat true threshold
    pub fn might_beat(&self, proxy: f32, true_threshold: f32, confidence: f32) -> bool {
        let error_bound = self.error_at_quantile(confidence);
        proxy - error_bound > true_threshold
    }
}

impl Default for ErrorEnvelope {
    fn default() -> Self {
        Self {
            list_idx: 0,
            quantiles: HashMap::new(),
            mean_error: 0.0,
            std_error: 0.0,
            max_error: 0.0,
            min_error: 0.0,
            sample_count: 0,
        }
    }
}

// ============================================================================
// Error Calibrator
// ============================================================================

/// Collects error samples and computes envelopes
pub struct ErrorCalibrator {
    /// Samples per list
    samples: Vec<Vec<ErrorSample>>,
    /// Number of lists
    n_lists: usize,
    /// Quantiles to compute
    quantiles: Vec<f32>,
}

impl ErrorCalibrator {
    /// Create new calibrator for n_lists
    pub fn new(n_lists: usize) -> Self {
        Self {
            samples: vec![Vec::new(); n_lists],
            n_lists,
            quantiles: vec![0.50, 0.75, 0.90, 0.95, 0.99, 0.999],
        }
    }
    
    /// Create with custom quantiles
    pub fn with_quantiles(n_lists: usize, quantiles: Vec<f32>) -> Self {
        Self {
            samples: vec![Vec::new(); n_lists],
            n_lists,
            quantiles,
        }
    }
    
    /// Record an error sample for a list
    pub fn record_error(&mut self, list_idx: usize, proxy: f32, true_score: f32) {
        if list_idx < self.n_lists {
            self.samples[list_idx].push(ErrorSample::new(proxy, true_score));
        }
    }
    
    /// Record multiple samples for a list
    pub fn record_errors(&mut self, list_idx: usize, samples: &[(f32, f32)]) {
        if list_idx < self.n_lists {
            for &(proxy, true_score) in samples {
                self.samples[list_idx].push(ErrorSample::new(proxy, true_score));
            }
        }
    }
    
    /// Compute envelopes for all lists
    pub fn finalize(&self) -> ErrorEnvelopeSet {
        let envelopes: Vec<ErrorEnvelope> = (0..self.n_lists)
            .map(|i| self.compute_envelope(i))
            .collect();
        
        // Also compute global envelope
        let global = self.compute_global_envelope();
        
        ErrorEnvelopeSet { envelopes, global }
    }
    
    /// Compute envelope for a single list
    fn compute_envelope(&self, list_idx: usize) -> ErrorEnvelope {
        let samples = &self.samples[list_idx];
        
        if samples.is_empty() {
            return ErrorEnvelope {
                list_idx: list_idx as u32,
                ..Default::default()
            };
        }
        
        // Extract errors
        let mut errors: Vec<f32> = samples.iter().map(|s| s.error).collect();
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = errors.len();
        
        // Compute statistics
        let sum: f32 = errors.iter().sum();
        let mean = sum / n as f32;
        let variance: f32 = errors.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();
        
        // Compute quantiles
        let mut quantiles = HashMap::new();
        for &q in &self.quantiles {
            let idx = ((n as f32 * q) as usize).min(n - 1);
            let key = (q * 10000.0).round() as u32;
            quantiles.insert(key, errors[idx]);
        }
        
        ErrorEnvelope {
            list_idx: list_idx as u32,
            quantiles,
            mean_error: mean,
            std_error: std,
            max_error: errors[n - 1],
            min_error: errors[0],
            sample_count: n as u32,
        }
    }
    
    /// Compute global envelope across all lists
    fn compute_global_envelope(&self) -> ErrorEnvelope {
        let mut all_errors: Vec<f32> = self.samples.iter()
            .flat_map(|s| s.iter().map(|e| e.error))
            .collect();
        
        if all_errors.is_empty() {
            return ErrorEnvelope::default();
        }
        
        all_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all_errors.len();
        
        let sum: f32 = all_errors.iter().sum();
        let mean = sum / n as f32;
        let variance: f32 = all_errors.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();
        
        let mut quantiles = HashMap::new();
        for &q in &self.quantiles {
            let idx = ((n as f32 * q) as usize).min(n - 1);
            let key = (q * 10000.0).round() as u32;
            quantiles.insert(key, all_errors[idx]);
        }
        
        ErrorEnvelope {
            list_idx: u32::MAX, // Indicates global
            quantiles,
            mean_error: mean,
            std_error: std,
            max_error: all_errors[n - 1],
            min_error: all_errors[0],
            sample_count: n as u32,
        }
    }
}

// ============================================================================
// Error Envelope Set
// ============================================================================

/// Collection of error envelopes for all lists
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelopeSet {
    /// Per-list envelopes
    pub envelopes: Vec<ErrorEnvelope>,
    /// Global envelope
    pub global: ErrorEnvelope,
}

impl ErrorEnvelopeSet {
    /// Get envelope for a list, falling back to global if not available
    pub fn get(&self, list_idx: usize) -> &ErrorEnvelope {
        if list_idx < self.envelopes.len() && self.envelopes[list_idx].sample_count > 0 {
            &self.envelopes[list_idx]
        } else {
            &self.global
        }
    }
    
    /// Convert proxy kth to safe true threshold
    pub fn safe_true_threshold(&self, list_idx: usize, proxy: f32, confidence: f32) -> f32 {
        self.get(list_idx).safe_true_threshold(proxy, confidence)
    }
    
    /// Check if we can terminate: all remaining lists have bounds below kth true threshold
    pub fn can_terminate(
        &self,
        kth_proxy: f32,
        remaining_list_bounds: &[(usize, f32)],
        confidence: f32,
    ) -> bool {
        // Convert kth proxy to safe true threshold (lower bound on true kth)
        let kth_true_lower = self.global.safe_true_threshold(kth_proxy, confidence);
        
        // Check if all remaining list bounds are below kth true threshold
        remaining_list_bounds.iter().all(|(list_idx, bound)| {
            // Use per-list envelope for tighter bounds
            let envelope = self.get(*list_idx);
            // The bound is an upper bound on proxy scores in the list
            // Convert to upper bound on true scores
            let true_upper = *bound + envelope.max_error.abs();
            true_upper < kth_true_lower
        })
    }
    
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }
}

// ============================================================================
// Calibration Runner
// ============================================================================

/// Runs calibration using representative queries
pub struct CalibrationRunner {
    /// Number of lists
    n_lists: usize,
    /// Quantization function (takes vector, returns quantized codes)
    quantize_fn: Option<Box<dyn Fn(&[f32]) -> Vec<u8> + Send + Sync>>,
    /// Distance function for proxy (takes query, codes, returns score)
    proxy_distance_fn: Option<Box<dyn Fn(&[f32], &[u8]) -> f32 + Send + Sync>>,
    /// Distance function for true (takes query, vector, returns score)
    true_distance_fn: Option<Box<dyn Fn(&[f32], &[f32]) -> f32 + Send + Sync>>,
}

impl CalibrationRunner {
    /// Create new calibration runner
    pub fn new(n_lists: usize) -> Self {
        Self {
            n_lists,
            quantize_fn: None,
            proxy_distance_fn: None,
            true_distance_fn: None,
        }
    }
    
    /// Run calibration with given queries and vectors per list
    ///
    /// For each query, computes proxy and true scores for vectors in each list,
    /// collecting error samples.
    pub fn calibrate(
        &self,
        queries: &[Vec<f32>],
        lists: &[Vec<Vec<f32>>],
        quantized_lists: &[Vec<Vec<u8>>],
    ) -> ErrorEnvelopeSet {
        let mut calibrator = ErrorCalibrator::new(self.n_lists);
        
        for query in queries {
            for (list_idx, (vectors, codes)) in lists.iter().zip(quantized_lists.iter()).enumerate() {
                for (vec, code) in vectors.iter().zip(codes.iter()) {
                    // Compute true and proxy scores
                    let true_score = dot_product(query, vec);
                    let proxy_score = if let Some(ref f) = self.proxy_distance_fn {
                        f(query, code)
                    } else {
                        true_score // Fallback: no quantization error
                    };
                    
                    calibrator.record_error(list_idx, proxy_score, true_score);
                }
            }
        }
        
        calibrator.finalize()
    }
    
    /// Simplified calibration using synthetic error model
    ///
    /// Generates error samples based on assumed error distribution.
    pub fn calibrate_synthetic(
        n_lists: usize,
        mean_error: f32,
        std_error: f32,
        samples_per_list: usize,
    ) -> ErrorEnvelopeSet {
        let mut calibrator = ErrorCalibrator::new(n_lists);
        
        // Use simple random number generation for reproducibility
        let mut rng_state: u64 = 12345;
        let mut rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32
        };
        
        for list_idx in 0..n_lists {
            for _ in 0..samples_per_list {
                // Box-Muller transform for normal distribution
                let u1 = rand();
                let u2 = rand();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let error = mean_error + std_error * z;
                
                let true_score = 0.5 + rand() * 0.5; // Random true score in [0.5, 1.0]
                let proxy_score = true_score + error;
                
                calibrator.record_error(list_idx, proxy_score, true_score);
            }
        }
        
        calibrator.finalize()
    }
}

/// Dot product helper
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_sample() {
        let sample = ErrorSample::new(0.92, 0.90);
        assert!((sample.error - 0.02).abs() < 1e-6);
    }
    
    #[test]
    fn test_calibrator() {
        let mut calibrator = ErrorCalibrator::new(3);
        
        // Add samples for list 0
        calibrator.record_error(0, 0.90, 0.88);
        calibrator.record_error(0, 0.85, 0.82);
        calibrator.record_error(0, 0.92, 0.91);
        calibrator.record_error(0, 0.88, 0.85);
        calibrator.record_error(0, 0.95, 0.90);
        
        let envelopes = calibrator.finalize();
        
        assert!(envelopes.envelopes[0].sample_count == 5);
        assert!(envelopes.envelopes[0].mean_error > 0.0);
        assert!(envelopes.envelopes[0].max_error > envelopes.envelopes[0].mean_error);
    }
    
    #[test]
    fn test_envelope_threshold() {
        let mut quantiles = HashMap::new();
        quantiles.insert(9500, 0.05); // 95% quantile: error ≤ 0.05
        quantiles.insert(9900, 0.08); // 99% quantile: error ≤ 0.08
        
        let envelope = ErrorEnvelope {
            list_idx: 0,
            quantiles,
            mean_error: 0.03,
            std_error: 0.02,
            max_error: 0.10,
            min_error: 0.00,
            sample_count: 100,
        };
        
        // Proxy = 0.90, 95% confidence
        // Safe true threshold = 0.90 - 0.05 = 0.85
        let threshold = envelope.safe_true_threshold(0.90, 0.95);
        assert!((threshold - 0.85).abs() < 0.01);
        
        // 99% confidence needs larger margin
        let threshold99 = envelope.safe_true_threshold(0.90, 0.99);
        assert!((threshold99 - 0.82).abs() < 0.01);
    }
    
    #[test]
    fn test_can_terminate() {
        let envelopes = CalibrationRunner::calibrate_synthetic(5, 0.03, 0.01, 100);
        
        // If kth proxy is high enough, should be able to terminate
        let kth_proxy = 0.95;
        let remaining = vec![(1, 0.70), (2, 0.65)]; // Low bounds
        
        let can_term = envelopes.can_terminate(kth_proxy, &remaining, 0.99);
        assert!(can_term, "Should be able to terminate with high kth and low bounds");
        
        // If bounds are high, should not terminate
        let remaining_high = vec![(1, 0.94), (2, 0.93)];
        let cannot_term = envelopes.can_terminate(kth_proxy, &remaining_high, 0.99);
        assert!(!cannot_term, "Should not terminate with close bounds");
    }
    
    #[test]
    fn test_synthetic_calibration() {
        let envelopes = CalibrationRunner::calibrate_synthetic(10, 0.02, 0.01, 500);
        
        assert_eq!(envelopes.envelopes.len(), 10);
        assert!(envelopes.global.sample_count > 0);
        
        // Mean should be close to synthetic mean
        assert!((envelopes.global.mean_error - 0.02).abs() < 0.01);
    }
}
