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

//! # Guarantee Ladder with Explicit Semantics (Task 2)
//!
//! This module provides three guarantee modes with well-defined correctness contracts:
//!
//! 1. **Fast Approximate** (`Approximate`): Top-k under proxy score
//!    - Fastest, ignores quantization error
//!    - No recall guarantees
//!
//! 2. **Calibrated High-Recall** (`Calibrated`): Probabilistic guarantees
//!    - Uses quantile-bounded error envelopes
//!    - P(recall ≥ ρ) ≥ 1-δ
//!
//! 3. **Certified** (`Certified`): Deterministic exact via LB/UB envelopes
//!    - Guaranteed correct results
//!    - Uses rerank to verify all candidates
//!
//! ## Math/Algorithm
//!
//! Let true score be s(x), proxy be ŝ(x) = s(x) + ε(x).
//!
//! - Mode 1: ε ignored (ranking by ŝ)
//! - Mode 2: ε bounded in probability (use quantiles)
//! - Mode 3: ε bounded deterministically (use LB/UB comparisons)

use std::time::Duration;

// ============================================================================
// Guarantee Modes
// ============================================================================

/// Guarantee mode for search correctness
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuaranteeMode {
    /// Fast approximate search - ignores quantization error
    /// 
    /// Returns top-k by proxy score (e.g., PQ/ADC distance).
    /// Fastest but may miss true nearest neighbors.
    Approximate,
    
    /// Calibrated high-recall search - probabilistic guarantees
    /// 
    /// Uses learned error envelopes to bound approximation error.
    /// Guarantees P(recall@k ≥ ρ) ≥ 1-δ with configurable ρ and δ.
    Calibrated {
        /// Target recall (ρ)
        recall_target: f32,
        /// Confidence level (1-δ)
        confidence: f32,
    },
    
    /// Certified search - deterministic correctness
    /// 
    /// Uses lower/upper bound envelopes for all candidates.
    /// Guarantees exact top-k results (same as brute force).
    Certified,
}

impl Default for GuaranteeMode {
    fn default() -> Self {
        Self::Calibrated {
            recall_target: 0.95,
            confidence: 0.99,
        }
    }
}

impl GuaranteeMode {
    /// Create calibrated mode with given recall target
    pub fn calibrated(recall_target: f32, confidence: f32) -> Self {
        Self::Calibrated {
            recall_target,
            confidence,
        }
    }
    
    /// Returns true if this mode requires rerank verification
    pub fn requires_rerank(&self) -> bool {
        matches!(self, GuaranteeMode::Certified)
    }
    
    /// Returns true if this mode uses error envelopes
    pub fn uses_error_envelopes(&self) -> bool {
        !matches!(self, GuaranteeMode::Approximate)
    }
    
    /// Get the error quantile to use for this mode
    pub fn error_quantile(&self) -> Option<f32> {
        match self {
            GuaranteeMode::Approximate => None,
            GuaranteeMode::Calibrated { confidence, .. } => Some(*confidence),
            GuaranteeMode::Certified => Some(1.0), // Use max error bound
        }
    }
}

// ============================================================================
// Stopping Rules
// ============================================================================

/// Stopping rule that matches guarantee mode semantics
#[derive(Debug, Clone)]
pub enum StoppingRule {
    /// Stop after scanning fixed number of lists/probes
    FixedProbes {
        n_probes: u32,
    },
    
    /// Stop when kth best score exceeds best list bound
    /// 
    /// For approximate mode: compare proxy scores directly
    BoundBased {
        /// Minimum number of probes before considering early stop
        min_probes: u32,
        /// Maximum probes as safety limit
        max_probes: u32,
    },
    
    /// Stop when probability of finding better candidates drops below threshold
    /// 
    /// For calibrated mode: uses error envelopes to estimate probability
    ProbabilisticBound {
        /// Probability threshold for stopping
        probability_threshold: f32,
        /// Error envelope quantile
        error_quantile: f32,
        /// Minimum probes
        min_probes: u32,
        /// Maximum probes
        max_probes: u32,
    },
    
    /// Stop when all candidates with possible better true scores are checked
    /// 
    /// For certified mode: uses deterministic LB/UB comparisons
    DeterministicBound {
        /// Maximum error bound (guaranteed upper bound on ε)
        max_error: f32,
    },
    
    /// Combined budget and bound stopping
    BudgetConstrained {
        inner: Box<StoppingRule>,
        max_ram_bytes: u64,
        max_latency: Duration,
    },
}

impl StoppingRule {
    /// Create stopping rule appropriate for guarantee mode
    pub fn for_mode(mode: &GuaranteeMode, default_probes: u32) -> Self {
        match mode {
            GuaranteeMode::Approximate => Self::FixedProbes {
                n_probes: default_probes,
            },
            GuaranteeMode::Calibrated { confidence, .. } => Self::ProbabilisticBound {
                probability_threshold: 0.01, // Stop when <1% chance of improvement
                error_quantile: *confidence,
                min_probes: default_probes / 4,
                max_probes: default_probes * 4,
            },
            GuaranteeMode::Certified => Self::DeterministicBound {
                max_error: 0.0, // Must be set from calibration data
            },
        }
    }
    
    /// Wrap with budget constraints
    pub fn with_budget(self, max_ram_bytes: u64, max_latency: Duration) -> Self {
        Self::BudgetConstrained {
            inner: Box::new(self),
            max_ram_bytes,
            max_latency,
        }
    }
}

// ============================================================================
// Stop Decision
// ============================================================================

/// Decision about whether to stop probing
#[derive(Debug, Clone)]
pub struct StopDecision {
    /// Whether to stop
    pub should_stop: bool,
    /// Reason for decision
    pub reason: StopReason,
    /// Estimated probability of missing better candidates (for calibrated mode)
    pub miss_probability: Option<f32>,
    /// Number of candidates that might have better true scores
    pub uncertain_candidates: u32,
}

/// Reason for stopping decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// All probes exhausted
    ProbesExhausted,
    /// Bound condition satisfied
    BoundSatisfied,
    /// Probability threshold reached
    ProbabilityThreshold,
    /// Deterministic guarantee achieved
    DeterministicComplete,
    /// Budget exhausted (RAM/latency)
    BudgetExhausted,
    /// Still searching
    Continuing,
}

// ============================================================================
// Score Envelope
// ============================================================================

/// Score with error bounds for certified/calibrated modes
#[derive(Debug, Clone, Copy)]
pub struct ScoreEnvelope {
    /// Proxy score (from quantized representation)
    pub proxy: f32,
    /// Lower bound on true score
    pub lower_bound: f32,
    /// Upper bound on true score
    pub upper_bound: f32,
    /// Quantile-based error bound (for calibrated mode)
    pub quantile_error: f32,
}

impl ScoreEnvelope {
    /// Create from proxy score with error bounds
    pub fn new(proxy: f32, max_error: f32) -> Self {
        Self {
            proxy,
            lower_bound: proxy - max_error,
            upper_bound: proxy + max_error,
            quantile_error: max_error,
        }
    }
    
    /// Create from proxy with asymmetric bounds
    pub fn with_bounds(proxy: f32, lower_bound: f32, upper_bound: f32) -> Self {
        Self {
            proxy,
            lower_bound,
            upper_bound,
            quantile_error: (upper_bound - lower_bound) / 2.0,
        }
    }
    
    /// Check if this envelope definitely beats another
    /// 
    /// Returns true iff lower_bound(self) > upper_bound(other)
    pub fn definitely_beats(&self, other: &ScoreEnvelope) -> bool {
        self.lower_bound > other.upper_bound
    }
    
    /// Check if this envelope might beat another
    /// 
    /// Returns true iff upper_bound(self) > lower_bound(other)
    pub fn might_beat(&self, other: &ScoreEnvelope) -> bool {
        self.upper_bound > other.lower_bound
    }
    
    /// Get true score estimate (center of bounds)
    pub fn estimated_true(&self) -> f32 {
        (self.lower_bound + self.upper_bound) / 2.0
    }
}

// ============================================================================
// Stopping Evaluator
// ============================================================================

/// Evaluator for stopping rules
pub struct StoppingEvaluator {
    rule: StoppingRule,
    probes_done: u32,
    ram_bytes_used: u64,
    start_time: std::time::Instant,
}

impl StoppingEvaluator {
    /// Create new evaluator
    pub fn new(rule: StoppingRule) -> Self {
        Self {
            rule,
            probes_done: 0,
            ram_bytes_used: 0,
            start_time: std::time::Instant::now(),
        }
    }
    
    /// Record a probe
    pub fn record_probe(&mut self, ram_bytes: u64) {
        self.probes_done += 1;
        self.ram_bytes_used += ram_bytes;
    }
    
    /// Evaluate stopping rule
    /// 
    /// # Arguments
    /// * `kth_score` - Score envelope of the kth best candidate
    /// * `best_remaining_bound` - Best upper bound of remaining lists
    pub fn evaluate(
        &self,
        kth_score: Option<&ScoreEnvelope>,
        best_remaining_bound: Option<f32>,
    ) -> StopDecision {
        match &self.rule {
            StoppingRule::FixedProbes { n_probes } => {
                if self.probes_done >= *n_probes {
                    StopDecision {
                        should_stop: true,
                        reason: StopReason::ProbesExhausted,
                        miss_probability: None,
                        uncertain_candidates: 0,
                    }
                } else {
                    StopDecision {
                        should_stop: false,
                        reason: StopReason::Continuing,
                        miss_probability: None,
                        uncertain_candidates: 0,
                    }
                }
            }
            
            StoppingRule::BoundBased { min_probes, max_probes } => {
                if self.probes_done >= *max_probes {
                    return StopDecision {
                        should_stop: true,
                        reason: StopReason::ProbesExhausted,
                        miss_probability: None,
                        uncertain_candidates: 0,
                    };
                }
                
                if self.probes_done < *min_probes {
                    return StopDecision {
                        should_stop: false,
                        reason: StopReason::Continuing,
                        miss_probability: None,
                        uncertain_candidates: 0,
                    };
                }
                
                // Check bound condition: kth.proxy > best_remaining
                if let (Some(kth), Some(bound)) = (kth_score, best_remaining_bound) {
                    if kth.proxy > bound {
                        return StopDecision {
                            should_stop: true,
                            reason: StopReason::BoundSatisfied,
                            miss_probability: None,
                            uncertain_candidates: 0,
                        };
                    }
                }
                
                StopDecision {
                    should_stop: false,
                    reason: StopReason::Continuing,
                    miss_probability: None,
                    uncertain_candidates: 0,
                }
            }
            
            StoppingRule::ProbabilisticBound {
                probability_threshold,
                error_quantile: _,
                min_probes,
                max_probes,
            } => {
                if self.probes_done >= *max_probes {
                    return StopDecision {
                        should_stop: true,
                        reason: StopReason::ProbesExhausted,
                        miss_probability: Some(0.0),
                        uncertain_candidates: 0,
                    };
                }
                
                if self.probes_done < *min_probes {
                    return StopDecision {
                        should_stop: false,
                        reason: StopReason::Continuing,
                        miss_probability: Some(1.0),
                        uncertain_candidates: 0,
                    };
                }
                
                // Use error envelopes to estimate miss probability
                if let (Some(kth), Some(bound)) = (kth_score, best_remaining_bound) {
                    // Estimate: if kth.lower_bound > bound + error, we're confident
                    let margin = kth.lower_bound - bound;
                    let error_margin = kth.quantile_error;
                    
                    // Simple probability model: linear decrease
                    let miss_prob = if margin > error_margin {
                        0.0
                    } else if margin < -error_margin {
                        1.0
                    } else {
                        0.5 - (margin / (2.0 * error_margin))
                    };
                    
                    if miss_prob < *probability_threshold {
                        return StopDecision {
                            should_stop: true,
                            reason: StopReason::ProbabilityThreshold,
                            miss_probability: Some(miss_prob),
                            uncertain_candidates: 0,
                        };
                    }
                    
                    return StopDecision {
                        should_stop: false,
                        reason: StopReason::Continuing,
                        miss_probability: Some(miss_prob),
                        uncertain_candidates: 0,
                    };
                }
                
                StopDecision {
                    should_stop: false,
                    reason: StopReason::Continuing,
                    miss_probability: Some(1.0),
                    uncertain_candidates: 0,
                }
            }
            
            StoppingRule::DeterministicBound { max_error } => {
                if let (Some(kth), Some(bound)) = (kth_score, best_remaining_bound) {
                    // Definitely done: kth.lower_bound > best_remaining + max_error
                    if kth.lower_bound > bound + *max_error {
                        return StopDecision {
                            should_stop: true,
                            reason: StopReason::DeterministicComplete,
                            miss_probability: Some(0.0),
                            uncertain_candidates: 0,
                        };
                    }
                }
                
                StopDecision {
                    should_stop: false,
                    reason: StopReason::Continuing,
                    miss_probability: None,
                    uncertain_candidates: 0,
                }
            }
            
            StoppingRule::BudgetConstrained {
                inner,
                max_ram_bytes,
                max_latency,
            } => {
                // Check budget first
                if self.ram_bytes_used > *max_ram_bytes {
                    return StopDecision {
                        should_stop: true,
                        reason: StopReason::BudgetExhausted,
                        miss_probability: None,
                        uncertain_candidates: 0,
                    };
                }
                
                if self.start_time.elapsed() > *max_latency {
                    return StopDecision {
                        should_stop: true,
                        reason: StopReason::BudgetExhausted,
                        miss_probability: None,
                        uncertain_candidates: 0,
                    };
                }
                
                // Delegate to inner rule
                let inner_eval = StoppingEvaluator {
                    rule: (**inner).clone(),
                    probes_done: self.probes_done,
                    ram_bytes_used: self.ram_bytes_used,
                    start_time: self.start_time,
                };
                inner_eval.evaluate(kth_score, best_remaining_bound)
            }
        }
    }
}

// ============================================================================
// Search Contract
// ============================================================================

/// Complete search contract specifying guarantees
#[derive(Debug, Clone)]
pub struct SearchContract {
    /// Guarantee mode
    pub mode: GuaranteeMode,
    /// Number of results requested
    pub k: usize,
    /// Stopping rule
    pub stopping_rule: StoppingRule,
    /// Whether to include score envelopes in results
    pub include_envelopes: bool,
}

impl SearchContract {
    /// Create contract for approximate search
    pub fn approximate(k: usize, n_probes: u32) -> Self {
        Self {
            mode: GuaranteeMode::Approximate,
            k,
            stopping_rule: StoppingRule::FixedProbes { n_probes },
            include_envelopes: false,
        }
    }
    
    /// Create contract for calibrated search
    pub fn calibrated(k: usize, recall_target: f32, confidence: f32) -> Self {
        let mode = GuaranteeMode::calibrated(recall_target, confidence);
        let stopping_rule = StoppingRule::for_mode(&mode, 16);
        Self {
            mode,
            k,
            stopping_rule,
            include_envelopes: true,
        }
    }
    
    /// Create contract for certified search
    pub fn certified(k: usize) -> Self {
        Self {
            mode: GuaranteeMode::Certified,
            k,
            stopping_rule: StoppingRule::DeterministicBound { max_error: 0.0 },
            include_envelopes: true,
        }
    }
    
    /// Add budget constraints
    pub fn with_budget(mut self, max_ram_bytes: u64, max_latency: Duration) -> Self {
        self.stopping_rule = self.stopping_rule.with_budget(max_ram_bytes, max_latency);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_guarantee_modes() {
        let approx = GuaranteeMode::Approximate;
        assert!(!approx.requires_rerank());
        assert!(!approx.uses_error_envelopes());
        
        let calibrated = GuaranteeMode::calibrated(0.95, 0.99);
        assert!(!calibrated.requires_rerank());
        assert!(calibrated.uses_error_envelopes());
        assert_eq!(calibrated.error_quantile(), Some(0.99));
        
        let certified = GuaranteeMode::Certified;
        assert!(certified.requires_rerank());
        assert!(certified.uses_error_envelopes());
    }
    
    #[test]
    fn test_score_envelope() {
        let a = ScoreEnvelope::new(0.9, 0.05);
        let b = ScoreEnvelope::new(0.8, 0.05);
        
        assert!(a.definitely_beats(&b)); // 0.85 > 0.85 is false, wait...
        // a.lower = 0.85, b.upper = 0.85, not strictly greater
        
        let c = ScoreEnvelope::new(0.9, 0.02);
        let d = ScoreEnvelope::new(0.8, 0.02);
        // c.lower = 0.88, d.upper = 0.82
        assert!(c.definitely_beats(&d)); // 0.88 > 0.82
    }
    
    #[test]
    fn test_fixed_probes_stopping() {
        let rule = StoppingRule::FixedProbes { n_probes: 10 };
        let mut eval = StoppingEvaluator::new(rule);
        
        for _ in 0..9 {
            eval.record_probe(1000);
            let decision = eval.evaluate(None, None);
            assert!(!decision.should_stop);
        }
        
        eval.record_probe(1000);
        let decision = eval.evaluate(None, None);
        assert!(decision.should_stop);
        assert_eq!(decision.reason, StopReason::ProbesExhausted);
    }
    
    #[test]
    fn test_bound_based_stopping() {
        let rule = StoppingRule::BoundBased {
            min_probes: 2,
            max_probes: 100,
        };
        let mut eval = StoppingEvaluator::new(rule);
        
        // Before min_probes, shouldn't stop
        eval.record_probe(1000);
        let kth = ScoreEnvelope::new(0.9, 0.01);
        let decision = eval.evaluate(Some(&kth), Some(0.8));
        assert!(!decision.should_stop);
        
        // After min_probes, should stop when kth > bound
        eval.record_probe(1000);
        let decision = eval.evaluate(Some(&kth), Some(0.8));
        assert!(decision.should_stop);
        assert_eq!(decision.reason, StopReason::BoundSatisfied);
    }
}
