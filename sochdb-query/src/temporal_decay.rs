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

//! Temporal Decay Scoring (Task 4)
//!
//! This module implements recency-biased relevance scoring for memory retrieval.
//! It applies exponential decay to blend temporal and semantic signals.
//!
//! ## Formula
//!
//! ```text
//! decay(Δt) = λ^(Δt/τ)
//! final_score = α × semantic_score + (1-α) × decay_score
//! ```
//!
//! Where:
//! - Δt = time since document creation/update
//! - τ = decay half-life (time for score to halve)
//! - λ = decay rate (typically 0.5 for half-life)
//! - α = semantic weight (0.0 to 1.0)
//!
//! ## Complexity
//!
//! - Decay computation: O(1) per document
//! - Resorting: O(K log K) for top-K candidates
//! - Selection heap: O(K) if using heap-based selection

use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for temporal decay scoring
#[derive(Debug, Clone)]
pub struct TemporalDecayConfig {
    /// Decay rate (λ): 0.5 = half-life decay
    pub decay_rate: f32,
    
    /// Half-life in seconds (τ): time for score to halve
    pub half_life_secs: f64,
    
    /// Semantic weight (α): 0.0 = pure recency, 1.0 = pure semantic
    pub semantic_weight: f32,
    
    /// Minimum decay score (floor)
    pub min_decay: f32,
    
    /// Whether to apply decay before or after other scoring
    pub apply_stage: DecayStage,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.5,
            half_life_secs: 3600.0 * 24.0, // 24 hours
            semantic_weight: 0.7,
            min_decay: 0.01,
            apply_stage: DecayStage::PostRetrieval,
        }
    }
}

impl TemporalDecayConfig {
    /// Create config for short-term memory (fast decay)
    pub fn short_term() -> Self {
        Self {
            decay_rate: 0.5,
            half_life_secs: 3600.0, // 1 hour
            semantic_weight: 0.5,
            min_decay: 0.01,
            apply_stage: DecayStage::PostRetrieval,
        }
    }
    
    /// Create config for long-term memory (slow decay)
    pub fn long_term() -> Self {
        Self {
            decay_rate: 0.5,
            half_life_secs: 3600.0 * 24.0 * 7.0, // 1 week
            semantic_weight: 0.85,
            min_decay: 0.05,
            apply_stage: DecayStage::PostRetrieval,
        }
    }
    
    /// Create config for working memory (very fast decay)
    pub fn working_memory() -> Self {
        Self {
            decay_rate: 0.5,
            half_life_secs: 300.0, // 5 minutes
            semantic_weight: 0.3,
            min_decay: 0.0,
            apply_stage: DecayStage::PostRetrieval,
        }
    }
    
    /// Create config with custom half-life
    pub fn with_half_life(half_life_secs: f64, semantic_weight: f32) -> Self {
        Self {
            half_life_secs,
            semantic_weight,
            ..Default::default()
        }
    }
}

/// When to apply decay scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayStage {
    /// Apply decay during index search (modifies distance)
    DuringSearch,
    /// Apply decay after retrieval (reranking)
    PostRetrieval,
    /// Apply decay as final step before returning
    Final,
}

// ============================================================================
// Temporal Scorer
// ============================================================================

/// Temporal decay scorer
#[derive(Debug, Clone)]
pub struct TemporalScorer {
    config: TemporalDecayConfig,
    /// Reference time (usually current time)
    reference_time: f64,
}

impl TemporalScorer {
    /// Create a new temporal scorer with current time as reference
    pub fn new(config: TemporalDecayConfig) -> Self {
        let reference_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        Self {
            config,
            reference_time,
        }
    }
    
    /// Create with specific reference time
    pub fn with_reference_time(config: TemporalDecayConfig, reference_time: f64) -> Self {
        Self {
            config,
            reference_time,
        }
    }
    
    /// Create with default config
    pub fn default_scorer() -> Self {
        Self::new(TemporalDecayConfig::default())
    }
    
    /// Calculate decay score for a given timestamp
    ///
    /// Returns a value between min_decay and 1.0
    pub fn decay_score(&self, timestamp_secs: f64) -> f32 {
        let delta_t = (self.reference_time - timestamp_secs).max(0.0);
        
        // decay = λ^(Δt/τ)
        let exponent = delta_t / self.config.half_life_secs;
        let decay = self.config.decay_rate.powf(exponent as f32);
        
        decay.max(self.config.min_decay)
    }
    
    /// Calculate decay score from Duration
    pub fn decay_score_duration(&self, age: Duration) -> f32 {
        let delta_t = age.as_secs_f64();
        let exponent = delta_t / self.config.half_life_secs;
        let decay = self.config.decay_rate.powf(exponent as f32);
        
        decay.max(self.config.min_decay)
    }
    
    /// Blend semantic and decay scores
    ///
    /// final = α × semantic + (1-α) × decay
    pub fn blend_scores(&self, semantic_score: f32, decay_score: f32) -> f32 {
        let alpha = self.config.semantic_weight;
        alpha * semantic_score + (1.0 - alpha) * decay_score
    }
    
    /// Calculate final score from semantic score and timestamp
    pub fn final_score(&self, semantic_score: f32, timestamp_secs: f64) -> f32 {
        let decay = self.decay_score(timestamp_secs);
        self.blend_scores(semantic_score, decay)
    }
    
    /// Apply temporal decay to a list of scored results
    ///
    /// Each result is (id, semantic_score, timestamp)
    /// Returns (id, final_score) sorted by final_score descending
    pub fn apply_decay<I>(
        &self,
        results: I,
    ) -> Vec<(String, f32)>
    where
        I: IntoIterator<Item = (String, f32, f64)>,
    {
        let mut scored: Vec<_> = results
            .into_iter()
            .map(|(id, semantic, timestamp)| {
                let final_score = self.final_score(semantic, timestamp);
                (id, final_score)
            })
            .collect();
        
        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        scored
    }
    
    /// Apply decay to typed results
    pub fn apply_decay_typed<T, F>(
        &self,
        results: &mut [T],
        get_score: impl Fn(&T) -> f32,
        get_timestamp: impl Fn(&T) -> f64,
        set_score: F,
    ) where
        F: Fn(&mut T, f32),
    {
        for result in results.iter_mut() {
            let semantic = get_score(result);
            let timestamp = get_timestamp(result);
            let final_score = self.final_score(semantic, timestamp);
            set_score(result, final_score);
        }
    }
    
    /// Get the half-life in human-readable format
    pub fn half_life_display(&self) -> String {
        let secs = self.config.half_life_secs;
        
        if secs < 60.0 {
            format!("{:.0} seconds", secs)
        } else if secs < 3600.0 {
            format!("{:.1} minutes", secs / 60.0)
        } else if secs < 86400.0 {
            format!("{:.1} hours", secs / 3600.0)
        } else {
            format!("{:.1} days", secs / 86400.0)
        }
    }
}

// ============================================================================
// Scored Result Types
// ============================================================================

/// A result with temporal decay applied
#[derive(Debug, Clone)]
pub struct TemporallyDecayedResult {
    /// Result identifier
    pub id: String,
    
    /// Original semantic/similarity score
    pub semantic_score: f32,
    
    /// Decay factor based on age
    pub decay_factor: f32,
    
    /// Final blended score
    pub final_score: f32,
    
    /// Document timestamp (seconds since epoch)
    pub timestamp: f64,
    
    /// Age of the document
    pub age_secs: f64,
}

impl TemporallyDecayedResult {
    /// Create from components
    pub fn new(
        id: String,
        semantic_score: f32,
        timestamp: f64,
        scorer: &TemporalScorer,
    ) -> Self {
        let decay_factor = scorer.decay_score(timestamp);
        let final_score = scorer.blend_scores(semantic_score, decay_factor);
        let age_secs = scorer.reference_time - timestamp;
        
        Self {
            id,
            semantic_score,
            decay_factor,
            final_score,
            timestamp,
            age_secs,
        }
    }
    
    /// Format age as human-readable string
    pub fn age_display(&self) -> String {
        let age = self.age_secs;
        
        if age < 60.0 {
            format!("{:.0}s ago", age)
        } else if age < 3600.0 {
            format!("{:.0}m ago", age / 60.0)
        } else if age < 86400.0 {
            format!("{:.1}h ago", age / 3600.0)
        } else {
            format!("{:.1}d ago", age / 86400.0)
        }
    }
}

// ============================================================================
// Decay Curve Analysis
// ============================================================================

/// Analyze decay curve for debugging/visualization
#[derive(Debug, Clone)]
pub struct DecayCurve {
    /// Points on the curve: (age_secs, decay_score)
    pub points: Vec<(f64, f32)>,
    
    /// Half-life in seconds
    pub half_life: f64,
    
    /// Configuration used
    pub config: TemporalDecayConfig,
}

impl DecayCurve {
    /// Generate decay curve points
    pub fn generate(config: &TemporalDecayConfig, max_age_secs: f64, num_points: usize) -> Self {
        let scorer = TemporalScorer::with_reference_time(config.clone(), max_age_secs);
        
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let age = (i as f64) * max_age_secs / (num_points as f64);
            let timestamp = max_age_secs - age;
            let score = scorer.decay_score(timestamp);
            points.push((age, score));
        }
        
        Self {
            points,
            half_life: config.half_life_secs,
            config: config.clone(),
        }
    }
    
    /// Find age where score drops to threshold
    pub fn age_at_threshold(&self, threshold: f32) -> Option<f64> {
        for (age, score) in &self.points {
            if *score <= threshold {
                return Some(*age);
            }
        }
        None
    }
    
    /// Format as ASCII chart
    pub fn ascii_chart(&self, width: usize, height: usize) -> String {
        let mut chart = vec![vec![' '; width]; height];
        
        for (age, score) in &self.points {
            let x = ((age / self.points.last().unwrap().0) * (width - 1) as f64) as usize;
            let y = ((1.0 - *score) * (height - 1) as f32) as usize;
            
            if x < width && y < height {
                chart[y][x] = '█';
            }
        }
        
        // Add axes
        for row in &mut chart {
            row[0] = '│';
        }
        chart[height - 1] = vec!['─'; width];
        chart[height - 1][0] = '└';
        
        chart.iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ============================================================================
// Integration with Search Results
// ============================================================================

/// Extension trait for applying temporal decay to search results
pub trait TemporalDecayExt {
    /// Apply temporal decay and return sorted results
    fn with_temporal_decay(self, scorer: &TemporalScorer) -> Vec<TemporallyDecayedResult>;
}

impl<I> TemporalDecayExt for I
where
    I: IntoIterator<Item = (String, f32, f64)>,
{
    fn with_temporal_decay(self, scorer: &TemporalScorer) -> Vec<TemporallyDecayedResult> {
        let mut results: Vec<_> = self
            .into_iter()
            .map(|(id, semantic_score, timestamp)| {
                TemporallyDecayedResult::new(id, semantic_score, timestamp, scorer)
            })
            .collect();
        
        // Sort by final score descending
        results.sort_by(|a, b| {
            b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        results
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Calculate decay score with default configuration
pub fn quick_decay(age_secs: f64) -> f32 {
    let scorer = TemporalScorer::new(TemporalDecayConfig::default());
    scorer.decay_score_duration(Duration::from_secs_f64(age_secs))
}

/// Calculate final score with default configuration
pub fn quick_temporal_score(semantic_score: f32, age_secs: f64) -> f32 {
    let scorer = TemporalScorer::new(TemporalDecayConfig::default());
    let decay = scorer.decay_score_duration(Duration::from_secs_f64(age_secs));
    scorer.blend_scores(semantic_score, decay)
}

/// Apply temporal decay to search results with default configuration
pub fn apply_default_decay<I>(results: I) -> Vec<(String, f32)>
where
    I: IntoIterator<Item = (String, f32, f64)>,
{
    let scorer = TemporalScorer::new(TemporalDecayConfig::default());
    scorer.apply_decay(results)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decay_at_half_life() {
        let config = TemporalDecayConfig {
            decay_rate: 0.5,
            half_life_secs: 3600.0, // 1 hour
            semantic_weight: 0.5,
            min_decay: 0.0,
            apply_stage: DecayStage::PostRetrieval,
        };
        
        let scorer = TemporalScorer::with_reference_time(config, 3600.0);
        
        // At reference time (age = 0), decay should be 1.0
        let decay_now = scorer.decay_score(3600.0);
        assert!((decay_now - 1.0).abs() < 0.01);
        
        // At half-life (age = 1 hour), decay should be 0.5
        let decay_half = scorer.decay_score(0.0);
        assert!((decay_half - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_decay_double_half_life() {
        let config = TemporalDecayConfig {
            decay_rate: 0.5,
            half_life_secs: 3600.0,
            semantic_weight: 0.5,
            min_decay: 0.0,
            apply_stage: DecayStage::PostRetrieval,
        };
        
        let scorer = TemporalScorer::with_reference_time(config, 7200.0);
        
        // At 2x half-life (age = 2 hours), decay should be 0.25
        let decay = scorer.decay_score(0.0);
        assert!((decay - 0.25).abs() < 0.01);
    }
    
    #[test]
    fn test_blend_scores() {
        let config = TemporalDecayConfig {
            semantic_weight: 0.7,
            ..Default::default()
        };
        
        let scorer = TemporalScorer::new(config);
        
        // semantic = 0.8, decay = 0.5
        // final = 0.7 * 0.8 + 0.3 * 0.5 = 0.56 + 0.15 = 0.71
        let final_score = scorer.blend_scores(0.8, 0.5);
        assert!((final_score - 0.71).abs() < 0.01);
    }
    
    #[test]
    fn test_min_decay_floor() {
        let config = TemporalDecayConfig {
            decay_rate: 0.5,
            half_life_secs: 1.0, // Very fast decay
            min_decay: 0.1,
            semantic_weight: 0.5,
            apply_stage: DecayStage::PostRetrieval,
        };
        
        let scorer = TemporalScorer::with_reference_time(config, 1000.0);
        
        // Very old document should hit min_decay floor
        let decay = scorer.decay_score(0.0);
        assert!((decay - 0.1).abs() < 0.01);
    }
    
    #[test]
    fn test_apply_decay_reorders() {
        let config = TemporalDecayConfig {
            decay_rate: 0.5,
            half_life_secs: 100.0,
            semantic_weight: 0.5,
            min_decay: 0.0,
            apply_stage: DecayStage::PostRetrieval,
        };
        
        let scorer = TemporalScorer::with_reference_time(config, 200.0);
        
        // Old document with high semantic score vs new document with lower semantic score
        let results = vec![
            ("old_high".to_string(), 0.9, 0.0),    // Age = 200s, decay ≈ 0.25
            ("new_low".to_string(), 0.6, 190.0),   // Age = 10s, decay ≈ 0.93
        ];
        
        let decayed = scorer.apply_decay(results);
        
        // New document should rank higher despite lower semantic score
        assert_eq!(decayed[0].0, "new_low");
    }
    
    #[test]
    fn test_decay_curve_generation() {
        let config = TemporalDecayConfig::default();
        let curve = DecayCurve::generate(&config, 86400.0 * 7.0, 100);
        
        assert_eq!(curve.points.len(), 100);
        
        // First point should have score near 1.0
        assert!(curve.points[0].1 > 0.9);
        
        // Last point should have lower score
        assert!(curve.points.last().unwrap().1 < curve.points[0].1);
    }
    
    #[test]
    fn test_temporally_decayed_result() {
        let config = TemporalDecayConfig::short_term();
        let scorer = TemporalScorer::with_reference_time(config, 7200.0);
        
        let result = TemporallyDecayedResult::new(
            "doc1".to_string(),
            0.85,
            3600.0, // 1 hour old
            &scorer,
        );
        
        assert_eq!(result.id, "doc1");
        assert!((result.semantic_score - 0.85).abs() < 0.01);
        assert!(result.decay_factor < 1.0);
        assert!(result.age_secs > 0.0);
    }
    
    #[test]
    fn test_half_life_display() {
        let config = TemporalDecayConfig {
            half_life_secs: 7200.0, // 2 hours
            ..Default::default()
        };
        
        let scorer = TemporalScorer::new(config);
        let display = scorer.half_life_display();
        
        assert!(display.contains("hours") || display.contains("2.0"));
    }
}
