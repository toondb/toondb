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

//! Token Budget Enforcement
//!
//! This module implements token estimation and budget tracking for
//! CONTEXT SELECT queries.
//!
//! ## Token Estimation Model
//!
//! The token count for a row is estimated as:
//!
//! $$T_{row} = \sum_{i=1}^{C} T(v_i) + (C - 1) \times T_{sep}$$
//!
//! Where:
//! - $T(v_i)$ = tokens for value $i$
//! - $C$ = number of columns
//! - $T_{sep}$ = separator token cost (~1 token per separator)
//!
//! ## Type-Specific Estimation
//!
//! Different data types have different token characteristics:
//!
//! | Type | Factor | Notes |
//! |------|--------|-------|
//! | Integer | 1.0 | ~1 token per 3-4 digits |
//! | Float | 1.2 | Decimal point adds overhead |
//! | String | 1.1 | Potential subword splits |
//! | Binary (hex) | 2.5 | 0x prefix + hex expansion |
//! | Boolean | 1.0 | "true"/"false" are single tokens |
//! | Null | 1.0 | "null" is a single token |

use crate::soch_ql::SochValue;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Token Estimator
// ============================================================================

/// Token estimation configuration
#[derive(Debug, Clone)]
pub struct TokenEstimatorConfig {
    /// Multiplier for integer values
    pub int_factor: f32,
    /// Multiplier for float values
    pub float_factor: f32,
    /// Multiplier for string values
    pub string_factor: f32,
    /// Multiplier for binary (hex) values
    pub hex_factor: f32,
    /// Bytes per token (approximate)
    pub bytes_per_token: f32,
    /// Separator cost in tokens
    pub separator_tokens: usize,
    /// Newline cost in tokens
    pub newline_tokens: usize,
    /// Header overhead tokens
    pub header_tokens: usize,
}

impl Default for TokenEstimatorConfig {
    fn default() -> Self {
        Self {
            int_factor: 1.0,
            float_factor: 1.2,
            string_factor: 1.1,
            hex_factor: 2.5,
            bytes_per_token: 4.0, // ~4 chars per token for English
            separator_tokens: 1,
            newline_tokens: 1,
            header_tokens: 10, // table[N]{cols}: header
        }
    }
}

impl TokenEstimatorConfig {
    /// Create config tuned for GPT-4 tokenizer
    pub fn gpt4() -> Self {
        Self {
            bytes_per_token: 3.8,
            ..Default::default()
        }
    }

    /// Create config tuned for Claude tokenizer
    pub fn claude() -> Self {
        Self {
            bytes_per_token: 4.2,
            ..Default::default()
        }
    }

    /// Create config with high precision (conservative)
    pub fn conservative() -> Self {
        Self {
            int_factor: 1.2,
            float_factor: 1.4,
            string_factor: 1.3,
            hex_factor: 3.0,
            bytes_per_token: 3.5,
            ..Default::default()
        }
    }
}

/// Token estimator
pub struct TokenEstimator {
    config: TokenEstimatorConfig,
}

impl TokenEstimator {
    /// Create a new estimator with default config
    pub fn new() -> Self {
        Self {
            config: TokenEstimatorConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: TokenEstimatorConfig) -> Self {
        Self { config }
    }

    /// Estimate tokens for a single value
    pub fn estimate_value(&self, value: &SochValue) -> usize {
        match value {
            SochValue::Null => 1,
            SochValue::Bool(_) => 1, // "true" or "false" is typically 1 token
            SochValue::Int(n) => {
                // Count digits + sign
                let digits = if *n == 0 {
                    1
                } else {
                    ((*n).abs() as f64).log10().ceil() as usize + if *n < 0 { 1 } else { 0 }
                };
                ((digits as f32 * self.config.int_factor) / self.config.bytes_per_token).ceil()
                    as usize
            }
            SochValue::UInt(n) => {
                let digits = if *n == 0 {
                    1
                } else {
                    ((*n as f64).log10().ceil() as usize).max(1)
                };
                ((digits as f32 * self.config.int_factor) / self.config.bytes_per_token).ceil()
                    as usize
            }
            SochValue::Float(f) => {
                // Format to 2 decimal places
                let s = format!("{:.2}", f);
                ((s.len() as f32 * self.config.float_factor) / self.config.bytes_per_token).ceil()
                    as usize
            }
            SochValue::Text(s) => {
                // Account for potential subword splitting
                ((s.len() as f32 * self.config.string_factor) / self.config.bytes_per_token).ceil()
                    as usize
            }
            SochValue::Binary(b) => {
                // Hex encoding: 0x + 2 chars per byte
                let hex_len = 2 + b.len() * 2;
                ((hex_len as f32 * self.config.hex_factor) / self.config.bytes_per_token).ceil()
                    as usize
            }
            SochValue::Array(arr) => {
                // Sum tokens for array elements plus brackets and separators
                let elem_tokens: usize = arr.iter().map(|v| self.estimate_value(v)).sum();
                let separator_tokens = if arr.is_empty() { 0 } else { arr.len() - 1 };
                2 + elem_tokens + separator_tokens // 2 for [ and ]
            }
        }
    }

    /// Estimate tokens for a row (multiple values)
    pub fn estimate_row(&self, values: &[SochValue]) -> usize {
        if values.is_empty() {
            return 0;
        }

        let value_tokens: usize = values.iter().map(|v| self.estimate_value(v)).sum();
        let separator_tokens = (values.len() - 1) * self.config.separator_tokens;
        let newline = self.config.newline_tokens;

        value_tokens + separator_tokens + newline
    }

    /// Estimate tokens for a table header
    pub fn estimate_header(&self, table: &str, columns: &[String], row_count: usize) -> usize {
        // Format: table[N]{col1,col2,...}:
        let base = self.config.header_tokens;
        let table_tokens = ((table.len() as f32) / self.config.bytes_per_token).ceil() as usize;
        let count_tokens = ((row_count as f64).log10().ceil() as usize).max(1);
        let col_tokens: usize = columns
            .iter()
            .map(|c| ((c.len() as f32) / self.config.bytes_per_token).ceil() as usize)
            .sum();

        base + table_tokens + count_tokens + col_tokens
    }

    /// Estimate tokens for a complete TOON table
    pub fn estimate_table(
        &self,
        table: &str,
        columns: &[String],
        rows: &[Vec<SochValue>],
    ) -> usize {
        let header = self.estimate_header(table, columns, rows.len());
        let row_tokens: usize = rows.iter().map(|r| self.estimate_row(r)).sum();
        header + row_tokens
    }

    /// Estimate tokens for plain text
    pub fn estimate_text(&self, text: &str) -> usize {
        ((text.len() as f32) / self.config.bytes_per_token).ceil() as usize
    }

    /// Truncate text to fit within token budget
    ///
    /// Uses binary search to find the optimal truncation point.
    pub fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> String {
        truncate_to_tokens(text, max_tokens, self, "...")
    }
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Token Budget Enforcer
// ============================================================================

/// Token budget enforcement result
#[derive(Debug, Clone)]
pub struct BudgetAllocation {
    /// Sections that fit fully
    pub full_sections: Vec<String>,
    /// Sections that were truncated (name, original_tokens, allocated_tokens)
    pub truncated_sections: Vec<(String, usize, usize)>,
    /// Sections that were dropped
    pub dropped_sections: Vec<String>,
    /// Total tokens allocated
    pub tokens_allocated: usize,
    /// Remaining budget
    pub tokens_remaining: usize,
    /// Detailed allocation decisions for EXPLAIN CONTEXT
    pub explain: Vec<AllocationDecision>,
}

/// Detailed explanation of a single allocation decision
#[derive(Debug, Clone)]
pub struct AllocationDecision {
    /// Section name
    pub section: String,
    /// Priority value
    pub priority: i32,
    /// Requested tokens
    pub requested: usize,
    /// Allocated tokens
    pub allocated: usize,
    /// Decision outcome
    pub outcome: AllocationOutcome,
    /// Human-readable reason
    pub reason: String,
}

/// Outcome of an allocation decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationOutcome {
    /// Section included in full
    Full,
    /// Section truncated to fit
    Truncated,
    /// Section dropped entirely
    Dropped,
}

/// Section for budget allocation
#[derive(Debug, Clone)]
pub struct BudgetSection {
    /// Section name
    pub name: String,
    /// Priority (lower = higher priority)
    pub priority: i32,
    /// Estimated token count
    pub estimated_tokens: usize,
    /// Minimum tokens needed (for truncation)
    pub minimum_tokens: Option<usize>,
    /// Is this section required?
    pub required: bool,
    /// Weight for proportional allocation (default: 1.0)
    pub weight: f32,
}

impl Default for BudgetSection {
    fn default() -> Self {
        Self {
            name: String::new(),
            priority: 0,
            estimated_tokens: 0,
            minimum_tokens: None,
            required: false,
            weight: 1.0,
        }
    }
}

/// Allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationStrategy {
    /// Greedy by priority (default) - process sections in priority order
    #[default]
    GreedyPriority,
    /// Proportional / water-filling - allocate proportionally by weight
    Proportional,
    /// Strict priority with minimum guarantees
    StrictPriority,
}

/// Token budget enforcer
///
/// Implements greedy token allocation by priority with optional
/// truncation support.
pub struct TokenBudgetEnforcer {
    /// Total budget
    budget: usize,
    /// Current allocation
    allocated: AtomicUsize,
    /// Estimator for token counting
    estimator: TokenEstimator,
    /// Reserved tokens (for overhead, etc.)
    reserved: usize,
    /// Allocation strategy
    strategy: AllocationStrategy,
}

/// Configuration for TokenBudgetEnforcer
#[derive(Debug, Clone)]
pub struct TokenBudgetConfig {
    /// Total token budget
    pub total_budget: usize,
    /// Reserved tokens for overhead
    pub reserved_tokens: usize,
    /// Enable strict budget enforcement
    pub strict: bool,
    /// Default priority for unspecified sections
    pub default_priority: i32,
    /// Allocation strategy
    pub strategy: AllocationStrategy,
}

impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            total_budget: 4096,
            reserved_tokens: 100,
            strict: false,
            default_priority: 10,
            strategy: AllocationStrategy::GreedyPriority,
        }
    }
}

impl TokenBudgetEnforcer {
    /// Create a new budget enforcer
    pub fn new(config: TokenBudgetConfig) -> Self {
        Self {
            budget: config.total_budget,
            allocated: AtomicUsize::new(0),
            estimator: TokenEstimator::new(),
            reserved: config.reserved_tokens,
            strategy: config.strategy,
        }
    }

    /// Create with simple budget (for backwards compatibility)
    pub fn with_budget(budget: usize) -> Self {
        Self {
            budget,
            allocated: AtomicUsize::new(0),
            estimator: TokenEstimator::new(),
            reserved: 0,
            strategy: AllocationStrategy::GreedyPriority,
        }
    }

    /// Create with custom estimator
    pub fn with_estimator(budget: usize, estimator: TokenEstimator) -> Self {
        Self {
            budget,
            allocated: AtomicUsize::new(0),
            estimator,
            reserved: 0,
            strategy: AllocationStrategy::GreedyPriority,
        }
    }

    /// Set allocation strategy
    pub fn with_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Reserve tokens for overhead (headers, separators, etc.)
    pub fn reserve(&mut self, tokens: usize) {
        self.reserved = tokens;
    }

    /// Get available budget (total - reserved - allocated)
    pub fn available(&self) -> usize {
        let allocated = self.allocated.load(Ordering::Acquire);
        self.budget.saturating_sub(self.reserved + allocated)
    }

    /// Get total budget
    pub fn total_budget(&self) -> usize {
        self.budget
    }

    /// Get allocated tokens
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Acquire)
    }

    /// Try to allocate tokens (returns true if successful)
    pub fn try_allocate(&self, tokens: usize) -> bool {
        loop {
            let current = self.allocated.load(Ordering::Acquire);
            let new_total = current + tokens;

            if new_total + self.reserved > self.budget {
                return false;
            }

            if self
                .allocated
                .compare_exchange(current, new_total, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
            // Retry on contention
        }
    }

    /// Allocate sections by priority (dispatches to strategy-specific method)
    pub fn allocate_sections(&self, sections: &[BudgetSection]) -> BudgetAllocation {
        match self.strategy {
            AllocationStrategy::GreedyPriority => self.allocate_greedy(sections),
            AllocationStrategy::Proportional => self.allocate_proportional(sections),
            AllocationStrategy::StrictPriority => self.allocate_strict(sections),
        }
    }

    /// Greedy allocation by priority order
    fn allocate_greedy(&self, sections: &[BudgetSection]) -> BudgetAllocation {
        // Sort by priority (lower = higher priority)
        let mut sorted: Vec<_> = sections.iter().collect();
        sorted.sort_by_key(|s| s.priority);

        let mut allocation = BudgetAllocation {
            full_sections: Vec::new(),
            truncated_sections: Vec::new(),
            dropped_sections: Vec::new(),
            tokens_allocated: 0,
            tokens_remaining: self.budget.saturating_sub(self.reserved),
            explain: Vec::new(),
        };

        for section in sorted {
            let remaining = allocation.tokens_remaining;

            if section.estimated_tokens <= remaining {
                // Section fits fully
                allocation.full_sections.push(section.name.clone());
                allocation.tokens_allocated += section.estimated_tokens;
                allocation.tokens_remaining -= section.estimated_tokens;
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: section.estimated_tokens,
                    outcome: AllocationOutcome::Full,
                    reason: format!("Fits in remaining budget ({} tokens)", remaining),
                });
            } else if let Some(min) = section.minimum_tokens {
                // Try truncated version
                if min <= remaining {
                    let truncated_to = remaining;
                    allocation.truncated_sections.push((
                        section.name.clone(),
                        section.estimated_tokens,
                        truncated_to,
                    ));
                    allocation.tokens_allocated += truncated_to;
                    allocation.explain.push(AllocationDecision {
                        section: section.name.clone(),
                        priority: section.priority,
                        requested: section.estimated_tokens,
                        allocated: truncated_to,
                        outcome: AllocationOutcome::Truncated,
                        reason: format!(
                            "Truncated from {} to {} tokens (min: {})",
                            section.estimated_tokens, truncated_to, min
                        ),
                    });
                    allocation.tokens_remaining = 0;
                } else {
                    allocation.dropped_sections.push(section.name.clone());
                    allocation.explain.push(AllocationDecision {
                        section: section.name.clone(),
                        priority: section.priority,
                        requested: section.estimated_tokens,
                        allocated: 0,
                        outcome: AllocationOutcome::Dropped,
                        reason: format!(
                            "Minimum {} exceeds remaining {} tokens",
                            min, remaining
                        ),
                    });
                }
            } else {
                // No truncation, must drop
                allocation.dropped_sections.push(section.name.clone());
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: 0,
                    outcome: AllocationOutcome::Dropped,
                    reason: format!(
                        "Requested {} exceeds remaining {} (no truncation allowed)",
                        section.estimated_tokens, remaining
                    ),
                });
            }
        }

        allocation
    }

    /// Proportional / water-filling allocation
    ///
    /// Allocates tokens proportionally by weight:
    /// $$b_i = \lfloor B \cdot w_i / \sum w \rfloor$$
    ///
    /// With minimum guarantees and iterative redistribution.
    fn allocate_proportional(&self, sections: &[BudgetSection]) -> BudgetAllocation {
        let available = self.budget.saturating_sub(self.reserved);
        let total_weight: f32 = sections.iter().map(|s| s.weight).sum();

        if total_weight == 0.0 {
            return self.allocate_greedy(sections);
        }

        let mut allocation = BudgetAllocation {
            full_sections: Vec::new(),
            truncated_sections: Vec::new(),
            dropped_sections: Vec::new(),
            tokens_allocated: 0,
            tokens_remaining: available,
            explain: Vec::new(),
        };

        // Phase 1: Calculate proportional allocations
        let mut allocations: Vec<(usize, usize, bool)> = sections
            .iter()
            .map(|s| {
                let proportional = ((available as f32) * s.weight / total_weight).floor() as usize;
                let capped = proportional.min(s.estimated_tokens);
                let min = s.minimum_tokens.unwrap_or(0);
                (capped.max(min), s.estimated_tokens, capped < s.estimated_tokens)
            })
            .collect();

        // Phase 2: Adjust to fit budget (water-filling)
        let mut total: usize = allocations.iter().map(|(a, _, _)| *a).sum();
        
        // If over budget, reduce proportionally from largest allocations
        while total > available {
            // Find the section with largest allocation that can be reduced
            let max_idx = allocations
                .iter()
                .enumerate()
                .filter(|(i, (a, _, _))| {
                    *a > sections[*i].minimum_tokens.unwrap_or(0)
                })
                .max_by_key(|(_, (a, _, _))| *a)
                .map(|(i, _)| i);

            match max_idx {
                Some(idx) => {
                    let reduce = (total - available).min(allocations[idx].0 - sections[idx].minimum_tokens.unwrap_or(0));
                    allocations[idx].0 -= reduce;
                    total -= reduce;
                }
                None => break, // Can't reduce further
            }
        }

        // Phase 3: Record results
        for (i, section) in sections.iter().enumerate() {
            let (allocated, requested, truncated) = allocations[i];
            
            if allocated == 0 {
                allocation.dropped_sections.push(section.name.clone());
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested,
                    allocated: 0,
                    outcome: AllocationOutcome::Dropped,
                    reason: "No budget available after proportional allocation".to_string(),
                });
            } else if truncated {
                allocation.truncated_sections.push((
                    section.name.clone(),
                    requested,
                    allocated,
                ));
                allocation.tokens_allocated += allocated;
                allocation.tokens_remaining = allocation.tokens_remaining.saturating_sub(allocated);
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested,
                    allocated,
                    outcome: AllocationOutcome::Truncated,
                    reason: format!(
                        "Proportional allocation: {:.1}% of budget (weight {:.1})",
                        (allocated as f32 / available as f32) * 100.0,
                        section.weight
                    ),
                });
            } else {
                allocation.full_sections.push(section.name.clone());
                allocation.tokens_allocated += allocated;
                allocation.tokens_remaining = allocation.tokens_remaining.saturating_sub(allocated);
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested,
                    allocated,
                    outcome: AllocationOutcome::Full,
                    reason: format!(
                        "Full allocation within proportional budget (weight {:.1})",
                        section.weight
                    ),
                });
            }
        }

        allocation
    }

    /// Strict priority with guaranteed minimums for required sections
    fn allocate_strict(&self, sections: &[BudgetSection]) -> BudgetAllocation {
        let mut sorted: Vec<_> = sections.iter().collect();
        sorted.sort_by_key(|s| (if s.required { 0 } else { 1 }, s.priority));
        
        // First pass: allocate minimums for required sections
        let mut allocation = BudgetAllocation {
            full_sections: Vec::new(),
            truncated_sections: Vec::new(),
            dropped_sections: Vec::new(),
            tokens_allocated: 0,
            tokens_remaining: self.budget.saturating_sub(self.reserved),
            explain: Vec::new(),
        };

        // Allocate required sections first (at minimum or full)
        for section in sorted.iter().filter(|s| s.required) {
            let remaining = allocation.tokens_remaining;
            let min = section.minimum_tokens.unwrap_or(section.estimated_tokens);

            if section.estimated_tokens <= remaining {
                allocation.full_sections.push(section.name.clone());
                allocation.tokens_allocated += section.estimated_tokens;
                allocation.tokens_remaining -= section.estimated_tokens;
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: section.estimated_tokens,
                    outcome: AllocationOutcome::Full,
                    reason: "Required section - full allocation".to_string(),
                });
            } else if min <= remaining {
                allocation.truncated_sections.push((
                    section.name.clone(),
                    section.estimated_tokens,
                    remaining,
                ));
                allocation.tokens_allocated += remaining;
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: remaining,
                    outcome: AllocationOutcome::Truncated,
                    reason: "Required section - truncated to fit".to_string(),
                });
                allocation.tokens_remaining = 0;
            }
            // Required sections can't be dropped - would be an error condition
        }

        // Then allocate optional sections
        for section in sorted.iter().filter(|s| !s.required) {
            let remaining = allocation.tokens_remaining;

            if remaining == 0 {
                allocation.dropped_sections.push(section.name.clone());
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: 0,
                    outcome: AllocationOutcome::Dropped,
                    reason: "No budget remaining after required sections".to_string(),
                });
                continue;
            }

            if section.estimated_tokens <= remaining {
                allocation.full_sections.push(section.name.clone());
                allocation.tokens_allocated += section.estimated_tokens;
                allocation.tokens_remaining -= section.estimated_tokens;
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: section.estimated_tokens,
                    outcome: AllocationOutcome::Full,
                    reason: "Optional section - fits in remaining budget".to_string(),
                });
            } else if let Some(min) = section.minimum_tokens {
                if min <= remaining {
                    allocation.truncated_sections.push((
                        section.name.clone(),
                        section.estimated_tokens,
                        remaining,
                    ));
                    allocation.tokens_allocated += remaining;
                    allocation.explain.push(AllocationDecision {
                        section: section.name.clone(),
                        priority: section.priority,
                        requested: section.estimated_tokens,
                        allocated: remaining,
                        outcome: AllocationOutcome::Truncated,
                        reason: "Optional section - truncated to fit".to_string(),
                    });
                    allocation.tokens_remaining = 0;
                } else {
                    allocation.dropped_sections.push(section.name.clone());
                    allocation.explain.push(AllocationDecision {
                        section: section.name.clone(),
                        priority: section.priority,
                        requested: section.estimated_tokens,
                        allocated: 0,
                        outcome: AllocationOutcome::Dropped,
                        reason: format!("Minimum {} exceeds remaining {}", min, remaining),
                    });
                }
            } else {
                allocation.dropped_sections.push(section.name.clone());
                allocation.explain.push(AllocationDecision {
                    section: section.name.clone(),
                    priority: section.priority,
                    requested: section.estimated_tokens,
                    allocated: 0,
                    outcome: AllocationOutcome::Dropped,
                    reason: format!("Requested {} exceeds remaining {}", section.estimated_tokens, remaining),
                });
            }
        }

        allocation
    }

    /// Reset allocation
    pub fn reset(&self) {
        self.allocated.store(0, Ordering::Release);
    }

    /// Get the estimator
    pub fn estimator(&self) -> &TokenEstimator {
        &self.estimator
    }
}

// ============================================================================
// EXPLAIN CONTEXT Output
// ============================================================================

impl BudgetAllocation {
    /// Generate human-readable explanation of budget allocation
    pub fn explain_text(&self) -> String {
        let mut output = String::new();
        output.push_str("=== CONTEXT BUDGET ALLOCATION ===\n\n");
        output.push_str(&format!(
            "Total Allocated: {} tokens\n",
            self.tokens_allocated
        ));
        output.push_str(&format!("Remaining: {} tokens\n\n", self.tokens_remaining));

        output.push_str("SECTIONS:\n");
        for decision in &self.explain {
            let status = match decision.outcome {
                AllocationOutcome::Full => "✓ FULL",
                AllocationOutcome::Truncated => "◐ TRUNCATED",
                AllocationOutcome::Dropped => "✗ DROPPED",
            };
            output.push_str(&format!(
                "  [{:^12}] {} (priority {})\n",
                status, decision.section, decision.priority
            ));
            output.push_str(&format!(
                "               Requested: {}, Allocated: {}\n",
                decision.requested, decision.allocated
            ));
            output.push_str(&format!("               Reason: {}\n", decision.reason));
        }

        output
    }

    /// Generate JSON explanation for programmatic use
    pub fn explain_json(&self) -> String {
        serde_json::to_string_pretty(&ExplainOutput {
            tokens_allocated: self.tokens_allocated,
            tokens_remaining: self.tokens_remaining,
            full_sections: self.full_sections.clone(),
            truncated_sections: self.truncated_sections.clone(),
            dropped_sections: self.dropped_sections.clone(),
            decisions: self.explain.iter().map(|d| ExplainDecision {
                section: d.section.clone(),
                priority: d.priority,
                requested: d.requested,
                allocated: d.allocated,
                outcome: format!("{:?}", d.outcome),
                reason: d.reason.clone(),
            }).collect(),
        }).unwrap_or_else(|_| "{}".to_string())
    }
}

#[derive(serde::Serialize)]
struct ExplainOutput {
    tokens_allocated: usize,
    tokens_remaining: usize,
    full_sections: Vec<String>,
    truncated_sections: Vec<(String, usize, usize)>,
    dropped_sections: Vec<String>,
    decisions: Vec<ExplainDecision>,
}

#[derive(serde::Serialize)]
struct ExplainDecision {
    section: String,
    priority: i32,
    requested: usize,
    allocated: usize,
    outcome: String,
    reason: String,
}

// ============================================================================
// Token-Aware Truncation
// ============================================================================

/// Truncate a string to fit within a token budget
pub fn truncate_to_tokens(
    text: &str,
    max_tokens: usize,
    estimator: &TokenEstimator,
    suffix: &str,
) -> String {
    let current = estimator.estimate_text(text);

    if current <= max_tokens {
        return text.to_string();
    }

    let suffix_tokens = estimator.estimate_text(suffix);
    let target_tokens = max_tokens.saturating_sub(suffix_tokens);

    if target_tokens == 0 {
        return suffix.to_string();
    }

    // Binary search for the right truncation point
    let mut low = 0;
    let mut high = text.len();

    while low < high {
        let mid = (low + high).div_ceil(2);

        // Find character boundary
        let boundary = text
            .char_indices()
            .take_while(|(i, _)| *i < mid)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);

        let truncated = &text[..boundary];
        let tokens = estimator.estimate_text(truncated);

        if tokens <= target_tokens {
            low = boundary;
        } else {
            high = boundary.saturating_sub(1);
        }
    }

    // Find word boundary
    let truncated = &text[..low];
    let word_boundary = truncated.rfind(|c: char| c.is_whitespace()).unwrap_or(low);

    format!("{}{}", &text[..word_boundary], suffix)
}

/// Truncate rows to fit within token budget
pub fn truncate_rows(
    rows: &[Vec<SochValue>],
    max_tokens: usize,
    estimator: &TokenEstimator,
) -> Vec<Vec<SochValue>> {
    let mut result = Vec::new();
    let mut used = 0;

    for row in rows {
        let row_tokens = estimator.estimate_row(row);

        if used + row_tokens <= max_tokens {
            result.push(row.clone());
            used += row_tokens;
        } else {
            break; // No more room
        }
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_value_int() {
        let est = TokenEstimator::new();

        // Small integers
        assert!(est.estimate_value(&SochValue::Int(0)) >= 1);
        assert!(est.estimate_value(&SochValue::Int(42)) >= 1);

        // Large integers use more tokens
        let small = est.estimate_value(&SochValue::Int(42));
        let large = est.estimate_value(&SochValue::Int(1_000_000_000));
        assert!(large >= small);
    }

    #[test]
    fn test_estimate_value_text() {
        let est = TokenEstimator::new();

        let short = est.estimate_value(&SochValue::Text("hello".to_string()));
        let long = est.estimate_value(&SochValue::Text(
            "hello world this is a longer string".to_string(),
        ));

        assert!(long > short);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_estimate_row() {
        let est = TokenEstimator::new();

        let row = vec![
            SochValue::Int(1),
            SochValue::Text("Alice".to_string()),
            SochValue::Float(3.14),
        ];

        let tokens = est.estimate_row(&row);

        // Should be sum of values + separators + newline
        assert!(tokens >= 3); // At least 1 per value
    }

    #[test]
    fn test_estimate_table() {
        let est = TokenEstimator::new();

        let columns = vec!["id".to_string(), "name".to_string()];
        let rows = vec![
            vec![SochValue::Int(1), SochValue::Text("Alice".to_string())],
            vec![SochValue::Int(2), SochValue::Text("Bob".to_string())],
        ];

        let tokens = est.estimate_table("users", &columns, &rows);

        // Should include header + rows
        assert!(tokens > est.estimate_row(&rows[0]) * 2);
    }

    #[test]
    fn test_budget_enforcer_allocation() {
        let enforcer = TokenBudgetEnforcer::with_budget(1000);

        assert!(enforcer.try_allocate(500));
        assert_eq!(enforcer.allocated(), 500);
        assert_eq!(enforcer.available(), 500);

        assert!(enforcer.try_allocate(400));
        assert_eq!(enforcer.allocated(), 900);

        // This should fail (only 100 left)
        assert!(!enforcer.try_allocate(200));
        assert_eq!(enforcer.allocated(), 900);
    }

    #[test]
    fn test_budget_enforcer_reset() {
        let enforcer = TokenBudgetEnforcer::with_budget(1000);

        enforcer.try_allocate(800);
        assert_eq!(enforcer.allocated(), 800);

        enforcer.reset();
        assert_eq!(enforcer.allocated(), 0);
    }

    #[test]
    fn test_allocate_sections() {
        let enforcer = TokenBudgetEnforcer::with_budget(1000);

        let sections = vec![
            BudgetSection {
                name: "A".to_string(),
                priority: 0,
                estimated_tokens: 300,
                minimum_tokens: None,
                required: true,
                weight: 1.0,
            },
            BudgetSection {
                name: "B".to_string(),
                priority: 1,
                estimated_tokens: 400,
                minimum_tokens: Some(200),
                required: false,
                weight: 1.0,
            },
            BudgetSection {
                name: "C".to_string(),
                priority: 2,
                estimated_tokens: 500,
                minimum_tokens: None,
                required: false,
                weight: 1.0,
            },
        ];

        let allocation = enforcer.allocate_sections(&sections);

        // A fits fully
        assert!(allocation.full_sections.contains(&"A".to_string()));

        // B might fit (300 remaining after A)
        // C won't fit (500 tokens, only 300 remaining)
        assert!(allocation.dropped_sections.contains(&"C".to_string()));

        assert!(allocation.tokens_allocated <= 1000);
    }

    #[test]
    fn test_allocate_by_priority() {
        let enforcer = TokenBudgetEnforcer::with_budget(500);

        let sections = vec![
            BudgetSection {
                name: "LowPriority".to_string(),
                priority: 10,
                estimated_tokens: 200,
                minimum_tokens: None,
                required: false,
                weight: 1.0,
            },
            BudgetSection {
                name: "HighPriority".to_string(),
                priority: 0,
                estimated_tokens: 400,
                minimum_tokens: None,
                required: true,
                weight: 1.0,
            },
        ];

        let allocation = enforcer.allocate_sections(&sections);

        // High priority goes first
        assert!(
            allocation
                .full_sections
                .contains(&"HighPriority".to_string())
        );

        // Low priority dropped (only 100 remaining)
        assert!(
            allocation
                .dropped_sections
                .contains(&"LowPriority".to_string())
        );
    }

    #[test]
    fn test_truncate_to_tokens() {
        let est = TokenEstimator::new();

        let text = "This is a long text that needs to be truncated to fit within the token budget";
        let truncated = truncate_to_tokens(text, 10, &est, "...");

        // Should be shorter
        assert!(truncated.len() < text.len());

        // Should end with suffix
        assert!(truncated.ends_with("..."));

        // Should fit budget
        assert!(est.estimate_text(&truncated) <= 10);
    }

    #[test]
    fn test_truncate_rows() {
        let est = TokenEstimator::new();

        let rows: Vec<Vec<SochValue>> = (0..100)
            .map(|i| vec![SochValue::Int(i), SochValue::Text(format!("row{}", i))])
            .collect();

        let truncated = truncate_rows(&rows, 50, &est);

        // Should have fewer rows
        assert!(truncated.len() < rows.len());

        // Total tokens should be under budget
        let total: usize = truncated.iter().map(|r| est.estimate_row(r)).sum();
        assert!(total <= 50);
    }

    #[test]
    fn test_reserved_budget() {
        let mut enforcer = TokenBudgetEnforcer::with_budget(1000);
        enforcer.reserve(200);

        assert_eq!(enforcer.available(), 800);

        assert!(enforcer.try_allocate(700));
        assert_eq!(enforcer.available(), 100);

        // Cannot exceed available (reserves are protected)
        assert!(!enforcer.try_allocate(200));
    }

    #[test]
    fn test_estimator_configs() {
        let default = TokenEstimator::new();
        let gpt4 = TokenEstimator::with_config(TokenEstimatorConfig::gpt4());
        let conservative = TokenEstimator::with_config(TokenEstimatorConfig::conservative());

        let text = "Hello, this is a test string for comparing token estimation across different configurations.";

        let default_est = default.estimate_text(text);
        let gpt4_est = gpt4.estimate_text(text);
        let conservative_est = conservative.estimate_text(text);

        // Conservative should give highest estimate
        assert!(conservative_est >= default_est);

        // All should be positive
        assert!(default_est > 0);
        assert!(gpt4_est > 0);
        assert!(conservative_est > 0);
    }

    #[test]
    fn test_section_with_truncation() {
        let enforcer = TokenBudgetEnforcer::with_budget(600);

        let sections = vec![
            BudgetSection {
                name: "Required".to_string(),
                priority: 0,
                estimated_tokens: 500,
                minimum_tokens: None,
                required: true,
                weight: 1.0,
            },
            BudgetSection {
                name: "Optional".to_string(),
                priority: 1,
                estimated_tokens: 300,
                minimum_tokens: Some(50), // Can be truncated
                required: false,
                weight: 1.0,
            },
        ];

        let allocation = enforcer.allocate_sections(&sections);

        // Required fits
        assert!(allocation.full_sections.contains(&"Required".to_string()));

        // Optional gets truncated (only 100 remaining, min is 50)
        assert!(
            allocation
                .truncated_sections
                .iter()
                .any(|(n, _, _)| n == "Optional")
        );
    }
}
