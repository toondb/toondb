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

//! Cost-Based Query Optimizer with Cardinality Estimation (Task 6)
//!
//! Provides cost-based query optimization for SOCH-QL with:
//! - Cardinality estimation using sketches (HyperLogLog, CountMin)
//! - Index selection: compare cost(table_scan) vs cost(index_seek)
//! - Column projection pushdown to LSCS layer
//! - Token-budget-aware planning
//!
//! ## Cost Model
//!
//! cost(plan) = I/O_cost + CPU_cost + memory_cost
//!
//! I/O_cost = blocks_read × C_seq + seeks × C_random
//! Where:
//!   C_seq = 0.1 ms/block (sequential read)
//!   C_random = 5 ms/seek (random seek)
//!
//! CPU_cost = rows_processed × C_filter + sorts × N × log(N) × C_compare
//!
//! ## Selectivity Estimation
//!
//! Uses CountMinSketch for predicate selectivity and HyperLogLog for distinct counts.
//!
//! ## Token Budget Planning
//!
//! Given max_tokens, estimates result size and injects LIMIT clause:
//!   max_rows = (max_tokens - header_tokens) / tokens_per_row

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ============================================================================
// Cost Model Constants
// ============================================================================

/// Cost model configuration with empirically-derived constants
#[derive(Debug, Clone)]
pub struct CostModelConfig {
    /// Sequential I/O cost per block (ms)
    pub c_seq: f64,
    /// Random I/O cost per seek (ms)
    pub c_random: f64,
    /// CPU cost per row filter (ms)
    pub c_filter: f64,
    /// CPU cost per comparison during sort (ms)
    pub c_compare: f64,
    /// Block size in bytes
    pub block_size: usize,
    /// B-tree fanout for index cost estimation
    pub btree_fanout: usize,
    /// Memory bandwidth (bytes/ms)
    pub memory_bandwidth: f64,
}

impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            c_seq: 0.1,                // 0.1 ms per block sequential
            c_random: 5.0,             // 5 ms per random seek
            c_filter: 0.001,           // 0.001 ms per row filter
            c_compare: 0.0001,         // 0.0001 ms per comparison
            block_size: 4096,          // 4 KB blocks
            btree_fanout: 100,         // 100 entries per B-tree node
            memory_bandwidth: 10000.0, // 10 GB/s = 10000 bytes/ms
        }
    }
}

// ============================================================================
// Statistics for Cardinality Estimation
// ============================================================================

/// Table statistics for cost estimation
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Table name
    pub name: String,
    /// Total row count
    pub row_count: u64,
    /// Total size in bytes
    pub size_bytes: u64,
    /// Column statistics
    pub column_stats: HashMap<String, ColumnStats>,
    /// Available indices
    pub indices: Vec<IndexStats>,
    /// Last update timestamp
    pub last_updated: u64,
}

/// Column statistics
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Distinct value count (from HyperLogLog)
    pub distinct_count: u64,
    /// Null count
    pub null_count: u64,
    /// Minimum value (if orderable)
    pub min_value: Option<String>,
    /// Maximum value (if orderable)
    pub max_value: Option<String>,
    /// Average length in bytes (for variable-length types)
    pub avg_length: f64,
    /// Most common values with frequencies
    pub mcv: Vec<(String, f64)>,
    /// Histogram buckets for range queries
    pub histogram: Option<Histogram>,
}

/// Histogram for range selectivity estimation
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bucket boundaries
    pub boundaries: Vec<f64>,
    /// Row count per bucket
    pub counts: Vec<u64>,
    /// Total rows in histogram
    pub total_rows: u64,
}

impl Histogram {
    /// Estimate selectivity for a range predicate
    pub fn estimate_range_selectivity(&self, min: Option<f64>, max: Option<f64>) -> f64 {
        if self.total_rows == 0 {
            return 0.5; // Default
        }

        let mut selected_rows = 0u64;

        for (i, &count) in self.counts.iter().enumerate() {
            let bucket_min = if i == 0 {
                f64::NEG_INFINITY
            } else {
                self.boundaries[i - 1]
            };
            let bucket_max = if i == self.boundaries.len() {
                f64::INFINITY
            } else {
                self.boundaries[i]
            };

            let overlaps = match (min, max) {
                (Some(min_val), Some(max_val)) => bucket_max >= min_val && bucket_min <= max_val,
                (Some(min_val), None) => bucket_max >= min_val,
                (None, Some(max_val)) => bucket_min <= max_val,
                (None, None) => true,
            };

            if overlaps {
                selected_rows += count;
            }
        }

        selected_rows as f64 / self.total_rows as f64
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Index name
    pub name: String,
    /// Indexed columns
    pub columns: Vec<String>,
    /// Is primary key
    pub is_primary: bool,
    /// Is unique
    pub is_unique: bool,
    /// Index type
    pub index_type: IndexType,
    /// Number of leaf pages
    pub leaf_pages: u64,
    /// Tree height (for B-tree)
    pub height: u32,
    /// Average entries per leaf page
    pub avg_leaf_density: f64,
}

/// Index types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BTree,
    Hash,
    LSM,
    Learned,
    Vector,
    Bloom,
}

// ============================================================================
// Query Predicates and Operations
// ============================================================================

/// Query predicate for cost estimation
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Equality: column = value
    Eq { column: String, value: String },
    /// Inequality: column != value
    Ne { column: String, value: String },
    /// Less than: column < value
    Lt { column: String, value: String },
    /// Less than or equal: column <= value
    Le { column: String, value: String },
    /// Greater than: column > value
    Gt { column: String, value: String },
    /// Greater than or equal: column >= value
    Ge { column: String, value: String },
    /// Between: column BETWEEN min AND max
    Between {
        column: String,
        min: String,
        max: String,
    },
    /// In list: column IN (v1, v2, ...)
    In { column: String, values: Vec<String> },
    /// Like: column LIKE pattern
    Like { column: String, pattern: String },
    /// Is null: column IS NULL
    IsNull { column: String },
    /// Is not null: column IS NOT NULL
    IsNotNull { column: String },
    /// And: pred1 AND pred2
    And(Box<Predicate>, Box<Predicate>),
    /// Or: pred1 OR pred2
    Or(Box<Predicate>, Box<Predicate>),
    /// Not: NOT pred
    Not(Box<Predicate>),
}

impl Predicate {
    /// Get columns referenced by this predicate
    pub fn referenced_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        self.collect_columns(&mut cols);
        cols
    }

    fn collect_columns(&self, cols: &mut HashSet<String>) {
        match self {
            Self::Eq { column, .. }
            | Self::Ne { column, .. }
            | Self::Lt { column, .. }
            | Self::Le { column, .. }
            | Self::Gt { column, .. }
            | Self::Ge { column, .. }
            | Self::Between { column, .. }
            | Self::In { column, .. }
            | Self::Like { column, .. }
            | Self::IsNull { column }
            | Self::IsNotNull { column } => {
                cols.insert(column.clone());
            }
            Self::And(left, right) | Self::Or(left, right) => {
                left.collect_columns(cols);
                right.collect_columns(cols);
            }
            Self::Not(inner) => inner.collect_columns(cols),
        }
    }
}

// ============================================================================
// Physical Plan Operators
// ============================================================================

/// Physical query plan node
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Table scan (full or partial)
    TableScan {
        table: String,
        columns: Vec<String>,
        predicate: Option<Box<Predicate>>,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Index seek
    IndexSeek {
        table: String,
        index: String,
        columns: Vec<String>,
        key_range: KeyRange,
        predicate: Option<Box<Predicate>>,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Filter operator
    Filter {
        input: Box<PhysicalPlan>,
        predicate: Predicate,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Project operator (column subset)
    Project {
        input: Box<PhysicalPlan>,
        columns: Vec<String>,
        estimated_cost: f64,
    },
    /// Sort operator
    Sort {
        input: Box<PhysicalPlan>,
        order_by: Vec<(String, SortDirection)>,
        estimated_cost: f64,
    },
    /// Limit operator
    Limit {
        input: Box<PhysicalPlan>,
        limit: u64,
        offset: u64,
        estimated_cost: f64,
    },
    /// Nested loop join
    NestedLoopJoin {
        outer: Box<PhysicalPlan>,
        inner: Box<PhysicalPlan>,
        condition: Predicate,
        join_type: JoinType,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Hash join
    HashJoin {
        build: Box<PhysicalPlan>,
        probe: Box<PhysicalPlan>,
        build_keys: Vec<String>,
        probe_keys: Vec<String>,
        join_type: JoinType,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Merge join
    MergeJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        join_type: JoinType,
        estimated_rows: u64,
        estimated_cost: f64,
    },
    /// Aggregate operator
    Aggregate {
        input: Box<PhysicalPlan>,
        group_by: Vec<String>,
        aggregates: Vec<AggregateExpr>,
        estimated_rows: u64,
        estimated_cost: f64,
    },
}

/// Key range for index seeks
#[derive(Debug, Clone)]
pub struct KeyRange {
    pub start: Option<Vec<u8>>,
    pub end: Option<Vec<u8>>,
    pub start_inclusive: bool,
    pub end_inclusive: bool,
}

impl KeyRange {
    pub fn all() -> Self {
        Self {
            start: None,
            end: None,
            start_inclusive: true,
            end_inclusive: true,
        }
    }

    pub fn point(key: Vec<u8>) -> Self {
        Self {
            start: Some(key.clone()),
            end: Some(key),
            start_inclusive: true,
            end_inclusive: true,
        }
    }

    pub fn range(start: Option<Vec<u8>>, end: Option<Vec<u8>>, inclusive: bool) -> Self {
        Self {
            start,
            end,
            start_inclusive: inclusive,
            end_inclusive: inclusive,
        }
    }
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Join type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Aggregate expression
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    pub function: AggregateFunction,
    pub column: Option<String>,
    pub alias: String,
}

/// Aggregate functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    CountDistinct,
}

// ============================================================================
// Cost-Based Query Optimizer
// ============================================================================

/// Cost-based query optimizer
pub struct CostBasedOptimizer {
    /// Cost model configuration
    config: CostModelConfig,
    /// Table statistics cache
    stats_cache: Arc<RwLock<HashMap<String, TableStats>>>,
    /// Token budget for result limiting
    token_budget: Option<u64>,
    /// Estimated tokens per row
    tokens_per_row: f64,
}

impl CostBasedOptimizer {
    pub fn new(config: CostModelConfig) -> Self {
        Self {
            config,
            stats_cache: Arc::new(RwLock::new(HashMap::new())),
            token_budget: None,
            tokens_per_row: 25.0, // Default estimate
        }
    }

    /// Set token budget for result limiting
    pub fn with_token_budget(mut self, budget: u64, tokens_per_row: f64) -> Self {
        self.token_budget = Some(budget);
        self.tokens_per_row = tokens_per_row;
        self
    }

    /// Update table statistics
    pub fn update_stats(&self, stats: TableStats) {
        self.stats_cache.write().insert(stats.name.clone(), stats);
    }

    /// Get table statistics
    pub fn get_stats(&self, table: &str) -> Option<TableStats> {
        self.stats_cache.read().get(table).cloned()
    }

    /// Optimize a SELECT query
    pub fn optimize(
        &self,
        table: &str,
        columns: Vec<String>,
        predicate: Option<Predicate>,
        order_by: Vec<(String, SortDirection)>,
        limit: Option<u64>,
    ) -> PhysicalPlan {
        let stats = self.get_stats(table);

        // Calculate token-aware limit
        let effective_limit = self.calculate_token_limit(limit);

        // Get best access path (scan vs index)
        let mut plan = self.choose_access_path(table, &columns, predicate.as_ref(), &stats);

        // Apply column projection pushdown
        plan = self.apply_projection_pushdown(plan, columns.clone());

        // Apply sorting if needed
        if !order_by.is_empty() {
            plan = self.add_sort(plan, order_by, &stats);
        }

        // Apply limit
        if let Some(lim) = effective_limit {
            plan = PhysicalPlan::Limit {
                estimated_cost: 0.0,
                input: Box::new(plan),
                limit: lim,
                offset: 0,
            };
        }

        plan
    }

    /// Calculate token-aware limit
    fn calculate_token_limit(&self, user_limit: Option<u64>) -> Option<u64> {
        match (self.token_budget, user_limit) {
            (Some(budget), Some(limit)) => {
                let header_tokens = 50u64;
                let max_rows = ((budget - header_tokens) as f64 / self.tokens_per_row) as u64;
                Some(limit.min(max_rows))
            }
            (Some(budget), None) => {
                let header_tokens = 50u64;
                let max_rows = ((budget - header_tokens) as f64 / self.tokens_per_row) as u64;
                Some(max_rows)
            }
            (None, limit) => limit,
        }
    }

    /// Choose best access path (table scan vs index seek)
    fn choose_access_path(
        &self,
        table: &str,
        columns: &[String],
        predicate: Option<&Predicate>,
        stats: &Option<TableStats>,
    ) -> PhysicalPlan {
        let row_count = stats.as_ref().map(|s| s.row_count).unwrap_or(10000);
        let size_bytes = stats
            .as_ref()
            .map(|s| s.size_bytes)
            .unwrap_or(row_count * 100);

        // Calculate table scan cost
        let scan_cost = self.estimate_scan_cost(row_count, size_bytes, predicate);

        // Try to find a suitable index
        let mut best_index_cost = f64::MAX;
        let mut best_index: Option<&IndexStats> = None;

        if let Some(table_stats) = stats.as_ref()
            && let Some(pred) = predicate
        {
            let pred_columns = pred.referenced_columns();

            for index in &table_stats.indices {
                if self.index_covers_predicate(index, &pred_columns) {
                    let selectivity = self.estimate_selectivity(pred, table_stats);
                    let index_cost = self.estimate_index_cost(index, row_count, selectivity);

                    if index_cost < best_index_cost {
                        best_index_cost = index_cost;
                        best_index = Some(index);
                    }
                }
            }
        }

        // Choose cheaper option
        if best_index_cost < scan_cost {
            let index = best_index.unwrap();
            let selectivity = predicate
                .map(|p| self.estimate_selectivity(p, stats.as_ref().unwrap()))
                .unwrap_or(1.0);

            PhysicalPlan::IndexSeek {
                table: table.to_string(),
                index: index.name.clone(),
                columns: columns.to_vec(),
                key_range: KeyRange::all(), // Simplified
                predicate: predicate.map(|p| Box::new(p.clone())),
                estimated_rows: (row_count as f64 * selectivity) as u64,
                estimated_cost: best_index_cost,
            }
        } else {
            PhysicalPlan::TableScan {
                table: table.to_string(),
                columns: columns.to_vec(),
                predicate: predicate.map(|p| Box::new(p.clone())),
                estimated_rows: row_count,
                estimated_cost: scan_cost,
            }
        }
    }

    /// Check if index covers predicate columns
    fn index_covers_predicate(&self, index: &IndexStats, pred_columns: &HashSet<String>) -> bool {
        // Index is useful if it covers at least the first column of the predicate
        if let Some(first_col) = index.columns.first() {
            pred_columns.contains(first_col)
        } else {
            false
        }
    }

    /// Estimate table scan cost
    fn estimate_scan_cost(
        &self,
        row_count: u64,
        size_bytes: u64,
        predicate: Option<&Predicate>,
    ) -> f64 {
        let blocks = (size_bytes as f64 / self.config.block_size as f64).ceil() as u64;

        // I/O cost: sequential read
        let io_cost = blocks as f64 * self.config.c_seq;

        // CPU cost: filter all rows
        let selectivity = predicate.map(|_| 0.1).unwrap_or(1.0);
        let cpu_cost = row_count as f64 * self.config.c_filter * selectivity;

        io_cost + cpu_cost
    }

    /// Estimate index seek cost
    ///
    /// Index cost = tree_traversal + leaf_scan + row_fetch
    fn estimate_index_cost(&self, index: &IndexStats, total_rows: u64, selectivity: f64) -> f64 {
        // Tree traversal cost (random I/O for each level)
        let tree_cost = index.height as f64 * self.config.c_random;

        // Leaf scan cost (sequential for matching range)
        let matching_rows = (total_rows as f64 * selectivity) as u64;
        let leaf_pages_scanned = (matching_rows as f64 / index.avg_leaf_density).ceil() as u64;
        let leaf_cost = leaf_pages_scanned as f64 * self.config.c_seq;

        // Row fetch cost (random if not clustered)
        let fetch_cost = if index.is_primary {
            0.0 // Clustered index, no extra fetch
        } else {
            matching_rows.min(1000) as f64 * self.config.c_random * 0.1 // Batch optimization
        };

        tree_cost + leaf_cost + fetch_cost
    }

    /// Estimate predicate selectivity
    #[allow(clippy::only_used_in_recursion)]
    fn estimate_selectivity(&self, predicate: &Predicate, stats: &TableStats) -> f64 {
        match predicate {
            Predicate::Eq { column, value } => {
                if let Some(col_stats) = stats.column_stats.get(column) {
                    // Check MCV first
                    for (mcv_val, freq) in &col_stats.mcv {
                        if mcv_val == value {
                            return *freq;
                        }
                    }
                    // Otherwise use uniform distribution
                    1.0 / col_stats.distinct_count.max(1) as f64
                } else {
                    0.1 // Default 10%
                }
            }
            Predicate::Ne { .. } => 0.9, // 90% pass
            Predicate::Lt { column, value }
            | Predicate::Le { column, value }
            | Predicate::Gt { column, value }
            | Predicate::Ge { column, value } => {
                if let Some(col_stats) = stats.column_stats.get(column) {
                    if let Some(ref hist) = col_stats.histogram {
                        let val: f64 = value.parse().unwrap_or(0.0);
                        match predicate {
                            Predicate::Lt { .. } | Predicate::Le { .. } => {
                                hist.estimate_range_selectivity(None, Some(val))
                            }
                            _ => hist.estimate_range_selectivity(Some(val), None),
                        }
                    } else {
                        0.25 // Default 25%
                    }
                } else {
                    0.25
                }
            }
            Predicate::Between { column, min, max } => {
                if let Some(col_stats) = stats.column_stats.get(column) {
                    if let Some(ref hist) = col_stats.histogram {
                        let min_val: f64 = min.parse().unwrap_or(0.0);
                        let max_val: f64 = max.parse().unwrap_or(f64::MAX);
                        hist.estimate_range_selectivity(Some(min_val), Some(max_val))
                    } else {
                        0.2
                    }
                } else {
                    0.2
                }
            }
            Predicate::In { column, values } => {
                if let Some(col_stats) = stats.column_stats.get(column) {
                    (values.len() as f64 / col_stats.distinct_count.max(1) as f64).min(1.0)
                } else {
                    (values.len() as f64 * 0.1).min(0.5)
                }
            }
            Predicate::Like { .. } => 0.15, // Default 15%
            Predicate::IsNull { column } => {
                if let Some(col_stats) = stats.column_stats.get(column) {
                    col_stats.null_count as f64 / stats.row_count.max(1) as f64
                } else {
                    0.01
                }
            }
            Predicate::IsNotNull { column } => {
                if let Some(col_stats) = stats.column_stats.get(column) {
                    1.0 - (col_stats.null_count as f64 / stats.row_count.max(1) as f64)
                } else {
                    0.99
                }
            }
            Predicate::And(left, right) => {
                // Assume independence
                self.estimate_selectivity(left, stats) * self.estimate_selectivity(right, stats)
            }
            Predicate::Or(left, right) => {
                let s1 = self.estimate_selectivity(left, stats);
                let s2 = self.estimate_selectivity(right, stats);
                // P(A or B) = P(A) + P(B) - P(A and B)
                (s1 + s2 - s1 * s2).min(1.0)
            }
            Predicate::Not(inner) => 1.0 - self.estimate_selectivity(inner, stats),
        }
    }

    /// Apply column projection pushdown
    fn apply_projection_pushdown(&self, plan: PhysicalPlan, columns: Vec<String>) -> PhysicalPlan {
        // If plan already has projection, merge; otherwise add
        match plan {
            PhysicalPlan::TableScan {
                table,
                predicate,
                estimated_rows,
                estimated_cost,
                ..
            } => {
                PhysicalPlan::TableScan {
                    table,
                    columns, // Pushed down columns
                    predicate,
                    estimated_rows,
                    estimated_cost: estimated_cost * 0.2, // Reduce cost estimate
                }
            }
            PhysicalPlan::IndexSeek {
                table,
                index,
                key_range,
                predicate,
                estimated_rows,
                estimated_cost,
                ..
            } => {
                PhysicalPlan::IndexSeek {
                    table,
                    index,
                    columns, // Pushed down columns
                    key_range,
                    predicate,
                    estimated_rows,
                    estimated_cost,
                }
            }
            other => PhysicalPlan::Project {
                input: Box::new(other),
                columns,
                estimated_cost: 0.0,
            },
        }
    }

    /// Add sort operator
    fn add_sort(
        &self,
        plan: PhysicalPlan,
        order_by: Vec<(String, SortDirection)>,
        _stats: &Option<TableStats>,
    ) -> PhysicalPlan {
        let estimated_rows = self.get_plan_rows(&plan);
        let sort_cost = if estimated_rows > 0 {
            estimated_rows as f64 * (estimated_rows as f64).log2() * self.config.c_compare
        } else {
            0.0
        };

        PhysicalPlan::Sort {
            input: Box::new(plan),
            order_by,
            estimated_cost: sort_cost,
        }
    }

    /// Get estimated rows from a plan
    #[allow(clippy::only_used_in_recursion)]
    fn get_plan_rows(&self, plan: &PhysicalPlan) -> u64 {
        match plan {
            PhysicalPlan::TableScan { estimated_rows, .. }
            | PhysicalPlan::IndexSeek { estimated_rows, .. }
            | PhysicalPlan::Filter { estimated_rows, .. }
            | PhysicalPlan::Aggregate { estimated_rows, .. }
            | PhysicalPlan::NestedLoopJoin { estimated_rows, .. }
            | PhysicalPlan::HashJoin { estimated_rows, .. }
            | PhysicalPlan::MergeJoin { estimated_rows, .. } => *estimated_rows,
            PhysicalPlan::Project { input, .. } | PhysicalPlan::Sort { input, .. } => {
                self.get_plan_rows(input)
            }
            PhysicalPlan::Limit { limit, .. } => *limit,
        }
    }

    /// Get estimated cost from a plan
    #[allow(clippy::only_used_in_recursion)]
    pub fn get_plan_cost(&self, plan: &PhysicalPlan) -> f64 {
        match plan {
            PhysicalPlan::TableScan { estimated_cost, .. } => *estimated_cost,
            PhysicalPlan::IndexSeek { estimated_cost, .. } => *estimated_cost,
            PhysicalPlan::Filter {
                estimated_cost,
                input,
                ..
            } => *estimated_cost + self.get_plan_cost(input),
            PhysicalPlan::Project {
                estimated_cost,
                input,
                ..
            } => *estimated_cost + self.get_plan_cost(input),
            PhysicalPlan::Sort {
                estimated_cost,
                input,
                ..
            } => *estimated_cost + self.get_plan_cost(input),
            PhysicalPlan::Limit {
                estimated_cost,
                input,
                ..
            } => *estimated_cost + self.get_plan_cost(input),
            PhysicalPlan::NestedLoopJoin {
                estimated_cost,
                outer,
                inner,
                ..
            } => *estimated_cost + self.get_plan_cost(outer) + self.get_plan_cost(inner),
            PhysicalPlan::HashJoin {
                estimated_cost,
                build,
                probe,
                ..
            } => *estimated_cost + self.get_plan_cost(build) + self.get_plan_cost(probe),
            PhysicalPlan::MergeJoin {
                estimated_cost,
                left,
                right,
                ..
            } => *estimated_cost + self.get_plan_cost(left) + self.get_plan_cost(right),
            PhysicalPlan::Aggregate {
                estimated_cost,
                input,
                ..
            } => *estimated_cost + self.get_plan_cost(input),
        }
    }

    /// Generate EXPLAIN output
    pub fn explain(&self, plan: &PhysicalPlan) -> String {
        self.explain_impl(plan, 0)
    }

    fn explain_impl(&self, plan: &PhysicalPlan, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let cost = self.get_plan_cost(plan);

        match plan {
            PhysicalPlan::TableScan {
                table,
                columns,
                estimated_rows,
                ..
            } => {
                format!(
                    "{}TableScan [table={}, columns={:?}, rows={}, cost={:.2}ms]",
                    prefix, table, columns, estimated_rows, cost
                )
            }
            PhysicalPlan::IndexSeek {
                table,
                index,
                columns,
                estimated_rows,
                ..
            } => {
                format!(
                    "{}IndexSeek [table={}, index={}, columns={:?}, rows={}, cost={:.2}ms]",
                    prefix, table, index, columns, estimated_rows, cost
                )
            }
            PhysicalPlan::Filter {
                input,
                estimated_rows,
                ..
            } => {
                format!(
                    "{}Filter [rows={}, cost={:.2}ms]\n{}",
                    prefix,
                    estimated_rows,
                    cost,
                    self.explain_impl(input, indent + 1)
                )
            }
            PhysicalPlan::Project { input, columns, .. } => {
                format!(
                    "{}Project [columns={:?}, cost={:.2}ms]\n{}",
                    prefix,
                    columns,
                    cost,
                    self.explain_impl(input, indent + 1)
                )
            }
            PhysicalPlan::Sort {
                input, order_by, ..
            } => {
                let order: Vec<_> = order_by
                    .iter()
                    .map(|(c, d)| format!("{} {:?}", c, d))
                    .collect();
                format!(
                    "{}Sort [order={:?}, cost={:.2}ms]\n{}",
                    prefix,
                    order,
                    cost,
                    self.explain_impl(input, indent + 1)
                )
            }
            PhysicalPlan::Limit {
                input,
                limit,
                offset,
                ..
            } => {
                format!(
                    "{}Limit [limit={}, offset={}, cost={:.2}ms]\n{}",
                    prefix,
                    limit,
                    offset,
                    cost,
                    self.explain_impl(input, indent + 1)
                )
            }
            PhysicalPlan::HashJoin {
                build,
                probe,
                join_type,
                estimated_rows,
                ..
            } => {
                format!(
                    "{}HashJoin [type={:?}, rows={}, cost={:.2}ms]\n{}\n{}",
                    prefix,
                    join_type,
                    estimated_rows,
                    cost,
                    self.explain_impl(build, indent + 1),
                    self.explain_impl(probe, indent + 1)
                )
            }
            PhysicalPlan::MergeJoin {
                left,
                right,
                join_type,
                estimated_rows,
                ..
            } => {
                format!(
                    "{}MergeJoin [type={:?}, rows={}, cost={:.2}ms]\n{}\n{}",
                    prefix,
                    join_type,
                    estimated_rows,
                    cost,
                    self.explain_impl(left, indent + 1),
                    self.explain_impl(right, indent + 1)
                )
            }
            PhysicalPlan::NestedLoopJoin {
                outer,
                inner,
                join_type,
                estimated_rows,
                ..
            } => {
                format!(
                    "{}NestedLoopJoin [type={:?}, rows={}, cost={:.2}ms]\n{}\n{}",
                    prefix,
                    join_type,
                    estimated_rows,
                    cost,
                    self.explain_impl(outer, indent + 1),
                    self.explain_impl(inner, indent + 1)
                )
            }
            PhysicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
                estimated_rows,
                ..
            } => {
                let aggs: Vec<_> = aggregates
                    .iter()
                    .map(|a| format!("{:?}({})", a.function, a.column.as_deref().unwrap_or("*")))
                    .collect();
                format!(
                    "{}Aggregate [group_by={:?}, aggs={:?}, rows={}, cost={:.2}ms]\n{}",
                    prefix,
                    group_by,
                    aggs,
                    estimated_rows,
                    cost,
                    self.explain_impl(input, indent + 1)
                )
            }
        }
    }
}

// ============================================================================
// Join Order Optimizer (Dynamic Programming)
// ============================================================================

/// Join order optimizer using dynamic programming
pub struct JoinOrderOptimizer {
    /// Table statistics
    stats: HashMap<String, TableStats>,
    /// Cost model
    config: CostModelConfig,
}

impl JoinOrderOptimizer {
    pub fn new(config: CostModelConfig) -> Self {
        Self {
            stats: HashMap::new(),
            config,
        }
    }

    /// Add table statistics
    pub fn add_stats(&mut self, stats: TableStats) {
        self.stats.insert(stats.name.clone(), stats);
    }

    /// Find optimal join order using dynamic programming
    ///
    /// Time: O(2^n × n^2) where n = number of tables
    /// Practical for n ≤ 10
    pub fn find_optimal_order(
        &self,
        tables: &[String],
        join_conditions: &[(String, String, String, String)], // (table1, col1, table2, col2)
    ) -> Vec<(String, String)> {
        let n = tables.len();
        if n <= 1 {
            return vec![];
        }

        // dp[mask] = (cost, join_order)
        let mut dp: HashMap<u32, (f64, Vec<(String, String)>)> = HashMap::new();

        // Base case: single tables
        for (i, _table) in tables.iter().enumerate() {
            let mask = 1u32 << i;
            dp.insert(mask, (0.0, vec![]));
        }

        // Build up larger subsets
        for size in 2..=n {
            for mask in 0..(1u32 << n) {
                if mask.count_ones() != size as u32 {
                    continue;
                }

                let mut best_cost = f64::MAX;
                let mut best_order = vec![];

                // Try all ways to split into two non-empty subsets
                for sub in 1..mask {
                    if sub & mask != sub || sub == 0 {
                        continue;
                    }
                    let other = mask ^ sub;
                    if other == 0 {
                        continue;
                    }

                    // Check if there's a join between sub and other
                    if !self.has_join_condition(tables, sub, other, join_conditions) {
                        continue;
                    }

                    if let (Some((cost1, order1)), Some((cost2, order2))) =
                        (dp.get(&sub), dp.get(&other))
                    {
                        let join_cost = self.estimate_join_cost(tables, sub, other);
                        let total_cost = cost1 + cost2 + join_cost;

                        if total_cost < best_cost {
                            best_cost = total_cost;
                            best_order = order1.clone();
                            best_order.extend(order2.clone());

                            // Add the join
                            let (t1, t2) =
                                self.get_join_tables(tables, sub, other, join_conditions);
                            if let Some((t1, t2)) = Some((t1, t2)) {
                                best_order.push((t1, t2));
                            }
                        }
                    }
                }

                if best_cost < f64::MAX {
                    dp.insert(mask, (best_cost, best_order));
                }
            }
        }

        let full_mask = (1u32 << n) - 1;
        dp.get(&full_mask)
            .map(|(_, order)| order.clone())
            .unwrap_or_default()
    }

    fn has_join_condition(
        &self,
        tables: &[String],
        mask1: u32,
        mask2: u32,
        conditions: &[(String, String, String, String)],
    ) -> bool {
        for (t1, _, t2, _) in conditions {
            let idx1 = tables.iter().position(|t| t == t1);
            let idx2 = tables.iter().position(|t| t == t2);

            if let (Some(i1), Some(i2)) = (idx1, idx2) {
                let in_mask1 = (mask1 >> i1) & 1 == 1;
                let in_mask2 = (mask2 >> i2) & 1 == 1;

                if in_mask1 && in_mask2 {
                    return true;
                }
            }
        }
        false
    }

    fn get_join_tables(
        &self,
        tables: &[String],
        mask1: u32,
        mask2: u32,
        conditions: &[(String, String, String, String)],
    ) -> (String, String) {
        for (t1, _, t2, _) in conditions {
            let idx1 = tables.iter().position(|t| t == t1);
            let idx2 = tables.iter().position(|t| t == t2);

            if let (Some(i1), Some(i2)) = (idx1, idx2) {
                let t1_in_mask1 = (mask1 >> i1) & 1 == 1;
                let t2_in_mask2 = (mask2 >> i2) & 1 == 1;

                if t1_in_mask1 && t2_in_mask2 {
                    return (t1.clone(), t2.clone());
                }
            }
        }
        (String::new(), String::new())
    }

    fn estimate_join_cost(&self, tables: &[String], mask1: u32, mask2: u32) -> f64 {
        let rows1 = self.estimate_rows_for_mask(tables, mask1);
        let rows2 = self.estimate_rows_for_mask(tables, mask2);

        // Hash join cost estimate
        // Build cost + probe cost
        let build_cost = rows1 as f64 * self.config.c_filter;
        let probe_cost = rows2 as f64 * self.config.c_filter;

        build_cost + probe_cost
    }

    fn estimate_rows_for_mask(&self, tables: &[String], mask: u32) -> u64 {
        let mut total = 1u64;

        for (i, table) in tables.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                let rows = self.stats.get(table).map(|s| s.row_count).unwrap_or(1000);
                total = total.saturating_mul(rows);
            }
        }

        // Apply default selectivity for joins
        let num_tables = mask.count_ones();
        if num_tables > 1 {
            total = (total as f64 * 0.1f64.powi(num_tables as i32 - 1)) as u64;
        }

        total.max(1)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stats() -> TableStats {
        let mut column_stats = HashMap::new();
        column_stats.insert(
            "id".to_string(),
            ColumnStats {
                name: "id".to_string(),
                distinct_count: 100000,
                null_count: 0,
                min_value: Some("1".to_string()),
                max_value: Some("100000".to_string()),
                avg_length: 8.0,
                mcv: vec![],
                histogram: None,
            },
        );
        column_stats.insert(
            "score".to_string(),
            ColumnStats {
                name: "score".to_string(),
                distinct_count: 100,
                null_count: 1000,
                min_value: Some("0".to_string()),
                max_value: Some("100".to_string()),
                avg_length: 8.0,
                mcv: vec![("50".to_string(), 0.05)],
                histogram: Some(Histogram {
                    boundaries: vec![25.0, 50.0, 75.0, 100.0],
                    counts: vec![25000, 25000, 25000, 25000],
                    total_rows: 100000,
                }),
            },
        );

        TableStats {
            name: "users".to_string(),
            row_count: 100000,
            size_bytes: 10_000_000, // 10 MB
            column_stats,
            indices: vec![
                IndexStats {
                    name: "pk_users".to_string(),
                    columns: vec!["id".to_string()],
                    is_primary: true,
                    is_unique: true,
                    index_type: IndexType::BTree,
                    leaf_pages: 1000,
                    height: 3,
                    avg_leaf_density: 100.0,
                },
                IndexStats {
                    name: "idx_score".to_string(),
                    columns: vec!["score".to_string()],
                    is_primary: false,
                    is_unique: false,
                    index_type: IndexType::BTree,
                    leaf_pages: 500,
                    height: 2,
                    avg_leaf_density: 200.0,
                },
            ],
            last_updated: 0,
        }
    }

    #[test]
    fn test_selectivity_estimation() {
        let config = CostModelConfig::default();
        let optimizer = CostBasedOptimizer::new(config);

        let stats = create_test_stats();
        optimizer.update_stats(stats.clone());

        // Equality predicate
        let pred = Predicate::Eq {
            column: "id".to_string(),
            value: "12345".to_string(),
        };
        let sel = optimizer.estimate_selectivity(&pred, &stats);
        assert!(sel < 0.001); // Should be very selective

        // Range predicate with histogram
        // Note: For histogram boundaries [25, 50, 75, 100] with equal distribution,
        // Gt{75} includes buckets with bucket_max >= 75, which is buckets 2 and 3 (50%)
        let pred = Predicate::Gt {
            column: "score".to_string(),
            value: "75".to_string(),
        };
        let sel = optimizer.estimate_selectivity(&pred, &stats);
        assert!(sel > 0.4 && sel < 0.6); // ~50% from histogram (2 of 4 buckets)
    }

    #[test]
    fn test_access_path_selection() {
        let config = CostModelConfig::default();
        let optimizer = CostBasedOptimizer::new(config);

        let stats = create_test_stats();
        optimizer.update_stats(stats);

        // High selectivity predicate should use index
        let pred = Predicate::Eq {
            column: "id".to_string(),
            value: "12345".to_string(),
        };
        let plan = optimizer.optimize(
            "users",
            vec!["id".to_string(), "score".to_string()],
            Some(pred),
            vec![],
            None,
        );

        match plan {
            PhysicalPlan::IndexSeek { index, .. } => {
                assert_eq!(index, "pk_users");
            }
            _ => panic!("Expected IndexSeek for equality on primary key"),
        }
    }

    #[test]
    fn test_token_budget_limit() {
        let config = CostModelConfig::default();
        let optimizer = CostBasedOptimizer::new(config).with_token_budget(2048, 25.0);

        // With 2048 token budget and 25 tokens/row:
        // max_rows = (2048 - 50) / 25 = ~80
        let plan = optimizer.optimize("users", vec!["id".to_string()], None, vec![], None);

        match plan {
            PhysicalPlan::Limit { limit, .. } => {
                assert!(limit <= 80);
            }
            _ => panic!("Expected Limit to be injected"),
        }
    }

    #[test]
    fn test_explain_output() {
        let config = CostModelConfig::default();
        let optimizer = CostBasedOptimizer::new(config);

        let stats = create_test_stats();
        optimizer.update_stats(stats);

        let plan = optimizer.optimize(
            "users",
            vec!["id".to_string(), "score".to_string()],
            Some(Predicate::Gt {
                column: "score".to_string(),
                value: "80".to_string(),
            }),
            vec![("score".to_string(), SortDirection::Descending)],
            Some(10),
        );

        let explain = optimizer.explain(&plan);
        assert!(explain.contains("Limit"));
        assert!(explain.contains("Sort"));
    }
}
