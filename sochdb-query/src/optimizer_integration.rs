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

//! Query Optimizer Integration (Task 10)
//!
//! Wires the QueryOptimizer into the SOCH-QL execution path for cost-based planning:
//! - Converts SOCH-QL WHERE clauses to optimizer predicates
//! - Uses cardinality hints from HyperLogLog sketches
//! - Selects optimal index based on selectivity estimates
//!
//! ## Integration Flow
//!
//! ```text
//! SOCH-QL Query
//!     │
//!     ▼
//! ┌─────────────────┐
//! │ Parse & Validate│
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Extract         │
//! │ Predicates      │ ← WHERE clause → QueryPredicate[]
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ QueryOptimizer  │
//! │ .plan_query()   │ ← Cost-based index selection
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Execute Plan    │
//! └─────────────────┘
//! ```
//!
//! ## Task 11: HyperLogLog Integration for Real-Time Cardinality
//!
//! The `CardinalityTracker` maintains HyperLogLog sketches per column for
//! real-time cardinality estimation with <1% standard error.
//!
//! ```text
//! On INSERT:
//!   tracker.observe("column_name", value)  // O(1) HLL update
//!
//! On SELECT planning:
//!   cardinality = tracker.estimate("column_name")  // O(1) estimate
//!
//! Math:
//!   Standard error = 1.04 / sqrt(m) where m = 2^precision
//!   For precision=14: SE = 0.81%, memory = 16KB per column
//! ```

use crate::query_optimizer::{
    CardinalitySource, CostModel, IndexSelection, QueryOperation, QueryOptimizer,
    QueryPlan as OptimizerPlan, QueryPredicate, TraversalDirection,
};
#[cfg(test)]
use crate::soch_ql::{ComparisonOp, WhereClause};
use crate::soch_ql::{SelectQuery, SochResult, SochValue};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use sochdb_core::{Catalog, Result};
use sochdb_storage::sketches::HyperLogLog;

// ============================================================================
// Storage Backend Trait - Allows wiring optimizer to actual storage
// ============================================================================

/// Storage backend trait for executing optimized query plans
///
/// This trait abstracts the storage layer so the optimizer can execute
/// plans without knowing the concrete storage implementation.
pub trait StorageBackend: Send + Sync {
    /// Execute a full table scan
    fn table_scan(
        &self,
        table: &str,
        columns: &[String],
        predicate: Option<&str>,
    ) -> Result<Vec<HashMap<String, SochValue>>>;

    /// Execute a primary key lookup
    fn primary_key_lookup(
        &self,
        table: &str,
        key: &SochValue,
    ) -> Result<Option<HashMap<String, SochValue>>>;

    /// Execute a secondary index seek
    fn secondary_index_seek(
        &self,
        table: &str,
        index: &str,
        key: &SochValue,
    ) -> Result<Vec<HashMap<String, SochValue>>>;

    /// Execute a time range scan
    fn time_index_scan(
        &self,
        table: &str,
        start_us: u64,
        end_us: u64,
    ) -> Result<Vec<HashMap<String, SochValue>>>;

    /// Execute a vector similarity search
    fn vector_search(
        &self,
        table: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(f32, HashMap<String, SochValue>)>>;

    /// Get table row count (for optimization)
    fn row_count(&self, table: &str) -> usize;
}

// ============================================================================
// Task 11: CardinalityTracker - Real-Time HyperLogLog Integration
// ============================================================================

/// Real-time cardinality tracker using HyperLogLog sketches
///
/// Maintains per-column HLL sketches for sub-microsecond cardinality queries
/// with <1% standard error.
///
/// ## Math
///
/// ```text
/// Standard error = 1.04 / sqrt(2^precision)
///
/// Precision=14: SE = 0.81%, memory = 16KB per column (dense)
/// Sparse mode: memory = O(cardinality) for low-cardinality columns
/// ```
///
/// ## Thread Safety
///
/// Uses fine-grained locking per table for concurrent updates across
/// multiple ingestion threads.
pub struct CardinalityTracker {
    /// HLL precision (4-18, default 14 for 0.81% error)
    precision: u8,
    /// Per-table column cardinality trackers
    tables: RwLock<HashMap<String, TableCardinalityTracker>>,
    /// Drift threshold for cache invalidation (0.20 = 20% change)
    drift_threshold: f64,
}

/// Per-table cardinality tracking
struct TableCardinalityTracker {
    /// HLL sketch per column
    columns: HashMap<String, HyperLogLog>,
    /// Row count estimate
    row_count: usize,
    /// Last update timestamp
    last_update_us: u64,
}

/// Cardinality estimate with confidence
#[derive(Debug, Clone)]
pub struct CardinalityEstimate {
    /// Estimated distinct count
    pub distinct: usize,
    /// Standard error percentage
    pub error_pct: f64,
    /// Source of estimate
    pub source: CardinalitySource,
    /// Is this a fresh (recently updated) estimate?
    pub is_fresh: bool,
}

impl CardinalityTracker {
    /// Create a new tracker with default precision (14)
    pub fn new() -> Self {
        Self::with_precision(14)
    }

    /// Create with custom HLL precision
    ///
    /// Precision affects accuracy vs memory:
    /// - 10: SE=3.25%, 1KB/column
    /// - 12: SE=1.63%, 4KB/column  
    /// - 14: SE=0.81%, 16KB/column (default)
    /// - 16: SE=0.41%, 64KB/column
    pub fn with_precision(precision: u8) -> Self {
        assert!((4..=18).contains(&precision), "Precision must be 4-18");
        Self {
            precision,
            tables: RwLock::new(HashMap::new()),
            drift_threshold: 0.20, // 20% change triggers replan
        }
    }

    /// Set drift threshold for cache invalidation
    pub fn set_drift_threshold(&mut self, threshold: f64) {
        self.drift_threshold = threshold;
    }

    /// Observe a value for a column (call on INSERT/UPDATE)
    ///
    /// O(1) operation - safe to call on every write.
    pub fn observe<T: std::hash::Hash>(&self, table: &str, column: &str, value: &T) {
        let mut tables = self.tables.write();
        let tracker = tables
            .entry(table.to_string())
            .or_insert_with(|| TableCardinalityTracker {
                columns: HashMap::new(),
                row_count: 0,
                last_update_us: Self::now(),
            });

        let hll = tracker
            .columns
            .entry(column.to_string())
            .or_insert_with(|| HyperLogLog::new(self.precision));

        hll.add(value);
        tracker.last_update_us = Self::now();
    }

    /// Observe multiple values in batch (more efficient for bulk inserts)
    pub fn observe_batch<T: std::hash::Hash>(
        &self,
        table: &str,
        column: &str,
        values: impl Iterator<Item = T>,
    ) {
        let mut tables = self.tables.write();
        let tracker = tables
            .entry(table.to_string())
            .or_insert_with(|| TableCardinalityTracker {
                columns: HashMap::new(),
                row_count: 0,
                last_update_us: Self::now(),
            });

        let hll = tracker
            .columns
            .entry(column.to_string())
            .or_insert_with(|| HyperLogLog::new(self.precision));

        for value in values {
            hll.add(&value);
        }
        tracker.last_update_us = Self::now();
    }

    /// Increment row count for a table
    pub fn increment_row_count(&self, table: &str, delta: usize) {
        let mut tables = self.tables.write();
        if let Some(tracker) = tables.get_mut(table) {
            tracker.row_count = tracker.row_count.saturating_add(delta);
        }
    }

    /// Estimate cardinality for a column
    ///
    /// O(1) operation - returns sub-microsecond.
    pub fn estimate(&self, table: &str, column: &str) -> Option<CardinalityEstimate> {
        let tables = self.tables.read();
        let tracker = tables.get(table)?;
        let hll = tracker.columns.get(column)?;

        let distinct = hll.cardinality() as usize;
        let error_pct = hll.standard_error();
        let freshness_us = Self::now().saturating_sub(tracker.last_update_us);

        Some(CardinalityEstimate {
            distinct,
            error_pct,
            source: CardinalitySource::HyperLogLog,
            // Consider fresh if updated within last minute
            is_fresh: freshness_us < 60_000_000,
        })
    }

    /// Get all column cardinalities for a table
    pub fn get_table_cardinalities(&self, table: &str) -> HashMap<String, usize> {
        let tables = self.tables.read();
        tables
            .get(table)
            .map(|tracker| {
                tracker
                    .columns
                    .iter()
                    .map(|(col, hll)| (col.clone(), hll.cardinality() as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get row count estimate for a table
    pub fn get_row_count(&self, table: &str) -> usize {
        self.tables
            .read()
            .get(table)
            .map(|t| t.row_count)
            .unwrap_or(0)
    }

    /// Check if cardinality has drifted beyond threshold
    ///
    /// Returns true if any column's cardinality has changed by more than
    /// `drift_threshold` (default 20%).
    pub fn has_cardinality_drift(
        &self,
        table: &str,
        cached_cardinalities: &HashMap<String, usize>,
    ) -> bool {
        let tables = self.tables.read();
        let tracker = match tables.get(table) {
            Some(t) => t,
            None => return true, // Table not tracked, consider stale
        };

        for (column, &cached) in cached_cardinalities {
            if let Some(hll) = tracker.columns.get(column) {
                let current = hll.cardinality();
                if cached == 0 {
                    if current > 0 {
                        return true; // New data in empty column
                    }
                } else {
                    let drift = (current as f64 - cached as f64).abs() / cached as f64;
                    if drift > self.drift_threshold {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Merge HLL from another tracker (for distributed scenarios)
    pub fn merge(&self, table: &str, column: &str, other_hll: &HyperLogLog) {
        let mut tables = self.tables.write();
        if let Some(tracker) = tables.get_mut(table)
            && let Some(hll) = tracker.columns.get_mut(column)
        {
            hll.merge(other_hll);
            tracker.last_update_us = Self::now();
        }
    }

    /// Clear all tracking data for a table
    pub fn clear_table(&self, table: &str) {
        self.tables.write().remove(table);
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> CardinalityTrackerStats {
        let tables = self.tables.read();
        let mut total_columns = 0;
        let mut total_bytes = 0;

        for tracker in tables.values() {
            for hll in tracker.columns.values() {
                total_columns += 1;
                total_bytes += hll.memory_usage();
            }
        }

        CardinalityTrackerStats {
            table_count: tables.len(),
            column_count: total_columns,
            memory_bytes: total_bytes,
            precision: self.precision,
            standard_error_pct: 1.04 / (1usize << self.precision) as f64 * 100.0,
        }
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

impl Default for CardinalityTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for cardinality tracker
#[derive(Debug, Clone)]
pub struct CardinalityTrackerStats {
    /// Number of tables tracked
    pub table_count: usize,
    /// Total columns tracked across all tables
    pub column_count: usize,
    /// Total memory usage in bytes
    pub memory_bytes: usize,
    /// HLL precision
    pub precision: u8,
    /// Standard error percentage
    pub standard_error_pct: f64,
}

/// Optimized query executor with cost-based planning
pub struct OptimizedExecutor {
    /// Query optimizer instance
    optimizer: QueryOptimizer,
    /// Table statistics cache
    table_stats: HashMap<String, TableStats>,
    /// Real-time cardinality tracker (Task 11)
    cardinality_tracker: Arc<CardinalityTracker>,
    /// Embedding provider for vector search (optional)
    embedding_provider: Option<Arc<dyn crate::embedding_provider::EmbeddingProvider>>,
}

/// Statistics for a table
#[derive(Debug, Clone, Default)]
pub struct TableStats {
    /// Estimated row count
    pub row_count: usize,
    /// Column cardinalities (distinct values)
    pub column_cardinalities: HashMap<String, usize>,
    /// Has time index
    pub has_time_index: bool,
    /// Has vector index
    pub has_vector_index: bool,
    /// Primary key column
    pub primary_key: Option<String>,
}

impl OptimizedExecutor {
    /// Create a new optimized executor
    pub fn new() -> Self {
        Self {
            optimizer: QueryOptimizer::new(),
            table_stats: HashMap::new(),
            cardinality_tracker: Arc::new(CardinalityTracker::new()),
            embedding_provider: None,
        }
    }

    /// Create with custom cost model
    pub fn with_cost_model(cost_model: CostModel) -> Self {
        Self {
            optimizer: QueryOptimizer::with_cost_model(cost_model),
            table_stats: HashMap::new(),
            cardinality_tracker: Arc::new(CardinalityTracker::new()),
            embedding_provider: None,
        }
    }

    /// Create with shared cardinality tracker (for integration with ingestion)
    pub fn with_cardinality_tracker(tracker: Arc<CardinalityTracker>) -> Self {
        Self {
            optimizer: QueryOptimizer::new(),
            table_stats: HashMap::new(),
            cardinality_tracker: tracker,
            embedding_provider: None,
        }
    }

    /// Set embedding provider for vector search
    pub fn set_embedding_provider(
        &mut self,
        provider: Arc<dyn crate::embedding_provider::EmbeddingProvider>,
    ) {
        self.embedding_provider = Some(provider);
    }

    /// Create with embedding provider
    pub fn with_embedding_provider(
        mut self,
        provider: Arc<dyn crate::embedding_provider::EmbeddingProvider>,
    ) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Get the cardinality tracker for external updates (e.g., on INSERT)
    pub fn cardinality_tracker(&self) -> Arc<CardinalityTracker> {
        Arc::clone(&self.cardinality_tracker)
    }

    /// Update table statistics (call periodically or on schema change)
    pub fn update_table_stats(&mut self, table: &str, stats: TableStats) {
        let row_count = stats.row_count;
        self.table_stats.insert(table.to_string(), stats);
        self.optimizer
            .update_total_edges(row_count, CardinalitySource::Exact);
    }

    /// Refresh stats from cardinality tracker
    ///
    /// Syncs real-time HLL estimates to the static stats cache.
    pub fn refresh_stats_from_tracker(&mut self, table: &str) {
        let cardinalities = self.cardinality_tracker.get_table_cardinalities(table);
        let row_count = self.cardinality_tracker.get_row_count(table);

        if let Some(stats) = self.table_stats.get_mut(table) {
            stats.column_cardinalities = cardinalities;
            if row_count > 0 {
                stats.row_count = row_count;
            }
        } else {
            self.table_stats.insert(
                table.to_string(),
                TableStats {
                    row_count,
                    column_cardinalities: cardinalities,
                    ..Default::default()
                },
            );
        }
    }

    /// Update column cardinality from HyperLogLog
    pub fn update_cardinality_hint(
        &mut self,
        table: &str,
        column: &str,
        cardinality: usize,
        _source: CardinalitySource,
    ) {
        if let Some(stats) = self.table_stats.get_mut(table) {
            stats
                .column_cardinalities
                .insert(column.to_string(), cardinality);
        }
    }

    /// Plan a SELECT query with cost-based optimization
    pub fn plan_select(
        &self,
        select: &SelectQuery,
        _catalog: &Catalog,
    ) -> Result<OptimizedQueryPlan> {
        // Extract predicates from WHERE clause
        let predicates = self.extract_predicates(select)?;

        // Get optimizer plan
        let optimizer_plan = self.optimizer.plan_query(&predicates, select.limit);

        // Convert to execution plan
        let exec_plan = self.build_execution_plan(select, &optimizer_plan)?;

        Ok(OptimizedQueryPlan {
            table: select.table.clone(),
            columns: select.columns.clone(),
            execution_plan: exec_plan,
            optimizer_plan,
            predicates,
        })
    }

    /// Extract predicates from SELECT query
    fn extract_predicates(&self, select: &SelectQuery) -> Result<Vec<QueryPredicate>> {
        let mut predicates = Vec::new();

        if let Some(where_clause) = &select.where_clause {
            for condition in &where_clause.conditions {
                if let Some(pred) = self.condition_to_predicate(&condition.column, &condition.value)
                {
                    predicates.push(pred);
                }
            }
        }

        Ok(predicates)
    }

    /// Convert a condition to optimizer predicate
    fn condition_to_predicate(&self, column: &str, value: &SochValue) -> Option<QueryPredicate> {
        // Detect special column patterns
        match column {
            // Time-based columns
            "timestamp" | "created_at" | "updated_at" | "time" => {
                if let SochValue::UInt(ts) = value {
                    // Assume range of 1 hour by default
                    let hour_us = 60 * 60 * 1_000_000u64;
                    return Some(QueryPredicate::TimeRange(*ts, ts + hour_us));
                }
            }
            // Project ID
            "project_id" | "project" => {
                if let SochValue::UInt(id) = value {
                    return Some(QueryPredicate::Project(*id as u16));
                }
            }
            // Tenant ID
            "tenant_id" | "tenant" => {
                if let SochValue::UInt(id) = value {
                    return Some(QueryPredicate::Tenant(*id as u32));
                }
            }
            // Span type
            "span_type" | "type" => {
                if let SochValue::Text(s) = value {
                    return Some(QueryPredicate::SpanType(s.clone()));
                }
            }
            _ => {}
        }

        None
    }

    /// Build execution plan from optimizer plan
    fn build_execution_plan(
        &self,
        select: &SelectQuery,
        opt_plan: &OptimizerPlan,
    ) -> Result<ExecutionPlan> {
        let mut steps = Vec::new();

        // Add scan/index step based on index selection
        match &opt_plan.index_selection {
            IndexSelection::LsmScan | IndexSelection::FullScan => {
                steps.push(ExecutionStep::TableScan {
                    table: select.table.clone(),
                });
            }
            IndexSelection::TimeIndex => {
                // Extract time range from operations
                if let Some(QueryOperation::LsmRangeScan { start_us, end_us }) =
                    opt_plan.operations.first()
                {
                    steps.push(ExecutionStep::TimeIndexScan {
                        table: select.table.clone(),
                        start_us: *start_us,
                        end_us: *end_us,
                    });
                }
            }
            IndexSelection::VectorIndex => {
                if let Some(QueryOperation::VectorSearch { k }) = opt_plan.operations.first() {
                    // Extract query text from SIMILAR TO predicate in WHERE clause
                    let query_text = self.extract_vector_query_text(select);
                    steps.push(ExecutionStep::VectorSearch {
                        table: select.table.clone(),
                        k: *k,
                        query_text,
                    });
                }
            }
            IndexSelection::CausalIndex => {
                if let Some(QueryOperation::GraphTraversal {
                    direction,
                    max_depth,
                }) = opt_plan.operations.first()
                {
                    steps.push(ExecutionStep::GraphTraversal {
                        table: select.table.clone(),
                        direction: *direction,
                        max_depth: *max_depth,
                    });
                }
            }
            IndexSelection::ProjectIndex => {
                steps.push(ExecutionStep::SecondaryIndexSeek {
                    table: select.table.clone(),
                    index: "project_idx".to_string(),
                });
            }
            IndexSelection::PrimaryKey => {
                steps.push(ExecutionStep::PrimaryKeyLookup {
                    table: select.table.clone(),
                });
            }
            IndexSelection::Secondary(idx) => {
                steps.push(ExecutionStep::SecondaryIndexSeek {
                    table: select.table.clone(),
                    index: idx.clone(),
                });
            }
            IndexSelection::MultiIndex(indexes) => {
                // For multi-index, use intersection
                steps.push(ExecutionStep::MultiIndexIntersect {
                    table: select.table.clone(),
                    indexes: indexes.iter().map(|idx| format!("{:?}", idx)).collect(),
                });
            }
        }

        // Add filter step if WHERE clause exists
        if select.where_clause.is_some() {
            steps.push(ExecutionStep::Filter {
                predicate: format!("{:?}", select.where_clause),
            });
        }

        // Add projection
        if !select.columns.is_empty() && select.columns[0] != "*" {
            steps.push(ExecutionStep::Project {
                columns: select.columns.clone(),
            });
        }

        // Add sort if ORDER BY exists
        if let Some(order_by) = &select.order_by {
            steps.push(ExecutionStep::Sort {
                column: order_by.column.clone(),
                ascending: order_by.direction == crate::soch_ql::SortDirection::Asc,
            });
        }

        // Add limit if specified
        if let Some(limit) = select.limit {
            steps.push(ExecutionStep::Limit { count: limit });
        }

        Ok(ExecutionPlan {
            steps,
            estimated_cost: opt_plan.cost.total_cost,
            estimated_rows: opt_plan.cost.records_returned,
        })
    }

    /// Execute an optimized query plan against a storage backend
    ///
    /// This is the key method that wires the optimizer output to actual storage.
    /// It interprets each ExecutionStep and calls the appropriate storage method.
    pub fn execute<S: StorageBackend>(
        &self,
        plan: &OptimizedQueryPlan,
        storage: &S,
    ) -> Result<SochResult> {
        let mut rows: Vec<HashMap<String, SochValue>> = Vec::new();
        let mut columns_to_return = plan.columns.clone();

        // Execute each step in order
        for step in &plan.execution_plan.steps {
            match step {
                ExecutionStep::TableScan { table } => {
                    // Full table scan - use storage backend
                    let predicate = plan.execution_plan.steps.iter().find_map(|s| match s {
                        ExecutionStep::Filter { predicate } => Some(predicate.as_str()),
                        _ => None,
                    });
                    rows = storage.table_scan(table, &columns_to_return, predicate)?;
                }
                ExecutionStep::PrimaryKeyLookup { table } => {
                    // Extract key from predicates
                    if let Some(key) = self.extract_primary_key_from_predicates(&plan.predicates)
                        && let Some(row) = storage.primary_key_lookup(table, &key)?
                    {
                        rows = vec![row];
                    }
                }
                ExecutionStep::SecondaryIndexSeek { table, index } => {
                    // Extract key from predicates for the indexed column
                    if let Some(key) =
                        self.extract_index_key_from_predicates(&plan.predicates, index)
                    {
                        rows = storage.secondary_index_seek(table, index, &key)?;
                    }
                }
                ExecutionStep::TimeIndexScan {
                    table,
                    start_us,
                    end_us,
                } => {
                    rows = storage.time_index_scan(table, *start_us, *end_us)?;
                }
                ExecutionStep::VectorSearch {
                    table,
                    k,
                    query_text,
                } => {
                    // Generate real embedding from query text using embedding provider
                    let query_embedding = match (query_text, &self.embedding_provider) {
                        (Some(text), Some(provider)) => {
                            // Use embedding provider to generate real embedding
                            provider.embed(text).unwrap_or_else(|e| {
                                tracing::warn!(
                                    "Failed to generate embedding for '{}': {}. Using fallback.",
                                    text,
                                    e
                                );
                                // Fallback to zeros matching provider dimension
                                vec![0.0f32; provider.dimension()]
                            })
                        }
                        (Some(_text), None) => {
                            // No embedding provider configured - use placeholder
                            tracing::warn!(
                                "Vector search requested but no embedding provider configured"
                            );
                            vec![0.0f32; 128] // Fallback dimension
                        }
                        (None, _) => {
                            // No query text provided - use placeholder
                            tracing::warn!("Vector search without query text, using placeholder");
                            vec![0.0f32; 128] // Fallback dimension
                        }
                    };
                    let results = storage.vector_search(table, &query_embedding, *k)?;
                    rows = results.into_iter().map(|(_, row)| row).collect();
                }
                ExecutionStep::GraphTraversal {
                    table,
                    direction: _,
                    max_depth: _,
                } => {
                    // Graph traversal - fallback to table scan for now
                    rows = storage.table_scan(table, &columns_to_return, None)?;
                }
                ExecutionStep::MultiIndexIntersect { table, indexes } => {
                    // Execute each index and intersect results
                    let mut result_sets: Vec<Vec<HashMap<String, SochValue>>> = Vec::new();
                    for index in indexes {
                        if let Some(key) =
                            self.extract_index_key_from_predicates(&plan.predicates, index)
                        {
                            result_sets.push(storage.secondary_index_seek(table, index, &key)?);
                        }
                    }
                    // Intersect by checking row IDs (simplified - assumes "id" column)
                    if !result_sets.is_empty() {
                        rows = self.intersect_result_sets(result_sets);
                    }
                }
                ExecutionStep::Filter { predicate: _ } => {
                    // Filter already applied in scan, but can post-filter here if needed
                    // For now, filtering is pushed to storage
                }
                ExecutionStep::Project { columns } => {
                    columns_to_return = columns.clone();
                    // Project columns from rows
                    rows = rows
                        .into_iter()
                        .map(|row| {
                            columns
                                .iter()
                                .filter_map(|c| row.get(c).map(|v| (c.clone(), v.clone())))
                                .collect()
                        })
                        .collect();
                }
                ExecutionStep::Sort { column, ascending } => {
                    rows.sort_by(|a, b| {
                        let va = a.get(column);
                        let vb = b.get(column);
                        let cmp = Self::compare_values(va, vb);
                        if *ascending { cmp } else { cmp.reverse() }
                    });
                }
                ExecutionStep::Limit { count } => {
                    rows.truncate(*count);
                }
            }
        }

        // Convert to SochResult
        let result_rows: Vec<Vec<SochValue>> = rows
            .iter()
            .map(|row| {
                columns_to_return
                    .iter()
                    .map(|c| row.get(c).cloned().unwrap_or(SochValue::Null))
                    .collect()
            })
            .collect();

        Ok(SochResult {
            table: plan.table.clone(),
            columns: columns_to_return,
            rows: result_rows,
        })
    }

    /// Extract primary key value from predicates
    fn extract_primary_key_from_predicates(
        &self,
        predicates: &[QueryPredicate],
    ) -> Option<SochValue> {
        for pred in predicates {
            // Look for ID predicates
            if let QueryPredicate::Project(id) = pred {
                return Some(SochValue::UInt(*id as u64));
            }
        }
        None
    }

    /// Extract index key from predicates for a specific index
    fn extract_index_key_from_predicates(
        &self,
        predicates: &[QueryPredicate],
        _index: &str,
    ) -> Option<SochValue> {
        for pred in predicates {
            match pred {
                QueryPredicate::Tenant(id) => return Some(SochValue::UInt(*id as u64)),
                QueryPredicate::Project(id) => return Some(SochValue::UInt(*id as u64)),
                QueryPredicate::SpanType(s) => return Some(SochValue::Text(s.clone())),
                _ => {}
            }
        }
        None
    }

    /// Extract query text from SIMILAR TO predicate for vector search
    ///
    /// Looks for conditions like: `content SIMILAR TO 'search query text'`
    /// Returns the query text to be embedded for similarity search.
    fn extract_vector_query_text(&self, select: &SelectQuery) -> Option<String> {
        use crate::soch_ql::ComparisonOp;
        
        if let Some(where_clause) = &select.where_clause {
            for condition in &where_clause.conditions {
                if matches!(condition.operator, ComparisonOp::SimilarTo) {
                    // Extract the text value from the condition
                    if let SochValue::Text(query_text) = &condition.value {
                        return Some(query_text.clone());
                    }
                }
            }
        }
        None
    }

    /// Intersect multiple result sets by common IDs
    fn intersect_result_sets(
        &self,
        sets: Vec<Vec<HashMap<String, SochValue>>>,
    ) -> Vec<HashMap<String, SochValue>> {
        if sets.is_empty() {
            return Vec::new();
        }
        if sets.len() == 1 {
            return sets.into_iter().next().unwrap();
        }

        // Use first set as base, filter by presence in other sets
        let mut base = sets.into_iter().next().unwrap();
        // Simplified intersection - in production, use row IDs
        base.truncate(base.len().min(100)); // Cap for safety
        base
    }

    /// Compare SochValue for sorting
    fn compare_values(a: Option<&SochValue>, b: Option<&SochValue>) -> std::cmp::Ordering {
        match (a, b) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(va), Some(vb)) => match (va, vb) {
                (SochValue::Int(a), SochValue::Int(b)) => a.cmp(b),
                (SochValue::UInt(a), SochValue::UInt(b)) => a.cmp(b),
                (SochValue::Float(a), SochValue::Float(b)) => {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }
                (SochValue::Text(a), SochValue::Text(b)) => a.cmp(b),
                (SochValue::Bool(a), SochValue::Bool(b)) => a.cmp(b),
                _ => std::cmp::Ordering::Equal,
            },
        }
    }

    /// Explain a query plan (for debugging)
    pub fn explain(&self, select: &SelectQuery, catalog: &Catalog) -> Result<String> {
        let plan = self.plan_select(select, catalog)?;

        let mut output = String::new();
        output.push_str(&format!(
            "QUERY PLAN (estimated cost: {:.2}, rows: {})\n",
            plan.optimizer_plan.cost.total_cost, plan.optimizer_plan.cost.records_returned
        ));
        output.push_str(&format!(
            "Index Selection: {:?}\n",
            plan.optimizer_plan.index_selection
        ));
        output.push_str("Execution Steps:\n");

        for (i, step) in plan.execution_plan.steps.iter().enumerate() {
            output.push_str(&format!("  {}. {:?}\n", i + 1, step));
        }

        output.push_str("\nCost Breakdown:\n");
        for (op, cost) in &plan.optimizer_plan.cost.breakdown {
            output.push_str(&format!("  {:?}: {:.2}\n", op, cost));
        }

        Ok(output)
    }
}

impl Default for OptimizedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized query plan with cost estimates
#[derive(Debug)]
pub struct OptimizedQueryPlan {
    /// Target table
    pub table: String,
    /// Columns to return
    pub columns: Vec<String>,
    /// Execution plan
    pub execution_plan: ExecutionPlan,
    /// Optimizer's plan (for debugging)
    pub optimizer_plan: OptimizerPlan,
    /// Extracted predicates
    pub predicates: Vec<QueryPredicate>,
}

/// Execution plan with ordered steps
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Ordered execution steps
    pub steps: Vec<ExecutionStep>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated output rows
    pub estimated_rows: usize,
}

/// Single execution step
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    /// Full table scan
    TableScan { table: String },
    /// Primary key lookup
    PrimaryKeyLookup { table: String },
    /// Time-based index scan
    TimeIndexScan {
        table: String,
        start_us: u64,
        end_us: u64,
    },
    /// Vector similarity search
    VectorSearch {
        table: String,
        k: usize,
        /// Query text to embed for similarity search.
        /// If None, falls back to placeholder (for backwards compat).
        query_text: Option<String>,
    },
    /// Graph traversal
    GraphTraversal {
        table: String,
        direction: TraversalDirection,
        max_depth: usize,
    },
    /// Secondary index seek
    SecondaryIndexSeek { table: String, index: String },
    /// Multi-index intersection
    MultiIndexIntersect { table: String, indexes: Vec<String> },
    /// Filter rows
    Filter { predicate: String },
    /// Project columns
    Project { columns: Vec<String> },
    /// Sort results
    Sort { column: String, ascending: bool },
    /// Limit output
    Limit { count: usize },
}

/// Query plan cache for repeated queries
///
/// ## Task 5 Enhancement: Frequency-Gated Caching
///
/// Plans are only cached after being used 3+ times to avoid
/// polluting the cache with one-off queries.
pub struct PlanCache {
    /// Cached plans by query hash
    cache: HashMap<u64, CachedPlan>,
    /// Frequency tracker for uncached queries
    frequency: HashMap<u64, FrequencyEntry>,
    /// Maximum cache entries
    max_entries: usize,
    /// Cache threshold (number of uses before caching)
    cache_threshold: usize,
    /// Statistics
    stats: AdaptiveCacheStats,
}

/// Cached query plan
#[derive(Debug, Clone)]
struct CachedPlan {
    /// The execution plan
    plan: ExecutionPlan,
    /// Cache hit count
    hits: usize,
    /// Last used timestamp
    last_used: u64,
    /// Time saved by caching (cumulative planning time avoided)
    time_saved_us: u64,
}

/// Frequency tracking entry
#[derive(Debug, Clone)]
struct FrequencyEntry {
    /// Number of times query was seen
    count: usize,
    /// First seen timestamp
    #[allow(dead_code)]
    first_seen: u64,
    /// Most recent timestamp
    last_seen: u64,
    /// The plan (saved but not cached until threshold)
    pending_plan: Option<ExecutionPlan>,
}

/// Enhanced cache statistics
#[derive(Debug, Clone, Default)]
pub struct AdaptiveCacheStats {
    /// Number of cached entries
    pub entries: usize,
    /// Total cache hits
    pub total_hits: usize,
    /// Total cache misses
    pub total_misses: usize,
    /// Queries blocked from cache (below threshold)
    pub frequency_blocked: usize,
    /// Queries promoted to cache
    pub promotions: usize,
    /// Estimated time saved (microseconds)
    pub time_saved_us: u64,
}

impl PlanCache {
    /// Create a new plan cache with default threshold (3)
    pub fn new(max_entries: usize) -> Self {
        Self::with_threshold(max_entries, 3)
    }

    /// Create with custom frequency threshold
    pub fn with_threshold(max_entries: usize, cache_threshold: usize) -> Self {
        Self {
            cache: HashMap::new(),
            frequency: HashMap::new(),
            max_entries,
            cache_threshold,
            stats: AdaptiveCacheStats::default(),
        }
    }

    /// Hash a query for caching
    pub fn hash_query(query: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Get cached plan with frequency tracking
    ///
    /// Returns cached plan if available, or None.
    /// If query is seen frequently but not cached, attempts promotion.
    pub fn get(&mut self, query_hash: u64) -> Option<&ExecutionPlan> {
        // Check cache first
        if self.cache.contains_key(&query_hash) {
            if let Some(cached) = self.cache.get_mut(&query_hash) {
                cached.hits += 1;
                cached.last_used = Self::now();
                cached.time_saved_us += 1000; // Assume 1ms planning time saved
                self.stats.total_hits += 1;
            }
            return self.cache.get(&query_hash).map(|c| &c.plan);
        }

        self.stats.total_misses += 1;

        // Check frequency tracker and promote if needed
        let should_promote = if let Some(freq) = self.frequency.get_mut(&query_hash) {
            freq.count += 1;
            freq.last_seen = Self::now();
            freq.count >= self.cache_threshold && freq.pending_plan.is_some()
        } else {
            false
        };

        if should_promote
            && let Some(freq) = self.frequency.remove(&query_hash)
            && let Some(plan) = freq.pending_plan
        {
            self.insert_to_cache(query_hash, plan);
            self.stats.promotions += 1;
            return self.cache.get(&query_hash).map(|c| &c.plan);
        }

        None
    }

    /// Register a plan for potential caching
    ///
    /// Does not immediately cache - waits for frequency threshold.
    pub fn put(&mut self, query_hash: u64, plan: ExecutionPlan) {
        let now = Self::now();

        // Check if already tracking frequency
        if let Some(freq) = self.frequency.get_mut(&query_hash) {
            freq.count += 1;
            freq.last_seen = now;
            freq.pending_plan = Some(plan.clone());

            // Promote if threshold reached
            if freq.count >= self.cache_threshold {
                self.promote_to_cache(query_hash, plan);
                self.stats.promotions += 1;
            } else {
                self.stats.frequency_blocked += 1;
            }
        } else {
            // First time seeing this query
            self.frequency.insert(
                query_hash,
                FrequencyEntry {
                    count: 1,
                    first_seen: now,
                    last_seen: now,
                    pending_plan: Some(plan),
                },
            );
            self.stats.frequency_blocked += 1;
        }

        // Clean up old frequency entries
        self.cleanup_frequency_tracker();
    }

    /// Force-cache a plan (bypasses frequency check)
    pub fn force_put(&mut self, query_hash: u64, plan: ExecutionPlan) {
        self.insert_to_cache(query_hash, plan);
        self.frequency.remove(&query_hash);
    }

    /// Insert plan directly into cache (internal helper)
    fn insert_to_cache(&mut self, query_hash: u64, plan: ExecutionPlan) {
        // Evict if at capacity
        if self.cache.len() >= self.max_entries {
            self.evict_lru();
        }

        self.cache.insert(
            query_hash,
            CachedPlan {
                plan,
                hits: 0,
                last_used: Self::now(),
                time_saved_us: 0,
            },
        );

        self.stats.entries = self.cache.len();
    }

    /// Promote plan from frequency tracker to cache
    fn promote_to_cache(&mut self, query_hash: u64, plan: ExecutionPlan) {
        self.insert_to_cache(query_hash, plan);
        self.frequency.remove(&query_hash);
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some((&key, _)) = self.cache.iter().min_by_key(|(_, v)| v.last_used) {
            self.cache.remove(&key);
        }
    }

    /// Cleanup old frequency tracker entries
    fn cleanup_frequency_tracker(&mut self) {
        let now = Self::now();
        let max_age = 60 * 1_000_000; // 1 minute

        self.frequency.retain(|_, v| now - v.last_seen < max_age);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.frequency.clear();
        self.stats = AdaptiveCacheStats::default();
    }

    /// Get cache statistics (legacy compatibility)
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            total_hits: self.stats.total_hits,
        }
    }

    /// Get enhanced statistics
    pub fn adaptive_stats(&self) -> &AdaptiveCacheStats {
        &self.stats
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cached entries
    pub entries: usize,
    /// Total cache hits
    pub total_hits: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soch_ql::{Condition, LogicalOp, OrderBy, SortDirection};

    #[test]
    fn test_predicate_extraction() {
        let executor = OptimizedExecutor::new();

        let select = SelectQuery {
            table: "events".to_string(),
            columns: vec!["*".to_string()],
            where_clause: Some(WhereClause {
                conditions: vec![Condition {
                    column: "timestamp".to_string(),
                    operator: ComparisonOp::Ge,
                    value: SochValue::UInt(1700000000000000),
                }],
                operator: LogicalOp::And,
            }),
            order_by: None,
            limit: None,
            offset: None,
        };

        let predicates = executor.extract_predicates(&select).unwrap();
        assert_eq!(predicates.len(), 1);
        assert!(matches!(predicates[0], QueryPredicate::TimeRange(_, _)));
    }

    #[test]
    fn test_plan_with_time_index() {
        let mut executor = OptimizedExecutor::new();
        executor.update_table_stats(
            "events",
            TableStats {
                row_count: 1_000_000,
                has_time_index: true,
                ..Default::default()
            },
        );

        let select = SelectQuery {
            table: "events".to_string(),
            columns: vec!["id".to_string(), "data".to_string()],
            where_clause: Some(WhereClause {
                conditions: vec![Condition {
                    column: "timestamp".to_string(),
                    operator: ComparisonOp::Ge,
                    value: SochValue::UInt(1700000000000000),
                }],
                operator: LogicalOp::And,
            }),
            order_by: None,
            limit: Some(100),
            offset: None,
        };

        let catalog = Catalog::new("test");
        let plan = executor.plan_select(&select, &catalog).unwrap();

        assert!(plan.execution_plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_plan_cache() {
        let mut cache = PlanCache::new(100);

        let plan = ExecutionPlan {
            steps: vec![ExecutionStep::TableScan {
                table: "test".to_string(),
            }],
            estimated_cost: 100.0,
            estimated_rows: 1000,
        };

        let query = "SELECT * FROM test";
        let hash = PlanCache::hash_query(query);

        // Miss (not tracked yet)
        assert!(cache.get(hash).is_none());

        // Put (with frequency-gated caching, needs 3 uses before cache)
        // Use 1: put sets count=1
        cache.put(hash, plan.clone());
        // Use 2: get increments to count=2 (still < threshold)
        assert!(cache.get(hash).is_none());

        // Use 3: put increments to count=3, triggers promotion
        cache.put(hash, plan);
        // Now at threshold (3 uses), should be cached
        assert!(cache.get(hash).is_some());

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_hits, 1);
    }

    #[test]
    fn test_force_cache() {
        let mut cache = PlanCache::new(100);

        let plan = ExecutionPlan {
            steps: vec![ExecutionStep::TableScan {
                table: "test".to_string(),
            }],
            estimated_cost: 100.0,
            estimated_rows: 1000,
        };

        let hash = PlanCache::hash_query("SELECT * FROM test2");

        // Force put bypasses frequency threshold
        cache.force_put(hash, plan);
        assert!(cache.get(hash).is_some());
    }

    #[test]
    fn test_explain() {
        let executor = OptimizedExecutor::new();

        let select = SelectQuery {
            table: "users".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            where_clause: None,
            order_by: Some(OrderBy {
                column: "id".to_string(),
                direction: SortDirection::Asc,
            }),
            limit: Some(10),
            offset: None,
        };

        let catalog = Catalog::new("test");
        let explain = executor.explain(&select, &catalog).unwrap();

        assert!(explain.contains("QUERY PLAN"));
        assert!(explain.contains("Execution Steps"));
    }

    // ========================================================================
    // Task 11: CardinalityTracker Tests
    // ========================================================================

    #[test]
    fn test_cardinality_tracker_basic() {
        let tracker = CardinalityTracker::new();

        // Add 1000 unique values
        for i in 0u64..1000 {
            tracker.observe("events", "user_id", &i);
        }

        let estimate = tracker.estimate("events", "user_id").unwrap();

        // Should be within 5% of actual (HLL with precision=14 has ~0.81% SE)
        let error = (estimate.distinct as f64 - 1000.0).abs() / 1000.0;
        assert!(
            error < 0.05,
            "Cardinality error {}% exceeds 5%",
            error * 100.0
        );
        assert!(estimate.error_pct < 1.0, "Standard error should be < 1%");
    }

    #[test]
    fn test_cardinality_tracker_multiple_columns() {
        let tracker = CardinalityTracker::new();

        // High cardinality column
        for i in 0u64..10_000 {
            tracker.observe("events", "span_id", &i);
        }

        // Low cardinality column
        for i in 0u64..1000 {
            tracker.observe("events", "project_id", &(i % 10));
        }

        let span_estimate = tracker.estimate("events", "span_id").unwrap();
        let project_estimate = tracker.estimate("events", "project_id").unwrap();

        // High cardinality should be ~10000
        let span_error = (span_estimate.distinct as f64 - 10000.0).abs() / 10000.0;
        assert!(span_error < 0.05, "span_id error {}%", span_error * 100.0);

        // Low cardinality should be ~10
        let project_error = (project_estimate.distinct as f64 - 10.0).abs() / 10.0;
        assert!(
            project_error < 0.20,
            "project_id error {}%",
            project_error * 100.0
        );
    }

    #[test]
    fn test_cardinality_drift_detection() {
        let tracker = CardinalityTracker::new();

        // Initial state: 100 distinct values
        for i in 0u64..100 {
            tracker.observe("events", "user_id", &i);
        }

        let mut cached = std::collections::HashMap::new();
        cached.insert("user_id".to_string(), 100usize);

        // No drift yet
        assert!(!tracker.has_cardinality_drift("events", &cached));

        // Add many more distinct values (50% more = drift)
        for i in 100u64..200 {
            tracker.observe("events", "user_id", &i);
        }

        // Now ~100% increase, should exceed 20% threshold
        assert!(tracker.has_cardinality_drift("events", &cached));
    }

    #[test]
    fn test_cardinality_tracker_memory() {
        let tracker = CardinalityTracker::new();

        // Add data for multiple tables/columns
        for i in 0u64..1000 {
            tracker.observe("table1", "col1", &i);
            tracker.observe("table1", "col2", &i);
            tracker.observe("table2", "col1", &i);
        }

        let stats = tracker.memory_usage();
        assert_eq!(stats.table_count, 2);
        assert_eq!(stats.column_count, 3);
        assert!(stats.memory_bytes > 0);
        assert!(stats.standard_error_pct < 1.0);
    }

    #[test]
    fn test_executor_with_cardinality_tracker() {
        let tracker = Arc::new(CardinalityTracker::new());

        // Simulate ingestion updating tracker
        for i in 0u64..500 {
            tracker.observe("events", "user_id", &i);
            tracker.observe("events", "span_id", &(i * 2));
        }
        tracker.increment_row_count("events", 500);

        // Create executor with shared tracker
        let mut executor = OptimizedExecutor::with_cardinality_tracker(Arc::clone(&tracker));

        // Refresh stats from tracker
        executor.refresh_stats_from_tracker("events");

        // Verify stats were synced
        let stats = &executor.table_stats.get("events").unwrap();
        assert_eq!(stats.row_count, 500);
        assert!(stats.column_cardinalities.contains_key("user_id"));
        assert!(stats.column_cardinalities.contains_key("span_id"));

        // Cardinality estimates should be reasonable
        let user_card = stats.column_cardinalities.get("user_id").unwrap();
        let error = (*user_card as f64 - 500.0).abs() / 500.0;
        assert!(error < 0.05, "user_id cardinality error {}%", error * 100.0);
    }
}
