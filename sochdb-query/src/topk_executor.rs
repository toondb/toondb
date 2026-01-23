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

//! ORDER BY + LIMIT Optimization with Streaming Top-K
//!
//! This module fixes a critical semantic bug in the query execution path and
//! provides efficient top-K query support for queue-like access patterns.
//!
//! ## The Bug: Incorrect ORDER BY + LIMIT Semantics
//!
//! The previous implementation applied LIMIT during scan collection, then sorted:
//!
//! ```text
//! ❌ WRONG:
//!    for row in scan:
//!        output.push(row)
//!        if output.len() >= limit:
//!            break
//!    output.sort(order_by)  # BUG: Sorting wrong subset!
//! ```
//!
//! This is NOT semantically equivalent to `ORDER BY ... LIMIT K` unless the
//! scan order matches the requested order (generally false).
//!
//! Example of the bug:
//! ```text
//! Table: [priority=5, priority=1, priority=3, priority=2, priority=4]
//! Query: ORDER BY priority ASC LIMIT 1
//!
//! Wrong (limit-then-sort): Returns priority=5 (first row)
//! Correct: Returns priority=1 (minimum)
//! ```
//!
//! ## The Fix: Three Strategies
//!
//! | Strategy       | Time       | Space | When to Use                     |
//! |----------------|------------|-------|----------------------------------|
//! | IndexPushdown  | O(log N+K) | O(K)  | Ordered index on ORDER BY col   |
//! | StreamingTopK  | O(N log K) | O(K)  | No index, K << N                |
//! | FullSort       | O(N log N) | O(N)  | No index, K ≈ N                 |
//!
//! ## Streaming Top-K Algorithm
//!
//! For ORDER BY col ASC LIMIT K:
//! 1. Maintain a max-heap of size K
//! 2. For each row, if row.col < heap.max, evict max and insert row
//! 3. At end, drain heap in sorted order
//!
//! This gives O(N log K) time and O(K) space, vs O(N log N) and O(N) for full sort.
//!
//! ## Queue Optimization
//!
//! For "get highest priority task" (ORDER BY priority ASC LIMIT 1):
//! - With 10K tasks: ~10K comparisons with O(1) memory
//! - With ordered index: ~14 comparisons (log₂ 10000)

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use sochdb_core::{SochRow, SochValue};

// ============================================================================
// OrderBySpec - Sort Specification
// ============================================================================

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// A single ORDER BY column specification
#[derive(Debug, Clone)]
pub struct OrderByColumn {
    /// Column index or name
    pub column: ColumnRef,
    /// Sort direction
    pub direction: SortDirection,
    /// Null handling (NULLS FIRST or NULLS LAST)
    pub nulls_first: bool,
}

/// Reference to a column
#[derive(Debug, Clone)]
pub enum ColumnRef {
    /// Column by index
    Index(usize),
    /// Column by name
    Name(String),
}

impl ColumnRef {
    /// Resolve to an index given a column name mapping
    pub fn resolve(&self, columns: &[String]) -> Option<usize> {
        match self {
            ColumnRef::Index(i) => Some(*i),
            ColumnRef::Name(name) => columns.iter().position(|c| c == name),
        }
    }
}

/// Full ORDER BY specification
#[derive(Debug, Clone)]
pub struct OrderBySpec {
    /// Columns to sort by (in order of priority)
    pub columns: Vec<OrderByColumn>,
}

impl OrderBySpec {
    /// Create from a single column
    pub fn single(column: ColumnRef, direction: SortDirection) -> Self {
        Self {
            columns: vec![OrderByColumn {
                column,
                direction,
                nulls_first: false,
            }],
        }
    }

    /// Add another column to the sort
    pub fn then_by(mut self, column: ColumnRef, direction: SortDirection) -> Self {
        self.columns.push(OrderByColumn {
            column,
            direction,
            nulls_first: false,
        });
        self
    }

    /// Create a comparator function for rows
    pub fn comparator(&self, column_names: &[String]) -> impl Fn(&SochRow, &SochRow) -> Ordering {
        let resolved: Vec<_> = self.columns
            .iter()
            .filter_map(|col| {
                col.column.resolve(column_names).map(|idx| (idx, col.direction, col.nulls_first))
            })
            .collect();
        
        move |a: &SochRow, b: &SochRow| {
            for &(idx, direction, nulls_first) in &resolved {
                let val_a = a.values.get(idx);
                let val_b = b.values.get(idx);
                
                let ordering = compare_values(val_a, val_b, nulls_first);
                
                if ordering != Ordering::Equal {
                    return match direction {
                        SortDirection::Ascending => ordering,
                        SortDirection::Descending => ordering.reverse(),
                    };
                }
            }
            Ordering::Equal
        }
    }

    /// Check if an index matches this ORDER BY spec
    pub fn matches_index(&self, index_columns: &[(String, SortDirection)]) -> bool {
        if self.columns.len() > index_columns.len() {
            return false;
        }

        self.columns.iter().zip(index_columns.iter()).all(|(col, (idx_col, idx_dir))| {
            match &col.column {
                ColumnRef::Name(name) => name == idx_col && col.direction == *idx_dir,
                ColumnRef::Index(_) => false, // Can't match by index
            }
        })
    }
}

/// Compare two optional SochValues with null handling
fn compare_values(a: Option<&SochValue>, b: Option<&SochValue>, nulls_first: bool) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => if nulls_first { Ordering::Less } else { Ordering::Greater },
        (Some(_), None) => if nulls_first { Ordering::Greater } else { Ordering::Less },
        (Some(SochValue::Null), Some(SochValue::Null)) => Ordering::Equal,
        (Some(SochValue::Null), Some(_)) => if nulls_first { Ordering::Less } else { Ordering::Greater },
        (Some(_), Some(SochValue::Null)) => if nulls_first { Ordering::Greater } else { Ordering::Less },
        (Some(a), Some(b)) => compare_soch_values(a, b),
    }
}

/// Compare two SochValues
fn compare_soch_values(a: &SochValue, b: &SochValue) -> Ordering {
    match (a, b) {
        (SochValue::Int(a), SochValue::Int(b)) => a.cmp(b),
        (SochValue::UInt(a), SochValue::UInt(b)) => a.cmp(b),
        (SochValue::Float(a), SochValue::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (SochValue::Text(a), SochValue::Text(b)) => a.cmp(b),
        (SochValue::Bool(a), SochValue::Bool(b)) => a.cmp(b),
        _ => Ordering::Equal, // Incompatible types compare as equal
    }
}

// ============================================================================
// TopKHeap - Generic Streaming Top-K
// ============================================================================

/// A bounded heap for streaming top-K selection
///
/// This maintains the K smallest (or largest) elements seen so far,
/// without storing all N elements.
///
/// ## Complexity
/// - Push: O(log K) when heap is full, O(log K) insertion
/// - Drain: O(K log K) to produce sorted output
/// - Space: O(K)
pub struct TopKHeap<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    /// The heap (max-heap for smallest K, min-heap for largest K)
    heap: BinaryHeap<ComparableWrapper<T, F>>,
    /// Maximum size
    k: usize,
    /// Comparator (defines desired output order)
    comparator: F,
    /// Whether we want smallest K (true) or largest K (false)
    want_smallest: bool,
}

/// Wrapper to make items comparable via the provided function
struct ComparableWrapper<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    value: T,
    comparator: *const F,
    inverted: bool,
}

// Safety: We ensure the comparator pointer remains valid for the lifetime of the heap
unsafe impl<T: Send, F> Send for ComparableWrapper<T, F> where F: Fn(&T, &T) -> Ordering {}
unsafe impl<T: Sync, F> Sync for ComparableWrapper<T, F> where F: Fn(&T, &T) -> Ordering {}

impl<T, F> PartialEq for ComparableWrapper<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T, F> Eq for ComparableWrapper<T, F> where F: Fn(&T, &T) -> Ordering {}

impl<T, F> PartialOrd for ComparableWrapper<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, F> Ord for ComparableWrapper<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Safety: comparator pointer is valid for heap lifetime
        let cmp = unsafe { &*self.comparator };
        let result = cmp(&self.value, &other.value);
        
        if self.inverted {
            result.reverse()
        } else {
            result
        }
    }
}

impl<T, F> TopKHeap<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    /// Create a new top-K heap
    ///
    /// - `k`: Number of elements to keep
    /// - `comparator`: Defines the desired output order (Less = should come first)
    /// - `want_smallest`: If true, keep the K elements that compare as smallest
    pub fn new(k: usize, comparator: F, want_smallest: bool) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            comparator,
            want_smallest,
        }
    }

    /// Push an element into the heap
    ///
    /// Complexity: O(log K)
    pub fn push(&mut self, value: T) {
        if self.k == 0 {
            return;
        }

        let wrapper = ComparableWrapper {
            value,
            comparator: &self.comparator as *const F,
            // For smallest K, we want a max-heap (so we can evict the largest)
            // For largest K, we want a min-heap (so we can evict the smallest)
            inverted: !self.want_smallest,
        };

        if self.heap.len() < self.k {
            self.heap.push(wrapper);
        } else if let Some(top) = self.heap.peek() {
            // Check if new element should replace the current boundary
            let should_replace = if self.want_smallest {
                // For smallest K: replace if new < current max
                (self.comparator)(&wrapper.value, &top.value) == Ordering::Less
            } else {
                // For largest K: replace if new > current min
                (self.comparator)(&wrapper.value, &top.value) == Ordering::Greater
            };

            if should_replace {
                self.heap.pop();
                self.heap.push(wrapper);
            }
        }
    }

    /// Get the current boundary value
    ///
    /// This is the value that new elements must beat to be included.
    pub fn threshold(&self) -> Option<&T> {
        self.heap.peek().map(|w| &w.value)
    }

    /// Check if the heap is at capacity
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Drain the heap into a sorted vector
    ///
    /// Complexity: O(K log K)
    pub fn into_sorted_vec(self) -> Vec<T> {
        let mut values: Vec<_> = self.heap.into_iter().map(|w| w.value).collect();
        if self.want_smallest {
            values.sort_by(&self.comparator);
        } else {
            // For largest K, sort in descending order
            values.sort_by(|a, b| (&self.comparator)(b, a));
        }
        values
    }

    /// Current number of elements
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ============================================================================
// OrderByLimitExecutor - The Fixed Implementation
// ============================================================================

/// Strategy for ORDER BY + LIMIT execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Use index pushdown (storage provides ordered results)
    IndexPushdown,
    /// Streaming top-K with heap
    StreamingTopK,
    /// Full sort then limit
    FullSort,
}

impl ExecutionStrategy {
    /// Choose the optimal strategy
    pub fn choose(
        has_matching_index: bool,
        estimated_rows: Option<usize>,
        limit: usize,
    ) -> Self {
        // If we have a matching index, always use pushdown
        if has_matching_index {
            return ExecutionStrategy::IndexPushdown;
        }

        // If we don't know the row count, use streaming (safe default)
        let n = match estimated_rows {
            Some(n) if n > 0 => n,
            _ => return ExecutionStrategy::StreamingTopK,
        };

        // Heuristic: streaming is better when K < sqrt(N) or K is "small"
        // Break-even is approximately when K * log(K) < N, but we use simpler heuristic
        let k = limit;
        
        if k <= 100 {
            // Small K: streaming is almost always better
            ExecutionStrategy::StreamingTopK
        } else if (k as f64) < (n as f64).sqrt() {
            // K < sqrt(N): streaming wins
            ExecutionStrategy::StreamingTopK
        } else {
            // Large K relative to N: full sort may be better
            // (avoids heap overhead when keeping most of the data)
            ExecutionStrategy::FullSort
        }
    }

    /// Get estimated complexity description
    pub fn complexity(&self, n: usize, k: usize) -> String {
        match self {
            ExecutionStrategy::IndexPushdown => {
                format!("O(log {} + {}) = O({})", n, k, (n as f64).log2() as usize + k)
            }
            ExecutionStrategy::StreamingTopK => {
                let log_k = (k as f64).log2().max(1.0) as usize;
                format!("O({} * log {}) ≈ O({})", n, k, n * log_k)
            }
            ExecutionStrategy::FullSort => {
                let log_n = (n as f64).log2().max(1.0) as usize;
                format!("O({} * log {}) ≈ O({})", n, n, n * log_n)
            }
        }
    }
}

/// Result statistics from ORDER BY + LIMIT execution
#[derive(Debug, Clone, Default)]
pub struct OrderByLimitStats {
    /// Strategy used
    pub strategy: Option<ExecutionStrategy>,
    /// Input rows processed
    pub input_rows: usize,
    /// Output rows produced
    pub output_rows: usize,
    /// Heap operations performed
    pub heap_operations: usize,
    /// Comparisons performed
    pub comparisons: usize,
    /// Rows skipped by offset
    pub offset_skipped: usize,
}

/// Executor for ORDER BY + LIMIT queries
///
/// This is the CORRECT implementation that ensures semantic equivalence
/// with `ORDER BY ... LIMIT K OFFSET M`.
pub struct OrderByLimitExecutor {
    /// ORDER BY specification
    order_by: OrderBySpec,
    /// LIMIT value
    limit: usize,
    /// OFFSET value
    offset: usize,
    /// Column names for resolution
    column_names: Vec<String>,
    /// Execution strategy
    strategy: ExecutionStrategy,
}

impl OrderByLimitExecutor {
    /// Create a new executor
    pub fn new(
        order_by: OrderBySpec,
        limit: usize,
        offset: usize,
        column_names: Vec<String>,
        has_matching_index: bool,
        estimated_rows: Option<usize>,
    ) -> Self {
        // For OFFSET, we need to fetch limit + offset rows
        let effective_limit = limit.saturating_add(offset);
        let strategy = ExecutionStrategy::choose(has_matching_index, estimated_rows, effective_limit);
        
        Self {
            order_by,
            limit,
            offset,
            column_names,
            strategy,
        }
    }

    /// Get the chosen strategy
    pub fn strategy(&self) -> ExecutionStrategy {
        self.strategy
    }

    /// Execute on an iterator of rows
    ///
    /// This is the main entry point. It:
    /// 1. Applies the chosen strategy to get top (limit + offset) rows
    /// 2. Applies offset
    /// 3. Returns the final result
    pub fn execute<I>(&self, rows: I) -> (Vec<SochRow>, OrderByLimitStats)
    where
        I: Iterator<Item = SochRow>,
    {
        let mut stats = OrderByLimitStats {
            strategy: Some(self.strategy),
            ..Default::default()
        };

        let effective_limit = self.limit.saturating_add(self.offset);
        
        let result = match self.strategy {
            ExecutionStrategy::IndexPushdown => {
                // With index, just take the first limit+offset rows
                // (caller must provide rows in correct order!)
                let collected: Vec<_> = rows.take(effective_limit).collect();
                stats.input_rows = collected.len();
                collected
            }
            ExecutionStrategy::StreamingTopK => {
                self.execute_streaming(rows, effective_limit, &mut stats)
            }
            ExecutionStrategy::FullSort => {
                self.execute_full_sort(rows, effective_limit, &mut stats)
            }
        };

        // Apply offset
        let final_result: Vec<_> = result
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();
        
        stats.offset_skipped = self.offset.min(stats.input_rows);
        stats.output_rows = final_result.len();
        
        (final_result, stats)
    }

    /// Execute using streaming top-K algorithm
    fn execute_streaming<I>(
        &self,
        rows: I,
        k: usize,
        stats: &mut OrderByLimitStats,
    ) -> Vec<SochRow>
    where
        I: Iterator<Item = SochRow>,
    {
        let comparator = self.order_by.comparator(&self.column_names);
        
        // We want the K smallest according to the comparator
        let mut heap = TopKHeap::new(k, comparator, true);
        
        for row in rows {
            stats.input_rows += 1;
            stats.heap_operations += 1;
            heap.push(row);
        }

        heap.into_sorted_vec()
    }

    /// Execute using full sort
    fn execute_full_sort<I>(
        &self,
        rows: I,
        k: usize,
        stats: &mut OrderByLimitStats,
    ) -> Vec<SochRow>
    where
        I: Iterator<Item = SochRow>,
    {
        let comparator = self.order_by.comparator(&self.column_names);
        
        let mut all_rows: Vec<_> = rows.collect();
        stats.input_rows = all_rows.len();
        
        all_rows.sort_by(&comparator);
        
        all_rows.truncate(k);
        all_rows
    }
}

// ============================================================================
// IndexAwareTopK - For Index Pushdown with Partial Match
// ============================================================================

/// Top-K with index awareness
///
/// When the index only partially matches the ORDER BY (e.g., index on col1
/// but ORDER BY col1, col2), we can still use the index for the first column
/// and apply top-K for the rest.
pub struct IndexAwareTopK<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    /// Current batch of items with same index key
    current_batch: Vec<T>,
    /// Best items seen so far
    result: Vec<T>,
    /// Maximum items to keep
    k: usize,
    /// Secondary comparator (for columns not in index)
    secondary_cmp: F,
}

impl<T, F> IndexAwareTopK<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    /// Create new index-aware top-K
    pub fn new(k: usize, secondary_cmp: F) -> Self {
        Self {
            current_batch: Vec::new(),
            result: Vec::with_capacity(k),
            k,
            secondary_cmp,
        }
    }

    /// Process an item from an index-ordered scan
    ///
    /// Items must be provided in index order. When the index key changes,
    /// the previous batch is finalized.
    pub fn push(&mut self, item: T, same_index_key_as_previous: bool) {
        if !same_index_key_as_previous {
            self.finalize_batch();
        }
        
        self.current_batch.push(item);
    }

    /// Finalize the current batch (sort by secondary key)
    fn finalize_batch(&mut self) {
        if self.current_batch.is_empty() {
            return;
        }

        // Sort batch by secondary key
        self.current_batch.sort_by(&self.secondary_cmp);

        // Take as many as we need
        let remaining = self.k.saturating_sub(self.result.len());
        let to_take = remaining.min(self.current_batch.len());
        
        self.result.extend(self.current_batch.drain(..to_take));
        self.current_batch.clear();
    }

    /// Check if we have enough results
    pub fn is_complete(&self) -> bool {
        self.result.len() >= self.k
    }

    /// Drain into final result
    pub fn into_result(mut self) -> Vec<T> {
        self.finalize_batch();
        self.result
    }
}

// ============================================================================
// SingleColumnTopK - Optimized for Single Column
// ============================================================================

/// Optimized top-K for single-column ORDER BY
///
/// Avoids the overhead of multi-column comparison when only one column is involved.
pub struct SingleColumnTopK {
    /// The heap
    heap: BinaryHeap<SingleColEntry>,
    /// K value
    k: usize,
    /// Column index
    col_idx: usize,
    /// Whether ascending order
    ascending: bool,
}

struct SingleColEntry {
    row: SochRow,
    key: OrderableValue,
    ascending: bool,
}

/// Wrapper to make SochValue orderable
#[derive(Clone)]
enum OrderableValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    Bool(bool),
    Null,
}

impl From<&SochValue> for OrderableValue {
    fn from(v: &SochValue) -> Self {
        match v {
            SochValue::Int(i) => OrderableValue::Int(*i),
            SochValue::UInt(u) => OrderableValue::UInt(*u),
            SochValue::Float(f) => OrderableValue::Float(*f),
            SochValue::Text(s) => OrderableValue::Text(s.clone()),
            SochValue::Bool(b) => OrderableValue::Bool(*b),
            SochValue::Null => OrderableValue::Null,
            _ => OrderableValue::Null,
        }
    }
}

impl PartialEq for OrderableValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OrderableValue {}

impl PartialOrd for OrderableValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderableValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (OrderableValue::Null, OrderableValue::Null) => Ordering::Equal,
            (OrderableValue::Null, _) => Ordering::Greater, // NULLS LAST
            (_, OrderableValue::Null) => Ordering::Less,
            (OrderableValue::Int(a), OrderableValue::Int(b)) => a.cmp(b),
            (OrderableValue::UInt(a), OrderableValue::UInt(b)) => a.cmp(b),
            (OrderableValue::Float(a), OrderableValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (OrderableValue::Text(a), OrderableValue::Text(b)) => a.cmp(b),
            (OrderableValue::Bool(a), OrderableValue::Bool(b)) => a.cmp(b),
            _ => Ordering::Equal, // Incompatible types
        }
    }
}

impl PartialEq for SingleColEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for SingleColEntry {}

impl PartialOrd for SingleColEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SingleColEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        let base = self.key.cmp(&other.key);
        
        // For ascending + max-heap, we want to evict the largest,
        // so we don't invert. For descending, we invert.
        if self.ascending {
            base
        } else {
            base.reverse()
        }
    }
}

impl SingleColumnTopK {
    /// Create new single-column top-K
    pub fn new(k: usize, col_idx: usize, ascending: bool) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            col_idx,
            ascending,
        }
    }

    /// Push a row
    pub fn push(&mut self, row: SochRow) {
        if self.k == 0 {
            return;
        }

        let key = row.values
            .get(self.col_idx)
            .map(OrderableValue::from)
            .unwrap_or(OrderableValue::Null);

        let entry = SingleColEntry {
            row,
            key,
            ascending: self.ascending,
        };

        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if let Some(top) = self.heap.peek() {
            let should_replace = if self.ascending {
                entry.key < top.key
            } else {
                entry.key > top.key
            };

            if should_replace {
                self.heap.pop();
                self.heap.push(entry);
            }
        }
    }

    /// Drain into sorted vector
    pub fn into_sorted_vec(self) -> Vec<SochRow> {
        let mut entries: Vec<_> = self.heap.into_iter().collect();
        
        entries.sort_by(|a, b| {
            let base = a.key.cmp(&b.key);
            if self.ascending { base } else { base.reverse() }
        });
        
        entries.into_iter().map(|e| e.row).collect()
    }

    /// Current count
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(values: Vec<SochValue>) -> SochRow {
        SochRow::new(values)
    }

    #[test]
    fn test_strategy_selection() {
        // With index: always pushdown
        assert_eq!(
            ExecutionStrategy::choose(true, Some(1_000_000), 10),
            ExecutionStrategy::IndexPushdown
        );

        // Small K without index: streaming
        assert_eq!(
            ExecutionStrategy::choose(false, Some(1_000_000), 10),
            ExecutionStrategy::StreamingTopK
        );

        // Large K relative to N (K > sqrt(N) and K > 100): full sort
        // For N=1000, sqrt(N) ≈ 31.6, so K=500 > sqrt(1000) and K > 100
        assert_eq!(
            ExecutionStrategy::choose(false, Some(1000), 500),
            ExecutionStrategy::FullSort
        );
        
        // K <= 100: always streaming even if K > sqrt(N)
        assert_eq!(
            ExecutionStrategy::choose(false, Some(100), 90),
            ExecutionStrategy::StreamingTopK
        );
    }

    #[test]
    fn test_order_by_spec_comparator() {
        let spec = OrderBySpec::single(
            ColumnRef::Name("priority".to_string()),
            SortDirection::Ascending,
        );
        
        let columns = vec!["id".to_string(), "priority".to_string(), "name".to_string()];
        let cmp = spec.comparator(&columns);
        
        let row1 = make_row(vec![
            SochValue::Int(1),
            SochValue::Int(5),
            SochValue::Text("A".to_string()),
        ]);
        let row2 = make_row(vec![
            SochValue::Int(2),
            SochValue::Int(3),
            SochValue::Text("B".to_string()),
        ]);
        
        // row2 has lower priority, should come first in ASC
        assert_eq!(cmp(&row2, &row1), Ordering::Less);
    }

    #[test]
    fn test_topk_heap_ascending() {
        let cmp = |a: &i32, b: &i32| a.cmp(b);
        let mut heap = TopKHeap::new(3, cmp, true);
        
        for i in [5, 2, 8, 1, 9, 3, 7, 4, 6] {
            heap.push(i);
        }
        
        let result = heap.into_sorted_vec();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_topk_heap_descending() {
        let cmp = |a: &i32, b: &i32| a.cmp(b);
        let mut heap = TopKHeap::new(3, cmp, false);
        
        for i in [5, 2, 8, 1, 9, 3, 7, 4, 6] {
            heap.push(i);
        }
        
        let result = heap.into_sorted_vec();
        // Descending order
        assert_eq!(result, vec![9, 8, 7]);
    }

    #[test]
    fn test_executor_streaming() {
        let columns = vec!["priority".to_string(), "name".to_string()];
        let order_by = OrderBySpec::single(
            ColumnRef::Name("priority".to_string()),
            SortDirection::Ascending,
        );
        
        let executor = OrderByLimitExecutor::new(
            order_by,
            3,      // limit
            0,      // offset
            columns.clone(),
            false,  // no index
            Some(10),
        );
        
        // Create rows with priorities: 5, 3, 1, 4, 2
        let rows = vec![
            make_row(vec![SochValue::Int(5), SochValue::Text("E".to_string())]),
            make_row(vec![SochValue::Int(3), SochValue::Text("C".to_string())]),
            make_row(vec![SochValue::Int(1), SochValue::Text("A".to_string())]),
            make_row(vec![SochValue::Int(4), SochValue::Text("D".to_string())]),
            make_row(vec![SochValue::Int(2), SochValue::Text("B".to_string())]),
        ];
        
        let (result, stats) = executor.execute(rows.into_iter());
        
        // Should get priorities 1, 2, 3 (the 3 smallest)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].values[0], SochValue::Int(1));
        assert_eq!(result[1].values[0], SochValue::Int(2));
        assert_eq!(result[2].values[0], SochValue::Int(3));
        
        assert_eq!(stats.input_rows, 5);
        assert_eq!(stats.output_rows, 3);
    }

    #[test]
    fn test_executor_with_offset() {
        let columns = vec!["priority".to_string()];
        let order_by = OrderBySpec::single(
            ColumnRef::Name("priority".to_string()),
            SortDirection::Ascending,
        );
        
        let executor = OrderByLimitExecutor::new(
            order_by,
            2,      // limit
            2,      // offset
            columns,
            false,
            Some(10),
        );
        
        // Priorities: 5, 3, 1, 4, 2 → sorted: 1, 2, 3, 4, 5
        // With offset 2, limit 2: should get 3, 4
        let rows = vec![
            make_row(vec![SochValue::Int(5)]),
            make_row(vec![SochValue::Int(3)]),
            make_row(vec![SochValue::Int(1)]),
            make_row(vec![SochValue::Int(4)]),
            make_row(vec![SochValue::Int(2)]),
        ];
        
        let (result, _) = executor.execute(rows.into_iter());
        
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].values[0], SochValue::Int(3));
        assert_eq!(result[1].values[0], SochValue::Int(4));
    }

    #[test]
    fn test_single_column_topk() {
        let mut topk = SingleColumnTopK::new(3, 0, true); // Column 0, ascending
        
        topk.push(make_row(vec![SochValue::Int(5)]));
        topk.push(make_row(vec![SochValue::Int(3)]));
        topk.push(make_row(vec![SochValue::Int(1)]));
        topk.push(make_row(vec![SochValue::Int(4)]));
        topk.push(make_row(vec![SochValue::Int(2)]));
        
        let result = topk.into_sorted_vec();
        
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].values[0], SochValue::Int(1));
        assert_eq!(result[1].values[0], SochValue::Int(2));
        assert_eq!(result[2].values[0], SochValue::Int(3));
    }

    #[test]
    fn test_correctness_vs_buggy_implementation() {
        // This test demonstrates the bug in the old implementation
        let columns = vec!["priority".to_string()];
        let order_by = OrderBySpec::single(
            ColumnRef::Name("priority".to_string()),
            SortDirection::Ascending,
        );
        
        // Rows in scan order: [5, 2, 8, 1, 9, 3]
        let rows: Vec<_> = [5, 2, 8, 1, 9, 3]
            .iter()
            .map(|&p| make_row(vec![SochValue::Int(p)]))
            .collect();
        
        // BUGGY: collect first 3, then sort
        let buggy: Vec<_> = rows.iter().take(3).cloned().collect();
        // buggy contains: [5, 2, 8] → sorted: [2, 5, 8]
        // Bug: would return priority 2 as "smallest" but actual min is 1!
        
        // CORRECT: streaming top-K
        let executor = OrderByLimitExecutor::new(
            order_by,
            3,
            0,
            columns,
            false,
            Some(6),
        );
        let (correct, _) = executor.execute(rows.into_iter());
        
        // Correct result: [1, 2, 3]
        assert_eq!(correct[0].values[0], SochValue::Int(1));
        assert_eq!(correct[1].values[0], SochValue::Int(2));
        assert_eq!(correct[2].values[0], SochValue::Int(3));
    }

    #[test]
    fn test_multi_column_order_by() {
        let columns = vec!["priority".to_string(), "created_at".to_string()];
        
        let order_by = OrderBySpec::single(
            ColumnRef::Name("priority".to_string()),
            SortDirection::Ascending,
        ).then_by(
            ColumnRef::Name("created_at".to_string()),
            SortDirection::Descending,
        );
        
        let executor = OrderByLimitExecutor::new(
            order_by,
            3,
            0,
            columns,
            false,
            Some(5),
        );
        
        // Rows: (priority, created_at)
        let rows = vec![
            make_row(vec![SochValue::Int(1), SochValue::Int(100)]),
            make_row(vec![SochValue::Int(1), SochValue::Int(200)]), // Same priority, later
            make_row(vec![SochValue::Int(2), SochValue::Int(150)]),
            make_row(vec![SochValue::Int(1), SochValue::Int(150)]),
            make_row(vec![SochValue::Int(3), SochValue::Int(100)]),
        ];
        
        let (result, _) = executor.execute(rows.into_iter());
        
        // Should be: priority=1 rows ordered by created_at DESC
        // So: (1, 200), (1, 150), (1, 100)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].values[0], SochValue::Int(1));
        assert_eq!(result[0].values[1], SochValue::Int(200));
        assert_eq!(result[1].values[0], SochValue::Int(1));
        assert_eq!(result[1].values[1], SochValue::Int(150));
        assert_eq!(result[2].values[0], SochValue::Int(1));
        assert_eq!(result[2].values[1], SochValue::Int(100));
    }
}
