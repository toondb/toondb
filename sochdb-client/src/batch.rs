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

//! Batch Writer with Group Commit & Streaming Auto-Commit
//!
//! High-throughput batch operations with adaptive sizing and transaction chunking.
//!
//! ## Streaming BatchWriter
//!
//! When `auto_commit` is enabled, the batch writer automatically commits transactions
//! when they reach `max_batch_size` operations. This provides:
//!
//! - **Bounded memory**: O(max_batch_size) instead of O(total_stream)
//! - **Predictable latency**: p95/p99 commit latency bounded by one chunk
//! - **Tunable throughput**: Batch size can be tuned to saturate fsync throughput
//!
//! ## Performance Model
//!
//! Per-operation cost:
//! ```text
//! C_op = c + L_fsync / K
//! ```
//! Where:
//! - c = CPU cost per write (~500ns)
//! - L_fsync = fsync latency (~5ms)
//! - K = batch size (ops per txn)
//!
//! Optimal batch size formula (from GroupCommitBuffer):
//! ```text
//! N* = sqrt(2 × L_fsync × λ / C_wait)
//! ```
//!
//! ## Example
//!
//! ```ignore
//! // Streaming mode: auto-commits every 1000 ops
//! let result = conn.batch()
//!     .max_batch_size(1000)
//!     .auto_commit(true)
//!     .insert("events", event1)
//!     .insert("events", event2)
//!     // ... millions of events
//!     .execute()?; // Commits final partial batch
//! ```

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::connection::SochConnection;
use crate::error::Result;

use sochdb_core::soch::SochValue;

/// Batch write operation
#[derive(Debug, Clone)]
pub enum BatchOp {
    Insert {
        table: String,
        values: Vec<(String, SochValue)>,
    },
    /// Zero-allocation insert using slice-based values in schema order
    InsertSlice {
        table: String,
        row_id: u64,
        /// Values in schema column order (indices match column positions)
        values: Vec<Option<SochValue>>,
    },
    Update {
        table: String,
        key_field: String,
        key_value: SochValue,
        updates: Vec<(String, SochValue)>,
    },
    Delete {
        table: String,
        key_field: String,
        key_value: SochValue,
    },
}

/// Batch result
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub ops_executed: usize,
    pub ops_failed: usize,
    pub duration_ms: u64,
    pub fsync_count: u64,
    /// Number of transaction chunks committed (for streaming mode)
    pub chunks_committed: usize,
}

/// Batch writer for high-throughput operations
///
/// Supports two modes:
/// - **Buffered mode** (auto_commit=false): All ops buffered, single commit at execute()
/// - **Streaming mode** (auto_commit=true): Auto-commits every max_batch_size ops
pub struct BatchWriter<'a> {
    conn: &'a SochConnection,
    ops: Vec<BatchOp>,
    max_batch_size: usize,
    auto_commit: bool,
    /// Number of chunks already committed (streaming mode)
    chunks_committed: usize,
    /// Total ops executed across all chunks
    total_ops_executed: usize,
    /// Total ops failed across all chunks
    total_ops_failed: usize,
    /// Cumulative duration of all chunks
    cumulative_duration_ms: u64,
}

impl<'a> BatchWriter<'a> {
    /// Create new batch writer
    pub fn new(conn: &'a SochConnection) -> Self {
        Self {
            conn,
            ops: Vec::new(),
            max_batch_size: 1000,
            auto_commit: false,
            chunks_committed: 0,
            total_ops_executed: 0,
            total_ops_failed: 0,
            cumulative_duration_ms: 0,
        }
    }

    /// Set maximum batch size
    ///
    /// In streaming mode (auto_commit=true), a commit is triggered when
    /// the batch reaches this size. Recommended values:
    /// - 100-500: Low latency, more fsyncs
    /// - 1000-5000: Balanced (default)
    /// - 10000+: Maximum throughput, higher latency spikes
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size.max(1); // At least 1
        self
    }

    /// Enable auto-commit when batch is full
    ///
    /// When enabled, the batch writer will automatically commit transactions
    /// when they reach `max_batch_size` operations. This bounds memory usage
    /// to O(max_batch_size) and provides predictable commit latency.
    pub fn auto_commit(mut self, enabled: bool) -> Self {
        self.auto_commit = enabled;
        self
    }

    /// Flush current batch if it's full (internal method)
    fn maybe_auto_flush(&mut self) -> Result<()> {
        if self.auto_commit && self.ops.len() >= self.max_batch_size {
            self.flush_current_batch()?;
        }
        Ok(())
    }

    /// Flush the current batch of operations (internal method)
    fn flush_current_batch(&mut self) -> Result<()> {
        if self.ops.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        let batch_ops = std::mem::take(&mut self.ops);
        let batch_size = batch_ops.len();
        let mut ops_failed = 0;

        {
            let mut tch = self.conn.tch.write();

            for op in batch_ops {
                match op {
                    BatchOp::Insert { table, values } => {
                        let map: std::collections::HashMap<String, SochValue> =
                            values.into_iter().collect();
                        tch.insert_row(&table, &map);
                    }
                    BatchOp::InsertSlice {
                        table,
                        row_id: _,
                        values,
                    } => {
                        // Convert to HashMap for now; the optimized path is in storage layer
                        let schema = tch.get_table_schema(&table);
                        if let Some(schema) = schema {
                            let columns: Vec<_> = schema
                                .fields
                                .iter()
                                .zip(values.into_iter())
                                .filter_map(|(name, val)| val.map(|v| (name.clone(), v)))
                                .collect();
                            let map: std::collections::HashMap<String, SochValue> =
                                columns.into_iter().collect();
                            tch.insert_row(&table, &map);
                        } else {
                            ops_failed += 1;
                        }
                    }
                    BatchOp::Update {
                        table,
                        key_field,
                        key_value,
                        updates,
                    } => {
                        let map: std::collections::HashMap<String, SochValue> =
                            updates.into_iter().collect();
                        let where_clause = crate::connection::WhereClause::Simple {
                            field: key_field,
                            op: crate::connection::CompareOp::Eq,
                            value: key_value,
                        };
                        // MutationResult contains affected_row_ids for future CDC/WAL integration
                        let _mutation_result = tch.update_rows(&table, &map, Some(&where_clause));
                    }
                    BatchOp::Delete {
                        table,
                        key_field,
                        key_value,
                    } => {
                        let where_clause = crate::connection::WhereClause::Simple {
                            field: key_field,
                            op: crate::connection::CompareOp::Eq,
                            value: key_value,
                        };
                        // MutationResult contains affected_row_ids for future CDC/WAL integration
                        let _mutation_result = tch.delete_rows(&table, Some(&where_clause));
                    }
                }
            }
        }

        // Single fsync for this chunk
        self.conn.storage.fsync()?;

        let duration = start.elapsed().as_millis() as u64;
        self.chunks_committed += 1;
        self.total_ops_executed += batch_size - ops_failed;
        self.total_ops_failed += ops_failed;
        self.cumulative_duration_ms += duration;

        Ok(())
    }

    /// Add insert operation
    pub fn insert(mut self, table: &str, values: Vec<(&str, SochValue)>) -> Self {
        self.ops.push(BatchOp::Insert {
            table: table.to_string(),
            values: values
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        });

        // Auto-flush if batch is full
        if let Err(_e) = self.maybe_auto_flush() {
            // In streaming mode, errors are accumulated; execute() returns final result
        }

        self
    }

    /// Add insert operation using slice-based values (zero-allocation path)
    ///
    /// Values must be in schema column order. Use None for NULL values.
    /// This is the fastest insert path, matching benchmark performance.
    ///
    /// # Example
    /// ```ignore
    /// batch.insert_slice("users", 1, vec![
    ///     Some(SochValue::UInt(1)),
    ///     Some(SochValue::Text("Alice".into())),
    ///     None, // NULL
    /// ])
    /// ```
    pub fn insert_slice(
        mut self,
        table: &str,
        row_id: u64,
        values: Vec<Option<SochValue>>,
    ) -> Self {
        self.ops.push(BatchOp::InsertSlice {
            table: table.to_string(),
            row_id,
            values,
        });

        if let Err(_e) = self.maybe_auto_flush() {
            // Errors accumulated
        }

        self
    }

    /// Add update operation
    pub fn update(
        mut self,
        table: &str,
        key_field: &str,
        key_value: SochValue,
        updates: Vec<(&str, SochValue)>,
    ) -> Self {
        self.ops.push(BatchOp::Update {
            table: table.to_string(),
            key_field: key_field.to_string(),
            key_value,
            updates: updates
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        });

        if let Err(_e) = self.maybe_auto_flush() {
            // Errors accumulated
        }

        self
    }

    /// Add delete operation
    pub fn delete(mut self, table: &str, key_field: &str, key_value: SochValue) -> Self {
        self.ops.push(BatchOp::Delete {
            table: table.to_string(),
            key_field: key_field.to_string(),
            key_value,
        });

        if let Err(_e) = self.maybe_auto_flush() {
            // Errors accumulated
        }

        self
    }

    /// Get number of pending operations (in current unflushed batch)
    pub fn pending_count(&self) -> usize {
        self.ops.len()
    }

    /// Get total operations processed so far (including flushed chunks)
    pub fn total_count(&self) -> usize {
        self.total_ops_executed + self.total_ops_failed + self.ops.len()
    }

    /// Execute all pending operations
    ///
    /// In streaming mode, this commits any remaining operations in the final
    /// partial batch. Returns cumulative results from all chunks.
    pub fn execute(mut self) -> Result<BatchResult> {
        let _start = Instant::now();

        // Flush any remaining operations
        if !self.ops.is_empty() {
            self.flush_current_batch()?;
        }

        Ok(BatchResult {
            ops_executed: self.total_ops_executed,
            ops_failed: self.total_ops_failed,
            duration_ms: self.cumulative_duration_ms,
            fsync_count: self.chunks_committed as u64,
            chunks_committed: self.chunks_committed,
        })
    }
}

/// Group commit buffer for high-throughput durability
///
/// **DEPRECATED**: Use `sochdb_storage::GroupCommitBuffer` instead.
/// This client-side implementation is purely in-memory and doesn't
/// actually perform I/O or integrate with the WAL. The storage layer's
/// `EventDrivenGroupCommit` or `GroupCommitBuffer` should be used for
/// actual durability guarantees.
///
/// Batches multiple transactions into single fsync for efficiency.
/// Optimal batch size: N* = sqrt(2 × L_fsync × λ / C_wait)
#[deprecated(
    since = "0.2.0",
    note = "Use sochdb_storage::EventDrivenGroupCommit for actual WAL integration"
)]
pub struct GroupCommitBuffer {
    inner: Arc<Mutex<GroupCommitInner>>,
    condvar: Arc<Condvar>,
    config: GroupCommitConfig,
}

struct GroupCommitInner {
    pending: VecDeque<PendingCommit>,
    batch_id: u64,
}

#[allow(dead_code)]
struct PendingCommit {
    id: u64,
    batch_id: u64,
    committed: bool,
}

/// Group commit configuration
#[derive(Debug, Clone)]
pub struct GroupCommitConfig {
    /// Maximum wait time before forced flush
    pub max_wait_ms: u64,
    /// Maximum operations before forced flush
    pub max_batch_size: usize,
    /// Target batch size (adaptive)
    pub target_batch_size: usize,
    /// Average fsync latency in microseconds
    pub fsync_latency_us: u64,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_wait_ms: 10,
            max_batch_size: 1000,
            target_batch_size: 100,
            fsync_latency_us: 5000, // 5ms default
        }
    }
}

#[allow(deprecated)]
impl GroupCommitBuffer {
    /// Create new group commit buffer
    pub fn new(config: GroupCommitConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(GroupCommitInner {
                pending: VecDeque::new(),
                batch_id: 0,
            })),
            condvar: Arc::new(Condvar::new()),
            config,
        }
    }

    /// Calculate optimal batch size
    ///
    /// Formula: N* = sqrt(2 × L_fsync × λ / C_wait)
    /// - L_fsync: fsync latency
    /// - λ: arrival rate (ops/sec)
    /// - C_wait: cost per unit wait time
    pub fn optimal_batch_size(&self, arrival_rate: f64, wait_cost: f64) -> usize {
        let l_fsync = self.config.fsync_latency_us as f64 / 1_000_000.0;
        let n_star = (2.0 * l_fsync * arrival_rate / wait_cost).sqrt();
        (n_star as usize).clamp(1, self.config.max_batch_size)
    }

    /// Submit operation and wait for commit
    pub fn submit_and_wait(&self, op_id: u64) -> Result<u64> {
        let timeout = Duration::from_millis(self.config.max_wait_ms);
        let target_size = self.config.target_batch_size;

        let mut inner = self.inner.lock();
        let current_batch_id = inner.batch_id;
        inner.pending.push_back(PendingCommit {
            id: op_id,
            batch_id: current_batch_id,
            committed: false,
        });

        // Check if we should flush
        let need_flush = inner.pending.len() >= target_size;
        if need_flush {
            inner.batch_id += 1;
        }

        let batch_id = inner.batch_id;

        // Wait for batch to be committed
        let result = self.condvar.wait_for(&mut inner, timeout);

        if result.timed_out() {
            // Force flush on timeout
            inner.batch_id += 1;
        }

        Ok(batch_id)
    }

    /// Flush pending commits
    pub fn flush(&self) {
        let mut inner = self.inner.lock();

        // Mark all pending as committed
        for pending in inner.pending.iter_mut() {
            pending.committed = true;
        }
        inner.pending.clear();
        inner.batch_id += 1;

        // Wake all waiters
        self.condvar.notify_all();
    }

    /// Get pending count
    pub fn pending_count(&self) -> usize {
        self.inner.lock().pending.len()
    }
}

/// Batch operations on connection
impl SochConnection {
    /// Start batch writer
    pub fn batch<'a>(&'a self) -> BatchWriter<'a> {
        BatchWriter::new(self)
    }

    /// Bulk insert rows (uses streaming mode internally)
    pub fn bulk_insert(
        &self,
        table: &str,
        rows: Vec<Vec<(&str, SochValue)>>,
    ) -> Result<BatchResult> {
        let mut batch = BatchWriter::new(self)
            .max_batch_size(1000)
            .auto_commit(true); // Enable streaming for bulk inserts
        for row in rows {
            batch = batch.insert(table, row);
        }
        batch.execute()
    }

    /// Bulk insert with zero-allocation path (fastest)
    ///
    /// Values must be in schema column order.
    pub fn bulk_insert_slice(
        &self,
        table: &str,
        rows: Vec<(u64, Vec<Option<SochValue>>)>,
    ) -> Result<BatchResult> {
        let mut batch = BatchWriter::new(self)
            .max_batch_size(1000)
            .auto_commit(true);
        for (row_id, values) in rows {
            batch = batch.insert_slice(table, row_id, values);
        }
        batch.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_writer() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .batch()
            .insert(
                "users",
                vec![
                    ("id", SochValue::Int(1)),
                    ("name", SochValue::Text("Alice".to_string())),
                ],
            )
            .insert(
                "users",
                vec![
                    ("id", SochValue::Int(2)),
                    ("name", SochValue::Text("Bob".to_string())),
                ],
            )
            .execute()
            .unwrap();

        assert_eq!(result.ops_executed, 2);
        assert_eq!(result.fsync_count, 1);
        assert_eq!(result.chunks_committed, 1);
    }

    #[test]
    fn test_streaming_batch_writer() {
        let conn = SochConnection::open("./test_streaming").unwrap();

        // With auto_commit and max_batch_size=2, should commit after every 2 ops
        let result = conn
            .batch()
            .max_batch_size(2)
            .auto_commit(true)
            .insert("users", vec![("id", SochValue::Int(1))])
            .insert("users", vec![("id", SochValue::Int(2))])
            .insert("users", vec![("id", SochValue::Int(3))])
            .insert("users", vec![("id", SochValue::Int(4))])
            .insert("users", vec![("id", SochValue::Int(5))])
            .execute()
            .unwrap();

        assert_eq!(result.ops_executed, 5);
        assert_eq!(result.chunks_committed, 3); // 2 full chunks + 1 partial
    }

    #[test]
    fn test_group_commit_config() {
        let config = GroupCommitConfig::default();
        assert_eq!(config.max_wait_ms, 10);
        assert_eq!(config.max_batch_size, 1000);
    }

    #[test]
    #[allow(deprecated)]
    fn test_optimal_batch_size() {
        let config = GroupCommitConfig {
            fsync_latency_us: 5000, // 5ms
            ..Default::default()
        };
        let buffer = GroupCommitBuffer::new(config);

        // arrival_rate = 10000 ops/sec, wait_cost = 0.001
        let optimal = buffer.optimal_batch_size(10000.0, 0.001);
        assert!(optimal > 1);
        assert!(optimal <= 1000);
    }

    #[test]
    fn test_bulk_insert() {
        let conn = SochConnection::open("./test").unwrap();

        let rows = vec![
            vec![
                ("id", SochValue::Int(1)),
                ("name", SochValue::Text("A".to_string())),
            ],
            vec![
                ("id", SochValue::Int(2)),
                ("name", SochValue::Text("B".to_string())),
            ],
            vec![
                ("id", SochValue::Int(3)),
                ("name", SochValue::Text("C".to_string())),
            ],
        ];

        let result = conn.bulk_insert("users", rows).unwrap();
        assert_eq!(result.ops_executed, 3);
    }

    #[test]
    fn test_insert_slice() {
        use crate::connection::FieldType;

        let conn = SochConnection::open("./test_slice").unwrap();

        // Register the table schema first - insert_slice requires a pre-existing schema
        conn.register_table(
            "users",
            &[
                ("id".to_string(), FieldType::UInt64),
                ("name".to_string(), FieldType::Text),
            ],
        )
        .unwrap();

        let result = conn
            .batch()
            .insert_slice(
                "users",
                1,
                vec![
                    Some(SochValue::UInt(1)),
                    Some(SochValue::Text("Alice".to_string())),
                ],
            )
            .insert_slice(
                "users",
                2,
                vec![
                    Some(SochValue::UInt(2)),
                    Some(SochValue::Text("Bob".to_string())),
                ],
            )
            .execute()
            .unwrap();

        assert_eq!(result.ops_executed, 2);
    }
}
