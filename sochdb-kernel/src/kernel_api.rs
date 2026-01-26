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

//! Kernel API Traits
//!
//! These traits define the stable API surface of the kernel.
//! Extensions and plugins implement or consume these traits.
//!
//! ## API Stability Guarantee
//!
//! Once the kernel reaches 1.0, these traits will follow semver:
//! - 1.x.x: No breaking changes to trait signatures
//! - 2.0.0: May introduce breaking changes with migration guide

use crate::error::KernelResult;
use crate::transaction::TransactionId;
use crate::wal::LogSequenceNumber;

/// Page identifier
pub type PageId = u64;

/// Row identifier  
pub type RowId = u64;

/// Table identifier
pub type TableId = u32;

/// Column identifier
pub type ColumnId = u16;

/// Core storage operations
///
/// This trait provides the minimal storage interface that all
/// storage backends must implement.
pub trait KernelStorage: Send + Sync {
    /// Read a page from storage
    fn read_page(&self, page_id: PageId) -> KernelResult<Vec<u8>>;

    /// Write a page to storage
    ///
    /// Returns the LSN of the write operation for WAL tracking
    fn write_page(&self, page_id: PageId, data: &[u8]) -> KernelResult<LogSequenceNumber>;

    /// Allocate a new page
    fn allocate_page(&self) -> KernelResult<PageId>;

    /// Free a page
    fn free_page(&self, page_id: PageId) -> KernelResult<()>;

    /// Sync all pending writes to durable storage
    fn sync(&self) -> KernelResult<()>;

    /// Get the current durable LSN (all writes up to this LSN are on disk)
    fn durable_lsn(&self) -> LogSequenceNumber;
}

/// Transaction operations
///
/// Provides ACID transaction semantics.
pub trait KernelTransaction: Send + Sync {
    /// Begin a new transaction
    fn begin(&self) -> KernelResult<TransactionId>;

    /// Begin a transaction with specific isolation level
    fn begin_with_isolation(
        &self,
        isolation: crate::transaction::IsolationLevel,
    ) -> KernelResult<TransactionId>;

    /// Commit a transaction
    fn commit(&self, txn_id: TransactionId) -> KernelResult<()>;

    /// Abort a transaction
    fn abort(&self, txn_id: TransactionId) -> KernelResult<()>;

    /// Check if a transaction is active
    fn is_active(&self, txn_id: TransactionId) -> bool;

    /// Get the snapshot timestamp for a transaction
    fn snapshot_ts(&self, txn_id: TransactionId) -> KernelResult<u64>;
}

/// Catalog operations
///
/// Schema management and metadata.
pub trait KernelCatalog: Send + Sync {
    /// Create a new table
    fn create_table(&self, name: &str, schema: &TableSchema) -> KernelResult<TableId>;

    /// Drop a table
    fn drop_table(&self, table_id: TableId) -> KernelResult<()>;

    /// Get table schema
    fn get_schema(&self, table_id: TableId) -> KernelResult<TableSchema>;

    /// List all tables
    fn list_tables(&self) -> KernelResult<Vec<TableInfo>>;

    /// Rename a table
    fn rename_table(&self, table_id: TableId, new_name: &str) -> KernelResult<()>;
}

/// Table schema definition
#[derive(Debug, Clone)]
pub struct TableSchema {
    /// Table name
    pub name: String,
    /// Column definitions
    pub columns: Vec<ColumnDef>,
    /// Primary key column indices
    pub primary_key: Vec<usize>,
}

/// Column definition
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Column name
    pub name: String,
    /// Column type
    pub data_type: DataType,
    /// Is nullable
    pub nullable: bool,
    /// Default value (serialized)
    pub default_value: Option<Vec<u8>>,
}

/// Supported data types (minimal set for kernel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Boolean
    Bool,
    /// 64-bit signed integer
    Int64,
    /// 64-bit float
    Float64,
    /// Variable-length string
    String,
    /// Variable-length bytes
    Bytes,
    /// Timestamp (microseconds since epoch)
    Timestamp,
    /// Null type
    Null,
}

/// Table information
#[derive(Debug, Clone)]
pub struct TableInfo {
    /// Table ID
    pub id: TableId,
    /// Table name
    pub name: String,
    /// Row count (approximate)
    pub row_count: u64,
    /// Creation timestamp
    pub created_at: u64,
}

/// Recovery operations
///
/// Crash recovery and checkpoint management.
pub trait KernelRecovery: Send + Sync {
    /// Perform crash recovery
    ///
    /// Returns the number of transactions recovered (committed + aborted)
    fn recover(&self) -> KernelResult<RecoveryStats>;

    /// Create a checkpoint
    fn checkpoint(&self) -> KernelResult<LogSequenceNumber>;

    /// Get the last checkpoint LSN
    fn last_checkpoint_lsn(&self) -> Option<LogSequenceNumber>;
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of committed transactions redone
    pub txns_redone: u64,
    /// Number of uncommitted transactions undone
    pub txns_undone: u64,
    /// Number of pages recovered
    pub pages_recovered: u64,
    /// Recovery duration in milliseconds
    pub duration_ms: u64,
}

/// Health check for monitoring
///
/// Minimal health interface - detailed metrics are in ObservabilityExtension
pub trait KernelHealth: Send + Sync {
    /// Check if the kernel is healthy
    fn is_healthy(&self) -> bool;

    /// Get basic health info (for plugins to consume)
    fn health_info(&self) -> HealthInfo;
}

/// Basic health information
///
/// This is the minimal info the kernel exposes.
/// Detailed metrics are handled by ObservabilityExtension plugins.
#[derive(Debug, Clone)]
pub struct HealthInfo {
    /// Is the kernel operational
    pub operational: bool,
    /// Current WAL size in bytes
    pub wal_size_bytes: u64,
    /// Active transaction count
    pub active_txns: u64,
    /// Buffer pool usage (0.0 - 1.0)
    pub buffer_pool_usage: f64,
    /// Last checkpoint age in seconds
    pub checkpoint_age_secs: u64,
}

impl Default for HealthInfo {
    fn default() -> Self {
        Self {
            operational: true,
            wal_size_bytes: 0,
            active_txns: 0,
            buffer_pool_usage: 0.0,
            checkpoint_age_secs: 0,
        }
    }
}
