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

//! SochDB Database Kernel
//!
//! The shared core that powers both embedded mode (`SochConnection::open`) and
//! server mode (`sochdb-server`). This is the "SQLite engine" equivalent.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                        Database Kernel                            │
//! │  Arc<Database> - shared by all connections                       │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                                                                   │
//! │  ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐ │
//! │  │  DurableStorage │   │     Catalog     │   │  Vector Index  │ │
//! │  │  (WAL + MVCC)   │   │  (Schema Mgmt)  │   │  (HNSW/Vamana) │ │
//! │  └────────┬────────┘   └────────┬────────┘   └───────┬────────┘ │
//! │           │                     │                     │          │
//! │           └─────────────────────┴─────────────────────┘          │
//! │                                 │                                 │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │              Query Executor (Path-Native)                    │ │
//! │  │  - Path resolution: O(|path|)                                │ │
//! │  │  - Column projection: 80% I/O reduction                     │ │
//! │  │  - Context selection: Token-aware chunking                  │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! │                                                                   │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! Deployment Modes:
//! ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
//! │  Embedded   │   │  IPC Server │   │  TCP Server │
//! │  (in-proc)  │   │  (Unix sock)│   │  (remote)   │
//! └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
//!        │                 │                 │
//!        └─────────────────┴─────────────────┘
//!                          │
//!                   Arc<Database>
//! ```
//!
//! ## Latency Model
//!
//! Let K = kernel processing cost for a query
//!
//! - Embedded: L_emb ≈ K (function call overhead negligible)
//! - IPC: L_ipc ≈ K + δ_ipc (δ_ipc = ~10-50µs for Unix socket)
//! - TCP: L_tcp ≈ K + δ_net (δ_net = 100µs-10ms depending on network)
//!
//! For LLM context queries where K >> δ_ipc, IPC is "nearly embedded".

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::durable_storage::{DurableStorage, TransactionMode};
use crate::index_policy::{IndexPolicy, TableIndexConfig, TableIndexRegistry};
use crate::key_buffer::KeyBuffer;
use crate::packed_row::{PackedColumnDef, PackedColumnType, PackedRow, PackedTableSchema};
use sochdb_core::catalog::Catalog;
use sochdb_core::{Result, SochDBError, SochValue};

// Re-export key types
pub use crate::durable_storage::RecoveryStats;

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Enable group commit for better write throughput
    pub group_commit: bool,
    /// Maximum memory for memtables before flush (bytes)
    pub memtable_size_limit: usize,
    /// Enable WAL for durability
    pub wal_enabled: bool,
    /// Sync mode: fsync after every commit vs periodic
    pub sync_mode: SyncMode,
    /// Read-only mode
    pub read_only: bool,
    
    /// Enable ordered index for O(log N) prefix scans
    ///
    /// # Deprecation Notice
    /// 
    /// **DEPRECATED since 0.2.0**: Use `default_index_policy` instead for per-table control.
    /// This field will be removed in v0.3.0.
    ///
    /// ## Migration Guide
    ///
    /// Replace:
    /// ```ignore
    /// DatabaseConfig { enable_ordered_index: true, .. }  // Old API
    /// DatabaseConfig { enable_ordered_index: false, .. } // Old API
    /// ```
    ///
    /// With:
    /// ```ignore
    /// DatabaseConfig { default_index_policy: IndexPolicy::ScanOptimized, .. }  // Ordered index enabled
    /// DatabaseConfig { default_index_policy: IndexPolicy::WriteOptimized, .. } // Ordered index disabled
    /// ```
    ///
    /// ## Behavior
    ///
    /// When false, saves ~134 ns/op on writes (20% speedup)
    /// but scan_prefix becomes O(N) instead of O(log N + K).
    /// 
    /// Set to false for write-heavy workloads without range scans.
    #[deprecated(
        since = "0.2.0",
        note = "Use `default_index_policy` field instead. This field will be removed in v0.3.0. \
                Set IndexPolicy::ScanOptimized for ordered index, WriteOptimized to disable."
    )]
    ///
    /// Set to false for write-heavy workloads without range scans.
    pub enable_ordered_index: bool,
    /// Group commit configuration
    pub group_commit_config: GroupCommitSettings,
    /// Default index policy for tables not explicitly configured
    ///
    /// This replaces the global `enable_ordered_index` toggle with
    /// fine-grained per-table control. Use `index_registry` to configure
    /// individual tables.
    ///
    /// | Policy         | Insert Cost | Scan Cost      | Use Case              |
    /// |----------------|-------------|----------------|------------------------|
    /// | WriteOptimized | O(1)        | O(N)           | High-write, rare scan  |
    /// | Balanced       | O(1) amort  | O(output+logK) | Mixed OLTP            |
    /// | ScanOptimized  | O(log N)    | O(logN + K)    | Analytics, range query |
    /// | AppendOnly     | O(1)        | O(N)           | Time-series logs       |
    pub default_index_policy: IndexPolicy,
}

/// Group commit settings - mirrors SQLite's WAL mode tuning
///
/// ## Performance Model
///
/// Without group commit: Throughput = 1 / L_fsync ≈ 200 commits/sec (L=5ms)
/// With group commit (batch size K): Throughput = K / L_fsync = K × 200 commits/sec
///
/// For K=100: 20,000 commits/sec (100× speedup)
///
/// ## SQLite Comparison
///
/// | Setting                    | SQLite Equivalent           |
/// |----------------------------|-----------------------------|
/// | batch_size = 1             | PRAGMA synchronous = FULL   |
/// | batch_size = 100           | WAL mode with batching      |
/// | max_wait_us = 0            | No delay, immediate flush   |
/// | max_wait_us = 10000        | Up to 10ms delay for batch  |
#[derive(Debug, Clone)]
pub struct GroupCommitSettings {
    /// Minimum batch size before flush (default: 1)
    pub min_batch_size: usize,
    /// Maximum batch size (default: 1000)
    pub max_batch_size: usize,
    /// Maximum wait time before flush in microseconds (default: 10000 = 10ms)
    pub max_wait_us: u64,
    /// Expected fsync latency in microseconds (for adaptive sizing)
    pub fsync_latency_us: u64,
}

impl Default for GroupCommitSettings {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 1000,
            max_wait_us: 10_000,     // 10ms
            fsync_latency_us: 5_000, // 5ms
        }
    }
}

impl GroupCommitSettings {
    /// High throughput preset - maximizes batching
    pub fn high_throughput() -> Self {
        Self {
            min_batch_size: 50,
            max_batch_size: 5000,
            max_wait_us: 50_000, // 50ms
            fsync_latency_us: 5_000,
        }
    }

    /// Low latency preset - minimal batching
    pub fn low_latency() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 10,
            max_wait_us: 1_000, // 1ms
            fsync_latency_us: 5_000,
        }
    }

    /// Calculate optimal batch size using Little's Law
    ///
    /// N* = sqrt(2 × L_fsync × λ / C_wait)
    ///
    /// # Arguments
    /// * `arrival_rate` - Operations per second
    /// * `wait_cost` - Cost coefficient for waiting (0.0-1.0)
    pub fn optimal_batch_size(&self, arrival_rate: f64, wait_cost: f64) -> usize {
        let l_fsync = self.fsync_latency_us as f64 / 1_000_000.0;
        let n_star = (2.0 * l_fsync * arrival_rate / wait_cost.max(0.001)).sqrt();
        (n_star as usize).clamp(self.min_batch_size, self.max_batch_size)
    }
}

impl Default for DatabaseConfig {
    #[allow(deprecated)]
    fn default() -> Self {
        Self {
            group_commit: true,
            memtable_size_limit: 64 * 1024 * 1024, // 64MB
            wal_enabled: true,
            sync_mode: SyncMode::Normal,
            read_only: false,
            enable_ordered_index: true, // Default: enabled for compatibility
            group_commit_config: GroupCommitSettings::default(),
            default_index_policy: IndexPolicy::Balanced, // New default: balanced OLTP policy
        }
    }
}

impl DatabaseConfig {
    /// Create config optimized for throughput (Fast Mode)
    ///
    /// - Disables ordered index (saves ~134 ns/op)
    /// - Uses high-throughput group commit settings
    /// - Suitable for append-only workloads
    #[allow(deprecated)]
    pub fn throughput_optimized() -> Self {
        Self {
            group_commit: true,
            memtable_size_limit: 128 * 1024 * 1024, // 128MB
            wal_enabled: true,
            sync_mode: SyncMode::Normal,
            read_only: false,
            enable_ordered_index: false,
            group_commit_config: GroupCommitSettings::high_throughput(),
            default_index_policy: IndexPolicy::WriteOptimized, // No ordered index overhead
        }
    }

    /// Create config optimized for latency
    ///
    /// - Keeps ordered index for fast range scans
    /// - Uses low-latency group commit settings
    /// - Suitable for OLTP workloads
    #[allow(deprecated)]
    pub fn latency_optimized() -> Self {
        Self {
            group_commit: true,
            memtable_size_limit: 32 * 1024 * 1024, // 32MB
            wal_enabled: true,
            sync_mode: SyncMode::Full,
            read_only: false,
            enable_ordered_index: true,
            group_commit_config: GroupCommitSettings::low_latency(),
            default_index_policy: IndexPolicy::ScanOptimized, // Fast range scans
        }
    }

    /// Create config matching SQLite defaults
    #[allow(deprecated)]
    pub fn sqlite_compatible() -> Self {
        Self {
            group_commit: false, // SQLite default is single-commit
            memtable_size_limit: 64 * 1024 * 1024,
            wal_enabled: true,
            sync_mode: SyncMode::Normal, // PRAGMA synchronous = NORMAL
            read_only: false,
            enable_ordered_index: true,
            group_commit_config: GroupCommitSettings::default(),
            default_index_policy: IndexPolicy::Balanced, // Good default for mixed workloads
        }
    }

    /// Get effective ordered index setting, derived from `default_index_policy`.
    /// 
    /// This is the shim method for the deprecated `enable_ordered_index` field.
    /// It returns `true` if the policy requires an ordered index (ScanOptimized),
    /// and `false` otherwise (WriteOptimized, Balanced, AppendOnly).
    ///
    /// # Policy Mapping
    ///
    /// | IndexPolicy      | Returns |
    /// |------------------|---------|
    /// | ScanOptimized    | true    |
    /// | Balanced         | false   |
    /// | WriteOptimized   | false   |
    /// | AppendOnly       | false   |
    ///
    /// Note: `Balanced` uses lazy compaction rather than a live ordered index,
    /// so it returns `false` for the low-level memtable config but still supports
    /// efficient range scans via sorted runs.
    pub fn effective_ordered_index(&self) -> bool {
        matches!(self.default_index_policy, IndexPolicy::ScanOptimized)
    }
}

/// WAL sync mode - matches SQLite's PRAGMA synchronous semantics
///
/// | SochDB     | SQLite       | Description                                    |
/// |------------|--------------|------------------------------------------------|
/// | Off        | OFF (0)      | No fsync, risk of data loss on crash           |
/// | Normal     | NORMAL (1)   | Fsync at checkpoints, not every commit         |
/// | Full       | FULL (2)     | Fsync every commit (safest, slowest)           |
///
/// # Performance vs Durability Trade-offs
///
/// - **Off**: ~10x faster than Full, but may lose last ~100ms of data on crash
/// - **Normal**: ~5x faster than Full, durable at checkpoint boundaries
/// - **Full**: Every commit is fsync'd, no data loss possible
///
/// # SQLite Compatibility
///
/// ```sql
/// -- SQLite equivalent settings
/// PRAGMA synchronous = OFF;    -- SyncMode::Off
/// PRAGMA synchronous = NORMAL; -- SyncMode::Normal  
/// PRAGMA synchronous = FULL;   -- SyncMode::Full
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// No fsync (equivalent to SQLite PRAGMA synchronous = OFF)
    ///
    /// Writes buffered in OS, may lose data on power failure.
    /// Use for non-critical data or bulk loading.
    Off = 0,

    /// Fsync at checkpoints (equivalent to SQLite PRAGMA synchronous = NORMAL)
    ///
    /// Default mode. Syncs WAL at checkpoint boundaries.
    /// Good balance of performance and durability.
    Normal = 1,

    /// Fsync every commit (equivalent to SQLite PRAGMA synchronous = FULL)
    ///
    /// Safest mode. Every commit is immediately durable.
    /// Required for financial/critical data.
    Full = 2,
}

impl SyncMode {
    /// Convert from SQLite synchronous pragma value
    pub fn from_sqlite_pragma(value: u32) -> Self {
        match value {
            0 => SyncMode::Off,
            1 => SyncMode::Normal,
            _ => SyncMode::Full, // 2+ treated as Full
        }
    }

    /// Convert to SQLite synchronous pragma value
    pub fn to_sqlite_pragma(self) -> u32 {
        self as u32
    }

    /// Parse from string (case-insensitive)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "OFF" | "0" => Some(SyncMode::Off),
            "NORMAL" | "1" => Some(SyncMode::Normal),
            "FULL" | "2" => Some(SyncMode::Full),
            _ => None,
        }
    }
}

/// Table schema for the kernel
#[derive(Debug, Clone)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnDef>,
}

/// Column definition
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: ColumnType,
    pub nullable: bool,
}

/// Column types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int64,
    UInt64,
    Float64,
    Text,
    Binary,
    Bool,
}

/// Transaction handle for kernel operations
#[derive(Debug, Clone, Copy)]
pub struct TxnHandle {
    pub txn_id: u64,
    pub snapshot_ts: u64,
}

/// Query result from the kernel
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Column names
    pub columns: Vec<String>,
    /// Row data (each row is a map of column -> value)
    pub rows: Vec<HashMap<String, SochValue>>,
    /// Number of rows scanned (for stats)
    pub rows_scanned: usize,
    /// Bytes read from storage
    pub bytes_read: usize,
}

impl QueryResult {
    /// Empty result
    pub fn empty() -> Self {
        Self {
            columns: vec![],
            rows: vec![],
            rows_scanned: 0,
            bytes_read: 0,
        }
    }

    /// Convert to TOON format for token efficiency
    pub fn to_toon(&self) -> String {
        if self.rows.is_empty() {
            return "[]".to_string();
        }

        // TOON format: table[N]{cols}: row1; row2; ...
        let n = self.rows.len();
        let cols = self.columns.join(",");

        let rows_str: Vec<String> = self
            .rows
            .iter()
            .map(|row| {
                self.columns
                    .iter()
                    .map(|c| {
                        row.get(c)
                            .map(format_soch_value)
                            .unwrap_or_else(|| "∅".to_string())
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();

        format!("result[{}]{{{}}}:{}", n, cols, rows_str.join(";"))
    }
}

fn format_soch_value(v: &SochValue) -> String {
    match v {
        SochValue::Null => "∅".to_string(),
        SochValue::Int(i) => i.to_string(),
        SochValue::UInt(u) => u.to_string(),
        SochValue::Float(f) => format!("{:.6}", f),
        SochValue::Text(s) => {
            if s.contains(',') || s.contains(';') {
                format!("\"{}\"", s.replace('"', "\\\""))
            } else {
                s.clone()
            }
        }
        SochValue::Bool(b) => if *b { "T" } else { "F" }.to_string(),
        SochValue::Binary(b) => format!("b64:{}", base64_encode(b)),
        _ => format!("{:?}", v),
    }
}

fn base64_encode(data: &[u8]) -> String {
    // Simple base64 encoding
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

// ============================================================================
// Columnar Query Results - SIMD-friendly result format
// ============================================================================

use sochdb_core::TypedColumn as CoreTypedColumn;

/// Columnar query result - SIMD-friendly format for analytics
///
/// Instead of row-oriented `Vec<HashMap<String, SochValue>>`, this returns
/// column-oriented `Vec<TypedColumn>` for efficient vectorized operations.
///
/// ## Memory Layout
///
/// Row-oriented (standard):
/// ```text
/// Row 0: [id=1, name="Alice", score=85]
/// Row 1: [id=2, name="Bob", score=92]
/// Row 2: [id=3, name="Carol", score=78]
/// ```
///
/// Column-oriented (this format):
/// ```text
/// id:    [1, 2, 3]           ← contiguous i64 array (SIMD-friendly)
/// name:  ["Alice", "Bob", "Carol"] ← Arrow-style string encoding
/// score: [85, 92, 78]        ← contiguous i64 array
/// ```
///
/// ## Performance Benefits
///
/// - SIMD: Column sums use vectorized instructions (~8× faster)
/// - Cache: Sequential access pattern maximizes L1/L2 cache hits
/// - Compression: Same-type data compresses better (5-10× typical)
/// - Filtering: Bitmap operations instead of row iteration
///
/// ## Usage
///
/// ```ignore
/// let result = db.query(txn, "users")
///     .columns(&["id", "score"])
///     .as_columnar()?;
///
/// // SIMD sum
/// let total_score = result.column("score")
///     .map(|c| c.sum_i64())
///     .unwrap_or(0);
///
/// // Stats
/// println!("Rows: {}, Memory: {} bytes", result.row_count(), result.memory_size());
/// ```
#[derive(Debug, Clone)]
pub struct ColumnarQueryResult {
    /// Column names in order
    pub columns: Vec<String>,
    /// Column data - each TypedColumn contains all values for one column
    pub data: Vec<CoreTypedColumn>,
    /// Number of rows
    pub row_count: usize,
    /// Bytes read from storage
    pub bytes_read: usize,
}

impl ColumnarQueryResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            columns: vec![],
            data: vec![],
            row_count: 0,
            bytes_read: 0,
        }
    }

    /// Get column by name
    pub fn column(&self, name: &str) -> Option<&CoreTypedColumn> {
        self.columns
            .iter()
            .position(|c| c == name)
            .and_then(|idx| self.data.get(idx))
    }

    /// Get column index by name
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Number of rows
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of columns
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Total memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.data.iter().map(|c| c.memory_size()).sum()
    }

    /// Sum of an i64 column (SIMD-optimized)
    pub fn sum_i64(&self, column: &str) -> Option<i64> {
        self.column(column).map(|c| c.sum_i64())
    }

    /// Sum of an f64 column (SIMD-optimized)
    pub fn sum_f64(&self, column: &str) -> Option<f64> {
        self.column(column).map(|c| c.sum_f64())
    }

    /// Get column statistics (min, max, null count)
    pub fn column_stats(&self, column: &str) -> Option<&sochdb_core::columnar::ColumnStats> {
        self.column(column).map(|c| c.stats())
    }

    /// Convert to TOON format (token-efficient)
    pub fn to_toon(&self) -> String {
        if self.row_count == 0 {
            return "[]".to_string();
        }

        let n = self.row_count;
        let cols = self.columns.join(",");

        // Build rows from columns
        let mut rows_str = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<String> = self
                .data
                .iter()
                .map(|col| format_columnar_value(col, i))
                .collect();
            rows_str.push(row.join(","));
        }

        format!("result[{}]{{{}}}:{}", n, cols, rows_str.join(";"))
    }
}

/// Format a single value from a TypedColumn at index
fn format_columnar_value(col: &CoreTypedColumn, idx: usize) -> String {
    match col {
        CoreTypedColumn::Int64 {
            values, validity, ..
        } => {
            if validity.is_valid(idx) && idx < values.len() {
                values[idx].to_string()
            } else {
                "∅".to_string()
            }
        }
        CoreTypedColumn::UInt64 {
            values, validity, ..
        } => {
            if validity.is_valid(idx) && idx < values.len() {
                values[idx].to_string()
            } else {
                "∅".to_string()
            }
        }
        CoreTypedColumn::Float64 {
            values, validity, ..
        } => {
            if validity.is_valid(idx) && idx < values.len() {
                format!("{:.6}", values[idx])
            } else {
                "∅".to_string()
            }
        }
        CoreTypedColumn::Text {
            offsets,
            data,
            validity,
            ..
        } => {
            if validity.is_valid(idx) && idx + 1 < offsets.len() {
                let start = offsets[idx] as usize;
                let end = offsets[idx + 1] as usize;
                std::str::from_utf8(&data[start..end])
                    .map(|s| {
                        if s.contains(',') || s.contains(';') {
                            format!("\"{}\"", s.replace('"', "\\\""))
                        } else {
                            s.to_string()
                        }
                    })
                    .unwrap_or_else(|_| "∅".to_string())
            } else {
                "∅".to_string()
            }
        }
        CoreTypedColumn::Binary {
            offsets,
            data,
            validity,
            ..
        } => {
            if validity.is_valid(idx) && idx + 1 < offsets.len() {
                let start = offsets[idx] as usize;
                let end = offsets[idx + 1] as usize;
                format!("b64:{}", base64_encode(&data[start..end]))
            } else {
                "∅".to_string()
            }
        }
        CoreTypedColumn::Bool {
            values,
            validity,
            len,
            ..
        } => {
            if validity.is_valid(idx) && idx < *len {
                let word = idx / 64;
                let bit = idx % 64;
                if (values[word] >> bit) & 1 == 1 {
                    "T"
                } else {
                    "F"
                }
                .to_string()
            } else {
                "∅".to_string()
            }
        }
    }
}

/// Vector search result
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: u64,
    pub distance: f32,
    pub metadata: Option<HashMap<String, SochValue>>,
}

/// The SochDB Database Kernel
///
/// This is the shared core used by both embedded (`SochConnection`) and
/// server (`sochdb-server`) modes. It owns all storage, catalog, and
/// indexing components.
///
/// # Thread Safety
///
/// The Database is fully thread-safe via internal synchronization:
/// - Multiple readers can operate concurrently (MVCC snapshots)
/// - Writers coordinate through WAL and group commit
/// - All state is behind Arc/RwLock for shared access
///
/// # Example
///
/// ```ignore
/// // Open a database (SQLite-style)
/// let db = Database::open("./my_data")?;
///
/// // Begin a transaction
/// let txn = db.begin_transaction()?;
///
/// // Write data
/// db.put(txn, b"user:1:name", b"Alice")?;
///
/// // Commit
/// db.commit(txn)?;
/// ```
#[allow(dead_code)]
pub struct Database {
    /// Path to database directory
    path: PathBuf,
    /// Durable storage layer (WAL + MVCC + memtable)
    storage: Arc<DurableStorage>,
    /// Schema catalog
    catalog: Arc<RwLock<Catalog>>,
    /// Registered table schemas (name -> schema) - lock-free for reads
    tables: DashMap<String, TableSchema>,
    /// Cached packed schemas for fast insert (name -> packed schema)
    packed_schemas: DashMap<String, PackedTableSchema>,
    /// Per-table index policy registry
    index_registry: Arc<TableIndexRegistry>,
    /// Configuration
    config: DatabaseConfig,
    /// Statistics
    stats: DatabaseStats,
    /// Shutdown flag
    shutdown: AtomicU64,
}

/// Database statistics
struct DatabaseStats {
    transactions_started: AtomicU64,
    transactions_committed: AtomicU64,
    transactions_aborted: AtomicU64,
    queries_executed: AtomicU64,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
}

impl DatabaseStats {
    fn new() -> Self {
        Self {
            transactions_started: AtomicU64::new(0),
            transactions_committed: AtomicU64::new(0),
            transactions_aborted: AtomicU64::new(0),
            queries_executed: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }
}

/// Public statistics snapshot
#[derive(Debug, Clone)]
pub struct Stats {
    pub transactions_started: u64,
    pub transactions_committed: u64,
    pub transactions_aborted: u64,
    pub queries_executed: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
}

impl Database {
    /// Open or create a database at the given path.
    ///
    /// This is the primary entry point, similar to `sqlite3_open()`.
    /// If the database exists, it will be opened and WAL recovery performed.
    /// If it doesn't exist, a new database will be created.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for the database files
    ///
    /// # Returns
    ///
    /// An `Arc<Database>` that can be shared across threads and connections.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Arc<Self>> {
        Self::open_with_config(path, DatabaseConfig::default())
    }

    /// Open without locking (for testing crash recovery scenarios)
    ///
    /// # Safety
    /// This should ONLY be used in tests that simulate crashes by forgetting
    /// the storage instance. In production, always use `open()`.
    #[cfg(test)]
    pub fn open_without_lock<P: AsRef<Path>>(path: P) -> Result<Arc<Self>> {
        let path = path.as_ref().to_path_buf();
        let config = DatabaseConfig::default();

        let storage = Arc::new(DurableStorage::open_without_lock(&path)?);

        let index_registry = Arc::new(TableIndexRegistry::with_default_policy(
            config.default_index_policy,
        ));

        let db = Arc::new(Self {
            path: path.clone(),
            storage,
            catalog: Arc::new(RwLock::new(Catalog::new("sochdb"))),
            tables: DashMap::new(),
            packed_schemas: DashMap::new(),
            index_registry,
            config,
            stats: DatabaseStats::new(),
            shutdown: AtomicU64::new(0),
        });

        db.recover()?;
        Ok(db)
    }

    /// Open with custom configuration
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: DatabaseConfig) -> Result<Arc<Self>> {
        let path = path.as_ref().to_path_buf();

        // Use IndexPolicy-based storage configuration for automatic memtable selection
        // This derives ordered index and memtable type from the policy
        let storage = Arc::new(DurableStorage::open_with_policy(
            &path,
            config.default_index_policy,
            config.group_commit,
        )?);

        // Create index registry with default policy from config
        let index_registry = Arc::new(TableIndexRegistry::with_default_policy(
            config.default_index_policy,
        ));

        let db = Arc::new(Self {
            path: path.clone(),
            storage,
            catalog: Arc::new(RwLock::new(Catalog::new("sochdb"))),
            tables: DashMap::new(),
            packed_schemas: DashMap::new(),
            index_registry,
            config,
            stats: DatabaseStats::new(),
            shutdown: AtomicU64::new(0),
        });

        // Perform crash recovery if needed
        db.recover()?;

        Ok(db)
    }

    /// Perform crash recovery
    fn recover(&self) -> Result<RecoveryStats> {
        self.storage.recover()
    }

    /// Get database path
    pub fn path(&self) -> &Path {
        &self.path
    }

    // =========================================================================
    // Transaction API
    // =========================================================================

    /// Begin a new transaction
    pub fn begin_transaction(&self) -> Result<TxnHandle> {
        self.stats
            .transactions_started
            .fetch_add(1, Ordering::Relaxed);
        let txn_id = self.storage.begin_transaction()?;

        // Get snapshot timestamp from MVCC
        // For now, use txn_id as a proxy (the real snapshot_ts is managed internally)
        Ok(TxnHandle {
            txn_id,
            snapshot_ts: txn_id,
        })
    }

    /// Begin a read-only transaction (optimized: no SSI tracking)
    ///
    /// Read-only transactions skip SSI read tracking, reducing overhead
    /// from ~82ns to ~32ns per read (2.6x faster).
    ///
    /// Use this for:
    /// - SELECT queries that don't modify data
    /// - Analytics and reporting queries
    /// - Snapshot reads for backup
    pub fn begin_read_only(&self) -> Result<TxnHandle> {
        self.stats
            .transactions_started
            .fetch_add(1, Ordering::Relaxed);
        let txn_id = self.storage.begin_with_mode(TransactionMode::ReadOnly)?;
        Ok(TxnHandle {
            txn_id,
            snapshot_ts: txn_id,
        })
    }

    /// Begin a write-only transaction (optimized: no read tracking)
    ///
    /// Write-only transactions skip read tracking, improving insert
    /// throughput for bulk loading scenarios.
    ///
    /// Use this for:
    /// - Bulk data imports
    /// - Append-only logging tables
    /// - ETL pipelines
    pub fn begin_write_only(&self) -> Result<TxnHandle> {
        self.stats
            .transactions_started
            .fetch_add(1, Ordering::Relaxed);
        let txn_id = self.storage.begin_with_mode(TransactionMode::WriteOnly)?;
        Ok(TxnHandle {
            txn_id,
            snapshot_ts: txn_id,
        })
    }

    /// Commit a transaction
    pub fn commit(&self, txn: TxnHandle) -> Result<u64> {
        self.stats
            .transactions_committed
            .fetch_add(1, Ordering::Relaxed);
        self.storage.commit(txn.txn_id)
    }

    /// Abort a transaction
    pub fn abort(&self, txn: TxnHandle) -> Result<()> {
        self.stats
            .transactions_aborted
            .fetch_add(1, Ordering::Relaxed);
        self.storage.abort(txn.txn_id)
    }

    // =========================================================================
    // Per-Table Index Policy API
    // =========================================================================

    /// Configure index policy for a table
    ///
    /// This allows fine-grained control over write/scan trade-offs per table:
    ///
    /// | Policy         | Insert Cost | Scan Cost      | Use Case              |
    /// |----------------|-------------|----------------|------------------------|
    /// | WriteOptimized | O(1)        | O(N)           | High-write, rare scan  |
    /// | Balanced       | O(1) amort  | O(output+logK) | Mixed OLTP            |
    /// | ScanOptimized  | O(log N)    | O(logN + K)    | Analytics, range query |
    /// | AppendOnly     | O(1)        | O(N)           | Time-series logs       |
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Fast inserts for logs table (no ordered index overhead)
    /// db.set_table_index_policy("logs", IndexPolicy::WriteOptimized);
    ///
    /// // Efficient range scans for analytics table
    /// db.set_table_index_policy("analytics", IndexPolicy::ScanOptimized);
    ///
    /// // Balanced for OLTP tables
    /// db.set_table_index_policy("users", IndexPolicy::Balanced);
    /// ```
    pub fn set_table_index_policy(&self, table: &str, policy: IndexPolicy) {
        self.index_registry.configure_table(
            TableIndexConfig::new(table, policy)
        );
    }

    /// Get the index policy for a table
    pub fn get_table_index_policy(&self, table: &str) -> IndexPolicy {
        self.index_registry.get_policy(table)
    }

    /// Get the index registry for advanced configuration
    pub fn index_registry(&self) -> &Arc<TableIndexRegistry> {
        &self.index_registry
    }

    // =========================================================================
    // Key-Value API (Low-level)
    // =========================================================================

    /// Put a key-value pair
    pub fn put(&self, txn: TxnHandle, key: &[u8], value: &[u8]) -> Result<()> {
        self.stats
            .bytes_written
            .fetch_add((key.len() + value.len()) as u64, Ordering::Relaxed);
        // Use write_refs to avoid unnecessary allocations
        self.storage.write_refs(txn.txn_id, key, value)
    }

    /// Batch put multiple key-value pairs with reduced overhead
    ///
    /// This amortizes per-operation costs over the entire batch:
    /// - Single DashMap lookup
    /// - Batch MVCC tracking
    /// - Batch memtable writes
    ///
    /// For 100+ entries, this is 2-3x faster than individual puts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let writes: Vec<(&[u8], &[u8])> = vec![
    ///     (b"key1", b"value1"),
    ///     (b"key2", b"value2"),
    ///     (b"key3", b"value3"),
    /// ];
    /// db.put_batch(txn, &writes)?;
    /// ```
    pub fn put_batch(&self, txn: TxnHandle, writes: &[(&[u8], &[u8])]) -> Result<()> {
        let bytes: u64 = writes
            .iter()
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum();
        self.stats.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        self.storage.write_batch_refs(txn.txn_id, writes)
    }

    /// Get a value by key
    pub fn get(&self, txn: TxnHandle, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let result = self.storage.read(txn.txn_id, key)?;
        if let Some(ref data) = result {
            self.stats
                .bytes_read
                .fetch_add(data.len() as u64, Ordering::Relaxed);
        }
        Ok(result)
    }

    /// Delete a key
    pub fn delete(&self, txn: TxnHandle, key: &[u8]) -> Result<()> {
        self.storage.delete(txn.txn_id, key.to_vec())
    }

    /// Minimum prefix length for scan operations.
    /// Prevents expensive full-table scans by requiring a meaningful prefix.
    pub const MIN_SCAN_PREFIX_LEN: usize = 2;

    /// Scan keys with a prefix (enforces minimum prefix length for safety).
    ///
    /// # Prefix Safety
    /// 
    /// To prevent accidental full-table scans, this method requires a minimum
    /// prefix length of 2 bytes. Use `scan_unchecked` for internal operations
    /// that need empty/short prefixes.
    ///
    /// # Errors
    ///
    /// Returns `SochDBError::InvalidInput` if prefix is too short.
    pub fn scan(&self, txn: TxnHandle, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        if prefix.len() < Self::MIN_SCAN_PREFIX_LEN {
            return Err(SochDBError::InvalidArgument(format!(
                "Prefix too short: {} bytes (minimum {} required). \
                 Use scan_unchecked() for unrestricted scans.",
                prefix.len(),
                Self::MIN_SCAN_PREFIX_LEN
            )));
        }
        self.scan_unchecked(txn, prefix)
    }

    /// Scan keys with a prefix without length validation.
    ///
    /// # Warning
    ///
    /// This method allows empty/short prefixes which can cause expensive
    /// full-table scans. Use `scan()` unless you specifically need unrestricted
    /// prefix access for internal operations.
    pub fn scan_unchecked(&self, txn: TxnHandle, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let results = self.storage.scan(txn.txn_id, prefix)?;
        let bytes: u64 = results
            .iter()
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum();
        self.stats.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        Ok(results)
    }

    /// Scan keys in range
    pub fn scan_range(
        &self,
        txn: TxnHandle,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let results = self.storage.scan_range(txn.txn_id, start, end)?;
        let bytes: u64 = results
            .iter()
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum();
        self.stats.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        Ok(results)
    }

    /// Streaming scan for very large result sets
    /// 
    /// Returns an iterator that yields (key, value) pairs without
    /// materializing the entire result set. Use this for large scans
    /// where memory efficiency is important.
    /// 
    /// ## Performance
    /// 
    /// - Memory: O(1) per iteration vs O(N) for scan_range
    /// - Latency: First result available immediately vs waiting for all results
    /// - Throughput: Slightly lower due to per-item overhead
    /// 
    /// ## Usage
    /// 
    /// ```ignore
    /// for result in db.scan_range_iter(txn, b"start", b"end") {
    ///     let (key, value) = result?;
    ///     // Process immediately - no need to wait for all results
    /// }
    /// ```
    pub fn scan_range_iter<'a>(
        &'a self,
        txn: TxnHandle,
        start: &'a [u8],
        end: &'a [u8],
    ) -> impl Iterator<Item = Result<(Vec<u8>, Vec<u8>)>> + 'a {
        let stats = &self.stats;
        self.storage
            .scan_range_iter(txn.txn_id, start, end)
            .map(move |item| {
                stats.bytes_read.fetch_add(
                    (item.0.len() + item.1.len()) as u64,
                    Ordering::Relaxed,
                );
                Ok(item)
            })
    }

    /// Flush memtable to WAL/Disk
    pub fn flush(&self) -> Result<()> {
        self.storage.fsync()
    }

    // =========================================================================
    // Path-Native API (SochDB's differentiator)
    // =========================================================================

    /// Get storage statistics
    pub fn storage_stats(&self) -> crate::durable_storage::StorageStats {
        self.storage.stats()
    }

    /// Put a value at a path
    ///
    /// Path format: "collection/doc_id/field" or "table.row_id.column"
    /// Resolution is O(|path|), not O(log N) like B-tree.
    pub fn put_path(&self, txn: TxnHandle, path: &str, value: &[u8]) -> Result<()> {
        self.put(txn, path.as_bytes(), value)
    }

    /// Get a value at a path
    pub fn get_path(&self, txn: TxnHandle, path: &str) -> Result<Option<Vec<u8>>> {
        self.get(txn, path.as_bytes())
    }

    /// Delete at a path
    pub fn delete_path(&self, txn: TxnHandle, path: &str) -> Result<()> {
        self.delete(txn, path.as_bytes())
    }

    /// Scan a path prefix
    ///
    /// Returns all key-value pairs where key starts with prefix.
    /// Useful for: "users/123/" -> all fields of user 123
    pub fn scan_path(&self, txn: TxnHandle, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        self.stats.queries_executed.fetch_add(1, Ordering::Relaxed);

        let results = self.scan(txn, prefix.as_bytes())?;

        Ok(results
            .into_iter()
            .filter_map(|(k, v)| String::from_utf8(k).ok().map(|path| (path, v)))
            .collect())
    }

    // =========================================================================
    // Query API
    // =========================================================================

    /// Execute a path query and return results
    ///
    /// This is the main query interface for LLM context retrieval.
    /// Supports:
    /// - Path prefix matching
    /// - Column projection (for I/O reduction)
    /// - Limit/offset
    pub fn query(&self, txn: TxnHandle, path_prefix: &str) -> QueryBuilder<'_> {
        QueryBuilder::new(self, txn, path_prefix.to_string())
    }

    // =========================================================================
    // Table API (Higher-level abstraction)
    // =========================================================================

    /// Register a table schema
    pub fn register_table(&self, schema: TableSchema) -> Result<()> {
        if self.tables.contains_key(&schema.name) {
            return Err(SochDBError::InvalidArgument(format!(
                "Table '{}' already exists",
                schema.name
            )));
        }
        // Cache the packed schema for fast inserts
        let packed_schema = Self::to_packed_schema(&schema);
        self.packed_schemas
            .insert(schema.name.clone(), packed_schema);
        self.tables.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Get table schema
    pub fn get_table_schema(&self, name: &str) -> Option<TableSchema> {
        self.tables.get(name).map(|s| s.clone())
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<String> {
        self.tables.iter().map(|e| e.key().clone()).collect()
    }
    /// Convert TableSchema to PackedTableSchema for efficient storage
    fn to_packed_schema(schema: &TableSchema) -> PackedTableSchema {
        let columns = schema
            .columns
            .iter()
            .map(|col| PackedColumnDef {
                name: col.name.clone(),
                col_type: match col.col_type {
                    ColumnType::Int64 => PackedColumnType::Int64,
                    ColumnType::UInt64 => PackedColumnType::UInt64,
                    ColumnType::Float64 => PackedColumnType::Float64,
                    ColumnType::Text => PackedColumnType::Text,
                    ColumnType::Binary => PackedColumnType::Binary,
                    ColumnType::Bool => PackedColumnType::Bool,
                },
                nullable: col.nullable,
            })
            .collect();

        PackedTableSchema::new(&schema.name, columns)
    }

    /// Insert a row into a table
    ///
    /// Uses packed row format: stores entire row as single key-value pair.
    /// This reduces write amplification from 4× to 1× for a 4-column table.
    ///
    /// # Performance
    /// - Before: 4 columns × (WAL entry + MVCC version) = 4 writes
    /// - After: 1 packed row = 1 write
    /// - Improvement: ~4× fewer WAL entries, ~48% less I/O overhead
    pub fn insert_row(
        &self,
        txn: TxnHandle,
        table: &str,
        row_id: u64,
        values: &HashMap<String, SochValue>,
    ) -> Result<()> {
        // Use cached packed schema - single DashMap lookup, no clone
        let packed_schema = self
            .packed_schemas
            .get(table)
            .ok_or_else(|| SochDBError::InvalidArgument(format!("Table '{}' not found", table)))?;

        // Pack the row using cached schema
        let packed_row = PackedRow::pack(&packed_schema, values);

        // Build key using KeyBuffer - optimized stack allocation (~12-15ns vs ~30-35ns for write!())
        let key = KeyBuffer::format_row_key(table, row_id);

        self.put(txn, key.as_bytes(), packed_row.as_bytes())?;

        Ok(())
    }

    /// Read a row from a table
    ///
    /// Reads packed row and extracts requested columns in O(k) time.
    /// Column projection happens in memory, not storage - all columns are fetched.
    pub fn read_row(
        &self,
        txn: TxnHandle,
        table: &str,
        row_id: u64,
        columns: Option<&[&str]>,
    ) -> Result<Option<HashMap<String, SochValue>>> {
        let schema = self
            .tables
            .get(table)
            .ok_or_else(|| SochDBError::InvalidArgument(format!("Table '{}' not found", table)))?;

        // Read the packed row with a single key lookup using KeyBuffer
        let key = KeyBuffer::format_row_key(table, row_id);
        let bytes = match self.get(txn, key.as_bytes())? {
            Some(b) => b,
            None => return Ok(None),
        };

        // Use cached packed schema
        let packed_schema = self
            .packed_schemas
            .get(table)
            .ok_or_else(|| SochDBError::Internal("Packed schema not found".into()))?;
        let packed_row = PackedRow::from_bytes(bytes, packed_schema.num_columns())?;

        // Determine which columns to return
        let cols_to_read: Vec<&str> = match columns {
            Some(c) => c.to_vec(),
            None => schema.columns.iter().map(|c| c.name.as_str()).collect(),
        };

        let mut row = HashMap::new();
        for col_name in cols_to_read {
            if let Some(idx) = packed_schema.column_index(col_name)
                && let Some(col_def) = packed_schema.column(idx)
                && let Some(value) = packed_row.get_column(idx, col_def.col_type)
            {
                row.insert(col_name.to_string(), value);
            }
        }

        Ok(Some(row))
    }

    /// Insert multiple rows efficiently in a batch
    ///
    /// This method accumulates all rows and writes them with fewer WAL syncs.
    /// Ideal for bulk loading scenarios.
    ///
    /// # Performance
    /// - Uses group commit to batch fsync operations
    /// - Expected throughput: 500K-1M rows/sec depending on row size
    pub fn insert_rows_batch(
        &self,
        txn: TxnHandle,
        table: &str,
        rows: &[(u64, HashMap<String, SochValue>)],
    ) -> Result<usize> {
        // Use cached packed schema
        let packed_schema = self
            .packed_schemas
            .get(table)
            .ok_or_else(|| SochDBError::InvalidArgument(format!("Table '{}' not found", table)))?;

        let mut count = 0;

        for (row_id, values) in rows {
            // Pack and write using KeyBuffer for efficient key construction
            let packed_row = PackedRow::pack(&packed_schema, values);
            let key = KeyBuffer::format_row_key(table, *row_id);
            self.put(txn, key.as_bytes(), packed_row.as_bytes())?;
            count += 1;
        }

        Ok(count)
    }

    /// Ultra-fast raw put - bypasses all validation
    ///
    /// Use when you've already validated the data and just need speed.
    /// This is ~10× faster than insert_row() for bulk inserts.
    #[inline]
    pub fn put_raw(&self, txn: TxnHandle, key: &[u8], value: &[u8]) -> Result<()> {
        self.storage.write_refs(txn.txn_id, key, value)
    }

    /// Zero-allocation insert - fastest path for bulk inserts
    ///
    /// Takes values as a slice in schema column order, avoiding HashMap overhead.
    ///
    /// # Arguments
    /// * `txn` - Transaction handle
    /// * `table` - Table name
    /// * `row_id` - Row identifier
    /// * `values` - Values in schema column order (None = NULL)
    ///
    /// # Performance
    /// - Eliminates ~6 allocations per row vs insert_row()
    /// - Expected: 1.2M-1.5M inserts/sec
    ///
    /// # Example
    /// ```ignore
    /// let values: &[Option<&SochValue>] = &[
    ///     Some(&SochValue::Int(1)),
    ///     Some(&SochValue::Text("Alice".into())),
    ///     None, // NULL
    /// ];
    /// db.insert_row_slice(txn, "users", 1, values)?;
    /// ```
    #[inline]
    pub fn insert_row_slice(
        &self,
        txn: TxnHandle,
        table: &str,
        row_id: u64,
        values: &[Option<&SochValue>],
    ) -> Result<()> {
        // Use cached packed schema - single DashMap lookup
        let packed_schema = self
            .packed_schemas
            .get(table)
            .ok_or_else(|| SochDBError::InvalidArgument(format!("Table '{}' not found", table)))?;

        // Validate column count matches
        if values.len() != packed_schema.num_columns() {
            return Err(SochDBError::InvalidArgument(format!(
                "Expected {} columns, got {}",
                packed_schema.num_columns(),
                values.len()
            )));
        }

        // Pack using zero-allocation path
        let packed_row = PackedRow::pack_slice(&packed_schema, values);

        // Build key using KeyBuffer - optimized stack allocation (~12-15ns vs ~30-35ns for write!())
        let key = KeyBuffer::format_row_key(table, row_id);

        self.put(txn, key.as_bytes(), packed_row.as_bytes())?;
        Ok(())
    }

    // =========================================================================
    // Maintenance
    // =========================================================================

    /// Force fsync to disk
    pub fn fsync(&self) -> Result<()> {
        self.storage.fsync()
    }

    /// Create a checkpoint
    pub fn checkpoint(&self) -> Result<u64> {
        self.storage.checkpoint()
    }

    /// Run garbage collection
    pub fn gc(&self) -> usize {
        self.storage.gc()
    }

    /// Get database statistics
    pub fn stats(&self) -> Stats {
        Stats {
            transactions_started: self.stats.transactions_started.load(Ordering::Relaxed),
            transactions_committed: self.stats.transactions_committed.load(Ordering::Relaxed),
            transactions_aborted: self.stats.transactions_aborted.load(Ordering::Relaxed),
            queries_executed: self.stats.queries_executed.load(Ordering::Relaxed),
            bytes_written: self.stats.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.stats.bytes_read.load(Ordering::Relaxed),
        }
    }

    /// Shutdown the database gracefully
    pub fn shutdown(&self) -> Result<()> {
        if self.shutdown.swap(1, Ordering::SeqCst) == 1 {
            return Ok(()); // Already shutting down
        }

        // Flush any pending writes
        self.fsync()?;

        // Create clean shutdown marker
        let marker = self.path.join(".clean_shutdown");
        std::fs::write(&marker, b"ok")?;

        Ok(())
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Try graceful shutdown if not already done
        if self.shutdown.load(Ordering::SeqCst) == 0 {
            let _ = self.fsync();
            let marker = self.path.join(".clean_shutdown");
            let _ = std::fs::write(&marker, b"ok");
        }
    }
}

/// Query builder for fluent query construction
pub struct QueryBuilder<'a> {
    db: &'a Database,
    txn: TxnHandle,
    path_prefix: String,
    columns: Option<Vec<String>>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl<'a> QueryBuilder<'a> {
    fn new(db: &'a Database, txn: TxnHandle, path_prefix: String) -> Self {
        Self {
            db,
            txn,
            path_prefix,
            columns: None,
            limit: None,
            offset: None,
        }
    }

    /// Select specific columns (for I/O reduction)
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = Some(cols.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Skip results
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Execute the query
    ///
    /// Scans packed rows and unpacks them. Each key is "table/row_id" pointing to a packed row.
    pub fn execute(self) -> Result<QueryResult> {
        self.db
            .stats
            .queries_executed
            .fetch_add(1, Ordering::Relaxed);

        // Get schema for the table if we're querying a table
        let table_name = self
            .path_prefix
            .split('/')
            .next()
            .unwrap_or(&self.path_prefix);
        let schema = self.db.tables.get(table_name).map(|s| s.clone());

        // Scan the path prefix
        let results = self.db.scan_path(self.txn, &self.path_prefix)?;

        let mut rows: Vec<HashMap<String, SochValue>> = Vec::new();
        let mut bytes_read = 0usize;

        if let Some(ref schema) = schema {
            // We have a table schema - use cached packed schema
            let packed_schema = self
                .db
                .packed_schemas
                .get(table_name)
                .map(|ps| ps.clone())
                .unwrap_or_else(|| Database::to_packed_schema(schema));

            for (path, value_bytes) in results {
                // Parse path: table/row_id
                let parts: Vec<&str> = path.split('/').collect();
                if parts.len() == 2 {
                    // This is a packed row
                    bytes_read += value_bytes.len();

                    if let Ok(packed_row) =
                        PackedRow::from_bytes(value_bytes, packed_schema.num_columns())
                    {
                        // Unpack all columns or just requested columns
                        let mut row = HashMap::new();

                        if let Some(ref cols) = self.columns {
                            // Only extract requested columns
                            for col_name in cols {
                                if let Some(idx) = packed_schema.column_index(col_name)
                                    && let Some(col_def) = packed_schema.column(idx)
                                    && let Some(value) =
                                        packed_row.get_column(idx, col_def.col_type)
                                {
                                    row.insert(col_name.clone(), value);
                                }
                            }
                        } else {
                            // Extract all columns
                            row = packed_row.unpack(&packed_schema);
                        }

                        if !row.is_empty() {
                            rows.push(row);
                        }
                    }
                }
            }
        } else {
            // Fallback: no schema, try legacy column-per-key format
            let mut rows_map: HashMap<String, HashMap<String, SochValue>> = HashMap::new();

            for (path, value_bytes) in results {
                let parts: Vec<&str> = path.split('/').collect();
                if parts.len() >= 3 {
                    let row_key = format!("{}/{}", parts[0], parts[1]);
                    let col_name = parts[2..].join("/");

                    if let Some(ref cols) = self.columns
                        && !cols.contains(&col_name)
                    {
                        continue;
                    }

                    bytes_read += value_bytes.len();
                    let row = rows_map.entry(row_key).or_default();
                    row.insert(col_name, deserialize_value(&value_bytes));
                }
            }

            rows = rows_map.into_values().collect();
        }

        // Apply offset
        if let Some(offset) = self.offset {
            rows = rows.into_iter().skip(offset).collect();
        }

        // Apply limit
        if let Some(limit) = self.limit {
            rows.truncate(limit);
        }

        // Collect column names
        let columns: Vec<String> = self.columns.unwrap_or_else(|| {
            rows.iter()
                .flat_map(|r| r.keys().cloned())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect()
        });

        Ok(QueryResult {
            columns,
            rows_scanned: rows.len(),
            bytes_read,
            rows,
        })
    }

    /// Execute and return TOON format (for LLM efficiency)
    pub fn to_toon(self) -> Result<String> {
        let result = self.execute()?;
        Ok(result.to_toon())
    }

    /// Execute with lazy iteration - avoids materializing all rows
    ///
    /// Returns an iterator over rows as `Vec<SochValue>` in schema column order.
    /// This is more memory-efficient than `execute()` for large result sets.
    ///
    /// # Performance
    /// - No upfront materialization of all rows
    /// - ~40% less memory for large result sets
    /// - Ideal for streaming to network or aggregations
    ///
    /// # Example
    /// ```ignore
    /// for row_result in db.query(txn, "users").execute_iter()? {
    ///     let row = row_result?;
    ///     // row is Vec<SochValue> in column order
    /// }
    /// ```
    pub fn execute_iter(self) -> Result<QueryRowIterator> {
        self.db
            .stats
            .queries_executed
            .fetch_add(1, Ordering::Relaxed);

        let table_name = self
            .path_prefix
            .split('/')
            .next()
            .unwrap_or(&self.path_prefix)
            .to_string();

        // Get packed schema (clone needed for iterator ownership)
        let packed_schema = self.db.packed_schemas.get(&table_name).map(|ps| ps.clone());

        // Scan the path prefix
        let results = self.db.scan_path(self.txn, &self.path_prefix)?;

        Ok(QueryRowIterator {
            results: results.into_iter(),
            packed_schema,
            columns: self.columns,
            offset: self.offset.unwrap_or(0),
            limit: self.limit,
            yielded: 0,
            skipped: 0,
        })
    }

    /// Execute and return columnar (SIMD-friendly) result format
    ///
    /// Instead of row-oriented `Vec<HashMap<String, SochValue>>`, returns
    /// column-oriented `Vec<TypedColumn>` for vectorized operations.
    ///
    /// ## Performance Benefits
    ///
    /// - SIMD: Aggregate operations (sum, avg) use vectorized instructions
    /// - Cache: Sequential access maximizes L1/L2 hits
    /// - Memory: ~30% less overhead than row-based format
    /// - Analytics: Ideal for ML preprocessing and statistics
    ///
    /// ## Example
    ///
    /// ```ignore
    /// let result = db.query(txn, "users")
    ///     .columns(&["id", "score"])
    ///     .as_columnar()?;
    ///
    /// // SIMD-optimized sum
    /// let total = result.sum_i64("score").unwrap_or(0);
    ///
    /// // Direct column access
    /// if let Some(scores) = result.column("score") {
    ///     for i in 0..scores.len() {
    ///         if let Some(v) = scores.get_i64(i) {
    ///             println!("Score: {}", v);
    ///         }
    ///     }
    /// }
    /// ```
    pub fn as_columnar(self) -> Result<ColumnarQueryResult> {
        self.db
            .stats
            .queries_executed
            .fetch_add(1, Ordering::Relaxed);

        let table_name = self
            .path_prefix
            .split('/')
            .next()
            .unwrap_or(&self.path_prefix);
        let schema = self.db.tables.get(table_name).map(|s| s.clone());

        // Get packed schema
        let packed_schema = match self.db.packed_schemas.get(table_name) {
            Some(ps) => ps.clone(),
            None => return Ok(ColumnarQueryResult::empty()),
        };

        // Determine columns to fetch
        let column_names: Vec<String> = self.columns.clone().unwrap_or_else(|| {
            schema
                .as_ref()
                .map(|s| s.columns.iter().map(|c| c.name.clone()).collect())
                .unwrap_or_default()
        });

        if column_names.is_empty() {
            return Ok(ColumnarQueryResult::empty());
        }

        // Initialize TypedColumns based on schema types
        let mut columns: Vec<CoreTypedColumn> = column_names
            .iter()
            .map(|col_name| {
                packed_schema
                    .column_index(col_name)
                    .and_then(|idx| packed_schema.column(idx))
                    .map(|col_def| match col_def.col_type {
                        PackedColumnType::Int64 => CoreTypedColumn::new_int64(),
                        PackedColumnType::UInt64 => CoreTypedColumn::new_uint64(),
                        PackedColumnType::Float64 => CoreTypedColumn::new_float64(),
                        PackedColumnType::Text => CoreTypedColumn::new_text(),
                        PackedColumnType::Binary => CoreTypedColumn::new_binary(),
                        PackedColumnType::Bool => CoreTypedColumn::new_bool(),
                        PackedColumnType::Null => CoreTypedColumn::new_text(), // Null column = fallback to text
                    })
                    .unwrap_or_else(CoreTypedColumn::new_text) // fallback
            })
            .collect();

        // Scan the path prefix
        let results = self.db.scan_path(self.txn, &self.path_prefix)?;

        let mut row_count = 0;
        let mut bytes_read = 0;
        let mut skipped = 0;

        for (path, value_bytes) in results {
            // Parse path: table/row_id
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() != 2 {
                continue;
            }

            // Apply offset
            if let Some(offset) = self.offset
                && skipped < offset
            {
                skipped += 1;
                continue;
            }

            // Apply limit
            if let Some(limit) = self.limit
                && row_count >= limit
            {
                break;
            }

            bytes_read += value_bytes.len();

            if let Ok(packed_row) = PackedRow::from_bytes(value_bytes, packed_schema.num_columns())
            {
                // Extract each column and push to corresponding TypedColumn
                for (col_idx, col_name) in column_names.iter().enumerate() {
                    if let Some(schema_idx) = packed_schema.column_index(col_name) {
                        if let Some(col_def) = packed_schema.column(schema_idx) {
                            let value = packed_row.get_column(schema_idx, col_def.col_type);
                            push_value_to_typed_column(&mut columns[col_idx], value);
                        } else {
                            push_null_to_typed_column(&mut columns[col_idx]);
                        }
                    } else {
                        push_null_to_typed_column(&mut columns[col_idx]);
                    }
                }
                row_count += 1;
            }
        }

        Ok(ColumnarQueryResult {
            columns: column_names,
            data: columns,
            row_count,
            bytes_read,
        })
    }
}

/// Lazy iterator over query results
///
/// Unpacks rows on-demand, avoiding upfront materialization.
pub struct QueryRowIterator {
    results: std::vec::IntoIter<(String, Vec<u8>)>,
    packed_schema: Option<PackedTableSchema>,
    columns: Option<Vec<String>>,
    offset: usize,
    limit: Option<usize>,
    yielded: usize,
    skipped: usize,
}

impl Iterator for QueryRowIterator {
    type Item = Result<Vec<SochValue>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check limit
        if let Some(limit) = self.limit
            && self.yielded >= limit
        {
            return None;
        }

        loop {
            let (path, value_bytes) = self.results.next()?;

            // Parse path: table/row_id
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() != 2 {
                continue; // Skip non-row entries
            }

            // Apply offset
            if self.skipped < self.offset {
                self.skipped += 1;
                continue;
            }

            if let Some(ref schema) = self.packed_schema {
                match PackedRow::from_bytes(value_bytes, schema.num_columns()) {
                    Ok(packed_row) => {
                        let row = if let Some(ref cols) = self.columns {
                            // Project specific columns
                            cols.iter()
                                .map(|col_name| {
                                    schema
                                        .column_index(col_name)
                                        .and_then(|idx| schema.column(idx))
                                        .and_then(|col_def| {
                                            packed_row.get_column(
                                                schema.column_index(col_name).unwrap(),
                                                col_def.col_type,
                                            )
                                        })
                                        .unwrap_or(SochValue::Null)
                                })
                                .collect()
                        } else {
                            // All columns in order
                            packed_row.unpack_to_vec(schema)
                        };

                        self.yielded += 1;
                        return Some(Ok(row));
                    }
                    Err(e) => return Some(Err(e)),
                }
            } else {
                // No schema - return raw bytes as binary
                self.yielded += 1;
                return Some(Ok(vec![SochValue::Binary(value_bytes)]));
            }
        }
    }
}

// Helper functions for serialization (kept for backward compatibility with legacy data)

#[allow(dead_code)]
fn serialize_value(value: &SochValue) -> Vec<u8> {
    // Simple serialization - in production use proper format
    match value {
        SochValue::Null => vec![0],
        SochValue::Int(i) => {
            let mut buf = vec![1];
            buf.extend_from_slice(&i.to_le_bytes());
            buf
        }
        SochValue::UInt(u) => {
            let mut buf = vec![2];
            buf.extend_from_slice(&u.to_le_bytes());
            buf
        }
        SochValue::Float(f) => {
            let mut buf = vec![3];
            buf.extend_from_slice(&f.to_le_bytes());
            buf
        }
        SochValue::Text(s) => {
            let mut buf = vec![4];
            buf.extend_from_slice(s.as_bytes());
            buf
        }
        SochValue::Bool(b) => vec![5, if *b { 1 } else { 0 }],
        SochValue::Binary(b) => {
            let mut buf = vec![6];
            buf.extend_from_slice(b);
            buf
        }
        _ => {
            // Fallback: serialize as text
            let s = format!("{:?}", value);
            let mut buf = vec![4];
            buf.extend_from_slice(s.as_bytes());
            buf
        }
    }
}

fn deserialize_value(bytes: &[u8]) -> SochValue {
    if bytes.is_empty() {
        return SochValue::Null;
    }

    match bytes[0] {
        0 => SochValue::Null,
        1 if bytes.len() >= 9 => {
            let i = i64::from_le_bytes(bytes[1..9].try_into().unwrap());
            SochValue::Int(i)
        }
        2 if bytes.len() >= 9 => {
            let u = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
            SochValue::UInt(u)
        }
        3 if bytes.len() >= 9 => {
            let f = f64::from_le_bytes(bytes[1..9].try_into().unwrap());
            SochValue::Float(f)
        }
        4 => {
            let s = String::from_utf8_lossy(&bytes[1..]).to_string();
            SochValue::Text(s)
        }
        5 if bytes.len() >= 2 => SochValue::Bool(bytes[1] != 0),
        6 => SochValue::Binary(bytes[1..].to_vec()),
        _ => {
            // Treat as text
            let s = String::from_utf8_lossy(bytes).to_string();
            SochValue::Text(s)
        }
    }
}

// ============================================================================
// Helper functions for columnar query result building
// ============================================================================

/// Push a SochValue into a TypedColumn
fn push_value_to_typed_column(col: &mut CoreTypedColumn, value: Option<SochValue>) {
    match value {
        None => push_null_to_typed_column(col),
        Some(v) => match (col, v) {
            (
                CoreTypedColumn::Int64 {
                    values,
                    validity,
                    stats,
                },
                SochValue::Int(i),
            ) => {
                values.push(i);
                validity.push(true);
                stats.update_i64(i);
            }
            (
                CoreTypedColumn::Int64 {
                    values,
                    validity,
                    stats,
                },
                SochValue::UInt(u),
            ) => {
                values.push(u as i64);
                validity.push(true);
                stats.update_i64(u as i64);
            }
            (
                CoreTypedColumn::UInt64 {
                    values,
                    validity,
                    stats,
                },
                SochValue::UInt(u),
            ) => {
                values.push(u);
                validity.push(true);
                stats.update_i64(u as i64);
            }
            (
                CoreTypedColumn::UInt64 {
                    values,
                    validity,
                    stats,
                },
                SochValue::Int(i),
            ) => {
                values.push(i as u64);
                validity.push(true);
                stats.update_i64(i);
            }
            (
                CoreTypedColumn::Float64 {
                    values,
                    validity,
                    stats,
                },
                SochValue::Float(f),
            ) => {
                values.push(f);
                validity.push(true);
                stats.update_f64(f);
            }
            (
                CoreTypedColumn::Float64 {
                    values,
                    validity,
                    stats,
                },
                SochValue::Int(i),
            ) => {
                values.push(i as f64);
                validity.push(true);
                stats.update_f64(i as f64);
            }
            (
                CoreTypedColumn::Text {
                    offsets,
                    data,
                    validity,
                    stats,
                },
                SochValue::Text(s),
            ) => {
                data.extend_from_slice(s.as_bytes());
                offsets.push(data.len() as u32);
                validity.push(true);
                stats.row_count += 1;
            }
            (
                CoreTypedColumn::Binary {
                    offsets,
                    data,
                    validity,
                    stats,
                },
                SochValue::Binary(b),
            ) => {
                data.extend_from_slice(&b);
                offsets.push(data.len() as u32);
                validity.push(true);
                stats.row_count += 1;
            }
            (
                CoreTypedColumn::Bool {
                    values,
                    validity,
                    stats,
                    len,
                },
                SochValue::Bool(b),
            ) => {
                let idx = *len;
                *len += 1;
                let num_words = (*len).div_ceil(64);
                while values.len() < num_words {
                    values.push(0);
                }
                if b {
                    let word = idx / 64;
                    let bit = idx % 64;
                    values[word] |= 1 << bit;
                }
                validity.push(true);
                stats.row_count += 1;
            }
            // Type mismatch - push as null
            (col, _) => push_null_to_typed_column(col),
        },
    }
}

/// Push a null value into a TypedColumn
fn push_null_to_typed_column(col: &mut CoreTypedColumn) {
    match col {
        CoreTypedColumn::Int64 {
            values,
            validity,
            stats,
        } => {
            values.push(0);
            validity.push(false);
            stats.update_null();
        }
        CoreTypedColumn::UInt64 {
            values,
            validity,
            stats,
        } => {
            values.push(0);
            validity.push(false);
            stats.update_null();
        }
        CoreTypedColumn::Float64 {
            values,
            validity,
            stats,
        } => {
            values.push(0.0);
            validity.push(false);
            stats.update_null();
        }
        CoreTypedColumn::Text {
            offsets,
            data: _,
            validity,
            stats,
        } => {
            offsets.push(offsets.last().copied().unwrap_or(0));
            validity.push(false);
            stats.update_null();
        }
        CoreTypedColumn::Binary {
            offsets,
            data: _,
            validity,
            stats,
        } => {
            offsets.push(offsets.last().copied().unwrap_or(0));
            validity.push(false);
            stats.update_null();
        }
        CoreTypedColumn::Bool {
            values,
            validity,
            stats,
            len,
        } => {
            *len += 1;
            let num_words = (*len).div_ceil(64);
            while values.len() < num_words {
                values.push(0);
            }
            validity.push(false);
            stats.update_null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_database_open_close() {
        let dir = tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        // Should be able to begin a transaction
        let txn = db.begin_transaction().unwrap();
        assert!(txn.txn_id > 0);

        db.abort(txn).unwrap();
        db.shutdown().unwrap();
    }

    #[test]
    fn test_database_put_get() {
        let dir = tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        let txn = db.begin_transaction().unwrap();
        db.put(txn, b"key1", b"value1").unwrap();

        let val = db.get(txn, b"key1").unwrap();
        assert_eq!(val, Some(b"value1".to_vec()));

        db.commit(txn).unwrap();

        // New transaction should see committed data
        let txn2 = db.begin_transaction().unwrap();
        let val = db.get(txn2, b"key1").unwrap();
        assert_eq!(val, Some(b"value1".to_vec()));
        db.abort(txn2).unwrap();
    }

    #[test]
    fn test_database_path_api() {
        let dir = tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        let txn = db.begin_transaction().unwrap();

        // Write using path API
        db.put_path(txn, "users/1/name", b"Alice").unwrap();
        db.put_path(txn, "users/1/email", b"alice@example.com")
            .unwrap();
        db.put_path(txn, "users/2/name", b"Bob").unwrap();

        db.commit(txn).unwrap();

        // Scan path prefix
        let txn2 = db.begin_transaction().unwrap();
        let results = db.scan_path(txn2, "users/1/").unwrap();
        assert_eq!(results.len(), 2);

        db.abort(txn2).unwrap();
    }

    #[test]
    fn test_database_table_api() {
        let dir = tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        // Register table
        db.register_table(TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnDef {
                    name: "name".to_string(),
                    col_type: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "age".to_string(),
                    col_type: ColumnType::Int64,
                    nullable: true,
                },
            ],
        })
        .unwrap();

        // Insert row
        let txn = db.begin_transaction().unwrap();
        let mut values = HashMap::new();
        values.insert("name".to_string(), SochValue::Text("Alice".to_string()));
        values.insert("age".to_string(), SochValue::Int(30));

        db.insert_row(txn, "users", 1, &values).unwrap();
        db.commit(txn).unwrap();

        // Read row
        let txn2 = db.begin_transaction().unwrap();
        let row = db.read_row(txn2, "users", 1, None).unwrap();
        assert!(row.is_some());

        let row = row.unwrap();
        assert_eq!(row.get("name"), Some(&SochValue::Text("Alice".to_string())));

        db.abort(txn2).unwrap();
    }

    #[test]
    fn test_database_query_builder() {
        let dir = tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        // Insert test data
        let txn = db.begin_transaction().unwrap();
        db.put_path(txn, "docs/1/title", b"Hello").unwrap();
        db.put_path(txn, "docs/1/content", b"World").unwrap();
        db.put_path(txn, "docs/2/title", b"Foo").unwrap();
        db.put_path(txn, "docs/2/content", b"Bar").unwrap();
        db.commit(txn).unwrap();

        // Query with limit
        let txn2 = db.begin_transaction().unwrap();
        let result = db.query(txn2, "docs/").limit(1).execute().unwrap();

        assert_eq!(result.rows.len(), 1);
        db.abort(txn2).unwrap();
    }

    #[test]
    fn test_database_crash_recovery() {
        let dir = tempdir().unwrap();

        // Write and commit
        {
            // Use open_without_lock for crash simulation tests
            let db = Database::open_without_lock(dir.path()).unwrap();
            // Set sync mode to FULL to ensure data is persisted before "crash"
            db.storage.set_sync_mode(2);
            let txn = db.begin_transaction().unwrap();
            db.put(txn, b"persist", b"this").unwrap();
            db.commit(txn).unwrap();
            // Don't call shutdown - simulate crash
            std::mem::forget(db);
        }

        // Reopen - should recover
        {
            let db = Database::open_without_lock(dir.path()).unwrap();
            let txn = db.begin_transaction().unwrap();
            let val = db.get(txn, b"persist").unwrap();
            assert_eq!(val, Some(b"this".to_vec()));
            db.abort(txn).unwrap();
        }
    }
}
