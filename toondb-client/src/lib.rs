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

//! # ToonDB Client SDK
//!
//! LLM-optimized database client with 40-66% token savings vs JSON.
//!
//! ## Key Features
//!
//! - **Path-based access**: O(|path|) resolution independent of data size
//! - **Token-efficient**: TOON format uses 40-66% fewer tokens than JSON
//! - **ACID transactions**: Full MVCC with snapshot isolation
//! - **Vector search**: Scale-aware backend (HNSW for small, Vamana+PQ for large)
//! - **Columnar storage**: 80% I/O reduction via projection pushdown
//!
//! ## Connection Types
//!
//! The SDK provides two connection types:
//!
//! - **`Connection`** (alias for `DurableConnection`): Production-grade with WAL durability,
//!   MVCC transactions, crash recovery. **Use this for production.**
//!
//! - **`InMemoryConnection`** (alias for `ToonConnection`): Fast in-memory storage for testing.
//!   Data is not persisted. **Use only for tests or ephemeral data.**
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use toondb::prelude::*;
//!
//! // Open a durable connection (default - uses WAL for persistence)
//! let conn = Connection::open("./data")?;
//!
//! // Or for testing, use in-memory
//! let test_conn = InMemoryConnection::open("./test_data")?;
//!
//! // Query with TOON output (66% fewer tokens)
//! let result = client.query("users")
//!     .filter("score", Gt, 80)
//!     .limit(100)
//!     .to_toon()?;
//!
//! println!("Tokens: {}", result.metrics().toon_tokens);
//! println!("Savings: {:.1}%", result.metrics().token_savings_percent());
//! ```
//!
//! ## CONTEXT SELECT for LLM Context
//!
//! ```rust,ignore
//! let context = client.context_query()
//!     .from_session("session_id")
//!     .with_token_limit(4000)
//!     .user_section(|s| s.columns(&["query", "preferences"]).priority(1))
//!     .history_section(|s| s.columns(&["recent"]).priority(2))
//!     .execute()?;
//! ```

pub mod atomic_memory;
pub mod batch;
pub mod checkpoint;
pub mod column_access;
pub mod connection;
pub mod context_query;
pub mod crud;
pub mod error;
pub mod format;
pub mod graph;
pub mod path_query;
pub mod policy;
pub mod query;
pub mod recovery;
pub mod result;
pub mod routing;
pub mod schema;
pub mod semantic_cache;
pub mod storage;
pub mod temporal_graph;
pub mod trace;
pub mod transaction;
pub mod vectors;

use crate::error::Result;

/// Trait for database connection operations.
///
/// This trait defines the core operations required for graph overlay,
/// policy engine, and tool routing to work with any connection type.
pub trait ConnectionTrait {
    /// Put a key-value pair
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// Get a value by key
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// Delete a key
    fn delete(&self, key: &[u8]) -> Result<()>;
    
    /// Scan keys with a prefix
    fn scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
}

// Primary connection API - DurableConnection is the default
pub use connection::DurableConnection;
/// Type alias for the default connection - uses durable storage with WAL
pub type Connection = DurableConnection;

/// Type alias for Database (for users expecting `use toondb::Database`)
/// This is the same as `Connection` - a durable database connection.
pub type Database = DurableConnection;

// For backwards compatibility and testing
pub use connection::ToonConnection;
/// Alias for in-memory connection (for testing)
pub type InMemoryConnection = ToonConnection;

pub use batch::{BatchOp, BatchResult, BatchWriter};
pub use column_access::{ColumnView, TypedColumn};
#[cfg(feature = "embedded")]
pub use connection::EmbeddedConnection;
pub use connection::{ConnectionConfig, DurableStats, RecoveryResult, SyncModeClient};
pub use context_query::{ContextQueryBuilder, ContextQueryResult, SectionBuilder, SectionContent};
pub use crud::{DeleteResult, InsertResult, RowBuilder, UpdateResult};
pub use format::{CanonicalFormat, ContextFormat, FormatCapabilities, FormatConversionError, WireFormat};
pub use path_query::PathQuery;
pub use result::{ResultMetrics, ToonResult};
pub use schema::{SchemaBuilder, TableDescription};
pub use transaction::{ClientTransaction, IsolationLevel, SnapshotReader};
pub use vectors::{SearchResult, VectorCollection};
// Re-export new modules
pub use atomic_memory::{AtomicMemoryWriter, AtomicWriteResult, MemoryOp, MemoryWriteBuilder};
pub use checkpoint::{Checkpoint, CheckpointMeta, CheckpointStore, DefaultCheckpointStore, RunMetadata, RunStatus, WorkflowEvent};
pub use trace::{TraceRun, TraceSpan, TraceStore, TraceValue, SpanKind, SpanStatusCode};
pub use policy::{CompiledPolicySet, EvaluationResult, PolicyOutcome, PolicyRule};
// Re-export deprecated GroupCommitBuffer with warning
#[allow(deprecated)]
pub use batch::{GroupCommitBuffer, GroupCommitConfig};
pub use error::ClientError;
pub use query::{QueryExecutor, QueryResult};
pub use recovery::{CheckpointResult, RecoveryManager, RecoveryStatus, WalVerificationResult};

// Re-export columnar query result from storage layer
pub use toondb_storage::ColumnarQueryResult;

use std::path::Path;
use std::sync::Arc;

/// ToonDB Client - LLM-optimized database access
///
/// # Token Efficiency
///
/// TOON format achieves 40-66% token reduction vs JSON:
/// - JSON: `{"field1": "val1", "field2": "val2"}`
/// - TOON: `table[N]{f1,f2}: v1,v2`
///
/// For 100 rows × 5 fields:
/// - JSON: ~7,500 tokens
/// - TOON: ~2,550 tokens (66% savings)
pub struct ToonClient {
    connection: Arc<ToonConnection>,
    config: ClientConfig,
}

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Maximum tokens per response (for LLM context management)
    pub token_budget: Option<usize>,
    /// Enable streaming output
    pub streaming: bool,
    /// Default output format
    pub output_format: OutputFormat,
    /// Connection pool size
    pub pool_size: usize,
}

/// Output format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// TOON format (default, 40-66% fewer tokens)
    Toon,
    /// JSON (for compatibility)
    Json,
    /// Raw columnar (for analytics)
    Columnar,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            token_budget: None,
            streaming: false,
            output_format: OutputFormat::Toon,
            pool_size: 10,
        }
    }
}

impl ToonClient {
    /// Open database at path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = ToonConnection::open(path)?;
        Ok(Self {
            connection: Arc::new(connection),
            config: ClientConfig::default(),
        })
    }

    /// Open with custom configuration
    pub fn open_with_config(
        path: impl AsRef<Path>,
        config: ClientConfig,
    ) -> Result<Self> {
        let connection = ToonConnection::open(path)?;
        Ok(Self {
            connection: Arc::new(connection),
            config,
        })
    }

    /// Set token budget for responses
    pub fn with_token_budget(mut self, budget: usize) -> Self {
        self.config.token_budget = Some(budget);
        self
    }

    /// Start a path-based query (ToonDB's unique access pattern)
    /// O(|path|) resolution, not O(N) scan
    pub fn query(&self, path: &str) -> PathQuery<'_> {
        PathQuery::from_path(&self.connection, path)
    }

    /// Access vector collection
    pub fn vectors(&self, name: &str) -> Result<VectorCollection> {
        VectorCollection::open(&self.connection, name)
    }

    /// Begin transaction with default isolation (snapshot)
    pub fn begin(&self) -> Result<ClientTransaction<'_>> {
        ClientTransaction::begin(&self.connection, IsolationLevel::SnapshotIsolation)
    }

    /// Begin transaction with specified isolation level
    pub fn begin_with_isolation(
        &self,
        isolation: IsolationLevel,
    ) -> Result<ClientTransaction<'_>> {
        ClientTransaction::begin(&self.connection, isolation)
    }

    /// Create a read-only snapshot at current time
    pub fn snapshot(&self) -> Result<SnapshotReader<'_>> {
        SnapshotReader::now(&self.connection)
    }

    /// Execute raw TOON-QL query
    pub fn execute(&self, sql: &str) -> Result<QueryResult> {
        self.connection.query_sql(sql)
    }

    /// Get connection for direct access
    pub fn connection(&self) -> &ToonConnection {
        &self.connection
    }

    /// Get client statistics
    pub fn stats(&self) -> ClientStats {
        self.connection.stats()
    }

    /// Get token budget
    pub fn token_budget(&self) -> Option<usize> {
        self.config.token_budget
    }

    /// Get output format
    pub fn output_format(&self) -> OutputFormat {
        self.config.output_format
    }
}

// ============================================================================
// DurableToonClient - WAL-backed ToonClient
// ============================================================================

/// Durable ToonClient backed by EmbeddedConnection with WAL/MVCC
///
/// Unlike `ToonClient` which uses in-memory `ToonConnection`, this uses
/// `EmbeddedConnection` which wraps the full Database kernel with:
/// - Write-Ahead Logging (WAL) for durability
/// - MVCC with SSI for proper transaction isolation
/// - Crash recovery
///
/// Use this for production workloads requiring ACID guarantees.
#[cfg(feature = "embedded")]
pub struct DurableToonClient {
    connection: Arc<EmbeddedConnection>,
    config: ClientConfig,
}

#[cfg(feature = "embedded")]
impl DurableToonClient {
    /// Open durable database at path with WAL/MVCC
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = EmbeddedConnection::open(path)?;
        Ok(Self {
            connection: Arc::new(connection),
            config: ClientConfig::default(),
        })
    }

    /// Create from existing connection
    pub fn from_connection(connection: Arc<EmbeddedConnection>) -> Self {
        Self {
            connection,
            config: ClientConfig::default(),
        }
    }

    /// Open with custom configuration
    pub fn open_with_config(
        path: impl AsRef<Path>,
        config: ClientConfig,
        db_config: toondb_storage::database::DatabaseConfig,
    ) -> Result<Self> {
        let connection = EmbeddedConnection::open_with_config(path, db_config)?;
        Ok(Self {
            connection: Arc::new(connection),
            config,
        })
    }

    /// Set token budget for responses
    pub fn with_token_budget(mut self, budget: usize) -> Self {
        self.config.token_budget = Some(budget);
        self
    }

    /// Begin a transaction
    pub fn begin(&self) -> Result<()> {
        self.connection.begin()
    }

    /// Commit the active transaction
    pub fn commit(&self) -> Result<u64> {
        self.connection.commit()
    }

    /// Abort the active transaction
    pub fn abort(&self) -> Result<()> {
        self.connection.abort()
    }

    /// Put bytes at a path
    pub fn put(&self, path: &str, value: &[u8]) -> Result<()> {
        self.connection.put(path, value)
    }

    /// Get bytes at a path
    pub fn get(&self, path: &str) -> Result<Option<Vec<u8>>> {
        self.connection.get(path)
    }

    /// Delete a path
    pub fn delete(&self, path: &str) -> Result<()> {
        self.connection.delete(path)
    }

    /// Scan paths with prefix
    pub fn scan(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        self.connection.scan(prefix)
    }

    /// Get database statistics
    pub fn stats(&self) -> ClientStats {
        self.connection.stats()
    }

    /// Force fsync
    pub fn fsync(&self) -> Result<()> {
        self.connection.fsync()
    }

    /// Get the underlying connection
    pub fn connection(&self) -> &EmbeddedConnection {
        &self.connection
    }

    /// Get token budget
    pub fn token_budget(&self) -> Option<usize> {
        self.config.token_budget
    }

    /// Get output format
    pub fn output_format(&self) -> OutputFormat {
        self.config.output_format
    }
}

/// Client statistics
#[derive(Debug, Clone)]
pub struct ClientStats {
    /// Total queries executed
    pub queries_executed: u64,
    /// Total TOON tokens emitted
    pub toon_tokens_emitted: u64,
    /// Equivalent JSON tokens
    pub json_tokens_equivalent: u64,
    /// Token savings percentage
    pub token_savings_percent: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Prelude for convenient imports
pub mod prelude {
    #[cfg(feature = "embedded")]
    pub use crate::DurableToonClient;
    pub use crate::path_query::CompareOp;
    pub use crate::{
        ClientConfig,
        ClientError,
        ClientStats,
        ClientTransaction,
        // Connection types
        Connection,
        DeleteResult,
        DurableConnection,
        InMemoryConnection,
        InsertResult,
        IsolationLevel,
        OutputFormat,
        PathQuery,
        ResultMetrics,
        RowBuilder,
        SchemaBuilder,
        SearchResult,
        SnapshotReader,
        TableDescription,
        ToonClient,
        ToonResult,
        UpdateResult,
        VectorCollection,
    };
    pub use toondb_core::toon::{ToonType, ToonValue};
}
