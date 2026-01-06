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

//! ToonDB Core
//!
//! TOON-native ACID database - fundamental types and data structures.
//!
//! ToonDB is to TOON what MongoDB is to JSON - a database that natively
//! understands and stores TOON (Tabular Object-Oriented Notation) documents.
//!
//! # Core Components
//!
//! - **TOON Format**: Native data format with schema support
//! - **Transaction Manager**: ACID transaction support via WAL
//! - **Virtual Filesystem**: POSIX-like VFS backed by WAL
//! - **Schema Catalog**: Table and index metadata management
//!
//! # Memory Allocation
//!
//! **Recommended**: Enable jemalloc for production workloads:
//!
//! ```toml
//! toondb-core = { version = "...", features = ["jemalloc"] }
//! ```
//!
//! jemalloc provides:
//! - Thread-local caching (no lock contention)
//! - Superior fragmentation handling
//! - Automatic memory return to OS
//! - Battle-tested in production (Firefox, Redis, RocksDB)
//!
//! **Note**: The `buddy_allocator` module is deprecated. See its documentation
//! for migration guidance.
//!
//! # Features
//!
//! - `jemalloc` - Use jemalloc as the global allocator for better performance
//!
//! # Example
//!
//! ```rust,ignore
//! use toondb_core::toon::{ToonSchema, ToonType, ToonTable, ToonRow, ToonValue};
//!
//! // Define a schema
//! let schema = ToonSchema::new("users")
//!     .field("id", ToonType::UInt)
//!     .field("name", ToonType::Text)
//!     .field("email", ToonType::Text)
//!     .primary_key("id");
//!
//! // Create a table
//! let mut table = ToonTable::new(schema);
//! table.push(ToonRow::new(vec![
//!     ToonValue::UInt(1),
//!     ToonValue::Text("Alice".into()),
//!     ToonValue::Text("alice@example.com".into()),
//! ]));
//!
//! // Format as TOON
//! println!("{}", table.format());
//! // Output: users[1]{id,name,email}:
//! //         1,Alice,alice@example.com
//! ```

// Use jemalloc as global allocator when feature is enabled
// This provides better performance for allocation-heavy workloads
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod block_storage;
#[deprecated(
    since = "0.2.0",
    note = "Use jemalloc feature instead. See buddy_allocator module docs for migration."
)]
#[allow(deprecated)]
pub mod buddy_allocator;
pub mod catalog;
pub mod columnar; // True Columnar Storage with Arrow-compatible layout (mm.md Task 1)
pub mod concurrency; // Hierarchical Lock Architecture (mm.md Task 2)
pub mod epoch_gc;
pub mod error;
pub mod format_migration;
pub mod key;
pub mod learned_index;
pub mod lockfree_interner; // Lock-free string interner (mm.md Task 6)
pub mod memory_schema; // Canonical Episode/Entity/Event schema
pub mod path_trie;
pub mod predefined_views;
pub mod reclamation;
pub mod schema_bridge;
pub mod schema_evolution;
pub mod sharded_block_store;
pub mod string_interner; // String interning for path segments
pub mod tbp; // TOON Binary Protocol for zero-copy wire format (mm.md Task 3.1)
pub mod toon;
pub mod toon_codec;
pub mod toonfs_metadata;
pub mod transaction_typestate; // Type-State Transaction API (compile-time safe transaction lifecycle)
pub mod txn;
pub mod version_chain; // Unified MVCC version chain trait
pub mod vfs;
pub mod zero_copy; // Predefined ToonQL views (Task 7)

// Analytics - anonymous usage tracking (disabled with TOONDB_DISABLE_ANALYTICS=true)
#[cfg(feature = "analytics")]
pub mod analytics;

// Re-export core types
pub use block_storage::{
    BlockCompression, BlockRef, BlockStore, BlockStoreStats, FileBlockManager,
};
pub use catalog::{Catalog, CatalogEntry, CatalogEntryType, McpToolDescriptor, OperationImpl};
pub use columnar::{
    ColumnChunk, ColumnStats, ColumnType as ColumnarColumnType, ColumnValue as ColumnarColumnValue,
    ColumnarStore, ColumnarTable, MemoryComparison, TypedColumn, ValidityBitmap,
};
pub use error::{Result, ToonDBError};
pub use key::{CausalKey, TemporalKey};
pub use learned_index::LearnedSparseIndex;
pub use lockfree_interner::{InternerStats, LockFreeInterner, Symbol};
pub use memory_schema::{
    Entity, EntityFacts, EntityKind, EntitySearchResult, Episode, EpisodeSearchResult, EpisodeType,
    Event, EventMetrics, EventRole, MemoryStore, TableRole, TableSemanticMetadata,
};
pub use path_trie::{ColumnGroupAffinity, ColumnType as PathTrieColumnType, PathTrie, TrieNode};
pub use predefined_views::{
    ViewDefinition, build_view_map, get_predefined_views, get_view, naming,
};
pub use toon::{ToonField, ToonIndex, ToonRow, ToonSchema, ToonTable, ToonType, ToonValue};
pub use toon_codec::{
    ToonDbBinaryCodec, ToonDocument, ToonParseError, ToonTextEncoder, ToonTextParser,
    ToonTokenCounter,
};
pub use toonfs_metadata::{DirEntryRow, FsMetadataStore, FsWalOp, InodeRow, ToonFS};
pub use txn::{
    AriesCheckpointData, AriesDirtyPageEntry, AriesTransactionEntry, IsolationLevel, Lsn, PageId,
    Transaction, TransactionManager, TxnId, TxnState, TxnStats, TxnWalEntry, TxnWrite,
    WalRecordType,
};
pub use transaction_typestate::{
    Transaction as TypestateTransaction, Active, Committed, Aborted,
    ReadOnly, ReadWrite, WriteOnly, TransactionStorage, TransactionMode,
};
pub use version_chain::{
    MvccVersionChain, MvccVersionChainMut, VersionMeta, VisibilityContext, WriteConflictDetection,
};
pub use vfs::{
    BlockId, DirEntry, Directory, FileStat, FileType, Inode, InodeId, Permissions, Superblock,
    VfsOp,
};

/// Database version
pub const TOONDB_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for ToonDB files
pub const TOONDB_MAGIC: [u8; 4] = *b"TOON";

/// Current schema version
pub const SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toon_roundtrip() {
        let schema = ToonSchema::new("test")
            .field("id", ToonType::UInt)
            .field("value", ToonType::Text);

        let mut table = ToonTable::new(schema);
        table.push(ToonRow::new(vec![
            ToonValue::UInt(1),
            ToonValue::Text("hello".into()),
        ]));

        let formatted = table.format();
        let parsed = ToonTable::parse(&formatted).unwrap();

        assert_eq!(parsed.schema.name, "test");
        assert_eq!(parsed.rows.len(), 1);
    }
}
