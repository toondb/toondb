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

//! SochDB Core
//!
//! TOON-native ACID database - fundamental types and data structures.
//!
//! SochDB is to TOON what MongoDB is to JSON - a database that natively
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
//! sochdb-core = { version = "...", features = ["jemalloc"] }
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
//! use sochdb_core::soch::{SochSchema, SochType, SochTable, SochRow, SochValue};
//!
//! // Define a schema
//! let schema = SochSchema::new("users")
//!     .field("id", SochType::UInt)
//!     .field("name", SochType::Text)
//!     .field("email", SochType::Text)
//!     .primary_key("id");
//!
//! // Create a table
//! let mut table = SochTable::new(schema);
//! table.push(SochRow::new(vec![
//!     SochValue::UInt(1),
//!     SochValue::Text("Alice".into()),
//!     SochValue::Text("alice@example.com".into()),
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
pub mod soch;
pub mod soch_codec;
pub mod sochfs_metadata;
pub mod transaction_typestate; // Type-State Transaction API (compile-time safe transaction lifecycle)
pub mod txn;
pub mod version_chain; // Unified MVCC version chain trait
pub mod vfs;
pub mod zero_copy; // Predefined SochQL views (Task 7)

// Analytics - anonymous usage tracking (disabled with SOCHDB_DISABLE_ANALYTICS=true)
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
pub use error::{Result, SochDBError};
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
pub use soch::{SochField, SochIndex, SochRow, SochSchema, SochTable, SochType, SochValue};
pub use soch_codec::{
    SochDbBinaryCodec, SochDocument, SochParseError, SochTextEncoder, SochTextParser,
    SochTokenCounter,
};
pub use sochfs_metadata::{DirEntryRow, FsMetadataStore, FsWalOp, InodeRow, SochFS};
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
pub const SOCHDB_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for SochDB files
pub const SOCHDB_MAGIC: [u8; 4] = *b"TOON";

/// Current schema version
pub const SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soch_roundtrip() {
        let schema = SochSchema::new("test")
            .field("id", SochType::UInt)
            .field("value", SochType::Text);

        let mut table = SochTable::new(schema);
        table.push(SochRow::new(vec![
            SochValue::UInt(1),
            SochValue::Text("hello".into()),
        ]));

        let formatted = table.format();
        let parsed = SochTable::parse(&formatted).unwrap();

        assert_eq!(parsed.schema.name, "test");
        assert_eq!(parsed.rows.len(), 1);
    }
}
