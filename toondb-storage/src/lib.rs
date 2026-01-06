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

//! ToonDB Storage Layer
//!
//! Log-Structured Column Store (LSCS) with transaction-aware WAL for TOON-native data.
//!
//! ## Runtime Modes
//!
//! This crate supports two runtime modes:
//!
//! ### Embedded Sync Mode (like SQLite)
//!
//! For embedded deployments without async runtime:
//!
//! ```toml
//! toondb-storage = { version = "...", default-features = false, features = ["embedded-sync"] }
//! ```
//!
//! Benefits:
//! - ~500KB smaller binary
//! - No async runtime overhead
//! - Simpler embedded integration
//!
//! ### Async Mode (default, for servers)
//!
//! For server deployments with async I/O:
//!
//! ```toml
//! toondb-storage = { version = "..." }  # async enabled by default
//! ```
//!
//! Benefits:
//! - Better scalability for concurrent connections
//! - Non-blocking I/O for server workloads
//!
//! ## Novel Components
//!
//! - **LSCS** (`lscs`): Log-Structured Column Store - columnar variant of LSM with
//!   schema-aware compression and column-aware compaction for reduced write amplification.
//!
//! - **Transaction WAL** (`txn_wal`): ACID-compliant Write-Ahead Log with transaction
//!   boundaries, commit/abort markers, and crash recovery.
//!
//! - **StorageEngine Trait** (`storage_engine`): Pluggable storage backend abstraction
//!   enabling 80% I/O reduction for columnar projections (Task 1).
//!
//! - **Page Manager** (`page_manager`): TOON file format with magic header and O(1)
//!   page allocation (Task 8).
//!
//! - **Columnar Compression** (`columnar_compression`): Type-aware encoding with
//!   dictionary, RLE, and delta compression for 2-4× storage reduction (Task 9).
//!
//! ## Utility Components
//!
//! - **Bloom Filters** (`bloom`): Probabilistic existence checks
//! - **Block Checksums** (`block_checksum`): Data integrity validation
//! - **Compression** (`compression`): LZ4/Zstd compression
//! - **Sketches** (`sketches`): Approximate algorithms (HyperLogLog, CountMin, DDSketch)

// New TOON-native storage components
pub mod actor; // Actor-based connection manager (mm.md Task 7.2)
pub mod aries_recovery; // ARIES-style crash recovery (Task 1)
pub mod checkpoint; // ARIES-style checkpointing with WAL truncation (mm.md Task 1.4)
pub mod columnar_compression;
pub mod database; // Database Kernel (shared by embedded + server)
pub mod durable_storage; // Fully wired durable storage with MVCC
pub mod ffi;
pub mod group_commit; // Event-driven Group Commit (Task 4)
pub mod hlc; // Hybrid Logical Clock for commit timestamps (mm.md Task 1.3)
pub mod hybrid_store; // PAX hybrid row-column storage (mm.md Task 4.1)
pub mod ipc; // IPC Protocol with multiplexing (mm.md Task 7.1)
#[cfg(unix)]
pub mod ipc_server; // Unix Socket IPC Server (Task 3)
pub mod learned_index_integration;
pub mod lscs;
pub mod mvcc_new;
pub mod mvcc_snapshot;
pub mod page_manager;
pub mod production_wal; // Production WAL with ARIES recovery (mm.md Task 3)
pub mod ssi; // Serializable Snapshot Isolation (Task 2)
pub mod storage_engine;
pub mod streaming_iterator; // Streaming Iterator Architecture (mm.md Task 4)
pub mod transaction; // Unified Transaction Coordinator trait and types
pub mod txn_arena; // Transaction-scoped arena with zero-copy key/value plumbing
pub mod txn_wal;
pub mod wal_integration;
pub mod zero_copy_safety; // Zero-Copy Validation Layer (Task 5) // FFI bindings for Python SDK

// Performance optimization modules
pub mod adaptive_learned_index;
pub mod adaptive_memtable; // Adaptive memtable sizing with memory pressure (Task 10)
pub mod deferred_index; // Deferred sorted index with LSM-style compaction (Rec 2)
pub mod dirty_tracking; // Batched dirty tracking with MPSC queue
pub mod index_policy; // Per-table index policy
pub mod batch_wal; // Batched WAL with vectored I/O (Task 3)
pub mod key_buffer; // Cache-line aligned key buffer (Task 2)
pub mod lockfree_memtable; // Lock-free read path with hazard pointers (Task 4)
pub mod packed_row; // Unified row storage with delta encoding (Task 1)

// PhD-Level Architectural Optimizations (December 2025)
pub mod clr_learned_index; // CLR Learned Index for sorted runs (Task 3)
pub mod lockfree_epoch; // Lock-Free Epoch Tracking (Task 3)
pub mod hierarchical_ts; // Hierarchical Timestamp Oracle (Task 9)
pub mod shard_coalesced; // Shard-Coalesced Batch DashMap (Task 6)
pub mod polymorphic_value; // Polymorphic Value Encoding (Task 12)
pub mod epoch_arena; // Epoch-Partitioned Key Arena (Task 1)
pub mod stratified_skiplist; // Stratified SkipList with Deferred Promotion (Task 2)
pub mod columnar_wal; // Columnar WAL Layout (Task 4)
pub mod generational_slab; // Generational Slab Allocator (Task 5)
pub mod rl_workload; // RL Workload Classifier (Task 10)
#[cfg(unix)]
pub mod io_uring_wal; // io_uring WAL Submission (Task 11)

// New performance modules (Recommendations 1-9)
pub mod cow_btree; // Copy-on-Write B-Tree for ordered access (Recommendation 5)
pub mod epoch_mvcc; // Epoch-based MVCC for O(log E) version lookup (Recommendation 7)
pub mod page_cache; // Application-level page cache with Clock-Pro (Recommendation 8)
pub mod row_format; // Slot-based columnar row storage (Recommendation 1)
pub mod tiered_memtable; // Tiered MemTable with deferred sorting (Recommendation 3)
pub mod tournament_tree; // K-way merge with tournament tree (Task 2)
pub mod vectorized_scan; // SIMD-accelerated vectorized scan engine (Recommendation 2)
pub mod zero_copy_serde; // Zero-copy serialization for WAL (Recommendation 6)

// Namespace and multi-tenancy support (Task 3)
pub mod namespace; // Namespace routing and on-disk layout

// Core utilities
pub mod backend;
pub mod backup;
pub mod block_checksum;
pub mod bloom;
pub mod compression;
pub mod dict_compression;
pub mod direct_io;
#[cfg(unix)]
pub mod io_uring;
pub mod manifest;
pub mod memory;
pub mod parallel_merge;
pub mod payload;
pub mod prefetch;
pub mod sketches;
pub mod two_level_index;
pub mod validation;
pub mod version_store;
pub mod zero_copy;

// Re-exports for new components
pub use columnar_compression::{
    ColumnEncoder, DeltaEncoder, DictionaryEncoder, EncodingStats, EncodingType, RleEncoder,
};
pub use learned_index_integration::{
    HybridIndex, IndexManager, IndexType, KeyStats, PointLookupExecutor,
};
pub use lscs::{
    ColumnDef, ColumnGroup, ColumnType, ColumnarMemtable, Lscs, LscsConfig, LscsRecoveryStats,
    LscsStats, TableSchema,
};
#[allow(deprecated)]
pub use mvcc_snapshot::{
    MvccStore, Snapshot as MvccSnapshot, Timestamp, TransactionManager, TxnId, TxnStatus,
    VersionChain, VersionInfo,
};
pub use page_manager::{
    DEFAULT_PAGE_SIZE, DbHeader, FORMAT_VERSION, FreePageHeader, PageId, PageManager,
    PageManagerStats, PageType, TOONDB_MAGIC,
};
pub use storage_engine::{
    ColumnId, ColumnIterator, Row, RowId, StorageEngine, StorageEngineType, StorageStats,
    TxnHandle, open_storage_engine,
};
pub use transaction::{
    DurabilityLevel, IsolationLevel, RecoveryStats as TxnRecoveryStats, TransactionCoordinator,
    TransactionHandle,
};
pub use txn_wal::{CrashRecoveryStats, TxnWal, TxnWalBuffer, TxnWalEntry, TxnWalStats};
pub use wal_integration::{
    GroupCommitBuffer, MvccTransactionManager, RecoveryStats, Transaction, TxnState, 
    WalStorageManager,
};

// Re-exports for performance optimization modules
pub use adaptive_learned_index::{AdaptiveLearnedIndex, LearnedIndexStats, PiecewiseLinearModel};
pub use adaptive_memtable::{
    AdaptiveMemtableConfig, AdaptiveMemtableSizer, AdaptiveMemtableStats,
    DEFAULT_BASE_SIZE, MAX_MEMTABLE_SIZE, MIN_MEMTABLE_SIZE,
};
pub use batch_wal::{
    BatchAccumulator, BatchedWalReader, BatchedWalStats, BatchedWalWriter, ConcurrentBatchedWal,
    DEFAULT_MAX_BATCH_BYTES, DEFAULT_MAX_BATCH_SIZE,
};
pub use clr_learned_index::{ClrIndex, ClrLookupResult, ClrStats, IndexedSortedRun};
pub use key_buffer::{
    ArenaKey,
    ArenaKeyHandle,
    BatchKeyGenerator,
    InternedTablePrefix,
    // Arena allocation for high-throughput key operations
    KeyArena,
    KeyBuffer,
    MAX_KEY_LENGTH,
};
pub use lockfree_memtable::{
    HazardDomain,
    INLINE_VALUE_SIZE,
    LockFreeMemTable,
    LockFreeVersion,
    LockFreeVersionChain,
    // Inline value storage for reduced memory indirection
    ValueStorage,
};
pub use packed_row::{
    PackedColumnDef, PackedColumnType, PackedRow, PackedRowBuilder, PackedTableSchema,
};

// Re-exports for utilities
pub use backend::{LocalFsBackend, ObjectMetadata, StorageBackend};
pub use backup::{BackupManager, BackupMetadata};
pub use block_checksum::{
    BlockChecksumConfig, BlockChecksumStats, BlockType as BlockChecksumType, BlockWriter, ChecksummedBlock,
};
pub use bloom::{BlockedBloomFilter, BloomFilter, LevelAdaptiveFPR, UnifiedBloomFilter};
pub use compression::{CompressionEngine, CompressionStats, StorageTier};
pub use manifest::{FileMetadata, LsmState, Manifest, VersionEdit};
pub use memory::{MemoryBudget, MemoryTracker, WriteBufferManager, WriteBufferStats};
pub use mvcc_new::{
    ColumnGroupRef, ReadVersion, Snapshot, SnapshotGuard, VersionGuard, VersionSet,
    VersionSetStats, VersionSetStatsSnapshot,
};
pub use payload::{CompressionType, PayloadStats, PayloadStore};
pub use sketches::{AdaptiveSketch, CountMinSketch, DDSketch, ExponentialHistogram, HyperLogLog};
pub use two_level_index::{
    BlockIndexEntry, BlockIndexReader, FencePointer, TemporalKey, TwoLevelIndex,
};
pub use validation::{SSTableValidator, validate_sstable_file};

// Re-exports for durable storage
pub use durable_storage::{ArenaMvccMemTable, DurableStorage, MvccMemTable, TransactionMode};

// Super Version and Copy-on-Write Version Set (mm.md Task 1)
pub mod version_set;
pub mod concurrent_art;
pub mod sstable;
pub mod wal_segment;
pub mod compaction_policy;
pub mod optimized_scan;

// Re-exports for new performance modules (Recommendations 1-9)
pub use version_set::{
    FileMetadata as VersionFileMetadata, ImmutableMemTable, ImmutableMemTableRef,
    LevelMetadata, SuperVersion, SuperVersionHandle, VersionSet as CowVersionSet,
};
pub use concurrent_art::ConcurrentART;
pub use sstable::{
    BlockBuilder, BlockIterator, BlockHandle, BlockType,
    FilterPolicy, BloomFilterPolicy, RibbonFilterPolicy, XorFilterPolicy, FilterReader,
    SSTableFormat, Header, Footer, Section, SectionType,
    SSTableBuilder, SSTableBuilderOptions, SSTableBuilderResult,
    SSTable, TableMetadata, ReadOptions, BlockCache,
};
pub use wal_segment::{
    WalSegmentManager, SegmentConfig, SegmentHeader, SegmentMetadata,
    CheckpointRecord, SegmentStats, RecoveryIterator, WalEntry,
};
pub use compaction_policy::{
    CompactionConfig, CompactionFile, CompactionJob, CompactionPicker,
    CompactionPriority, CompactionReason, CompactionState, CompactionStats,
    CompactionStrategy, LeveledCompactionPicker, RetentionConfig,
    UniversalCompactionPicker, VersionPruner,
};
pub use optimized_scan::{
    EntrySource, FileRange, LevelFiles, RangeScanner, ScanConfig, ScanStats,
    TournamentTree, VersionedEntry,
};
pub use cow_btree::{BTreeEntry, BTreeSnapshot, CowBTree, Node, SearchResult};
pub use epoch_mvcc::{
    CommitResult, EpochManager, EpochMvccStore, EpochSnapshot, EpochTransaction,
    EpochVersionChain, GcStats, StoreStats, VersionEntry,
};
pub use page_cache::{CacheStats, ClockProCache, CachedPage, PageId as CachePageId, PageState};
pub use row_format::{Slot, SlotRow, SlotRowArena, SlotRowHandle, SlotRowFlags};
pub use tiered_memtable::{HotEntry, SortedBatch, TieredMemTable};
pub use vectorized_scan::{
    ColumnVector, ComparisonOp, Int64Comparison, VectorBatch, VectorPredicate,
    VectorizedScanConfig, VectorizedScanStats, DEFAULT_BATCH_SIZE,
    // SoA + Late Materialization (80/20 optimization)
    SimdVisibilityFilter, SoaBatch, SoaScanIterator, SoaScanStats, SoaSource,
    StreamingScanIterator, ValueHandle, VersionedSlice,
};
pub use zero_copy_serde::{
    FieldDescriptor, MmapWalReader, SerdeStats, WalBatchReader, WalBatchWriter,
    WalEntryBuilder, WalEntryHeader, WalEntryReader, WalEntryType, ZeroCopyHeader,
    FORMAT_VERSION as SERDE_FORMAT_VERSION, HEADER_SIZE as SERDE_HEADER_SIZE, ZERO_COPY_MAGIC,
};

// Re-exports for transaction arena and zero-copy plumbing
pub use txn_arena::{
    ArenaWriteSet, BytesRef, KeyFingerprint, TxnArena, TxnWriteBuffer, WriteOp,
};

// Re-exports for dirty tracking with batching
pub use dirty_tracking::{
    BatchedDirtyTracker, DirtyEvent, DirtyTrackingStats, TxnDirtyBuffer,
};

// Re-exports for per-table index policy
pub use index_policy::{
    BalancedTableIndex, IndexPolicy, SortedRun, TableIndexConfig, TableIndexRegistry,
};

// Re-exports for database kernel
pub use database::{
    ColumnDef as DbColumnDef,
    ColumnType as DbColumnType,
    ColumnarQueryResult, // SIMD-friendly columnar result format
    Database,
    DatabaseConfig,
    GroupCommitSettings,
    QueryBuilder,
    QueryResult,
    QueryRowIterator,
    RecoveryStats as DbRecoveryStats,
    Stats as DbStats,
    SyncMode,
    TableSchema as DbTableSchema,
    TxnHandle as KernelTxnHandle,
    VectorSearchResult,
};
