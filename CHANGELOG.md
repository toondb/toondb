# Changelog

All notable changes to SochDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.1] - 2026-01-19

### Added

#### 🔒 Concurrency Safety (Critical Fix)

This release addresses the **66.5% data loss** issue from concurrent multi-process access by implementing comprehensive database locking and fencing mechanisms.

- **Advisory File Locking (P0)** — Cross-platform exclusive database access
  - POSIX `flock()` on Unix, `LockFileEx()` on Windows
  - New `DatabaseLock` and `RwDatabaseLock` types in `sochdb-storage/lock.rs`
  - Stale lock detection with PID tracking
  - Configurable timeout and retry via `LockConfig`
  - Lock acquired automatically in `DurableStorage::open_with_full_config()`

- **WAL Sequence Fencing (P0)** — Split-brain detection
  - New 64-byte `WalHeader` with magic, epoch, and writer_id (UUID)
  - Epoch-based fencing prevents stale writers from corrupting data
  - CRC chain verification for entry integrity
  - New `FencedWal` type in `sochdb-storage/wal_fencing.rs`

- **Reader-Writer Lock Protocol (P1)** — Multiple readers OR single writer
  - `RwDatabaseLock` supports concurrent reads with exclusive writes
  - Uses `.write_lock` file for writer exclusivity

- **Lock Timeout & Deadlock Detection (P1)** — Robust error handling
  - `LockConfig` with `timeout` and `retry_interval` settings
  - Stale lock cleanup when holder PID no longer exists
  - New error types: `LockError`, `DatabaseLocked`, `EpochMismatch`, `SplitBrain`

- **Connection Mode Enforcement (P2)** — Type-safe read-only connections
  - New `ReadOnlyConnection` type with compile-time write prevention
  - `ReadableConnection` and `WritableConnection` traits
  - `ConnectionModeClient` enum for runtime mode tracking

#### 📦 SDK Updates

- **Python SDK v0.4.1**
  - New error types: `LockError`, `DatabaseLockedError`, `LockTimeoutError`, `EpochMismatchError`, `SplitBrainError`
  - New error codes (10xxx range) for lock/concurrency errors
  - Updated `from_rust_error()` mapping for new error types

- **Node.js SDK v0.4.1**
  - New `ErrorCode` enum with lock error codes
  - New error classes: `LockError`, `DatabaseLockedError`, `LockTimeoutError`, `EpochMismatchError`, `SplitBrainError`
  - All errors now include `code`, `remediation` properties

- **Go SDK v0.4.1**
  - New sentinel errors: `ErrDatabaseLocked`, `ErrLockTimeout`, `ErrEpochMismatch`, `ErrSplitBrain`
  - New error types: `DatabaseLockedError`, `LockTimeoutError`, `EpochMismatchError`, `SplitBrainError`
  - Implements `errors.Is()` for sentinel matching

### Fixed

- Crash recovery tests now properly release locks before reopening database
- Added `open_without_lock()` test helper for crash simulation scenarios

---

## [Unreleased]

### Added

#### 🎯 Core Infrastructure Improvements

- **Monotonic Commit Timestamps** — End-to-end HLC (Hybrid Logical Clock) integration
  - FFI: New `C_CommitResult` struct with `commit_ts` and `error_code` in `sochdb_commit()`
  - Python: `Transaction.commit()` now returns real `u64` commit timestamp (was hardcoded `0`)
  - Go: Updated `Transaction.Commit()` to retrieve commit timestamp from FFI
  - Node.js: `Transaction.commit()` returns timestamp
  - Rust: Already correct in `sochdb-client`
  - Enables MVCC observability, replication with causal ordering, and deterministic replay

- **Configuration Plumbing** — 13 tunable parameters now applied via FFI
  - New `C_DatabaseConfig` struct with: `wal_enabled`, `sync_mode`, `memtable_size_bytes`, `bloom_filter_bits_per_key`, `block_cache_size_mb`, `compaction_trigger_mb`, `max_file_size_mb`, `compression_enabled`, `enable_statistics`, `max_open_files`, `use_direct_io`, `enable_checksum`, `auto_checkpoint_interval_s`
  - New `sochdb_open_with_config()` FFI function
  - Python: `Database.open()` applies config via FFI (previously accepted but ignored)
  - Go: Config support added to `OpenWithConfig()`
  - Node.js: Config interface and application in `Database.open()`
  - Predictable durability guarantees and tunable write amplification

- **Prefix-Bounded Scans** — Multi-tenant safety by construction
  - Storage: `MIN_SAFE_PREFIX_LEN = 3` validation in `scan()`
  - Storage: New `scan_unsafe()` for internal use bypassing validation
  - Python: `scan_prefix()` validates minimum length, `scan_prefix_unchecked()` for power users
  - Go: Similar validation in `ScanPrefix()`
  - Node.js: TypeScript validation in `scanPrefix()`
  - Rust: Enforced in `scan()` method
  - Prevents accidental cross-tenant queries and data leakage

#### 🧠 Query Execution & Optimization

- **Production-Grade Context Query Engine** — Token-aware retrieval for LLM applications
  - Python: `ContextQuery` class with token estimator, deduplication strategies
  - Go: `ContextQuery` implementation with token budget tracking
  - Node.js: TypeScript async context assembly with `ContextQuery` builder
  - Rust: Already complete in `context_query.rs`
  - Features: Token budgeting, semantic deduplication, provenance tracking, priority-ordered selection
  - Supports GPT-4 (128K), Claude (200K), and custom token estimators (tiktoken)

- **Real Vector Search in Query Executor** — Functional embedding generation
  - Query: Added `ComparisonOp::SimilarTo` for vector predicates in `toon_ql.rs`
  - Optimizer: `ExecutionStep::VectorSearch` now includes `query_text`
  - Optimizer: `OptimizedExecutor` holds `EmbeddingProvider`
  - Optimizer: `extract_vector_query_text()` parses WHERE clause
  - Query syntax: `WHERE embedding SIMILAR_TO 'search query'`
  - Enables production RAG pipelines with measurable recall/latency

- **Index-Aware UPDATE/DELETE** — Secondary index infrastructure
  - Python: New `IndexInfo` dataclass and `_indexes` dict
  - Python: `_create_index()` and `_drop_index()` methods
  - Python: `_update()` uses index when WHERE matches indexed column (O(log N) vs O(N))
  - Python: `_delete()` uses index similarly
  - Python: `_insert()` maintains all indexes
  - Python: SQL parser handles `CREATE INDEX` and `DROP INDEX`
  - **Go: Added complete SQL indexing support** (CREATE/DROP INDEX, index-aware operations)
  - **Node.js: Added complete SQL indexing support** (CREATE/DROP INDEX, index-aware operations)
  - Syntax: `CREATE INDEX idx_name ON table(column)`, `DROP INDEX idx_name`

- **Hardened MCP Query Execution** — Real parser with prefix safety
  - MCP (Rust): Complete rewrite of `exec_query()` in `sochdb-mcp/src/tools.rs`
  - MCP (Rust): New `SqlParser` with PEG-style grammar
  - MCP (Rust): `ParsedQuery` struct with validated table/columns/conditions
  - MCP (Rust): Scan operations enforce prefix bounds at MCP boundary
  - **Go: Added MCP-compatible query parser and execution**
  - **Node.js: Added MCP-compatible query parser and execution**
  - Grammar-based parsing, injection-resistant, multi-tenant safe by construction

#### 🚀 Deployment & Multi-Tenancy

- **Unified Deployment Surfaces** — Single `connect()` API across embedded/IPC/gRPC
  - Python: New `connect(uri)` function with auto-detection
  - Go: Unified connection in `Connect()`
  - Node.js: `connect()` with TypeScript types
  - URI patterns: `file://./data`, `ipc:///tmp/sochdb.sock`, `grpc://localhost:50051`, `grpcs://prod.example.com:443`
  - Easy migration from laptop → server deployments

#### 🕸️ Agent-Specific Features

- **Graph Overlay** — Lightweight graph layer on KV storage for agent memory
  - Python: `GraphOverlay`, `GraphNode`, `GraphEdge` in `graph.py`
  - Go: Complete implementation in `graph.go`
  - Node.js: TypeScript implementation in `graph.ts`
  - Rust: Implementation in `graph.rs`
  - Features: Typed edges, bidirectional indexes, BFS/DFS traversal, shortest path, property storage
  - Storage: `graph:{ns}:nodes:{id}`, `graph:{ns}:out:{id}:{edge}`, `graph:{ns}:in:{id}:{edge}`
  - Complexity: O(1) for node/edge ops, O(degree) for neighbors, O(V+E) for traversals

- **Policy & Safety Hooks** — Trigger-based guardrails for agent operations
  - Python: `PolicyEngine`, `PolicyAction`, decorators in `policy.py`
  - Go: Implementation with function callbacks in `policy.go`
  - Node.js: TypeScript with async hooks in `policy.ts`
  - Rust: Implementation in `policy.rs`
  - Features: `@before_write`, `@after_read`, `@before_delete`, `@after_commit` hooks
  - Actions: `ALLOW`, `DENY`, `MODIFY`, `AUDIT`
  - Pattern matching: Glob patterns like `users/*/email`
  - Rate limiting: Token bucket algorithm per agent/session
  - Audit logging: All operations with timestamp + context

- **Tool Routing** — Context-driven dynamic binding for multi-agent systems
  - Python: `AgentRegistry`, `ToolRouter`, `ToolDispatcher` in `routing.py`
  - Go: Implementation in `routing.go`
  - Node.js: TypeScript implementation in `routing.ts`
  - Rust: Implementation in `routing.rs`
  - Features: Tool registry, 6 routing strategies (ROUND_ROBIN, PRIORITY, LEAST_LOADED, STICKY, RANDOM, FASTEST)
  - Categories: CODE, SEARCH, DATABASE, MEMORY, VECTOR, GRAPH, ANALYTICS, API, FILE, SYSTEM
  - Local & remote agents with automatic failover

### Changed

- **Transaction Commit Return Type** — Now returns actual commit timestamp
  - Python: `Transaction.commit()` returns `int` (u64 HLC timestamp) instead of `None`
  - Go: `Transaction.Commit()` returns `(uint64, error)` instead of `error`
  - Node.js: `Transaction.commit()` returns `Promise<bigint>`
  - **Breaking**: Code checking `commit() == 0` should check for exceptions instead

- **Scan API Safety** — Minimum prefix length now enforced
  - Storage: `scan()` validates `prefix.len() >= MIN_SAFE_PREFIX_LEN`
  - Python: `scan_prefix()` requires ≥3 bytes, added `scan_prefix_unchecked()` for unsafe ops
  - Go: `ScanPrefix()` enforces validation, `ScanPrefixUnchecked()` for unsafe ops
  - Node.js: `scanPrefix()` enforces validation, `scanPrefixUnchecked()` for unsafe ops
  - **Breaking**: Short prefixes (< 3 bytes) now return error

- **Database Configuration** — Config parameters now actually applied
  - Python: `Database.open(config=...)` applies all config params via FFI (previously ignored)
  - Go: `OpenWithConfig(config)` applies configuration
  - Node.js: `Database.open(path, config)` applies configuration
  - **Breaking**: Config now has real effect on durability/performance (review production configs)

### Documentation

- **New Guides**
  - `docs/guides/policy-hooks.md` — Complete guide with examples in all 4 SDKs (600+ lines)
  - `docs/guides/tool-routing.md` — Multi-agent orchestration guide (550+ lines)
  - `docs/guides/context-query.md` — Token-aware retrieval for LLMs (500+ lines)
  - `docs/guides/graph-overlay.md` — Graph-on-KV design and usage (450+ lines)

- **SDK README Updates**
  - Python: Added Graph Overlay, Context Query, Policy Hooks, Tool Routing sections
  - Go: Added Graph Overlay, Context Query, Policy Hooks, Tool Routing sections
  - Node.js: Added Graph Overlay, Context Query, Policy Hooks, Tool Routing sections
  - Rust: Updated lib.rs with new module exports

- **Main Documentation Updates**
  - `docs/index.md`: New "Agent-Optimized Features" section
  - `docs/index.md`: Added links to 4 new guides
  - `docs/index.md`: Updated Quick Links table with Graph and Context Query

### Performance

- **Commit Timestamps**: No overhead (HLC is O(1), already existed but unused)
- **Config Validation**: ~50µs at `Database.open()`
- **Prefix Validation**: ~1µs per scan for length check
- **Context Query**: O(n log n) for n chunks, ~10ms for 100 chunks
- **Real Vector Search**: Embedding generation ~50-200ms (external API), HNSW ~1ms
- **Index-Aware SQL**: UPDATE/DELETE improved from O(N) to O(log N + m)
- **Graph Overlay**: 2× storage (in/out edges), ~10µs per edge operation
- **Policy Hooks**: ~5-50µs per hook (depends on complexity)
- **Tool Routing**: ~1ms for routing decision (10-100 tools)

### Security

- **Multi-Tenant Isolation**: Prefix-bounded scans prevent cross-tenant queries
- **SQL Injection Resistant**: MCP query parser uses grammar-based parsing
- **PII Protection**: Policy hooks enforce field-level redaction/encryption
- **Rate Limiting**: Token bucket prevents agent DoS attacks
- **Audit Trails**: All operations logged with commit timestamps
- **Fail-Safe Defaults**: Unsafe operations require explicit opt-in

### Fixed

- **Python Transaction Commit**: Now returns real HLC timestamp instead of hardcoded 0
- **Config Application**: Database config parameters are now actually applied via FFI
- **Vector Search Placeholder**: Replaced `vec![0.0f32; 128]` with real embedding generation
- **MCP Query Safety**: Fixed naive string parsing vulnerable to injection-like issues
- **SQL Performance**: Added secondary indexes to avoid O(N) table scans

---

## [0.3.1] - 2026-01-04

### Added
- **Anonymous Usage Analytics** — Optional, privacy-respecting usage information collection
  - `database_opened` event — Track database initialization across SDKs
  - `error` event — Static error tracking (error type + code location only, no sensitive data)
  - Stable anonymous machine ID (SHA-256 hash, no PII)
  - Environment variable opt-out: `SOCHDB_DISABLE_ANALYTICS=true`
  - Graceful degradation when analytics dependencies unavailable
  - Python: Optional `posthog` dependency in `[analytics]` extra
  - JavaScript: Optional `posthog-node` in `optionalDependencies`
  - Rust: Feature flag `analytics` for `sochdb-core`

### Changed
- **Analytics Privacy-First Design** — No sensitive data collection
  - Error tracking sends only static `error_type` and `location` (e.g., "query_error" @ "sql.execute")
  - No dynamic error messages, user data, file paths, or identifiable information
  - SDK context: version, OS, architecture only
  - All data sent to PostHog with user-controlled opt-out
- **Documentation Updates** — Version references updated to 0.3.1 across all guides
  - Updated benchmark versions in README
  - Updated installation instructions in all SDK guides
  - Consistent versioning across Python, JavaScript, Rust, and Go documentation

### Fixed
- **JavaScript Analytics Import** — Fixed ESM import path in `database.ts` (`.js` extension required)
- **Rust Analytics Feature** — Enabled `analytics` feature by default in client crates
  - Added to `sochdb-client`, `sochdb-python`, `sochdb-grpc`
  - Added `json` feature to `ureq` dependency for PostHog integration
- **Test Version Mismatch** — Updated JavaScript test expectations to match 0.3.1 version

---

## [0.3.0] - 2026-01-03

### Added
- **Namespace Isolation** — Type-safe multi-tenancy with logical database namespaces
  - `Namespace` and `Collection` APIs for tenant-scoped data isolation
  - No key prefixing required — true isolation at the storage layer
  - Per-namespace configuration and collection management
  - Available in Python, Go, Node.js, and Rust SDKs
- **Hybrid Search** — Dense vector (HNSW) + sparse BM25 text search
  - `HybridSearchEngine` with Reciprocal Rank Fusion (RRF)
  - Configurable alpha weighting (vector vs. keyword balance)
  - `InvertedIndex` for full-text search with BM25 scoring
  - Automatic tokenization with stop-word filtering
- **Multi-Vector Documents** — Multiple embeddings per document
  - Store title, abstract, and content vectors separately
  - Aggregation strategies: max-pooling, mean-pooling, weighted sum
  - `MultiVectorMapping` for chunk-level semantic search
- **Context-Aware Queries** — LLM-optimized retrieval with token budgeting
  - `ContextQuery` builder with fluent API
  - Token estimation for GPT-4, Claude, Gemini models
  - Semantic deduplication to avoid redundant results
  - Automatic content extraction with configurable fields
- **Tombstone-Based Deletion** — Logical deletion for vector indices
  - `TombstoneManager` for efficient soft deletes
  - Filtering during search without rebuilding index
  - Lazy compaction for performance
- **Enhanced Error Taxonomy** — Structured error handling across all SDKs
  - `ErrorCode` enum with 40+ specific error types
  - Hierarchical error categories (Storage, Index, Query, IPC)
  - Remediation hints for common errors
  - Python SDK: Enhanced exception classes with error codes

### Changed
- **SDK Documentation** — Comprehensive updates across all languages
  - Removed version numbers from installation instructions (easier maintenance)
  - Added "What's New in Latest Release" section to all READMEs
  - Developer-friendly examples for namespace, hybrid search, and multi-vector
  - Consistent documentation structure across Python, Go, Node.js, and Rust SDKs
- **Python SDK** — Namespace API integration in Database class
  - `create_namespace()`, `namespace()`, `list_namespaces()` methods
  - Namespace examples in Quick Start section
- **Go SDK** — Updated with 0.3.0 feature examples
- **Node.js SDK** — TypeScript examples for all new features
- **Rust SDK** — Updated with NamespaceHandle and new vector APIs

### Fixed
- **Node.js VERSION Bug** — VERSION constant was stuck at 0.2.8 instead of matching package.json
  - Fixed `sochdb-js/src/index.ts` VERSION export
  - Updated `database.test.ts` to test correct version

### Breaking Changes
- Namespace API requires explicit namespace creation before use
- Collection configuration is immutable after creation
- Python SDK: Import path remains `sochdb` (package name: `sochdb-client`)

---

## [0.2.9] - 2026-01-02

### Added
- **Comprehensive benchmark suite** with real-world LLM embeddings (Azure OpenAI)
  - SochDB vs ChromaDB: **3× faster** vector search
  - SochDB vs LanceDB: **22× faster** vector search
  - Recall@k benchmarks showing **>98% recall** with sub-millisecond latency
  - End-to-end RAG bottleneck analysis (API is 333× slower than database)
- **Full SQL engine support in Go SDK** with DDL/DML operations matching Python/JS SDKs
- **Community health files** for open source project
  - CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
  - SECURITY.md with vulnerability reporting policy and response timelines
  - Issue templates (bug report, feature request, support) with YAML validation
- **Unified release workflow** with automated SDK publishing to crates.io, PyPI, and npm
- **360° performance report** with retrieval quality, latency, throughput, and resource efficiency metrics

### Changed
- Simplified release workflow with improved error handling and protected branch support
- Updated documentation with consistent SDK guides across all languages
- Enhanced benchmark reports with real-world embedding comparisons

### Fixed
- Rust compilation errors in storage.rs
- Go SDK test output formatting (removed redundant newlines)
- Path dependencies for crates.io publishing
- Wire protocol documentation for all SDKs

---

## [0.2.7] - 2026-01-01

### Added
- **Full SQL engine support** in Python SDK with DDL/DML operations (CREATE/DROP TABLE, INSERT, SELECT, UPDATE, DELETE)
- **Full SQL engine support** in JavaScript SDK with complete SQL parser and executor
- **Go embedded server mode** - automatically starts/stops sochdb-server without external setup
- **Transaction SQL support** - execute() method added to Transaction class in Python SDK
- SQL storage using KV backend with `_sql/tables/` prefix (tables and rows stored as JSON)
- WHERE clause support with operators: =, !=, <, >, >=, <=, LIKE, NOT LIKE
- ORDER BY with ASC/DESC, LIMIT, and OFFSET support
- Data types: INT, TEXT, FLOAT, BOOL, BLOB

### Changed
- JavaScript stats() response format changed from key=value to valid JSON
- Go SDK now defaults to embedded mode (Config.Embedded = true)
- Python SDK execute_sql() added as alias for execute() for documentation consistency

### Fixed
- **Critical**: Python SQL API now returns actual query results (was returning empty rows)
- **Critical**: IPC server stats command now returns valid JSON format
- **Critical**: Go SDK no longer requires external server process
- JavaScript SDK ESM imports now use explicit .js extensions

---

## [0.2.6] - 2026-01-01

### Added
- **Enhanced scan() method** across all SDKs for efficient prefix-based iteration and multi-tenant isolation
- **Full SQL support** in Python SDK (CREATE TABLE, INSERT, SELECT, JOIN, WHERE, GROUP BY, aggregations)
- **SQL integration** in Rust SDK via sochdb-query crate with async IPC
- **Bulk vector API** in Python SDK (~1,600 vec/s, 12× faster than FFI loop)
- **Zero-copy reads** in Rust SDK for large value optimization
- **Async IPC client** in Rust SDK with Tokio runtime
- **SQL examples** in all SDK READMEs and examples/ directory
- **Comprehensive SDK guides** in docs/guides/ for Go, Python, JavaScript/Node.js, and Rust
- **SQL API documentation** in docs/api-reference/sql-api.md

### Changed
- Updated all SDK READMEs with v0.2.6 features and complete examples
- Improved documentation structure with consistent formatting across all guides
- Enhanced TypeScript definitions in JavaScript SDK
- Updated wire protocol documentation for Little Endian format
- Improved error messages across all SDKs

### Fixed
- **Path operations** (putPath/getPath) in JavaScript SDK now use correct wire format
- Binary encoding issues in Go SDK path operations
- Python SDK _bin/ directory now properly excluded from git (built during CI/CD)
- Scan range calculation simplified using semicolon trick (';' after '/' in ASCII)

---

## [0.2.3] - 2025-12-31

### Added
- Go SDK in-repo (`sochdb-go/`) with examples
- Rust SDK guide and multi-language docs index updates

### Fixed
- TBP writer/reader null bitmap sizing and fixed-row offset correctness
- Crates.io publishability for workspace path dependencies

---

## [0.1.0] - 2024-12-27

### Added

#### Core Database
- **ACID Transactions** with MVCC and Serializable Snapshot Isolation (SSI)
- **Write-Ahead Log (WAL)** with group commit optimization
- **Path-based data model** with O(|path|) resolution via Trie-Columnar Hybrid (TCH)
- **Columnar storage** with automatic projection pushdown

#### LLM Features
- **TOON Format** — 40-66% token reduction compared to JSON
- **Context Query Builder** — token budgeting and priority-based truncation
- **Vector Search** — HNSW index with F32/F16/BF16 quantization

#### Python SDK
- **Embedded mode** via FFI for single-process applications
- **IPC mode** via Unix sockets for multi-process scenarios
- **Bulk API** for high-throughput vector operations (~1,600 vec/s vs ~130 vec/s FFI)
- Pre-built wheels for Linux (x86_64, aarch64), macOS (universal2), Windows (x64)

#### MCP Integration
- **sochdb-mcp** server for Claude Desktop, Cursor, and Goose
- Tools: `sochdb_put`, `sochdb_get`, `sochdb_scan`, `sochdb_delete`, `sochdb_context_query`

#### Indexing
- **HNSW** vector index with configurable M, ef_construction, ef_search
- **B-Tree** index for ordered key access
- **Bloom filters** for existence checks

### Performance
- Ordered index can be disabled for ~20% faster writes (point lookups only)
- Group commit reduces fsync overhead for high-throughput writes
- Columnar storage minimizes I/O for selective queries

### Known Limitations
- Single-node only (no distributed mode, replication, or clustering)
- Python SDK requires `SOCHDB_LIB_PATH` environment variable for FFI mode

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.3 | 2025-12-31 | Multi-SDK + publish readiness |
| 0.1.0 | 2024-12-27 | Initial release |

---

## Upgrade Guide

### Upgrading to 0.1.0

This is the initial release. No upgrade path required.

---

## Contributors

Thanks to all contributors who helped make SochDB possible!

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## Links

- [Documentation](docs/index.md)
- [Quick Start](docs/QUICKSTART.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
