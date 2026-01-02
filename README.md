# 🎬 ToonDB

### The LLM‑Native Database

**Token‑optimized context • Columnar storage • Built‑in vector search • Embedded-first**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org/)

* **40–66% fewer tokens** for tabular context via **TOON** (Tabular Object‑Oriented Notation)
* **SQL support** with full SQL-92 syntax for relational queries
* **Context Query Builder**: assemble *system + user + history + retrieval* under a token budget
* **Native HNSW vector search** (F32/F16/BF16) with optional quantization
* **ACID transactions** (MVCC + WAL + Serializable Snapshot Isolation)
* **Two access modes**: **Embedded (FFI)** and **IPC (Unix sockets)** via Python SDK

**Quick links:** [📚 Documentation](https://docs.toondb.dev) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [TOON Format](#-toon-format) • [Benchmarks](#-benchmarks) • [RFD](docs/rfds/RFD-001-ai-native-database.md)

---

## Why ToonDB exists

Most "agent stacks" still glue together:

* a KV store (sessions / state)
* a vector DB (retrieval)
* a prompt packer (context budgeting, truncation)
* a relational DB (metadata)

…and then spend weeks maintaining brittle context assembly and token budgeting.

**ToonDB collapses that stack into one LLM‑native substrate**: you store structured data + embeddings + history *and* ask the DB to produce a token‑efficient context payload.

---

## What you can rely on today (verified features)

### ✅ LLM / Agent primitives

* **TOON output format** for compact, model-friendly context
* **Context Query Builder** with token budgeting + priority-based truncation
* **Vector search** (HNSW), integrated into retrieval workflows

### ✅ Database fundamentals

* **SQL support** with full SQL-92 syntax (SELECT, INSERT, UPDATE, DELETE, JOINs)
* **ACID transactions** with **MVCC**
* **WAL durability** + **group commit**
* **Serializable Snapshot Isolation (SSI)**
* **Columnar storage** with projection pushdown (read only the columns you need)

### ✅ Developer experience

* **Rust client** (`toondb-client`)
* **Python SDK** with:

  * **Embedded mode (FFI)** for lowest latency
  * **IPC mode (Unix sockets)** for multi-process / service scenarios
* **Bulk vector operations** for high-throughput ingestion

### Known limits

* **Single-node only** (no replication / clustering yet)

---

## ToonDB in one picture

| Problem           | Typical approach               | ToonDB approach                     |
| ----------------- | ------------------------------ | ----------------------------------- |
| Token waste       | JSON/SQL payload bloat         | **TOON**: dense, table-like output  |
| RAG plumbing      | External vector DB + glue      | **Built-in HNSW** + quantization    |
| Context assembly  | multiple reads + custom packer | **One context query** with a budget |
| I/O amplification | row store reads all columns    | **columnar** + projection pushdown  |

---

## 📦 Quick Start

### Installation

Choose your preferred SDK:

```bash
# Python
pip install toondb-client

# Node.js / TypeScript
npm install @sushanth/toondb

# Go
go get github.com/toondb/toondb/toondb-go@v0.2.7

# Rust - add to Cargo.toml
# toondb = "0.2"
```

### Hello World

#### Python

```python
from toondb import Database

db = Database.open("./my_db")
db.put(b"users/alice", b"Alice Smith")
print(db.get(b"users/alice").decode())  # "Alice Smith"
db.close()
```

#### Node.js / TypeScript

```typescript
import { ToonDatabase } from '@sushanth/toondb';

const db = new ToonDatabase('./my_db');
await db.put('users/alice', 'Alice Smith');
console.log(await db.get('users/alice'));  // "Alice Smith"
await db.close();
```

#### Go

```go
package main

import (
    "fmt"
    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, _ := toondb.Open("./my_db")
    defer db.Close()
    
    db.Put([]byte("users/alice"), []byte("Alice Smith"))
    value, _ := db.Get([]byte("users/alice"))
    fmt.Println(string(value))  // "Alice Smith"
}
```

#### Rust

```rust
use toondb::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("./my_db")?;
    
    db.put(b"users/alice", b"Alice Smith")?;
    if let Some(value) = db.get(b"users/alice")? {
        println!("{}", String::from_utf8_lossy(&value));  // "Alice Smith"
    }
    Ok(())
}
```

### Vector Search Example

#### Python

```python
from toondb import VectorIndex
import numpy as np

# Create HNSW index
index = VectorIndex(dimension=384)

# Add vectors
embeddings = np.random.randn(1000, 384).astype(np.float32)
index.bulk_add(embeddings, ids=list(range(1000)))

# Search
query = np.random.randn(384).astype(np.float32)
results = index.search(query, k=10)
print(results)  # [(id, distance), ...]
```

#### Node.js / TypeScript

```typescript
import { VectorIndex } from '@sushanth/toondb';

const index = new VectorIndex({ dimension: 384 });

// Add vectors
await index.add('doc1', embedding1);
await index.add('doc2', embedding2);

// Search
const results = await index.search(queryEmbedding, 10);
console.log(results);  // [{ id, distance }, ...]
```

### SDK Feature Matrix

| Feature | Python | Node.js | Go | Rust |
|---------|--------|---------|-----|------|
| Basic KV | ✅ | ✅ | ✅ | ✅ |
| Transactions | ✅ | ✅ | ✅ | ✅ |
| Vector Search | ✅ | ✅ | ✅ | ✅ |
| Path API | ✅ | ✅ | ✅ | ✅ |
| Bulk Operations | ✅ | ⏳ | ⏳ | ✅ |
| IPC Mode | ✅ | ⏳ | ⏳ | ✅ |

---

## 🏗 Architecture

```text
App / Agent Runtime
   │
   ├─ toondb-client (Rust / Python)
   │
   ├─ toondb-query   (planner + TOON encoder + context builder)
   └─ toondb-kernel  (MVCC + WAL + catalog)
        ├─ toondb-storage (columnar LSCS + mmap)
        └─ toondb-index   (B-Tree + HNSW)
```

### Crate Overview

| Crate | Description | Key Components |
|-------|-------------|----------------|
| `toondb-core` | Core types and TOON format | `ToonValue`, `ToonSchema`, `ToonTable`, codec |
| `toondb-kernel` | Database kernel | WAL, MVCC, transactions, catalog |
| `toondb-storage` | Storage engine | LSCS columnar, mmap, block checksums |
| `toondb-index` | Index structures | B-Tree, HNSW vector index |
| `toondb-query` | Query execution | Cost optimizer, context builder, TOON-QL |
| `toondb-client` | Client SDK | `ToonConnection`, `PathQuery`, `BatchWriter` |
| `toondb-plugin-logging` | Logging plugin | Structured logging, tracing |

---

## 📄 TOON Format

TOON (Tabular Object-Oriented Notation) is ToonDB's compact serialization format designed specifically for LLM context windows—a token-optimized format that dramatically reduces token consumption.

### Format Specification

```ebnf
document     ::= table_header newline row*
table_header ::= name "[" count "]" "{" fields "}" ":"
name         ::= identifier
count        ::= integer
fields       ::= field ("," field)*
field        ::= identifier
row          ::= value ("," value)* newline
value        ::= null | bool | number | string | array | ref
```

### Token Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                      JSON (156 tokens)                          │
├─────────────────────────────────────────────────────────────────┤
│ [                                                               │
│   {"id": 1, "name": "Alice", "email": "alice@example.com"},    │
│   {"id": 2, "name": "Bob", "email": "bob@example.com"},        │
│   {"id": 3, "name": "Charlie", "email": "charlie@example.com"} │
│ ]                                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      TOON (52 tokens) — 67% reduction!          │
├─────────────────────────────────────────────────────────────────┤
│ users[3]{id,name,email}:                                        │
│ 1,Alice,alice@example.com                                       │
│ 2,Bob,bob@example.com                                           │
│ 3,Charlie,charlie@example.com                                   │
└─────────────────────────────────────────────────────────────────┘
```

### TOON Value Types

| Type | TOON Syntax | Example |
|------|-------------|---------|
| Null | `∅` | `∅` |
| Boolean | `T` / `F` | `T` |
| Integer | number | `42`, `-17` |
| Float | decimal | `3.14159` |
| String | text or `"quoted"` | `Alice`, `"hello, world"` |
| Array | `[items]` | `[1,2,3]` |
| Reference | `ref(table,id)` | `ref(users,42)` |
| Binary | `b64:data` | `b64:SGVsbG8=` |

---

## 🔍 Vector Search

ToonDB includes an HNSW (Hierarchical Navigable Small World) index for similarity search.

### Configuration

```rust
use toondb_index::{HNSWIndex, HNSWConfig, DistanceMetric};

// Create index with custom parameters
let config = HNSWConfig {
    m: 16,                          // Max connections per layer
    m_max: 32,                      // Max connections at layer 0
    ef_construction: 200,           // Build-time search width
    ef_search: 50,                  // Query-time search width
    metric: DistanceMetric::Cosine, // Or Euclidean, DotProduct
    ..Default::default()
};

let index = HNSWIndex::with_config(config);
```

### Vector Operations

```rust
use toondb::{ToonConnection, VectorCollection, SearchResult};

let conn = ToonConnection::open("./vectors")?;

// Insert vectors
let embedding: Vec<f32> = get_embedding("Hello world");
conn.vector_insert("documents", 1, &embedding, Some(metadata))?;

// Search similar vectors
let query_embedding = get_embedding("Hi there");
let results: Vec<SearchResult> = conn.vector_search("documents", &query_embedding, 10)?;

for result in results {
    println!("ID: {}, Distance: {:.4}", result.id, result.distance);
}
```

### Distance Metrics

| Metric | Use Case | Formula |
|--------|----------|---------|
| `Cosine` | Text embeddings, normalized vectors | `1 - (a·b)/(‖a‖‖b‖)` |
| `Euclidean` | Spatial data, unnormalized | `√Σ(aᵢ-bᵢ)²` |
| `DotProduct` | When vectors are pre-normalized | `-a·b` |

### Vector Quantization

ToonDB supports optional quantization to reduce memory usage with minimal recall loss:

| Precision | Memory | Search Latency | Use Case |
|-----------|--------|----------------|----------|
| `F32` | 100% (baseline) | Baseline | Maximum precision |
| `F16` | 50% | ~Same | General embeddings |
| `BF16` | 50% | ~Same | ML model compatibility |

> **Tip**: F16 typically provides 50% memory reduction with <1% recall degradation for most embedding models.

---

## 🔐 Transactions

ToonDB provides **ACID transactions** with MVCC (Multi-Version Concurrency Control) and WAL durability.

### ACID Guarantees

| Property | Implementation |
|----------|----------------|
| **Atomicity** | Buffered writes with all-or-nothing commit |
| **Consistency** | Schema validation before commit |
| **Isolation** | MVCC snapshots with read/write set tracking |
| **Durability** | WAL with fsync, group commit support |

### Transaction Modes

```rust
use toondb::{ToonConnection, ClientTransaction, IsolationLevel};

// Auto-commit (implicit transaction per operation)
conn.put("users/1/name", b"Alice")?;

// Explicit transaction with isolation level
let txn = conn.begin_with_isolation(IsolationLevel::Serializable)?;
conn.put_in_txn(txn, "users/1/name", b"Alice")?;
conn.put_in_txn(txn, "users/1/email", b"alice@example.com")?;
conn.commit(txn)?;  // SSI validation happens here

// Rollback on error
let txn = conn.begin()?;
if let Err(e) = do_something(&conn, txn) {
    conn.rollback(txn)?;
    return Err(e);
}
conn.commit(txn)?;
```

### Isolation Levels

| Level | Description | Status |
|-------|-------------|--------|
| `ReadCommitted` | Sees committed data at statement start | ✅ Implemented |
| `SnapshotIsolation` | Reads see consistent point-in-time view | ✅ Implemented |
| `Serializable` | SSI with rw-antidependency cycle detection | ✅ Implemented |

### WAL Sync Modes

```rust
use toondb_kernel::SyncMode;

let config = DatabaseConfig {
    sync_mode: SyncMode::Normal,  // Group commit (recommended)
    // sync_mode: SyncMode::Full, // Fsync every commit (safest)
    // sync_mode: SyncMode::Off,  // Periodic fsync (fastest)
    ..Default::default()
};
```

### Durability Presets

ToonDB provides pre-configured durability settings for common use cases:

| Preset | Sync Mode | Group Commit | Best For |
|--------|-----------|--------------|----------|
| `throughput_optimized()` | Normal | Large batches | High-volume ingestion |
| `latency_optimized()` | Full | Small batches | Real-time applications |
| `max_durability()` | Full | Disabled | Financial/critical data |

```rust
use toondb::ConnectionConfig;

// High-throughput batch processing
let config = ConnectionConfig::throughput_optimized();

// Low-latency real-time access
let config = ConnectionConfig::latency_optimized();

// Maximum durability (fsync every commit, no batching)
let config = ConnectionConfig::max_durability();
```

---

## 🌳 Path API

ToonDB's unique path-based API provides **O(|path|)** resolution via the Trie-Columnar Hybrid (TCH) structure.

### Path Format

```
collection/document_id/field
table/row_id/column
```

### Operations

```rust
use toondb::{ToonConnection, PathQuery};

let conn = ToonConnection::open("./data")?;

// Put a value at a path
conn.put("users/1/name", b"Alice")?;
conn.put("users/1/profile/avatar", avatar_bytes)?;

// Get a value
let name = conn.get("users/1/name")?;

// Delete at path
conn.delete("users/1/profile/avatar")?;

// Scan by prefix (returns all matching key-value pairs)
let user_data = conn.scan("users/1/")?;
for (key, value) in user_data {
    println!("{}: {:?}", key, value);
}

// Query using PathQuery builder
let results = PathQuery::from_path(&conn, "users")
    .select(&["id", "name", "email"])
    .where_eq("status", "active")
    .order_by("created_at", Order::Desc)
    .limit(10)
    .execute()?;
```

### Path Resolution

```
Path: "users/1/name"
      
      TCH Resolution (O(3) = O(|path|))
      ┌─────────────────────────────────┐
      │  users  →  1  →  name           │
      │    ↓       ↓       ↓            │
      │  Table   Row   Column           │
      │  Lookup  Index  Access          │
      └─────────────────────────────────┘
      
vs    B-Tree (O(log N))
      ┌─────────────────────────────────┐
      │  Binary search through          │
      │  potentially millions of keys   │
      └─────────────────────────────────┘
```

### Optional Ordered Index

ToonDB's ordered index can be disabled for write-optimized workloads:

```rust
use toondb::ConnectionConfig;

// Default: ordered index enabled (O(log N) prefix scans)
let config = ConnectionConfig::default();

// Write-optimized: disable ordered index (~20% faster writes)
let mut config = ConnectionConfig::default();
config.enable_ordered_index = false;
// Note: scan_prefix becomes O(N) instead of O(log N + K)
```

| Mode | Write Speed | Prefix Scan | Use Case |
|------|-------------|-------------|----------|
| Ordered index **on** | Baseline | O(log N + K) | Read-heavy, prefix queries |
| Ordered index **off** | ~20% faster | O(N) | Write-heavy, point lookups |

---

## 📊 Context Query Builder

Build LLM context with automatic token budget management.

```rust
use toondb_query::{ContextSection, ContextSelectQuery};
use toondb::ContextQueryBuilder;

let context = ContextQueryBuilder::new()
    .for_session("session_123")
    .with_budget(4096)  // Token budget
    
    // System prompt (highest priority)
    .literal("SYSTEM", -1, "You are a helpful assistant")
    
    // User profile from database
    .section("USER", 0)
        .get("user.profile.{name, email, preferences}")
        .done()
    
    // Recent conversation history
    .section("HISTORY", 1)
        .last(10, "messages")
        .where_eq("session_id", session_id)
        .done()
    
    // Relevant documents via vector search
    .section("DOCS", 2)
        .search("knowledge_base", "query_embedding", 5)
        .min_score(0.7)
        .done()
    
    .truncation(TruncationStrategy::PriorityDrop)
    .format(ContextFormat::Toon)
    .execute()?;

println!("Tokens used: {}/{}", context.token_count, 4096);
println!("Context:\n{}", context.context);
```

---

## 🔌 Plugin System

ToonDB uses a plugin architecture for extensibility without dependency bloat.

### Extension Types

| Extension | Purpose | Example |
|-----------|---------|---------|
| `StorageExtension` | Alternative backends | RocksDB, LSCS |
| `IndexExtension` | Custom indexes | Learned index, full-text |
| `ObservabilityExtension` | Metrics/tracing | Prometheus, DataDog |
| `CompressionExtension` | Compression algos | LZ4, Zstd |

### Implementing a Plugin

```rust
use toondb_kernel::{Extension, ExtensionInfo, ObservabilityExtension};

struct PrometheusMetrics { /* ... */ }

impl Extension for PrometheusMetrics {
    fn info(&self) -> ExtensionInfo {
        ExtensionInfo {
            name: "prometheus-metrics".into(),
            version: "1.0.0".into(),
            description: "Prometheus metrics export".into(),
            author: "Your Name".into(),
            capabilities: vec![ExtensionCapability::Observability],
        }
    }
    
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl ObservabilityExtension for PrometheusMetrics {
    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        // Push to Prometheus
    }
    
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        // Set gauge value
    }
    
    fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        // Record histogram
    }
    
    // ... tracing methods
}

// Register the plugin
db.plugins().register_observability(Box::new(PrometheusMetrics::new()))?;
```

---

## 🧮 Batch Operations

High-throughput batch operations with group commit optimization.

```rust
use toondb::{ToonConnection, BatchWriter, GroupCommitConfig};

let conn = ToonConnection::open("./data")?;

// Batch insert with auto-commit
let result = conn.batch()
    .max_batch_size(1000)
    .auto_commit(true)
    .insert("events", vec![("id", id1), ("data", data1)])
    .insert("events", vec![("id", id2), ("data", data2)])
    // ... more inserts
    .execute()?;

println!("Executed: {}, Failed: {}, Duration: {}ms", 
    result.ops_executed, result.ops_failed, result.duration_ms);

// Bulk insert for large datasets
let rows: Vec<Vec<(&str, ToonValue)>> = generate_rows(10_000);
let result = conn.bulk_insert("events", rows)?;
```

### Group Commit Formula

ToonDB calculates optimal batch size using:

```
N* = √(2 × L_fsync × λ / C_wait)

Where:
- L_fsync = fsync latency (~5ms typical)
- λ = arrival rate (ops/sec)
- C_wait = cost per unit wait time
```

---

## 📈 Benchmarks

### Token Efficiency (TOON vs JSON)

| Dataset | JSON Tokens | TOON Tokens | Reduction |
|---------|-------------|-------------|-----------|
| Users (100 rows, 5 cols) | 2,340 | 782 | **66.6%** |
| Events (1000 rows, 3 cols) | 18,200 | 7,650 | **58.0%** |
| Products (500 rows, 8 cols) | 15,600 | 5,980 | **61.7%** |

### Vector Search (HNSW)

| Vectors | Dimensions | QPS (ef=50) | Recall@10 |
|---------|------------|-------------|-----------|
| 100K | 384 | 12,400 | 0.98 |
| 1M | 384 | 8,200 | 0.97 |
| 10M | 384 | 4,100 | 0.96 |

### I/O Reduction (Columnar)

| Query | Row Store | ToonDB Columnar | Reduction |
|-------|-----------|-----------------|-----------|
| SELECT 2 of 10 cols | 100% | 20% | **80%** |
| SELECT 1 of 20 cols | 100% | 5% | **95%** |

### Performance (vs SQLite)

> **Benchmark Methodology**: This benchmark compares ToonDB to SQLite under similar durability settings (`WAL` mode, `synchronous=NORMAL`), both in Rust and Python. Results depend on hardware, build flags, dataset size, and workload patterns.

Benchmarks running on 100k records (Apple M1/M2 class hardware):

| Database | Mode | Insert Rate | Notes |
|----------|------|-------------|-------|
| **SQLite** | File (WAL) | ~1.16M ops/sec | Industry standard |
| **ToonDB** | Embedded (WAL) | ~760k ops/sec | Group commit disabled for sequential |
| **ToonDB** | put_raw | ~1.30M ops/sec | Direct storage layer bypass |
| **ToonDB** | insert_row_slice | ~1.29M ops/sec | Zero-allocation row API |

> **Note**: Performance varies by workload. ToonDB excels in LLM context assembly scenarios (token-efficient output, vector search, context budget management). SQLite remains the gold standard for general-purpose relational workloads. See [Before Heavy Production Use](#️-before-heavy-production-use) for current limitations.

---

## 🛠 Configuration Reference

### DatabaseConfig

```rust
pub struct DatabaseConfig {
    /// Enable group commit for better throughput
    pub group_commit: bool,           // default: true
    
    /// WAL sync mode
    pub sync_mode: SyncMode,          // default: Normal
    
    /// Maximum WAL size before checkpoint
    pub max_wal_size: u64,            // default: 64MB
    
    /// Memtable size before flush
    pub memtable_size: usize,         // default: 4MB
    
    /// Block cache size
    pub block_cache_size: usize,      // default: 64MB
    
    /// Compression algorithm
    pub compression: Compression,      // default: LZ4
}
```

### HNSWConfig

```rust
pub struct HNSWConfig {
    /// Max connections per node per layer
    pub m: usize,                     // default: 16
    
    /// Max connections at layer 0
    pub m_max: usize,                 // default: 32
    
    /// Construction-time search width
    pub ef_construction: usize,       // default: 200
    
    /// Query-time search width (adjustable)
    pub ef_search: usize,             // default: 50
    
    /// Distance metric
    pub metric: DistanceMetric,       // default: Cosine
    
    /// Level multiplier (mL = 1/ln(M))
    pub ml: f32,                      // default: calculated
}
```

---

## 📚 API Reference

### ToonConnection

| Method | Description | Returns |
|--------|-------------|---------|
| `open(path)` | Open/create database | `Result<ToonConnection>` |
| `create_table(schema)` | Create a new table | `Result<CreateResult>` |
| `drop_table(name)` | Drop a table | `Result<DropResult>` |
| `batch()` | Start a batch writer | `BatchWriter` |
| `put(path, value)` | Put value at path | `Result<()>` |
| `get(path)` | Get value at path | `Result<Option<Vec<u8>>>` |
| `delete(path)` | Delete at path | `Result<()>` |
| `scan(prefix)` | Scan path prefix | `Result<Vec<(String, Vec<u8>)>>` |
| `begin()` | Begin transaction | `Result<TxnHandle>` |
| `commit(txn)` | Commit transaction | `Result<()>` |
| `rollback(txn)` | Rollback transaction | `Result<()>` |
| `vector_insert(...)` | Insert vector | `Result<()>` |
| `vector_search(...)` | Search similar vectors | `Result<Vec<SearchResult>>` |
| `fsync()` | Force sync to disk | `Result<()>` |
| `checkpoint()` | Create checkpoint | `Result<u64>` |
| `stats()` | Get statistics | `ClientStats` |

### PathQuery

| Method | Description | Returns |
|--------|-------------|---------|
| `from_path(conn, path)` | Create query from path | `PathQuery` |
| `select(cols)` | Select columns | `Self` |
| `project(cols)` | Alias for select | `Self` |
| `where_eq(field, val)` | Equality filter | `Self` |
| `where_gt(field, val)` | Greater than filter | `Self` |
| `where_like(field, pat)` | Pattern match | `Self` |
| `order_by(field, dir)` | Sort results | `Self` |
| `limit(n)` | Limit results | `Self` |
| `offset(n)` | Skip results | `Self` |
| `execute()` | Execute query | `Result<QueryResult>` |
| `execute_toon()` | Execute and return TOON | `Result<String>` |

### ToonValue

| Variant | Rust Type | Description |
|---------|-----------|-------------|
| `Null` | — | Null value |
| `Bool(bool)` | `bool` | Boolean |
| `Int(i64)` | `i64` | Signed integer |
| `UInt(u64)` | `u64` | Unsigned integer |
| `Float(f64)` | `f64` | 64-bit float |
| `Text(String)` | `String` | UTF-8 string |
| `Binary(Vec<u8>)` | `Vec<u8>` | Binary data |
| `Array(Vec<ToonValue>)` | `Vec<ToonValue>` | Array of values |
| `Object(HashMap<String, ToonValue>)` | `HashMap` | Key-value object |
| `Ref { table, id }` | — | Foreign key reference |

### ToonType

| Type | Description |
|------|-------------|
| `Int` | 64-bit signed integer |
| `UInt` | 64-bit unsigned integer |
| `Float` | 64-bit float |
| `Text` | UTF-8 string |
| `Bool` | Boolean |
| `Bytes` | Binary data |
| `Vector(dim)` | Float vector with dimension |
| `Array(inner)` | Array of inner type |
| `Optional(inner)` | Nullable type |
| `Ref(table)` | Foreign key to table |

---

## 🔧 Building from Source

### Prerequisites

- Rust 2024 edition (1.75+)
- Clang/LLVM (for SIMD optimizations)

### Build

```bash
# Clone the repository
git clone https://github.com/toondb/toondb.git
cd toondb

# Build all crates
cargo build --release

# Run tests
cargo test --all

# Run benchmarks
cargo bench
```

### Feature Flags

| Feature | Crate | Description |
|---------|-------|-------------|
| `simd` | toondb-client | SIMD optimizations for column access |
| `embedded` | toondb-client | Use kernel directly (no IPC) |
| `full` | toondb-kernel | All kernel features |

---

## ⚠️ Before heavy production use

* **Single node** (no replication / clustering)
* **WAL growth**: call `checkpoint()` periodically for long-running services
* **Group commit**: tune per workload (disable for strictly sequential writes)

---

## 🚧 Roadmap (high level)

* Cost-based optimizer: experimental
* Agent flow metadata schema: planned
* Agent runtime library: planned
* Adaptive group commit: planned
* WAL compaction / auto-truncation: planned

---

## 🤖 Vision: ToonDB as an Agentic Framework Foundation

ToonDB is designed to be the **brain, memory, and registry** for AI agents—not by embedding a programming language, but by storing agent metadata that external runtimes interpret.

### The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Agent Runtime│    │    ToonDB    │    │     LLM      │   │
│  │  (executor)  │◄──►│  (metadata)  │    │   (worker)   │   │
│  └──────┬───────┘    └──────────────┘    └──────▲───────┘   │
│         │                                        │           │
│         │  1. Load flow from DB                  │           │
│         │  2. Build prompt from node config      │           │
│         │  3. Call LLM ─────────────────────────►│           │
│         │  4. Parse result, update state         │           │
│         │  5. Choose next edge, repeat           │           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### What ToonDB Stores

| Table | Purpose |
|-------|---------|
| `agent_flows` | Flow definitions: name, entry node, version |
| `agent_nodes` | Nodes: LLM steps, tool calls, decisions, loops, reflections |
| `agent_edges` | Edges with conditions for routing |
| `agent_sessions` | Runtime state per user/conversation |
| `agent_reflections` | Feedback and learning data |

### Node Types

Flows are graphs where each node has a `kind`:

- **`llm_step`** — Call the LLM with a prompt template
- **`tool_call`** — Execute a tool (API, function, DB query)
- **`decision`** — Branch based on previous output
- **`loop_start` / `loop_end`** — Iteration with exit conditions
- **`reflection`** — Ask LLM to evaluate and improve
- **`subflow`** — Invoke another flow

### Example: Support Agent Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Classify   │────►│  Retrieve   │────►│   Answer    │
│   Intent    │     │   Context   │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐            │
                    │   Reflect   │◄───────────┘
                    │  (optional) │
                    └─────────────┘
```

The LLM only sees **one node at a time**:

```text
flow: support_assistant
node: classify_intent
goal: classify the user's message
input:
  user_message: "I can't access my account"
context:
  last_episodes: [...]
allowed_outputs: ["billing", "bug", "feature", "other"]
```

This keeps prompts small and stable. The runtime handles control flow.

### Why This Approach

| Benefit | Description |
|---------|-------------|
| **Separation of concerns** | ToonDB = data, Runtime = execution, LLM = reasoning |
| **Language-agnostic** | Rust, Python, TypeScript runtimes share the same flows |
| **Debuggable** | Every step, state change, and decision is in the DB |
| **Learnable** | Reflection nodes + stored feedback enable continuous improvement |
| **No prompt injection risk** | LLM never sees "execute this code"—just structured tasks |

### Built-in Patterns (Planned)

Templates for common agentic patterns:

- **Reflection loop** — Execute, evaluate, retry if needed
- **Tree-of-thought** — Parallel exploration with best-path selection
- **Self-correction** — Validate output, fix errors automatically
- **Tool-first-then-answer** — Gather data before responding

These ship as rows in `agent_flows` / `agent_nodes` that you can clone and customize.

---

## ☁️ Cloud Roadmap

> **Local-first success unlocks the cloud.**

ToonDB is currently a **local-first, embedded database** — and it's working great! Based on the success of this MVP, I'm exploring a cloud offering:

| Phase | Status | Description |
|-------|--------|-------------|
| **Local MVP** | ✅ Live | Embedded + IPC modes, full ACID, vector search |
| **Cloud (ToonDB Cloud)** | 🚧 On the way | Hosted, managed ToonDB with sync |

**Your feedback shapes the cloud roadmap.** If you're interested in a hosted solution, let us know what you need!

---

## 💬 A Note from the Creator

> **This is an MVP — and your support makes it better.**

ToonDB started as an experiment: *what if databases were designed for LLMs from day one?* The result is what you see here — a working, tested, and (I hope) useful database.

But here's the thing: **software gets better with users.** Every bug report, feature request, and "hey, this broke" message helps ToonDB become more robust. You might find rough edges. You might encounter surprises. That's expected — and fixable!

**What I need from you:**
- 🐛 **Report bugs** — even small ones
- 💡 **Request features** — what's missing for your use case?
- ⭐ **Star the repo** — it helps others discover ToonDB
- 📣 **Share your experience** — blog posts, tweets, anything

Your usage and feedback don't just help me — they help everyone building with ToonDB. Let's make this great together.

> **Note:** ToonDB is a **single-person project** built over weekends and spare time. I'm the sole developer, architect, and maintainer. This means you might find rough edges, incomplete features, or areas that need polish. The good news? Your contributions can make a real impact. More hands on this project means more advanced features, better stability, and faster progress. Every PR, issue report, and suggestion directly shapes what ToonDB becomes.

*— Sushanth*

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
cargo install cargo-watch cargo-criterion

# Run in watch mode
cargo watch -x "test --all"

# Run specific benchmark
cargo criterion --bench vector_search
```

---

## License

Apache-2.0

---

## 🙏 Acknowledgments

- HNSW algorithm: [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- MVCC implementation inspired by PostgreSQL and SQLite
- Columnar storage design influenced by Apache Arrow
- Vamana (DiskANN): Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS 2019
- HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2018
- PGM-Index: Ferragina & Vinciguerra, "The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds", VLDB 2020
- ARIES: Mohan et al., "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging", ACM TODS 1992
- SSI: Cahill et al., "Serializable Isolation for Snapshot Databases", ACM SIGMOD 2008
- LSM-Tree: O'Neil et al., "The Log-Structured Merge-Tree (LSM-Tree)", Acta Informatica 1996
- Toon https://github.com/toon-format/toon

---

**Built with ❤️ for the AI era**

[GitHub](https://github.com/toondb/toondb) • [Documentation](https://docs.toondb.dev)
