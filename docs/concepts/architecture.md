# ToonDB Complete Architecture & API Reference

**Version:** 0.2.7  
**License:** Apache-2.0  
**Repository:** https://github.com/toondb/toondb

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Executive Summary](#executive-summary)
3. [System Architecture](#system-architecture)
4. [Module Structure](#module-structure)
5. [Storage Engine](#storage-engine)
6. [Client SDK API](#client-sdk-api)
7. [TOON Format Internals](#toon-format-internals)
8. [MCP Protocol & Tools API](#mcp-protocol--tools-api)
9. [Vector Search API](#vector-search-api)
10. [Transaction & MVCC API](#transaction--mvcc-api)
11. [Context Query API](#context-query-api)
12. [Query Processing Pipeline](#query-processing-pipeline)
13. [Memory Management](#memory-management)
14. [Concurrency Model](#concurrency-model)
15. [Python SDK Architecture](#python-sdk-architecture)
16. [Performance Characteristics](#performance-characteristics)
17. [Configuration Reference](#configuration-reference)

---

## Design Philosophy

ToonDB is built around four core principles:

### 1. Token Efficiency First
Every design decision prioritizes minimizing tokens when data is consumed by LLMs:
```
Traditional: LLM ← JSON ← SQL Result ← Query Optimizer ← B-Tree
             ~150 tokens for 3 rows

ToonDB:      LLM ← TOON ← Columnar Scan ← Path Resolution
             ~50 tokens for 3 rows (66% reduction)
```

### 2. Path-Based Access
O(|path|) resolution instead of O(log N) tree traversal:
```
Path: "users/42/profile/avatar"

TCH Resolution:
├─ users     → Table lookup (O(1) hash)
│  └─ 42     → Row index (O(1) direct)
│     └─ profile → Column group (O(1))
│        └─ avatar → Column offset (O(1))

Total: O(4) = O(|path|), regardless of table size
```

### 3. Columnar by Default
Read only what you need - 50% I/O reduction for typical queries.

### 4. Embeddable & Extensible
Single-file deployment (~1.5MB) with optional plugin architecture.

---

## Executive Summary

ToonDB is an **AI-native database** designed from the ground up for LLM applications and autonomous agents. Key differentiators:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Trie-Columnar Hybrid (TCH)** | O(\|path\|) lookups via radix-compressed trie | Constant-time path resolution |
| **HNSW/Vamana Vector Indexes** | Scale-aware routing (HNSW &lt;100K, Vamana &gt;1M) | Optimal latency at any scale |
| **TOON Format** | 40-60% token reduction vs JSON | Significant LLM cost savings |
| **MCP Protocol** | Native LLM tool integration | Seamless agent orchestration |
| **Embedded Architecture** | SQLite-like deployment (single binary) | Zero external dependencies |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ToonDB Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │  MCP Server │   │ Agent Ctx   │   │  gRPC API   │  Interfaces  │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘              │
│          │                 │                 │                      │
│   ┌──────┴─────────────────┴─────────────────┴──────┐              │
│   │              Query Processing Engine             │              │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │              │
│   │  │ ToonQL  │  │ Context │  │ Token Budgeting │  │              │
│   │  │ Parser  │  │ Builder │  │    Engine       │  │              │
│   │  └────┬────┘  └────┬────┘  └────────┬────────┘  │              │
│   └───────┼────────────┼────────────────┼───────────┘              │
│           │            │                │                           │
│   ┌───────┴────────────┴────────────────┴───────────┐              │
│   │              Unified Storage Layer               │              │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │              │
│   │  │   TCH    │  │  Vector  │  │    MVCC      │   │              │
│   │  │  Storage │  │  Index   │  │ Transactions │   │              │
│   │  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │              │
│   └───────┼─────────────┼───────────────┼───────────┘              │
│           │             │               │                           │
│   ┌───────┴─────────────┴───────────────┴───────────┐              │
│   │              Durability Layer (WAL)              │              │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │              │
│   │  │   WAL    │  │  Group   │  │   Crash      │   │              │
│   │  │  Writer  │  │  Commit  │  │   Recovery   │   │              │
│   │  └──────────┘  └──────────┘  └──────────────┘   │              │
│   └─────────────────────────────────────────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Sync-First Storage**: Core storage uses synchronous I/O (like SQLite) for predictable latency and simpler embedding
2. **Optional Async**: Tokio runtime only required for server mode (MCP, gRPC)
3. **Lock-Free Reads**: Hazard pointer protection for true lock-free read paths
4. **AI-Native Formatting**: TOON format as first-class output for token efficiency

---

## Module Structure

### Crate Dependency Graph

```
toondb-studio (GUI)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                     Application Layer                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │ toondb-mcp │  │toondb-grpc │  │toondb-wasm │              │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘              │
└────────┼───────────────┼───────────────┼─────────────────────┘
         ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                      Client Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐                    │
│  │  toondb-client  │  │  toondb-python  │                    │
│  └────────┬────────┘  └────────┬────────┘                    │
└───────────┼────────────────────┼─────────────────────────────┘
            ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│                      Query Layer                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │toondb-query│  │toondb-tools│  │  toon-fmt  │              │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘              │
└────────┼───────────────┼───────────────┼─────────────────────┘
         ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Execution Layer                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │toondb-index│  │toondb-vector│ │toondb-kernel│             │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘              │
└────────┼───────────────┼───────────────┼─────────────────────┘
         ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Storage Layer                             │
│               ┌─────────────────────┐                        │
│               │   toondb-storage    │                        │
│               │  (WAL + LSCS + GC)  │                        │
│               └──────────┬──────────┘                        │
└──────────────────────────┼───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Core Layer                               │
│               ┌─────────────────────┐                        │
│               │    toondb-core      │                        │
│               │ (Types, Codec, Trie)│                        │
│               └─────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### Crate Responsibilities

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| `toondb-core` | Foundational types, codecs, trie | `ToonValue`, `ToonSchema`, `ToonCodec` |
| `toondb-storage` | WAL, LSCS, GC, durability | `Database`, `WalManager`, `GarbageCollector` |
| `toondb-kernel` | Query execution, table operations | `Kernel`, `TableHandle`, `ScanIterator` |
| `toondb-index` | B-tree, learned indexes | `BTreeIndex`, `LearnedSparseIndex` |
| `toondb-vector` | HNSW, Vamana, PQ | `HnswIndex`, `VamanaIndex`, `ProductQuantizer` |
| `toondb-query` | ToonQL parser, optimizer | `Parser`, `Planner`, `Optimizer` |
| `toondb-client` | High-level SDK, context queries | `DurableConnection`, `ContextQueryBuilder` |
| `toondb-mcp` | MCP protocol server | `McpServer`, `ToolExecutor` |
| `toondb-python` | Python bindings (FFI) | `PyToonDB`, `PyVectorIndex` |
| `toondb-wasm` | Browser WASM build | `WasmVectorIndex` |

---

## Storage Engine

### Log-Structured Column Store (LSCS)

```
┌────────────────────────────────────────────────────────────┐
│                    Storage Engine                           │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Write-Ahead Log (WAL)                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Active   │  │ Sealed   │  │ Sealed   │  │ Archived │   │
│  │ Segment  │  │ Segment  │  │ Segment  │  │ Segments │   │
│  │ (writes) │  │ (full)   │  │ (full)   │  │ (backup) │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       ▼             ▼             ▼             ▼          │
│  ┌────────────────────────────────────────────────────┐   │
│  │              Group Commit Buffer                    │   │
│  │  Batches transactions for efficient fsync           │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Log-Structured Column Store (LSCS)                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │  MemTable   │   │  MemTable   │   │  MemTable   │      │
│  │  (Active)   │   │ (Immutable) │   │ (Flushing)  │      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         ▼                 ▼                 ▼              │
│  ┌───────────────────────────────────────────────────┐    │
│  │                  Sorted Runs                       │    │
│  │  L0: ████ ████ ████ (recently flushed)            │    │
│  │  L1: ████████████████ (merged)                    │    │
│  │  L2: ████████████████████████████ (compacted)     │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  Background Workers                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Compactor│  │   GC     │  │Checkpoint│                 │
│  │ Thread   │  │  Thread  │  │  Thread  │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Lock-Free MemTable

The MemTable uses hazard pointers for true lock-free reads:

```rust
pub struct LockFreeMemTable {
    data: DashMap<Vec<u8>, LockFreeVersionChain>,
    hazard_domain: HazardDomain,
    size_bytes: AtomicUsize,
}

impl LockFreeMemTable {
    /// Read with zero-copy callback (optimal path)
    pub fn read_with<F, R>(
        &self,
        key: &[u8],
        snapshot_ts: u64,
        txn_id: Option<u64>,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&[u8]) -> R;
    
    /// Write a value (creates uncommitted version)
    pub fn write(&self, key: Vec<u8>, value: Option<Vec<u8>>, txn_id: u64) -> Result<()>;
    
    /// Commit a transaction's writes
    pub fn commit(&self, txn_id: u64, commit_ts: u64, keys: &[Vec<u8>]);
}
```

**Scalability**: Lock-free design achieves 23% better scaling vs RwLock at 8 threads.

### SST File Format

```
┌─────────────────────────────────────────────────────────────┐
│                    SST File Structure                        │
├─────────────────────────────────────────────────────────────┤
│ Data Blocks                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Block 0: [key1:val1][key2:val2]...[keyN:valN][trailer]  │ │
│ │ Block 1: [key1:val1][key2:val2]...[keyN:valN][trailer]  │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Meta Blocks                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Bloom Filter Block                                       │ │
│ │ Column Stats Block                                       │ │
│ │ Compression Dict Block (optional)                        │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Index Block                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [first_key_0, offset_0, size_0]                          │ │
│ │ [first_key_1, offset_1, size_1]                          │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Footer (48 bytes)                                            │
│ ┌────────────────┬────────────────┬────────────────────────┐│
│ │ Meta Index     │ Index Handle   │ Magic + Version        ││
│ │ BlockHandle    │ BlockHandle    │                        ││
│ └────────────────┴────────────────┴────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Compaction Strategy

```
Level 0 (L0): Recent flushes, may overlap
┌────┐ ┌────┐ ┌────┐ ┌────┐
│SST1│ │SST2│ │SST3│ │SST4│  ← 4 files, overlapping key ranges
└────┘ └────┘ └────┘ └────┘
          │
          ▼ Compaction (merge sort)
Level 1 (L1): Non-overlapping, sorted
┌──────────────────────────────────────┐
│              SST (merged)             │
└──────────────────────────────────────┘
          │
          ▼ Size-triggered compaction
Level 2 (L2): 10x larger budget
┌────────────────────────────────────────────────────────────┐
│                      SST files                              │
└────────────────────────────────────────────────────────────┘
```

### WAL Record Format

```
┌────────────────────────────────────────────────────────────────┐
│                       WAL Record Format                         │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐ │
│  │  CRC32   │  Length  │   Type   │  TxnID   │    Data      │ │
│  │ (4 bytes)│ (4 bytes)│ (1 byte) │ (8 bytes)│  (variable)  │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘ │
│                                                                 │
│  Record Types:                                                  │
│  • 0x01: PUT (key, value)                                      │
│  • 0x02: DELETE (key)                                          │
│  • 0x03: BEGIN_TXN (txn_id)                                    │
│  • 0x04: COMMIT_TXN (txn_id, commit_ts)                        │
│  • 0x05: ABORT_TXN (txn_id)                                    │
│  • 0x06: CHECKPOINT (LSN, active_txns)                         │
└────────────────────────────────────────────────────────────────┘
```

### Recovery Process

```rust
fn recover(&self) -> Result<RecoveryStats> {
    // 1. Find latest checkpoint
    let checkpoint = self.find_latest_checkpoint()?;
    
    // 2. Replay WAL from checkpoint
    let mut wal_reader = WalReader::open_from(checkpoint.lsn)?;
    let mut active_txns: HashSet<u64> = checkpoint.active_txns;
    
    while let Some(record) = wal_reader.next()? {
        match record.record_type {
            RecordType::BeginTxn => {
                active_txns.insert(record.txn_id);
            }
            RecordType::Put => {
                if active_txns.contains(&record.txn_id) {
                    self.replay_put(&record)?;
                }
            }
            RecordType::CommitTxn => {
                active_txns.remove(&record.txn_id);
            }
            RecordType::AbortTxn => {
                self.rollback_txn(record.txn_id)?;
                active_txns.remove(&record.txn_id);
            }
            _ => {}
        }
    }
    
    // 3. Abort incomplete transactions
    for txn_id in active_txns {
        self.rollback_txn(txn_id)?;
    }
    
    Ok(stats)
}
```

---

## Client SDK API

### ToonClient (In-Memory)

For testing and development without durability requirements:

```rust
use toondb::prelude::*;

// Open database
let client = ToonClient::open("./mydb")?;

// Configure token budget for LLM responses
let client = client.with_token_budget(4096);

// Path-based queries (O(|path|) resolution)
let result = client.query("/users/123").execute()?;

// Execute ToonQL
let rows = client.execute("SELECT * FROM users WHERE active = true")?;

// Begin transaction
let txn = client.begin()?;

// Vector operations
let vectors = client.vectors("embeddings")?;
vectors.add(&["doc1", "doc2"], &[vec1, vec2])?;
let results = vectors.search(&query_embedding, 10)?;
```

### DurableToonClient (Production)

Full WAL/MVCC support for production workloads:

```rust
use toondb::prelude::*;

// Open with durability
let client = DurableToonClient::open("./mydb")?;

// Path-based CRUD
client.put("/users/123", b"{\"name\": \"Alice\"}")?;
let data = client.get("/users/123")?;
client.delete("/users/123")?;

// Scan with prefix
let results = client.scan("/users/")?;

// Transaction support
client.begin()?;
client.put("/users/1", value1)?;
client.put("/users/2", value2)?;
let commit_ts = client.commit()?;

// Force durability
client.fsync()?;
```

### PathQuery Builder

Leverages TCH's O(|path|) resolution:

```rust
use toondb::path_query::{PathQuery, CompareOp};

// Fluent query builder
let results = client.query("/users")
    .filter("score", CompareOp::Gt, ToonValue::Int(80))
    .filter("active", CompareOp::Eq, ToonValue::Bool(true))
    .select(&["name", "email", "score"])
    .order_by("score", SortDirection::Desc)
    .limit(10)
    .execute()?;
```

### Comparison Operators

```rust
pub enum CompareOp {
    Eq,        // =
    Ne,        // !=
    Lt,        // <
    Le,        // <=
    Gt,        // >
    Ge,        // >=
    Like,      // LIKE pattern matching
    In,        // IN (array)
    IsNull,    // IS NULL
    IsNotNull, // IS NOT NULL
}
```

### Output Formats

```rust
pub enum OutputFormat {
    Toon,      // Default: 40-60% fewer tokens than JSON
    Json,      // Standard JSON for compatibility
    Columnar,  // Raw columnar for analytics
}
```

---

## TOON Format Internals

### Text Format Grammar

```ebnf
document     ::= table_header newline row*
table_header ::= name "[" count "]" "{" fields "}" ":"
row          ::= value ("," value)* newline
value        ::= null | bool | number | string | array | ref

null         ::= "∅"
bool         ::= "T" | "F"
number       ::= integer | float
string       ::= raw_string | quoted_string
array        ::= "[" value ("," value)* "]"
ref          ::= "ref(" identifier "," integer ")"
```

### Binary Format Structure

```
┌──────────────────────────────────────────────────────────────┐
│                    TOON Binary Format                         │
├──────────────────────────────────────────────────────────────┤
│ Header (16 bytes)                                            │
│ ┌──────────┬──────────┬──────────┬──────────┐               │
│ │  Magic   │ Version  │  Flags   │ Row Count│               │
│ │ (4 bytes)│ (2 bytes)│ (2 bytes)│ (8 bytes)│               │
│ └──────────┴──────────┴──────────┴──────────┘               │
├──────────────────────────────────────────────────────────────┤
│ Schema Section                                               │
│ ┌──────────┬──────────────────────────────────┐             │
│ │ Name Len │ Table Name (UTF-8)               │             │
│ ├──────────┼──────────────────────────────────┤             │
│ │ Col Count│ [Column Definitions...]           │             │
│ └──────────┴──────────────────────────────────┘             │
├──────────────────────────────────────────────────────────────┤
│ Data Section (columnar)                                      │
│ ┌──────────────────────────────────────────────┐            │
│ │ Column 0: [type_tag][values...]              │            │
│ │ Column 1: [type_tag][values...]              │            │
│ └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### Type Tags

```rust
#[repr(u8)]
pub enum ToonTypeTag {
    Null      = 0x00,
    False     = 0x01,
    True      = 0x02,
    PosFixint = 0x10,  // 0-15 in lower nibble
    NegFixint = 0x20,  // -16 to -1 in lower nibble
    Int8      = 0x30,
    Int16     = 0x31,
    Int32     = 0x32,
    Int64     = 0x33,
    Float32   = 0x40,
    Float64   = 0x41,
    FixStr    = 0x50,  // 0-15 char length in lower nibble
    Str8      = 0x60,
    Str16     = 0x61,
    Str32     = 0x62,
    Array     = 0x70,
    Ref       = 0x80,
}
```

### Varint Encoding

```rust
fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn decode_varint(buf: &[u8]) -> (u64, usize) {
    let mut result = 0u64;
    let mut shift = 0;
    for (i, &byte) in buf.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
        shift += 7;
    }
    (result, buf.len())
}
```

---

## MCP Protocol & Tools API

### Server Lifecycle

```
Client                           Server
  │                                │
  │──── initialize ──────────────►│
  │                                │ Create ToonDB connection
  │◄─── capabilities ─────────────│
  │                                │
  │──── initialized ─────────────►│
  │                                │
  │──── tools/list ──────────────►│
  │                                │ Return 15 built-in tools
  │◄─── tool definitions ─────────│
  │                                │
  │──── tools/call ──────────────►│
  │     { "name": "toondb_query", │
  │       "arguments": {...} }    │
  │                                │ Execute query
  │◄─── result (TOON format) ─────│
```

### MCP Server Implementation

```rust
impl McpServer {
    pub fn new(conn: Arc<EmbeddedConnection>) -> Self;
    
    /// Dispatch JSON-RPC request
    pub fn dispatch(&self, req: &RpcRequest) -> RpcResponse;
    
    /// Get database statistics
    pub fn db_stats(&self) -> DatabaseStats;
}
```

### Built-in Tools Reference

#### Core Database Tools

| Tool | Description | Required Args |
|------|-------------|---------------|
| `toondb_context_query` | AI-optimized context with token budgeting | `sections` |
| `toondb_query` | Execute ToonQL query | `query` |
| `toondb_get` | Get value at path | `path` |
| `toondb_put` | Set value at path | `path`, `value` |
| `toondb_delete` | Delete at path | `path` |
| `toondb_list_tables` | List tables with metadata | - |
| `toondb_describe` | Get table schema | `table` |

#### Memory Tools (Episode/Entity Schema)

| Tool | Description | Required Args |
|------|-------------|---------------|
| `memory_search_episodes` | Semantic episode search | `query` |
| `memory_get_episode_timeline` | Event timeline for episode | `episode_id` |
| `memory_search_entities` | Entity search | `query` |
| `memory_get_entity_facts` | Entity details | `entity_id` |
| `memory_build_context` | One-shot context packing | `goal`, `token_budget` |

#### Log Tools

| Tool | Description | Required Args |
|------|-------------|---------------|
| `logs_tail` | Get last N rows | `table` |
| `logs_timeline` | Time-range query | `table`, `start`, `end` |

### Tool Schemas

#### toondb_context_query

```json
{
  "name": "toondb_context_query",
  "description": "Fetch AI-optimized context from ToonDB with token budgeting",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sections": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "kind": { "type": "string", "enum": ["literal", "get", "last", "search"] },
            "text": { "type": "string", "description": "For kind=literal" },
            "path": { "type": "string", "description": "For kind=get" },
            "table": { "type": "string", "description": "For kind=last/search" },
            "query": { "type": "string", "description": "For kind=search" },
            "top_k": { "type": "integer", "default": 10 }
          },
          "required": ["name", "kind"]
        }
      },
      "token_budget": { "type": "integer", "default": 4096 },
      "format": { "type": "string", "enum": ["toon", "json", "markdown"], "default": "toon" },
      "truncation": { "type": "string", "enum": ["tail_drop", "head_drop", "proportional"], "default": "tail_drop" }
    },
    "required": ["sections"]
  }
}
```

#### toondb_query

```json
{
  "name": "toondb_query",
  "description": "Execute a ToonQL query. Returns results in TOON format.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "ToonQL query" },
      "format": { "type": "string", "enum": ["toon", "json"], "default": "toon" },
      "limit": { "type": "integer", "default": 100 }
    },
    "required": ["query"]
  }
}
```

#### memory_search_episodes

```json
{
  "name": "memory_search_episodes",
  "description": "Search for similar past episodes by semantic similarity",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Natural language query" },
      "k": { "type": "integer", "default": 5 },
      "episode_type": { 
        "type": "string", 
        "enum": ["conversation", "task", "workflow", "debug", "agent_interaction"] 
      },
      "entity_id": { "type": "string", "description": "Filter by entity" }
    },
    "required": ["query"]
  }
}
```

#### memory_build_context

```json
{
  "name": "memory_build_context",
  "description": "Build optimized LLM context from memory automatically",
  "inputSchema": {
    "type": "object",
    "properties": {
      "goal": { "type": "string", "description": "What the context will be used for" },
      "token_budget": { "type": "integer", "default": 4096 },
      "session_id": { "type": "string" },
      "episode_id": { "type": "string" },
      "entity_ids": { "type": "array", "items": { "type": "string" } },
      "include_schema": { "type": "boolean", "default": false }
    },
    "required": ["goal", "token_budget"]
  }
}
```

---

## Vector Search API

### Scale-Aware Router

```
┌────────────────────────────────────────────────────────────┐
│                   Scale-Aware Routing                       │
│                                                             │
│   Vectors < 100K ─────► HNSW (in-memory, low latency)      │
│   Vectors 100K-1M ────► Hybrid (HNSW + Vamana migration)   │
│   Vectors > 1M ───────► Vamana + PQ (disk-based, scalable) │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### VectorCollection API

```rust
pub struct VectorCollection {
    dimension: usize,
    backend: VectorBackend,
    pq: Option<ProductQuantizer>,
    id_map: RwLock<HashMap<String, usize>>,
    reverse_map: RwLock<HashMap<usize, String>>,
}

impl VectorCollection {
    /// Open or create a vector collection
    pub fn open(conn: &ToonConnection, name: &str) -> Result<Self>;
    
    /// Add vectors in batch
    pub fn add(&mut self, ids: &[&str], vectors: &[Vec<f32>]) -> Result<()>;
    
    /// Add a single vector
    pub fn add_one(&mut self, id: &str, vector: Vec<f32>) -> Result<()>;
    
    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    
    /// Get vector by ID
    pub fn get(&self, id: &str) -> Option<Vec<f32>>;
    
    /// Delete vector by ID
    pub fn delete(&mut self, id: &str) -> Result<bool>;
    
    /// Get collection statistics
    pub fn stats(&self) -> VectorStats;
    
    /// Get compression ratio (if PQ trained)
    pub fn compression_ratio(&self) -> Option<f32>;
    
    /// Migrate batch during idle time (for hybrid mode)
    pub fn migrate_batch(&mut self) -> Result<usize>;
}

pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<Value>,
}

pub struct VectorStats {
    pub count: usize,
    pub dimension: usize,
    pub backend: String,
    pub memory_bytes: usize,
    pub pq_enabled: bool,
    pub migration_progress: Option<f32>,
}
```

### Index Parameters

#### HNSW (Small Collections &lt;100K)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max connections per node |
| `M_max0` | 32 | Max connections at layer 0 |
| `ef_construction` | 200 | Build-time search width |
| `ef_search` | 100 | Query-time search width |
| **Memory** | ~1.5KB/vec | Higher memory, lowest latency |
| **Latency** | &lt;1ms | Sub-millisecond queries |

#### Vamana (Large Collections >1M)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `R` | 64 | Graph out-degree |
| `L` | 100 | Search list size |
| `α` | 1.2 | Pruning parameter |
| **Memory** | ~0.5KB/vec + mmap | Disk-friendly |
| **Latency** | 2-5ms | Good for large scale |

#### Product Quantizer (Compression)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 8 | Number of subquantizers |
| `Ksub` | 256 | Centroids per subquantizer |
| `nbits` | 8 | Bits per code |
| **Memory** | 32B/vec | 96% compression for 768d |

### HNSW Graph Structure

```
┌────────────────────────────────────────────────────────────────┐
│                    HNSW Graph Structure                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 2:       ●───────────────────────●                      │
│                 │                       │                       │
│  Layer 1:       ●───●───────●───────────●───────●              │
│                 │   │       │           │       │               │
│  Layer 0:   ●───●───●───●───●───●───●───●───●───●───●          │
│            v0  v1  v2  v3  v4  v5  v6  v7  v8  v9  v10         │
│                                                                 │
│  Search: Start at top layer, greedily descend                  │
│  Insert: Random level, connect at each layer                    │
└────────────────────────────────────────────────────────────────┘
```

### Level Generation Algorithm

```rust
/// Per HNSW paper: level = floor(-ln(uniform(0,1)) * mL)
fn random_level(&self) -> usize {
    let uniform: f32 = rand::random();
    let level = (-uniform.ln() * self.level_multiplier).floor() as usize;
    level.min(16) // Cap at 16 layers
}

// Distribution for M=16, mL≈0.36:
// P(level=0) ≈ 70%
// P(level=1) ≈ 21%
// P(level=2) ≈ 6%
// P(level≥3) ≈ 3%
```

### Search Algorithm

```rust
fn search_internal(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let entry_point = self.entry_point?;
    
    // Navigate from top layer to layer 1
    let mut current = entry_point;
    for layer in (1..=self.max_layer).rev() {
        current = self.search_layer_single(query, current, layer);
    }
    
    // Search layer 0 with ef_search candidates
    let candidates = self.search_layer(query, current, self.ef_search.max(k), 0);
    
    // Return top-k results
    candidates.into_iter()
        .take(k)
        .map(|(dist, idx)| (self.nodes[idx].id, dist))
        .collect()
}
```

### WASM Vector Index (Browser)

```javascript
import init, { WasmVectorIndex } from 'toondb-wasm';

async function main() {
  await init();
  
  // Create index: dimension=768, M=16, ef_construction=100
  const index = new WasmVectorIndex(768, 16, 100);
  
  // Insert vectors
  const ids = BigUint64Array.from([1n, 2n, 3n]);
  const vectors = new Float32Array(768 * 3);
  const inserted = index.insertBatch(ids, vectors);
  
  // Search
  const query = new Float32Array(768);
  const results = index.search(query, 10);
}
```

---

## Transaction & MVCC API

### Transaction Lifecycle

```rust
// Begin transaction
client.begin()?;

// Operations within transaction
client.put("/users/1", value1)?;
client.put("/users/2", value2)?;
let data = client.get("/users/1")?;

// Commit or abort
let commit_ts = client.commit()?;  // Returns commit timestamp
// OR
client.abort()?;  // Discard all changes
```

### Isolation Levels

```rust
pub enum IsolationLevel {
    ReadCommitted,    // See committed data at statement start
    RepeatableRead,   // See committed data at transaction start
    Serializable,     // SSI - Full serializability
}

// Begin with specific isolation
let txn = client.begin_with_isolation(IsolationLevel::Serializable)?;
```

### MVCC Internals

Version chain structure:

```rust
pub struct LockFreeVersion {
    storage: VersionStorage,      // Inline (&lt;48B) or heap
    txn_id: AtomicU64,            // Writing transaction
    commit_ts: AtomicU64,         // Commit timestamp (0 if uncommitted)
    next: AtomicPtr<LockFreeVersion>, // Next older version
}
```

Visibility rules:
- **Own writes**: Visible if `txn_id` matches current transaction
- **Committed writes**: Visible if `commit_ts <= snapshot_ts`
- **Conflict detection**: SSI validation checks for rw-antidependency cycles

### MVCC Version Chain

```rust
struct VersionedValue {
    value: Option<Vec<u8>>,  // None = tombstone
    txn_id: u64,             // Transaction that wrote this
    timestamp: u64,          // Commit timestamp
    next: Option<Box<VersionedValue>>,  // Older versions
}

impl MVCCStore {
    fn get(&self, key: &Key, snapshot_ts: u64) -> Option<&[u8]> {
        let mut version = self.current.get(key)?;
        
        // Find visible version
        while version.timestamp > snapshot_ts {
            version = version.next.as_ref()?;
        }
        
        version.value.as_deref()
    }
}
```

### Group Commit Optimization

```rust
/// Optimal batch size: N* = √(2 × L_fsync × λ / C_wait)
struct GroupCommitBuffer {
    pending: VecDeque<PendingCommit>,
    config: GroupCommitConfig,
}

impl GroupCommitBuffer {
    fn optimal_batch_size(&self, arrival_rate: f64, wait_cost: f64) -> usize {
        let l_fsync = self.config.fsync_latency_us as f64 / 1_000_000.0;
        let n_star = (2.0 * l_fsync * arrival_rate / wait_cost).sqrt();
        (n_star as usize).clamp(1, self.config.max_batch_size)
    }
}
```

---

## Context Query API

### ContextQueryBuilder

Build AI-optimized context with automatic token budgeting:

```rust
pub struct ContextQueryBuilder {
    sections: Vec<ContextSection>,
    token_budget: usize,
    format: ContextFormat,
    truncation: TruncationStrategy,
}

pub enum SectionKind {
    Literal { text: String },
    Get { path: String },
    Last { table: String, top_k: usize, filter: Option<Filter> },
    Search { query: String, collection: String, top_k: usize },
    ToolRegistry { include_schema: bool },
}

pub enum TruncationStrategy {
    TailDrop,      // Drop lowest priority sections first
    HeadDrop,      // Drop highest priority sections first
    Proportional,  // Reduce all sections proportionally
}
```

### Usage Example

```rust
let context = ContextQueryBuilder::new()
    .section("system", SectionKind::Literal { 
        text: "You are a helpful assistant.".to_string() 
    })
    .section("history", SectionKind::Last { 
        table: "messages".to_string(), 
        top_k: 10,
        filter: None,
    })
    .section("knowledge", SectionKind::Search { 
        query: user_message.clone(),
        collection: "docs".to_string(),
        top_k: 5,
    })
    .with_budget(4096)
    .with_format(ContextFormat::Toon)
    .with_truncation(TruncationStrategy::TailDrop)
    .execute()?;
```

### TOON Format Token Savings

```
JSON (52 chars ≈ 13 tokens):
{"user":{"name":"Alice","age":30,"active":true}}

TOON (47 chars ≈ 12 tokens, 8% savings):
user.name="Alice" user.age=30 user.active=true

TOON Table Format (40%+ savings for arrays):
[users]
name    | age | active
"Alice" | 30  | true
"Bob"   | 25  | false
```

---

## Performance Characteristics

### Complexity Analysis

| Operation | ToonDB | B-Tree (SQLite) | Notes |
|-----------|--------|-----------------|-------|
| Point Read | O(\|path\|) | O(log N) | TCH path-based |
| Point Write | O(\|path\|) | O(log N) | + WAL |
| Range Scan | O(\|path\| + K) | O(log N + K) | K = result count |
| Vector Search (HNSW) | O(log N) | N/A | ef-dependent |
| Vector Search (Vamana) | O(log N) | N/A | + PQ overhead |
| Full Scan | O(N) | O(N) | Columnar advantage |

### Memory Budget

| Component | Default | Configuration |
|-----------|---------|---------------|
| MemTable | 64MB | `memtable_size` |
| Block Cache | 128MB | `block_cache_size` |
| HNSW Index | ~1.5KB/vec | M, ef_construction |
| Vamana Index | ~0.5KB/vec | R, L, α |
| PQ Codes | 32B/vec | M, Ksub |
| WAL Buffer | 16MB | `wal_buffer_size` |

### Latency Targets

| Operation | Target | P99 |
|-----------|--------|-----|
| Point Read (cached) | &lt;100μs | &lt;500μs |
| Point Read (disk) | &lt;1ms | &lt;5ms |
| Point Write | &lt;100μs | &lt;1ms |
| Transaction Commit | &lt;1ms | &lt;5ms |
| Vector Search (10K) | &lt;1ms | &lt;5ms |
| Vector Search (1M) | &lt;10ms | &lt;50ms |
| Context Query | &lt;50ms | &lt;200ms |

---

## Configuration Reference

### DatabaseConfig

```rust
pub struct DatabaseConfig {
    // WAL settings
    pub wal_segment_size: usize,      // Default: 64MB
    pub wal_sync_mode: SyncMode,      // Fsync, FsyncDelayed, None
    pub group_commit: bool,           // Batch commits for throughput
    pub group_commit_delay_us: u64,   // Max wait time
    
    // LSCS settings
    pub memtable_size: usize,         // Default: 64MB
    pub level_ratio: usize,           // Default: 10
    pub max_levels: usize,            // Default: 7
    
    // GC settings
    pub gc_interval_secs: u64,        // Default: 60
    pub min_versions_to_keep: usize,  // Default: 2
}
```

### ClientConfig

```rust
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
```

### SyncMode Options

```rust
pub enum SyncMode {
    Fsync,        // fsync after every commit (safest)
    FsyncDelayed, // fsync after group_commit_delay_us
    None,         // No fsync (fastest, risk of data loss)
}
```

---

## Workspace Dependencies

Key dependencies from `Cargo.toml`:

```toml
[workspace.dependencies]
# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Compression
lz4 = "1.24"
zstd = "0.13"

# Data structures
crossbeam-skiplist = "0.1"
parking_lot = "0.12"
dashmap = "5.5"

# Hashing
blake3 = "1.5"
twox-hash = "1.6"

# Vectors
ndarray = "0.15"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Async (optional)
tokio = { version = "1.35", optional = true }
```

---

## Query Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      Query Pipeline                               │
├──────────────────────────────────────────────────────────────────┤
│  1. PARSE                                                         │
│     Input: conn.query("users").where_eq("status", "active")      │
│     Output: QueryAST { table, predicates, projections }          │
│                                                                   │
│  2. PLAN                                                          │
│     • Choose access method (scan vs index)                        │
│     • Push down predicates                                        │
│     Output: LogicalPlan                                           │
│                                                                   │
│  3. OPTIMIZE                                                      │
│     • Cost-based index selection                                  │
│     • Predicate ordering by selectivity                          │
│     Output: PhysicalPlan                                          │
│                                                                   │
│  4. EXECUTE                                                       │
│     • Open column readers                                         │
│     • Apply predicates (vectorized)                               │
│     Output: QueryResult                                           │
│                                                                   │
│  5. FORMAT                                                        │
│     • TOON: users[N]{cols}:row1;row2;...                         │
│     • JSON: [{"col": val}, ...]                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Predicate Pushdown

```rust
fn push_predicates(plan: &mut ScanPlan, predicates: &[Predicate]) {
    for pred in predicates {
        match pred {
            // Push to index lookup
            Predicate::Eq(col, val) if is_indexed(col) => {
                plan.index_lookup = Some(IndexLookup { column: col, value: val });
            }
            // Push to block-level filtering
            Predicate::Range(col, min, max) => {
                plan.block_filters.push(BlockFilter { column: col, min, max });
            }
            // Late filter (after scan)
            _ => plan.late_filters.push(pred.clone()),
        }
    }
}
```

---

## Memory Management

### Buddy Allocator

```
┌────────────────────────────────────────────────────────────────┐
│                    Buddy Allocator                              │
├────────────────────────────────────────────────────────────────┤
│  Order 10 (1KB):  [████████████████████████████████]           │
│                            │                                    │
│                   ┌────────┴────────┐                          │
│  Order 9 (512B):  [████████████████] [________________]         │
│                         │                                       │
│                   ┌─────┴─────┐                                 │
│  Order 8 (256B):  [████████] [████]  ...                       │
│                                                                 │
│  Allocation: Find smallest power-of-2 block, split if needed   │
│  Deallocation: Coalesce with buddy if both free                │
└────────────────────────────────────────────────────────────────┘
```

### Arena Allocator

```rust
struct BuddyArena {
    buddy: BuddyAllocator,
    current_block: Mutex<Option<ArenaBlock>>,
    block_size: usize,
}

impl BuddyArena {
    fn allocate(&self, size: usize, align: usize) -> Result<usize> {
        let mut current = self.current_block.lock();
        
        // Try current block first
        if let Some(ref mut block) = *current {
            let aligned = (block.offset + align - 1) & !(align - 1);
            if aligned + size <= block.size {
                block.offset = aligned + size;
                return Ok(block.base + aligned);
            }
        }
        
        // Allocate new block from buddy allocator
        let new_size = size.max(self.block_size).next_power_of_two();
        let base = self.buddy.allocate(new_size)?;
        *current = Some(ArenaBlock { base, offset: size, size: new_size });
        Ok(base)
    }
    
    fn reset(&self) {
        // Free all blocks at once - O(1) reset
        self.current_block.lock().take();
    }
}
```

---

## Concurrency Model

### Lock Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Lock Acquisition Order                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Catalog Lock (RwLock) - Table schema changes                │
│  2. Table Lock (per-table RwLock) - DDL operations              │
│  3. Transaction Manager Lock (Mutex) - Begin/commit/abort       │
│  4. WAL Lock (Mutex) - Append to write-ahead log                │
│  5. Memtable Lock (RwLock) - In-memory writes                   │
│  6. Index Lock (per-index RwLock) - Index modifications         │
│                                                                  │
│  ALWAYS acquire in this order to prevent deadlocks              │
└─────────────────────────────────────────────────────────────────┘
```

### Lock-Free Reads via MVCC

```rust
impl Database {
    fn read(&self, key: &[u8], snapshot: Snapshot) -> Option<Value> {
        // No locks needed - snapshot isolation
        let version = self.mvcc.get_visible(key, snapshot.timestamp);
        version.map(|v| v.value.clone())
    }
}

// Snapshot is just a timestamp
struct Snapshot {
    timestamp: u64,
    txn_id: u64,
}
```

---

## Python SDK Architecture

### Access Modes

```
┌──────────────────────────────────────────────────────────────────┐
│                     Python Application                            │
├──────────────────────────────────────────────────────────────────┤
│                        toondb (PyPI)                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────────────┐ │
│  │  Embedded  │  │    IPC     │  │        Bulk API             │ │
│  │    FFI     │  │   Client   │  │  (subprocess → toondb-bulk) │ │
│  └─────┬──────┘  └─────┬──────┘  └──────────────┬──────────────┘ │
└────────┼───────────────┼─────────────────────────┼────────────────┘
         │               │                         │
    ┌────▼────┐    ┌─────▼─────┐           ┌───────▼───────┐
    │  Rust   │    │    IPC    │           │  toondb-bulk  │
    │  FFI    │    │  Server   │           │    binary     │
    │  (.so)  │    │           │           │               │
    └─────────┘    └───────────┘           └───────────────┘
```

### Distribution Model (uv-style wheels)

Wheels contain pre-built Rust binaries - no compilation required:

```
toondb-0.2.3-py3-none-manylinux_2_17_x86_64.whl
├── toondb/
│   ├── __init__.py
│   ├── database.py      # Embedded FFI
│   ├── ipc.py           # IPC client
│   ├── bulk.py          # Bulk operations
│   └── _bin/
│       └── linux-x86_64/
│           └── toondb-bulk  # Pre-built binary
```

**Platform matrix:**
- `manylinux_2_17_x86_64` - Linux glibc ≥ 2.17
- `manylinux_2_17_aarch64` - Linux ARM64
- `macosx_11_0_universal2` - macOS Intel + Apple Silicon
- `win_amd64` - Windows x64

### Bulk API FFI Bypass

For vector-heavy workloads, the Bulk API avoids FFI overhead:

```
Python FFI path (130 vec/s):
┌─────────┐    memcpy    ┌──────┐
│ numpy   │ ────────────→│ Rust │ → repeated N times
└─────────┘   per batch  └──────┘

Bulk API path (1,600 vec/s):
┌─────────┐   mmap    ┌──────────────┐   fork    ┌──────────────┐
│ numpy   │ ────────→ │  temp file   │ ────────→ │ toondb-bulk  │
└─────────┘  1 write  └──────────────┘  1 proc   └───────────────┘
```

**Result**: 12× throughput improvement for bulk vector operations.

---

## Summary

ToonDB's architecture delivers:

1. **AI-Native Design**: 
   - TOON format (40-60% token savings)
   - Context queries with automatic token budgeting
   - MCP integration for seamless LLM tool use

2. **Embedded Simplicity**: 
   - SQLite-like deployment (~1.5MB binary)
   - Sync-first storage for predictable latency
   - Zero external dependencies

3. **Scale-Aware Adaptation**: 
   - HNSW for small collections (&lt;100K vectors, &lt;1ms latency)
   - Vamana+PQ for large collections (>1M vectors, 32B/vector)
   - Automatic migration between backends

4. **Production-Ready**: 
   - WAL durability with group commit
   - MVCC transactions with SSI isolation
   - Crash recovery with checkpoint/replay

5. **High Performance**:
   - O(|path|) lookups via TCH (Trie-Columnar Hybrid)
   - Lock-free reads via hazard pointers
   - 12× Python throughput via Bulk API

This enables ToonDB to replace the traditional AI stack (PostgreSQL + Pinecone + Redis + custom RAG) with a single, optimized system.
