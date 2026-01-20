# SochDB API Reference

Complete API documentation for SochDB developers.

---

## Table of Contents

1. [Core Types](#core-types)
2. [Client SDK](#client-sdk)
3. [Query API](#query-api)
4. [Vector Operations](#vector-operations)
5. [Storage Layer](#storage-layer)
6. [Kernel API](#kernel-api)
7. [Plugin System](#plugin-system)

---

## Core Types

### SochValue

The universal value type for SochDB data.

```rust
pub enum SochValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    Binary(Vec<u8>),
    Array(Vec<SochValue>),
    Object(HashMap<String, SochValue>),
    Ref { table: String, id: u64 },
}
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_null()` | `fn is_null(&self) -> bool` | Check if value is null |
| `as_int()` | `fn as_int(&self) -> Option<i64>` | Extract integer |
| `as_text()` | `fn as_text(&self) -> Option<&str>` | Extract string reference |
| `to_toon()` | `fn to_toon(&self) -> String` | Convert to TOON format |
| `type_tag()` | `fn type_tag(&self) -> SochTypeTag` | Get binary type tag |

#### Conversions

```rust
// From Rust types
let v = SochValue::from(42i64);
let v = SochValue::from("hello");
let v = SochValue::from(vec![1, 2, 3]);

// To Rust types
let i: i64 = v.try_into()?;
let s: String = v.try_into()?;
```

---

### SochType

Schema type definitions.

```rust
pub enum SochType {
    Int,
    UInt,
    Float,
    Text,
    Bool,
    Bytes,
    Vector(usize),                    // Dimensionality
    Array(Box<SochType>),
    Optional(Box<SochType>),
    Ref(String),                      // Referenced table
    Object(Vec<(String, SochType)>),
}
```

#### Type Parsing

```rust
// Parse from string
SochType::parse("int")          // Some(SochType::Int)
SochType::parse("text?")        // Some(SochType::Optional(Box::new(SochType::Text)))
SochType::parse("ref(users)")   // Some(SochType::Ref("users".into()))
SochType::parse("vec(384)")     // Some(SochType::Vector(384))
```

---

### SochSchema

Table schema definition.

```rust
pub struct SochSchema {
    pub name: String,
    pub fields: Vec<SochField>,
    pub primary_key: Option<String>,
    pub indexes: Vec<IndexDef>,
}

pub struct SochField {
    pub name: String,
    pub field_type: SochType,
    pub nullable: bool,
    pub default: Option<SochValue>,
}
```

#### Builder Pattern

```rust
let schema = SochSchema::new("users")
    .field("id", SochType::UInt)
    .field("name", SochType::Text)
    .field("email", SochType::Text)
    .field("age", SochType::Optional(Box::new(SochType::Int)))
    .field("embedding", SochType::Vector(384))
    .primary_key("id")
    .index("email", IndexType::BTree)
    .index("embedding", IndexType::HNSW);
```

---

### SochTable

In-memory table representation.

```rust
pub struct SochTable {
    pub schema: SochSchema,
    pub rows: Vec<SochRow>,
}

pub struct SochRow {
    pub values: Vec<SochValue>,
}
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `new(schema)` | `fn new(schema: SochSchema) -> Self` | Create empty table |
| `push(row)` | `fn push(&mut self, row: SochRow)` | Add a row |
| `len()` | `fn len(&self) -> usize` | Number of rows |
| `format()` | `fn format(&self) -> String` | TOON string format |
| `parse(s)` | `fn parse(s: &str) -> Result<Self>` | Parse from TOON |

---

## Client SDK

### SochConnection

Primary interface for database operations.

```rust
pub struct SochConnection {
    // Internal fields
}
```

#### Opening Connections

```rust
// Basic open
let conn = SochConnection::open("./database")?;

// With configuration
let conn = SochConnection::open_with_config(
    "./database",
    ConnectionConfig {
        read_only: false,
        cache_size: 64 * 1024 * 1024,  // 64MB
        ..Default::default()
    }
)?;
```

#### Table Operations

```rust
// Create table
conn.create_table(schema)?;

// Drop table
conn.drop_table("table_name")?;

// List tables
let tables: Vec<String> = conn.list_tables()?;
```

#### CRUD Operations

```rust
// Insert (returns row ID)
let id = conn.insert("users", &values)?;

// Select
let results = conn.query("users")
    .columns(&["id", "name"])
    .where_eq("status", "active")
    .limit(10)
    .execute()?;

// Update
let updated = conn.update("users")
    .set("status", SochValue::Text("inactive".into()))
    .where_eq("id", 42)
    .execute()?;

// Delete
let deleted = conn.delete_where("users")
    .where_gt("age", 100)
    .execute()?;
```

#### Path API

```rust
// Put raw bytes
conn.put("users/1/profile", &serialized_data)?;

// Get raw bytes
let data: Option<Vec<u8>> = conn.get("users/1/profile")?;

// Delete
conn.delete("users/1/profile")?;

// Scan prefix
let entries = conn.scan("users/1/")?;
for (path, value) in entries {
    println!("{}: {} bytes", path, value.len());
}
```

#### Transactions

```rust
// Begin explicit transaction
let txn = conn.begin()?;

// Operations within transaction
conn.put_in_txn(txn, "key1", b"value1")?;
conn.put_in_txn(txn, "key2", b"value2")?;

// Commit (or rollback on error)
match validate_state(&conn, txn) {
    Ok(_) => conn.commit(txn)?,
    Err(e) => {
        conn.rollback(txn)?;
        return Err(e);
    }
}
```

---

### SchemaBuilder

Fluent schema construction.

```rust
let schema = SchemaBuilder::table("orders")
    .field("id", SochType::UInt)
    .field("user_id", SochType::Ref("users".into()))
    .field("amount", SochType::Float)
    .field("status", SochType::Text)
    .field("created_at", SochType::UInt)
    .primary_key("id")
    .foreign_key("user_id", "users", "id")
    .index("created_at", IndexType::BTree)
    .constraint("amount > 0")
    .build();
```

---

### BatchWriter

High-throughput batch operations.

```rust
pub struct BatchWriter<'a> {
    // ...
}
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `new(conn)` | Create batch writer | `BatchWriter` |
| `max_batch_size(n)` | Set max operations per batch | `Self` |
| `auto_commit(bool)` | Enable auto-flush when full | `Self` |
| `insert(table, values)` | Add insert operation | `Self` |
| `update(table, key, updates)` | Add update operation | `Self` |
| `delete(table, key)` | Add delete operation | `Self` |
| `pending_count()` | Get pending operation count | `usize` |
| `execute()` | Execute all operations | `Result<BatchResult>` |

#### Example

```rust
let result = conn.batch()
    .max_batch_size(1000)
    .auto_commit(true)
    .insert("events", vec![("id", 1.into()), ("data", "event1".into())])
    .insert("events", vec![("id", 2.into()), ("data", "event2".into())])
    .insert("events", vec![("id", 3.into()), ("data", "event3".into())])
    .execute()?;

println!("Executed: {}, Fsyncs: {}", result.ops_executed, result.fsync_count);
```

---

### GroupCommitBuffer

Advanced durability control with adaptive batching.

```rust
let config = GroupCommitConfig {
    max_wait_ms: 10,
    max_batch_size: 1000,
    target_batch_size: 100,
    fsync_latency_us: 5000,
};

let buffer = GroupCommitBuffer::new(config);

// Calculate optimal batch size
// N* = √(2 × L_fsync × λ / C_wait)
let optimal = buffer.optimal_batch_size(
    10_000.0,  // λ: 10k ops/sec arrival rate
    1.0        // C_wait: cost per unit wait time
);
```

---

## Query API

### QueryBuilder

Fluent query construction.

```rust
pub struct QueryBuilder<'a> {
    // ...
}
```

#### Methods

```rust
conn.query("users")
    // Column selection
    .columns(&["id", "name", "email"])
    .all_columns()
    
    // Filtering
    .where_eq("status", "active")
    .where_ne("role", "admin")
    .where_gt("age", 18)
    .where_gte("score", 80)
    .where_lt("balance", 0)
    .where_lte("attempts", 3)
    .where_like("name", "A%")
    .where_in("country", &["US", "UK", "CA"])
    .where_between("created", start, end)
    .where_null("deleted_at")
    .where_not_null("verified_at")
    
    // Logical operators
    .and(|q| q.where_eq("a", 1).where_eq("b", 2))
    .or(|q| q.where_eq("x", 1).where_eq("y", 2))
    
    // Ordering
    .order_by("created_at", Order::Desc)
    .order_by("name", Order::Asc)
    
    // Pagination
    .limit(100)
    .offset(200)
    
    // Execute
    .execute()?;        // QueryResult
    .to_toon()?;        // String (TOON format)
    .to_json()?;        // String (JSON format)
    .count()?;          // u64
    .first()?;          // Option<Row>
    .exists()?;         // bool
```

---

### ContextQueryBuilder

Build optimized LLM context with budget management.

```rust
pub struct ContextQueryBuilder {
    // ...
}
```

#### Full Example

```rust
use sochdb::{
    ContextQueryBuilder, ContextFormat, TruncationStrategy, ContextValue
};

let result = ContextQueryBuilder::new()
    // Session and budget
    .for_session("session_abc123")
    .with_budget(8192)  // 8K token budget
    
    // Variables for dynamic content
    .set_var("user_id", ContextValue::Int(42))
    .set_var("query", ContextValue::String("How do I reset my password?".into()))
    
    // System prompt (highest priority: -1)
    .literal("SYSTEM", -1, "You are a helpful customer support agent.")
    
    // User profile (priority 0)
    .section("USER", 0)
        .get("users.{name, email, plan, created_at}")
        .where_eq("id", "$user_id")
        .done()
    
    // Conversation history (priority 1)
    .section("HISTORY", 1)
        .last(20, "messages")
        .where_eq("session_id", "session_abc123")
        .format_as("${role}: ${content}")
        .done()
    
    // Similar past tickets via vector search (priority 2)
    .section("SIMILAR_TICKETS", 2)
        .search("tickets", "embedding", 5)
        .min_score(0.75)
        .fields(&["title", "resolution"])
        .done()
    
    // Knowledge base articles (priority 3)
    .section("KNOWLEDGE", 3)
        .search("articles", "embedding", 3)
        .min_score(0.8)
        .fields(&["title", "content"])
        .max_tokens(2000)  // Per-section budget
        .done()
    
    // Output configuration
    .format(ContextFormat::Soch)
    .truncation(TruncationStrategy::PriorityDrop)
    .include_schema(false)
    
    // Execute
    .execute()?;

// Access results
println!("Token count: {}/{}", result.token_count, 8192);
println!("Sections included: {:?}", result.included_sections());
println!("Sections dropped: {:?}", result.dropped_sections());
println!("\n{}", result.context);
```

#### Truncation Strategies

| Strategy | Behavior |
|----------|----------|
| `Strict` | Error if budget exceeded |
| `TailDrop` | Drop content from end |
| `HeadDrop` | Drop content from beginning |
| `MiddleDrop` | Drop from middle, keep start/end |
| `PriorityDrop` | Drop lowest priority sections first |

---

## Vector Operations

### HNSWIndex

Hierarchical Navigable Small World graph for vector similarity search.

```rust
pub struct HNSWIndex {
    // ...
}
```

#### Configuration

```rust
let config = HNSWConfig {
    m: 16,                         // Connections per layer
    m_max: 32,                     // Connections at layer 0
    ef_construction: 200,          // Build quality
    ef_search: 50,                 // Search quality
    metric: DistanceMetric::Cosine,
    ..Default::default()
};

let index = HNSWIndex::with_config(config);
```

#### Operations

```rust
// Insert vectors
index.insert(edge_id, &vector)?;

// Search
let results = index.search(&query_vector, k)?;
for (id, distance) in results {
    println!("ID: {}, Distance: {:.4}", id, distance);
}

// Persistence
index.save_to_disk("vectors.hnsw")?;
let index = HNSWIndex::load_from_disk("vectors.hnsw")?;

// Stats
println!("Vectors: {}", index.len());
println!("Max level: {}", index.max_level());
```

#### Distance Metrics

```rust
pub enum DistanceMetric {
    /// Cosine distance: 1 - cos(a, b)
    /// Best for: Normalized embeddings, text similarity
    Cosine,
    
    /// Euclidean distance: ||a - b||₂
    /// Best for: Spatial data, unnormalized vectors
    Euclidean,
    
    /// Negative dot product: -a · b
    /// Best for: Pre-normalized vectors, MIPS
    DotProduct,
}
```

#### Client Integration

```rust
// Via SochConnection
conn.vector_insert("documents", doc_id, &embedding, Some(metadata))?;

let results = conn.vector_search("documents", &query_embedding, 10)?;
for result in results {
    println!("ID: {}, Distance: {:.4}, Meta: {:?}", 
        result.id, result.distance, result.metadata);
}

// Batch insert
conn.vector_batch_insert("documents", vectors)?;
```

---

## Storage Layer

### DurableStorage

Handles WAL, MVCC, and durability.

```rust
pub struct DurableStorage {
    // ...
}
```

#### Operations

```rust
// Open storage
let storage = DurableStorage::open("./data")?;

// Or with group commit
let storage = DurableStorage::open_with_group_commit("./data")?;

// Basic operations
storage.put(key, value, txn_id)?;
let value = storage.get(key, snapshot_ts)?;
storage.delete(key, txn_id)?;

// Scan
let iter = storage.scan(start_key, end_key, snapshot_ts)?;

// Durability
storage.fsync()?;
storage.checkpoint()?;

// Recovery
let stats = storage.recover()?;
println!("Recovered {} transactions", stats.transactions_recovered);
```

---

### Block Checksums

CRC32C checksums for data integrity.

```rust
// Creating checksummed blocks
let block = ChecksummedBlock::new(data, BlockType::Data);
let bytes = block.to_bytes();

// Verifying blocks
let verified = ChecksummedBlock::from_bytes(&bytes)?;
assert_eq!(verified.checksum, block.checksum);

// Quick verification without parsing
let is_valid = ChecksummedBlock::verify(&bytes);
```

#### Block Types

```rust
pub enum BlockType {
    Data,
    Index,
    Bloom,
    ColumnData,
    ColumnMeta,
    TemporalIndex,
    Unknown,
}
```

---

### Columnar Storage

Column-oriented storage for analytical queries.

```rust
pub struct ColumnarTable {
    pub name: String,
    pub columns: HashMap<String, Column>,
    pub row_count: usize,
}

pub struct Column {
    pub name: String,
    pub dtype: ColumnType,
    pub data: ColumnData,
    pub validity: Option<Vec<u8>>,  // Null bitmap
    pub len: usize,
}
```

#### Column Types

```rust
pub enum ColumnType {
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    String,
    Binary,
    Struct(Vec<(String, Box<ColumnType>)>),
    List(Box<ColumnType>),
}
```

---

## Kernel API

### Database

The core database kernel.

```rust
pub struct Database {
    // ...
}
```

#### Opening

```rust
// Basic open
let db = Database::open("./data")?;

// With configuration
let db = Database::open_with_config("./data", DatabaseConfig {
    group_commit: true,
    sync_mode: SyncMode::Normal,
    max_wal_size: 64 * 1024 * 1024,
    memtable_size: 4 * 1024 * 1024,
    block_cache_size: 64 * 1024 * 1024,
    compression: Compression::LZ4,
})?;
```

#### Transaction Management

```rust
// Begin transaction
let txn = db.begin_transaction()?;

// Operations
db.put_path(txn, "users/1/name", b"Alice")?;
let value = db.get_path(txn, "users/1/name")?;
db.delete_path(txn, "users/1/temp")?;

// Commit or rollback
db.commit(txn)?;
// or
db.abort(txn)?;
```

#### Query Execution

```rust
let result = db.query(txn, "users")
    .columns(&["id", "name"])
    .limit(100)
    .execute()?;

println!("TOON: {}", result.to_toon());
```

#### Maintenance

```rust
// Force sync
db.fsync()?;

// Checkpoint (returns LSN)
let lsn = db.checkpoint()?;

// Garbage collection
let freed = db.gc();

// Statistics
let stats = db.stats();
println!("Queries: {}", stats.queries_executed);
println!("Bytes written: {}", stats.bytes_written);

// Shutdown
db.shutdown()?;
```

---

### Transaction Handle

```rust
#[derive(Debug, Clone, Copy)]
pub struct TxnHandle {
    pub txn_id: u64,
    pub snapshot_ts: u64,
}
```

---

### Query Result

```rust
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, SochValue>>,
    pub rows_scanned: usize,
    pub bytes_read: usize,
}

impl QueryResult {
    /// Convert to TOON format
    pub fn to_toon(&self) -> String { ... }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool { ... }
    
    /// Row count
    pub fn len(&self) -> usize { ... }
}
```

---

## Plugin System

### Extension Trait

Base trait for all plugins.

```rust
pub trait Extension: Send + Sync {
    fn info(&self) -> ExtensionInfo;
    fn init(&mut self) -> KernelResult<()>;
    fn shutdown(&mut self) -> KernelResult<()>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub struct ExtensionInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub capabilities: Vec<ExtensionCapability>,
}
```

---

### Storage Extension

Custom storage backends.

```rust
pub trait StorageExtension: Extension {
    fn get(&self, table_id: TableId, key: &[u8]) -> KernelResult<Option<Vec<u8>>>;
    fn put(&self, table_id: TableId, key: &[u8], value: &[u8], txn_id: TransactionId) -> KernelResult<()>;
    fn delete(&self, table_id: TableId, key: &[u8], txn_id: TransactionId) -> KernelResult<()>;
    fn scan(&self, table_id: TableId, start: &[u8], end: &[u8], limit: usize) -> KernelResult<Vec<(Vec<u8>, Vec<u8>)>>;
    fn flush(&self) -> KernelResult<()>;
    fn compact(&self) -> KernelResult<()>;
    fn stats(&self) -> StorageStats;
}
```

---

### Index Extension

Custom index types.

```rust
pub trait IndexExtension: Extension {
    fn index_type(&self) -> &str;
    fn build(&mut self, table_id: TableId, column_id: u16, data: &[(RowId, Vec<u8>)]) -> KernelResult<()>;
    fn insert(&mut self, key: &[u8], row_id: RowId) -> KernelResult<()>;
    fn delete(&mut self, key: &[u8], row_id: RowId) -> KernelResult<()>;
    fn lookup(&self, key: &[u8]) -> KernelResult<Vec<RowId>>;
    fn range(&self, start: &[u8], end: &[u8], limit: usize) -> KernelResult<Vec<RowId>>;
    fn nearest(&self, query: &[u8], k: usize) -> KernelResult<Vec<(RowId, f32)>>;
    fn size_bytes(&self) -> u64;
}
```

---

### Observability Extension

Metrics, tracing, and logging.

```rust
pub trait ObservabilityExtension: Extension {
    // Metrics
    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]);
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    
    // Tracing
    fn span_start(&self, name: &str, parent: Option<u64>) -> u64;
    fn span_end(&self, span_id: u64);
    fn span_event(&self, span_id: u64, name: &str, attributes: &[(&str, &str)]);
    
    // Logging
    fn log(&self, level: LogLevel, message: &str, fields: &[(&str, &str)]);
}
```

---

### Registering Plugins

```rust
// Get plugin manager
let plugins = db.plugins();

// Register storage plugin
plugins.register_storage(Box::new(MyStoragePlugin::new()))?;

// Register index plugin
plugins.register_index(Box::new(MyIndexPlugin::new()))?;

// Register observability plugin
plugins.register_observability(Box::new(PrometheusPlugin::new()))?;

// List registered plugins
for info in plugins.list() {
    println!("{} v{}: {}", info.name, info.version, info.description);
}
```

---

## Error Types

### ClientError

```rust
pub enum ClientError {
    ConnectionFailed(String),
    QueryFailed(String),
    SchemaError(String),
    Storage(String),
    Transaction(String),
    InvalidPath(String),
    NotFound(String),
}
```

### KernelError

```rust
pub enum KernelError {
    Io(std::io::Error),
    Storage(String),
    Transaction { txn_id: u64, message: String },
    TableNotFound { name: String },
    ColumnNotFound { table: String, column: String },
    TypeMismatch { expected: String, found: String },
    ConstraintViolation(String),
    Plugin { message: String },
    Internal(String),
}
```

---

## Constants

### Performance Tuning

```rust
// Default buffer/cache sizes
pub const DEFAULT_MEMTABLE_SIZE: usize = 4 * 1024 * 1024;      // 4MB
pub const DEFAULT_BLOCK_CACHE_SIZE: usize = 64 * 1024 * 1024;  // 64MB
pub const DEFAULT_WAL_SIZE: u64 = 64 * 1024 * 1024;            // 64MB

// HNSW defaults
pub const DEFAULT_HNSW_M: usize = 16;
pub const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 200;
pub const DEFAULT_HNSW_EF_SEARCH: usize = 50;

// Block sizes
pub const BLOCK_SIZE: usize = 4096;
pub const BLOCK_TRAILER_SIZE: usize = 5;  // 4 byte CRC + 1 byte type

// Validation
pub const MAX_KEY_SIZE: usize = 64 * 1024;
pub const MAX_VALUE_SIZE: usize = 1024 * 1024 * 1024;  // 1GB
```

---

## Version Information

```rust
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_HASH: &str = env!("GIT_HASH");

pub fn version_info() -> VersionInfo {
    VersionInfo {
        version: VERSION.to_string(),
        git_hash: GIT_HASH.to_string(),
        rust_version: rustc_version(),
        features: enabled_features(),
    }
}
```

---

*For the latest documentation, see [sochdb.dev](https://sochdb.dev)*
