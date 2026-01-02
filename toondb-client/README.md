# ToonDB Rust Client

[![Crates.io](https://img.shields.io/crates/v/toondb.svg)](https://crates.io/crates/toondb)
[![Documentation](https://docs.rs/toondb/badge.svg)](https://docs.rs/toondb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

The official Rust client SDK for **ToonDB** — a high-performance embedded document database with HNSW vector search, built-in multi-tenancy, and SQL support.

## Features

- ✅ **Zero-Copy Reads** — Direct access to memory-mapped data
- ✅ **Native Vector Search** — Built-in HNSW index for embeddings
- ✅ **SQL Support** — Full SQL via toondb-query integration
- ✅ **IPC Client** — Connect to ToonDB server (async)
- ✅ **Multi-Tenancy** — Efficient prefix scanning for data isolation
- ✅ **ACID Transactions** — Snapshot isolation with automatic commit/abort
- ✅ **Thread-Safe** — Safe concurrent access with MVCC
- ✅ **Columnar Storage** — Efficient for analytical queries

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
toondb = "0.2.9"
tokio = { version = "1", features = ["full"] }  # For async IPC
```

## Quick Start

### IPC Client (Async)

```rust
use toondb::IpcClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to ToonDB server
    let mut client = IpcClient::connect("./my_database/toondb.sock").await?;
    
    // Put and Get
    client.put(b"user:123", b r#"{"name":"Alice","age":30}"#).await?;
    let value = client.get(b"user:123").await?;
    
    if let Some(data) = value {
        println!("{}", String::from_utf8_lossy(&data));
        // Output: {"name":"Alice","age":30}
    }
    
    Ok(())
}
```

**Start server first:**
```bash
toondb-server --db ./my_database
# Output: [IpcServer] Listening on "./my_database/toondb.sock"
```

### Embedded Mode (Direct FFI)

For single-process applications with maximum performance:

```rust
use toondb_core::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open database
    let db = Database::open("./my_database")?;
    
    // Key-value operations
    db.put(b"key", b"value")?;
    let value = db.get(b"key")?;
    
    if let Some(data) = value {
        println!("{}", String::from_utf8_lossy(&data));
        // Output: value
    }
    
    Ok(())
}
```

## Core Operations

### Basic Key-Value

```rust
// Put
client.put(b"key", b"value").await?;

// Get
match client.get(b"key").await? {
    Some(value) => println!("{}", String::from_utf8_lossy(&value)),
    None => println!("Key not found"),
}

// Delete
client.delete(b"key").await?;
```

**Output:**
```
value
Key not found (after delete)
```

### Path Operations

```rust
// Hierarchical data storage
client.put_path("users/alice/email", b"alice@example.com").await?;
client.put_path("users/alice/age", b"30").await?;
client.put_path("users/bob/email", b"bob@example.com").await?;

// Retrieve by path
if let Some(email) = client.get_path("users/alice/email").await? {
    println!("Alice's email: {}", String::from_utf8_lossy(&email));
}
```

**Output:**
```
Alice's email: alice@example.com
```

### Prefix Scanning ⭐

The most efficient way to iterate keys with a common prefix:

```rust
use toondb::IpcClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = IpcClient::connect("./db/toondb.sock").await?;
    
    // Insert multi-tenant data
    client.put(b"tenants/acme/users/1", b r#"{"name":"Alice"}"#).await?;
    client.put(b"tenants/acme/users/2", b r#"{"name":"Bob"}"#).await?;
    client.put(b"tenants/acme/orders/1", b r#"{"total":100}"#).await?;
    client.put(b"tenants/globex/users/1", b r#"{"name":"Charlie"}"#).await?;
    
    // Scan only ACME Corp data (tenant isolation)
    let results = client.scan("tenants/acme/").await?;
    println!("ACME Corp has {} items:", results.len());
    
    for kv in results {
        println!("  {}: {}", 
            String::from_utf8_lossy(&kv.key),
            String::from_utf8_lossy(&kv.value)
        );
    }
    
    Ok(())
}
```

**Output:**
```
ACME Corp has 3 items:
  tenants/acme/orders/1: {"total":100}
  tenants/acme/users/1: {"name":"Alice"}
  tenants/acme/users/2: {"name":"Bob"}
```

**Why use scan():**
- **Fast**: O(|prefix|) performance
- **Isolated**: Perfect for multi-tenant apps
- **Efficient**: Zero-copy reads from storage

## Transactions

### Automatic Transactions

```rust
// Transaction with automatic commit/abort
client.with_transaction(|txn| async move {
    txn.put(b"account:1:balance", b"1000").await?;
    txn.put(b"account:2:balance", b"500").await?;
    Ok(())
}).await?;
```

**Output:**
```
✅ Transaction committed
```

### Manual Transaction Control

```rust
let txn = client.begin_transaction().await?;

txn.put(b"key1", b"value1").await?;
txn.put(b"key2", b"value2").await?;

// Commit or abort
if success {
    client.commit_transaction(txn).await?;
} else {
    client.abort_transaction(txn).await?;
}
```

## SQL Operations

ToonDB supports full SQL via the `toondb-query` crate:

```rust
use toondb_query::QueryEngine;
use toondb_core::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("./sql_db")?;
    let query_engine = QueryEngine::new(db);
    
    // Create table
    query_engine.execute(r#"
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        )
    "#)?;
    
    // Insert data
    query_engine.execute(r#"
        INSERT INTO users (id, name, email, age)
        VALUES (1, 'Alice', 'alice@example.com', 30)
    "#)?;
    
    query_engine.execute(r#"
        INSERT INTO users (id, name, email, age)
        VALUES (2, 'Bob', 'bob@example.com', 25)
    "#)?;
    
    // Query
    let results = query_engine.execute("SELECT * FROM users WHERE age > 26")?;
    for row in results {
        println!("{:?}", row);
    }
    
    Ok(())
}
```

**Output:**
```
Row { id: 1, name: "Alice", email: "alice@example.com", age: 30 }
```

### Complex SQL Queries

```rust
// JOIN query
let results = query_engine.execute(r#"
    SELECT users.name, orders.product, orders.amount
    FROM users
    JOIN orders ON users.id = orders.user_id
    WHERE orders.amount > 50
    ORDER BY orders.amount DESC
"#)?;

for row in results {
    println!("{} bought {} for ${}", 
        row.get_str("name")?, 
        row.get_str("product")?, 
        row.get_f64("amount")?
    );
}
```

**Output:**
```
Alice bought Laptop for $999.99
Bob bought Keyboard for $75
```

### Aggregations

```rust
// GROUP BY with aggregations
let results = query_engine.execute(r#"
    SELECT users.name, COUNT(*) as order_count, SUM(orders.amount) as total
    FROM users
    JOIN orders ON users.id = orders.user_id
    GROUP BY users.name
    ORDER BY total DESC
"#)?;

for row in results {
    println!("{}: {} orders, ${} total",
        row.get_str("name")?,
        row.get_i64("order_count")?,
        row.get_f64("total")?
    );
}
```

**Output:**
```
Alice: 2 orders, $1024.99 total
Bob: 1 orders, $75 total
```

## Vector Search

### HNSW Index

```rust
use toondb_index::{HnswIndex, DistanceMetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create HNSW index
    let mut index = HnswIndex::new(
        384,                      // dimension
        DistanceMetric::Cosine,   // metric
        16,                       // m
        100                       // ef_construction
    )?;
    
    // Build from embeddings
    let embeddings = vec![
        vec![0.1, 0.2, 0.3, /* ... 384 dims */],
        vec![0.4, 0.5, 0.6, /* ... 384 dims */],
    ];
    let labels = vec!["doc1", "doc2"];
    
    index.bulk_build(&embeddings, &labels)?;
    
    // Search
    let query = vec![0.15, 0.25, 0.35, /* ... 384 dims */];
    let results = index.query(&query, 10, 50)?; // k=10, ef_search=50
    
    for (i, result) in results.iter().enumerate() {
        println!("{}. {} (distance: {:.4})", 
            i + 1, 
            result.label, 
            result.distance
        );
    }
    
    Ok(())
}
```

**Output:**
```
1. doc1 (distance: 0.0234)
2. doc2 (distance: 0.1567)
```

## Complete Example: Multi-Tenant SaaS App

```rust
use toondb::IpcClient;
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = IpcClient::connect("./saas_db/toondb.sock").await?;
    
    // Insert tenant data
    client.put(
        b"tenants/acme/users/alice",
        br#"{"role":"admin","email":"alice@acme.com"}"#
    ).await?;
    
    client.put(
        b"tenants/acme/users/bob",
        br#"{"role":"user","email":"bob@acme.com"}"#
    ).await?;
    
    client.put(
        b"tenants/globex/users/charlie",
        br#"{"role":"admin","email":"charlie@globex.com"}"#
    ).await?;
    
    // Scan ACME Corp data only (tenant isolation)
    let acme_data = client.scan("tenants/acme/").await?;
    println!("ACME Corp: {} users", acme_data.len());
    
    for kv in acme_data {
        let user: Value = serde_json::from_slice(&kv.value)?;
        println!("  {}: {} ({})",
            String::from_utf8_lossy(&kv.key),
            user["email"].as_str().unwrap(),
            user["role"].as_str().unwrap()
        );
    }
    
    // Scan Globex Corp data
    let globex_data = client.scan("tenants/globex/").await?;
    println!("\nGlobex Inc: {} users", globex_data.len());
    
    for kv in globex_data {
        let user: Value = serde_json::from_slice(&kv.value)?;
        println!("  {}: {} ({})",
            String::from_utf8_lossy(&kv.key),
            user["email"].as_str().unwrap(),
            user["role"].as_str().unwrap()
        );
    }
    
    Ok(())
}
```

**Output:**
```
ACME Corp: 2 users
  tenants/acme/users/alice: alice@acme.com (admin)
  tenants/acme/users/bob: bob@acme.com (user)

Globex Inc: 1 users
  tenants/globex/users/charlie: charlie@globex.com (admin)
```

## API Reference

### IpcClient (Async)

| Method | Description |
|--------|-------------|
| `IpcClient::connect(path)` | Connect to IPC server |
| `put(key, value)` | Store key-value pair |
| `get(key)` | Retrieve value (Option) |
| `delete(key)` | Delete a key |
| `put_path(path, value)` | Store at hierarchical path |
| `get_path(path)` | Retrieve by path |
| `scan(prefix)` | Scan keys with prefix |
| `begin_transaction()` | Start transaction |
| `commit_transaction(txn)` | Commit transaction |
| `abort_transaction(txn)` | Abort transaction |

### Database (Embedded)

| Method | Description |
|--------|-------------|
| `Database::open(path)` | Open/create database |
| `put(key, value)` | Store key-value pair |
| `get(key)` | Retrieve value (Option) |
| `delete(key)` | Delete a key |
| `scan(start, end)` | Iterate key range |
| `checkpoint()` | Force durability checkpoint |
| `stats()` | Get storage statistics |

## Configuration

```rust
use toondb::Config;

let config = Config {
    create_if_missing: true,
    wal_enabled: true,
    sync_mode: SyncMode::Normal,  // Full, Normal, Off
    memtable_size_bytes: 64 * 1024 * 1024,  // 64MB
    ..Default::default()
};

let db = Database::open_with_config("./my_db", config)?;
```

## Error Handling

```rust
use toondb::{IpcClient, Error};

match client.get(b"key").await {
    Ok(Some(value)) => {
        println!("Found: {}", String::from_utf8_lossy(&value));
    }
    Ok(None) => {
        println!("Key not found");
    }
    Err(Error::ConnectionFailed) => {
        eprintln!("Server not running!");
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Best Practices

✅ **Use IPC for multi-process** — Better for microservices
✅ **Use embedded for single-process** — Maximum performance
✅ **Use scan() for multi-tenancy** — Efficient prefix-based isolation
✅ **Use transactions** — Atomic multi-key operations
✅ **Use async/await** — Non-blocking I/O for IPC
✅ **Handle errors properly** — Match on Error variants

## Crate Organization

| Crate | Purpose |
|-------|---------|
| `toondb` | High-level client SDK (this crate) |
| `toondb-core` | Core database engine |
| `toondb-storage` | Storage layer with IPC server |
| `toondb-index` | Vector search (HNSW) |
| `toondb-query` | SQL query engine |
| `toondb-client` | Low-level client bindings |

## Building from Source

```bash
# Clone repository
git clone https://github.com/toondb/toondb
cd toondb

# Build all crates
cargo build --release

# Run tests
cargo test --all

# Build specific crate
cargo build --release -p toondb-client
```

## Examples

See the [examples directory](../examples/rust) for more:

- `basic_operations.rs` - Simple key-value operations
- `multi_tenant.rs` - Multi-tenant data isolation
- `transactions.rs` - ACID transactions
- `vector_search.rs` - HNSW vector search
- `sql_queries.rs` - SQL operations

## Platform Support

- Linux (x86_64, aarch64)
- macOS (Intel, Apple Silicon)
- Windows (x64)

Requires Rust 1.70 or later.

## License

Apache License 2.0

## Links

- [Documentation](https://docs.rs/toondb)
- [Crates.io](https://crates.io/crates/toondb)
- [Python SDK](../toondb-python-sdk)
- [Go SDK](../toondb-go)
- [JavaScript SDK](../toondb-js)
- [GitHub](https://github.com/toondb/toondb)

## Support

- GitHub Issues: https://github.com/toondb/toondb/issues
- Email: sushanth@toondb.dev

## Author

**Sushanth** - [GitHub](https://github.com/sushanthpy)
