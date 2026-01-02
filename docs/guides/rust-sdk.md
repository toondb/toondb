# Rust SDK Guide

> **Version:** 0.2.7  
> **Time:** 25 minutes  
> **Difficulty:** Intermediate  
> **Prerequisites:** Rust 1.70+, Tokio runtime

Complete guide to ToonDB's Rust SDK with async IPC client, SQL integration, and zero-copy operations.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Async IPC Client](#async-ipc-client)
4. [SQL Integration](#sql-integration)
5. [Key-Value Operations](#key-value-operations)
6. [Path API](#path-api)
7. [Prefix Scanning](#prefix-scanning)
8. [Transactions](#transactions)
9. [Query Builder](#query-builder)
10. [Zero-Copy Reads](#zero-copy-reads)
11. [Best Practices](#best-practices)
12. [Complete Examples](#complete-examples)

---

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
toondb-client = "0.2.7"
tokio = { version = "1", features = ["full"] }

# Optional: For SQL support
toondb-query = "0.2.7"
```

**What's New in 0.2.6:**
- ✅ Async IPC client with Tokio
- ✅ Enhanced scan() method for multi-tenant isolation
- ✅ SQL integration via toondb-query
- ✅ Zero-copy read optimizations
- ✅ Improved error types

---

## Quick Start

### Async IPC Client

```rust
use toondb_client::IpcClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to ToonDB server
    let client = IpcClient::connect("./my_database/toondb.sock").await?;
    
    // Put and Get
    client.put(b"user:123", b"{\"name\":\"Alice\",\"age\":30}").await?;
    let value = client.get(b"user:123").await?;
    
    if let Some(val) = value {
        println!("{}", String::from_utf8_lossy(&val));
        // Output: {"name":"Alice","age":30}
    }
    
    Ok(())
}
```

**Output:**
```
{"name":"Alice","age":30}
```

---

## Async IPC Client

⭐ **ToonDB uses async IPC** — no blocking operations:

### Start Server

```bash
# Terminal 1: Start ToonDB server
toondb-server --db ./my_database
# Server listens at: ./my_database/toondb.sock
```

### Connect from Rust

```rust
use toondb_client::IpcClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IpcClient::connect("./my_database/toondb.sock").await?;
    
    // Check latency
    let latency = client.ping().await?;
    println!("Ping: {:.2}ms", latency * 1000.0);
    
    Ok(())
}
```

**Output:**
```
Ping: 0.12ms
```

**Wire Protocol:**
- All integers: Little Endian
- Format: `[opcode:1][length:4 LE][payload]`
- Binary-safe: Handles null bytes, UTF-8

---

## SQL Integration

⭐ **Full SQL via toondb-query:**

### CREATE TABLE

```rust
use toondb_client::IpcClient;
use toondb_query::SqlExecutor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IpcClient::connect("./my_db/toondb.sock").await?;
    let executor = SqlExecutor::new(client);
    
    // Create table
    executor.execute(r#"
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        )
    "#).await?;
    
    // Insert data
    executor.execute(r#"
        INSERT INTO users (id, name, email, age)
        VALUES (1, 'Alice', 'alice@example.com', 30)
    "#).await?;
    
    executor.execute(r#"
        INSERT INTO users (id, name, email, age)
        VALUES (2, 'Bob', 'bob@example.com', 25)
    "#).await?;
    
    println!("✅ Table created and data inserted");
    
    Ok(())
}
```

**Output:**
```
✅ Table created and data inserted
```

### SELECT Queries

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct User {
    id: i32,
    name: String,
    email: String,
    age: i32,
}

// Query all users
let results: Vec<User> = executor
    .query("SELECT * FROM users")
    .await?;

for user in results {
    println!("{}: {} ({} years old)", user.id, user.name, user.age);
}
```

**Output:**
```
1: Alice (30 years old)
2: Bob (25 years old)
```

### JOIN Queries

```rust
// Create orders table
executor.execute(r#"
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product TEXT,
        amount REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
"#).await?;

// Insert orders
executor.execute("INSERT INTO orders VALUES (1, 1, 'Laptop', 999.99)").await?;
executor.execute("INSERT INTO orders VALUES (2, 1, 'Mouse', 25.00)").await?;
executor.execute("INSERT INTO orders VALUES (3, 2, 'Keyboard', 75.00)").await?;

// JOIN query
#[derive(Debug, Deserialize)]
struct OrderSummary {
    name: String,
    product: String,
    amount: f64,
}

let results: Vec<OrderSummary> = executor.query(r#"
    SELECT users.name, orders.product, orders.amount
    FROM users
    JOIN orders ON users.id = orders.user_id
    WHERE orders.amount > 50
    ORDER BY orders.amount DESC
"#).await?;

for order in results {
    println!("{} bought {} for ${}", order.name, order.product, order.amount);
}
```

**Output:**
```
Alice bought Laptop for $999.99
Bob bought Keyboard for $75
```

### Aggregations

```rust
#[derive(Debug, Deserialize)]
struct UserStats {
    name: String,
    order_count: i32,
    total: f64,
}

let results: Vec<UserStats> = executor.query(r#"
    SELECT 
        users.name,
        COUNT(*) as order_count,
        SUM(orders.amount) as total
    FROM users
    JOIN orders ON users.id = orders.user_id
    GROUP BY users.name
    ORDER BY total DESC
"#).await?;

for stat in results {
    println!("{}: {} orders, ${:.2} total", stat.name, stat.order_count, stat.total);
}
```

**Output:**
```
Alice: 2 orders, $1024.99 total
Bob: 1 orders, $75.00 total
```

---

## Key-Value Operations

### Basic Operations

```rust
use toondb_client::IpcClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IpcClient::connect("./my_db/toondb.sock").await?;
    
    // Put
    client.put(b"key", b"value").await?;
    
    // Get
    if let Some(value) = client.get(b"key").await? {
        println!("{}", String::from_utf8_lossy(&value));
    }
    
    // Delete
    client.delete(b"key").await?;
    
    // Get after delete
    let deleted = client.get(b"key").await?;
    println!("After delete: {:?}", deleted);
    
    Ok(())
}
```

**Output:**
```
value
After delete: None
```

### JSON Operations

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct User {
    name: String,
    email: String,
    age: u32,
}

// Store JSON
let user = User {
    name: "Alice".to_string(),
    email: "alice@example.com".to_string(),
    age: 30,
};

let json = serde_json::to_vec(&user)?;
client.put(b"users/alice", &json).await?;

// Retrieve JSON
if let Some(value) = client.get(b"users/alice").await? {
    let user: User = serde_json::from_slice(&value)?;
    println!("Name: {}, Age: {}", user.name, user.age);
}
```

**Output:**
```
Name: Alice, Age: 30
```

---

## Path API

```rust
// Store hierarchical data
client.put_path(&["users", "alice", "email"], b"alice@example.com").await?;
client.put_path(&["users", "alice", "age"], b"30").await?;
client.put_path(&["users", "alice", "settings", "theme"], b"dark").await?;

// Retrieve by path
if let Some(email) = client.get_path(&["users", "alice", "email"]).await? {
    println!("Alice's email: {}", String::from_utf8_lossy(&email));
}
```

**Output:**
```
Alice's email: alice@example.com
```

**Wire Format:**
```
[path_count: 2 bytes LE]
[path_len_1: 2 bytes LE][path_1: UTF-8]
[path_len_2: 2 bytes LE][path_2: UTF-8]
...
```

---

## Prefix Scanning

⭐ **New in 0.2.6** — Multi-tenant isolation:

```rust
// Insert multi-tenant data
client.put(b"tenants/acme/users/1", b"{\"name\":\"Alice\"}").await?;
client.put(b"tenants/acme/users/2", b"{\"name\":\"Bob\"}").await?;
client.put(b"tenants/acme/orders/1", b"{\"total\":100}").await?;
client.put(b"tenants/globex/users/1", b"{\"name\":\"Charlie\"}").await?;

// Scan only ACME Corp data (tenant isolation)
let results = client.scan(b"tenants/acme/", b"tenants/acme;").await?;

println!("ACME Corp has {} items:", results.len());
for (key, value) in results {
    println!("  {}: {}", 
        String::from_utf8_lossy(&key), 
        String::from_utf8_lossy(&value)
    );
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
- **Fast**: O(|prefix|) — only reads matching keys
- **Isolated**: Perfect for multi-tenancy
- **Efficient**: Binary-safe iteration

**Range trick:**
```rust
// Scan "users/" to "users;" (semicolon is after '/' in ASCII)
let results = client.scan(b"users/", b"users;").await?;
// Matches: users/1, users/2, users/alice, ...
// Excludes: user, users, usersabc
```

---

## Transactions

### Atomic Operations

```rust
let txn = client.begin_transaction().await?;

// Perform operations
txn.put(b"account:1:balance", b"1000").await?;
txn.put(b"account:2:balance", b"500").await?;

// Commit or abort
if some_condition {
    txn.commit().await?;
    println!("✅ Transaction committed");
} else {
    txn.abort().await?;
    println!("❌ Transaction aborted");
}
```

**Output:**
```
✅ Transaction committed
```

### Transaction with Scan

```rust
let txn = client.begin_transaction().await?;

txn.put(b"key1", b"value1").await?;
txn.put(b"key2", b"value2").await?;

// Scan within transaction
let results = txn.scan(b"key", b"key~").await?;
for (key, value) in results {
    println!("{}: {}", 
        String::from_utf8_lossy(&key), 
        String::from_utf8_lossy(&value)
    );
}

txn.commit().await?;
```

**Output:**
```
key1: value1
key2: value2
```

---

## Query Builder

Returns results in **TOON format** (token-optimized):

```rust
use toondb_client::query::QueryBuilder;

// Insert structured data
client.put(
    b"products/laptop",
    b"{\"name\":\"Laptop\",\"price\":999,\"stock\":5}"
).await?;

client.put(
    b"products/mouse",
    b"{\"name\":\"Mouse\",\"price\":25,\"stock\":20}"
).await?;

// Query with column selection
let results = QueryBuilder::new(&client)
    .prefix("products/")
    .select(&["name", "price"])
    .limit(10)
    .execute()
    .await?;

for (key, value) in results {
    println!("{}: {}", 
        String::from_utf8_lossy(&key), 
        String::from_utf8_lossy(&value)
    );
}
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
```

**TOON benefits:**
- Fewer tokens for LLMs
- Structured output
- Easy parsing

---

## Zero-Copy Reads

⭐ **Avoid allocations with Bytes:**

```rust
use bytes::Bytes;

// Zero-copy read (returns Bytes backed by socket buffer)
let value: Option<Bytes> = client.get_zero_copy(b"large_blob").await?;

if let Some(bytes) = value {
    // No heap allocation — direct view into socket buffer
    println!("Read {} bytes (zero-copy)", bytes.len());
    
    // Can slice without copying
    let slice = &bytes[0..100];
}
```

**Performance:**
- Regular `get()`: Allocates `Vec<u8>`
- Zero-copy `get_zero_copy()`: Returns Bytes (shared buffer)
- Best for: Large values, read-heavy workloads

---

## Best Practices

### 1. Use SQL for Structured Data

```rust
// ✅ Good: Use SQL for relational data
executor.execute("CREATE TABLE users (...)").await?;
let users: Vec<User> = executor.query("SELECT * FROM users WHERE age > 25").await?;
```

### 2. Use K-V for Unstructured Data

```rust
// ✅ Good: Use K-V for documents, blobs, cache
client.put(b"cache:user:123", &json_bytes).await?;
client.put(b"blob:image:456", &image_bytes).await?;
```

### 3. Use scan() for Multi-Tenancy

```rust
// ✅ Good: Efficient tenant isolation
let tenant_id = "acme";
let prefix = format!("tenants/{}/", tenant_id);
let end = format!("tenants/{};", tenant_id);
let data = client.scan(prefix.as_bytes(), end.as_bytes()).await?;
```

### 4. Use Transactions for Atomicity

```rust
// ✅ Good: Atomic operations
let txn = client.begin_transaction().await?;
txn.put(b"key1", b"value1").await?;
txn.put(b"key2", b"value2").await?;
txn.commit().await?;
```

### 5. Use Zero-Copy for Large Reads

```rust
// ✅ Good: Zero-copy for large blobs
let blob = client.get_zero_copy(b"large_blob").await?;

// ❌ Bad: Unnecessary allocation
let blob = client.get(b"large_blob").await?; // Copies to `Vec<u8>`
```

### 6. Handle Errors Properly

```rust
// ✅ Good: Proper error handling
match client.get(b"key").await {
    Ok(Some(value)) => println!("Value: {:?}", value),
    Ok(None) => println!("Key not found"),
    Err(e) => eprintln!("Error: {}", e),
}

// ❌ Bad: Unwrap can panic
let value = client.get(b"key").await.unwrap().unwrap();
```

---

## Complete Examples

### Example 1: Multi-Tenant SaaS with SQL + K-V

```rust
use toondb_client::IpcClient;
use toondb_query::SqlExecutor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Tenant {
    id: i32,
    name: String,
    created_at: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct TenantUser {
    role: String,
    email: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IpcClient::connect("./saas_db/toondb.sock").await?;
    let executor = SqlExecutor::new(client.clone());
    
    // SQL for tenant metadata
    executor.execute(r#"
        CREATE TABLE IF NOT EXISTS tenants (
            id INTEGER PRIMARY KEY,
            name TEXT,
            created_at TEXT
        )
    "#).await?;
    
    executor.execute("INSERT INTO tenants VALUES (1, 'ACME Corp', '2026-01-01')").await?;
    executor.execute("INSERT INTO tenants VALUES (2, 'Globex Inc', '2026-01-01')").await?;
    
    // K-V for tenant-specific data
    let alice = TenantUser {
        role: "admin".to_string(),
        email: "alice@acme.com".to_string(),
    };
    client.put(b"tenants/1/users/alice", &serde_json::to_vec(&alice)?).await?;
    
    let bob = TenantUser {
        role: "user".to_string(),
        email: "bob@acme.com".to_string(),
    };
    client.put(b"tenants/1/users/bob", &serde_json::to_vec(&bob)?).await?;
    
    let charlie = TenantUser {
        role: "admin".to_string(),
        email: "charlie@globex.com".to_string(),
    };
    client.put(b"tenants/2/users/charlie", &serde_json::to_vec(&charlie)?).await?;
    
    // Query SQL
    let tenants: Vec<Tenant> = executor.query("SELECT * FROM tenants ORDER BY name").await?;
    
    for tenant in tenants {
        // Scan tenant-specific K-V data
        let prefix = format!("tenants/{}/", tenant.id);
        let end = format!("tenants/{};", tenant.id);
        let users = client.scan(prefix.as_bytes(), end.as_bytes()).await?;
        
        println!("\n{} ({} users):", tenant.name, users.len());
        for (key, value) in users {
            let user: TenantUser = serde_json::from_slice(&value)?;
            println!("  {}: {} ({})", 
                String::from_utf8_lossy(&key),
                user.email,
                user.role
            );
        }
    }
    
    Ok(())
}
```

**Output:**
```
ACME Corp (2 users):
  tenants/1/users/alice: alice@acme.com (admin)
  tenants/1/users/bob: bob@acme.com (user)

Globex Inc (1 users):
  tenants/2/users/charlie: charlie@globex.com (admin)
```

### Example 2: E-commerce with SQL

```rust
use toondb_client::IpcClient;
use toondb_query::SqlExecutor;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CategoryStats {
    category: String,
    order_count: i32,
    revenue: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IpcClient::connect("./ecommerce/toondb.sock").await?;
    let executor = SqlExecutor::new(client);
    
    // Create schema
    executor.execute(r#"
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            category TEXT
        )
    "#).await?;
    
    executor.execute(r#"
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            total REAL
        )
    "#).await?;
    
    // Insert data
    executor.execute("INSERT INTO products VALUES (1, 'Laptop', 999.99, 'Electronics')").await?;
    executor.execute("INSERT INTO products VALUES (2, 'Mouse', 25.00, 'Electronics')").await?;
    executor.execute("INSERT INTO products VALUES (3, 'Desk', 299.99, 'Furniture')").await?;
    
    executor.execute("INSERT INTO orders VALUES (1, 1, 2, 1999.98)").await?;
    executor.execute("INSERT INTO orders VALUES (2, 2, 5, 125.00)").await?;
    
    // Analytics query
    let results: Vec<CategoryStats> = executor.query(r#"
        SELECT 
            products.category,
            COUNT(orders.id) as order_count,
            SUM(orders.total) as revenue
        FROM products
        JOIN orders ON products.id = orders.product_id
        GROUP BY products.category
        ORDER BY revenue DESC
    "#).await?;
    
    println!("Category Performance:");
    for stat in results {
        println!("{}: {} orders, ${:.2}", stat.category, stat.order_count, stat.revenue);
    }
    
    Ok(())
}
```

**Output:**
```
Category Performance:
Electronics: 2 orders, $2124.98
```

### Example 3: Session Cache

```rust
use toondb_client::IpcClient;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
struct Session {
    user_id: String,
    token: String,
    expires_at: u64,
}

struct SessionStore {
    client: IpcClient,
}

impl SessionStore {
    fn new(client: IpcClient) -> Self {
        Self { client }
    }
    
    async fn create(&self, user_id: &str, token: &str, ttl_ms: u64) -> Result<(), Box<dyn std::error::Error>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let session = Session {
            user_id: user_id.to_string(),
            token: token.to_string(),
            expires_at: now + ttl_ms,
        };
        
        let key = format!("sessions/{}", token);
        self.client.put(key.as_bytes(), &serde_json::to_vec(&session)?).await?;
        Ok(())
    }
    
    async fn get(&self, token: &str) -> Result<Option<Session>, Box<dyn std::error::Error>> {
        let key = format!("sessions/{}", token);
        if let Some(value) = self.client.get(key.as_bytes()).await? {
            let session: Session = serde_json::from_slice(&value)?;
            
            let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
            if now > session.expires_at {
                self.delete(token).await?;
                return Ok(None);
            }
            
            return Ok(Some(session));
        }
        Ok(None)
    }
    
    async fn delete(&self, token: &str) -> Result<(), Box<dyn std::error::Error>> {
        let key = format!("sessions/{}", token);
        self.client.delete(key.as_bytes()).await?;
        Ok(())
    }
    
    async fn cleanup(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let results = self.client.scan(b"sessions/", b"sessions;").await?;
        
        let mut removed = 0;
        for (key, value) in results {
            let session: Session = serde_json::from_slice(&value)?;
            if now > session.expires_at {
                self.client.delete(&key).await?;
                removed += 1;
            }
        }
        
        Ok(removed)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IpcClient::connect("./sessions/toondb.sock").await?;
    let store = SessionStore::new(client);
    
    // Create sessions
    store.create("user1", "token123", 60000).await?; // 1 minute
    store.create("user2", "token456", 120000).await?; // 2 minutes
    
    println!("Created 2 sessions");
    
    // Retrieve session
    if let Some(session) = store.get("token123").await? {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let expires_in = (session.expires_at - now) / 1000;
        println!("Session for {}: expires in {}s", session.user_id, expires_in);
    }
    
    // Cleanup expired
    let removed = store.cleanup().await?;
    println!("Cleaned up {} expired sessions", removed);
    
    Ok(())
}
```

**Output:**
```
Created 2 sessions
Session for user1: expires in 60s
Cleaned up 0 expired sessions
```

---

## API Reference

### IpcClient

| Method | Description |
|--------|-------------|
| `IpcClient::connect(path)` | Connect to server |
| `ping()` | Check latency |
| `put(key, value)` | Store key-value |
| `get(key)` | Retrieve value (None if not found) |
| `get_zero_copy(key)` | Zero-copy read ⭐ |
| `delete(key)` | Delete key |
| `put_path(path, value)` | Store by path |
| `get_path(path)` | Get by path |
| `scan(start, end)` | Iterate range ⭐ |
| `begin_transaction()` | Begin transaction |

### Transaction

| Method | Description |
|--------|-------------|
| `put(key, value)` | Store in transaction |
| `get(key)` | Retrieve from transaction |
| `delete(key)` | Delete in transaction |
| `scan(start, end)` | Scan in transaction |
| `commit()` | Commit changes |
| `abort()` | Rollback changes |

### SqlExecutor

| Method | Description |
|--------|-------------|
| `SqlExecutor::new(client)` | Create executor |
| `execute(query)` | Execute DDL/DML |
| `query<T>(query)` | Execute SELECT (returns `Vec<T>`) |

---

## Resources

- [Rust SDK GitHub](https://github.com/toondb/toondb/tree/main/toondb-client)
- [Crates.io](https://crates.io/crates/toondb-client)
- [API Documentation](https://docs.rs/toondb-client/)
- [Go SDK](./go-sdk.md)
- [Python SDK](./python-sdk.md)
- [JavaScript SDK](./nodejs-sdk.md)

---

*Last updated: January 2026 (v0.2.7)*
