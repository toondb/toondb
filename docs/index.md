---
sidebar_position: 1
slug: /
title: Introduction
---

# ToonDB Documentation

Welcome to the official ToonDB documentation. ToonDB is **The LLM-Native Database** — a high-performance embedded database designed specifically for AI applications.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **40-66% Fewer Tokens** | TOON format optimized for LLM consumption |
| **Blazing Fast** | Rust-powered with zero-copy and SIMD |
| **Vector Search** | Built-in HNSW indexing for embeddings |
| **Embeddable** | In-process or client-server mode |
| **Multi-Language** | Native SDKs for Rust, Python, Node.js, Go |
| **MCP Ready** | Seamless Claude/LLM agent integration |

---

## Quick Install

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="rust" label="Rust" default>

```bash
cargo add toondb
```

```rust
use toondb::Database;

fn main() -> anyhow::Result<()> {
    let db = Database::open("./my_app_db")?;

    db.with_transaction(|txn| {
        txn.put(b"users/alice", br#"{"name": "Alice", "role": "admin"}"#)?;
        Ok(())
    })?;

    if let Some(user) = db.get(b"users/alice")? {
        println!("{}", String::from_utf8_lossy(&user));
    }

    Ok(())
}
```

</TabItem>
<TabItem value="python" label="Python">

```bash
pip install toondb-client
```

```python
from toondb import Database

db = Database.open("./my_app_db")

with db.transaction() as txn:
    txn.put(b"users/alice", b'{"name": "Alice", "role": "admin"}')

user = db.get(b"users/alice")
print(user.decode())  # {"name": "Alice", "role": "admin"}

db.close()
```

</TabItem>
<TabItem value="nodejs" label="Node.js">

```bash
npm install @sushanth/toondb
```

```typescript
import { Database } from '@sushanth/toondb';

const db = await Database.open('./my_app_db');

await db.withTransaction(async (txn) => {
  await txn.put('users/alice', '{"name": "Alice", "role": "admin"}');
});

const user = await db.get('users/alice');
console.log(user?.toString());

await db.close();
```

</TabItem>
<TabItem value="go" label="Go">

```bash
go get github.com/toondb/toondb/toondb-go
```

```go
package main

import (
    "fmt"
    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, _ := toondb.Open("./my_app_db")
    defer db.Close()

    db.WithTransaction(func(txn *toondb.Transaction) error {
        return txn.Put("users/alice", []byte(`{"name": "Alice", "role": "admin"}`))
    })

    user, _ := db.Get("users/alice")
    fmt.Println(string(user))
}
```

</TabItem>
</Tabs>

→ [Full Quick Start Guide](/getting-started/quickstart)

---

## Documentation Sections

### 🚀 Getting Started
Step-by-step guides to get you up and running quickly.

- [Quick Start](/getting-started/quickstart) — 5-minute intro
- [Installation](/getting-started/installation) — All platforms
- [First App](/getting-started/first-app) — Build something real

### 📖 Guides
Task-oriented guides for specific use cases.

**Language SDKs:**
- [Rust SDK](/guides/rust-sdk) — Native Rust guide
- [Python SDK](/guides/python-sdk) — Complete Python guide
- [Node.js SDK](/guides/nodejs-sdk) — TypeScript/JavaScript guide
- [Go SDK](/guides/go-sdk) — Go client guide

**Features:**
- [SQL Guide](/guides/sql-guide) — Working with SQL queries
- [Vector Search](/guides/vector-search) — HNSW indexing
- [Bulk Operations](/guides/bulk-operations) — Batch processing
- [Deployment](/guides/deployment) — Production setup

### 💡 Concepts
Deep dives into ToonDB's architecture and design.

- [Architecture](/concepts/architecture) — System design
- [TOON Format](/concepts/toon-format) — Token-optimized format
- [Performance](/concepts/performance) — Optimization guide

### 📋 API Reference
Complete technical specifications.

- [SQL API](/api-reference/sql-api) — SQL query reference
- [Rust API](/api-reference/rust-api) — Crate documentation
- [Python API](/api-reference/python-api) — Full Python API docs
- [Node.js API](/api-reference/nodejs-api) — TypeScript/JavaScript API
- [Go API](/api-reference/go-api) — Go package documentation

### 🛠️ Server Reference
Deep technical documentation for ToonDB servers and tools.

- [IPC Server](/servers/IPC_SERVER.md) — Wire protocol & architecture
- [gRPC Server](/servers/GRPC_SERVER.md) — Vector search service
- [Bulk Operations](/servers/BULK_OPERATIONS.md) — High-performance tools

### 🍳 Cookbook
Recipes for common tasks.

- [Vector Indexing](/cookbook/vector-indexing) — Embedding workflows
- [MCP Integration](/cookbook/mcp-integration) — Claude integration
- [Logging](/cookbook/logging) — Observability setup

---

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started in 5 minutes | [Quick Start](/getting-started/quickstart) |
| Use SQL queries | [SQL Guide](/guides/sql-guide) |
| Use the Rust SDK | [Rust Guide](/guides/rust-sdk) |
| Use the Python SDK | [Python Guide](/guides/python-sdk) |
| Use the Node.js SDK | [Node.js Guide](/guides/nodejs-sdk) |
| Use the Go SDK | [Go Guide](/guides/go-sdk) |
| Add vector search | [Vector Search](/guides/vector-search) |
| Understand the architecture | [Architecture](/concepts/architecture) |
| See the SQL API reference | [SQL API](/api-reference/sql-api) |

---

## External Links

- [**toondb.dev**](https://toondb.dev) — Main website
- [**GitHub**](https://github.com/toondb/toondb) — Source code
- [**Discussions**](https://github.com/toondb/toondb/discussions) — Community Q&A
