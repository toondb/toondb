---
sidebar_position: 1
slug: /
title: Introduction
---

# SochDB Documentation

Welcome to the official SochDB documentation. SochDB is **The LLM-Native Database** — a high-performance embedded database designed specifically for AI applications.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **40-66% Fewer Tokens** | TOON format optimized for LLM consumption |
| **Graph Overlay** (v0.3.3) | Lightweight graph layer for agent memory with BFS/DFS traversal |
| **Namespace Isolation** (v0.3.0) | Type-safe multi-tenancy with per-tenant data isolation |
| **Hybrid Search** (v0.3.0) | Vector + BM25 keyword search with RRF fusion |
| **ContextQuery Builder** (v0.3.0+) | Token-aware retrieval with enhanced deduplication (v0.3.3) |
| **Policy Hooks** (v0.3.3) | Agent safety controls with pre-built templates |
| **Tool Routing** (v0.3.3) | Multi-agent coordination with dynamic discovery |
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
cargo add sochdb
```

```rust
use sochdb::Database;

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
pip install sochdb-client
```

```python
from sochdb import Database

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
npm install @sochdb/sochdb
```

```typescript
import { Database } from '@sochdb/sochdb';

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
go get github.com/sochdb/sochdb-go
```

```go
package main

import (
    "fmt"
    sochdb "github.com/sochdb/sochdb-go"
)

func main() {
    db, _ := sochdb.Open("./my_app_db")
    defer db.Close()

    db.WithTransaction(func(txn *sochdb.Transaction) error {
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

**AI Agent Safety:**
- [Policy & Safety Hooks](/guides/policy-hooks) — Pre/post operation validation
- [Multi-Agent Tool Routing](/guides/tool-routing) — Route tools across agents

**Agent Memory & Context:**
- [Graph Overlay](/guides/graph-overlay) — Lightweight graph for agent memory
- [Context Query](/guides/context-query) — Token-aware retrieval for LLMs

### 💡 Concepts
Deep dives into SochDB's architecture and design.

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
Deep technical documentation for SochDB servers and tools.

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
| Enforce agent safety policies | [Policy Hooks](/guides/policy-hooks) |
| Route tools across agents | [Tool Routing](/guides/tool-routing) |
| Model agent memory relationships | [Graph Overlay](/guides/graph-overlay) |
| Build token-aware context | [Context Query](/guides/context-query) |
| Understand the architecture | [Architecture](/concepts/architecture) |
| See the SQL API reference | [SQL API](/api-reference/sql-api) |

---

## External Links

- [**sochdb.dev**](https://sochdb.dev) — Main website
- [**GitHub**](https://github.com/sochdb/sochdb) — Source code
- [**Discussions**](https://github.com/sochdb/sochdb/discussions) — Community Q&A
