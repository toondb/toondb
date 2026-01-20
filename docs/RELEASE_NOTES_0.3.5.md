# SochDB v0.3.5 Release Notes

**Release Date**: January 2025  
**Focus**: Sync-First Architecture, Enhanced SQL Support, SDK Improvements

---

## Overview

SochDB v0.3.5 represents a significant architectural evolution, moving to a **sync-first core** design that makes the async runtime (tokio) truly optional. This release follows SQLite's proven design pattern: synchronous storage core with async only at the edges.

## Major Changes

### 1. Sync-First Architecture

**What Changed:**
- Removed `tokio` from workspace-level dependencies
- Made async runtime opt-in via `--features async` flag
- Core storage, MVCC, WAL, and indexes are now fully synchronous
- Async only required for gRPC server and async client APIs

**Benefits:**
```
┌──────────────────────────────────────────────────┐
│ Binary Size Reduction                             │
├──────────────────────────────────────────────────┤
│ sochdb-storage (default):      732 KB            │
│ sochdb-storage (with async): 1,200 KB            │
│ Savings: ~500 KB (40% reduction)                 │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Dependency Reduction                              │
├──────────────────────────────────────────────────┤
│ Default build:     ~60 dependencies              │
│ With async:       ~100 dependencies              │
│ Reduction: ~40 fewer dependencies                │
└──────────────────────────────────────────────────┘
```

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────┐
│             Application Layer                        │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐          ┌──────────────────┐
│ Embedded FFI │          │ gRPC Server      │
│ (sync only)  │          │ (needs tokio)    │
└──────┬───────┘          └────────┬─────────┘
       │                           │
       └──────────┬────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │    SochDB Core (Sync)       │
    │  ✓ Storage Engine           │
    │  ✓ MVCC + WAL               │
    │  ✓ SQL Query Engine         │
    │  ✓ Vector Index (HNSW)      │
    │  ✓ No tokio dependency      │
    └─────────────────────────────┘
```

**Use Cases:**
- **Embedded databases** in WASM/mobile apps → smaller binaries
- **FFI boundaries** (Python, Node.js) → no async runtime mismatch
- **Sync-only applications** → no unnecessary dependencies
- **Edge computing** → minimal footprint

**Crate-Level Changes:**

| Crate               | tokio Status       | Binary Size (approx) |
|---------------------|-------------------|---------------------|
| `sochdb-storage`    | Optional          | 732 KB (no tokio)   |
| `sochdb-core`       | Not included      | 450 KB              |
| `sochdb-query`      | Not included      | 380 KB              |
| `sochdb-index`      | Not included      | 2.1 MB              |
| `sochdb-kernel`     | Optional          | 1.8 MB              |
| `sochdb-grpc`       | Required          | 3.5 MB              |

### 2. Enhanced SQL Support (AST-Based Executor)

**New Features:**

#### a) Multi-Dialect Support
SochDB now understands syntax from multiple SQL dialects and normalizes them internally:

```sql
-- PostgreSQL syntax
SELECT * FROM users WHERE name ILIKE 'alice%';

-- MySQL syntax
SELECT * FROM users WHERE name LIKE 'alice%';

-- SQLite syntax
SELECT * FROM users WHERE name GLOB 'alice*';
```

**Supported Dialects:**
- PostgreSQL (primary)
- MySQL
- SQLite
- Generic SQL-92

**Implementation:**
- Unified AST representation using `sqlparser` crate
- Dialect-specific parser selection
- AST normalization layer
- Consistent execution regardless of input dialect

#### b) Idempotent DDL

```sql
-- Safe to run multiple times
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);

-- Won't fail if table doesn't exist
DROP TABLE IF EXISTS old_users;

-- Check existence before altering
ALTER TABLE IF EXISTS users ADD COLUMN age INTEGER;
```

**Benefits:**
- Migration scripts can be run repeatedly
- No manual existence checks required
- Better compatibility with ORMs

#### c) Improved Error Messages

**Before (v0.3.4):**
```
Error: syntax error
```

**After (v0.3.5):**
```
SQL Syntax Error at position 25:
  SELECT * FROM users WHER name = 'alice'
                      ^^^^
  Unexpected token 'WHER', did you mean 'WHERE'?
```

**Error Categories:**
1. **Syntax errors**: Position, expected tokens, suggestions
2. **Semantic errors**: Type mismatches, undefined columns
3. **Execution errors**: Constraint violations, deadlocks

### 3. Python SDK Enhancements

#### Vector Index Convenience Methods

**Problem**: Users had to manage separate `VectorIndex` objects, leading to verbose code:

```python
# Old way (v0.3.4)
from sochdb import Database, VectorIndex

db = Database.open("./my_db")
index = VectorIndex("./vectors/embeddings", dimension=384, metric="cosine")
index.add("doc1", embedding1)
index.build()
results = index.search(query, k=10)
```

**Solution**: Integrate vector operations into `Database` class:

```python
# New way (v0.3.5)
from sochdb import Database

db = Database.open("./my_db")

# Create index
db.create_index("embeddings", dimension=384, max_connections=16, ef_construction=200)

# Bulk insert
ids = ["doc1", "doc2", "doc3"]
vectors = [emb1, emb2, emb3]
db.insert_vectors("embeddings", ids, vectors)

# Search
results = db.search("embeddings", query, k=10)
```

**New Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_index()` | `(name: str, dimension: int, max_connections: int, ef_construction: int)` | Create named vector index |
| `insert_vectors()` | `(index_name: str, ids: List[str], vectors: List[List[float]])` | Bulk insert vectors |
| `search()` | `(index_name: str, query: List[float], k: int) -> List[Dict]` | Search k-nearest neighbors |

**Implementation Details:**
- Index metadata stored in SochDB's internal tables
- Automatic index loading on database open
- Reference counting for shared indexes
- Thread-safe access via Rust mutex

**Migration:**
```python
# Your old code still works!
index = VectorIndex("./vectors/embeddings", dimension=384)

# But you can now simplify to:
db.create_index("embeddings", dimension=384)
```

### 4. Node.js SDK Graph Overlay

Full TypeScript support for graph operations previously only available in Python:

```typescript
import { Database } from '@sochdb/sochdb';

const db = await Database.open('./my_db');

// Add nodes
await db.addNode('user_1', { type: 'user', name: 'Alice' });
await db.addNode('user_2', { type: 'user', name: 'Bob' });
await db.addNode('proj_1', { type: 'project', name: 'SochDB' });

// Add relationships
await db.addEdge('user_1', 'proj_1', { role: 'maintainer' });
await db.addEdge('user_2', 'proj_1', { role: 'contributor' });

// Traverse (BFS)
const path = await db.traverse('user_1', 'user_2', { 
    algorithm: 'bfs',
    maxDepth: 5 
});

// Get neighbors with filter
const maintainers = await db.getNeighbors('proj_1', {
    edgeFilter: { role: 'maintainer' }
});

await db.close();
```

**Graph Operations:**

| Method | Description |
|--------|-------------|
| `addNode(id, attributes)` | Add node with metadata |
| `addEdge(from, to, attributes)` | Add directed edge |
| `removeNode(id)` | Remove node and incident edges |
| `removeEdge(from, to)` | Remove specific edge |
| `getNode(id)` | Get node attributes |
| `getNeighbors(id, filter?)` | Get adjacent nodes |
| `traverse(start, end, opts)` | BFS/DFS pathfinding |

---

## Performance Improvements

### 1. Reduced Binary Size

| Configuration | Binary Size | Change |
|---------------|-------------|--------|
| v0.3.4 (with tokio) | 1.2 MB | baseline |
| v0.3.5 (default) | 732 KB | **-40%** |
| v0.3.5 (with async) | 1.2 MB | 0% |

### 2. Dependency Tree

```bash
# v0.3.4
cargo tree -p sochdb-storage | wc -l
# Output: 102 crates

# v0.3.5 (default)
cargo tree -p sochdb-storage --no-default-features | wc -l  
# Output: 62 crates (-40%)

# v0.3.5 (with async)
cargo tree -p sochdb-storage --features async | wc -l
# Output: 102 crates
```

### 3. Compilation Time

| Configuration | Clean Build | Incremental |
|---------------|-------------|-------------|
| v0.3.4 | 145s | 12s |
| v0.3.5 (default) | **98s** | **8s** |
| v0.3.5 (async) | 145s | 12s |

---

## Migration Guide

### Rust Projects

#### If You Don't Use Async

**No changes needed!** Your code will automatically benefit from smaller binaries:

```toml
# Cargo.toml
[dependencies]
sochdb = "0.3.5"
```

```rust
// Your code stays the same
use sochdb::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("./my_db")?;
    db.put(b"key", b"value")?;
    Ok(())
}
```

#### If You Use Async Features

Enable the `async` feature:

```toml
# Cargo.toml
[dependencies]
sochdb = { version = "0.3.5", features = ["async"] }
```

```rust
// Async code works as before
use sochdb::Database;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("./my_db")?;
    db.put_async(b"key", b"value").await?;
    Ok(())
}
```

### Python SDK

#### Using Vector Index

**Old API (still works):**
```python
from sochdb import VectorIndex

index = VectorIndex("./vectors", dimension=384)
index.add("doc1", embedding)
```

**New API (recommended):**
```python
from sochdb import Database

db = Database.open("./my_db")
db.create_index("vectors", dimension=384)
db.insert_vectors("vectors", ["doc1"], [embedding])
```

**Migration Steps:**
1. Replace `VectorIndex()` with `db.create_index()`
2. Replace `index.add()` with `db.insert_vectors()`
3. Replace `index.search()` with `db.search()`
4. Remove separate index object management

### Node.js SDK

#### Graph Overlay

**New in v0.3.5:** Graph operations now available in TypeScript/JavaScript:

```typescript
// Previously: only available in Python
// Now: full TypeScript support

import { Database } from '@sochdb/sochdb';

const db = await Database.open('./my_db');
await db.addNode('node1', { attr: 'value' });
await db.addEdge('node1', 'node2', { rel: 'knows' });
```

---

## Testing

All changes verified with comprehensive test suite:

```bash
# Total tests
cargo test --workspace
# Result: 1,697 tests passed

# Storage without async
cargo test -p sochdb-storage --no-default-features
# Result: 636 tests passed

# Storage with async
cargo test -p sochdb-storage --features async
# Result: 636 tests passed

# Python SDK
cd sochdb-python-sdk && python -m pytest
# Result: 156 tests passed

# Node.js SDK
cd sochdb-nodejs-sdk && npm test
# Result: 89 tests passed
```

**Test Coverage:**
- ✅ Sync-first storage operations
- ✅ Async feature flag behavior
- ✅ Python vector index convenience methods
- ✅ Node.js graph overlay operations
- ✅ SQL dialect compatibility
- ✅ Idempotent DDL operations
- ✅ Error message formatting

---

## Breaking Changes

### None! 🎉

This release is **fully backward compatible**:
- Old APIs continue to work
- Async feature is opt-in
- Existing databases open without migration
- Python/Node.js SDKs maintain compatibility

---

## Known Issues

1. **Windows Support**: Async features on Windows may require additional runtime setup
2. **WASM**: Async features not yet tested in WASM environments
3. **Python 3.8**: Vector convenience methods require Python 3.9+ (type hints)

---

## Deprecation Notices

None in this release. All existing APIs remain supported.

---

## Future Roadmap (v0.3.6+)

- **Streaming queries**: Async iterator support for large result sets
- **Connection pooling**: Optional async connection pool
- **Distributed transactions**: Two-phase commit for multi-node setups
- **WASM optimization**: Further size reduction for web deployments

---

## Credits

**Contributors:**
- Core team: Sync-first architecture design and implementation
- Community: SQL dialect testing, bug reports

**Inspirations:**
- SQLite: Sync-first design philosophy
- DuckDB: Embedded analytics architecture
- Polars: Zero-copy columnar operations

---

## Resources

- **Documentation**: [https://sochdb.dev](https://sochdb.dev)
- **GitHub**: [https://github.com/sochdb/sochdb](https://github.com/sochdb/sochdb)
- **Benchmarks**: [sochdb-benchmarks](https://github.com/sochdb/sochdb-benchmarks)
- **Discord**: [Community chat](https://discord.gg/sochdb)

---

## Feedback

We'd love to hear about your experience with v0.3.5:
- Open issues on [GitHub](https://github.com/sochdb/sochdb/issues)
- Join our [Discord](https://discord.gg/sochdb)
- Email: team@sochdb.dev
