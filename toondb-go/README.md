# ToonDB Go SDK

[![Go Reference](https://pkg.go.dev/badge/github.com/toondb/toondb/toondb-go.svg)](https://pkg.go.dev/github.com/toondb/toondb/toondb-go)
[![CI](https://github.com/sushanthpy/toondb/actions/workflows/go-ci.yml/badge.svg)](https://github.com/sushanthpy/toondb/actions/workflows/go-ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/toondb/toondb/toondb-go)](https://goreportcard.com/report/github.com/toondb/toondb/toondb-go)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The official Go client SDK for **ToonDB** — a high-performance embedded document database with HNSW vector search.

## Architecture

The Go SDK is an **IPC client** that communicates with the ToonDB server via Unix domain sockets:

```
┌─────────────────┐         IPC/Socket         ┌──────────────────┐
│   Go SDK        │ ◄────────────────────────► │  ToonDB Server   │
│  (toondb-go)    │    (toondb.sock)           │  (toondb-mcp)    │
└─────────────────┘                            └──────────────────┘
                                                        │
                                                        ▼
                                                ┌──────────────────┐
                                                │  Storage Engine  │
                                                │  (LSM, WAL, etc) │
                                                └──────────────────┘
```

> **Note:** Unlike embedded databases like SQLite, the Go SDK requires a ToonDB server process to be running. The Python and JavaScript SDKs include automatic server management, but the Go SDK currently requires manual server startup.

## Features

- ✅ **Key-Value Store** — Simple `Get`/`Put`/`Delete` operations
- ✅ **Path-Native API** — Hierarchical keys like `users/alice/email`
- ✅ **Transactions** — ACID-compliant with automatic commit/abort
- ✅ **Query Builder** — Fluent API for prefix scans
- ✅ **Vector Search** — HNSW approximate nearest neighbor search
- ✅ **Type-Safe** — Full Go type safety with generics
- ✅ **Zero CGO** — Pure Go IPC client (no C dependencies)

## Prerequisites

Before using the Go SDK, you must:

1. **Install the ToonDB server binary** (one of these methods):
   
   ```bash
   # Option 1: Download pre-built binaries
   # Download from: https://github.com/toondb/toondb/releases
   
   # Option 2: Build from source (requires Rust)
   cargo build --release -p toondb-mcp
   ```

2. **Start the ToonDB server** before running your Go application:
   
   ```bash
   # Start server for a database directory
   ./target/release/toondb-mcp serve --db ./my_database
   ```

3. **(Optional) For vector search**, also build the toondb-bulk tool:
   
   ```bash
   cargo build --release -p toondb-tools
   ```

## Installation

```bash
go get github.com/toondb/toondb/toondb-go@v0.2.4
```

**Requirements:**
- Go 1.21+
- ToonDB server running (see Prerequisites)

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    // Open database
    db, err := toondb.Open("./my_database")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Key-value operations
    err = db.Put([]byte("user:123"), []byte(`{"name": "Alice"}`))
    if err != nil {
        log.Fatal(err)
    }

    value, err := db.Get([]byte("user:123"))
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(value)) // {"name": "Alice"}

    // Path-native API
    err = db.PutPath("users/alice/email", []byte("alice@example.com"))
    if err != nil {
        log.Fatal(err)
    }

    email, err := db.GetPath("users/alice/email")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(email)) // alice@example.com
}
```

## API Reference

### Database

```go
// Open a database
db, err := toondb.Open("./my_database")

// Open with custom configuration
db, err := toondb.OpenWithConfig(&toondb.Config{
    Path:            "./my_database",
    WALEnabled:      true,
    SyncMode:        "full",
    CreateIfMissing: true,
})

// Key-value operations
err = db.Put(key, value)
value, err := db.Get(key)
err = db.Delete(key)

// String convenience methods
err = db.PutString("key", "value")
value, err := db.GetString("key")

// Path-native operations
err = db.PutPath("users/alice/email", value)
value, err := db.GetPath("users/alice/email")

// Close
err = db.Close()
```

### Transactions

```go
// Recommended: Use WithTransaction for automatic commit/abort
err := db.WithTransaction(func(txn *toondb.Transaction) error {
    if err := txn.Put([]byte("key1"), []byte("value1")); err != nil {
        return err  // Transaction will abort
    }
    if err := txn.Put([]byte("key2"), []byte("value2")); err != nil {
        return err
    }
    return nil  // Transaction will commit
})

// Manual transaction control
txn, err := db.BeginTransaction()
if err != nil {
    log.Fatal(err)
}

err = txn.Put([]byte("key"), []byte("value"))
if err != nil {
    txn.Abort()
    log.Fatal(err)
}

err = txn.Commit()
```

### Query Builder

```go
// Prefix scan with fluent API
results, err := db.Query("users/").
    Limit(10).
    Offset(0).
    Select("name", "email").
    Execute()

for _, kv := range results {
    fmt.Printf("Key: %s, Value: %s\n", kv.Key, kv.Value)
}

// Get first result only
first, err := db.Query("users/").First()

// Check existence
exists, err := db.Query("users/alice/").Exists()

// Count results
count, err := db.Query("users/").Count()

// Iterate with callback
err = db.Query("users/").ForEach(func(kv toondb.KeyValue) error {
    fmt.Println(string(kv.Key))
    return nil
})
```

### Vector Search

```go
// Create vector index
config := &toondb.VectorIndexConfig{
    Dimension:      384,
    Metric:         toondb.Cosine,
    M:              16,
    EfConstruction: 100,
}
index := toondb.NewVectorIndex("./vectors", config)

// Build index from vectors
vectors := [][]float32{
    {0.1, 0.2, 0.3, ...},  // 384-dim embedding
    {0.4, 0.5, 0.6, ...},
}
labels := []string{"doc1", "doc2"}
err := index.BulkBuild(vectors, labels)

// Query nearest neighbors
queryVec := []float32{0.15, 0.25, 0.35, ...}
results, err := index.Query(queryVec, 10, 50)  // k=10, ef_search=50

for _, r := range results {
    fmt.Printf("ID: %d, Distance: %.4f, Label: %s\n",
        r.ID, r.Distance, r.Label)
}

// Get index info
info, err := index.Info()
fmt.Printf("Vectors: %d, Dimension: %d\n", info.NumVectors, info.Dimension)
```

### Utility Functions

```go
// Distance calculations
dist := toondb.ComputeCosineDistance(vecA, vecB)
dist := toondb.ComputeEuclideanDistance(vecA, vecB)

// Vector normalization
normalized := toondb.NormalizeVector(vec)

// Read FVECS format (common benchmark format)
vectors, err := toondb.ReadVectorsFromFVECS("./vectors.fvecs")
```

## Error Handling

```go
import "errors"

value, err := db.Get(key)
if err != nil {
    if errors.Is(err, toondb.ErrClosed) {
        // Database is closed
    }
    // Handle other errors
}

if value == nil {
    // Key not found (not an error)
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `ErrNotFound` | Key was not found |
| `ErrClosed` | Database is closed |
| `ErrTxnCommitted` | Transaction already committed |
| `ErrTxnAborted` | Transaction already aborted |
| `ErrConnectionFailed` | Failed to connect to server |
| `ErrProtocol` | Wire protocol error |
| `ErrVectorDimension` | Vector dimension mismatch |

## Configuration

```go
config := &toondb.Config{
    // Path to database directory (required)
    Path: "./my_database",

    // Create directory if it doesn't exist (default: true)
    CreateIfMissing: true,

    // Enable Write-Ahead Logging (default: true)
    WALEnabled: true,

    // Sync mode: "full", "normal", or "off" (default: "normal")
    SyncMode: "normal",

    // Maximum memtable size before flush (default: 64MB)
    MemtableSizeBytes: 64 * 1024 * 1024,
}
```

## Best Practices

### 1. Always Close the Database

```go
db, err := toondb.Open("./my_database")
if err != nil {
    log.Fatal(err)
}
defer db.Close()  // Always defer close
```

### 2. Use WithTransaction for Atomic Operations

```go
// Good: automatic commit/abort
err := db.WithTransaction(func(txn *toondb.Transaction) error {
    // Operations...
    return nil
})

// Avoid: manual transaction handling (error-prone)
txn, _ := db.BeginTransaction()
// ...
txn.Commit()
```

### 3. Handle nil Values

```go
value, err := db.Get(key)
if err != nil {
    return err
}
if value == nil {
    // Key doesn't exist - this is not an error!
    return fmt.Errorf("key not found: %s", key)
}
```

### 4. Batch Operations in Transactions

```go
// Efficient: batch multiple writes
err := db.WithTransaction(func(txn *toondb.Transaction) error {
    for _, item := range items {
        if err := txn.Put(item.Key, item.Value); err != nil {
            return err
        }
    }
    return nil
})
```

## Testing

```bash
cd toondb-go
go test -v ./...
```

## Benchmarks

```bash
go test -bench=. -benchmem ./...
```

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

## Links

- [ToonDB Documentation](https://toondb.io/docs)
- [Python SDK](../toondb-python-sdk)
- [TypeScript SDK](../toondb-js)
- [Rust Crate](../toondb-client)
- [GitHub Repository](https://github.com/sushanthpy/toondb)
