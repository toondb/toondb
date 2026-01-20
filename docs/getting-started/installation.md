---
sidebar_position: 2
---

# Installation

Complete installation guide for SochDB across different platforms and use cases.

## Python SDK

```bash
pip install sochdb-client
```

**Pre-built binaries** are available for Linux (x86_64), macOS (Apple Silicon), and Windows (x86_64).

### Verify Installation

```python
from sochdb import Database

db = Database.open("./test_db")
db.put(b"test", b"hello")
value = db.get(b"test")
print(f"SochDB installed! Value: {value.decode()}")
db.close()
```

---

## Node.js / TypeScript SDK

```bash
npm install @sochdb/sochdb
```

### Verify Installation

```typescript
import { SochDatabase } from '@sochdb/sochdb';

const db = new SochDatabase('./test_db');
await db.put('test', 'hello');
const value = await db.get('test');
console.log(`SochDB installed! Value: ${value}`);
await db.close();
```

---

## Go SDK

```bash
go get github.com/sochdb/sochdb-go@v0.3.1
```

### Verify Installation

```go
package main

import (
    "fmt"
    sochdb "github.com/sochdb/sochdb-go"
)

func main() {
    db, _ := sochdb.Open("./test_db")
    defer db.Close()
    
    db.Put([]byte("test"), []byte("hello"))
    value, _ := db.Get([]byte("test"))
    fmt.Printf("SochDB installed! Value: %s\n", value)
}
```

---

## Rust Crate

Add to your `Cargo.toml`:

```toml
[dependencies]
sochdb = "0.2"
```

### Verify Installation

```rust
use sochdb::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("./test_db")?;
    
    db.put(b"test", b"hello")?;
    if let Some(value) = db.get(b"test")? {
        println!("SochDB installed! Value: {}", String::from_utf8_lossy(&value));
    }
    Ok(())
}
```

---

## Build from Source

### Prerequisites

| Requirement | Minimum Version |
|-------------|-----------------|
| Rust | 1.75+ (2024 edition) |
| Git | Any recent |
| C Compiler | GCC 9+ or Clang 11+ |

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/sochdb/sochdb
cd sochdb

# Build release binaries
cargo build --release

# Run tests
cargo test --release

# Install CLI (optional)
cargo install --path sochdb-cli
```

### Build Python Bindings from Source

```bash
cd sochdb-python-sdk

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install build dependencies
pip install maturin

# Build and install
maturin develop --release
```

---

## Platform-Specific Notes

### macOS

On Apple Silicon Macs, ensure you're using a native ARM64 Python:

```bash
# Check architecture
python -c "import platform; print(platform.machine())"
# Should output: arm64
```

### Linux

For best performance, ensure your kernel supports `io_uring`:

```bash
# Check kernel version (5.1+ recommended)
uname -r
```

### Windows

Use PowerShell or Windows Terminal for the best experience:

```powershell
# Install via pip
pip install sochdb-client
```

---

## Verifying Your Installation

Run the diagnostic script to verify everything is working:

```python
from sochdb import Database, VectorIndex
import numpy as np

# Test basic operations
db = Database(":memory:")

with db.transaction() as txn:
    txn.put("test/key", b"value")
    
value = db.get("test/key")
assert value == b"value", "Basic KV operations failed"
print("✓ Key-value operations working")

# Test vector index (if available)
try:
    index = VectorIndex(dimension=128)
    vectors = np.random.rand(10, 128).astype(np.float32)
    index.insert_batch(vectors, list(range(10)))
    
    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, k=5)
    print(f"✓ Vector search working (found {len(results)} results)")
except Exception as e:
    print(f"⚠ Vector search not available: {e}")

print("\n🎉 SochDB is ready to use!")
```

---

## Next Steps

- [Quick Start Guide](/getting-started/quickstart) — Your first SochDB application
- [Python SDK Guide](/guides/python-sdk) — Complete Python tutorial
- [Vector Search](/guides/vector-search) — HNSW indexing guide
