---
sidebar_position: 1
---

# Quick Start

Get ToonDB running in 5 minutes.

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| **Rust** | 2024 edition (≥1.75) | `rustc --version` |
| **Python** (optional) | ≥3.9 | `python --version` |
| **Git** | Any recent | `git --version` |

---

## Installation

### Python

```bash
pip install toondb-client
```

### Node.js / TypeScript

```bash
npm install @sushanth/toondb
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
toondb = "0.2"
```

### Go

```bash
go get github.com/toondb/toondb/toondb-go@v0.2.9
```

### Build from Source

```bash
git clone https://github.com/toondb/toondb
cd toondb
cargo build --release
```

---

## Hello World

### Python

```python
from toondb import Database

# Open database (creates automatically)
db = Database.open("./my_first_db")

# Store data
db.put(b"users/alice/name", b"Alice Smith")
db.put(b"users/alice/email", b"alice@example.com")

# Retrieve data
name = db.get(b"users/alice/name")
print(f"Name: {name.decode()}")  # Output: Name: Alice Smith

db.close()
```

### Node.js / TypeScript

```typescript
import { ToonDatabase } from '@sushanth/toondb';

// Open database
const db = new ToonDatabase('./my_first_db');

// Store data
await db.put('users/alice/name', 'Alice Smith');
await db.put('users/alice/email', 'alice@example.com');

// Retrieve data
const name = await db.get('users/alice/name');
console.log(`Name: ${name}`);  // Output: Name: Alice Smith

await db.close();
```

### Go

```go
package main

import (
    "fmt"
    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    // Open database
    db, err := toondb.Open("./my_first_db")
    if err != nil {
        panic(err)
    }
    defer db.Close()

    // Store data
    db.Put([]byte("users/alice/name"), []byte("Alice Smith"))
    db.Put([]byte("users/alice/email"), []byte("alice@example.com"))

    // Retrieve data
    name, _ := db.Get([]byte("users/alice/name"))
    fmt.Printf("Name: %s\n", name)  // Output: Name: Alice Smith
}
```

### Rust

```rust
use toondb::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open database
    let db = Database::open("./my_first_db")?;

    // Store data
    db.put(b"users/alice/name", b"Alice Smith")?;
    db.put(b"users/alice/email", b"alice@example.com")?;

    // Retrieve data
    if let Some(name) = db.get(b"users/alice/name")? {
        println!("Name: {}", String::from_utf8_lossy(&name));
    }

    Ok(())
}
```

---

## Verify Installation

### Python

```bash
python -c "from toondb import Database; print('ToonDB Python SDK installed!')"
```

### Node.js

```bash
node -e "const {ToonDatabase} = require('@sushanth/toondb'); console.log('ToonDB Node.js SDK installed!')"
```

### Go

```bash
go run -e 'package main; import _ "github.com/toondb/toondb/toondb-go"; func main() { println("ToonDB Go SDK installed!") }'
```

### Rust

```bash
cargo build --release && echo "ToonDB Rust SDK installed!"
```

---

## Configuration

ToonDB works out of the box with sensible defaults. For customization:

### Environment Variables

```bash
# Enable debug logging
export RUST_LOG=toondb=debug

# Set default database path
export TOONDB_PATH=./data

# Library path for Python FFI
export TOONDB_LIB_PATH=/path/to/toondb/target/release
```

### Configuration File

Create `toondb.toml` in your database directory:

```toml
[storage]
path = "./data"
sync_mode = "normal"  # "full", "normal", or "off"

[index]
hnsw_m = 16           # HNSW graph connectivity
hnsw_ef = 200         # Construction search width

[server]
socket_path = "/tmp/toondb.sock"
max_connections = 100
```

---

## Next Steps

| Goal | Resource |
|------|----------|
| Build a complete app | [First App Tutorial](/getting-started/first-app) |
| Learn vector search | [Vector Search Tutorial](/guides/vector-search) |
| Use with LLM agents | [MCP Integration](/cookbook/mcp-integration) |
| Understand internals | [Architecture](/concepts/architecture) |
| Contribute | [Contributing Guide](/contributing/style-guide) |

---

## Troubleshooting

### Common Issues

#### Python: `ModuleNotFoundError: No module named 'toondb'`

```bash
pip install --upgrade toondb-client
```

#### Rust: `error: linking with 'cc' failed`

Install build tools:

```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt install build-essential

# Fedora
sudo dnf install gcc
```

#### Permission denied on Unix socket

```bash
chmod 755 /tmp/toondb.sock
```

### Get Help

- [GitHub Issues](https://github.com/toondb/toondb/issues)
- [Documentation Home](/)
- [API Reference](/api-reference/python-api)

