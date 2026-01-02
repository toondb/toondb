# ToonDB Python SDK Documentation

A comprehensive Python client SDK for **ToonDB** - the database optimized for LLM context retrieval.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Embedded Mode (FFI)](#embedded-mode-ffi)
4. [IPC Client Mode](#ipc-client-mode)
5. [Advanced Features](#advanced-features)
   - [TOON Format](#toon-format-token-optimized-output-notation)
   - [High-Performance Batched Scanning](#high-performance-batched-scanning)
   - [Database Statistics & Monitoring](#database-statistics--monitoring)
   - [Manual Checkpoint](#manual-checkpoint)
   - [Python Plugin System](#python-plugin-system)
   - [Transaction Advanced Features](#transaction-advanced-features)
   - [IPC Server & Multi-Process Access](#ipc-server--multi-process-access)
   - [CLI Tools](#cli-tools)
6. [API Reference](#api-reference)
7. [Use Cases](#use-cases)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI

```bash
pip install toondb-client
```

### From Source

```bash
cd toondb-python-sdk
pip install -e .
```

### Native Library (Required for Embedded Mode)

For embedded mode, build the Rust native library:

```bash
cd /path/to/toon_database
cargo build --release
```

Set the library path:

```bash
export TOONDB_LIB_PATH=/path/to/toon_database/target/release
```

---

## Quick Start

### Embedded Mode (Single Process)

```python
from toondb import Database

# Open database (creates if not exists)
db = Database.open("./my_database")

# Store and retrieve data
db.put(b"user:1", b'{"name": "Alice"}')
value = db.get(b"user:1")
print(value)  # b'{"name": "Alice"}'

# Clean up
db.close()
```

### With Context Manager

```python
from toondb import Database

with Database.open("./my_database") as db:
    db.put(b"key", b"value")
    value = db.get(b"key")
# Database automatically closed
```

---

## Embedded Mode (FFI)

The embedded mode provides direct access to ToonDB via FFI to the Rust library. This is the recommended mode for single-process applications.

### Opening a Database

```python
from toondb import Database

# Basic open
db = Database.open("./data")

# Context manager (recommended)
with Database.open("./data") as db:
    # operations here
    pass
```

### Key-Value Operations

```python
# Put (create/update)
db.put(b"key", b"value")

# Get (returns None if not found)
value = db.get(b"key")

# Delete
db.delete(b"key")
```

### Path-Native API

ToonDB supports hierarchical data organization using paths:

```python
# Store at path
db.put_path("users/alice/email", b"alice@example.com")
db.put_path("config/app/theme", b"dark")

# Retrieve by path
email = db.get_path("users/alice/email")
```

### Transactions

Transactions provide ACID guarantees:

```python
# Auto-commit with context manager
with db.transaction() as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    # Automatically commits on success
    # Automatically aborts on exception

# Manual control
txn = db.transaction()
txn.put(b"key", b"value")
txn.commit()  # or txn.abort()
```

### Range Scans

```python
# Scan all keys
for key, value in db.scan():
    print(key, value)

# Scan range [start, end)
for key, value in db.scan(b"user:", b"user;"):
    print(key, value)

# Prefix scan pattern
prefix = b"log:2024-01-15:"
end = prefix[:-1] + bytes([prefix[-1] + 1])
for key, value in db.scan(prefix, end):
    print(key, value)
```

### Administrative Operations

```python
# Force checkpoint to disk
lsn = db.checkpoint()

# Get storage statistics
stats = db.stats()
print(stats)
# {'memtable_size_bytes': 1024, 'wal_size_bytes': 4096, ...}
```

---

## IPC Client Mode

IPC mode allows multi-process access to ToonDB via Unix domain sockets.

### Connecting

```python
from toondb import IpcClient

client = IpcClient.connect("/tmp/toondb.sock", timeout=30.0)
```

### Basic Operations

```python
# Ping (returns latency in seconds)
latency = client.ping()

# Put/Get/Delete - same as embedded
client.put(b"key", b"value")
value = client.get(b"key")
client.delete(b"key")
```

### Path API

```python
# Note: IPC uses list of path segments
client.put_path(["users", "alice", "email"], b"alice@example.com")
email = client.get_path(["users", "alice", "email"])
```

### Transactions

```python
# Begin transaction
txn_id = client.begin_transaction()

# Perform operations
# (Note: IPC operations don't use txn_id directly yet)

# Commit or abort
commit_ts = client.commit(txn_id)
# or: client.abort(txn_id)
```

### Query Builder

```python
from toondb import Query

# Fluent query interface
results = client.query("users/") \
    .limit(10) \
    .offset(0) \
    .select(["name", "email"]) \
    .execute()  # Returns TOON string

# Parse to list of dicts
results_list = client.query("users/") \
    .limit(10) \
    .to_list()
```

### Scan

```python
# Scan with prefix
results = client.scan("users/")
# Returns: [{"key": b"users/1", "value": b"..."}, ...]
```

---

## Bulk Vector Operations

The Bulk API provides high-throughput vector operations by bypassing Python FFI overhead.
Instead of crossing the Python/Rust boundary for each vector, it:

1. Writes vectors to a memory-mapped file
2. Spawns the native `toondb-bulk` binary as a subprocess
3. Returns results via stdout/file

### Why Bulk Operations?

| Method | 768D Throughput | Overhead |
|--------|-----------------|----------|
| Python FFI | ~130 vec/s | 12× slower |
| Bulk API | ~1,600 vec/s | 1.0× baseline |

### Building an Index

```python
from toondb.bulk import bulk_build_index
import numpy as np

# Your embeddings (N × D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,                    # HNSW max connections
    ef_construction=100,     # Construction search depth
    threads=0,               # 0 = auto
    quiet=False,             # Show progress
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

### Querying an Index

```python
from toondb.bulk import bulk_query_index
import numpy as np

# Single query
query = np.random.randn(768).astype(np.float32)
results = bulk_query_index(
    index="my_index.hnsw",
    query=query,
    k=10,
    ef_search=64,
)

for neighbor in results:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
```

### Binary Resolution

The SDK automatically finds the `toondb-bulk` binary:

```python
from toondb.bulk import get_toondb_bulk_path

# Returns path to bundled or installed binary
path = get_toondb_bulk_path()
print(f"Using binary: {path}")
```

Resolution order:
1. **Bundled in wheel**: `_bin/<platform>/toondb-bulk`
2. **System PATH**: `which toondb-bulk`
3. **Cargo target**: `../target/release/toondb-bulk` (development)

### Bulk API Reference

| Function | Description |
|----------|-------------|
| `bulk_build_index(embeddings, output, ...)` | Build HNSW index from numpy array |
| `bulk_query_index(index, query, k, ...)` | Query HNSW index for k nearest neighbors |
| `bulk_info(index)` | Get index metadata |
| `convert_embeddings_to_raw(embeddings, path)` | Convert to raw f32 format |
| `get_toondb_bulk_path()` | Get path to toondb-bulk binary |

### BulkBuildStats

```python
@dataclass
class BulkBuildStats:
    vectors: int          # Total vectors indexed
    dimension: int        # Vector dimension
    elapsed_seconds: float
    rate: float           # vec/s
    index_size_bytes: int
```

---

## Advanced Features

### TOON Format (Token-Optimized Output Notation)

TOON is a columnar text format designed for LLM context efficiency, achieving **40-66% token reduction** compared to JSON.

#### Why TOON?

When passing database results to LLMs, JSON's verbosity wastes tokens:

```json
[
  {"name": "Alice", "email": "alice@example.com"},
  {"name": "Bob", "email": "bob@example.com"}
]
```
**JSON tokens**: ~165 tokens (pretty), ~140 tokens (compact)

```
users[2]{name,email}:Alice,alice@example.com;Bob,bob@example.com
```
**TOON tokens**: ~70 tokens (**59% reduction!**)

#### Converting to TOON

```python
from toondb import Database

records = [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
]

# Convert all fields
toon_str = Database.to_toon("users", records)
print(toon_str)
# users[2]{id,name,email,age}:1,Alice,alice@example.com,30;2,Bob,bob@example.com,25

# Convert specific fields only
toon_str = Database.to_toon("users", records, ["name", "email"])
print(toon_str)
# users[2]{name,email}:Alice,alice@example.com;Bob,bob@example.com
```

#### Parsing from TOON

```python
toon_str = "users[2]{name,email}:Alice,alice@ex.com;Bob,bob@ex.com"

table_name, fields, records = Database.from_toon(toon_str)

print(table_name)  # "users"
print(fields)      # ["name", "email"]
print(records)     
# [
#   {"name": "Alice", "email": "alice@ex.com"},
#   {"name": "Bob", "email": "bob@ex.com"}
# ]
```

#### Use Case: RAG with LLMs

```python
from toondb import Database
import openai

with Database.open("./knowledge_base") as db:
    # Query relevant documents
    results = db.execute_sql("""
        SELECT title, content, relevance_score 
        FROM documents 
        WHERE category = 'technical'
        ORDER BY relevance_score DESC
        LIMIT 10
    """)
    
    # Convert to TOON for efficient context
    records = [dict(row) for row in results]
    toon_context = Database.to_toon("documents", records, ["title", "content"])
    
    # Send to LLM (saves tokens!)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context:\n{toon_context}"},
            {"role": "user", "content": "Summarize the technical documents"}
        ]
    )
```

### High-Performance Batched Scanning

The `scan_batched()` method dramatically reduces FFI overhead by fetching multiple results per call.

#### The Problem with Regular Scan

```python
# Regular scan: 1 FFI call per result
for key, value in txn.scan(b"prefix:", b"prefix;"):
    process(key, value)
# 10,000 results = 10,000 FFI calls
# At 500ns per call = 5ms overhead
```

#### The Solution: Batched Scan

```python
# Batched scan: 1 FFI call per batch
for key, value in txn.scan_batched(b"prefix:", b"prefix;", batch_size=1000):
    process(key, value)
# 10,000 results = 10 FFI calls
# At 500ns per call = 5µs overhead (1000× faster!)
```

#### Complete Example

```python
from toondb import Database
import time

with Database.open("./my_db") as db:
    # Insert 100K test records
    print("Inserting 100K records...")
    with db.transaction() as txn:
        for i in range(100000):
            txn.put(f"user:{i:06d}".encode(), f"data_{i}".encode())
    
    # Benchmark regular scan
    start = time.time()
    txn = db.transaction()
    count = sum(1 for _ in txn.scan(b"user:", b"user;"))
    regular_time = time.time() - start
    txn.abort()
    print(f"Regular scan: {count} items in {regular_time:.3f}s")
    
    # Benchmark batched scan
    start = time.time()
    txn = db.transaction()
    count = sum(1 for _ in txn.scan_batched(b"user:", b"user;", batch_size=1000))
    batched_time = time.time() - start
    txn.abort()
    print(f"Batched scan: {count} items in {batched_time:.3f}s")
    
    speedup = regular_time / batched_time
    print(f"Speedup: {speedup:.1f}×")
```

**Performance Comparison:**

| Dataset Size | Regular Scan | Batched Scan | Speedup |
|--------------|--------------|--------------|---------|
| 10K items | 15ms | 2ms | 7.5× |
| 100K items | 150ms | 12ms | 12.5× |
| 1M items | 1.5s | 120ms | 12.5× |

**Best Practices:**
- Use `batch_size=1000` for most workloads
- Increase to `batch_size=5000` for very large scans
- Decrease to `batch_size=100` if processing each item is expensive

### Database Statistics & Monitoring

Monitor database performance and health with runtime statistics.

```python
from toondb import Database

with Database.open("./my_db") as db:
    # Perform operations
    for i in range(1000):
        db.put(f"key:{i}".encode(), f"value:{i}".encode())
    
    # Get comprehensive statistics
    stats = db.stats()
    
    # Storage metrics
    print(f"Total keys: {stats.get('keys_count', 0):,}")
    print(f"Bytes written: {stats.get('bytes_written', 0):,}")
    print(f"Bytes read: {stats.get('bytes_read', 0):,}")
    
    # Transaction metrics
    print(f"Transactions committed: {stats.get('transactions_committed', 0)}")
    print(f"Transactions aborted: {stats.get('transactions_aborted', 0)}")
    
    # Query metrics
    print(f"Queries executed: {stats.get('queries_executed', 0)}")
    
    # Cache metrics
    hits = stats.get('cache_hits', 0)
    misses = stats.get('cache_misses', 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0
    print(f"Cache hit rate: {hit_rate:.1f}%")
```

**Available Statistics:**

| Metric | Type | Description |
|--------|------|-------------|
| `keys_count` | int | Total number of keys in database |
| `bytes_written` | int | Cumulative bytes written |
| `bytes_read` | int | Cumulative bytes read |
| `transactions_committed` | int | Number of successful transactions |
| `transactions_aborted` | int | Number of aborted transactions |
| `queries_executed` | int | Number of SQL queries executed |
| `cache_hits` | int | Number of cache hits |
| `cache_misses` | int | Number of cache misses |

**Monitoring Example:**

```python
import time
from toondb import Database

def monitor_database(db_path: str, interval: int = 5):
    """Monitor database statistics every N seconds."""
    with Database.open(db_path) as db:
        prev_stats = db.stats()
        
        while True:
            time.sleep(interval)
            curr_stats = db.stats()
            
            # Calculate deltas
            writes = curr_stats['bytes_written'] - prev_stats['bytes_written']
            reads = curr_stats['bytes_read'] - prev_stats['bytes_read']
            
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Writes: {writes:,} bytes/s, "
                  f"Reads: {reads:,} bytes/s")
            
            prev_stats = curr_stats

# Run monitor
monitor_database("./my_db", interval=5)
```

### Manual Checkpoint

Force a checkpoint to ensure all in-memory data is flushed to disk.

```python
from toondb import Database

with Database.open("./my_db") as db:
    # Bulk import
    print("Importing 100K records...")
    with db.transaction() as txn:
        for i in range(100000):
            txn.put(f"bulk:{i}".encode(), f"data:{i}".encode())
    
    # Force checkpoint for durability
    print("Checkpointing...")
    lsn = db.checkpoint()
    print(f"Checkpoint complete at LSN {lsn}")
    print("All data is now durable on disk!")
```

**When to Use Checkpoints:**

1. **Before Backups**
   ```python
   db.checkpoint()
   os.system("tar -czf backup.tar.gz ./my_db")
   ```

2. **After Bulk Imports**
   ```python
   # Import 1M records
   for batch in batches:
       with db.transaction() as txn:
           for record in batch:
               txn.put(record.key, record.value)
   
   # Ensure durability
   db.checkpoint()
   ```

3. **Before System Shutdown**
   ```python
   import signal
   
   def shutdown_handler(signum, frame):
       db.checkpoint()
       db.close()
       sys.exit(0)
   
   signal.signal(signal.SIGTERM, shutdown_handler)
   ```

4. **Periodic Durability**
   ```python
   import threading
   
   def periodic_checkpoint(db, interval=300):
       while True:
           time.sleep(interval)
           db.checkpoint()
   
   # Checkpoint every 5 minutes
   thread = threading.Thread(target=periodic_checkpoint, args=(db,))
   thread.daemon = True
   thread.start()
   ```

### Python Plugin System

Run Python code as database triggers with full package support.

#### Basic Plugin

```python
from toondb.plugins import PythonPlugin, PluginRegistry, TriggerEvent

# Define a simple validation plugin
plugin = PythonPlugin(
    name="email_validator",
    code='''
def on_before_insert(row: dict) -> dict:
    """Validate and normalize email addresses."""
    if "email" in row:
        email = row["email"].strip().lower()
        if "@" not in email:
            raise TriggerAbort("Invalid email format", code="INVALID_EMAIL")
        row["email"] = email
    return row
''',
    triggers={"users": ["BEFORE INSERT"]}
)

# Register plugin
registry = PluginRegistry()
registry.register(plugin)

# Test trigger
row = {"name": "Alice", "email": "  ALICE@EXAMPLE.COM  "}
result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
print(result["email"])  # "alice@example.com"
```

#### Advanced Plugin with Packages

```python
from toondb.plugins import PythonPlugin, TriggerAbort

# ML-powered fraud detection
fraud_detector = PythonPlugin(
    name="fraud_detector",
    code='''
import numpy as np

def on_before_insert(row: dict) -> dict:
    """Detect fraudulent transactions using ML."""
    amount = row.get("amount", 0)
    
    # Simple heuristic (replace with real ML model)
    if amount > 10000:
        raise TriggerAbort(
            f"Transaction amount ${amount} exceeds limit",
            code="FRAUD_DETECTED"
        )
    
    # Add fraud score
    row["fraud_score"] = min(amount / 10000, 1.0)
    
    return row
''',
    version="1.0.0",
    packages=["numpy"],
    triggers={"transactions": ["BEFORE INSERT"]}
)

registry.register(fraud_detector)
```

#### Available Trigger Events

```python
from toondb.plugins import TriggerEvent

# Row-level triggers
TriggerEvent.BEFORE_INSERT  # Before inserting a row
TriggerEvent.AFTER_INSERT   # After inserting a row
TriggerEvent.BEFORE_UPDATE  # Before updating a row
TriggerEvent.AFTER_UPDATE   # After updating a row
TriggerEvent.BEFORE_DELETE  # Before deleting a row
TriggerEvent.AFTER_DELETE   # After deleting a row

# Batch triggers
TriggerEvent.ON_BATCH       # On batch operations
```

#### Plugin Registry API

```python
from toondb.plugins import PluginRegistry

registry = PluginRegistry()

# Register plugin
registry.register(my_plugin)

# List all plugins
plugins = registry.list_plugins()
print(plugins)  # ["email_validator", "fraud_detector"]

# Get plugin by name
plugin = registry.get("email_validator")

# Unregister plugin
registry.unregister("email_validator")

# Fire trigger manually
row = {"email": "test@example.com"}
result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
```

#### Error Handling

```python
from toondb.plugins import TriggerAbort

try:
    result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
except TriggerAbort as e:
    print(f"Trigger aborted: {e}")
    print(f"Error code: {e.code}")
except RuntimeError as e:
    print(f"Plugin execution failed: {e}")
```

### Transaction Advanced Features

#### Transaction ID

```python
with Database.open("./my_db") as db:
    txn = db.transaction()
    print(f"Transaction ID: {txn.id}")  # e.g., 42
    txn.commit()
```

#### Commit Returns LSN

Track durability with Log Sequence Numbers:

```python
txn = db.transaction()
txn.put(b"key", b"value")
lsn = txn.commit()
print(f"Data committed at LSN {lsn}")

# Later, verify checkpoint includes this LSN
stats = db.stats()
if stats.get('last_checkpoint_lsn', 0) >= lsn:
    print("Data is checkpointed and durable")
```

#### SQL in Transactions

```python
with Database.open("./my_db") as db:
    # Create table
    db.execute_sql("CREATE TABLE users (id INT, name TEXT)")
    
    # Atomic multi-table transaction
    txn = db.transaction()
    
    # Insert via SQL
    txn.execute("INSERT INTO users VALUES (1, 'Alice')")
    
    # Insert via KV (same transaction!)
    txn.put(b"user:1:metadata", b'{"verified": true}')
    
    # Commit both SQL and KV operations atomically
    txn.commit()
```

### IPC Server & Multi-Process Access

Enable multiple processes to access the same database via Unix domain sockets.

#### Starting the Server

```bash
# Basic usage
toondb-server --db ./my_database

# Custom socket path
toondb-server --db ./my_database --socket /tmp/custom.sock

# Production settings
toondb-server \
    --db ./production_db \
    --max-clients 200 \
    --timeout-ms 60000 \
    --log-level info
```

**Server Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--db <PATH>` | `./toondb_data` | Database directory |
| `--socket <PATH>` | `<db>/toondb.sock` | Unix socket path |
| `--max-clients <N>` | 100 | Maximum concurrent connections |
| `--timeout-ms <MS>` | 30000 | Connection timeout (milliseconds) |
| `--log-level <LEVEL>` | `info` | Log level (trace/debug/info/warn/error) |

#### Connecting from Python

```python
from toondb import IpcClient

# Connect to server
client = IpcClient.connect("./my_database/toondb.sock", timeout=30.0)

# Use like embedded database
client.put(b"key", b"value")
value = client.get(b"key")

# Check latency
latency = client.ping()
print(f"Round-trip latency: {latency*1000:.2f}ms")

# Clean up
client.close()
```

#### IPC Protocol Details

The IPC protocol uses a binary message format:

**Message Structure:**
```
[OpCode: 1 byte][Length: 4 bytes LE][Payload: N bytes]
```

**Request OpCodes:**

| Code | Name | Payload |
|------|------|---------|
| 0x01 | PUT | key_len(4) + key + val_len(4) + val |
| 0x02 | GET | key_len(4) + key |
| 0x03 | DELETE | key_len(4) + key |
| 0x04 | BEGIN_TXN | (empty) |
| 0x05 | COMMIT_TXN | txn_id(8) |
| 0x06 | ABORT_TXN | txn_id(8) |
| 0x07 | QUERY | JSON query params |
| 0x0C | CHECKPOINT | (empty) |
| 0x0D | STATS | (empty) |
| 0x0E | PING | (empty) |

**Response OpCodes:**

| Code | Name | Payload |
|------|------|---------|
| 0x80 | OK | (empty) |
| 0x81 | ERROR | error_msg |
| 0x82 | VALUE | val_len(4) + val |
| 0x83 | TXN_ID | txn_id(8) |
| 0x86 | STATS_RESP | JSON stats |
| 0x87 | PONG | timestamp(8) |

#### Multi-Process Example

```python
# Process 1: Writer
from toondb import IpcClient

client = IpcClient.connect("./shared_db/toondb.sock")
for i in range(1000):
    client.put(f"log:{i}".encode(), f"entry_{i}".encode())
client.close()
```

```python
# Process 2: Reader
from toondb import IpcClient

client = IpcClient.connect("./shared_db/toondb.sock")
results = client.scan("log:")
print(f"Found {len(results)} log entries")
client.close()
```

### CLI Tools

The SDK includes globally available CLI tools for managing servers, bulk operations, and high-performance vector search.

#### toondb-server
IPC server management.
```bash
toondb-server --db ./database
```

#### toondb-grpc-server
Dedicated gRPC server for vector operations.
```bash
toondb-grpc-server --port 50051
```

#### toondb-bulk

High-performance bulk operations that bypass Python FFI overhead.

**Build HNSW Index:**
```bash
toondb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw \
    --dimension 768 \
    --max-connections 16 \
    --ef-construction 100 \
    --threads 0 \
    --batch-size 1000
```

**Advanced Options:**
```bash
toondb-bulk build-index \
    --input vectors.npy \
    --output index.hnsw \
    --dimension 1536 \
    --max-connections 32 \
    --ef-construction 200 \
    --threads 8 \
    --direct-read \        # Use direct I/O instead of mmap
    --prefault \           # Prefault mmap pages
    --telemetry \          # Enable page fault telemetry
    --ordering kmeans      # Reorder for locality (random_projection/kmeans/none)
```

**Query Index:**
```bash
toondb-bulk query \
    --index index.hnsw \
    --query query_vector.raw \
    --k 10 \
    --ef 64
```

**Get Index Info:**
```bash
toondb-bulk info --index index.hnsw
# Output:
# Dimension: 768
# Vectors: 100000
# Max connections: 16
# Entry point: 42
```

**Convert Formats:**
```bash
toondb-bulk convert \
    --input vectors.npy \
    --output vectors.raw \
    --to-format raw_f32 \
    --dimension 768
```

#### toondb-grpc-server

gRPC server for remote vector operations.

```bash
# Start server
toondb-grpc-server --host 0.0.0.0 --port 50051 --debug

# Options:
#   --host <HOST>    Bind address [default: 127.0.0.1]
#   -p, --port <N>   Listen port [default: 50051]
#   -d, --debug      Enable debug logging
```

**Use from Python:**
```python
import grpc
from toondb_pb2 import VectorSearchRequest
from toondb_pb2_grpc import VectorServiceStub

channel = grpc.insecure_channel('localhost:50051')
stub = VectorServiceStub(channel)

request = VectorSearchRequest(
    index_path="index.hnsw",
    query_vector=query_vec.tobytes(),
    k=10,
    ef_search=64
)

response = stub.Search(request)
for neighbor in response.neighbors:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance}")
```

---

## API Reference

### class Database

| Method | Description |
|--------|-------------|
| `open(path: str) -> Database` | Open/create database at path |
| `close()` | Close the database |
| `put(key: bytes, value: bytes)` | Store key-value pair |
| `get(key: bytes) -> Optional[bytes]` | Retrieve value by key |
| `delete(key: bytes)` | Delete a key |
| `put_path(path: str, value: bytes)` | Store at hierarchical path |
| `get_path(path: str) -> Optional[bytes]` | Retrieve by path |
| `delete_path(path: str)` | Delete at hierarchical path |
| `scan(start: bytes, end: bytes)` | Iterate key range |
| `scan_prefix(prefix: bytes)` | Scan keys matching prefix |
| `transaction() -> Transaction` | Begin new transaction |
| `execute_sql(query: str) -> SQLQueryResult` | Execute SQL statement |
| `execute(query: str) -> SQLQueryResult` | Alias for execute_sql() |
| `checkpoint() -> int` | Force checkpoint, returns LSN |
| `stats() -> dict` | Get storage statistics |
| `to_toon(table: str, records: list, fields: list) -> str` | Convert records to TOON format (static) |
| `from_toon(toon_str: str) -> tuple` | Parse TOON format (static) |

### class Transaction

| Method | Description |
|--------|-------------|
| `id` | Get transaction ID (property) |
| `put(key: bytes, value: bytes)` | Put within transaction |
| `get(key: bytes) -> Optional[bytes]` | Get with snapshot isolation |
| `delete(key: bytes)` | Delete within transaction |
| `put_path(path: str, value: bytes)` | Put at path |
| `get_path(path: str) -> Optional[bytes]` | Get at path |
| `scan(start: bytes, end: bytes)` | Scan within transaction |
| `scan_prefix(prefix: bytes)` | Scan keys matching prefix |
| `scan_batched(start: bytes, end: bytes, batch_size: int)` | High-performance batched scan |
| `execute(sql: str) -> SQLQueryResult` | Execute SQL within transaction |
| `commit() -> int` | Commit transaction, returns LSN |
| `abort()` | Abort/rollback transaction |

### class IpcClient

| Method | Description |
|--------|-------------|
| `connect(path: str, timeout: float) -> IpcClient` | Connect to IPC server |
| `close()` | Close connection |
| `ping() -> float` | Ping, returns latency |
| `put(key: bytes, value: bytes)` | Store key-value |
| `get(key: bytes) -> Optional[bytes]` | Retrieve value |
| `delete(key: bytes)` | Delete key |
| `put_path(path: List[str], value: bytes)` | Store at path |
| `get_path(path: List[str]) -> Optional[bytes]` | Get at path |
| `query(prefix: str) -> Query` | Create query builder |
| `scan(prefix: str) -> List[dict]` | Scan with prefix |
| `begin_transaction() -> int` | Begin transaction |
| `commit(txn_id: int) -> int` | Commit transaction |
| `abort(txn_id: int)` | Abort transaction |
| `checkpoint()` | Force checkpoint |
| `stats() -> dict` | Get statistics |

### class PythonPlugin

| Attribute/Method | Description |
|------------------|-------------|
| `name: str` | Plugin name (unique identifier) |
| `code: str` | Python code to execute |
| `version: str` | Plugin version (default: "1.0.0") |
| `packages: List[str]` | Required Python packages |
| `triggers: Dict[str, List[str]]` | Table -> trigger events mapping |
| `with_trigger(table: str, event: str) -> PythonPlugin` | Add trigger binding (fluent) |
| `to_dict() -> dict` | Convert to dictionary |

### class PluginRegistry

| Method | Description |
|--------|-------------|
| `register(plugin: PythonPlugin)` | Register a plugin |
| `unregister(name: str)` | Unregister a plugin |
| `list_plugins() -> List[str]` | List all registered plugin names |
| `get(name: str) -> Optional[PythonPlugin]` | Get plugin by name |
| `fire(table: str, event: TriggerEvent, row: dict) -> dict` | Fire triggers for an event |

### enum TriggerEvent

| Value | Description |
|-------|-------------|
| `BEFORE_INSERT` | Before inserting a row |
| `AFTER_INSERT` | After inserting a row |
| `BEFORE_UPDATE` | Before updating a row |
| `AFTER_UPDATE` | After updating a row |
| `BEFORE_DELETE` | Before deleting a row |
| `AFTER_DELETE` | After deleting a row |
| `ON_BATCH` | On batch operations |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `ToonDBError` | Base exception |
| `ConnectionError` | Connection failed |
| `TransactionError` | Transaction operation failed |
| `ProtocolError` | Wire protocol error |
| `DatabaseError` | Database operation failed |
| `TriggerAbort` | Raised by trigger code to abort operation |

---

## Use Cases

### 1. Session Cache

```python
from toondb import Database
import json
from datetime import datetime, timedelta

class SessionCache:
    def __init__(self, db, ttl_hours=24):
        self.db = db
        self.ttl = timedelta(hours=ttl_hours)
    
    def set(self, session_id: str, user_data: dict):
        expires = (datetime.utcnow() + self.ttl).isoformat()
        value = {"data": user_data, "expires": expires}
        self.db.put(f"session:{session_id}".encode(), 
                    json.dumps(value).encode())
    
    def get(self, session_id: str) -> dict | None:
        raw = self.db.get(f"session:{session_id}".encode())
        if not raw:
            return None
        value = json.loads(raw.decode())
        if datetime.fromisoformat(value["expires"]) < datetime.utcnow():
            self.delete(session_id)
            return None
        return value["data"]
    
    def delete(self, session_id: str):
        self.db.delete(f"session:{session_id}".encode())
```

### 2. User Management with Secondary Index

```python
class UserStore:
    def __init__(self, db):
        self.db = db
    
    def create_user(self, email: str, name: str) -> str:
        # Check uniqueness
        if self.db.get(f"idx:email:{email}".encode()):
            raise ValueError("Email exists")
        
        user_id = f"user_{int(time.time()*1000)}"
        user = {"id": user_id, "email": email, "name": name}
        
        with self.db.transaction() as txn:
            txn.put(f"users:{user_id}".encode(), 
                    json.dumps(user).encode())
            txn.put(f"idx:email:{email}".encode(), 
                    user_id.encode())
        
        return user_id
    
    def get_by_email(self, email: str) -> dict | None:
        user_id = self.db.get(f"idx:email:{email}".encode())
        if not user_id:
            return None
        data = self.db.get(f"users:{user_id.decode()}".encode())
        return json.loads(data.decode()) if data else None
```

### 3. Document Store

```python
class DocumentStore:
    def __init__(self, db, collection: str):
        self.db = db
        self.collection = collection
    
    def insert(self, doc: dict, doc_id: str = None) -> str:
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]
        key = f"doc:{self.collection}:{doc_id}".encode()
        self.db.put(key, json.dumps(doc).encode())
        return doc_id
    
    def find_all(self) -> list[dict]:
        prefix = f"doc:{self.collection}:".encode()
        docs = []
        for key, val in self.db.scan(prefix, prefix[:-1] + b";"):
            docs.append(json.loads(val.decode()))
        return docs
```

### 4. Feature Flags

```python
class FeatureFlags:
    def __init__(self, db, environment: str):
        self.db = db
        self.env = environment
    
    def set(self, feature: str, enabled: bool):
        path = f"features/{self.env}/{feature}"
        self.db.put_path(path, b"true" if enabled else b"false")
    
    def is_enabled(self, feature: str) -> bool:
        path = f"features/{self.env}/{feature}"
        val = self.db.get_path(path)
        return val and val.decode().lower() == "true"
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# ✓ Good
with Database.open("./data") as db:
    db.put(b"key", b"value")

# ✗ Avoid
db = Database.open("./data")
db.put(b"key", b"value")
# Easy to forget db.close()
```

### 2. Batch Operations in Transactions

```python
# ✓ Good - single transaction
with db.transaction() as txn:
    for item in items:
        txn.put(item.key, item.value)

# ✗ Slow - many small transactions
for item in items:
    db.put(item.key, item.value)
```

### 3. Use Appropriate Key Prefixes

```python
# ✓ Good - organized, scannable
db.put(b"user:123:profile", data)
db.put(b"user:123:settings", data)
db.put(b"order:456:items", data)

# ✗ Bad - no structure
db.put(b"user123profile", data)
```

### 4. Handle Missing Keys

```python
# ✓ Good
value = db.get(b"key")
if value is None:
    # Handle missing key
    pass

# ✗ Bad - assumes key exists
value = db.get(b"key").decode()  # AttributeError if None
```

### 5. Error Handling

```python
from toondb.errors import DatabaseError, TransactionError

try:
    with db.transaction() as txn:
        txn.put(b"key", b"value")
except TransactionError as e:
    print(f"Transaction failed: {e}")
except DatabaseError as e:
    print(f"Database error: {e}")
```

---

## Troubleshooting

### Library Not Found

```
DatabaseError: Could not find libtoondb_storage.dylib
```

**Solution**: Set `TOONDB_LIB_PATH` environment variable:
```bash
export TOONDB_LIB_PATH=/path/to/target/release
```

### Connection Refused (IPC)

```
ConnectionError: Failed to connect to /tmp/toondb.sock
```

**Solution**: Ensure IPC server is running:
```bash
cargo run --bin ipc_server -- --socket /tmp/toondb.sock
```

### Transaction Already Completed

```
TransactionError: Transaction already committed
```

**Solution**: Don't reuse transaction objects after commit/abort:
```python
txn = db.transaction()
txn.commit()
# txn.put(...)  # ✗ Error!

# Create new transaction instead
txn2 = db.transaction()
txn2.put(...)
```

---

## Version Compatibility

| SDK Version | ToonDB Version | Python |
|-------------|----------------|--------|
| 0.1.x       | 0.1.x          | 3.9+   |
| 0.2.x       | 0.2.x          | 3.9+   |

---

## License

Apache License 2.0 - Same as ToonDB core.
