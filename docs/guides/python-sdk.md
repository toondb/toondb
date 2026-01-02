# Python SDK Guide

> **Version:** 0.2.9  
> **Time:** 35 minutes  
> **Difficulty:** Beginner to Intermediate  
> **Prerequisites:** Python 3.9+

Complete guide to ToonDB's Python SDK covering SQL, key-value operations, advanced features (TOON format, batched scanning, plugins), bulk operations, and multi-process modes.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [SQL Database](#sql-database)
4. [Key-Value Operations](#key-value-operations)
5. [Path API](#path-api)
6. [Prefix Scanning](#prefix-scanning)
7. [Transactions](#transactions)
8. [Query Builder](#query-builder)
9. [Vector Search](#vector-search)
10. [IPC Mode](#ipc-mode)
11. [CLI Tools](#cli-tools)
    - [toondb-server](#toondb-server-options)
    - [toondb-bulk](#toondb-bulk)
    - [toondb-grpc-server](#toondb-grpc-server)
12. [Advanced Features](#advanced-features)
    - [TOON Format](#toon-format)
    - [Batched Scanning](#batched-scanning)
    - [Statistics & Monitoring](#statistics--monitoring)
    - [Manual Checkpoint](#manual-checkpoint)
    - [Python Plugins](#python-plugins)
    - [Transaction Advanced](#transaction-advanced)
13. [Best Practices](#best-practices)
14. [Complete Examples](#complete-examples)

---

## Installation

```bash
pip install toondb-client
```

**What's New in 0.2.7:**
- ✅ Full SQL engine support (CREATE, INSERT, SELECT, UPDATE, DELETE)
- ✅ SQL in transactions via Transaction.execute()
- ✅ SQL WHERE clauses with multiple operators
- ✅ SQL ORDER BY, LIMIT, OFFSET support

**What's New in 0.2.6:**
- ✅ Enhanced `scan_prefix()` method for multi-tenant isolation
- ✅ Bulk vector operations (~1,600 vec/s)
- ✅ Zero-compilation with pre-built binaries
- ✅ Improved FFI performance

> **Import Note:** Install with `pip install toondb-client`, import as `from toondb import Database`

**Pre-built for:**
- Linux (x86_64, aarch64)
- macOS (Intel, Apple Silicon)
- Windows (x64)

---

## Quick Start

### Embedded Mode

```python
from toondb import Database

# Open database
with Database.open("./my_database") as db:
    # Put and Get
    db.put(b"user:123", b'{"name":"Alice","age":30}')
    value = db.get(b"user:123")
    print(value.decode())
    # Output: {"name":"Alice","age":30}
```

**Output:**
```
{"name":"Alice","age":30}
```

---

## SQL Database

### CREATE TABLE

```python
from toondb import Database

with Database.open("./sql_db") as db:
    # Create table
    db.execute_sql("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        )
    """)
    
    # Insert data
    db.execute_sql("""
        INSERT INTO users (id, name, email, age)
        VALUES (1, 'Alice', 'alice@example.com', 30)
    """)
    
    db.execute_sql("""
        INSERT INTO users (id, name, email, age)
        VALUES (2, 'Bob', 'bob@example.com', 25)
    """)
```

**Output:**
```
Table 'users' created
2 rows inserted
```

### SELECT Queries

```python
# Select all
results = db.execute_sql("SELECT * FROM users")
for row in results:
    print(row)

# Output:
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 30}
# {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'age': 25}

# WHERE clause
results = db.execute_sql("SELECT name, age FROM users WHERE age > 26")
for row in results:
    print(f"{row['name']}: {row['age']} years old")

# Output:
# Alice: 30 years old
```

### JOIN Queries

```python
# Create orders table
db.execute_sql("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product TEXT,
        amount REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")

# Insert orders
db.execute_sql("INSERT INTO orders VALUES (1, 1, 'Laptop', 999.99)")
db.execute_sql("INSERT INTO orders VALUES (2, 1, 'Mouse', 25.00)")
db.execute_sql("INSERT INTO orders VALUES (3, 2, 'Keyboard', 75.00)")

# JOIN query
results = db.execute_sql("""
    SELECT users.name, orders.product, orders.amount
    FROM users
    JOIN orders ON users.id = orders.user_id
    WHERE orders.amount > 50
    ORDER BY orders.amount DESC
""")

for row in results:
    print(f"{row['name']} bought {row['product']} for ${row['amount']}")
```

**Output:**
```
Alice bought Laptop for $999.99
Bob bought Keyboard for $75.0
```

### Aggregations

```python
# GROUP BY with aggregations
results = db.execute_sql("""
    SELECT users.name, COUNT(*) as order_count, SUM(orders.amount) as total
    FROM users
    JOIN orders ON users.id = orders.user_id
    GROUP BY users.name
    ORDER BY total DESC
""")

for row in results:
    print(f"{row['name']}: {row['order_count']} orders, ${row['total']} total")
```

**Output:**
```
Alice: 2 orders, $1024.99 total
Bob: 1 orders, $75.0 total
```

### UPDATE and DELETE

```python
# Update
db.execute_sql("UPDATE users SET age = 31 WHERE name = 'Alice'")

# Delete
db.execute_sql("DELETE FROM users WHERE age < 26")

# Verify
results = db.execute_sql("SELECT name, age FROM users")
for row in results:
    print(row)

# Output:
# {'name': 'Alice', 'age': 31}
```

---

## Key-Value Operations

### Basic Operations

```python
# Put
db.put(b"key", b"value")

# Get
value = db.get(b"key")
if value:
    print(value.decode())
else:
    print("Key not found")

# Delete
db.delete(b"key")

# Output:
# value
# Key not found (after delete)
```

### JSON Data

```python
import json

# Store JSON
user = {"name": "Alice", "email": "alice@example.com", "age": 30}
db.put(b"users/alice", json.dumps(user).encode())

# Retrieve JSON
value = db.get(b"users/alice")
if value:
    user = json.loads(value.decode())
    print(f"Name: {user['name']}, Age: {user['age']}")

# Output:
# Name: Alice, Age: 30
```

---

## Path API

```python
# Store hierarchical data
db.put_path("users/alice/email", b"alice@example.com")
db.put_path("users/alice/age", b"30")
db.put_path("users/alice/settings/theme", b"dark")

# Retrieve by path
email = db.get_path("users/alice/email")
print(f"Alice's email: {email.decode()}")

# Output:
# Alice's email: alice@example.com
```

---

## Prefix Scanning

⭐ **Most efficient way to iterate keys:**

```python
# Insert multi-tenant data
db.put(b"tenants/acme/users/1", b'{"name":"Alice"}')
db.put(b"tenants/acme/users/2", b'{"name":"Bob"}')
db.put(b"tenants/acme/orders/1", b'{"total":100}')
db.put(b"tenants/globex/users/1", b'{"name":"Charlie"}')

# Scan only ACME Corp data (tenant isolation)
results = list(db.scan(b"tenants/acme/", b"tenants/acme;"))
print(f"ACME Corp has {len(results)} items:")
for key, value in results:
    print(f"  {key.decode()}: {value.decode()}")
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
- **Efficient**: Binary-safe iteration

---

## Transactions

### Automatic Transactions

```python
# Context manager handles commit/abort
with db.transaction() as txn:
    txn.put(b"account:1:balance", b"1000")
    txn.put(b"account:2:balance", b"500")
    # Commits on success, aborts on exception
```

**Output:**
```
✅ Transaction committed
```

### Manual Control

```python
txn = db.begin_transaction()
try:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    
    # Scan within transaction
    for key, value in txn.scan(b"key", b"key~"):
        print(f"{key.decode()}: {value.decode()}")
    
    txn.commit()
except Exception as e:
    txn.abort()
    raise
```

**Output:**
```
key1: value1
key2: value2
✅ Transaction committed
```

---

## Query Builder

Returns results in **TOON format** (token-optimized for LLMs):

```python
# Insert structured data
db.put(b"products/laptop", b'{"name":"Laptop","price":999,"stock":5}')
db.put(b"products/mouse", b'{"name":"Mouse","price":25,"stock":20}')

# Query with column selection
results = db.query("products/") \
    .select(["name", "price"]) \
    .limit(10) \
    .to_list()

for key, value in results:
    print(f"{key.decode()}: {value.decode()}")
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
```

---

## Vector Search

### Bulk HNSW Index Building

```python
from toondb.bulk import bulk_build_index, bulk_query_index
import numpy as np

# Generate embeddings (10K × 768D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index at ~1,600 vec/s
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
    metric="cosine"
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

**Output:**
```
Built 10000 vectors at 1598 vec/s
Index size: 45.2 MB
```

### Query HNSW Index

```python
# Single query vector
query = np.random.randn(768).astype(np.float32)

results = bulk_query_index(
    index="my_index.hnsw",
    query=query,
    k=10,
    ef_search=64
)

print(f"Top {len(results)} nearest neighbors:")
for i, neighbor in enumerate(results):
    print(f"{i+1}. ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
```

**Output:**
```
Top 10 nearest neighbors:
1. ID: 3421, Distance: 0.1234
2. ID: 7892, Distance: 0.1456
3. ID: 1205, Distance: 0.1678
...
```

**Performance:**
- Python FFI: ~130 vec/s
- Bulk API: ~1,600 vec/s (12× faster)

---

## IPC Mode

For multi-process applications, ToonDB provides a high-performance IPC server with Unix domain socket communication.

> **Deep Dive:** See [IPC Server Capabilities](../servers/IPC_SERVER.md) for wire protocol details, internals, and architecture.

### Quick Start

```bash
# Start the IPC server (globally available after pip install)
toondb-server --db ./my_database

# Check status
toondb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)
```

```python
# Connect from Python (or any other process)
from toondb import IpcClient

client = IpcClient.connect("./my_database/toondb.sock")

client.put(b"key", b"value")
value = client.get(b"key")
print(value.decode())
# Output: value
```

### toondb-server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db PATH` | `./toondb_data` | Database directory |
| `--socket PATH` | `<db>/toondb.sock` | Unix socket path |
| `--max-clients N` | `100` | Maximum concurrent connections |
| `--timeout-ms MS` | `30000` | Connection timeout (30s) |
| `--log-level LEVEL` | `info` | trace/debug/info/warn/error |

### Server Commands

```bash
# Start server
toondb-server --db ./my_database

# Check if running
toondb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)
#         Socket: ./my_database/toondb.sock
#         Database: /absolute/path/to/my_database

# Stop server gracefully
toondb-server stop --db ./my_database
```

### Production Configuration

```bash
# High-traffic production setup
toondb-server \
    --db /var/lib/toondb/production \
    --socket /var/run/toondb.sock \
    --max-clients 500 \
    --timeout-ms 60000 \
    --log-level info
```

### Wire Protocol

The IPC server uses a binary protocol for high-performance communication. See the [Deep Dive](../servers/IPC_SERVER.md) for full opcode usage.

### Server Statistics

The IPC server tracks real-time metrics accessible via `client.stats()`:

```python
from toondb import IpcClient

client = IpcClient.connect("./my_database/toondb.sock")
stats = client.stats()

print(f"Connections: {stats['connections_active']}/{stats['connections_total']}")
print(f"Requests: {stats['requests_success']} success, {stats['requests_error']} errors")
print(f"Throughput: {stats['bytes_received']} bytes in, {stats['bytes_sent']} bytes out")
print(f"Uptime: {stats['uptime_secs']} seconds")
print(f"Active transactions: {stats['active_transactions']}")
```

---

## CLI Tools

Three CLI tools are available globally after `pip install toondb-client`:

### toondb-bulk

High-performance bulk vector operations (~1,600 vec/s).

> **Deep Dive:** See [Bulk Operations Capabilities](../servers/BULK_OPERATIONS.md) for benchmarks, file formats, and internals.

```bash
# Build HNSW index from embeddings
toondb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw \
    --dimension 768 \
    --max-connections 16 \
    --ef-construction 100 \
    --metric cosine

# Query k-nearest neighbors
toondb-bulk query \
    --index index.hnsw \
    --query query_vector.raw \
    --k 10 \
    --ef 64

# Get index metadata
toondb-bulk info --index index.hnsw
# Output:
# Dimension: 768
# Vectors: 100000
# Max connections: 16

# Convert between formats
toondb-bulk convert \
    --input vectors.npy \
    --output vectors.raw \
    --to-format raw_f32 \
    --dimension 768
```

### toondb-grpc-server

gRPC server for remote vector search operations.

> **Deep Dive:** See [gRPC Server Capabilities](../servers/GRPC_SERVER.md) for service methods, HNSW configuration, and proto definitions.

```bash
# Start gRPC server
toondb-grpc-server --host 0.0.0.0 --port 50051

# Check status
toondb-grpc-server status --port 50051
```

**gRPC Service Methods:** See [gRPC Deep Dive](../servers/GRPC_SERVER.md) for full method signatures.

**Python gRPC Client Example:**

```python
import grpc
from toondb_pb2 import (
    CreateIndexRequest, SearchRequest, HnswConfig
)
from toondb_pb2_grpc import VectorIndexServiceStub

# Connect to gRPC server
channel = grpc.insecure_channel('localhost:50051')
stub = VectorIndexServiceStub(channel)

# Create index
response = stub.CreateIndex(CreateIndexRequest(
    name="my_index",
    dimension=768,
    metric=1,  # COSINE
    config=HnswConfig(
        max_connections=16,
        ef_construction=200,
        ef_search=50
    )
))
print(f"Created: {response.info.name}")

# Search
import numpy as np
query = np.random.randn(768).astype(np.float32)
response = stub.Search(SearchRequest(
    index_name="my_index",
    query=query.tolist(),
    k=10
))
for result in response.results:
    print(f"ID: {result.id}, Distance: {result.distance:.4f}")
```

### Environment Variables

Override bundled binaries with custom paths:

```bash
export TOONDB_SERVER_PATH=/path/to/toondb-server
export TOONDB_BULK_PATH=/path/to/toondb-bulk
export TOONDB_GRPC_SERVER_PATH=/path/to/toondb-grpc-server
```

---

## Advanced Features

### TOON Format

**Token-Optimized Output Notation** - Achieve **40-66% token reduction** for LLM context.

```python
from toondb import Database

# Sample records
records = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
]

# Convert to TOON format
toon_str = Database.to_toon("users", records, ["name", "email"])
print(toon_str)
# Output: users[2]{name,email}:Alice,alice@example.com;Bob,bob@example.com

# Parse TOON back to records
table_name, fields, records = Database.from_toon(toon_str)
print(records)
# Output: [{"name": "Alice", "email": "alice@example.com"}, ...]
```

**Token Comparison:**
- JSON (compact): ~165 tokens
- TOON format: ~70 tokens (**59% reduction!**)

**Use Case: RAG with LLMs**

```python
from toondb import Database
import openai

with Database.open("./knowledge_base") as db:
    # Query relevant documents
    results = db.execute_sql("""
        SELECT title, content 
        FROM documents 
        WHERE category = 'technical'
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
            {"role": "user", "content": "Summarize the documents"}
        ]
    )
```

---

### Batched Scanning

**1000× fewer FFI calls** for large dataset scans.

```python
from toondb import Database

with Database.open("./my_db") as db:
    # Insert 10K test records
    with db.transaction() as txn:
        for i in range(10000):
            txn.put(f"item:{i:05d}".encode(), f"value:{i}".encode())
    
    # Regular scan: 10,000 FFI calls
    txn = db.transaction()
    count = sum(1 for _ in txn.scan(b"item:", b"item;"))
    txn.abort()
    print(f"Regular scan: {count} items")
    
    # Batched scan: 10 FFI calls (1000× fewer!)
    txn = db.transaction()
    count = sum(1 for _ in txn.scan_batched(
        start=b"item:",
        end=b"item;",
        batch_size=1000  # Fetch 1000 results per FFI call
    ))
    txn.abort()
    print(f"Batched scan: {count} items (much faster!)")
```

**Performance:**

| Dataset | Regular Scan | Batched Scan | Speedup |
|---------|--------------|--------------|---------|
| 10K items | 15ms | 2ms | 7.5× |
| 100K items | 150ms | 12ms | 12.5× |

---

### Statistics & Monitoring

```python
from toondb import Database

with Database.open("./my_db") as db:
    # Perform operations
    for i in range(1000):
        db.put(f"key:{i}".encode(), f"value:{i}".encode())
    
    # Get runtime statistics
    stats = db.stats()
    
    print(f"Keys: {stats['keys_count']:,}")
    print(f"Bytes written: {stats['bytes_written']:,}")
    print(f"Bytes read: {stats['bytes_read']:,}")
    print(f"Transactions: {stats['transactions_committed']}")
    
    # Cache metrics
    hits = stats['cache_hits']
    misses = stats['cache_misses']
    hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0
    print(f"Cache hit rate: {hit_rate:.1f}%")
```

**Available Metrics:**
- `keys_count` - Total keys
- `bytes_written` - Cumulative writes
- `bytes_read` - Cumulative reads
- `transactions_committed` - Successful transactions
- `cache_hits` / `cache_misses` - Cache performance

---

### Manual Checkpoint

Force durability checkpoint to flush data to disk.

```python
from toondb import Database

with Database.open("./my_db") as db:
    # Bulk import
    print("Importing 10K records...")
    with db.transaction() as txn:
        for i in range(10000):
            txn.put(f"bulk:{i}".encode(), f"data:{i}".encode())
    
    # Force checkpoint
    lsn = db.checkpoint()
    print(f"Checkpoint complete at LSN {lsn}")
    print("All data is durable on disk!")
```

**When to Use:**
- ✅ Before backups
- ✅ After bulk imports
- ✅ Before system shutdown
- ✅ Periodic durability (every 5 minutes)

---

### Python Plugins

Run Python code as database triggers.

```python
from toondb.plugins import PythonPlugin, PluginRegistry, TriggerEvent, TriggerAbort

# Define validation plugin
plugin = PythonPlugin(
    name="user_validator",
    code='''
def on_before_insert(row: dict) -> dict:
    """Validate and transform data."""
    # Normalize email
    if "email" in row:
        row["email"] = row["email"].lower().strip()
    
    # Validate age
    if row.get("age", 0) < 0:
        raise TriggerAbort("Age cannot be negative", code="INVALID_AGE")
    
    # Add timestamp
    import time
    row["created_at"] = time.time()
    
    return row
''',
    triggers={"users": ["BEFORE INSERT"]}
)

# Register and use
registry = PluginRegistry()
registry.register(plugin)

# Fire trigger
row = {"name": "Alice", "email": "  ALICE@EXAMPLE.COM  ", "age": 30}
result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
print(result["email"])  # "alice@example.com"
print(result["created_at"])  # 1704182400.0
```

**Available Events:**
- `BEFORE_INSERT`, `AFTER_INSERT`
- `BEFORE_UPDATE`, `AFTER_UPDATE`
- `BEFORE_DELETE`, `AFTER_DELETE`

---

### Transaction Advanced

```python
from toondb import Database

with Database.open("./my_db") as db:
    # Get transaction ID
    txn = db.transaction()
    print(f"Transaction ID: {txn.id}")
    
    # Perform operations
    txn.put(b"key", b"value")
    
    # Commit returns LSN (Log Sequence Number)
    lsn = txn.commit()
    print(f"Committed at LSN: {lsn}")
    
    # Execute SQL within transaction
    txn2 = db.transaction()
    txn2.execute("INSERT INTO users VALUES (1, 'Alice')")
    txn2.put(b"user:1:metadata", b'{"verified": true}')
    txn2.commit()  # Atomic SQL + KV operation
```

---

## Best Practices

### 1. Use SQL for Structured Data

```python
# ✅ Good: Use SQL for relational data
db.execute_sql("CREATE TABLE users (...)")
db.execute_sql("INSERT INTO users VALUES (...)")
results = db.execute_sql("SELECT * FROM users WHERE age > 25")
```

### 2. Use K-V for Unstructured Data

```python
# ✅ Good: Use K-V for documents, blobs, cache
db.put(b"cache:user:123", json.dumps(user).encode())
db.put(b"blob:image:456", image_bytes)
```

### 3. Use scan() for Multi-Tenancy

```python
# ✅ Good: Efficient tenant isolation
tenant_id = "acme"
prefix = f"tenants/{tenant_id}/".encode()
end = f"tenants/{tenant_id};".encode()
data = list(db.scan(prefix, end))
```

### 4. Use Transactions

```python
# ✅ Good: Atomic operations
with db.transaction() as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
```

### 5. Use Bulk API for Vectors

```python
# ✅ Good: Fast bulk operations
bulk_build_index(embeddings, "index.hnsw")

# ❌ Bad: Slow FFI loop
for vec in vectors:
    index.insert(vec)  # 12× slower!
```

### 6. Use Batched Scanning for Large Datasets

```python
# ✅ Good: Fast batched scan
txn = db.transaction()
for key, value in txn.scan_batched(b"prefix:", b"prefix;", batch_size=1000):
    process(key, value)
txn.abort()

# ❌ Bad: Slow regular scan for large datasets
for key, value in txn.scan(b"prefix:", b"prefix;"):
    process(key, value)  # 1000× more FFI calls!
```

### 7. Use TOON Format for LLM Context

```python
# ✅ Good: Token-efficient for LLMs
results = db.execute_sql("SELECT * FROM users LIMIT 100")
records = [dict(row) for row in results]
toon_context = Database.to_toon("users", records, ["name", "email"])
# Send to LLM - saves 40-66% tokens!

# ❌ Bad: Wasteful JSON for LLM context
json_context = json.dumps(records)  # Uses 2× more tokens
```

### 8. Always Use Context Managers

```python
# ✅ Good: Automatic cleanup
with Database.open("./db") as db:
    db.put(b"key", b"value")

# ❌ Bad: Manual cleanup required
db = Database.open("./db")
db.put(b"key", b"value")
db.close()
```

---

## Complete Examples

### Example 1: Multi-Tenant SaaS with SQL + K-V

```python
from toondb import Database
import json

def main():
    with Database.open("./saas_db") as db:
        # SQL for tenant metadata
        db.execute_sql("""
            CREATE TABLE IF NOT EXISTS tenants (
                id INTEGER PRIMARY KEY,
                name TEXT,
                created_at TEXT
            )
        """)
        
        db.execute_sql("INSERT INTO tenants VALUES (1, 'ACME Corp', '2026-01-01')")
        db.execute_sql("INSERT INTO tenants VALUES (2, 'Globex Inc', '2026-01-01')")
        
        # K-V for tenant-specific data
        db.put(b"tenants/1/users/alice", b'{"role":"admin","email":"alice@acme.com"}')
        db.put(b"tenants/1/users/bob", b'{"role":"user","email":"bob@acme.com"}')
        db.put(b"tenants/2/users/charlie", b'{"role":"admin","email":"charlie@globex.com"}')
        
        # Query SQL
        tenants = db.execute_sql("SELECT * FROM tenants ORDER BY name")
        
        for tenant in tenants:
            tenant_id = tenant['id']
            tenant_name = tenant['name']
            
            # Scan tenant-specific K-V data
            prefix = f"tenants/{tenant_id}/".encode()
            end = f"tenants/{tenant_id};".encode()
            users = list(db.scan(prefix, end))
            
            print(f"\n{tenant_name} ({len(users)} users):")
            for key, value in users:
                user_data = json.loads(value.decode())
                print(f"  {key.decode()}: {user_data['email']} ({user_data['role']})")

if __name__ == "__main__":
    main()
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

```python
from toondb import Database

with Database.open("./ecommerce") as db:
    # Create schema
    db.execute_sql("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            category TEXT
        )
    """)
    
    db.execute_sql("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            total REAL
        )
    """)
    
    # Insert data
    db.execute_sql("INSERT INTO products VALUES (1, 'Laptop', 999.99, 'Electronics')")
    db.execute_sql("INSERT INTO products VALUES (2, 'Mouse', 25.00, 'Electronics')")
    db.execute_sql("INSERT INTO products VALUES (3, 'Desk', 299.99, 'Furniture')")
    
    db.execute_sql("INSERT INTO orders VALUES (1, 1, 2, 1999.98)")
    db.execute_sql("INSERT INTO orders VALUES (2, 2, 5, 125.00)")
    
    # Analytics query
    results = db.execute_sql("""
        SELECT 
            products.category,
            COUNT(orders.id) as order_count,
            SUM(orders.total) as revenue
        FROM products
        JOIN orders ON products.id = orders.product_id
        GROUP BY products.category
        ORDER BY revenue DESC
    """)
    
    print("Category Performance:")
    for row in results:
        print(f"{row['category']}: {row['order_count']} orders, ${row['revenue']:.2f}")
```

**Output:**
```
Category Performance:
Electronics: 2 orders, $2124.98
```

---

## API Reference

### Database (Embedded)

| Method | Description |
|--------|-------------|
| `Database.open(path)` | Open/create database |
| `put(key: bytes, value: bytes)` | Store key-value |
| `get(key: bytes) -> bytes \| None` | Retrieve value |
| `delete(key: bytes)` | Delete key |
| `put_path(path: str, value: bytes)` | Store by path |
| `get_path(path: str) -> bytes \| None` | Get by path |
| `delete_path(path: str)` | Delete by path |
| `scan(start: bytes, end: bytes)` | Iterate range |
| `scan_prefix(prefix: bytes)` | Scan keys matching prefix |
| `transaction()` | Begin transaction |
| `execute_sql(query: str)` | Execute SQL ⭐ |
| `execute(query: str)` | Alias for execute_sql() |
| `checkpoint() -> int` | Force checkpoint, returns LSN |
| `stats() -> dict` | Get runtime statistics |
| `to_toon(table, records, fields) -> str` | Convert to TOON format (static) |
| `from_toon(toon_str) -> tuple` | Parse TOON format (static) |

### Transaction

| Method | Description |
|--------|-------------|
| `id` | Transaction ID (property) |
| `put(key: bytes, value: bytes)` | Put within transaction |
| `get(key: bytes) -> bytes \| None` | Get with snapshot isolation |
| `delete(key: bytes)` | Delete within transaction |
| `scan(start: bytes, end: bytes)` | Scan within transaction |
| `scan_prefix(prefix: bytes)` | Scan keys matching prefix |
| `scan_batched(start, end, batch_size)` | High-performance batched scan |
| `execute(sql: str)` | Execute SQL within transaction |
| `commit() -> int` | Commit, returns LSN |
| `abort()` | Abort/rollback |

### IpcClient

| Method | Description |
|--------|-------------|
| `IpcClient.connect(path)` | Connect to server |
| `ping() -> float` | Check latency |
| `query(prefix: str)` | Create query builder |
| `scan(prefix: str)` | Scan prefix |

### Bulk API

| Function | Description |
|----------|-------------|
| `bulk_build_index(...)` | Build HNSW (~1,600 vec/s) |
| `bulk_query_index(...)` | Query k-NN |
| `bulk_info(index)` | Get index metadata |

---

## Configuration

```python
db = Database.open("./my_db", config={
    "create_if_missing": True,
    "wal_enabled": True,
    "sync_mode": "normal",  # "full", "normal", "off"
    "memtable_size_bytes": 64 * 1024 * 1024,
})
```

---

## Testing

```bash
# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_sql.py -v

# With coverage
pytest --cov=toondb tests/
```

---

## Resources

- [Python SDK GitHub](https://github.com/toondb/toondb/tree/main/toondb-python-sdk)
- [PyPI Package](https://pypi.org/project/toondb-client/)
- [API Reference](../api-reference/python-api.md)
- [Go SDK](./go-sdk.md)
- [JavaScript SDK](./nodejs-sdk.md)
- [Rust SDK](./rust-sdk.md)

---

*Last updated: January 2026 (v0.2.9)*
