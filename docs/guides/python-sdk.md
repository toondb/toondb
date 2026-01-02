# Python SDK Guide

> **Version:** 0.2.7  
> **Time:** 25 minutes  
> **Difficulty:** Beginner  
> **Prerequisites:** Python 3.9+

Complete guide to ToonDB's Python SDK with key-value operations, bulk operations, and multi-process modes.

> **Note:** SQL support is planned for a future release. This guide covers key-value and path-based operations.

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
11. [Best Practices](#best-practices)
12. [Complete Examples](#complete-examples)

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

For multi-process applications:

```bash
# Terminal 1: Start server
toondb-server --db ./my_database
```

```python
# Terminal 2: Connect
from toondb import IpcClient

client = IpcClient.connect("./my_database/toondb.sock")

client.put(b"key", b"value")
value = client.get(b"key")
print(value.decode())

# Output: value
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

### 6. Always Use Context Managers

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
| `scan(start: bytes, end: bytes)` | Iterate range |
| `transaction()` | Begin transaction |
| `execute_sql(query: str)` | Execute SQL ⭐ |
| `checkpoint()` | Force checkpoint |

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

*Last updated: January 2026 (v0.2.7)*
