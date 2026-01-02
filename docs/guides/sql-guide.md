# Working with SQL in ToonDB

This guide shows you how to use SQL queries in ToonDB for relational data operations.

## Introduction

ToonDB supports full SQL-92 syntax, allowing you to work with relational data using familiar SQL statements. The SQL engine is built on top of ToonDB's high-performance storage layer, giving you both SQL flexibility and ToonDB's speed.

## Setting Up

### Installation

SQL support is built into all ToonDB SDKs:

```bash
# Python
pip install toondb

# Node.js
npm install @sushanth/toondb

# Rust (add to Cargo.toml)
toondb-client = "0.2.7"

# Go
go get github.com/toondb/toondb/toondb-go@v0.2.7
```

### Opening a Database

```python
# Python
from toondb import Database
db = Database("./mydb")
```

```typescript
// TypeScript
import { Database } from '@sushanth/toondb';
const db = new Database('./mydb');
```

```rust
// Rust
use toondb_client::Client;
let mut client = Client::open("./mydb")?;
```

## Creating Tables

Define your schema with CREATE TABLE:

```python
db.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL,
        created_at TEXT,
        is_active BOOLEAN DEFAULT TRUE
    )
""")

db.execute("""
    CREATE TABLE posts (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        content TEXT,
        likes INTEGER DEFAULT 0,
        created_at TEXT
    )
""")
```

### Supported Data Types

| Type | Description | Example |
|------|-------------|---------|
| `INTEGER` | 64-bit signed integer | `42`, `-100` |
| `REAL` | 64-bit floating point | `3.14`, `-0.5` |
| `TEXT` | UTF-8 string | `'Hello'`, `'日本語'` |
| `BLOB` | Binary data | `x'48656c6c6f'` |
| `BOOLEAN` | True/false | `TRUE`, `FALSE` |
| `NULL` | Null value | `NULL` |

### Constraints

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,           -- Primary key
    sku TEXT NOT NULL UNIQUE,         -- Required and unique
    name TEXT NOT NULL,               -- Required
    price REAL CHECK (price > 0),     -- Validation
    stock INTEGER DEFAULT 0,          -- Default value
    category TEXT DEFAULT 'general'   -- Default string
)
```

## Inserting Data

### Single Row Insert

```python
db.execute("""
    INSERT INTO users (id, username, email, created_at)
    VALUES (1, 'alice', 'alice@example.com', '2024-01-01')
""")
```

### Multiple Rows

```python
db.execute("""
    INSERT INTO posts (id, user_id, title, content, likes) VALUES
        (1, 1, 'First Post', 'Hello World!', 10),
        (2, 1, 'Second Post', 'SQL is great', 25),
        (3, 1, 'Third Post', 'More content', 15)
""")
```

### Using Transactions

For better performance with multiple inserts:

```python
txn = db.begin_transaction()
try:
    for i in range(1000):
        db.execute(f"""
            INSERT INTO logs (id, message, timestamp)
            VALUES ({i}, 'Log entry {i}', '{datetime.now()}')
        """)
    txn.commit()
except Exception as e:
    txn.rollback()
    raise
```

## Querying Data

### Basic SELECT

```python
# All columns
result = db.execute("SELECT * FROM users")

# Specific columns
result = db.execute("SELECT id, username, email FROM users")

# Process results
for row in result.rows:
    print(f"User: {row['username']} ({row['email']})")
```

### WHERE Clause

Filter results with conditions:

```python
# Simple condition
result = db.execute("SELECT * FROM users WHERE is_active = TRUE")

# Multiple conditions
result = db.execute("""
    SELECT * FROM posts 
    WHERE likes > 20 AND user_id = 1
""")

# Pattern matching
result = db.execute("""
    SELECT * FROM users 
    WHERE email LIKE '%@gmail.com'
""")

# Range queries
result = db.execute("""
    SELECT * FROM products 
    WHERE price BETWEEN 10.0 AND 50.0
""")

# IN clause
result = db.execute("""
    SELECT * FROM users 
    WHERE id IN (1, 2, 3, 5, 8)
""")
```

### Sorting Results

```python
# Ascending order
result = db.execute("""
    SELECT username, email FROM users 
    ORDER BY username ASC
""")

# Descending order
result = db.execute("""
    SELECT title, likes FROM posts 
    ORDER BY likes DESC
""")

# Multiple columns
result = db.execute("""
    SELECT * FROM products 
    ORDER BY category ASC, price DESC
""")
```

### Limiting Results

```python
# Get top 10
result = db.execute("""
    SELECT * FROM posts 
    ORDER BY likes DESC 
    LIMIT 10
""")

# Pagination (skip 20, take 10)
result = db.execute("""
    SELECT * FROM posts 
    ORDER BY created_at DESC 
    LIMIT 10 OFFSET 20
""")
```

### Aggregate Functions

```python
# Count rows
result = db.execute("SELECT COUNT(*) as total FROM users")
total_users = result.rows[0]['total']

# Average
result = db.execute("SELECT AVG(likes) as avg_likes FROM posts")

# Min/Max
result = db.execute("""
    SELECT 
        MIN(price) as min_price,
        MAX(price) as max_price
    FROM products
""")

# Sum
result = db.execute("SELECT SUM(quantity) as total_stock FROM inventory")
```

### GROUP BY

```python
# Count posts per user
result = db.execute("""
    SELECT user_id, COUNT(*) as post_count
    FROM posts
    GROUP BY user_id
""")

# Average likes per user
result = db.execute("""
    SELECT user_id, AVG(likes) as avg_likes
    FROM posts
    GROUP BY user_id
    HAVING avg_likes > 15
""")
```

## Updating Data

### Update Single Row

```python
db.execute("""
    UPDATE users 
    SET email = 'newemail@example.com' 
    WHERE id = 1
""")
```

### Update Multiple Fields

```python
db.execute("""
    UPDATE products 
    SET price = 29.99, stock = 100 
    WHERE sku = 'WIDGET-001'
""")
```

### Update with Expression

```python
# Increment likes
db.execute("""
    UPDATE posts 
    SET likes = likes + 1 
    WHERE id = 5
""")

# Apply discount
db.execute("""
    UPDATE products 
    SET price = price * 0.9 
    WHERE category = 'clearance'
""")
```

### Conditional Update

```python
db.execute("""
    UPDATE users 
    SET is_active = FALSE 
    WHERE last_login < '2023-01-01'
""")
```

## Deleting Data

### Delete Specific Rows

```python
db.execute("DELETE FROM users WHERE id = 5")

db.execute("""
    DELETE FROM posts 
    WHERE created_at < '2023-01-01'
""")
```

### Delete with Multiple Conditions

```python
db.execute("""
    DELETE FROM products 
    WHERE stock = 0 AND discontinued = TRUE
""")
```

### Truncate Table

```python
# Delete all rows
db.execute("DELETE FROM temp_table")

# Or drop and recreate
db.execute("DROP TABLE temp_table")
db.execute("CREATE TABLE temp_table (...)")
```

## Joins

### Inner Join

```python
result = db.execute("""
    SELECT 
        users.username,
        posts.title,
        posts.likes
    FROM users
    INNER JOIN posts ON users.id = posts.user_id
    WHERE posts.likes > 10
""")

for row in result.rows:
    print(f"{row['username']}: {row['title']} ({row['likes']} likes)")
```

### Left Join

```python
# Get all users, including those without posts
result = db.execute("""
    SELECT 
        users.username,
        COUNT(posts.id) as post_count
    FROM users
    LEFT JOIN posts ON users.id = posts.user_id
    GROUP BY users.id, users.username
""")
```

### Multiple Joins

```python
result = db.execute("""
    SELECT 
        users.username,
        posts.title,
        comments.content
    FROM users
    INNER JOIN posts ON users.id = posts.user_id
    INNER JOIN comments ON posts.id = comments.post_id
    WHERE posts.created_at > '2024-01-01'
""")
```

## Indexes

Create indexes to speed up queries:

```python
# Single column index
db.execute("CREATE INDEX idx_users_email ON users (email)")

# Composite index
db.execute("CREATE INDEX idx_posts_user_date ON posts (user_id, created_at)")

# Unique index
db.execute("CREATE UNIQUE INDEX idx_products_sku ON products (sku)")
```

Drop indexes when no longer needed:

```python
db.execute("DROP INDEX idx_users_email")
```

## Transactions

Ensure atomicity for multiple operations:

### Python

```python
txn = db.begin_transaction()
try:
    # Transfer money between accounts
    db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    db.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
    db.execute("INSERT INTO transactions (from_id, to_id, amount) VALUES (1, 2, 100)")
    
    txn.commit()
    print("Transfer completed")
except Exception as e:
    txn.rollback()
    print(f"Transfer failed: {e}")
```

### TypeScript

```typescript
const txn = await db.beginTransaction();
try {
    await db.execute("UPDATE inventory SET stock = stock - 1 WHERE id = 1");
    await db.execute("INSERT INTO orders (product_id, quantity) VALUES (1, 1)");
    
    await txn.commit();
    console.log('Order placed');
} catch (error) {
    await txn.rollback();
    console.error('Order failed:', error);
}
```

### Rust

```rust
client.with_transaction(|txn| {
    client.execute("DELETE FROM old_data WHERE created_at < '2023-01-01'")?;
    client.execute("INSERT INTO archive SELECT * FROM old_data")?;
    Ok(())
})?;
```

## Best Practices

### 1. Use Prepared Statements

For security and performance (when available):

```python
# Bad - SQL injection risk
user_input = "admin' OR '1'='1"
db.execute(f"SELECT * FROM users WHERE username = '{user_input}'")

# Good - use parameterized queries (implementation-dependent)
db.execute("SELECT * FROM users WHERE username = ?", [user_input])
```

### 2. Batch Operations in Transactions

```python
# Slow - individual transactions
for item in items:
    db.execute(f"INSERT INTO items VALUES ({item.id}, '{item.name}')")

# Fast - single transaction
txn = db.begin_transaction()
for item in items:
    db.execute(f"INSERT INTO items VALUES ({item.id}, '{item.name}')")
txn.commit()
```

### 3. Create Indexes for Frequent Queries

```python
# If you often query by email
db.execute("CREATE INDEX idx_email ON users (email)")

# If you often filter by date range
db.execute("CREATE INDEX idx_created_at ON posts (created_at)")
```

### 4. Select Only Required Columns

```python
# Inefficient
result = db.execute("SELECT * FROM users")

# Efficient
result = db.execute("SELECT id, username FROM users")
```

### 5. Use LIMIT for Large Results

```python
# Prevent loading millions of rows
result = db.execute("""
    SELECT * FROM logs 
    ORDER BY timestamp DESC 
    LIMIT 100
""")
```

## Real-World Examples

### User Authentication System

```python
# Create schema
db.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        created_at TEXT,
        last_login TEXT
    )
""")

# Register user
db.execute("""
    INSERT INTO users (id, username, password_hash, email, created_at)
    VALUES (1, 'alice', '<hash>', 'alice@example.com', '2024-01-01')
""")

# Login check
result = db.execute("""
    SELECT id, username, password_hash 
    FROM users 
    WHERE username = 'alice'
""")

if result.rows and verify_password(result.rows[0]['password_hash'], input_password):
    # Update last login
    db.execute(f"""
        UPDATE users 
        SET last_login = '{datetime.now()}' 
        WHERE id = {result.rows[0]['id']}
    """)
    print("Login successful")
```

### E-commerce Order System

```python
# Create tables
db.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        total REAL NOT NULL,
        status TEXT DEFAULT 'pending',
        created_at TEXT
    )
""")

db.execute("""
    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL
    )
""")

# Place order with transaction
txn = db.begin_transaction()
try:
    # Create order
    db.execute("""
        INSERT INTO orders (id, customer_id, total, created_at)
        VALUES (101, 1, 299.98, '2024-01-15')
    """)
    
    # Add items
    db.execute("""
        INSERT INTO order_items (id, order_id, product_id, quantity, price)
        VALUES (1, 101, 5, 2, 149.99)
    """)
    
    # Update inventory
    db.execute("UPDATE products SET stock = stock - 2 WHERE id = 5")
    
    txn.commit()
    print("Order placed successfully")
except Exception as e:
    txn.rollback()
    print(f"Order failed: {e}")

# Get customer's orders
result = db.execute("""
    SELECT o.id, o.total, o.status, o.created_at,
           GROUP_CONCAT(oi.product_id) as products
    FROM orders o
    LEFT JOIN order_items oi ON o.id = oi.order_id
    WHERE o.customer_id = 1
    GROUP BY o.id
    ORDER BY o.created_at DESC
""")
```

## Troubleshooting

### Query Too Slow?

1. Add indexes on filtered columns
2. Use LIMIT to reduce result size
3. Avoid SELECT * - choose specific columns
4. Check query plan with EXPLAIN

### Transaction Deadlocks?

1. Keep transactions short
2. Always access tables in same order
3. Use appropriate isolation level

### Data Not Appearing?

1. Check if transaction was committed
2. Verify WHERE conditions
3. Check for silent failures

## Next Steps

- Learn about [Vector Search](./vector-search.md) for semantic queries
- Explore [Hybrid Retrieval](../concepts/hybrid-retrieval.md) combining SQL and vectors
- See [Performance Guide](../concepts/performance.md) for optimization tips
- Check [SQL Examples](../../examples/) for more code

## See Also

- [SQL API Reference](../api-reference/sql-api.md)
- [Transaction Guide](./transactions.md)
- [Query Optimization](../concepts/query-optimization.md)
