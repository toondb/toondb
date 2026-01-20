# SQL API Reference

SochDB provides full SQL support for relational data operations alongside its native key-value and vector search capabilities.

## Overview

SochDB's SQL engine supports:
- **SQL-92** core syntax
- **DDL**: CREATE TABLE, DROP TABLE, CREATE INDEX, DROP INDEX
- **DML**: SELECT, INSERT, UPDATE, DELETE
- **Transactions**: BEGIN, COMMIT, ROLLBACK
- **Query features**: WHERE, ORDER BY, LIMIT, OFFSET, GROUP BY, JOINs
- **Data types**: INTEGER, REAL, TEXT, BLOB, BOOLEAN, NULL

## Architecture

```
SQL Query → Parser → AST → Optimizer → Executor → Storage Layer
```

The SQL engine is built in `sochdb-query/src/sql/` and integrates seamlessly with SochDB's storage layer.

## Quick Start

### Python

```python
from sochdb import Database

db = Database("./mydb")

# Create table
db.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        age INTEGER
    )
""")

# Insert data
db.execute("INSERT INTO users (id, name, email, age) VALUES (1, 'Alice', 'alice@example.com', 30)")

# Query data
result = db.execute("SELECT * FROM users WHERE age > 25")
for row in result.rows:
    print(f"{row['name']}: {row['email']}")

# Update
db.execute("UPDATE users SET age = 31 WHERE name = 'Alice'")

# Delete
db.execute("DELETE FROM users WHERE age < 18")

db.close()
```

### Rust

```rust
use sochdb_client::Client;

let mut client = Client::open("./mydb")?;

// Create table
client.execute(r#"
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL,
        stock INTEGER DEFAULT 0
    )
"#)?;

// Insert data
client.execute("INSERT INTO products (id, name, price, stock) VALUES (1, 'Widget', 29.99, 100)")?;

// Query
let result = client.execute("SELECT name, price FROM products WHERE price < 50.0")?;
for row in &result.rows {
    println!("{:?}", row);
}

Ok(())
```

### TypeScript/Node.js

```typescript
import { Database } from '@sochdb/sochdb';

const db = new Database('./mydb');

// Create table
await db.execute(`
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        total REAL,
        status TEXT
    )
`);

// Insert
await db.execute("INSERT INTO orders (id, customer_id, total, status) VALUES (1, 101, 299.99, 'pending')");

// Query
const result = await db.execute("SELECT * FROM orders WHERE status = 'pending'");
result.rows.forEach(row => {
    console.log(`Order ${row.id}: $${row.total}`);
});

await db.close();
```

### Go

```go
import "github.com/sochdb/sochdb-go"

db, err := sochdb.Open("./mydb")
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Create table
_, err = db.Execute(`
    CREATE TABLE inventory (
        id INTEGER PRIMARY KEY,
        item TEXT NOT NULL,
        quantity INTEGER
    )
`)

// Insert
_, err = db.Execute("INSERT INTO inventory (id, item, quantity) VALUES (1, 'Laptop', 50)")

// Query
result, err := db.Execute("SELECT item, quantity FROM inventory WHERE quantity > 0")
for _, row := range result.Rows {
    fmt.Printf("%s: %d\n", row["item"], row["quantity"])
}
```

## DDL (Data Definition Language)

### CREATE TABLE

Create a new table with schema.

```sql
CREATE TABLE table_name (
    column1 datatype constraints,
    column2 datatype constraints,
    ...
    table_constraints
)
```

**Example:**

```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    department TEXT,
    salary REAL,
    hired_date TEXT,
    is_active BOOLEAN DEFAULT TRUE
)
```

**Supported Data Types:**
- `INTEGER` - 64-bit signed integer
- `REAL` - 64-bit floating point
- `TEXT` - UTF-8 string
- `BLOB` - Binary data
- `BOOLEAN` - True/false
- `NULL` - Null value

**Constraints:**
- `PRIMARY KEY` - Unique identifier
- `NOT NULL` - Cannot be null
- `UNIQUE` - Values must be unique
- `DEFAULT value` - Default value if not specified

### DROP TABLE

Remove a table and all its data.

```sql
DROP TABLE table_name
```

**Example:**

```sql
DROP TABLE employees
```

### CREATE INDEX

Create an index for faster queries.

```sql
CREATE INDEX index_name ON table_name (column1, column2, ...)
```

**Example:**

```sql
CREATE INDEX idx_email ON employees (email)
CREATE INDEX idx_dept_salary ON employees (department, salary)
```

### DROP INDEX

Remove an index.

```sql
DROP INDEX index_name
```

## DML (Data Manipulation Language)

### SELECT

Query data from tables.

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition
ORDER BY column [ASC|DESC]
LIMIT count
OFFSET skip
```

**Examples:**

```sql
-- Select all columns
SELECT * FROM users

-- Select specific columns
SELECT name, email FROM users

-- With WHERE clause
SELECT * FROM users WHERE age >= 18 AND status = 'active'

-- With ORDER BY
SELECT name, salary FROM employees ORDER BY salary DESC

-- With LIMIT
SELECT * FROM products ORDER BY price ASC LIMIT 10

-- With OFFSET and LIMIT (pagination)
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 40

-- With aggregate functions
SELECT COUNT(*) as total FROM users
SELECT AVG(salary) as avg_salary FROM employees
SELECT MAX(price) as max_price FROM products

-- With GROUP BY
SELECT department, COUNT(*) as count 
FROM employees 
GROUP BY department

-- With HAVING
SELECT department, AVG(salary) as avg_sal
FROM employees
GROUP BY department
HAVING avg_sal > 50000
```

**WHERE Operators:**
- `=`, `!=`, `<>` - Equality/inequality
- `<`, `<=`, `>`, `>=` - Comparison
- `AND`, `OR`, `NOT` - Logical
- `LIKE`, `NOT LIKE` - Pattern matching
- `IN`, `NOT IN` - Set membership
- `BETWEEN` - Range check
- `IS NULL`, `IS NOT NULL` - Null check

**Pattern Matching:**

```sql
-- Starts with 'A'
SELECT * FROM users WHERE name LIKE 'A%'

-- Ends with '.com'
SELECT * FROM users WHERE email LIKE '%.com'

-- Contains 'admin'
SELECT * FROM users WHERE email LIKE '%admin%'
```

### INSERT

Add new rows to a table.

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
```

**Examples:**

```sql
-- Insert single row
INSERT INTO users (id, name, email, age)
VALUES (1, 'Alice', 'alice@example.com', 30)

-- Insert multiple rows
INSERT INTO users (id, name, email, age) VALUES
    (2, 'Bob', 'bob@example.com', 25),
    (3, 'Charlie', 'charlie@example.com', 35),
    (4, 'Diana', 'diana@example.com', 28)

-- Insert with defaults
INSERT INTO products (id, name) VALUES (1, 'Widget')
-- Other columns get DEFAULT values
```

### UPDATE

Modify existing rows.

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition
```

**Examples:**

```sql
-- Update single field
UPDATE users SET age = 31 WHERE id = 1

-- Update multiple fields
UPDATE employees 
SET salary = 65000, department = 'Engineering'
WHERE id = 123

-- Update with expression
UPDATE products SET price = price * 1.1 WHERE category = 'electronics'

-- Update all rows (use with caution!)
UPDATE users SET status = 'active'
```

### DELETE

Remove rows from a table.

```sql
DELETE FROM table_name
WHERE condition
```

**Examples:**

```sql
-- Delete specific row
DELETE FROM users WHERE id = 1

-- Delete with condition
DELETE FROM sessions WHERE expires_at < '2024-01-01'

-- Delete all rows (use with caution!)
DELETE FROM temp_data
```

## Transactions

Execute multiple statements atomically.

### Python

```python
# Explicit transaction
txn = db.begin_transaction()
try:
    db.execute("INSERT INTO accounts (id, balance) VALUES (1, 1000)")
    db.execute("INSERT INTO accounts (id, balance) VALUES (2, 500)")
    txn.commit()
except Exception as e:
    txn.rollback()
    raise
```

### Rust

```rust
// Transaction with closure
client.with_transaction(|txn| {
    client.execute("INSERT INTO logs (message) VALUES ('Started')")?;
    client.execute("UPDATE counters SET value = value + 1")?;
    Ok(())
})?;
```

### TypeScript

```typescript
// Transaction with async/await
const txn = await db.beginTransaction();
try {
    await db.execute("INSERT INTO orders (id, total) VALUES (1, 100)");
    await db.execute("UPDATE inventory SET stock = stock - 1");
    await txn.commit();
} catch (error) {
    await txn.rollback();
    throw error;
}
```

## Advanced Features

### JOINs

Combine rows from multiple tables.

```sql
-- INNER JOIN
SELECT users.name, posts.title
FROM users
INNER JOIN posts ON users.id = posts.user_id

-- LEFT JOIN
SELECT users.name, COUNT(posts.id) as post_count
FROM users
LEFT JOIN posts ON users.id = posts.user_id
GROUP BY users.id

-- Multiple joins
SELECT u.name, p.title, c.content
FROM users u
INNER JOIN posts p ON u.id = p.user_id
INNER JOIN comments c ON p.id = c.post_id
```

### Subqueries

Use query results in another query.

```sql
-- Subquery in WHERE
SELECT name FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total > 1000)

-- Subquery in FROM
SELECT dept, avg_salary
FROM (
    SELECT department as dept, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
) WHERE avg_salary > 50000
```

### Common Table Expressions (CTEs)

Named subqueries for readability.

```sql
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 100000
)
SELECT department, COUNT(*) as count
FROM high_earners
GROUP BY department
```

## SQL Parser API

For advanced use cases, parse SQL directly:

### Rust

```rust
use sochdb_query::sql::{Parser, Statement};

// Parse SQL into AST
let sql = "SELECT * FROM users WHERE age > 25";
let stmt = Parser::parse(sql)?;

match stmt {
    Statement::Select(select) => {
        println!("Columns: {:?}", select.columns);
        println!("From: {:?}", select.from);
        println!("Where: {:?}", select.where_clause);
    }
    _ => {}
}

// Parse multiple statements
let stmts = Parser::parse_statements("INSERT INTO t VALUES (1); INSERT INTO t VALUES (2);")?;
for stmt in stmts {
    // Process each statement
}
```

### SQL Executor

Standalone SQL execution without database:

```rust
use sochdb_query::sql::{SqlExecutor, ExecutionResult};

let mut executor = SqlExecutor::new();

// Create table in memory
executor.execute("CREATE TABLE temp (id INTEGER, value TEXT)")?;

// Insert data
executor.execute("INSERT INTO temp VALUES (1, 'hello')")?;
executor.execute("INSERT INTO temp VALUES (2, 'world')")?;

// Query
let result = executor.execute("SELECT * FROM temp WHERE id > 0")?;
match result {
    ExecutionResult::Rows { columns, rows } => {
        for row in rows {
            println!("{:?}", row);
        }
    }
    _ => {}
}
```

## Performance Tips

1. **Use indexes** for frequently queried columns
2. **Batch inserts** in transactions for better throughput
3. **Select specific columns** instead of `SELECT *`
4. **Use LIMIT** for large result sets
5. **Analyze query plans** with EXPLAIN (when available)

## Error Handling

### Python

```python
try:
    result = db.execute("SELECT * FROM nonexistent")
except sochdb.DatabaseError as e:
    print(f"SQL error: {e}")
```

### Rust

```rust
match client.execute("SELECT * FROM users") {
    Ok(result) => { /* process result */ }
    Err(e) => eprintln!("Query failed: {}", e),
}
```

### TypeScript

```typescript
try {
    const result = await db.execute("SELECT * FROM users");
} catch (error) {
    console.error('Query error:', error);
}
```

## SQL vs Key-Value API

SochDB supports both paradigms:

| Feature | SQL API | Key-Value API |
|---------|---------|---------------|
| **Schema** | Required (CREATE TABLE) | Schema-free |
| **Queries** | Rich (WHERE, JOIN, etc.) | Prefix scans, range queries |
| **Use Case** | Structured data, analytics | Hierarchical keys, JSON docs |
| **Performance** | Optimized for complex queries | Ultra-fast point lookups |

**When to use SQL:**
- Structured, relational data
- Complex queries with multiple conditions
- Data analytics and reporting
- Standard SQL tools integration

**When to use Key-Value:**
- Hierarchical data (paths)
- High-throughput simple operations
- Document storage with flexible schema
- LLM context retrieval

## See Also

- [Query API Reference](./query-api.md)
- [Transaction Guide](../guides/transactions.md)
- [Performance Tuning](../concepts/performance.md)
- [SQL Examples](../../examples/)
