# ToonDB SQL Surface & Compatibility

This document defines ToonDB's SQL dialect support and the canonical pipeline for SQL execution.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SQL Query Lifecycle                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │   SQL Text  │ --> │   Lexer     │ --> │   Parser    │                │
│  │ (any dialect)│    │ (tokenize)  │     │ (parse)     │                │
│  └─────────────┘     └─────────────┘     └─────────────┘                │
│                                                 │                        │
│                                                 v                        │
│                             ┌───────────────────────────────────────┐   │
│                             │        Canonical AST                   │   │
│                             │  (dialect-normalized representation)   │   │
│                             └───────────────────────────────────────┘   │
│                                                 │                        │
│                   ┌─────────────────────────────┼─────────────────────┐ │
│                   │                             │                     │ │
│                   v                             v                     v │
│  ┌─────────────────────────┐   ┌─────────────────────────┐   ┌───────┐ │
│  │     Validator           │   │      Planner            │   │Executor│ │
│  │ (semantic checks)       │   │ (optimize + plan)       │   │ (run)  │ │
│  └─────────────────────────┘   └─────────────────────────┘   └───────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core SQL Support (Guaranteed)

### Data Manipulation Language (DML)

| Statement | Support | Example |
|-----------|---------|---------|
| `SELECT` | ✅ Full | `SELECT id, name FROM users WHERE age > 21` |
| `INSERT` | ✅ Full | `INSERT INTO users (id, name) VALUES (1, 'Alice')` |
| `UPDATE` | ✅ Full | `UPDATE users SET name = 'Bob' WHERE id = 1` |
| `DELETE` | ✅ Full | `DELETE FROM users WHERE id = 1` |

### Data Definition Language (DDL)

| Statement | Support | Example |
|-----------|---------|---------|
| `CREATE TABLE` | ✅ Full | `CREATE TABLE users (id INT PRIMARY KEY, name TEXT)` |
| `DROP TABLE` | ✅ Full | `DROP TABLE users` |
| `ALTER TABLE` | 🔄 Partial | `ALTER TABLE users ADD COLUMN email TEXT` |
| `CREATE INDEX` | ✅ Full | `CREATE INDEX idx_name ON users (name)` |
| `DROP INDEX` | ✅ Full | `DROP INDEX idx_name` |

### Idempotent DDL

| Statement | Support | Behavior |
|-----------|---------|----------|
| `CREATE TABLE IF NOT EXISTS` | ✅ Full | No-op if table exists |
| `DROP TABLE IF EXISTS` | ✅ Full | No-op if table doesn't exist |
| `CREATE INDEX IF NOT EXISTS` | ✅ Full | No-op if index exists |
| `DROP INDEX IF EXISTS` | ✅ Full | No-op if index doesn't exist |

### Transactions

| Statement | Support | Notes |
|-----------|---------|-------|
| `BEGIN` | ✅ Full | Start transaction |
| `COMMIT` | ✅ Full | Commit transaction |
| `ROLLBACK` | ✅ Full | Rollback transaction |
| `SAVEPOINT` | 🔄 Partial | Named savepoints |

## Dialect Compatibility (Conflict/Upsert Family)

ToonDB normalizes dialect-specific INSERT variants to a canonical AST representation.

### PostgreSQL Style
```sql
-- Do nothing on conflict
INSERT INTO users (id, name) VALUES (1, 'Alice')
ON CONFLICT DO NOTHING;

-- Update on conflict (specific columns)
INSERT INTO users (id, name) VALUES (1, 'Alice')
ON CONFLICT (id) DO UPDATE SET name = 'Bob';
```

### MySQL Style
```sql
-- Ignore on duplicate (equivalent to ON CONFLICT DO NOTHING)
INSERT IGNORE INTO users (id, name) VALUES (1, 'Alice');

-- Update on duplicate key
INSERT INTO users (id, name) VALUES (1, 'Alice')
ON DUPLICATE KEY UPDATE name = 'Bob';
```

### SQLite Style
```sql
-- Ignore on conflict
INSERT OR IGNORE INTO users (id, name) VALUES (1, 'Alice');

-- Replace on conflict (delete + insert)
INSERT OR REPLACE INTO users (id, name) VALUES (1, 'Alice');

-- Abort on conflict (default behavior)
INSERT OR ABORT INTO users (id, name) VALUES (1, 'Alice');

-- Fail on conflict (fail but continue batch)
INSERT OR FAIL INTO users (id, name) VALUES (1, 'Alice');
```

### Internal Representation

All dialect forms normalize to:
```rust
InsertStmt {
    on_conflict: Some(OnConflict {
        target: Option<ConflictTarget>,  // (id) or ON CONSTRAINT name
        action: ConflictAction,          // DoNothing, DoUpdate(...), DoReplace, etc.
    })
}
```

## Parameterized Queries

ToonDB supports two placeholder styles:

### Positional Placeholders (`$1`, `$2`, ...)
```sql
SELECT * FROM users WHERE id = $1 AND name = $2
```

### Question Mark Placeholders (`?`)
```sql
SELECT * FROM users WHERE id = ? AND name = ?
```

Question marks are automatically indexed (1, 2, 3...) during lexing.

### Parameter Binding
```rust
let result = executor.execute_with_params(
    "SELECT * FROM users WHERE id = $1",
    &[ToonValue::Int(42)]
)?;
```

## Query Features

### SELECT Clauses
- `DISTINCT` / `ALL`
- `FROM` with table aliases
- `WHERE` with complex predicates
- `GROUP BY` with `HAVING`
- `ORDER BY` with `ASC`/`DESC`/`NULLS FIRST`/`NULLS LAST`
- `LIMIT` and `OFFSET`
- Set operations: `UNION`, `INTERSECT`, `EXCEPT`

### Expressions
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`
- Logical: `AND`, `OR`, `NOT`
- `IS NULL`, `IS NOT NULL`
- `IN (...)`, `NOT IN (...)`
- `BETWEEN ... AND ...`
- `LIKE` with wildcards
- `CASE WHEN ... THEN ... ELSE ... END`
- `CAST(expr AS type)`
- Function calls: `COUNT()`, `SUM()`, `AVG()`, etc.

### ToonDB Extensions
- `VECTOR(dimensions)` data type
- `EMBEDDING(dimensions)` data type
- `VECTOR_SEARCH(column, query, k, metric)` function
- `CONTEXT_WINDOW` for LLM context management

## Explicit Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-table JOINs | 🔄 Partial | Two-table INNER JOIN only |
| LEFT/RIGHT/FULL JOIN | 📋 Planned | Not yet implemented |
| Correlated subqueries | ❌ | Out of scope |
| Window functions | 📋 Planned | Future enhancement |
| CTEs (WITH clause) | 📋 Planned | Future enhancement |
| Stored procedures | ❌ | Out of scope |
| Triggers | ❌ | Out of scope |

## Complexity Analysis

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Lexing | O(n) | n = input length |
| Parsing | O(n) | n = token count |
| AST rewriting | O(|AST|) | Linear in AST size |
| SELECT (no index) | O(N) | N = table rows |
| SELECT (with index) | O(log N + K) | K = result rows |
| INSERT | O(log N) | B-Tree index update |
| INSERT (conflict check) | O(log N) | Uniqueness check |
| UPDATE (no index) | O(N) | Full scan |
| UPDATE (with index) | O(log N + K) | K = affected rows |

## Client Usage

### Using the AST-Based Query Executor

The recommended way to execute SQL is via the AST-based query executor:

```rust
use toondb::connection::ToonConnection;
use toondb::ast_query::QueryResult;

let conn = ToonConnection::open("./data")?;

// Execute SQL using AST-based parser (recommended)
match conn.query_ast("SELECT * FROM users WHERE active = true")? {
    QueryResult::Select(rows) => {
        for row in rows {
            println!("{:?}", row);
        }
    }
    _ => {}
}

// Execute with parameters
let result = conn.query_ast_params(
    "INSERT INTO users (id, name) VALUES ($1, $2)",
    &[ToonValue::Int(1), ToonValue::Text("Alice".to_string())]
)?;

// Execute non-query SQL (INSERT, UPDATE, DELETE)
let rows_affected = conn.execute_ast("DELETE FROM users WHERE id = 1")?;
```

### Dialect Support

The AST-based executor automatically normalizes dialect-specific syntax:

```rust
// All of these normalize to the same canonical AST:

// MySQL style
conn.execute_ast("INSERT IGNORE INTO users VALUES (1, 'Alice')")?;

// PostgreSQL style  
conn.execute_ast("INSERT INTO users VALUES (1, 'Alice') ON CONFLICT DO NOTHING")?;

// SQLite style
conn.execute_ast("INSERT OR IGNORE INTO users VALUES (1, 'Alice')")?;
```

## Files

- `toondb-query/src/sql/compatibility.rs` - Feature matrix and dialect detection
- `toondb-query/src/sql/token.rs` - Token types including dialect keywords
- `toondb-query/src/sql/lexer.rs` - Tokenizer with placeholder indexing
- `toondb-query/src/sql/ast.rs` - Canonical AST definitions
- `toondb-query/src/sql/parser.rs` - Recursive descent parser
- `toondb-query/src/sql/bridge.rs` - Unified execution pipeline
- `toondb-query/src/sql/error.rs` - Error types
- `toondb-query/src/sql/mod.rs` - Module exports
- `toondb-client/src/ast_query.rs` - AST-based client query executor
