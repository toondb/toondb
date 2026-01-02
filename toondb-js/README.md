# ToonDB JavaScript SDK

[![npm version](https://badge.fury.io/js/%40toondb%2Ftoondb-js.svg)](https://www.npmjs.com/package/@toondb/toondb-js)
[![CI](https://github.com/toondb/toondb/actions/workflows/js-ci.yml/badge.svg)](https://github.com/toondb/toondb/actions/workflows/js-ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The official JavaScript/TypeScript SDK for **ToonDB** — a high-performance embedded document database with HNSW vector search and built-in multi-tenancy support.

## Features

- ✅ **Key-Value Store** — Simple `get()`/`put()`/`delete()` operations
- ✅ **Path-Native API** — Hierarchical keys like `users/alice/email`
- ✅ **Prefix Scanning** — Fast `scan()` for multi-tenant data isolation
- ✅ **SQL Support** — Full DDL/DML with CREATE, INSERT, SELECT, UPDATE, DELETE
- ✅ **Transactions** — ACID-compliant with automatic commit/abort
- ✅ **Query Builder** — Fluent API for complex queries (returns TOON format)
- ✅ **Vector Search** — HNSW approximate nearest neighbor search
- ✅ **TypeScript First** — Full type safety with `.d.ts` definitions
- ✅ **Dual Mode** — Embedded server or external server connection
- ✅ **Node.js + Bun** — Works with Node.js 18+ and Bun

## Installation

```bash
npm install @sushanth/toondb@0.2.9
# or
yarn add @sushanth/toondb@0.2.9
# or
bun add @sushanth/toondb@0.2.9
```

**Requirements:**
- Node.js 18+ or Bun 1.0+
- ToonDB server binaries (automatically installed)

## CLI Tools

Three CLI tools are globally available after installation (`npm install -g sushanth-toondb`):

### 1. toondb-server
Start the DB server for IPC (multi-process) mode.
```bash
toondb-server --db ./my_database
```

### 2. toondb-bulk
High-performance tool for heavy offline tasks (building indexes, converting formats).
```bash
toondb-bulk build-index --input vectors.npy --output index.hnsw
```

### 3. toondb-grpc-server
Dedicated gRPC server for high-throughput vector search.
```bash
toondb-grpc-server --port 50051
```

## Quick Start

### Embedded Mode (Recommended)

Database runs in the same process:

```typescript
import { Database } from 'sushanth-toondb';

const db = new Database('./my_database', {
  mode: 'embedded',
  createIfMissing: true
});

await db.open();

// Put and Get
await db.put(Buffer.from('user:123'), Buffer.from('{"name":"Alice","age":30}'));
const value = await db.get(Buffer.from('user:123'));
console.log(value?.toString());
// Output: {"name":"Alice","age":30}

await db.close();
```

### External Mode

Connect to a running ToonDB server:

```bash
# Terminal 1: Start server
./toondb-server --db ./my_database
# Output: [IpcServer] Listening on "./my_database/toondb.sock"
```

```typescript
import { Database } from 'sushanth-toondb';

const db = new Database('./my_database', {
  mode: 'external' // Connect to existing server
});

await db.open();
// Use db...
await db.close();
```

## Core Operations

### Basic Key-Value

```typescript
// Put
await db.put(Buffer.from('key'), Buffer.from('value'));

// Get
const value = await db.get(Buffer.from('key'));
if (!value) {
  console.log('Key not found');
} else {
  console.log(value.toString());
}

// Delete
await db.delete(Buffer.from('key'));
```

**Output:**
```
value
Key not found (after delete)
```

### Path Operations ⭐ Fixed in 0.2.6

```typescript
// Store hierarchical data
await db.putPath('users/alice/email', Buffer.from('alice@example.com'));
await db.putPath('users/alice/age', Buffer.from('30'));
await db.putPath('users/bob/email', Buffer.from('bob@example.com'));

// Retrieve by path
const email = await db.getPath('users/alice/email');
console.log(`Alice's email: ${email?.toString()}`);
```

**Output:**
```
Alice's email: alice@example.com
```

**Note:** In v0.2.5, this threw "Path segment truncated" error. Now fixed!

### Prefix Scanning ⭐ New in 0.2.6

The most efficient way to iterate keys with a common prefix:

```typescript
// Insert multi-tenant data
await db.put(Buffer.from('tenants/acme/users/1'), Buffer.from('{"name":"Alice"}'));
await db.put(Buffer.from('tenants/acme/users/2'), Buffer.from('{"name":"Bob"}'));
await db.put(Buffer.from('tenants/acme/orders/1'), Buffer.from('{"total":100}'));
await db.put(Buffer.from('tenants/globex/users/1'), Buffer.from('{"name":"Charlie"}'));

// Scan only ACME Corp's data
const results = await db.scan('tenants/acme/');
console.log(`ACME Corp has ${results.length} items:`);
results.forEach(kv => {
  console.log(`  ${kv.key.toString()}: ${kv.value.toString()}`);
});
```

**Output:**
```
ACME Corp has 3 items:
  tenants/acme/orders/1: {"total":100}
  tenants/acme/users/1: {"name":"Alice"}
  tenants/acme/users/2: {"name":"Bob"}
```

**Why use scan():**
- **Fast**: Binary protocol, O(|prefix|) performance
- **Isolated**: Perfect for multi-tenant apps
- **Efficient**: Returns raw Buffers (no JSON parsing)

## Transactions

```typescript
// Automatic commit/abort
await db.transaction(async (txn) => {
  await txn.put(Buffer.from('account:1:balance'), Buffer.from('1000'));
  await txn.put(Buffer.from('account:2:balance'), Buffer.from('500'));
  // Commits on success, aborts on error
});
```

**Output:**
```
✅ Transaction committed
```

**Manual control:**
```typescript
const txn = await db.beginTransaction();
try {
  await txn.put(Buffer.from('key1'), Buffer.from('value1'));
  await txn.put(Buffer.from('key2'), Buffer.from('value2'));
  await txn.commit();
} catch (err) {
  await txn.abort();
  throw err;
}
```

## Query Builder

Returns results in **TOON format** (token-optimized for LLMs):

```typescript
// Insert structured data
await db.put(Buffer.from('products/laptop'), Buffer.from('{"name":"Laptop","price":999}'));
await db.put(Buffer.from('products/mouse'), Buffer.from('{"name":"Mouse","price":25}'));

// Query with column selection
const results = await db.query('products/')
  .select(['name', 'price'])
  .limit(10)
  .execute();

results.forEach(kv => {
  console.log(`${kv.key.toString()}: ${kv.value.toString()}`);
});
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
```

**Other query methods:**
```typescript
const first = await db.query('products/').first();     // Get first result
const count = await db.query('products/').count();     // Count results
const exists = await db.query('products/').exists();   // Check existence
```

## SQL-Like Operations

While JavaScript SDK focuses on key-value operations, you can use query() for SQL-like operations:

```typescript
// INSERT-like: Store structured data
await db.put(Buffer.from('products/001'), Buffer.from('{"id":1,"name":"Laptop","price":999}'));
await db.put(Buffer.from('products/002'), Buffer.from('{"id":2,"name":"Mouse","price":25}'));

// SELECT-like: Query with column selection
const results = await db.query('products/')
  .select(['name', 'price'])  // SELECT name, price
  .limit(10)                   // LIMIT 10
  .execute();
```

**Output:**
```
SELECT name, price FROM products LIMIT 10:
products/001: result[1]{name,price}:Laptop,999
products/002: result[1]{name,price}:Mouse,25
```

**UPDATE-like:**
```typescript
// Get current value
const current = await db.get(Buffer.from('products/001'));
const product = JSON.parse(current.toString());

// Update
product.price = 899;
await db.put(Buffer.from('products/001'), Buffer.from(JSON.stringify(product)));
```

**DELETE-like:**
```typescript
await db.delete(Buffer.from('products/001'));
```

> **Note:** For full SQL support, use the SQLExecutor class (see SQL Operations section below).

## SQL Operations

**New in v0.2.7:** Full SQL engine with CREATE TABLE, INSERT, SELECT, UPDATE, DELETE.

```typescript
import { Database, SQLExecutor } from 'sushanth-toondb';

const db = await Database.open('./sql_db');
const sqlExecutor = new SQLExecutor(db);

// CREATE TABLE
const createResult = await sqlExecutor.execute(`
  CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER
  )
`);

// INSERT
await sqlExecutor.execute("INSERT INTO users (id, name, email, age) VALUES (1, 'Alice', 'alice@example.com', 30)");
await sqlExecutor.execute("INSERT INTO users (id, name, email, age) VALUES (2, 'Bob', 'bob@example.com', 25)");

// SELECT - Returns SQLQueryResult with .rows, .columns, .rowsAffected
const result = await sqlExecutor.execute('SELECT * FROM users');
console.log(`Found ${result.rows.length} users:`);
result.rows.forEach(row => {
  console.log(`  ${row.name}: ${row.email}`);
});

// SELECT with WHERE
const filtered = await sqlExecutor.execute("SELECT name, age FROM users WHERE age > 26");
filtered.rows.forEach(row => {
  console.log(`${row.name}: ${row.age} years old`);
});

// UPDATE
await sqlExecutor.execute("UPDATE users SET age = 31 WHERE name = 'Alice'");

// DELETE
await sqlExecutor.execute("DELETE FROM users WHERE age < 26");
```

**Output:**
```
Found 2 users:
  Alice: alice@example.com
  Bob: bob@example.com
Alice: 30 years old
```

**Important:** `execute()` returns a `SQLQueryResult` object with:
- `.rows` - Array of row objects (for SELECT queries)
- `.columns` - Array of column names
- `.rowsAffected` - Number of rows modified (for INSERT/UPDATE/DELETE)

## Vector Search

```typescript
import { VectorIndex } from 'sushanth-toondb';

// Create HNSW index using VectorIndex class
const indexPath = './my-vector-index';
const index = new VectorIndex(indexPath, {
  dimensions: 384,
  maxElements: 10000,
  m: 16,
  efConstruction: 100
});

// Build from embeddings - accepts number[][] (not Float32Array[])
const vectors = [
  [0.1, 0.2, 0.3, /* ... 384 dims */],
  [0.4, 0.5, 0.6, /* ... 384 dims */]
];
const labels = ['doc1', 'doc2'];
await index.bulkBuild(vectors, labels);

// Search - accepts number[] (converted internally to Float32Array)
const query = [0.15, 0.25, 0.35, /* ... 384 dims */];
const results = await index.query(query, 10); // k=10 (ef defaults to 64)

results.forEach((r, i) => {
  console.log(`${i + 1}. ${r.label} (distance: ${r.distance.toFixed(4)})`);
});
```

**Output:**
```
1. doc1 (distance: 0.0234)
2. doc2 (distance: 0.1567)
```

## Complete Example: Multi-Tenant App

```typescript
import { Database } from 'sushanth-toondb';

async function main() {
  const db = new Database('./multi_tenant_db', {
    mode: 'embedded',
    createIfMissing: true
  });
  await db.open();

  // Insert data for two tenants
  await db.put(
    Buffer.from('tenants/acme/users/alice'),
    Buffer.from('{"role":"admin"}')
  );
  await db.put(
    Buffer.from('tenants/acme/users/bob'),
    Buffer.from('{"role":"user"}')
  );
  await db.put(
    Buffer.from('tenants/globex/users/charlie'),
    Buffer.from('{"role":"admin"}')
  );

  // Scan ACME Corp data only (tenant isolation)
  const acmeData = await db.scan('tenants/acme/');
  console.log(`ACME Corp: ${acmeData.length} users`);
  acmeData.forEach(kv => {
    console.log(`  ${kv.key.toString()}: ${kv.value.toString()}`);
  });

  // Scan Globex Corp data
  const globexData = await db.scan('tenants/globex/');
  console.log(`\nGlobex Corp: ${globexData.length} users`);
  globexData.forEach(kv => {
    console.log(`  ${kv.key.toString()}: ${kv.value.toString()}`);
  });

  await db.close();
}

main();
```

**Output:**
```
ACME Corp: 2 users
  tenants/acme/users/alice: {"role":"admin"}
  tenants/acme/users/bob: {"role":"user"}

Globex Corp: 1 users
  tenants/globex/users/charlie: {"role":"admin"}
```

## Embedded vs External Mode

### Embedded Mode (Default)
✅ **Pros:**
- No separate server process needed
- Automatic lifecycle management
- Simpler deployment
- Better for single-app scenarios

❌ **Cons:**
- Database locked to one process
- Can't share across apps

```typescript
const db = new Database('./db', { mode: 'embedded' });
```

### External Mode
✅ **Pros:**
- Multiple clients can connect
- Server runs independently
- Better for microservices

❌ **Cons:**
- Must manage server process
- Extra network hop (Unix socket)

```typescript
const db = new Database('./db', { mode: 'external' });
```

## Error Handling

```typescript
try {
  const value = await db.get(Buffer.from('key'));
  if (!value) {
    console.log('Key not found (not an error)');
  }
} catch (err) {
  if (err.message.includes('Database is closed')) {
    console.error('Database not open!');
  } else if (err.message.includes('Connection failed')) {
    console.error('Server not running!');
  } else {
    console.error('Unknown error:', err);
  }
}
```

## Configuration Options

```typescript
const db = new Database('./my_database', {
  mode: 'embedded',           // 'embedded' | 'external'
  createIfMissing: true,      // Auto-create database
  walEnabled: true,           // Write-ahead logging
  syncMode: 'normal',         // 'full' | 'normal' | 'off'
  memtableSizeBytes: 64 * 1024 * 1024,  // 64MB
  serverPath: './toondb-server',        // Custom server binary
  timeout: 30000              // Connection timeout (ms)
});
```

## TypeScript Types

```typescript
import { Database, QueryBuilder, Transaction } from 'sushanth-toondb';

interface User {
  name: string;
  email: string;
}

// Type-safe helpers
async function getUser(db: Database, key: string): Promise<User | null> {
  const value = await db.get(Buffer.from(key));
  return value ? JSON.parse(value.toString()) : null;
}

async function putUser(db: Database, key: string, user: User): Promise<void> {
  await db.put(Buffer.from(key), Buffer.from(JSON.stringify(user)));
}
```

## Best Practices

✅ **Always close:** `await db.close()` to prevent resource leaks
✅ **Use transactions:** For atomic multi-key operations
✅ **Check null:** `value === null` means key doesn't exist
✅ **Use scan():** For prefix iteration (not query)
✅ **Multi-tenant:** Prefix keys with tenant ID
✅ **Buffer keys:** Always use Buffer for binary safety

## Testing

```bash
# Run tests
npm test

# Build
npm run build

# Type check
npm run typecheck
```

## Troubleshooting

**"Database is closed" error:**
```typescript
await db.open(); // Must call open() first!
```

**"Path segment truncated" (v0.2.5):**
- **Fixed in v0.2.6!** Upgrade: `npm install @sushanth/toondb@0.2.9`

**Server not found:**
```typescript
// Specify custom server path
const db = new Database('./db', {
  mode: 'embedded',
  serverPath: '/path/to/toondb-server'
});
```

## Migration from 0.2.5 → 0.2.6

**No breaking changes!** Just upgrade:

```bash
npm install @sushanth/toondb
```

**New features:**
- `scan()` method now available
- `putPath()` / `getPath()` now work correctly

## Building the Package

```bash
# Clone repo
git clone https://github.com/toondb/toondb
cd toondb/toondb-js

# Install dependencies
npm install

# Build
npm run build

# Create tarball
npm pack
# Creates: sushanth-toondb-0.2.9.tgz
```

## License

Apache License 2.0

## Links

- [Documentation](https://docs.toondb.dev)
- [Python SDK](../toondb-python-sdk)
- [Go SDK](../toondb-go)
- [GitHub](https://github.com/toondb/toondb)
- [npm Package](https://www.npmjs.com/package/toondb)

## Support

- GitHub Issues: https://github.com/toondb/toondb/issues
- Email: sushanth@toondb.dev
