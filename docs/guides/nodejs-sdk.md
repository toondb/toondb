# JavaScript/Node.js SDK Guide

> **🔧 Skill Level:** Beginner  
> **⏱️ Time Required:** 20 minutes  
> **📦 Requirements:** Node.js 18+, TypeScript 5+

Complete guide to ToonDB's JavaScript SDK with embedded/external modes, TypeScript support, and multi-tenant patterns.

---

## 📦 Installation

```bash
npm install toondb
# or
yarn add toondb
```

**Package includes:**
- Native binaries for all major platforms
- Full TypeScript definitions
- CLI tools: `toondb-server`, `toondb-bulk`, `toondb-grpc-server`

---

## 🚀 Quick Start

### Embedded Mode (Recommended)

```javascript
import { Database } from 'toondb';

async function main() {
  // Open database with embedded engine
  const db = await Database.open('./my_database');
  
  try {
    // Put and Get
    await db.put(Buffer.from('user:123'), Buffer.from('{"name":"Alice","age":30}'));
    const value = await db.get(Buffer.from('user:123'));
    console.log(value.toString());
    // Output: {"name":"Alice","age":30}
  } finally {
    await db.close();
  }
}

main();
```

**Output:**
```
{"name":"Alice","age":30}
```

---

## Embedded vs External

### Embedded Mode (Default)

Runs ToonDB engine in-process:

```javascript
const db = await Database.open('./my_db');
// ✅ Fast: No IPC overhead
// ✅ Simple: Single process
// ❌ Limited: One connection per database
```

### External Mode

Connects to standalone server:

```bash
# Terminal 1: Start server
toondb-server --db ./my_database --host 127.0.0.1 --port 5555
```

```javascript
// Terminal 2: Connect
import { IpcClient } from 'toondb';

const client = await IpcClient.connect({
  host: '127.0.0.1',
  port: 5555
});

await client.put(Buffer.from('key'), Buffer.from('value'));
// ✅ Multi-process: Many clients
// ✅ Remote: Network access
// ❌ Slower: IPC overhead
```

**When to use:**
- Embedded: Single app, local data, fast operations
- External: Microservices, multi-process, remote data

---

## Core Operations

### Basic K-V Operations

```javascript
const db = await Database.open('./my_db');

// Put
await db.put(Buffer.from('key'), Buffer.from('value'));

// Get
const value = await db.get(Buffer.from('key'));
console.log(value?.toString());
// Output: value

// Delete
await db.delete(Buffer.from('key'));

// Get after delete
const deletedValue = await db.get(Buffer.from('key'));
console.log(deletedValue);
// Output: null
```

**Output:**
```
value
null
```

### JSON Operations

```javascript
// Store JSON
const user = { name: 'Alice', email: 'alice@example.com', age: 30 };
await db.put(
  Buffer.from('users/alice'),
  Buffer.from(JSON.stringify(user))
);

// Retrieve JSON
const value = await db.get(Buffer.from('users/alice'));
if (value) {
  const retrievedUser = JSON.parse(value.toString());
  console.log(`Name: ${retrievedUser.name}, Age: ${retrievedUser.age}`);
}
```

**Output:**
```
Name: Alice, Age: 30
```

---

## Path API

⭐ **Fixed in 0.2.6** — Now uses correct wire format:

```javascript
// Store hierarchical data
await db.putPath('users/alice/email', Buffer.from('alice@example.com'));
await db.putPath('users/alice/age', Buffer.from('30'));
await db.putPath('users/alice/settings/theme', Buffer.from('dark'));

// Retrieve by path
const email = await db.getPath('users/alice/email');
console.log(`Alice's email: ${email?.toString()}`);

// Output: Alice's email: alice@example.com
```

**Output:**
```
Alice's email: alice@example.com
```

**Path Format (Wire Protocol):**
```
[path_count: 2 bytes LE]
[path_length_1: 2 bytes LE][path_1: UTF-8]
[path_length_2: 2 bytes LE][path_2: UTF-8]
...
```

---

## Prefix Scanning

⭐ **New in 0.2.6** — Multi-tenant isolation:

```javascript
// Insert multi-tenant data
await db.put(Buffer.from('tenants/acme/users/1'), Buffer.from('{"name":"Alice"}'));
await db.put(Buffer.from('tenants/acme/users/2'), Buffer.from('{"name":"Bob"}'));
await db.put(Buffer.from('tenants/acme/orders/1'), Buffer.from('{"total":100}'));
await db.put(Buffer.from('tenants/globex/users/1'), Buffer.from('{"name":"Charlie"}'));

// Scan only ACME Corp data
const acmeData = [];
for await (const [key, value] of db.scan(
  Buffer.from('tenants/acme/'),
  Buffer.from('tenants/acme;')
)) {
  acmeData.push([key.toString(), value.toString()]);
}

console.log(`ACME Corp has ${acmeData.length} items:`);
for (const [key, value] of acmeData) {
  console.log(`  ${key}: ${value}`);
}
```

**Output:**
```
ACME Corp has 3 items:
  tenants/acme/orders/1: {"total":100}
  tenants/acme/users/1: {"name":"Alice"}
  tenants/acme/users/2: {"name":"Bob"}
```

**Why use scan():**
- **Fast**: O(|prefix|) — only reads matching keys
- **Isolated**: Perfect for multi-tenancy
- **Efficient**: No full-table scan

**Range trick:**
```javascript
// Scan "users/" to "users;" (semicolon is after '/' in ASCII)
const start = Buffer.from('users/');
const end = Buffer.from('users;');
// Matches: users/1, users/2, users/alice, ...
// Excludes: user, users, usersabc
```

---

## Transactions

### Automatic Transactions

```javascript
// Atomic operations
const txn = await db.beginTransaction();
try {
  await txn.put(Buffer.from('account:1:balance'), Buffer.from('1000'));
  await txn.put(Buffer.from('account:2:balance'), Buffer.from('500'));
  
  await txn.commit();
  console.log('✅ Transaction committed');
} catch (error) {
  await txn.abort();
  console.error('❌ Transaction aborted:', error);
  throw error;
}
```

**Output:**
```
✅ Transaction committed
```

### Transaction with Scan

```javascript
const txn = await db.beginTransaction();
try {
  await txn.put(Buffer.from('key1'), Buffer.from('value1'));
  await txn.put(Buffer.from('key2'), Buffer.from('value2'));
  
  // Scan within transaction
  for await (const [key, value] of txn.scan(
    Buffer.from('key'),
    Buffer.from('key~')
  )) {
    console.log(`${key.toString()}: ${value.toString()}`);
  }
  
  await txn.commit();
} catch (error) {
  await txn.abort();
  throw error;
}
```

**Output:**
```
key1: value1
key2: value2
```

---

## Query Builder

Returns results in **TOON format** (token-optimized):

```javascript
// Insert structured data
await db.put(
  Buffer.from('products/laptop'),
  Buffer.from('{"name":"Laptop","price":999,"stock":5}')
);
await db.put(
  Buffer.from('products/mouse'),
  Buffer.from('{"name":"Mouse","price":25,"stock":20}')
);

// Query with column selection
const results = await db.query('products/')
  .select(['name', 'price'])
  .limit(10)
  .toList();

for (const [key, value] of results) {
  console.log(`${key.toString()}: ${value.toString()}`);
}
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
```

**TOON format benefits:**
- Fewer tokens for LLMs
- Structured output
- Easy parsing

---

## TypeScript Usage

### Type-Safe Operations

```typescript
import { Database, Transaction } from 'toondb';

interface User {
  name: string;
  email: string;
  age: number;
}

async function main() {
  const db = await Database.open('./my_db');
  
  try {
    // Store with type safety
    const user: User = {
      name: 'Alice',
      email: 'alice@example.com',
      age: 30
    };
    
    await db.put(
      Buffer.from('users/alice'),
      Buffer.from(JSON.stringify(user))
    );
    
    // Retrieve with type safety
    const value = await db.get(Buffer.from('users/alice'));
    if (value) {
      const retrievedUser: User = JSON.parse(value.toString());
      console.log(`User: ${retrievedUser.name} (${retrievedUser.email})`);
    }
  } finally {
    await db.close();
  }
}
```

**Output:**
```
User: Alice (alice@example.com)
```

### Generic Helper Functions

```typescript
class TypedDatabase {
  constructor(private db: Database) {}
  
  async putJSON<T>(key: string, value: T): Promise<void> {
    await this.db.put(
      Buffer.from(key),
      Buffer.from(JSON.stringify(value))
    );
  }
  
  async getJSON<T>(key: string): Promise<T | null> {
    const value = await this.db.get(Buffer.from(key));
    return value ? JSON.parse(value.toString()) : null;
  }
  
  async scanJSON<T>(prefix: string): Promise<Array<[string, T]>> {
    const results: Array<[string, T]> = [];
    for await (const [key, value] of this.db.scan(
      Buffer.from(prefix),
      Buffer.from(prefix.replace(/\/$/, ';'))
    )) {
      results.push([
        key.toString(),
        JSON.parse(value.toString())
      ]);
    }
    return results;
  }
}

// Usage
const db = await Database.open('./my_db');
const typedDb = new TypedDatabase(db);

await typedDb.putJSON('users/alice', { name: 'Alice', age: 30 });
const user = await typedDb.getJSON<User>('users/alice');
```

---

## Best Practices

### 1. Always Close Database

```javascript
// ✅ Good: Use try-finally
const db = await Database.open('./my_db');
try {
  await db.put(Buffer.from('key'), Buffer.from('value'));
} finally {
  await db.close();
}

// ❌ Bad: Might leak resources
const db = await Database.open('./my_db');
await db.put(Buffer.from('key'), Buffer.from('value'));
// Forgot to close!
```

### 2. Use scan() for Prefix Queries

```javascript
// ✅ Good: Efficient prefix scan
const results = [];
for await (const [key, value] of db.scan(
  Buffer.from('users/'),
  Buffer.from('users;')
)) {
  results.push([key, value]);
}

// ❌ Bad: Load all keys into memory
const allKeys = await db.getAllKeys(); // Don't do this!
const filtered = allKeys.filter(k => k.startsWith('users/'));
```

### 3. Use Transactions for Atomicity

```javascript
// ✅ Good: Atomic updates
const txn = await db.beginTransaction();
try {
  await txn.put(Buffer.from('counter'), Buffer.from('1'));
  await txn.put(Buffer.from('timestamp'), Buffer.from(Date.now().toString()));
  await txn.commit();
} catch (error) {
  await txn.abort();
  throw error;
}

// ❌ Bad: Partial updates possible
await db.put(Buffer.from('counter'), Buffer.from('1'));
// If error here, counter is updated but timestamp isn't
await db.put(Buffer.from('timestamp'), Buffer.from(Date.now().toString()));
```

### 4. Handle Errors Properly

```javascript
// ✅ Good: Proper error handling
try {
  const value = await db.get(Buffer.from('key'));
  if (value === null) {
    console.log('Key not found');
  } else {
    console.log('Value:', value.toString());
  }
} catch (error) {
  console.error('Database error:', error);
}

// ❌ Bad: Assuming success
const value = await db.get(Buffer.from('key'));
console.log(value.toString()); // Crashes if null!
```

### 5. Use Buffer for Binary Data

```javascript
// ✅ Good: Binary-safe
await db.put(Buffer.from('key'), Buffer.from([0x00, 0x01, 0x02]));

// ❌ Bad: String encoding issues
await db.put('key', '\x00\x01\x02'); // May corrupt data
```

---

## Complete Examples

### Example 1: Multi-Tenant SaaS Application

```javascript
import { Database } from 'toondb';

interface TenantUser {
  id: string;
  role: string;
  email: string;
}

async function main() {
  const db = await Database.open('./saas_db');
  
  try {
    // Insert tenant-specific data
    const tenants = [
      { id: 'acme', name: 'ACME Corp' },
      { id: 'globex', name: 'Globex Inc' }
    ];
    
    // ACME Corp users
    await db.put(
      Buffer.from('tenants/acme/users/alice'),
      Buffer.from(JSON.stringify({ id: 'alice', role: 'admin', email: 'alice@acme.com' }))
    );
    await db.put(
      Buffer.from('tenants/acme/users/bob'),
      Buffer.from(JSON.stringify({ id: 'bob', role: 'user', email: 'bob@acme.com' }))
    );
    
    // Globex Inc users
    await db.put(
      Buffer.from('tenants/globex/users/charlie'),
      Buffer.from(JSON.stringify({ id: 'charlie', role: 'admin', email: 'charlie@globex.com' }))
    );
    
    // Query each tenant's data in isolation
    for (const tenant of tenants) {
      const prefix = Buffer.from(`tenants/${tenant.id}/users/`);
      const end = Buffer.from(`tenants/${tenant.id}/users;`);
      
      const users: TenantUser[] = [];
      for await (const [key, value] of db.scan(prefix, end)) {
        users.push(JSON.parse(value.toString()));
      }
      
      console.log(`\n${tenant.name} (${users.length} users):`);
      for (const user of users) {
        console.log(`  ${user.email} (${user.role})`);
      }
    }
  } finally {
    await db.close();
  }
}

main().catch(console.error);
```

**Output:**
```
ACME Corp (2 users):
  alice@acme.com (admin)
  bob@acme.com (user)

Globex Inc (1 users):
  charlie@globex.com (admin)
```

### Example 2: SQL-Like Operations with K-V

```javascript
import { Database } from 'toondb';

interface Product {
  id: string;
  name: string;
  price: number;
  category: string;
}

async function main() {
  const db = await Database.open('./ecommerce');
  
  try {
    // INSERT: Store products
    const products: Product[] = [
      { id: '1', name: 'Laptop', price: 999.99, category: 'Electronics' },
      { id: '2', name: 'Mouse', price: 25.00, category: 'Electronics' },
      { id: '3', name: 'Desk', price: 299.99, category: 'Furniture' }
    ];
    
    for (const product of products) {
      await db.put(
        Buffer.from(`products/${product.id}`),
        Buffer.from(JSON.stringify(product))
      );
    }
    
    // SELECT: Retrieve all products
    console.log('All Products:');
    for await (const [key, value] of db.scan(
      Buffer.from('products/'),
      Buffer.from('products;')
    )) {
      const product: Product = JSON.parse(value.toString());
      console.log(`  ${product.name}: $${product.price}`);
    }
    
    // WHERE: Filter by category
    console.log('\nElectronics:');
    for await (const [key, value] of db.scan(
      Buffer.from('products/'),
      Buffer.from('products;')
    )) {
      const product: Product = JSON.parse(value.toString());
      if (product.category === 'Electronics') {
        console.log(`  ${product.name}: $${product.price}`);
      }
    }
    
    // UPDATE: Modify price
    const laptopValue = await db.get(Buffer.from('products/1'));
    if (laptopValue) {
      const laptop: Product = JSON.parse(laptopValue.toString());
      laptop.price = 899.99;
      await db.put(
        Buffer.from('products/1'),
        Buffer.from(JSON.stringify(laptop))
      );
      console.log(`\nUpdated ${laptop.name} price to $${laptop.price}`);
    }
    
    // DELETE: Remove product
    await db.delete(Buffer.from('products/2'));
    console.log('Deleted Mouse');
    
  } finally {
    await db.close();
  }
}

main().catch(console.error);
```

**Output:**
```
All Products:
  Laptop: $999.99
  Mouse: $25
  Desk: $299.99

Electronics:
  Laptop: $999.99
  Mouse: $25

Updated Laptop price to $899.99
Deleted Mouse
```

### Example 3: Session Cache

```javascript
import { Database } from 'toondb';

interface Session {
  userId: string;
  token: string;
  expiresAt: number;
}

class SessionStore {
  constructor(private db: Database) {}
  
  async create(userId: string, token: string, ttlMs: number): Promise<void> {
    const session: Session = {
      userId,
      token,
      expiresAt: Date.now() + ttlMs
    };
    
    await this.db.put(
      Buffer.from(`sessions/${token}`),
      Buffer.from(JSON.stringify(session))
    );
  }
  
  async get(token: string): Promise<Session | null> {
    const value = await this.db.get(Buffer.from(`sessions/${token}`));
    if (!value) return null;
    
    const session: Session = JSON.parse(value.toString());
    
    // Check expiration
    if (Date.now() > session.expiresAt) {
      await this.delete(token);
      return null;
    }
    
    return session;
  }
  
  async delete(token: string): Promise<void> {
    await this.db.delete(Buffer.from(`sessions/${token}`));
  }
  
  async cleanup(): Promise<number> {
    let removed = 0;
    const now = Date.now();
    
    for await (const [key, value] of this.db.scan(
      Buffer.from('sessions/'),
      Buffer.from('sessions;')
    )) {
      const session: Session = JSON.parse(value.toString());
      if (now > session.expiresAt) {
        await this.db.delete(key);
        removed++;
      }
    }
    
    return removed;
  }
}

async function main() {
  const db = await Database.open('./sessions');
  const store = new SessionStore(db);
  
  try {
    // Create sessions
    await store.create('user1', 'token123', 60000); // 1 minute
    await store.create('user2', 'token456', 120000); // 2 minutes
    
    console.log('Created 2 sessions');
    
    // Retrieve session
    const session = await store.get('token123');
    console.log(`Session for ${session?.userId}: expires in ${Math.round((session!.expiresAt - Date.now()) / 1000)}s`);
    
    // Cleanup expired
    const removed = await store.cleanup();
    console.log(`Cleaned up ${removed} expired sessions`);
    
  } finally {
    await db.close();
  }
}

main().catch(console.error);
```

**Output:**
```
Created 2 sessions
Session for user1: expires in 60s
Cleaned up 0 expired sessions
```

---

## API Reference

### Database

| Method | Description |
|--------|-------------|
| `Database.open(path)` | Open/create database |
| `put(key, value)` | Store key-value |
| `get(key)` | Retrieve value (null if not found) |
| `delete(key)` | Delete key |
| `putPath(path, value)` | Store by path ⭐ |
| `getPath(path)` | Get by path ⭐ |
| `scan(start, end)` | Iterate range (async iterator) ⭐ |
| `beginTransaction()` | Begin transaction |
| `query(prefix)` | Create query builder |
| `checkpoint()` | Force checkpoint |
| `close()` | Close database |

### Transaction

| Method | Description |
|--------|-------------|
| `put(key, value)` | Store in transaction |
| `get(key)` | Retrieve from transaction |
| `delete(key)` | Delete in transaction |
| `scan(start, end)` | Scan in transaction |
| `commit()` | Commit changes |
| `abort()` | Rollback changes |

### IpcClient

| Method | Description |
|--------|-------------|
| `IpcClient.connect(opts)` | Connect to server |
| `ping()` | Check latency |
| `put(key, value)` | Store key-value |
| `get(key)` | Retrieve value |
| `scan(prefix)` | Scan prefix |

---

## Migration from 0.2.5

### Path Operations

```javascript
// ❌ 0.2.5: Incorrect wire format
await db.putPath('users/alice', value);
// May have produced incorrect keys

// ✅ 0.2.6: Fixed wire format
await db.putPath('users/alice', value);
// Now correctly encodes path segments
```

### Scan Range

```javascript
// ❌ 0.2.5: Manual range calculation
const start = Buffer.from('users/');
const end = Buffer.from('users/' + '\xFF'.repeat(100));

// ✅ 0.2.6: Simple semicolon trick
const start = Buffer.from('users/');
const end = Buffer.from('users;'); // ';' is after '/' in ASCII
```

---

## Resources

- [JavaScript SDK GitHub](https://github.com/toondb/toondb/tree/main/toondb-js)
- [npm Package](https://www.npmjs.com/package/toondb)
- [TypeScript Definitions](https://github.com/toondb/toondb/blob/main/toondb-js/src/index.ts)
- [Go SDK](./go-sdk.md)
- [Python SDK](./python-sdk.md)
- [Rust SDK](./rust-sdk.md)

---

*Last updated: January 2026 (v0.2.9)*
