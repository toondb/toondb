# ToonDB Node.js SDK

[![npm version](https://badge.fury.io/js/%40sushanth%2Ftoondb.svg)](https://badge.fury.io/js/%40sushanth%2Ftoondb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Node.js 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)

**ToonDB is an AI-native database with token-optimized output, O(|path|) lookups, built-in vector search, and durable transactions.**

JavaScript/TypeScript client SDK for [ToonDB](https://github.com/toondb/toondb) - the database optimized for LLM context retrieval.

## Installation

```bash
npm install @sushanth/toondb
```

**Zero compilation required** - pre-built binaries are bundled for all major platforms:
- Linux x86_64 (glibc ≥ 2.17)
- macOS ARM64 (Apple Silicon)
- macOS x86_64 (Intel)
- Windows x64

## Architecture

The SDK supports two modes of operation:

### Embedded Mode (Default)
The SDK automatically starts and manages a ToonDB server process:

```
┌─────────────────┐  auto-start   ┌──────────────────┐
│  Your Node.js   │ ──────────►   │  ToonDB Server   │
│   Application   │               │  (toondb-mcp)    │
└─────────────────┘               └──────────────────┘
         │                                 │
         │         IPC/Socket              │
         └────────────────────────────────►│
                                           ▼
                                   ┌──────────────────┐
                                   │  Storage Engine  │
                                   │  (LSM, WAL, etc) │
                                   └──────────────────┘
```

### External Server Mode
Connect to an existing ToonDB server (for multi-process access):

```typescript
const db = await Database.open({
  path: './my_database',
  embedded: false  // Don't start embedded server
});
```

## Features

### Core Database
- 🚀 **Embedded Mode**: Automatically manages server lifecycle (v0.2.4+)
- 📡 **IPC Mode**: Multi-process access via Unix domain sockets
- 📁 **Path-Native API**: Hierarchical data organization with O(|path|) lookups
- 💾 **ACID Transactions**: Full transaction support with snapshot isolation
- 🔍 **Range Scans**: Efficient prefix and range queries
- 🎯 **Token-Optimized**: TOON format output designed for LLM context windows

### High-Performance Vector Operations
- ⚡ **Bulk API**: High-throughput vector ingestion via native binary
- 🔢 **SIMD Kernels**: Auto-dispatched AVX2/NEON optimizations for distance calculations
- 📊 **Multi-Format Input**: Support for Float32Array and typed arrays
- 🔎 **HNSW Indexing**: Build and query approximate nearest neighbor indexes

### Distribution
- 📦 **Zero-Compile Install**: Pre-built Rust binaries bundled in package
- 🌍 **Cross-Platform**: Linux, macOS, Windows with automatic platform detection
- 📝 **TypeScript**: Full type definitions included
- ✅ **ESM & CommonJS**: Works with both module systems

## Quick Start

### Database Operations

```typescript
import { Database } from '@sushanth/toondb';

// Open a database (creates if doesn't exist, starts embedded server)
const db = await Database.open('./my_database');

// Simple key-value operations
await db.put(Buffer.from('user:123'), Buffer.from('{"name": "Alice", "email": "alice@example.com"}'));
const value = await db.get(Buffer.from('user:123'));

// Path-native API
await db.putPath('users/alice/email', Buffer.from('alice@example.com'));
const email = await db.getPath('users/alice/email');

// Transactions
await db.withTransaction(async (txn) => {
  await txn.put(Buffer.from('key1'), Buffer.from('value1'));
  await txn.put(Buffer.from('key2'), Buffer.from('value2'));
  // Automatically commits on success, aborts on exception
});

// Clean up (stops embedded server automatically)
await db.close();
```

### IPC Mode (For multi-process access)

```typescript
import { IpcClient } from '@sushanth/toondb';

// Connect to a running ToonDB IPC server
const client = await IpcClient.connect('/tmp/toondb.sock');

// Same API as Database
await client.put(Buffer.from('key'), Buffer.from('value'));
const value = await client.get(Buffer.from('key'));

// Query Builder
const results = await client.queryBuilder('users/')
  .limit(10)
  .select(['name', 'email'])
  .toList();

await client.close();
```

### Vector Search

```typescript
import { VectorIndex } from '@sushanth/toondb';

// Generate or load embeddings (10K × 768D)
const embeddings = new Float32Array(10000 * 768);
// ... fill with your embeddings

// Build HNSW index
const stats = await VectorIndex.bulkBuild(embeddings, {
  output: 'my_index.hnsw',
  dimension: 768,
  m: 16,
  efConstruction: 100,
});

console.log(`Built ${stats.vectors} vectors at ${stats.rate.toFixed(0)} vec/s`);

// Query the index
const query = new Float32Array(768);
// ... fill with query vector

const results = await VectorIndex.query('my_index.hnsw', query, {
  k: 10,
  efSearch: 64,
});

for (const neighbor of results) {
  console.log(`ID: ${neighbor.id}, Distance: ${neighbor.distance.toFixed(4)}`);
}
```

## API Reference

### Database

```typescript
class Database {
  // Open a database
  static async open(path: string | DatabaseConfig): Promise<Database>;
  
  // Key-value operations
  async get(key: Buffer | string): Promise<Buffer | null>;
  async put(key: Buffer | string, value: Buffer | string): Promise<void>;
  async delete(key: Buffer | string): Promise<void>;
  
  // Path operations
  async getPath(path: string): Promise<Buffer | null>;
  async putPath(path: string, value: Buffer | string): Promise<void>;
  
  // Query builder
  query(pathPrefix: string): Query;
  
  // Transactions
  async withTransaction<T>(fn: (txn: Transaction) => Promise<T>): Promise<T>;
  
  // Maintenance
  async checkpoint(): Promise<void>;
  async stats(): Promise<StorageStats>;
  async close(): Promise<void>;
}
```

### IpcClient

```typescript
class IpcClient {
  // Connect to IPC server
  static async connect(socketPath: string): Promise<IpcClient>;
  
  // Same operations as Database
  async get(key: Buffer): Promise<Buffer | null>;
  async put(key: Buffer, value: Buffer): Promise<void>;
  async delete(key: Buffer): Promise<void>;
  async getPath(path: string): Promise<Buffer | null>;
  async putPath(path: string, value: Buffer): Promise<void>;
  
  // Query
  async query(pathPrefix: string, options?: QueryOptions): Promise<string>;
  queryBuilder(pathPrefix: string): Query;
  
  // Transaction management
  async beginTransaction(): Promise<bigint>;
  async commitTransaction(txnId: bigint): Promise<void>;
  async abortTransaction(txnId: bigint): Promise<void>;
  
  // Utilities
  async ping(): Promise<boolean>;
  async close(): Promise<void>;
}
```

### Query

```typescript
class Query {
  limit(n: number): Query;
  offset(n: number): Query;
  select(columns: string[]): Query;
  
  async execute(): Promise<string>;      // Returns TOON format
  async toList(): Promise<object[]>;     // Returns parsed objects
  async first(): Promise<object | null>; // Returns first result
  async count(): Promise<number>;        // Returns count
}
```

### VectorIndex

```typescript
class VectorIndex {
  // Build an HNSW index
  static async bulkBuild(
    vectors: Float32Array,
    options: {
      output: string;
      dimension: number;
      m?: number;           // default: 16
      efConstruction?: number; // default: 100
      metric?: 'cosine' | 'euclidean' | 'dot';
    }
  ): Promise<BulkBuildStats>;
  
  // Query an index
  static async query(
    indexPath: string,
    query: Float32Array,
    options?: {
      k?: number;        // default: 10
      efSearch?: number; // default: 64
    }
  ): Promise<VectorSearchResult[]>;
  
  // Get index metadata
  static async info(indexPath: string): Promise<IndexInfo>;
}
```

## SDKs

| Platform | Package | Install |
|----------|---------|---------|
| Rust | [`toondb`](https://crates.io/crates/toondb) | `cargo add toondb` |
| Python | [`toondb-client`](https://pypi.org/project/toondb-client/) | `pip install toondb-client` |
| JavaScript | [`@sushanth/toondb`](https://www.npmjs.com/package/@sushanth/toondb) | `npm install @sushanth/toondb` |

## TypeScript Support

Full TypeScript definitions are included. Import types as needed:

```typescript
import {
  Database,
  DatabaseConfig,
  IpcClient,
  Query,
  QueryResult,
  VectorIndex,
  VectorSearchResult,
  ToonDBError,
  ConnectionError,
  TransactionError,
} from '@sushanth/toondb';
```

## Error Handling

All errors extend from `ToonDBError`:

```typescript
import { ToonDBError, ConnectionError, TransactionError } from '@sushanth/toondb';

try {
  await db.get(Buffer.from('key'));
} catch (error) {
  if (error instanceof ConnectionError) {
    console.error('Connection failed:', error.message);
  } else if (error instanceof TransactionError) {
    console.error('Transaction failed:', error.message);
  } else if (error instanceof ToonDBError) {
    console.error('Database error:', error.message);
  }
}
```

## Development

```bash
# Clone the repository
git clone https://github.com/toondb/toondb.git
cd toondb/toondb-js

# Install dependencies
npm install

# Build
npm run build

# Test
npm test

# Lint
npm run lint
```

## License

Apache-2.0
