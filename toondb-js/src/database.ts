/**
 * ToonDB Embedded Database
 *
 * Direct database access via IPC to the ToonDB server.
 * This provides the same API as the Python SDK's Database class.
 *
 * @packageDocumentation
 */

// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import * as fs from 'fs';
import * as path from 'path';
import { DatabaseError, TransactionError } from './errors';
import { IpcClient } from './ipc-client';
import { Query } from './query';
import { startEmbeddedServer, stopEmbeddedServer } from './server-manager';

/**
 * Configuration options for the Database.
 */
export interface DatabaseConfig {
  /** Path to the database directory */
  path: string;
  /** Whether to create the database if it doesn't exist (default: true) */
  createIfMissing?: boolean;
  /** Enable WAL (Write-Ahead Logging) for durability (default: true) */
  walEnabled?: boolean;
  /** Sync mode: 'full' | 'normal' | 'off' (default: 'normal') */
  syncMode?: 'full' | 'normal' | 'off';
  /** Maximum size of the memtable before flushing (default: 64MB) */
  memtableSizeBytes?: number;
  /** 
   * Whether to automatically start an embedded server (default: true)
   * Set to false if connecting to an existing external server
   */
  embedded?: boolean;
}

/**
 * Result of a SQL query execution.
 */
export interface SQLQueryResult {
  /** Result rows */
  rows: Array<Record<string, any>>;
  /** Column names */
  columns: string[];
  /** Number of rows affected (for INSERT/UPDATE/DELETE) */
  rowsAffected: number;
}

/**
 * Transaction handle for atomic operations.
 */
export class Transaction {
  private _db: Database;
  private _txnId: bigint | null = null;
  private _committed = false;
  private _aborted = false;

  constructor(db: Database) {
    this._db = db;
  }

  /**
   * Begin the transaction.
   * @internal
   */
  async begin(): Promise<void> {
    this._txnId = await this._db['_beginTransaction']();
  }

  /**
   * Get a value by key within this transaction.
   */
  async get(key: Buffer | string): Promise<Buffer | null> {
    this._ensureActive();
    return this._db.get(key);
  }

  /**
   * Put a key-value pair within this transaction.
   */
  async put(key: Buffer | string, value: Buffer | string): Promise<void> {
    this._ensureActive();
    return this._db.put(key, value);
  }

  /**
   * Delete a key within this transaction.
   */
  async delete(key: Buffer | string): Promise<void> {
    this._ensureActive();
    return this._db.delete(key);
  }

  /**
   * Scan keys with a prefix within this transaction.
   * @param prefix - The prefix to scan for
   * @param end - Optional end boundary (exclusive)
   */
  async *scan(prefix: string | Buffer, end?: string | Buffer): AsyncGenerator<[Buffer, Buffer]> {
    this._ensureActive();
    // Delegate to database's scan method
    // Transactional isolation is maintained by the underlying storage
    for await (const entry of this._db.scanGenerator(prefix, end)) {
      yield entry;
    }
  }

  /**
   * Get a value by path within this transaction.
   */
  async getPath(pathStr: string): Promise<Buffer | null> {
    this._ensureActive();
    return this._db.getPath(pathStr);
  }

  /**
   * Put a value at a path within this transaction.
   */
  async putPath(pathStr: string, value: Buffer | string): Promise<void> {
    this._ensureActive();
    return this._db.putPath(pathStr, value);
  }

  /**
   * Commit the transaction.
   * 
   * After committing, an optional checkpoint is triggered to ensure writes
   * are durable. This prevents race conditions where subsequent reads might
   * not see committed data due to async flush timing.
   */
  async commit(): Promise<void> {
    this._ensureActive();
    if (this._txnId !== null) {
      await this._db['_commitTransaction'](this._txnId);
      // Trigger checkpoint to ensure durability (addresses race condition)
      // Note: This is a trade-off between consistency and performance.
      // For high-throughput scenarios, consider batching checkpoints.
      await this._db.checkpoint();
    }
    this._committed = true;
  }

  /**
   * Abort/rollback the transaction.
   */
  async abort(): Promise<void> {
    if (this._committed || this._aborted) return;
    if (this._txnId !== null) {
      await this._db['_abortTransaction'](this._txnId);
    }
    this._aborted = true;
  }

  private _ensureActive(): void {
    if (this._committed) {
      throw new TransactionError('Transaction already committed');
    }
    if (this._aborted) {
      throw new TransactionError('Transaction already aborted');
    }
  }
}

/**
 * ToonDB Database client.
 *
 * Provides access to ToonDB with full transaction support.
 *
 * @example
 * ```typescript
 * import { Database } from '@sushanth/toondb';
 *
 * // Open a database
 * const db = await Database.open('./my_database');
 *
 * // Simple key-value operations
 * await db.put(Buffer.from('user:123'), Buffer.from('{"name": "Alice"}'));
 * const value = await db.get(Buffer.from('user:123'));
 *
 * // Path-native API
 * await db.putPath('users/alice/email', Buffer.from('alice@example.com'));
 * const email = await db.getPath('users/alice/email');
 *
 * // Transactions
 * await db.withTransaction(async (txn) => {
 *   await txn.put(Buffer.from('key1'), Buffer.from('value1'));
 *   await txn.put(Buffer.from('key2'), Buffer.from('value2'));
 * });
 *
 * // Clean up
 * await db.close();
 * ```
 */
export class Database {
  private _client: IpcClient | null = null;
  private _config: DatabaseConfig;
  private _closed = false;
  private _embeddedServerStarted = false;

  private constructor(config: DatabaseConfig) {
    this._config = {
      createIfMissing: true,
      walEnabled: true,
      syncMode: 'normal',
      memtableSizeBytes: 64 * 1024 * 1024,
      embedded: true,  // Default to embedded mode
      ...config,
    };
  }

  /**
   * Open a database at the specified path.
   *
   * @param pathOrConfig - Path to the database directory or configuration object
   * @returns A new Database instance
   *
   * @example
   * ```typescript
   * // Simple usage (embedded mode - starts server automatically)
   * const db = await Database.open('./my_database');
   *
   * // With configuration
   * const db = await Database.open({
   *   path: './my_database',
   *   walEnabled: true,
   *   syncMode: 'full',
   * });
   * 
   * // Connect to existing external server
   * const db = await Database.open({
   *   path: './my_database',
   *   embedded: false,  // Don't start embedded server
   * });
   * ```
   */
  static async open(pathOrConfig: string | DatabaseConfig): Promise<Database> {
    const config: DatabaseConfig =
      typeof pathOrConfig === 'string' ? { path: pathOrConfig } : pathOrConfig;

    // Ensure database directory exists
    if (config.createIfMissing !== false) {
      if (!fs.existsSync(config.path)) {
        fs.mkdirSync(config.path, { recursive: true });
      }
    }

    const db = new Database(config);

    // Start embedded server if configured (default: true)
    let socketPath: string;
    if (db._config.embedded !== false) {
      // Start embedded server and get socket path
      socketPath = await startEmbeddedServer(config.path);
      db._embeddedServerStarted = true;
    } else {
      // Connect to existing server socket
      socketPath = path.join(config.path, 'toondb.sock');
    }

    db._client = await IpcClient.connect(socketPath);

    return db;
  }

  /**
   * Get a value by key.
   *
   * @param key - The key to look up (Buffer or string)
   * @returns The value as a Buffer, or null if not found
   */
  async get(key: Buffer | string): Promise<Buffer | null> {
    this._ensureOpen();
    const keyBuf = typeof key === 'string' ? Buffer.from(key) : key;
    return this._client!.get(keyBuf);
  }

  /**
   * Put a key-value pair.
   *
   * @param key - The key (Buffer or string)
   * @param value - The value (Buffer or string)
   */
  async put(key: Buffer | string, value: Buffer | string): Promise<void> {
    this._ensureOpen();
    const keyBuf = typeof key === 'string' ? Buffer.from(key) : key;
    const valueBuf = typeof value === 'string' ? Buffer.from(value) : value;
    return this._client!.put(keyBuf, valueBuf);
  }

  /**
   * Delete a key.
   *
   * @param key - The key to delete (Buffer or string)
   */
  async delete(key: Buffer | string): Promise<void> {
    this._ensureOpen();
    const keyBuf = typeof key === 'string' ? Buffer.from(key) : key;
    return this._client!.delete(keyBuf);
  }

  /**
   * Get a value by path.
   *
   * @param pathStr - The path (e.g., "users/alice/email")
   * @returns The value as a Buffer, or null if not found
   */
  async getPath(pathStr: string): Promise<Buffer | null> {
    this._ensureOpen();
    return this._client!.getPath(pathStr);
  }

  /**
   * Put a value at a path.
   *
   * @param pathStr - The path (e.g., "users/alice/email")
   * @param value - The value (Buffer or string)
   */
  async putPath(pathStr: string, value: Buffer | string): Promise<void> {
    this._ensureOpen();
    const valueBuf = typeof value === 'string' ? Buffer.from(value) : value;
    return this._client!.putPath(pathStr, valueBuf);
  }

  /**
   * Create a query builder for the given path prefix.
   *
   * @param pathPrefix - The path prefix to query (e.g., "users/")
   * @returns A Query builder instance
   *
   * @example
   * ```typescript
   * const results = await db.query('users/')
   *   .limit(10)
   *   .select(['name', 'email'])
   *   .execute();
   * ```
   */
  query(pathPrefix: string): Query {
    this._ensureOpen();
    return new Query(this._client!, pathPrefix);
  }

  /**
   * Scan for keys with a prefix, returning key-value pairs.
   * This is the preferred method for simple prefix-based iteration.
   *
   * @param prefix - The prefix to scan for (e.g., "users/", "tenants/tenant1/")
   * @returns Array of key-value pairs
   *
   * @example
   * ```typescript
   * const results = await db.scan('tenants/tenant1/');
   * for (const { key, value } of results) {
   *   console.log(`${key.toString()}: ${value.toString()}`);
   * }
   * ```
   */
  async scan(prefix: string): Promise<Array<{ key: Buffer; value: Buffer }>> {
    this._ensureOpen();
    return this._client!.scan(prefix);
  }

  /**
   * Scan for keys with a prefix using an async generator.
   * This allows for memory-efficient iteration over large result sets.
   *
   * @param prefix - The prefix to scan for
   * @param end - Optional end boundary (exclusive)
   * @returns Async generator yielding [key, value] tuples
   *
   * @example
   * ```typescript
   * for await (const [key, value] of db.scanGenerator('users/')) {
   *   console.log(`${key.toString()}: ${value.toString()}`);
   * }
   * ```
   */
  async *scanGenerator(prefix: string | Buffer, end?: string | Buffer): AsyncGenerator<[Buffer, Buffer]> {
    this._ensureOpen();
    const prefixBuf = typeof prefix === 'string' ? Buffer.from(prefix) : prefix;
    const endBuf = end ? (typeof end === 'string' ? Buffer.from(end) : end) : undefined;
    
    const results = await this._client!.scan(prefixBuf.toString());
    for (const { key, value } of results) {
      // Filter by end boundary if provided
      if (endBuf && Buffer.compare(key, endBuf) >= 0) {
        break;
      }
      yield [key, value];
    }
  }

  /**
   * Execute operations within a transaction.
   *
   * The transaction automatically commits on success or aborts on error.
   *
   * @param fn - Async function that receives a Transaction object
   *
   * @example
   * ```typescript
   * await db.withTransaction(async (txn) => {
   *   await txn.put(Buffer.from('key1'), Buffer.from('value1'));
   *   await txn.put(Buffer.from('key2'), Buffer.from('value2'));
   *   // Automatically commits
   * });
   * ```
   */
  async withTransaction<T>(fn: (txn: Transaction) => Promise<T>): Promise<T> {
    this._ensureOpen();
    const txn = new Transaction(this);
    await txn.begin();
    try {
      const result = await fn(txn);
      await txn.commit();
      return result;
    } catch (error) {
      await txn.abort();
      throw error;
    }
  }

  /**
   * Create a new transaction.
   *
   * @returns A new Transaction instance
   * @deprecated Use withTransaction() for automatic commit/abort handling
   */
  async transaction(): Promise<Transaction> {
    this._ensureOpen();
    const txn = new Transaction(this);
    await txn.begin();
    return txn;
  }

  /**
   * Force a checkpoint to persist memtable to disk.
   */
  async checkpoint(): Promise<void> {
    this._ensureOpen();
    return this._client!.checkpoint();
  }

  /**
   * Get storage statistics.
   */
  async stats(): Promise<{
    memtableSizeBytes: number;
    walSizeBytes: number;
    activeTransactions: number;
  }> {
    this._ensureOpen();
    return this._client!.stats();
  }

  /**
   * Execute a SQL query.
   * 
   * ToonDB supports a subset of SQL for relational data stored on top of 
   * the key-value engine. Tables and rows are stored as:
   * - Schema: _sql/tables/{table_name}/schema
   * - Rows: _sql/tables/{table_name}/rows/{row_id}
   * 
   * Supported SQL:
   * - CREATE TABLE table_name (col1 TYPE, col2 TYPE, ...)
   * - DROP TABLE table_name
   * - INSERT INTO table_name (cols) VALUES (vals)
   * - SELECT cols FROM table_name [WHERE ...] [ORDER BY ...] [LIMIT ...]
   * - UPDATE table_name SET col=val [WHERE ...]
   * - DELETE FROM table_name [WHERE ...]
   * 
   * Supported types: INT, TEXT, FLOAT, BOOL, BLOB
   * 
   * @param sql - SQL query string
   * @returns SQLQueryResult with rows and metadata
   * 
   * @example
   * ```typescript
   * // Create a table
   * await db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)");
   * 
   * // Insert data
   * await db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)");
   * 
   * // Query data
   * const result = await db.execute("SELECT * FROM users WHERE age > 26");
   * result.rows.forEach(row => console.log(row));
   * ```
   */
  async execute(sql: string): Promise<SQLQueryResult> {
    this._ensureOpen();
    
    // Import the SQL executor
    const { SQLExecutor } = await import('./sql-engine.js');
    
    // Create a database adapter for the SQL executor
    const dbAdapter = {
      get: (key: Buffer | string) => this.get(key),
      put: (key: Buffer | string, value: Buffer | string) => this.put(key, value),
      delete: (key: Buffer | string) => this.delete(key),
      scan: (prefix: string) => this.scan(prefix),
    };
    
    const executor = new SQLExecutor(dbAdapter);
    return executor.execute(sql);
  }

  /**
   * Close the database connection.
   * If running in embedded mode, also stops the embedded server.
   */
  async close(): Promise<void> {
    if (this._closed) return;
    if (this._client) {
      await this._client.close();
      this._client = null;
    }
    // Stop embedded server if we started it
    if (this._embeddedServerStarted) {
      await stopEmbeddedServer(this._config.path);
      this._embeddedServerStarted = false;
    }
    this._closed = true;
  }

  // Internal methods for transaction management
  private async _beginTransaction(): Promise<bigint> {
    return this._client!.beginTransaction();
  }

  private async _commitTransaction(txnId: bigint): Promise<void> {
    return this._client!.commitTransaction(txnId);
  }

  private async _abortTransaction(txnId: bigint): Promise<void> {
    return this._client!.abortTransaction(txnId);
  }

  private _ensureOpen(): void {
    if (this._closed) {
      throw new DatabaseError('Database is closed');
    }
    if (!this._client) {
      throw new DatabaseError('Database not connected');
    }
  }
}
