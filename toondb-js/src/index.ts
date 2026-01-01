/**
 * ToonDB Node.js/TypeScript SDK
 *
 * A JavaScript/TypeScript client for ToonDB - the database optimized for LLM context retrieval.
 *
 * Provides two modes of access:
 * - Embedded: Direct database access via FFI (single process)
 * - IPC: Client-server access via Unix sockets (multi-process)
 * - Vector: HNSW vector search
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
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

export { Database, Transaction, DatabaseConfig } from './database';
export { IpcClient, IpcClientConfig, OpCode } from './ipc-client';
export { Query, QueryResult } from './query';
export { VectorIndex, VectorSearchResult, VectorIndexConfig } from './vector';
export {
  ToonDBError,
  ConnectionError,
  TransactionError,
  ProtocolError,
  DatabaseError,
} from './errors';
export {
  startEmbeddedServer,
  stopEmbeddedServer,
  stopAllEmbeddedServers,
  isServerRunning,
} from './server-manager';

export const VERSION = '0.2.4';
