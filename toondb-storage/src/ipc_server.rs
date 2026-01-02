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

//! Unix Domain Socket IPC Server for ToonDB
//!
//! Provides a local IPC server that wraps the Database kernel for
//! multi-process access to a ToonDB database.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                      IPC Server Process                        │
//! │  ┌────────────────────────────────────────────────────────┐   │
//! │  │              Database Kernel (Arc<Database>)           │   │
//! │  └────────────────────────────────────────────────────────┘   │
//! │           ▲                    ▲                    ▲         │
//! │           │                    │                    │         │
//! │  ┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐ │
//! │  │ ClientHandler 1 │ │ ClientHandler 2 │ │ ClientHandler N │ │
//! │  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘ │
//! │           │                    │                    │         │
//! │  ┌────────┴────────────────────┴────────────────────┴────────┐│
//! │  │              Unix Domain Socket Listener                  ││
//! │  │                  /tmp/toondb-<id>.sock                    ││
//! │  └───────────────────────────────────────────────────────────┘│
//! └────────────────────────────────────────────────────────────────┘
//!          ▲                    ▲                    ▲
//!          │ Unix Socket        │ Unix Socket        │ Unix Socket
//!   ┌──────┴──────┐      ┌──────┴──────┐      ┌──────┴──────┐
//!   │  Client 1   │      │  Client 2   │      │  Client N   │
//!   │  (Process)  │      │  (Process)  │      │  (Process)  │
//!   └─────────────┘      └─────────────┘      └─────────────┘
//! ```
//!
//! # Wire Protocol
//!
//! All messages use a simple length-prefixed binary format:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  OpCode (1 byte)  │  Length (4 bytes LE)  │  Payload (N)    │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## OpCodes
//!
//! | Code | Name          | Direction | Description                    |
//! |------|---------------|-----------|--------------------------------|
//! | 0x01 | PUT           | C→S       | Put key-value pair             |
//! | 0x02 | GET           | C→S       | Get value by key               |
//! | 0x03 | DELETE        | C→S       | Delete key                     |
//! | 0x04 | BEGIN_TXN     | C→S       | Start transaction              |
//! | 0x05 | COMMIT_TXN    | C→S       | Commit transaction             |
//! | 0x06 | ABORT_TXN     | C→S       | Abort transaction              |
//! | 0x07 | QUERY         | C→S       | Execute query                  |
//! | 0x08 | CREATE_TABLE  | C→S       | Create table                   |
//! | 0x09 | PUT_PATH      | C→S       | Put hierarchical path          |
//! | 0x0A | GET_PATH      | C→S       | Get by hierarchical path       |
//! | 0x0B | SCAN          | C→S       | Scan key range                 |
//! | 0x0C | CHECKPOINT    | C→S       | Force checkpoint               |
//! | 0x0D | STATS         | C→S       | Get database stats             |
//! |------|---------------|-----------|--------------------------------|
//! | 0x80 | OK            | S→C       | Success response               |
//! | 0x81 | ERROR         | S→C       | Error response                 |
//! | 0x82 | VALUE         | S→C       | Value response                 |
//! | 0x83 | TXN_ID        | S→C       | Transaction ID response        |
//! | 0x84 | ROW           | S→C       | Query result row (streaming)   |
//! | 0x85 | END_STREAM    | S→C       | End of streaming results       |
//! | 0x86 | STATS_RESP    | S→C       | Stats response                 |

use crate::database::{Database, TxnHandle};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use thiserror::Error;

// ============================================================================
// Wire Protocol Constants
// ============================================================================

/// Client → Server opcodes
mod opcode {
    pub const PUT: u8 = 0x01;
    pub const GET: u8 = 0x02;
    pub const DELETE: u8 = 0x03;
    pub const BEGIN_TXN: u8 = 0x04;
    pub const COMMIT_TXN: u8 = 0x05;
    pub const ABORT_TXN: u8 = 0x06;
    pub const QUERY: u8 = 0x07;
    pub const CREATE_TABLE: u8 = 0x08;
    pub const PUT_PATH: u8 = 0x09;
    pub const GET_PATH: u8 = 0x0A;
    pub const SCAN: u8 = 0x0B;
    pub const CHECKPOINT: u8 = 0x0C;
    pub const STATS: u8 = 0x0D;
    pub const PING: u8 = 0x0E;
    pub const EXECUTE_SQL: u8 = 0x0F;

    /// Server → Client response opcodes
    pub const OK: u8 = 0x80;
    pub const ERROR: u8 = 0x81;
    pub const VALUE: u8 = 0x82;
    pub const TXN_ID: u8 = 0x83;
    #[allow(dead_code)]
    pub const ROW: u8 = 0x84;
    #[allow(dead_code)]
    pub const END_STREAM: u8 = 0x85;
    pub const STATS_RESP: u8 = 0x86;
    pub const PONG: u8 = 0x87;
}

// Maximum message size (16 MB)
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Error)]
pub enum IpcError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Server already running")]
    AlreadyRunning,

    #[error("Server not running")]
    NotRunning,

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Message too large: {0} bytes (max: {1})")]
    MessageTooLarge(usize, usize),

    #[error("Invalid opcode: {0:#x}")]
    InvalidOpcode(u8),

    #[error("Transaction not found: {0}")]
    TxnNotFound(u64),
}

pub type Result<T> = std::result::Result<T, IpcError>;

// ============================================================================
// Wire Protocol Implementation
// ============================================================================

/// Message frame for the wire protocol
#[derive(Debug, Clone)]
pub struct Message {
    pub opcode: u8,
    pub payload: Vec<u8>,
}

impl Message {
    pub fn new(opcode: u8, payload: Vec<u8>) -> Self {
        Self { opcode, payload }
    }

    pub fn ok() -> Self {
        Self::new(opcode::OK, vec![])
    }

    pub fn error(msg: &str) -> Self {
        Self::new(opcode::ERROR, msg.as_bytes().to_vec())
    }

    pub fn value(data: Vec<u8>) -> Self {
        Self::new(opcode::VALUE, data)
    }

    pub fn txn_id(id: u64) -> Self {
        Self::new(opcode::TXN_ID, id.to_le_bytes().to_vec())
    }

    /// Read a message from a stream
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        // Read opcode (1 byte)
        let mut opcode_buf = [0u8; 1];
        match reader.read_exact(&mut opcode_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(IpcError::ConnectionClosed);
            }
            Err(e) => return Err(e.into()),
        }
        let opcode = opcode_buf[0];

        // Read length (4 bytes, little-endian)
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        // Validate length
        if len > MAX_MESSAGE_SIZE {
            return Err(IpcError::MessageTooLarge(len, MAX_MESSAGE_SIZE));
        }

        // Read payload
        let mut payload = vec![0u8; len];
        if len > 0 {
            reader.read_exact(&mut payload)?;
        }

        Ok(Self { opcode, payload })
    }

    /// Write a message to a stream
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write opcode
        writer.write_all(&[self.opcode])?;

        // Write length
        let len = self.payload.len() as u32;
        writer.write_all(&len.to_le_bytes())?;

        // Write payload
        if !self.payload.is_empty() {
            writer.write_all(&self.payload)?;
        }

        writer.flush()?;
        Ok(())
    }
}

// ============================================================================
// Request/Response Encoding
// ============================================================================

/// Encode a PUT request payload: key_len (4) + key + value
fn encode_put(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + key.len() + value.len());
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);
    buf
}

/// Decode a PUT request payload
fn decode_put(payload: &[u8]) -> Result<(&[u8], &[u8])> {
    if payload.len() < 4 {
        return Err(IpcError::Protocol("PUT payload too short".into()));
    }
    let key_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    if payload.len() < 4 + key_len {
        return Err(IpcError::Protocol("PUT payload key truncated".into()));
    }
    let key = &payload[4..4 + key_len];
    let value = &payload[4 + key_len..];
    Ok((key, value))
}

/// Encode a path PUT request: path_count (2) + [path_len (2) + path]... + value
fn encode_put_path(path: &[&str], value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(path.len() as u16).to_le_bytes());
    for segment in path {
        let seg_bytes = segment.as_bytes();
        buf.extend_from_slice(&(seg_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(seg_bytes);
    }
    buf.extend_from_slice(value);
    buf
}

/// Decode a path request
fn decode_path(payload: &[u8]) -> Result<(Vec<String>, &[u8])> {
    if payload.len() < 2 {
        return Err(IpcError::Protocol("Path payload too short".into()));
    }
    let count = u16::from_le_bytes(payload[0..2].try_into().unwrap()) as usize;
    let mut offset = 2;
    let mut path = Vec::with_capacity(count);

    for _ in 0..count {
        if offset + 2 > payload.len() {
            return Err(IpcError::Protocol("Path segment length truncated".into()));
        }
        let seg_len = u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + seg_len > payload.len() {
            return Err(IpcError::Protocol("Path segment truncated".into()));
        }
        let segment = std::str::from_utf8(&payload[offset..offset + seg_len])
            .map_err(|_| IpcError::Protocol("Invalid UTF-8 in path".into()))?;
        path.push(segment.to_string());
        offset += seg_len;
    }

    Ok((path, &payload[offset..]))
}

// ============================================================================
// Server Statistics
// ============================================================================

#[derive(Debug, Default)]
pub struct ServerStats {
    pub connections_total: AtomicU64,
    pub connections_active: AtomicU64,
    pub requests_total: AtomicU64,
    pub requests_success: AtomicU64,
    pub requests_error: AtomicU64,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub start_time: Mutex<Option<Instant>>,
}

impl ServerStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> ServerStatsSnapshot {
        ServerStatsSnapshot {
            connections_total: self.connections_total.load(Ordering::Relaxed),
            connections_active: self.connections_active.load(Ordering::Relaxed),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            requests_success: self.requests_success.load(Ordering::Relaxed),
            requests_error: self.requests_error.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            uptime_secs: self
                .start_time
                .lock()
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerStatsSnapshot {
    pub connections_total: u64,
    pub connections_active: u64,
    pub requests_total: u64,
    pub requests_success: u64,
    pub requests_error: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub uptime_secs: u64,
}

// ============================================================================
// Client Connection Handler
// ============================================================================

struct ClientHandler {
    db: Arc<Database>,
    stream: UnixStream,
    stats: Arc<ServerStats>,
    active_txns: HashMap<u64, TxnHandle>, // client_txn_id → TxnHandle
    next_client_txn_id: u64,
}

impl ClientHandler {
    fn new(db: Arc<Database>, stream: UnixStream, stats: Arc<ServerStats>) -> Self {
        Self {
            db,
            stream,
            stats,
            active_txns: HashMap::new(),
            next_client_txn_id: 1,
        }
    }

    fn handle(&mut self) -> Result<()> {
        // Set read timeout for graceful shutdown detection
        self.stream
            .set_read_timeout(Some(Duration::from_secs(30)))?;

        let mut reader = BufReader::new(self.stream.try_clone()?);
        let mut writer = BufWriter::new(self.stream.try_clone()?);

        loop {
            // Read request
            let request = match Message::read_from(&mut reader) {
                Ok(msg) => msg,
                Err(IpcError::ConnectionClosed) => {
                    // Clean shutdown - abort any pending transactions
                    self.cleanup_transactions();
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            self.stats.requests_total.fetch_add(1, Ordering::Relaxed);
            self.stats
                .bytes_received
                .fetch_add((5 + request.payload.len()) as u64, Ordering::Relaxed);

            // Process request
            let response = self.process_request(&request);

            // Track success/error
            if response.opcode == opcode::ERROR {
                self.stats.requests_error.fetch_add(1, Ordering::Relaxed);
            } else {
                self.stats.requests_success.fetch_add(1, Ordering::Relaxed);
            }

            // Send response
            self.stats
                .bytes_sent
                .fetch_add((5 + response.payload.len()) as u64, Ordering::Relaxed);
            response.write_to(&mut writer)?;
        }
    }

    fn process_request(&mut self, request: &Message) -> Message {
        match request.opcode {
            opcode::PING => Message::new(opcode::PONG, vec![]),

            opcode::PUT => self.handle_put(&request.payload),
            opcode::GET => self.handle_get(&request.payload),
            opcode::DELETE => self.handle_delete(&request.payload),

            opcode::BEGIN_TXN => self.handle_begin_txn(),
            opcode::COMMIT_TXN => self.handle_commit_txn(&request.payload),
            opcode::ABORT_TXN => self.handle_abort_txn(&request.payload),

            opcode::PUT_PATH => self.handle_put_path(&request.payload),
            opcode::GET_PATH => self.handle_get_path(&request.payload),

            opcode::QUERY => self.handle_query(&request.payload),
            opcode::CREATE_TABLE => self.handle_create_table(&request.payload),
            opcode::SCAN => self.handle_scan(&request.payload),
            opcode::EXECUTE_SQL => self.handle_execute_sql(&request.payload),

            opcode::CHECKPOINT => self.handle_checkpoint(),
            opcode::STATS => self.handle_stats(),

            _ => Message::error(&format!("Unknown opcode: {:#x}", request.opcode)),
        }
    }

    fn handle_execute_sql(&self, payload: &[u8]) -> Message {
        // Payload: SQL query string (UTF-8)
        let sql = match std::str::from_utf8(payload) {
            Ok(s) => s,
            Err(_) => return Message::error("Invalid UTF-8 in SQL query"),
        };

        // For now, return error indicating SQL execution happens client-side
        // The Go SDK will need to implement SQL-to-KV mapping like Python does
        let result = serde_json::json!({
            "error": "SQL execution must be implemented client-side. Use Python SDK for full SQL support.",
            "sql": sql
        });

        match serde_json::to_vec(&result) {
            Ok(json) => Message::value(json),
            Err(e) => Message::error(&format!("Failed to serialize error: {}", e)),
        }
    }

    fn handle_query(&self, payload: &[u8]) -> Message {
        // Payload: path_len(2) + path + limit(4) + offset(4) + cols_count(2) + [col_len(2) + col]...
        let mut offset = 0;

        if payload.len() < 2 {
            return Message::error("Query payload too short");
        }

        // Path
        let path_len = u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + path_len > payload.len() {
            return Message::error("Query path truncated");
        }
        let path = match std::str::from_utf8(&payload[offset..offset + path_len]) {
            Ok(s) => s,
            Err(_) => return Message::error("Invalid UTF-8 in query path"),
        };
        offset += path_len;

        // Limit & Offset
        if offset + 8 > payload.len() {
            return Message::error("Query limit/offset truncated");
        }
        let limit_val =
            u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let offset_val =
            u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        // Columns
        if offset + 2 > payload.len() {
            return Message::error("Query columns count truncated");
        }
        let cols_count =
            u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        let mut columns = Vec::with_capacity(cols_count);
        for _ in 0..cols_count {
            if offset + 2 > payload.len() {
                return Message::error("Query column length truncated");
            }
            let col_len =
                u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
            offset += 2;

            if offset + col_len > payload.len() {
                return Message::error("Query column truncated");
            }
            let col = match std::str::from_utf8(&payload[offset..offset + col_len]) {
                Ok(s) => s.to_string(),
                Err(_) => return Message::error("Invalid UTF-8 in query column"),
            };
            columns.push(col);
            offset += col_len;
        }

        // Execute query
        // Note: Query is read-only, so we can use a read transaction
        let txn = match self.db.begin_transaction() {
            Ok(t) => t,
            Err(e) => return Message::error(&e.to_string()),
        };

        let mut builder = self.db.query(txn, path);

        if limit_val > 0 {
            builder = builder.limit(limit_val);
        }
        if offset_val > 0 {
            builder = builder.offset(offset_val);
        }

        if !columns.is_empty() {
            let cols_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            builder = builder.columns(&cols_refs);
        }

        let result = builder.to_toon();
        let _ = self.db.abort(txn); // Read-only

        match result {
            Ok(toon_str) => Message::new(opcode::VALUE, toon_str.into_bytes()),
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_scan(&self, payload: &[u8]) -> Message {
        let prefix = match std::str::from_utf8(payload) {
            Ok(s) => s,
            Err(_) => return Message::error("Invalid UTF-8 in scan prefix"),
        };

        let txn = match self.db.begin_transaction() {
            Ok(t) => t,
            Err(e) => return Message::error(&e.to_string()),
        };

        let result = self.db.scan_path(txn, prefix);
        let _ = self.db.abort(txn);

        match result {
            Ok(items) => {
                // Format as simple newline-separated key=value for now
                // Or maybe JSON? Let's use a simple custom format:
                // count(4) + [key_len(2) + key + val_len(4) + val]...
                let mut buf = Vec::new();
                buf.extend_from_slice(&(items.len() as u32).to_le_bytes());

                for (key, val) in items {
                    let key_bytes = key.as_bytes();
                    buf.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
                    buf.extend_from_slice(key_bytes);
                    buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
                    buf.extend_from_slice(&val);
                }

                Message::new(opcode::VALUE, buf)
            }
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_create_table(&self, payload: &[u8]) -> Message {
        // Payload: JSON schema definition
        let _schema_json = match std::str::from_utf8(payload) {
            Ok(s) => s,
            Err(_) => return Message::error("Invalid UTF-8 in schema"),
        };

        // We need a way to parse TableSchema from JSON.
        // Since we don't have serde_json derived for TableSchema in database.rs (it's in toondb-storage, but TableSchema is in database.rs),
        // we might need to manually parse or assume it's passed as a specific format.
        // Let's assume for now we can use serde_json if we add the dependency or if it's already there.
        // Checking Cargo.toml... serde_json is there.
        // But TableSchema struct in database.rs doesn't derive Deserialize.
        // I'll need to define a local struct or use a helper.

        // For now, let's implement a simple manual parser or just error out saying "Not implemented fully"
        // but the plan said "Parse payload: Schema definition".
        // Let's try to use a simple custom binary format for schema to avoid JSON dependency issues if structs aren't serializable.
        // Format: name_len(2) + name + col_count(2) + [col_name_len(2) + col_name + type(1) + nullable(1)]...

        let mut offset = 0;
        if payload.len() < 2 {
            return Message::error("Schema payload too short");
        }

        let name_len = u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + name_len > payload.len() {
            return Message::error("Schema name truncated");
        }
        let name = match std::str::from_utf8(&payload[offset..offset + name_len]) {
            Ok(s) => s.to_string(),
            Err(_) => return Message::error("Invalid UTF-8 in schema name"),
        };
        offset += name_len;

        if offset + 2 > payload.len() {
            return Message::error("Schema column count truncated");
        }
        let col_count =
            u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        let mut columns = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            if offset + 2 > payload.len() {
                return Message::error("Column name length truncated");
            }
            let col_name_len =
                u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as usize;
            offset += 2;

            if offset + col_name_len > payload.len() {
                return Message::error("Column name truncated");
            }
            let col_name = match std::str::from_utf8(&payload[offset..offset + col_name_len]) {
                Ok(s) => s.to_string(),
                Err(_) => return Message::error("Invalid UTF-8 in column name"),
            };
            offset += col_name_len;

            if offset + 2 > payload.len() {
                return Message::error("Column type/nullable truncated");
            }
            let type_byte = payload[offset];
            offset += 1;
            let nullable_byte = payload[offset];
            offset += 1;

            let col_type = match type_byte {
                0 => crate::database::ColumnType::Int64,
                1 => crate::database::ColumnType::UInt64,
                2 => crate::database::ColumnType::Float64,
                3 => crate::database::ColumnType::Text,
                4 => crate::database::ColumnType::Binary,
                5 => crate::database::ColumnType::Bool,
                _ => return Message::error("Invalid column type"),
            };

            columns.push(crate::database::ColumnDef {
                name: col_name,
                col_type,
                nullable: nullable_byte != 0,
            });
        }

        let schema = crate::database::TableSchema { name, columns };

        match self.db.register_table(schema) {
            Ok(_) => Message::ok(),
            Err(e) => Message::error(&e.to_string()),
        }
    }

    /// Auto-commit PUT - creates a transaction, writes, commits
    fn handle_put(&self, payload: &[u8]) -> Message {
        match decode_put(payload) {
            Ok((key, value)) => {
                // Auto-transaction for simple PUT
                let txn = match self.db.begin_transaction() {
                    Ok(t) => t,
                    Err(e) => return Message::error(&e.to_string()),
                };

                if let Err(e) = self.db.put(txn, key, value) {
                    let _ = self.db.abort(txn);
                    return Message::error(&e.to_string());
                }

                match self.db.commit(txn) {
                    Ok(_) => Message::ok(),
                    Err(e) => Message::error(&e.to_string()),
                }
            }
            Err(e) => Message::error(&e.to_string()),
        }
    }

    /// Auto-commit GET - creates a read transaction
    fn handle_get(&self, payload: &[u8]) -> Message {
        // Auto-transaction for simple GET
        let txn = match self.db.begin_transaction() {
            Ok(t) => t,
            Err(e) => return Message::error(&e.to_string()),
        };

        let result = self.db.get(txn, payload);
        let _ = self.db.abort(txn); // Abort is fine for read-only

        match result {
            Ok(Some(value)) => Message::value(value),
            Ok(None) => Message::new(opcode::VALUE, vec![]),
            Err(e) => Message::error(&e.to_string()),
        }
    }

    /// Auto-commit DELETE
    fn handle_delete(&self, payload: &[u8]) -> Message {
        let txn = match self.db.begin_transaction() {
            Ok(t) => t,
            Err(e) => return Message::error(&e.to_string()),
        };

        if let Err(e) = self.db.delete(txn, payload) {
            let _ = self.db.abort(txn);
            return Message::error(&e.to_string());
        }

        match self.db.commit(txn) {
            Ok(_) => Message::ok(),
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_begin_txn(&mut self) -> Message {
        match self.db.begin_transaction() {
            Ok(txn) => {
                let client_txn_id = self.next_client_txn_id;
                self.next_client_txn_id += 1;
                self.active_txns.insert(client_txn_id, txn);
                Message::txn_id(client_txn_id)
            }
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_commit_txn(&mut self, payload: &[u8]) -> Message {
        if payload.len() < 8 {
            return Message::error("COMMIT_TXN requires txn_id");
        }
        let client_txn_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        match self.active_txns.remove(&client_txn_id) {
            Some(txn) => match self.db.commit(txn) {
                Ok(commit_ts) => Message::txn_id(commit_ts),
                Err(e) => Message::error(&e.to_string()),
            },
            None => Message::error(&format!("Transaction not found: {}", client_txn_id)),
        }
    }

    fn handle_abort_txn(&mut self, payload: &[u8]) -> Message {
        if payload.len() < 8 {
            return Message::error("ABORT_TXN requires txn_id");
        }
        let client_txn_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        match self.active_txns.remove(&client_txn_id) {
            Some(txn) => match self.db.abort(txn) {
                Ok(_) => Message::ok(),
                Err(e) => Message::error(&e.to_string()),
            },
            None => Message::error(&format!("Transaction not found: {}", client_txn_id)),
        }
    }

    fn handle_put_path(&self, payload: &[u8]) -> Message {
        match decode_path(payload) {
            Ok((path, value)) => {
                let txn = match self.db.begin_transaction() {
                    Ok(t) => t,
                    Err(e) => return Message::error(&e.to_string()),
                };

                let path_str = path.join("/");
                if let Err(e) = self.db.put_path(txn, &path_str, value) {
                    let _ = self.db.abort(txn);
                    return Message::error(&e.to_string());
                }

                match self.db.commit(txn) {
                    Ok(_) => Message::ok(),
                    Err(e) => Message::error(&e.to_string()),
                }
            }
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_get_path(&self, payload: &[u8]) -> Message {
        match decode_path(payload) {
            Ok((path, _)) => {
                let txn = match self.db.begin_transaction() {
                    Ok(t) => t,
                    Err(e) => return Message::error(&e.to_string()),
                };

                let path_str = path.join("/");
                let result = self.db.get_path(txn, &path_str);
                let _ = self.db.abort(txn);

                match result {
                    Ok(Some(value)) => Message::value(value),
                    Ok(None) => Message::new(opcode::VALUE, vec![]),
                    Err(e) => Message::error(&e.to_string()),
                }
            }
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_checkpoint(&self) -> Message {
        match self.db.checkpoint() {
            Ok(_) => Message::ok(),
            Err(e) => Message::error(&e.to_string()),
        }
    }

    fn handle_stats(&self) -> Message {
        let stats = self.stats.snapshot();
        // Encode stats as JSON for SDK compatibility
        let stats_json = format!(
            r#"{{"connections_total":{},"connections_active":{},"requests_total":{},"requests_success":{},"requests_error":{},"bytes_received":{},"bytes_sent":{},"uptime_secs":{},"memtable_size_bytes":0,"wal_size_bytes":0,"active_transactions":{}}}"#,
            stats.connections_total,
            stats.connections_active,
            stats.requests_total,
            stats.requests_success,
            stats.requests_error,
            stats.bytes_received,
            stats.bytes_sent,
            stats.uptime_secs,
            self.active_txns.len()
        );
        Message::new(opcode::STATS_RESP, stats_json.into_bytes())
    }

    fn cleanup_transactions(&mut self) {
        // Abort all pending transactions for this client
        for (_client_id, txn) in self.active_txns.drain() {
            let _ = self.db.abort(txn);
        }
    }
}

// ============================================================================
// IPC Server
// ============================================================================

/// Configuration for the IPC server
#[derive(Debug, Clone)]
pub struct IpcServerConfig {
    /// Path to the Unix socket file
    pub socket_path: PathBuf,

    /// Maximum number of concurrent connections
    pub max_connections: usize,

    /// Thread pool size for handling connections
    pub thread_pool_size: usize,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
}

impl Default for IpcServerConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/toondb.sock"),
            max_connections: 100,
            thread_pool_size: 4,
            connection_timeout_secs: 300, // 5 minutes
        }
    }
}

impl IpcServerConfig {
    pub fn with_socket_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.socket_path = path.as_ref().to_path_buf();
        self
    }

    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }
}

/// Unix Domain Socket IPC Server
pub struct IpcServer {
    db: Arc<Database>,
    config: IpcServerConfig,
    stats: Arc<ServerStats>,
    running: Arc<AtomicBool>,
    listener_handle: Mutex<Option<JoinHandle<()>>>,
}

impl IpcServer {
    /// Create a new IPC server for the given database
    pub fn new(db: Arc<Database>, config: IpcServerConfig) -> Self {
        Self {
            db,
            config,
            stats: Arc::new(ServerStats::new()),
            running: Arc::new(AtomicBool::new(false)),
            listener_handle: Mutex::new(None),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(db: Arc<Database>) -> Self {
        Self::new(db, IpcServerConfig::default())
    }

    /// Start the server (blocking)
    pub fn run(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(IpcError::AlreadyRunning);
        }

        // Remove existing socket file if present
        if self.config.socket_path.exists() {
            std::fs::remove_file(&self.config.socket_path)?;
        }

        // Create listener
        let listener = UnixListener::bind(&self.config.socket_path)?;
        listener.set_nonblocking(false)?;

        // Record start time
        *self.stats.start_time.lock() = Some(Instant::now());

        eprintln!("[IpcServer] Listening on {:?}", self.config.socket_path);

        // Accept connections
        while self.running.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _addr)) => {
                    // Check connection limit
                    let active = self.stats.connections_active.load(Ordering::Relaxed);
                    if active >= self.config.max_connections as u64 {
                        eprintln!("[IpcServer] Connection limit reached, rejecting");
                        continue;
                    }

                    self.stats.connections_total.fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .connections_active
                        .fetch_add(1, Ordering::Relaxed);

                    let db = Arc::clone(&self.db);
                    let stats = Arc::clone(&self.stats);

                    // Spawn handler thread
                    thread::spawn(move || {
                        let mut handler = ClientHandler::new(db, stream, Arc::clone(&stats));
                        if let Err(e) = handler.handle() {
                            eprintln!("[IpcServer] Client error: {}", e);
                        }
                        stats.connections_active.fetch_sub(1, Ordering::Relaxed);
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Non-blocking timeout, check if we should stop
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    eprintln!("[IpcServer] Accept error: {}", e);
                }
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&self.config.socket_path);

        Ok(())
    }

    /// Start the server in a background thread
    pub fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(IpcError::AlreadyRunning);
        }

        let db = Arc::clone(&self.db);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let running = Arc::clone(&self.running);

        let handle = thread::spawn(move || {
            // Run the server loop directly (flag already set by start())
            // Remove existing socket file if present
            if config.socket_path.exists() {
                let _ = std::fs::remove_file(&config.socket_path);
            }

            // Create listener
            let listener = match UnixListener::bind(&config.socket_path) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[IpcServer] Failed to bind: {}", e);
                    running.store(false, Ordering::SeqCst);
                    return;
                }
            };
            let _ = listener.set_nonblocking(false);

            // Record start time
            *stats.start_time.lock() = Some(Instant::now());

            eprintln!("[IpcServer] Listening on {:?}", config.socket_path);

            // Accept connections
            while running.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _addr)) => {
                        // Check connection limit
                        let active = stats.connections_active.load(Ordering::Relaxed);
                        if active >= config.max_connections as u64 {
                            eprintln!("[IpcServer] Connection limit reached, rejecting");
                            continue;
                        }

                        stats.connections_total.fetch_add(1, Ordering::Relaxed);
                        stats.connections_active.fetch_add(1, Ordering::Relaxed);

                        let db_clone = Arc::clone(&db);
                        let stats_clone = Arc::clone(&stats);

                        // Spawn handler thread
                        thread::spawn(move || {
                            let mut handler =
                                ClientHandler::new(db_clone, stream, Arc::clone(&stats_clone));
                            if let Err(e) = handler.handle() {
                                eprintln!("[IpcServer] Client error: {}", e);
                            }
                            stats_clone
                                .connections_active
                                .fetch_sub(1, Ordering::Relaxed);
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // Non-blocking timeout, check if we should stop
                        thread::sleep(Duration::from_millis(100));
                    }
                    Err(e) => {
                        eprintln!("[IpcServer] Accept error: {}", e);
                        break;
                    }
                }
            }

            // Cleanup
            let _ = std::fs::remove_file(&config.socket_path);
        });

        *self.listener_handle.lock() = Some(handle);
        Ok(())
    }

    /// Stop the server
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        // Connect to socket to wake up accept() if blocking
        let _ = UnixStream::connect(&self.config.socket_path);

        // Wait for listener thread
        if let Some(handle) = self.listener_handle.lock().take() {
            let _ = handle.join();
        }
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get server statistics
    pub fn stats(&self) -> ServerStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get socket path
    pub fn socket_path(&self) -> &Path {
        &self.config.socket_path
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// IPC Client (for connecting to server from another process)
// ============================================================================

/// Client for connecting to an IPC server
pub struct IpcClient {
    stream: UnixStream,
}

impl IpcClient {
    /// Connect to an IPC server
    pub fn connect<P: AsRef<Path>>(socket_path: P) -> Result<Self> {
        let stream = UnixStream::connect(socket_path)?;
        Ok(Self { stream })
    }

    /// Connect with timeout
    pub fn connect_with_timeout<P: AsRef<Path>>(socket_path: P, timeout: Duration) -> Result<Self> {
        let stream = UnixStream::connect(socket_path)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        Ok(Self { stream })
    }

    /// Send a request and receive response
    fn request(&mut self, msg: Message) -> Result<Message> {
        msg.write_to(&mut self.stream)?;
        Message::read_from(&mut self.stream)
    }

    /// Ping the server
    pub fn ping(&mut self) -> Result<Duration> {
        let start = Instant::now();
        let resp = self.request(Message::new(opcode::PING, vec![]))?;
        if resp.opcode != opcode::PONG {
            return Err(IpcError::Protocol("Expected PONG".into()));
        }
        Ok(start.elapsed())
    }

    /// Put a key-value pair
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let payload = encode_put(key, value);
        let resp = self.request(Message::new(opcode::PUT, payload))?;
        self.check_ok(resp)
    }

    /// Get a value by key
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let resp = self.request(Message::new(opcode::GET, key.to_vec()))?;
        match resp.opcode {
            opcode::VALUE if resp.payload.is_empty() => Ok(None),
            opcode::VALUE => Ok(Some(resp.payload)),
            opcode::ERROR => Err(IpcError::Database(
                String::from_utf8_lossy(&resp.payload).to_string(),
            )),
            _ => Err(IpcError::Protocol(format!(
                "Unexpected opcode: {:#x}",
                resp.opcode
            ))),
        }
    }

    /// Delete a key
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        let resp = self.request(Message::new(opcode::DELETE, key.to_vec()))?;
        self.check_ok(resp)
    }

    /// Begin a transaction, returns transaction ID
    pub fn begin_txn(&mut self) -> Result<u64> {
        let resp = self.request(Message::new(opcode::BEGIN_TXN, vec![]))?;
        match resp.opcode {
            opcode::TXN_ID => {
                if resp.payload.len() >= 8 {
                    Ok(u64::from_le_bytes(resp.payload[0..8].try_into().unwrap()))
                } else {
                    Err(IpcError::Protocol("TXN_ID response too short".into()))
                }
            }
            opcode::ERROR => Err(IpcError::Database(
                String::from_utf8_lossy(&resp.payload).to_string(),
            )),
            _ => Err(IpcError::Protocol(format!(
                "Unexpected opcode: {:#x}",
                resp.opcode
            ))),
        }
    }

    /// Commit a transaction, returns commit timestamp
    pub fn commit_txn(&mut self, txn_id: u64) -> Result<u64> {
        let resp = self.request(Message::new(
            opcode::COMMIT_TXN,
            txn_id.to_le_bytes().to_vec(),
        ))?;
        match resp.opcode {
            opcode::TXN_ID => {
                if resp.payload.len() >= 8 {
                    Ok(u64::from_le_bytes(resp.payload[0..8].try_into().unwrap()))
                } else {
                    Err(IpcError::Protocol("TXN_ID response too short".into()))
                }
            }
            opcode::ERROR => Err(IpcError::Database(
                String::from_utf8_lossy(&resp.payload).to_string(),
            )),
            _ => Err(IpcError::Protocol(format!(
                "Unexpected opcode: {:#x}",
                resp.opcode
            ))),
        }
    }

    /// Abort a transaction
    pub fn abort_txn(&mut self, txn_id: u64) -> Result<()> {
        let resp = self.request(Message::new(
            opcode::ABORT_TXN,
            txn_id.to_le_bytes().to_vec(),
        ))?;
        self.check_ok(resp)
    }

    /// Put by hierarchical path
    pub fn put_path(&mut self, path: &[&str], value: &[u8]) -> Result<()> {
        let payload = encode_put_path(path, value);
        let resp = self.request(Message::new(opcode::PUT_PATH, payload))?;
        self.check_ok(resp)
    }

    /// Get by hierarchical path
    pub fn get_path(&mut self, path: &[&str]) -> Result<Option<Vec<u8>>> {
        let payload = encode_put_path(path, &[]);
        let resp = self.request(Message::new(opcode::GET_PATH, payload))?;
        match resp.opcode {
            opcode::VALUE if resp.payload.is_empty() => Ok(None),
            opcode::VALUE => Ok(Some(resp.payload)),
            opcode::ERROR => Err(IpcError::Database(
                String::from_utf8_lossy(&resp.payload).to_string(),
            )),
            _ => Err(IpcError::Protocol(format!(
                "Unexpected opcode: {:#x}",
                resp.opcode
            ))),
        }
    }

    /// Force a checkpoint
    pub fn checkpoint(&mut self) -> Result<()> {
        let resp = self.request(Message::new(opcode::CHECKPOINT, vec![]))?;
        self.check_ok(resp)
    }

    /// Get server statistics
    pub fn stats(&mut self) -> Result<HashMap<String, u64>> {
        let resp = self.request(Message::new(opcode::STATS, vec![]))?;
        match resp.opcode {
            opcode::STATS_RESP => {
                let stats_str = String::from_utf8_lossy(&resp.payload);
                
                // Parse JSON response
                let stats: HashMap<String, u64> = serde_json::from_str(&stats_str)
                    .map_err(|e| IpcError::Protocol(format!("Failed to parse stats JSON: {}", e)))?;
                
                Ok(stats)
            }
            opcode::ERROR => Err(IpcError::Database(
                String::from_utf8_lossy(&resp.payload).to_string(),
            )),
            _ => Err(IpcError::Protocol(format!(
                "Unexpected opcode: {:#x}",
                resp.opcode
            ))),
        }
    }

    fn check_ok(&self, resp: Message) -> Result<()> {
        match resp.opcode {
            opcode::OK => Ok(()),
            opcode::ERROR => Err(IpcError::Database(
                String::from_utf8_lossy(&resp.payload).to_string(),
            )),
            _ => Err(IpcError::Protocol(format!(
                "Unexpected opcode: {:#x}",
                resp.opcode
            ))),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    fn setup_test_server() -> (Arc<Database>, TempDir, PathBuf) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let socket_path = temp_dir.path().join("test.sock");

        let db = Database::open(&db_path).unwrap();
        (db, temp_dir, socket_path)
    }

    #[test]
    fn test_message_roundtrip() {
        let original = Message::new(0x01, b"hello world".to_vec());

        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(buffer);
        let decoded = Message::read_from(&mut cursor).unwrap();

        assert_eq!(decoded.opcode, original.opcode);
        assert_eq!(decoded.payload, original.payload);
    }

    #[test]
    fn test_encode_decode_put() {
        let key = b"test-key";
        let value = b"test-value";

        let encoded = encode_put(key, value);
        let (decoded_key, decoded_value) = decode_put(&encoded).unwrap();

        assert_eq!(decoded_key, key);
        assert_eq!(decoded_value, value);
    }

    #[test]
    fn test_encode_decode_path() {
        let path = vec!["users", "alice", "settings"];
        let value = b"preferences";

        let encoded = encode_put_path(&path, value);
        let (decoded_path, decoded_value) = decode_path(&encoded).unwrap();

        let expected_path: Vec<String> = path.iter().map(|s| s.to_string()).collect();
        assert_eq!(decoded_path, expected_path);
        assert_eq!(decoded_value, value);
    }

    #[test]
    fn test_server_client_basic() {
        let (db, _temp_dir, socket_path) = setup_test_server();

        // Start server
        let config = IpcServerConfig::default().with_socket_path(&socket_path);
        let server = IpcServer::new(Arc::clone(&db), config);
        server.start().unwrap();

        // Wait for server to be ready
        thread::sleep(Duration::from_millis(100));

        // Connect client
        let mut client = IpcClient::connect(&socket_path).unwrap();

        // Test ping
        let latency = client.ping().unwrap();
        assert!(latency < Duration::from_secs(1));

        // Test put/get
        client.put(b"key1", b"value1").unwrap();
        let value = client.get(b"key1").unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));

        // Test get non-existent
        let value = client.get(b"nonexistent").unwrap();
        assert_eq!(value, None);

        // Test delete
        client.delete(b"key1").unwrap();
        let value = client.get(b"key1").unwrap();
        assert_eq!(value, None);

        // Stop server
        server.stop();
    }

    #[test]
    fn test_server_client_transactions() {
        let (db, _temp_dir, socket_path) = setup_test_server();

        let config = IpcServerConfig::default().with_socket_path(&socket_path);
        let server = IpcServer::new(Arc::clone(&db), config);
        server.start().unwrap();

        thread::sleep(Duration::from_millis(100));

        let mut client = IpcClient::connect(&socket_path).unwrap();

        // Begin transaction
        let txn_id = client.begin_txn().unwrap();
        assert!(txn_id > 0);

        // Commit
        let commit_ts = client.commit_txn(txn_id).unwrap();
        assert!(commit_ts > 0);

        // Begin another and abort
        let txn_id2 = client.begin_txn().unwrap();
        client.abort_txn(txn_id2).unwrap();

        server.stop();
    }

    #[test]
    fn test_server_client_paths() {
        let (db, _temp_dir, socket_path) = setup_test_server();

        let config = IpcServerConfig::default().with_socket_path(&socket_path);
        let server = IpcServer::new(Arc::clone(&db), config);
        server.start().unwrap();

        thread::sleep(Duration::from_millis(100));

        let mut client = IpcClient::connect(&socket_path).unwrap();

        // Put by path
        client
            .put_path(&["users", "alice", "email"], b"alice@example.com")
            .unwrap();

        // Get by path
        let value = client.get_path(&["users", "alice", "email"]).unwrap();
        assert_eq!(value, Some(b"alice@example.com".to_vec()));

        // Get non-existent path
        let value = client.get_path(&["users", "bob", "email"]).unwrap();
        assert_eq!(value, None);

        server.stop();
    }

    #[test]
    fn test_server_stats() {
        let (db, _temp_dir, socket_path) = setup_test_server();

        let config = IpcServerConfig::default().with_socket_path(&socket_path);
        let server = IpcServer::new(Arc::clone(&db), config);
        server.start().unwrap();

        thread::sleep(Duration::from_millis(100));

        let mut client = IpcClient::connect(&socket_path).unwrap();

        // Make some requests
        client.ping().unwrap();
        client.put(b"k", b"v").unwrap();
        client.get(b"k").unwrap();

        // Get stats
        let stats = client.stats().unwrap();
        assert!(stats.contains_key("requests_total"));
        assert!(*stats.get("requests_total").unwrap() >= 4);

        // Check server-side stats
        let server_stats = server.stats();
        assert!(server_stats.requests_total >= 4);
        assert!(server_stats.connections_active >= 1);

        server.stop();
    }

    #[test]
    fn test_multiple_clients() {
        let (db, _temp_dir, socket_path) = setup_test_server();

        let config = IpcServerConfig::default()
            .with_socket_path(&socket_path)
            .with_max_connections(10);
        let server = IpcServer::new(Arc::clone(&db), config);
        server.start().unwrap();

        thread::sleep(Duration::from_millis(100));

        // Connect multiple clients
        let mut handles = Vec::new();
        let socket_path_clone = socket_path.clone();

        for i in 0..5 {
            let path = socket_path_clone.clone();
            let handle = thread::spawn(move || {
                let mut client = IpcClient::connect(&path).unwrap();
                let key = format!("key-{}", i);
                let value = format!("value-{}", i);

                client.put(key.as_bytes(), value.as_bytes()).unwrap();
                let result = client.get(key.as_bytes()).unwrap();
                assert_eq!(result, Some(value.into_bytes()));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = server.stats();
        assert_eq!(stats.connections_total, 5);

        server.stop();
    }
}
