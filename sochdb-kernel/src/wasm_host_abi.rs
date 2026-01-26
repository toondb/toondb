// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! WASM Host Function ABI
//!
//! This module defines the host functions that WASM plugins can call
//! to interact with SochDB.
//!
//! ## Design Principles
//!
//! 1. **Capability-Checked**: All operations verify permissions before execution
//! 2. **Memory-Safe**: All data crosses the WASM boundary via explicit serialization
//! 3. **Low-Overhead**: Designed for ~100ns per call overhead target
//! 4. **Auditable**: Every host call is logged for security auditing
//!
//! ## Available Host Functions
//!
//! | Function | Description | Capability Required |
//! |----------|-------------|---------------------|
//! | `soch_read` | Read rows from a table | `can_read_table` |
//! | `soch_write` | Write rows to a table | `can_write_table` |
//! | `soch_delete` | Delete rows from a table | `can_write_table` |
//! | `vector_search` | Similarity search | `can_vector_search` |
//! | `index_lookup` | Point/range index lookup | `can_index_search` |
//! | `emit_metric` | Emit observability metric | (always allowed) |
//! | `log_message` | Log a message | (always allowed) |
//! | `get_config` | Read plugin config | (always allowed) |
//!
//! ## Memory Model
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                  WASM Linear Memory                  │
//! │  ┌─────────────────────────────────────────────┐    │
//! │  │         Plugin Code & Data                   │    │
//! │  └─────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────┐    │
//! │  │     Input Buffer (host writes here)         │    │
//! │  └─────────────────────────────────────────────┘    │
//! │  ┌─────────────────────────────────────────────┐    │
//! │  │     Output Buffer (plugin writes here)       │    │
//! │  └─────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────┘
//! ```

use crate::error::{KernelError, KernelResult};
use crate::kernel_api::RowId;
use crate::wasm_runtime::WasmPluginCapabilities;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Host Function Results
// ============================================================================

/// Result of a host function call
#[derive(Debug, Clone, PartialEq)]
pub enum HostCallResult {
    /// Operation succeeded with data
    Success(Vec<u8>),
    /// Operation succeeded with no data
    Ok,
    /// Permission denied
    PermissionDenied(String),
    /// Resource not found
    NotFound(String),
    /// Invalid arguments
    InvalidArgs(String),
    /// Internal error
    Error(String),
}

impl HostCallResult {
    /// Convert to status code for WASM
    pub fn status_code(&self) -> i32 {
        match self {
            HostCallResult::Success(_) => 0,
            HostCallResult::Ok => 0,
            HostCallResult::PermissionDenied(_) => -1,
            HostCallResult::NotFound(_) => -2,
            HostCallResult::InvalidArgs(_) => -3,
            HostCallResult::Error(_) => -4,
        }
    }

    /// Get data if successful
    pub fn data(&self) -> Option<&[u8]> {
        match self {
            HostCallResult::Success(data) => Some(data),
            _ => None,
        }
    }
}

// ============================================================================
// Host Function Context
// ============================================================================

/// Context passed to host function implementations
///
/// Contains the plugin's capabilities and access to database resources.
pub struct HostFunctionContext {
    /// Plugin name (for logging)
    pub plugin_name: String,
    /// Plugin capabilities
    pub capabilities: WasmPluginCapabilities,
    /// Audit log for this call chain
    pub audit_log: Vec<AuditEntry>,
    /// Current transaction ID (if any)
    pub transaction_id: Option<u64>,
    /// Session variables
    pub session_vars: HashMap<String, Vec<u8>>,
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Function name
    pub function: String,
    /// Table accessed (if any)
    pub table: Option<String>,
    /// Result status code
    pub status: i32,
    /// Rows affected
    pub rows_affected: u64,
}

impl HostFunctionContext {
    /// Create a new context for a plugin
    pub fn new(plugin_name: &str, capabilities: WasmPluginCapabilities) -> Self {
        Self {
            plugin_name: plugin_name.to_string(),
            capabilities,
            audit_log: Vec::new(),
            transaction_id: None,
            session_vars: HashMap::new(),
        }
    }

    /// Check if the plugin can read from a table
    pub fn check_read(&self, table: &str) -> KernelResult<()> {
        if !self.capabilities.can_read(table) {
            return Err(KernelError::Plugin {
                message: format!(
                    "plugin '{}' not authorized to read table '{}'",
                    self.plugin_name, table
                ),
            });
        }
        Ok(())
    }

    /// Check if the plugin can write to a table
    pub fn check_write(&self, table: &str) -> KernelResult<()> {
        if !self.capabilities.can_write(table) {
            return Err(KernelError::Plugin {
                message: format!(
                    "plugin '{}' not authorized to write table '{}'",
                    self.plugin_name, table
                ),
            });
        }
        Ok(())
    }

    /// Check if the plugin can perform vector search
    pub fn check_vector_search(&self) -> KernelResult<()> {
        if !self.capabilities.can_vector_search {
            return Err(KernelError::Plugin {
                message: format!(
                    "plugin '{}' not authorized for vector search",
                    self.plugin_name
                ),
            });
        }
        Ok(())
    }

    /// Check if the plugin can perform index search
    pub fn check_index_search(&self) -> KernelResult<()> {
        if !self.capabilities.can_index_search {
            return Err(KernelError::Plugin {
                message: format!(
                    "plugin '{}' not authorized for index search",
                    self.plugin_name
                ),
            });
        }
        Ok(())
    }

    /// Add an audit entry
    pub fn audit(&mut self, function: &str, table: Option<&str>, status: i32, rows: u64) {
        self.audit_log.push(AuditEntry {
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            function: function.to_string(),
            table: table.map(|s| s.to_string()),
            status,
            rows_affected: rows,
        });
    }
}

// ============================================================================
// Host Function Trait
// ============================================================================

/// Trait for host function implementations
///
/// Each host function that WASM plugins can call implements this trait.
pub trait HostFunction: Send + Sync {
    /// Function name as seen from WASM
    fn name(&self) -> &str;

    /// Execute the function
    fn execute(&self, ctx: &mut HostFunctionContext, args: &[u8]) -> HostCallResult;

    /// Human-readable description
    fn description(&self) -> &str;
}

// ============================================================================
// Host Function: soch_read
// ============================================================================

/// Read rows from a table
///
/// ## Arguments (serialized)
/// - `table_name: String` - Name of the table to read
/// - `key: Option<Vec<u8>>` - Optional key for point lookup
/// - `limit: u32` - Maximum rows to return
///
/// ## Returns
/// - Serialized rows in TOON format
pub struct SochRead {
    /// Storage backend accessor (simulated)
    _marker: std::marker::PhantomData<()>,
}

impl SochRead {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl Default for SochRead {
    fn default() -> Self {
        Self::new()
    }
}

impl HostFunction for SochRead {
    fn name(&self) -> &str {
        "soch_read"
    }

    fn description(&self) -> &str {
        "Read rows from a table with optional key filter"
    }

    fn execute(&self, ctx: &mut HostFunctionContext, args: &[u8]) -> HostCallResult {
        // Parse arguments (simplified - in production use proper serialization)
        let args_str = match std::str::from_utf8(args) {
            Ok(s) => s,
            Err(_) => {
                ctx.audit("soch_read", None, -3, 0);
                return HostCallResult::InvalidArgs("invalid UTF-8 in arguments".to_string());
            }
        };

        // Parse table name (first line)
        let table = args_str.lines().next().unwrap_or("");

        // Check permission
        if let Err(e) = ctx.check_read(table) {
            ctx.audit("soch_read", Some(table), -1, 0);
            return HostCallResult::PermissionDenied(e.to_string());
        }

        // Simulate reading data
        // In production, this would call into the storage layer
        let mock_data = "table[1]{id,name}:\n(1,\"mock_row\")"
            .to_string()
            .into_bytes();

        ctx.audit("soch_read", Some(table), 0, 1);
        HostCallResult::Success(mock_data)
    }
}

// ============================================================================
// Host Function: soch_write
// ============================================================================

/// Write rows to a table
///
/// ## Arguments (serialized)
/// - `table_name: String` - Name of the table to write
/// - `rows: Vec<SochRow>` - Rows to insert/update
///
/// ## Returns
/// - Number of rows written
pub struct SochWrite {
    _marker: std::marker::PhantomData<()>,
}

impl SochWrite {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl Default for SochWrite {
    fn default() -> Self {
        Self::new()
    }
}

impl HostFunction for SochWrite {
    fn name(&self) -> &str {
        "soch_write"
    }

    fn description(&self) -> &str {
        "Write rows to a table"
    }

    fn execute(&self, ctx: &mut HostFunctionContext, args: &[u8]) -> HostCallResult {
        let args_str = match std::str::from_utf8(args) {
            Ok(s) => s,
            Err(_) => {
                ctx.audit("soch_write", None, -3, 0);
                return HostCallResult::InvalidArgs("invalid UTF-8 in arguments".to_string());
            }
        };

        let table = args_str.lines().next().unwrap_or("");

        if let Err(e) = ctx.check_write(table) {
            ctx.audit("soch_write", Some(table), -1, 0);
            return HostCallResult::PermissionDenied(e.to_string());
        }

        // Count rows (simplified - each line after table name is a row)
        let row_count = args_str.lines().skip(1).count() as u64;

        ctx.audit("soch_write", Some(table), 0, row_count);
        HostCallResult::Success(row_count.to_le_bytes().to_vec())
    }
}

// ============================================================================
// Host Function: vector_search
// ============================================================================

/// Vector similarity search
///
/// ## Arguments (serialized)
/// - `collection: String` - Vector collection name
/// - `query: Vec<f32>` - Query embedding
/// - `k: u32` - Number of results
/// - `filter: Option<String>` - Optional filter expression
///
/// ## Returns
/// - Serialized vector of (row_id, distance) pairs
pub struct VectorSearch {
    _marker: std::marker::PhantomData<()>,
}

impl VectorSearch {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl Default for VectorSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl HostFunction for VectorSearch {
    fn name(&self) -> &str {
        "vector_search"
    }

    fn description(&self) -> &str {
        "Perform vector similarity search"
    }

    fn execute(&self, ctx: &mut HostFunctionContext, args: &[u8]) -> HostCallResult {
        if let Err(e) = ctx.check_vector_search() {
            ctx.audit("vector_search", None, -1, 0);
            return HostCallResult::PermissionDenied(e.to_string());
        }

        // Parse collection name from args (simplified)
        let args_str = std::str::from_utf8(args).unwrap_or("");
        let collection = args_str.lines().next().unwrap_or("default");

        // Simulate search results
        // In production, this would call the HNSW/Vamana index
        let mock_results: Vec<(RowId, f32)> = vec![(1, 0.1), (2, 0.2), (3, 0.3)];

        // Serialize results
        let mut result = Vec::new();
        for (row_id, distance) in mock_results {
            result.extend_from_slice(&row_id.to_le_bytes());
            result.extend_from_slice(&distance.to_le_bytes());
        }

        ctx.audit("vector_search", Some(collection), 0, 3);
        HostCallResult::Success(result)
    }
}

// ============================================================================
// Host Function: emit_metric
// ============================================================================

/// Emit an observability metric
///
/// ## Arguments (serialized)
/// - `metric_type: u8` - 0=counter, 1=gauge, 2=histogram
/// - `name: String` - Metric name
/// - `value: f64` - Metric value
/// - `labels: Vec<(String, String)>` - Label pairs
///
/// ## Returns
/// - Ok
pub struct EmitMetric {
    /// Counter for total metrics emitted
    metrics_emitted: AtomicU64,
}

impl EmitMetric {
    pub fn new() -> Self {
        Self {
            metrics_emitted: AtomicU64::new(0),
        }
    }

    pub fn total_emitted(&self) -> u64 {
        self.metrics_emitted.load(Ordering::Relaxed)
    }
}

impl Default for EmitMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl HostFunction for EmitMetric {
    fn name(&self) -> &str {
        "emit_metric"
    }

    fn description(&self) -> &str {
        "Emit an observability metric (counter, gauge, or histogram)"
    }

    fn execute(&self, ctx: &mut HostFunctionContext, args: &[u8]) -> HostCallResult {
        // Metrics are always allowed (no permission check needed)

        if args.is_empty() {
            ctx.audit("emit_metric", None, -3, 0);
            return HostCallResult::InvalidArgs("empty metric data".to_string());
        }

        // In production, this would forward to ObservabilityExtension
        self.metrics_emitted.fetch_add(1, Ordering::Relaxed);

        ctx.audit("emit_metric", None, 0, 1);
        HostCallResult::Ok
    }
}

// ============================================================================
// Host Function: log_message
// ============================================================================

/// Log a message
///
/// ## Arguments (serialized)
/// - `level: u8` - 0=debug, 1=info, 2=warn, 3=error
/// - `message: String` - Log message
///
/// ## Returns
/// - Ok
pub struct LogMessage {
    /// Buffer for captured logs (useful for testing)
    logs: parking_lot::RwLock<Vec<(u8, String)>>,
}

impl LogMessage {
    pub fn new() -> Self {
        Self {
            logs: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Get captured logs
    pub fn captured_logs(&self) -> Vec<(u8, String)> {
        self.logs.read().clone()
    }

    /// Clear captured logs
    pub fn clear_logs(&self) {
        self.logs.write().clear();
    }
}

impl Default for LogMessage {
    fn default() -> Self {
        Self::new()
    }
}

impl HostFunction for LogMessage {
    fn name(&self) -> &str {
        "log_message"
    }

    fn description(&self) -> &str {
        "Log a message at specified level"
    }

    fn execute(&self, ctx: &mut HostFunctionContext, args: &[u8]) -> HostCallResult {
        // Logging is always allowed

        if args.is_empty() {
            return HostCallResult::InvalidArgs("empty log data".to_string());
        }

        let level = args[0];
        let message = std::str::from_utf8(&args[1..]).unwrap_or("<invalid UTF-8>");

        // Capture the log
        self.logs.write().push((level, message.to_string()));

        // In production, forward to logging system
        ctx.audit("log_message", None, 0, 0);
        HostCallResult::Ok
    }
}

// ============================================================================
// Host Function Registry
// ============================================================================

/// Registry of available host functions
pub struct HostFunctionRegistry {
    /// Functions by name
    functions: HashMap<String, Arc<dyn HostFunction>>,
}

impl Default for HostFunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HostFunctionRegistry {
    /// Create a new registry with standard functions
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };

        // Register standard functions
        registry.register(Arc::new(SochRead::new()));
        registry.register(Arc::new(SochWrite::new()));
        registry.register(Arc::new(VectorSearch::new()));
        registry.register(Arc::new(EmitMetric::new()));
        registry.register(Arc::new(LogMessage::new()));

        registry
    }

    /// Register a host function
    pub fn register(&mut self, func: Arc<dyn HostFunction>) {
        self.functions.insert(func.name().to_string(), func);
    }

    /// Get a host function by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn HostFunction>> {
        self.functions.get(name).cloned()
    }

    /// List all available functions
    pub fn list(&self) -> Vec<(&str, &str)> {
        self.functions
            .values()
            .map(|f| (f.name(), f.description()))
            .collect()
    }

    /// Execute a host function
    pub fn execute(
        &self,
        name: &str,
        ctx: &mut HostFunctionContext,
        args: &[u8],
    ) -> HostCallResult {
        match self.functions.get(name) {
            Some(func) => func.execute(ctx, args),
            None => HostCallResult::NotFound(format!("host function '{}' not found", name)),
        }
    }
}

// ============================================================================
// Wire Format for WASM Boundary
// ============================================================================

/// Serialization helpers for crossing WASM boundary
pub mod wire {
    /// Serialize a string for WASM
    pub fn encode_string(s: &str) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + s.len());
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    /// Deserialize a string from WASM
    pub fn decode_string(data: &[u8]) -> Option<(&str, &[u8])> {
        if data.len() < 4 {
            return None;
        }
        let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + len {
            return None;
        }
        let s = std::str::from_utf8(&data[4..4 + len]).ok()?;
        Some((s, &data[4 + len..]))
    }

    /// Serialize a row ID
    pub fn encode_row_id(id: u64) -> [u8; 8] {
        id.to_le_bytes()
    }

    /// Deserialize a row ID
    pub fn decode_row_id(data: &[u8]) -> Option<(u64, &[u8])> {
        if data.len() < 8 {
            return None;
        }
        let id = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        Some((id, &data[8..]))
    }

    /// Serialize an f32 vector
    pub fn encode_f32_vec(v: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + v.len() * 4);
        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for f in v {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    /// Deserialize an f32 vector
    pub fn decode_f32_vec(data: &[u8]) -> Option<(Vec<f32>, &[u8])> {
        if data.len() < 4 {
            return None;
        }
        let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + len * 4 {
            return None;
        }
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let offset = 4 + i * 4;
            let f = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            vec.push(f);
        }
        Some((vec, &data[4 + len * 4..]))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_call_result_status() {
        assert_eq!(HostCallResult::Ok.status_code(), 0);
        assert_eq!(HostCallResult::Success(vec![]).status_code(), 0);
        assert_eq!(
            HostCallResult::PermissionDenied("".to_string()).status_code(),
            -1
        );
        assert_eq!(HostCallResult::NotFound("".to_string()).status_code(), -2);
        assert_eq!(
            HostCallResult::InvalidArgs("".to_string()).status_code(),
            -3
        );
        assert_eq!(HostCallResult::Error("".to_string()).status_code(), -4);
    }

    #[test]
    fn test_host_context_permission_checks() {
        let caps = WasmPluginCapabilities {
            can_read_table: vec!["users".to_string()],
            can_write_table: vec!["logs".to_string()],
            can_vector_search: true,
            can_index_search: false,
            ..Default::default()
        };

        let ctx = HostFunctionContext::new("test_plugin", caps);

        assert!(ctx.check_read("users").is_ok());
        assert!(ctx.check_read("other").is_err());
        assert!(ctx.check_write("logs").is_ok());
        assert!(ctx.check_write("users").is_err());
        assert!(ctx.check_vector_search().is_ok());
        assert!(ctx.check_index_search().is_err());
    }

    #[test]
    fn test_soch_read_permission() {
        let caps = WasmPluginCapabilities {
            can_read_table: vec!["allowed_table".to_string()],
            ..Default::default()
        };

        let mut ctx = HostFunctionContext::new("test", caps);
        let read_fn = SochRead::new();

        // Allowed table
        let result = read_fn.execute(&mut ctx, b"allowed_table\n");
        assert_eq!(result.status_code(), 0);

        // Denied table
        let result = read_fn.execute(&mut ctx, b"denied_table\n");
        assert_eq!(result.status_code(), -1);
    }

    #[test]
    fn test_soch_write_permission() {
        let caps = WasmPluginCapabilities {
            can_write_table: vec!["writable".to_string()],
            ..Default::default()
        };

        let mut ctx = HostFunctionContext::new("test", caps);
        let write_fn = SochWrite::new();

        let result = write_fn.execute(&mut ctx, b"writable\nrow1\nrow2\n");
        assert_eq!(result.status_code(), 0);
        assert_eq!(result.data().unwrap(), &2u64.to_le_bytes());

        let result = write_fn.execute(&mut ctx, b"readonly\nrow1\n");
        assert_eq!(result.status_code(), -1);
    }

    #[test]
    fn test_vector_search() {
        let caps = WasmPluginCapabilities {
            can_vector_search: true,
            ..Default::default()
        };

        let mut ctx = HostFunctionContext::new("test", caps);
        let search_fn = VectorSearch::new();

        let result = search_fn.execute(&mut ctx, b"collection\n");
        assert_eq!(result.status_code(), 0);

        // Should contain 3 results (row_id, distance pairs)
        let data = result.data().unwrap();
        assert_eq!(data.len(), 3 * (8 + 4)); // 3 * (u64 + f32)
    }

    #[test]
    fn test_emit_metric() {
        let caps = WasmPluginCapabilities::default();
        let mut ctx = HostFunctionContext::new("test", caps);
        let metric_fn = EmitMetric::new();

        let result = metric_fn.execute(&mut ctx, b"\x01metric_name\x00\x00\x00\x00");
        assert_eq!(result.status_code(), 0);
        assert_eq!(metric_fn.total_emitted(), 1);
    }

    #[test]
    fn test_log_message() {
        let caps = WasmPluginCapabilities::default();
        let mut ctx = HostFunctionContext::new("test", caps);
        let log_fn = LogMessage::new();

        let result = log_fn.execute(&mut ctx, b"\x01hello world");
        assert_eq!(result.status_code(), 0);

        let logs = log_fn.captured_logs();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].0, 1); // info level
        assert_eq!(logs[0].1, "hello world");
    }

    #[test]
    fn test_host_function_registry() {
        let registry = HostFunctionRegistry::new();

        // All standard functions should be registered
        assert!(registry.get("soch_read").is_some());
        assert!(registry.get("soch_write").is_some());
        assert!(registry.get("vector_search").is_some());
        assert!(registry.get("emit_metric").is_some());
        assert!(registry.get("log_message").is_some());

        // Unknown function
        assert!(registry.get("unknown").is_none());

        // List should have all functions
        let list = registry.list();
        assert!(list.len() >= 5);
    }

    #[test]
    fn test_registry_execute() {
        let registry = HostFunctionRegistry::new();
        let caps = WasmPluginCapabilities {
            can_read_table: vec!["test".to_string()],
            ..Default::default()
        };
        let mut ctx = HostFunctionContext::new("plugin", caps);

        let result = registry.execute("soch_read", &mut ctx, b"test\n");
        assert_eq!(result.status_code(), 0);

        let result = registry.execute("nonexistent", &mut ctx, b"");
        assert_eq!(result.status_code(), -2);
    }

    #[test]
    fn test_audit_log() {
        let caps = WasmPluginCapabilities {
            can_read_table: vec!["audit_test".to_string()],
            ..Default::default()
        };
        let mut ctx = HostFunctionContext::new("test", caps);
        let read_fn = SochRead::new();

        let _ = read_fn.execute(&mut ctx, b"audit_test\n");

        assert_eq!(ctx.audit_log.len(), 1);
        assert_eq!(ctx.audit_log[0].function, "soch_read");
        assert_eq!(ctx.audit_log[0].table, Some("audit_test".to_string()));
        assert_eq!(ctx.audit_log[0].status, 0);
    }

    // Wire format tests
    mod wire_tests {
        use super::super::wire::*;

        #[test]
        fn test_encode_decode_string() {
            let s = "hello world";
            let encoded = encode_string(s);
            let (decoded, rest) = decode_string(&encoded).unwrap();
            assert_eq!(decoded, s);
            assert!(rest.is_empty());
        }

        #[test]
        fn test_encode_decode_row_id() {
            let id = 0x123456789ABCDEF0u64;
            let encoded = encode_row_id(id);
            let (decoded, rest) = decode_row_id(&encoded).unwrap();
            assert_eq!(decoded, id);
            assert!(rest.is_empty());
        }

        #[test]
        fn test_encode_decode_f32_vec() {
            let v = vec![1.0, 2.0, 3.0, 4.0];
            let encoded = encode_f32_vec(&v);
            let (decoded, rest) = decode_f32_vec(&encoded).unwrap();
            assert_eq!(decoded, v);
            assert!(rest.is_empty());
        }

        #[test]
        fn test_decode_empty() {
            assert!(decode_string(&[]).is_none());
            assert!(decode_row_id(&[]).is_none());
            assert!(decode_f32_vec(&[]).is_none());
        }
    }
}
