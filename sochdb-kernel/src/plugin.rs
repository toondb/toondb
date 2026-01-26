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

//! Plugin Architecture
//!
//! SochDB uses a plugin architecture to keep the kernel minimal while
//! allowing rich functionality through extensions.
//!
//! ## Design Philosophy
//!
//! 1. **Core is Minimal**: The kernel contains only ACID-critical code
//! 2. **Extensions Add Features**: Storage backends, indices, observability are plugins
//! 3. **No Bloat**: Users only pay for what they use
//! 4. **Vendor Neutral**: No lock-in to specific monitoring/storage systems
//!
//! ## Plugin Categories
//!
//! - `StorageExtension`: Alternative storage backends (LSCS, RocksDB, etc.)
//! - `IndexExtension`: Custom index types (vector, learned, etc.)
//! - `ObservabilityExtension`: Metrics, tracing, logging backends
//! - `CompressionExtension`: Compression algorithms
//!
//! ## Example: Adding Prometheus Metrics
//!
//! ```ignore
//! // In a separate crate: sochdb-prometheus-plugin
//! struct PrometheusPlugin { /* ... */ }
//!
//! impl ObservabilityExtension for PrometheusPlugin {
//!     fn record_metric(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
//!         // Push to Prometheus
//!     }
//! }
//!
//! // Usage:
//! let db = KernelDB::open(path)?;
//! db.plugins().register_observability(Box::new(PrometheusPlugin::new()))?;
//! ```

use crate::error::{KernelError, KernelResult};
use crate::kernel_api::{HealthInfo, RowId, TableId};
use crate::transaction::TransactionId;
use parking_lot::RwLock;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Extension Trait Definitions
// ============================================================================

/// Information about an extension
#[derive(Debug, Clone)]
pub struct ExtensionInfo {
    /// Unique extension name (e.g., "prometheus-metrics")
    pub name: String,
    /// Extension version
    pub version: String,
    /// Human-readable description
    pub description: String,
    /// Extension author
    pub author: String,
    /// Capabilities provided
    pub capabilities: Vec<ExtensionCapability>,
}

/// Capabilities an extension can provide
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExtensionCapability {
    /// Alternative storage backend
    Storage,
    /// Custom index type
    Index,
    /// Metrics/tracing/logging
    Observability,
    /// Compression algorithm
    Compression,
    /// Query optimization
    QueryOptimizer,
    /// Authentication/Authorization
    Auth,
    /// Custom - for third-party extensions
    Custom(String),
}

/// Base trait for all extensions
pub trait Extension: Send + Sync {
    /// Get extension information
    fn info(&self) -> ExtensionInfo;

    /// Initialize the extension
    fn init(&mut self) -> KernelResult<()> {
        Ok(())
    }

    /// Shutdown the extension gracefully
    fn shutdown(&mut self) -> KernelResult<()> {
        Ok(())
    }

    /// Cast to Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Cast to mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ============================================================================
// Storage Extension
// ============================================================================

/// Storage backend extension
///
/// Implement this to provide alternative storage engines (LSCS, RocksDB, etc.)
pub trait StorageExtension: Extension {
    /// Read data for a key
    fn get(&self, table_id: TableId, key: &[u8]) -> KernelResult<Option<Vec<u8>>>;

    /// Write data for a key
    fn put(
        &self,
        table_id: TableId,
        key: &[u8],
        value: &[u8],
        txn_id: TransactionId,
    ) -> KernelResult<()>;

    /// Delete a key
    fn delete(&self, table_id: TableId, key: &[u8], txn_id: TransactionId) -> KernelResult<()>;

    /// Scan a range of keys
    fn scan(
        &self,
        table_id: TableId,
        start: &[u8],
        end: &[u8],
        limit: usize,
    ) -> KernelResult<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Flush pending writes
    fn flush(&self) -> KernelResult<()>;

    /// Compact storage (if applicable)
    fn compact(&self) -> KernelResult<()> {
        Ok(()) // Default: no-op
    }

    /// Get storage statistics
    fn stats(&self) -> StorageStats {
        StorageStats::default()
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total bytes stored
    pub bytes_stored: u64,
    /// Number of keys
    pub key_count: u64,
    /// Pending compaction bytes
    pub pending_compaction_bytes: u64,
    /// Write amplification factor
    pub write_amplification: f64,
}

// ============================================================================
// Index Extension
// ============================================================================

/// Index extension
///
/// Implement this for custom index types (vector, learned, full-text, etc.)
pub trait IndexExtension: Extension {
    /// Index type name (e.g., "hnsw", "learned", "btree")
    fn index_type(&self) -> &str;

    /// Build index on existing data
    fn build(
        &mut self,
        table_id: TableId,
        column_id: u16,
        data: &[(RowId, Vec<u8>)],
    ) -> KernelResult<()>;

    /// Insert a key-value pair into the index
    fn insert(&mut self, key: &[u8], row_id: RowId) -> KernelResult<()>;

    /// Delete a key from the index
    fn delete(&mut self, key: &[u8], row_id: RowId) -> KernelResult<()>;

    /// Point lookup
    fn lookup(&self, key: &[u8]) -> KernelResult<Vec<RowId>>;

    /// Range scan
    fn range(&self, start: &[u8], end: &[u8], limit: usize) -> KernelResult<Vec<RowId>>;

    /// Nearest neighbor search (for vector indices)
    fn nearest(&self, _query: &[u8], _k: usize) -> KernelResult<Vec<(RowId, f32)>> {
        Err(KernelError::Plugin {
            message: "nearest neighbor not supported by this index type".into(),
        })
    }

    /// Get index size in bytes
    fn size_bytes(&self) -> u64;
}

// ============================================================================
// Observability Extension (PLUGIN ARCHITECTURE)
// ============================================================================

/// Observability extension
///
/// Implement this for metrics, tracing, and logging backends.
///
/// ## Why Plugin Architecture?
///
/// 1. **No Dependency Bloat**: Core doesn't pull in Prometheus, DataDog, etc.
/// 2. **Vendor Neutral**: Users choose their observability stack
/// 3. **Flexible**: Can run without any observability in embedded scenarios
///
/// ## Available Plugins (separate crates):
///
/// - `sochdb-prometheus`: Prometheus metrics
/// - `sochdb-datadog`: DataDog integration  
/// - `sochdb-opentelemetry`: OpenTelemetry support
/// - `sochdb-logging-json`: JSON structured logging
/// - `sochdb-logging-logfmt`: logfmt style logging
pub trait ObservabilityExtension: Extension {
    // -------------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------------

    /// Record a counter increment
    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]);

    /// Record a gauge value
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]);

    /// Record a histogram observation
    fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]);

    // -------------------------------------------------------------------------
    // Tracing
    // -------------------------------------------------------------------------

    /// Start a new span
    fn span_start(&self, name: &str, parent: Option<u64>) -> u64;

    /// End a span
    fn span_end(&self, span_id: u64);

    /// Add an event to a span
    fn span_event(&self, span_id: u64, name: &str, attributes: &[(&str, &str)]);

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    /// Log at debug level
    fn log_debug(&self, message: &str, fields: &[(&str, &str)]) {
        let _ = (message, fields); // Default: no-op
    }

    /// Log at info level
    fn log_info(&self, message: &str, fields: &[(&str, &str)]) {
        let _ = (message, fields); // Default: no-op
    }

    /// Log at warn level
    fn log_warn(&self, message: &str, fields: &[(&str, &str)]) {
        let _ = (message, fields); // Default: no-op
    }

    /// Log at error level
    fn log_error(&self, message: &str, fields: &[(&str, &str)]) {
        let _ = (message, fields); // Default: no-op
    }

    // -------------------------------------------------------------------------
    // Health Reporting
    // -------------------------------------------------------------------------

    /// Report health status (called periodically by kernel)
    fn report_health(&self, health: &HealthInfo) {
        let _ = health; // Default: no-op
    }
}

// ============================================================================
// Compression Extension
// ============================================================================

/// Compression algorithm extension
pub trait CompressionExtension: Extension {
    /// Algorithm name (e.g., "lz4", "zstd", "snappy")
    fn algorithm(&self) -> &str;

    /// Compress data
    fn compress(&self, input: &[u8]) -> KernelResult<Vec<u8>>;

    /// Decompress data
    fn decompress(&self, input: &[u8]) -> KernelResult<Vec<u8>>;

    /// Compression level (if applicable)
    fn set_level(&mut self, _level: i32) -> KernelResult<()> {
        Ok(())
    }
}

// ============================================================================
// Plugin Manager
// ============================================================================

/// Plugin manager - registry for all extensions
///
/// The kernel uses this to discover and invoke extensions.
pub struct PluginManager {
    /// Storage extensions by name
    storage: RwLock<HashMap<String, Arc<dyn StorageExtension>>>,
    /// Index extensions by name
    indices: RwLock<HashMap<String, Arc<RwLock<dyn IndexExtension>>>>,
    /// Observability extensions (can have multiple)
    observability: RwLock<Vec<Arc<dyn ObservabilityExtension>>>,
    /// Compression extensions by algorithm name
    compression: RwLock<HashMap<String, Arc<dyn CompressionExtension>>>,
    /// Active storage backend name
    active_storage: RwLock<Option<String>>,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(HashMap::new()),
            indices: RwLock::new(HashMap::new()),
            observability: RwLock::new(Vec::new()),
            compression: RwLock::new(HashMap::new()),
            active_storage: RwLock::new(None),
        }
    }

    // -------------------------------------------------------------------------
    // Storage Extensions
    // -------------------------------------------------------------------------

    /// Register a storage extension
    pub fn register_storage(&self, ext: Arc<dyn StorageExtension>) -> KernelResult<()> {
        let name = ext.info().name.clone();
        let mut storage = self.storage.write();

        if storage.contains_key(&name) {
            return Err(KernelError::Plugin {
                message: format!("storage extension '{}' already registered", name),
            });
        }

        storage.insert(name.clone(), ext);

        // Set as active if it's the first one
        let mut active = self.active_storage.write();
        if active.is_none() {
            *active = Some(name);
        }

        Ok(())
    }

    /// Set the active storage backend
    pub fn set_active_storage(&self, name: &str) -> KernelResult<()> {
        let storage = self.storage.read();
        if !storage.contains_key(name) {
            return Err(KernelError::Plugin {
                message: format!("storage extension '{}' not found", name),
            });
        }
        *self.active_storage.write() = Some(name.to_string());
        Ok(())
    }

    /// Get the active storage backend
    pub fn storage(&self) -> Option<Arc<dyn StorageExtension>> {
        let active = self.active_storage.read();
        active
            .as_ref()
            .and_then(|name| self.storage.read().get(name).cloned())
    }

    // -------------------------------------------------------------------------
    // Index Extensions
    // -------------------------------------------------------------------------

    /// Register an index extension
    pub fn register_index(&self, ext: Arc<RwLock<dyn IndexExtension>>) -> KernelResult<()> {
        let name = ext.read().info().name.clone();
        let mut indices = self.indices.write();

        if indices.contains_key(&name) {
            return Err(KernelError::Plugin {
                message: format!("index extension '{}' already registered", name),
            });
        }

        indices.insert(name, ext);
        Ok(())
    }

    /// Get an index extension by name
    pub fn index(&self, name: &str) -> Option<Arc<RwLock<dyn IndexExtension>>> {
        self.indices.read().get(name).cloned()
    }

    /// List registered index types
    pub fn list_index_types(&self) -> Vec<String> {
        self.indices.read().keys().cloned().collect()
    }

    // -------------------------------------------------------------------------
    // Observability Extensions
    // -------------------------------------------------------------------------

    /// Register an observability extension
    ///
    /// Multiple observability extensions can be registered (fan-out to all)
    pub fn register_observability(&self, ext: Arc<dyn ObservabilityExtension>) -> KernelResult<()> {
        self.observability.write().push(ext);
        Ok(())
    }

    /// Record a counter across all observability extensions
    pub fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        for ext in self.observability.read().iter() {
            ext.counter_inc(name, value, labels);
        }
    }

    /// Record a gauge across all observability extensions
    pub fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        for ext in self.observability.read().iter() {
            ext.gauge_set(name, value, labels);
        }
    }

    /// Record a histogram observation across all observability extensions
    pub fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        for ext in self.observability.read().iter() {
            ext.histogram_observe(name, value, labels);
        }
    }

    /// Report health to all observability extensions
    pub fn report_health(&self, health: &HealthInfo) {
        for ext in self.observability.read().iter() {
            ext.report_health(health);
        }
    }

    /// Check if any observability is configured
    pub fn has_observability(&self) -> bool {
        !self.observability.read().is_empty()
    }

    // -------------------------------------------------------------------------
    // Compression Extensions
    // -------------------------------------------------------------------------

    /// Register a compression extension
    pub fn register_compression(&self, ext: Arc<dyn CompressionExtension>) -> KernelResult<()> {
        let algo = ext.algorithm().to_string();
        let mut compression = self.compression.write();

        if compression.contains_key(&algo) {
            return Err(KernelError::Plugin {
                message: format!("compression '{}' already registered", algo),
            });
        }

        compression.insert(algo, ext);
        Ok(())
    }

    /// Get a compression extension by algorithm name
    pub fn compression(&self, algorithm: &str) -> Option<Arc<dyn CompressionExtension>> {
        self.compression.read().get(algorithm).cloned()
    }

    /// List available compression algorithms
    pub fn list_compression(&self) -> Vec<String> {
        self.compression.read().keys().cloned().collect()
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /// Shutdown all extensions gracefully
    pub fn shutdown_all(&self) -> KernelResult<()> {
        // Extensions are immutable through Arc, so we can't call shutdown
        // In a real implementation, we'd use Arc<RwLock<dyn Extension>>
        // For now, this is a no-op placeholder
        Ok(())
    }

    /// Get information about all registered extensions
    pub fn list_extensions(&self) -> Vec<ExtensionInfo> {
        let mut result = Vec::new();

        for ext in self.storage.read().values() {
            result.push(ext.info());
        }

        for ext in self.indices.read().values() {
            result.push(ext.read().info());
        }

        for ext in self.observability.read().iter() {
            result.push(ext.info());
        }

        for ext in self.compression.read().values() {
            result.push(ext.info());
        }

        result
    }
}

// ============================================================================
// Null Observability (Default - No-Op)
// ============================================================================

/// Null observability extension - does nothing
///
/// Used when no observability is configured. Zero overhead.
pub struct NullObservability;

impl Extension for NullObservability {
    fn info(&self) -> ExtensionInfo {
        ExtensionInfo {
            name: "null-observability".into(),
            version: "0.0.0".into(),
            description: "No-op observability (default)".into(),
            author: "SochDB".into(),
            capabilities: vec![ExtensionCapability::Observability],
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ObservabilityExtension for NullObservability {
    fn counter_inc(&self, _name: &str, _value: u64, _labels: &[(&str, &str)]) {}
    fn gauge_set(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
    fn histogram_observe(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
    fn span_start(&self, _name: &str, _parent: Option<u64>) -> u64 {
        0
    }
    fn span_end(&self, _span_id: u64) {}
    fn span_event(&self, _span_id: u64, _name: &str, _attributes: &[(&str, &str)]) {}
}

// ============================================================================
// Dynamic Plugin Loading (Optional Feature)
// ============================================================================

#[cfg(feature = "dynamic-plugins")]
pub mod dynamic {
    //! Dynamic plugin loading support
    //!
    //! Enabled with the `dynamic-plugins` feature.
    //! Allows loading plugins from shared libraries at runtime.

    use super::*;
    use libloading::{Library, Symbol};
    use std::path::Path;

    /// Dynamic plugin loader
    pub struct DynamicPluginLoader {
        /// Loaded libraries (kept alive)
        _libraries: Vec<Library>,
    }

    impl DynamicPluginLoader {
        /// Create a new dynamic plugin loader
        pub fn new() -> Self {
            Self {
                _libraries: Vec::new(),
            }
        }

        /// Load an observability plugin from a shared library
        ///
        /// The library must export a function:
        /// ```c
        /// extern "C" fn create_observability_plugin() -> *mut dyn ObservabilityExtension
        /// ```
        pub fn load_observability(
            &mut self,
            path: &Path,
        ) -> KernelResult<Arc<dyn ObservabilityExtension>> {
            unsafe {
                let lib = Library::new(path).map_err(|e| KernelError::Plugin {
                    message: format!("failed to load library: {}", e),
                })?;

                let create_fn: Symbol<fn() -> Box<dyn ObservabilityExtension>> = lib
                    .get(b"create_observability_plugin")
                    .map_err(|e| KernelError::Plugin {
                        message: format!("symbol not found: {}", e),
                    })?;

                let plugin = create_fn();
                self._libraries.push(lib);

                Ok(Arc::from(plugin))
            }
        }
    }

    impl Default for DynamicPluginLoader {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_manager_creation() {
        let pm = PluginManager::new();
        assert!(!pm.has_observability());
        assert!(pm.storage().is_none());
    }

    #[test]
    fn test_null_observability() {
        let null = NullObservability;
        // Should not panic
        null.counter_inc("test", 1, &[]);
        null.gauge_set("test", 1.0, &[]);
        null.histogram_observe("test", 1.0, &[]);
        let span = null.span_start("test", None);
        null.span_event(span, "event", &[]);
        null.span_end(span);
    }

    #[test]
    fn test_register_observability() {
        let pm = PluginManager::new();
        let null = Arc::new(NullObservability);

        assert!(!pm.has_observability());
        pm.register_observability(null).unwrap();
        assert!(pm.has_observability());
    }
}
