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

//! Lightweight Namespace Routing + On-Disk Layout (Task 3)
//!
//! This module provides physical namespace isolation with:
//! - Namespace registry (`_namespaces.meta`)
//! - Per-namespace directory layout
//! - Database router for (namespace, collection) → storage handles
//!
//! ## On-Disk Layout
//!
//! ```text
//! data/
//! ├── _namespaces.meta          # Registry of all namespaces
//! ├── _global/                  # Shared metadata
//! │   └── catalog.db
//! └── namespaces/
//!     ├── tenant_a/
//!     │   ├── _meta.json        # Namespace metadata
//!     │   ├── collections/
//!     │   │   ├── docs/
//!     │   │   │   ├── vectors.idx
//!     │   │   │   └── data.db
//!     │   │   └── images/
//!     │   │       ├── vectors.idx
//!     │   │       └── data.db
//!     │   └── kv/                # Key-value storage
//!     │       └── data.db
//!     └── tenant_b/
//!         └── ...
//! ```
//!
//! ## Name Resolution
//!
//! Resolution is O(1) via hash-map lookup:
//!
//! ```text
//! (namespace="tenant_a", collection="docs")
//!     → NamespaceHandle { root: "data/namespaces/tenant_a" }
//!         → CollectionHandle { path: "data/namespaces/tenant_a/collections/docs" }
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ============================================================================
// Namespace Error Types
// ============================================================================

/// Errors related to namespace operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum NamespaceStorageError {
    #[error("namespace not found: {0}")]
    NotFound(String),

    #[error("namespace already exists: {0}")]
    AlreadyExists(String),

    #[error("invalid namespace name: {0}")]
    InvalidName(String),

    #[error("collection not found: {namespace}/{collection}")]
    CollectionNotFound { namespace: String, collection: String },

    #[error("collection already exists: {namespace}/{collection}")]
    CollectionAlreadyExists { namespace: String, collection: String },

    #[error("storage I/O error: {0}")]
    IoError(String),

    #[error("namespace is read-only: {0}")]
    ReadOnly(String),

    #[error("namespace registry corrupted: {0}")]
    RegistryCorrupted(String),
}

impl From<std::io::Error> for NamespaceStorageError {
    fn from(e: std::io::Error) -> Self {
        NamespaceStorageError::IoError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, NamespaceStorageError>;

// ============================================================================
// Namespace Metadata
// ============================================================================

/// Metadata for a namespace (stored in _meta.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceMeta {
    /// Unique namespace identifier
    pub name: String,
    
    /// Human-readable display name
    pub display_name: Option<String>,
    
    /// Creation timestamp (Unix epoch millis)
    pub created_at: u64,
    
    /// Last modified timestamp (Unix epoch millis)
    pub updated_at: u64,
    
    /// Whether the namespace is read-only
    pub read_only: bool,
    
    /// Custom metadata/labels
    pub labels: HashMap<String, String>,
    
    /// Collections in this namespace
    #[serde(default)]
    pub collections: Vec<String>,
}

impl NamespaceMeta {
    /// Create a new namespace metadata
    pub fn new(name: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Self {
            name: name.into(),
            display_name: None,
            created_at: now,
            updated_at: now,
            read_only: false,
            labels: HashMap::new(),
            collections: Vec::new(),
        }
    }
    
    /// Set display name
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }
    
    /// Set labels
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }
}

/// Namespace registry (stored in _namespaces.meta)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NamespaceRegistry {
    /// Version for forward compatibility
    pub version: u32,
    
    /// List of registered namespaces
    pub namespaces: Vec<NamespaceEntry>,
}

/// Entry in the namespace registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceEntry {
    /// Namespace name (directory name)
    pub name: String,
    
    /// Creation timestamp
    pub created_at: u64,
    
    /// Whether the namespace is active
    pub active: bool,
}

// ============================================================================
// Collection Config
// ============================================================================

/// Collection configuration (stored in collection directory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Collection name
    pub name: String,
    
    /// Vector dimension (if applicable)
    pub dimension: Option<usize>,
    
    /// Distance metric
    pub metric: DistanceMetric,
    
    /// Index configuration
    pub index_config: IndexConfig,
    
    /// Creation timestamp
    pub created_at: u64,
    
    /// Whether the config is frozen (immutable after creation)
    pub frozen: bool,
}

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// HNSW M parameter
    pub m: usize,
    
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    
    /// Enable quantization
    pub quantization: Option<QuantizationType>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            quantization: None,
        }
    }
}

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    Scalar,  // int8 quantization
    PQ,      // Product quantization
}

// ============================================================================
// Namespace Handle
// ============================================================================

/// Handle to a namespace's storage
#[derive(Debug, Clone)]
pub struct NamespaceHandle {
    /// Namespace name
    pub name: String,
    
    /// Root directory for this namespace
    pub root: PathBuf,
    
    /// Metadata
    pub meta: Arc<RwLock<NamespaceMeta>>,
}

impl NamespaceHandle {
    /// Get the collections directory
    pub fn collections_dir(&self) -> PathBuf {
        self.root.join("collections")
    }
    
    /// Get the key-value storage directory
    pub fn kv_dir(&self) -> PathBuf {
        self.root.join("kv")
    }
    
    /// Get a collection path
    pub fn collection_path(&self, collection: &str) -> PathBuf {
        self.collections_dir().join(collection)
    }
    
    /// Check if collection exists
    pub fn has_collection(&self, collection: &str) -> bool {
        self.collection_path(collection).exists()
    }
    
    /// List collections
    pub fn list_collections(&self) -> Result<Vec<String>> {
        let collections_dir = self.collections_dir();
        if !collections_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut collections = Vec::new();
        for entry in fs::read_dir(&collections_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    collections.push(name.to_string());
                }
            }
        }
        Ok(collections)
    }
}

/// Handle to a collection's storage
#[derive(Debug, Clone)]
pub struct CollectionHandle {
    /// Namespace this collection belongs to
    pub namespace: String,
    
    /// Collection name
    pub name: String,
    
    /// Root directory for this collection
    pub root: PathBuf,
    
    /// Configuration (immutable after creation)
    pub config: Arc<CollectionConfig>,
}

impl CollectionHandle {
    /// Get the vector index path
    pub fn vectors_path(&self) -> PathBuf {
        self.root.join("vectors.idx")
    }
    
    /// Get the data storage path
    pub fn data_path(&self) -> PathBuf {
        self.root.join("data.db")
    }
    
    /// Get the metadata index path
    pub fn metadata_path(&self) -> PathBuf {
        self.root.join("metadata.idx")
    }
    
    /// Get the tombstones path
    pub fn tombstones_path(&self) -> PathBuf {
        self.root.join("tombstones.bin")
    }
}

// ============================================================================
// Namespace Router (Main Interface)
// ============================================================================

/// Database router for namespace resolution
///
/// Resolves (namespace, collection) → storage handles with O(1) lookup.
pub struct NamespaceRouter {
    /// Base data directory
    data_dir: PathBuf,
    
    /// Namespace registry (cached)
    registry: RwLock<NamespaceRegistry>,
    
    /// Loaded namespace handles (cache)
    namespaces: RwLock<HashMap<String, Arc<NamespaceHandle>>>,
    
    /// Loaded collection handles (cache)
    collections: RwLock<HashMap<(String, String), Arc<CollectionHandle>>>,
}

impl NamespaceRouter {
    /// Create a new namespace router
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        
        // Ensure directories exist
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(data_dir.join("namespaces"))?;
        fs::create_dir_all(data_dir.join("_global"))?;
        
        // Load or create registry
        let registry_path = data_dir.join("_namespaces.meta");
        let registry = if registry_path.exists() {
            let file = File::open(&registry_path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)
                .map_err(|e| NamespaceStorageError::RegistryCorrupted(e.to_string()))?
        } else {
            NamespaceRegistry {
                version: 1,
                namespaces: Vec::new(),
            }
        };
        
        Ok(Self {
            data_dir,
            registry: RwLock::new(registry),
            namespaces: RwLock::new(HashMap::new()),
            collections: RwLock::new(HashMap::new()),
        })
    }
    
    /// Save the registry to disk
    fn save_registry(&self) -> Result<()> {
        let registry_path = self.data_dir.join("_namespaces.meta");
        let file = File::create(&registry_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &*self.registry.read())
            .map_err(|e| NamespaceStorageError::IoError(e.to_string()))?;
        Ok(())
    }
    
    // ========================================================================
    // Namespace Operations
    // ========================================================================
    
    /// Create a new namespace
    pub fn create_namespace(&self, meta: NamespaceMeta) -> Result<Arc<NamespaceHandle>> {
        let name = meta.name.clone();
        
        // Validate name
        Self::validate_namespace_name(&name)?;
        
        // Check if already exists
        {
            let registry = self.registry.read();
            if registry.namespaces.iter().any(|e| e.name == name) {
                return Err(NamespaceStorageError::AlreadyExists(name));
            }
        }
        
        // Create directory structure
        let namespace_dir = self.data_dir.join("namespaces").join(&name);
        fs::create_dir_all(&namespace_dir)?;
        fs::create_dir_all(namespace_dir.join("collections"))?;
        fs::create_dir_all(namespace_dir.join("kv"))?;
        
        // Write namespace metadata
        let meta_path = namespace_dir.join("_meta.json");
        let file = File::create(&meta_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &meta)
            .map_err(|e| NamespaceStorageError::IoError(e.to_string()))?;
        
        // Update registry
        {
            let mut registry = self.registry.write();
            registry.namespaces.push(NamespaceEntry {
                name: name.clone(),
                created_at: meta.created_at,
                active: true,
            });
        }
        self.save_registry()?;
        
        // Create and cache handle
        let handle = Arc::new(NamespaceHandle {
            name: name.clone(),
            root: namespace_dir,
            meta: Arc::new(RwLock::new(meta)),
        });
        
        self.namespaces.write().insert(name, handle.clone());
        
        Ok(handle)
    }
    
    /// Get a namespace handle
    pub fn get_namespace(&self, name: &str) -> Result<Arc<NamespaceHandle>> {
        // Check cache first
        if let Some(handle) = self.namespaces.read().get(name) {
            return Ok(handle.clone());
        }
        
        // Check registry
        {
            let registry = self.registry.read();
            if !registry.namespaces.iter().any(|e| e.name == name && e.active) {
                return Err(NamespaceStorageError::NotFound(name.to_string()));
            }
        }
        
        // Load from disk
        let namespace_dir = self.data_dir.join("namespaces").join(name);
        if !namespace_dir.exists() {
            return Err(NamespaceStorageError::NotFound(name.to_string()));
        }
        
        let meta_path = namespace_dir.join("_meta.json");
        let meta: NamespaceMeta = if meta_path.exists() {
            let file = File::open(&meta_path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)
                .map_err(|e| NamespaceStorageError::RegistryCorrupted(e.to_string()))?
        } else {
            NamespaceMeta::new(name)
        };
        
        let handle = Arc::new(NamespaceHandle {
            name: name.to_string(),
            root: namespace_dir,
            meta: Arc::new(RwLock::new(meta)),
        });
        
        self.namespaces.write().insert(name.to_string(), handle.clone());
        
        Ok(handle)
    }
    
    /// List all namespaces
    pub fn list_namespaces(&self) -> Vec<String> {
        self.registry
            .read()
            .namespaces
            .iter()
            .filter(|e| e.active)
            .map(|e| e.name.clone())
            .collect()
    }
    
    /// Delete a namespace (marks as inactive, doesn't delete files)
    pub fn delete_namespace(&self, name: &str) -> Result<()> {
        {
            let mut registry = self.registry.write();
            let entry = registry
                .namespaces
                .iter_mut()
                .find(|e| e.name == name)
                .ok_or_else(|| NamespaceStorageError::NotFound(name.to_string()))?;
            entry.active = false;
        }
        
        self.save_registry()?;
        self.namespaces.write().remove(name);
        
        // Remove cached collections for this namespace
        self.collections
            .write()
            .retain(|(ns, _), _| ns != name);
        
        Ok(())
    }
    
    /// Validate namespace name
    fn validate_namespace_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(NamespaceStorageError::InvalidName(
                "namespace name cannot be empty".to_string(),
            ));
        }
        
        if name.len() > 256 {
            return Err(NamespaceStorageError::InvalidName(
                "namespace name too long (max 256 chars)".to_string(),
            ));
        }
        
        // Must start with alphanumeric
        let first = name.chars().next().unwrap();
        if !first.is_alphanumeric() {
            return Err(NamespaceStorageError::InvalidName(format!(
                "namespace name must start with alphanumeric, got '{}'",
                first
            )));
        }
        
        // Only allow alphanumeric, underscore, hyphen, dot
        for ch in name.chars() {
            if !ch.is_alphanumeric() && ch != '_' && ch != '-' && ch != '.' {
                return Err(NamespaceStorageError::InvalidName(format!(
                    "invalid character '{}' in namespace name",
                    ch
                )));
            }
        }
        
        // Reserved names
        let reserved = ["_global", "_namespaces", "_meta", "_system"];
        if reserved.contains(&name) {
            return Err(NamespaceStorageError::InvalidName(format!(
                "'{}' is a reserved name",
                name
            )));
        }
        
        Ok(())
    }
    
    // ========================================================================
    // Collection Operations
    // ========================================================================
    
    /// Create a collection in a namespace
    pub fn create_collection(
        &self,
        namespace: &str,
        config: CollectionConfig,
    ) -> Result<Arc<CollectionHandle>> {
        let ns_handle = self.get_namespace(namespace)?;
        
        // Check if namespace is read-only
        if ns_handle.meta.read().read_only {
            return Err(NamespaceStorageError::ReadOnly(namespace.to_string()));
        }
        
        let collection_name = config.name.clone();
        let collection_dir = ns_handle.collection_path(&collection_name);
        
        // Check if already exists
        if collection_dir.exists() {
            return Err(NamespaceStorageError::CollectionAlreadyExists {
                namespace: namespace.to_string(),
                collection: collection_name,
            });
        }
        
        // Create directory
        fs::create_dir_all(&collection_dir)?;
        
        // Freeze config and write
        let frozen_config = CollectionConfig {
            frozen: true,
            ..config
        };
        
        let config_path = collection_dir.join("config.json");
        let file = File::create(&config_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &frozen_config)
            .map_err(|e| NamespaceStorageError::IoError(e.to_string()))?;
        
        // Update namespace metadata
        {
            let mut meta = ns_handle.meta.write();
            meta.collections.push(collection_name.clone());
            meta.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
        }
        
        // Write updated namespace meta
        let ns_meta_path = ns_handle.root.join("_meta.json");
        let file = File::create(&ns_meta_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &*ns_handle.meta.read())
            .map_err(|e| NamespaceStorageError::IoError(e.to_string()))?;
        
        // Create and cache handle
        let handle = Arc::new(CollectionHandle {
            namespace: namespace.to_string(),
            name: collection_name.clone(),
            root: collection_dir,
            config: Arc::new(frozen_config),
        });
        
        self.collections
            .write()
            .insert((namespace.to_string(), collection_name), handle.clone());
        
        Ok(handle)
    }
    
    /// Get a collection handle
    pub fn get_collection(
        &self,
        namespace: &str,
        collection: &str,
    ) -> Result<Arc<CollectionHandle>> {
        let key = (namespace.to_string(), collection.to_string());
        
        // Check cache
        if let Some(handle) = self.collections.read().get(&key) {
            return Ok(handle.clone());
        }
        
        let ns_handle = self.get_namespace(namespace)?;
        let collection_dir = ns_handle.collection_path(collection);
        
        if !collection_dir.exists() {
            return Err(NamespaceStorageError::CollectionNotFound {
                namespace: namespace.to_string(),
                collection: collection.to_string(),
            });
        }
        
        // Load config
        let config_path = collection_dir.join("config.json");
        let config: CollectionConfig = if config_path.exists() {
            let file = File::open(&config_path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)
                .map_err(|e| NamespaceStorageError::RegistryCorrupted(e.to_string()))?
        } else {
            // Default config for legacy collections
            CollectionConfig {
                name: collection.to_string(),
                dimension: None,
                metric: DistanceMetric::Cosine,
                index_config: IndexConfig::default(),
                created_at: 0,
                frozen: true,
            }
        };
        
        let handle = Arc::new(CollectionHandle {
            namespace: namespace.to_string(),
            name: collection.to_string(),
            root: collection_dir,
            config: Arc::new(config),
        });
        
        self.collections.write().insert(key, handle.clone());
        
        Ok(handle)
    }
    
    /// Delete a collection
    pub fn delete_collection(&self, namespace: &str, collection: &str) -> Result<()> {
        let ns_handle = self.get_namespace(namespace)?;
        
        // Check if namespace is read-only
        if ns_handle.meta.read().read_only {
            return Err(NamespaceStorageError::ReadOnly(namespace.to_string()));
        }
        
        let collection_dir = ns_handle.collection_path(collection);
        if !collection_dir.exists() {
            return Err(NamespaceStorageError::CollectionNotFound {
                namespace: namespace.to_string(),
                collection: collection.to_string(),
            });
        }
        
        // Remove from cache
        self.collections
            .write()
            .remove(&(namespace.to_string(), collection.to_string()));
        
        // Update namespace metadata
        {
            let mut meta = ns_handle.meta.write();
            meta.collections.retain(|c| c != collection);
            meta.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
        }
        
        // Write updated namespace meta
        let ns_meta_path = ns_handle.root.join("_meta.json");
        let file = File::create(&ns_meta_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &*ns_handle.meta.read())
            .map_err(|e| NamespaceStorageError::IoError(e.to_string()))?;
        
        // Remove directory (optional - could mark as deleted instead)
        fs::remove_dir_all(&collection_dir)?;
        
        Ok(())
    }
    
    /// Resolve (namespace, collection) to storage paths
    ///
    /// This is the main routing method, O(1) via hash-map lookup.
    pub fn resolve(
        &self,
        namespace: &str,
        collection: &str,
    ) -> Result<Arc<CollectionHandle>> {
        self.get_collection(namespace, collection)
    }
}

// ============================================================================
// Prefix Iteration Safety
// ============================================================================

/// Safe prefix iterator that guarantees all keys match the prefix
///
/// This is the correct primitive for namespace-scoped iteration.
/// Unlike range scans, this cannot accidentally include keys from other namespaces.
pub struct PrefixIterator<I> {
    inner: I,
    prefix: Vec<u8>,
    exhausted: bool,
}

impl<I> PrefixIterator<I> {
    /// Create a new prefix iterator
    pub fn new(inner: I, prefix: Vec<u8>) -> Self {
        Self {
            inner,
            prefix,
            exhausted: false,
        }
    }
    
    /// Get the prefix
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }
}

impl<I, K, V> Iterator for PrefixIterator<I>
where
    I: Iterator<Item = (K, V)>,
    K: AsRef<[u8]>,
{
    type Item = (K, V);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        
        match self.inner.next() {
            Some((key, value)) => {
                if key.as_ref().starts_with(&self.prefix) {
                    Some((key, value))
                } else {
                    // Key doesn't match prefix - we're done
                    // This is the safety guarantee: we stop at the first non-matching key
                    self.exhausted = true;
                    None
                }
            }
            None => {
                self.exhausted = true;
                None
            }
        }
    }
}

/// Compute the exclusive end key for a prefix scan
///
/// Given a prefix, returns the smallest key that is greater than all keys
/// starting with that prefix. This enables efficient range scans.
///
/// # Example
/// ```
/// use sochdb_storage::namespace::next_prefix;
/// assert_eq!(next_prefix(b"abc"), Some(b"abd".to_vec()));
/// assert_eq!(next_prefix(b"\xff\xff"), None); // No successor
/// ```
pub fn next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() {
        return None;
    }
    
    let mut result = prefix.to_vec();
    
    // Find rightmost byte that isn't 0xFF
    while let Some(&last) = result.last() {
        if last == 0xFF {
            result.pop();
        } else {
            // Increment this byte
            *result.last_mut().unwrap() += 1;
            return Some(result);
        }
    }
    
    // All bytes were 0xFF - no successor exists
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    fn setup_router() -> (TempDir, NamespaceRouter) {
        let temp_dir = TempDir::new().unwrap();
        let router = NamespaceRouter::new(temp_dir.path()).unwrap();
        (temp_dir, router)
    }
    
    #[test]
    fn test_create_namespace() {
        let (_temp, router) = setup_router();
        
        let meta = NamespaceMeta::new("tenant_a")
            .with_display_name("Tenant A")
            .with_labels([("env".to_string(), "production".to_string())].into());
        
        let handle = router.create_namespace(meta).unwrap();
        assert_eq!(handle.name, "tenant_a");
        assert!(handle.root.exists());
        assert!(handle.collections_dir().exists());
    }
    
    #[test]
    fn test_namespace_already_exists() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        
        let result = router.create_namespace(NamespaceMeta::new("tenant_a"));
        assert!(matches!(result, Err(NamespaceStorageError::AlreadyExists(_))));
    }
    
    #[test]
    fn test_get_namespace() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        
        let handle = router.get_namespace("tenant_a").unwrap();
        assert_eq!(handle.name, "tenant_a");
    }
    
    #[test]
    fn test_namespace_not_found() {
        let (_temp, router) = setup_router();
        
        let result = router.get_namespace("nonexistent");
        assert!(matches!(result, Err(NamespaceStorageError::NotFound(_))));
    }
    
    #[test]
    fn test_list_namespaces() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        router.create_namespace(NamespaceMeta::new("tenant_b")).unwrap();
        
        let mut namespaces = router.list_namespaces();
        namespaces.sort();
        
        assert_eq!(namespaces, vec!["tenant_a", "tenant_b"]);
    }
    
    #[test]
    fn test_delete_namespace() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        router.delete_namespace("tenant_a").unwrap();
        
        let namespaces = router.list_namespaces();
        assert!(!namespaces.contains(&"tenant_a".to_string()));
    }
    
    #[test]
    fn test_create_collection() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        
        let config = CollectionConfig {
            name: "documents".to_string(),
            dimension: Some(384),
            metric: DistanceMetric::Cosine,
            index_config: IndexConfig::default(),
            created_at: 0,
            frozen: false,
        };
        
        let handle = router.create_collection("tenant_a", config).unwrap();
        assert_eq!(handle.name, "documents");
        assert!(handle.root.exists());
        assert!(handle.config.frozen); // Should be frozen after creation
    }
    
    #[test]
    fn test_get_collection() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        
        let config = CollectionConfig {
            name: "documents".to_string(),
            dimension: Some(384),
            metric: DistanceMetric::Cosine,
            index_config: IndexConfig::default(),
            created_at: 0,
            frozen: false,
        };
        
        router.create_collection("tenant_a", config).unwrap();
        
        let handle = router.get_collection("tenant_a", "documents").unwrap();
        assert_eq!(handle.name, "documents");
        assert_eq!(handle.namespace, "tenant_a");
    }
    
    #[test]
    fn test_resolve() {
        let (_temp, router) = setup_router();
        
        router.create_namespace(NamespaceMeta::new("tenant_a")).unwrap();
        
        let config = CollectionConfig {
            name: "documents".to_string(),
            dimension: Some(384),
            metric: DistanceMetric::Cosine,
            index_config: IndexConfig::default(),
            created_at: 0,
            frozen: false,
        };
        
        router.create_collection("tenant_a", config).unwrap();
        
        // O(1) resolution
        let handle = router.resolve("tenant_a", "documents").unwrap();
        assert_eq!(handle.vectors_path(), handle.root.join("vectors.idx"));
    }
    
    #[test]
    fn test_invalid_namespace_names() {
        let (_temp, router) = setup_router();
        
        // Empty name
        assert!(router.create_namespace(NamespaceMeta::new("")).is_err());
        
        // Starts with hyphen
        assert!(router.create_namespace(NamespaceMeta::new("-bad")).is_err());
        
        // Contains space
        assert!(router.create_namespace(NamespaceMeta::new("bad name")).is_err());
        
        // Reserved name
        assert!(router.create_namespace(NamespaceMeta::new("_global")).is_err());
    }
    
    #[test]
    fn test_next_prefix() {
        assert_eq!(next_prefix(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(next_prefix(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(next_prefix(b"\xff\xff\xff"), None);
        assert_eq!(next_prefix(b""), None);
        assert_eq!(next_prefix(b"tenant_a/"), Some(b"tenant_a0".to_vec()));
    }
    
    #[test]
    fn test_prefix_iterator() {
        let data = vec![
            (b"tenant_a/doc1".to_vec(), b"v1".to_vec()),
            (b"tenant_a/doc2".to_vec(), b"v2".to_vec()),
            (b"tenant_b/doc1".to_vec(), b"v3".to_vec()), // Should not be returned
        ];
        
        let iter = PrefixIterator::new(data.into_iter(), b"tenant_a/".to_vec());
        let results: Vec<_> = iter.collect();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].0.starts_with(b"tenant_a/"));
        assert!(results[1].0.starts_with(b"tenant_a/"));
    }
}
