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

//! Trie-Columnar Hybrid (TCH) - Novel Data Structure for TOON
//!
//! Combines two access patterns:
//! 1. **Path access**: `user.profile.settings.theme` → Trie traversal O(|path|)
//! 2. **Tabular scan**: `users[N]{fields}` → Columnar sequential I/O
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Trie-Columnar Hybrid (TCH)                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   Path Trie (Radix Compressed)          Column Store             │
//! │   ─────────────────────────             ────────────             │
//! │                                                                  │
//! │        [root]                           Field: "id"              │
//! │        /    \                           ┌─────────────┐          │
//! │    [users] [config]                     │ 1,2,3,4,5...│ (packed) │
//! │      │       │                          └─────────────┘          │
//! │   [_array_] [theme]                                              │
//! │      │       │                          Field: "name"            │
//! │   ┌──┴──┐   "dark"                      ┌─────────────┐          │
//! │ [id][name]                              │Alice,Bob,...│          │
//! │   │    │                                └─────────────┘          │
//! │   ▼    ▼                                                         │
//! │ ColRef ColRef ───────────────────────►  Field: "score"          │
//! │                                         ┌─────────────┐          │
//! │                                         │95.5,87.2,...│          │
//! │                                         └─────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Complexity
//!
//! - Path lookup: O(|path|) - length of path string, NOT data size
//! - Column scan: O(N/B) - sequential I/O
//! - Point query: O(|path| + log N) with optional index

use moka::sync::Cache;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a column
pub type ColumnId = u32;

/// Reference to a column in the column store
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    /// Column identifier
    pub id: ColumnId,
    /// Offset within the column (for nested arrays)
    pub offset: u32,
    /// Length of the column (number of values)
    pub len: u32,
}

/// Type of a trie node
#[derive(Debug, Clone, PartialEq)]
pub enum TrieNodeType {
    /// Root node
    Root,
    /// Object with named children
    Object,
    /// Array with indexed elements
    Array {
        /// Schema defining the array element type
        schema: Arc<ArraySchema>,
        /// References to columns for each field
        column_refs: Vec<ColumnRef>,
    },
    /// Leaf value (scalar or direct reference)
    Value {
        /// Column containing this value
        column_ref: ColumnRef,
    },
}

/// Schema for array elements in TOON
#[derive(Debug, Clone, PartialEq)]
pub struct ArraySchema {
    /// Field names in order
    pub fields: Vec<String>,
    /// Field types
    pub types: Vec<FieldType>,
    /// Row count
    pub row_count: u32,
}

/// Type of a field in the schema
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Bool,
    Int64,
    UInt64,
    Float64,
    Text,
    Binary,
    Ref,
}

impl FieldType {
    /// Size in bytes (fixed-size types) or 0 for variable
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            FieldType::Bool => Some(1),
            FieldType::Int64 | FieldType::UInt64 | FieldType::Float64 => Some(8),
            FieldType::Ref => Some(8),
            FieldType::Text | FieldType::Binary => None,
        }
    }
}

/// A node in the path trie (radix compressed)
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Compressed label for this edge (radix compression)
    pub label: String,
    /// Children indexed by first character
    pub children: HashMap<char, Box<TrieNode>>,
    /// Node type and associated data
    pub node_type: TrieNodeType,
}

impl TrieNode {
    /// Create a new root node
    pub fn root() -> Self {
        Self {
            label: String::new(),
            children: HashMap::new(),
            node_type: TrieNodeType::Root,
        }
    }

    /// Create an object node
    pub fn object(label: &str) -> Self {
        Self {
            label: label.to_string(),
            children: HashMap::new(),
            node_type: TrieNodeType::Object,
        }
    }

    /// Create an array node
    pub fn array(label: &str, schema: ArraySchema, column_refs: Vec<ColumnRef>) -> Self {
        Self {
            label: label.to_string(),
            children: HashMap::new(),
            node_type: TrieNodeType::Array {
                schema: Arc::new(schema),
                column_refs,
            },
        }
    }

    /// Create a value node
    pub fn value(label: &str, column_ref: ColumnRef) -> Self {
        Self {
            label: label.to_string(),
            children: HashMap::new(),
            node_type: TrieNodeType::Value { column_ref },
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, child: TrieNode) {
        if let Some(first_char) = child.label.chars().next() {
            self.children.insert(first_char, Box::new(child));
        }
    }

    /// Find child by prefix
    pub fn find_child(&self, prefix: &str) -> Option<&TrieNode> {
        let first_char = prefix.chars().next()?;
        let child = self.children.get(&first_char)?;

        // Check if the child's label matches the prefix
        if prefix.starts_with(&child.label) || child.label.starts_with(prefix) {
            Some(child)
        } else {
            None
        }
    }

    /// Find child by prefix (mutable)
    pub fn find_child_mut(&mut self, prefix: &str) -> Option<&mut TrieNode> {
        let first_char = prefix.chars().next()?;
        let child = self.children.get_mut(&first_char)?;

        if prefix.starts_with(&child.label) || child.label.starts_with(prefix) {
            Some(child)
        } else {
            None
        }
    }
}

/// Result of a path resolution
#[derive(Debug, Clone)]
pub enum PathResolution {
    /// Found an array - returns schema and column references
    Array {
        schema: Arc<ArraySchema>,
        columns: Vec<ColumnRef>,
    },
    /// Found a scalar value - returns column reference
    Value(ColumnRef),
    /// Path not found
    NotFound,
    /// Partial match (path is a prefix)
    Partial { matched: String, remaining: String },
}

/// Column store for storing column data
#[derive(Debug)]
pub struct ColumnStore {
    /// Next available column ID
    next_id: ColumnId,
    /// Column data by ID
    columns: HashMap<ColumnId, ColumnData>,
}

use memmap2::Mmap;

/// Storage backend for column data
#[derive(Debug)]
pub enum DataStorage {
    /// In-memory vector (mutable)
    InMemory(Vec<u8>),
    /// Memory-mapped file (read-only)
    Mmap(Arc<Mmap>),
    /// Slice of a memory-mapped file
    MmapSlice {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl DataStorage {
    /// Get data as slice
    pub fn as_slice(&self) -> &[u8] {
        match self {
            DataStorage::InMemory(vec) => vec,
            DataStorage::Mmap(mmap) => mmap,
            DataStorage::MmapSlice { mmap, offset, len } => &mmap[*offset..*offset + *len],
        }
    }

    /// Get mutable reference to in-memory data
    pub fn as_mut_vec(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            DataStorage::InMemory(vec) => Some(vec),
            _ => None,
        }
    }
}

impl Clone for DataStorage {
    fn clone(&self) -> Self {
        match self {
            DataStorage::InMemory(vec) => DataStorage::InMemory(vec.clone()),
            DataStorage::Mmap(mmap) => DataStorage::Mmap(mmap.clone()),
            DataStorage::MmapSlice { mmap, offset, len } => DataStorage::MmapSlice {
                mmap: mmap.clone(),
                offset: *offset,
                len: *len,
            },
        }
    }
}

/// Data for a single column
#[derive(Debug, Clone)]
pub struct ColumnData {
    /// Column ID
    pub id: ColumnId,
    /// Field type
    pub field_type: FieldType,
    /// Raw data (format depends on type)
    pub data: DataStorage,
    /// Null bitmap (1 bit per value, 1 = non-null)
    pub null_bitmap: Option<Vec<u8>>,
    /// Offsets for variable-length types (Text, Binary)
    pub offsets: Option<Vec<u32>>,
    /// Number of values
    pub count: u32,
}

impl ColumnStore {
    /// Create a new empty column store
    pub fn new() -> Self {
        Self {
            next_id: 0,
            columns: HashMap::new(),
        }
    }

    /// Allocate a new column
    pub fn allocate_column(&mut self, field_type: FieldType) -> ColumnId {
        let id = self.next_id;
        self.next_id += 1;
        self.columns.insert(
            id,
            ColumnData {
                id,
                field_type,
                data: DataStorage::InMemory(Vec::new()),
                null_bitmap: None,
                offsets: if field_type.fixed_size().is_none() {
                    Some(vec![0]) // Initial offset for variable-length
                } else {
                    None
                },
                count: 0,
            },
        );
        id
    }

    /// Get a column by ID
    pub fn get_column(&self, id: ColumnId) -> Option<&ColumnData> {
        self.columns.get(&id)
    }

    /// Get a column by ID (mutable)
    pub fn get_column_mut(&mut self, id: ColumnId) -> Option<&mut ColumnData> {
        self.columns.get_mut(&id)
    }

    /// Append an i64 value to a column
    pub fn append_i64(&mut self, col_id: ColumnId, value: i64) -> bool {
        if let Some(col) = self.columns.get_mut(&col_id) {
            if col.field_type != FieldType::Int64 {
                return false;
            }
            if let Some(vec) = col.data.as_mut_vec() {
                vec.extend_from_slice(&value.to_le_bytes());
                col.count += 1;
                return true;
            }
        }
        false
    }

    /// Append a u64 value to a column
    pub fn append_u64(&mut self, col_id: ColumnId, value: u64) -> bool {
        if let Some(col) = self.columns.get_mut(&col_id) {
            if col.field_type != FieldType::UInt64 {
                return false;
            }
            if let Some(vec) = col.data.as_mut_vec() {
                vec.extend_from_slice(&value.to_le_bytes());
                col.count += 1;
                return true;
            }
        }
        false
    }

    /// Append a f64 value to a column
    pub fn append_f64(&mut self, col_id: ColumnId, value: f64) -> bool {
        if let Some(col) = self.columns.get_mut(&col_id) {
            if col.field_type != FieldType::Float64 {
                return false;
            }
            if let Some(vec) = col.data.as_mut_vec() {
                vec.extend_from_slice(&value.to_le_bytes());
                col.count += 1;
                return true;
            }
        }
        false
    }

    /// Append a text value to a column
    pub fn append_text(&mut self, col_id: ColumnId, value: &str) -> bool {
        if let Some(col) = self.columns.get_mut(&col_id) {
            if col.field_type != FieldType::Text {
                return false;
            }
            if let Some(vec) = col.data.as_mut_vec() {
                vec.extend_from_slice(value.as_bytes());
                if let Some(offsets) = &mut col.offsets {
                    offsets.push(vec.len() as u32);
                }
                col.count += 1;
                return true;
            }
        }
        false
    }

    /// Read an i64 from a column at index
    pub fn read_i64(&self, col_id: ColumnId, index: usize) -> Option<i64> {
        let col = self.columns.get(&col_id)?;
        if col.field_type != FieldType::Int64 || index >= col.count as usize {
            return None;
        }
        let offset = index * 8;
        let data = col.data.as_slice();
        let bytes: [u8; 8] = data[offset..offset + 8].try_into().ok()?;
        Some(i64::from_le_bytes(bytes))
    }

    /// Read a u64 from a column at index
    pub fn read_u64(&self, col_id: ColumnId, index: usize) -> Option<u64> {
        let col = self.columns.get(&col_id)?;
        if col.field_type != FieldType::UInt64 || index >= col.count as usize {
            return None;
        }
        let offset = index * 8;
        let data = col.data.as_slice();
        let bytes: [u8; 8] = data[offset..offset + 8].try_into().ok()?;
        Some(u64::from_le_bytes(bytes))
    }

    /// Read a f64 from a column at index
    pub fn read_f64(&self, col_id: ColumnId, index: usize) -> Option<f64> {
        let col = self.columns.get(&col_id)?;
        if col.field_type != FieldType::Float64 || index >= col.count as usize {
            return None;
        }
        let offset = index * 8;
        let data = col.data.as_slice();
        let bytes: [u8; 8] = data[offset..offset + 8].try_into().ok()?;
        Some(f64::from_le_bytes(bytes))
    }

    /// Read text from a column at index
    pub fn read_text(&self, col_id: ColumnId, index: usize) -> Option<String> {
        let col = self.columns.get(&col_id)?;
        if col.field_type != FieldType::Text || index >= col.count as usize {
            return None;
        }
        let offsets = col.offsets.as_ref()?;
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        let data = col.data.as_slice();
        String::from_utf8(data[start..end].to_vec()).ok()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.columns
            .values()
            .map(|c| {
                c.data.as_slice().len()
                    + c.null_bitmap.as_ref().map(|b| b.len()).unwrap_or(0)
                    + c.offsets.as_ref().map(|o| o.len() * 4).unwrap_or(0)
            })
            .sum()
    }
}

impl Default for ColumnStore {
    fn default() -> Self {
        Self::new()
    }
}

/// The main Trie-Columnar Hybrid structure
#[derive(Debug)]
pub struct TrieColumnarHybrid {
    /// Path trie (radix compressed)
    trie: TrieNode,
    /// Column store
    columns: ColumnStore,
}

impl TrieColumnarHybrid {
    /// Create a new empty TCH
    pub fn new() -> Self {
        Self {
            trie: TrieNode::root(),
            columns: ColumnStore::new(),
        }
    }

    /// Resolve a path to columns or values
    ///
    /// Path format: "table.field" or "table" for array access
    /// O(|path|) complexity - depends on path length, not data size
    pub fn resolve(&self, path: &str) -> PathResolution {
        let parts: Vec<&str> = path.split('.').collect();
        self.resolve_parts(&parts)
    }

    /// Resolve path parts recursively
    fn resolve_parts(&self, parts: &[&str]) -> PathResolution {
        if parts.is_empty() {
            return PathResolution::NotFound;
        }

        let mut current = &self.trie;
        let mut matched = String::new();

        for (i, part) in parts.iter().enumerate() {
            match current.find_child(part) {
                Some(child) => {
                    if !matched.is_empty() {
                        matched.push('.');
                    }
                    matched.push_str(&child.label);
                    current = child;
                }
                None => {
                    // If we haven't matched anything, it's NotFound
                    if matched.is_empty() {
                        return PathResolution::NotFound;
                    }
                    return PathResolution::Partial {
                        matched,
                        remaining: parts[i..].join("."),
                    };
                }
            }
        }

        // Return based on node type
        match &current.node_type {
            TrieNodeType::Array {
                schema,
                column_refs,
            } => PathResolution::Array {
                schema: schema.clone(),
                columns: column_refs.clone(),
            },
            TrieNodeType::Value { column_ref } => PathResolution::Value(*column_ref),
            _ => PathResolution::NotFound,
        }
    }

    /// Get the column store
    pub fn column_store(&self) -> &ColumnStore {
        &self.columns
    }

    /// Get the column store (mutable)
    pub fn column_store_mut(&mut self) -> &mut ColumnStore {
        &mut self.columns
    }

    /// Get the trie root
    pub fn trie(&self) -> &TrieNode {
        &self.trie
    }

    /// Get the trie root (mutable)
    pub fn trie_mut(&mut self) -> &mut TrieNode {
        &mut self.trie
    }

    /// Register a table with schema
    pub fn register_table(&mut self, name: &str, fields: &[(String, FieldType)]) -> Vec<ColumnRef> {
        // Allocate columns for each field
        let mut column_refs = Vec::with_capacity(fields.len());
        for (_, field_type) in fields {
            let col_id = self.columns.allocate_column(*field_type);
            column_refs.push(ColumnRef {
                id: col_id,
                offset: 0,
                len: 0,
            });
        }

        // Create array schema
        let schema = ArraySchema {
            fields: fields.iter().map(|(n, _)| n.clone()).collect(),
            types: fields.iter().map(|(_, t)| *t).collect(),
            row_count: 0,
        };

        // Add to trie
        let table_node = TrieNode::array(name, schema, column_refs.clone());
        self.trie.add_child(table_node);

        column_refs
    }

    /// Memory statistics
    pub fn memory_stats(&self) -> TchStats {
        TchStats {
            column_bytes: self.columns.memory_bytes(),
            num_columns: self.columns.columns.len(),
            num_trie_nodes: self.count_trie_nodes(&self.trie),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn count_trie_nodes(&self, node: &TrieNode) -> usize {
        1 + node
            .children
            .values()
            .map(|c| self.count_trie_nodes(c))
            .sum::<usize>()
    }
}

impl Default for TrieColumnarHybrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for TCH
#[derive(Debug, Clone)]
pub struct TchStats {
    /// Bytes used by column data
    pub column_bytes: usize,
    /// Number of columns
    pub num_columns: usize,
    /// Number of trie nodes
    pub num_trie_nodes: usize,
}

/// Cached path resolution result (for LRU cache storage)
#[derive(Clone, Debug)]
pub enum CachedResolution {
    /// Found an array - returns schema and column references
    Array {
        schema: Arc<ArraySchema>,
        columns: Vec<ColumnRef>,
    },
    /// Found a scalar value - returns column reference
    Value(ColumnRef),
    /// Path not found
    NotFound,
    /// Partial match
    Partial { matched: String, remaining: String },
}

impl From<PathResolution> for CachedResolution {
    fn from(res: PathResolution) -> Self {
        match res {
            PathResolution::Array { schema, columns } => {
                CachedResolution::Array { schema, columns }
            }
            PathResolution::Value(col) => CachedResolution::Value(col),
            PathResolution::NotFound => CachedResolution::NotFound,
            PathResolution::Partial { matched, remaining } => {
                CachedResolution::Partial { matched, remaining }
            }
        }
    }
}

impl From<CachedResolution> for PathResolution {
    fn from(res: CachedResolution) -> Self {
        match res {
            CachedResolution::Array { schema, columns } => {
                PathResolution::Array { schema, columns }
            }
            CachedResolution::Value(col) => PathResolution::Value(col),
            CachedResolution::NotFound => PathResolution::NotFound,
            CachedResolution::Partial { matched, remaining } => {
                PathResolution::Partial { matched, remaining }
            }
        }
    }
}

/// Cache statistics for monitoring hit rates
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: AtomicU64,
    /// Number of cache misses
    pub misses: AtomicU64,
}

impl CacheStats {
    /// Get hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

/// Trie-Columnar Hybrid with LRU caching for path resolution
///
/// Wraps `TrieColumnarHybrid` with a thread-safe LRU cache using Moka.
/// Useful for workloads with repeated path lookups (e.g., query execution).
pub struct CachedTrieColumnarHybrid {
    /// Inner TCH (immutable for reads)
    inner: TrieColumnarHybrid,
    /// LRU cache for path resolutions
    cache: Cache<String, CachedResolution>,
    /// Cache statistics
    stats: Arc<CacheStats>,
}

impl CachedTrieColumnarHybrid {
    /// Create a new cached TCH with default cache size (10,000 entries)
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create a cached TCH with specified cache capacity
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            inner: TrieColumnarHybrid::new(),
            cache: Cache::new(capacity),
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Wrap an existing TCH with caching
    pub fn from_tch(tch: TrieColumnarHybrid) -> Self {
        Self::from_tch_with_capacity(tch, 10_000)
    }

    /// Wrap an existing TCH with caching and specified capacity
    pub fn from_tch_with_capacity(tch: TrieColumnarHybrid, capacity: u64) -> Self {
        Self {
            inner: tch,
            cache: Cache::new(capacity),
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Resolve a path with caching
    ///
    /// Returns cached result if available, otherwise performs trie traversal
    /// and caches the result.
    pub fn resolve(&self, path: &str) -> PathResolution {
        // Check cache first
        if let Some(cached) = self.cache.get(path) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return cached.into();
        }

        // Cache miss - perform actual resolution
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let result = self.inner.resolve(path);

        // Cache the result
        self.cache.insert(path.to_string(), result.clone().into());

        result
    }

    /// Invalidate cache entry for a path
    ///
    /// Call this after modifying the trie structure (e.g., registering tables)
    pub fn invalidate(&self, path: &str) {
        self.cache.invalidate(path);
    }

    /// Invalidate all cache entries
    pub fn invalidate_all(&self) {
        self.cache.invalidate_all();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        self.stats.hit_rate()
    }

    /// Get inner TCH reference (for read-only access)
    pub fn inner(&self) -> &TrieColumnarHybrid {
        &self.inner
    }

    /// Get mutable inner TCH reference
    ///
    /// Note: Mutations may invalidate cached entries. Call `invalidate_all()`
    /// after modifying the trie structure.
    pub fn inner_mut(&mut self) -> &mut TrieColumnarHybrid {
        &mut self.inner
    }

    /// Register a table (invalidates cache)
    pub fn register_table(&mut self, name: &str, fields: &[(String, FieldType)]) -> Vec<ColumnRef> {
        let refs = self.inner.register_table(name, fields);
        // Invalidate all cached paths since new table might affect resolution
        self.cache.invalidate_all();
        refs
    }

    /// Get column store reference
    pub fn column_store(&self) -> &ColumnStore {
        self.inner.column_store()
    }

    /// Get mutable column store reference
    pub fn column_store_mut(&mut self) -> &mut ColumnStore {
        self.inner.column_store_mut()
    }

    /// Memory statistics including cache
    pub fn memory_stats(&self) -> CachedTchStats {
        let inner_stats = self.inner.memory_stats();
        CachedTchStats {
            column_bytes: inner_stats.column_bytes,
            num_columns: inner_stats.num_columns,
            num_trie_nodes: inner_stats.num_trie_nodes,
            cache_entries: self.cache.entry_count(),
            cache_hit_rate: self.hit_rate(),
        }
    }
}

impl Default for CachedTrieColumnarHybrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended statistics for cached TCH
#[derive(Debug, Clone)]
pub struct CachedTchStats {
    /// Bytes used by column data
    pub column_bytes: usize,
    /// Number of columns
    pub num_columns: usize,
    /// Number of trie nodes
    pub num_trie_nodes: usize,
    /// Number of cached path resolutions
    pub cache_entries: u64,
    /// Cache hit rate percentage
    pub cache_hit_rate: f64,
}

use sochdb_core::soch::ColumnAccess;

/// Accessor for TCH columns implementing ColumnAccess trait
pub struct TchColumnAccess<'a> {
    store: &'a ColumnStore,
    columns: &'a [ColumnRef],
    schema: &'a ArraySchema,
}

impl<'a> TchColumnAccess<'a> {
    pub fn new(store: &'a ColumnStore, columns: &'a [ColumnRef], schema: &'a ArraySchema) -> Self {
        Self {
            store,
            columns,
            schema,
        }
    }
}

impl<'a> ColumnAccess for TchColumnAccess<'a> {
    fn row_count(&self) -> usize {
        self.schema.row_count as usize
    }

    fn col_count(&self) -> usize {
        self.columns.len()
    }

    fn field_names(&self) -> Vec<&str> {
        self.schema.fields.iter().map(|s| s.as_str()).collect()
    }

    fn write_value(
        &self,
        col_idx: usize,
        row_idx: usize,
        f: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result {
        if col_idx >= self.columns.len() {
            return Err(std::fmt::Error);
        }
        let col_ref = &self.columns[col_idx];
        self.store.write_value(col_ref.id, row_idx, f)
    }
}

impl ColumnStore {
    /// Write value at index to formatter
    pub fn write_value(
        &self,
        col_id: ColumnId,
        index: usize,
        f: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result {
        let col = self.columns.get(&col_id).ok_or(std::fmt::Error)?;
        if index >= col.count as usize {
            return Err(std::fmt::Error);
        }

        match col.field_type {
            FieldType::Int64 => {
                let offset = index * 8;
                let data = col.data.as_slice();
                if offset + 8 > data.len() {
                    return Err(std::fmt::Error);
                }
                let bytes: [u8; 8] = data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| std::fmt::Error)?;
                write!(f, "{}", i64::from_le_bytes(bytes))
            }
            FieldType::UInt64 => {
                let offset = index * 8;
                let data = col.data.as_slice();
                if offset + 8 > data.len() {
                    return Err(std::fmt::Error);
                }
                let bytes: [u8; 8] = data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| std::fmt::Error)?;
                write!(f, "{}", u64::from_le_bytes(bytes))
            }
            FieldType::Float64 => {
                let offset = index * 8;
                let data = col.data.as_slice();
                if offset + 8 > data.len() {
                    return Err(std::fmt::Error);
                }
                let bytes: [u8; 8] = data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| std::fmt::Error)?;
                write!(f, "{}", f64::from_le_bytes(bytes))
            }
            FieldType::Text => {
                let offsets = col.offsets.as_ref().ok_or(std::fmt::Error)?;
                if index + 1 >= offsets.len() {
                    return Err(std::fmt::Error);
                }
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                let data = col.data.as_slice();
                if end > data.len() {
                    return Err(std::fmt::Error);
                }
                let s = std::str::from_utf8(&data[start..end]).map_err(|_| std::fmt::Error)?;

                // Escape if needed
                if s.contains(',') || s.contains('\n') || s.contains('"') {
                    write!(f, "\"{}\"", s.replace('"', "\"\""))
                } else {
                    write!(f, "{}", s)
                }
            }
            FieldType::Bool => {
                let byte_idx = index / 8;
                let bit_idx = index % 8;
                let data = col.data.as_slice();
                if byte_idx >= data.len() {
                    return Err(std::fmt::Error);
                }
                let val = (data[byte_idx] >> bit_idx) & 1 != 0;
                write!(f, "{}", val)
            }
            FieldType::Binary => {
                // Binary is stored as length-prefixed bytes
                let data = col.data.as_slice();
                if data.is_empty() {
                    write!(f, "b64:<empty>")
                } else {
                    write!(f, "b64:<{}bytes>", data.len())
                }
            }
            FieldType::Ref => {
                // Reference is stored as (table_hash: u32, id: u64)
                let data = col.data.as_slice();
                if data.len() >= 12 && index * 12 + 12 <= data.len() {
                    let offset = index * 12;
                    let id = u64::from_le_bytes(data[offset + 4..offset + 12].try_into().unwrap());
                    write!(f, "ref({})", id)
                } else {
                    write!(f, "null")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_core::soch::SochCursor;

    #[test]
    fn test_tch_cursor() {
        let mut tch = TrieColumnarHybrid::new();

        let fields = vec![
            ("id".to_string(), FieldType::UInt64),
            ("name".to_string(), FieldType::Text),
        ];

        let cols = tch.register_table("users", &fields);

        // Add data manually to columns
        let id_col = cols[0].id;
        let name_col = cols[1].id;

        tch.columns.append_u64(id_col, 1);
        tch.columns.append_text(name_col, "Alice");

        tch.columns.append_u64(id_col, 2);
        tch.columns.append_text(name_col, "Bob");

        // Update schema row count (hack for test since register_table sets it to 0 and doesn't update)
        // In real usage, we'd update the schema in the trie.
        // For this test, we construct TchColumnAccess manually with correct row count.
        let schema = ArraySchema {
            fields: vec!["id".to_string(), "name".to_string()],
            types: vec![FieldType::UInt64, FieldType::Text],
            row_count: 2,
        };

        let access = TchColumnAccess::new(&tch.columns, &cols, &schema);
        let cursor = SochCursor::new(&access, "users".to_string());

        let lines: Vec<String> = cursor.collect();
        assert_eq!(lines.len(), 3); // Header + 2 rows
        assert_eq!(lines[0], "users[2]{id,name}:");
        assert_eq!(lines[1], "1,Alice");
        assert_eq!(lines[2], "2,Bob");
    }

    #[test]
    fn test_register_table() {
        let mut tch = TrieColumnarHybrid::new();

        let fields = vec![
            ("id".to_string(), FieldType::UInt64),
            ("name".to_string(), FieldType::Text),
            ("score".to_string(), FieldType::Float64),
        ];

        let cols = tch.register_table("users", &fields);
        assert_eq!(cols.len(), 3);

        // Should be resolvable
        match tch.resolve("users") {
            PathResolution::Array { schema, columns } => {
                assert_eq!(schema.fields, vec!["id", "name", "score"]);
                assert_eq!(columns.len(), 3);
            }
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_column_store_i64() {
        let mut store = ColumnStore::new();
        let col = store.allocate_column(FieldType::Int64);

        store.append_i64(col, 42);
        store.append_i64(col, -100);
        store.append_i64(col, i64::MAX);

        assert_eq!(store.read_i64(col, 0), Some(42));
        assert_eq!(store.read_i64(col, 1), Some(-100));
        assert_eq!(store.read_i64(col, 2), Some(i64::MAX));
        assert_eq!(store.read_i64(col, 3), None); // Out of bounds
    }

    #[test]
    fn test_column_store_text() {
        let mut store = ColumnStore::new();
        let col = store.allocate_column(FieldType::Text);

        store.append_text(col, "Hello");
        store.append_text(col, "World");
        store.append_text(col, "");
        store.append_text(col, "Long string with many characters");

        assert_eq!(store.read_text(col, 0), Some("Hello".to_string()));
        assert_eq!(store.read_text(col, 1), Some("World".to_string()));
        assert_eq!(store.read_text(col, 2), Some("".to_string()));
        assert_eq!(
            store.read_text(col, 3),
            Some("Long string with many characters".to_string())
        );
    }

    #[test]
    fn test_column_store_mixed_types() {
        let mut store = ColumnStore::new();

        let id_col = store.allocate_column(FieldType::UInt64);
        let name_col = store.allocate_column(FieldType::Text);
        let score_col = store.allocate_column(FieldType::Float64);

        // Simulate a row insert
        store.append_u64(id_col, 1);
        store.append_text(name_col, "Alice");
        store.append_f64(score_col, 95.5);

        store.append_u64(id_col, 2);
        store.append_text(name_col, "Bob");
        store.append_f64(score_col, 87.2);

        // Verify
        assert_eq!(store.read_u64(id_col, 0), Some(1));
        assert_eq!(store.read_text(name_col, 0), Some("Alice".to_string()));
        assert_eq!(store.read_f64(score_col, 0), Some(95.5));

        assert_eq!(store.read_u64(id_col, 1), Some(2));
        assert_eq!(store.read_text(name_col, 1), Some("Bob".to_string()));
        assert_eq!(store.read_f64(score_col, 1), Some(87.2));
    }

    #[test]
    fn test_path_resolution() {
        let mut tch = TrieColumnarHybrid::new();

        tch.register_table(
            "users",
            &[
                ("id".to_string(), FieldType::UInt64),
                ("name".to_string(), FieldType::Text),
            ],
        );

        tch.register_table(
            "orders",
            &[
                ("id".to_string(), FieldType::UInt64),
                ("total".to_string(), FieldType::Float64),
            ],
        );

        // Should find both tables
        assert!(matches!(tch.resolve("users"), PathResolution::Array { .. }));
        assert!(matches!(
            tch.resolve("orders"),
            PathResolution::Array { .. }
        ));

        // Non-existent paths
        assert!(matches!(
            tch.resolve("nonexistent"),
            PathResolution::NotFound | PathResolution::Partial { .. }
        ));
    }

    #[test]
    fn test_trie_node_radix() {
        let mut root = TrieNode::root();

        // Add children with common prefixes
        root.add_child(TrieNode::object("users"));
        root.add_child(TrieNode::object("orders"));
        root.add_child(TrieNode::object("config"));

        assert!(root.find_child("users").is_some());
        assert!(root.find_child("orders").is_some());
        assert!(root.find_child("config").is_some());
        assert!(root.find_child("unknown").is_none());
    }

    #[test]
    fn test_memory_stats() {
        let mut tch = TrieColumnarHybrid::new();

        tch.register_table("test", &[("data".to_string(), FieldType::Int64)]);

        let stats = tch.memory_stats();
        assert_eq!(stats.num_columns, 1);
        assert!(stats.num_trie_nodes >= 2); // root + table
    }

    #[test]
    fn test_cached_tch_basic() {
        let mut cached_tch = CachedTrieColumnarHybrid::new();

        // Register a table
        cached_tch.register_table(
            "users",
            &[
                ("id".to_string(), FieldType::UInt64),
                ("name".to_string(), FieldType::Text),
            ],
        );

        // First lookup - cache miss
        let result = cached_tch.resolve("users");
        assert!(matches!(result, PathResolution::Array { .. }));

        // Check stats - should have 1 miss
        assert_eq!(
            cached_tch
                .cache_stats()
                .misses
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            cached_tch
                .cache_stats()
                .hits
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        // Second lookup - cache hit
        let result2 = cached_tch.resolve("users");
        assert!(matches!(result2, PathResolution::Array { .. }));

        // Check stats - should have 1 hit now
        assert_eq!(
            cached_tch
                .cache_stats()
                .hits
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // Hit rate should be 50% (1 hit, 1 miss)
        assert!((cached_tch.hit_rate() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_cached_tch_invalidation() {
        let mut cached_tch = CachedTrieColumnarHybrid::new();

        cached_tch.register_table("orders", &[("id".to_string(), FieldType::UInt64)]);

        // Populate cache
        cached_tch.resolve("orders");
        assert_eq!(
            cached_tch
                .cache_stats()
                .misses
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // Invalidate
        cached_tch.invalidate("orders");

        // Should be a miss again
        cached_tch.resolve("orders");
        assert_eq!(
            cached_tch
                .cache_stats()
                .misses
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn test_cached_tch_stats() {
        let mut cached_tch = CachedTrieColumnarHybrid::with_capacity(100);

        cached_tch.register_table("test", &[("data".to_string(), FieldType::Int64)]);

        // Populate cache
        cached_tch.resolve("test");

        let stats = cached_tch.memory_stats();
        assert_eq!(stats.num_columns, 1);
        // Cache entries may not be immediately visible due to async write-back
        // Just verify hit rate is tracked correctly (1 miss = 0% hit rate)
        assert!(stats.cache_hit_rate < 1.0);
    }
}
