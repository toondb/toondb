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

//! Path Trie for TOON Document Path Resolution (Task 4)
//!
//! Implements O(|path|) path resolution for nested TOON documents,
//! regardless of schema size.
//!
//! ## Performance Analysis
//!
//! ```text
//! Traditional BTreeMap: O(d × log F) where d=depth, F=fields
//! Path Trie:           O(d) where d=path depth
//!
//! For path="users.profile.settings.theme" (d=4) with F=100 fields:
//! - BTreeMap: 4 × log(100) ≈ 26.6 comparisons
//! - PathTrie: 4 lookups = 4 hash lookups
//! ```
//!
//! ## Memory Model
//!
//! Memory: O(N × avg_path_length) where N = total columns
//! For 100 tables × 50 cols × 3 levels = 15,000 nodes × ~100 bytes = 1.5MB

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Column identifier (matches storage layer)
pub type ColumnId = u32;

/// A node in the path trie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieNode {
    /// Column ID if this path is a leaf (terminal column)
    pub column_id: Option<ColumnId>,
    /// Column type for leaf nodes
    pub column_type: Option<ColumnType>,
    /// Children nodes keyed by path segment
    pub children: HashMap<String, Box<TrieNode>>,
}

impl TrieNode {
    /// Create an empty node
    pub fn new() -> Self {
        Self {
            column_id: None,
            column_type: None,
            children: HashMap::new(),
        }
    }

    /// Create a leaf node with column information
    pub fn leaf(column_id: ColumnId, column_type: ColumnType) -> Self {
        Self {
            column_id: Some(column_id),
            column_type: Some(column_type),
            children: HashMap::new(),
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.column_id.is_some()
    }

    /// Count total nodes in subtree
    pub fn count_nodes(&self) -> usize {
        1 + self
            .children
            .values()
            .map(|c| c.count_nodes())
            .sum::<usize>()
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Column type for physical storage optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnType {
    /// Boolean (1 byte, SIMD groupable)
    Bool,
    /// Signed 64-bit integer (8 bytes, SIMD groupable)
    Int64,
    /// Unsigned 64-bit integer (8 bytes, SIMD groupable)
    UInt64,
    /// 64-bit float (8 bytes, SIMD groupable)
    Float64,
    /// Variable-length text
    Text,
    /// Variable-length binary
    Binary,
    /// Timestamp (8 bytes, SIMD groupable)
    Timestamp,
    /// Nested TOON document
    Nested,
    /// Array of values
    Array,
}

impl ColumnType {
    /// Check if type is fixed-size (for SIMD optimization)
    pub fn is_fixed_size(&self) -> bool {
        matches!(
            self,
            ColumnType::Bool
                | ColumnType::Int64
                | ColumnType::UInt64
                | ColumnType::Float64
                | ColumnType::Timestamp
        )
    }

    /// Get fixed size in bytes (None for variable-length)
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            ColumnType::Bool => Some(1),
            ColumnType::Int64
            | ColumnType::UInt64
            | ColumnType::Float64
            | ColumnType::Timestamp => Some(8),
            _ => None,
        }
    }
}

/// Path Trie for O(|path|) TOON path resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathTrie {
    /// Root node
    root: TrieNode,
    /// Total number of columns registered
    total_columns: u32,
    /// Next column ID to assign
    next_column_id: ColumnId,
}

impl PathTrie {
    /// Create a new empty trie
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            total_columns: 0,
            next_column_id: 0,
        }
    }

    /// Insert a path and get its column ID
    ///
    /// Path format: "users.profile.settings.theme"
    /// Returns the assigned column ID
    pub fn insert(&mut self, path: &str, column_type: ColumnType) -> ColumnId {
        let segments: Vec<&str> = path.split('.').collect();
        let column_id = self.next_column_id;
        self.next_column_id += 1;

        let mut current = &mut self.root;

        for (i, segment) in segments.iter().enumerate() {
            let is_last = i == segments.len() - 1;

            current = current
                .children
                .entry(segment.to_string())
                .or_insert_with(|| Box::new(TrieNode::new()));

            if is_last {
                current.column_id = Some(column_id);
                current.column_type = Some(column_type);
            }
        }

        self.total_columns += 1;
        column_id
    }

    /// Resolve a path to its column ID in O(|path|) time
    ///
    /// Returns None if path doesn't exist
    pub fn resolve(&self, path: &str) -> Option<ColumnId> {
        let segments: Vec<&str> = path.split('.').collect();
        let mut current = &self.root;

        for segment in segments {
            current = current.children.get(segment)?;
        }

        current.column_id
    }

    /// Resolve with type information
    pub fn resolve_with_type(&self, path: &str) -> Option<(ColumnId, ColumnType)> {
        let segments: Vec<&str> = path.split('.').collect();
        let mut current = &self.root;

        for segment in segments {
            current = current.children.get(segment)?;
        }

        Some((current.column_id?, current.column_type?))
    }

    /// Get all paths that start with a prefix
    ///
    /// Useful for wildcard queries like "users.profile.*"
    pub fn prefix_match(&self, prefix: &str) -> Vec<(String, ColumnId)> {
        let mut results = Vec::new();

        // Navigate to prefix node
        let segments: Vec<&str> = if prefix.is_empty() {
            vec![]
        } else {
            prefix.split('.').collect()
        };

        let mut current = &self.root;
        for segment in &segments {
            if let Some(child) = current.children.get(*segment) {
                current = child;
            } else {
                return results;
            }
        }

        // Collect all paths under this node
        self.collect_paths(current, prefix.to_string(), &mut results);
        results
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_paths(&self, node: &TrieNode, path: String, results: &mut Vec<(String, ColumnId)>) {
        if let Some(col_id) = node.column_id {
            results.push((path.clone(), col_id));
        }

        for (segment, child) in &node.children {
            let child_path = if path.is_empty() {
                segment.clone()
            } else {
                format!("{}.{}", path, segment)
            };
            self.collect_paths(child, child_path, results);
        }
    }

    /// Get total number of columns
    pub fn total_columns(&self) -> u32 {
        self.total_columns
    }

    /// Get total number of nodes (memory usage indicator)
    pub fn total_nodes(&self) -> usize {
        self.root.count_nodes()
    }

    /// Estimate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Rough estimate: ~100 bytes per node (HashMap overhead + strings)
        self.total_nodes() * 100
    }
}

impl Default for PathTrie {
    fn default() -> Self {
        Self::new()
    }
}

/// Column group affinity for compression optimization
///
/// Groups columns by:
/// 1. Type compatibility (all Int64 together for SIMD)
/// 2. Access correlation (co-accessed columns together)
/// 3. Null density (sparse columns separate)
#[derive(Debug, Clone)]
pub struct ColumnGroupAffinity {
    /// Columns grouped by type for SIMD optimization
    pub type_groups: HashMap<ColumnType, Vec<ColumnId>>,
    /// Access frequency per column (for hot/cold separation)
    pub access_frequency: HashMap<ColumnId, u64>,
    /// Null density per column (0.0 = never null, 1.0 = always null)
    pub null_density: HashMap<ColumnId, f64>,
}

impl ColumnGroupAffinity {
    /// Create from a path trie
    pub fn from_trie(trie: &PathTrie) -> Self {
        let mut type_groups: HashMap<ColumnType, Vec<ColumnId>> = HashMap::new();

        fn collect_columns(node: &TrieNode, groups: &mut HashMap<ColumnType, Vec<ColumnId>>) {
            if let (Some(col_id), Some(col_type)) = (node.column_id, node.column_type) {
                groups.entry(col_type).or_default().push(col_id);
            }
            for child in node.children.values() {
                collect_columns(child, groups);
            }
        }

        collect_columns(&trie.root, &mut type_groups);

        Self {
            type_groups,
            access_frequency: HashMap::new(),
            null_density: HashMap::new(),
        }
    }

    /// Record a column access (for access correlation)
    pub fn record_access(&mut self, column_id: ColumnId) {
        *self.access_frequency.entry(column_id).or_insert(0) += 1;
    }

    /// Update null density for a column
    pub fn update_null_density(&mut self, column_id: ColumnId, null_count: u64, total_count: u64) {
        if total_count > 0 {
            self.null_density
                .insert(column_id, null_count as f64 / total_count as f64);
        }
    }

    /// Get columns suitable for SIMD processing (fixed-size, same type)
    pub fn simd_groups(&self) -> Vec<(ColumnType, Vec<ColumnId>)> {
        self.type_groups
            .iter()
            .filter(|(t, _)| t.is_fixed_size())
            .map(|(t, cols)| (*t, cols.clone()))
            .collect()
    }

    /// Get sparse columns (null_density > threshold)
    pub fn sparse_columns(&self, threshold: f64) -> Vec<ColumnId> {
        self.null_density
            .iter()
            .filter(|(_, density)| **density > threshold)
            .map(|(col_id, _)| *col_id)
            .collect()
    }

    /// Get hot columns (top N by access frequency)
    pub fn hot_columns(&self, n: usize) -> Vec<ColumnId> {
        let mut sorted: Vec<_> = self.access_frequency.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        sorted
            .into_iter()
            .take(n)
            .map(|(col_id, _)| *col_id)
            .collect()
    }
}

// ============================================================================
// CONCURRENT PATH TRIE WITH EPOCH-BASED RECLAMATION
// ============================================================================

/// A node in the concurrent path trie using epoch-based reclamation
#[derive(Debug)]
pub struct ConcurrentTrieNode {
    /// Column ID if this path is a leaf (terminal column)
    pub column_id: Option<ColumnId>,
    /// Column type for leaf nodes
    pub column_type: Option<ColumnType>,
    /// Children nodes keyed by path segment (lock-free via DashMap)
    pub children: DashMap<String, Arc<ConcurrentTrieNode>>,
    /// Epoch when this node was created (for reclamation)
    pub created_epoch: u64,
}

impl ConcurrentTrieNode {
    /// Create an empty node
    pub fn new(epoch: u64) -> Self {
        Self {
            column_id: None,
            column_type: None,
            children: DashMap::new(),
            created_epoch: epoch,
        }
    }

    /// Create a leaf node with column information
    pub fn leaf(column_id: ColumnId, column_type: ColumnType, epoch: u64) -> Self {
        Self {
            column_id: Some(column_id),
            column_type: Some(column_type),
            children: DashMap::new(),
            created_epoch: epoch,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.column_id.is_some()
    }

    /// Count total nodes in subtree
    pub fn count_nodes(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|r| r.value().count_nodes())
            .sum::<usize>()
    }
}

/// Concurrent Path Trie with lock-free reads and epoch-based reclamation
///
/// This trie supports:
/// - Lock-free concurrent reads via DashMap
/// - Concurrent inserts with fine-grained locking
/// - Epoch-based garbage collection for safe memory reclamation
///
/// ## Performance
///
/// - Read: O(|path|) with no locking (DashMap provides lock-free reads)
/// - Insert: O(|path|) with minimal lock contention
/// - Memory reclamation: Deferred until no readers access old nodes
#[derive(Debug)]
pub struct ConcurrentPathTrie {
    /// Root node (never changes, only children do)
    root: Arc<ConcurrentTrieNode>,
    /// Total number of columns registered
    total_columns: AtomicU32,
    /// Next column ID to assign
    next_column_id: AtomicU32,
    /// Current epoch for versioning
    current_epoch: AtomicU64,
    /// Minimum active reader epoch (for GC)
    min_reader_epoch: AtomicU64,
    /// Reader count per epoch (for tracking when to collect)
    reader_epochs: DashMap<u64, AtomicU32>,
}

impl ConcurrentPathTrie {
    /// Create a new empty concurrent trie
    pub fn new() -> Self {
        Self {
            root: Arc::new(ConcurrentTrieNode::new(0)),
            total_columns: AtomicU32::new(0),
            next_column_id: AtomicU32::new(0),
            current_epoch: AtomicU64::new(1),
            min_reader_epoch: AtomicU64::new(0),
            reader_epochs: DashMap::new(),
        }
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Acquire)
    }

    /// Advance epoch (call periodically to allow GC)
    pub fn advance_epoch(&self) -> u64 {
        self.current_epoch.fetch_add(1, Ordering::AcqRel)
    }

    /// Begin a read operation (returns epoch guard)
    /// The guard must be held for the duration of the read
    pub fn begin_read(&self) -> ReadGuard<'_> {
        let epoch = self.current_epoch.load(Ordering::Acquire);

        // Increment reader count for this epoch
        self.reader_epochs
            .entry(epoch)
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(1, Ordering::Relaxed);

        ReadGuard { trie: self, epoch }
    }

    /// Insert a path and get its column ID (thread-safe)
    ///
    /// Path format: "users.profile.settings.theme"
    /// Returns the assigned column ID
    pub fn insert(&self, path: &str, column_type: ColumnType) -> ColumnId {
        let segments: Vec<&str> = path.split('.').collect();
        let column_id = self.next_column_id.fetch_add(1, Ordering::Relaxed);
        let epoch = self.current_epoch.load(Ordering::Acquire);

        let mut current = self.root.clone();

        for (i, segment) in segments.iter().enumerate() {
            let is_last = i == segments.len() - 1;

            if is_last {
                // Create or update leaf node
                let leaf = Arc::new(ConcurrentTrieNode::leaf(column_id, column_type, epoch));
                current.children.insert(segment.to_string(), leaf);
            } else {
                // Get or create intermediate node
                let next = current
                    .children
                    .entry(segment.to_string())
                    .or_insert_with(|| Arc::new(ConcurrentTrieNode::new(epoch)))
                    .clone();
                current = next;
            }
        }

        self.total_columns.fetch_add(1, Ordering::Relaxed);
        column_id
    }

    /// Resolve a path to its column ID in O(|path|) time (lock-free)
    ///
    /// Returns None if path doesn't exist
    pub fn resolve(&self, path: &str) -> Option<ColumnId> {
        let segments: Vec<&str> = path.split('.').collect();
        let mut current = self.root.clone();

        for segment in segments {
            let next = current.children.get(segment)?.clone();
            current = next;
        }

        current.column_id
    }

    /// Resolve with type information (lock-free)
    pub fn resolve_with_type(&self, path: &str) -> Option<(ColumnId, ColumnType)> {
        let segments: Vec<&str> = path.split('.').collect();
        let mut current = self.root.clone();

        for segment in segments {
            let next = current.children.get(segment)?.clone();
            current = next;
        }

        Some((current.column_id?, current.column_type?))
    }

    /// Get all paths that start with a prefix (lock-free)
    pub fn prefix_match(&self, prefix: &str) -> Vec<(String, ColumnId)> {
        let mut results = Vec::new();

        let segments: Vec<&str> = if prefix.is_empty() {
            vec![]
        } else {
            prefix.split('.').collect()
        };

        let mut current = self.root.clone();
        for segment in &segments {
            let next = match current.children.get(*segment) {
                Some(child) => child.clone(),
                None => return results,
            };
            current = next;
        }

        self.collect_paths(&current, prefix.to_string(), &mut results);
        results
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_paths(
        &self,
        node: &ConcurrentTrieNode,
        path: String,
        results: &mut Vec<(String, ColumnId)>,
    ) {
        if let Some(col_id) = node.column_id {
            results.push((path.clone(), col_id));
        }

        for entry in node.children.iter() {
            let child_path = if path.is_empty() {
                entry.key().clone()
            } else {
                format!("{}.{}", path, entry.key())
            };
            self.collect_paths(entry.value(), child_path, results);
        }
    }

    /// Get total number of columns
    pub fn total_columns(&self) -> u32 {
        self.total_columns.load(Ordering::Relaxed)
    }

    /// Get total number of nodes (memory usage indicator)
    pub fn total_nodes(&self) -> usize {
        self.root.count_nodes()
    }

    /// Update minimum reader epoch (call after readers complete)
    pub fn update_min_reader_epoch(&self) {
        let mut min_epoch = self.current_epoch.load(Ordering::Acquire);

        // Find minimum epoch with active readers
        for entry in self.reader_epochs.iter() {
            if entry.value().load(Ordering::Relaxed) > 0 && *entry.key() < min_epoch {
                min_epoch = *entry.key();
            }
        }

        self.min_reader_epoch.store(min_epoch, Ordering::Release);

        // Clean up old epoch entries
        let threshold = min_epoch.saturating_sub(10);
        self.reader_epochs
            .retain(|epoch, count| *epoch >= threshold || count.load(Ordering::Relaxed) > 0);
    }

    /// Get minimum reader epoch (nodes older than this can be reclaimed)
    pub fn min_reader_epoch(&self) -> u64 {
        self.min_reader_epoch.load(Ordering::Acquire)
    }
}

impl Default for ConcurrentPathTrie {
    fn default() -> Self {
        Self::new()
    }
}

// Make ConcurrentPathTrie Send + Sync
unsafe impl Send for ConcurrentPathTrie {}
unsafe impl Sync for ConcurrentPathTrie {}

/// RAII guard for read operations
/// Decrements reader count when dropped
pub struct ReadGuard<'a> {
    trie: &'a ConcurrentPathTrie,
    epoch: u64,
}

impl<'a> ReadGuard<'a> {
    /// Get the epoch this read is pinned to
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl<'a> Drop for ReadGuard<'a> {
    fn drop(&mut self) {
        if let Some(count) = self.trie.reader_epochs.get(&self.epoch) {
            count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_trie_insert_resolve() {
        let mut trie = PathTrie::new();

        let id1 = trie.insert("users.id", ColumnType::UInt64);
        let id2 = trie.insert("users.name", ColumnType::Text);
        let id3 = trie.insert("users.profile.email", ColumnType::Text);
        let id4 = trie.insert("users.profile.settings.theme", ColumnType::Text);

        assert_eq!(trie.resolve("users.id"), Some(id1));
        assert_eq!(trie.resolve("users.name"), Some(id2));
        assert_eq!(trie.resolve("users.profile.email"), Some(id3));
        assert_eq!(trie.resolve("users.profile.settings.theme"), Some(id4));
        assert_eq!(trie.resolve("nonexistent"), None);
        assert_eq!(trie.resolve("users.profile"), None); // Not a leaf

        assert_eq!(trie.total_columns(), 4);
    }

    #[test]
    fn test_path_trie_prefix_match() {
        let mut trie = PathTrie::new();

        trie.insert("users.id", ColumnType::UInt64);
        trie.insert("users.name", ColumnType::Text);
        trie.insert("users.profile.email", ColumnType::Text);
        trie.insert("orders.id", ColumnType::UInt64);

        let user_cols = trie.prefix_match("users");
        assert_eq!(user_cols.len(), 3);

        let profile_cols = trie.prefix_match("users.profile");
        assert_eq!(profile_cols.len(), 1);

        let all_cols = trie.prefix_match("");
        assert_eq!(all_cols.len(), 4);
    }

    #[test]
    fn test_resolve_with_type() {
        let mut trie = PathTrie::new();

        trie.insert("score", ColumnType::Float64);
        trie.insert("name", ColumnType::Text);

        let (id, col_type) = trie.resolve_with_type("score").unwrap();
        assert_eq!(id, 0);
        assert_eq!(col_type, ColumnType::Float64);

        let (id, col_type) = trie.resolve_with_type("name").unwrap();
        assert_eq!(id, 1);
        assert_eq!(col_type, ColumnType::Text);
    }

    #[test]
    fn test_column_group_affinity() {
        let mut trie = PathTrie::new();

        trie.insert("id", ColumnType::UInt64);
        trie.insert("score", ColumnType::Float64);
        trie.insert("age", ColumnType::Int64);
        trie.insert("name", ColumnType::Text);
        trie.insert("timestamp", ColumnType::Timestamp);

        let mut affinity = ColumnGroupAffinity::from_trie(&trie);

        // Check SIMD groups
        let simd = affinity.simd_groups();
        assert!(!simd.is_empty());

        // Record accesses
        affinity.record_access(0);
        affinity.record_access(0);
        affinity.record_access(1);

        let hot = affinity.hot_columns(2);
        assert_eq!(hot.len(), 2);
        assert_eq!(hot[0], 0); // Most accessed

        // Check sparse columns
        affinity.update_null_density(3, 90, 100);
        let sparse = affinity.sparse_columns(0.5);
        assert_eq!(sparse, vec![3]);
    }

    #[test]
    fn test_memory_estimate() {
        let mut trie = PathTrie::new();

        for i in 0..100 {
            trie.insert(
                &format!("table{}.column{}", i / 10, i % 10),
                ColumnType::UInt64,
            );
        }

        // ~10 tables × ~10 columns × 2-3 levels = ~300 nodes
        assert!(trie.total_nodes() > 100);
        assert!(trie.memory_bytes() > 10000); // At least 10KB
    }

    // ========================================================================
    // CONCURRENT PATH TRIE TESTS
    // ========================================================================

    #[test]
    fn test_concurrent_trie_basic() {
        let trie = ConcurrentPathTrie::new();

        let id1 = trie.insert("users.id", ColumnType::UInt64);
        let id2 = trie.insert("users.name", ColumnType::Text);
        let id3 = trie.insert("users.profile.email", ColumnType::Text);

        assert_eq!(trie.resolve("users.id"), Some(id1));
        assert_eq!(trie.resolve("users.name"), Some(id2));
        assert_eq!(trie.resolve("users.profile.email"), Some(id3));
        assert_eq!(trie.resolve("nonexistent"), None);

        assert_eq!(trie.total_columns(), 3);
    }

    #[test]
    fn test_concurrent_trie_resolve_with_type() {
        let trie = ConcurrentPathTrie::new();

        trie.insert("score", ColumnType::Float64);
        trie.insert("name", ColumnType::Text);

        let (id, col_type) = trie.resolve_with_type("score").unwrap();
        assert_eq!(id, 0);
        assert_eq!(col_type, ColumnType::Float64);

        let (id, col_type) = trie.resolve_with_type("name").unwrap();
        assert_eq!(id, 1);
        assert_eq!(col_type, ColumnType::Text);
    }

    #[test]
    fn test_concurrent_trie_prefix_match() {
        let trie = ConcurrentPathTrie::new();

        trie.insert("users.id", ColumnType::UInt64);
        trie.insert("users.name", ColumnType::Text);
        trie.insert("users.profile.email", ColumnType::Text);
        trie.insert("orders.id", ColumnType::UInt64);

        let user_cols = trie.prefix_match("users");
        assert_eq!(user_cols.len(), 3);

        let all_cols = trie.prefix_match("");
        assert_eq!(all_cols.len(), 4);
    }

    #[test]
    fn test_concurrent_trie_epoch_management() {
        let trie = ConcurrentPathTrie::new();

        assert_eq!(trie.current_epoch(), 1);

        // Advance epoch
        let old = trie.advance_epoch();
        assert_eq!(old, 1);
        assert_eq!(trie.current_epoch(), 2);

        // Begin read - should pin to current epoch
        let guard = trie.begin_read();
        assert_eq!(guard.epoch(), 2);

        // Advance epoch while read is active
        trie.advance_epoch();
        assert_eq!(trie.current_epoch(), 3);

        // Reader still at epoch 2
        assert_eq!(guard.epoch(), 2);

        // Drop guard and update min epoch
        drop(guard);
        trie.update_min_reader_epoch();
    }

    #[test]
    fn test_concurrent_trie_multithreaded() {
        use std::sync::Arc;
        use std::thread;

        let trie = Arc::new(ConcurrentPathTrie::new());
        let mut handles = vec![];

        // Spawn writer threads
        for i in 0..4 {
            let trie = Arc::clone(&trie);
            let handle = thread::spawn(move || {
                for j in 0..25 {
                    let path = format!("table{}.column{}", i, j);
                    trie.insert(&path, ColumnType::UInt64);
                }
            });
            handles.push(handle);
        }

        // Spawn reader threads
        for _ in 0..4 {
            let trie = Arc::clone(&trie);
            let handle = thread::spawn(move || {
                let _guard = trie.begin_read();
                // Read some paths (may or may not exist yet)
                for i in 0..4 {
                    for j in 0..25 {
                        let path = format!("table{}.column{}", i, j);
                        let _ = trie.resolve(&path);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All 100 columns should be inserted
        assert_eq!(trie.total_columns(), 100);

        // Verify all can be resolved
        for i in 0..4 {
            for j in 0..25 {
                let path = format!("table{}.column{}", i, j);
                assert!(trie.resolve(&path).is_some(), "Missing path: {}", path);
            }
        }
    }

    #[test]
    fn test_concurrent_trie_read_guard() {
        let trie = ConcurrentPathTrie::new();

        // Insert some data
        trie.insert("test.path", ColumnType::UInt64);

        // Multiple concurrent reads
        {
            let guard1 = trie.begin_read();
            let guard2 = trie.begin_read();

            assert_eq!(trie.resolve("test.path"), Some(0));
            assert_eq!(guard1.epoch(), guard2.epoch());

            // Guards dropped here
        }

        // Epoch tracking should work
        trie.update_min_reader_epoch();
    }
}
