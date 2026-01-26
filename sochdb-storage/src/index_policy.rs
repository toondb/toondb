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

//! Per-Table Index Policy + Scan-Optimized Structure
//!
//! This module replaces the global `enable_ordered_index` toggle with
//! a per-table index policy, allowing fine-grained control over write
//! throughput vs scan performance trade-offs.
//!
//! ## Problem: Global Toggle is Too Coarse
//!
//! The current `DatabaseConfig::enable_ordered_index` is a single global toggle.
//! This forces a choice:
//! - Enabled: Pay ~134 ns/op on ALL writes for O(log N) scans
//! - Disabled: Fast writes but O(N) scans on ALL tables
//!
//! Real workloads have mixed access patterns:
//! - Write-heavy logs tables → don't need ordered index
//! - Scan-heavy analytics tables → need ordered index
//! - OLTP tables → need balanced policy
//!
//! ## Solution: Per-Table Index Policy
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Table Index Registry                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  "users"    → WriteOptimized (no ordered index)                 │
//! │  "orders"   → Balanced (lazy compaction to sorted runs)          │
//! │  "analytics"→ ScanOptimized (maintain ordered index)             │
//! │  "logs"     → AppendOnly (no index, time-ordered writes)         │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Index Policies
//!
//! | Policy         | Insert Cost | Scan Cost      | Use Case              |
//! |----------------|-------------|----------------|------------------------|
//! | WriteOptimized | O(1)        | O(N)           | High-write, rare scan  |
//! | Balanced       | O(1) amort  | O(output+logK) | Mixed OLTP            |
//! | ScanOptimized  | O(log N)    | O(logN + K)    | Analytics, range query |
//! | AppendOnly     | O(1)        | O(N)           | Time-series logs       |
//!
//! ## LSM-Style Balanced Policy
//!
//! For `Balanced` tables, we use an LSM-style approach:
//! - Unsorted append-friendly memtable (O(1) inserts)
//! - Periodic compaction to sorted runs
//! - K-way merge for range scans (O(output + log K))
//!
//! This retains range-scan capability without paying O(log N) on every write.

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::key_buffer::ArenaKeyHandle;

// ============================================================================
// IndexPolicy - Per-Table Index Configuration
// ============================================================================

/// Index policy for a table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexPolicy {
    /// No ordered index - fastest writes, O(N) scans
    /// Use for write-heavy tables that rarely need range scans.
    WriteOptimized,
    
    /// LSM-style: unsorted memtable + periodic sorted runs
    /// Amortized O(1) inserts, O(output + log K) scans where K = run count.
    /// Good balance for mixed OLTP workloads.
    Balanced,
    
    /// Maintain ordered index on every write
    /// O(log N) inserts, O(log N + K) scans.
    /// Use for analytics tables with frequent range queries.
    ScanOptimized,
    
    /// Append-only with no indexing
    /// O(1) inserts, O(N) scans (but efficient forward iteration).
    /// Use for time-series logs where data is naturally ordered.
    AppendOnly,
}

impl Default for IndexPolicy {
    fn default() -> Self {
        IndexPolicy::Balanced
    }
}

impl IndexPolicy {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "write_optimized" | "write-optimized" | "write" => Some(IndexPolicy::WriteOptimized),
            "balanced" | "default" => Some(IndexPolicy::Balanced),
            "scan_optimized" | "scan-optimized" | "scan" => Some(IndexPolicy::ScanOptimized),
            "append_only" | "append-only" | "append" => Some(IndexPolicy::AppendOnly),
            _ => None,
        }
    }

    /// Get write cost description
    pub fn write_cost(&self) -> &'static str {
        match self {
            IndexPolicy::WriteOptimized => "O(1)",
            IndexPolicy::Balanced => "O(1) amortized",
            IndexPolicy::ScanOptimized => "O(log N)",
            IndexPolicy::AppendOnly => "O(1)",
        }
    }

    /// Get scan cost description
    pub fn scan_cost(&self) -> &'static str {
        match self {
            IndexPolicy::WriteOptimized => "O(N)",
            IndexPolicy::Balanced => "O(output + log K)",
            IndexPolicy::ScanOptimized => "O(log N + K)",
            IndexPolicy::AppendOnly => "O(N)",
        }
    }

    /// Whether this policy maintains an ordered index
    pub fn has_ordered_index(&self) -> bool {
        matches!(self, IndexPolicy::ScanOptimized)
    }

    /// Whether this policy supports efficient range scans
    pub fn supports_efficient_scans(&self) -> bool {
        matches!(self, IndexPolicy::ScanOptimized | IndexPolicy::Balanced)
    }
}

// ============================================================================
// TableIndexConfig - Configuration for a Single Table
// ============================================================================

/// Index configuration for a single table
#[derive(Debug, Clone)]
pub struct TableIndexConfig {
    /// Table name
    pub table_name: String,
    /// Index policy
    pub policy: IndexPolicy,
    /// Maximum sorted runs before compaction (for Balanced policy)
    pub max_sorted_runs: usize,
    /// Target sorted run size in bytes
    pub target_run_size: usize,
    /// Enable bloom filters for point queries
    pub enable_bloom_filter: bool,
}

impl TableIndexConfig {
    /// Create a new table index config
    pub fn new(table_name: impl Into<String>, policy: IndexPolicy) -> Self {
        Self {
            table_name: table_name.into(),
            policy,
            max_sorted_runs: 4,
            target_run_size: 16 * 1024 * 1024, // 16MB
            enable_bloom_filter: true,
        }
    }

    /// Builder: set max sorted runs
    pub fn with_max_sorted_runs(mut self, max: usize) -> Self {
        self.max_sorted_runs = max;
        self
    }

    /// Builder: set target run size
    pub fn with_target_run_size(mut self, size: usize) -> Self {
        self.target_run_size = size;
        self
    }

    /// Builder: enable/disable bloom filter
    pub fn with_bloom_filter(mut self, enable: bool) -> Self {
        self.enable_bloom_filter = enable;
        self
    }
}

// ============================================================================
// TableIndexRegistry - Central Registry for Per-Table Policies
// ============================================================================

/// Registry of per-table index policies
///
/// Allows setting different index policies for different tables,
/// with a default policy for tables not explicitly configured.
pub struct TableIndexRegistry {
    /// Per-table configurations
    configs: DashMap<String, TableIndexConfig>,
    /// Default policy for unconfigured tables
    default_policy: RwLock<IndexPolicy>,
}

impl TableIndexRegistry {
    /// Create a new registry with default Balanced policy
    pub fn new() -> Self {
        Self {
            configs: DashMap::new(),
            default_policy: RwLock::new(IndexPolicy::Balanced),
        }
    }

    /// Create with a specific default policy
    pub fn with_default_policy(policy: IndexPolicy) -> Self {
        Self {
            configs: DashMap::new(),
            default_policy: RwLock::new(policy),
        }
    }

    /// Set the default policy for unconfigured tables
    pub fn set_default_policy(&self, policy: IndexPolicy) {
        *self.default_policy.write() = policy;
    }

    /// Get the default policy
    pub fn default_policy(&self) -> IndexPolicy {
        *self.default_policy.read()
    }

    /// Configure a table with a specific policy
    pub fn configure_table(&self, config: TableIndexConfig) {
        self.configs.insert(config.table_name.clone(), config);
    }

    /// Get the policy for a table
    pub fn get_policy(&self, table_name: &str) -> IndexPolicy {
        self.configs
            .get(table_name)
            .map(|c| c.policy)
            .unwrap_or_else(|| *self.default_policy.read())
    }

    /// Get the full config for a table (or default)
    pub fn get_config(&self, table_name: &str) -> TableIndexConfig {
        self.configs
            .get(table_name)
            .map(|c| c.clone())
            .unwrap_or_else(|| {
                TableIndexConfig::new(table_name, *self.default_policy.read())
            })
    }

    /// Check if a table has an explicitly configured policy
    pub fn has_explicit_config(&self, table_name: &str) -> bool {
        self.configs.contains_key(table_name)
    }

    /// Remove a table's configuration (reverts to default)
    pub fn remove_config(&self, table_name: &str) -> Option<TableIndexConfig> {
        self.configs.remove(table_name).map(|(_, c)| c)
    }

    /// List all configured tables
    pub fn configured_tables(&self) -> Vec<String> {
        self.configs.iter().map(|e| e.key().clone()).collect()
    }
}

impl Default for TableIndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SortedRun - Immutable Sorted Segment for Balanced Policy
// ============================================================================

/// An immutable sorted segment of key-value pairs
///
/// Used by the Balanced policy for LSM-style scan optimization.
/// Each run is sorted by key, enabling efficient k-way merge.
/// 
/// ## Key Range Metadata
/// 
/// Each run stores `min_key` and `max_key` bounds for O(1) overlap checking.
/// This enables prefix scan pruning: runs that don't overlap the prefix
/// range can be skipped entirely.
#[derive(Debug)]
pub struct SortedRun<K, V> {
    /// Sorted entries
    entries: Vec<(K, V)>,
    /// Minimum key in this run (for scan pruning)
    min_key: Option<K>,
    /// Maximum key in this run (for scan pruning)
    max_key: Option<K>,
    /// Size in bytes (approximate)
    size_bytes: usize,
    /// Creation timestamp
    #[allow(dead_code)]
    created_at: std::time::Instant,
    /// Run level in LSM hierarchy
    level: usize,
}

impl<K: Ord + Clone, V: Clone> SortedRun<K, V> {
    /// Create a new sorted run from unsorted entries
    pub fn from_unsorted(mut entries: Vec<(K, V)>, level: usize) -> Self {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let size_bytes = std::mem::size_of_val(&entries);
        let min_key = entries.first().map(|(k, _)| k.clone());
        let max_key = entries.last().map(|(k, _)| k.clone());
        Self {
            entries,
            min_key,
            max_key,
            size_bytes,
            created_at: std::time::Instant::now(),
            level,
        }
    }

    /// Create from already-sorted entries
    pub fn from_sorted(entries: Vec<(K, V)>, level: usize) -> Self {
        let size_bytes = std::mem::size_of_val(&entries);
        let min_key = entries.first().map(|(k, _)| k.clone());
        let max_key = entries.last().map(|(k, _)| k.clone());
        Self {
            entries,
            min_key,
            max_key,
            size_bytes,
            created_at: std::time::Instant::now(),
            level,
        }
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get the level
    pub fn level(&self) -> usize {
        self.level
    }

    /// Binary search for a key
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Range scan from start key
    pub fn range_from<'a>(&'a self, start: &K) -> impl Iterator<Item = &'a (K, V)> {
        let idx = self.entries
            .binary_search_by(|(k, _)| k.cmp(start))
            .unwrap_or_else(|i| i);
        self.entries[idx..].iter()
    }

    /// Range scan with bounds
    pub fn range<'a>(&'a self, start: &K, end: &K) -> impl Iterator<Item = &'a (K, V)> {
        let start_idx = self.entries
            .binary_search_by(|(k, _)| k.cmp(start))
            .unwrap_or_else(|i| i);
        let end_idx = self.entries
            .binary_search_by(|(k, _)| k.cmp(end))
            .unwrap_or_else(|i| i);
        self.entries[start_idx..end_idx].iter()
    }

    /// Iterate all entries
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }

    /// Direct access to underlying entries for O(1) indexing
    /// 
    /// Required for efficient k-way merge. Without this accessor,
    /// callers are forced to use `iter().nth()` which is O(n) per call.
    #[inline]
    pub fn entries(&self) -> &[(K, V)] {
        &self.entries
    }

    /// Check if this run might contain keys with the given prefix.
    ///
    /// Uses stored min_key and max_key bounds for O(1) overlap check.
    /// Returns `true` if the run may contain matching keys (conservative).
    /// Returns `false` only if we can prove no keys match.
    ///
    /// # Prefix Overlap Logic
    ///
    /// For a run with [min_key, max_key] to overlap prefix range [prefix, prefix++):
    /// - If max_key < prefix → run is entirely before prefix → no overlap
    /// - If min_key starts with prefix OR min_key < prefix and max_key >= prefix → overlap
    ///
    /// We use a conservative check: return true unless max_key < prefix.
    #[inline]
    pub fn overlaps_prefix(&self, prefix: &K) -> bool {
        match &self.max_key {
            Some(max) if max < prefix => false, // Run entirely before prefix
            _ => true, // Could overlap (conservative)
        }
    }

    /// Check if this run might contain keys in the given range.
    ///
    /// Uses stored min_key and max_key bounds for O(1) overlap check.
    /// Returns `true` if the run may contain matching keys (conservative).
    #[inline]
    pub fn overlaps_range(&self, start: &K, end: &K) -> bool {
        // Check if run is entirely outside the range
        match (&self.min_key, &self.max_key) {
            (Some(min), _) if min >= end => false,  // Run entirely after range
            (_, Some(max)) if max < start => false, // Run entirely before range
            _ => true, // Could overlap
        }
    }

    /// Get the minimum key in this run, if any
    #[inline]
    pub fn min_key(&self) -> Option<&K> {
        self.min_key.as_ref()
    }

    /// Get the maximum key in this run, if any
    #[inline]
    pub fn max_key(&self) -> Option<&K> {
        self.max_key.as_ref()
    }
}

// ============================================================================
// BalancedTableIndex - LSM-Style Index for Balanced Policy
// ============================================================================

/// LSM-style index for the Balanced policy
///
/// Combines:
/// - Unsorted DashMap for O(1) writes
/// - Periodic compaction to sorted runs
/// - K-way merge for range scans
pub struct BalancedTableIndex<V: Clone + Send + Sync + Eq + 'static> {
    /// Unsorted memtable for fast writes
    memtable: DashMap<ArenaKeyHandle, V>,
    /// Sorted runs for efficient scans
    sorted_runs: RwLock<Vec<Arc<SortedRun<ArenaKeyHandle, V>>>>,
    /// Configuration
    config: TableIndexConfig,
    /// Size of memtable in bytes
    memtable_size: std::sync::atomic::AtomicUsize,
}

impl<V: Clone + Send + Sync + Eq + 'static> BalancedTableIndex<V> {
    /// Create a new balanced table index
    pub fn new(config: TableIndexConfig) -> Self {
        Self {
            memtable: DashMap::new(),
            sorted_runs: RwLock::new(Vec::new()),
            config,
            memtable_size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Insert a key-value pair (O(1))
    pub fn insert(&self, key: ArenaKeyHandle, value: V) {
        let key_size = key.len();
        let value_size = std::mem::size_of::<V>();
        
        self.memtable.insert(key, value);
        self.memtable_size.fetch_add(key_size + value_size, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get a value by key
    /// 
    /// Uses run metadata for O(1) pruning: skips runs where 
    /// `max_key < key` (run entirely before search key).
    pub fn get(&self, key: &ArenaKeyHandle) -> Option<V> {
        // Check memtable first
        if let Some(v) = self.memtable.get(key) {
            return Some(v.clone());
        }

        // Check sorted runs (newest first)
        // Use metadata pruning to skip runs that can't contain the key
        let runs = self.sorted_runs.read();
        for run in runs.iter().rev() {
            // Prune: skip runs that are entirely before the key
            if run.overlaps_prefix(key) {
                if let Some(v) = run.get(key) {
                    return Some(v.clone());
                }
            }
        }

        None
    }

    /// Scan entries with a given key prefix
    /// 
    /// Uses run metadata for O(1) pruning: only scans runs whose
    /// key range overlaps with the prefix. Returns merged results
    /// in key order with duplicates resolved (newest wins).
    /// 
    /// # Pruning Benefit
    /// 
    /// For selective prefixes (e.g., "user:123:" in a table with 1M keys),
    /// most runs will have `max_key < prefix` or `min_key > prefix_end`,
    /// allowing them to be skipped entirely.
    pub fn scan_prefix(&self, prefix: &ArenaKeyHandle) -> Vec<(ArenaKeyHandle, V)> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        #[derive(Eq, PartialEq)]
        struct PrefixHeapEntry<V: Clone> {
            key: ArenaKeyHandle,
            value: V,
            source_idx: usize,  // 0 = memtable, 1+ = sorted runs
        }

        impl<V: Clone + Eq + PartialEq> Ord for PrefixHeapEntry<V> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match self.key.cmp(&other.key) {
                    std::cmp::Ordering::Equal => self.source_idx.cmp(&other.source_idx),
                    other => other,
                }
            }
        }

        impl<V: Clone + Eq + PartialEq> PartialOrd for PrefixHeapEntry<V> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap: BinaryHeap<Reverse<PrefixHeapEntry<V>>> = BinaryHeap::new();

        // Collect matching entries from memtable
        for entry in self.memtable.iter() {
            let key = entry.key();
            if key.as_bytes().starts_with(prefix.as_bytes()) {
                heap.push(Reverse(PrefixHeapEntry {
                    key: key.clone(),
                    value: entry.value().clone(),
                    source_idx: 0,
                }));
            }
        }

        // Collect from sorted runs, using metadata pruning
        let runs = self.sorted_runs.read();
        for (run_idx, run) in runs.iter().enumerate() {
            // Prune: skip runs that can't contain any matching keys
            if !run.overlaps_prefix(prefix) {
                continue;  // Run is entirely before prefix
            }

            // Scan matching entries in this run
            for (key, value) in run.range_from(prefix) {
                if !key.as_bytes().starts_with(prefix.as_bytes()) {
                    break;  // Past prefix range
                }
                heap.push(Reverse(PrefixHeapEntry {
                    key: key.clone(),
                    value: value.clone(),
                    source_idx: run_idx + 1,  // 1-indexed (0 = memtable)
                }));
            }
        }

        // Merge and deduplicate
        let mut result = Vec::with_capacity(heap.len());
        let mut last_key: Option<ArenaKeyHandle> = None;

        while let Some(Reverse(entry)) = heap.pop() {
            let is_new_key = last_key.as_ref().map(|k| k != &entry.key).unwrap_or(true);
            if is_new_key {
                last_key = Some(entry.key.clone());
                result.push((entry.key, entry.value));
            }
        }

        result
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        let memtable_size = self.memtable_size.load(std::sync::atomic::Ordering::Relaxed);
        let runs = self.sorted_runs.read();
        
        memtable_size >= self.config.target_run_size
            || runs.len() >= self.config.max_sorted_runs
    }

    /// Compact memtable to a new sorted run
    pub fn compact_memtable(&self) {
        // Drain memtable
        let entries: Vec<_> = self.memtable.iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        
        if entries.is_empty() {
            return;
        }

        // Clear memtable
        self.memtable.clear();
        self.memtable_size.store(0, std::sync::atomic::Ordering::Relaxed);

        // Create new sorted run
        let run = Arc::new(SortedRun::from_unsorted(entries, 0));
        
        let mut runs = self.sorted_runs.write();
        runs.push(run);
    }

    /// Merge multiple sorted runs (compaction)
    pub fn merge_runs(&self, levels_to_merge: usize) {
        let mut runs = self.sorted_runs.write();
        
        if runs.len() < levels_to_merge {
            return;
        }

        // Take the oldest runs to merge
        let to_merge: Vec<_> = runs.drain(..levels_to_merge).collect();
        
        // K-way merge
        let merged = self.k_way_merge(&to_merge);
        
        // Create new run at next level
        let new_run = Arc::new(SortedRun::from_sorted(merged, to_merge.len()));
        runs.insert(0, new_run);
    }

    /// K-way merge of sorted runs
    /// 
    /// Complexity: O(N log K) where N = total entries, K = number of runs
    /// 
    /// Key insight: Use direct indexing into the underlying Vec instead of 
    /// creating new iterators. `iter().nth(n)` is O(n), making the old
    /// implementation O(N²/K).
    fn k_way_merge(&self, runs: &[Arc<SortedRun<ArenaKeyHandle, V>>]) -> Vec<(ArenaKeyHandle, V)> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        #[derive(Eq, PartialEq)]
        struct HeapEntry<V: Clone> {
            key: ArenaKeyHandle,
            value: V,
            run_idx: usize,
            entry_idx: usize,
        }

        impl<V: Clone + Eq + PartialEq> Ord for HeapEntry<V> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Primary: key order (min-heap via Reverse wrapper)
                // Secondary: run_idx for stability (lower = older = superseded)
                match self.key.cmp(&other.key) {
                    std::cmp::Ordering::Equal => self.run_idx.cmp(&other.run_idx),
                    other => other,
                }
            }
        }

        impl<V: Clone + Eq + PartialEq> PartialOrd for HeapEntry<V> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap: BinaryHeap<Reverse<HeapEntry<V>>> = BinaryHeap::new();
        
        // Track current position in each run (index into the underlying Vec)
        let mut run_positions: Vec<usize> = vec![0; runs.len()];
        
        // Initialize heap with first entry from each run using direct indexing
        for (run_idx, run) in runs.iter().enumerate() {
            let entries = run.entries();
            if !entries.is_empty() {
                let (key, value) = &entries[0];  // O(1) direct access
                heap.push(Reverse(HeapEntry {
                    key: key.clone(),
                    value: value.clone(),
                    run_idx,
                    entry_idx: 0,
                }));
            }
        }

        // Pre-allocate result based on estimated total entries
        let estimated_size: usize = runs.iter().map(|r| r.len()).sum();
        let mut result = Vec::with_capacity(estimated_size);
        let mut last_key: Option<ArenaKeyHandle> = None;

        while let Some(Reverse(entry)) = heap.pop() {
            // Duplicate suppression: keep only the first occurrence (newest due to run ordering)
            let is_new_key = last_key.as_ref().map(|k| k != &entry.key).unwrap_or(true);
            if is_new_key {
                last_key = Some(entry.key.clone());
                result.push((entry.key.clone(), entry.value));
            }

            // Advance position in this run
            run_positions[entry.run_idx] += 1;
            let next_idx = run_positions[entry.run_idx];
            
            // FIX: Direct indexing is O(1), not O(n) like iter().nth()
            let run_entries = runs[entry.run_idx].entries();
            if next_idx < run_entries.len() {
                let (key, value) = &run_entries[next_idx];  // O(1) access
                heap.push(Reverse(HeapEntry {
                    key: key.clone(),
                    value: value.clone(),
                    run_idx: entry.run_idx,
                    entry_idx: next_idx,
                }));
            }
        }

        result
    }

    /// Get table config
    pub fn config(&self) -> &TableIndexConfig {
        &self.config
    }

    /// Get memtable size
    pub fn memtable_size(&self) -> usize {
        self.memtable_size.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get number of sorted runs
    pub fn run_count(&self) -> usize {
        self.sorted_runs.read().len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_policy_from_str() {
        assert_eq!(IndexPolicy::from_str("write_optimized"), Some(IndexPolicy::WriteOptimized));
        assert_eq!(IndexPolicy::from_str("balanced"), Some(IndexPolicy::Balanced));
        assert_eq!(IndexPolicy::from_str("scan-optimized"), Some(IndexPolicy::ScanOptimized));
        assert_eq!(IndexPolicy::from_str("append_only"), Some(IndexPolicy::AppendOnly));
        assert_eq!(IndexPolicy::from_str("invalid"), None);
    }

    #[test]
    fn test_registry_default_policy() {
        let registry = TableIndexRegistry::new();
        
        // Unconfigured table gets default policy
        assert_eq!(registry.get_policy("unknown"), IndexPolicy::Balanced);
        
        // Configure a table
        registry.configure_table(TableIndexConfig::new("users", IndexPolicy::WriteOptimized));
        assert_eq!(registry.get_policy("users"), IndexPolicy::WriteOptimized);
        
        // Other tables still get default
        assert_eq!(registry.get_policy("orders"), IndexPolicy::Balanced);
    }

    #[test]
    fn test_registry_change_default() {
        let registry = TableIndexRegistry::new();
        
        registry.set_default_policy(IndexPolicy::ScanOptimized);
        assert_eq!(registry.get_policy("any_table"), IndexPolicy::ScanOptimized);
    }

    #[test]
    fn test_sorted_run() {
        let entries = vec![
            (ArenaKeyHandle::new(b"c"), 3),
            (ArenaKeyHandle::new(b"a"), 1),
            (ArenaKeyHandle::new(b"b"), 2),
        ];
        
        let run = SortedRun::from_unsorted(entries, 0);
        
        assert_eq!(run.len(), 3);
        assert_eq!(run.get(&ArenaKeyHandle::new(b"a")), Some(&1));
        assert_eq!(run.get(&ArenaKeyHandle::new(b"b")), Some(&2));
        assert_eq!(run.get(&ArenaKeyHandle::new(b"c")), Some(&3));
        assert_eq!(run.get(&ArenaKeyHandle::new(b"d")), None);
    }

    #[test]
    fn test_balanced_table_index() {
        let config = TableIndexConfig::new("test", IndexPolicy::Balanced);
        let index: BalancedTableIndex<i32> = BalancedTableIndex::new(config);
        
        index.insert(ArenaKeyHandle::new(b"key1"), 1);
        index.insert(ArenaKeyHandle::new(b"key2"), 2);
        
        assert_eq!(index.get(&ArenaKeyHandle::new(b"key1")), Some(1));
        assert_eq!(index.get(&ArenaKeyHandle::new(b"key2")), Some(2));
        assert_eq!(index.get(&ArenaKeyHandle::new(b"key3")), None);
    }

    #[test]
    fn test_balanced_compaction() {
        let config = TableIndexConfig::new("test", IndexPolicy::Balanced)
            .with_target_run_size(100); // Small size to trigger compaction
        
        let index: BalancedTableIndex<i32> = BalancedTableIndex::new(config);
        
        for i in 0..10 {
            let key = format!("key{:03}", i);
            index.insert(ArenaKeyHandle::new(key.as_bytes()), i as i32);
        }
        
        // Compact memtable
        index.compact_memtable();
        
        assert_eq!(index.run_count(), 1);
        assert_eq!(index.memtable_size(), 0);
        
        // Values should still be accessible
        assert_eq!(index.get(&ArenaKeyHandle::new(b"key005")), Some(5));
    }

    #[test]
    fn test_k_way_merge_scaling() {
        // Verify O(N log K) complexity by checking that merge time scales linearly
        // with N (not quadratically as the old iter().nth() implementation would)
        use std::time::Instant;
        
        let sizes = [100, 500, 1000];
        let mut times_ns: Vec<u128> = Vec::new();
        
        for size in sizes {
            // Create 5 runs with `size` entries each
            let runs: Vec<Arc<SortedRun<ArenaKeyHandle, i32>>> = (0..5)
                .map(|run_id| {
                    let entries: Vec<(ArenaKeyHandle, i32)> = (0..size)
                        .map(|i| {
                            let key = format!("key_{:08}_{}", i * 5 + run_id, run_id);
                            (ArenaKeyHandle::new(key.as_bytes()), (i * 5 + run_id) as i32)
                        })
                        .collect();
                    Arc::new(SortedRun::from_sorted(entries, run_id))
                })
                .collect();
            
            let config = TableIndexConfig::new("test", IndexPolicy::Balanced);
            let index: BalancedTableIndex<i32> = BalancedTableIndex::new(config);
            
            let start = Instant::now();
            let merged = index.k_way_merge(&runs);
            let elapsed = start.elapsed();
            
            times_ns.push(elapsed.as_nanos());
            
            // Verify merge produced correct output
            let total_entries = size * 5;
            assert_eq!(merged.len(), total_entries, "Merge should produce all unique entries");
        }
        
        // For O(N log K) scaling, time should roughly double when N doubles
        // (since log K is constant). For O(N²), time would quadruple.
        // We check that the ratio is closer to linear than quadratic.
        if times_ns.len() >= 2 && times_ns[0] > 0 {
            let ratio_1_to_2 = times_ns[1] as f64 / times_ns[0] as f64;
            let ratio_2_to_3 = times_ns[2] as f64 / times_ns[1] as f64;
            
            // For linear scaling with 5x size increase, expect ~5x time increase
            // For quadratic, expect ~25x. We assert it's closer to linear.
            assert!(ratio_1_to_2 < 15.0, 
                "Merge scaling should be sub-quadratic: ratio={:.1}x for 5x size", ratio_1_to_2);
            assert!(ratio_2_to_3 < 10.0,
                "Merge scaling should be sub-quadratic: ratio={:.1}x for 2x size", ratio_2_to_3);
        }
    }

    #[test]
    fn test_sorted_run_metadata_pruning() {
        // Test that min_key and max_key are correctly computed
        let entries = vec![
            (ArenaKeyHandle::new(b"apple"), 1),
            (ArenaKeyHandle::new(b"banana"), 2),
            (ArenaKeyHandle::new(b"cherry"), 3),
        ];
        let run = SortedRun::from_sorted(entries, 0);
        
        // Verify min/max are set correctly
        assert_eq!(run.min_key().map(|k| k.as_bytes()), Some(b"apple".as_slice()));
        assert_eq!(run.max_key().map(|k| k.as_bytes()), Some(b"cherry".as_slice()));
        
        // Test overlaps_prefix pruning
        assert!(run.overlaps_prefix(&ArenaKeyHandle::new(b"banana"))); // In range
        assert!(run.overlaps_prefix(&ArenaKeyHandle::new(b"apple")));  // At start
        assert!(run.overlaps_prefix(&ArenaKeyHandle::new(b"cherry"))); // At end
        
        // Prefix BEFORE range should not overlap (max_key < prefix)
        assert!(!run.overlaps_prefix(&ArenaKeyHandle::new(b"date")));  // After range
        assert!(!run.overlaps_prefix(&ArenaKeyHandle::new(b"zebra"))); // Way after
        
        // Test overlaps_range pruning
        assert!(run.overlaps_range(
            &ArenaKeyHandle::new(b"banana"),
            &ArenaKeyHandle::new(b"cherry")
        ));
        assert!(!run.overlaps_range(
            &ArenaKeyHandle::new(b"date"),
            &ArenaKeyHandle::new(b"fig")
        )); // Entirely after
        assert!(!run.overlaps_range(
            &ArenaKeyHandle::new(b"aaa"),
            &ArenaKeyHandle::new(b"aab")
        )); // Entirely before
    }

    #[test]
    fn test_scan_prefix() {
        let config = TableIndexConfig::new("test", IndexPolicy::Balanced)
            .with_target_run_size(50); // Small to trigger compaction
        let index: BalancedTableIndex<i32> = BalancedTableIndex::new(config);
        
        // Insert entries with different prefixes
        let prefixes = ["user:1:", "user:2:", "order:1:", "order:2:"];
        for (i, prefix) in prefixes.iter().enumerate() {
            for j in 0..5 {
                let key = format!("{}{}", prefix, j);
                index.insert(ArenaKeyHandle::new(key.as_bytes()), (i * 10 + j) as i32);
            }
        }
        
        // Compact to create sorted runs
        index.compact_memtable();
        
        // Add more entries to memtable
        index.insert(ArenaKeyHandle::new(b"user:1:99"), 199);
        index.insert(ArenaKeyHandle::new(b"order:1:99"), 299);
        
        // Scan for user:1: prefix
        let results = index.scan_prefix(&ArenaKeyHandle::new(b"user:1:"));
        assert_eq!(results.len(), 6); // 5 from run + 1 from memtable
        
        // Verify all results have the correct prefix
        for (key, _value) in &results {
            assert!(key.as_bytes().starts_with(b"user:1:"), 
                "Key {:?} should start with user:1:", String::from_utf8_lossy(key.as_bytes()));
        }
        
        // Verify results are sorted
        for window in results.windows(2) {
            assert!(window[0].0 <= window[1].0, "Results should be sorted by key");
        }
        
        // Scan for order: prefix
        let results = index.scan_prefix(&ArenaKeyHandle::new(b"order:"));
        assert_eq!(results.len(), 11); // 10 from run + 1 from memtable
    }

    #[test]
    fn test_empty_sorted_run_metadata() {
        // Empty run should have None for min/max
        let entries: Vec<(ArenaKeyHandle, i32)> = vec![];
        let run = SortedRun::from_sorted(entries, 0);
        
        assert!(run.min_key().is_none());
        assert!(run.max_key().is_none());
        assert!(run.overlaps_prefix(&ArenaKeyHandle::new(b"anything"))); // Conservative: true
        assert!(run.overlaps_range(
            &ArenaKeyHandle::new(b"a"),
            &ArenaKeyHandle::new(b"z")
        )); // Conservative: true
    }
}
