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

//! Queue-Optimized Index Policy
//!
//! This module extends the per-table index policy with queue-specific
//! optimizations that ensure efficient priority queue operations.
//!
//! ## Queue Access Patterns
//!
//! Queues have specific access patterns that differ from general tables:
//!
//! | Operation | Pattern                              | Requirement          |
//! |-----------|--------------------------------------|----------------------|
//! | Enqueue   | Insert at any position               | O(log N) or better   |
//! | Dequeue   | Find minimum key, delete it          | O(log N) find + O(1) delete |
//! | Peek      | Read minimum key without deletion    | O(log N)             |
//! | Count     | Get queue size                       | O(1)                 |
//!
//! ## Why Queue Tables Need ScanOptimized Policy
//!
//! The dequeue operation requires "find minimum key", which is:
//! - O(log N) with ordered index (ScanOptimized)
//! - O(N) without ordered index (WriteOptimized/Balanced with deferred sort)
//!
//! For a queue with 10,000 tasks:
//! - With ScanOptimized: ~14 comparisons per dequeue
//! - With WriteOptimized: ~10,000 comparisons per dequeue
//!
//! ## Avoiding Deferred-Sort Latency Spikes
//!
//! The Balanced policy uses "deferred sorting" where writes are O(1) append
//! and scans trigger O(N log N) sort-on-demand. This creates latency spikes:
//!
//! ```text
//! Pop #1: 0.1ms (memtable small)
//! Pop #2: 0.1ms
//! ...
//! Pop #1000: 50ms (sort triggered!) ← Latency spike
//! Pop #1001: 0.2ms (now sorted)
//! ```
//!
//! ScanOptimized maintains order on every write, giving predictable latency.
//!
//! ## Queue Index Configuration
//!
//! ```rust
//! let config = QueueIndexConfig::new("task_queue")
//!     .with_priority_column("priority")
//!     .with_timestamp_column("ready_at")
//!     .with_fifo_column("sequence")
//!     .build();
//! ```

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::index_policy::{IndexPolicy, TableIndexConfig, TableIndexRegistry};
use crate::key_buffer::ArenaKeyHandle;

// ============================================================================
// QueueIndexConfig - Queue-Specific Configuration
// ============================================================================

/// Configuration for queue tables
#[derive(Debug, Clone)]
pub struct QueueIndexConfig {
    /// Base table configuration
    pub base: TableIndexConfig,
    /// Name of the priority column (for composite key ordering)
    pub priority_column: Option<String>,
    /// Name of the timestamp column (for ready-time ordering)
    pub timestamp_column: Option<String>,
    /// Name of the sequence column (for FIFO within same priority)
    pub fifo_column: Option<String>,
    /// Whether to maintain min-key cache for O(1) peek
    pub enable_min_key_cache: bool,
    /// Whether to track queue size for O(1) count
    pub enable_size_tracking: bool,
}

impl QueueIndexConfig {
    /// Create a new queue index config
    pub fn new(queue_name: impl Into<String>) -> Self {
        Self {
            base: TableIndexConfig::new(queue_name, IndexPolicy::ScanOptimized),
            priority_column: None,
            timestamp_column: None,
            fifo_column: None,
            enable_min_key_cache: true,
            enable_size_tracking: true,
        }
    }

    /// Set the priority column name
    pub fn with_priority_column(mut self, column: impl Into<String>) -> Self {
        self.priority_column = Some(column.into());
        self
    }

    /// Set the timestamp column name
    pub fn with_timestamp_column(mut self, column: impl Into<String>) -> Self {
        self.timestamp_column = Some(column.into());
        self
    }

    /// Set the FIFO sequence column name
    pub fn with_fifo_column(mut self, column: impl Into<String>) -> Self {
        self.fifo_column = Some(column.into());
        self
    }

    /// Enable or disable min-key cache
    pub fn with_min_key_cache(mut self, enable: bool) -> Self {
        self.enable_min_key_cache = enable;
        self
    }

    /// Enable or disable size tracking
    pub fn with_size_tracking(mut self, enable: bool) -> Self {
        self.enable_size_tracking = enable;
        self
    }

    /// Get the composite key columns for this queue
    pub fn key_columns(&self) -> Vec<&str> {
        let mut columns = Vec::new();
        if let Some(ref col) = self.priority_column {
            columns.push(col.as_str());
        }
        if let Some(ref col) = self.timestamp_column {
            columns.push(col.as_str());
        }
        if let Some(ref col) = self.fifo_column {
            columns.push(col.as_str());
        }
        columns
    }
}

// ============================================================================
// QueueIndex - Queue-Optimized Index Structure
// ============================================================================

/// A queue-optimized ordered index
///
/// This provides efficient priority queue operations by:
/// 1. Maintaining a BTreeMap for O(log N) min-key access
/// 2. Caching the minimum key for O(1) peek
/// 3. Tracking size for O(1) count
///
/// ## Internal Structure
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────┐
/// │                         QueueIndex                                   │
/// ├─────────────────────────────────────────────────────────────────────┤
/// │ entries: BTreeMap<CompositeKey, Value>  ← O(log N) ordered ops      │
/// │ min_key_cache: Option<CompositeKey>     ← O(1) peek                 │
/// │ size: AtomicUsize                       ← O(1) count                │
/// │ version: AtomicU64                      ← For cache invalidation    │
/// └─────────────────────────────────────────────────────────────────────┘
/// ```
pub struct QueueIndex<V: Clone + Send + Sync> {
    /// The ordered entries
    entries: RwLock<BTreeMap<CompositeQueueKey, V>>,
    /// Cached minimum key (invalidated on mutation)
    min_key_cache: RwLock<Option<CompositeQueueKey>>,
    /// Current size
    size: AtomicUsize,
    /// Version counter for cache invalidation
    version: AtomicU64,
    /// Configuration
    config: QueueIndexConfig,
}

/// Composite key for queue ordering
///
/// Encodes: priority + timestamp + sequence for deterministic ordering.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompositeQueueKey {
    /// Primary sort: priority (lower = more urgent)
    pub priority: i64,
    /// Secondary sort: ready timestamp
    pub timestamp: u64,
    /// Tertiary sort: sequence number (for FIFO within same priority/time)
    pub sequence: u64,
    /// Task identifier
    pub task_id: String,
}

impl CompositeQueueKey {
    /// Create a new composite key
    pub fn new(priority: i64, timestamp: u64, sequence: u64, task_id: impl Into<String>) -> Self {
        Self {
            priority,
            timestamp,
            sequence,
            task_id: task_id.into(),
        }
    }

    /// Encode to bytes for storage
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32 + self.task_id.len());
        
        // Priority: map i64 to u64 preserving order
        let priority_encoded = (self.priority as i128 + i64::MAX as i128 + 1) as u64;
        bytes.extend_from_slice(&priority_encoded.to_be_bytes());
        
        // Timestamp: big-endian
        bytes.extend_from_slice(&self.timestamp.to_be_bytes());
        
        // Sequence: big-endian
        bytes.extend_from_slice(&self.sequence.to_be_bytes());
        
        // Task ID
        bytes.extend_from_slice(self.task_id.as_bytes());
        
        bytes
    }

    /// Decode from bytes
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        
        let priority_encoded = u64::from_be_bytes(bytes[0..8].try_into().ok()?);
        let priority = (priority_encoded as i128 - i64::MAX as i128 - 1) as i64;
        
        let timestamp = u64::from_be_bytes(bytes[8..16].try_into().ok()?);
        let sequence = u64::from_be_bytes(bytes[16..24].try_into().ok()?);
        let task_id = String::from_utf8(bytes[24..].to_vec()).ok()?;
        
        Some(Self {
            priority,
            timestamp,
            sequence,
            task_id,
        })
    }
}

impl<V: Clone + Send + Sync> QueueIndex<V> {
    /// Create a new queue index
    pub fn new(config: QueueIndexConfig) -> Self {
        Self {
            entries: RwLock::new(BTreeMap::new()),
            min_key_cache: RwLock::new(None),
            size: AtomicUsize::new(0),
            version: AtomicU64::new(0),
            config,
        }
    }

    /// Insert a task into the queue
    ///
    /// Complexity: O(log N)
    pub fn insert(&self, key: CompositeQueueKey, value: V) {
        let is_new_min = {
            let entries = self.entries.read();
            entries.first_key_value()
                .map(|(min, _)| &key < min)
                .unwrap_or(true)
        };
        
        {
            let mut entries = self.entries.write();
            let was_absent = entries.insert(key.clone(), value).is_none();
            
            if was_absent {
                self.size.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Update min cache if this is the new minimum
        if is_new_min && self.config.enable_min_key_cache {
            *self.min_key_cache.write() = Some(key);
        }
        
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Peek at the minimum key without removing it
    ///
    /// Complexity: O(1) if cache hit, O(log N) if cache miss
    pub fn peek_min(&self) -> Option<(CompositeQueueKey, V)> {
        // Try cache first
        if self.config.enable_min_key_cache {
            let cache = self.min_key_cache.read();
            if let Some(ref cached_key) = *cache {
                let entries = self.entries.read();
                if let Some(value) = entries.get(cached_key) {
                    return Some((cached_key.clone(), value.clone()));
                }
            }
        }
        
        // Cache miss - scan
        let entries = self.entries.read();
        let result = entries.first_key_value()
            .map(|(k, v)| (k.clone(), v.clone()));
        
        // Update cache
        if self.config.enable_min_key_cache {
            if let Some((ref key, _)) = result {
                *self.min_key_cache.write() = Some(key.clone());
            }
        }
        
        result
    }

    /// Remove and return the minimum entry
    ///
    /// Complexity: O(log N)
    pub fn pop_min(&self) -> Option<(CompositeQueueKey, V)> {
        let result = {
            let mut entries = self.entries.write();
            entries.pop_first()
        };
        
        if result.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
            
            // Invalidate cache
            if self.config.enable_min_key_cache {
                *self.min_key_cache.write() = None;
            }
            
            self.version.fetch_add(1, Ordering::Release);
        }
        
        result
    }

    /// Remove a specific entry by key
    ///
    /// Complexity: O(log N)
    pub fn remove(&self, key: &CompositeQueueKey) -> Option<V> {
        let result = {
            let mut entries = self.entries.write();
            entries.remove(key)
        };
        
        if result.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
            
            // Invalidate cache if we removed the cached min
            if self.config.enable_min_key_cache {
                let should_invalidate = {
                    let cache = self.min_key_cache.read();
                    cache.as_ref().map(|c| c == key).unwrap_or(false)
                };
                if should_invalidate {
                    *self.min_key_cache.write() = None;
                }
            }
            
            self.version.fetch_add(1, Ordering::Release);
        }
        
        result
    }

    /// Get an entry by key
    ///
    /// Complexity: O(log N)
    pub fn get(&self, key: &CompositeQueueKey) -> Option<V> {
        self.entries.read().get(key).cloned()
    }

    /// Check if a key exists
    ///
    /// Complexity: O(log N)
    pub fn contains(&self, key: &CompositeQueueKey) -> bool {
        self.entries.read().contains_key(key)
    }

    /// Get queue size
    ///
    /// Complexity: O(1)
    pub fn len(&self) -> usize {
        if self.config.enable_size_tracking {
            self.size.load(Ordering::Relaxed)
        } else {
            self.entries.read().len()
        }
    }

    /// Check if queue is empty
    ///
    /// Complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get current version (for change detection)
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Scan entries with priority <= threshold
    ///
    /// Useful for batch processing of high-priority tasks.
    ///
    /// Complexity: O(log N + K) where K is result count
    pub fn scan_by_priority(&self, max_priority: i64, limit: usize) -> Vec<(CompositeQueueKey, V)> {
        let entries = self.entries.read();
        
        entries.iter()
            .take_while(|(k, _)| k.priority <= max_priority)
            .take(limit)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Scan entries ready at or before the given timestamp
    ///
    /// Complexity: O(N) in worst case, but typically O(K) if data is time-ordered
    pub fn scan_ready(&self, now: u64, limit: usize) -> Vec<(CompositeQueueKey, V)> {
        let entries = self.entries.read();
        
        entries.iter()
            .filter(|(k, _)| k.timestamp <= now)
            .take(limit)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get the configuration
    pub fn config(&self) -> &QueueIndexConfig {
        &self.config
    }
}

// ============================================================================
// QueueTableRegistry - Queue-Aware Table Registry
// ============================================================================

/// Registry extension for queue tables
pub struct QueueTableRegistry {
    /// Base registry
    base: TableIndexRegistry,
    /// Queue-specific configs
    queue_configs: RwLock<std::collections::HashMap<String, QueueIndexConfig>>,
}

impl QueueTableRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            base: TableIndexRegistry::with_default_policy(IndexPolicy::Balanced),
            queue_configs: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Register a table as a queue
    pub fn register_queue(&self, config: QueueIndexConfig) {
        // Register base config with ScanOptimized policy
        self.base.configure_table(config.base.clone());
        
        // Store queue-specific config
        self.queue_configs.write().insert(
            config.base.table_name.clone(),
            config,
        );
    }

    /// Check if a table is registered as a queue
    pub fn is_queue(&self, table_name: &str) -> bool {
        self.queue_configs.read().contains_key(table_name)
    }

    /// Get queue config
    pub fn get_queue_config(&self, table_name: &str) -> Option<QueueIndexConfig> {
        self.queue_configs.read().get(table_name).cloned()
    }

    /// Get the base registry
    pub fn base(&self) -> &TableIndexRegistry {
        &self.base
    }
}

impl Default for QueueTableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// QueueStats - Queue Statistics
// ============================================================================

/// Statistics for a queue index
#[derive(Debug, Clone, Default)]
pub struct QueueIndexStats {
    /// Current size
    pub size: usize,
    /// Number of inserts
    pub inserts: u64,
    /// Number of pops
    pub pops: u64,
    /// Number of peeks
    pub peeks: u64,
    /// Cache hit rate for peek operations
    pub cache_hit_rate: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_key_ordering() {
        let k1 = CompositeQueueKey::new(1, 100, 1, "task1");
        let k2 = CompositeQueueKey::new(2, 100, 1, "task2");
        let k3 = CompositeQueueKey::new(1, 200, 1, "task3");
        let k4 = CompositeQueueKey::new(1, 100, 2, "task4");
        
        // Lower priority comes first
        assert!(k1 < k2);
        
        // Same priority, earlier timestamp comes first
        assert!(k1 < k3);
        
        // Same priority and timestamp, lower sequence comes first
        assert!(k1 < k4);
    }

    #[test]
    fn test_composite_key_encode_decode() {
        let original = CompositeQueueKey::new(-100, 12345, 999, "my-task-id");
        let encoded = original.encode();
        let decoded = CompositeQueueKey::decode(&encoded).unwrap();
        
        assert_eq!(decoded.priority, original.priority);
        assert_eq!(decoded.timestamp, original.timestamp);
        assert_eq!(decoded.sequence, original.sequence);
        assert_eq!(decoded.task_id, original.task_id);
    }

    #[test]
    fn test_queue_index_insert_pop() {
        let config = QueueIndexConfig::new("test_queue");
        let index: QueueIndex<String> = QueueIndex::new(config);
        
        // Insert with different priorities
        index.insert(CompositeQueueKey::new(3, 100, 1, "low"), "low priority".to_string());
        index.insert(CompositeQueueKey::new(1, 100, 1, "high"), "high priority".to_string());
        index.insert(CompositeQueueKey::new(2, 100, 1, "medium"), "medium priority".to_string());
        
        assert_eq!(index.len(), 3);
        
        // Pop should return highest priority (lowest number) first
        let (key, value) = index.pop_min().unwrap();
        assert_eq!(key.priority, 1);
        assert_eq!(value, "high priority");
        
        let (key, _) = index.pop_min().unwrap();
        assert_eq!(key.priority, 2);
        
        let (key, _) = index.pop_min().unwrap();
        assert_eq!(key.priority, 3);
        
        assert!(index.is_empty());
    }

    #[test]
    fn test_queue_index_peek() {
        let config = QueueIndexConfig::new("test_queue");
        let index: QueueIndex<i32> = QueueIndex::new(config);
        
        index.insert(CompositeQueueKey::new(2, 100, 1, "task1"), 1);
        index.insert(CompositeQueueKey::new(1, 100, 1, "task2"), 2);
        
        // Peek should return min without removing
        let (key, value) = index.peek_min().unwrap();
        assert_eq!(key.priority, 1);
        assert_eq!(value, 2);
        
        // Should still have 2 items
        assert_eq!(index.len(), 2);
        
        // Peek again (should hit cache)
        let (key, _) = index.peek_min().unwrap();
        assert_eq!(key.priority, 1);
    }

    #[test]
    fn test_queue_index_remove() {
        let config = QueueIndexConfig::new("test_queue");
        let index: QueueIndex<i32> = QueueIndex::new(config);
        
        let key1 = CompositeQueueKey::new(1, 100, 1, "task1");
        let key2 = CompositeQueueKey::new(2, 100, 1, "task2");
        
        index.insert(key1.clone(), 1);
        index.insert(key2.clone(), 2);
        
        // Remove by key
        let removed = index.remove(&key1);
        assert_eq!(removed, Some(1));
        assert_eq!(index.len(), 1);
        
        // Pop should return remaining item
        let (key, _) = index.pop_min().unwrap();
        assert_eq!(key.task_id, "task2");
    }

    #[test]
    fn test_scan_by_priority() {
        let config = QueueIndexConfig::new("test_queue");
        let index: QueueIndex<i32> = QueueIndex::new(config);
        
        for i in 1..=10 {
            index.insert(CompositeQueueKey::new(i, 100, 1, format!("task{}", i)), i as i32);
        }
        
        // Scan priority <= 3
        let results = index.scan_by_priority(3, 100);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.priority, 1);
        assert_eq!(results[1].0.priority, 2);
        assert_eq!(results[2].0.priority, 3);
    }

    #[test]
    fn test_scan_ready() {
        let config = QueueIndexConfig::new("test_queue");
        let index: QueueIndex<i32> = QueueIndex::new(config);
        
        // Insert tasks with different ready times
        index.insert(CompositeQueueKey::new(1, 100, 1, "ready1"), 1);
        index.insert(CompositeQueueKey::new(1, 200, 1, "ready2"), 2);
        index.insert(CompositeQueueKey::new(1, 300, 1, "future"), 3);
        
        // Scan ready at timestamp 200
        let results = index.scan_ready(200, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_queue_registry() {
        let registry = QueueTableRegistry::new();
        
        let queue_config = QueueIndexConfig::new("task_queue")
            .with_priority_column("priority")
            .with_timestamp_column("ready_at");
        
        registry.register_queue(queue_config);
        
        assert!(registry.is_queue("task_queue"));
        assert!(!registry.is_queue("regular_table"));
        
        let config = registry.get_queue_config("task_queue").unwrap();
        assert_eq!(config.priority_column, Some("priority".to_string()));
    }

    #[test]
    fn test_fifo_within_priority() {
        let config = QueueIndexConfig::new("test_queue");
        let index: QueueIndex<String> = QueueIndex::new(config);
        
        // Insert tasks with same priority, different sequence
        index.insert(CompositeQueueKey::new(1, 100, 3, "third"), "third".to_string());
        index.insert(CompositeQueueKey::new(1, 100, 1, "first"), "first".to_string());
        index.insert(CompositeQueueKey::new(1, 100, 2, "second"), "second".to_string());
        
        // Should pop in sequence order (FIFO)
        let (_, v1) = index.pop_min().unwrap();
        let (_, v2) = index.pop_min().unwrap();
        let (_, v3) = index.pop_min().unwrap();
        
        assert_eq!(v1, "first");
        assert_eq!(v2, "second");
        assert_eq!(v3, "third");
    }
}
