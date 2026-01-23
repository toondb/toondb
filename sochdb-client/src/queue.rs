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

//! First-Class Queue API with Ordered-Key Task Entries
//!
//! This module replaces the anti-pattern of storing queues as blobbed JSON with
//! a proper ordered-key representation that leverages SochDB's strengths.
//!
//! ## Problem: Blobbed JSON Queue Anti-Pattern
//!
//! ```text
//! ❌ Current: Single KV value holding JSON list
//!    queue:tasks → [{"id":1,"priority":5,...}, {"id":2,"priority":1,...}, ...]
//!    
//!    Dequeue: Read → Parse → Pop → Serialize → Write
//!    Complexity: O(N) parse + O(N) rewrite per operation
//!    Issues: Cache-hostile, contention on single key, WAL amplification
//! ```
//!
//! ## Solution: Ordered-Key Task Entries
//!
//! ```text
//! ✅ New: Each task as its own record with priority-encoded key
//!    queue/<queue_id>/<priority_be>/<ready_ts_be>/<seq_be>/<task_id> → payload
//!    
//!    Dequeue: Range/prefix scan → take first key → delete
//!    Complexity: O(log N) to locate + O(1) delete
//!    Benefits: Localized KV ops, better cache locality, lower WAL writes
//! ```
//!
//! ## Key Encoding
//!
//! Keys use big-endian fixed-width encoding so lexicographic order matches
//! numeric order. This allows the "head" of the queue to be the smallest key.
//!
//! ```text
//! Key Layout (bytes):
//! ┌────────────────┬─────────────┬──────────────┬─────────────┬───────────┐
//! │ queue/<q_id>/  │ priority_be │ ready_ts_be  │ seq_be      │ task_id   │
//! │ (prefix)       │ (8 bytes)   │ (8 bytes)    │ (8 bytes)   │ (uuid)    │
//! └────────────────┴─────────────┴──────────────┴─────────────┴───────────┘
//! ```
//!
//! ## Atomic Claim Protocol
//!
//! To prevent double-delivery under concurrency, we use a claim-based approach:
//!
//! 1. Worker scans for minimal candidate key
//! 2. Worker attempts CAS: `queue_claim/<task_hash>` → `{owner, lease_expiry}`
//! 3. Only one worker wins; losers retry with next candidate
//! 4. Winner moves task to `inflight/` namespace or deletes original
//! 5. Lease expiry allows recovery from worker crashes
//!
//! ## Visibility Timeout
//!
//! Tasks remain "invisible" while being processed. If the worker crashes or
//! doesn't ACK within the timeout, the task becomes visible again.
//!
//! ```text
//! Task Lifecycle:
//! ┌──────────┐     ┌───────────┐     ┌────────────┐     ┌─────────┐
//! │ PENDING  │────▶│ CLAIMED   │────▶│ PROCESSING │────▶│ DONE    │
//! │ (ready)  │     │ (inflight)│     │ (worker)   │     │ (acked) │
//! └──────────┘     └───────────┘     └────────────┘     └─────────┘
//!       ▲                │                  │
//!       │                │ lease expired    │ nack
//!       └────────────────┴──────────────────┘
//! ```

use std::collections::{BinaryHeap, HashMap};
use std::cmp::{Ordering, Reverse};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};

// ============================================================================
// Key Encoding - Big-Endian for Lexicographic Ordering
// ============================================================================

/// Encode a u64 as big-endian bytes for lexicographic ordering
/// 
/// Big-endian encoding ensures that lexicographic byte comparison
/// matches numeric comparison: smaller numbers come first.
#[inline]
pub fn encode_u64_be(value: u64) -> [u8; 8] {
    value.to_be_bytes()
}

/// Decode a big-endian u64 from bytes
#[inline]
pub fn decode_u64_be(bytes: &[u8]) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[..8]);
    u64::from_be_bytes(arr)
}

/// Encode a priority with inversion for min-heap behavior
/// 
/// For ascending order (lower priority = higher urgency),
/// we store `u64::MAX - priority` so smaller values come first.
#[inline]
pub fn encode_priority(priority: i64, ascending: bool) -> [u8; 8] {
    if ascending {
        // For ASC: lower priority values should come first
        // Map i64 to u64 in a way that preserves order
        let mapped = (priority as i128 + i64::MAX as i128 + 1) as u64;
        encode_u64_be(mapped)
    } else {
        // For DESC: higher priority values should come first
        // Invert the mapping
        let mapped = (i64::MAX as i128 - priority as i128) as u64;
        encode_u64_be(mapped)
    }
}

/// Decode a priority from encoded bytes
#[inline]
pub fn decode_priority(bytes: &[u8], ascending: bool) -> i64 {
    let mapped = decode_u64_be(bytes);
    if ascending {
        (mapped as i128 - i64::MAX as i128 - 1) as i64
    } else {
        (i64::MAX as i128 - mapped as i128) as i64
    }
}

// ============================================================================
// QueueKey - Composite Key for Queue Entries
// ============================================================================

/// Composite key for queue entries
/// 
/// Layout ensures lexicographic order matches desired queue order:
/// 1. Queue ID (namespace separation)
/// 2. Priority (big-endian, optionally inverted)
/// 3. Ready timestamp (when task becomes visible)
/// 4. Sequence number (tie-breaker for FIFO within same priority)
/// 5. Task ID (unique identifier)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueueKey {
    /// Queue identifier
    pub queue_id: String,
    /// Task priority (lower = more urgent for ASC ordering)
    pub priority: i64,
    /// Timestamp when task becomes ready (for delayed tasks)
    pub ready_ts: u64,
    /// Sequence number for FIFO ordering within same priority
    pub sequence: u64,
    /// Unique task identifier
    pub task_id: String,
}

impl QueueKey {
    /// Create a new queue key
    pub fn new(
        queue_id: impl Into<String>,
        priority: i64,
        ready_ts: u64,
        sequence: u64,
        task_id: impl Into<String>,
    ) -> Self {
        Self {
            queue_id: queue_id.into(),
            priority,
            ready_ts,
            sequence,
            task_id: task_id.into(),
        }
    }

    /// Encode key to bytes for storage
    /// 
    /// Format: `queue/<queue_id>/<priority_be>/<ready_ts_be>/<seq_be>/<task_id>`
    pub fn encode(&self, ascending_priority: bool) -> Vec<u8> {
        let mut key = Vec::with_capacity(64);
        
        // Prefix
        key.extend_from_slice(b"queue/");
        key.extend_from_slice(self.queue_id.as_bytes());
        key.push(b'/');
        
        // Priority (big-endian encoded)
        key.extend_from_slice(&encode_priority(self.priority, ascending_priority));
        key.push(b'/');
        
        // Ready timestamp (big-endian)
        key.extend_from_slice(&encode_u64_be(self.ready_ts));
        key.push(b'/');
        
        // Sequence number (big-endian)
        key.extend_from_slice(&encode_u64_be(self.sequence));
        key.push(b'/');
        
        // Task ID
        key.extend_from_slice(self.task_id.as_bytes());
        
        key
    }

    /// Generate prefix for scanning all tasks in a queue
    pub fn queue_prefix(queue_id: &str) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(32);
        prefix.extend_from_slice(b"queue/");
        prefix.extend_from_slice(queue_id.as_bytes());
        prefix.push(b'/');
        prefix
    }
}

impl Ord for QueueKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // First by queue_id
        self.queue_id.cmp(&other.queue_id)
            // Then by priority (ascending: lower = first)
            .then(self.priority.cmp(&other.priority))
            // Then by ready_ts
            .then(self.ready_ts.cmp(&other.ready_ts))
            // Then by sequence
            .then(self.sequence.cmp(&other.sequence))
            // Finally by task_id for uniqueness
            .then(self.task_id.cmp(&other.task_id))
    }
}

impl PartialOrd for QueueKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ============================================================================
// Task - Queue Task with Payload
// ============================================================================

/// Queue task state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is pending (ready to be dequeued)
    Pending,
    /// Task is claimed by a worker (inflight)
    Claimed,
    /// Task is completed (will be deleted)
    Completed,
    /// Task failed and is dead-lettered
    DeadLettered,
}

/// A task in the queue
#[derive(Debug, Clone)]
pub struct Task {
    /// Task key (determines ordering)
    pub key: QueueKey,
    /// Task payload (arbitrary bytes)
    pub payload: Vec<u8>,
    /// Task state
    pub state: TaskState,
    /// Number of delivery attempts
    pub attempts: u32,
    /// Maximum delivery attempts before dead-lettering
    pub max_attempts: u32,
    /// Created timestamp (epoch millis)
    pub created_at: u64,
    /// Last claimed timestamp (epoch millis)
    pub claimed_at: Option<u64>,
    /// Claim owner (worker ID)
    pub claimed_by: Option<String>,
    /// Lease expiry timestamp (epoch millis)
    pub lease_expires_at: Option<u64>,
}

impl Task {
    /// Create a new pending task
    pub fn new(key: QueueKey, payload: Vec<u8>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Self {
            key,
            payload,
            state: TaskState::Pending,
            attempts: 0,
            max_attempts: 3,
            created_at: now,
            claimed_at: None,
            claimed_by: None,
            lease_expires_at: None,
        }
    }

    /// Create with custom max attempts
    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Check if the task is visible (ready to be claimed)
    pub fn is_visible(&self, now_millis: u64) -> bool {
        match self.state {
            TaskState::Pending => self.key.ready_ts <= now_millis,
            TaskState::Claimed => {
                // Visible again if lease expired
                self.lease_expires_at
                    .map(|exp| now_millis >= exp)
                    .unwrap_or(false)
            }
            TaskState::Completed | TaskState::DeadLettered => false,
        }
    }

    /// Check if the task should be dead-lettered
    pub fn should_dead_letter(&self) -> bool {
        self.attempts >= self.max_attempts
    }
}

// ============================================================================
// Claim - Lease-Based Ownership
// ============================================================================

/// A claim on a task (lease-based ownership)
#[derive(Debug, Clone)]
pub struct Claim {
    /// Task ID being claimed
    pub task_id: String,
    /// Worker claiming the task
    pub owner: String,
    /// When the claim was created
    pub claimed_at: u64,
    /// When the claim expires
    pub expires_at: u64,
}

impl Claim {
    /// Create a new claim
    pub fn new(task_id: impl Into<String>, owner: impl Into<String>, lease_ms: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Self {
            task_id: task_id.into(),
            owner: owner.into(),
            claimed_at: now,
            expires_at: now + lease_ms,
        }
    }

    /// Check if the claim has expired
    pub fn is_expired(&self, now_millis: u64) -> bool {
        now_millis >= self.expires_at
    }

    /// Encode claim key for storage
    pub fn encode_key(queue_id: &str, task_id: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(b"queue_claim/");
        key.extend_from_slice(queue_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(task_id.as_bytes());
        key
    }
}

// ============================================================================
// DequeueResult - Result of a Dequeue Operation
// ============================================================================

/// Result of a dequeue operation
#[derive(Debug)]
pub enum DequeueResult {
    /// Successfully claimed a task
    Success(Task),
    /// Queue is empty (no visible tasks)
    Empty,
    /// Contention - all visible tasks were claimed by other workers
    /// Contains the number of tasks that were attempted
    Contention(usize),
    /// Error during dequeue
    Error(String),
}

// ============================================================================
// QueueConfig - Queue Configuration
// ============================================================================

/// Queue configuration
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Queue identifier
    pub queue_id: String,
    /// Default visibility timeout in milliseconds
    pub default_visibility_timeout_ms: u64,
    /// Maximum attempts before dead-lettering
    pub max_attempts: u32,
    /// Whether to use ascending priority (lower = more urgent)
    pub ascending_priority: bool,
    /// Dead letter queue ID (if any)
    pub dead_letter_queue_id: Option<String>,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            queue_id: "default".to_string(),
            default_visibility_timeout_ms: 30_000, // 30 seconds
            max_attempts: 3,
            ascending_priority: true,
            dead_letter_queue_id: None,
        }
    }
}

impl QueueConfig {
    /// Create a new queue config
    pub fn new(queue_id: impl Into<String>) -> Self {
        Self {
            queue_id: queue_id.into(),
            ..Default::default()
        }
    }

    /// Builder: set visibility timeout
    pub fn with_visibility_timeout(mut self, timeout_ms: u64) -> Self {
        self.default_visibility_timeout_ms = timeout_ms;
        self
    }

    /// Builder: set max attempts
    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Builder: set priority ordering
    pub fn with_ascending_priority(mut self, ascending: bool) -> Self {
        self.ascending_priority = ascending;
        self
    }

    /// Builder: set dead letter queue
    pub fn with_dead_letter_queue(mut self, dlq_id: impl Into<String>) -> Self {
        self.dead_letter_queue_id = Some(dlq_id.into());
        self
    }
}

// ============================================================================
// PriorityQueue - In-Memory Implementation for Reference
// ============================================================================

/// In-memory priority queue implementation
/// 
/// This serves as a reference implementation and for testing.
/// Production use should integrate with SochDB's storage layer.
pub struct PriorityQueue {
    /// Queue configuration
    config: QueueConfig,
    /// Tasks indexed by key (ordered storage)
    tasks: RwLock<std::collections::BTreeMap<QueueKey, Task>>,
    /// Active claims (task_id -> claim)
    claims: RwLock<HashMap<String, Claim>>,
    /// Sequence counter for FIFO ordering
    sequence: AtomicU64,
}

impl PriorityQueue {
    /// Create a new priority queue
    pub fn new(config: QueueConfig) -> Self {
        Self {
            config,
            tasks: RwLock::new(std::collections::BTreeMap::new()),
            claims: RwLock::new(HashMap::new()),
            sequence: AtomicU64::new(0),
        }
    }

    /// Get the queue ID
    pub fn queue_id(&self) -> &str {
        &self.config.queue_id
    }

    /// Enqueue a task with the given priority and payload
    /// 
    /// Complexity: O(log N) insertion into ordered structure
    pub fn enqueue(&self, priority: i64, payload: Vec<u8>) -> Task {
        self.enqueue_delayed(priority, payload, 0)
    }

    /// Enqueue a task with a delay before it becomes visible
    /// 
    /// Complexity: O(log N)
    pub fn enqueue_delayed(&self, priority: i64, payload: Vec<u8>, delay_ms: u64) -> Task {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        let sequence = self.sequence.fetch_add(1, AtomicOrdering::SeqCst);
        let task_id = format!("{:016x}{:016x}", now, sequence);
        
        let key = QueueKey::new(
            &self.config.queue_id,
            priority,
            now + delay_ms,
            sequence,
            task_id,
        );
        
        let task = Task::new(key.clone(), payload)
            .with_max_attempts(self.config.max_attempts);
        
        self.tasks.write().insert(key, task.clone());
        
        task
    }

    /// Dequeue the next visible task
    /// 
    /// This implements the atomic claim protocol:
    /// 1. Find first visible task
    /// 2. Attempt to claim it (CAS semantics)
    /// 3. If claimed, return task; otherwise retry with next candidate
    /// 
    /// Complexity: O(log N) to find first visible + O(1) claim
    pub fn dequeue(&self, worker_id: impl Into<String>) -> DequeueResult {
        self.dequeue_with_timeout(worker_id, self.config.default_visibility_timeout_ms)
    }

    /// Dequeue with custom visibility timeout
    pub fn dequeue_with_timeout(
        &self,
        worker_id: impl Into<String>,
        visibility_timeout_ms: u64,
    ) -> DequeueResult {
        let worker = worker_id.into();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        // Clean up expired claims first
        self.cleanup_expired_claims(now);
        
        let mut tasks = self.tasks.write();
        let mut claims = self.claims.write();
        let mut contention_count = 0;
        
        // Find first visible, unclaimed task
        for (key, task) in tasks.iter_mut() {
            if !task.is_visible(now) {
                continue;
            }
            
            // Check if already claimed (by another worker whose claim hasn't expired)
            if let Some(claim) = claims.get(&key.task_id) {
                if !claim.is_expired(now) && claim.owner != worker {
                    contention_count += 1;
                    continue;
                }
            }
            
            // Attempt to claim
            let claim = Claim::new(&key.task_id, &worker, visibility_timeout_ms);
            
            // Update task state
            task.state = TaskState::Claimed;
            task.attempts += 1;
            task.claimed_at = Some(now);
            task.claimed_by = Some(worker.clone());
            task.lease_expires_at = Some(claim.expires_at);
            
            // Store claim
            claims.insert(key.task_id.clone(), claim);
            
            return DequeueResult::Success(task.clone());
        }
        
        if contention_count > 0 {
            DequeueResult::Contention(contention_count)
        } else {
            DequeueResult::Empty
        }
    }

    /// Acknowledge successful processing of a task (delete it)
    /// 
    /// Complexity: O(log N)
    pub fn ack(&self, task_id: &str) -> Result<(), String> {
        let mut tasks = self.tasks.write();
        let mut claims = self.claims.write();
        
        // Find and remove the task
        let key = tasks.iter()
            .find(|(_, t)| t.key.task_id == task_id)
            .map(|(k, _)| k.clone());
        
        if let Some(key) = key {
            tasks.remove(&key);
            claims.remove(task_id);
            Ok(())
        } else {
            Err(format!("Task not found: {}", task_id))
        }
    }

    /// Negative acknowledgment - return task to queue
    /// 
    /// Optionally adjust priority or add delay for retry
    /// 
    /// Complexity: O(log N) for reposition
    pub fn nack(&self, task_id: &str, new_priority: Option<i64>, delay_ms: Option<u64>) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        let mut tasks = self.tasks.write();
        let mut claims = self.claims.write();
        
        // Find the task
        let entry = tasks.iter()
            .find(|(_, t)| t.key.task_id == task_id)
            .map(|(k, t)| (k.clone(), t.clone()));
        
        if let Some((old_key, mut task)) = entry {
            // Check if should dead-letter
            if task.should_dead_letter() {
                task.state = TaskState::DeadLettered;
                // In real implementation, move to DLQ
                tasks.remove(&old_key);
                claims.remove(task_id);
                return Err(format!("Task dead-lettered after {} attempts", task.attempts));
            }
            
            // Create new key with updated priority/ready_ts
            let new_priority = new_priority.unwrap_or(task.key.priority);
            let new_ready_ts = delay_ms.map(|d| now + d).unwrap_or(now);
            let new_sequence = self.sequence.fetch_add(1, AtomicOrdering::SeqCst);
            
            let new_key = QueueKey::new(
                &self.config.queue_id,
                new_priority,
                new_ready_ts,
                new_sequence,
                &task.key.task_id,
            );
            
            // Update task state
            task.key = new_key.clone();
            task.state = TaskState::Pending;
            task.claimed_at = None;
            task.claimed_by = None;
            task.lease_expires_at = None;
            
            // Remove old entry, insert new
            tasks.remove(&old_key);
            tasks.insert(new_key, task);
            claims.remove(task_id);
            
            Ok(())
        } else {
            Err(format!("Task not found: {}", task_id))
        }
    }

    /// Extend the visibility timeout for a task
    /// 
    /// Useful when processing takes longer than expected
    pub fn extend_visibility(&self, task_id: &str, additional_ms: u64) -> Result<(), String> {
        let mut claims = self.claims.write();
        let mut tasks = self.tasks.write();
        
        if let Some(claim) = claims.get_mut(task_id) {
            claim.expires_at += additional_ms;
            
            // Also update the task's lease
            let entry = tasks.iter_mut()
                .find(|(_, t)| t.key.task_id == task_id);
            
            if let Some((_, task)) = entry {
                task.lease_expires_at = Some(claim.expires_at);
            }
            
            Ok(())
        } else {
            Err(format!("No active claim for task: {}", task_id))
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        let tasks = self.tasks.read();
        let claims = self.claims.read();
        
        let mut pending = 0;
        let mut delayed = 0;
        let mut inflight = 0;
        
        for task in tasks.values() {
            match task.state {
                TaskState::Pending => {
                    if task.key.ready_ts > now {
                        delayed += 1;
                    } else {
                        pending += 1;
                    }
                }
                TaskState::Claimed => {
                    if task.lease_expires_at.map(|exp| now < exp).unwrap_or(false) {
                        inflight += 1;
                    } else {
                        pending += 1; // Lease expired, counts as pending
                    }
                }
                _ => {}
            }
        }
        
        QueueStats {
            queue_id: self.config.queue_id.clone(),
            pending,
            delayed,
            inflight,
            total: tasks.len(),
            active_claims: claims.len(),
        }
    }

    /// Clean up expired claims (return tasks to pending state)
    fn cleanup_expired_claims(&self, now_millis: u64) {
        let expired: Vec<_> = {
            let claims = self.claims.read();
            claims.iter()
                .filter(|(_, c)| c.is_expired(now_millis))
                .map(|(id, _)| id.clone())
                .collect()
        };
        
        let mut tasks = self.tasks.write();
        let mut claims = self.claims.write();
        
        for task_id in expired {
            claims.remove(&task_id);
            
            // Find and reset the task
            for (_, task) in tasks.iter_mut() {
                if task.key.task_id == task_id && task.state == TaskState::Claimed {
                    task.state = TaskState::Pending;
                    task.claimed_at = None;
                    task.claimed_by = None;
                    task.lease_expires_at = None;
                    break;
                }
            }
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Queue identifier
    pub queue_id: String,
    /// Number of pending (visible) tasks
    pub pending: usize,
    /// Number of delayed tasks
    pub delayed: usize,
    /// Number of tasks currently being processed
    pub inflight: usize,
    /// Total number of tasks in queue
    pub total: usize,
    /// Number of active claims
    pub active_claims: usize,
}

// ============================================================================
// Streaming Top-K for ORDER BY ... LIMIT Optimization
// ============================================================================

/// Streaming Top-K collector using a bounded heap
/// 
/// This implements correct ORDER BY ... LIMIT K semantics without
/// requiring O(N) memory or O(N log N) full sort.
/// 
/// ## Complexity
/// 
/// - Space: O(K)
/// - Time: O(N log K) for N insertions
/// - For K=1: O(N) comparisons with O(1) memory
/// 
/// ## Algorithm
/// 
/// For ascending order (smallest K elements):
/// - Maintain a max-heap of size K
/// - For each element, if heap is full and element < heap.max, replace max
/// - At end, drain heap in sorted order
/// 
/// For descending order (largest K elements):
/// - Maintain a min-heap of size K
/// - For each element, if heap is full and element > heap.min, replace min
pub struct StreamingTopK<T> {
    /// Bounded heap (max-heap for ASC, min-heap for DESC)
    heap: BinaryHeap<HeapEntry<T>>,
    /// Maximum size (K)
    k: usize,
    /// Whether we want ascending order (smallest K)
    ascending: bool,
}

/// Heap entry wrapper for ordering control
struct HeapEntry<T> {
    value: T,
    /// If true, use natural ordering (max-heap behavior)
    /// If false, use reversed ordering (min-heap behavior)
    natural_order: bool,
}

impl<T: Ord> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Ord> Eq for HeapEntry<T> {}

impl<T: Ord> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.natural_order {
            self.value.cmp(&other.value)
        } else {
            other.value.cmp(&self.value)
        }
    }
}

impl<T: Ord + Clone> StreamingTopK<T> {
    /// Create a new streaming top-K collector
    /// 
    /// - `k`: Number of elements to keep
    /// - `ascending`: If true, keep smallest K; if false, keep largest K
    pub fn new(k: usize, ascending: bool) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            ascending,
        }
    }

    /// Push a value into the collector
    /// 
    /// Complexity: O(log K)
    pub fn push(&mut self, value: T) {
        if self.k == 0 {
            return;
        }

        let entry = HeapEntry {
            value,
            // For ascending (smallest K), we want max-heap behavior
            // so we can efficiently evict the largest
            natural_order: self.ascending,
        };

        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else {
            // Check if new value should replace the current extreme
            if let Some(top) = self.heap.peek() {
                let should_replace = if self.ascending {
                    // For smallest K: replace if new < current max
                    entry.value < top.value
                } else {
                    // For largest K: replace if new > current min
                    entry.value > top.value
                };

                if should_replace {
                    self.heap.pop();
                    self.heap.push(entry);
                }
            }
        }
    }

    /// Get the current threshold (for early termination with sorted input)
    /// 
    /// Returns the value that new elements must beat to be included.
    pub fn threshold(&self) -> Option<&T> {
        self.heap.peek().map(|e| &e.value)
    }

    /// Check if the heap is at capacity
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Drain the collector into a sorted vector
    /// 
    /// Complexity: O(K log K)
    pub fn into_sorted_vec(self) -> Vec<T> {
        let mut values: Vec<_> = self.heap.into_iter().map(|e| e.value).collect();
        
        if self.ascending {
            values.sort();
        } else {
            values.sort_by(|a, b| b.cmp(a));
        }
        
        values
    }

    /// Get the number of elements currently held
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

/// Multi-column top-K with custom comparator
/// 
/// Supports ORDER BY col1 ASC, col2 DESC LIMIT K semantics.
pub struct MultiColumnTopK<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    /// Bounded heap
    heap: Vec<T>,
    /// Maximum size (K)
    k: usize,
    /// Custom comparator
    comparator: F,
}

impl<T: Clone, F: Fn(&T, &T) -> Ordering> MultiColumnTopK<T, F> {
    /// Create with custom comparator
    /// 
    /// The comparator should return `Ordering::Less` if the first argument
    /// should come before the second in the final result.
    pub fn new(k: usize, comparator: F) -> Self {
        Self {
            heap: Vec::with_capacity(k + 1),
            k,
            comparator,
        }
    }

    /// Push a value
    /// 
    /// Complexity: O(K) in worst case (linear scan), but typically O(1) for
    /// random input after heap is full. For truly O(log K), use a proper
    /// heap with custom comparator.
    pub fn push(&mut self, value: T) {
        if self.k == 0 {
            return;
        }

        self.heap.push(value);
        
        if self.heap.len() > self.k {
            // Find and remove the worst element
            let mut worst_idx = 0;
            for i in 1..self.heap.len() {
                if (self.comparator)(&self.heap[i], &self.heap[worst_idx]) == Ordering::Greater {
                    worst_idx = i;
                }
            }
            self.heap.swap_remove(worst_idx);
        }
    }

    /// Drain into sorted vector
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        self.heap.sort_by(&self.comparator);
        self.heap
    }

    /// Current length
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ============================================================================
// OrderByLimitStrategy - Query Execution Strategy Selection
// ============================================================================

/// Strategy for executing ORDER BY ... LIMIT queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderByLimitStrategy {
    /// Use index pushdown (ordered scan from storage)
    /// 
    /// Complexity: O(log N) to position + O(K) to retrieve
    /// Requires: Matching ordered index on ORDER BY column(s)
    IndexPushdown,
    
    /// Use streaming top-K heap
    /// 
    /// Complexity: O(N log K) time, O(K) space
    /// Use when: No matching index, K << N
    StreamingTopK,
    
    /// Full sort then limit
    /// 
    /// Complexity: O(N log N) time, O(N) space
    /// Use when: No index, K ≈ N, or complex ORDER BY
    FullSort,
}

impl OrderByLimitStrategy {
    /// Choose the best strategy based on available information
    pub fn choose(
        has_matching_index: bool,
        estimated_rows: usize,
        limit: usize,
    ) -> Self {
        if has_matching_index {
            return OrderByLimitStrategy::IndexPushdown;
        }

        // Heuristic: use streaming top-K if K < sqrt(N) or K < 1000
        let use_streaming = limit < 1000 || (limit as f64) < (estimated_rows as f64).sqrt();
        
        if use_streaming {
            OrderByLimitStrategy::StreamingTopK
        } else {
            OrderByLimitStrategy::FullSort
        }
    }

    /// Get description for explain plan
    pub fn description(&self) -> &'static str {
        match self {
            OrderByLimitStrategy::IndexPushdown => 
                "Index Pushdown: O(log N + K) using ordered index",
            OrderByLimitStrategy::StreamingTopK => 
                "Streaming Top-K: O(N log K) time, O(K) space",
            OrderByLimitStrategy::FullSort => 
                "Full Sort: O(N log N) time, O(N) space",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_encoding() {
        let key1 = QueueKey::new("tasks", 1, 1000, 1, "task-1");
        let key2 = QueueKey::new("tasks", 2, 1000, 1, "task-2");
        let key3 = QueueKey::new("tasks", 1, 2000, 1, "task-3");

        let encoded1 = key1.encode(true);
        let encoded2 = key2.encode(true);
        let encoded3 = key3.encode(true);

        // Lower priority should come first (ascending)
        assert!(encoded1 < encoded2, "Priority 1 should come before priority 2");
        
        // Same priority, earlier ready_ts comes first
        assert!(encoded1 < encoded3, "Earlier ready_ts should come first");
    }

    #[test]
    fn test_priority_encoding() {
        // Test ascending order (lower priority = more urgent)
        let p1 = encode_priority(-100, true);
        let p2 = encode_priority(0, true);
        let p3 = encode_priority(100, true);

        assert!(p1 < p2, "Negative priority should come first in ascending");
        assert!(p2 < p3, "Zero should come before positive in ascending");

        // Test descending order
        let d1 = encode_priority(-100, false);
        let d2 = encode_priority(0, false);
        let d3 = encode_priority(100, false);

        assert!(d3 < d2, "Higher priority should come first in descending");
        assert!(d2 < d1, "Zero should come before negative in descending");
    }

    #[test]
    fn test_enqueue_dequeue() {
        let config = QueueConfig::new("test-queue");
        let queue = PriorityQueue::new(config);

        // Enqueue tasks with different priorities
        queue.enqueue(3, b"low priority".to_vec());
        queue.enqueue(1, b"high priority".to_vec());
        queue.enqueue(2, b"medium priority".to_vec());

        // Dequeue should return highest priority first (priority 1)
        match queue.dequeue("worker-1") {
            DequeueResult::Success(task) => {
                assert_eq!(task.key.priority, 1);
                assert_eq!(task.payload, b"high priority");
            }
            _ => panic!("Expected success"),
        }

        // Next should be priority 2
        match queue.dequeue("worker-1") {
            DequeueResult::Success(task) => {
                assert_eq!(task.key.priority, 2);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_ack_removes_task() {
        let config = QueueConfig::new("test-queue");
        let queue = PriorityQueue::new(config);

        queue.enqueue(1, b"task 1".to_vec());
        
        let task = match queue.dequeue("worker-1") {
            DequeueResult::Success(t) => t,
            _ => panic!("Expected success"),
        };

        assert!(queue.ack(&task.key.task_id).is_ok());

        // Queue should now be empty
        match queue.dequeue("worker-1") {
            DequeueResult::Empty => {}
            _ => panic!("Expected empty"),
        }
    }

    #[test]
    fn test_nack_returns_task() {
        let config = QueueConfig::new("test-queue")
            .with_visibility_timeout(100); // Short timeout for test
        let queue = PriorityQueue::new(config);

        queue.enqueue(1, b"task 1".to_vec());
        
        let task = match queue.dequeue("worker-1") {
            DequeueResult::Success(t) => t,
            _ => panic!("Expected success"),
        };

        // Nack with higher priority
        assert!(queue.nack(&task.key.task_id, Some(0), None).is_ok());

        // Task should be available again
        match queue.dequeue("worker-2") {
            DequeueResult::Success(t) => {
                assert_eq!(t.key.priority, 0);
                assert_eq!(t.attempts, 2); // Second attempt
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_streaming_topk_ascending() {
        let mut topk = StreamingTopK::new(3, true);
        
        for i in [5, 2, 8, 1, 9, 3, 7, 4, 6] {
            topk.push(i);
        }

        let result = topk.into_sorted_vec();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_streaming_topk_descending() {
        let mut topk = StreamingTopK::new(3, false);
        
        for i in [5, 2, 8, 1, 9, 3, 7, 4, 6] {
            topk.push(i);
        }

        let result = topk.into_sorted_vec();
        assert_eq!(result, vec![9, 8, 7]);
    }

    #[test]
    fn test_streaming_topk_k1() {
        // Special case: finding minimum
        let mut topk = StreamingTopK::new(1, true);
        
        for i in [5, 2, 8, 1, 9, 3] {
            topk.push(i);
        }

        let result = topk.into_sorted_vec();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_multi_column_topk() {
        // ORDER BY priority ASC, created_at DESC LIMIT 3
        #[derive(Clone, Debug, PartialEq)]
        struct Task {
            priority: i64,
            created_at: u64,
            id: String,
        }

        let comparator = |a: &Task, b: &Task| {
            match a.priority.cmp(&b.priority) {
                Ordering::Equal => b.created_at.cmp(&a.created_at), // DESC
                other => other, // ASC
            }
        };

        let mut topk = MultiColumnTopK::new(3, comparator);

        topk.push(Task { priority: 1, created_at: 100, id: "a".into() });
        topk.push(Task { priority: 2, created_at: 200, id: "b".into() });
        topk.push(Task { priority: 1, created_at: 200, id: "c".into() }); // Same priority, later
        topk.push(Task { priority: 1, created_at: 150, id: "d".into() });
        topk.push(Task { priority: 3, created_at: 100, id: "e".into() });

        let result = topk.into_sorted_vec();
        
        // Should be: priority 1 tasks (ordered by created_at DESC), then priority 2
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, "c"); // priority=1, created_at=200
        assert_eq!(result[1].id, "d"); // priority=1, created_at=150
        assert_eq!(result[2].id, "a"); // priority=1, created_at=100
    }

    #[test]
    fn test_strategy_selection() {
        // With matching index, always use pushdown
        assert_eq!(
            OrderByLimitStrategy::choose(true, 1_000_000, 10),
            OrderByLimitStrategy::IndexPushdown
        );

        // Small K without index → streaming
        assert_eq!(
            OrderByLimitStrategy::choose(false, 1_000_000, 10),
            OrderByLimitStrategy::StreamingTopK
        );

        // Large K (> 1000) without index, K > sqrt(N) → full sort
        // For N=10000, sqrt(N) = 100, so K=5000 > sqrt(10000) and K > 1000
        assert_eq!(
            OrderByLimitStrategy::choose(false, 10_000, 5_000),
            OrderByLimitStrategy::FullSort
        );
        
        // K < 1000 always uses streaming even if K > sqrt(N)
        assert_eq!(
            OrderByLimitStrategy::choose(false, 1_000, 900),
            OrderByLimitStrategy::StreamingTopK
        );
    }

    #[test]
    fn test_queue_stats() {
        let config = QueueConfig::new("test-queue");
        let queue = PriorityQueue::new(config);

        queue.enqueue(1, b"task 1".to_vec());
        queue.enqueue(2, b"task 2".to_vec());
        queue.enqueue(3, b"task 3".to_vec());

        let stats = queue.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.pending, 3);
        assert_eq!(stats.inflight, 0);

        // Dequeue one
        let _ = queue.dequeue("worker-1");

        let stats = queue.stats();
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.inflight, 1);
    }

    #[test]
    fn test_delayed_task() {
        let config = QueueConfig::new("test-queue");
        let queue = PriorityQueue::new(config);

        // Enqueue with 1 hour delay
        queue.enqueue_delayed(1, b"delayed task".to_vec(), 3_600_000);

        // Should not be visible yet
        match queue.dequeue("worker-1") {
            DequeueResult::Empty => {}
            _ => panic!("Delayed task should not be visible"),
        }

        let stats = queue.stats();
        assert_eq!(stats.delayed, 1);
        assert_eq!(stats.pending, 0);
    }
}
