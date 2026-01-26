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

//! Batched Dirty Tracking with MPSC Queue
//!
//! This module replaces per-write mutex acquisition for dirty tracking
//! with a batched MPSC approach that dramatically reduces lock contention.
//!
//! ## Problem: Lock Convoying
//!
//! Per-write mutex acquisition is the canonical scalability killer:
//! - Serializes otherwise-parallel writers
//! - Causes lock convoying under contention
//! - N writers → N lock acquisitions per batch
//!
//! ## Solution: Batched MPSC + Thread-Local Buffering
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Writer Threads                               │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
//! │  │ Thread 1 │  │ Thread 2 │  │ Thread 3 │  │ Thread N │       │
//! │  │  Buffer  │  │  Buffer  │  │  Buffer  │  │  Buffer  │       │
//! │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
//! │       │             │             │             │              │
//! │       └─────────────┼─────────────┼─────────────┘              │
//! │                     │             │                            │
//! │                     ▼             ▼                            │
//! │              ┌──────────────────────────────┐                  │
//! │              │      MPSC Channel            │                  │
//! │              │  (crossbeam-channel)         │                  │
//! │              └──────────────┬───────────────┘                  │
//! │                             │                                  │
//! │                             ▼                                  │
//! │              ┌──────────────────────────────┐                  │
//! │              │   Aggregator Thread          │                  │
//! │              │   (drains every 10ms or     │                  │
//! │              │    every 1000 entries)       │                  │
//! │              └──────────────────────────────┘                  │
//! └────────────────────────────────────────────────────────────────┘
//!
//! Lock acquisitions: O(W) → O(W/B) where W=writes, B=batch size
//! ```
//!
//! ## Performance
//!
//! - Thread-local buffer: Zero contention during writes
//! - MPSC send: ~20ns (vs ~200ns for mutex under contention)
//! - Batch flush: Amortized over B writes
//!
//! ## Usage
//!
//! ```ignore
//! let tracker = BatchedDirtyTracker::new();
//!
//! // In transaction:
//! tracker.mark_dirty(key);  // Lock-free, buffers locally
//!
//! // At commit:
//! tracker.flush_buffer();   // Sends batch to aggregator
//! ```

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{self, Receiver, Sender};
use parking_lot::Mutex;

use crate::txn_arena::KeyFingerprint;

/// Default batch size for thread-local buffer
const DEFAULT_BATCH_SIZE: usize = 64;

/// Maximum wait time before flushing (in milliseconds)
const MAX_FLUSH_INTERVAL_MS: u64 = 10;

// ============================================================================
// DirtyEvent - Batched Dirty Key Notification
// ============================================================================

/// Event sent through MPSC channel
#[derive(Debug)]
pub enum DirtyEvent {
    /// Batch of dirty key fingerprints from a transaction
    Batch {
        txn_id: u64,
        keys: Vec<KeyFingerprint>,
    },
    /// Epoch advance request
    AdvanceEpoch,
    /// Shutdown signal
    Shutdown,
}

// ============================================================================
// ThreadLocalBuffer - Per-Thread Dirty Key Accumulator
// ============================================================================

/// Thread-local buffer for accumulating dirty keys
struct ThreadLocalBuffer {
    /// Current transaction ID
    txn_id: u64,
    /// Accumulated dirty key fingerprints
    keys: Vec<KeyFingerprint>,
    /// Sender to aggregator
    sender: Sender<DirtyEvent>,
}

impl ThreadLocalBuffer {
    fn new(sender: Sender<DirtyEvent>) -> Self {
        Self {
            txn_id: 0,
            keys: Vec::with_capacity(DEFAULT_BATCH_SIZE),
            sender,
        }
    }

    /// Mark a key as dirty (no lock, no send)
    #[inline]
    fn mark_dirty(&mut self, txn_id: u64, key_fingerprint: KeyFingerprint) {
        if self.txn_id != txn_id {
            // New transaction - flush old buffer if any
            self.flush();
            self.txn_id = txn_id;
        }
        self.keys.push(key_fingerprint);
    }

    /// Flush accumulated keys to aggregator
    fn flush(&mut self) {
        if !self.keys.is_empty() {
            let keys = std::mem::take(&mut self.keys);
            // Best-effort send - don't block if channel is full
            let _ = self.sender.try_send(DirtyEvent::Batch {
                txn_id: self.txn_id,
                keys,
            });
            self.keys = Vec::with_capacity(DEFAULT_BATCH_SIZE);
        }
    }
}

// ============================================================================
// BatchedDirtyTracker - Lock-Free Dirty Tracking
// ============================================================================

/// Batched dirty tracker with MPSC queue
///
/// Provides lock-free dirty tracking for multi-threaded writes.
/// Each thread accumulates dirty keys locally, then sends them
/// in batches through an MPSC channel to an aggregator.
pub struct BatchedDirtyTracker {
    /// MPSC sender (cloned for each thread)
    sender: Sender<DirtyEvent>,
    /// MPSC receiver (for aggregator thread)
    receiver: Receiver<DirtyEvent>,
    /// Aggregator thread handle
    aggregator_handle: Mutex<Option<JoinHandle<()>>>,
    /// Running flag
    running: AtomicBool,
    /// Current epoch
    current_epoch: AtomicU64,
    /// Aggregated dirty keys per epoch
    epochs: [Mutex<HashSet<KeyFingerprint>>; 4],
    /// Statistics
    stats: DirtyTrackingStats,
}

/// Dirty tracking statistics
pub struct DirtyTrackingStats {
    /// Total events received
    pub events_received: AtomicU64,
    /// Total keys tracked
    pub keys_tracked: AtomicU64,
    /// Total batches received
    pub batches_received: AtomicU64,
    /// Current epoch
    pub current_epoch: AtomicU64,
}

impl Default for DirtyTrackingStats {
    fn default() -> Self {
        Self {
            events_received: AtomicU64::new(0),
            keys_tracked: AtomicU64::new(0),
            batches_received: AtomicU64::new(0),
            current_epoch: AtomicU64::new(0),
        }
    }
}

const EPOCH_RING_SIZE: usize = 4;

impl BatchedDirtyTracker {
    /// Create a new batched dirty tracker
    pub fn new() -> Arc<Self> {
        let (sender, receiver) = crossbeam_channel::bounded(1024);
        
        let tracker = Arc::new(Self {
            sender,
            receiver,
            aggregator_handle: Mutex::new(None),
            running: AtomicBool::new(false),
            current_epoch: AtomicU64::new(0),
            epochs: [
                Mutex::new(HashSet::new()),
                Mutex::new(HashSet::new()),
                Mutex::new(HashSet::new()),
                Mutex::new(HashSet::new()),
            ],
            stats: DirtyTrackingStats::default(),
        });
        
        tracker
    }

    /// Start the aggregator thread
    pub fn start(self: &Arc<Self>) {
        if self.running.swap(true, Ordering::SeqCst) {
            return; // Already running
        }

        let tracker = Arc::clone(self);
        let handle = thread::spawn(move || {
            tracker.aggregator_loop();
        });

        *self.aggregator_handle.lock() = Some(handle);
    }

    /// Stop the aggregator thread
    pub fn stop(&self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return; // Already stopped
        }

        // Send shutdown signal
        let _ = self.sender.send(DirtyEvent::Shutdown);

        // Wait for aggregator to finish
        if let Some(handle) = self.aggregator_handle.lock().take() {
            let _ = handle.join();
        }
    }

    /// Get a sender for a thread to use
    pub fn get_sender(&self) -> Sender<DirtyEvent> {
        self.sender.clone()
    }

    /// Mark a key as dirty using a thread-local buffer
    ///
    /// This is the zero-contention hot path used by writers.
    #[inline]
    pub fn mark_dirty(&self, txn_id: u64, key_fingerprint: KeyFingerprint) {
        thread_local! {
            static BUFFER: RefCell<Option<ThreadLocalBuffer>> = const { RefCell::new(None) };
        }

        BUFFER.with(|cell| {
            let mut buffer = cell.borrow_mut();
            if buffer.is_none() {
                *buffer = Some(ThreadLocalBuffer::new(self.sender.clone()));
            }
            buffer.as_mut().unwrap().mark_dirty(txn_id, key_fingerprint);
        });
    }

    /// Flush the current thread's buffer
    pub fn flush_thread_buffer(&self) {
        thread_local! {
            static BUFFER: RefCell<Option<ThreadLocalBuffer>> = const { RefCell::new(None) };
        }

        BUFFER.with(|cell| {
            if let Some(buffer) = cell.borrow_mut().as_mut() {
                buffer.flush();
            }
        });
    }

    /// Send a batch of dirty keys directly (for transaction commit)
    #[inline]
    pub fn send_batch(&self, txn_id: u64, keys: Vec<KeyFingerprint>) {
        if keys.is_empty() {
            return;
        }
        let _ = self.sender.try_send(DirtyEvent::Batch { txn_id, keys });
    }

    /// Advance to next epoch, returning the old epoch's dirty keys
    pub fn advance_epoch(&self) -> (u64, Vec<KeyFingerprint>) {
        // Send epoch advance event to ensure all pending events are processed
        let _ = self.sender.try_send(DirtyEvent::AdvanceEpoch);
        
        let old_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
        let old_idx = (old_epoch as usize) % EPOCH_RING_SIZE;
        
        // Drain the old epoch
        let mut guard = self.epochs[old_idx].lock();
        let keys: Vec<_> = guard.drain().collect();
        
        self.stats.current_epoch.store(old_epoch + 1, Ordering::Relaxed);
        
        (old_epoch, keys)
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Relaxed)
    }

    /// Get statistics
    pub fn stats(&self) -> &DirtyTrackingStats {
        &self.stats
    }

    /// Aggregator loop - runs in background thread
    fn aggregator_loop(&self) {
        use crossbeam_channel::RecvTimeoutError;
        
        let timeout = std::time::Duration::from_millis(MAX_FLUSH_INTERVAL_MS);
        
        while self.running.load(Ordering::Relaxed) {
            match self.receiver.recv_timeout(timeout) {
                Ok(event) => {
                    self.process_event(event);
                }
                Err(RecvTimeoutError::Timeout) => {
                    // No events for a while - that's fine
                }
                Err(RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        
        // Drain remaining events on shutdown
        while let Ok(event) = self.receiver.try_recv() {
            if matches!(event, DirtyEvent::Shutdown) {
                break;
            }
            self.process_event(event);
        }
    }

    /// Process a single event
    fn process_event(&self, event: DirtyEvent) {
        match event {
            DirtyEvent::Batch { txn_id: _, keys } => {
                let epoch = self.current_epoch.load(Ordering::Relaxed);
                let idx = (epoch as usize) % EPOCH_RING_SIZE;
                
                let mut guard = self.epochs[idx].lock();
                let key_count = keys.len();
                guard.extend(keys);
                
                self.stats.events_received.fetch_add(1, Ordering::Relaxed);
                self.stats.keys_tracked.fetch_add(key_count as u64, Ordering::Relaxed);
                self.stats.batches_received.fetch_add(1, Ordering::Relaxed);
            }
            DirtyEvent::AdvanceEpoch => {
                // Epoch advance is handled by the caller
            }
            DirtyEvent::Shutdown => {
                // Will exit the loop
            }
        }
    }
}

impl Default for BatchedDirtyTracker {
    fn default() -> Self {
        let (sender, receiver) = crossbeam_channel::bounded(1024);
        Self {
            sender,
            receiver,
            aggregator_handle: Mutex::new(None),
            running: AtomicBool::new(false),
            current_epoch: AtomicU64::new(0),
            epochs: [
                Mutex::new(HashSet::new()),
                Mutex::new(HashSet::new()),
                Mutex::new(HashSet::new()),
                Mutex::new(HashSet::new()),
            ],
            stats: DirtyTrackingStats::default(),
        }
    }
}

impl Drop for BatchedDirtyTracker {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// TxnDirtyBuffer - Transaction-Local Dirty Key Buffer
// ============================================================================

/// Transaction-local dirty key buffer for commit-time batching
///
/// Instead of tracking dirty keys globally during the transaction,
/// this buffer accumulates them locally and flushes once at commit.
/// This is simpler than the thread-local approach and works well
/// for single-threaded transaction execution.
pub struct TxnDirtyBuffer {
    /// Transaction ID
    txn_id: u64,
    /// Accumulated dirty key fingerprints
    keys: Vec<KeyFingerprint>,
}

impl TxnDirtyBuffer {
    /// Create a new transaction dirty buffer
    #[inline]
    pub fn new(txn_id: u64) -> Self {
        Self {
            txn_id,
            keys: Vec::with_capacity(64),
        }
    }

    /// Create with expected capacity
    #[inline]
    pub fn with_capacity(txn_id: u64, capacity: usize) -> Self {
        Self {
            txn_id,
            keys: Vec::with_capacity(capacity),
        }
    }

    /// Record a dirty key (no lock, just local append)
    #[inline]
    pub fn record(&mut self, key_fingerprint: KeyFingerprint) {
        self.keys.push(key_fingerprint);
    }

    /// Record multiple dirty keys
    #[inline]
    pub fn record_many(&mut self, key_fingerprints: impl IntoIterator<Item = KeyFingerprint>) {
        self.keys.extend(key_fingerprints);
    }

    /// Get the transaction ID
    #[inline]
    pub fn txn_id(&self) -> u64 {
        self.txn_id
    }

    /// Get the number of dirty keys
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Drain the buffer and return all keys
    #[inline]
    pub fn drain(&mut self) -> Vec<KeyFingerprint> {
        std::mem::take(&mut self.keys)
    }

    /// Flush to a BatchedDirtyTracker
    #[inline]
    pub fn flush_to(&mut self, tracker: &BatchedDirtyTracker) {
        if !self.keys.is_empty() {
            tracker.send_batch(self.txn_id, std::mem::take(&mut self.keys));
            self.keys = Vec::with_capacity(64);
        }
    }

    /// Clear the buffer without flushing
    #[inline]
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_txn_dirty_buffer() {
        let mut buffer = TxnDirtyBuffer::new(1);
        
        buffer.record(KeyFingerprint::from_bytes(b"key1"));
        buffer.record(KeyFingerprint::from_bytes(b"key2"));
        buffer.record(KeyFingerprint::from_bytes(b"key3"));
        
        assert_eq!(buffer.len(), 3);
        
        let keys = buffer.drain();
        assert_eq!(keys.len(), 3);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_batched_tracker_basic() {
        let tracker = BatchedDirtyTracker::new();
        tracker.start();
        
        // Send some events directly
        tracker.send_batch(1, vec![
            KeyFingerprint::from_bytes(b"key1"),
            KeyFingerprint::from_bytes(b"key2"),
        ]);
        
        // Give aggregator time to process
        thread::sleep(Duration::from_millis(50));
        
        // Advance epoch to collect
        let (_epoch, keys) = tracker.advance_epoch();
        
        // Keys should have been processed
        assert!(tracker.stats().batches_received.load(Ordering::Relaxed) >= 1);
        
        tracker.stop();
    }

    #[test]
    fn test_epoch_rotation() {
        let tracker = BatchedDirtyTracker::new();
        
        // Directly insert into epochs without starting the aggregator thread
        {
            let mut guard = tracker.epochs[0].lock();
            guard.insert(KeyFingerprint::from_bytes(b"key1"));
            guard.insert(KeyFingerprint::from_bytes(b"key2"));
        }
        
        let (epoch, keys) = tracker.advance_epoch();
        assert_eq!(epoch, 0);
        assert_eq!(keys.len(), 2);
        
        // New epoch should be empty
        let (epoch2, keys2) = tracker.advance_epoch();
        assert_eq!(epoch2, 1);
        assert!(keys2.is_empty());
    }
}
