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

//! Event-Driven Group Commit Buffer
//!
//! This module implements a proper group commit mechanism with:
//! - Event-driven wait using condition variables (not polling)
//! - Single fsync per batch with durability guarantee
//! - Adaptive batch sizing based on Little's Law
//!
//! ## Algorithm
//!
//! Group Commit Queueing Model:
//!
//! Little's Law: N = λ × W
//!   Where: N = avg number of requests in system
//!          λ = arrival rate (req/sec)
//!          W = avg time in system (sec)
//!
//! Optimal Batch Size: N* = sqrt(2 × L_fsync × λ / C_wait)
//!   Where: L_fsync = fsync latency
//!          C_wait = normalized waiting cost
//!
//! Example: L = 5ms, λ = 1000 req/s, C_wait = 1.0
//!   N* = sqrt(2 × 0.005 × 1000 / 1.0) ≈ 3.16 → 3 commits/batch
//!
//! ## Throughput Analysis
//!
//! Without group commit:
//!   Throughput = 1 / L = 200 commits/sec (for L = 5ms)
//!
//! With group commit (batch size N):
//!   Throughput = N / L = N × 200 commits/sec
//!
//! For N = 100:
//!   Throughput = 20,000 commits/sec
//!   Speedup = 100x

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Pending commit with notification channel
pub struct PendingCommitV2 {
    /// Transaction ID
    pub txn_id: u64,
    /// Enqueue timestamp
    pub enqueue_time: Instant,
    /// Notification channel (oneshot-style via Arc<Condvar>)
    pub notifier: Arc<(Mutex<CommitResult>, Condvar)>,
}

/// Result of a commit operation
#[derive(Debug, Clone)]
pub enum CommitResult {
    /// Commit pending (initial state)
    Pending,
    /// Commit succeeded with timestamp
    Success(u64),
    /// Commit failed with error message
    Error(String),
}

/// Event-driven group commit buffer with proper synchronization
#[allow(dead_code)]
pub struct EventDrivenGroupCommit {
    /// Pending commits queue
    pending: Mutex<VecDeque<PendingCommitV2>>,
    /// Signal that new commits are available
    commit_available: Condvar,
    /// Configuration
    config: GroupCommitConfig,
    /// Metrics
    metrics: GroupCommitMetrics,
    /// Flush callback (performs actual WAL fsync)
    #[allow(clippy::type_complexity)]
    flush_fn: Arc<dyn Fn(&[u64]) -> Result<u64, String> + Send + Sync>,
    /// Running flag
    running: AtomicU64, // 1 = running, 0 = stopped
    /// Flush thread handle
    flush_thread: Mutex<Option<JoinHandle<()>>>,
}

/// Group commit configuration
#[derive(Clone)]
pub struct GroupCommitConfig {
    /// Minimum batch size before flush
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time before flush (microseconds)
    pub max_wait_us: u64,
    /// Initial fsync latency estimate (microseconds)
    pub fsync_latency_us: u64,
    /// Arrival rate EMA alpha (0.0-1.0)
    pub ema_alpha: f64,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 1000,
            max_wait_us: 10_000,     // 10ms max wait
            fsync_latency_us: 5_000, // 5ms default
            ema_alpha: 0.1,
        }
    }
}

/// Metrics for group commit monitoring
pub struct GroupCommitMetrics {
    /// Current adaptive batch size
    pub adaptive_batch_size: AtomicU64,
    /// Estimated arrival rate (req/s × 1000 for precision)
    pub arrival_rate_ema: AtomicU64,
    /// Estimated fsync latency (microseconds)
    pub fsync_latency_us: AtomicU64,
    /// Total commits processed
    pub total_commits: AtomicU64,
    /// Total batches processed
    pub total_batches: AtomicU64,
    /// Total fsync time (microseconds)
    pub total_fsync_time_us: AtomicU64,
    /// Last arrival timestamp (microseconds since epoch)
    pub last_arrival_us: AtomicU64,
}

impl Default for GroupCommitMetrics {
    fn default() -> Self {
        Self {
            adaptive_batch_size: AtomicU64::new(10),
            arrival_rate_ema: AtomicU64::new(100_000), // 100 req/s initial
            fsync_latency_us: AtomicU64::new(5_000),
            total_commits: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            total_fsync_time_us: AtomicU64::new(0),
            last_arrival_us: AtomicU64::new(0),
        }
    }
}

impl EventDrivenGroupCommit {
    /// Create a new event-driven group commit buffer
    ///
    /// # Arguments
    /// * `flush_fn` - Callback that performs WAL fsync. Takes list of txn_ids, returns commit timestamp.
    pub fn new<F>(flush_fn: F) -> Self
    where
        F: Fn(&[u64]) -> Result<u64, String> + Send + Sync + 'static,
    {
        Self::with_config(flush_fn, GroupCommitConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config<F>(flush_fn: F, config: GroupCommitConfig) -> Self
    where
        F: Fn(&[u64]) -> Result<u64, String> + Send + Sync + 'static,
    {
        let gc = Self {
            pending: Mutex::new(VecDeque::new()),
            commit_available: Condvar::new(),
            config,
            metrics: GroupCommitMetrics::default(),
            flush_fn: Arc::new(flush_fn),
            running: AtomicU64::new(0),
            flush_thread: Mutex::new(None),
        };

        gc.metrics
            .fsync_latency_us
            .store(gc.config.fsync_latency_us, Ordering::Relaxed);
        gc
    }

    /// Start the background flush thread
    pub fn start(&self) -> Result<(), String> {
        if self
            .running
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err("Already running".into());
        }

        // We can't easily start a thread that references self
        // In a real implementation, we'd use Arc<Self> pattern
        // For now, document that flush_loop should be called from the owner
        Ok(())
    }

    /// Stop the flush thread
    pub fn stop(&self) {
        self.running.store(0, Ordering::SeqCst);

        // Wake up the flush thread
        let _lock = self.pending.lock().unwrap();
        self.commit_available.notify_all();
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst) == 1
    }

    /// Submit a commit and wait for it to complete
    ///
    /// This blocks until the transaction's batch has been fsynced.
    /// Returns the commit timestamp on success.
    pub fn submit_and_wait(&self, txn_id: u64) -> Result<u64, String> {
        // Update arrival rate
        self.update_arrival_rate();

        // Create notification channel
        let notifier = Arc::new((Mutex::new(CommitResult::Pending), Condvar::new()));
        let commit = PendingCommitV2 {
            txn_id,
            enqueue_time: Instant::now(),
            notifier: notifier.clone(),
        };

        // Enqueue and check if we should trigger flush
        let should_flush = {
            let mut pending = self.pending.lock().unwrap();
            pending.push_back(commit);

            let batch_size = self.optimal_batch_size();
            pending.len() >= batch_size
        };

        // Signal availability
        self.commit_available.notify_one();

        // If we should flush immediately and no background thread, flush inline
        if should_flush && !self.is_running() {
            self.flush_batch();
        }

        // Wait for result
        let (lock, cvar) = &*notifier;
        let mut result = lock.lock().unwrap();

        while matches!(*result, CommitResult::Pending) {
            // Wait with timeout for defensive programming
            let timeout = Duration::from_micros(self.config.max_wait_us * 2);
            let (new_result, timeout_result) = cvar.wait_timeout(result, timeout).unwrap();
            result = new_result;

            if timeout_result.timed_out() {
                // Timeout - try flushing ourselves if no background thread
                if !self.is_running() {
                    drop(result);
                    self.flush_batch();
                    result = lock.lock().unwrap();
                }
            }
        }

        match &*result {
            CommitResult::Success(ts) => Ok(*ts),
            CommitResult::Error(e) => Err(e.clone()),
            CommitResult::Pending => Err("Unexpected pending state".into()),
        }
    }

    /// Flush one batch of pending commits
    ///
    /// Called by background flush thread or inline when needed.
    pub fn flush_batch(&self) {
        let batch = {
            let mut pending = self.pending.lock().unwrap();
            if pending.is_empty() {
                return;
            }

            let batch_size = self.optimal_batch_size().min(pending.len());
            pending.drain(..batch_size).collect::<Vec<_>>()
        };

        if batch.is_empty() {
            return;
        }

        let txn_ids: Vec<_> = batch.iter().map(|c| c.txn_id).collect();
        let batch_size = batch.len();

        // Measure fsync time
        let start = Instant::now();
        let result = (self.flush_fn)(&txn_ids);
        let elapsed_us = start.elapsed().as_micros() as u64;

        // Update metrics
        self.update_fsync_latency(elapsed_us);
        self.metrics.total_batches.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_commits
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.metrics
            .total_fsync_time_us
            .fetch_add(elapsed_us, Ordering::Relaxed);

        // Notify all waiters
        for commit in batch {
            let (lock, cvar) = &*commit.notifier;
            let mut result_lock = lock.lock().unwrap();
            *result_lock = match &result {
                Ok(ts) => CommitResult::Success(*ts),
                Err(e) => CommitResult::Error(e.clone()),
            };
            cvar.notify_one();
        }
    }

    /// Background flush loop (call from owner thread)
    pub fn flush_loop(&self) {
        while self.is_running() {
            let should_flush = {
                let pending = self.pending.lock().unwrap();
                let batch_size = self.optimal_batch_size();

                if pending.len() >= batch_size {
                    true
                } else if pending.is_empty() {
                    // Wait for commits
                    let _pending = self
                        .commit_available
                        .wait_timeout(pending, Duration::from_micros(self.config.max_wait_us))
                        .unwrap()
                        .0;
                    false
                } else {
                    // Have some commits, check if we should wait longer
                    let oldest = pending
                        .front()
                        .map(|c| c.enqueue_time.elapsed().as_micros() as u64)
                        .unwrap_or(0);

                    if oldest > self.config.max_wait_us {
                        true
                    } else {
                        // Wait for more commits
                        let remaining =
                            Duration::from_micros(self.config.max_wait_us.saturating_sub(oldest));
                        let _pending = self
                            .commit_available
                            .wait_timeout(pending, remaining)
                            .unwrap()
                            .0;
                        true // Flush after wait
                    }
                }
            };

            if should_flush {
                self.flush_batch();
            }
        }
    }

    /// Compute optimal batch size using Little's Law
    ///
    /// N* = sqrt(2 × L_fsync × λ / C_wait)
    fn optimal_batch_size(&self) -> usize {
        let lambda = self.metrics.arrival_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0;
        let l_fsync = self.metrics.fsync_latency_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let c_wait = 1.0; // Normalized waiting cost

        let n_opt = (2.0 * l_fsync * lambda / c_wait).sqrt();
        let batch_size = (n_opt as usize)
            .max(self.config.min_batch_size)
            .min(self.config.max_batch_size);

        self.metrics
            .adaptive_batch_size
            .store(batch_size as u64, Ordering::Relaxed);
        batch_size
    }

    /// Update arrival rate using exponential moving average
    fn update_arrival_rate(&self) {
        let now_us = Self::now_us();
        let last = self.metrics.last_arrival_us.swap(now_us, Ordering::Relaxed);

        if last > 0 {
            let delta_us = now_us.saturating_sub(last);
            if delta_us > 0 {
                // Rate = 1_000_000 / delta_us (requests per second)
                // Stored as rate × 1000 for precision
                let instant_rate = 1_000_000_000 / delta_us;

                let old_rate = self.metrics.arrival_rate_ema.load(Ordering::Relaxed);
                let alpha = (self.config.ema_alpha * 1000.0) as u64;
                let new_rate = (old_rate * (1000 - alpha) + instant_rate * alpha) / 1000;
                self.metrics
                    .arrival_rate_ema
                    .store(new_rate, Ordering::Relaxed);
            }
        }
    }

    /// Update fsync latency estimate
    fn update_fsync_latency(&self, latency_us: u64) {
        let old = self.metrics.fsync_latency_us.load(Ordering::Relaxed);
        let alpha = (self.config.ema_alpha * 1000.0) as u64;
        let new = (old * (1000 - alpha) + latency_us * alpha) / 1000;
        self.metrics.fsync_latency_us.store(new, Ordering::Relaxed);
    }

    /// Get current time in microseconds
    fn now_us() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Get statistics for monitoring
    pub fn stats(&self) -> GroupCommitStatsV2 {
        GroupCommitStatsV2 {
            adaptive_batch_size: self.metrics.adaptive_batch_size.load(Ordering::Relaxed) as usize,
            arrival_rate: self.metrics.arrival_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0,
            fsync_latency_us: self.metrics.fsync_latency_us.load(Ordering::Relaxed),
            pending_count: self.pending.lock().unwrap().len(),
            total_commits: self.metrics.total_commits.load(Ordering::Relaxed),
            total_batches: self.metrics.total_batches.load(Ordering::Relaxed),
            avg_batch_size: {
                let batches = self.metrics.total_batches.load(Ordering::Relaxed);
                let commits = self.metrics.total_commits.load(Ordering::Relaxed);
                if batches > 0 {
                    commits as f64 / batches as f64
                } else {
                    0.0
                }
            },
            avg_fsync_time_us: {
                let batches = self.metrics.total_batches.load(Ordering::Relaxed);
                let time = self.metrics.total_fsync_time_us.load(Ordering::Relaxed);
                if batches > 0 { time / batches } else { 0 }
            },
        }
    }
}

/// Statistics for event-driven group commit
#[derive(Debug, Clone)]
pub struct GroupCommitStatsV2 {
    /// Current adaptive batch size
    pub adaptive_batch_size: usize,
    /// Estimated arrival rate (requests/second)
    pub arrival_rate: f64,
    /// Estimated fsync latency (microseconds)
    pub fsync_latency_us: u64,
    /// Current pending commit count
    pub pending_count: usize,
    /// Total commits processed
    pub total_commits: u64,
    /// Total batches processed
    pub total_batches: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average fsync time (microseconds)
    pub avg_fsync_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_single_commit() {
        let commit_ts = AtomicU64::new(100);
        let gc = EventDrivenGroupCommit::new(move |_txn_ids| {
            Ok(commit_ts.fetch_add(1, Ordering::SeqCst))
        });

        let result = gc.submit_and_wait(1);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 100);
    }

    #[test]
    fn test_batch_commit() {
        use parking_lot::RwLock;
        use std::sync::Arc;
        use std::thread;

        let _commit_ts = Arc::new(AtomicU64::new(100));
        let batch_sizes = Arc::new(RwLock::new(Vec::new()));
        let batch_sizes_clone = batch_sizes.clone();

        let gc = Arc::new(EventDrivenGroupCommit::with_config(
            move |txn_ids| {
                batch_sizes_clone.write().push(txn_ids.len());
                Ok(100)
            },
            GroupCommitConfig {
                min_batch_size: 3,
                max_wait_us: 1_000_000, // 1 second - long enough to batch
                ..Default::default()
            },
        ));

        // Submit 3 commits in parallel
        let mut handles = vec![];
        for i in 0..3 {
            let gc = Arc::clone(&gc);
            handles.push(thread::spawn(move || gc.submit_and_wait(i)));
        }

        // Wait for all
        for h in handles {
            assert!(h.join().unwrap().is_ok());
        }

        // Should have been batched
        let sizes = batch_sizes.read();
        assert!(!sizes.is_empty());
        // The 3 commits should have been batched together
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_adaptive_sizing() {
        let gc = EventDrivenGroupCommit::with_config(
            |_| Ok(1),
            GroupCommitConfig {
                fsync_latency_us: 5000, // 5ms
                ..Default::default()
            },
        );

        // Simulate high arrival rate (1000 req/s)
        gc.metrics
            .arrival_rate_ema
            .store(1_000_000, Ordering::Relaxed);

        let batch_size = gc.optimal_batch_size();

        // N* = sqrt(2 × 0.005 × 1000 / 1) ≈ 3.16
        assert!((3..=10).contains(&batch_size));
    }

    #[test]
    fn test_stats() {
        let gc = EventDrivenGroupCommit::new(|_| Ok(1));

        gc.metrics.total_commits.store(100, Ordering::Relaxed);
        gc.metrics.total_batches.store(10, Ordering::Relaxed);
        gc.metrics
            .total_fsync_time_us
            .store(50_000, Ordering::Relaxed);

        let stats = gc.stats();
        assert_eq!(stats.total_commits, 100);
        assert_eq!(stats.total_batches, 10);
        assert_eq!(stats.avg_batch_size, 10.0);
        assert_eq!(stats.avg_fsync_time_us, 5000);
    }
}
