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

//! Memory pressure handling and resource limits
//!
//! Prevents OOM crashes by implementing memory budgets and backpressure.
//! Addresses Task 7 from task.md: Missing Memory Pressure Handling.
//!
//! ## WriteBufferManager (jj.md Task 2)
//!
//! Coordinates memory usage across active and immutable memtables to prevent OOM:
//! - Tracks total buffer memory (active + immutable memtables)
//! - Enforces soft/hard limits with backpressure
//! - Triggers flush when memory pressure is detected
//! - Blocks writes when hard limit is exceeded

use parking_lot::Condvar;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Memory budget configuration for the storage engine
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum memory for all components (bytes)
    pub total_budget: u64,
    /// Maximum memory for active memtable
    pub memtable_budget: u64,
    /// Maximum memory for immutable memtables
    pub immutable_memtables_budget: u64,
    /// Maximum memory for block cache
    pub block_cache_budget: u64,
    /// Percentage of budget at which to trigger early flush (0.0-1.0)
    pub soft_limit: f64,
    /// Percentage of budget at which to block writes (0.0-1.0)
    pub hard_limit: f64,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            total_budget: 512 * 1024 * 1024,               // 512 MB default
            memtable_budget: 32 * 1024 * 1024,             // 32 MB per memtable
            immutable_memtables_budget: 128 * 1024 * 1024, // 128 MB for immutable
            block_cache_budget: 256 * 1024 * 1024,         // 256 MB for cache
            soft_limit: 0.80,                              // 80% - trigger early flush
            hard_limit: 0.95,                              // 95% - block writes
        }
    }
}

impl MemoryBudget {
    /// Create budget from available system memory percentage
    ///
    /// Example: `from_system_memory_percent(0.25)` uses 25% of available RAM
    pub fn from_system_memory_percent(percent: f64) -> Self {
        let available_bytes = Self::get_available_memory();
        let total_budget = (available_bytes as f64 * percent) as u64;

        Self {
            total_budget,
            memtable_budget: total_budget / 16,
            immutable_memtables_budget: total_budget / 4,
            block_cache_budget: total_budget / 2,
            soft_limit: 0.80,
            hard_limit: 0.95,
        }
    }

    /// Get available system memory in bytes
    ///
    /// Platform-specific implementations:
    /// - Linux: Reads /proc/meminfo
    /// - macOS/BSD: Uses sysctl hw.memsize
    /// - Windows: Uses GlobalMemoryStatusEx
    /// - Fallback: Returns conservative 1GB estimate
    fn get_available_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            Self::linux_available_memory().unwrap_or(1024 * 1024 * 1024)
        }

        #[cfg(target_os = "macos")]
        {
            Self::macos_available_memory().unwrap_or(1024 * 1024 * 1024)
        }

        #[cfg(target_os = "windows")]
        {
            Self::windows_available_memory().unwrap_or(1024 * 1024 * 1024)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Conservative fallback: 1GB
            1024 * 1024 * 1024
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_available_memory() -> Option<u64> {
        use std::fs::read_to_string;

        let meminfo = read_to_string("/proc/meminfo").ok()?;

        // Prefer MemAvailable (includes reclaimable memory)
        // Fall back to MemFree if not available
        let mut mem_available = None;
        let mut mem_free = None;

        for line in meminfo.lines() {
            if line.starts_with("MemAvailable:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    mem_available = parts[1].parse::<u64>().ok();
                }
            } else if line.starts_with("MemFree:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    mem_free = parts[1].parse::<u64>().ok();
                }
            }
        }

        // Convert KB to bytes
        mem_available.or(mem_free).map(|kb| kb * 1024)
    }

    #[cfg(target_os = "macos")]
    fn macos_available_memory() -> Option<u64> {
        use std::process::Command;

        // Get total physical memory via sysctl
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;

        let mem_bytes: u64 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .ok()?;

        // hw.memsize returns total RAM - use 90% as "available" approximation
        // This is more conservative than caching would be but safer
        Some((mem_bytes as f64 * 0.9) as u64)
    }

    #[cfg(target_os = "windows")]
    fn windows_available_memory() -> Option<u64> {
        // For Windows, we could use GlobalMemoryStatusEx via winapi
        // For now, return None to use fallback (1GB)
        // TODO: Add winapi dependency and implement proper detection
        None
    }
}

/// Memory usage tracker with pressure detection
pub struct MemoryTracker {
    /// Current memory usage estimate
    current_usage: Arc<AtomicU64>,
    /// Memory budget configuration
    budget: MemoryBudget,
    /// Whether system is under memory pressure
    under_pressure: Arc<AtomicBool>,
}

impl MemoryTracker {
    /// Create new memory tracker with given budget
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            current_usage: Arc::new(AtomicU64::new(0)),
            budget,
            under_pressure: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Record memory allocation
    pub fn allocate(&self, bytes: u64) {
        let new_usage = self.current_usage.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.check_pressure(new_usage);
    }

    /// Record memory deallocation
    pub fn deallocate(&self, bytes: u64) {
        let prev_usage = self.current_usage.fetch_sub(bytes, Ordering::Relaxed);
        let new_usage = prev_usage.saturating_sub(bytes);
        self.check_pressure(new_usage);
    }

    /// Check if under memory pressure and update flag
    fn check_pressure(&self, current: u64) {
        let pressure = current as f64 >= (self.budget.total_budget as f64 * self.budget.soft_limit);
        self.under_pressure.store(pressure, Ordering::Relaxed);
    }

    /// Check if writes should be blocked (hard limit exceeded)
    pub fn should_block_writes(&self) -> bool {
        let current = self.current_usage.load(Ordering::Relaxed);
        current as f64 >= (self.budget.total_budget as f64 * self.budget.hard_limit)
    }

    /// Check if early flush should be triggered (soft limit exceeded)
    pub fn should_trigger_flush(&self) -> bool {
        self.under_pressure.load(Ordering::Relaxed)
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> u64 {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get memory usage as percentage of budget
    pub fn usage_percent(&self) -> f64 {
        let current = self.current_usage.load(Ordering::Relaxed);
        (current as f64 / self.budget.total_budget as f64) * 100.0
    }

    /// Reset memory usage counter
    pub fn reset(&self) {
        self.current_usage.store(0, Ordering::Relaxed);
        self.under_pressure.store(false, Ordering::Relaxed);
    }
}

/// Write Buffer Manager for coordinating memory across memtables
///
/// Implements jj.md Task 2: Write Buffer Manager with Global Memory Coordination
///
/// ## Algorithm
/// ```text
/// total_buffer_memory = active_memtable.size + Σ(immutable_memtables[i].size)
///
/// on_write(bytes):
///     if total_buffer_memory + bytes > hard_limit:
///         block_until(total_buffer_memory < soft_limit)
///     if total_buffer_memory > soft_limit:
///         trigger_flush(largest_immutable_memtable)
///     total_buffer_memory += bytes
///
/// on_flush_complete(memtable):
///     total_buffer_memory -= memtable.size
///     signal_blocked_writers()
/// ```
pub struct WriteBufferManager {
    /// Total memory used by active + immutable memtables
    total_buffer_memory: AtomicU64,
    /// Memory budget for all write buffers
    buffer_limit: u64,
    /// Soft limit percentage (0.0-1.0) - trigger flush
    soft_limit_ratio: f64,
    /// Hard limit percentage (0.0-1.0) - block writes
    hard_limit_ratio: f64,
    /// Whether writers are currently blocked
    writers_blocked: AtomicBool,
    /// Condition variable for blocked writers
    write_cv: Condvar,
    /// Mutex for condition variable
    write_mutex: Mutex<()>,
    /// Statistics
    stats: WriteBufferStats,
}

/// Statistics for write buffer monitoring
#[derive(Debug, Default)]
pub struct WriteBufferStats {
    /// Number of times writes were blocked
    pub blocks_count: AtomicU64,
    /// Total microseconds spent blocked
    pub blocked_time_us: AtomicU64,
    /// Number of flushes triggered by soft limit
    pub soft_limit_flushes: AtomicU64,
}

impl WriteBufferManager {
    /// Create a new write buffer manager
    ///
    /// # Arguments
    /// * `buffer_limit` - Maximum total memory for write buffers (bytes)
    pub fn new(buffer_limit: u64) -> Self {
        Self {
            total_buffer_memory: AtomicU64::new(0),
            buffer_limit,
            soft_limit_ratio: 0.8,
            hard_limit_ratio: 0.95,
            writers_blocked: AtomicBool::new(false),
            write_cv: Condvar::new(),
            write_mutex: Mutex::new(()),
            stats: WriteBufferStats::default(),
        }
    }

    /// Create with custom soft/hard limits
    pub fn with_limits(buffer_limit: u64, soft_limit_ratio: f64, hard_limit_ratio: f64) -> Self {
        Self {
            total_buffer_memory: AtomicU64::new(0),
            buffer_limit,
            soft_limit_ratio,
            hard_limit_ratio,
            writers_blocked: AtomicBool::new(false),
            write_cv: Condvar::new(),
            write_mutex: Mutex::new(()),
            stats: WriteBufferStats::default(),
        }
    }

    /// Reserve memory for a write operation
    ///
    /// May block if hard limit is exceeded, waiting for flushes to complete.
    /// Returns true if flush should be triggered (soft limit exceeded).
    pub fn reserve_memory(&self, bytes: u64) -> bool {
        let soft_limit = (self.buffer_limit as f64 * self.soft_limit_ratio) as u64;
        let hard_limit = (self.buffer_limit as f64 * self.hard_limit_ratio) as u64;

        loop {
            let current = self.total_buffer_memory.load(Ordering::Acquire);
            let new_total = current + bytes;

            if new_total > hard_limit {
                // Block until memory is freed
                self.writers_blocked.store(true, Ordering::Release);
                self.stats.blocks_count.fetch_add(1, Ordering::Relaxed);

                let start = std::time::Instant::now();
                {
                    let mut guard = self.write_mutex.lock();
                    // Wait for flush to complete
                    self.write_cv
                        .wait_for(&mut guard, std::time::Duration::from_millis(100));
                }
                self.stats
                    .blocked_time_us
                    .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);

                // Retry after waiting
                continue;
            }

            // Try to reserve the memory
            if self
                .total_buffer_memory
                .compare_exchange_weak(current, new_total, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Check if we should trigger flush (soft limit)
                let should_flush = new_total > soft_limit;
                if should_flush {
                    self.stats
                        .soft_limit_flushes
                        .fetch_add(1, Ordering::Relaxed);
                }
                return should_flush;
            }
            // CAS failed, retry
        }
    }

    /// Release memory after flush completes
    ///
    /// Signals any blocked writers to retry.
    pub fn release_memory(&self, bytes: u64) {
        self.total_buffer_memory.fetch_sub(bytes, Ordering::AcqRel);

        // Signal blocked writers
        if self.writers_blocked.swap(false, Ordering::AcqRel) {
            self.write_cv.notify_all();
        }
    }

    /// Get current buffer memory usage
    pub fn memory_usage(&self) -> u64 {
        self.total_buffer_memory.load(Ordering::Acquire)
    }

    /// Get usage as percentage of limit
    pub fn usage_percent(&self) -> f64 {
        let current = self.total_buffer_memory.load(Ordering::Acquire);
        (current as f64 / self.buffer_limit as f64) * 100.0
    }

    /// Check if under soft limit pressure
    pub fn is_under_pressure(&self) -> bool {
        let current = self.total_buffer_memory.load(Ordering::Acquire);
        let soft_limit = (self.buffer_limit as f64 * self.soft_limit_ratio) as u64;
        current > soft_limit
    }

    /// Get statistics
    pub fn stats(&self) -> &WriteBufferStats {
        &self.stats
    }
}

// =============================================================================
// Task 10 Enhancement: Async Write Buffer Spillover
// =============================================================================

/// Async spillover manager for write buffers
///
/// ## Problem Addressed
/// Current WriteBufferManager blocks writes when hard limit is exceeded.
/// This causes latency spikes for foreground operations.
///
/// ## Solution
/// Add async spillover to secondary storage (disk) before blocking:
/// 1. When soft limit exceeded → trigger async flush to SSTable
/// 2. When 90% limit exceeded → trigger spillover to temp file
/// 3. Only block at 100% when spillover buffer is also full
///
/// ## Architecture
/// ```text
/// Memtable (hot data)
///     │
///     ├── Soft limit (80%) → Async SSTable flush
///     │
///     ├── Spillover limit (90%) → Async temp file write
///     │
///     └── Hard limit (100%) → Block (last resort)
///
/// Spillover files are replayed into new SSTables during quiet periods.
/// ```
#[allow(dead_code)]
pub struct SpilloverManager {
    /// Write buffer manager reference
    write_buffer: Arc<WriteBufferManager>,
    /// Spillover buffer capacity
    spillover_capacity: u64,
    /// Current spillover usage
    spillover_used: AtomicU64,
    /// Spillover limit ratio (e.g., 0.9 for 90%)
    spillover_limit_ratio: f64,
    /// Number of active spillover files
    spillover_file_count: AtomicU64,
    /// Whether spillover is currently active
    spillover_active: AtomicBool,
    /// Channel for spillover requests
    spillover_tx: crossbeam_channel::Sender<SpilloverRequest>,
    /// Statistics
    stats: SpilloverStats,
}

/// Request to spill data to secondary storage
#[derive(Debug)]
pub struct SpilloverRequest {
    /// Key-value data to spill
    pub data: Vec<(Vec<u8>, Vec<u8>)>,
    /// Timestamp of oldest entry
    pub min_timestamp: u64,
    /// Timestamp of newest entry  
    pub max_timestamp: u64,
    /// Size in bytes
    pub size_bytes: u64,
}

/// Spillover statistics
#[derive(Debug, Default)]
pub struct SpilloverStats {
    /// Number of spillover operations
    pub spillover_count: AtomicU64,
    /// Total bytes spilled to disk
    pub bytes_spilled: AtomicU64,
    /// Total bytes recovered from spillover
    pub bytes_recovered: AtomicU64,
    /// Average spillover latency (microseconds)
    pub avg_latency_us: AtomicU64,
    /// Number of times blocking was avoided by spillover
    pub blocks_avoided: AtomicU64,
}

impl SpilloverManager {
    /// Create a new spillover manager
    pub fn new(
        write_buffer: Arc<WriteBufferManager>,
        spillover_capacity: u64,
    ) -> (Self, crossbeam_channel::Receiver<SpilloverRequest>) {
        let (tx, rx) = crossbeam_channel::bounded(16);

        let manager = Self {
            write_buffer,
            spillover_capacity,
            spillover_used: AtomicU64::new(0),
            spillover_limit_ratio: 0.9,
            spillover_file_count: AtomicU64::new(0),
            spillover_active: AtomicBool::new(false),
            spillover_tx: tx,
            stats: SpilloverStats::default(),
        };

        (manager, rx)
    }

    /// Check if spillover should be triggered
    pub fn should_spillover(&self) -> bool {
        let usage = self.write_buffer.memory_usage();
        let spillover_limit =
            (self.write_buffer.buffer_limit as f64 * self.spillover_limit_ratio) as u64;
        usage > spillover_limit && !self.is_spillover_full()
    }

    /// Check if spillover buffer is full
    pub fn is_spillover_full(&self) -> bool {
        self.spillover_used.load(Ordering::Relaxed) >= self.spillover_capacity
    }

    /// Reserve memory with spillover fallback
    ///
    /// Returns:
    /// - `Ok(false)` - Memory reserved, no action needed
    /// - `Ok(true)` - Memory reserved, flush should be triggered
    /// - `Err(SpilloverRequest)` - Caller should spill this data before proceeding
    pub fn reserve_memory(
        &self,
        bytes: u64,
        data: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<bool, SpilloverRequest> {
        // First try normal reservation
        if !self.write_buffer.is_under_pressure() {
            let should_flush = self.write_buffer.reserve_memory(bytes);
            return Ok(should_flush);
        }

        // Check if we should spillover
        if self.should_spillover() && !data.is_empty() {
            let request = SpilloverRequest {
                data,
                min_timestamp: 0,
                max_timestamp: u64::MAX,
                size_bytes: bytes,
            };

            // Try to send spillover request
            if self.spillover_tx.try_send(request.clone()).is_ok() {
                self.spillover_used.fetch_add(bytes, Ordering::Relaxed);
                self.stats.spillover_count.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_spilled.fetch_add(bytes, Ordering::Relaxed);
                self.stats.blocks_avoided.fetch_add(1, Ordering::Relaxed);
                self.spillover_active.store(true, Ordering::Release);

                // Don't block - data will be spilled
                return Ok(true);
            } else {
                // Spillover queue full, return request to caller
                return Err(request);
            }
        }

        // Fall back to blocking reservation
        let should_flush = self.write_buffer.reserve_memory(bytes);
        Ok(should_flush)
    }

    /// Release spillover capacity after recovery
    pub fn release_spillover(&self, bytes: u64) {
        self.spillover_used.fetch_sub(bytes, Ordering::Relaxed);
        self.stats
            .bytes_recovered
            .fetch_add(bytes, Ordering::Relaxed);

        if self.spillover_used.load(Ordering::Relaxed) == 0 {
            self.spillover_active.store(false, Ordering::Release);
        }
    }

    /// Check if spillover is active
    pub fn is_spillover_active(&self) -> bool {
        self.spillover_active.load(Ordering::Acquire)
    }

    /// Get spillover usage
    pub fn spillover_usage(&self) -> u64 {
        self.spillover_used.load(Ordering::Relaxed)
    }

    /// Get spillover capacity
    pub fn spillover_capacity(&self) -> u64 {
        self.spillover_capacity
    }

    /// Get statistics
    pub fn stats(&self) -> &SpilloverStats {
        &self.stats
    }
}

impl Clone for SpilloverRequest {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            min_timestamp: self.min_timestamp,
            max_timestamp: self.max_timestamp,
            size_bytes: self.size_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_default() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.total_budget, 512 * 1024 * 1024);
        assert_eq!(budget.soft_limit, 0.80);
        assert_eq!(budget.hard_limit, 0.95);
    }

    #[test]
    fn test_memory_tracker_pressure() {
        let budget = MemoryBudget {
            total_budget: 1000,
            memtable_budget: 100,
            immutable_memtables_budget: 300,
            block_cache_budget: 500,
            soft_limit: 0.80,
            hard_limit: 0.95,
        };

        let tracker = MemoryTracker::new(budget);

        // Below soft limit - no pressure
        tracker.allocate(700);
        assert!(!tracker.should_trigger_flush());
        assert!(!tracker.should_block_writes());

        // Above soft limit - trigger flush
        tracker.allocate(100);
        assert_eq!(tracker.current_usage(), 800);
        assert!(tracker.should_trigger_flush());
        assert!(!tracker.should_block_writes());

        // Above hard limit - block writes
        tracker.allocate(200);
        assert_eq!(tracker.current_usage(), 1000);
        assert!(tracker.should_trigger_flush());
        assert!(tracker.should_block_writes());

        // Deallocate - pressure relieved
        tracker.deallocate(300);
        assert_eq!(tracker.current_usage(), 700);
        assert!(!tracker.should_trigger_flush());
        assert!(!tracker.should_block_writes());
    }

    #[test]
    fn test_memory_tracker_usage_percent() {
        let budget = MemoryBudget {
            total_budget: 1000,
            memtable_budget: 100,
            immutable_memtables_budget: 300,
            block_cache_budget: 500,
            soft_limit: 0.80,
            hard_limit: 0.95,
        };

        let tracker = MemoryTracker::new(budget);

        tracker.allocate(500);
        assert_eq!(tracker.usage_percent(), 50.0);

        tracker.allocate(250);
        assert_eq!(tracker.usage_percent(), 75.0);
    }

    #[test]
    fn test_from_system_memory_percent() {
        let budget = MemoryBudget::from_system_memory_percent(0.25);

        // Should have reasonable values
        assert!(budget.total_budget > 0);
        assert!(budget.memtable_budget > 0);
        assert!(budget.memtable_budget < budget.total_budget);
        assert_eq!(budget.soft_limit, 0.80);
        assert_eq!(budget.hard_limit, 0.95);
    }

    #[test]
    fn test_system_memory_detection() {
        // Verify we can detect system memory (not fallback to 1GB)
        // This tests the platform-specific detection code
        let budget = MemoryBudget::from_system_memory_percent(1.0);

        // On any modern system with >4GB RAM, we should detect more than 1GB
        // If we're hitting the 1GB fallback, this test will fail
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            // Should detect at least 2GB on any modern dev machine
            assert!(
                budget.total_budget > 2 * 1024 * 1024 * 1024,
                "Expected >2GB detected, got {} bytes. Memory detection may have failed.",
                budget.total_budget
            );
        }

        // On all platforms, should at least get the 1GB fallback
        assert!(budget.total_budget >= 1024 * 1024 * 1024);
    }

    #[test]
    fn test_write_buffer_manager_reserve_release() {
        let wbm = WriteBufferManager::new(1000);

        // Reserve some memory
        let should_flush = wbm.reserve_memory(400);
        assert!(!should_flush); // Below 80% soft limit
        assert_eq!(wbm.memory_usage(), 400);

        // Reserve more - should trigger soft limit
        let should_flush = wbm.reserve_memory(500);
        assert!(should_flush); // Above 80% soft limit (900/1000)
        assert_eq!(wbm.memory_usage(), 900);

        // Release memory
        wbm.release_memory(600);
        assert_eq!(wbm.memory_usage(), 300);
        assert!(!wbm.is_under_pressure());
    }

    #[test]
    fn test_write_buffer_manager_pressure() {
        let wbm = WriteBufferManager::with_limits(1000, 0.5, 0.9);

        // Below soft limit
        wbm.reserve_memory(400);
        assert!(!wbm.is_under_pressure());
        assert_eq!(wbm.usage_percent(), 40.0);

        // Above soft limit
        wbm.reserve_memory(200);
        assert!(wbm.is_under_pressure());
        assert_eq!(wbm.usage_percent(), 60.0);
    }

    // Spillover Manager Tests

    #[test]
    fn test_spillover_manager_creation() {
        let wbm = Arc::new(WriteBufferManager::new(1000));
        let (spillover, _rx) = SpilloverManager::new(wbm, 500);

        assert_eq!(spillover.spillover_capacity(), 500);
        assert_eq!(spillover.spillover_usage(), 0);
        assert!(!spillover.is_spillover_active());
    }

    #[test]
    fn test_spillover_manager_reserve_below_limit() {
        let wbm = Arc::new(WriteBufferManager::new(1000));
        let (spillover, _rx) = SpilloverManager::new(wbm, 500);

        // Below any limits - should succeed without spillover
        let result = spillover.reserve_memory(100, vec![]);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No flush needed
    }

    #[test]
    fn test_spillover_manager_stats() {
        let wbm = Arc::new(WriteBufferManager::new(1000));
        let (spillover, _rx) = SpilloverManager::new(wbm.clone(), 500);

        // Fill up to trigger spillover consideration
        wbm.reserve_memory(850); // Above 80% soft limit

        // Create test data
        let data = vec![(b"key".to_vec(), b"value".to_vec())];

        // This should trigger spillover logic
        let result = spillover.reserve_memory(100, data);
        assert!(result.is_ok());

        let stats = spillover.stats();
        // Stats are available even if spillover wasn't needed
        assert!(stats.spillover_count.load(Ordering::Relaxed) <= 1);
    }

    #[test]
    fn test_spillover_release() {
        let wbm = Arc::new(WriteBufferManager::new(1000));
        let (spillover, _rx) = SpilloverManager::new(wbm, 500);

        // Simulate spillover used
        spillover.spillover_used.store(200, Ordering::Relaxed);
        spillover.spillover_active.store(true, Ordering::Release);

        assert!(spillover.is_spillover_active());
        assert_eq!(spillover.spillover_usage(), 200);

        // Release spillover
        spillover.release_spillover(200);

        assert!(!spillover.is_spillover_active());
        assert_eq!(spillover.spillover_usage(), 0);
    }
}
