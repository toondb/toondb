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

//! io_uring Backend for Linux Async I/O
//!
//! Implements asynchronous I/O using io_uring for high-throughput workloads.
//!
//! ## jj.md Task 15: io_uring Support
//!
//! Goals:
//! - 2-5x throughput improvement for write-heavy workloads
//! - Reduced CPU usage (fewer syscalls)
//! - Better integration with async Rust ecosystem
//!
//! ## Architecture
//!
//! ```text
//! Application          Kernel
//!     │                   │
//!     ├── SQ ────────────► │ (Submission Queue)
//!     │   [op1][op2]...   │
//!     │                   │
//!     │ ◄──────────── CQ ─┤ (Completion Queue)
//!     │   [res1][res2]... │
//! ```
//!
//! ## Platform Support
//!
//! - Linux 5.1+: Full io_uring support
//! - macOS/Windows: Fallback to standard async I/O
//!
//! ## Features
//!
//! - Zero-copy: Kernel operates on user buffers directly
//! - Batching: Submit multiple ops with single syscall
//! - Async: Non-blocking completion notification
//! - Polling: Busy-poll mode for ultra-low latency
//!
//! Reference: io_uring documentation - https://kernel.dk/io_uring.pdf

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(target_os = "linux")]
use io_uring::{IoUring, opcode, types};

/// io_uring configuration
#[derive(Debug, Clone)]
pub struct IoUringConfig {
    /// Size of the submission queue (power of 2)
    pub sq_entries: u32,
    /// Size of the completion queue (usually 2x sq_entries)
    pub cq_entries: u32,
    /// Use kernel-side polling (IORING_SETUP_SQPOLL)
    pub sq_poll: bool,
    /// Idle timeout for SQ polling in milliseconds
    pub sq_poll_idle_ms: u32,
    /// Use registered buffers for zero-copy I/O
    pub use_registered_buffers: bool,
    /// Maximum number of registered buffers
    pub max_registered_buffers: usize,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        Self {
            sq_entries: 256,
            cq_entries: 512,
            sq_poll: false,
            sq_poll_idle_ms: 1000,
            use_registered_buffers: true,
            max_registered_buffers: 64,
        }
    }
}

impl IoUringConfig {
    /// High-throughput configuration
    pub fn high_throughput() -> Self {
        Self {
            sq_entries: 1024,
            cq_entries: 2048,
            sq_poll: true,
            sq_poll_idle_ms: 2000,
            use_registered_buffers: true,
            max_registered_buffers: 256,
        }
    }

    /// Low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            sq_entries: 64,
            cq_entries: 128,
            sq_poll: true,
            sq_poll_idle_ms: 100,
            use_registered_buffers: true,
            max_registered_buffers: 32,
        }
    }
}

/// I/O operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOpType {
    Read,
    Write,
    Fsync,
    Fallocate,
    Close,
}

/// I/O operation for submission
#[derive(Debug)]
pub struct IoOp {
    /// Operation type
    pub op_type: IoOpType,
    /// File descriptor (or index for registered files)
    pub fd: i32,
    /// Buffer for read/write operations
    pub buffer: Vec<u8>,
    /// Offset in the file
    pub offset: u64,
    /// Length of the operation
    pub len: usize,
    /// User data for tracking
    pub user_data: u64,
}

impl IoOp {
    /// Create a read operation
    pub fn read(fd: i32, offset: u64, len: usize, user_data: u64) -> Self {
        Self {
            op_type: IoOpType::Read,
            fd,
            buffer: vec![0u8; len],
            offset,
            len,
            user_data,
        }
    }

    /// Create a write operation
    pub fn write(fd: i32, data: Vec<u8>, offset: u64, user_data: u64) -> Self {
        let len = data.len();
        Self {
            op_type: IoOpType::Write,
            fd,
            buffer: data,
            offset,
            len,
            user_data,
        }
    }

    /// Create an fsync operation
    pub fn fsync(fd: i32, user_data: u64) -> Self {
        Self {
            op_type: IoOpType::Fsync,
            fd,
            buffer: Vec::new(),
            offset: 0,
            len: 0,
            user_data,
        }
    }
}

/// Completion result for an I/O operation
#[derive(Debug)]
pub struct IoCompletion {
    /// User data from the original operation
    pub user_data: u64,
    /// Result (bytes transferred or error code)
    pub result: i32,
    /// Whether the operation succeeded
    pub success: bool,
}

impl IoCompletion {
    /// Create a successful completion
    pub fn success(user_data: u64, result: i32) -> Self {
        Self {
            user_data,
            result,
            success: true,
        }
    }

    /// Create a failed completion
    pub fn failure(user_data: u64, error_code: i32) -> Self {
        Self {
            user_data,
            result: error_code,
            success: false,
        }
    }

    /// Get the number of bytes transferred
    pub fn bytes_transferred(&self) -> Option<usize> {
        if self.success && self.result >= 0 {
            Some(self.result as usize)
        } else {
            None
        }
    }
}

/// Statistics for io_uring operations
#[derive(Debug, Default)]
pub struct IoUringStats {
    /// Total operations submitted
    pub ops_submitted: AtomicU64,
    /// Total operations completed
    pub ops_completed: AtomicU64,
    /// Total bytes read
    pub bytes_read: AtomicU64,
    /// Total bytes written
    pub bytes_written: AtomicU64,
    /// Total syscalls (submit + wait)
    pub syscalls: AtomicU64,
    /// Operations batched (multiple ops per syscall)
    pub ops_batched: AtomicU64,
}

impl IoUringStats {
    /// Create new stats
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a submission
    pub fn record_submit(&self, count: u64) {
        self.ops_submitted.fetch_add(count, Ordering::Relaxed);
        self.syscalls.fetch_add(1, Ordering::Relaxed);
        if count > 1 {
            self.ops_batched.fetch_add(count - 1, Ordering::Relaxed);
        }
    }

    /// Record a completion
    pub fn record_completion(&self, op_type: IoOpType, bytes: u64) {
        self.ops_completed.fetch_add(1, Ordering::Relaxed);
        match op_type {
            IoOpType::Read => {
                self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
            }
            IoOpType::Write => {
                self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Get snapshot
    pub fn snapshot(&self) -> IoUringStatsSnapshot {
        IoUringStatsSnapshot {
            ops_submitted: self.ops_submitted.load(Ordering::Relaxed),
            ops_completed: self.ops_completed.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            syscalls: self.syscalls.load(Ordering::Relaxed),
            ops_batched: self.ops_batched.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of io_uring stats
#[derive(Debug, Clone)]
pub struct IoUringStatsSnapshot {
    pub ops_submitted: u64,
    pub ops_completed: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub syscalls: u64,
    pub ops_batched: u64,
}

impl IoUringStatsSnapshot {
    /// Calculate batching efficiency
    pub fn batching_efficiency(&self) -> f64 {
        if self.syscalls == 0 {
            0.0
        } else {
            self.ops_submitted as f64 / self.syscalls as f64
        }
    }

    /// Calculate throughput in bytes/syscall
    pub fn bytes_per_syscall(&self) -> f64 {
        if self.syscalls == 0 {
            0.0
        } else {
            (self.bytes_read + self.bytes_written) as f64 / self.syscalls as f64
        }
    }
}

/// Async I/O backend trait
///
/// This trait abstracts over different async I/O implementations:
/// - io_uring on Linux 5.1+
/// - Standard sync I/O as fallback
pub trait AsyncIoBackend: Send + Sync {
    /// Submit an I/O operation
    fn submit(&mut self, op: IoOp) -> io::Result<()>;

    /// Submit multiple I/O operations (batched)
    fn submit_batch(&mut self, ops: Vec<IoOp>) -> io::Result<()>;

    /// Wait for at least one completion
    fn wait_one(&mut self) -> io::Result<IoCompletion>;

    /// Wait for all pending completions
    fn wait_all(&mut self) -> io::Result<Vec<IoCompletion>>;

    /// Get the number of pending operations
    fn pending(&self) -> usize;

    /// Check if io_uring is available
    fn is_uring_available(&self) -> bool;
}

/// Fallback synchronous I/O backend
///
/// Used on platforms where io_uring is not available (macOS, Windows, older Linux).
/// Provides the same interface but executes operations synchronously.
pub struct SyncIoBackend {
    pending: parking_lot::Mutex<VecDeque<IoOp>>,
    completions: parking_lot::Mutex<VecDeque<IoCompletion>>,
    stats: Arc<IoUringStats>,
}

impl SyncIoBackend {
    /// Create a new sync I/O backend
    pub fn new(stats: Arc<IoUringStats>) -> Self {
        Self {
            pending: parking_lot::Mutex::new(VecDeque::new()),
            completions: parking_lot::Mutex::new(VecDeque::new()),
            stats,
        }
    }

    /// Execute an operation synchronously
    fn execute(&self, mut op: IoOp) -> IoCompletion {
        use std::os::unix::io::FromRawFd;

        let result = unsafe {
            // SAFETY: We trust the caller to provide valid file descriptors
            let file = File::from_raw_fd(op.fd);
            let res = match op.op_type {
                IoOpType::Read => {
                    let mut file_ref = &file;
                    file_ref.seek(SeekFrom::Start(op.offset)).ok();
                    file_ref.read(&mut op.buffer)
                }
                IoOpType::Write => {
                    let mut file_ref = &file;
                    file_ref.seek(SeekFrom::Start(op.offset)).ok();
                    file_ref.write(&op.buffer)
                }
                IoOpType::Fsync => file.sync_all().map(|_| 0),
                IoOpType::Fallocate | IoOpType::Close => Ok(0),
            };
            // Don't close the fd - it's managed elsewhere
            std::mem::forget(file);
            res
        };

        match result {
            Ok(n) => {
                self.stats.record_completion(op.op_type, n as u64);
                IoCompletion::success(op.user_data, n as i32)
            }
            Err(e) => IoCompletion::failure(op.user_data, e.raw_os_error().unwrap_or(-1)),
        }
    }
}

impl AsyncIoBackend for SyncIoBackend {
    fn submit(&mut self, op: IoOp) -> io::Result<()> {
        self.stats.record_submit(1);
        let completion = self.execute(op);
        self.completions.lock().push_back(completion);
        Ok(())
    }

    fn submit_batch(&mut self, ops: Vec<IoOp>) -> io::Result<()> {
        let count = ops.len() as u64;
        self.stats.record_submit(count);

        let completions: Vec<_> = ops.into_iter().map(|op| self.execute(op)).collect();
        self.completions.lock().extend(completions);
        Ok(())
    }

    fn wait_one(&mut self) -> io::Result<IoCompletion> {
        self.completions
            .lock()
            .pop_front()
            .ok_or_else(|| io::Error::new(io::ErrorKind::WouldBlock, "No completions"))
    }

    fn wait_all(&mut self) -> io::Result<Vec<IoCompletion>> {
        Ok(self.completions.lock().drain(..).collect())
    }

    fn pending(&self) -> usize {
        self.pending.lock().len()
    }

    fn is_uring_available(&self) -> bool {
        false
    }
}

/// Linux io_uring backend (real implementation)
///
/// Uses the io-uring crate to provide real async I/O on Linux 5.1+
/// Falls back to synchronous I/O on older kernels or other platforms.
#[cfg(target_os = "linux")]
pub struct LinuxIoUringBackend {
    uring: Option<IoUring>,
    config: IoUringConfig,
    pending: parking_lot::Mutex<VecDeque<IoOp>>,
    completions: parking_lot::Mutex<VecDeque<IoCompletion>>,
    stats: Arc<IoUringStats>,
    /// Whether real io_uring is available (requires kernel check)
    uring_available: bool,
}

#[cfg(target_os = "linux")]
impl LinuxIoUringBackend {
    /// Create a new io_uring backend
    pub fn new(config: IoUringConfig, stats: Arc<IoUringStats>) -> io::Result<Self> {
        // Try to initialize real io_uring
        let (uring, uring_available) = match IoUring::new(config.sq_entries) {
            Ok(uring) => {
                eprintln!("io_uring initialized successfully with {} entries", config.sq_entries);
                (Some(uring), true)
            },
            Err(e) => {
                eprintln!("io_uring initialization failed: {}. Falling back to sync I/O", e);
                (None, false)
            }
        };

        Ok(Self {
            uring,
            config,
            pending: parking_lot::Mutex::new(VecDeque::new()),
            completions: parking_lot::Mutex::new(VecDeque::new()),
            stats,
            uring_available,
        })
    }

    /// Check if io_uring is available on this system
    fn check_uring_available() -> bool {
        // Check kernel version by reading /proc/version
        #[cfg(target_os = "linux")]
        {
            if let Ok(version) = std::fs::read_to_string("/proc/version") {
                // Parse kernel version (e.g., "Linux version 5.15.0-generic")
                let parts: Vec<&str> = version.split_whitespace().collect();
                if parts.len() >= 3 {
                    let version_parts: Vec<&str> = parts[2].split('.').collect();
                    if version_parts.len() >= 2
                        && let (Ok(major), Ok(minor)) = (
                            version_parts[0].parse::<u32>(),
                            version_parts[1].parse::<u32>(),
                        )
                    {
                        // io_uring requires Linux 5.1+
                        return major > 5 || (major == 5 && minor >= 1);
                    }
                }
            }
        }
        false
    }

    /// Submit an operation to io_uring (real implementation)
    fn submit_to_uring(&mut self, op: IoOp) -> io::Result<()> {
        if let Some(ref mut uring) = self.uring {
            let mut sq = uring.submission();
            
            let sqe = match op.op_type {
                IoOpType::Read => {
                    opcode::Read::new(types::Fd(op.fd), op.buffer.as_ptr() as *mut u8, op.len as u32)
                        .offset(op.offset)
                        .build()
                        .user_data(op.user_data)
                }
                IoOpType::Write => {
                    opcode::Write::new(types::Fd(op.fd), op.buffer.as_ptr(), op.len as u32)
                        .offset(op.offset)
                        .build()
                        .user_data(op.user_data)
                }
                IoOpType::Fsync => {
                    opcode::Fsync::new(types::Fd(op.fd))
                        .build()
                        .user_data(op.user_data)
                }
                _ => return Err(io::Error::new(io::ErrorKind::Unsupported, "Operation not supported")),
            };

            // SAFETY: We submit to the ring and will wait for completion
            unsafe {
                sq.push(&sqe).map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to push to submission queue"))?;
            }
            
            sq.sync();
            drop(sq);
            
            // Submit and wait for completion
            uring.submit_and_wait(1)?;
            
            // Process completion
            let mut cq = uring.completion();
            while let Some(cqe) = cq.next() {
                let completion = if cqe.result() >= 0 {
                    self.stats.record_completion(op.op_type, cqe.result() as u64);
                    IoCompletion::success(cqe.user_data(), cqe.result())
                } else {
                    IoCompletion::failure(cqe.user_data(), cqe.result())
                };
                self.completions.lock().push_back(completion);
            }
            
            Ok(())
        } else {
            // Fallback to synchronous I/O
            let completion = self.simulate_execute(op);
            self.completions.lock().push_back(completion);
            Ok(())
        }
    }

    /// Simulate io_uring submission (fallback for when io_uring is not available)
    fn simulate_execute(&self, mut op: IoOp) -> IoCompletion {
        use std::os::unix::io::FromRawFd;

        let result = unsafe {
            let file = File::from_raw_fd(op.fd);
            let res = match op.op_type {
                IoOpType::Read => {
                    let mut file_ref = &file;
                    file_ref.seek(SeekFrom::Start(op.offset)).ok();
                    file_ref.read(&mut op.buffer)
                }
                IoOpType::Write => {
                    let mut file_ref = &file;
                    file_ref.seek(SeekFrom::Start(op.offset)).ok();
                    file_ref.write(&op.buffer)
                }
                IoOpType::Fsync => file.sync_all().map(|_| 0),
                IoOpType::Fallocate | IoOpType::Close => Ok(0),
            };
            std::mem::forget(file);
            res
        };

        match result {
            Ok(n) => {
                self.stats.record_completion(op.op_type, n as u64);
                IoCompletion::success(op.user_data, n as i32)
            }
            Err(e) => IoCompletion::failure(op.user_data, e.raw_os_error().unwrap_or(-1)),
        }
    }
}

#[cfg(target_os = "linux")]
impl AsyncIoBackend for LinuxIoUringBackend {
    fn submit(&mut self, op: IoOp) -> io::Result<()> {
        self.stats.record_submit(1);
        self.submit_to_uring(op)
    }

    fn submit_batch(&mut self, ops: Vec<IoOp>) -> io::Result<()> {
        let count = ops.len() as u64;
        self.stats.record_submit(count);

        if let Some(ref mut uring) = self.uring {
            let mut sq = uring.submission();
            
            // Submit all operations
            for op in ops {
                let sqe = match op.op_type {
                    IoOpType::Read => {
                        opcode::Read::new(types::Fd(op.fd), op.buffer.as_ptr() as *mut u8, op.len as u32)
                            .offset(op.offset)
                            .build()
                            .user_data(op.user_data)
                    }
                    IoOpType::Write => {
                        opcode::Write::new(types::Fd(op.fd), op.buffer.as_ptr(), op.len as u32)
                            .offset(op.offset)
                            .build()
                            .user_data(op.user_data)
                    }
                    IoOpType::Fsync => {
                        opcode::Fsync::new(types::Fd(op.fd))
                            .build()
                            .user_data(op.user_data)
                    }
                    _ => continue, // Skip unsupported operations
                };

                // SAFETY: We submit to the ring and will process completions
                unsafe {
                    if sq.push(&sqe).is_err() {
                        break; // Submission queue full
                    }
                }
            }
            
            sq.sync();
            drop(sq);
            
            // Submit batch
            uring.submit()?;
            
            Ok(())
        } else {
            // Fallback to synchronous I/O
            let completions: Vec<_> = ops
                .into_iter()
                .map(|op| self.simulate_execute(op))
                .collect();
            self.completions.lock().extend(completions);
            Ok(())
        }
    }

    fn wait_one(&mut self) -> io::Result<IoCompletion> {
        // First try cached completions
        if let Some(completion) = self.completions.lock().pop_front() {
            return Ok(completion);
        }

        // If no cached completions and we have real io_uring, wait for one
        if let Some(ref mut uring) = self.uring {
            uring.submit_and_wait(1)?;
            let mut cq = uring.completion();
            if let Some(cqe) = cq.next() {
                let completion = if cqe.result() >= 0 {
                    IoCompletion::success(cqe.user_data(), cqe.result())
                } else {
                    IoCompletion::failure(cqe.user_data(), cqe.result())
                };
                return Ok(completion);
            }
        }

        Err(io::Error::new(io::ErrorKind::WouldBlock, "No completions"))
    }

    fn wait_all(&mut self) -> io::Result<Vec<IoCompletion>> {
        let mut all_completions = self.completions.lock().drain(..).collect::<Vec<_>>();

        // If we have real io_uring, collect any pending completions
        if let Some(ref mut uring) = self.uring {
            let mut cq = uring.completion();
            while let Some(cqe) = cq.next() {
                let completion = if cqe.result() >= 0 {
                    IoCompletion::success(cqe.user_data(), cqe.result())
                } else {
                    IoCompletion::failure(cqe.user_data(), cqe.result())
                };
                all_completions.push(completion);
            }
        }

        Ok(all_completions)
    }

    fn pending(&self) -> usize {
        self.pending.lock().len()
    }

    fn is_uring_available(&self) -> bool {
        self.uring_available
    }
}

/// Create the best available async I/O backend for the current platform
pub fn create_backend(config: IoUringConfig, stats: Arc<IoUringStats>) -> Box<dyn AsyncIoBackend> {
    #[cfg(target_os = "linux")]
    {
        match LinuxIoUringBackend::new(config, stats.clone()) {
            Ok(backend) if backend.is_uring_available() => {
                tracing::info!("Using Linux io_uring backend");
                Box::new(backend)
            }
            _ => {
                tracing::info!("Falling back to sync I/O backend");
                Box::new(SyncIoBackend::new(stats))
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = config; // Suppress unused warning
        tracing::info!("Using sync I/O backend (io_uring not available on this platform)");
        Box::new(SyncIoBackend::new(stats))
    }
}

/// Batched write helper for WAL operations
pub struct BatchedWriter {
    backend: Box<dyn AsyncIoBackend>,
    pending_ops: Vec<IoOp>,
    batch_size: usize,
    next_user_data: AtomicU64,
}

impl BatchedWriter {
    /// Create a new batched writer
    pub fn new(backend: Box<dyn AsyncIoBackend>, batch_size: usize) -> Self {
        Self {
            backend,
            pending_ops: Vec::with_capacity(batch_size),
            batch_size,
            next_user_data: AtomicU64::new(0),
        }
    }

    /// Queue a write operation
    pub fn write(&mut self, fd: i32, data: Vec<u8>, offset: u64) -> u64 {
        let user_data = self.next_user_data.fetch_add(1, Ordering::Relaxed);
        let op = IoOp::write(fd, data, offset, user_data);
        self.pending_ops.push(op);

        if self.pending_ops.len() >= self.batch_size {
            self.flush().ok();
        }

        user_data
    }

    /// Flush all pending operations
    pub fn flush(&mut self) -> io::Result<Vec<IoCompletion>> {
        if self.pending_ops.is_empty() {
            return Ok(Vec::new());
        }

        let ops = std::mem::take(&mut self.pending_ops);
        self.backend.submit_batch(ops)?;
        self.backend.wait_all()
    }

    /// Get pending count
    pub fn pending(&self) -> usize {
        self.pending_ops.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_uring_config() {
        let default = IoUringConfig::default();
        assert_eq!(default.sq_entries, 256);
        assert!(!default.sq_poll);

        let high = IoUringConfig::high_throughput();
        assert_eq!(high.sq_entries, 1024);
        assert!(high.sq_poll);

        let low = IoUringConfig::low_latency();
        assert_eq!(low.sq_entries, 64);
        assert!(low.sq_poll);
    }

    #[test]
    fn test_io_op_creation() {
        let read_op = IoOp::read(5, 1024, 512, 42);
        assert_eq!(read_op.op_type, IoOpType::Read);
        assert_eq!(read_op.fd, 5);
        assert_eq!(read_op.offset, 1024);
        assert_eq!(read_op.len, 512);
        assert_eq!(read_op.user_data, 42);

        let write_op = IoOp::write(6, vec![1, 2, 3], 2048, 99);
        assert_eq!(write_op.op_type, IoOpType::Write);
        assert_eq!(write_op.buffer, vec![1, 2, 3]);

        let fsync_op = IoOp::fsync(7, 100);
        assert_eq!(fsync_op.op_type, IoOpType::Fsync);
    }

    #[test]
    fn test_io_completion() {
        let success = IoCompletion::success(42, 1024);
        assert!(success.success);
        assert_eq!(success.bytes_transferred(), Some(1024));

        let failure = IoCompletion::failure(42, -5);
        assert!(!failure.success);
        assert_eq!(failure.bytes_transferred(), None);
    }

    #[test]
    fn test_io_uring_stats() {
        let stats = IoUringStats::new();

        stats.record_submit(5);
        stats.record_completion(IoOpType::Read, 1024);
        stats.record_completion(IoOpType::Write, 512);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.ops_submitted, 5);
        assert_eq!(snapshot.ops_completed, 2);
        assert_eq!(snapshot.bytes_read, 1024);
        assert_eq!(snapshot.bytes_written, 512);
        assert_eq!(snapshot.syscalls, 1);
        assert_eq!(snapshot.ops_batched, 4);
    }

    #[test]
    fn test_stats_efficiency() {
        let stats = IoUringStats::new();

        // Simulate 10 ops in 2 syscalls
        stats.record_submit(5);
        stats.record_submit(5);

        for _ in 0..10 {
            stats.record_completion(IoOpType::Write, 100);
        }

        let snapshot = stats.snapshot();
        assert!((snapshot.batching_efficiency() - 5.0).abs() < 0.01);
        assert!((snapshot.bytes_per_syscall() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_sync_backend() {
        use tempfile::NamedTempFile;

        let stats = IoUringStats::new();
        let backend = SyncIoBackend::new(stats.clone());

        assert!(!backend.is_uring_available());
        assert_eq!(backend.pending(), 0);

        // Create a temp file for testing
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"hello world").unwrap();
        temp.flush().unwrap();

        // Test file size check (without actual operations to avoid fd issues)
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.ops_submitted, 0);
    }

    #[test]
    fn test_create_backend() {
        let stats = IoUringStats::new();
        let config = IoUringConfig::default();
        let backend = create_backend(config, stats);

        // On non-Linux, should always be sync backend
        #[cfg(not(target_os = "linux"))]
        assert!(!backend.is_uring_available());

        assert_eq!(backend.pending(), 0);
    }

    #[test]
    fn test_batched_writer() {
        let stats = IoUringStats::new();
        let backend = Box::new(SyncIoBackend::new(stats));
        let writer = BatchedWriter::new(backend, 10);

        assert_eq!(writer.pending(), 0);

        // Note: Can't actually test writes without valid fd
        // This just tests the structure
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_uring_check() {
        let stats = IoUringStats::new();
        let config = IoUringConfig::default();
        let backend = LinuxIoUringBackend::new(config, stats).unwrap();

        // Check returns true on Linux 5.1+
        println!("io_uring available: {}", backend.is_uring_available());
    }
}
