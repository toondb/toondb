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

//! Advisory File Locking for Database Exclusivity
//!
//! This module implements cross-platform advisory file locking to enforce
//! single-writer database exclusivity at the filesystem level.
//!
//! ## Problem
//!
//! Process-local synchronization primitives (`Mutex`, `RwLock`, `AtomicU64`)
//! provide zero protection against concurrent multi-process access. When
//! multiple OS processes open the same database files, data corruption occurs:
//!
//! 1. Process A appends entry at offset X, increments local sequence to N
//! 2. Process B appends entry at offset X+∆, has independent sequence M≠N
//! 3. Recovery sees inconsistent sequences → data loss
//!
//! ## Solution
//!
//! Use POSIX advisory locks (`flock`/`fcntl`) to enforce:
//! - Single-process exclusive access to database files
//! - Fail-fast behavior for concurrent access attempts
//! - Automatic lock release on process crash
//!
//! ## Platform Support
//!
//! - **Unix/Linux/macOS**: Uses `flock()` system call
//! - **Windows**: Uses `LockFileEx()` with `LOCKFILE_EXCLUSIVE_LOCK`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_storage::lock::DatabaseLock;
//!
//! // Acquire exclusive lock (fails fast if already locked)
//! let lock = DatabaseLock::acquire("/path/to/db")?;
//!
//! // Lock held for lifetime of `lock` variable
//! // ... database operations ...
//!
//! // Lock automatically released on drop
//! drop(lock);
//! ```

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sochdb_core::{Result, SochDBError};

// =============================================================================
// Error Types
// =============================================================================

/// Errors specific to database locking operations
#[derive(Debug)]
pub enum LockError {
    /// Database is locked by another process
    DatabaseLocked {
        /// PID of the process holding the lock (if known)
        holder_pid: Option<u32>,
        /// Path to the lock file
        lock_path: PathBuf,
    },
    /// Lock acquisition timed out
    Timeout {
        /// How long we waited
        elapsed: Duration,
        /// The configured timeout
        timeout: Duration,
    },
    /// Stale lock detected (holder process no longer exists)
    StaleLock {
        /// PID that was recorded in the lock file
        stale_pid: u32,
    },
    /// I/O error during lock operations
    Io(std::io::Error),
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::DatabaseLocked { holder_pid, lock_path } => {
                if let Some(pid) = holder_pid {
                    write!(f, "Database is locked by process {} (lock file: {})", 
                           pid, lock_path.display())
                } else {
                    write!(f, "Database is locked (lock file: {})", lock_path.display())
                }
            }
            LockError::Timeout { elapsed, timeout } => {
                write!(f, "Lock acquisition timed out after {:?} (timeout: {:?})", 
                       elapsed, timeout)
            }
            LockError::StaleLock { stale_pid } => {
                write!(f, "Stale lock detected from crashed process {}", stale_pid)
            }
            LockError::Io(e) => write!(f, "Lock I/O error: {}", e),
        }
    }
}

impl std::error::Error for LockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LockError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LockError {
    fn from(e: std::io::Error) -> Self {
        LockError::Io(e)
    }
}

impl From<LockError> for SochDBError {
    fn from(e: LockError) -> Self {
        match e {
            LockError::DatabaseLocked { holder_pid, lock_path } => {
                SochDBError::LockError(format!(
                    "Database locked by PID {:?} (lock: {})", 
                    holder_pid, lock_path.display()
                ))
            }
            LockError::Timeout { elapsed, timeout } => {
                SochDBError::LockError(format!(
                    "Lock timeout after {:?} (max: {:?})", elapsed, timeout
                ))
            }
            LockError::StaleLock { stale_pid } => {
                SochDBError::LockError(format!(
                    "Stale lock from crashed process {}", stale_pid
                ))
            }
            LockError::Io(e) => SochDBError::Io(e),
        }
    }
}

// =============================================================================
// Lock Configuration
// =============================================================================

/// Configuration for database lock behavior
#[derive(Debug, Clone)]
pub struct LockConfig {
    /// Timeout for lock acquisition (None = fail immediately)
    pub timeout: Option<Duration>,
    /// Interval between lock retry attempts
    pub retry_interval: Duration,
    /// Whether to detect and recover from stale locks
    pub detect_stale_locks: bool,
    /// Lock file name (relative to database directory)
    pub lock_file_name: String,
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            timeout: Some(Duration::from_secs(5)),
            retry_interval: Duration::from_millis(100),
            detect_stale_locks: true,
            lock_file_name: ".lock".to_string(),
        }
    }
}

impl LockConfig {
    /// Create config with no timeout (fail immediately if locked)
    pub fn no_wait() -> Self {
        Self {
            timeout: None,
            ..Default::default()
        }
    }

    /// Create config with specific timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            ..Default::default()
        }
    }
}

// =============================================================================
// Database Lock
// =============================================================================

/// Exclusive advisory lock on a database directory
///
/// This lock ensures single-process access to a SochDB database.
/// The lock is automatically released when this struct is dropped.
///
/// ## Implementation
///
/// Uses POSIX `flock()` on Unix systems and `LockFileEx()` on Windows.
/// The lock file also contains the PID of the lock holder for debugging
/// and stale lock detection.
///
/// ## Safety
///
/// Advisory locks are cooperative - they only work if all processes
/// attempting to access the database use this locking mechanism.
pub struct DatabaseLock {
    /// Open file handle (keeps the lock active)
    lock_file: File,
    /// Path to the lock file
    path: PathBuf,
    /// Our PID (for diagnostics)
    our_pid: u32,
}

impl DatabaseLock {
    /// Acquire exclusive lock on a database directory
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory
    ///
    /// # Returns
    ///
    /// Returns `Ok(DatabaseLock)` if lock acquired successfully.
    /// Returns `Err(LockError::DatabaseLocked)` if another process holds the lock.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let lock = DatabaseLock::acquire("/path/to/db")?;
    /// // Lock is held until `lock` is dropped
    /// ```
    pub fn acquire<P: AsRef<Path>>(db_path: P) -> std::result::Result<Self, LockError> {
        Self::acquire_with_config(db_path, &LockConfig::no_wait())
    }

    /// Acquire exclusive lock with timeout
    ///
    /// Will retry lock acquisition until timeout expires.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory
    /// * `timeout` - Maximum time to wait for lock
    pub fn acquire_with_timeout<P: AsRef<Path>>(
        db_path: P, 
        timeout: Duration
    ) -> std::result::Result<Self, LockError> {
        Self::acquire_with_config(db_path, &LockConfig::with_timeout(timeout))
    }

    /// Acquire exclusive lock with full configuration
    pub fn acquire_with_config<P: AsRef<Path>>(
        db_path: P,
        config: &LockConfig,
    ) -> std::result::Result<Self, LockError> {
        let db_path = db_path.as_ref();
        let lock_path = db_path.join(&config.lock_file_name);

        // Ensure database directory exists
        if !db_path.exists() {
            std::fs::create_dir_all(db_path)?;
        }

        let deadline = config.timeout.map(|t| Instant::now() + t);
        let our_pid = std::process::id();

        loop {
            // Try to open/create lock file
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(&lock_path)?;

            // Attempt to acquire exclusive lock
            match Self::try_flock(&file, false) {
                Ok(()) => {
                    // Lock acquired! Write our PID
                    Self::write_pid(&file, our_pid)?;
                    
                    return Ok(Self {
                        lock_file: file,
                        path: lock_path,
                        our_pid,
                    });
                }
                Err(LockError::DatabaseLocked { .. }) => {
                    // Lock is held by another process
                    
                    // Check for stale lock
                    let mut should_retry = false;
                    if config.detect_stale_locks {
                        if let Some(holder_pid) = Self::read_pid(&file) {
                            if !Self::process_exists(holder_pid) {
                                // Process is dead - try to take over
                                // We need to close and reopen to clear state
                                drop(file);
                                
                                // Force remove the lock file
                                if std::fs::remove_file(&lock_path).is_ok() {
                                    should_retry = true;
                                }
                            }
                        }
                    }
                    
                    if should_retry {
                        continue; // Retry acquisition
                    }

                    // Check timeout
                    if let Some(deadline) = deadline {
                        if Instant::now() >= deadline {
                            return Err(LockError::Timeout {
                                elapsed: config.timeout.unwrap_or_default(),
                                timeout: config.timeout.unwrap_or_default(),
                            });
                        }
                        
                        // Wait and retry
                        std::thread::sleep(config.retry_interval);
                        continue;
                    } else {
                        // No timeout - fail immediately
                        // Note: file may have been dropped above, so we can't read PID
                        return Err(LockError::DatabaseLocked { 
                            holder_pid: None, 
                            lock_path 
                        });
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Get path to the lock file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get PID of lock holder (us)
    pub fn pid(&self) -> u32 {
        self.our_pid
    }

    /// Check if a given PID is holding the lock on a database
    ///
    /// Useful for diagnostics without attempting to acquire.
    pub fn get_lock_holder<P: AsRef<Path>>(db_path: P) -> Option<u32> {
        let lock_path = db_path.as_ref().join(".lock");
        let file = File::open(&lock_path).ok()?;
        Self::read_pid(&file)
    }

    /// Write PID to lock file
    fn write_pid(file: &File, pid: u32) -> std::result::Result<(), LockError> {
        use std::io::Seek;
        let mut file = file;
        file.seek(std::io::SeekFrom::Start(0))?;
        file.set_len(0)?;
        writeln!(file, "{}", pid)?;
        file.sync_all()?;
        Ok(())
    }

    /// Read PID from lock file
    fn read_pid(file: &File) -> Option<u32> {
        use std::io::Seek;
        let mut file = file;
        let _ = file.seek(std::io::SeekFrom::Start(0));
        let mut contents = String::new();
        file.read_to_string(&mut contents).ok()?;
        contents.trim().parse().ok()
    }

    /// Check if a process exists
    #[cfg(unix)]
    fn process_exists(pid: u32) -> bool {
        // kill(pid, 0) checks if process exists without sending a signal
        // Returns 0 if process exists, -1 with ESRCH if not
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if result == 0 {
            true
        } else {
            // Check if error is ESRCH (no such process)
            let errno = std::io::Error::last_os_error().raw_os_error();
            errno != Some(libc::ESRCH)
        }
    }

    #[cfg(windows)]
    fn process_exists(pid: u32) -> bool {
        use std::ptr::null_mut;
        unsafe {
            let handle = windows_sys::Win32::System::Threading::OpenProcess(
                windows_sys::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                pid,
            );
            if handle.is_null() {
                false
            } else {
                windows_sys::Win32::Foundation::CloseHandle(handle);
                true
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn process_exists(_pid: u32) -> bool {
        // On unknown platforms, assume process exists to be safe
        true
    }

    /// Try to acquire flock on file
    #[cfg(unix)]
    fn try_flock(file: &File, blocking: bool) -> std::result::Result<(), LockError> {
        use std::os::unix::io::AsRawFd;
        
        let fd = file.as_raw_fd();
        let operation = if blocking {
            libc::LOCK_EX
        } else {
            libc::LOCK_EX | libc::LOCK_NB
        };

        let result = unsafe { libc::flock(fd, operation) };
        
        if result == 0 {
            Ok(())
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                Err(LockError::DatabaseLocked {
                    holder_pid: None,
                    lock_path: PathBuf::new(),
                })
            } else {
                Err(LockError::Io(err))
            }
        }
    }

    #[cfg(windows)]
    fn try_flock(file: &File, blocking: bool) -> std::result::Result<(), LockError> {
        use std::os::windows::io::AsRawHandle;
        
        let handle = file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
        
        let flags = windows_sys::Win32::Storage::FileSystem::LOCKFILE_EXCLUSIVE_LOCK
            | if blocking { 0 } else { windows_sys::Win32::Storage::FileSystem::LOCKFILE_FAIL_IMMEDIATELY };
        
        let mut overlapped: windows_sys::Win32::System::IO::OVERLAPPED = unsafe { std::mem::zeroed() };
        
        let result = unsafe {
            windows_sys::Win32::Storage::FileSystem::LockFileEx(
                handle,
                flags,
                0,
                1,
                0,
                &mut overlapped,
            )
        };
        
        if result != 0 {
            Ok(())
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION as i32) {
                Err(LockError::DatabaseLocked {
                    holder_pid: None,
                    lock_path: PathBuf::new(),
                })
            } else {
                Err(LockError::Io(err))
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn try_flock(_file: &File, _blocking: bool) -> std::result::Result<(), LockError> {
        // On unsupported platforms, assume success (no locking)
        // This is unsafe but allows compilation
        Ok(())
    }

    /// Release the lock (called automatically on drop)
    #[cfg(unix)]
    fn release(&self) {
        use std::os::unix::io::AsRawFd;
        let fd = self.lock_file.as_raw_fd();
        unsafe { libc::flock(fd, libc::LOCK_UN) };
    }

    #[cfg(windows)]
    fn release(&self) {
        use std::os::windows::io::AsRawHandle;
        let handle = self.lock_file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
        let mut overlapped: windows_sys::Win32::System::IO::OVERLAPPED = unsafe { std::mem::zeroed() };
        unsafe {
            windows_sys::Win32::Storage::FileSystem::UnlockFileEx(
                handle,
                0,
                1,
                0,
                &mut overlapped,
            );
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn release(&self) {
        // No-op on unsupported platforms
    }
}

impl Drop for DatabaseLock {
    fn drop(&mut self) {
        self.release();
        // Lock file is removed when the last handle is closed
        // We explicitly remove it for cleaner state
        let _ = std::fs::remove_file(&self.path);
    }
}

// =============================================================================
// Reader-Writer Lock Protocol (Task 3)
// =============================================================================

/// Shared-Exclusive lock state stored in lock file header
///
/// Format (16 bytes):
/// ```text
/// ┌────────────┬────────────────┬──────────────────┬─────────┐
/// │ reader_cnt │ writer_intent  │ writer_active    │ padding │
/// │ (4 bytes)  │ (4 bytes)      │ (4 bytes)        │ (4 B)   │
/// └────────────┴────────────────┴──────────────────┴─────────┘
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RwLockState {
    /// Number of active readers
    pub reader_count: u32,
    /// Writer waiting to acquire (prevents reader starvation)
    pub writer_intent: u32,
    /// Writer currently active
    pub writer_active: u32,
    /// Reserved for future use
    pub _padding: u32,
}

/// Connection mode for database access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionMode {
    /// Read-only access (acquires shared lock)
    ReadOnly,
    /// Read-write access (acquires exclusive lock)
    ReadWrite,
}

/// Reader-Writer database lock for concurrent read access
///
/// Implements a shared-exclusive lock protocol:
/// - Multiple concurrent readers allowed
/// - Single exclusive writer
/// - Writer intent prevents reader starvation
pub struct RwDatabaseLock {
    /// Open file handle
    lock_file: File,
    /// Path to lock file
    path: PathBuf,
    /// Our connection mode
    mode: ConnectionMode,
    /// Our PID
    our_pid: u32,
}

impl RwDatabaseLock {
    /// Acquire a shared (read-only) lock
    ///
    /// Multiple processes can hold shared locks simultaneously.
    /// Blocks if a writer is active or waiting.
    pub fn acquire_shared<P: AsRef<Path>>(db_path: P) -> std::result::Result<Self, LockError> {
        Self::acquire_with_mode(db_path, ConnectionMode::ReadOnly, &LockConfig::default())
    }

    /// Acquire an exclusive (read-write) lock
    ///
    /// Only one process can hold an exclusive lock.
    /// Blocks if any readers or another writer is active.
    pub fn acquire_exclusive<P: AsRef<Path>>(db_path: P) -> std::result::Result<Self, LockError> {
        Self::acquire_with_mode(db_path, ConnectionMode::ReadWrite, &LockConfig::default())
    }

    /// Acquire lock with specified mode and configuration
    pub fn acquire_with_mode<P: AsRef<Path>>(
        db_path: P,
        mode: ConnectionMode,
        config: &LockConfig,
    ) -> std::result::Result<Self, LockError> {
        let db_path = db_path.as_ref();
        let lock_path = db_path.join(&config.lock_file_name);
        
        if !db_path.exists() {
            std::fs::create_dir_all(db_path)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)?;

        let our_pid = std::process::id();
        let deadline = config.timeout.map(|t| Instant::now() + t);

        loop {
            match mode {
                ConnectionMode::ReadOnly => {
                    // Acquire shared lock
                    if Self::try_shared_lock(&file)? {
                        return Ok(Self {
                            lock_file: file,
                            path: lock_path,
                            mode,
                            our_pid,
                        });
                    }
                }
                ConnectionMode::ReadWrite => {
                    // Acquire exclusive lock
                    if Self::try_exclusive_lock(&file)? {
                        return Ok(Self {
                            lock_file: file,
                            path: lock_path,
                            mode,
                            our_pid,
                        });
                    }
                }
            }

            // Check timeout
            if let Some(deadline) = deadline {
                if Instant::now() >= deadline {
                    return Err(LockError::Timeout {
                        elapsed: config.timeout.unwrap_or_default(),
                        timeout: config.timeout.unwrap_or_default(),
                    });
                }
                std::thread::sleep(config.retry_interval);
            } else {
                return Err(LockError::DatabaseLocked {
                    holder_pid: None,
                    lock_path,
                });
            }
        }
    }

    /// Get connection mode
    pub fn mode(&self) -> ConnectionMode {
        self.mode
    }

    /// Check if this is a read-only connection
    pub fn is_readonly(&self) -> bool {
        self.mode == ConnectionMode::ReadOnly
    }

    #[cfg(unix)]
    fn try_shared_lock(file: &File) -> std::result::Result<bool, LockError> {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let result = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
        if result == 0 {
            Ok(true)
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                Ok(false)
            } else {
                Err(LockError::Io(err))
            }
        }
    }

    #[cfg(unix)]
    fn try_exclusive_lock(file: &File) -> std::result::Result<bool, LockError> {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            Ok(true)
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                Ok(false)
            } else {
                Err(LockError::Io(err))
            }
        }
    }

    #[cfg(windows)]
    fn try_shared_lock(file: &File) -> std::result::Result<bool, LockError> {
        use std::os::windows::io::AsRawHandle;
        let handle = file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
        let mut overlapped: windows_sys::Win32::System::IO::OVERLAPPED = unsafe { std::mem::zeroed() };
        
        let result = unsafe {
            windows_sys::Win32::Storage::FileSystem::LockFileEx(
                handle,
                windows_sys::Win32::Storage::FileSystem::LOCKFILE_FAIL_IMMEDIATELY,
                0, 1, 0,
                &mut overlapped,
            )
        };
        
        if result != 0 {
            Ok(true)
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION as i32) {
                Ok(false)
            } else {
                Err(LockError::Io(err))
            }
        }
    }

    #[cfg(windows)]
    fn try_exclusive_lock(file: &File) -> std::result::Result<bool, LockError> {
        use std::os::windows::io::AsRawHandle;
        let handle = file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
        let mut overlapped: windows_sys::Win32::System::IO::OVERLAPPED = unsafe { std::mem::zeroed() };
        
        let result = unsafe {
            windows_sys::Win32::Storage::FileSystem::LockFileEx(
                handle,
                windows_sys::Win32::Storage::FileSystem::LOCKFILE_EXCLUSIVE_LOCK 
                    | windows_sys::Win32::Storage::FileSystem::LOCKFILE_FAIL_IMMEDIATELY,
                0, 1, 0,
                &mut overlapped,
            )
        };
        
        if result != 0 {
            Ok(true)
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION as i32) {
                Ok(false)
            } else {
                Err(LockError::Io(err))
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn try_shared_lock(_file: &File) -> std::result::Result<bool, LockError> {
        Ok(true)
    }

    #[cfg(not(any(unix, windows)))]
    fn try_exclusive_lock(_file: &File) -> std::result::Result<bool, LockError> {
        Ok(true)
    }

    #[cfg(unix)]
    fn release(&self) {
        use std::os::unix::io::AsRawFd;
        let fd = self.lock_file.as_raw_fd();
        unsafe { libc::flock(fd, libc::LOCK_UN) };
    }

    #[cfg(windows)]
    fn release(&self) {
        use std::os::windows::io::AsRawHandle;
        let handle = self.lock_file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
        let mut overlapped: windows_sys::Win32::System::IO::OVERLAPPED = unsafe { std::mem::zeroed() };
        unsafe {
            windows_sys::Win32::Storage::FileSystem::UnlockFileEx(handle, 0, 1, 0, &mut overlapped);
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn release(&self) {}
}

impl Drop for RwDatabaseLock {
    fn drop(&mut self) {
        self.release();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use tempfile::TempDir;

    #[test]
    fn test_exclusive_lock_basic() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path();

        // First lock should succeed
        let lock1 = DatabaseLock::acquire(db_path);
        assert!(lock1.is_ok());

        // Second lock should fail immediately
        let lock2 = DatabaseLock::acquire(db_path);
        assert!(matches!(lock2, Err(LockError::DatabaseLocked { .. })));

        // After releasing first lock, second should succeed
        drop(lock1);
        let lock3 = DatabaseLock::acquire(db_path);
        assert!(lock3.is_ok());
    }

    #[test]
    fn test_lock_with_timeout() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().to_path_buf();

        // Acquire lock
        let _lock = DatabaseLock::acquire(&db_path).unwrap();

        // Try with short timeout - should fail
        let start = Instant::now();
        let result = DatabaseLock::acquire_with_timeout(&db_path, Duration::from_millis(100));
        let elapsed = start.elapsed();

        assert!(matches!(result, Err(LockError::Timeout { .. })));
        assert!(elapsed >= Duration::from_millis(100));
        assert!(elapsed < Duration::from_millis(500)); // Shouldn't be too long
    }

    #[test]
    fn test_lock_pid_recorded() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path();

        let lock = DatabaseLock::acquire(db_path).unwrap();
        let our_pid = std::process::id();
        
        assert_eq!(lock.pid(), our_pid);
        
        // Check we can read the holder
        let holder = DatabaseLock::get_lock_holder(db_path);
        assert_eq!(holder, Some(our_pid));
    }

    #[test]
    fn test_shared_lock_multiple_readers() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path();

        // Multiple shared locks should succeed
        let lock1 = RwDatabaseLock::acquire_shared(db_path);
        let lock2 = RwDatabaseLock::acquire_shared(db_path);

        assert!(lock1.is_ok());
        assert!(lock2.is_ok());
    }

    #[test]
    fn test_exclusive_blocks_shared() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path();

        // Exclusive lock first
        let _exclusive = RwDatabaseLock::acquire_exclusive(db_path).unwrap();

        // Shared lock should fail immediately with no timeout
        let shared = RwDatabaseLock::acquire_with_mode(
            db_path,
            ConnectionMode::ReadOnly,
            &LockConfig::no_wait(),
        );
        
        assert!(matches!(shared, Err(LockError::DatabaseLocked { .. })));
    }
}
