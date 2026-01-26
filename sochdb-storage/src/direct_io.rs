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

//! Direct I/O Support for Cache-Bypass Scenarios
//!
//! Implements O_DIRECT path for SSTable reads to prevent page cache pollution
//! during large scans, compaction, and backup operations.
//!
//! ## jj.md Task 14: Direct I/O
//!
//! Goals:
//! - Prevent cache pollution from large scans
//! - Predictable latency (no page cache contention)
//! - Better control of memory usage
//!
//! ## When to Use Direct I/O
//!
//! - Full table scans (would evict entire cache)
//! - Compaction reads (data processed once)
//! - Backup operations (one-time sequential read)
//! - User-configured per-query
//!
//! ## Platform Support
//!
//! - Linux: O_DIRECT flag
//! - macOS: F_NOCACHE via fcntl
//! - Windows: FILE_FLAG_NO_BUFFERING (not implemented yet)

use std::alloc::{Layout, alloc, dealloc};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Alignment required for Direct I/O
#[cfg(target_os = "linux")]
pub const DIRECT_IO_ALIGNMENT: usize = 512;

#[cfg(target_os = "macos")]
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

/// Page size for Direct I/O buffers
pub const PAGE_SIZE: usize = 4096;

/// Direct I/O mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DirectIoMode {
    /// Use buffered I/O (default)
    #[default]
    Buffered,
    /// Use Direct I/O (bypass page cache)
    Direct,
    /// Auto-detect based on access pattern
    Auto,
}

/// Configuration for Direct I/O operations
#[derive(Debug, Clone)]
pub struct DirectIoConfig {
    /// I/O mode
    pub mode: DirectIoMode,
    /// Buffer size for Direct I/O reads (must be aligned)
    pub buffer_size: usize,
    /// Threshold for switching to Direct I/O in Auto mode (bytes)
    pub auto_threshold: usize,
}

impl Default for DirectIoConfig {
    fn default() -> Self {
        Self {
            mode: DirectIoMode::Buffered,
            buffer_size: 256 * 1024,          // 256KB
            auto_threshold: 64 * 1024 * 1024, // 64MB
        }
    }
}

impl DirectIoConfig {
    /// Create a Direct I/O config
    pub fn direct() -> Self {
        Self {
            mode: DirectIoMode::Direct,
            ..Default::default()
        }
    }

    /// Create an auto-detect config
    pub fn auto() -> Self {
        Self {
            mode: DirectIoMode::Auto,
            ..Default::default()
        }
    }
}

/// Aligned buffer for Direct I/O operations
///
/// Direct I/O requires buffers to be aligned to the filesystem block size
/// (typically 512 bytes or 4KB).
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl AlignedBuffer {
    /// Create a new aligned buffer
    pub fn new(capacity: usize) -> Self {
        Self::with_alignment(capacity, DIRECT_IO_ALIGNMENT)
    }

    /// Create with specific alignment
    pub fn with_alignment(capacity: usize, alignment: usize) -> Self {
        // Round up capacity to alignment boundary
        let aligned_capacity = capacity.div_ceil(alignment) * alignment;

        let layout =
            Layout::from_size_align(aligned_capacity, alignment).expect("Invalid alignment");

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Failed to allocate aligned buffer");
        }

        Self {
            ptr,
            len: 0,
            capacity: aligned_capacity,
            alignment,
        }
    }

    /// Get the buffer contents
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get mutable buffer contents
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }

    /// Set the length of valid data
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity);
        self.len = len;
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Check if pointer is properly aligned
    pub fn is_aligned(&self) -> bool {
        (self.ptr as usize).is_multiple_of(self.alignment)
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, self.alignment)
            .expect("Invalid alignment in drop");
        unsafe { dealloc(self.ptr, layout) };
    }
}

// AlignedBuffer is Send + Sync because the pointer is exclusively owned
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

/// Open a file with Direct I/O enabled (if supported)
#[cfg(target_os = "linux")]
pub fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

#[cfg(target_os = "macos")]
pub fn open_direct(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new().read(true).open(path)?;

    // On macOS, we use F_NOCACHE to disable caching
    unsafe {
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        if libc::fcntl(fd, libc::F_NOCACHE, 1) == -1 {
            return Err(io::Error::last_os_error());
        }
    }

    Ok(file)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn open_direct(path: &Path) -> io::Result<File> {
    // Fallback to buffered I/O on unsupported platforms
    OpenOptions::new().read(true).open(path)
}

/// Open a file for Direct I/O writing
#[cfg(target_os = "linux")]
pub fn open_direct_write(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

#[cfg(target_os = "macos")]
pub fn open_direct_write(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    unsafe {
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        if libc::fcntl(fd, libc::F_NOCACHE, 1) == -1 {
            return Err(io::Error::last_os_error());
        }
    }

    Ok(file)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn open_direct_write(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

/// Direct I/O reader with aligned buffers
pub struct DirectReader {
    file: File,
    buffer: AlignedBuffer,
    file_offset: u64,
    buffer_offset: usize,
    buffer_valid: usize,
    file_size: u64,
}

impl DirectReader {
    /// Create a new Direct I/O reader
    pub fn open(path: &Path, buffer_size: usize) -> io::Result<Self> {
        let file = open_direct(path)?;
        let file_size = file.metadata()?.len();

        Ok(Self {
            file,
            buffer: AlignedBuffer::new(buffer_size),
            file_offset: 0,
            buffer_offset: 0,
            buffer_valid: 0,
            file_size,
        })
    }

    /// Create from an existing file
    pub fn from_file(file: File, buffer_size: usize) -> io::Result<Self> {
        let file_size = file.metadata()?.len();

        Ok(Self {
            file,
            buffer: AlignedBuffer::new(buffer_size),
            file_offset: 0,
            buffer_offset: 0,
            buffer_valid: 0,
            file_size,
        })
    }

    /// Get the file size
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Refill the buffer from the file
    fn refill_buffer(&mut self) -> io::Result<usize> {
        // Align the file offset
        let aligned_offset =
            (self.file_offset / DIRECT_IO_ALIGNMENT as u64) * DIRECT_IO_ALIGNMENT as u64;
        let skip = (self.file_offset - aligned_offset) as usize;

        self.file.seek(SeekFrom::Start(aligned_offset))?;

        let buf = self.buffer.as_mut_slice();
        let bytes_read = self.file.read(buf)?;

        self.buffer.set_len(bytes_read);
        self.buffer_valid = bytes_read;
        self.buffer_offset = skip;

        Ok(bytes_read.saturating_sub(skip))
    }
}

impl Read for DirectReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut total_read = 0;

        while total_read < buf.len() {
            // Check if we have data in buffer
            let available = self.buffer_valid.saturating_sub(self.buffer_offset);

            if available == 0 {
                // Buffer exhausted, refill
                if self.file_offset >= self.file_size {
                    break; // EOF
                }
                let refilled = self.refill_buffer()?;
                if refilled == 0 {
                    break; // EOF
                }
                continue;
            }

            // Copy from buffer
            let to_copy = (buf.len() - total_read).min(available);
            let src = &self.buffer.as_slice()[self.buffer_offset..self.buffer_offset + to_copy];
            buf[total_read..total_read + to_copy].copy_from_slice(src);

            self.buffer_offset += to_copy;
            self.file_offset += to_copy as u64;
            total_read += to_copy;
        }

        Ok(total_read)
    }
}

impl Seek for DirectReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p,
            SeekFrom::End(p) => {
                if p >= 0 {
                    self.file_size + p as u64
                } else {
                    self.file_size.saturating_sub((-p) as u64)
                }
            }
            SeekFrom::Current(p) => {
                if p >= 0 {
                    self.file_offset + p as u64
                } else {
                    self.file_offset.saturating_sub((-p) as u64)
                }
            }
        };

        // Invalidate buffer if seeking outside buffered range
        let buffer_start = self.file_offset - self.buffer_offset as u64;
        let buffer_end = buffer_start + self.buffer_valid as u64;

        if new_pos < buffer_start || new_pos >= buffer_end {
            self.buffer_offset = 0;
            self.buffer_valid = 0;
        } else {
            self.buffer_offset = (new_pos - buffer_start) as usize;
        }

        self.file_offset = new_pos;
        Ok(new_pos)
    }
}

/// Statistics for Direct I/O operations
#[derive(Debug, Default, Clone)]
pub struct DirectIoStats {
    /// Total bytes read with Direct I/O
    pub direct_bytes_read: u64,
    /// Total bytes read with buffered I/O
    pub buffered_bytes_read: u64,
    /// Total bytes written with Direct I/O
    pub direct_bytes_written: u64,
    /// Total bytes written with buffered I/O
    pub buffered_bytes_written: u64,
    /// Number of Direct I/O reads
    pub direct_reads: u64,
    /// Number of buffered reads
    pub buffered_reads: u64,
}

impl DirectIoStats {
    /// Record a Direct I/O read
    pub fn record_direct_read(&mut self, bytes: u64) {
        self.direct_bytes_read += bytes;
        self.direct_reads += 1;
    }

    /// Record a buffered read
    pub fn record_buffered_read(&mut self, bytes: u64) {
        self.buffered_bytes_read += bytes;
        self.buffered_reads += 1;
    }

    /// Get total bytes read
    pub fn total_bytes_read(&self) -> u64 {
        self.direct_bytes_read + self.buffered_bytes_read
    }

    /// Get Direct I/O ratio
    pub fn direct_io_ratio(&self) -> f64 {
        let total = self.total_bytes_read();
        if total == 0 {
            0.0
        } else {
            self.direct_bytes_read as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_aligned_buffer() {
        let buf = AlignedBuffer::new(1024);
        assert!(buf.is_aligned());
        assert!(buf.capacity() >= 1024);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_aligned_buffer_write_read() {
        let mut buf = AlignedBuffer::new(100);
        {
            let slice = buf.as_mut_slice();
            for (i, item) in slice.iter_mut().enumerate().take(50) {
                *item = i as u8;
            }
        }
        buf.set_len(50);

        assert_eq!(buf.len(), 50);
        assert_eq!(buf.as_slice()[0], 0);
        assert_eq!(buf.as_slice()[49], 49);
    }

    #[test]
    fn test_direct_io_config() {
        let default = DirectIoConfig::default();
        assert_eq!(default.mode, DirectIoMode::Buffered);

        let direct = DirectIoConfig::direct();
        assert_eq!(direct.mode, DirectIoMode::Direct);

        let auto = DirectIoConfig::auto();
        assert_eq!(auto.mode, DirectIoMode::Auto);
    }

    #[test]
    fn test_direct_reader() {
        // Create a temporary file with some data
        let mut temp = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        temp.write_all(&data).unwrap();
        temp.flush().unwrap();

        // Read with Direct I/O
        let mut reader = DirectReader::open(temp.path(), 4096).unwrap();
        let mut read_buf = vec![0u8; 10000];
        let bytes_read = reader.read(&mut read_buf).unwrap();

        assert_eq!(bytes_read, 10000);
        assert_eq!(read_buf, data);
    }

    #[test]
    fn test_direct_reader_seek() {
        let mut temp = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        temp.write_all(&data).unwrap();
        temp.flush().unwrap();

        let mut reader = DirectReader::open(temp.path(), 4096).unwrap();

        // Seek to middle
        reader.seek(SeekFrom::Start(5000)).unwrap();

        let mut read_buf = vec![0u8; 100];
        reader.read_exact(&mut read_buf).unwrap();

        // Verify we read the right data
        assert_eq!(read_buf, data[5000..5100]);
    }

    #[test]
    fn test_direct_io_stats() {
        let mut stats = DirectIoStats::default();

        stats.record_direct_read(1000);
        stats.record_buffered_read(500);

        assert_eq!(stats.total_bytes_read(), 1500);
        assert!((stats.direct_io_ratio() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_open_direct() {
        let temp = NamedTempFile::new().unwrap();
        let result = open_direct(temp.path());
        assert!(result.is_ok());
    }
}
