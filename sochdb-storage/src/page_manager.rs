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

//! Page-Based File Layout with Database Header (Task 8)
//!
//! Implements proper database file format with:
//! - Magic header on page 0
//! - Schema catalog page reference
//! - Free list management
//! - O(1) page allocation
//!
//! ## File Layout
//!
//! ```text
//! Page 0: DbHeader (128 bytes reserved)
//! Page 1: Schema Catalog (root of catalog B-tree)
//! Page 2+: Data pages (column groups, indexes)
//! ```
//!
//! ## Space Amplification Target
//!
//! SA = file_size / logical_data_size < 1.3 (30% overhead)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Default page size (4KB)
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// Magic bytes for SochDB files
pub const SOCHDB_MAGIC: [u8; 4] = *b"TOON";

/// Current format version
pub const FORMAT_VERSION: u32 = 1;

/// Page ID type
pub type PageId = u64;

/// Database header (stored on page 0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbHeader {
    /// Magic bytes "TOON"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Page size in bytes
    pub page_size: u32,
    /// Page ID of schema catalog root
    pub schema_page: PageId,
    /// First free page ID (head of free list)
    pub free_list_head: PageId,
    /// Total number of allocated pages
    pub total_pages: u64,
    /// Database creation timestamp (microseconds)
    pub created_us: u64,
    /// Last modified timestamp (microseconds)
    pub modified_us: u64,
    /// Header checksum
    pub checksum: u32,
}

impl DbHeader {
    /// Size of header in bytes
    pub const SIZE: usize = 128;

    /// Create a new database header
    pub fn new(page_size: u32) -> Self {
        let now = now_micros();
        let mut header = Self {
            magic: SOCHDB_MAGIC,
            version: FORMAT_VERSION,
            page_size,
            schema_page: 1,    // Page 1 is always schema catalog
            free_list_head: 0, // No free pages initially
            total_pages: 2,    // Header + Schema catalog
            created_us: now,
            modified_us: now,
            checksum: 0,
        };
        header.checksum = header.compute_checksum();
        header
    }

    /// Compute checksum (CRC32 of all fields except checksum)
    fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.magic);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.page_size.to_le_bytes());
        hasher.update(&self.schema_page.to_le_bytes());
        hasher.update(&self.free_list_head.to_le_bytes());
        hasher.update(&self.total_pages.to_le_bytes());
        hasher.update(&self.created_us.to_le_bytes());
        hasher.update(&self.modified_us.to_le_bytes());
        hasher.finalize()
    }

    /// Validate checksum
    pub fn validate(&self) -> bool {
        self.magic == SOCHDB_MAGIC && self.checksum == self.compute_checksum()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        let mut cursor = io::Cursor::new(&mut buf[..]);

        cursor.write_all(&self.magic).unwrap();
        cursor.write_u32::<LittleEndian>(self.version).unwrap();
        cursor.write_u32::<LittleEndian>(self.page_size).unwrap();
        cursor.write_u64::<LittleEndian>(self.schema_page).unwrap();
        cursor
            .write_u64::<LittleEndian>(self.free_list_head)
            .unwrap();
        cursor.write_u64::<LittleEndian>(self.total_pages).unwrap();
        cursor.write_u64::<LittleEndian>(self.created_us).unwrap();
        cursor.write_u64::<LittleEndian>(self.modified_us).unwrap();
        cursor.write_u32::<LittleEndian>(self.checksum).unwrap();

        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Buffer too short for header",
            ));
        }

        let mut cursor = io::Cursor::new(buf);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;

        let version = cursor.read_u32::<LittleEndian>()?;
        let page_size = cursor.read_u32::<LittleEndian>()?;
        let schema_page = cursor.read_u64::<LittleEndian>()?;
        let free_list_head = cursor.read_u64::<LittleEndian>()?;
        let total_pages = cursor.read_u64::<LittleEndian>()?;
        let created_us = cursor.read_u64::<LittleEndian>()?;
        let modified_us = cursor.read_u64::<LittleEndian>()?;
        let checksum = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            magic,
            version,
            page_size,
            schema_page,
            free_list_head,
            total_pages,
            created_us,
            modified_us,
            checksum,
        })
    }
}

/// Free page header (linked list node)
#[derive(Debug, Clone)]
pub struct FreePageHeader {
    /// Next free page ID (0 = end of list)
    pub next_free: PageId,
    /// Number of contiguous free pages (for coalescing)
    pub count: u32,
}

impl FreePageHeader {
    /// Size in bytes
    pub const SIZE: usize = 12;

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        let mut cursor = io::Cursor::new(&mut buf[..]);
        cursor.write_u64::<LittleEndian>(self.next_free).unwrap();
        cursor.write_u32::<LittleEndian>(self.count).unwrap();
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(buf: &[u8]) -> io::Result<Self> {
        let mut cursor = io::Cursor::new(buf);
        Ok(Self {
            next_free: cursor.read_u64::<LittleEndian>()?,
            count: cursor.read_u32::<LittleEndian>()?,
        })
    }
}

/// Page type markers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    /// Database header (page 0 only)
    Header = 0,
    /// Schema catalog
    Catalog = 1,
    /// Column group data
    ColumnGroup = 2,
    /// Index interior node
    IndexInterior = 3,
    /// Index leaf node
    IndexLeaf = 4,
    /// Free page
    Free = 5,
    /// Overflow page (for large values)
    Overflow = 6,
}

/// Statistics for page manager
#[derive(Debug, Clone, Default)]
pub struct PageManagerStats {
    /// Total pages allocated
    pub total_pages: u64,
    /// Pages currently in use
    pub used_pages: u64,
    /// Free pages
    pub free_pages: u64,
    /// Page allocations
    pub allocations: u64,
    /// Page deallocations
    pub deallocations: u64,
    /// Page size
    pub page_size: u32,
    /// Total file size
    pub file_size: u64,
    /// Space amplification ratio
    pub space_amplification: f64,
}

/// Page manager for database file
pub struct PageManager {
    /// Database file path
    path: PathBuf,
    /// Database file handle
    file: RwLock<File>,
    /// Database header
    header: RwLock<DbHeader>,
    /// Page size
    page_size: u32,
    /// Stats counters
    allocations: AtomicU64,
    deallocations: AtomicU64,
}

impl PageManager {
    /// Create a new database file
    pub fn create<P: AsRef<Path>>(path: P, page_size: u32) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create and initialize file
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        // Write header on page 0
        let header = DbHeader::new(page_size);
        let header_bytes = header.to_bytes();
        file.write_all(&header_bytes)?;

        // Pad to page size
        let padding = vec![0u8; page_size as usize - DbHeader::SIZE];
        file.write_all(&padding)?;

        // Initialize page 1 (schema catalog)
        let catalog_page = vec![0u8; page_size as usize];
        file.write_all(&catalog_page)?;

        file.sync_all()?;

        Ok(Self {
            path,
            file: RwLock::new(file),
            header: RwLock::new(header),
            page_size,
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
        })
    }

    /// Open an existing database file
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Read header
        let mut header_buf = [0u8; DbHeader::SIZE];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header_buf)?;

        let header = DbHeader::from_bytes(&header_buf)?;

        if !header.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid database header or checksum",
            ));
        }

        let page_size = header.page_size;

        Ok(Self {
            path,
            file: RwLock::new(file),
            header: RwLock::new(header),
            page_size,
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
        })
    }

    /// Allocate a new page - O(1) amortized
    ///
    /// First tries to pop from free list, otherwise extends file
    pub fn allocate_page(&self) -> io::Result<PageId> {
        let mut header = self.header.write();
        let mut file = self.file.write();

        self.allocations.fetch_add(1, Ordering::Relaxed);

        if header.free_list_head != 0 {
            // Pop from free list
            let page_id = header.free_list_head;

            // Read free page header to get next
            let offset = page_id * self.page_size as u64;
            file.seek(SeekFrom::Start(offset))?;

            let mut free_header_buf = [0u8; FreePageHeader::SIZE];
            file.read_exact(&mut free_header_buf)?;
            let free_header = FreePageHeader::from_bytes(&free_header_buf)?;

            // Update free list head
            header.free_list_head = free_header.next_free;
            header.modified_us = now_micros();
            header.checksum = header.compute_checksum();

            // Write updated header
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;

            Ok(page_id)
        } else {
            // Extend file
            let page_id = header.total_pages;
            header.total_pages += 1;
            header.modified_us = now_micros();
            header.checksum = header.compute_checksum();

            // Write updated header
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;

            // Extend file with zeroed page
            let offset = page_id * self.page_size as u64;
            file.seek(SeekFrom::Start(offset))?;
            let zero_page = vec![0u8; self.page_size as usize];
            file.write_all(&zero_page)?;

            Ok(page_id)
        }
    }

    /// Deallocate a page - O(1)
    ///
    /// Adds page to head of free list
    pub fn deallocate_page(&self, page_id: PageId) -> io::Result<()> {
        if page_id < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deallocate header or catalog pages",
            ));
        }

        let mut header = self.header.write();
        let mut file = self.file.write();

        self.deallocations.fetch_add(1, Ordering::Relaxed);

        // Create free page header
        let free_header = FreePageHeader {
            next_free: header.free_list_head,
            count: 1,
        };

        // Write free page header
        let offset = page_id * self.page_size as u64;
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(&free_header.to_bytes())?;

        // Update database header
        header.free_list_head = page_id;
        header.modified_us = now_micros();
        header.checksum = header.compute_checksum();

        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header.to_bytes())?;

        Ok(())
    }

    /// Read a page
    pub fn read_page(&self, page_id: PageId) -> io::Result<Vec<u8>> {
        let mut file = self.file.write();
        let offset = page_id * self.page_size as u64;

        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; self.page_size as usize];
        file.read_exact(&mut buf)?;

        Ok(buf)
    }

    /// Write a page
    pub fn write_page(&self, page_id: PageId, data: &[u8]) -> io::Result<()> {
        if data.len() != self.page_size as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Page data must be exactly {} bytes", self.page_size),
            ));
        }

        let mut file = self.file.write();
        let offset = page_id * self.page_size as u64;

        file.seek(SeekFrom::Start(offset))?;
        file.write_all(data)?;

        // Update modified time
        {
            let mut header = self.header.write();
            header.modified_us = now_micros();
            header.checksum = header.compute_checksum();
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;
        }

        Ok(())
    }

    /// Sync all changes to disk
    pub fn sync(&self) -> io::Result<()> {
        self.file.read().sync_all()
    }

    /// Get page size
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Get total pages
    pub fn total_pages(&self) -> u64 {
        self.header.read().total_pages
    }

    /// Get database file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get statistics
    pub fn stats(&self) -> io::Result<PageManagerStats> {
        let header = self.header.read();
        let file = self.file.read();
        let file_size = file.metadata()?.len();

        // Count free pages by walking free list
        let mut free_count = 0u64;
        let mut current = header.free_list_head;

        drop(header);
        drop(file);

        while current != 0 {
            free_count += 1;
            let page_data = self.read_page(current)?;
            let free_header = FreePageHeader::from_bytes(&page_data)?;
            current = free_header.next_free;

            // Safety: prevent infinite loops
            if free_count > 1_000_000 {
                break;
            }
        }

        let header = self.header.read();
        let used_pages = header.total_pages - free_count;
        let logical_size = used_pages * self.page_size as u64;
        let space_amp = if logical_size > 0 {
            file_size as f64 / logical_size as f64
        } else {
            1.0
        };

        Ok(PageManagerStats {
            total_pages: header.total_pages,
            used_pages,
            free_pages: free_count,
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            page_size: self.page_size,
            file_size,
            space_amplification: space_amp,
        })
    }
}

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sochdb");

        // Create
        {
            let pm = PageManager::create(&path, DEFAULT_PAGE_SIZE).unwrap();
            assert_eq!(pm.total_pages(), 2);
            assert_eq!(pm.page_size(), DEFAULT_PAGE_SIZE);
        }

        // Reopen
        {
            let pm = PageManager::open(&path).unwrap();
            assert_eq!(pm.total_pages(), 2);
        }
    }

    #[test]
    fn test_allocate_and_deallocate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sochdb");

        let pm = PageManager::create(&path, DEFAULT_PAGE_SIZE).unwrap();

        // Allocate pages
        let p1 = pm.allocate_page().unwrap();
        let p2 = pm.allocate_page().unwrap();
        let p3 = pm.allocate_page().unwrap();

        assert_eq!(p1, 2); // Pages 0,1 are header and catalog
        assert_eq!(p2, 3);
        assert_eq!(p3, 4);
        assert_eq!(pm.total_pages(), 5);

        // Deallocate p2
        pm.deallocate_page(p2).unwrap();

        // Next allocation should reuse p2
        let p4 = pm.allocate_page().unwrap();
        assert_eq!(p4, 3); // Reused from free list

        // Check stats
        let stats = pm.stats().unwrap();
        assert_eq!(stats.total_pages, 5);
        assert_eq!(stats.free_pages, 0);
        assert!(stats.space_amplification < 1.5);
    }

    #[test]
    fn test_read_write_page() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sochdb");

        let pm = PageManager::create(&path, DEFAULT_PAGE_SIZE).unwrap();
        let page_id = pm.allocate_page().unwrap();

        // Write data
        let mut data = vec![0u8; DEFAULT_PAGE_SIZE as usize];
        data[0..4].copy_from_slice(b"TEST");
        data[100..108].copy_from_slice(&12345u64.to_le_bytes());
        pm.write_page(page_id, &data).unwrap();

        // Read back
        let read_data = pm.read_page(page_id).unwrap();
        assert_eq!(&read_data[0..4], b"TEST");

        let value = u64::from_le_bytes(read_data[100..108].try_into().unwrap());
        assert_eq!(value, 12345);
    }

    #[test]
    fn test_header_validation() {
        let header = DbHeader::new(4096);
        assert!(header.validate());

        let bytes = header.to_bytes();
        let restored = DbHeader::from_bytes(&bytes).unwrap();
        assert!(restored.validate());

        // Tamper with data
        let mut bad_bytes = bytes;
        bad_bytes[10] = 0xFF;
        let bad_header = DbHeader::from_bytes(&bad_bytes).unwrap();
        assert!(!bad_header.validate());
    }

    #[test]
    fn test_cannot_deallocate_system_pages() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sochdb");

        let pm = PageManager::create(&path, DEFAULT_PAGE_SIZE).unwrap();

        // Cannot deallocate page 0 (header) or 1 (catalog)
        assert!(pm.deallocate_page(0).is_err());
        assert!(pm.deallocate_page(1).is_err());
    }
}
