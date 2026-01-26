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

//! Page Management
//!
//! Basic page abstraction for the kernel.
//! Provides page header structure with LSN tracking for ARIES recovery.

use crate::error::{KernelError, KernelResult, PageErrorKind};
use crate::kernel_api::PageId;
use crate::wal::LogSequenceNumber;
use bytes::{BufMut, Bytes, BytesMut};

/// Default page size: 8KB (matches common filesystem block size)
pub const PAGE_SIZE: usize = 8192;

/// Page header size
pub const PAGE_HEADER_SIZE: usize = 32;

/// Usable page space (after header)
pub const PAGE_DATA_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

/// Page magic number for validation
pub const PAGE_MAGIC: u32 = 0x544F4F4E; // "TOON"

/// Page types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    /// Free/unallocated page
    Free = 0,
    /// Data page (rows)
    Data = 1,
    /// Index page (B-tree node)
    Index = 2,
    /// Overflow page (for large values)
    Overflow = 3,
    /// Metadata page (catalog info)
    Metadata = 4,
}

impl TryFrom<u8> for PageType {
    type Error = KernelError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Free),
            1 => Ok(Self::Data),
            2 => Ok(Self::Index),
            3 => Ok(Self::Overflow),
            4 => Ok(Self::Metadata),
            _ => Err(KernelError::Page {
                kind: PageErrorKind::InvalidSize,
            }),
        }
    }
}

/// Page header
///
/// Every page starts with this header for ARIES recovery support.
///
/// Layout (32 bytes):
/// - magic: u32 (4 bytes) - validation magic number
/// - page_id: u64 (8 bytes) - page identifier
/// - page_lsn: u64 (8 bytes) - LSN of last modification (for recovery)
/// - page_type: u8 (1 byte) - page type
/// - flags: u8 (1 byte) - page flags
/// - free_space: u16 (2 bytes) - free space in page
/// - checksum: u32 (4 bytes) - page checksum
/// - reserved: u32 (4 bytes) - reserved for future use
#[derive(Debug, Clone, Copy)]
pub struct PageHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Page identifier
    pub page_id: PageId,
    /// LSN of last modification
    pub page_lsn: LogSequenceNumber,
    /// Page type
    pub page_type: PageType,
    /// Flags
    pub flags: u8,
    /// Free space in page
    pub free_space: u16,
    /// Checksum
    pub checksum: u32,
}

impl PageHeader {
    /// Create a new page header
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        Self {
            magic: PAGE_MAGIC,
            page_id,
            page_lsn: LogSequenceNumber::INVALID,
            page_type,
            flags: 0,
            free_space: PAGE_DATA_SIZE as u16,
            checksum: 0,
        }
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> [u8; PAGE_HEADER_SIZE] {
        let mut buf = [0u8; PAGE_HEADER_SIZE];
        let mut cursor = 0;

        buf[cursor..cursor + 4].copy_from_slice(&self.magic.to_le_bytes());
        cursor += 4;

        buf[cursor..cursor + 8].copy_from_slice(&self.page_id.to_le_bytes());
        cursor += 8;

        buf[cursor..cursor + 8].copy_from_slice(&self.page_lsn.0.to_le_bytes());
        cursor += 8;

        buf[cursor] = self.page_type as u8;
        cursor += 1;

        buf[cursor] = self.flags;
        cursor += 1;

        buf[cursor..cursor + 2].copy_from_slice(&self.free_space.to_le_bytes());
        cursor += 2;

        buf[cursor..cursor + 4].copy_from_slice(&self.checksum.to_le_bytes());
        // cursor += 4; // reserved bytes remain zero

        buf
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> KernelResult<Self> {
        if data.len() < PAGE_HEADER_SIZE {
            return Err(KernelError::Page {
                kind: PageErrorKind::InvalidSize,
            });
        }

        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != PAGE_MAGIC {
            return Err(KernelError::Corruption {
                details: format!(
                    "invalid page magic: expected {:#x}, got {:#x}",
                    PAGE_MAGIC, magic
                ),
            });
        }

        let page_id = u64::from_le_bytes(data[4..12].try_into().unwrap());
        let page_lsn = LogSequenceNumber(u64::from_le_bytes(data[12..20].try_into().unwrap()));
        let page_type = PageType::try_from(data[20])?;
        let flags = data[21];
        let free_space = u16::from_le_bytes(data[22..24].try_into().unwrap());
        let checksum = u32::from_le_bytes(data[24..28].try_into().unwrap());

        Ok(Self {
            magic,
            page_id,
            page_lsn,
            page_type,
            flags,
            free_space,
            checksum,
        })
    }
}

/// Page - a fixed-size storage unit
pub struct Page {
    /// Page header
    pub header: PageHeader,
    /// Page data (excluding header)
    pub data: BytesMut,
}

impl Page {
    /// Create a new empty page
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        Self {
            header: PageHeader::new(page_id, page_type),
            data: BytesMut::zeroed(PAGE_DATA_SIZE),
        }
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8]) -> KernelResult<Self> {
        if bytes.len() != PAGE_SIZE {
            return Err(KernelError::Page {
                kind: PageErrorKind::InvalidSize,
            });
        }

        let header = PageHeader::deserialize(&bytes[..PAGE_HEADER_SIZE])?;
        let data = BytesMut::from(&bytes[PAGE_HEADER_SIZE..]);

        Ok(Self { header, data })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(PAGE_SIZE);
        buf.put_slice(&self.header.serialize());
        buf.put_slice(&self.data);
        buf.freeze()
    }

    /// Get page ID
    pub fn page_id(&self) -> PageId {
        self.header.page_id
    }

    /// Get page LSN
    pub fn lsn(&self) -> LogSequenceNumber {
        self.header.page_lsn
    }

    /// Set page LSN (after modification)
    pub fn set_lsn(&mut self, lsn: LogSequenceNumber) {
        self.header.page_lsn = lsn;
    }

    /// Check if page needs redo during recovery
    ///
    /// Returns true if the page's LSN is less than the WAL record's LSN,
    /// meaning the WAL record's changes haven't been applied yet.
    pub fn needs_redo(&self, record_lsn: LogSequenceNumber) -> bool {
        self.header.page_lsn < record_lsn
    }

    /// Compute checksum for the page
    pub fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        // Include header fields except checksum itself
        hasher.update(&self.header.magic.to_le_bytes());
        hasher.update(&self.header.page_id.to_le_bytes());
        hasher.update(&self.header.page_lsn.0.to_le_bytes());
        hasher.update(&[self.header.page_type as u8, self.header.flags]);
        hasher.update(&self.header.free_space.to_le_bytes());
        hasher.update(&self.data);
        hasher.finalize()
    }

    /// Validate page checksum
    pub fn validate_checksum(&self) -> bool {
        self.header.checksum == self.compute_checksum()
    }

    /// Update checksum before writing to disk
    pub fn update_checksum(&mut self) {
        self.header.checksum = self.compute_checksum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_header_roundtrip() {
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_id: 42,
            page_lsn: LogSequenceNumber(100),
            page_type: PageType::Data,
            flags: 0x01,
            free_space: 1234,
            checksum: 0xDEADBEEF,
        };

        let serialized = header.serialize();
        let deserialized = PageHeader::deserialize(&serialized).unwrap();

        assert_eq!(header.magic, deserialized.magic);
        assert_eq!(header.page_id, deserialized.page_id);
        assert_eq!(header.page_lsn, deserialized.page_lsn);
        assert_eq!(header.page_type, deserialized.page_type);
        assert_eq!(header.flags, deserialized.flags);
        assert_eq!(header.free_space, deserialized.free_space);
        assert_eq!(header.checksum, deserialized.checksum);
    }

    #[test]
    fn test_page_roundtrip() {
        let mut page = Page::new(1, PageType::Data);
        page.data[0..5].copy_from_slice(b"hello");
        page.set_lsn(LogSequenceNumber(50));
        page.update_checksum();

        let bytes = page.to_bytes();
        let restored = Page::from_bytes(&bytes).unwrap();

        assert_eq!(restored.page_id(), 1);
        assert_eq!(restored.lsn(), LogSequenceNumber(50));
        assert!(restored.validate_checksum());
    }

    #[test]
    fn test_needs_redo() {
        let mut page = Page::new(1, PageType::Data);
        page.set_lsn(LogSequenceNumber(100));

        // Record with lower LSN - already applied
        assert!(!page.needs_redo(LogSequenceNumber(50)));

        // Record with same LSN - already applied
        assert!(!page.needs_redo(LogSequenceNumber(100)));

        // Record with higher LSN - needs redo
        assert!(page.needs_redo(LogSequenceNumber(150)));
    }
}
