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

//! Zero-Copy Serialization (Recommendation 6)
//!
//! ## Problem
//!
//! Traditional serialization (bincode/serde) requires:
//! 1. Serialize: Marshal data into bytes
//! 2. Write: Write bytes to WAL
//! 3. Read: Read bytes from WAL
//! 4. Deserialize: Unmarshal bytes back to structs
//!
//! Deserialization is expensive:
//! ```text
//! bincode::deserialize<WalEntry>:
//!   - Parse header: ~50ns
//!   - Allocate strings: ~200ns (heap allocation)
//!   - Copy data: ~100ns
//!   - Total: ~350ns per entry
//! ```
//!
//! ## Solution
//!
//! Zero-copy serialization where the serialized format IS the in-memory format.
//! Data can be accessed directly from memory-mapped files without deserialization.
//!
//! ```text
//! Zero-copy read:
//!   - Validate header: ~10ns
//!   - Return reference: ~5ns
//!   - Total: ~15ns per entry (23x faster)
//! ```
//!
//! ## Design
//!
//! Fixed-size header followed by variable-length data:
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Magic (4B) │ Version (2B) │ Flags (2B) │ Length (4B) │ CRC │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Fixed-size fields                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Offset table (variable-length field offsets)                │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Variable-length data                     │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::mem::size_of;

/// Magic number for zero-copy format
pub const ZERO_COPY_MAGIC: u32 = 0x5A43_4F50; // "ZCOP"

/// Current format version
pub const FORMAT_VERSION: u16 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 16;

// =============================================================================
// Zero-Copy Header
// =============================================================================

/// Header for zero-copy serialized data
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyHeader {
    /// Magic number for format validation
    pub magic: u32,
    /// Format version
    pub version: u16,
    /// Flags (compression, etc.)
    pub flags: u16,
    /// Total length including header
    pub total_length: u32,
    /// CRC32 of data (excluding header)
    pub crc: u32,
}

impl ZeroCopyHeader {
    /// Create new header
    pub fn new(data_length: usize, flags: u16, crc: u32) -> Self {
        Self {
            magic: ZERO_COPY_MAGIC,
            version: FORMAT_VERSION,
            flags,
            total_length: (HEADER_SIZE + data_length) as u32,
            crc,
        }
    }

    /// Validate header
    #[inline]
    pub fn validate(&self) -> bool {
        self.magic == ZERO_COPY_MAGIC && self.version <= FORMAT_VERSION
    }

    /// Write header to buffer
    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= HEADER_SIZE);
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.total_length.to_le_bytes());
        buf[12..16].copy_from_slice(&self.crc.to_le_bytes());
    }

    /// Read header from buffer
    pub fn read_from(buf: &[u8]) -> Option<Self> {
        if buf.len() < HEADER_SIZE {
            return None;
        }
        Some(Self {
            magic: u32::from_le_bytes(buf[0..4].try_into().ok()?),
            version: u16::from_le_bytes(buf[4..6].try_into().ok()?),
            flags: u16::from_le_bytes(buf[6..8].try_into().ok()?),
            total_length: u32::from_le_bytes(buf[8..12].try_into().ok()?),
            crc: u32::from_le_bytes(buf[12..16].try_into().ok()?),
        })
    }
}

// =============================================================================
// Zero-Copy WAL Entry
// =============================================================================

/// WAL entry type
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalEntryType {
    /// Insert operation
    Insert = 1,
    /// Update operation
    Update = 2,
    /// Delete operation
    Delete = 3,
    /// Begin transaction
    BeginTxn = 4,
    /// Commit transaction
    CommitTxn = 5,
    /// Abort transaction
    AbortTxn = 6,
    /// Checkpoint marker
    Checkpoint = 7,
}

impl WalEntryType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Insert),
            2 => Some(Self::Update),
            3 => Some(Self::Delete),
            4 => Some(Self::BeginTxn),
            5 => Some(Self::CommitTxn),
            6 => Some(Self::AbortTxn),
            7 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

/// Fixed-size WAL entry header (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WalEntryHeader {
    /// Transaction ID
    pub txn_id: u64,
    /// Log sequence number
    pub lsn: u64,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp: u64,
    /// Entry type
    pub entry_type: u8,
    /// Number of variable-length fields
    pub field_count: u8,
    /// Reserved for alignment
    pub _reserved: [u8; 6],
}

pub const WAL_ENTRY_HEADER_SIZE: usize = size_of::<WalEntryHeader>();

impl WalEntryHeader {
    pub fn new(txn_id: u64, lsn: u64, entry_type: WalEntryType, field_count: u8) -> Self {
        Self {
            txn_id,
            lsn,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            entry_type: entry_type as u8,
            field_count,
            _reserved: [0; 6],
        }
    }

    /// Write to buffer
    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= WAL_ENTRY_HEADER_SIZE);
        buf[0..8].copy_from_slice(&self.txn_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.lsn.to_le_bytes());
        buf[16..24].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[24] = self.entry_type;
        buf[25] = self.field_count;
        buf[26..32].copy_from_slice(&self._reserved);
    }

    /// Read from buffer (zero-copy)
    #[inline]
    pub fn read_from(buf: &[u8]) -> Option<&Self> {
        if buf.len() < WAL_ENTRY_HEADER_SIZE {
            return None;
        }
        // Safety: We've verified length and WalEntryHeader is repr(C)
        // This is the zero-copy read - no deserialization!
        unsafe {
            let ptr = buf.as_ptr() as *const Self;
            // Verify alignment
            if ptr as usize % std::mem::align_of::<Self>() != 0 {
                return None;
            }
            Some(&*ptr)
        }
    }

    /// Read from buffer (safe copy version for unaligned data)
    pub fn read_from_copy(buf: &[u8]) -> Option<Self> {
        if buf.len() < WAL_ENTRY_HEADER_SIZE {
            return None;
        }
        Some(Self {
            txn_id: u64::from_le_bytes(buf[0..8].try_into().ok()?),
            lsn: u64::from_le_bytes(buf[8..16].try_into().ok()?),
            timestamp: u64::from_le_bytes(buf[16..24].try_into().ok()?),
            entry_type: buf[24],
            field_count: buf[25],
            _reserved: buf[26..32].try_into().ok()?,
        })
    }
}

// =============================================================================
// Zero-Copy WAL Entry Builder
// =============================================================================

/// Field descriptor for variable-length fields
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FieldDescriptor {
    /// Offset from start of data section
    pub offset: u32,
    /// Length of field
    pub length: u32,
}

pub const FIELD_DESCRIPTOR_SIZE: usize = size_of::<FieldDescriptor>();

/// Builder for zero-copy WAL entries
pub struct WalEntryBuilder {
    /// Fixed header
    header: WalEntryHeader,
    /// Field descriptors
    fields: Vec<FieldDescriptor>,
    /// Variable-length data
    data: Vec<u8>,
}

impl WalEntryBuilder {
    /// Create new builder
    pub fn new(txn_id: u64, lsn: u64, entry_type: WalEntryType) -> Self {
        Self {
            header: WalEntryHeader::new(txn_id, lsn, entry_type, 0),
            fields: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add a variable-length field
    pub fn add_field(&mut self, data: &[u8]) -> &mut Self {
        let offset = self.data.len() as u32;
        let length = data.len() as u32;
        self.fields.push(FieldDescriptor { offset, length });
        self.data.extend_from_slice(data);
        self.header.field_count = self.fields.len() as u8;
        self
    }

    /// Add key field
    pub fn with_key(&mut self, key: &[u8]) -> &mut Self {
        self.add_field(key)
    }

    /// Add value field
    pub fn with_value(&mut self, value: &[u8]) -> &mut Self {
        self.add_field(value)
    }

    /// Calculate total size
    pub fn total_size(&self) -> usize {
        HEADER_SIZE + 
        WAL_ENTRY_HEADER_SIZE + 
        self.fields.len() * FIELD_DESCRIPTOR_SIZE + 
        self.data.len()
    }

    /// Build into bytes
    pub fn build(&self) -> Vec<u8> {
        let data_len = WAL_ENTRY_HEADER_SIZE + 
            self.fields.len() * FIELD_DESCRIPTOR_SIZE + 
            self.data.len();
        
        let mut buf = vec![0u8; HEADER_SIZE + data_len];
        
        // Calculate CRC of data portion
        let crc = crc32fast::hash(&buf[HEADER_SIZE..]);
        
        // Write header
        let header = ZeroCopyHeader::new(data_len, 0, crc);
        header.write_to(&mut buf[0..HEADER_SIZE]);
        
        // Write WAL entry header
        let offset = HEADER_SIZE;
        self.header.write_to(&mut buf[offset..offset + WAL_ENTRY_HEADER_SIZE]);
        
        // Write field descriptors
        let mut offset = HEADER_SIZE + WAL_ENTRY_HEADER_SIZE;
        for field in &self.fields {
            buf[offset..offset + 4].copy_from_slice(&field.offset.to_le_bytes());
            buf[offset + 4..offset + 8].copy_from_slice(&field.length.to_le_bytes());
            offset += FIELD_DESCRIPTOR_SIZE;
        }
        
        // Write data
        buf[offset..].copy_from_slice(&self.data);
        
        // Update CRC
        let crc = crc32fast::hash(&buf[HEADER_SIZE..]);
        buf[12..16].copy_from_slice(&crc.to_le_bytes());
        
        buf
    }
}

// =============================================================================
// Zero-Copy WAL Entry Reader
// =============================================================================

/// Zero-copy WAL entry reader
/// 
/// Provides direct access to WAL entry fields without deserialization.
/// The backing data must remain valid for the lifetime of this reader.
pub struct WalEntryReader<'a> {
    /// Raw backing data
    data: &'a [u8],
    /// Parsed header
    header: &'a WalEntryHeader,
    /// Field count
    field_count: usize,
    /// Offset to field descriptors
    fields_offset: usize,
    /// Offset to data section
    data_offset: usize,
}

impl<'a> WalEntryReader<'a> {
    /// Create reader from raw bytes (zero-copy)
    pub fn from_bytes(bytes: &'a [u8]) -> Option<Self> {
        // Validate outer header
        let outer_header = ZeroCopyHeader::read_from(bytes)?;
        if !outer_header.validate() {
            return None;
        }
        
        // Validate CRC
        let expected_crc = outer_header.crc;
        let actual_crc = crc32fast::hash(&bytes[HEADER_SIZE..]);
        if expected_crc != actual_crc {
            return None;
        }
        
        // Read WAL entry header (zero-copy if aligned)
        let entry_data = &bytes[HEADER_SIZE..];
        let header = WalEntryHeader::read_from(entry_data)?;
        
        let field_count = header.field_count as usize;
        let fields_offset = WAL_ENTRY_HEADER_SIZE;
        let data_offset = fields_offset + field_count * FIELD_DESCRIPTOR_SIZE;
        
        Some(Self {
            data: entry_data,
            header,
            field_count,
            fields_offset,
            data_offset,
        })
    }

    /// Get transaction ID
    #[inline]
    pub fn txn_id(&self) -> u64 {
        self.header.txn_id
    }

    /// Get LSN
    #[inline]
    pub fn lsn(&self) -> u64 {
        self.header.lsn
    }

    /// Get timestamp
    #[inline]
    pub fn timestamp(&self) -> u64 {
        self.header.timestamp
    }

    /// Get entry type
    #[inline]
    pub fn entry_type(&self) -> Option<WalEntryType> {
        WalEntryType::from_u8(self.header.entry_type)
    }

    /// Get number of fields
    #[inline]
    pub fn field_count(&self) -> usize {
        self.field_count
    }

    /// Get field by index (zero-copy)
    #[inline]
    pub fn get_field(&self, index: usize) -> Option<&'a [u8]> {
        if index >= self.field_count {
            return None;
        }
        
        let desc_offset = self.fields_offset + index * FIELD_DESCRIPTOR_SIZE;
        let desc_bytes = self.data.get(desc_offset..desc_offset + FIELD_DESCRIPTOR_SIZE)?;
        
        let offset = u32::from_le_bytes(desc_bytes[0..4].try_into().ok()?) as usize;
        let length = u32::from_le_bytes(desc_bytes[4..8].try_into().ok()?) as usize;
        
        let start = self.data_offset + offset;
        self.data.get(start..start + length)
    }

    /// Get key field (first field by convention)
    #[inline]
    pub fn key(&self) -> Option<&'a [u8]> {
        self.get_field(0)
    }

    /// Get value field (second field by convention)
    #[inline]
    pub fn value(&self) -> Option<&'a [u8]> {
        self.get_field(1)
    }

    /// Iterate over all fields
    pub fn fields(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        (0..self.field_count).filter_map(|i| self.get_field(i))
    }
}

// =============================================================================
// Zero-Copy Batch Writer
// =============================================================================

/// Batch writer for multiple WAL entries
/// 
/// Optimized for group commit scenarios where multiple entries
/// are written together.
pub struct WalBatchWriter {
    /// Accumulated entries
    entries: Vec<Vec<u8>>,
    /// Total size
    total_size: usize,
}

impl WalBatchWriter {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_size: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            total_size: 0,
        }
    }

    /// Add entry to batch
    pub fn add(&mut self, entry: WalEntryBuilder) {
        let bytes = entry.build();
        self.total_size += bytes.len();
        self.entries.push(bytes);
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total size in bytes
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Build into single contiguous buffer
    pub fn build(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.total_size + 8);
        
        // Write entry count
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        // Write total size
        buf.extend_from_slice(&(self.total_size as u32).to_le_bytes());
        
        // Write all entries
        for entry in &self.entries {
            buf.extend_from_slice(entry);
        }
        
        buf
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_size = 0;
    }
}

impl Default for WalBatchWriter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Zero-Copy Batch Reader
// =============================================================================

/// Batch reader for multiple WAL entries
pub struct WalBatchReader<'a> {
    data: &'a [u8],
    entry_count: usize,
    #[allow(dead_code)]
    current_offset: usize,
}

impl<'a> WalBatchReader<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        
        let entry_count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let _total_size = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        
        Some(Self {
            data,
            entry_count,
            current_offset: 8,
        })
    }

    /// Get number of entries in batch
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Iterate over entries
    pub fn entries(&self) -> WalBatchIter<'a> {
        WalBatchIter {
            data: self.data,
            offset: 8,
            remaining: self.entry_count,
        }
    }
}

/// Iterator over batch entries
pub struct WalBatchIter<'a> {
    data: &'a [u8],
    offset: usize,
    remaining: usize,
}

impl<'a> Iterator for WalBatchIter<'a> {
    type Item = WalEntryReader<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        
        let entry_data = &self.data[self.offset..];
        let header = ZeroCopyHeader::read_from(entry_data)?;
        
        let entry_len = header.total_length as usize;
        let entry = WalEntryReader::from_bytes(&entry_data[..entry_len])?;
        
        self.offset += entry_len;
        self.remaining -= 1;
        
        Some(entry)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for WalBatchIter<'a> {}

// =============================================================================
// Memory-Mapped Zero-Copy Access
// =============================================================================

/// Memory-mapped WAL file for zero-copy access
pub struct MmapWalReader {
    /// Memory-mapped data
    mmap: memmap2::Mmap,
    /// File size
    size: usize,
}

impl MmapWalReader {
    /// Open WAL file for zero-copy reading
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        let size = metadata.len() as usize;
        
        // Safety: File is opened read-only
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        
        Ok(Self { mmap, size })
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get file size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Read entry at offset (zero-copy)
    pub fn read_entry_at(&self, offset: usize) -> Option<WalEntryReader<'_>> {
        if offset >= self.size {
            return None;
        }
        WalEntryReader::from_bytes(&self.mmap[offset..])
    }

    /// Iterate over all entries
    pub fn entries(&self) -> MmapWalIter<'_> {
        MmapWalIter {
            data: &self.mmap,
            offset: 0,
            size: self.size,
        }
    }
}

/// Iterator over memory-mapped WAL entries
pub struct MmapWalIter<'a> {
    data: &'a [u8],
    offset: usize,
    size: usize,
}

impl<'a> Iterator for MmapWalIter<'a> {
    type Item = WalEntryReader<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.size {
            return None;
        }
        
        let entry_data = &self.data[self.offset..];
        if entry_data.len() < HEADER_SIZE {
            return None;
        }
        
        let header = ZeroCopyHeader::read_from(entry_data)?;
        if !header.validate() {
            return None;
        }
        
        let entry_len = header.total_length as usize;
        if self.offset + entry_len > self.size {
            return None;
        }
        
        let entry = WalEntryReader::from_bytes(&entry_data[..entry_len])?;
        self.offset += entry_len;
        
        Some(entry)
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Serialization statistics
#[derive(Debug, Default)]
pub struct SerdeStats {
    /// Total entries written
    pub entries_written: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total entries read
    pub entries_read: u64,
    /// Total bytes read (zero-copy, not actually copied)
    pub bytes_read: u64,
    /// CRC validation failures
    pub crc_failures: u64,
}

impl SerdeStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_write(&mut self, bytes: usize) {
        self.entries_written += 1;
        self.bytes_written += bytes as u64;
    }

    pub fn record_read(&mut self, bytes: usize) {
        self.entries_read += 1;
        self.bytes_read += bytes as u64;
    }

    pub fn record_crc_failure(&mut self) {
        self.crc_failures += 1;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_roundtrip() {
        let mut builder = WalEntryBuilder::new(42, 100, WalEntryType::Insert);
        builder.with_key(b"test_key").with_value(b"test_value");
        
        let bytes = builder.build();
        let reader = WalEntryReader::from_bytes(&bytes).unwrap();
        
        assert_eq!(reader.txn_id(), 42);
        assert_eq!(reader.lsn(), 100);
        assert_eq!(reader.entry_type(), Some(WalEntryType::Insert));
        assert_eq!(reader.field_count(), 2);
        assert_eq!(reader.key(), Some(b"test_key".as_slice()));
        assert_eq!(reader.value(), Some(b"test_value".as_slice()));
    }

    #[test]
    fn test_wal_entry_zero_copy_header() {
        let header = WalEntryHeader::new(123, 456, WalEntryType::Update, 3);
        let mut buf = vec![0u8; WAL_ENTRY_HEADER_SIZE];
        header.write_to(&mut buf);
        
        // Test zero-copy read (if aligned)
        if let Some(read_header) = WalEntryHeader::read_from(&buf) {
            assert_eq!(read_header.txn_id, 123);
            assert_eq!(read_header.lsn, 456);
            assert_eq!(read_header.entry_type, WalEntryType::Update as u8);
            assert_eq!(read_header.field_count, 3);
        }
        
        // Test copy read
        let read_header = WalEntryHeader::read_from_copy(&buf).unwrap();
        assert_eq!(read_header.txn_id, 123);
        assert_eq!(read_header.lsn, 456);
    }

    #[test]
    fn test_wal_entry_crc_validation() {
        let mut builder = WalEntryBuilder::new(1, 1, WalEntryType::Insert);
        builder.with_key(b"key");
        
        let mut bytes = builder.build();
        
        // Valid entry should parse
        assert!(WalEntryReader::from_bytes(&bytes).is_some());
        
        // Corrupt data
        if bytes.len() > 20 {
            bytes[20] ^= 0xFF;
        }
        
        // Corrupted entry should fail CRC
        assert!(WalEntryReader::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_batch_writer_reader() {
        let mut batch = WalBatchWriter::new();
        
        for i in 0..10 {
            let mut entry = WalEntryBuilder::new(i, i * 10, WalEntryType::Insert);
            entry.with_key(format!("key_{}", i).as_bytes());
            entry.with_value(format!("value_{}", i).as_bytes());
            batch.add(entry);
        }
        
        assert_eq!(batch.len(), 10);
        
        let bytes = batch.build();
        let reader = WalBatchReader::from_bytes(&bytes).unwrap();
        
        assert_eq!(reader.entry_count(), 10);
        
        for (i, entry) in reader.entries().enumerate() {
            assert_eq!(entry.txn_id(), i as u64);
            assert_eq!(entry.key(), Some(format!("key_{}", i).as_bytes()));
        }
    }

    #[test]
    fn test_multiple_fields() {
        let mut builder = WalEntryBuilder::new(1, 1, WalEntryType::Update);
        builder.add_field(b"field_0");
        builder.add_field(b"field_1");
        builder.add_field(b"field_2");
        builder.add_field(b"field_3");
        
        let bytes = builder.build();
        let reader = WalEntryReader::from_bytes(&bytes).unwrap();
        
        assert_eq!(reader.field_count(), 4);
        
        let fields: Vec<_> = reader.fields().collect();
        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0], b"field_0");
        assert_eq!(fields[1], b"field_1");
        assert_eq!(fields[2], b"field_2");
        assert_eq!(fields[3], b"field_3");
    }

    #[test]
    fn test_empty_fields() {
        let builder = WalEntryBuilder::new(1, 1, WalEntryType::BeginTxn);
        
        let bytes = builder.build();
        let reader = WalEntryReader::from_bytes(&bytes).unwrap();
        
        assert_eq!(reader.field_count(), 0);
        assert_eq!(reader.entry_type(), Some(WalEntryType::BeginTxn));
    }

    #[test]
    fn test_large_value() {
        let large_value = vec![0xAB; 1024 * 1024]; // 1MB
        
        let mut builder = WalEntryBuilder::new(1, 1, WalEntryType::Insert);
        builder.with_key(b"large_key").with_value(&large_value);
        
        let bytes = builder.build();
        let reader = WalEntryReader::from_bytes(&bytes).unwrap();
        
        assert_eq!(reader.value(), Some(large_value.as_slice()));
    }

    #[test]
    fn test_header_validation() {
        let header = ZeroCopyHeader::new(100, 0, 12345);
        assert!(header.validate());
        
        let mut bad_header = header;
        bad_header.magic = 0xDEADBEEF;
        assert!(!bad_header.validate());
    }
}
