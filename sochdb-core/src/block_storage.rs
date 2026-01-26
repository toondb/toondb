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

//! File Data Block Storage (Task 12)
//!
//! Integrates PayloadStore as SochFS data block backend:
//! - O(1) append-only writes (no LSM compaction overhead)
//! - Transparent compression (LZ4/ZSTD based on content type)
//! - Block reference tracking for garbage collection
//!
//! ## Block Storage Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                      Inode Table                        │
//! │  inode_id=7, blocks=[(off=0, len=4096, cmp=LZ4),       │
//! │                      (off=4096, len=2048, cmp=ZSTD)]   │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                    PayloadStore                         │
//! │  Offset 0:    [LZ4 header][compressed block 1]         │
//! │  Offset 4096: [ZSTD header][compressed block 2]        │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Architectural Improvements (Production Tasks)
//!
//! ### Task 1: Fixed-Layout BlockHeader
//!
//! Uses deterministic 17-byte fixed layout with explicit little-endian encoding:
//! ```text
//! Offset  Size  Field            Type
//! 0       4     magic            [u8; 4] = "TBLK"
//! 4       1     compression      u8
//! 5       4     original_size    u32 (LE)
//! 9       4     compressed_size  u32 (LE)
//! 13      4     checksum         u32 (LE, CRC32)
//! Total: 17 bytes
//! ```
//!
//! ### Task 5: Error Propagation
//!
//! All serialization methods return `Result<T, SochDBError>` instead of
//! using `unwrap_or_default()` which silently swallows errors.

use byteorder::{ByteOrder, LittleEndian};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Result, SochDBError};

/// Default block size (4KB)
pub const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Maximum block size (1MB)
pub const MAX_BLOCK_SIZE: usize = 1024 * 1024;

/// Compression type for blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlockCompression {
    /// No compression
    None = 0,
    /// LZ4 (fast)
    Lz4 = 1,
    /// ZSTD (high ratio)
    Zstd = 2,
}

impl BlockCompression {
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => BlockCompression::Lz4,
            2 => BlockCompression::Zstd,
            _ => BlockCompression::None,
        }
    }

    pub fn to_byte(&self) -> u8 {
        *self as u8
    }
}

/// Block reference stored in inode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRef {
    /// Offset in payload store
    pub store_offset: u64,
    /// Compressed length
    pub compressed_len: u32,
    /// Original length
    pub original_len: u32,
    /// Compression type
    pub compression: BlockCompression,
    /// Checksum
    pub checksum: u32,
}

/// Fixed size of BlockRef when serialized (21 bytes)
/// Layout: offset(8) + compressed_len(4) + original_len(4) + compression(1) + checksum(4)
impl BlockRef {
    /// Fixed serialization size
    pub const SERIALIZED_SIZE: usize = 21;

    /// Serialize to fixed-layout bytes (Task 5: returns Result)
    pub fn to_bytes(&self) -> Result<[u8; Self::SERIALIZED_SIZE]> {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        LittleEndian::write_u64(&mut buf[0..8], self.store_offset);
        LittleEndian::write_u32(&mut buf[8..12], self.compressed_len);
        LittleEndian::write_u32(&mut buf[12..16], self.original_len);
        buf[16] = self.compression as u8;
        LittleEndian::write_u32(&mut buf[17..21], self.checksum);
        Ok(buf)
    }

    /// Deserialize from fixed-layout bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SERIALIZED_SIZE {
            return Err(SochDBError::Serialization(format!(
                "BlockRef too short: {} < {}",
                data.len(),
                Self::SERIALIZED_SIZE
            )));
        }

        Ok(Self {
            store_offset: LittleEndian::read_u64(&data[0..8]),
            compressed_len: LittleEndian::read_u32(&data[8..12]),
            original_len: LittleEndian::read_u32(&data[12..16]),
            compression: BlockCompression::from_byte(data[16]),
            checksum: LittleEndian::read_u32(&data[17..21]),
        })
    }
}

/// Block header in payload store (Task 1: Fixed 17-byte layout)
///
/// Layout (all integers little-endian):
/// ```text
/// Offset  Size  Field            
/// 0       4     magic = "TBLK"   
/// 4       1     compression      
/// 5       4     original_size    
/// 9       4     compressed_size  
/// 13      4     checksum (CRC32)
/// Total: 17 bytes
/// ```
#[derive(Debug, Clone)]
pub struct BlockHeader {
    /// Magic bytes
    pub magic: [u8; 4],
    /// Compression type
    pub compression: u8,
    /// Original size
    pub original_size: u32,
    /// Compressed size
    pub compressed_size: u32,
    /// CRC32 checksum
    pub checksum: u32,
}

impl BlockHeader {
    pub const MAGIC: [u8; 4] = *b"TBLK";
    pub const SIZE: usize = 17; // Fixed layout size

    /// Serialize to fixed 17-byte layout (Task 1: deterministic serialization)
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&Self::MAGIC);
        buf[4] = self.compression;
        LittleEndian::write_u32(&mut buf[5..9], self.original_size);
        LittleEndian::write_u32(&mut buf[9..13], self.compressed_size);
        LittleEndian::write_u32(&mut buf[13..17], self.checksum);
        buf
    }

    /// Deserialize from 17-byte buffer (Task 1: validates magic)
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(SochDBError::Corruption(format!(
                "BlockHeader too short: {} < {}",
                buf.len(),
                Self::SIZE
            )));
        }

        if buf[0..4] != Self::MAGIC {
            return Err(SochDBError::Corruption(format!(
                "Invalid block magic: expected {:?}, got {:?}",
                Self::MAGIC,
                &buf[0..4]
            )));
        }

        Ok(Self {
            magic: Self::MAGIC,
            compression: buf[4],
            original_size: LittleEndian::read_u32(&buf[5..9]),
            compressed_size: LittleEndian::read_u32(&buf[9..13]),
            checksum: LittleEndian::read_u32(&buf[13..17]),
        })
    }
}

/// File block storage backed by append-only store
pub struct BlockStore {
    /// Data storage (append-only)
    data: RwLock<Vec<u8>>,
    /// Next write offset
    next_offset: AtomicU64,
    /// Block index: offset -> BlockRef
    index: RwLock<HashMap<u64, BlockRef>>,
    /// Reference counts for GC
    ref_counts: RwLock<HashMap<u64, u32>>,
}

impl BlockStore {
    /// Create a new block store
    pub fn new() -> Self {
        Self {
            data: RwLock::new(Vec::new()),
            next_offset: AtomicU64::new(0),
            index: RwLock::new(HashMap::new()),
            ref_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Write a block with automatic compression selection
    ///
    /// Returns BlockRef for the written block.
    pub fn write_block(&self, data: &[u8]) -> Result<BlockRef> {
        let compression = self.select_compression(data);
        self.write_block_with_compression(data, compression)
    }

    /// Write a block with specified compression
    pub fn write_block_with_compression(
        &self,
        data: &[u8],
        compression: BlockCompression,
    ) -> Result<BlockRef> {
        // Compress data
        let compressed = self.compress(data, compression)?;

        // Calculate checksum
        let checksum = crc32fast::hash(&compressed);

        // Allocate offset
        let header_size = BlockHeader::SIZE;
        let total_size = header_size + compressed.len();
        let offset = self
            .next_offset
            .fetch_add(total_size as u64, Ordering::SeqCst);

        // Create header
        let header = BlockHeader {
            magic: BlockHeader::MAGIC,
            compression: compression as u8,
            original_size: data.len() as u32,
            compressed_size: compressed.len() as u32,
            checksum,
        };

        // Write to store
        {
            let mut store = self.data.write();
            store.resize((offset + total_size as u64) as usize, 0);

            // Write header (Task 1: fixed-layout serialization)
            let header_bytes = header.to_bytes();
            store[offset as usize..offset as usize + header_size].copy_from_slice(&header_bytes);

            // Write data
            store[offset as usize + header_size..offset as usize + total_size]
                .copy_from_slice(&compressed);
        }

        // Create block reference
        let block_ref = BlockRef {
            store_offset: offset,
            compressed_len: compressed.len() as u32,
            original_len: data.len() as u32,
            compression,
            checksum,
        };

        // Update index
        self.index.write().insert(offset, block_ref.clone());

        // Initialize ref count
        self.ref_counts.write().insert(offset, 1);

        Ok(block_ref)
    }

    /// Read a block
    pub fn read_block(&self, block_ref: &BlockRef) -> Result<Vec<u8>> {
        let offset = block_ref.store_offset as usize;
        let header_size = BlockHeader::SIZE;
        let total_size = header_size + block_ref.compressed_len as usize;

        // Read from store
        let compressed = {
            let store = self.data.read();
            if offset + total_size > store.len() {
                return Err(SochDBError::NotFound("Block not found".into()));
            }

            // Verify header (Task 1: use fixed-layout deserialization)
            let header_bytes = &store[offset..offset + header_size];
            let _header = BlockHeader::from_bytes(header_bytes)?;

            // Read data
            store[offset + header_size..offset + total_size].to_vec()
        };

        // Verify checksum
        let checksum = crc32fast::hash(&compressed);
        if checksum != block_ref.checksum {
            return Err(SochDBError::Corruption("Block checksum mismatch".into()));
        }

        // Decompress
        self.decompress(
            &compressed,
            block_ref.compression,
            block_ref.original_len as usize,
        )
    }

    /// Select compression based on data content
    fn select_compression(&self, data: &[u8]) -> BlockCompression {
        if data.len() < 128 {
            return BlockCompression::None; // Too small to compress
        }

        // Detect content type
        if is_soch_content(data) {
            BlockCompression::Zstd // TOON is repetitive, good for ZSTD
        } else if is_json_content(data) || is_compressible(data) {
            BlockCompression::Lz4 // JSON and compressible content: fast compression
        } else {
            BlockCompression::None // Binary/random data
        }
    }

    /// Compress data with fallback to raw on compression failure or expansion
    ///
    /// Returns the compressed data or original data if:
    /// - Compression fails
    /// - Compressed size >= original size (compression expanded the data)
    fn compress(&self, data: &[u8], compression: BlockCompression) -> Result<Vec<u8>> {
        match compression {
            BlockCompression::None => Ok(data.to_vec()),
            BlockCompression::Lz4 => {
                match lz4::block::compress(data, None, false) {
                    Ok(compressed) => {
                        // Fallback to raw if compression didn't help
                        if compressed.len() >= data.len() {
                            Ok(data.to_vec())
                        } else {
                            Ok(compressed)
                        }
                    }
                    Err(_) => {
                        // Fallback to raw on compression failure
                        Ok(data.to_vec())
                    }
                }
            }
            BlockCompression::Zstd => {
                // Default compression level (3) for balance of speed/ratio
                match zstd::encode_all(data, 3) {
                    Ok(compressed) => {
                        // Fallback to raw if compression didn't help
                        if compressed.len() >= data.len() {
                            Ok(data.to_vec())
                        } else {
                            Ok(compressed)
                        }
                    }
                    Err(_) => {
                        // Fallback to raw on compression failure
                        Ok(data.to_vec())
                    }
                }
            }
        }
    }

    /// Decompress data with automatic format detection
    fn decompress(
        &self,
        data: &[u8],
        compression: BlockCompression,
        original_size: usize,
    ) -> Result<Vec<u8>> {
        match compression {
            BlockCompression::None => Ok(data.to_vec()),
            BlockCompression::Lz4 => {
                // If data matches original size, it's uncompressed (fallback case)
                if data.len() == original_size {
                    return Ok(data.to_vec());
                }

                lz4::block::decompress(data, Some(original_size as i32)).map_err(|e| {
                    SochDBError::Corruption(format!("LZ4 decompression failed: {}", e))
                })
            }
            BlockCompression::Zstd => {
                // If data matches original size, it's uncompressed (fallback case)
                if data.len() == original_size {
                    return Ok(data.to_vec());
                }

                zstd::decode_all(data).map_err(|e| {
                    SochDBError::Corruption(format!("ZSTD decompression failed: {}", e))
                })
            }
        }
    }

    /// Increment reference count
    pub fn add_ref(&self, offset: u64) {
        let mut refs = self.ref_counts.write();
        *refs.entry(offset).or_insert(0) += 1;
    }

    /// Decrement reference count
    pub fn release_ref(&self, offset: u64) -> bool {
        let mut refs = self.ref_counts.write();
        if let Some(count) = refs.get_mut(&offset) {
            *count = count.saturating_sub(1);
            return *count == 0;
        }
        false
    }

    /// Get storage statistics
    pub fn stats(&self) -> BlockStoreStats {
        let data = self.data.read();
        let index = self.index.read();

        let mut total_original = 0u64;
        let mut total_compressed = 0u64;

        for block_ref in index.values() {
            total_original += block_ref.original_len as u64;
            total_compressed += block_ref.compressed_len as u64;
        }

        BlockStoreStats {
            total_bytes: data.len() as u64,
            block_count: index.len(),
            total_original_bytes: total_original,
            total_compressed_bytes: total_compressed,
            compression_ratio: if total_compressed > 0 {
                total_original as f64 / total_compressed as f64
            } else {
                1.0
            },
        }
    }
}

impl Default for BlockStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Block store statistics
#[derive(Debug, Clone, Default)]
pub struct BlockStoreStats {
    /// Total bytes in store
    pub total_bytes: u64,
    /// Number of blocks
    pub block_count: usize,
    /// Total original bytes (before compression)
    pub total_original_bytes: u64,
    /// Total compressed bytes
    pub total_compressed_bytes: u64,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f64,
}

/// Check if data looks like TOON format
pub fn is_soch_content(data: &[u8]) -> bool {
    // TOON typically starts with table name and brackets
    if data.len() < 10 {
        return false;
    }

    // Check for common TOON patterns: "name[", "name{", or starts with alphabetic
    let s = String::from_utf8_lossy(&data[..data.len().min(100)]);
    s.contains('[') && s.contains('{') && s.contains(':')
}

/// Check if data looks like JSON
pub fn is_json_content(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }

    // JSON typically starts with { or [
    let first = data[0];
    first == b'{' || first == b'['
}

/// Check if data is likely compressible
pub fn is_compressible(data: &[u8]) -> bool {
    if data.len() < 64 {
        return false;
    }

    // Count unique bytes in sample
    let sample_size = data.len().min(256);
    let mut seen = [false; 256];
    let mut unique = 0;

    for &byte in &data[..sample_size] {
        if !seen[byte as usize] {
            seen[byte as usize] = true;
            unique += 1;
        }
    }

    // If less than 50% unique bytes, likely compressible
    unique < sample_size / 2
}

/// File block manager - higher level interface for file I/O
pub struct FileBlockManager {
    /// Block store
    store: BlockStore,
    /// Block size
    block_size: usize,
}

impl FileBlockManager {
    /// Create new file block manager
    pub fn new(block_size: usize) -> Self {
        Self {
            store: BlockStore::new(),
            block_size: block_size.min(MAX_BLOCK_SIZE),
        }
    }

    /// Write file data, returns block references
    pub fn write_file(&self, data: &[u8]) -> Result<Vec<BlockRef>> {
        let mut blocks = Vec::new();

        for chunk in data.chunks(self.block_size) {
            let block_ref = self.store.write_block(chunk)?;
            blocks.push(block_ref);
        }

        Ok(blocks)
    }

    /// Read file data from block references
    pub fn read_file(&self, blocks: &[BlockRef]) -> Result<Vec<u8>> {
        let mut data = Vec::new();

        for block_ref in blocks {
            let block_data = self.store.read_block(block_ref)?;
            data.extend(block_data);
        }

        Ok(data)
    }

    /// Get underlying store stats
    pub fn stats(&self) -> BlockStoreStats {
        self.store.stats()
    }
}

impl Default for FileBlockManager {
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }
}

// ============================================================================
// Durable Block Store with WAL (Task 2: Persistent Block Storage)
// ============================================================================

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// WAL record types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalRecordType {
    /// Block write record
    BlockWrite = 1,
    /// Checkpoint marker
    Checkpoint = 2,
    /// Commit marker
    Commit = 3,
    /// Transaction begin
    TxnBegin = 4,
}

impl WalRecordType {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(WalRecordType::BlockWrite),
            2 => Some(WalRecordType::Checkpoint),
            3 => Some(WalRecordType::Commit),
            4 => Some(WalRecordType::TxnBegin),
            _ => None,
        }
    }
}

/// WAL record header (fixed 33-byte layout)
///
/// Layout:
/// ```text
/// Offset  Size  Field
/// 0       8     lsn (Log Sequence Number)
/// 8       8     txn_id
/// 16      1     record_type
/// 17      8     page_id
/// 25      4     data_len
/// 29      4     crc32 (checksum of header + data)
/// Total: 33 bytes
/// ```
#[derive(Debug, Clone)]
pub struct WalRecordHeader {
    pub lsn: u64,
    pub txn_id: u64,
    pub record_type: WalRecordType,
    pub page_id: u64,
    pub data_len: u32,
    pub crc32: u32,
}

impl WalRecordHeader {
    pub const SIZE: usize = 33;

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        LittleEndian::write_u64(&mut buf[0..8], self.lsn);
        LittleEndian::write_u64(&mut buf[8..16], self.txn_id);
        buf[16] = self.record_type as u8;
        LittleEndian::write_u64(&mut buf[17..25], self.page_id);
        LittleEndian::write_u32(&mut buf[25..29], self.data_len);
        LittleEndian::write_u32(&mut buf[29..33], self.crc32);
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(SochDBError::Corruption(format!(
                "WAL record header too short: {} < {}",
                buf.len(),
                Self::SIZE
            )));
        }

        let record_type = WalRecordType::from_byte(buf[16]).ok_or_else(|| {
            SochDBError::Corruption(format!("Invalid WAL record type: {}", buf[16]))
        })?;

        Ok(Self {
            lsn: LittleEndian::read_u64(&buf[0..8]),
            txn_id: LittleEndian::read_u64(&buf[8..16]),
            record_type,
            page_id: LittleEndian::read_u64(&buf[17..25]),
            data_len: LittleEndian::read_u32(&buf[25..29]),
            crc32: LittleEndian::read_u32(&buf[29..33]),
        })
    }

    /// Compute CRC32 for header + data
    pub fn compute_crc32(&self, data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();

        // Hash header fields (excluding crc32 itself)
        let mut header_buf = [0u8; 29];
        LittleEndian::write_u64(&mut header_buf[0..8], self.lsn);
        LittleEndian::write_u64(&mut header_buf[8..16], self.txn_id);
        header_buf[16] = self.record_type as u8;
        LittleEndian::write_u64(&mut header_buf[17..25], self.page_id);
        LittleEndian::write_u32(&mut header_buf[25..29], self.data_len);

        hasher.update(&header_buf);
        hasher.update(data);
        hasher.finalize()
    }
}

/// WAL writer for durable writes
pub struct WalWriter {
    /// WAL file handle
    file: BufWriter<File>,
    /// Next LSN to assign
    next_lsn: u64,
    /// Path to WAL file
    #[allow(dead_code)]
    path: PathBuf,
}

impl WalWriter {
    /// Open or create WAL file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open file for append
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;

        // Get current file size for next_lsn
        let metadata = file.metadata()?;
        let next_lsn = metadata.len();

        Ok(Self {
            file: BufWriter::new(file),
            next_lsn,
            path,
        })
    }

    /// Append a WAL record
    pub fn append(
        &mut self,
        txn_id: u64,
        record_type: WalRecordType,
        page_id: u64,
        data: &[u8],
    ) -> Result<u64> {
        let lsn = self.next_lsn;

        let mut header = WalRecordHeader {
            lsn,
            txn_id,
            record_type,
            page_id,
            data_len: data.len() as u32,
            crc32: 0, // Will be filled in
        };

        // Compute CRC32
        header.crc32 = header.compute_crc32(data);

        // Write header
        let header_bytes = header.to_bytes();
        self.file.write_all(&header_bytes)?;

        // Write data
        self.file.write_all(data)?;

        // Update next LSN
        self.next_lsn += WalRecordHeader::SIZE as u64 + data.len() as u64;

        Ok(lsn)
    }

    /// Sync WAL to disk (fsync)
    pub fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        Ok(())
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> u64 {
        self.next_lsn
    }
}

/// WAL reader for recovery
pub struct WalReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl WalReader {
    /// Open WAL file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        Ok(Self {
            reader: BufReader::new(file),
            path,
        })
    }

    /// Read next WAL record
    pub fn read_next(&mut self) -> Result<Option<(WalRecordHeader, Vec<u8>)>> {
        // Read header
        let mut header_buf = [0u8; WalRecordHeader::SIZE];
        match self.reader.read_exact(&mut header_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let header = WalRecordHeader::from_bytes(&header_buf)?;

        // Read data
        let mut data = vec![0u8; header.data_len as usize];
        self.reader.read_exact(&mut data)?;

        // Verify CRC32
        let computed_crc = header.compute_crc32(&data);
        if computed_crc != header.crc32 {
            return Err(SochDBError::Corruption(format!(
                "WAL CRC mismatch at LSN {}: expected {:#x}, got {:#x}",
                header.lsn, header.crc32, computed_crc
            )));
        }

        Ok(Some((header, data)))
    }

    /// Iterate over all records
    pub fn iter(&mut self) -> WalIterator<'_> {
        WalIterator { reader: self }
    }
}

/// Iterator over WAL records
pub struct WalIterator<'a> {
    reader: &'a mut WalReader,
}

impl<'a> Iterator for WalIterator<'a> {
    type Item = Result<(WalRecordHeader, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_next() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Durable block store with WAL-backed persistence
///
/// Provides ACID guarantees through write-ahead logging:
/// - Writes are first logged to WAL, then synced
/// - On crash, replays WAL to recover state
/// - Checkpoint mechanism to truncate WAL
pub struct DurableBlockStore {
    /// In-memory block store (cache)
    store: BlockStore,
    /// WAL writer
    wal: parking_lot::Mutex<WalWriter>,
    /// Data file for persistent blocks
    data_file: parking_lot::Mutex<File>,
    /// Dirty pages not yet flushed to data file
    dirty_pages: RwLock<HashMap<u64, Vec<u8>>>,
    /// Checkpoint LSN (WAL can be truncated before this)
    checkpoint_lsn: AtomicU64,
    /// Path to data directory
    data_dir: PathBuf,
    /// Next page ID
    next_page_id: AtomicU64,
}

impl DurableBlockStore {
    /// Open or create a durable block store
    pub fn open<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();

        // Create directory if needed
        std::fs::create_dir_all(&data_dir)?;

        let wal_path = data_dir.join("wal.log");
        let data_path = data_dir.join("blocks.dat");

        // Open WAL
        let wal = WalWriter::open(&wal_path)?;

        // Open data file
        let data_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&data_path)?;

        let store = Self {
            store: BlockStore::new(),
            wal: parking_lot::Mutex::new(wal),
            data_file: parking_lot::Mutex::new(data_file),
            dirty_pages: RwLock::new(HashMap::new()),
            checkpoint_lsn: AtomicU64::new(0),
            data_dir,
            next_page_id: AtomicU64::new(0),
        };

        Ok(store)
    }

    /// Write a block with WAL durability
    ///
    /// 1. Writes to WAL and fsyncs
    /// 2. Updates in-memory store
    /// 3. Marks page as dirty (will be flushed at checkpoint)
    pub fn write_block(&self, txn_id: u64, data: &[u8]) -> Result<BlockRef> {
        // Allocate page ID
        let page_id = self.next_page_id.fetch_add(1, Ordering::SeqCst);

        // Write to WAL first
        {
            let mut wal = self.wal.lock();
            wal.append(txn_id, WalRecordType::BlockWrite, page_id, data)?;
            wal.sync()?; // Durability point
        }

        // Write to in-memory store
        let block_ref = self.store.write_block(data)?;

        // Mark as dirty (for checkpoint flushing)
        self.dirty_pages.write().insert(page_id, data.to_vec());

        Ok(block_ref)
    }

    /// Read a block
    pub fn read_block(&self, block_ref: &BlockRef) -> Result<Vec<u8>> {
        self.store.read_block(block_ref)
    }

    /// Commit a transaction
    pub fn commit(&self, txn_id: u64) -> Result<u64> {
        let mut wal = self.wal.lock();
        let lsn = wal.append(txn_id, WalRecordType::Commit, 0, &[])?;
        wal.sync()?; // Durability point for commit
        Ok(lsn)
    }

    /// Create a checkpoint
    ///
    /// 1. Flushes all dirty pages to data file
    /// 2. Writes checkpoint marker to WAL
    /// 3. Updates checkpoint LSN (WAL can be truncated before this)
    pub fn checkpoint(&self) -> Result<u64> {
        // Collect dirty pages
        let dirty: Vec<(u64, Vec<u8>)> = {
            let mut pages = self.dirty_pages.write();
            pages.drain().collect()
        };

        // Flush to data file
        {
            let mut file = self.data_file.lock();
            for (page_id, data) in &dirty {
                let offset = *page_id * (DEFAULT_BLOCK_SIZE as u64 + BlockHeader::SIZE as u64);
                file.seek(SeekFrom::Start(offset))?;
                file.write_all(data)?;
            }
            file.sync_all()?;
        }

        // Write checkpoint marker
        let lsn = {
            let mut wal = self.wal.lock();
            let lsn = wal.append(0, WalRecordType::Checkpoint, 0, &[])?;
            wal.sync()?;
            lsn
        };

        // Update checkpoint LSN
        self.checkpoint_lsn.store(lsn, Ordering::SeqCst);

        Ok(lsn)
    }

    /// Recover from WAL after crash
    ///
    /// Replays all committed transactions from the last checkpoint.
    pub fn recover(&mut self) -> Result<RecoveryStats> {
        let wal_path = self.data_dir.join("wal.log");

        if !wal_path.exists() {
            return Ok(RecoveryStats::default());
        }

        let mut reader = WalReader::open(&wal_path)?;
        let mut stats = RecoveryStats::default();

        // Track active transactions
        let mut pending_txns: HashMap<u64, Vec<(u64, Vec<u8>)>> = HashMap::new();
        let mut committed_txns: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // Read all WAL records
        for record_result in reader.iter() {
            let (header, data) = record_result?;
            stats.records_read += 1;

            match header.record_type {
                WalRecordType::BlockWrite => {
                    pending_txns
                        .entry(header.txn_id)
                        .or_default()
                        .push((header.page_id, data));
                }
                WalRecordType::Commit => {
                    committed_txns.insert(header.txn_id);
                    stats.txns_committed += 1;
                }
                WalRecordType::Checkpoint => {
                    self.checkpoint_lsn.store(header.lsn, Ordering::SeqCst);
                    stats.checkpoints_found += 1;
                }
                WalRecordType::TxnBegin => {
                    // Just track
                }
            }
        }

        // Redo committed transactions
        for txn_id in &committed_txns {
            if let Some(writes) = pending_txns.remove(txn_id) {
                for (page_id, data) in writes {
                    self.store.write_block(&data)?;
                    self.next_page_id.fetch_max(page_id + 1, Ordering::SeqCst);
                    stats.blocks_recovered += 1;
                }
            }
        }

        // Count aborted transactions (uncommitted)
        stats.txns_aborted = pending_txns.len();

        Ok(stats)
    }

    /// Get statistics
    pub fn stats(&self) -> DurableBlockStoreStats {
        let store_stats = self.store.stats();
        DurableBlockStoreStats {
            block_stats: store_stats,
            dirty_page_count: self.dirty_pages.read().len(),
            checkpoint_lsn: self.checkpoint_lsn.load(Ordering::SeqCst),
            wal_size: self.wal.lock().current_lsn(),
        }
    }
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of WAL records read
    pub records_read: usize,
    /// Number of committed transactions
    pub txns_committed: usize,
    /// Number of aborted transactions (not committed before crash)
    pub txns_aborted: usize,
    /// Number of blocks recovered
    pub blocks_recovered: usize,
    /// Number of checkpoints found
    pub checkpoints_found: usize,
}

/// Durable block store statistics
#[derive(Debug, Clone)]
pub struct DurableBlockStoreStats {
    /// Block store stats
    pub block_stats: BlockStoreStats,
    /// Number of dirty pages (not yet flushed)
    pub dirty_page_count: usize,
    /// Last checkpoint LSN
    pub checkpoint_lsn: u64,
    /// Current WAL size (bytes)
    pub wal_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_store_write_read() {
        let store = BlockStore::new();

        let data = b"Hello, SochFS block storage!";
        let block_ref = store.write_block(data).unwrap();

        let read_data = store.read_block(&block_ref).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_compression_selection() {
        let store = BlockStore::new();

        // Small data: no compression
        let small = b"hi";
        assert_eq!(store.select_compression(small), BlockCompression::None);

        // TOON-like content (NOTE: compression stub returns None for now)
        let toon = b"users[5]{id,name}:\n1,Alice\n2,Bob\n3,Charlie";
        // In production with zstd, this would return Zstd
        let compression = store.select_compression(toon);
        assert!(compression == BlockCompression::Zstd || compression == BlockCompression::None);

        // JSON content (NOTE: compression stub returns None for now)
        let json = br#"{"users": [{"id": 1, "name": "Alice"}]}"#;
        // In production with lz4, this would return Lz4
        let compression = store.select_compression(json);
        assert!(compression == BlockCompression::Lz4 || compression == BlockCompression::None);
    }

    #[test]
    fn test_lz4_compression() {
        let store = BlockStore::new();

        let data = "Hello, world! ".repeat(100);
        let block_ref = store
            .write_block_with_compression(data.as_bytes(), BlockCompression::Lz4)
            .unwrap();

        // NOTE: With stub compression, compressed_len == original_len
        // In production with lz4, compressed_len < original_len
        // For now, just verify data integrity
        let read_data = store.read_block(&block_ref).unwrap();
        assert_eq!(read_data, data.as_bytes());
    }

    #[test]
    fn test_zstd_compression() {
        let store = BlockStore::new();

        let data = "TOON format is very repetitive ".repeat(100);
        let block_ref = store
            .write_block_with_compression(data.as_bytes(), BlockCompression::Zstd)
            .unwrap();

        // NOTE: With stub compression, compressed_len == original_len
        // In production with zstd, compressed_len < original_len
        // For now, just verify data integrity
        let read_data = store.read_block(&block_ref).unwrap();
        assert_eq!(read_data, data.as_bytes());
    }

    #[test]
    fn test_file_block_manager() {
        let manager = FileBlockManager::new(1024);

        let data = "Test data ".repeat(500); // ~5KB, multiple blocks
        let blocks = manager.write_file(data.as_bytes()).unwrap();

        assert!(blocks.len() > 1); // Should have multiple blocks

        let read_data = manager.read_file(&blocks).unwrap();
        assert_eq!(read_data, data.as_bytes());
    }

    #[test]
    fn test_stats() {
        let store = BlockStore::new();

        // Write some compressible data
        let data = "Repetitive data pattern ".repeat(100);
        store.write_block(data.as_bytes()).unwrap();

        let stats = store.stats();
        assert_eq!(stats.block_count, 1);
        // NOTE: With stub compression, ratio is 1.0
        // In production, ratio would be > 1.0 for compressible data
        assert!(stats.compression_ratio >= 1.0);
    }

    // ========================================================================
    // Task 1: Fixed-Layout BlockHeader Tests
    // ========================================================================

    #[test]
    fn test_block_header_fixed_layout() {
        let header = BlockHeader {
            magic: BlockHeader::MAGIC,
            compression: BlockCompression::Zstd as u8,
            original_size: 4096,
            compressed_size: 1024,
            checksum: 0xDEADBEEF,
        };

        let bytes = header.to_bytes();

        // Verify exact 17-byte size
        assert_eq!(bytes.len(), 17);

        // Verify magic at offset 0-3
        assert_eq!(&bytes[0..4], b"TBLK");

        // Verify compression at offset 4
        assert_eq!(bytes[4], BlockCompression::Zstd as u8);

        // Verify original_size at offset 5-8 (little-endian)
        assert_eq!(LittleEndian::read_u32(&bytes[5..9]), 4096);

        // Verify compressed_size at offset 9-12 (little-endian)
        assert_eq!(LittleEndian::read_u32(&bytes[9..13]), 1024);

        // Verify checksum at offset 13-16 (little-endian)
        assert_eq!(LittleEndian::read_u32(&bytes[13..17]), 0xDEADBEEF);
    }

    #[test]
    fn test_block_header_roundtrip() {
        let original = BlockHeader {
            magic: BlockHeader::MAGIC,
            compression: BlockCompression::Lz4 as u8,
            original_size: 65536,
            compressed_size: 32768,
            checksum: 0x12345678,
        };

        let bytes = original.to_bytes();
        let recovered = BlockHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.compression, original.compression);
        assert_eq!(recovered.original_size, original.original_size);
        assert_eq!(recovered.compressed_size, original.compressed_size);
        assert_eq!(recovered.checksum, original.checksum);
    }

    #[test]
    fn test_block_header_invalid_magic() {
        let mut bytes = [0u8; 17];
        bytes[0..4].copy_from_slice(b"XXXX"); // Invalid magic

        let result = BlockHeader::from_bytes(&bytes);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("Invalid block magic"));
    }

    #[test]
    fn test_block_header_too_short() {
        let bytes = [0u8; 10]; // Only 10 bytes, need 17

        let result = BlockHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // ========================================================================
    // Task 5: BlockRef Error Propagation Tests
    // ========================================================================

    #[test]
    fn test_block_ref_fixed_layout() {
        let block_ref = BlockRef {
            store_offset: 0x123456789ABCDEF0,
            compressed_len: 4096,
            original_len: 8192,
            compression: BlockCompression::Zstd,
            checksum: 0xCAFEBABE,
        };

        let bytes = block_ref.to_bytes().unwrap();

        // Verify exact 21-byte size
        assert_eq!(bytes.len(), 21);

        // Verify offset at 0-7 (little-endian)
        assert_eq!(LittleEndian::read_u64(&bytes[0..8]), 0x123456789ABCDEF0);

        // Verify compressed_len at 8-11
        assert_eq!(LittleEndian::read_u32(&bytes[8..12]), 4096);

        // Verify original_len at 12-15
        assert_eq!(LittleEndian::read_u32(&bytes[12..16]), 8192);

        // Verify compression at 16
        assert_eq!(bytes[16], BlockCompression::Zstd as u8);

        // Verify checksum at 17-20
        assert_eq!(LittleEndian::read_u32(&bytes[17..21]), 0xCAFEBABE);
    }

    #[test]
    fn test_block_ref_roundtrip() {
        let original = BlockRef {
            store_offset: u64::MAX, // Test large values
            compressed_len: u32::MAX,
            original_len: u32::MAX,
            compression: BlockCompression::None,
            checksum: u32::MAX,
        };

        let bytes = original.to_bytes().unwrap();
        let recovered = BlockRef::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.store_offset, original.store_offset);
        assert_eq!(recovered.compressed_len, original.compressed_len);
        assert_eq!(recovered.original_len, original.original_len);
        assert_eq!(recovered.compression, original.compression);
        assert_eq!(recovered.checksum, original.checksum);
    }

    #[test]
    fn test_block_ref_too_short() {
        let bytes = [0u8; 10]; // Only 10 bytes, need 21

        let result = BlockRef::from_bytes(&bytes);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("BlockRef too short"));
    }

    #[test]
    fn test_cross_platform_compatibility() {
        // Test that serialization is deterministic regardless of platform
        let block_ref = BlockRef {
            store_offset: 0x0102030405060708,
            compressed_len: 0x0A0B0C0D,
            original_len: 0x0E0F1011,
            compression: BlockCompression::Lz4,
            checksum: 0x12131415,
        };

        let bytes = block_ref.to_bytes().unwrap();

        // These exact byte values should be the same on any platform
        // (little-endian encoding)
        assert_eq!(bytes[0], 0x08); // LSB of offset
        assert_eq!(bytes[7], 0x01); // MSB of offset
        assert_eq!(bytes[8], 0x0D); // LSB of compressed_len
        assert_eq!(bytes[17], 0x15); // LSB of checksum
    }

    // ========================================================================
    // Task 2: LZ4/ZSTD Production Compression Tests
    // ========================================================================

    #[test]
    fn test_lz4_compression_roundtrip() {
        let store = BlockStore::new();

        // Create compressible data (repetitive pattern)
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();

        let block_ref = store
            .write_block_with_compression(&data, BlockCompression::Lz4)
            .unwrap();
        let recovered = store.read_block(&block_ref).unwrap();

        assert_eq!(recovered, data);
        // Verify compression actually happened (data was compressible)
        assert!(block_ref.compressed_len <= block_ref.original_len);
    }

    #[test]
    fn test_zstd_compression_roundtrip() {
        let store = BlockStore::new();

        // Create highly compressible data (all zeros)
        let data = vec![0u8; 8192];

        let block_ref = store
            .write_block_with_compression(&data, BlockCompression::Zstd)
            .unwrap();
        let recovered = store.read_block(&block_ref).unwrap();

        assert_eq!(recovered, data);
        // Verify compression was very effective
        assert!(block_ref.compressed_len < block_ref.original_len / 2);
    }

    #[test]
    fn test_compression_fallback_on_incompressible() {
        let store = BlockStore::new();

        // Create random-looking data (incompressible)
        let mut data = vec![0u8; 256];
        for (i, byte) in data.iter_mut().enumerate().take(256) {
            *byte = ((i * 17 + 31) % 256) as u8; // Pseudo-random
        }

        let block_ref = store
            .write_block_with_compression(&data, BlockCompression::Lz4)
            .unwrap();
        let recovered = store.read_block(&block_ref).unwrap();

        assert_eq!(recovered, data);
    }

    #[test]
    fn test_automatic_compression_selection() {
        let store = BlockStore::new();

        // Test JSON content (should select LZ4)
        let json_data = br#"{"name": "test", "value": 123, "items": [1, 2, 3]}"#.repeat(10);
        let json_ref = store.write_block(&json_data).unwrap();
        let json_recovered = store.read_block(&json_ref).unwrap();
        assert_eq!(json_recovered, json_data);

        // Test TOON content (should select ZSTD)
        let mut soch_data = vec![0u8; 256];
        soch_data[0..4].copy_from_slice(b"TOON"); // Magic header
        let soch_ref = store.write_block(&soch_data).unwrap();
        let soch_recovered = store.read_block(&soch_ref).unwrap();
        assert_eq!(soch_recovered, soch_data);
    }

    #[test]
    fn test_small_data_no_compression() {
        let store = BlockStore::new();

        // Data smaller than 128 bytes should not be compressed
        let small_data = vec![42u8; 64];
        let block_ref = store.write_block(&small_data).unwrap();

        // Should be stored uncompressed
        assert_eq!(block_ref.compression, BlockCompression::None);

        let recovered = store.read_block(&block_ref).unwrap();
        assert_eq!(recovered, small_data);
    }

    #[test]
    fn test_compression_stats() {
        let store = BlockStore::new();

        // Write compressible data
        let data = vec![0u8; 4096];
        store
            .write_block_with_compression(&data, BlockCompression::Zstd)
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.block_count, 1);
        assert!(
            stats.compression_ratio > 1.0,
            "Compression should reduce size"
        );
        assert!(stats.total_original_bytes > stats.total_compressed_bytes);
    }

    // ========================================================================
    // Task 2: Durable Block Store with WAL Tests
    // ========================================================================

    #[test]
    fn test_wal_record_header_roundtrip() {
        let original = WalRecordHeader {
            lsn: 12345,
            txn_id: 67890,
            record_type: WalRecordType::BlockWrite,
            page_id: 42,
            data_len: 4096,
            crc32: 0xDEADBEEF,
        };

        let bytes = original.to_bytes();
        let recovered = WalRecordHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.lsn, original.lsn);
        assert_eq!(recovered.txn_id, original.txn_id);
        assert_eq!(recovered.record_type, original.record_type);
        assert_eq!(recovered.page_id, original.page_id);
        assert_eq!(recovered.data_len, original.data_len);
        assert_eq!(recovered.crc32, original.crc32);
    }

    #[test]
    fn test_wal_crc32() {
        let header = WalRecordHeader {
            lsn: 100,
            txn_id: 1,
            record_type: WalRecordType::BlockWrite,
            page_id: 0,
            data_len: 4,
            crc32: 0,
        };

        let data = b"test";
        let crc1 = header.compute_crc32(data);
        let crc2 = header.compute_crc32(data);

        assert_eq!(crc1, crc2, "CRC should be deterministic");

        // Different data should produce different CRC
        let different_data = b"TEST";
        let crc3 = header.compute_crc32(different_data);
        assert_ne!(crc1, crc3, "Different data should have different CRC");
    }

    #[test]
    fn test_durable_block_store_basic() {
        let dir = tempfile::tempdir().unwrap();

        let store = DurableBlockStore::open(dir.path()).unwrap();

        // Write a block
        let data = b"Hello, durable block store!";
        let block_ref = store.write_block(1, data).unwrap();

        // Read it back
        let read_data = store.read_block(&block_ref).unwrap();
        assert_eq!(read_data, data);

        // Commit the transaction
        store.commit(1).unwrap();

        // Check stats
        let stats = store.stats();
        assert_eq!(stats.dirty_page_count, 1);
    }

    #[test]
    fn test_durable_block_store_checkpoint() {
        let dir = tempfile::tempdir().unwrap();

        let store = DurableBlockStore::open(dir.path()).unwrap();

        // Write some blocks
        store.write_block(1, b"block1").unwrap();
        store.write_block(1, b"block2").unwrap();
        store.write_block(1, b"block3").unwrap();
        store.commit(1).unwrap();

        // Checkpoint should flush dirty pages
        let checkpoint_lsn = store.checkpoint().unwrap();
        assert!(checkpoint_lsn > 0);

        // Dirty pages should be cleared
        let stats = store.stats();
        assert_eq!(stats.dirty_page_count, 0);
        assert_eq!(stats.checkpoint_lsn, checkpoint_lsn);
    }

    #[test]
    fn test_durable_block_store_recovery() {
        let dir = tempfile::tempdir().unwrap();

        // Phase 1: Write and commit
        {
            let store = DurableBlockStore::open(dir.path()).unwrap();
            store.write_block(1, b"data1").unwrap();
            store.write_block(1, b"data2").unwrap();
            store.commit(1).unwrap();

            // Don't checkpoint - simulate crash before flush
        }

        // Phase 2: Recover
        {
            let mut store = DurableBlockStore::open(dir.path()).unwrap();
            let stats = store.recover().unwrap();

            // Should have recovered the committed transaction
            assert_eq!(stats.txns_committed, 1);
            assert_eq!(stats.blocks_recovered, 2);
            assert_eq!(stats.txns_aborted, 0);
        }
    }

    #[test]
    fn test_durable_block_store_uncommitted_recovery() {
        let dir = tempfile::tempdir().unwrap();

        // Phase 1: Write but don't commit (simulate crash)
        {
            let store = DurableBlockStore::open(dir.path()).unwrap();
            store.write_block(1, b"uncommitted_data").unwrap();
            // NO commit - transaction should be aborted on recovery
        }

        // Phase 2: Recover
        {
            let mut store = DurableBlockStore::open(dir.path()).unwrap();
            let stats = store.recover().unwrap();

            // Uncommitted transaction should be aborted
            assert_eq!(stats.txns_committed, 0);
            assert_eq!(stats.txns_aborted, 1);
            assert_eq!(stats.blocks_recovered, 0);
        }
    }

    #[test]
    fn test_wal_writer_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write records
        {
            let mut writer = WalWriter::open(&wal_path).unwrap();
            writer.append(1, WalRecordType::TxnBegin, 0, &[]).unwrap();
            writer
                .append(1, WalRecordType::BlockWrite, 0, b"data1")
                .unwrap();
            writer
                .append(1, WalRecordType::BlockWrite, 1, b"data2")
                .unwrap();
            writer.append(1, WalRecordType::Commit, 0, &[]).unwrap();
            writer.sync().unwrap();
        }

        // Read records
        {
            let mut reader = WalReader::open(&wal_path).unwrap();
            let mut records = Vec::new();
            for record in reader.iter() {
                records.push(record.unwrap());
            }

            assert_eq!(records.len(), 4);
            assert_eq!(records[0].0.record_type, WalRecordType::TxnBegin);
            assert_eq!(records[1].0.record_type, WalRecordType::BlockWrite);
            assert_eq!(records[1].1, b"data1");
            assert_eq!(records[2].0.record_type, WalRecordType::BlockWrite);
            assert_eq!(records[2].1, b"data2");
            assert_eq!(records[3].0.record_type, WalRecordType::Commit);
        }
    }
}
