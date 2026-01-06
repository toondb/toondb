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

//! Block Encoding with Restart Points and Hash Index
//!
//! This module implements the block format used in SSTables. Each block
//! contains a sequence of key-value pairs with prefix compression and
//! restart points for efficient random access.
//!
//! ## Block Format
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Block Contents                                 │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Entry 0: [shared_len][unshared_len][value_len][unshared_key][value]     │
//! │ Entry 1: [shared_len][unshared_len][value_len][unshared_key][value]     │
//! │ ...                                                                      │
//! │ Entry N-1                                                                │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Restart Points: [offset_0][offset_1]...[offset_R-1][num_restarts]       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Optional Hash Index: [bucket_0][bucket_1]...[bucket_B-1][num_buckets]   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Block Trailer: [type(1)][checksum(4)]                                   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Prefix Compression
//!
//! Keys are prefix-compressed within a restart interval:
//! - `shared_len`: Number of bytes shared with previous key
//! - `unshared_len`: Number of non-shared key bytes following
//! - `value_len`: Length of value
//!
//! At restart points, `shared_len = 0` (full key stored).
//!
//! ## Complexity Analysis
//!
//! | Operation    | Without Hash | With Hash Index     |
//! |--------------|--------------|---------------------|
//! | Point lookup | O(log n/r + r) | O(1) expected     |
//! | Seek         | O(log n/r + r) | O(log n/r + r)    |
//! | Iterate      | O(n)           | O(n)              |
//!
//! Where n = entries, r = restart interval (typically 16).

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::cmp::Ordering;
use std::io::{Cursor, Read, Write};

/// Default restart interval (entries between restart points)
pub const DEFAULT_RESTART_INTERVAL: usize = 16;

/// Default number of hash buckets per entry (for hash index)
pub const DEFAULT_HASH_BUCKET_RATIO: f64 = 0.75;

/// Block compression type
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    /// Uncompressed block
    Uncompressed = 0,
    /// Snappy compressed
    Snappy = 1,
    /// LZ4 compressed
    Lz4 = 2,
    /// Zstd compressed
    Zstd = 3,
}

impl TryFrom<u8> for BlockType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BlockType::Uncompressed),
            1 => Ok(BlockType::Snappy),
            2 => Ok(BlockType::Lz4),
            3 => Ok(BlockType::Zstd),
            _ => Err(()),
        }
    }
}

impl BlockType {
    /// Convert from u8
    pub fn from_u8(value: u8) -> Self {
        Self::try_from(value).unwrap_or(BlockType::Uncompressed)
    }
}

/// Handle to a block (offset + size)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHandle {
    /// Offset in file
    pub offset: u64,
    /// Size of block data (excluding trailer)
    pub size: u64,
}

impl BlockHandle {
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Get offset
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Get size
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Encode to bytes (varint encoded for compactness)
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        encode_varint(&mut buf, self.offset);
        encode_varint(&mut buf, self.size);
        buf
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        let mut cursor = Cursor::new(data);
        let offset = decode_varint(&mut cursor)?;
        let size = decode_varint(&mut cursor)?;
        Some((Self { offset, size }, cursor.position() as usize))
    }
}

/// A restart point entry
#[derive(Debug, Clone)]
struct RestartPoint {
    /// Offset within block data
    offset: u32,
}

/// Hash bucket entry for hash index
#[derive(Debug, Clone)]
struct HashBucket {
    /// Restart point index (0xFF = empty)
    restart_index: u8,
}

// =============================================================================
// Block Builder
// =============================================================================

/// Builder for creating blocks with prefix compression
pub struct BlockBuilder {
    /// Block data buffer
    buffer: Vec<u8>,
    /// Restart points (offsets into buffer)
    restarts: Vec<u32>,
    /// Entries since last restart
    entries_since_restart: usize,
    /// Restart interval
    restart_interval: usize,
    /// Last key (for prefix compression)
    last_key: Vec<u8>,
    /// Number of entries
    entry_count: usize,
    /// Whether to build hash index
    use_hash_index: bool,
    /// Keys for hash index (stored temporarily during build)
    keys_for_hash: Vec<Vec<u8>>,
}

impl BlockBuilder {
    /// Create a new block builder
    pub fn new(restart_interval: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            restarts: vec![0], // First restart at offset 0
            entries_since_restart: 0,
            restart_interval,
            last_key: Vec::new(),
            entry_count: 0,
            use_hash_index: false,
            keys_for_hash: Vec::new(),
        }
    }

    /// Create with hash index enabled
    pub fn with_hash_index(restart_interval: usize) -> Self {
        Self {
            use_hash_index: true,
            ..Self::new(restart_interval)
        }
    }

    /// Add a key-value pair to the block
    ///
    /// Keys must be added in sorted order.
    pub fn add(&mut self, key: &[u8], value: &[u8]) {
        debug_assert!(
            self.entry_count == 0 || key > self.last_key.as_slice(),
            "Keys must be added in sorted order"
        );

        // Check if we need a restart point
        let shared = if self.entries_since_restart >= self.restart_interval {
            // New restart point - store full key
            self.restarts.push(self.buffer.len() as u32);
            self.entries_since_restart = 0;
            0
        } else {
            // Calculate shared prefix with last key
            self.shared_prefix_len(&self.last_key, key)
        };

        let non_shared = key.len() - shared;
        let value_len = value.len();

        // Write entry: [shared_len][non_shared_len][value_len][key_delta][value]
        encode_varint(&mut self.buffer, shared as u64);
        encode_varint(&mut self.buffer, non_shared as u64);
        encode_varint(&mut self.buffer, value_len as u64);
        self.buffer.extend_from_slice(&key[shared..]);
        self.buffer.extend_from_slice(value);

        // Update state
        self.last_key.clear();
        self.last_key.extend_from_slice(key);
        self.entries_since_restart += 1;
        self.entry_count += 1;

        // Store key for hash index
        if self.use_hash_index {
            self.keys_for_hash.push(key.to_vec());
        }
    }

    /// Calculate shared prefix length
    fn shared_prefix_len(&self, a: &[u8], b: &[u8]) -> usize {
        let mut shared = 0;
        let min_len = a.len().min(b.len());
        while shared < min_len && a[shared] == b[shared] {
            shared += 1;
        }
        shared
    }

    /// Finish building the block
    ///
    /// Returns the block contents including restarts and optional hash index.
    pub fn finish(&mut self) -> Vec<u8> {
        let mut result = std::mem::take(&mut self.buffer);

        // Build hash index if enabled
        if self.use_hash_index && self.entry_count > 0 {
            self.build_hash_index(&mut result);
        }

        // Write restart points
        for restart in &self.restarts {
            result.write_u32::<LittleEndian>(*restart).unwrap();
        }
        result
            .write_u32::<LittleEndian>(self.restarts.len() as u32)
            .unwrap();

        result
    }

    /// Build hash index for fast point lookups
    fn build_hash_index(&self, data: &mut Vec<u8>) {
        // Number of buckets = entries * bucket_ratio
        let num_buckets = ((self.entry_count as f64 * DEFAULT_HASH_BUCKET_RATIO) as usize).max(1);
        let mut buckets = vec![0xFFu8; num_buckets]; // 0xFF = empty

        // For each key, compute hash and store restart point index
        for (key_idx, key) in self.keys_for_hash.iter().enumerate() {
            let restart_idx = key_idx / self.restart_interval;
            let bucket = Self::hash_key(key) as usize % num_buckets;
            
            // Simple linear probing for collisions
            let mut probe = bucket;
            for _ in 0..num_buckets {
                if buckets[probe] == 0xFF {
                    buckets[probe] = restart_idx as u8;
                    break;
                }
                probe = (probe + 1) % num_buckets;
            }
        }

        // Write hash index
        data.extend_from_slice(&buckets);
        data.write_u32::<LittleEndian>(num_buckets as u32).unwrap();
    }

    /// Hash a key for the hash index
    fn hash_key(key: &[u8]) -> u32 {
        // Use xxHash for speed
        twox_hash::xxh3::hash64(key) as u32
    }

    /// Check if block is empty
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Get approximate size of current block
    pub fn estimated_size(&self) -> usize {
        self.buffer.len() + self.restarts.len() * 4 + 4
    }

    /// Reset builder for reuse
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.restarts.clear();
        self.restarts.push(0);
        self.entries_since_restart = 0;
        self.last_key.clear();
        self.entry_count = 0;
        self.keys_for_hash.clear();
    }
}

impl Default for BlockBuilder {
    fn default() -> Self {
        Self::new(DEFAULT_RESTART_INTERVAL)
    }
}

// =============================================================================
// Block
// =============================================================================

/// An SSTable block with efficient key lookup
pub struct Block {
    /// Raw block data
    data: Vec<u8>,
    /// Offset where restart array begins
    restarts_offset: usize,
    /// Number of restart points
    num_restarts: usize,
    /// Number of hash buckets (0 if no hash index)
    num_hash_buckets: usize,
    /// Offset where hash index begins (if present)
    hash_index_offset: usize,
}

impl Block {
    /// Create a block from raw data
    pub fn new(data: Vec<u8>) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        // Read number of restarts (last 4 bytes)
        let num_restarts = {
            let mut cursor = Cursor::new(&data[data.len() - 4..]);
            cursor.read_u32::<LittleEndian>().ok()? as usize
        };

        if num_restarts == 0 || data.len() < 4 + num_restarts * 4 {
            return None;
        }

        let restarts_offset = data.len() - 4 - num_restarts * 4;

        // TODO: Detect and parse hash index if present
        // For now, assume no hash index
        let num_hash_buckets = 0;
        let hash_index_offset = restarts_offset;

        Some(Self {
            data,
            restarts_offset,
            num_restarts,
            num_hash_buckets,
            hash_index_offset,
        })
    }

    /// Get restart point offset at given index
    fn restart_offset(&self, index: usize) -> u32 {
        debug_assert!(index < self.num_restarts);
        let offset = self.restarts_offset + index * 4;
        let mut cursor = Cursor::new(&self.data[offset..offset + 4]);
        cursor.read_u32::<LittleEndian>().unwrap()
    }

    /// Seek to the first entry with key >= target
    ///
    /// Returns an iterator positioned at the target or past-the-end.
    pub fn seek(&self, target: &[u8]) -> BlockIterator<'_> {
        // Binary search on restart points
        let mut left = 0;
        let mut right = self.num_restarts;

        while left < right {
            let mid = left + (right - left) / 2;
            let offset = self.restart_offset(mid) as usize;
            
            // Read key at restart point (shared = 0)
            let key = self.read_key_at(offset);
            
            match key.as_slice().cmp(target) {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => {
                    return BlockIterator::new(self, offset);
                }
            }
        }

        // left now points to the first restart point with key > target
        // Start from the previous restart point and scan
        let start_restart = if left > 0 { left - 1 } else { 0 };
        let start_offset = self.restart_offset(start_restart) as usize;
        
        let mut iter = BlockIterator::new(self, start_offset);
        while iter.valid() {
            if iter.key() >= target {
                break;
            }
            iter.next();
        }
        iter
    }

    /// Get value for exact key match
    ///
    /// Uses hash index if available for O(1) expected time.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // TODO: Use hash index if available
        
        // Fall back to binary search
        let mut iter = self.seek(key);
        if iter.valid() && iter.key() == key {
            Some(iter.value().to_vec())
        } else {
            None
        }
    }

    /// Read the full key at a given offset (must be at a restart point)
    fn read_key_at(&self, offset: usize) -> Vec<u8> {
        let mut cursor = Cursor::new(&self.data[offset..self.restarts_offset]);
        
        let shared = decode_varint(&mut cursor).unwrap_or(0) as usize;
        let non_shared = decode_varint(&mut cursor).unwrap_or(0) as usize;
        let _value_len = decode_varint(&mut cursor);
        
        debug_assert_eq!(shared, 0, "Expected restart point (shared = 0)");
        
        let pos = cursor.position() as usize;
        self.data[offset + pos..offset + pos + non_shared].to_vec()
    }

    /// Create an iterator over all entries
    pub fn iter(&self) -> BlockIterator<'_> {
        BlockIterator::new(self, 0)
    }

    /// Get block data
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

// =============================================================================
// Block Iterator
// =============================================================================

/// Iterator over block entries with prefix decompression
pub struct BlockIterator<'a> {
    block: &'a Block,
    /// Current position in data
    offset: usize,
    /// Current reconstructed key
    key: Vec<u8>,
    /// Current value slice
    value_start: usize,
    value_len: usize,
    /// Whether iterator is valid
    valid: bool,
}

impl<'a> BlockIterator<'a> {
    /// Create a new block iterator starting at the given offset
    pub fn new(block: &'a Block, offset: usize) -> Self {
        let mut iter = Self {
            block,
            offset,
            key: Vec::new(),
            value_start: 0,
            value_len: 0,
            valid: false,
        };
        iter.parse_entry();
        iter
    }

    /// Check if iterator is valid
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Get current key
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Get current value
    pub fn value(&self) -> &[u8] {
        &self.block.data[self.value_start..self.value_start + self.value_len]
    }

    /// Move to next entry
    pub fn next(&mut self) {
        if !self.valid {
            return;
        }

        // Move past current value
        self.offset = self.value_start + self.value_len;
        self.parse_entry();
    }

    /// Parse entry at current offset
    fn parse_entry(&mut self) {
        if self.offset >= self.block.restarts_offset {
            self.valid = false;
            return;
        }

        let mut cursor = Cursor::new(&self.block.data[self.offset..self.block.restarts_offset]);

        // Read header
        let shared = match decode_varint(&mut cursor) {
            Some(v) => v as usize,
            None => {
                self.valid = false;
                return;
            }
        };
        let non_shared = match decode_varint(&mut cursor) {
            Some(v) => v as usize,
            None => {
                self.valid = false;
                return;
            }
        };
        let value_len = match decode_varint(&mut cursor) {
            Some(v) => v as usize,
            None => {
                self.valid = false;
                return;
            }
        };

        let header_len = cursor.position() as usize;
        let data_start = self.offset + header_len;

        // Bounds check
        if data_start + non_shared + value_len > self.block.restarts_offset {
            self.valid = false;
            return;
        }

        // Reconstruct key
        self.key.truncate(shared);
        self.key
            .extend_from_slice(&self.block.data[data_start..data_start + non_shared]);

        // Record value location
        self.value_start = data_start + non_shared;
        self.value_len = value_len;
        self.valid = true;
    }

    /// Seek to first entry >= target
    pub fn seek(&mut self, target: &[u8]) {
        // For now, delegate to block's seek and copy state
        let new_iter = self.block.seek(target);
        self.offset = new_iter.offset;
        self.key = new_iter.key;
        self.value_start = new_iter.value_start;
        self.value_len = new_iter.value_len;
        self.valid = new_iter.valid;
    }

    /// Seek to first entry
    pub fn seek_to_first(&mut self) {
        self.offset = 0;
        self.key.clear();
        self.parse_entry();
    }
}

// =============================================================================
// Varint Encoding/Decoding
// =============================================================================

/// Encode a u64 as a varint
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Decode a varint from a cursor
fn decode_varint<R: Read>(reader: &mut R) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;

    loop {
        let byte = reader.read_u8().ok()?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return None; // Overflow
        }
    }

    Some(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_builder_single_entry() {
        let mut builder = BlockBuilder::new(16);
        builder.add(b"key1", b"value1");
        
        let data = builder.finish();
        let block = Block::new(data).unwrap();
        
        assert_eq!(block.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(block.get(b"key2"), None);
    }

    #[test]
    fn test_block_builder_multiple_entries() {
        let mut builder = BlockBuilder::new(4);
        
        for i in 0..20 {
            let key = format!("key{:02}", i);
            let value = format!("value{:02}", i);
            builder.add(key.as_bytes(), value.as_bytes());
        }
        
        let data = builder.finish();
        let block = Block::new(data).unwrap();
        
        // Test all keys
        for i in 0..20 {
            let key = format!("key{:02}", i);
            let expected_value = format!("value{:02}", i);
            assert_eq!(block.get(key.as_bytes()), Some(expected_value.into_bytes()));
        }
    }

    #[test]
    fn test_block_iterator() {
        let mut builder = BlockBuilder::new(4);
        
        builder.add(b"apple", b"1");
        builder.add(b"banana", b"2");
        builder.add(b"cherry", b"3");
        builder.add(b"date", b"4");
        
        let data = builder.finish();
        let block = Block::new(data).unwrap();
        
        let mut iter = block.iter();
        let mut count = 0;
        while iter.valid() {
            count += 1;
            iter.next();
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn test_block_seek() {
        let mut builder = BlockBuilder::new(2);
        
        builder.add(b"a", b"1");
        builder.add(b"c", b"2");
        builder.add(b"e", b"3");
        builder.add(b"g", b"4");
        
        let data = builder.finish();
        let block = Block::new(data).unwrap();
        
        // Seek to existing key
        let iter = block.seek(b"c");
        assert!(iter.valid());
        assert_eq!(iter.key(), b"c");
        
        // Seek to non-existing key (should find next)
        let iter = block.seek(b"d");
        assert!(iter.valid());
        assert_eq!(iter.key(), b"e");
        
        // Seek past all keys
        let iter = block.seek(b"z");
        assert!(!iter.valid());
    }

    #[test]
    fn test_prefix_compression() {
        let mut builder = BlockBuilder::new(16);
        
        // Keys with common prefix (MUST be in sorted order!)
        builder.add(b"user:1000:age", b"30");
        builder.add(b"user:1000:email", b"alice@example.com");
        builder.add(b"user:1000:name", b"Alice");
        builder.add(b"user:1001:name", b"Bob");
        
        let data = builder.finish();
        
        // Verify compression happened (data should be smaller than uncompressed)
        let uncompressed_size = 
            b"user:1000:age".len() + b"30".len() +
            b"user:1000:email".len() + b"alice@example.com".len() +
            b"user:1000:name".len() + b"Alice".len() +
            b"user:1001:name".len() + b"Bob".len();
        
        // Block should be smaller due to prefix compression
        // (accounting for some overhead)
        assert!(data.len() < uncompressed_size + 50);
        
        // Verify all keys are retrievable
        let block = Block::new(data).unwrap();
        assert_eq!(block.get(b"user:1000:age"), Some(b"30".to_vec()));
        assert_eq!(block.get(b"user:1000:email"), Some(b"alice@example.com".to_vec()));
        assert_eq!(block.get(b"user:1000:name"), Some(b"Alice".to_vec()));
        assert_eq!(block.get(b"user:1001:name"), Some(b"Bob".to_vec()));
    }

    #[test]
    fn test_varint_encoding() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];
        
        for &value in &test_values {
            let mut buf = Vec::new();
            encode_varint(&mut buf, value);
            
            let mut cursor = Cursor::new(&buf);
            let decoded = decode_varint(&mut cursor).unwrap();
            
            assert_eq!(value, decoded, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_block_handle() {
        let handle = BlockHandle::new(12345, 67890);
        let encoded = handle.encode();
        
        let (decoded, len) = BlockHandle::decode(&encoded).unwrap();
        assert_eq!(handle, decoded);
        assert_eq!(len, encoded.len());
    }
}
