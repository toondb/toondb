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

//! Two-Level Index for SSTables
//!
//! Implements a hierarchical index structure to reduce memory usage and
//! SSTable open time for large files.
//!
//! ## jj.md Task 1: Two-Level Index Structure
//!
//! Goals:
//! - Reduce SSTable open time from O(index_size) to O(1)
//! - Reduce memory usage by 10-50x for large SSTables
//! - Improve cold-start latency from seconds to milliseconds
//!
//! ## Architecture
//!
//! ```text
//! Two-Level Index Structure:
//! ├── Data Blocks (64KB each)
//! │   └── Sorted edges
//! ├── Block Index (loaded on-demand)
//! │   └── [min_key, max_key, offset] per block
//! └── Top-Level Index (in footer, always loaded)
//!     └── Fence pointers: every 1MB of block index
//!
//! Lookup: O(log(N/B)) where B = block size
//! Memory: O(fence_count) ≈ O(file_size / 1MB) instead of O(N/block_size)
//! ```
//!
//! Reference: RocksDB BlockBasedTableFormat - https://github.com/facebook/rocksdb/wiki/BlockBasedTable-Format

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read};

/// Size of each data block in bytes (64KB - cache friendly)
pub const DATA_BLOCK_SIZE: usize = 64 * 1024;

/// Distance between fence pointers in the block index (1MB of block index data)
pub const FENCE_INTERVAL_BYTES: usize = 1024 * 1024;

/// Size of each block index entry: min_ts(8) + min_edge_id(16) + max_ts(8) + max_edge_id(16) + offset(8) + length(4)
pub const BLOCK_INDEX_ENTRY_SIZE: usize = 60;

/// Size of each fence pointer entry: min_ts(8) + min_edge_id(16) + block_index_offset(8)
pub const FENCE_POINTER_SIZE: usize = 32;

/// A key in the temporal index (timestamp + edge_id for uniqueness)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TemporalKey {
    pub timestamp_us: u64,
    pub edge_id: u128,
}

impl TemporalKey {
    pub fn new(timestamp_us: u64, edge_id: u128) -> Self {
        Self {
            timestamp_us,
            edge_id,
        }
    }

    /// Minimum possible key
    pub fn min() -> Self {
        Self {
            timestamp_us: 0,
            edge_id: 0,
        }
    }

    /// Maximum possible key
    pub fn max() -> Self {
        Self {
            timestamp_us: u64::MAX,
            edge_id: u128::MAX,
        }
    }
}

/// Entry in the block-level index (Level 2)
///
/// Each entry describes a data block's key range and location.
#[derive(Debug, Clone, Copy)]
pub struct BlockIndexEntry {
    /// Minimum key in this block
    pub min_key: TemporalKey,
    /// Maximum key in this block
    pub max_key: TemporalKey,
    /// Offset of the block in the data section
    pub offset: u64,
    /// Length of the block in bytes
    pub length: u32,
}

impl BlockIndexEntry {
    /// Check if a key falls within this block's range
    pub fn contains_key(&self, key: &TemporalKey) -> bool {
        *key >= self.min_key && *key <= self.max_key
    }

    /// Check if a timestamp range overlaps with this block
    pub fn overlaps_range(&self, start_ts: u64, end_ts: u64) -> bool {
        self.max_key.timestamp_us >= start_ts && self.min_key.timestamp_us <= end_ts
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; BLOCK_INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; BLOCK_INDEX_ENTRY_SIZE];
        let mut cursor = Cursor::new(&mut buf[..]);

        cursor
            .write_u64::<LittleEndian>(self.min_key.timestamp_us)
            .unwrap();
        cursor
            .write_u128::<LittleEndian>(self.min_key.edge_id)
            .unwrap();
        cursor
            .write_u64::<LittleEndian>(self.max_key.timestamp_us)
            .unwrap();
        cursor
            .write_u128::<LittleEndian>(self.max_key.edge_id)
            .unwrap();
        cursor.write_u64::<LittleEndian>(self.offset).unwrap();
        cursor.write_u32::<LittleEndian>(self.length).unwrap();

        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        if bytes.len() < BLOCK_INDEX_ENTRY_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Block index entry too short",
            ));
        }

        let mut cursor = Cursor::new(bytes);

        let min_timestamp_us = cursor.read_u64::<LittleEndian>()?;
        let min_edge_id = cursor.read_u128::<LittleEndian>()?;
        let max_timestamp_us = cursor.read_u64::<LittleEndian>()?;
        let max_edge_id = cursor.read_u128::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        let length = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            min_key: TemporalKey::new(min_timestamp_us, min_edge_id),
            max_key: TemporalKey::new(max_timestamp_us, max_edge_id),
            offset,
            length,
        })
    }
}

/// Fence pointer in the top-level index (Level 1)
///
/// Points to a position in the block index. Used to quickly narrow
/// down which section of the block index to load.
#[derive(Debug, Clone, Copy)]
pub struct FencePointer {
    /// First key at this fence position
    pub key: TemporalKey,
    /// Offset within the block index section
    pub block_index_offset: u64,
}

impl FencePointer {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; FENCE_POINTER_SIZE] {
        let mut buf = [0u8; FENCE_POINTER_SIZE];
        let mut cursor = Cursor::new(&mut buf[..]);

        cursor
            .write_u64::<LittleEndian>(self.key.timestamp_us)
            .unwrap();
        cursor.write_u128::<LittleEndian>(self.key.edge_id).unwrap();
        cursor
            .write_u64::<LittleEndian>(self.block_index_offset)
            .unwrap();

        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        if bytes.len() < FENCE_POINTER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Fence pointer too short",
            ));
        }

        let mut cursor = Cursor::new(bytes);

        let timestamp_us = cursor.read_u64::<LittleEndian>()?;
        let edge_id = cursor.read_u128::<LittleEndian>()?;
        let block_index_offset = cursor.read_u64::<LittleEndian>()?;

        Ok(Self {
            key: TemporalKey::new(timestamp_us, edge_id),
            block_index_offset,
        })
    }
}

/// Two-level index for efficient lookups with minimal memory usage
///
/// The top-level fence pointers are always in memory (~1KB per GB of data).
/// Block index entries are loaded on-demand.
#[derive(Debug)]
pub struct TwoLevelIndex {
    /// Top-level fence pointers (always in memory)
    pub fence_pointers: Vec<FencePointer>,

    /// Total number of blocks in the SSTable
    pub total_blocks: u32,

    /// Offset of the block index section in the file
    pub block_index_offset: u64,

    /// Length of the block index section
    pub block_index_length: u64,
}

impl TwoLevelIndex {
    /// Create a new two-level index from block entries
    ///
    /// This is called during SSTable creation to build the index structure.
    pub fn build(blocks: &[BlockIndexEntry], block_index_offset: u64) -> Self {
        let mut fence_pointers = Vec::new();
        let mut current_offset = 0u64;

        // Create fence pointers at regular intervals
        for (i, block) in blocks.iter().enumerate() {
            let entry_offset = (i * BLOCK_INDEX_ENTRY_SIZE) as u64;

            // Add fence pointer at start and every FENCE_INTERVAL_BYTES
            if i == 0 || (entry_offset - current_offset) >= FENCE_INTERVAL_BYTES as u64 {
                fence_pointers.push(FencePointer {
                    key: block.min_key,
                    block_index_offset: entry_offset,
                });
                current_offset = entry_offset;
            }
        }

        let block_index_length = (blocks.len() * BLOCK_INDEX_ENTRY_SIZE) as u64;

        Self {
            fence_pointers,
            total_blocks: blocks.len() as u32,
            block_index_offset,
            block_index_length,
        }
    }

    /// Find the fence pointer range that may contain the target key
    ///
    /// Returns (start_offset, end_offset) within the block index section.
    pub fn find_fence_range(&self, key: &TemporalKey) -> (u64, u64) {
        if self.fence_pointers.is_empty() {
            return (0, self.block_index_length);
        }

        // Binary search for the fence pointer
        let idx = match self.fence_pointers.binary_search_by(|fp| fp.key.cmp(key)) {
            Ok(i) => i, // Exact match
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1 // Key is between fence_pointers[i-1] and fence_pointers[i]
                }
            }
        };

        let start_offset = self.fence_pointers[idx].block_index_offset;
        let end_offset = if idx + 1 < self.fence_pointers.len() {
            self.fence_pointers[idx + 1].block_index_offset
        } else {
            self.block_index_length
        };

        (start_offset, end_offset)
    }

    /// Find fence pointer range for a timestamp range query
    ///
    /// Returns (start_offset, end_offset) within the block index section.
    pub fn find_fence_range_for_timestamps(&self, start_ts: u64, end_ts: u64) -> (u64, u64) {
        if self.fence_pointers.is_empty() {
            return (0, self.block_index_length);
        }

        // Find first fence that could contain start_ts
        let start_key = TemporalKey::new(start_ts, 0);
        let start_idx = match self
            .fence_pointers
            .binary_search_by(|fp| fp.key.cmp(&start_key))
        {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
        };

        // Find last fence that could contain end_ts
        let end_key = TemporalKey::new(end_ts, u128::MAX);
        let end_idx = match self
            .fence_pointers
            .binary_search_by(|fp| fp.key.cmp(&end_key))
        {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        let start_offset = self.fence_pointers[start_idx].block_index_offset;
        let end_offset = if end_idx < self.fence_pointers.len() {
            self.fence_pointers[end_idx].block_index_offset
        } else {
            self.block_index_length
        };

        (start_offset, end_offset)
    }

    /// Serialize the fence pointers to bytes (stored in footer)
    pub fn fence_pointers_to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.fence_pointers.len() * FENCE_POINTER_SIZE + 8);

        // Write count
        buf.write_u32::<LittleEndian>(self.fence_pointers.len() as u32)
            .unwrap();
        buf.write_u32::<LittleEndian>(self.total_blocks).unwrap();

        // Write fence pointers
        for fp in &self.fence_pointers {
            buf.extend_from_slice(&fp.to_bytes());
        }

        buf
    }

    /// Deserialize fence pointers from bytes
    pub fn fence_pointers_from_bytes(
        bytes: &[u8],
        block_index_offset: u64,
        block_index_length: u64,
    ) -> std::io::Result<Self> {
        if bytes.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Fence pointer section too short",
            ));
        }

        let mut cursor = Cursor::new(bytes);
        let count = cursor.read_u32::<LittleEndian>()? as usize;
        let total_blocks = cursor.read_u32::<LittleEndian>()?;

        let expected_size = 8 + count * FENCE_POINTER_SIZE;
        if bytes.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Fence pointer section too short: {} < {}",
                    bytes.len(),
                    expected_size
                ),
            ));
        }

        let mut fence_pointers = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = [0u8; FENCE_POINTER_SIZE];
            cursor.read_exact(&mut buf)?;
            fence_pointers.push(FencePointer::from_bytes(&buf)?);
        }

        Ok(Self {
            fence_pointers,
            total_blocks,
            block_index_offset,
            block_index_length,
        })
    }

    /// Estimate memory usage of this index in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.fence_pointers.len() * std::mem::size_of::<FencePointer>()
    }

    /// Get the number of fence pointers
    pub fn fence_count(&self) -> usize {
        self.fence_pointers.len()
    }
}

/// Block index reader for loading block index entries on-demand
pub struct BlockIndexReader<'a> {
    /// Reference to the mmap'd block index section
    data: &'a [u8],
}

impl<'a> BlockIndexReader<'a> {
    /// Create a new block index reader from the block index section
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Read a single block index entry at the given offset
    pub fn read_entry(&self, offset: usize) -> std::io::Result<BlockIndexEntry> {
        if offset + BLOCK_INDEX_ENTRY_SIZE > self.data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Block index offset out of bounds",
            ));
        }

        BlockIndexEntry::from_bytes(&self.data[offset..offset + BLOCK_INDEX_ENTRY_SIZE])
    }

    /// Read a range of block index entries
    pub fn read_range(&self, start: usize, end: usize) -> std::io::Result<Vec<BlockIndexEntry>> {
        let start = start.min(self.data.len());
        let end = end.min(self.data.len());

        let mut entries = Vec::new();
        let mut offset = start;

        while offset + BLOCK_INDEX_ENTRY_SIZE <= end {
            entries.push(self.read_entry(offset)?);
            offset += BLOCK_INDEX_ENTRY_SIZE;
        }

        Ok(entries)
    }

    /// Binary search for a key in a range of block index entries
    pub fn find_block_for_key(
        &self,
        key: &TemporalKey,
        start_offset: usize,
        end_offset: usize,
    ) -> std::io::Result<Option<BlockIndexEntry>> {
        let entries = self.read_range(start_offset, end_offset)?;

        // Binary search for the block containing this key
        let idx = entries.partition_point(|e| e.max_key < *key);

        if idx < entries.len() && entries[idx].contains_key(key) {
            Ok(Some(entries[idx]))
        } else {
            Ok(None)
        }
    }

    /// Find all blocks that overlap with a timestamp range
    pub fn find_blocks_for_range(
        &self,
        start_ts: u64,
        end_ts: u64,
        start_offset: usize,
        end_offset: usize,
    ) -> std::io::Result<Vec<BlockIndexEntry>> {
        let entries = self.read_range(start_offset, end_offset)?;

        Ok(entries
            .into_iter()
            .filter(|e| e.overlaps_range(start_ts, end_ts))
            .collect())
    }

    /// Get total number of entries in this reader's range
    pub fn entry_count(&self) -> usize {
        self.data.len() / BLOCK_INDEX_ENTRY_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_blocks(count: usize) -> Vec<BlockIndexEntry> {
        (0..count)
            .map(|i| BlockIndexEntry {
                min_key: TemporalKey::new(i as u64 * 1000, i as u128),
                max_key: TemporalKey::new((i + 1) as u64 * 1000 - 1, i as u128),
                offset: (i * DATA_BLOCK_SIZE) as u64,
                length: DATA_BLOCK_SIZE as u32,
            })
            .collect()
    }

    #[test]
    fn test_temporal_key_ordering() {
        let k1 = TemporalKey::new(100, 1);
        let k2 = TemporalKey::new(100, 2);
        let k3 = TemporalKey::new(200, 1);

        assert!(k1 < k2); // Same timestamp, lower edge_id comes first
        assert!(k2 < k3); // Lower timestamp comes first
        assert!(k1 < k3);
    }

    #[test]
    fn test_block_index_entry_serialization() {
        let entry = BlockIndexEntry {
            min_key: TemporalKey::new(1000, 42),
            max_key: TemporalKey::new(2000, 100),
            offset: 65536,
            length: 64000,
        };

        let bytes = entry.to_bytes();
        let restored = BlockIndexEntry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.min_key, entry.min_key);
        assert_eq!(restored.max_key, entry.max_key);
        assert_eq!(restored.offset, entry.offset);
        assert_eq!(restored.length, entry.length);
    }

    #[test]
    fn test_fence_pointer_serialization() {
        let fp = FencePointer {
            key: TemporalKey::new(5000, 123),
            block_index_offset: 1024 * 1024,
        };

        let bytes = fp.to_bytes();
        let restored = FencePointer::from_bytes(&bytes).unwrap();

        assert_eq!(restored.key, fp.key);
        assert_eq!(restored.block_index_offset, fp.block_index_offset);
    }

    #[test]
    fn test_two_level_index_build() {
        let blocks = create_test_blocks(100);
        let index = TwoLevelIndex::build(&blocks, 0);

        // Should have at least one fence pointer (at start)
        assert!(!index.fence_pointers.is_empty());
        assert_eq!(index.total_blocks, 100);

        // First fence should point to start
        assert_eq!(index.fence_pointers[0].block_index_offset, 0);
    }

    #[test]
    fn test_two_level_index_fence_range() {
        let blocks = create_test_blocks(100);
        let index = TwoLevelIndex::build(&blocks, 0);

        // Key in first block
        let key = TemporalKey::new(500, 0);
        let (start, end) = index.find_fence_range(&key);
        assert_eq!(start, 0);
        assert!(end > start);

        // Key at end
        let key = TemporalKey::new(99000, 99);
        let (start, end) = index.find_fence_range(&key);
        assert!(start < end);
        assert_eq!(end, index.block_index_length);
    }

    #[test]
    fn test_two_level_index_serialization() {
        let blocks = create_test_blocks(50);
        let index = TwoLevelIndex::build(&blocks, 1024);

        let bytes = index.fence_pointers_to_bytes();
        let restored =
            TwoLevelIndex::fence_pointers_from_bytes(&bytes, 1024, index.block_index_length)
                .unwrap();

        assert_eq!(restored.fence_pointers.len(), index.fence_pointers.len());
        assert_eq!(restored.total_blocks, index.total_blocks);
    }

    #[test]
    fn test_block_index_reader() {
        let blocks = create_test_blocks(10);

        // Serialize blocks
        let mut data = Vec::new();
        for block in &blocks {
            data.extend_from_slice(&block.to_bytes());
        }

        let reader = BlockIndexReader::new(&data);

        // Read first entry
        let entry = reader.read_entry(0).unwrap();
        assert_eq!(entry.min_key, blocks[0].min_key);

        // Read range
        let range = reader.read_range(0, data.len()).unwrap();
        assert_eq!(range.len(), 10);
    }

    #[test]
    fn test_block_index_find_block_for_key() {
        let blocks = create_test_blocks(10);

        let mut data = Vec::new();
        for block in &blocks {
            data.extend_from_slice(&block.to_bytes());
        }

        let reader = BlockIndexReader::new(&data);

        // Find key in block 5
        let key = TemporalKey::new(5500, 5);
        let found = reader.find_block_for_key(&key, 0, data.len()).unwrap();

        assert!(found.is_some());
        let block = found.unwrap();
        assert!(block.contains_key(&key));
    }

    #[test]
    fn test_block_index_find_blocks_for_range() {
        let blocks = create_test_blocks(10);

        let mut data = Vec::new();
        for block in &blocks {
            data.extend_from_slice(&block.to_bytes());
        }

        let reader = BlockIndexReader::new(&data);

        // Find blocks overlapping timestamp range 2500-4500 (should include blocks 2,3,4)
        let found = reader
            .find_blocks_for_range(2500, 4500, 0, data.len())
            .unwrap();

        assert!(found.len() >= 2); // At least blocks containing these timestamps
    }

    #[test]
    fn test_memory_usage() {
        let blocks = create_test_blocks(1000);
        let index = TwoLevelIndex::build(&blocks, 0);

        let memory = index.memory_usage();

        // Should be much less than full block index
        let full_index_size = blocks.len() * BLOCK_INDEX_ENTRY_SIZE;
        assert!(memory < full_index_size / 10); // At least 10x reduction
    }

    #[test]
    fn test_block_contains_key() {
        let block = BlockIndexEntry {
            min_key: TemporalKey::new(1000, 0),
            max_key: TemporalKey::new(2000, 100),
            offset: 0,
            length: 64000,
        };

        assert!(block.contains_key(&TemporalKey::new(1500, 50)));
        assert!(block.contains_key(&TemporalKey::new(1000, 0))); // Min edge
        assert!(block.contains_key(&TemporalKey::new(2000, 100))); // Max edge
        assert!(!block.contains_key(&TemporalKey::new(999, 0))); // Below range
        assert!(!block.contains_key(&TemporalKey::new(2001, 0))); // Above range
    }

    #[test]
    fn test_block_overlaps_range() {
        let block = BlockIndexEntry {
            min_key: TemporalKey::new(1000, 0),
            max_key: TemporalKey::new(2000, 100),
            offset: 0,
            length: 64000,
        };

        assert!(block.overlaps_range(500, 1500)); // Overlaps start
        assert!(block.overlaps_range(1500, 2500)); // Overlaps end
        assert!(block.overlaps_range(1200, 1800)); // Fully inside
        assert!(block.overlaps_range(500, 2500)); // Fully outside
        assert!(!block.overlaps_range(100, 500)); // Before
        assert!(!block.overlaps_range(2500, 3000)); // After
    }
}
