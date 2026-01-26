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

//! Sharded Block Store (Task 3/7)
//!
//! Partitions block storage by hash(file_id) for parallel I/O:
//! - Each segment has independent file and index
//! - Lock-free offset allocation via atomic counters
//! - Per-segment reference counting for GC
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   ShardedBlockStore                         │
//! │  global_offset: AtomicU64                                   │
//! │  shard_count: 8 (default, based on CPU cores)              │
//! └──────────────────────┬──────────────────────────────────────┘
//!                        │
//!        ┌───────────────┼───────────────┐
//!        ▼               ▼               ▼
//! ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
//! │  Shard 0    │ │  Shard 1    │ │  Shard N    │
//! │  file_0.blk │ │  file_1.blk │ │  file_N.blk │
//! │  index: Map │ │  index: Map │ │  index: Map │
//! │  refs: Map  │ │  refs: Map  │ │  refs: Map  │
//! └─────────────┘ └─────────────┘ └─────────────┘
//! ```
//!
//! ## Sharding Algorithm
//!
//! Segment assignment: `shard_id = hash(file_id) % shard_count`
//!
//! For reads: `shard_id = (block_offset / segment_size) % shard_count`
//!
//! ## Throughput Model (Amdahl's Law)
//!
//! Speedup(S) = 1 / ((1-p) + p/S)
//!
//! With p ≈ 0.95 (fraction parallelizable), S=16 shards → ~10× speedup

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::block_storage::{
    BlockCompression, BlockHeader, BlockRef, is_compressible, is_json_content, is_soch_content,
};
use crate::{Result, SochDBError};

/// Default number of shards (based on typical CPU core count)
const DEFAULT_SHARD_COUNT: usize = 8;

/// Default segment size (64MB per shard)
const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

/// Individual shard with its own storage and index
pub struct BlockShard {
    /// Shard ID
    id: usize,
    /// Data storage (append-only)
    data: RwLock<Vec<u8>>,
    /// Next write offset within this shard
    next_offset: AtomicU64,
    /// Block index: local_offset -> BlockRef
    index: RwLock<HashMap<u64, BlockRef>>,
    /// Reference counts for GC
    ref_counts: RwLock<HashMap<u64, AtomicU32>>,
    /// Bytes written (for stats)
    bytes_written: AtomicU64,
    /// Blocks written (for stats)
    blocks_written: AtomicU64,
}

impl BlockShard {
    /// Create a new shard
    pub fn new(id: usize) -> Self {
        Self {
            id,
            data: RwLock::new(Vec::new()),
            next_offset: AtomicU64::new(0),
            index: RwLock::new(HashMap::new()),
            ref_counts: RwLock::new(HashMap::new()),
            bytes_written: AtomicU64::new(0),
            blocks_written: AtomicU64::new(0),
        }
    }

    /// Write a block to this shard
    pub fn write_block(&self, data: &[u8], compression: BlockCompression) -> Result<BlockRef> {
        // Compress data
        let compressed = self.compress(data, compression)?;

        // Calculate checksum
        let checksum = crc32fast::hash(&compressed);

        // Allocate offset atomically
        let header_size = BlockHeader::SIZE;
        let total_size = header_size + compressed.len();
        let local_offset = self
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

        // Write to shard storage
        {
            let mut store = self.data.write();
            let required_size = (local_offset + total_size as u64) as usize;
            if store.len() < required_size {
                store.resize(required_size, 0);
            }

            // Write header
            let header_bytes = header.to_bytes();
            store[local_offset as usize..local_offset as usize + header_size]
                .copy_from_slice(&header_bytes);

            // Write data
            store[local_offset as usize + header_size..local_offset as usize + total_size]
                .copy_from_slice(&compressed);
        }

        // Create block reference with shard-local offset
        let block_ref = BlockRef {
            store_offset: local_offset,
            compressed_len: compressed.len() as u32,
            original_len: data.len() as u32,
            compression,
            checksum,
        };

        // Update index
        self.index.write().insert(local_offset, block_ref.clone());

        // Initialize ref count
        self.ref_counts
            .write()
            .insert(local_offset, AtomicU32::new(1));

        // Update stats
        self.bytes_written
            .fetch_add(total_size as u64, Ordering::Relaxed);
        self.blocks_written.fetch_add(1, Ordering::Relaxed);

        Ok(block_ref)
    }

    /// Read a block from this shard
    pub fn read_block(&self, block_ref: &BlockRef) -> Result<Vec<u8>> {
        let offset = block_ref.store_offset as usize;
        let header_size = BlockHeader::SIZE;
        let total_size = header_size + block_ref.compressed_len as usize;

        // Read from shard storage
        let compressed = {
            let store = self.data.read();
            if offset + total_size > store.len() {
                return Err(SochDBError::Corruption(format!(
                    "Block at offset {} extends beyond shard {} data (size {})",
                    offset,
                    self.id,
                    store.len()
                )));
            }

            // Verify header
            let header = BlockHeader::from_bytes(&store[offset..offset + header_size])?;

            // Verify checksum matches
            if header.checksum != block_ref.checksum {
                return Err(SochDBError::Corruption(format!(
                    "Checksum mismatch in shard {}: expected {}, got {}",
                    self.id, block_ref.checksum, header.checksum
                )));
            }

            store[offset + header_size..offset + total_size].to_vec()
        };

        // Verify data checksum
        let computed_checksum = crc32fast::hash(&compressed);
        if computed_checksum != block_ref.checksum {
            return Err(SochDBError::Corruption(format!(
                "Data checksum mismatch in shard {}: expected {}, got {}",
                self.id, block_ref.checksum, computed_checksum
            )));
        }

        // Decompress
        self.decompress(
            &compressed,
            block_ref.compression,
            block_ref.original_len as usize,
        )
    }

    /// Increment reference count
    pub fn add_ref(&self, offset: u64) {
        let refs = self.ref_counts.read();
        if let Some(count) = refs.get(&offset) {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement reference count, returns true if block can be reclaimed
    pub fn release_ref(&self, offset: u64) -> bool {
        let refs = self.ref_counts.read();
        if let Some(count) = refs.get(&offset) {
            let prev = count.fetch_sub(1, Ordering::Relaxed);
            return prev == 1; // Was 1, now 0
        }
        false
    }

    /// Get shard statistics
    pub fn stats(&self) -> ShardStats {
        let index = self.index.read();
        let mut total_original = 0u64;
        let mut total_compressed = 0u64;

        for block_ref in index.values() {
            total_original += block_ref.original_len as u64;
            total_compressed += block_ref.compressed_len as u64;
        }

        ShardStats {
            shard_id: self.id,
            block_count: index.len(),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            total_original_bytes: total_original,
            total_compressed_bytes: total_compressed,
        }
    }

    /// Compress data with fallback
    fn compress(&self, data: &[u8], compression: BlockCompression) -> Result<Vec<u8>> {
        match compression {
            BlockCompression::None => Ok(data.to_vec()),
            BlockCompression::Lz4 => match lz4::block::compress(data, None, false) {
                Ok(compressed) if compressed.len() < data.len() => Ok(compressed),
                _ => Ok(data.to_vec()),
            },
            BlockCompression::Zstd => match zstd::encode_all(data, 3) {
                Ok(compressed) if compressed.len() < data.len() => Ok(compressed),
                _ => Ok(data.to_vec()),
            },
        }
    }

    /// Decompress data
    fn decompress(
        &self,
        data: &[u8],
        compression: BlockCompression,
        original_size: usize,
    ) -> Result<Vec<u8>> {
        match compression {
            BlockCompression::None => Ok(data.to_vec()),
            BlockCompression::Lz4 => {
                if data.len() == original_size {
                    return Ok(data.to_vec());
                }
                lz4::block::decompress(data, Some(original_size as i32))
                    .map_err(|e| SochDBError::Corruption(format!("LZ4 decompress failed: {}", e)))
            }
            BlockCompression::Zstd => {
                if data.len() == original_size {
                    return Ok(data.to_vec());
                }
                zstd::decode_all(data)
                    .map_err(|e| SochDBError::Corruption(format!("ZSTD decompress failed: {}", e)))
            }
        }
    }
}

/// Statistics for a single shard
#[derive(Debug, Clone)]
pub struct ShardStats {
    pub shard_id: usize,
    pub block_count: usize,
    pub bytes_written: u64,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
}

/// Sharded block store for parallel I/O
pub struct ShardedBlockStore {
    /// Individual shards
    shards: Vec<BlockShard>,
    /// Number of shards
    shard_count: usize,
    /// Segment size for offset-based shard lookup
    #[allow(dead_code)]
    segment_size: u64,
    /// Global write counter (for stats)
    total_writes: AtomicU64,
}

impl ShardedBlockStore {
    /// Create a new sharded block store with default settings
    pub fn new() -> Self {
        Self::with_shards(DEFAULT_SHARD_COUNT)
    }

    /// Create with specific number of shards
    pub fn with_shards(shard_count: usize) -> Self {
        let shards = (0..shard_count).map(BlockShard::new).collect();

        Self {
            shards,
            shard_count,
            segment_size: DEFAULT_SEGMENT_SIZE,
            total_writes: AtomicU64::new(0),
        }
    }

    /// Get shard for a file ID (for writes)
    #[inline]
    fn shard_for_file(&self, file_id: u64) -> usize {
        // Use FxHash-style mixing for fast hashing
        let mut h = file_id;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        (h as usize) % self.shard_count
    }

    /// Get shard for an offset (for reads)
    #[inline]
    #[allow(dead_code)]
    fn shard_for_offset(&self, offset: u64) -> usize {
        ((offset / self.segment_size) as usize) % self.shard_count
    }

    /// Write a block for a specific file
    pub fn write_block(&self, file_id: u64, data: &[u8]) -> Result<ShardedBlockRef> {
        let shard_id = self.shard_for_file(file_id);
        let compression = self.select_compression(data);

        let block_ref = self.shards[shard_id].write_block(data, compression)?;

        self.total_writes.fetch_add(1, Ordering::Relaxed);

        Ok(ShardedBlockRef {
            shard_id,
            block_ref,
        })
    }

    /// Write a block with specific compression
    pub fn write_block_with_compression(
        &self,
        file_id: u64,
        data: &[u8],
        compression: BlockCompression,
    ) -> Result<ShardedBlockRef> {
        let shard_id = self.shard_for_file(file_id);
        let block_ref = self.shards[shard_id].write_block(data, compression)?;

        self.total_writes.fetch_add(1, Ordering::Relaxed);

        Ok(ShardedBlockRef {
            shard_id,
            block_ref,
        })
    }

    /// Read a block
    pub fn read_block(&self, shard_ref: &ShardedBlockRef) -> Result<Vec<u8>> {
        if shard_ref.shard_id >= self.shard_count {
            return Err(SochDBError::Corruption(format!(
                "Invalid shard ID: {} (max {})",
                shard_ref.shard_id,
                self.shard_count - 1
            )));
        }
        self.shards[shard_ref.shard_id].read_block(&shard_ref.block_ref)
    }

    /// Increment reference count
    pub fn add_ref(&self, shard_ref: &ShardedBlockRef) {
        if shard_ref.shard_id < self.shard_count {
            self.shards[shard_ref.shard_id].add_ref(shard_ref.block_ref.store_offset);
        }
    }

    /// Decrement reference count
    pub fn release_ref(&self, shard_ref: &ShardedBlockRef) -> bool {
        if shard_ref.shard_id < self.shard_count {
            self.shards[shard_ref.shard_id].release_ref(shard_ref.block_ref.store_offset)
        } else {
            false
        }
    }

    /// Get aggregate statistics
    pub fn stats(&self) -> ShardedBlockStoreStats {
        let shard_stats: Vec<ShardStats> = self.shards.iter().map(|s| s.stats()).collect();

        let total_blocks: usize = shard_stats.iter().map(|s| s.block_count).sum();
        let total_bytes: u64 = shard_stats.iter().map(|s| s.bytes_written).sum();
        let total_original: u64 = shard_stats.iter().map(|s| s.total_original_bytes).sum();
        let total_compressed: u64 = shard_stats.iter().map(|s| s.total_compressed_bytes).sum();

        ShardedBlockStoreStats {
            shard_count: self.shard_count,
            total_blocks,
            total_bytes_written: total_bytes,
            total_original_bytes: total_original,
            total_compressed_bytes: total_compressed,
            compression_ratio: if total_compressed > 0 {
                total_original as f64 / total_compressed as f64
            } else {
                1.0
            },
            shard_stats,
        }
    }

    /// Select compression based on data content
    fn select_compression(&self, data: &[u8]) -> BlockCompression {
        if data.len() < 128 {
            return BlockCompression::None;
        }

        if is_soch_content(data) {
            BlockCompression::Zstd
        } else if is_json_content(data) || is_compressible(data) {
            BlockCompression::Lz4
        } else {
            BlockCompression::None
        }
    }
}

impl Default for ShardedBlockStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference to a block in a specific shard
#[derive(Debug, Clone)]
pub struct ShardedBlockRef {
    /// Which shard contains this block
    pub shard_id: usize,
    /// The block reference within that shard
    pub block_ref: BlockRef,
}

impl ShardedBlockRef {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + 21); // shard_id (u32) + BlockRef
        buf.extend(&(self.shard_id as u32).to_le_bytes());
        buf.extend(&self.block_ref.to_bytes().unwrap_or([0u8; 21]));
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 25 {
            return Err(SochDBError::Corruption("ShardedBlockRef too short".into()));
        }
        let shard_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let block_ref = BlockRef::from_bytes(&data[4..])?;
        Ok(Self {
            shard_id,
            block_ref,
        })
    }
}

/// Statistics for the sharded block store
#[derive(Debug, Clone)]
pub struct ShardedBlockStoreStats {
    pub shard_count: usize,
    pub total_blocks: usize,
    pub total_bytes_written: u64,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub compression_ratio: f64,
    pub shard_stats: Vec<ShardStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharded_store_basic() {
        let store = ShardedBlockStore::new();

        let data = b"Hello, sharded world!";
        let shard_ref = store.write_block(1, data).unwrap();

        let recovered = store.read_block(&shard_ref).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_sharded_store_multiple_files() {
        let store = ShardedBlockStore::new();

        // Write blocks for different files
        let mut refs = Vec::new();
        for file_id in 0..100u64 {
            let data = format!("Data for file {}", file_id).into_bytes();
            let shard_ref = store.write_block(file_id, &data).unwrap();
            refs.push((file_id, shard_ref, data));
        }

        // Verify all blocks can be read
        for (file_id, shard_ref, expected) in refs {
            let recovered = store.read_block(&shard_ref).unwrap();
            assert_eq!(recovered, expected, "File {} mismatch", file_id);
        }
    }

    #[test]
    fn test_sharded_store_distribution() {
        let store = ShardedBlockStore::with_shards(4);

        // Write many blocks and check distribution
        for i in 0..1000u64 {
            let data = vec![i as u8; 64];
            store.write_block(i, &data).unwrap();
        }

        let stats = store.stats();

        // Check that blocks are distributed across shards
        for shard_stat in &stats.shard_stats {
            // Each shard should have some blocks (probabilistic)
            assert!(
                shard_stat.block_count > 0,
                "Shard {} has no blocks",
                shard_stat.shard_id
            );
        }

        // Total should be 1000
        assert_eq!(stats.total_blocks, 1000);
    }

    #[test]
    fn test_sharded_ref_serialization() {
        let shard_ref = ShardedBlockRef {
            shard_id: 3,
            block_ref: BlockRef {
                store_offset: 12345,
                compressed_len: 100,
                original_len: 200,
                compression: BlockCompression::Lz4,
                checksum: 0xDEADBEEF,
            },
        };

        let bytes = shard_ref.to_bytes();
        let recovered = ShardedBlockRef::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.shard_id, 3);
        assert_eq!(recovered.block_ref.store_offset, 12345);
        assert_eq!(recovered.block_ref.compression, BlockCompression::Lz4);
    }

    #[test]
    fn test_sharded_store_compression() {
        let store = ShardedBlockStore::new();

        // Compressible data
        let data = vec![0u8; 4096];
        let shard_ref = store.write_block(1, &data).unwrap();

        // Verify compression happened
        assert!(shard_ref.block_ref.compressed_len < shard_ref.block_ref.original_len);

        // Verify data is correct
        let recovered = store.read_block(&shard_ref).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_ref_counting() {
        let store = ShardedBlockStore::new();

        let data = b"Reference counted block";
        let shard_ref = store.write_block(1, data).unwrap();

        // Add references
        store.add_ref(&shard_ref);
        store.add_ref(&shard_ref);

        // Release references
        assert!(!store.release_ref(&shard_ref)); // ref count: 2
        assert!(!store.release_ref(&shard_ref)); // ref count: 1
        assert!(store.release_ref(&shard_ref)); // ref count: 0, can reclaim
    }
}
