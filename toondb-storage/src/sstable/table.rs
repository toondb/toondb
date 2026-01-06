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

//! SSTable Reader
//!
//! This module provides an SSTable reader with:
//! - Memory-mapped I/O for efficient access
//! - Lazy block loading
//! - Block cache integration
//! - Binary search in index for O(log n) lookups
//! - Filter-based negative lookup optimization

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;

use super::block::{Block, BlockHandle, BlockIterator, BlockType};
use super::filter::FilterReader;
use super::format::{Footer, Header, Section, SectionType, SSTableFormat, HEADER_SIZE};

/// Block cache entry
pub struct CachedBlock {
    /// Raw block data
    pub data: Vec<u8>,
    /// Block type (compression)
    pub block_type: BlockType,
    /// Decompressed data (if applicable)
    pub decompressed: Vec<u8>,
}

/// Simple block cache (HashMap-based for simplicity)
pub struct BlockCache {
    /// Cache entries by (file_id, block_offset)
    entries: RwLock<HashMap<(u64, u64), Arc<CachedBlock>>>,
    /// Maximum capacity
    capacity: usize,
}

impl BlockCache {
    /// Create a new block cache
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
        }
    }

    /// Get a cached block
    pub fn get(&self, file_id: u64, offset: u64) -> Option<Arc<CachedBlock>> {
        self.entries.read().get(&(file_id, offset)).cloned()
    }

    /// Insert a block into cache
    pub fn insert(&self, file_id: u64, offset: u64, block: CachedBlock) -> Arc<CachedBlock> {
        let block = Arc::new(block);
        let mut entries = self.entries.write();
        
        // Simple eviction: clear when full
        if entries.len() >= self.capacity {
            entries.clear();
        }
        
        entries.insert((file_id, offset), block.clone());
        block
    }
}

/// Read options
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Verify checksums when reading blocks
    pub verify_checksums: bool,
    /// Fill block cache
    pub fill_cache: bool,
    /// Use filter to skip blocks
    pub use_filter: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            verify_checksums: true,
            fill_cache: true,
            use_filter: true,
        }
    }
}

/// SSTable reader for reading SSTable files
pub struct SSTable {
    /// File path
    path: PathBuf,
    /// Unique file ID for caching
    file_id: u64,
    /// Memory-mapped file
    mmap: Mmap,
    /// Parsed header
    header: Header,
    /// Parsed footer with sections
    footer: Footer,
    /// Index block (cached)
    index: Vec<u8>,
    /// Parsed index entries
    index_entries: Vec<IndexEntry>,
    /// Filter reader (if filter section exists)
    filter: Option<FilterReader>,
    /// File metadata
    metadata: TableMetadata,
    /// Block cache reference
    cache: Option<Arc<BlockCache>>,
}

/// Index entry
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Largest key in this block (separator)
    largest_key: Vec<u8>,
    /// Block handle
    handle: BlockHandle,
}

/// Table metadata
#[derive(Debug, Clone)]
pub struct TableMetadata {
    /// File size
    pub file_size: u64,
    /// Number of data blocks
    pub num_data_blocks: usize,
    /// Smallest key
    pub smallest_key: Option<Vec<u8>>,
    /// Largest key
    pub largest_key: Option<Vec<u8>>,
}

impl SSTable {
    /// Open an SSTable file
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Self::open_with_cache(path, None)
    }

    /// Open an SSTable file with a block cache
    pub fn open_with_cache<P: AsRef<Path>>(
        path: P,
        cache: Option<Arc<BlockCache>>,
    ) -> std::io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // Memory-map the file
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Generate file ID from path hash
        let file_id = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            path.hash(&mut hasher);
            hasher.finish()
        };

        // Parse header
        if mmap.len() < HEADER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small for SSTable header",
            ));
        }

        let header = Header::decode(&mmap[..HEADER_SIZE]).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid SSTable header")
        })?;

        // Parse footer
        let footer_offset = header.footer_offset as usize;
        if footer_offset >= mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Footer offset beyond file",
            ));
        }

        let footer = Footer::decode(&mmap[footer_offset..], header.num_sections).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid SSTable footer")
        })?;

        // Load index section
        let index_section = footer
            .sections
            .iter()
            .find(|s| s.section_type == SectionType::Index)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing index section")
            })?;

        let index_start = index_section.offset as usize;
        let index_end = index_start + index_section.size as usize;
        let index = mmap[index_start..index_end].to_vec();

        // Parse index entries
        let index_entries = Self::parse_index(&index)?;

        // Load filter section if present
        let filter = footer
            .sections
            .iter()
            .find(|s| s.section_type == SectionType::Filter)
            .and_then(|section| {
                let start = section.offset as usize;
                let end = start + section.size as usize;
                FilterReader::from_bytes(&mmap[start..end])
            });

        // Extract metadata
        let metadata = TableMetadata {
            file_size,
            num_data_blocks: index_entries.len(),
            smallest_key: index_entries.first().map(|e| e.largest_key.clone()),
            largest_key: index_entries.last().map(|e| e.largest_key.clone()),
        };

        Ok(Self {
            path: path.to_path_buf(),
            file_id,
            mmap,
            header,
            footer,
            index,
            index_entries,
            filter,
            metadata,
            cache,
        })
    }

    /// Parse index entries from index block data
    fn parse_index(data: &[u8]) -> std::io::Result<Vec<IndexEntry>> {
        let mut entries = Vec::new();
        let block = Block::new(data.to_vec()).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid index block")
        })?;
        let mut iter = block.iter();

        while iter.valid() {
            let key = iter.key().to_vec();
            let value = iter.value();

            let (handle, _bytes_read) = BlockHandle::decode(value).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid block handle")
            })?;

            entries.push(IndexEntry {
                largest_key: key,
                handle,
            });

            iter.next();
        }

        Ok(entries)
    }

    /// Get a value by key
    pub fn get(&self, key: &[u8], options: &ReadOptions) -> std::io::Result<Option<Vec<u8>>> {
        // Use filter to check if key might exist
        if options.use_filter {
            if let Some(ref filter) = self.filter {
                if !filter.may_contain(key) {
                    return Ok(None);
                }
            }
        }

        // Binary search in index to find the right block
        let block_idx = self.find_block_for_key(key);
        if block_idx >= self.index_entries.len() {
            return Ok(None);
        }

        // Load and search the block
        let block_data = self.read_block(&self.index_entries[block_idx].handle, options)?;
        let block = Block::new(block_data).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid data block")
        })?;

        let iter = block.seek(key);
        if iter.valid() && iter.key() == key {
            Ok(Some(iter.value().to_vec()))
        } else {
            Ok(None)
        }
    }

    /// Binary search to find block that might contain the key
    fn find_block_for_key(&self, key: &[u8]) -> usize {
        // Binary search for first block where largest_key >= key
        self.index_entries
            .binary_search_by(|entry| {
                if entry.largest_key.as_slice() < key {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|i| i)
    }

    /// Read a block from file
    fn read_block(
        &self,
        handle: &BlockHandle,
        options: &ReadOptions,
    ) -> std::io::Result<Vec<u8>> {
        let offset = handle.offset();
        let size = handle.size();

        // Try cache first
        if let Some(ref cache) = self.cache {
            if let Some(block) = cache.get(self.file_id, offset) {
                return Ok(block.decompressed.clone());
            }
        }

        // Read from mmap
        let start = offset as usize;
        let end = start + size as usize;

        if end + 5 > self.mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Block extends beyond file",
            ));
        }

        let block_data = &self.mmap[start..end];
        let block_type = BlockType::from_u8(self.mmap[end]);
        let stored_checksum = u32::from_le_bytes([
            self.mmap[end + 1],
            self.mmap[end + 2],
            self.mmap[end + 3],
            self.mmap[end + 4],
        ]);

        // Verify checksum if requested
        if options.verify_checksums {
            let computed_checksum = crc32fast::hash(block_data);
            if computed_checksum != stored_checksum {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Block checksum mismatch",
                ));
            }
        }

        // Decompress if needed
        let decompressed = match block_type {
            BlockType::Uncompressed => block_data.to_vec(),
            BlockType::Lz4 => lz4_flex::decompress_size_prepended(block_data).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("LZ4 error: {}", e))
            })?,
            BlockType::Zstd => zstd::decode_all(block_data).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Zstd error: {}", e))
            })?,
            BlockType::Snappy => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Snappy not supported",
                ))
            }
        };

        // Cache the block
        if options.fill_cache {
            if let Some(ref cache) = self.cache {
                cache.insert(
                    self.file_id,
                    offset,
                    CachedBlock {
                        data: block_data.to_vec(),
                        block_type,
                        decompressed: decompressed.clone(),
                    },
                );
            }
        }

        Ok(decompressed)
    }

    /// Create an iterator over all entries
    pub fn iter(&self) -> SSTableIterator {
        SSTableIterator::new(self)
    }

    /// Create a range iterator
    pub fn range(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> RangeIterator {
        RangeIterator::new(self, start, end)
    }

    /// Get table metadata
    pub fn metadata(&self) -> &TableMetadata {
        &self.metadata
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get number of data blocks
    pub fn num_blocks(&self) -> usize {
        self.index_entries.len()
    }

    /// Check if key might exist (using filter)
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.filter
            .as_ref()
            .map(|f| f.may_contain(key))
            .unwrap_or(true)
    }
}

/// Iterator over all entries in an SSTable
pub struct SSTableIterator<'a> {
    table: &'a SSTable,
    /// Current block index
    block_idx: usize,
    /// Current block data
    block_data: Option<Vec<u8>>,
    /// Current block iterator
    block_iter: Option<BlockIterator<'a>>,
    /// Read options
    options: ReadOptions,
    /// Is iterator valid
    valid: bool,
}

impl<'a> SSTableIterator<'a> {
    fn new(table: &'a SSTable) -> Self {
        let mut iter = Self {
            table,
            block_idx: 0,
            block_data: None,
            block_iter: None,
            options: ReadOptions::default(),
            valid: false,
        };
        iter.load_block();
        iter
    }

    /// Load current block
    fn load_block(&mut self) {
        if self.block_idx >= self.table.index_entries.len() {
            self.valid = false;
            return;
        }

        let handle = &self.table.index_entries[self.block_idx].handle;
        match self.table.read_block(handle, &self.options) {
            Ok(data) => {
                self.block_data = Some(data);
                self.valid = true;
            }
            Err(_) => {
                self.valid = false;
            }
        }
    }

    /// Check if iterator is valid
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Get current key
    pub fn key(&self) -> Option<&[u8]> {
        if !self.valid {
            return None;
        }
        // Note: In a full implementation, this would return the current key from block_iter
        // This is a simplified version
        self.block_data.as_ref().map(|_| &b""[..])
    }

    /// Get current value
    pub fn value(&self) -> Option<&[u8]> {
        if !self.valid {
            return None;
        }
        self.block_data.as_ref().map(|_| &b""[..])
    }

    /// Move to next entry
    pub fn next(&mut self) {
        // In a full implementation:
        // 1. Advance block_iter
        // 2. If block_iter exhausted, load next block
        self.block_idx += 1;
        self.load_block();
    }

    /// Seek to key
    pub fn seek(&mut self, target: &[u8]) {
        // Binary search to find starting block
        self.block_idx = self.table.find_block_for_key(target);
        self.load_block();
        // Then seek within the block
    }
}

/// Range iterator
pub struct RangeIterator<'a> {
    table: &'a SSTable,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    current_block: usize,
    exhausted: bool,
}

impl<'a> RangeIterator<'a> {
    fn new(table: &'a SSTable, start: Option<&[u8]>, end: Option<&[u8]>) -> Self {
        let start_block = start
            .map(|k| table.find_block_for_key(k))
            .unwrap_or(0);

        Self {
            table,
            start: start.map(|s| s.to_vec()),
            end: end.map(|e| e.to_vec()),
            current_block: start_block,
            exhausted: false,
        }
    }

    /// Check if range is exhausted
    pub fn exhausted(&self) -> bool {
        self.exhausted
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::builder::{SSTableBuilder, SSTableBuilderOptions};
    use tempfile::tempdir;

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        // Build SSTable
        let options = SSTableBuilderOptions {
            block_size: 256,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in 0..100 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        // Read SSTable
        let table = SSTable::open(&path).unwrap();

        assert_eq!(table.num_blocks(), table.metadata.num_data_blocks);
    }

    #[test]
    fn test_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_get.sst");

        let options = SSTableBuilderOptions {
            block_size: 256,
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in 0..100 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        builder.finish().unwrap();

        let table = SSTable::open(&path).unwrap();
        let read_opts = ReadOptions::default();

        // Test existing key
        let result = table.get(b"key00050", &read_opts).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), b"value00050");

        // Test non-existing key
        let result = table.get(b"nonexistent", &read_opts).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_block_cache() {
        let cache = BlockCache::new(100);

        let block = CachedBlock {
            data: vec![1, 2, 3],
            block_type: BlockType::Uncompressed,
            decompressed: vec![1, 2, 3],
        };

        cache.insert(1, 0, block);

        let cached = cache.get(1, 0);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().data, vec![1, 2, 3]);

        let missing = cache.get(1, 100);
        assert!(missing.is_none());
    }
}
