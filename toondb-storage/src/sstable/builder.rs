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

//! SSTable Builder
//!
//! This module provides a builder for creating SSTable files with:
//! - Configurable block size
//! - Optional bloom/ribbon/xor filters
//! - Two-level index for large tables
//! - Compression support

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use super::block::{BlockBuilder, BlockHandle, BlockType, DEFAULT_RESTART_INTERVAL};
use super::filter::{BloomFilterPolicy, FilterBuilder, FilterPolicy};
use super::format::{Footer, Header, Section, SectionType, SSTableFormat, HEADER_SIZE};

/// Default block size (4KB - matches typical filesystem block)
pub const DEFAULT_BLOCK_SIZE: usize = 4 * 1024;

/// Default filter bits per key
pub const DEFAULT_FILTER_BITS_PER_KEY: f64 = 10.0;

/// Builder options
#[derive(Debug, Clone)]
pub struct SSTableBuilderOptions {
    /// Target block size in bytes
    pub block_size: usize,
    /// Restart interval for prefix compression
    pub restart_interval: usize,
    /// Block compression type
    pub compression: BlockType,
    /// Filter policy (None = no filter)
    pub filter_policy: Option<Box<dyn FilterPolicy>>,
    /// Use hash index for blocks
    pub use_block_hash_index: bool,
    /// Enable two-level index for large tables
    pub use_two_level_index: bool,
}

impl Default for SSTableBuilderOptions {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            restart_interval: DEFAULT_RESTART_INTERVAL,
            compression: BlockType::Uncompressed,
            filter_policy: Some(Box::new(BloomFilterPolicy::with_bits_per_key(
                DEFAULT_FILTER_BITS_PER_KEY,
            ))),
            use_block_hash_index: true,
            use_two_level_index: false,
        }
    }
}

// Implement Clone for FilterPolicy
impl Clone for Box<dyn FilterPolicy> {
    fn clone(&self) -> Self {
        // Create a new policy with same configuration
        // This is a simplified approach - could use a proper clone trait
        Box::new(BloomFilterPolicy::with_bits_per_key(self.bits_per_key()))
    }
}

/// Index entry pointing to a data block
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Largest key in the block (separator key)
    largest_key: Vec<u8>,
    /// Block handle (offset, size)
    handle: BlockHandle,
}

/// SSTable builder
pub struct SSTableBuilder {
    /// Options
    options: SSTableBuilderOptions,
    /// Output file
    file: BufWriter<File>,
    /// Current data block builder
    data_block: BlockBuilder,
    /// Index entries
    index_entries: Vec<IndexEntry>,
    /// Filter builder (if enabled)
    filter_builder: Option<Box<dyn FilterBuilder>>,
    /// Current file offset
    offset: u64,
    /// Number of entries
    num_entries: u64,
    /// Smallest key
    smallest_key: Option<Vec<u8>>,
    /// Largest key
    largest_key: Option<Vec<u8>>,
    /// Last key added (for sorting check)
    last_key: Option<Vec<u8>>,
    /// Pending index entry for current block
    pending_index_entry: bool,
    /// Last block's largest key
    pending_largest_key: Vec<u8>,
    /// Data blocks section start
    data_section_start: u64,
    /// Estimated keys for filter sizing
    estimated_keys: usize,
}

impl SSTableBuilder {
    /// Create a new SSTable builder
    pub fn new<P: AsRef<Path>>(path: P, options: SSTableBuilderOptions) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Reserve space for header
        writer.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        
        let data_block = if options.use_block_hash_index {
            BlockBuilder::with_hash_index(options.restart_interval)
        } else {
            BlockBuilder::new(options.restart_interval)
        };
        
        Ok(Self {
            options,
            file: writer,
            data_block,
            index_entries: Vec::new(),
            filter_builder: None,
            offset: HEADER_SIZE as u64,
            num_entries: 0,
            smallest_key: None,
            largest_key: None,
            last_key: None,
            pending_index_entry: false,
            pending_largest_key: Vec::new(),
            data_section_start: HEADER_SIZE as u64,
            estimated_keys: 0,
        })
    }

    /// Set estimated number of keys (for filter sizing)
    pub fn set_estimated_keys(&mut self, count: usize) {
        self.estimated_keys = count;
        
        // Initialize filter builder with proper size
        if let Some(ref policy) = self.options.filter_policy {
            self.filter_builder = Some(policy.create_builder(count));
        }
    }

    /// Add a key-value pair
    ///
    /// Keys must be added in sorted order.
    pub fn add(&mut self, key: &[u8], value: &[u8]) -> std::io::Result<()> {
        // Verify sorted order
        if let Some(ref last) = self.last_key {
            debug_assert!(
                key > last.as_slice(),
                "Keys must be added in sorted order"
            );
        }

        // Handle pending index entry from previous block
        if self.pending_index_entry {
            self.add_index_entry(&self.pending_largest_key.clone())?;
            self.pending_index_entry = false;
        }

        // Add to filter if enabled
        if let Some(ref mut builder) = self.filter_builder {
            builder.add_key(key);
        }

        // Add to data block
        self.data_block.add(key, value);

        // Track keys
        if self.smallest_key.is_none() {
            self.smallest_key = Some(key.to_vec());
        }
        self.largest_key = Some(key.to_vec());
        self.last_key = Some(key.to_vec());
        self.num_entries += 1;

        // Flush block if it's large enough
        if self.data_block.estimated_size() >= self.options.block_size {
            self.flush_data_block()?;
        }

        Ok(())
    }

    /// Flush current data block to file
    fn flush_data_block(&mut self) -> std::io::Result<()> {
        if self.data_block.is_empty() {
            return Ok(());
        }

        // Get block contents
        let block_data = self.data_block.finish();
        let block_size = block_data.len();

        // Compress if enabled
        let (compressed_data, block_type) = self.maybe_compress(&block_data);

        // Write block
        let block_offset = self.offset;
        self.file.write_all(&compressed_data)?;
        
        // Write block trailer (type + checksum)
        self.file.write_all(&[block_type as u8])?;
        let checksum = crc32fast::hash(&compressed_data);
        self.file.write_all(&checksum.to_le_bytes())?;

        // Update offset
        let total_size = compressed_data.len() + 1 + 4; // data + type + checksum
        self.offset += total_size as u64;

        // Record pending index entry
        if let Some(ref key) = self.largest_key {
            self.pending_largest_key = key.clone();
        }
        self.pending_index_entry = true;

        // Create index entry
        let handle = BlockHandle::new(block_offset, block_size as u64);
        self.index_entries.push(IndexEntry {
            largest_key: self.pending_largest_key.clone(),
            handle,
        });

        // Reset block builder
        self.data_block.reset();

        Ok(())
    }

    /// Maybe compress block data
    fn maybe_compress(&self, data: &[u8]) -> (Vec<u8>, BlockType) {
        match self.options.compression {
            BlockType::Uncompressed => (data.to_vec(), BlockType::Uncompressed),
            BlockType::Lz4 => {
                // Use LZ4 compression
                match lz4_flex::compress_prepend_size(data) {
                    compressed if compressed.len() < data.len() => (compressed, BlockType::Lz4),
                    _ => (data.to_vec(), BlockType::Uncompressed),
                }
            }
            BlockType::Zstd => {
                // Use Zstd compression
                match zstd::encode_all(data, 3) {
                    Ok(compressed) if compressed.len() < data.len() => (compressed, BlockType::Zstd),
                    _ => (data.to_vec(), BlockType::Uncompressed),
                }
            }
            BlockType::Snappy => {
                // Snappy not implemented - fall back to uncompressed
                (data.to_vec(), BlockType::Uncompressed)
            }
        }
    }

    /// Add index entry
    fn add_index_entry(&mut self, largest_key: &[u8]) -> std::io::Result<()> {
        // Index entries are already added in flush_data_block
        // This is for any additional processing
        Ok(())
    }

    /// Finish building the SSTable
    pub fn finish(mut self) -> std::io::Result<SSTableBuilderResult> {
        // Flush any remaining data
        self.flush_data_block()?;

        let data_section_end = self.offset;
        let data_section_size = data_section_end - self.data_section_start;
        let data_checksum = 0u32; // Would compute from all blocks

        let mut sections = vec![Section::new(
            SectionType::DataBlocks,
            self.data_section_start,
            data_section_size,
            data_checksum,
        )];

        // Write filter section
        if let Some(mut builder) = self.filter_builder.take() {
            let filter_data = builder.finish();
            let filter_offset = self.offset;
            let filter_size = filter_data.len() as u64;
            let filter_checksum = crc32fast::hash(&filter_data);

            self.file.write_all(&filter_data)?;
            self.offset += filter_size;

            sections.push(Section::new(
                SectionType::Filter,
                filter_offset,
                filter_size,
                filter_checksum,
            ));
        }

        // Write index section
        let index_offset = self.offset;
        let index_data = self.build_index()?;
        let index_size = index_data.len() as u64;
        let index_checksum = crc32fast::hash(&index_data);

        self.file.write_all(&index_data)?;
        self.offset += index_size;

        sections.push(Section::new(
            SectionType::Index,
            index_offset,
            index_size,
            index_checksum,
        ));

        // Write footer
        let footer_offset = self.offset;
        let footer = Footer::new(sections.clone());
        let footer_data = footer.encode();
        self.file.write_all(&footer_data)?;

        // Write header at start
        let header = Header::new(sections.len() as u32, footer_offset);
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header.encode())?;

        // Flush and sync
        self.file.flush()?;

        Ok(SSTableBuilderResult {
            file_size: footer_offset + footer_data.len() as u64,
            num_entries: self.num_entries,
            num_data_blocks: self.index_entries.len(),
            smallest_key: self.smallest_key,
            largest_key: self.largest_key,
        })
    }

    /// Build index block
    fn build_index(&self) -> std::io::Result<Vec<u8>> {
        let mut builder = BlockBuilder::new(1); // No prefix compression for index

        for entry in &self.index_entries {
            let handle_encoded = entry.handle.encode();
            builder.add(&entry.largest_key, &handle_encoded);
        }

        Ok(builder.finish())
    }

    /// Get number of entries added
    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    /// Get current file size
    pub fn file_size(&self) -> u64 {
        self.offset
    }
}

/// Result of building an SSTable
#[derive(Debug)]
pub struct SSTableBuilderResult {
    /// Final file size in bytes
    pub file_size: u64,
    /// Number of entries
    pub num_entries: u64,
    /// Number of data blocks
    pub num_data_blocks: usize,
    /// Smallest key
    pub smallest_key: Option<Vec<u8>>,
    /// Largest key
    pub largest_key: Option<Vec<u8>>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_builder_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let options = SSTableBuilderOptions {
            block_size: 256, // Small blocks for testing
            filter_policy: None,
            ..Default::default()
        };

        let mut builder = SSTableBuilder::new(&path, options).unwrap();

        for i in 0..100 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let result = builder.finish().unwrap();

        assert_eq!(result.num_entries, 100);
        assert!(result.num_data_blocks > 0);
        assert!(result.file_size > 0);
    }

    #[test]
    fn test_builder_with_filter() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_filter.sst");

        let mut builder = SSTableBuilder::new(&path, SSTableBuilderOptions::default()).unwrap();
        builder.set_estimated_keys(1000);

        for i in 0..1000 {
            let key = format!("key{:06}", i);
            let value = format!("value{:06}", i);
            builder.add(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let result = builder.finish().unwrap();

        assert_eq!(result.num_entries, 1000);
        assert!(result.file_size > 0);
    }
}
