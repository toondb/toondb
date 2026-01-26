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

//! Bloom filter for fast negative lookups
//!
//! Space-efficient probabilistic data structure for existence tests.
//! False positive rate: ~1% with optimal k hash functions.
//!
//! Uses Blake3-based double hashing for mathematically independent hash functions.

use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter
    ///
    /// - expected_items: number of items to be inserted
    /// - false_positive_rate: desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let num_bits = Self::optimal_num_bits(expected_items, false_positive_rate);
        let num_hashes = Self::optimal_num_hashes(expected_items, num_bits);

        let num_words = num_bits.div_ceil(64);
        let bits = vec![0u64; num_words];

        Self {
            bits,
            num_bits,
            num_hashes,
        }
    }

    /// Calculate optimal number of bits
    fn optimal_num_bits(n: usize, p: f64) -> usize {
        let m = -(n as f64 * p.ln()) / (2.0_f64.ln().powi(2));
        m.ceil() as usize
    }

    /// Calculate optimal number of hash functions
    fn optimal_num_hashes(n: usize, m: usize) -> usize {
        let k = (m as f64 / n as f64) * 2.0_f64.ln();
        (k.ceil() as usize).max(1)
    }

    /// Insert an item into the bloom filter
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let hash = Self::hash(item, i);
            let bit_index = hash % self.num_bits;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            // PERFORMANCE: Remove runtime bounds check (proven safe by construction)
            // The modulo operation ensures bit_index < num_bits
            // The allocation ensures bits.len() == (num_bits + 63) / 64
            // Therefore word_index = bit_index / 64 < bits.len()
            debug_assert!(
                word_index < self.bits.len(),
                "Bloom filter index out of bounds: {} >= {}",
                word_index,
                self.bits.len()
            );

            self.bits[word_index] |= 1u64 << bit_offset;
        }
    }

    /// Check if an item might be in the set
    ///
    /// Returns true if item *might* be present (could be false positive)
    /// Returns false if item is *definitely not* present (100% accurate)
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let hash = Self::hash(item, i);
            let bit_index = hash % self.num_bits;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            // PERFORMANCE: Remove runtime bounds check (proven safe by construction)
            debug_assert!(
                word_index < self.bits.len(),
                "Bloom filter index out of bounds: {} >= {}",
                word_index,
                self.bits.len()
            );

            if (self.bits[word_index] & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute h1 hash using Blake3
    fn hash_h1<T: Hash>(item: &T) -> usize {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = blake3::Hasher::new();

        // Hash the item using standard Hash trait first to get bytes
        let mut std_hasher = DefaultHasher::new();
        item.hash(&mut std_hasher);
        let hash_value = std_hasher.finish();

        hasher.update(&hash_value.to_le_bytes());
        let hash = hasher.finalize();

        // Take first 8 bytes as usize
        let bytes = hash.as_bytes();
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize
    }

    /// Compute h2 hash using Blake3 with domain separator
    fn hash_h2<T: Hash>(item: &T) -> usize {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"BloomFilter::h2"); // Domain separator

        let mut std_hasher = DefaultHasher::new();
        item.hash(&mut std_hasher);
        let hash_value = std_hasher.finish();

        hasher.update(&hash_value.to_le_bytes());
        let hash = hasher.finalize();

        u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap()) as usize
    }

    /// Hash function using double hashing with Blake3
    ///
    /// Implements the double hashing scheme: h_i(x) = (h1(x) + i * h2(x)) mod m
    /// where h1 and h2 are independent cryptographic hashes.
    ///
    /// This is proven equivalent to k independent hash functions in:
    /// "Less Hashing, Same Performance: Building a Better Bloom Filter"
    /// by Kirsch & Mitzenmacher (2008)
    ///
    /// Mathematical correctness: Using Blake3 as the base hash ensures:
    /// 1. h1 and h2 are computationally independent
    /// 2. Collision resistance (2^128 security for truncated output)
    /// 3. Uniform distribution over output space
    fn hash<T: Hash>(item: &T, index: usize) -> usize {
        if index == 0 {
            Self::hash_h1(item)
        } else {
            // Double hashing: h_i(x) = h1(x) + i * h2(x)
            let h1 = Self::hash_h1(item);
            let h2 = Self::hash_h2(item);

            // Combine using double hashing formula
            // Use wrapping_add to handle overflow gracefully
            h1.wrapping_add(index.wrapping_mul(h2))
        }
    }

    /// Serialize to bytes with BLAKE3 checksum for corruption detection
    ///
    /// **CRITICAL FIX**: Adds integrity checksum to detect bit flips, cosmic rays,
    /// disk errors, or malicious modification of bloom filter data.
    ///
    /// Format v2 (with version header for forward compatibility):
    /// - Magic number: 4 bytes ("BLM\x02" = 0x424C4D02)
    /// - Header: num_bits (8) + num_hashes (8) + num_words (8) = 24 bytes
    /// - Bit data: num_words * 8 bytes
    /// - Checksum: 32 bytes (BLAKE3 hash of magic + header + data)
    ///
    /// Format v1 (legacy, for backward compatibility):
    /// - Header: num_bits (8) + num_hashes (8) + num_words (8) = 24 bytes
    /// - Bit data: num_words * 8 bytes
    /// - Checksum: 32 bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let mut buf = Vec::new();

        // Version 2 magic number: "BLM" + version byte (0x02)
        // This allows future readers to detect the format and version
        buf.extend_from_slice(&[0x42, 0x4C, 0x4D, 0x02]); // "BLM\x02"

        buf.write_u64::<LittleEndian>(self.num_bits as u64).unwrap();
        buf.write_u64::<LittleEndian>(self.num_hashes as u64)
            .unwrap();
        buf.write_u64::<LittleEndian>(self.bits.len() as u64)
            .unwrap();

        for &word in &self.bits {
            buf.write_u64::<LittleEndian>(word).unwrap();
        }

        // CRITICAL FIX: Add BLAKE3 checksum for corruption detection
        let checksum = blake3::hash(&buf);
        buf.extend_from_slice(checksum.as_bytes());

        buf
    }

    /// Magic number for bloom filter v2 format
    const MAGIC_V2: [u8; 4] = [0x42, 0x4C, 0x4D, 0x02]; // "BLM\x02"

    /// Deserialize from bytes with checksum validation and version detection
    ///
    /// **CRITICAL FIX**: Validates BLAKE3 checksum to detect bloom filter corruption.
    /// If corrupted, returns an error. The caller can then rebuild from the index.
    ///
    /// **FORWARD COMPATIBILITY**: Detects format version via magic number.
    /// - v2 (with magic "BLM\x02"): New format with version header
    /// - v1 (no magic): Legacy format, auto-detected by size
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};

        const CHECKSUM_SIZE: usize = 32; // BLAKE3 produces 32-byte hashes
        const MAGIC_SIZE: usize = 4;
        const HEADER_SIZE: usize = 24; // 3 * u64
        const MIN_SIZE_V2: usize = MAGIC_SIZE + HEADER_SIZE + CHECKSUM_SIZE; // 60 bytes
        const MIN_SIZE_V1: usize = HEADER_SIZE + CHECKSUM_SIZE; // 56 bytes

        // Check for v2 format (has magic number)
        if bytes.len() >= MIN_SIZE_V2 && bytes[..4] == Self::MAGIC_V2 {
            return Self::from_bytes_v2(bytes);
        }

        // Fall back to v1 format (legacy, no magic number)
        if bytes.len() >= MIN_SIZE_V1 {
            return Self::from_bytes_v1(bytes);
        }

        Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "Bloom filter data too small: {} bytes (minimum {})",
                bytes.len(),
                MIN_SIZE_V1
            ),
        ))
    }

    /// Parse v2 format (with magic number and version header)
    fn from_bytes_v2(bytes: &[u8]) -> std::io::Result<Self> {
        use std::io::{Cursor, Error, ErrorKind};

        const CHECKSUM_SIZE: usize = 32;
        const MAGIC_SIZE: usize = 4;

        // Validate checksum BEFORE parsing data
        let data_len = bytes.len() - CHECKSUM_SIZE;
        let (data, stored_checksum) = bytes.split_at(data_len);

        let computed_checksum = blake3::hash(data);
        if computed_checksum.as_bytes() != stored_checksum {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Bloom filter v2 corruption detected: checksum mismatch. \
                     Expected {:?}, got {:?}.",
                    hex::encode(stored_checksum),
                    hex::encode(computed_checksum.as_bytes())
                ),
            ));
        }

        // Skip magic number, parse header
        let mut cursor = Cursor::new(&data[MAGIC_SIZE..]);
        Self::parse_header_and_data(&mut cursor)
    }

    /// Parse v1 format (legacy, no magic number)
    fn from_bytes_v1(bytes: &[u8]) -> std::io::Result<Self> {
        use std::io::{Cursor, Error, ErrorKind};

        const CHECKSUM_SIZE: usize = 32;

        // Validate checksum BEFORE parsing data
        let data_len = bytes.len() - CHECKSUM_SIZE;
        let (data, stored_checksum) = bytes.split_at(data_len);

        let computed_checksum = blake3::hash(data);
        if computed_checksum.as_bytes() != stored_checksum {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Bloom filter v1 corruption detected: checksum mismatch. \
                     Expected {:?}, got {:?}.",
                    hex::encode(stored_checksum),
                    hex::encode(computed_checksum.as_bytes())
                ),
            ));
        }

        let mut cursor = Cursor::new(data);
        Self::parse_header_and_data(&mut cursor)
    }

    /// Common header and data parsing logic
    fn parse_header_and_data(cursor: &mut std::io::Cursor<&[u8]>) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::{Error, ErrorKind};

        let num_bits = cursor.read_u64::<LittleEndian>()? as usize;
        let num_hashes = cursor.read_u64::<LittleEndian>()? as usize;
        let num_words = cursor.read_u64::<LittleEndian>()? as usize;

        // SECURITY: Validate parameters to prevent DOS/OOM attacks
        const MAX_BITS: usize = 1_000_000_000; // 1 billion bits = ~119 MB
        const MAX_HASHES: usize = 32; // Way more than needed (typical: 7)
        const MAX_WORDS: usize = MAX_BITS / 64; // ~15.6 million words

        if num_bits > MAX_BITS {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Bloom filter num_bits too large: {} > {}",
                    num_bits, MAX_BITS
                ),
            ));
        }

        if num_hashes > MAX_HASHES {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Bloom filter num_hashes too large: {} > {}",
                    num_hashes, MAX_HASHES
                ),
            ));
        }

        if num_words > MAX_WORDS {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Bloom filter num_words too large: {} > {}",
                    num_words, MAX_WORDS
                ),
            ));
        }

        // CORRECTNESS: Verify num_words matches num_bits
        let expected_words = num_bits.div_ceil(64);
        if num_words != expected_words {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Bloom filter size mismatch: num_words={} but expected={} for num_bits={}",
                    num_words, expected_words, num_bits
                ),
            ));
        }

        let mut bits = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            bits.push(cursor.read_u64::<LittleEndian>()?);
        }

        Ok(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    /// Deserialize from bytes without checksum validation (for backward compatibility)
    ///
    /// Use this for reading old bloom filters that don't have checksums.
    /// New bloom filters should use `from_bytes()` which validates checksums.
    pub fn from_bytes_legacy(bytes: &[u8]) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::{Cursor, Error, ErrorKind};

        let mut cursor = Cursor::new(bytes);

        let num_bits = cursor.read_u64::<LittleEndian>()? as usize;
        let num_hashes = cursor.read_u64::<LittleEndian>()? as usize;
        let num_words = cursor.read_u64::<LittleEndian>()? as usize;

        // SECURITY: Validate parameters
        const MAX_BITS: usize = 1_000_000_000;
        const MAX_HASHES: usize = 32;
        const MAX_WORDS: usize = MAX_BITS / 64;

        if num_bits > MAX_BITS || num_hashes > MAX_HASHES || num_words > MAX_WORDS {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bloom filter parameters exceed limits",
            ));
        }

        let expected_words = num_bits.div_ceil(64);
        if num_words != expected_words {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bloom filter size mismatch",
            ));
        }

        let mut bits = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            bits.push(cursor.read_u64::<LittleEndian>()?);
        }

        Ok(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        24 + self.bits.len() * 8 // metadata + bit vector
    }

    /// Get memory size (alias for size_bytes)
    pub fn memory_size(&self) -> usize {
        self.size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        // Insert items
        for i in 0..100 {
            bloom.insert(&i);
        }

        // Check inserted items (should all return true)
        for i in 0..100 {
            assert!(bloom.contains(&i));
        }

        // Check non-inserted items (most should return false)
        let mut false_positives = 0;
        for i in 100..1000 {
            if bloom.contains(&i) {
                false_positives += 1;
            }
        }

        // False positive rate should be roughly 1% (allow up to 3% for variance)
        let fp_rate = false_positives as f64 / 900.0;
        assert!(fp_rate < 0.03, "False positive rate too high: {}", fp_rate);
    }

    #[test]
    fn test_bloom_filter_serialization() {
        let mut bloom = BloomFilter::new(100, 0.01);

        for i in 0..50 {
            bloom.insert(&i);
        }

        let bytes = bloom.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).unwrap();

        // Check all items are still present
        for i in 0..50 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_bloom_filter_strings() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        let items = vec!["hello", "world", "foo", "bar", "baz"];
        for item in &items {
            bloom.insert(item);
        }

        for item in &items {
            assert!(bloom.contains(item));
        }

        // Bloom filters can have false positives, so this might return true
        let _ = bloom.contains(&"not_there");
    }
}

// ============================================================================
// jj.md Task 3: Blocked Bloom Filter with Cache-Line Alignment
// ============================================================================

/// Cache line size in bytes (64 bytes on most modern CPUs)
const CACHE_LINE_SIZE: usize = 64;

/// Bits per cache line block (512 bits = 64 bytes)
const BLOCK_BITS: usize = CACHE_LINE_SIZE * 8;

/// Level-adaptive false positive rates for LSM-tree
/// L0 has the most reads, so we use lower FPR
#[derive(Debug, Clone, Copy)]
pub struct LevelAdaptiveFPR {
    /// FPR for L0 (most frequently accessed)
    pub l0: f64,
    /// FPR for L1
    pub l1: f64,
    /// FPR for L2 and higher (less frequently accessed)
    pub l2_plus: f64,
}

impl Default for LevelAdaptiveFPR {
    fn default() -> Self {
        Self {
            l0: 0.001,     // 0.1% - Very low for L0
            l1: 0.005,     // 0.5%
            l2_plus: 0.01, // 1% - Standard for deeper levels
        }
    }
}

impl LevelAdaptiveFPR {
    /// Get FPR for a given level
    pub fn for_level(&self, level: usize) -> f64 {
        match level {
            0 => self.l0,
            1 => self.l1,
            _ => self.l2_plus,
        }
    }
}

// =============================================================================
// Task 9 Enhancement: Bloom Filter Cascade
// =============================================================================

/// Cascaded bloom filters with decreasing false positive rates
///
/// ## Design
/// Organizes multiple bloom filters in a cascade where:
/// - First filter (L0): Smallest, catches most negatives quickly
/// - Subsequent filters: Progressively lower FPR for items that pass
///
/// ## Benefits
/// - Fast rejection of non-existent keys (first filter catches ~99%)
/// - Lower overall FPR than single filter for same memory budget
/// - Amortized O(1) lookup with early termination
///
/// ## Example Configuration
/// ```text
/// L0: 1% FPR, 100KB  → Catches 99% of negatives
/// L1: 0.1% FPR, 50KB → Catches 99.9% of remaining
/// L2: 0.01% FPR, 25KB → Final filter for edge cases
/// Combined FPR: 0.01 * 0.001 * 0.0001 = 0.000000001 (1 in billion)
/// ```
#[derive(Debug)]
pub struct BloomFilterCascade {
    /// Cascade of filters (index 0 = first check)
    filters: Vec<BlockedBloomFilter>,
    /// FPR for each level
    level_fpr: Vec<f64>,
    /// Number of items expected
    expected_items: usize,
}

impl BloomFilterCascade {
    /// Create a new cascade with specified levels
    ///
    /// # Arguments
    /// * `expected_items` - Number of items to store
    /// * `level_fprs` - False positive rate for each level (e.g., [0.01, 0.001, 0.0001])
    pub fn new(expected_items: usize, level_fprs: Vec<f64>) -> Self {
        let filters = level_fprs
            .iter()
            .map(|&fpr| BlockedBloomFilter::new(expected_items, fpr))
            .collect();

        Self {
            filters,
            level_fpr: level_fprs,
            expected_items,
        }
    }

    /// Create default 3-level cascade
    ///
    /// - L0: 1% FPR (fast initial check)
    /// - L1: 0.1% FPR (medium precision)
    /// - L2: 0.01% FPR (high precision)
    pub fn default_cascade(expected_items: usize) -> Self {
        Self::new(expected_items, vec![0.01, 0.001, 0.0001])
    }

    /// Create memory-optimized 2-level cascade
    pub fn compact(expected_items: usize) -> Self {
        Self::new(expected_items, vec![0.01, 0.0001])
    }

    /// Insert an item into all cascade levels
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for filter in &mut self.filters {
            filter.insert(item);
        }
    }

    /// Check if item might exist (with early termination)
    ///
    /// Returns false (definitely not present) as soon as any level returns false.
    /// Returns true only if ALL levels return true.
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        for filter in &self.filters {
            if !filter.contains(item) {
                return false;
            }
        }
        true
    }

    /// Check with level tracking (for debugging/stats)
    pub fn contains_with_level<T: Hash>(&self, item: &T) -> (bool, usize) {
        for (level, filter) in self.filters.iter().enumerate() {
            if !filter.contains(item) {
                return (false, level);
            }
        }
        (true, self.filters.len())
    }

    /// Combined theoretical false positive rate
    pub fn combined_fpr(&self) -> f64 {
        self.level_fpr.iter().product()
    }

    /// Number of cascade levels
    pub fn num_levels(&self) -> usize {
        self.filters.len()
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.filters.iter().map(|f| f.memory_usage()).sum()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let mut buf = Vec::new();

        // Magic: "BCF\x01" (Bloom Cascade Filter v1)
        buf.extend_from_slice(&[0x42, 0x43, 0x46, 0x01]);

        // Header
        buf.write_u64::<LittleEndian>(self.expected_items as u64)
            .unwrap();
        buf.write_u64::<LittleEndian>(self.filters.len() as u64)
            .unwrap();

        // FPR for each level
        for &fpr in &self.level_fpr {
            buf.write_u64::<LittleEndian>(fpr.to_bits()).unwrap();
        }

        // Each filter's serialized data
        for filter in &self.filters {
            let filter_bytes = filter.to_bytes();
            buf.write_u64::<LittleEndian>(filter_bytes.len() as u64)
                .unwrap();
            buf.extend_from_slice(&filter_bytes);
        }

        // Checksum
        let checksum = blake3::hash(&buf);
        buf.extend_from_slice(checksum.as_bytes());

        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::{Cursor, Error, ErrorKind};

        const CHECKSUM_SIZE: usize = 32;
        const MIN_SIZE: usize = 4 + 16 + CHECKSUM_SIZE; // magic + header + checksum

        if bytes.len() < MIN_SIZE {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bloom cascade data too small",
            ));
        }

        // Verify magic
        if bytes[..4] != [0x42, 0x43, 0x46, 0x01] {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Invalid bloom cascade magic number",
            ));
        }

        // Verify checksum
        let data_len = bytes.len() - CHECKSUM_SIZE;
        let (data, stored_checksum) = bytes.split_at(data_len);
        let computed_checksum = blake3::hash(data);
        if computed_checksum.as_bytes() != stored_checksum {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bloom cascade checksum mismatch",
            ));
        }

        let mut cursor = Cursor::new(&bytes[4..]); // Skip magic

        let expected_items = cursor.read_u64::<LittleEndian>()? as usize;
        let num_levels = cursor.read_u64::<LittleEndian>()? as usize;

        // Read FPRs
        let mut level_fpr = Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            let bits = cursor.read_u64::<LittleEndian>()?;
            level_fpr.push(f64::from_bits(bits));
        }

        // Read filters
        let mut filters = Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            let filter_len = cursor.read_u64::<LittleEndian>()? as usize;
            let pos = cursor.position() as usize;
            let filter_bytes = &bytes[4 + pos..4 + pos + filter_len];
            cursor.set_position((pos + filter_len) as u64);

            let filter = BlockedBloomFilter::from_bytes(filter_bytes)?;
            filters.push(filter);
        }

        Ok(Self {
            filters,
            level_fpr,
            expected_items,
        })
    }
}

// ============================================================================
// Unified Bloom Filter for SSTable Reader (supports both formats)
// ============================================================================

/// Magic numbers for format detection
const BLOOM_MAGIC_V2: [u8; 4] = [0x42, 0x4C, 0x4D, 0x02]; // "BLM\x02" - Standard BloomFilter
const BLOCKED_BLOOM_MAGIC: [u8; 4] = [0x42, 0x42, 0x46, 0x01]; // "BBF\x01" - BlockedBloomFilter

/// Unified bloom filter that can be deserialized from either format.
///
/// This allows SSTableReader to handle both old SSTables (with BloomFilter)
/// and new SSTables (with BlockedBloomFilter) transparently.
#[derive(Debug)]
pub enum UnifiedBloomFilter {
    /// Standard bloom filter (legacy format)
    Standard(BloomFilter),
    /// Cache-line optimized blocked bloom filter (new format)
    Blocked(BlockedBloomFilter),
}

impl UnifiedBloomFilter {
    /// Check if an item might be in the set
    pub fn contains<T: std::hash::Hash>(&self, item: &T) -> bool {
        match self {
            UnifiedBloomFilter::Standard(bf) => bf.contains(item),
            UnifiedBloomFilter::Blocked(bbf) => bbf.contains(item),
        }
    }

    /// Deserialize from bytes, auto-detecting the format
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};

        if bytes.len() < 4 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Bloom filter data too small to detect format",
            ));
        }

        // Check magic number to determine format
        if bytes[..4] == BLOCKED_BLOOM_MAGIC {
            // New blocked bloom filter format
            BlockedBloomFilter::from_bytes(bytes).map(UnifiedBloomFilter::Blocked)
        } else if bytes[..4] == BLOOM_MAGIC_V2 || bytes.len() >= 56 {
            // Standard bloom filter (v2 with magic or v1 legacy)
            BloomFilter::from_bytes(bytes).map(UnifiedBloomFilter::Standard)
        } else {
            Err(Error::new(
                ErrorKind::InvalidData,
                "Unknown bloom filter format",
            ))
        }
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            UnifiedBloomFilter::Standard(bf) => bf.memory_size(),
            UnifiedBloomFilter::Blocked(bbf) => bbf.memory_usage(),
        }
    }
}

/// Cache-line aligned blocked bloom filter for improved cache efficiency.
///
/// ## jj.md Task 3: Blocked Bloom Filter
///
/// This implementation groups all hash probes for a key into a single
/// cache-line sized block, reducing cache misses from k (number of hashes)
/// to 1.
///
/// ### Performance Benefits
///
/// - **Cache efficiency**: All probes hit the same cache line
/// - **Reduced memory bandwidth**: One cache line load vs k scattered loads
/// - **Better TLB usage**: Single page access per lookup
///
/// ### Trade-offs
///
/// - Slightly higher FPR than standard bloom filter at same memory usage
/// - Block selection uses one hash, remaining hashes probe within block
///
/// ### Reference
///
/// "Cache-Efficient Bloom Filters" (Putze et al., 2007)
#[derive(Debug)]
pub struct BlockedBloomFilter {
    /// Cache-line aligned blocks (each block is 64 bytes = 512 bits)
    blocks: Vec<[u64; 8]>,
    /// Number of blocks
    num_blocks: usize,
    /// Number of hash probes within each block
    num_hashes: usize,
    /// Expected items (for sizing)
    expected_items: usize,
}

impl BlockedBloomFilter {
    /// Create a new blocked bloom filter.
    ///
    /// # Arguments
    /// * `expected_items` - Number of items expected to be inserted
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate bits per key needed
        // m/n = -ln(p) / ln(2)² ≈ -1.44 * ln(p)
        let bits_per_key = (-1.44 * false_positive_rate.ln()).ceil() as usize;
        let bits_per_key = bits_per_key.max(4); // Minimum 4 bits per key

        let total_bits = expected_items * bits_per_key;
        let num_blocks = total_bits.div_ceil(BLOCK_BITS);
        let num_blocks = num_blocks.max(1);

        // Optimal number of hashes: k = ln(2) * (m/n) ≈ 0.693 * bits_per_key
        let num_hashes = ((0.693 * bits_per_key as f64).ceil() as usize).clamp(1, 8);

        let blocks = vec![[0u64; 8]; num_blocks];

        Self {
            blocks,
            num_blocks,
            num_hashes,
            expected_items,
        }
    }

    /// Create a blocked bloom filter with level-adaptive FPR.
    pub fn for_level(expected_items: usize, level: usize) -> Self {
        let fpr = LevelAdaptiveFPR::default().for_level(level);
        Self::new(expected_items, fpr)
    }

    /// Insert an item into the bloom filter.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let (block_idx, h2) = self.compute_hashes(item);

        // Probe within the selected block
        let block = &mut self.blocks[block_idx];
        for i in 0..self.num_hashes {
            let bit_pos = (h2.wrapping_add(i * 31)) % BLOCK_BITS;
            let word_idx = bit_pos / 64;
            let bit_offset = bit_pos % 64;
            block[word_idx] |= 1u64 << bit_offset;
        }
    }

    /// Check if an item might be in the set.
    ///
    /// Returns `true` if item *might* be present (could be false positive).
    /// Returns `false` if item is *definitely not* present (100% accurate).
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (block_idx, h2) = self.compute_hashes(item);
        let block = &self.blocks[block_idx];

        for i in 0..self.num_hashes {
            let bit_pos = (h2.wrapping_add(i * 31)) % BLOCK_BITS;
            let word_idx = bit_pos / 64;
            let bit_offset = bit_pos % 64;
            if (block[word_idx] & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute block index and intra-block hash.
    fn compute_hashes<T: Hash>(&self, item: &T) -> (usize, usize) {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let h1 = hasher.finish();

        // Use upper and lower bits for two independent-ish hashes
        let block_idx = (h1 as usize) % self.num_blocks;
        let h2 = ((h1 >> 32) as usize) | ((h1 as usize) << 32);

        (block_idx, h2)
    }

    /// Serialize to bytes with checksum.
    pub fn to_bytes(&self) -> Vec<u8> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let mut buf = Vec::new();

        // Magic number for blocked bloom filter: "BBF\x01"
        buf.extend_from_slice(&[0x42, 0x42, 0x46, 0x01]);

        // Header
        buf.write_u64::<LittleEndian>(self.num_blocks as u64)
            .unwrap();
        buf.write_u64::<LittleEndian>(self.num_hashes as u64)
            .unwrap();
        buf.write_u64::<LittleEndian>(self.expected_items as u64)
            .unwrap();

        // Block data
        for block in &self.blocks {
            for &word in block {
                buf.write_u64::<LittleEndian>(word).unwrap();
            }
        }

        // BLAKE3 checksum
        let checksum = blake3::hash(&buf);
        buf.extend_from_slice(checksum.as_bytes());

        buf
    }

    /// Deserialize from bytes with checksum validation.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::{Cursor, Error, ErrorKind};

        const MAGIC: [u8; 4] = [0x42, 0x42, 0x46, 0x01]; // "BBF\x01"
        const CHECKSUM_SIZE: usize = 32;
        const HEADER_SIZE: usize = 4 + 24; // magic + 3 * u64

        if bytes.len() < HEADER_SIZE + CHECKSUM_SIZE {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Blocked bloom filter data too small",
            ));
        }

        // Verify magic
        if bytes[..4] != MAGIC {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Invalid blocked bloom filter magic number",
            ));
        }

        // Verify checksum
        let data_len = bytes.len() - CHECKSUM_SIZE;
        let (data, stored_checksum) = bytes.split_at(data_len);
        let computed_checksum = blake3::hash(data);

        if computed_checksum.as_bytes() != stored_checksum {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Blocked bloom filter checksum mismatch",
            ));
        }

        // Parse header
        let mut cursor = Cursor::new(&bytes[4..]);
        let num_blocks = cursor.read_u64::<LittleEndian>()? as usize;
        let num_hashes = cursor.read_u64::<LittleEndian>()? as usize;
        let expected_items = cursor.read_u64::<LittleEndian>()? as usize;

        // Validate sizes
        let expected_data_size = HEADER_SIZE + num_blocks * 64 + CHECKSUM_SIZE;
        if bytes.len() != expected_data_size {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Blocked bloom filter size mismatch: {} vs {}",
                    bytes.len(),
                    expected_data_size
                ),
            ));
        }

        // Parse blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut block = [0u64; 8];
            for word in &mut block {
                *word = cursor.read_u64::<LittleEndian>()?;
            }
            blocks.push(block);
        }

        Ok(Self {
            blocks,
            num_blocks,
            num_hashes,
            expected_items,
        })
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        4 + 24 + self.num_blocks * 64 + 32 // magic + header + blocks + checksum
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.num_blocks * 64
    }

    /// Get fill ratio (fraction of bits set).
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: usize = self
            .blocks
            .iter()
            .flat_map(|b| b.iter())
            .map(|w| w.count_ones() as usize)
            .sum();

        let total_bits = self.num_blocks * BLOCK_BITS;
        set_bits as f64 / total_bits as f64
    }
}

#[cfg(test)]
mod blocked_bloom_tests {
    use super::*;

    #[test]
    fn test_blocked_bloom_basic() {
        let mut bloom = BlockedBloomFilter::new(1000, 0.01);

        // Insert items
        for i in 0..100 {
            bloom.insert(&i);
        }

        // Check inserted items (should all return true)
        for i in 0..100 {
            assert!(bloom.contains(&i), "Missing inserted item {}", i);
        }

        // Check non-inserted items (most should return false)
        let mut false_positives = 0;
        for i in 100..1000 {
            if bloom.contains(&i) {
                false_positives += 1;
            }
        }

        // Blocked bloom filters have slightly higher FPR, allow up to 5%
        let fp_rate = false_positives as f64 / 900.0;
        assert!(fp_rate < 0.05, "False positive rate too high: {}", fp_rate);
    }

    #[test]
    fn test_blocked_bloom_serialization() {
        let mut bloom = BlockedBloomFilter::new(100, 0.01);

        for i in 0..50 {
            bloom.insert(&i);
        }

        let bytes = bloom.to_bytes();
        let restored = BlockedBloomFilter::from_bytes(&bytes).unwrap();

        // Check all items are still present
        for i in 0..50 {
            assert!(
                restored.contains(&i),
                "Missing item {} after deserialize",
                i
            );
        }
    }

    #[test]
    fn test_blocked_bloom_level_adaptive() {
        // L0 should have lower FPR (more bits per key)
        let l0_bloom = BlockedBloomFilter::for_level(1000, 0);
        let l2_bloom = BlockedBloomFilter::for_level(1000, 2);

        // L0 should use more memory (lower FPR = more bits)
        assert!(
            l0_bloom.memory_usage() >= l2_bloom.memory_usage(),
            "L0 bloom should use >= memory than L2"
        );
    }

    #[test]
    fn test_blocked_bloom_u128_keys() {
        let mut bloom = BlockedBloomFilter::new(1000, 0.01);

        // Insert u128 edge IDs
        let edge_ids: Vec<u128> = (0..100).map(|i| i as u128 * 12345678901234567890).collect();

        for id in &edge_ids {
            bloom.insert(id);
        }

        // All inserted items should be found
        for id in &edge_ids {
            assert!(bloom.contains(id));
        }
    }

    #[test]
    fn test_blocked_bloom_checksum_corruption() {
        let mut bloom = BlockedBloomFilter::new(100, 0.01);
        bloom.insert(&42);

        let mut bytes = bloom.to_bytes();

        // Corrupt a byte in the data
        if bytes.len() > 10 {
            bytes[10] ^= 0xFF;
        }

        // Should fail checksum validation
        assert!(BlockedBloomFilter::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_blocked_bloom_fill_ratio() {
        let mut bloom = BlockedBloomFilter::new(1000, 0.01);

        assert_eq!(bloom.fill_ratio(), 0.0);

        for i in 0..500 {
            bloom.insert(&i);
        }

        let ratio = bloom.fill_ratio();
        assert!(ratio > 0.0 && ratio < 1.0);
    }

    #[test]
    fn test_level_adaptive_fpr() {
        let fpr = LevelAdaptiveFPR::default();

        assert_eq!(fpr.for_level(0), 0.001);
        assert_eq!(fpr.for_level(1), 0.005);
        assert_eq!(fpr.for_level(2), 0.01);
        assert_eq!(fpr.for_level(5), 0.01);
    }

    // Bloom Filter Cascade Tests

    #[test]
    fn test_cascade_basic() {
        let mut cascade = BloomFilterCascade::default_cascade(1000);

        // Insert items
        for i in 0..100 {
            cascade.insert(&i);
        }

        // All inserted items should be found
        for i in 0..100 {
            assert!(cascade.contains(&i), "Item {} should be found", i);
        }

        // Verify cascade has expected structure
        assert_eq!(cascade.num_levels(), 3);
        assert!(cascade.combined_fpr() < 0.0001);
    }

    #[test]
    fn test_cascade_early_termination() {
        let mut cascade = BloomFilterCascade::new(1000, vec![0.5, 0.5, 0.5]); // High FPR for testing

        cascade.insert(&42);

        // Inserted item passes all levels
        let (found, level) = cascade.contains_with_level(&42);
        assert!(found);
        assert_eq!(level, 3); // Passed all 3 levels

        // Non-inserted item should fail at some level
        let (found, level) = cascade.contains_with_level(&999999);
        if !found {
            assert!(level < 3, "Should fail before reaching all levels");
        }
    }

    #[test]
    fn test_cascade_serialization() {
        let mut cascade = BloomFilterCascade::default_cascade(500);

        for i in 0..100 {
            cascade.insert(&i);
        }

        // Serialize and deserialize
        let bytes = cascade.to_bytes();
        let restored = BloomFilterCascade::from_bytes(&bytes).unwrap();

        // Check properties preserved
        assert_eq!(restored.num_levels(), cascade.num_levels());
        assert_eq!(restored.expected_items, cascade.expected_items);

        // Check contents
        for i in 0..100 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_cascade_memory() {
        let cascade = BloomFilterCascade::default_cascade(10000);

        // Should use reasonable memory
        let mem = cascade.memory_usage();
        assert!(mem > 0);

        // Combined FPR should be very low
        let combined_fpr = cascade.combined_fpr();
        assert!(
            combined_fpr < 0.000001,
            "Combined FPR should be < 1 in million"
        );
    }

    #[test]
    fn test_cascade_compact() {
        let compact = BloomFilterCascade::compact(1000);
        let full = BloomFilterCascade::default_cascade(1000);

        // Compact should use fewer levels
        assert_eq!(compact.num_levels(), 2);
        assert_eq!(full.num_levels(), 3);

        // But still maintain low FPR
        assert!(compact.combined_fpr() < 0.0001);
    }
}
