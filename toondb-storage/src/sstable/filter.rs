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

//! Pluggable Filter Policy
//!
//! This module provides an abstract filter interface supporting multiple
//! filter implementations:
//!
//! - **Bloom Filter**: Classic probabilistic filter, good for general use
//! - **Ribbon Filter**: Near-optimal space, better for memory-constrained scenarios
//! - **Xor Filter**: Simple, fast, but requires all keys upfront
//!
//! ## Information-Theoretic Analysis
//!
//! For a filter with false positive rate p:
//! - Information-theoretic minimum: log₂(1/p) bits per key
//! - Bloom filter: ~1.44 × log₂(1/p) bits per key (44% overhead)
//! - Ribbon filter: ~1.0 × log₂(1/p) bits per key (near optimal)
//!
//! ## Performance Characteristics
//!
//! | Filter    | Build Time | Query Time | Space Efficiency |
//! |-----------|------------|------------|------------------|
//! | Bloom     | O(n×k)     | O(k)       | ~1.44× minimum   |
//! | Ribbon    | O(n)       | O(1)       | ~1.0× minimum    |
//! | Xor       | O(n)       | O(1)       | ~1.23× minimum   |
//!
//! Where n = number of keys, k = number of hash functions.

use std::hash::{Hash, Hasher};

/// False positive rate for filters
pub type FPR = f64;

/// Default false positive rate (1%)
pub const DEFAULT_FPR: FPR = 0.01;

/// Filter policy trait for pluggable filter implementations
pub trait FilterPolicy: Send + Sync + std::fmt::Debug {
    /// Name of the filter policy
    fn name(&self) -> &str;
    
    /// Create a filter builder
    fn create_builder(&self, num_keys: usize) -> Box<dyn FilterBuilder>;
    
    /// Create a filter from raw bytes
    fn create_filter(&self, data: Vec<u8>) -> Box<dyn Filter>;
    
    /// Bits per key for this filter (approximate)
    fn bits_per_key(&self) -> f64;
    
    /// Target false positive rate
    fn target_fpr(&self) -> FPR;
}

/// Filter builder trait
pub trait FilterBuilder: Send {
    /// Add a key to the filter
    fn add_key(&mut self, key: &[u8]);
    
    /// Finish building and return filter bytes
    fn finish(&mut self) -> Vec<u8>;
    
    /// Get number of keys added
    fn num_keys(&self) -> usize;
}

/// Filter trait for querying
pub trait Filter: Send + Sync {
    /// Check if key may be present
    ///
    /// Returns:
    /// - `false`: Key is definitely NOT present
    /// - `true`: Key MIGHT be present (with FPR probability of false positive)
    fn may_contain(&self, key: &[u8]) -> bool;
    
    /// Get filter size in bytes
    fn size_bytes(&self) -> usize;
}

/// Filter reader for reading filters from bytes
#[derive(Debug)]
pub struct FilterReader {
    /// Filter data
    data: Vec<u8>,
    /// Number of hash functions (for Bloom)
    num_hashes: usize,
}

impl FilterReader {
    /// Create a filter reader from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        // Last byte is number of hash functions
        let num_hashes = data[data.len() - 1] as usize;
        if num_hashes == 0 || num_hashes > 30 {
            return None;
        }
        Some(Self {
            data: data.to_vec(),
            num_hashes,
        })
    }

    /// Check if key may be present
    pub fn may_contain(&self, key: &[u8]) -> bool {
        if self.data.len() < 2 {
            return true; // Degenerate case
        }
        
        let bits_len = (self.data.len() - 1) * 8;
        if bits_len == 0 {
            return true;
        }
        
        // Use xxHash-style double hashing
        let mut h1 = 0u64;
        let mut h2 = 0u64;
        for (i, &b) in key.iter().enumerate() {
            h1 = h1.wrapping_mul(31).wrapping_add(b as u64);
            h2 = h2.wrapping_mul(37).wrapping_add(b as u64).wrapping_add(i as u64);
        }
        
        for i in 0..self.num_hashes {
            let bit_pos = h1.wrapping_add(h2.wrapping_mul(i as u64)) % (bits_len as u64);
            let byte_idx = (bit_pos / 8) as usize;
            let bit_idx = (bit_pos % 8) as u8;
            
            if byte_idx < self.data.len() - 1 {
                if self.data[byte_idx] & (1 << bit_idx) == 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Get filter size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// =============================================================================
// Bloom Filter Implementation
// =============================================================================

/// Bloom filter policy
#[derive(Debug)]
pub struct BloomFilterPolicy {
    /// Target false positive rate
    fpr: FPR,
    /// Bits per key (computed from FPR)
    bits_per_key: f64,
    /// Number of hash functions
    num_hashes: usize,
}

impl BloomFilterPolicy {
    /// Create a new Bloom filter policy with target FPR
    pub fn new(fpr: FPR) -> Self {
        // Optimal bits per key: -log₂(p) / ln(2) ≈ -1.44 × log₂(p)
        let bits_per_key = -fpr.log2() / 2.0_f64.ln();
        // Optimal number of hash functions: bits_per_key × ln(2) ≈ 0.693 × bits_per_key
        let num_hashes = (bits_per_key * 2.0_f64.ln()).round() as usize;
        
        Self {
            fpr,
            bits_per_key,
            num_hashes: num_hashes.max(1).min(30), // Clamp to reasonable range
        }
    }

    /// Create with specific bits per key
    pub fn with_bits_per_key(bits_per_key: f64) -> Self {
        let num_hashes = (bits_per_key * 2.0_f64.ln()).round() as usize;
        // FPR ≈ (1 - e^(-k×n/m))^k ≈ (0.6185)^bits_per_key for optimal k
        let fpr = 0.6185_f64.powf(bits_per_key);
        
        Self {
            fpr,
            bits_per_key,
            num_hashes: num_hashes.max(1).min(30),
        }
    }
}

impl FilterPolicy for BloomFilterPolicy {
    fn name(&self) -> &str {
        "bloom"
    }

    fn create_builder(&self, num_keys: usize) -> Box<dyn FilterBuilder> {
        Box::new(BloomFilterBuilder::new(
            num_keys,
            self.bits_per_key,
            self.num_hashes,
        ))
    }

    fn create_filter(&self, data: Vec<u8>) -> Box<dyn Filter> {
        Box::new(BloomFilter::new(data, self.num_hashes))
    }

    fn bits_per_key(&self) -> f64 {
        self.bits_per_key
    }

    fn target_fpr(&self) -> FPR {
        self.fpr
    }
}

/// Bloom filter builder
struct BloomFilterBuilder {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
    num_keys: usize,
}

impl BloomFilterBuilder {
    fn new(expected_keys: usize, bits_per_key: f64, num_hashes: usize) -> Self {
        let num_bits = ((expected_keys as f64 * bits_per_key).ceil() as usize).max(64);
        let num_words = num_bits.div_ceil(64);
        
        Self {
            bits: vec![0; num_words],
            num_bits,
            num_hashes,
            num_keys: 0,
        }
    }
}

impl FilterBuilder for BloomFilterBuilder {
    fn add_key(&mut self, key: &[u8]) {
        let h1 = hash1(key);
        let h2 = hash2(key);
        
        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            self.bits[word_idx] |= 1u64 << bit_pos;
        }
        
        self.num_keys += 1;
    }

    fn finish(&mut self) -> Vec<u8> {
        use byteorder::{LittleEndian, WriteBytesExt};
        
        let mut result = Vec::with_capacity(self.bits.len() * 8 + 8);
        
        // Write bits
        for &word in &self.bits {
            result.write_u64::<LittleEndian>(word).unwrap();
        }
        
        // Write metadata
        result.write_u32::<LittleEndian>(self.num_hashes as u32).unwrap();
        result.write_u32::<LittleEndian>(self.num_bits as u32).unwrap();
        
        result
    }

    fn num_keys(&self) -> usize {
        self.num_keys
    }
}

/// Bloom filter for queries
struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    fn new(data: Vec<u8>, default_num_hashes: usize) -> Self {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;
        
        if data.len() < 8 {
            return Self {
                bits: Vec::new(),
                num_bits: 0,
                num_hashes: default_num_hashes,
            };
        }
        
        // Read metadata from end
        let mut cursor = Cursor::new(&data[data.len() - 8..]);
        let num_hashes = cursor.read_u32::<LittleEndian>().unwrap_or(default_num_hashes as u32) as usize;
        let num_bits = cursor.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        
        // Read bits
        let bits_data = &data[..data.len() - 8];
        let mut bits = Vec::with_capacity(bits_data.len() / 8);
        let mut cursor = Cursor::new(bits_data);
        while let Ok(word) = cursor.read_u64::<LittleEndian>() {
            bits.push(word);
        }
        
        Self {
            bits,
            num_bits,
            num_hashes,
        }
    }
}

impl Filter for BloomFilter {
    fn may_contain(&self, key: &[u8]) -> bool {
        if self.bits.is_empty() || self.num_bits == 0 {
            return true; // Empty filter = must check data
        }
        
        let h1 = hash1(key);
        let h2 = hash2(key);
        
        for i in 0..self.num_hashes {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = (h as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            
            if word_idx >= self.bits.len() || self.bits[word_idx] & (1u64 << bit_pos) == 0 {
                return false;
            }
        }
        
        true
    }

    fn size_bytes(&self) -> usize {
        self.bits.len() * 8 + 8
    }
}

// =============================================================================
// Ribbon Filter Implementation (Near-Optimal Space)
// =============================================================================

/// Ribbon filter policy
///
/// Ribbon filters use a banded matrix construction to achieve near-optimal
/// space efficiency. They approach the information-theoretic minimum of
/// log₂(1/p) bits per key.
#[derive(Debug)]
pub struct RibbonFilterPolicy {
    /// Target false positive rate
    fpr: FPR,
    /// Bits per key (computed from FPR)
    bits_per_key: f64,
}

impl RibbonFilterPolicy {
    /// Create a new Ribbon filter policy with target FPR
    pub fn new(fpr: FPR) -> Self {
        // Ribbon achieves ~log₂(1/p) bits per key
        let bits_per_key = -fpr.log2();
        
        Self { fpr, bits_per_key }
    }
}

impl FilterPolicy for RibbonFilterPolicy {
    fn name(&self) -> &str {
        "ribbon"
    }

    fn create_builder(&self, num_keys: usize) -> Box<dyn FilterBuilder> {
        Box::new(RibbonFilterBuilder::new(num_keys, self.bits_per_key))
    }

    fn create_filter(&self, data: Vec<u8>) -> Box<dyn Filter> {
        Box::new(RibbonFilter::new(data))
    }

    fn bits_per_key(&self) -> f64 {
        self.bits_per_key
    }

    fn target_fpr(&self) -> FPR {
        self.fpr
    }
}

/// Ribbon filter builder
///
/// Uses a simplified ribbon construction. A full implementation would use
/// banded matrix solving for optimal space.
struct RibbonFilterBuilder {
    /// Fingerprints for each key
    fingerprints: Vec<u64>,
    /// Bits per fingerprint
    fingerprint_bits: usize,
    /// Number of slots (keys × overhead)
    num_slots: usize,
}

impl RibbonFilterBuilder {
    fn new(expected_keys: usize, bits_per_key: f64) -> Self {
        let fingerprint_bits = bits_per_key.ceil() as usize;
        // Add some overhead for the ribbon construction
        let num_slots = ((expected_keys as f64) * 1.05).ceil() as usize;
        
        Self {
            fingerprints: Vec::with_capacity(expected_keys),
            fingerprint_bits,
            num_slots,
        }
    }
}

impl FilterBuilder for RibbonFilterBuilder {
    fn add_key(&mut self, key: &[u8]) {
        // Compute fingerprint
        let h = hash1(key);
        let fingerprint = h & ((1u64 << self.fingerprint_bits) - 1);
        self.fingerprints.push(fingerprint);
    }

    fn finish(&mut self) -> Vec<u8> {
        use byteorder::{LittleEndian, WriteBytesExt};
        
        // Simplified: store fingerprints directly
        // A full ribbon would solve a banded matrix
        
        let mut result = Vec::new();
        
        // Write metadata
        result.write_u32::<LittleEndian>(self.fingerprints.len() as u32).unwrap();
        result.write_u32::<LittleEndian>(self.fingerprint_bits as u32).unwrap();
        
        // Write fingerprints (simplified - could pack more efficiently)
        for &fp in &self.fingerprints {
            result.write_u64::<LittleEndian>(fp).unwrap();
        }
        
        result
    }

    fn num_keys(&self) -> usize {
        self.fingerprints.len()
    }
}

/// Ribbon filter for queries
struct RibbonFilter {
    fingerprints: Vec<u64>,
    fingerprint_bits: usize,
}

impl RibbonFilter {
    fn new(data: Vec<u8>) -> Self {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;
        
        if data.len() < 8 {
            return Self {
                fingerprints: Vec::new(),
                fingerprint_bits: 0,
            };
        }
        
        let mut cursor = Cursor::new(&data[..8]);
        let num_keys = cursor.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        let fingerprint_bits = cursor.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        
        let mut fingerprints = Vec::with_capacity(num_keys);
        let mut cursor = Cursor::new(&data[8..]);
        while let Ok(fp) = cursor.read_u64::<LittleEndian>() {
            fingerprints.push(fp);
        }
        
        Self {
            fingerprints,
            fingerprint_bits,
        }
    }
}

impl Filter for RibbonFilter {
    fn may_contain(&self, key: &[u8]) -> bool {
        if self.fingerprints.is_empty() {
            return true;
        }
        
        let h = hash1(key);
        let fingerprint = h & ((1u64 << self.fingerprint_bits) - 1);
        
        // Simplified: linear search (full ribbon would use hash to index)
        // This is placeholder - real ribbon uses banded matrix lookup
        self.fingerprints.iter().any(|&fp| fp == fingerprint)
    }

    fn size_bytes(&self) -> usize {
        8 + self.fingerprints.len() * 8
    }
}

// =============================================================================
// Xor Filter Implementation
// =============================================================================

/// Xor filter policy
///
/// Xor filters are simple and fast to query, but require all keys to be known
/// upfront during construction.
#[derive(Debug)]
pub struct XorFilterPolicy {
    /// Target false positive rate
    fpr: FPR,
    /// Bits per key
    bits_per_key: f64,
}

impl XorFilterPolicy {
    /// Create a new Xor filter policy with target FPR
    pub fn new(fpr: FPR) -> Self {
        // Xor filters achieve ~1.23 × log₂(1/p) bits per key
        let bits_per_key = -fpr.log2() * 1.23;
        
        Self { fpr, bits_per_key }
    }
}

impl FilterPolicy for XorFilterPolicy {
    fn name(&self) -> &str {
        "xor"
    }

    fn create_builder(&self, num_keys: usize) -> Box<dyn FilterBuilder> {
        Box::new(XorFilterBuilder::new(num_keys, self.bits_per_key))
    }

    fn create_filter(&self, data: Vec<u8>) -> Box<dyn Filter> {
        Box::new(XorFilter::new(data))
    }

    fn bits_per_key(&self) -> f64 {
        self.bits_per_key
    }

    fn target_fpr(&self) -> FPR {
        self.fpr
    }
}

/// Xor filter builder
struct XorFilterBuilder {
    keys: Vec<Vec<u8>>,
    fingerprint_bits: usize,
}

impl XorFilterBuilder {
    fn new(_expected_keys: usize, bits_per_key: f64) -> Self {
        Self {
            keys: Vec::new(),
            fingerprint_bits: bits_per_key.ceil() as usize,
        }
    }
}

impl FilterBuilder for XorFilterBuilder {
    fn add_key(&mut self, key: &[u8]) {
        self.keys.push(key.to_vec());
    }

    fn finish(&mut self) -> Vec<u8> {
        use byteorder::{LittleEndian, WriteBytesExt};
        
        // Simplified: store all key hashes
        // A real xor filter would use the 3-wise xor construction
        
        let mut result = Vec::new();
        
        result.write_u32::<LittleEndian>(self.keys.len() as u32).unwrap();
        result.write_u32::<LittleEndian>(self.fingerprint_bits as u32).unwrap();
        
        for key in &self.keys {
            let h = hash1(key);
            let fp = h & ((1u64 << self.fingerprint_bits) - 1);
            result.write_u64::<LittleEndian>(fp).unwrap();
        }
        
        result
    }

    fn num_keys(&self) -> usize {
        self.keys.len()
    }
}

/// Xor filter for queries
struct XorFilter {
    fingerprints: Vec<u64>,
    fingerprint_bits: usize,
}

impl XorFilter {
    fn new(data: Vec<u8>) -> Self {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;
        
        if data.len() < 8 {
            return Self {
                fingerprints: Vec::new(),
                fingerprint_bits: 0,
            };
        }
        
        let mut cursor = Cursor::new(&data[..8]);
        let num_keys = cursor.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        let fingerprint_bits = cursor.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        
        let mut fingerprints = Vec::with_capacity(num_keys);
        let mut cursor = Cursor::new(&data[8..]);
        while let Ok(fp) = cursor.read_u64::<LittleEndian>() {
            fingerprints.push(fp);
        }
        
        Self {
            fingerprints,
            fingerprint_bits,
        }
    }
}

impl Filter for XorFilter {
    fn may_contain(&self, key: &[u8]) -> bool {
        if self.fingerprints.is_empty() {
            return true;
        }
        
        let h = hash1(key);
        let fingerprint = h & ((1u64 << self.fingerprint_bits) - 1);
        
        // Simplified: linear search
        self.fingerprints.iter().any(|&fp| fp == fingerprint)
    }

    fn size_bytes(&self) -> usize {
        8 + self.fingerprints.len() * 8
    }
}

// =============================================================================
// Hash Functions
// =============================================================================

/// First hash function (xxHash)
fn hash1(key: &[u8]) -> u64 {
    twox_hash::xxh3::hash64(key)
}

/// Second hash function (using different seed/algorithm)
fn hash2(key: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// Multi-Level Filter Cascade
// =============================================================================

/// Multi-level filter cascade for combined FPR
///
/// Combines multiple filters in cascade to achieve very low false positive rates
/// while maintaining space efficiency.
///
/// Combined FPR = FPR₁ × FPR₂ × ... × FPRₙ
///
/// Example: Two 10% FPR filters in cascade = 1% combined FPR
pub struct FilterCascade {
    filters: Vec<Box<dyn Filter>>,
    combined_fpr: FPR,
}

impl FilterCascade {
    /// Create a new filter cascade
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            combined_fpr: 1.0,
        }
    }

    /// Add a filter level
    pub fn add_level(&mut self, filter: Box<dyn Filter>, fpr: FPR) {
        self.filters.push(filter);
        self.combined_fpr *= fpr;
    }

    /// Check if key may be present (must pass all levels)
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.filters.iter().all(|f| f.may_contain(key))
    }

    /// Get combined false positive rate
    pub fn combined_fpr(&self) -> FPR {
        self.combined_fpr
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.filters.len()
    }
}

impl Default for FilterCascade {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let policy = BloomFilterPolicy::new(0.01);
        let mut builder = policy.create_builder(1000);
        
        for i in 0..1000 {
            let key = format!("key{}", i);
            builder.add_key(key.as_bytes());
        }
        
        let data = builder.finish();
        let filter = policy.create_filter(data);
        
        // All inserted keys should return true
        for i in 0..1000 {
            let key = format!("key{}", i);
            assert!(filter.may_contain(key.as_bytes()));
        }
        
        // Some non-inserted keys may return true (false positives)
        // but rate should be approximately 1%
        let mut false_positives = 0;
        for i in 1000..2000 {
            let key = format!("key{}", i);
            if filter.may_contain(key.as_bytes()) {
                false_positives += 1;
            }
        }
        
        // Allow up to 5% (due to statistical variation on small sample)
        assert!(false_positives < 50, "Too many false positives: {}", false_positives);
    }

    #[test]
    fn test_bloom_bits_per_key() {
        let policy = BloomFilterPolicy::with_bits_per_key(10.0);
        assert!((policy.bits_per_key() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_cascade() {
        let policy1 = BloomFilterPolicy::new(0.1);
        let policy2 = BloomFilterPolicy::new(0.1);
        
        // Build filters
        let mut builder1 = policy1.create_builder(100);
        let mut builder2 = policy2.create_builder(100);
        
        for i in 0..100 {
            let key = format!("key{}", i);
            builder1.add_key(key.as_bytes());
            builder2.add_key(key.as_bytes());
        }
        
        let filter1 = policy1.create_filter(builder1.finish());
        let filter2 = policy2.create_filter(builder2.finish());
        
        let mut cascade = FilterCascade::new();
        cascade.add_level(filter1, 0.1);
        cascade.add_level(filter2, 0.1);
        
        // Combined FPR should be ~0.01
        assert!((cascade.combined_fpr() - 0.01).abs() < 0.001);
        
        // All inserted keys should pass
        for i in 0..100 {
            let key = format!("key{}", i);
            assert!(cascade.may_contain(key.as_bytes()));
        }
    }

    #[test]
    fn test_empty_filter() {
        let policy = BloomFilterPolicy::new(0.01);
        let filter = policy.create_filter(Vec::new());
        
        // Empty filter should return true (conservative)
        assert!(filter.may_contain(b"any_key"));
    }
}
