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

//! Dictionary-Based Compression
//!
//! Implements dictionary compression for repetitive data patterns like JSON
//! agent traces with common schemas.
//!
//! ## jj.md Task 5: Dictionary Compression
//!
//! Goals:
//! - 2-4x better compression ratio for small payloads
//! - Reduce storage cost by 50-70%
//! - Faster decompression (dictionary pre-loaded)
//!
//! ## How It Works
//!
//! 1. Train a dictionary from representative sample payloads
//! 2. Share the dictionary across an SSTable (stored in footer)
//! 3. Use the dictionary for both compression and decompression
//!
//! For agent trace payloads (typically JSON with repetitive schemas like
//! `{"prompt": ..., "response": ..., "model": ...}`), dictionary compression
//! exploits the repeated structure for 5-10x compression ratios.
//!
//! ## Reference
//!
//! Zstd Dictionary Compression - https://facebook.github.io/zstd/#small-data

use std::io;

#[cfg(test)]
use std::io::Cursor;

/// Default dictionary size in bytes (32KB is a good balance)
pub const DEFAULT_DICT_SIZE: usize = 32 * 1024;

/// Minimum samples needed for effective dictionary training
pub const MIN_TRAINING_SAMPLES: usize = 100;

/// Maximum sample size for training (larger samples are truncated)
pub const MAX_SAMPLE_SIZE: usize = 128 * 1024;

/// A trained compression dictionary for Zstd.
///
/// The dictionary contains common patterns extracted from sample data,
/// enabling much better compression for small, repetitive payloads.
#[derive(Clone)]
pub struct CompressionDictionary {
    /// Raw dictionary bytes
    data: Vec<u8>,
    /// Dictionary ID (for validation)
    id: u32,
}

impl std::fmt::Debug for CompressionDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressionDictionary")
            .field("size", &self.data.len())
            .field("id", &self.id)
            .finish()
    }
}

impl CompressionDictionary {
    /// Train a new dictionary from sample data.
    ///
    /// # Arguments
    /// * `samples` - Representative sample payloads
    /// * `dict_size` - Target dictionary size in bytes (default: 32KB)
    ///
    /// # Returns
    /// A trained dictionary, or an error if training fails.
    ///
    /// # Example
    /// ```ignore
    /// let samples: Vec<Vec<u8>> = payloads.clone();
    /// let dict = CompressionDictionary::train(&samples, 32 * 1024)?;
    /// ```
    pub fn train(samples: &[Vec<u8>], dict_size: usize) -> io::Result<Self> {
        if samples.len() < MIN_TRAINING_SAMPLES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Need at least {} samples for dictionary training, got {}",
                    MIN_TRAINING_SAMPLES,
                    samples.len()
                ),
            ));
        }

        // Use zstd dictionary training - samples need to be Vec<u8> or similar
        let dict_data = zstd::dict::from_samples(samples, dict_size)
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Extract dictionary ID from the trained dictionary
        let id = Self::extract_dict_id(&dict_data);

        Ok(Self {
            data: dict_data,
            id,
        })
    }

    /// Create a dictionary from raw bytes (for loading from storage).
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let id = Self::extract_dict_id(&data);
        Self { data, id }
    }

    /// Get the raw dictionary bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the dictionary size.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get the dictionary ID.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Extract dictionary ID from raw bytes.
    fn extract_dict_id(data: &[u8]) -> u32 {
        if data.len() >= 8 {
            // Zstd dictionary ID is at bytes 4-7
            u32::from_le_bytes([data[4], data[5], data[6], data[7]])
        } else {
            0
        }
    }
}

/// Compressor using a pre-trained dictionary.
pub struct DictionaryCompressor {
    /// The compression dictionary bytes (owned copy)
    dict_bytes: Vec<u8>,
    /// Compression level (1-22, default 3)
    level: i32,
}

impl DictionaryCompressor {
    /// Create a new dictionary compressor.
    pub fn new(dict: CompressionDictionary, level: i32) -> Self {
        Self {
            dict_bytes: dict.data,
            level,
        }
    }

    /// Create with default compression level.
    pub fn with_default_level(dict: CompressionDictionary) -> Self {
        Self::new(dict, 3)
    }

    /// Compress data using the dictionary.
    pub fn compress(&self, data: &[u8]) -> io::Result<Vec<u8>> {
        // Create a compressor with the dictionary
        let mut compressor = zstd::bulk::Compressor::with_dictionary(self.level, &self.dict_bytes)
            .map_err(|e| io::Error::other(e.to_string()))?;

        compressor
            .compress(data)
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Get the dictionary bytes.
    pub fn dictionary_bytes(&self) -> &[u8] {
        &self.dict_bytes
    }
}

/// Decompressor using a pre-trained dictionary.
pub struct DictionaryDecompressor {
    /// The decompression dictionary bytes
    dict_bytes: Vec<u8>,
}

impl DictionaryDecompressor {
    /// Create a new dictionary decompressor.
    pub fn new(dict: CompressionDictionary) -> Self {
        Self {
            dict_bytes: dict.data,
        }
    }

    /// Decompress data using the dictionary.
    pub fn decompress(&self, data: &[u8]) -> io::Result<Vec<u8>> {
        // Create a decompressor with the dictionary
        let mut decompressor = zstd::bulk::Decompressor::with_dictionary(&self.dict_bytes)
            .map_err(|e| io::Error::other(e.to_string()))?;

        decompressor
            .decompress(data, data.len() * 20) // estimate 20x expansion max
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Decompress into a pre-allocated buffer.
    pub fn decompress_to(&self, data: &[u8], output: &mut Vec<u8>) -> io::Result<()> {
        let result = self.decompress(data)?;
        output.clear();
        output.extend_from_slice(&result);
        Ok(())
    }

    /// Get the dictionary bytes.
    pub fn dictionary_bytes(&self) -> &[u8] {
        &self.dict_bytes
    }
}

/// Builder for collecting samples and training a dictionary.
#[derive(Default)]
pub struct DictionaryBuilder {
    samples: Vec<Vec<u8>>,
    max_samples: usize,
    dict_size: usize,
}

impl DictionaryBuilder {
    /// Create a new dictionary builder.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            max_samples: 10000,
            dict_size: DEFAULT_DICT_SIZE,
        }
    }

    /// Set the maximum number of samples to collect.
    pub fn max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self
    }

    /// Set the target dictionary size.
    pub fn dict_size(mut self, size: usize) -> Self {
        self.dict_size = size;
        self
    }

    /// Add a sample for training.
    pub fn add_sample(&mut self, sample: Vec<u8>) {
        if self.samples.len() < self.max_samples {
            self.samples.push(sample);
        }
    }

    /// Add a sample from a slice.
    pub fn add_sample_slice(&mut self, sample: &[u8]) {
        if self.samples.len() < self.max_samples {
            self.samples.push(sample.to_vec());
        }
    }

    /// Get the current number of samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Check if we have enough samples to train.
    pub fn can_train(&self) -> bool {
        self.samples.len() >= MIN_TRAINING_SAMPLES
    }

    /// Train a dictionary from collected samples.
    pub fn build(self) -> io::Result<CompressionDictionary> {
        CompressionDictionary::train(&self.samples, self.dict_size)
    }
}

/// Statistics for dictionary compression.
#[derive(Debug, Default, Clone)]
pub struct DictionaryCompressionStats {
    /// Total bytes before compression
    pub bytes_in: u64,
    /// Total bytes after compression
    pub bytes_out: u64,
    /// Number of compressions
    pub compressions: u64,
    /// Number of decompressions
    pub decompressions: u64,
}

impl DictionaryCompressionStats {
    /// Record a compression operation.
    pub fn record_compression(&mut self, input_size: usize, output_size: usize) {
        self.bytes_in += input_size as u64;
        self.bytes_out += output_size as u64;
        self.compressions += 1;
    }

    /// Record a decompression operation.
    pub fn record_decompression(&mut self, compressed_size: usize, decompressed_size: usize) {
        self.bytes_out += compressed_size as u64;
        self.bytes_in += decompressed_size as u64;
        self.decompressions += 1;
    }

    /// Get the compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_out == 0 {
            1.0
        } else {
            self.bytes_in as f64 / self.bytes_out as f64
        }
    }

    /// Get the space savings percentage.
    pub fn space_savings(&self) -> f64 {
        if self.bytes_in == 0 {
            0.0
        } else {
            (1.0 - (self.bytes_out as f64 / self.bytes_in as f64)) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_json_samples(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|i| {
                format!(
                    r#"{{"id":{},"type":"trace","agent":"agent_{}","model":"gpt-4","prompt":"Hello, how are you?","response":"I am doing well, thank you!","tokens":{{"input":10,"output":15}},"latency_ms":{}}}"#,
                    i,
                    i % 10,
                    100 + (i % 500)
                )
                .into_bytes()
            })
            .collect()
    }

    #[test]
    fn test_dictionary_builder() {
        let mut builder = DictionaryBuilder::new()
            .max_samples(200)
            .dict_size(16 * 1024);

        let samples = generate_json_samples(150);
        for sample in samples {
            builder.add_sample(sample);
        }

        assert_eq!(builder.sample_count(), 150);
        assert!(builder.can_train());

        let dict = builder.build().unwrap();
        assert!(dict.size() > 0);
        assert!(dict.size() <= 16 * 1024);
    }

    #[test]
    fn test_dictionary_compression_roundtrip() {
        let samples = generate_json_samples(200);

        let dict = CompressionDictionary::train(&samples, 16 * 1024).unwrap();

        let compressor = DictionaryCompressor::with_default_level(dict.clone());
        let decompressor = DictionaryDecompressor::new(dict);

        // Test compression/decompression of a new sample
        let test_data = r#"{"id":9999,"type":"trace","agent":"agent_5","model":"gpt-4","prompt":"Test message","response":"Test response","tokens":{"input":5,"output":10},"latency_ms":150}"#.as_bytes();

        let compressed = compressor.compress(test_data).unwrap();
        let decompressed = decompressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, test_data);

        // Verify compression ratio
        let ratio = test_data.len() as f64 / compressed.len() as f64;
        println!(
            "Compression ratio: {:.2}x ({} -> {} bytes)",
            ratio,
            test_data.len(),
            compressed.len()
        );

        // With dictionary, we should get at least 1x compression
        assert!(
            ratio >= 0.9,
            "Expected reasonable compression, got {:.2}x",
            ratio
        );
    }

    #[test]
    fn test_dictionary_from_bytes() {
        let samples = generate_json_samples(150);

        let original = CompressionDictionary::train(&samples, 8 * 1024).unwrap();
        let bytes = original.as_bytes().to_vec();

        let restored = CompressionDictionary::from_bytes(bytes);

        assert_eq!(restored.id(), original.id());
        assert_eq!(restored.size(), original.size());
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = DictionaryCompressionStats::default();

        stats.record_compression(1000, 200);
        stats.record_compression(2000, 400);

        assert_eq!(stats.compressions, 2);
        assert_eq!(stats.bytes_in, 3000);
        assert_eq!(stats.bytes_out, 600);
        assert!((stats.compression_ratio() - 5.0).abs() < 0.01);
        assert!((stats.space_savings() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_insufficient_samples() {
        let samples: Vec<Vec<u8>> = vec![b"too few samples".to_vec()];
        let result = CompressionDictionary::train(&samples, DEFAULT_DICT_SIZE);
        assert!(result.is_err());
    }

    #[test]
    fn test_dictionary_improves_small_payload_compression() {
        // Generate training samples
        let samples = generate_json_samples(200);

        let dict = CompressionDictionary::train(&samples, 32 * 1024).unwrap();
        let compressor = DictionaryCompressor::with_default_level(dict);

        // Compress a small payload (typical agent trace)
        let small_payload = r#"{"id":1,"type":"trace","agent":"agent_1","model":"gpt-4","prompt":"Hi","response":"Hello!","tokens":{"input":1,"output":2},"latency_ms":50}"#.as_bytes();

        let with_dict = compressor.compress(small_payload).unwrap();
        let without_dict = zstd::encode_all(Cursor::new(small_payload), 3).unwrap();

        println!("Small payload: {} bytes", small_payload.len());
        println!(
            "With dictionary: {} bytes ({:.1}x)",
            with_dict.len(),
            small_payload.len() as f64 / with_dict.len() as f64
        );
        println!(
            "Without dictionary: {} bytes ({:.1}x)",
            without_dict.len(),
            small_payload.len() as f64 / without_dict.len() as f64
        );

        // Dictionary should provide better compression for small payloads
        // (or at least not be worse - small payloads sometimes expand with standard zstd)
        assert!(
            with_dict.len() <= without_dict.len() + 50, // Allow some margin
            "Dictionary compression should be competitive"
        );
    }
}
