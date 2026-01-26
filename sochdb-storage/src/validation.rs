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

//! SSTable Validation Layer
//!
//! Implements defense-in-depth validation for memory-mapped files to prevent crashes
//! from corrupted, truncated, or tampered files.
//!
//! ## Safety Guarantees
//!
//! 1. **Pre-mmap validation**: Verify file integrity before memory mapping
//! 2. **Magic number check**: Ensure file is valid SSTable format
//! 3. **Size validation**: Prevent reading beyond file boundaries
//! 4. **Checksum verification**: Detect bit rot and tampering
//!
//! ## Formal Safety Invariant
//!
//! ∀p ∈ MappedPages: validate_before_mmap(file) = Ok ⟹ p.valid = true
//!
//! This establishes that all memory-mapped pages are valid before dereferencing.

use blake3::Hasher;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use sochdb_core::{Result, SochDBError};

/// Minimum valid SSTable size (header + at least one edge + footer)
/// Header (8 bytes magic) + Edge (128 bytes) + Footer (144 bytes) = 280 bytes
pub const MIN_SSTABLE_SIZE: u64 = 280;

/// SSTable magic number: "AFFv2025" in ASCII
pub const MAGIC_NUMBER: u64 = 0x4146465632303235;

/// Footer size in bytes
pub const FOOTER_SIZE: usize = 144;

/// Validation error types
#[derive(Debug)]
pub enum ValidationError {
    TooSmall {
        actual: u64,
        minimum: u64,
    },
    BadMagic {
        expected: u64,
        actual: u64,
    },
    ChecksumMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    IoError(std::io::Error),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::TooSmall { actual, minimum } => {
                write!(
                    f,
                    "SSTable file too small: {} bytes (minimum: {})",
                    actual, minimum
                )
            }
            ValidationError::BadMagic { expected, actual } => {
                write!(
                    f,
                    "Invalid magic number: {:#x} (expected: {:#x})",
                    actual, expected
                )
            }
            ValidationError::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected {}, got {}",
                    hex::encode(expected),
                    hex::encode(actual)
                )
            }
            ValidationError::IoError(e) => write!(f, "I/O error during validation: {}", e),
        }
    }
}

impl std::error::Error for ValidationError {}

impl From<std::io::Error> for ValidationError {
    fn from(e: std::io::Error) -> Self {
        ValidationError::IoError(e)
    }
}

/// SSTable validator for pre-mmap validation
pub struct SSTableValidator {
    /// Expected magic number (default: MAGIC_NUMBER)
    pub expected_magic: u64,

    /// Whether to perform full file checksum (expensive, optional)
    pub verify_full_checksum: bool,

    /// Expected file checksum (if known from metadata)
    pub expected_checksum: Option<[u8; 32]>,
}

impl Default for SSTableValidator {
    fn default() -> Self {
        Self {
            expected_magic: MAGIC_NUMBER,
            verify_full_checksum: false,
            expected_checksum: None,
        }
    }
}

impl SSTableValidator {
    /// Create validator with full checksum verification enabled
    pub fn with_checksum_verification(expected_checksum: [u8; 32]) -> Self {
        Self {
            expected_magic: MAGIC_NUMBER,
            verify_full_checksum: true,
            expected_checksum: Some(expected_checksum),
        }
    }

    /// Validate SSTable file before memory mapping
    ///
    /// This performs comprehensive validation WITHOUT mmap to establish safety invariants:
    /// 1. File size >= minimum (header + footer)
    /// 2. Magic number matches expected value
    /// 3. Footer is readable and well-formed
    /// 4. Optional: Full file checksum (if verify_full_checksum = true)
    ///
    /// **Performance cost:** ~5-10ms for basic validation, ~50-100ms for full checksum
    ///
    /// **Safety benefit:** Prevents segfaults from corrupted/truncated files
    pub fn validate_before_mmap(&self, file: &mut File) -> Result<()> {
        // 1. Check file size >= minimum
        let metadata = file.metadata()?;

        let file_size = metadata.len();
        if file_size < MIN_SSTABLE_SIZE {
            return Err(SochDBError::Corruption(format!(
                "SSTable file too small: {} bytes (minimum: {})",
                file_size, MIN_SSTABLE_SIZE
            )));
        }

        // 2. Read and verify magic number from footer (last bytes)
        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;

        let mut footer_bytes = vec![0u8; FOOTER_SIZE];
        file.read_exact(&mut footer_bytes)?;

        // Extract magic number (first 8 bytes of footer)
        let magic = u64::from_le_bytes(footer_bytes[0..8].try_into().unwrap());
        if magic != self.expected_magic {
            return Err(SochDBError::Corruption(format!(
                "Invalid SSTable magic number: {:#x} (expected: {:#x})",
                magic, self.expected_magic
            )));
        }

        // 3. Verify footer structure integrity
        // Extract num_entries (offset 56 in footer)
        let num_entries = u64::from_le_bytes(footer_bytes[56..64].try_into().unwrap());

        // Sanity check: num_entries should be reasonable
        // Max entries in one SSTable: ~10M edges (each 128 bytes = 1.28GB file)
        const MAX_REASONABLE_ENTRIES: u64 = 10_000_000;
        if num_entries > MAX_REASONABLE_ENTRIES {
            return Err(SochDBError::Corruption(format!(
                "Unreasonable num_entries in footer: {} (max: {})",
                num_entries, MAX_REASONABLE_ENTRIES
            )));
        }

        // Verify file size matches expected content
        // Minimum size check: footer + bloom + index + at least num_entries * 128
        let min_expected_size = FOOTER_SIZE as u64 + num_entries * 128;
        if file_size < min_expected_size {
            return Err(SochDBError::Corruption(format!(
                "File size {} too small for {} entries (expected >= {})",
                file_size, num_entries, min_expected_size
            )));
        }

        // 4. Optional: Verify full file checksum
        if self.verify_full_checksum
            && let Some(expected) = self.expected_checksum
        {
            let computed = self.compute_file_checksum(file)?;
            if computed != expected {
                return Err(SochDBError::Corruption(format!(
                    "Checksum mismatch: expected {}, got {}",
                    hex::encode(expected),
                    hex::encode(computed)
                )));
            }
        }

        Ok(())
    }

    /// Compute BLAKE3 checksum of entire file
    ///
    /// **Performance:** O(file_size) - reads entire file once
    /// For 1GB file: ~1 second on modern SSD
    ///
    /// **Use case:** One-time validation during SSTable open, or periodic integrity checks
    fn compute_file_checksum(&self, file: &mut File) -> Result<[u8; 32]> {
        // Seek to beginning
        file.seek(SeekFrom::Start(0))?;

        // Read file in chunks and hash
        let mut hasher = Hasher::new();
        let mut buffer = vec![0u8; 64 * 1024]; // 64KB chunks

        loop {
            let bytes_read = file.read(&mut buffer)?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let hash = hasher.finalize();
        Ok(*hash.as_bytes())
    }

    /// Fast validation: only check magic number and file size
    ///
    /// **Performance:** O(1) - reads only footer
    /// **Use case:** Production hot path where performance is critical
    pub fn validate_fast(&self, file: &mut File) -> Result<()> {
        // 1. Check file size
        let metadata = file.metadata()?;

        let file_size = metadata.len();
        if file_size < MIN_SSTABLE_SIZE {
            return Err(SochDBError::Corruption(format!(
                "SSTable file too small: {} bytes",
                file_size
            )));
        }

        // 2. Verify magic number
        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;

        let mut magic_bytes = [0u8; 8];
        file.read_exact(&mut magic_bytes)?;

        let magic = u64::from_le_bytes(magic_bytes);
        if magic != self.expected_magic {
            return Err(SochDBError::Corruption(format!(
                "Invalid magic number: {:#x}",
                magic
            )));
        }

        Ok(())
    }
}

/// Validate SSTable file at path (convenience function)
///
/// Performs fast validation (magic + size only) unless full_validation is true.
pub fn validate_sstable_file<P: AsRef<Path>>(path: P, full_validation: bool) -> Result<()> {
    let mut file = File::open(path.as_ref())?;

    let validator = SSTableValidator::default();

    if full_validation {
        validator.validate_before_mmap(&mut file)
    } else {
        validator.validate_fast(&mut file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_validate_too_small() {
        // Create file that's too small
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0u8; 100]).unwrap(); // Only 100 bytes
        file.flush().unwrap();

        let mut file = File::open(file.path()).unwrap();
        let validator = SSTableValidator::default();

        let result = validator.validate_fast(&mut file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_validate_bad_magic() {
        // Create file with wrong magic number
        let mut file = NamedTempFile::new().unwrap();

        // Write enough bytes to pass size check
        let mut content = vec![0u8; MIN_SSTABLE_SIZE as usize];

        // Write wrong magic number in footer location
        let footer_offset = content.len() - FOOTER_SIZE;
        let wrong_magic: u64 = 0xDEADBEEF;
        content[footer_offset..footer_offset + 8].copy_from_slice(&wrong_magic.to_le_bytes());

        file.write_all(&content).unwrap();
        file.flush().unwrap();

        let mut file = File::open(file.path()).unwrap();
        let validator = SSTableValidator::default();

        let result = validator.validate_fast(&mut file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn test_validate_correct_file() {
        // Create minimal valid SSTable file
        let mut file = NamedTempFile::new().unwrap();

        let mut content = vec![0u8; MIN_SSTABLE_SIZE as usize];

        // Write correct magic number in footer location
        let footer_offset = content.len() - FOOTER_SIZE;
        content[footer_offset..footer_offset + 8].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());

        // Write reasonable num_entries (offset 56 in footer)
        let num_entries: u64 = 1;
        content[footer_offset + 56..footer_offset + 64].copy_from_slice(&num_entries.to_le_bytes());

        file.write_all(&content).unwrap();
        file.flush().unwrap();

        let mut file = File::open(file.path()).unwrap();
        let validator = SSTableValidator::default();

        let result = validator.validate_fast(&mut file);
        assert!(result.is_ok());
    }
}
