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

//! Block Format Migration - Backwards Compatible Format Versioning
//!
//! This module provides format versioning for SochDB's block storage,
//! enabling backwards-compatible upgrades without data loss.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Block Format Versions                     │
//! │                                                             │
//! │  v1: Original format (17-byte header + data)                │
//! │  v2: Extended header (21-byte: +format version, +flags)     │
//! │  v3: Future format with additional metadata                 │
//! │                                                             │
//! │  Detection Strategy:                                        │
//! │  - v1: magic="TBLK", byte[17] = data start                  │
//! │  - v2: magic="TBL2", byte[5] = format version               │
//! │  - All versions share first 4 bytes for magic detection     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Format Detection
//!
//! The reader auto-detects format version by inspecting magic bytes:
//! - `TBLK` = Format v1 (original 17-byte header)
//! - `TBL2` = Format v2 (extended 21-byte header)
//!
//! # Migration Path
//!
//! Blocks are migrated lazily on read:
//! 1. Read block with auto-detection
//! 2. If format < current, upgrade in memory
//! 3. Optionally rewrite upgraded block during compaction

use byteorder::{ByteOrder, LittleEndian};

use crate::block_storage::BlockCompression;
use crate::{Result, SochDBError};

/// Block format version
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum FormatVersion {
    /// Original format (17-byte header)
    V1 = 1,
    /// Extended format (21-byte header with flags)
    V2 = 2,
}

impl FormatVersion {
    /// Current format version used for new blocks
    pub const CURRENT: FormatVersion = FormatVersion::V2;

    /// Parse from byte
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(FormatVersion::V1),
            2 => Some(FormatVersion::V2),
            _ => None,
        }
    }

    /// Get header size for this version
    pub fn header_size(&self) -> usize {
        match self {
            FormatVersion::V1 => V1_HEADER_SIZE,
            FormatVersion::V2 => V2_HEADER_SIZE,
        }
    }
}

// Magic bytes for different versions
const V1_MAGIC: [u8; 4] = *b"TBLK";
const V2_MAGIC: [u8; 4] = *b"TBL2";

// Header sizes
const V1_HEADER_SIZE: usize = 17;
const V2_HEADER_SIZE: usize = 21;

/// Block flags for v2 format
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockFlags {
    /// Block contains encrypted data
    pub encrypted: bool,
    /// Block has extended checksums (SHA-256 trailer)
    pub extended_checksum: bool,
    /// Block is part of a multi-block span
    pub spanning: bool,
    /// Block metadata follows data
    pub has_metadata: bool,
}

impl BlockFlags {
    /// Pack flags into single byte
    pub fn to_byte(&self) -> u8 {
        let mut b = 0u8;
        if self.encrypted {
            b |= 0x01;
        }
        if self.extended_checksum {
            b |= 0x02;
        }
        if self.spanning {
            b |= 0x04;
        }
        if self.has_metadata {
            b |= 0x08;
        }
        b
    }

    /// Unpack flags from byte
    pub fn from_byte(b: u8) -> Self {
        Self {
            encrypted: (b & 0x01) != 0,
            extended_checksum: (b & 0x02) != 0,
            spanning: (b & 0x04) != 0,
            has_metadata: (b & 0x08) != 0,
        }
    }
}

/// V1 block header (original format, 17 bytes)
///
/// Layout:
/// - bytes 0-3: magic "TBLK"
/// - byte 4: compression type
/// - bytes 5-8: original size (u32 LE)
/// - bytes 9-12: compressed size (u32 LE)
/// - bytes 13-16: CRC32 checksum (u32 LE)
#[derive(Debug, Clone)]
pub struct V1Header {
    pub compression: BlockCompression,
    pub original_size: u32,
    pub compressed_size: u32,
    pub checksum: u32,
}

impl V1Header {
    /// Parse from bytes
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < V1_HEADER_SIZE {
            return Err(SochDBError::InvalidData(format!(
                "V1 header too short: {} < {}",
                buf.len(),
                V1_HEADER_SIZE
            )));
        }

        if &buf[0..4] != V1_MAGIC.as_slice() {
            return Err(SochDBError::InvalidData(format!(
                "Invalid V1 magic: {:?}",
                &buf[0..4]
            )));
        }

        Ok(Self {
            compression: BlockCompression::from_byte(buf[4]),
            original_size: LittleEndian::read_u32(&buf[5..9]),
            compressed_size: LittleEndian::read_u32(&buf[9..13]),
            checksum: LittleEndian::read_u32(&buf[13..17]),
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; V1_HEADER_SIZE] {
        let mut buf = [0u8; V1_HEADER_SIZE];
        buf[0..4].copy_from_slice(&V1_MAGIC);
        buf[4] = self.compression.to_byte();
        LittleEndian::write_u32(&mut buf[5..9], self.original_size);
        LittleEndian::write_u32(&mut buf[9..13], self.compressed_size);
        LittleEndian::write_u32(&mut buf[13..17], self.checksum);
        buf
    }

    /// Upgrade to V2
    pub fn upgrade_to_v2(&self) -> V2Header {
        V2Header {
            format_version: FormatVersion::V2,
            compression: self.compression,
            flags: BlockFlags::default(),
            original_size: self.original_size,
            compressed_size: self.compressed_size,
            checksum: self.checksum,
        }
    }
}

/// V2 block header (extended format, 21 bytes)
///
/// Layout:
/// - bytes 0-3: magic "TBL2"
/// - byte 4: format version
/// - byte 5: compression type
/// - byte 6: flags
/// - bytes 7-10: original size (u32 LE)
/// - bytes 11-14: compressed size (u32 LE)
/// - bytes 15-18: CRC32 checksum (u32 LE)
/// - bytes 19-20: reserved (u16 LE)
#[derive(Debug, Clone)]
pub struct V2Header {
    pub format_version: FormatVersion,
    pub compression: BlockCompression,
    pub flags: BlockFlags,
    pub original_size: u32,
    pub compressed_size: u32,
    pub checksum: u32,
}

impl V2Header {
    /// Parse from bytes
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < V2_HEADER_SIZE {
            return Err(SochDBError::InvalidData(format!(
                "V2 header too short: {} < {}",
                buf.len(),
                V2_HEADER_SIZE
            )));
        }

        if &buf[0..4] != V2_MAGIC.as_slice() {
            return Err(SochDBError::InvalidData(format!(
                "Invalid V2 magic: {:?}",
                &buf[0..4]
            )));
        }

        let format_version = FormatVersion::from_byte(buf[4]).ok_or_else(|| {
            SochDBError::InvalidData(format!("Unknown format version: {}", buf[4]))
        })?;

        Ok(Self {
            format_version,
            compression: BlockCompression::from_byte(buf[5]),
            flags: BlockFlags::from_byte(buf[6]),
            original_size: LittleEndian::read_u32(&buf[7..11]),
            compressed_size: LittleEndian::read_u32(&buf[11..15]),
            checksum: LittleEndian::read_u32(&buf[15..19]),
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; V2_HEADER_SIZE] {
        let mut buf = [0u8; V2_HEADER_SIZE];
        buf[0..4].copy_from_slice(&V2_MAGIC);
        buf[4] = self.format_version as u8;
        buf[5] = self.compression.to_byte();
        buf[6] = self.flags.to_byte();
        LittleEndian::write_u32(&mut buf[7..11], self.original_size);
        LittleEndian::write_u32(&mut buf[11..15], self.compressed_size);
        LittleEndian::write_u32(&mut buf[15..19], self.checksum);
        // bytes 19-20 reserved
        buf
    }

    /// Downgrade to V1 (loses flags and extended info)
    pub fn downgrade_to_v1(&self) -> V1Header {
        V1Header {
            compression: self.compression,
            original_size: self.original_size,
            compressed_size: self.compressed_size,
            checksum: self.checksum,
        }
    }
}

/// Version-agnostic block header
#[derive(Debug, Clone)]
pub enum BlockHeader {
    V1(V1Header),
    V2(V2Header),
}

impl BlockHeader {
    /// Detect version and parse header from bytes
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 4 {
            return Err(SochDBError::InvalidData(
                "Buffer too short for magic detection".to_string(),
            ));
        }

        let magic = &buf[0..4];

        if magic == V1_MAGIC.as_slice() {
            Ok(BlockHeader::V1(V1Header::from_bytes(buf)?))
        } else if magic == V2_MAGIC.as_slice() {
            Ok(BlockHeader::V2(V2Header::from_bytes(buf)?))
        } else {
            Err(SochDBError::InvalidData(format!(
                "Unknown block magic: {:?}",
                magic
            )))
        }
    }

    /// Get the format version
    pub fn version(&self) -> FormatVersion {
        match self {
            BlockHeader::V1(_) => FormatVersion::V1,
            BlockHeader::V2(_) => FormatVersion::V2,
        }
    }

    /// Get header size
    pub fn header_size(&self) -> usize {
        self.version().header_size()
    }

    /// Get compression type
    pub fn compression(&self) -> BlockCompression {
        match self {
            BlockHeader::V1(h) => h.compression,
            BlockHeader::V2(h) => h.compression,
        }
    }

    /// Get original (uncompressed) size
    pub fn original_size(&self) -> u32 {
        match self {
            BlockHeader::V1(h) => h.original_size,
            BlockHeader::V2(h) => h.original_size,
        }
    }

    /// Get compressed size
    pub fn compressed_size(&self) -> u32 {
        match self {
            BlockHeader::V1(h) => h.compressed_size,
            BlockHeader::V2(h) => h.compressed_size,
        }
    }

    /// Get CRC32 checksum
    pub fn checksum(&self) -> u32 {
        match self {
            BlockHeader::V1(h) => h.checksum,
            BlockHeader::V2(h) => h.checksum,
        }
    }

    /// Get flags (V2 only, returns default for V1)
    pub fn flags(&self) -> BlockFlags {
        match self {
            BlockHeader::V1(_) => BlockFlags::default(),
            BlockHeader::V2(h) => h.flags,
        }
    }

    /// Upgrade to current format version
    pub fn upgrade(&self) -> Self {
        match self {
            BlockHeader::V1(h) => BlockHeader::V2(h.upgrade_to_v2()),
            BlockHeader::V2(_) => self.clone(),
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            BlockHeader::V1(h) => h.to_bytes().to_vec(),
            BlockHeader::V2(h) => h.to_bytes().to_vec(),
        }
    }

    /// Check if this is the current format version
    pub fn is_current(&self) -> bool {
        self.version() == FormatVersion::CURRENT
    }

    /// Check if block needs migration
    pub fn needs_migration(&self) -> bool {
        self.version() < FormatVersion::CURRENT
    }
}

/// Complete block with header and data
#[derive(Debug, Clone)]
pub struct MigratableBlock {
    pub header: BlockHeader,
    pub data: Vec<u8>,
}

impl MigratableBlock {
    /// Create new block with current format
    pub fn new(
        data: Vec<u8>,
        compression: BlockCompression,
        original_size: u32,
        compressed_size: u32,
        checksum: u32,
    ) -> Self {
        Self {
            header: BlockHeader::V2(V2Header {
                format_version: FormatVersion::CURRENT,
                compression,
                flags: BlockFlags::default(),
                original_size,
                compressed_size,
                checksum,
            }),
            data,
        }
    }

    /// Create with flags
    pub fn with_flags(mut self, flags: BlockFlags) -> Self {
        if let BlockHeader::V2(ref mut h) = self.header {
            h.flags = flags;
        }
        self
    }

    /// Read block from bytes (auto-detects version)
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let header = BlockHeader::from_bytes(buf)?;
        let header_size = header.header_size();
        let data_size = header.compressed_size() as usize;

        if buf.len() < header_size + data_size {
            return Err(SochDBError::InvalidData(format!(
                "Block buffer too short: {} < {}",
                buf.len(),
                header_size + data_size
            )));
        }

        Ok(Self {
            header,
            data: buf[header_size..header_size + data_size].to_vec(),
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_bytes = self.header.to_bytes();
        let mut result = Vec::with_capacity(header_bytes.len() + self.data.len());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&self.data);
        result
    }

    /// Migrate to current format version
    pub fn migrate(&mut self) {
        self.header = self.header.upgrade();
    }

    /// Check if migration needed
    pub fn needs_migration(&self) -> bool {
        self.header.needs_migration()
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> Result<()> {
        let computed = crc32fast::hash(&self.data);
        let stored = self.header.checksum();

        if computed != stored {
            return Err(SochDBError::DataCorruption {
                details: format!(
                    "Checksum mismatch: computed {} != stored {}",
                    computed, stored
                ),
                location: "block data".to_string(),
                hint: "Block may be corrupted, try restoring from backup".to_string(),
            });
        }

        Ok(())
    }
}

/// Block format migration statistics
#[derive(Debug, Default)]
pub struct MigrationStats {
    pub blocks_read: u64,
    pub blocks_migrated: u64,
    pub v1_blocks_found: u64,
    pub v2_blocks_found: u64,
    pub checksum_failures: u64,
}

impl MigrationStats {
    pub fn record_read(&mut self, version: FormatVersion) {
        self.blocks_read += 1;
        match version {
            FormatVersion::V1 => self.v1_blocks_found += 1,
            FormatVersion::V2 => self.v2_blocks_found += 1,
        }
    }

    pub fn record_migration(&mut self) {
        self.blocks_migrated += 1;
    }

    pub fn record_checksum_failure(&mut self) {
        self.checksum_failures += 1;
    }

    /// Migration progress percentage
    pub fn migration_progress(&self) -> f64 {
        if self.v1_blocks_found == 0 {
            100.0
        } else {
            (self.blocks_migrated as f64 / self.v1_blocks_found as f64) * 100.0
        }
    }
}

/// Block format migrator for batch migrations
pub struct FormatMigrator {
    stats: MigrationStats,
    verify_checksums: bool,
}

impl FormatMigrator {
    pub fn new() -> Self {
        Self {
            stats: MigrationStats::default(),
            verify_checksums: true,
        }
    }

    /// Set whether to verify checksums during migration
    pub fn with_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_checksums = verify;
        self
    }

    /// Migrate a single block
    pub fn migrate_block(&mut self, block: &mut MigratableBlock) -> Result<bool> {
        self.stats.record_read(block.header.version());

        if self.verify_checksums
            && let Err(e) = block.verify_checksum()
        {
            self.stats.record_checksum_failure();
            return Err(e);
        }

        if block.needs_migration() {
            block.migrate();
            self.stats.record_migration();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Migrate multiple blocks
    pub fn migrate_blocks(&mut self, blocks: &mut [MigratableBlock]) -> Result<usize> {
        let mut migrated = 0;
        for block in blocks {
            if self.migrate_block(block)? {
                migrated += 1;
            }
        }
        Ok(migrated)
    }

    /// Get migration statistics
    pub fn stats(&self) -> &MigrationStats {
        &self.stats
    }
}

impl Default for FormatMigrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_header_roundtrip() {
        let header = V1Header {
            compression: BlockCompression::None,
            original_size: 1024,
            compressed_size: 1024,
            checksum: 0xDEADBEEF,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), V1_HEADER_SIZE);

        let parsed = V1Header::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.compression, BlockCompression::None);
        assert_eq!(parsed.original_size, 1024);
        assert_eq!(parsed.compressed_size, 1024);
        assert_eq!(parsed.checksum, 0xDEADBEEF);
    }

    #[test]
    fn test_v2_header_roundtrip() {
        let header = V2Header {
            format_version: FormatVersion::V2,
            compression: BlockCompression::Lz4,
            flags: BlockFlags {
                encrypted: true,
                extended_checksum: false,
                spanning: true,
                has_metadata: false,
            },
            original_size: 2048,
            compressed_size: 1500,
            checksum: 0xCAFEBABE,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), V2_HEADER_SIZE);

        let parsed = V2Header::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.format_version, FormatVersion::V2);
        assert_eq!(parsed.compression, BlockCompression::Lz4);
        assert!(parsed.flags.encrypted);
        assert!(!parsed.flags.extended_checksum);
        assert!(parsed.flags.spanning);
        assert!(!parsed.flags.has_metadata);
        assert_eq!(parsed.original_size, 2048);
        assert_eq!(parsed.compressed_size, 1500);
        assert_eq!(parsed.checksum, 0xCAFEBABE);
    }

    #[test]
    fn test_version_detection() {
        // V1 block
        let v1_header = V1Header {
            compression: BlockCompression::None,
            original_size: 100,
            compressed_size: 100,
            checksum: 0x12345678,
        };
        let v1_bytes = v1_header.to_bytes();

        let detected = BlockHeader::from_bytes(&v1_bytes).unwrap();
        assert_eq!(detected.version(), FormatVersion::V1);

        // V2 block
        let v2_header = V2Header {
            format_version: FormatVersion::V2,
            compression: BlockCompression::Zstd,
            flags: BlockFlags::default(),
            original_size: 200,
            compressed_size: 150,
            checksum: 0x87654321,
        };
        let v2_bytes = v2_header.to_bytes();

        let detected = BlockHeader::from_bytes(&v2_bytes).unwrap();
        assert_eq!(detected.version(), FormatVersion::V2);
    }

    #[test]
    fn test_v1_to_v2_upgrade() {
        let v1_header = V1Header {
            compression: BlockCompression::Lz4,
            original_size: 500,
            compressed_size: 300,
            checksum: 0xABCDEF00,
        };

        let v2_header = v1_header.upgrade_to_v2();

        assert_eq!(v2_header.format_version, FormatVersion::V2);
        assert_eq!(v2_header.compression, BlockCompression::Lz4);
        assert_eq!(v2_header.original_size, 500);
        assert_eq!(v2_header.compressed_size, 300);
        assert_eq!(v2_header.checksum, 0xABCDEF00);
        // Default flags
        assert!(!v2_header.flags.encrypted);
    }

    #[test]
    fn test_block_migration() {
        // Create V1 block
        let data = b"Hello, SochDB!";
        let checksum = crc32fast::hash(data);

        let v1_header = V1Header {
            compression: BlockCompression::None,
            original_size: data.len() as u32,
            compressed_size: data.len() as u32,
            checksum,
        };

        let mut buf = v1_header.to_bytes().to_vec();
        buf.extend_from_slice(data);

        // Parse and migrate
        let mut block = MigratableBlock::from_bytes(&buf).unwrap();
        assert!(block.needs_migration());
        assert_eq!(block.header.version(), FormatVersion::V1);

        block.migrate();
        assert!(!block.needs_migration());
        assert_eq!(block.header.version(), FormatVersion::V2);

        // Verify data preserved
        assert_eq!(block.data, data);
        block.verify_checksum().unwrap();
    }

    #[test]
    fn test_format_migrator() {
        let data1 = b"Block one data";
        let data2 = b"Block two data";

        // Create V1 blocks
        let mut blocks: Vec<MigratableBlock> = vec![
            MigratableBlock {
                header: BlockHeader::V1(V1Header {
                    compression: BlockCompression::None,
                    original_size: data1.len() as u32,
                    compressed_size: data1.len() as u32,
                    checksum: crc32fast::hash(data1),
                }),
                data: data1.to_vec(),
            },
            MigratableBlock {
                header: BlockHeader::V1(V1Header {
                    compression: BlockCompression::None,
                    original_size: data2.len() as u32,
                    compressed_size: data2.len() as u32,
                    checksum: crc32fast::hash(data2),
                }),
                data: data2.to_vec(),
            },
        ];

        let mut migrator = FormatMigrator::new();
        let migrated = migrator.migrate_blocks(&mut blocks).unwrap();

        assert_eq!(migrated, 2);
        assert_eq!(migrator.stats().blocks_read, 2);
        assert_eq!(migrator.stats().blocks_migrated, 2);
        assert_eq!(migrator.stats().v1_blocks_found, 2);

        for block in &blocks {
            assert_eq!(block.header.version(), FormatVersion::V2);
        }
    }

    #[test]
    fn test_checksum_verification_failure() {
        let data = b"Test data";
        let block = MigratableBlock {
            header: BlockHeader::V1(V1Header {
                compression: BlockCompression::None,
                original_size: data.len() as u32,
                compressed_size: data.len() as u32,
                checksum: 0xBADBAD, // Wrong checksum
            }),
            data: data.to_vec(),
        };

        let result = block.verify_checksum();
        assert!(result.is_err());
    }

    #[test]
    fn test_block_flags() {
        let flags = BlockFlags {
            encrypted: true,
            extended_checksum: true,
            spanning: false,
            has_metadata: true,
        };

        let byte = flags.to_byte();
        let parsed = BlockFlags::from_byte(byte);

        assert!(parsed.encrypted);
        assert!(parsed.extended_checksum);
        assert!(!parsed.spanning);
        assert!(parsed.has_metadata);
    }

    #[test]
    fn test_migration_progress() {
        let mut stats = MigrationStats::default();

        // No V1 blocks = 100% done
        assert_eq!(stats.migration_progress(), 100.0);

        // Some V1 blocks
        stats.v1_blocks_found = 10;
        stats.blocks_migrated = 5;
        assert_eq!(stats.migration_progress(), 50.0);

        stats.blocks_migrated = 10;
        assert_eq!(stats.migration_progress(), 100.0);
    }

    #[test]
    fn test_block_complete_roundtrip() {
        let data = b"Complete block test with some data";
        let checksum = crc32fast::hash(data);

        let block = MigratableBlock::new(
            data.to_vec(),
            BlockCompression::None,
            data.len() as u32,
            data.len() as u32,
            checksum,
        );

        // Serialize
        let bytes = block.to_bytes();

        // Deserialize
        let parsed = MigratableBlock::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.header.version(), FormatVersion::V2);
        assert_eq!(parsed.data, data);
        parsed.verify_checksum().unwrap();
    }

    #[test]
    fn test_block_with_flags() {
        let data = b"Encrypted data";
        let checksum = crc32fast::hash(data);

        let block = MigratableBlock::new(
            data.to_vec(),
            BlockCompression::Zstd,
            data.len() as u32,
            data.len() as u32,
            checksum,
        )
        .with_flags(BlockFlags {
            encrypted: true,
            extended_checksum: false,
            spanning: false,
            has_metadata: true,
        });

        let bytes = block.to_bytes();
        let parsed = MigratableBlock::from_bytes(&bytes).unwrap();

        assert!(parsed.header.flags().encrypted);
        assert!(parsed.header.flags().has_metadata);
        assert!(!parsed.header.flags().spanning);
    }

    #[test]
    fn test_unknown_magic_error() {
        let bad_magic = b"XXXX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = BlockHeader::from_bytes(bad_magic);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_too_short_error() {
        let short_buf = b"TBL";
        let result = BlockHeader::from_bytes(short_buf);
        assert!(result.is_err());
    }
}
