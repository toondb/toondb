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

//! Forward-Compatible SSTable Container Format
//!
//! This module defines the on-disk format for SSTables with support for
//! future extensibility and safe mmap operations.
//!
//! ## Design Goals
//!
//! 1. **Forward compatibility**: New sections can be added without breaking readers
//! 2. **Backward compatibility**: Old readers skip unknown sections
//! 3. **Safe mmap**: All data validated before memory mapping
//! 4. **Efficient access**: Direct offset-based section lookup
//!
//! ## File Format
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         SSTable File Layout                              │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Header (32 bytes):                                                       │
//! │   Magic (8 bytes): "TDBSSTab"                                           │
//! │   Version (4 bytes): Format version                                      │
//! │   Flags (4 bytes): Feature flags                                         │
//! │   Num Sections (4 bytes): Number of sections                            │
//! │   Footer Offset (8 bytes): Offset to footer                              │
//! │   Header Checksum (4 bytes): CRC32 of header                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Section 0: Data Blocks                                                   │
//! │   [Block 0][Block 1]...[Block N]                                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Section 1: Filter (optional)                                            │
//! │   [Filter Data]                                                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Section 2: Index                                                         │
//! │   [Index Block]                                                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Section 3: Metadata (optional)                                           │
//! │   [Properties, Stats, etc.]                                              │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ Footer (variable):                                                       │
//! │   Section Directory: [Type, Offset, Size, Checksum] × N                 │
//! │   Footer Checksum (4 bytes)                                              │
//! │   Magic (8 bytes): "TDBSSTab"                                           │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};

/// SSTable magic number: "TDBSSTab" in ASCII
pub const TABLE_MAGIC: [u8; 8] = [0x54, 0x44, 0x42, 0x53, 0x53, 0x54, 0x61, 0x62];

/// Current format version
pub const FORMAT_VERSION: u32 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 32;

/// Section entry size in footer (type + offset + size + checksum)
pub const SECTION_ENTRY_SIZE: usize = 24;

/// Table magic newtype for type safety
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableMagic([u8; 8]);

impl TableMagic {
    pub fn new() -> Self {
        Self(TABLE_MAGIC)
    }

    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    pub fn is_valid(&self) -> bool {
        self.0 == TABLE_MAGIC
    }
}

impl Default for TableMagic {
    fn default() -> Self {
        Self::new()
    }
}

/// Section types in an SSTable
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionType {
    /// Data blocks containing key-value pairs
    DataBlocks = 0,
    /// Bloom/Ribbon/Xor filter for the table
    Filter = 1,
    /// Index block for data block lookup
    Index = 2,
    /// Metadata (properties, stats, etc.)
    Metadata = 3,
    /// Range tombstones
    RangeTombstones = 4,
    /// Compression dictionary
    CompressionDict = 5,
    /// Reserved for future use
    Reserved = 0xFFFFFFFF,
}

impl TryFrom<u32> for SectionType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SectionType::DataBlocks),
            1 => Ok(SectionType::Filter),
            2 => Ok(SectionType::Index),
            3 => Ok(SectionType::Metadata),
            4 => Ok(SectionType::RangeTombstones),
            5 => Ok(SectionType::CompressionDict),
            _ => Err(()),
        }
    }
}

/// A section in the SSTable
#[derive(Debug, Clone)]
pub struct Section {
    /// Section type
    pub section_type: SectionType,
    /// Offset in file
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// CRC32 checksum of section data
    pub checksum: u32,
}

impl Section {
    pub fn new(section_type: SectionType, offset: u64, size: u64, checksum: u32) -> Self {
        Self {
            section_type,
            offset,
            size,
            checksum,
        }
    }

    /// Encode to bytes
    pub fn encode<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.section_type as u32)?;
        writer.write_u64::<LittleEndian>(self.offset)?;
        writer.write_u64::<LittleEndian>(self.size)?;
        writer.write_u32::<LittleEndian>(self.checksum)?;
        Ok(())
    }

    /// Decode from bytes
    pub fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let type_val = reader.read_u32::<LittleEndian>()?;
        let section_type = SectionType::try_from(type_val).unwrap_or(SectionType::Reserved);
        let offset = reader.read_u64::<LittleEndian>()?;
        let size = reader.read_u64::<LittleEndian>()?;
        let checksum = reader.read_u32::<LittleEndian>()?;
        
        Ok(Self {
            section_type,
            offset,
            size,
            checksum,
        })
    }
}

/// SSTable file header
#[derive(Debug, Clone)]
pub struct Header {
    /// Magic number
    pub magic: TableMagic,
    /// Format version
    pub version: u32,
    /// Feature flags
    pub flags: u32,
    /// Number of sections
    pub num_sections: u32,
    /// Offset to footer
    pub footer_offset: u64,
    /// Header checksum
    pub checksum: u32,
}

impl Header {
    pub fn new(num_sections: u32, footer_offset: u64) -> Self {
        let mut header = Self {
            magic: TableMagic::new(),
            version: FORMAT_VERSION,
            flags: 0,
            num_sections,
            footer_offset,
            checksum: 0,
        };
        header.checksum = header.compute_checksum();
        header
    }

    /// Encode header to bytes
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut cursor = Cursor::new(&mut buf[..]);
        
        cursor.write_all(self.magic.as_bytes()).unwrap();
        cursor.write_u32::<LittleEndian>(self.version).unwrap();
        cursor.write_u32::<LittleEndian>(self.flags).unwrap();
        cursor.write_u32::<LittleEndian>(self.num_sections).unwrap();
        cursor.write_u64::<LittleEndian>(self.footer_offset).unwrap();
        cursor.write_u32::<LittleEndian>(self.checksum).unwrap();
        
        buf
    }

    /// Decode header from bytes
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < HEADER_SIZE {
            return None;
        }
        
        let mut cursor = Cursor::new(data);
        
        let mut magic_bytes = [0u8; 8];
        cursor.read_exact(&mut magic_bytes).ok()?;
        let magic = TableMagic(magic_bytes);
        
        let version = cursor.read_u32::<LittleEndian>().ok()?;
        let flags = cursor.read_u32::<LittleEndian>().ok()?;
        let num_sections = cursor.read_u32::<LittleEndian>().ok()?;
        let footer_offset = cursor.read_u64::<LittleEndian>().ok()?;
        let checksum = cursor.read_u32::<LittleEndian>().ok()?;
        
        let header = Self {
            magic,
            version,
            flags,
            num_sections,
            footer_offset,
            checksum,
        };
        
        // Verify checksum
        if header.compute_checksum() != checksum {
            return None;
        }
        
        Some(header)
    }

    /// Compute checksum of header (excluding checksum field)
    fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(self.magic.as_bytes());
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.flags.to_le_bytes());
        hasher.update(&self.num_sections.to_le_bytes());
        hasher.update(&self.footer_offset.to_le_bytes());
        hasher.finalize()
    }

    /// Validate header
    pub fn is_valid(&self) -> bool {
        self.magic.is_valid() && 
        self.version <= FORMAT_VERSION &&
        self.compute_checksum() == self.checksum
    }
}

/// SSTable footer
#[derive(Debug, Clone)]
pub struct Footer {
    /// Section directory
    pub sections: Vec<Section>,
    /// Footer checksum
    pub checksum: u32,
    /// Magic number (repeated for validation)
    pub magic: TableMagic,
}

impl Footer {
    pub fn new(sections: Vec<Section>) -> Self {
        let mut footer = Self {
            sections,
            checksum: 0,
            magic: TableMagic::new(),
        };
        footer.checksum = footer.compute_checksum();
        footer
    }

    /// Encode footer to bytes
    pub fn encode(&self) -> Vec<u8> {
        let size = self.sections.len() * SECTION_ENTRY_SIZE + 4 + 8;
        let mut buf = Vec::with_capacity(size);
        
        for section in &self.sections {
            section.encode(&mut buf).unwrap();
        }
        
        buf.write_u32::<LittleEndian>(self.checksum).unwrap();
        buf.extend_from_slice(self.magic.as_bytes());
        
        buf
    }

    /// Decode footer from bytes
    pub fn decode(data: &[u8], num_sections: u32) -> Option<Self> {
        let expected_size = num_sections as usize * SECTION_ENTRY_SIZE + 4 + 8;
        if data.len() < expected_size {
            return None;
        }
        
        let mut cursor = Cursor::new(data);
        
        let mut sections = Vec::with_capacity(num_sections as usize);
        for _ in 0..num_sections {
            sections.push(Section::decode(&mut cursor).ok()?);
        }
        
        let checksum = cursor.read_u32::<LittleEndian>().ok()?;
        
        let mut magic_bytes = [0u8; 8];
        cursor.read_exact(&mut magic_bytes).ok()?;
        let magic = TableMagic(magic_bytes);
        
        let footer = Self {
            sections,
            checksum,
            magic,
        };
        
        // Verify checksum
        if footer.compute_checksum() != checksum {
            return None;
        }
        
        Some(footer)
    }

    /// Compute checksum of footer (excluding checksum and magic)
    fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        for section in &self.sections {
            hasher.update(&(section.section_type as u32).to_le_bytes());
            hasher.update(&section.offset.to_le_bytes());
            hasher.update(&section.size.to_le_bytes());
            hasher.update(&section.checksum.to_le_bytes());
        }
        hasher.finalize()
    }

    /// Get section by type
    pub fn get_section(&self, section_type: SectionType) -> Option<&Section> {
        self.sections.iter().find(|s| s.section_type == section_type)
    }

    /// Check if section exists
    pub fn has_section(&self, section_type: SectionType) -> bool {
        self.get_section(section_type).is_some()
    }
}

/// SSTable format reader/writer
pub struct SSTableFormat {
    pub header: Header,
    pub footer: Footer,
}

impl SSTableFormat {
    /// Create a new format with given sections
    pub fn new(sections: Vec<Section>) -> Self {
        let footer_offset = sections.iter().map(|s| s.offset + s.size).max().unwrap_or(HEADER_SIZE as u64);
        
        Self {
            header: Header::new(sections.len() as u32, footer_offset),
            footer: Footer::new(sections),
        }
    }

    /// Read format from file
    pub fn read<R: Read + Seek>(reader: &mut R) -> std::io::Result<Self> {
        // Read header
        let mut header_buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_buf)?;
        
        let header = Header::decode(&header_buf)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid header"))?;
        
        if !header.is_valid() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid header"));
        }
        
        // Seek to footer
        reader.seek(SeekFrom::Start(header.footer_offset))?;
        
        // Read footer
        let footer_size = header.num_sections as usize * SECTION_ENTRY_SIZE + 4 + 8;
        let mut footer_buf = vec![0u8; footer_size];
        reader.read_exact(&mut footer_buf)?;
        
        let footer = Footer::decode(&footer_buf, header.num_sections)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid footer"))?;
        
        Ok(Self { header, footer })
    }

    /// Write format to file (header and footer only)
    pub fn write<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write header at start
        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&self.header.encode())?;
        
        // Write footer at footer_offset
        writer.seek(SeekFrom::Start(self.header.footer_offset))?;
        writer.write_all(&self.footer.encode())?;
        
        Ok(())
    }

    /// Get section by type
    pub fn get_section(&self, section_type: SectionType) -> Option<&Section> {
        self.footer.get_section(section_type)
    }

    /// Validate section data against checksum
    pub fn validate_section<R: Read + Seek>(
        &self,
        reader: &mut R,
        section: &Section,
    ) -> std::io::Result<bool> {
        reader.seek(SeekFrom::Start(section.offset))?;
        
        let mut data = vec![0u8; section.size as usize];
        reader.read_exact(&mut data)?;
        
        let computed_checksum = crc32fast::hash(&data);
        Ok(computed_checksum == section.checksum)
    }

    /// Pre-validate all sections before mmap
    ///
    /// This establishes the safety invariant that all mapped pages are valid.
    pub fn validate_all_sections<R: Read + Seek>(&self, reader: &mut R) -> std::io::Result<bool> {
        for section in &self.footer.sections {
            if !self.validate_section(reader, section)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_table_magic() {
        let magic = TableMagic::new();
        assert!(magic.is_valid());
        assert_eq!(magic.as_bytes(), &TABLE_MAGIC);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = Header::new(3, 1024);
        let encoded = header.encode();
        
        let decoded = Header::decode(&encoded).unwrap();
        assert_eq!(decoded.version, FORMAT_VERSION);
        assert_eq!(decoded.num_sections, 3);
        assert_eq!(decoded.footer_offset, 1024);
        assert!(decoded.is_valid());
    }

    #[test]
    fn test_section_roundtrip() {
        let section = Section::new(SectionType::DataBlocks, 100, 500, 12345);
        
        let mut buf = Vec::new();
        section.encode(&mut buf).unwrap();
        
        let decoded = Section::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.section_type, SectionType::DataBlocks);
        assert_eq!(decoded.offset, 100);
        assert_eq!(decoded.size, 500);
        assert_eq!(decoded.checksum, 12345);
    }

    #[test]
    fn test_footer_roundtrip() {
        let sections = vec![
            Section::new(SectionType::DataBlocks, 32, 1000, 111),
            Section::new(SectionType::Filter, 1032, 200, 222),
            Section::new(SectionType::Index, 1232, 100, 333),
        ];
        
        let footer = Footer::new(sections);
        let encoded = footer.encode();
        
        let decoded = Footer::decode(&encoded, 3).unwrap();
        assert_eq!(decoded.sections.len(), 3);
        assert!(decoded.magic.is_valid());
    }

    #[test]
    fn test_format_roundtrip() {
        let sections = vec![
            Section::new(SectionType::DataBlocks, 32, 1000, 111),
            Section::new(SectionType::Index, 1032, 100, 222),
        ];
        
        let format = SSTableFormat::new(sections);
        
        let mut buf = vec![0u8; 2048];
        let mut cursor = Cursor::new(&mut buf[..]);
        format.write(&mut cursor).unwrap();
        
        let mut cursor = Cursor::new(&buf[..]);
        let read_format = SSTableFormat::read(&mut cursor).unwrap();
        
        assert_eq!(read_format.header.num_sections, 2);
        assert!(read_format.get_section(SectionType::DataBlocks).is_some());
        assert!(read_format.get_section(SectionType::Index).is_some());
        assert!(read_format.get_section(SectionType::Filter).is_none());
    }
}
