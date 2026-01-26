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

//! TOON Binary Protocol (TBP) - Zero-Copy Binary Wire Format
//!
//! From mm.md Task 3.1: Zero-Copy Binary Wire Format
//!
//! ## Problem
//!
//! Current TOON format is text-based with parsing overhead:
//! - O(n) string allocations per row
//! - UTF-8 validation on every parse
//! - No random access (must scan from start)
//! - Variable-length encoding requires sequential parsing
//!
//! ## Solution
//!
//! Binary protocol enables:
//! - O(1) field access via row index + column offset
//! - Zero-copy reads from mmap'd files
//! - Null bitmap for efficient NULL handling
//! - LLM-friendly text emission on demand
//!
//! ## Layout
//!
//! ```text
//! TBP Layout (Little-Endian, 32-byte header):
//! ┌─────────────────────────────────────────────────────┐
//! │ magic: u32 = 0x544F4F4E ("TOON")                    │
//! │ version: u16, flags: u16                            │
//! │ schema_id: u64 (hash for validation)                │
//! │ row_count: u32, column_count: u16                   │
//! │ null_bitmap_offset: u32, row_index_offset: u32      │
//! │ data_offset: u32                                    │
//! ├─────────────────────────────────────────────────────┤
//! │ Null Bitmap: ceil(rows × cols / 8) bytes            │
//! │ Row Index: [u32; row_count] offsets                 │
//! │ Data Section (columnar within blocks)               │
//! └─────────────────────────────────────────────────────┘
//!
//! Access complexity:
//! - Row access: O(1) via row_index[row]
//! - Field access: O(1) via column_type + fixed_offset
//! - Null check: O(1) via bitmap[row * cols + col]
//! ```

use std::io::{self, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// TBP magic number: "TOON" in ASCII
pub const TBP_MAGIC: u32 = 0x544F_4F4E;

/// Current TBP version
pub const TBP_VERSION: u16 = 1;

/// TBP header size in bytes
pub const TBP_HEADER_SIZE: usize = 32;

/// TBP flags
#[derive(Debug, Clone, Copy, Default)]
pub struct TbpFlags(pub u16);

impl TbpFlags {
    /// Null bitmap is present
    pub const HAS_NULLS: u16 = 1 << 0;
    /// Row index is present (for variable-length data)
    pub const HAS_ROW_INDEX: u16 = 1 << 1;
    /// Data is compressed
    pub const COMPRESSED: u16 = 1 << 2;
    /// Schema is embedded in the file
    pub const EMBEDDED_SCHEMA: u16 = 1 << 3;

    pub fn has_nulls(&self) -> bool {
        self.0 & Self::HAS_NULLS != 0
    }

    pub fn has_row_index(&self) -> bool {
        self.0 & Self::HAS_ROW_INDEX != 0
    }

    pub fn is_compressed(&self) -> bool {
        self.0 & Self::COMPRESSED != 0
    }

    pub fn has_embedded_schema(&self) -> bool {
        self.0 & Self::EMBEDDED_SCHEMA != 0
    }
}

/// Column type for TBP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TbpColumnType {
    /// Null (no data)
    Null = 0,
    /// Boolean (1 byte, 0 or 1)
    Bool = 1,
    /// Signed 8-bit integer
    Int8 = 2,
    /// Unsigned 8-bit integer
    UInt8 = 3,
    /// Signed 16-bit integer
    Int16 = 4,
    /// Unsigned 16-bit integer
    UInt16 = 5,
    /// Signed 32-bit integer
    Int32 = 6,
    /// Unsigned 32-bit integer
    UInt32 = 7,
    /// Signed 64-bit integer
    Int64 = 8,
    /// Unsigned 64-bit integer
    UInt64 = 9,
    /// 32-bit float
    Float32 = 10,
    /// 64-bit float
    Float64 = 11,
    /// Variable-length string (UTF-8)
    String = 12,
    /// Variable-length binary
    Binary = 13,
    /// Timestamp (microseconds since epoch)
    Timestamp = 14,
    /// Fixed-size binary (e.g., UUIDs)
    FixedBinary = 15,
}

impl TbpColumnType {
    /// Get the fixed size of this type, or None for variable-length types
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            TbpColumnType::Null => Some(0),
            TbpColumnType::Bool => Some(1),
            TbpColumnType::Int8 | TbpColumnType::UInt8 => Some(1),
            TbpColumnType::Int16 | TbpColumnType::UInt16 => Some(2),
            TbpColumnType::Int32 | TbpColumnType::UInt32 | TbpColumnType::Float32 => Some(4),
            TbpColumnType::Int64 | TbpColumnType::UInt64 | TbpColumnType::Float64 | TbpColumnType::Timestamp => Some(8),
            TbpColumnType::String | TbpColumnType::Binary => None,
            TbpColumnType::FixedBinary => None, // Size specified per column
        }
    }

    /// Check if this type is variable-length
    pub fn is_variable(&self) -> bool {
        self.fixed_size().is_none()
    }

    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Null),
            1 => Some(Self::Bool),
            2 => Some(Self::Int8),
            3 => Some(Self::UInt8),
            4 => Some(Self::Int16),
            5 => Some(Self::UInt16),
            6 => Some(Self::Int32),
            7 => Some(Self::UInt32),
            8 => Some(Self::Int64),
            9 => Some(Self::UInt64),
            10 => Some(Self::Float32),
            11 => Some(Self::Float64),
            12 => Some(Self::String),
            13 => Some(Self::Binary),
            14 => Some(Self::Timestamp),
            15 => Some(Self::FixedBinary),
            _ => None,
        }
    }
}

/// Column definition in TBP schema
#[derive(Debug, Clone)]
pub struct TbpColumn {
    /// Column name
    pub name: String,
    /// Column type
    pub col_type: TbpColumnType,
    /// Fixed size for FixedBinary type
    pub fixed_size: Option<u16>,
    /// Column is nullable
    pub nullable: bool,
}

impl TbpColumn {
    pub fn new(name: impl Into<String>, col_type: TbpColumnType) -> Self {
        Self {
            name: name.into(),
            col_type,
            fixed_size: None,
            nullable: true,
        }
    }

    pub fn with_fixed_size(mut self, size: u16) -> Self {
        self.fixed_size = Some(size);
        self
    }

    pub fn not_null(mut self) -> Self {
        self.nullable = false;
        self
    }
}

/// TBP schema
#[derive(Debug, Clone)]
pub struct TbpSchema {
    /// Table name
    pub name: String,
    /// Columns
    pub columns: Vec<TbpColumn>,
    /// Schema ID (hash for validation)
    pub schema_id: u64,
}

impl TbpSchema {
    pub fn new(name: impl Into<String>, columns: Vec<TbpColumn>) -> Self {
        let name = name.into();
        let schema_id = Self::compute_schema_id(&name, &columns);
        Self {
            name,
            columns,
            schema_id,
        }
    }

    /// Compute a hash of the schema for validation
    fn compute_schema_id(name: &str, columns: &[TbpColumn]) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        for col in columns {
            col.name.hash(&mut hasher);
            (col.col_type as u8).hash(&mut hasher);
            col.fixed_size.hash(&mut hasher);
            col.nullable.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Check if schema has any variable-length columns
    pub fn has_variable_columns(&self) -> bool {
        self.columns.iter().any(|c| c.col_type.is_variable())
    }

    /// Check if schema has any nullable columns
    pub fn has_nullable_columns(&self) -> bool {
        self.columns.iter().any(|c| c.nullable)
    }

    /// Get the fixed row size (if all columns are fixed-size)
    pub fn fixed_row_size(&self) -> Option<usize> {
        if self.has_variable_columns() {
            return None;
        }

        let mut size = 0;
        for col in &self.columns {
            match col.col_type {
                TbpColumnType::FixedBinary => {
                    size += col.fixed_size.unwrap_or(0) as usize;
                }
                _ => {
                    size += col.col_type.fixed_size()?;
                }
            }
        }
        Some(size)
    }
}

/// TBP header (32 bytes)
#[derive(Debug, Clone)]
pub struct TbpHeader {
    /// Magic number (should be TBP_MAGIC)
    pub magic: u32,
    /// Version number
    pub version: u16,
    /// Flags
    pub flags: TbpFlags,
    /// Schema ID for validation
    pub schema_id: u64,
    /// Number of rows
    pub row_count: u32,
    /// Number of columns
    pub column_count: u16,
    /// Reserved
    pub reserved: u16,
    /// Offset to null bitmap (0 if no nulls)
    pub null_bitmap_offset: u32,
    /// Offset to row index (0 if fixed-size rows)
    pub row_index_offset: u32,
}

impl TbpHeader {
    /// Write header to a buffer
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.magic)?;
        w.write_u16::<LittleEndian>(self.version)?;
        w.write_u16::<LittleEndian>(self.flags.0)?;
        w.write_u64::<LittleEndian>(self.schema_id)?;
        w.write_u32::<LittleEndian>(self.row_count)?;
        w.write_u16::<LittleEndian>(self.column_count)?;
        w.write_u16::<LittleEndian>(self.reserved)?;
        w.write_u32::<LittleEndian>(self.null_bitmap_offset)?;
        w.write_u32::<LittleEndian>(self.row_index_offset)?;
        Ok(())
    }

    /// Read header from a buffer
    pub fn read(data: &[u8]) -> io::Result<Self> {
        if data.len() < TBP_HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Header too short"));
        }

        let mut cursor = std::io::Cursor::new(data);
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != TBP_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid TBP magic"));
        }

        Ok(Self {
            magic,
            version: cursor.read_u16::<LittleEndian>()?,
            flags: TbpFlags(cursor.read_u16::<LittleEndian>()?),
            schema_id: cursor.read_u64::<LittleEndian>()?,
            row_count: cursor.read_u32::<LittleEndian>()?,
            column_count: cursor.read_u16::<LittleEndian>()?,
            reserved: cursor.read_u16::<LittleEndian>()?,
            null_bitmap_offset: cursor.read_u32::<LittleEndian>()?,
            row_index_offset: cursor.read_u32::<LittleEndian>()?,
        })
    }
}

/// Null bitmap for efficient null checking
#[derive(Debug, Clone, Copy)]
pub struct NullBitmap<'a> {
    data: &'a [u8],
    columns: usize,
}

impl<'a> NullBitmap<'a> {
    pub fn new(data: &'a [u8], columns: usize) -> Self {
        Self { data, columns }
    }

    /// Check if a cell is null - O(1)
    #[inline]
    pub fn is_null(&self, row: usize, col: usize) -> bool {
        let bit_idx = row * self.columns + col;
        let byte_idx = bit_idx / 8;
        let bit_pos = bit_idx % 8;

        if byte_idx >= self.data.len() {
            return false;
        }

        self.data[byte_idx] & (1 << bit_pos) != 0
    }

    /// Calculate required size for bitmap
    pub fn required_size(rows: usize, cols: usize) -> usize {
        (rows * cols + 7) / 8
    }
}

/// Mutable null bitmap for writing
pub struct NullBitmapMut {
    data: Vec<u8>,
    columns: usize,
}

impl NullBitmapMut {
    pub fn new(rows: usize, columns: usize) -> Self {
        let size = NullBitmap::required_size(rows, columns);
        Self {
            data: vec![0; size],
            columns,
        }
    }

    /// Set a cell as null
    #[inline]
    pub fn set_null(&mut self, row: usize, col: usize) {
        let bit_idx = row * self.columns + col;
        let byte_idx = bit_idx / 8;
        let bit_pos = bit_idx % 8;

        if byte_idx < self.data.len() {
            self.data[byte_idx] |= 1 << bit_pos;
        }
    }

    /// Get the raw bitmap data
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Into raw data
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

/// Zero-copy row view into TBP data
#[derive(Debug, Clone)]
pub struct RowView<'a> {
    /// Schema reference
    schema: &'a TbpSchema,
    /// Raw row data
    data: &'a [u8],
    /// Null bitmap reference
    null_bitmap: Option<&'a NullBitmap<'a>>,
    /// Row index for null bitmap access
    row_idx: usize,
}

impl<'a> RowView<'a> {
    pub fn new(
        schema: &'a TbpSchema,
        data: &'a [u8],
        null_bitmap: Option<&'a NullBitmap<'a>>,
        row_idx: usize,
    ) -> Self {
        Self {
            schema,
            data,
            null_bitmap,
            row_idx,
        }
    }

    /// Check if column is null - O(1)
    #[inline]
    pub fn is_null(&self, col: usize) -> bool {
        self.null_bitmap
            .map(|b| b.is_null(self.row_idx, col))
            .unwrap_or(false)
    }

    /// Get column offset for fixed-size columns
    fn column_offset(&self, col: usize) -> usize {
        let mut offset = 0;
        for c in &self.schema.columns[..col] {
            offset += match c.col_type {
                TbpColumnType::FixedBinary => c.fixed_size.unwrap_or(0) as usize,
                _ => c.col_type.fixed_size().unwrap_or(0),
            };
        }
        offset
    }

    /// Read a boolean column - O(1)
    pub fn read_bool(&self, col: usize) -> Option<bool> {
        if self.is_null(col) {
            return None;
        }
        let offset = self.column_offset(col);
        Some(self.data.get(offset).copied().unwrap_or(0) != 0)
    }

    /// Read an i64 column - O(1)
    pub fn read_i64(&self, col: usize) -> Option<i64> {
        if self.is_null(col) {
            return None;
        }
        let offset = self.column_offset(col);
        if offset + 8 > self.data.len() {
            return None;
        }
        let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().ok()?;
        Some(i64::from_le_bytes(bytes))
    }

    /// Read a u64 column - O(1)
    pub fn read_u64(&self, col: usize) -> Option<u64> {
        if self.is_null(col) {
            return None;
        }
        let offset = self.column_offset(col);
        if offset + 8 > self.data.len() {
            return None;
        }
        let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().ok()?;
        Some(u64::from_le_bytes(bytes))
    }

    /// Read an f64 column - O(1)
    pub fn read_f64(&self, col: usize) -> Option<f64> {
        if self.is_null(col) {
            return None;
        }
        let offset = self.column_offset(col);
        if offset + 8 > self.data.len() {
            return None;
        }
        let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().ok()?;
        Some(f64::from_le_bytes(bytes))
    }

    /// Read an i32 column - O(1)
    pub fn read_i32(&self, col: usize) -> Option<i32> {
        if self.is_null(col) {
            return None;
        }
        let offset = self.column_offset(col);
        if offset + 4 > self.data.len() {
            return None;
        }
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().ok()?;
        Some(i32::from_le_bytes(bytes))
    }

    /// Read an f32 column - O(1)
    pub fn read_f32(&self, col: usize) -> Option<f32> {
        if self.is_null(col) {
            return None;
        }
        let offset = self.column_offset(col);
        if offset + 4 > self.data.len() {
            return None;
        }
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().ok()?;
        Some(f32::from_le_bytes(bytes))
    }

    /// Get raw row data
    pub fn raw_data(&self) -> &[u8] {
        self.data
    }
}

/// TBP writer for creating binary tables
pub struct TbpWriter {
    schema: TbpSchema,
    null_bitmap: NullBitmapMut,
    row_index: Vec<u32>,
    data: Vec<u8>,
    row_count: usize,
}

impl TbpWriter {
    pub fn new(schema: TbpSchema, estimated_rows: usize) -> Self {
        Self {
            null_bitmap: NullBitmapMut::new(estimated_rows, schema.columns.len()),
            row_index: Vec::with_capacity(estimated_rows),
            data: Vec::with_capacity(estimated_rows * schema.fixed_row_size().unwrap_or(64)),
            row_count: 0,
            schema,
        }
    }

    /// Start a new row and return a row writer
    pub fn start_row(&mut self) -> TbpRowWriter<'_> {
        let offset = self.data.len() as u32;
        self.row_index.push(offset);
        TbpRowWriter {
            writer: self,
            col_idx: 0,
        }
    }

    /// Mark a cell as null
    fn set_null(&mut self, row: usize, col: usize) {
        self.null_bitmap.set_null(row, col);
    }

    /// Finish writing and produce the final buffer
    pub fn finish(self) -> Vec<u8> {
        let has_nulls = self.schema.has_nullable_columns();
        let has_variable = self.schema.has_variable_columns();

        let mut flags = TbpFlags(0);
        if has_nulls {
            flags.0 |= TbpFlags::HAS_NULLS;
        }
        if has_variable {
            flags.0 |= TbpFlags::HAS_ROW_INDEX;
        }

        // Calculate offsets
        let null_bitmap_offset = if has_nulls { TBP_HEADER_SIZE as u32 } else { 0 };
        let null_bitmap_size = if has_nulls {
            NullBitmap::required_size(self.row_count, self.schema.columns.len())
        } else {
            0
        };

        let row_index_offset = if has_variable {
            (TBP_HEADER_SIZE + null_bitmap_size) as u32
        } else {
            0
        };
        let row_index_size = if has_variable {
            self.row_count * 4
        } else {
            0
        };

        let data_offset = TBP_HEADER_SIZE + null_bitmap_size + row_index_size;

        let header = TbpHeader {
            magic: TBP_MAGIC,
            version: TBP_VERSION,
            flags,
            schema_id: self.schema.schema_id,
            row_count: self.row_count as u32,
            column_count: self.schema.columns.len() as u16,
            reserved: 0,
            null_bitmap_offset,
            row_index_offset,
        };

        let total_size = data_offset + self.data.len();
        let mut buffer = Vec::with_capacity(total_size);

        // Write header
        header.write(&mut buffer).unwrap();

        // Write null bitmap
        if has_nulls {
            let required = NullBitmap::required_size(self.row_count, self.schema.columns.len());
            buffer.extend_from_slice(&self.null_bitmap.as_bytes()[..required]);
        }

        // Write row index
        if has_variable {
            for offset in &self.row_index {
                buffer.write_u32::<LittleEndian>(*offset + data_offset as u32).unwrap();
            }
        }

        // Write data
        buffer.extend_from_slice(&self.data);

        buffer
    }
}

/// Row writer for TBP
pub struct TbpRowWriter<'a> {
    writer: &'a mut TbpWriter,
    col_idx: usize,
}

impl<'a> TbpRowWriter<'a> {
    /// Write a null value
    pub fn write_null(mut self) -> Self {
        self.writer.set_null(self.writer.row_count, self.col_idx);
        self.col_idx += 1;
        self
    }

    /// Write a boolean
    pub fn write_bool(mut self, value: bool) -> Self {
        self.writer.data.push(if value { 1 } else { 0 });
        self.col_idx += 1;
        self
    }

    /// Write an i64
    pub fn write_i64(mut self, value: i64) -> Self {
        self.writer.data.extend_from_slice(&value.to_le_bytes());
        self.col_idx += 1;
        self
    }

    /// Write a u64
    pub fn write_u64(mut self, value: u64) -> Self {
        self.writer.data.extend_from_slice(&value.to_le_bytes());
        self.col_idx += 1;
        self
    }

    /// Write an f64
    pub fn write_f64(mut self, value: f64) -> Self {
        self.writer.data.extend_from_slice(&value.to_le_bytes());
        self.col_idx += 1;
        self
    }

    /// Write an i32
    pub fn write_i32(mut self, value: i32) -> Self {
        self.writer.data.extend_from_slice(&value.to_le_bytes());
        self.col_idx += 1;
        self
    }

    /// Write an f32
    pub fn write_f32(mut self, value: f32) -> Self {
        self.writer.data.extend_from_slice(&value.to_le_bytes());
        self.col_idx += 1;
        self
    }

    /// Write a string (variable length)
    pub fn write_string(mut self, value: &str) -> Self {
        let bytes = value.as_bytes();
        self.writer.data.write_u32::<LittleEndian>(bytes.len() as u32).unwrap();
        self.writer.data.extend_from_slice(bytes);
        self.col_idx += 1;
        self
    }

    /// Write binary data (variable length)
    pub fn write_binary(mut self, value: &[u8]) -> Self {
        self.writer.data.write_u32::<LittleEndian>(value.len() as u32).unwrap();
        self.writer.data.extend_from_slice(value);
        self.col_idx += 1;
        self
    }

    /// Finish the row
    pub fn finish(self) {
        self.writer.row_count += 1;
    }
}

/// TBP reader for zero-copy access
pub struct TbpReader<'a> {
    data: &'a [u8],
    header: TbpHeader,
    schema: &'a TbpSchema,
}

impl<'a> TbpReader<'a> {
    /// Create a new reader
    pub fn new(data: &'a [u8], schema: &'a TbpSchema) -> io::Result<Self> {
        let header = TbpHeader::read(data)?;

        if header.schema_id != schema.schema_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Schema ID mismatch",
            ));
        }

        Ok(Self {
            data,
            header,
            schema,
        })
    }

    /// Get number of rows
    pub fn row_count(&self) -> usize {
        self.header.row_count as usize
    }

    /// Get a row by index - O(1)
    pub fn get_row(&self, row: usize) -> Option<RowView<'_>> {
        if row >= self.row_count() {
            return None;
        }

        // Get row offset
        let row_offset = if self.header.flags.has_row_index() {
            let idx_offset = self.header.row_index_offset as usize + row * 4;
            if idx_offset + 4 > self.data.len() {
                return None;
            }
            let bytes: [u8; 4] = self.data[idx_offset..idx_offset + 4].try_into().ok()?;
            u32::from_le_bytes(bytes) as usize
        } else {
            // Fixed-size rows
            let row_size = self.schema.fixed_row_size()?;
            let null_bitmap_size = if self.header.flags.has_nulls() {
                NullBitmap::required_size(self.row_count(), self.schema.columns.len())
            } else {
                0
            };
            TBP_HEADER_SIZE + null_bitmap_size + row * row_size
        };

        let row_data = &self.data[row_offset..];

        // TODO: properly construct null bitmap reference
        Some(RowView::new(self.schema, row_data, None, row))
    }

    /// Iterate over all rows - zero allocation per row
    pub fn iter(&'a self) -> impl Iterator<Item = RowView<'a>> {
        (0..self.row_count()).filter_map(move |i| self.get_row(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = TbpHeader {
            magic: TBP_MAGIC,
            version: TBP_VERSION,
            flags: TbpFlags(TbpFlags::HAS_NULLS | TbpFlags::HAS_ROW_INDEX),
            schema_id: 12345678,
            row_count: 100,
            column_count: 5,
            reserved: 0,
            null_bitmap_offset: 32,
            row_index_offset: 48,
        };

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();
        assert_eq!(buffer.len(), TBP_HEADER_SIZE);

        let parsed = TbpHeader::read(&buffer).unwrap();
        assert_eq!(parsed.magic, TBP_MAGIC);
        assert_eq!(parsed.version, TBP_VERSION);
        assert_eq!(parsed.row_count, 100);
        assert_eq!(parsed.column_count, 5);
    }

    #[test]
    fn test_null_bitmap() {
        let mut bitmap = NullBitmapMut::new(10, 5);
        bitmap.set_null(0, 0);
        bitmap.set_null(5, 3);
        bitmap.set_null(9, 4);

        let data = bitmap.as_bytes();
        let reader = NullBitmap::new(data, 5);

        assert!(reader.is_null(0, 0));
        assert!(!reader.is_null(0, 1));
        assert!(reader.is_null(5, 3));
        assert!(reader.is_null(9, 4));
        assert!(!reader.is_null(9, 3));
    }

    #[test]
    fn test_writer_reader_roundtrip() {
        let schema = TbpSchema::new(
            "test_table",
            vec![
                TbpColumn::new("id", TbpColumnType::Int64).not_null(),
                TbpColumn::new("value", TbpColumnType::Float64),
            ],
        );

        let mut writer = TbpWriter::new(schema.clone(), 100);

        // Write some rows
        for i in 0..10 {
            writer
                .start_row()
                .write_i64(i)
                .write_f64(i as f64 * 1.5)
                .finish();
        }

        let data = writer.finish();

        // Read back
        let reader = TbpReader::new(&data, &schema).unwrap();
        assert_eq!(reader.row_count(), 10);

        let row = reader.get_row(5).unwrap();
        assert_eq!(row.read_i64(0), Some(5));
        assert_eq!(row.read_f64(1), Some(7.5));
    }
}
