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

//! Adaptive Hybrid Storage (AHS) - PAX Block Layout
//!
//! From mm.md Task 4.1: Hybrid Row-Column Storage with PAX Blocks
//!
//! ## Problem
//!
//! Current storage uses pure row format requiring full row materialization
//! even for single-column queries. LLMs often need specific columns
//! (e.g., just `summary` field for context building).
//!
//! ## Solution
//!
//! PAX (Partition Attributes Across) provides the best of both worlds:
//! - Row-oriented at block level (good for point queries)
//! - Column-oriented within blocks (good for scans)
//!
//! ## Layout
//!
//! ```text
//! Block Size = 64KB (L2 cache friendly)
//!
//! ┌─────────────────────────────────────────┐
//! │ Block Header (64 bytes)                 │
//! │  - row_count, column_count              │
//! │  - minipage_offsets: [u32; col_count]   │
//! ├─────────────────────────────────────────┤
//! │ Null Bitmap (packed)                    │
//! ├─────────────────────────────────────────┤
//! │ Minipage 0 (Column 0 values)            │
//! │ Minipage 1 (Column 1 values)            │
//! │ ...                                     │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Cache-Oblivious Analysis
//!
//! ```text
//! Cache line = 64 bytes, i64 column: 8 values/line
//!
//! Row-store scan (all columns):
//!   Cache misses = O(N × cols / B) where B = block transfer
//!
//! PAX with column pruning (k columns):
//!   Cache misses = O(N × k / B)
//!
//! For 10-column table, selecting 2 columns:
//!   Bandwidth reduction = 10/2 = 5×
//! ```

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Default PAX block size (64KB - L2 cache friendly)
pub const PAX_BLOCK_SIZE: usize = 64 * 1024;

/// Maximum columns per block
pub const MAX_COLUMNS: usize = 256;

/// PAX block header size
pub const PAX_HEADER_SIZE: usize = 64;

/// Magic number for PAX blocks
pub const PAX_MAGIC: u32 = 0x50415821; // "PAX!"

/// Column type for PAX storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PaxColumnType {
    /// Boolean (bit-packed)
    Bool = 0,
    /// 8-bit integer
    Int8 = 1,
    /// 16-bit integer
    Int16 = 2,
    /// 32-bit integer
    Int32 = 3,
    /// 64-bit integer
    Int64 = 4,
    /// 32-bit float
    Float32 = 5,
    /// 64-bit float
    Float64 = 6,
    /// Variable-length binary (offset + data)
    VarBinary = 7,
    /// Fixed-size binary
    FixedBinary = 8,
}

impl PaxColumnType {
    /// Get fixed size in bytes, None for variable-length
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            PaxColumnType::Bool => Some(1), // Stored as byte for simplicity
            PaxColumnType::Int8 => Some(1),
            PaxColumnType::Int16 => Some(2),
            PaxColumnType::Int32 => Some(4),
            PaxColumnType::Int64 => Some(8),
            PaxColumnType::Float32 => Some(4),
            PaxColumnType::Float64 => Some(8),
            PaxColumnType::VarBinary => None,
            PaxColumnType::FixedBinary => None, // Size is per-column
        }
    }

    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Bool),
            1 => Some(Self::Int8),
            2 => Some(Self::Int16),
            3 => Some(Self::Int32),
            4 => Some(Self::Int64),
            5 => Some(Self::Float32),
            6 => Some(Self::Float64),
            7 => Some(Self::VarBinary),
            8 => Some(Self::FixedBinary),
            _ => None,
        }
    }
}

/// Column definition for PAX
#[derive(Debug, Clone)]
pub struct PaxColumnDef {
    /// Column name
    pub name: String,
    /// Column type
    pub col_type: PaxColumnType,
    /// Fixed size for FixedBinary columns
    pub fixed_size: Option<u16>,
    /// Whether column is nullable
    pub nullable: bool,
}

impl PaxColumnDef {
    pub fn new(name: impl Into<String>, col_type: PaxColumnType) -> Self {
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

/// PAX schema
#[derive(Debug, Clone)]
pub struct PaxSchema {
    pub columns: Vec<PaxColumnDef>,
}

impl PaxSchema {
    pub fn new(columns: Vec<PaxColumnDef>) -> Self {
        Self { columns }
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Check if schema has any nullable columns
    pub fn has_nullable(&self) -> bool {
        self.columns.iter().any(|c| c.nullable)
    }

    /// Check if schema has any variable-length columns
    pub fn has_variable(&self) -> bool {
        self.columns.iter().any(|c| c.col_type == PaxColumnType::VarBinary)
    }
}

/// PAX block header (64 bytes)
#[derive(Debug, Clone)]
pub struct PaxBlockHeader {
    /// Magic number
    pub magic: u32,
    /// Version
    pub version: u16,
    /// Flags
    pub flags: u16,
    /// Number of rows in this block
    pub row_count: u32,
    /// Number of columns
    pub column_count: u16,
    /// Reserved
    pub reserved: u16,
    /// Offset to null bitmap (from block start)
    pub null_bitmap_offset: u32,
    /// Size of null bitmap in bytes
    pub null_bitmap_size: u32,
    /// Offsets to each minipage (from block start)
    /// Stored after header, [u32; column_count]
    pub minipage_offsets: Vec<u32>,
    /// Sizes of each minipage
    pub minipage_sizes: Vec<u32>,
}

impl PaxBlockHeader {
    /// Header base size (without variable arrays)
    const BASE_SIZE: usize = 24;

    pub fn new(row_count: u32, column_count: usize) -> Self {
        Self {
            magic: PAX_MAGIC,
            version: 1,
            flags: 0,
            row_count,
            column_count: column_count as u16,
            reserved: 0,
            null_bitmap_offset: 0,
            null_bitmap_size: 0,
            minipage_offsets: vec![0; column_count],
            minipage_sizes: vec![0; column_count],
        }
    }

    /// Compute total header size including offset arrays
    pub fn total_size(&self) -> usize {
        Self::BASE_SIZE + (self.column_count as usize) * 8 // offsets + sizes
    }

    /// Write header to buffer
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.magic)?;
        w.write_u16::<LittleEndian>(self.version)?;
        w.write_u16::<LittleEndian>(self.flags)?;
        w.write_u32::<LittleEndian>(self.row_count)?;
        w.write_u16::<LittleEndian>(self.column_count)?;
        w.write_u16::<LittleEndian>(self.reserved)?;
        w.write_u32::<LittleEndian>(self.null_bitmap_offset)?;
        w.write_u32::<LittleEndian>(self.null_bitmap_size)?;

        for &offset in &self.minipage_offsets {
            w.write_u32::<LittleEndian>(offset)?;
        }
        for &size in &self.minipage_sizes {
            w.write_u32::<LittleEndian>(size)?;
        }

        Ok(())
    }

    /// Read header from buffer
    pub fn read<R: Read>(r: &mut R, _column_count: usize) -> io::Result<Self> {
        let magic = r.read_u32::<LittleEndian>()?;
        if magic != PAX_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid PAX magic"));
        }

        let version = r.read_u16::<LittleEndian>()?;
        let flags = r.read_u16::<LittleEndian>()?;
        let row_count = r.read_u32::<LittleEndian>()?;
        let col_count = r.read_u16::<LittleEndian>()?;
        let reserved = r.read_u16::<LittleEndian>()?;
        let null_bitmap_offset = r.read_u32::<LittleEndian>()?;
        let null_bitmap_size = r.read_u32::<LittleEndian>()?;

        let mut minipage_offsets = vec![0u32; col_count as usize];
        for offset in &mut minipage_offsets {
            *offset = r.read_u32::<LittleEndian>()?;
        }

        let mut minipage_sizes = vec![0u32; col_count as usize];
        for size in &mut minipage_sizes {
            *size = r.read_u32::<LittleEndian>()?;
        }

        Ok(Self {
            magic,
            version,
            flags,
            row_count,
            column_count: col_count,
            reserved,
            null_bitmap_offset,
            null_bitmap_size,
            minipage_offsets,
            minipage_sizes,
        })
    }
}

/// Minipage - columnar data for a single column within a block
#[derive(Debug)]
pub struct Minipage {
    /// Column index
    pub column_idx: usize,
    /// Raw data
    pub data: Vec<u8>,
    /// Column type
    pub col_type: PaxColumnType,
    /// Number of values
    pub value_count: usize,
}

impl Minipage {
    pub fn new(column_idx: usize, col_type: PaxColumnType, capacity: usize) -> Self {
        let data_capacity = match col_type.fixed_size() {
            Some(size) => capacity * size,
            None => capacity * 16, // Estimate for variable-length
        };

        Self {
            column_idx,
            data: Vec::with_capacity(data_capacity),
            col_type,
            value_count: 0,
        }
    }

    /// Write an i64 value
    pub fn write_i64(&mut self, value: i64) {
        self.data.extend_from_slice(&value.to_le_bytes());
        self.value_count += 1;
    }

    /// Write an i32 value
    pub fn write_i32(&mut self, value: i32) {
        self.data.extend_from_slice(&value.to_le_bytes());
        self.value_count += 1;
    }

    /// Write an f64 value
    pub fn write_f64(&mut self, value: f64) {
        self.data.extend_from_slice(&value.to_le_bytes());
        self.value_count += 1;
    }

    /// Write an f32 value
    pub fn write_f32(&mut self, value: f32) {
        self.data.extend_from_slice(&value.to_le_bytes());
        self.value_count += 1;
    }

    /// Write a bool value
    pub fn write_bool(&mut self, value: bool) {
        self.data.push(if value { 1 } else { 0 });
        self.value_count += 1;
    }

    /// Write variable-length binary
    pub fn write_var_binary(&mut self, value: &[u8]) {
        self.data.write_u32::<LittleEndian>(value.len() as u32).unwrap();
        self.data.extend_from_slice(value);
        self.value_count += 1;
    }

    /// Read i64 at index
    pub fn read_i64(&self, idx: usize) -> Option<i64> {
        let offset = idx * 8;
        if offset + 8 > self.data.len() {
            return None;
        }
        let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().ok()?;
        Some(i64::from_le_bytes(bytes))
    }

    /// Read i32 at index
    pub fn read_i32(&self, idx: usize) -> Option<i32> {
        let offset = idx * 4;
        if offset + 4 > self.data.len() {
            return None;
        }
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().ok()?;
        Some(i32::from_le_bytes(bytes))
    }

    /// Read f64 at index
    pub fn read_f64(&self, idx: usize) -> Option<f64> {
        let offset = idx * 8;
        if offset + 8 > self.data.len() {
            return None;
        }
        let bytes: [u8; 8] = self.data[offset..offset + 8].try_into().ok()?;
        Some(f64::from_le_bytes(bytes))
    }

    /// Read f32 at index
    pub fn read_f32(&self, idx: usize) -> Option<f32> {
        let offset = idx * 4;
        if offset + 4 > self.data.len() {
            return None;
        }
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().ok()?;
        Some(f32::from_le_bytes(bytes))
    }

    /// Read bool at index
    pub fn read_bool(&self, idx: usize) -> Option<bool> {
        self.data.get(idx).map(|&v| v != 0)
    }

    /// Get raw slice for SIMD operations
    pub fn as_i64_slice(&self) -> &[i64] {
        // Safety: Data is aligned and written as i64
        let ptr = self.data.as_ptr() as *const i64;
        let len = self.data.len() / 8;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get raw slice for SIMD operations
    pub fn as_f64_slice(&self) -> &[f64] {
        let ptr = self.data.as_ptr() as *const f64;
        let len = self.data.len() / 8;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get raw slice for SIMD operations
    pub fn as_i32_slice(&self) -> &[i32] {
        let ptr = self.data.as_ptr() as *const i32;
        let len = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get raw slice for SIMD operations
    pub fn as_f32_slice(&self) -> &[f32] {
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// PAX block writer
pub struct PaxBlockWriter {
    schema: PaxSchema,
    null_bitmap: Vec<u8>,
    minipages: Vec<Minipage>,
    row_count: usize,
    max_rows: usize,
}

impl PaxBlockWriter {
    pub fn new(schema: PaxSchema, max_rows: usize) -> Self {
        let col_count = schema.column_count();
        let null_bitmap_size = (max_rows * col_count + 7) / 8;

        let minipages = schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| Minipage::new(i, col.col_type, max_rows))
            .collect();

        Self {
            schema,
            null_bitmap: vec![0; null_bitmap_size],
            minipages,
            row_count: 0,
            max_rows,
        }
    }

    /// Check if block is full
    pub fn is_full(&self) -> bool {
        self.row_count >= self.max_rows
    }

    /// Get current row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Mark a cell as null
    fn set_null(&mut self, row: usize, col: usize) {
        let bit_idx = row * self.schema.column_count() + col;
        let byte_idx = bit_idx / 8;
        let bit_pos = bit_idx % 8;
        if byte_idx < self.null_bitmap.len() {
            self.null_bitmap[byte_idx] |= 1 << bit_pos;
        }
    }

    /// Start writing a new row
    pub fn start_row(&mut self) -> PaxRowWriter<'_> {
        PaxRowWriter {
            block: self,
            col_idx: 0,
        }
    }

    /// Finish and build the block
    pub fn finish(self) -> PaxBlock {
        let mut header = PaxBlockHeader::new(self.row_count as u32, self.schema.column_count());

        // Calculate offsets
        let header_size = header.total_size();
        let null_bitmap_size = (self.row_count * self.schema.column_count() + 7) / 8;

        header.null_bitmap_offset = header_size as u32;
        header.null_bitmap_size = null_bitmap_size as u32;

        let mut offset = header_size + null_bitmap_size;
        for (i, mp) in self.minipages.iter().enumerate() {
            header.minipage_offsets[i] = offset as u32;
            header.minipage_sizes[i] = mp.data.len() as u32;
            offset += mp.data.len();
        }

        PaxBlock {
            header,
            null_bitmap: self.null_bitmap[..null_bitmap_size].to_vec(),
            minipages: self.minipages,
            schema: self.schema,
        }
    }
}

/// Row writer for PAX blocks
pub struct PaxRowWriter<'a> {
    block: &'a mut PaxBlockWriter,
    col_idx: usize,
}

impl<'a> PaxRowWriter<'a> {
    /// Write null value
    pub fn write_null(mut self) -> Self {
        self.block.set_null(self.block.row_count, self.col_idx);
        // Write a zero placeholder
        match self.block.schema.columns[self.col_idx].col_type {
            PaxColumnType::Bool => self.block.minipages[self.col_idx].write_bool(false),
            PaxColumnType::Int8 => self.block.minipages[self.col_idx].data.push(0),
            PaxColumnType::Int16 => self.block.minipages[self.col_idx].data.extend_from_slice(&[0; 2]),
            PaxColumnType::Int32 | PaxColumnType::Float32 => {
                self.block.minipages[self.col_idx].data.extend_from_slice(&[0; 4]);
            }
            PaxColumnType::Int64 | PaxColumnType::Float64 => {
                self.block.minipages[self.col_idx].data.extend_from_slice(&[0; 8]);
            }
            PaxColumnType::VarBinary => {
                self.block.minipages[self.col_idx].write_var_binary(&[]);
            }
            PaxColumnType::FixedBinary => {
                let size = self.block.schema.columns[self.col_idx].fixed_size.unwrap_or(0) as usize;
                self.block.minipages[self.col_idx].data.extend(std::iter::repeat(0).take(size));
            }
        }
        self.block.minipages[self.col_idx].value_count += 1;
        self.col_idx += 1;
        self
    }

    /// Write i64 value
    pub fn write_i64(mut self, value: i64) -> Self {
        self.block.minipages[self.col_idx].write_i64(value);
        self.col_idx += 1;
        self
    }

    /// Write i32 value
    pub fn write_i32(mut self, value: i32) -> Self {
        self.block.minipages[self.col_idx].write_i32(value);
        self.col_idx += 1;
        self
    }

    /// Write f64 value
    pub fn write_f64(mut self, value: f64) -> Self {
        self.block.minipages[self.col_idx].write_f64(value);
        self.col_idx += 1;
        self
    }

    /// Write f32 value
    pub fn write_f32(mut self, value: f32) -> Self {
        self.block.minipages[self.col_idx].write_f32(value);
        self.col_idx += 1;
        self
    }

    /// Write bool value
    pub fn write_bool(mut self, value: bool) -> Self {
        self.block.minipages[self.col_idx].write_bool(value);
        self.col_idx += 1;
        self
    }

    /// Write variable-length binary
    pub fn write_bytes(mut self, value: &[u8]) -> Self {
        self.block.minipages[self.col_idx].write_var_binary(value);
        self.col_idx += 1;
        self
    }

    /// Write string
    pub fn write_string(self, value: &str) -> Self {
        self.write_bytes(value.as_bytes())
    }

    /// Finish the row
    pub fn finish(self) {
        self.block.row_count += 1;
    }
}

/// Complete PAX block
#[derive(Debug)]
pub struct PaxBlock {
    pub header: PaxBlockHeader,
    pub null_bitmap: Vec<u8>,
    pub minipages: Vec<Minipage>,
    pub schema: PaxSchema,
}

impl PaxBlock {
    /// Get row count
    pub fn row_count(&self) -> usize {
        self.header.row_count as usize
    }

    /// Check if a cell is null
    pub fn is_null(&self, row: usize, col: usize) -> bool {
        let bit_idx = row * self.schema.column_count() + col;
        let byte_idx = bit_idx / 8;
        let bit_pos = bit_idx % 8;
        if byte_idx >= self.null_bitmap.len() {
            return false;
        }
        self.null_bitmap[byte_idx] & (1 << bit_pos) != 0
    }

    /// Get a column minipage for columnar access
    pub fn get_column(&self, col: usize) -> Option<&Minipage> {
        self.minipages.get(col)
    }

    /// Read i64 from specific row/column
    pub fn read_i64(&self, row: usize, col: usize) -> Option<i64> {
        if self.is_null(row, col) {
            return None;
        }
        self.minipages.get(col)?.read_i64(row)
    }

    /// Read f64 from specific row/column
    pub fn read_f64(&self, row: usize, col: usize) -> Option<f64> {
        if self.is_null(row, col) {
            return None;
        }
        self.minipages.get(col)?.read_f64(row)
    }

    /// Read i32 from specific row/column
    pub fn read_i32(&self, row: usize, col: usize) -> Option<i32> {
        if self.is_null(row, col) {
            return None;
        }
        self.minipages.get(col)?.read_i32(row)
    }

    /// Read bool from specific row/column
    pub fn read_bool(&self, row: usize, col: usize) -> Option<bool> {
        if self.is_null(row, col) {
            return None;
        }
        self.minipages.get(col)?.read_bool(row)
    }

    /// Serialize block to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        self.header.write(&mut buffer).unwrap();
        buffer.extend_from_slice(&self.null_bitmap);
        for mp in &self.minipages {
            buffer.extend_from_slice(&mp.data);
        }
        buffer
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.header.total_size()
            + self.null_bitmap.len()
            + self.minipages.iter().map(|m| m.data.len()).sum::<usize>()
    }
}

/// Column projection for selective reading
#[derive(Debug, Clone)]
pub struct ColumnProjection {
    /// Indices of columns to read
    columns: Vec<usize>,
}

impl ColumnProjection {
    pub fn new(columns: Vec<usize>) -> Self {
        Self { columns }
    }

    /// Create projection for all columns
    pub fn all(column_count: usize) -> Self {
        Self {
            columns: (0..column_count).collect(),
        }
    }

    /// Get projected column indices
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// Calculate bandwidth savings ratio
    ///
    /// Returns N/k where N = total columns, k = selected columns
    pub fn bandwidth_savings(&self, total_columns: usize) -> f64 {
        if self.columns.is_empty() {
            return 1.0;
        }
        total_columns as f64 / self.columns.len() as f64
    }
}

/// Iterator over PAX block rows with column projection
pub struct PaxBlockIterator<'a> {
    block: &'a PaxBlock,
    projection: ColumnProjection,
    current_row: usize,
}

impl<'a> PaxBlockIterator<'a> {
    pub fn new(block: &'a PaxBlock, projection: ColumnProjection) -> Self {
        Self {
            block,
            projection,
            current_row: 0,
        }
    }
    
    /// Get the next row as a view
    pub fn next_row(&mut self) -> Option<PaxRowViewOwned> {
        if self.current_row >= self.block.row_count() {
            return None;
        }

        let row = PaxRowViewOwned {
            row_idx: self.current_row,
            projection: self.projection.clone(),
        };

        self.current_row += 1;
        Some(row)
    }
}

impl<'a> Iterator for PaxBlockIterator<'a> {
    type Item = usize; // Returns row index

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.block.row_count() {
            return None;
        }

        let row_idx = self.current_row;
        self.current_row += 1;
        Some(row_idx)
    }
}

/// Owned row view data (no lifetime issues)
#[derive(Debug, Clone)]
pub struct PaxRowViewOwned {
    pub row_idx: usize,
    pub projection: ColumnProjection,
}

/// Zero-allocation row view for PAX block
pub struct PaxRowView<'a> {
    block: &'a PaxBlock,
    row_idx: usize,
    projection: &'a ColumnProjection,
}

impl<'a> PaxRowView<'a> {
    /// Get row index
    pub fn row_index(&self) -> usize {
        self.row_idx
    }

    /// Check if projected column is null
    pub fn is_null(&self, proj_idx: usize) -> bool {
        let col = self.projection.columns.get(proj_idx).copied().unwrap_or(0);
        self.block.is_null(self.row_idx, col)
    }

    /// Read i64 from projected column
    pub fn read_i64(&self, proj_idx: usize) -> Option<i64> {
        let col = *self.projection.columns.get(proj_idx)?;
        self.block.read_i64(self.row_idx, col)
    }

    /// Read f64 from projected column
    pub fn read_f64(&self, proj_idx: usize) -> Option<f64> {
        let col = *self.projection.columns.get(proj_idx)?;
        self.block.read_f64(self.row_idx, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pax_block_write_read() {
        let schema = PaxSchema::new(vec![
            PaxColumnDef::new("id", PaxColumnType::Int64),
            PaxColumnDef::new("value", PaxColumnType::Float64),
            PaxColumnDef::new("flag", PaxColumnType::Bool),
        ]);

        let mut writer = PaxBlockWriter::new(schema.clone(), 100);

        // Write some rows
        for i in 0..10 {
            writer
                .start_row()
                .write_i64(i)
                .write_f64(i as f64 * 1.5)
                .write_bool(i % 2 == 0)
                .finish();
        }

        let block = writer.finish();
        assert_eq!(block.row_count(), 10);

        // Read back
        assert_eq!(block.read_i64(0, 0), Some(0));
        assert_eq!(block.read_f64(0, 1), Some(0.0));
        assert_eq!(block.read_bool(0, 2), Some(true));

        assert_eq!(block.read_i64(5, 0), Some(5));
        assert_eq!(block.read_f64(5, 1), Some(7.5));
        assert_eq!(block.read_bool(5, 2), Some(false));
    }

    #[test]
    fn test_column_projection() {
        let schema = PaxSchema::new(vec![
            PaxColumnDef::new("a", PaxColumnType::Int64),
            PaxColumnDef::new("b", PaxColumnType::Int64),
            PaxColumnDef::new("c", PaxColumnType::Int64),
            PaxColumnDef::new("d", PaxColumnType::Int64),
        ]);

        let mut writer = PaxBlockWriter::new(schema.clone(), 100);
        for i in 0..5 {
            writer
                .start_row()
                .write_i64(i)
                .write_i64(i * 10)
                .write_i64(i * 100)
                .write_i64(i * 1000)
                .finish();
        }

        let block = writer.finish();

        // Project only columns 0 and 2
        let projection = ColumnProjection::new(vec![0, 2]);
        assert_eq!(projection.bandwidth_savings(4), 2.0);

        let mut iter = PaxBlockIterator::new(&block, projection);

        let row_idx = iter.next().unwrap();
        // Read from the block using the row index and original column indices
        // Projection maps columns [0, 2] so we read original columns 0 and 2
        assert_eq!(block.read_i64(row_idx, 0), Some(0)); // original column 0
        assert_eq!(block.read_i64(row_idx, 2), Some(0)); // original column 2

        let row_idx = iter.next().unwrap();
        assert_eq!(block.read_i64(row_idx, 0), Some(1)); // original column 0
        assert_eq!(block.read_i64(row_idx, 2), Some(100)); // original column 2
    }

    #[test]
    fn test_null_handling() {
        let schema = PaxSchema::new(vec![
            PaxColumnDef::new("id", PaxColumnType::Int64).not_null(),
            PaxColumnDef::new("value", PaxColumnType::Float64),
        ]);

        let mut writer = PaxBlockWriter::new(schema.clone(), 100);

        writer.start_row().write_i64(1).write_f64(1.0).finish();
        writer.start_row().write_i64(2).write_null().finish();
        writer.start_row().write_i64(3).write_f64(3.0).finish();

        let block = writer.finish();

        assert!(!block.is_null(0, 0));
        assert!(!block.is_null(0, 1));
        assert!(!block.is_null(1, 0));
        assert!(block.is_null(1, 1));
        assert!(!block.is_null(2, 1));

        assert_eq!(block.read_f64(0, 1), Some(1.0));
        assert_eq!(block.read_f64(1, 1), None);
        assert_eq!(block.read_f64(2, 1), Some(3.0));
    }

    #[test]
    fn test_columnar_access() {
        let schema = PaxSchema::new(vec![
            PaxColumnDef::new("id", PaxColumnType::Int64),
        ]);

        let mut writer = PaxBlockWriter::new(schema.clone(), 1000);
        for i in 0..100 {
            writer.start_row().write_i64(i).finish();
        }

        let block = writer.finish();

        // Get column minipage for SIMD-friendly access
        let col = block.get_column(0).unwrap();
        let slice = col.as_i64_slice();

        assert_eq!(slice.len(), 100);
        assert_eq!(slice[0], 0);
        assert_eq!(slice[50], 50);
        assert_eq!(slice[99], 99);
    }
}
