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

//! TOON Format Codec
//!
//! This module implements the TOON (Token-Optimized Object Notation) format
//! specification using the official `toon-format` crate.
//!
//! ## TOON Format Grammar (Simplified)
//!
//! ```text
//! document     ::= top_level_value
//! value        ::= simple_object | array | primitive
//! simple_object::= (key ":" value newline)+ 
//! array        ::= header newline item*
//! header       ::= name "[" count "]" ( "{" fields "}" )? ":"
//! item         ::= "-" value newline | row newline
//! ```

use crate::soch::{SochValue}; // Use shared types from soch.rs
use std::collections::HashMap;
use toon_format::{self, EncodeOptions, DecodeOptions, Delimiter, Indent};
use toon_format::types::KeyFoldingMode;

// ============================================================================
// TOON Value Types
// ============================================================================

/// TOON value type tags for binary encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SochTypeTag {
    /// Null value
    Null = 0x00,
    /// Boolean false
    False = 0x01,
    /// Boolean true  
    True = 0x02,
    /// Positive fixint (0-15, embedded in lower nibble: 0x10-0x1F)
    PosFixint = 0x10,
    /// Negative fixint (-16 to -1, embedded: 0x20-0x2F)
    NegFixint = 0x20,
    /// 8-bit signed integer
    Int8 = 0x30,
    /// 16-bit signed integer
    Int16 = 0x31,
    /// 32-bit signed integer
    Int32 = 0x32,
    /// 64-bit signed integer
    Int64 = 0x33,
    /// 32-bit float
    Float32 = 0x40,
    /// 64-bit float
    Float64 = 0x41,
    /// Fixed-length string (length in lower 4 bits: 0x50-0x5F, 0-15 chars)
    FixStr = 0x50,
    /// String with 8-bit length prefix
    Str8 = 0x60,
    /// String with 16-bit length prefix
    Str16 = 0x61,
    /// String with 32-bit length prefix
    Str32 = 0x62,
    /// Array
    Array = 0x70,
    /// Reference to another table row
    Ref = 0x80,
    /// Object (Map)
    Object = 0x90,
    /// Binary data
    Binary = 0xA0,
    /// Unsigned Integer (varint)
    UInt = 0xB0,
}

// ============================================================================
// TOON Document Structure
// ============================================================================

/// TOON document
#[derive(Debug, Clone)]
pub struct SochDocument {
    /// Root value
    pub root: SochValue,
    /// Schema version
    pub version: u32,
}

impl SochDocument {
    /// Create a new TOON document from a value
    pub fn new(root: SochValue) -> Self {
        Self {
            root,
            version: 1,
        }
    }

    /// Create a table-like document (legacy helper)
    pub fn new_table(_name: impl Into<String>, fields: Vec<String>, rows: Vec<Vec<SochValue>>) -> Self {
        // Convert to Array of Objects for canonical representation
        let fields_str: Vec<String> = fields;
        let mut array = Vec::new();
        for row in rows {
            let mut obj = HashMap::new();
            for (i, val) in row.into_iter().enumerate() {
                if i < fields_str.len() {
                    obj.insert(fields_str[i].clone(), val);
                }
            }
            array.push(SochValue::Object(obj));
        }
        
        Self {
            root: SochValue::Array(array),
            version: 1,
        }
    }
}

// ============================================================================
// Text Format (Human-Readable)
// ============================================================================

/// TOON text format encoder (wraps toon-format crate)
pub struct SochTextEncoder;

impl SochTextEncoder {
    /// Encode a document to TOON text format
    pub fn encode(doc: &SochDocument) -> String {
        // Use default options for now, can be sophisticated later
        let options = EncodeOptions::new()
            .with_indent(Indent::Spaces(2))
            .with_delimiter(Delimiter::Comma)
            .with_key_folding(KeyFoldingMode::Safe);
        
        // Use toon_format to encode the SochValue
        // SochValue implements Serialize, so this works directly.
        toon_format::encode(&doc.root, &options).unwrap_or_else(|e| format!("Error encoding TOON: {}", e))
    }
}

/// TOON text format decoder/parser (wraps toon-format crate)
pub struct SochTextParser;

impl SochTextParser {
    pub fn parse(input: &str) -> Result<SochDocument, SochParseError> {
         Self::parse_with_options(input, DecodeOptions::default())
    }
    
    pub fn parse_with_options(input: &str, options: DecodeOptions) -> Result<SochDocument, SochParseError> {
        let root: SochValue = toon_format::decode(input, &options)
            .map_err(|e| SochParseError::RowError { line: 0, cause: e.to_string() })?;
            
        Ok(SochDocument::new(root))
    }
    
    // Legacy helper kept for compatibility if needed, but useless now
    pub fn parse_header(_line: &str) -> Result<(String, usize, Vec<String>), SochParseError> {
        Err(SochParseError::InvalidHeader)
    }
}

/// Token counter (dummy implementation for now)
pub struct SochTokenCounter;
impl SochTokenCounter {
    pub fn count(_doc: &SochDocument) -> usize {
        0
    }
}


/// Parse error types
#[derive(Debug, Clone)]
pub enum SochParseError {
    EmptyInput,
    InvalidHeader,
    InvalidRowCount,
    InvalidValue,
    RowCountMismatch { expected: usize, actual: usize },
    FieldCountMismatch { expected: usize, actual: usize },
    RowError { line: usize, cause: String },
}

impl std::fmt::Display for SochParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for SochParseError {}


// ============================================================================
// Binary Format (Compact)
// ============================================================================

/// TOON binary format magic bytes
pub const TOON_MAGIC: [u8; 4] = [0x54, 0x4F, 0x4F, 0x4E]; // "TOON"

/// TOON binary codec (Renamed from SochBinaryCodec to SochDbBinaryCodec)
pub struct SochDbBinaryCodec;

impl SochDbBinaryCodec {
    /// Encode a document to binary format
    pub fn encode(doc: &SochDocument) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&TOON_MAGIC);
        // Version
        Self::write_varint(&mut buf, doc.version as u64);
        // Root value
        Self::write_value(&mut buf, &doc.root);
        // Checksum
        let checksum = crc32fast::hash(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Decode binary format to document
    pub fn decode(data: &[u8]) -> Result<SochDocument, SochParseError> {
        if data.len() < 8 { return Err(SochParseError::InvalidHeader); }
        if data[0..4] != TOON_MAGIC { return Err(SochParseError::InvalidHeader); }
        
        // Verify checksum
        let stored_checksum = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        let computed_checksum = crc32fast::hash(&data[..data.len() - 4]);
        if stored_checksum != computed_checksum { return Err(SochParseError::InvalidValue); }
        
        let data = &data[..data.len() - 4];
        let mut cursor = 4;
        
        let (version, bytes) = Self::read_varint(&data[cursor..])?;
        cursor += bytes;
        
        let (root, _) = Self::read_value(&data[cursor..])?;
        
        Ok(SochDocument {
            root,
            version: version as u32,
        })
    }
    
    fn write_varint(buf: &mut Vec<u8>, mut n: u64) {
        while n > 127 {
            buf.push((n as u8 & 0x7F) | 0x80);
            n >>= 7;
        }
        buf.push(n as u8 & 0x7F);
    }
    
    fn read_varint(data: &[u8]) -> Result<(u64, usize), SochParseError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut i = 0;
        while i < data.len() {
            let byte = data[i];
            result |= ((byte & 0x7F) as u64) << shift;
            i += 1;
            if byte & 0x80 == 0 { return Ok((result, i)); }
            shift += 7;
        }
        Err(SochParseError::InvalidValue)
    }

    fn read_string(data: &[u8]) -> Result<(String, usize), SochParseError> {
        let (len, varint_bytes) = Self::read_varint(data)?;
        let len = len as usize;
        if data.len() < varint_bytes + len { return Err(SochParseError::InvalidValue); }
        let s = std::str::from_utf8(&data[varint_bytes..varint_bytes+len]).map_err(|_| SochParseError::InvalidValue)?.to_string();
        Ok((s, varint_bytes + len))
    }
    
    fn write_value(buf: &mut Vec<u8>, value: &SochValue) {
        match value {
            SochValue::Null => buf.push(SochTypeTag::Null as u8),
            SochValue::Bool(true) => buf.push(SochTypeTag::True as u8),
            SochValue::Bool(false) => buf.push(SochTypeTag::False as u8),
            SochValue::Int(n) => {
                 // Optimization: FixInts
                 buf.push(SochTypeTag::Int64 as u8);
                 buf.extend_from_slice(&n.to_le_bytes());
            },
            SochValue::UInt(n) => {
                 buf.push(SochTypeTag::UInt as u8);
                 Self::write_varint(buf, *n);
            },
            SochValue::Float(f) => {
                 buf.push(SochTypeTag::Float64 as u8);
                 buf.extend_from_slice(&f.to_le_bytes());
            },
            SochValue::Text(s) => {
                 buf.push(SochTypeTag::Str32 as u8);
                 let bytes = s.as_bytes();
                 buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                 buf.extend_from_slice(bytes);
            },
            SochValue::Binary(b) => {
                 buf.push(SochTypeTag::Binary as u8);
                 Self::write_varint(buf, b.len() as u64);
                 buf.extend_from_slice(b);
            },
            SochValue::Array(arr) => {
                 buf.push(SochTypeTag::Array as u8);
                 Self::write_varint(buf, arr.len() as u64);
                 for item in arr { Self::write_value(buf, item); }
            },
            SochValue::Object(map) => {
                 buf.push(SochTypeTag::Object as u8);
                 Self::write_varint(buf, map.len() as u64);
                 for (k, v) in map {
                     // Key string
                     let k_bytes = k.as_bytes();
                     Self::write_varint(buf, k_bytes.len() as u64);
                     buf.extend_from_slice(k_bytes);
                     // Value
                     Self::write_value(buf, v);
                 }
            },
            SochValue::Ref { table, id } => {
                 buf.push(SochTypeTag::Ref as u8);
                 // table name
                 let t_bytes = table.as_bytes();
                 Self::write_varint(buf, t_bytes.len() as u64);
                 buf.extend_from_slice(t_bytes);
                 // id
                 Self::write_varint(buf, *id);
            }
        }
    }
    
    fn read_value(data: &[u8]) -> Result<(SochValue, usize), SochParseError> {
        if data.is_empty() { return Err(SochParseError::InvalidValue); }
        let tag = data[0];
        let mut cursor = 1;
        
        match tag {
            0x00 => Ok((SochValue::Null, 1)),
            0x01 => Ok((SochValue::Bool(false), 1)),
            0x02 => Ok((SochValue::Bool(true), 1)),
            0x33 => { // Int64
                 if data.len() < cursor + 8 { return Err(SochParseError::InvalidValue); }
                 let n = i64::from_le_bytes(data[cursor..cursor+8].try_into().unwrap());
                 Ok((SochValue::Int(n), cursor+8))
            },
            0x41 => { // Float64
                 if data.len() < cursor + 8 { return Err(SochParseError::InvalidValue); }
                 let f = f64::from_le_bytes(data[cursor..cursor+8].try_into().unwrap());
                 Ok((SochValue::Float(f), cursor+8))
            },
            0x62 => { // Str32
                 if data.len() < cursor + 4 { return Err(SochParseError::InvalidValue); }
                 let len = u32::from_le_bytes(data[cursor..cursor+4].try_into().unwrap()) as usize;
                 cursor += 4;
                 if data.len() < cursor + len { return Err(SochParseError::InvalidValue); }
                 let s = std::str::from_utf8(&data[cursor..cursor+len]).unwrap().to_string();
                 Ok((SochValue::Text(s), cursor+len))
            },
            0x70 => { // Array
                 let (len, bytes) = Self::read_varint(&data[cursor..])?;
                 cursor += bytes;
                 let mut arr = Vec::new();
                 for _ in 0..len {
                     let (val, bytes_read) = Self::read_value(&data[cursor..])?;
                     cursor += bytes_read;
                     arr.push(val);
                 }
                 Ok((SochValue::Array(arr), cursor))
            },
            0xB0 => { // UInt
                 let (n, bytes) = Self::read_varint(&data[cursor..])?;
                 Ok((SochValue::UInt(n), cursor+bytes))
            },
            0x80 => { // Ref
                 let (table, table_bytes) = Self::read_string(&data[cursor..])?;
                 cursor += table_bytes;
                 let (id, id_bytes) = Self::read_varint(&data[cursor..])?;
                 Ok((SochValue::Ref { table, id }, cursor+id_bytes))
            },
            0x90 => { // Object
                 let (len, bytes_read) = Self::read_varint(&data[cursor..])?;
                 cursor += bytes_read;
                 let mut map = HashMap::new();
                 for _ in 0..len {
                     let (k, k_bytes) = Self::read_string(&data[cursor..])?;
                     cursor += k_bytes;
                     let (v, v_bytes) = Self::read_value(&data[cursor..])?;
                     cursor += v_bytes;
                     map.insert(k, v);
                 }
                 Ok((SochValue::Object(map), cursor))
            },
            // Add other cases as needed
            _ => Err(SochParseError::InvalidValue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_object() {
        let mut obj = HashMap::new();
        obj.insert("id".to_string(), SochValue::Int(1));
        obj.insert("name".to_string(), SochValue::Text("Alice".to_string()));
        let doc = SochDocument::new(SochValue::Object(obj));
        
        // This test now uses canonical encoder
        let encoded = SochTextEncoder::encode(&doc);
        // Canonical output might differ slightly (e.g. sorting), but should contain keys
        assert!(encoded.contains("id"));
        assert!(encoded.contains("1"));
        assert!(encoded.contains("name"));
        assert!(encoded.contains("Alice"));
        
        // Roundtrip binary with new codec name
        let bin = SochDbBinaryCodec::encode(&doc);
        let decoded = SochDbBinaryCodec::decode(&bin).unwrap();
        if let SochValue::Object(map) = decoded.root {
             // Accessing values. Note: SochValue doesn't impl PartialEq against literal ints easily matching on variant needed
             // Use string representation or direct match
            assert_eq!(map.get("id"), Some(&SochValue::Int(1)));
            assert_eq!(map.get("name"), Some(&SochValue::Text("Alice".to_string())));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_array() {
        let arr = vec![
            SochValue::Int(1),
            SochValue::Int(2),
        ];
        let doc = SochDocument::new(SochValue::Array(arr));
        
        let encoded = SochTextEncoder::encode(&doc);
        // Should contain values
        assert!(encoded.contains("1"));
        assert!(encoded.contains("2"));
        
        let bin = SochDbBinaryCodec::encode(&doc);
        let decoded = SochDbBinaryCodec::decode(&bin).unwrap();
        if let SochValue::Array(arr) = decoded.root {
             assert_eq!(arr.len(), 2);
             assert_eq!(arr[0], SochValue::Int(1));
        } else {
            panic!("Expected array");
        }
    }
}
