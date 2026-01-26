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

//! TOON (Tabular Object-Oriented Notation) - Native Data Format for SochDB
//!
//! TOON is a compact, schema-aware data format optimized for LLMs and databases.
//! It's the native format for SochDB, like JSON is for MongoDB.
//!
//! Format: `name[count]{fields}:\nrow1\nrow2\n...`
//!
//! Example:
//! ```text
//! users[3]{id,name,email}:
//! 1,Alice,alice@example.com
//! 2,Bob,bob@example.com
//! 3,Charlie,charlie@example.com
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// TOON Value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SochValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    Binary(Vec<u8>),
    Array(Vec<SochValue>),
    Object(HashMap<String, SochValue>),
    /// Reference to another table row: ref(table_name, id)
    Ref {
        table: String,
        id: u64,
    },
}

impl SochValue {
    pub fn is_null(&self) -> bool {
        matches!(self, SochValue::Null)
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            SochValue::Int(v) => Some(*v),
            SochValue::UInt(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_uint(&self) -> Option<u64> {
        match self {
            SochValue::UInt(v) => Some(*v),
            SochValue::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            SochValue::Float(v) => Some(*v),
            SochValue::Int(v) => Some(*v as f64),
            SochValue::UInt(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            SochValue::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SochValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

fn needs_quoting(s: &str) -> bool {
    if s.is_empty() { return true; }
    if s.starts_with(' ') || s.ends_with(' ') { return true; }
    if matches!(s, "true" | "false" | "null") { return true; }
    
    // Check for number-like patterns
    if s.parse::<f64>().is_ok() { return true; }
    if s == "-" || s.starts_with('-') { return true; }
    // Leading zeros check (e.g. 05 usually treated as number in some contexts or invalid)
    if s.len() > 1 && s.starts_with('0') && s.chars().nth(1).map_or(false, |c| c.is_ascii_digit()) && !s.contains('.') {
        return true;
    }

    // Check for special chars or delimiter (comma)
    // Spec §7.3: :, ", \, [, ], {, }, newline, return, tab, delimiter
    s.contains(|c| matches!(c, ':' | '"' | '\\' | '[' | ']' | '{' | '}' | '\n' | '\r' | '\t' | ','))
}

impl fmt::Display for SochValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SochValue::Null => write!(f, "null"),
            SochValue::Bool(b) => write!(f, "{}", b),
            SochValue::Int(i) => write!(f, "{}", i),
            SochValue::UInt(u) => write!(f, "{}", u),
            SochValue::Float(fl) => write!(f, "{}", fl),
            SochValue::Text(s) => {
                if needs_quoting(s) {
                    write!(f, "\"")?;
                    for c in s.chars() {
                        match c {
                            '"' => write!(f, "\\\"")?,
                            '\\' => write!(f, "\\\\")?,
                            '\n' => write!(f, "\\n")?,
                            '\r' => write!(f, "\\r")?,
                            '\t' => write!(f, "\\t")?,
                            c => write!(f, "{}", c)?,
                        }
                    }
                    write!(f, "\"")
                } else {
                    write!(f, "{}", s)
                }
            }
            SochValue::Binary(b) => write!(f, "0x{}", hex::encode(b)),
            SochValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            SochValue::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{}:{}", k, v)?;
                }
                write!(f, "}}")
            }
            SochValue::Ref { table, id } => write!(f, "@{}:{}", table, id),
        }
    }
}

/// Field type in a TOON schema
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SochType {
    Null,
    Bool,
    Int,
    UInt,
    Float,
    Text,
    Binary,
    Array(Box<SochType>),
    Object(Vec<(String, SochType)>),
    Ref(String), // Reference to table name
    /// Union of types (for nullable fields)
    Optional(Box<SochType>),
}

impl SochType {
    /// Check if a value matches this type
    pub fn matches(&self, value: &SochValue) -> bool {
        match (self, value) {
            (SochType::Null, SochValue::Null) => true,
            (SochType::Bool, SochValue::Bool(_)) => true,
            (SochType::Int, SochValue::Int(_)) => true,
            (SochType::UInt, SochValue::UInt(_)) => true,
            (SochType::Float, SochValue::Float(_)) => true,
            (SochType::Text, SochValue::Text(_)) => true,
            (SochType::Binary, SochValue::Binary(_)) => true,
            (SochType::Array(inner), SochValue::Array(arr)) => arr.iter().all(|v| inner.matches(v)),
            (SochType::Ref(table), SochValue::Ref { table: t, .. }) => table == t,
            (SochType::Optional(inner), value) => value.is_null() || inner.matches(value),
            _ => false,
        }
    }

    /// Parse type from string notation
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        match s {
            "null" => Some(SochType::Null),
            "bool" => Some(SochType::Bool),
            "int" | "i64" => Some(SochType::Int),
            "uint" | "u64" => Some(SochType::UInt),
            "float" | "f64" => Some(SochType::Float),
            "text" | "string" => Some(SochType::Text),
            "binary" | "bytes" => Some(SochType::Binary),
            _ if s.starts_with("ref(") && s.ends_with(')') => {
                let table = &s[4..s.len() - 1];
                Some(SochType::Ref(table.to_string()))
            }
            _ if s.starts_with("array(") && s.ends_with(')') => {
                let inner = &s[6..s.len() - 1];
                SochType::parse(inner).map(|t| SochType::Array(Box::new(t)))
            }
            _ if s.ends_with('?') => {
                let inner = &s[..s.len() - 1];
                SochType::parse(inner).map(|t| SochType::Optional(Box::new(t)))
            }
            _ => None,
        }
    }
}

impl fmt::Display for SochType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SochType::Null => write!(f, "null"),
            SochType::Bool => write!(f, "bool"),
            SochType::Int => write!(f, "int"),
            SochType::UInt => write!(f, "uint"),
            SochType::Float => write!(f, "float"),
            SochType::Text => write!(f, "text"),
            SochType::Binary => write!(f, "binary"),
            SochType::Array(inner) => write!(f, "array({})", inner),
            SochType::Object(fields) => {
                write!(f, "{{")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}:{}", name, ty)?;
                }
                write!(f, "}}")
            }
            SochType::Ref(table) => write!(f, "ref({})", table),
            SochType::Optional(inner) => write!(f, "{}?", inner),
        }
    }
}

/// A TOON schema definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SochSchema {
    /// Schema name (table name)
    pub name: String,
    /// Field definitions
    pub fields: Vec<SochField>,
    /// Primary key field name
    pub primary_key: Option<String>,
    /// Indexes on this schema
    pub indexes: Vec<SochIndex>,
}

/// A field in a TOON schema
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SochField {
    pub name: String,
    pub field_type: SochType,
    pub nullable: bool,
    pub default: Option<String>, // Default value as TOON string
}

/// An index definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SochIndex {
    pub name: String,
    pub fields: Vec<String>,
    pub unique: bool,
}

impl SochSchema {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            primary_key: None,
            indexes: Vec::new(),
        }
    }

    pub fn field(mut self, name: impl Into<String>, field_type: SochType) -> Self {
        self.fields.push(SochField {
            name: name.into(),
            field_type,
            nullable: false,
            default: None,
        });
        self
    }

    pub fn nullable_field(mut self, name: impl Into<String>, field_type: SochType) -> Self {
        self.fields.push(SochField {
            name: name.into(),
            field_type,
            nullable: true,
            default: None,
        });
        self
    }

    pub fn primary_key(mut self, field: impl Into<String>) -> Self {
        self.primary_key = Some(field.into());
        self
    }

    pub fn index(mut self, name: impl Into<String>, fields: Vec<String>, unique: bool) -> Self {
        self.indexes.push(SochIndex {
            name: name.into(),
            fields,
            unique,
        });
        self
    }

    /// Get field names for header
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Format schema header: name[0]{field1,field2,...}:
    pub fn format_header(&self) -> String {
        let fields: Vec<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
        format!("{}[0]{{{}}}:", self.name, fields.join(","))
    }
}

/// A TOON row - values for a single record
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SochRow {
    pub values: Vec<SochValue>,
}

impl SochRow {
    pub fn new(values: Vec<SochValue>) -> Self {
        Self { values }
    }

    /// Get value by index
    pub fn get(&self, index: usize) -> Option<&SochValue> {
        self.values.get(index)
    }

    /// Format row as TOON line
    pub fn format(&self) -> String {
        self.values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parse row from TOON line
    pub fn parse(line: &str, schema: &SochSchema) -> Result<Self, String> {
        let mut values = Vec::with_capacity(schema.fields.len());
        let mut chars = line.chars().peekable();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut field_idx = 0;

        while let Some(ch) = chars.next() {
            match ch {
                '"' if !in_quotes => {
                    in_quotes = true;
                }
                '"' if in_quotes => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        current.push('"');
                    } else {
                        in_quotes = false;
                    }
                }
                ',' if !in_quotes => {
                    let value = Self::parse_value(&current, field_idx, schema)?;
                    values.push(value);
                    current.clear();
                    field_idx += 1;
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        // Last field
        if !current.is_empty() || field_idx < schema.fields.len() {
            let value = Self::parse_value(&current, field_idx, schema)?;
            values.push(value);
        }

        Ok(Self { values })
    }

    fn parse_value(s: &str, field_idx: usize, schema: &SochSchema) -> Result<SochValue, String> {
        let s = s.trim();

        if s.is_empty() || s == "null" {
            return Ok(SochValue::Null);
        }

        let field = schema
            .fields
            .get(field_idx)
            .ok_or_else(|| format!("Field index {} out of bounds", field_idx))?;

        match &field.field_type {
            SochType::Bool => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" => Ok(SochValue::Bool(true)),
                "false" | "0" | "no" => Ok(SochValue::Bool(false)),
                _ => Err(format!("Invalid bool: {}", s)),
            },
            SochType::Int => s
                .parse::<i64>()
                .map(SochValue::Int)
                .map_err(|e| format!("Invalid int: {}", e)),
            SochType::UInt => s
                .parse::<u64>()
                .map(SochValue::UInt)
                .map_err(|e| format!("Invalid uint: {}", e)),
            SochType::Float => s
                .parse::<f64>()
                .map(SochValue::Float)
                .map_err(|e| format!("Invalid float: {}", e)),
            SochType::Text => Ok(SochValue::Text(s.to_string())),
            SochType::Binary => {
                if let Some(hex_str) = s.strip_prefix("0x") {
                    hex::decode(hex_str)
                        .map(SochValue::Binary)
                        .map_err(|e| format!("Invalid hex: {}", e))
                } else {
                    Err("Binary must start with 0x".to_string())
                }
            }
            SochType::Ref(table) => {
                // Format: @table:id or just id
                if let Some(ref_str) = s.strip_prefix('@') {
                    let parts: Vec<&str> = ref_str.split(':').collect();
                    if parts.len() == 2 {
                        let id = parts[1]
                            .parse::<u64>()
                            .map_err(|e| format!("Invalid ref id: {}", e))?;
                        Ok(SochValue::Ref {
                            table: parts[0].to_string(),
                            id,
                        })
                    } else {
                        Err(format!("Invalid ref format: {}", s))
                    }
                } else {
                    let id = s
                        .parse::<u64>()
                        .map_err(|e| format!("Invalid ref id: {}", e))?;
                    Ok(SochValue::Ref {
                        table: table.clone(),
                        id,
                    })
                }
            }
            SochType::Optional(inner) => {
                // Try to parse as inner type
                let temp_field = SochField {
                    name: field.name.clone(),
                    field_type: (**inner).clone(),
                    nullable: true,
                    default: None,
                };
                let temp_schema = SochSchema {
                    name: schema.name.clone(),
                    fields: vec![temp_field],
                    primary_key: None,
                    indexes: vec![],
                };
                Self::parse_value(s, 0, &temp_schema)
            }
            _ => Ok(SochValue::Text(s.to_string())),
        }
    }
}

/// A complete TOON table (header + rows)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SochTable {
    pub schema: SochSchema,
    pub rows: Vec<SochRow>,
}

impl SochTable {
    pub fn new(schema: SochSchema) -> Self {
        Self {
            schema,
            rows: Vec::new(),
        }
    }

    pub fn with_rows(schema: SochSchema, rows: Vec<SochRow>) -> Self {
        Self { schema, rows }
    }

    pub fn push(&mut self, row: SochRow) {
        self.rows.push(row);
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Format as TOON string
    pub fn format(&self) -> String {
        let fields: Vec<&str> = self.schema.fields.iter().map(|f| f.name.as_str()).collect();
        let header = format!(
            "{}[{}]{{{}}}:",
            self.schema.name,
            self.rows.len(),
            fields.join(",")
        );

        let mut output = header;
        for row in &self.rows {
            output.push('\n');
            output.push_str(&row.format());
        }
        output
    }

    /// Parse TOON string to table
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut lines = input.lines();

        // Parse header: name[count]{field1,field2,...}:
        let header = lines.next().ok_or("Empty input")?;
        let (schema, _count) = Self::parse_header(header)?;

        // Parse rows
        let mut rows = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let row = SochRow::parse(line, &schema)?;
            rows.push(row);
        }

        Ok(Self { schema, rows })
    }

    fn parse_header(header: &str) -> Result<(SochSchema, usize), String> {
        // name[count]{field1,field2,...}:
        let header = header.trim_end_matches(':');

        let bracket_start = header.find('[').ok_or("Missing [")?;
        let bracket_end = header.find(']').ok_or("Missing ]")?;
        let brace_start = header.find('{').ok_or("Missing {")?;
        let brace_end = header.find('}').ok_or("Missing }")?;

        let name = &header[..bracket_start];
        let count_str = &header[bracket_start + 1..bracket_end];
        let fields_str = &header[brace_start + 1..brace_end];

        let count = count_str
            .parse::<usize>()
            .map_err(|e| format!("Invalid count: {}", e))?;

        let field_names: Vec<&str> = fields_str.split(',').map(|s| s.trim()).collect();

        let mut schema = SochSchema::new(name);
        for field_name in field_names {
            // Check if type is specified: field_name:type
            if let Some(colon_pos) = field_name.find(':') {
                let fname = &field_name[..colon_pos];
                let ftype_str = &field_name[colon_pos + 1..];
                let ftype = SochType::parse(ftype_str).unwrap_or(SochType::Text);
                schema.fields.push(SochField {
                    name: fname.to_string(),
                    field_type: ftype,
                    nullable: false,
                    default: None,
                });
            } else {
                // Default to text type
                schema.fields.push(SochField {
                    name: field_name.to_string(),
                    field_type: SochType::Text,
                    nullable: false,
                    default: None,
                });
            }
        }

        Ok((schema, count))
    }
}

impl fmt::Display for SochTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Trait for accessing columnar data without allocation
pub trait ColumnAccess {
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn field_names(&self) -> Vec<&str>;
    fn write_value(
        &self,
        col_idx: usize,
        row_idx: usize,
        f: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result;
}

/// Cursor for iterating over columnar data and emitting TOON format
pub struct SochCursor<'a, C: ColumnAccess> {
    access: &'a C,
    current_row: usize,
    header_emitted: bool,
    schema_name: String,
}

impl<'a, C: ColumnAccess> SochCursor<'a, C> {
    pub fn new(access: &'a C, schema_name: String) -> Self {
        Self {
            access,
            current_row: 0,
            header_emitted: false,
            schema_name,
        }
    }
}

impl<'a, C: ColumnAccess> Iterator for SochCursor<'a, C> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.header_emitted {
            self.header_emitted = true;
            let fields = self.access.field_names().join(",");
            return Some(format!(
                "{}[{}]{{{}}}:",
                self.schema_name,
                self.access.row_count(),
                fields
            ));
        }

        if self.current_row >= self.access.row_count() {
            return None;
        }

        let mut row_str = String::new();
        for col_idx in 0..self.access.col_count() {
            if col_idx > 0 {
                row_str.push(',');
            }
            // We ignore write errors here as String write shouldn't fail
            let _ = self
                .access
                .write_value(col_idx, self.current_row, &mut row_str);
        }

        self.current_row += 1;
        Some(row_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soch_value_display() {
        assert_eq!(SochValue::Int(42).to_string(), "42");
        assert_eq!(SochValue::Text("hello".into()).to_string(), "hello");
        assert_eq!(
            SochValue::Text("hello, world".into()).to_string(),
            "\"hello, world\""
        );
        assert_eq!(SochValue::Bool(true).to_string(), "true");
        assert_eq!(SochValue::Null.to_string(), "null");
    }

    #[test]
    fn test_soch_schema() {
        let schema = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("email", SochType::Text)
            .primary_key("id");

        assert_eq!(schema.name, "users");
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.primary_key, Some("id".to_string()));
    }

    #[test]
    fn test_soch_table_format() {
        let schema = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("email", SochType::Text);

        let mut table = SochTable::new(schema);
        table.push(SochRow::new(vec![
            SochValue::UInt(1),
            SochValue::Text("Alice".into()),
            SochValue::Text("alice@example.com".into()),
        ]));
        table.push(SochRow::new(vec![
            SochValue::UInt(2),
            SochValue::Text("Bob".into()),
            SochValue::Text("bob@example.com".into()),
        ]));

        let formatted = table.format();
        assert!(formatted.contains("users[2]{id,name,email}:"));
        assert!(formatted.contains("1,Alice,alice@example.com"));
        assert!(formatted.contains("2,Bob,bob@example.com"));
    }

    #[test]
    fn test_soch_table_parse() {
        let input = r#"users[2]{id,name,email}:
1,Alice,alice@example.com
2,Bob,bob@example.com"#;

        let table = SochTable::parse(input).unwrap();
        assert_eq!(table.schema.name, "users");
        assert_eq!(table.rows.len(), 2);
    }

    #[test]
    fn test_soch_type_parse() {
        assert_eq!(SochType::parse("int"), Some(SochType::Int));
        assert_eq!(SochType::parse("text"), Some(SochType::Text));
        assert_eq!(
            SochType::parse("ref(users)"),
            Some(SochType::Ref("users".into()))
        );
        assert_eq!(
            SochType::parse("int?"),
            Some(SochType::Optional(Box::new(SochType::Int)))
        );
    }
}
