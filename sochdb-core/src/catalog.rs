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

//! Schema Catalog for SochDB
//!
//! Manages table schemas, indexes, operations, and database metadata.
//! The catalog itself is stored as a TOON document.
//!
//! ## MCP Integration (Task 7)
//!
//! Operations stored in the catalog can be exposed as MCP tools:
//! - Input/output schemas defined as TOON schemas
//! - Built-in, SOCH-QL, or external implementations
//! - Token savings: ~60% via TOON responses

use crate::soch::SochSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Catalog entry type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatalogEntryType {
    Table,
    Index,
    View,
    Sequence,
    /// MCP-compatible operation (Task 7)
    Operation,
}

/// Operation implementation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationImpl {
    /// Built-in operation (e.g., range_scan, semantic_search)
    BuiltIn(String),
    /// Stored procedure as SOCH-QL query
    SochQL(String),
    /// External function reference
    External(String),
}

/// A catalog entry (table, index, operation, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Entry name
    pub name: String,
    /// Entry type
    pub entry_type: CatalogEntryType,
    /// Schema definition (for tables)
    pub schema: Option<SochSchema>,
    /// Input schema (for operations)
    pub input_schema: Option<SochSchema>,
    /// Output schema (for operations)
    pub output_schema: Option<SochSchema>,
    /// Operation implementation (for operations)
    pub implementation: Option<OperationImpl>,
    /// Description (for MCP tool generation)
    pub description: Option<String>,
    /// Root page/block ID for data storage
    pub root_id: u64,
    /// Creation timestamp
    pub created_us: u64,
    /// Last modified timestamp
    pub modified_us: u64,
    /// Row count estimate
    pub row_count: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CatalogEntry {
    pub fn new_table(name: impl Into<String>, schema: SochSchema, root_id: u64) -> Self {
        let now = now_micros();
        Self {
            name: name.into(),
            entry_type: CatalogEntryType::Table,
            schema: Some(schema),
            input_schema: None,
            output_schema: None,
            implementation: None,
            description: None,
            root_id,
            created_us: now,
            modified_us: now,
            row_count: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn new_index(
        name: impl Into<String>,
        table_name: impl Into<String>,
        fields: Vec<String>,
        unique: bool,
        root_id: u64,
    ) -> Self {
        let now = now_micros();
        let mut metadata = HashMap::new();
        metadata.insert("table".to_string(), table_name.into());
        metadata.insert("fields".to_string(), fields.join(","));
        metadata.insert("unique".to_string(), unique.to_string());

        Self {
            name: name.into(),
            entry_type: CatalogEntryType::Index,
            schema: None,
            input_schema: None,
            output_schema: None,
            implementation: None,
            description: None,
            root_id,
            created_us: now,
            modified_us: now,
            row_count: 0,
            metadata,
        }
    }

    /// Create a new operation entry (Task 7: MCP Integration)
    pub fn new_operation(
        name: impl Into<String>,
        input_schema: SochSchema,
        output_schema: SochSchema,
        implementation: OperationImpl,
        description: impl Into<String>,
    ) -> Self {
        let now = now_micros();
        Self {
            name: name.into(),
            entry_type: CatalogEntryType::Operation,
            schema: None,
            input_schema: Some(input_schema),
            output_schema: Some(output_schema),
            implementation: Some(implementation),
            description: Some(description.into()),
            root_id: 0,
            created_us: now,
            modified_us: now,
            row_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Generate MCP tool descriptor from operation entry
    pub fn to_mcp_tool(&self) -> Option<McpToolDescriptor> {
        if self.entry_type != CatalogEntryType::Operation {
            return None;
        }

        Some(McpToolDescriptor {
            name: self.name.clone(),
            description: self.description.clone().unwrap_or_default(),
            input_schema: self.input_schema.as_ref()?.clone(),
            output_schema: self.output_schema.as_ref()?.clone(),
        })
    }
}

/// MCP Tool Descriptor for LLM tool calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input parameter schema
    pub input_schema: SochSchema,
    /// Output result schema
    pub output_schema: SochSchema,
}

/// The Schema Catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    /// Database name
    pub name: String,
    /// Version number
    pub version: u64,
    /// All catalog entries
    pub entries: HashMap<String, CatalogEntry>,
    /// Next auto-increment ID for each table
    pub auto_increment: HashMap<String, u64>,
}

impl Catalog {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: 1,
            entries: HashMap::new(),
            auto_increment: HashMap::new(),
        }
    }

    /// Create a new table
    pub fn create_table(&mut self, schema: SochSchema, root_id: u64) -> Result<(), String> {
        if self.entries.contains_key(&schema.name) {
            return Err(format!("Table '{}' already exists", schema.name));
        }

        let name = schema.name.clone();
        let entry = CatalogEntry::new_table(&name, schema, root_id);
        self.entries.insert(name.clone(), entry);
        self.auto_increment.insert(name, 0);
        self.version += 1;
        Ok(())
    }

    /// Drop a table
    pub fn drop_table(&mut self, name: &str) -> Result<CatalogEntry, String> {
        // Remove associated indexes first
        let indexes_to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                e.entry_type == CatalogEntryType::Index
                    && e.metadata.get("table") == Some(&name.to_string())
            })
            .map(|(k, _)| k.clone())
            .collect();

        for idx in indexes_to_remove {
            self.entries.remove(&idx);
        }

        self.auto_increment.remove(name);
        self.entries
            .remove(name)
            .ok_or_else(|| format!("Table '{}' not found", name))
    }

    /// Get a table schema
    pub fn get_table(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries
            .get(name)
            .filter(|e| e.entry_type == CatalogEntryType::Table)
    }

    /// Get a mutable table entry
    pub fn get_table_mut(&mut self, name: &str) -> Option<&mut CatalogEntry> {
        self.entries
            .get_mut(name)
            .filter(|e| e.entry_type == CatalogEntryType::Table)
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, e)| e.entry_type == CatalogEntryType::Table)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Create an index
    pub fn create_index(
        &mut self,
        name: impl Into<String>,
        table_name: &str,
        fields: Vec<String>,
        unique: bool,
        root_id: u64,
    ) -> Result<(), String> {
        let name = name.into();

        if !self.entries.contains_key(table_name) {
            return Err(format!("Table '{}' not found", table_name));
        }

        if self.entries.contains_key(&name) {
            return Err(format!("Index '{}' already exists", name));
        }

        // Validate fields exist in table
        if let Some(entry) = self.get_table(table_name)
            && let Some(schema) = &entry.schema
        {
            for field in &fields {
                if !schema.fields.iter().any(|f| &f.name == field) {
                    return Err(format!(
                        "Field '{}' not found in table '{}'",
                        field, table_name
                    ));
                }
            }
        }

        let entry = CatalogEntry::new_index(&name, table_name, fields, unique, root_id);
        self.entries.insert(name, entry);
        self.version += 1;
        Ok(())
    }

    /// Drop an index
    pub fn drop_index(&mut self, name: &str) -> Result<CatalogEntry, String> {
        if let Some(entry) = self.entries.get(name)
            && entry.entry_type != CatalogEntryType::Index
        {
            return Err(format!("'{}' is not an index", name));
        }
        self.entries
            .remove(name)
            .ok_or_else(|| format!("Index '{}' not found", name))
    }

    /// Get indexes for a table
    pub fn get_indexes(&self, table_name: &str) -> Vec<&CatalogEntry> {
        self.entries
            .values()
            .filter(|e| {
                e.entry_type == CatalogEntryType::Index
                    && e.metadata.get("table") == Some(&table_name.to_string())
            })
            .collect()
    }

    /// Get next auto-increment value
    pub fn next_auto_increment(&mut self, table_name: &str) -> u64 {
        let value = self
            .auto_increment
            .entry(table_name.to_string())
            .or_insert(0);
        *value += 1;
        *value
    }

    /// Update row count for a table
    pub fn update_row_count(&mut self, table_name: &str, count: u64) {
        if let Some(entry) = self.entries.get_mut(table_name) {
            entry.row_count = count;
            entry.modified_us = now_micros();
        }
    }

    /// Serialize catalog to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Deserialize catalog from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }

    /// Format catalog as TOON
    pub fn to_toon(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "soch_catalog[{}]{{name,type,fields,root,rows}}:",
            self.entries.len()
        ));

        for (name, entry) in &self.entries {
            let entry_type = match entry.entry_type {
                CatalogEntryType::Table => "table",
                CatalogEntryType::Index => "index",
                CatalogEntryType::View => "view",
                CatalogEntryType::Sequence => "sequence",
                CatalogEntryType::Operation => "operation",
            };

            let fields = if let Some(schema) = &entry.schema {
                schema
                    .fields
                    .iter()
                    .map(|f| format!("{}:{}", f.name, f.field_type))
                    .collect::<Vec<_>>()
                    .join(";")
            } else if let Some(input) = &entry.input_schema {
                // For operations, show input schema
                input
                    .fields
                    .iter()
                    .map(|f| format!("{}:{}", f.name, f.field_type))
                    .collect::<Vec<_>>()
                    .join(";")
            } else {
                entry.metadata.get("fields").cloned().unwrap_or_default()
            };

            lines.push(format!(
                "{},{},\"{}\",{},{}",
                name, entry_type, fields, entry.root_id, entry.row_count
            ));
        }

        lines.join("\n")
    }

    /// Create an operation (Task 7: MCP Integration)
    pub fn create_operation(
        &mut self,
        name: impl Into<String>,
        input_schema: SochSchema,
        output_schema: SochSchema,
        implementation: OperationImpl,
        description: impl Into<String>,
    ) -> Result<(), String> {
        let name = name.into();

        if self.entries.contains_key(&name) {
            return Err(format!("Operation '{}' already exists", name));
        }

        let entry = CatalogEntry::new_operation(
            &name,
            input_schema,
            output_schema,
            implementation,
            description,
        );
        self.entries.insert(name, entry);
        self.version += 1;
        Ok(())
    }

    /// Get an operation
    pub fn get_operation(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries
            .get(name)
            .filter(|e| e.entry_type == CatalogEntryType::Operation)
    }

    /// List all operations (for MCP tool discovery)
    pub fn list_operations(&self) -> Vec<&CatalogEntry> {
        self.entries
            .values()
            .filter(|e| e.entry_type == CatalogEntryType::Operation)
            .collect()
    }

    /// Generate all MCP tool descriptors
    pub fn generate_mcp_tools(&self) -> Vec<McpToolDescriptor> {
        self.list_operations()
            .iter()
            .filter_map(|e| e.to_mcp_tool())
            .collect()
    }
}

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soch::SochType;

    #[test]
    fn test_create_table() {
        let mut catalog = Catalog::new("test_db");

        let schema = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("email", SochType::Text)
            .primary_key("id");

        catalog.create_table(schema, 1).unwrap();

        assert!(catalog.get_table("users").is_some());
        assert_eq!(catalog.list_tables(), vec!["users"]);
    }

    #[test]
    fn test_create_index() {
        let mut catalog = Catalog::new("test_db");

        let schema = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("email", SochType::Text);

        catalog.create_table(schema, 1).unwrap();
        catalog
            .create_index("idx_users_email", "users", vec!["email".into()], true, 2)
            .unwrap();

        let indexes = catalog.get_indexes("users");
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].name, "idx_users_email");
    }

    #[test]
    fn test_auto_increment() {
        let mut catalog = Catalog::new("test_db");

        let schema = SochSchema::new("users").field("id", SochType::UInt);
        catalog.create_table(schema, 1).unwrap();

        assert_eq!(catalog.next_auto_increment("users"), 1);
        assert_eq!(catalog.next_auto_increment("users"), 2);
        assert_eq!(catalog.next_auto_increment("users"), 3);
    }

    #[test]
    fn test_drop_table_removes_indexes() {
        let mut catalog = Catalog::new("test_db");

        let schema = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("email", SochType::Text);

        catalog.create_table(schema, 1).unwrap();
        catalog
            .create_index("idx_users_email", "users", vec!["email".into()], true, 2)
            .unwrap();

        catalog.drop_table("users").unwrap();

        assert!(catalog.get_table("users").is_none());
        assert!(catalog.get_indexes("users").is_empty());
    }

    #[test]
    fn test_catalog_serialization() {
        let mut catalog = Catalog::new("test_db");

        let schema = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text);
        catalog.create_table(schema, 1).unwrap();

        let bytes = catalog.to_bytes().expect("Failed to serialize catalog");
        let restored = Catalog::from_bytes(&bytes).unwrap();

        assert_eq!(restored.name, "test_db");
        assert!(restored.get_table("users").is_some());
    }
}
