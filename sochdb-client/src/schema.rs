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

//! Schema Management (DDL)
//!
//! Provides fluent schema definition and DDL operations.

use crate::connection::{SochConnection, to_field_type};
use crate::error::{ClientError, Result};

use sochdb_core::soch::{SochSchema, SochType};

/// Schema builder with fluent API
pub struct SchemaBuilder {
    name: String,
    fields: Vec<FieldBuilder>,
    primary_key: Option<String>,
    indexes: Vec<IndexDef>,
}

/// Field builder
#[derive(Debug, Clone)]
pub struct FieldBuilder {
    pub name: String,
    pub field_type: SochType,
    pub nullable: bool,
    pub unique: bool,
    pub default: Option<String>,
}

/// Index definition
#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name: String,
    pub fields: Vec<String>,
    pub unique: bool,
}

impl SchemaBuilder {
    /// Start building a new table schema
    pub fn table(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fields: Vec::new(),
            primary_key: None,
            indexes: Vec::new(),
        }
    }

    /// Add a field
    pub fn field(mut self, name: &str, field_type: SochType) -> FieldConfig {
        self.fields.push(FieldBuilder {
            name: name.to_string(),
            field_type,
            nullable: true,
            unique: false,
            default: None,
        });
        FieldConfig { builder: self }
    }

    /// Set primary key
    pub fn primary_key(mut self, field: &str) -> Self {
        self.primary_key = Some(field.to_string());
        self
    }

    /// Add an index
    pub fn index(mut self, name: &str, fields: &[&str], unique: bool) -> Self {
        self.indexes.push(IndexDef {
            name: name.to_string(),
            fields: fields.iter().map(|s| s.to_string()).collect(),
            unique,
        });
        self
    }

    /// Build the SochSchema
    pub fn build(self) -> SochSchema {
        let mut schema = SochSchema::new(&self.name);
        for field in self.fields {
            schema = schema.field(&field.name, field.field_type);
        }
        if let Some(pk) = self.primary_key {
            schema = schema.primary_key(&pk);
        }
        schema
    }

    /// Get field definitions
    pub fn get_fields(&self) -> &[FieldBuilder] {
        &self.fields
    }
}

/// Field configuration (for chaining)
pub struct FieldConfig {
    pub(crate) builder: SchemaBuilder,
}

impl FieldConfig {
    /// Mark field as NOT NULL
    pub fn not_null(mut self) -> Self {
        if let Some(field) = self.builder.fields.last_mut() {
            field.nullable = false;
        }
        self
    }

    /// Mark field as UNIQUE
    pub fn unique(mut self) -> Self {
        if let Some(field) = self.builder.fields.last_mut() {
            field.unique = true;
        }
        self
    }

    /// Set default value
    pub fn default_value(mut self, value: &str) -> Self {
        if let Some(field) = self.builder.fields.last_mut() {
            field.default = Some(value.to_string());
        }
        self
    }

    /// Continue adding fields
    pub fn field(self, name: &str, field_type: SochType) -> FieldConfig {
        self.builder.field(name, field_type)
    }

    /// Finish with primary key
    pub fn primary_key(self, field: &str) -> SchemaBuilder {
        self.builder.primary_key(field)
    }

    /// Add index
    pub fn index(self, name: &str, fields: &[&str], unique: bool) -> SchemaBuilder {
        self.builder.index(name, fields, unique)
    }

    /// Build schema
    pub fn build(self) -> SochSchema {
        self.builder.build()
    }
}

/// Table description
#[derive(Debug, Clone)]
pub struct TableDescription {
    pub name: String,
    pub columns: Vec<ColumnDescription>,
    pub row_count: u64,
    pub indexes: Vec<String>,
}

/// Column description
#[derive(Debug, Clone)]
pub struct ColumnDescription {
    pub name: String,
    pub field_type: SochType,
    pub nullable: bool,
}

/// Create table result
#[derive(Debug, Clone)]
pub struct CreateTableResult {
    pub table_name: String,
    pub column_count: usize,
}

/// Drop table result
#[derive(Debug, Clone)]
pub struct DropTableResult {
    pub table_name: String,
    pub rows_deleted: u64,
}

/// Create index result
#[derive(Debug, Clone)]
pub struct CreateIndexResult {
    pub index_name: String,
    pub table_name: String,
    pub rows_indexed: usize,
}

/// DDL operations on connection
impl SochConnection {
    /// Create a new table from schema
    pub fn create_table(&self, schema: SochSchema) -> Result<CreateTableResult> {
        let name = schema.name.clone();
        let column_count = schema.fields.len();

        // Register in TCH
        let fields: Vec<_> = schema
            .fields
            .iter()
            .map(|f| (f.name.clone(), to_field_type(&f.field_type)))
            .collect();
        self.tch.write().register_table(&name, &fields);

        // Update catalog
        {
            let mut catalog = self.catalog.write();
            let root_id = 0; // Placeholder
            catalog
                .create_table(schema, root_id)
                .map_err(ClientError::Schema)?;
        }

        Ok(CreateTableResult {
            table_name: name,
            column_count,
        })
    }

    /// Drop a table
    pub fn drop_table(&self, name: &str) -> Result<DropTableResult> {
        let entry = {
            let mut catalog = self.catalog.write();
            catalog.drop_table(name).map_err(ClientError::Schema)?
        };

        Ok(DropTableResult {
            table_name: name.to_string(),
            rows_deleted: entry.row_count,
        })
    }

    /// Create an index
    pub fn create_index(
        &self,
        name: &str,
        table: &str,
        fields: &[&str],
        unique: bool,
    ) -> Result<CreateIndexResult> {
        {
            let mut catalog = self.catalog.write();
            let root_id = 0; // Placeholder
            catalog
                .create_index(
                    name,
                    table,
                    fields.iter().map(|s| s.to_string()).collect(),
                    unique,
                    root_id,
                )
                .map_err(ClientError::Schema)?;
        }

        Ok(CreateIndexResult {
            index_name: name.to_string(),
            table_name: table.to_string(),
            rows_indexed: 0, // Placeholder
        })
    }

    /// Drop an index
    pub fn drop_index(&self, name: &str) -> Result<()> {
        let mut catalog = self.catalog.write();
        catalog.drop_index(name).map_err(ClientError::Schema)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::table("users")
            .field("id", SochType::Int)
            .not_null()
            .field("name", SochType::Text)
            .not_null()
            .field("email", SochType::Text)
            .unique()
            .primary_key("id")
            .build();

        assert_eq!(schema.name, "users");
        assert_eq!(schema.fields.len(), 3);
    }

    #[test]
    fn test_schema_with_index() {
        let schema = SchemaBuilder::table("orders")
            .field("id", SochType::Int)
            .field("user_id", SochType::Int)
            .field("amount", SochType::Float)
            .primary_key("id")
            .index("idx_user", &["user_id"], false)
            .build();

        assert_eq!(schema.name, "orders");
    }

    #[test]
    fn test_create_table() {
        let conn = SochConnection::open("./test").unwrap();

        let schema = SchemaBuilder::table("test_table")
            .field("id", SochType::Int)
            .field("name", SochType::Text)
            .build();

        let result = conn.create_table(schema).unwrap();
        assert_eq!(result.table_name, "test_table");
        assert_eq!(result.column_count, 2);
    }

    #[test]
    fn test_drop_table() {
        let conn = SochConnection::open("./test").unwrap();

        let schema = SchemaBuilder::table("to_drop")
            .field("id", SochType::Int)
            .build();

        conn.create_table(schema).unwrap();
        let result = conn.drop_table("to_drop").unwrap();
        assert_eq!(result.table_name, "to_drop");
    }
}
