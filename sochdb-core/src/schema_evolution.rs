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

//! Schema Evolution - Online Schema Changes
//!
//! This module provides backwards-compatible schema evolution for SochDB,
//! allowing schema changes without full table rewrites through:
//!
//! - **Schema Versioning**: Each schema has a monotonic version number
//! - **Migration Registry**: Registered transformations between versions
//! - **Lazy Migration**: Rows are migrated on-read when version mismatch detected
//! - **Background Compaction**: Asynchronous migration during idle time
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Schema Version Graph                      │
//! │                                                             │
//! │  v1 ──────────────────────────────────────────────────────→ │
//! │   ↓                                                         │
//! │  v2 (add column "email")                                    │
//! │   ↓                                                         │
//! │  v3 (rename "name" → "full_name", add "created_at")        │
//! │   ↓                                                         │
//! │  v4 (drop "legacy_field")                                   │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Rows carry their schema version. On read:
//! 1. Check row version vs current schema version
//! 2. If mismatch, apply migration chain (v_row → v_current)
//! 3. Return migrated row (optionally rewrite in background)
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Result, SochDBError};
use crate::soch::{SochRow, SochSchema, SochType, SochValue};

/// Schema version identifier
pub type SchemaVersion = u64;

/// Unique schema identifier (table/collection name + version)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SchemaId {
    pub name: String,
    pub version: SchemaVersion,
}

impl SchemaId {
    pub fn new(name: impl Into<String>, version: SchemaVersion) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }
}

/// Describes a single schema change operation
#[derive(Debug, Clone)]
pub enum SchemaChange {
    /// Add a new column with default value
    AddColumn {
        name: String,
        column_type: SochType,
        default: SochValue,
        position: Option<usize>, // None = append at end
    },
    /// Drop an existing column
    DropColumn { name: String },
    /// Rename a column
    RenameColumn { old_name: String, new_name: String },
    /// Change column type with conversion function
    ChangeType {
        name: String,
        new_type: SochType,
        converter: TypeConverter,
    },
    /// Reorder columns
    ReorderColumns { new_order: Vec<String> },
}

/// Type conversion function for schema evolution
#[derive(Debug, Clone)]
pub enum TypeConverter {
    /// Identity (no conversion needed, types are compatible)
    Identity,
    /// Convert integer to text
    IntToText,
    /// Convert text to integer (may fail)
    TextToInt,
    /// Convert float to integer (truncate)
    FloatToInt,
    /// Convert integer to float
    IntToFloat,
    /// Custom conversion with closure index (stored in registry)
    Custom(usize),
}

impl TypeConverter {
    /// Apply the type conversion
    pub fn convert(
        &self,
        value: &SochValue,
        custom_converters: &[fn(&SochValue) -> SochValue],
    ) -> Result<SochValue> {
        match self {
            TypeConverter::Identity => Ok(value.clone()),
            TypeConverter::IntToText => match value {
                SochValue::Int(i) => Ok(SochValue::Text(i.to_string())),
                SochValue::UInt(u) => Ok(SochValue::Text(u.to_string())),
                _ => Err(SochDBError::SchemaEvolution(format!(
                    "Cannot convert {:?} to text via IntToText",
                    value
                ))),
            },
            TypeConverter::TextToInt => match value {
                SochValue::Text(s) => s.parse::<i64>().map(SochValue::Int).map_err(|_| {
                    SochDBError::SchemaEvolution(format!("Cannot parse '{}' as integer", s))
                }),
                _ => Err(SochDBError::SchemaEvolution(format!(
                    "Cannot convert {:?} to int via TextToInt",
                    value
                ))),
            },
            TypeConverter::FloatToInt => match value {
                SochValue::Float(f) => Ok(SochValue::Int(*f as i64)),
                _ => Err(SochDBError::SchemaEvolution(format!(
                    "Cannot convert {:?} to int via FloatToInt",
                    value
                ))),
            },
            TypeConverter::IntToFloat => match value {
                SochValue::Int(i) => Ok(SochValue::Float(*i as f64)),
                SochValue::UInt(u) => Ok(SochValue::Float(*u as f64)),
                _ => Err(SochDBError::SchemaEvolution(format!(
                    "Cannot convert {:?} to float via IntToFloat",
                    value
                ))),
            },
            TypeConverter::Custom(idx) => {
                if *idx < custom_converters.len() {
                    Ok(custom_converters[*idx](value))
                } else {
                    Err(SochDBError::SchemaEvolution(format!(
                        "Custom converter index {} out of bounds",
                        idx
                    )))
                }
            }
        }
    }
}

/// Migration between two schema versions
#[derive(Debug, Clone)]
pub struct Migration {
    pub from_version: SchemaVersion,
    pub to_version: SchemaVersion,
    pub changes: Vec<SchemaChange>,
}

impl Migration {
    pub fn new(from: SchemaVersion, to: SchemaVersion) -> Self {
        Self {
            from_version: from,
            to_version: to,
            changes: Vec::new(),
        }
    }

    pub fn add_column(
        mut self,
        name: impl Into<String>,
        column_type: SochType,
        default: SochValue,
    ) -> Self {
        self.changes.push(SchemaChange::AddColumn {
            name: name.into(),
            column_type,
            default,
            position: None,
        });
        self
    }

    pub fn add_column_at(
        mut self,
        name: impl Into<String>,
        column_type: SochType,
        default: SochValue,
        position: usize,
    ) -> Self {
        self.changes.push(SchemaChange::AddColumn {
            name: name.into(),
            column_type,
            default,
            position: Some(position),
        });
        self
    }

    pub fn drop_column(mut self, name: impl Into<String>) -> Self {
        self.changes
            .push(SchemaChange::DropColumn { name: name.into() });
        self
    }

    pub fn rename_column(
        mut self,
        old_name: impl Into<String>,
        new_name: impl Into<String>,
    ) -> Self {
        self.changes.push(SchemaChange::RenameColumn {
            old_name: old_name.into(),
            new_name: new_name.into(),
        });
        self
    }

    pub fn change_type(
        mut self,
        name: impl Into<String>,
        new_type: SochType,
        converter: TypeConverter,
    ) -> Self {
        self.changes.push(SchemaChange::ChangeType {
            name: name.into(),
            new_type,
            converter,
        });
        self
    }
}

/// Row with embedded schema version for lazy migration
#[derive(Debug, Clone)]
pub struct VersionedRow {
    pub version: SchemaVersion,
    pub data: SochRow,
}

impl VersionedRow {
    pub fn new(version: SchemaVersion, data: SochRow) -> Self {
        Self { version, data }
    }
}

/// Statistics for schema evolution operations
#[derive(Debug, Default)]
pub struct EvolutionStats {
    pub rows_migrated: AtomicU64,
    pub migrations_applied: AtomicU64,
    pub migration_errors: AtomicU64,
    pub lazy_migrations: AtomicU64,
    pub background_migrations: AtomicU64,
}

impl EvolutionStats {
    pub fn record_lazy_migration(&self) {
        self.lazy_migrations.fetch_add(1, Ordering::Relaxed);
        self.rows_migrated.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_background_migration(&self, count: u64) {
        self.background_migrations
            .fetch_add(count, Ordering::Relaxed);
        self.rows_migrated.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_migration_applied(&self) {
        self.migrations_applied.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.migration_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> EvolutionStatsSnapshot {
        EvolutionStatsSnapshot {
            rows_migrated: self.rows_migrated.load(Ordering::Relaxed),
            migrations_applied: self.migrations_applied.load(Ordering::Relaxed),
            migration_errors: self.migration_errors.load(Ordering::Relaxed),
            lazy_migrations: self.lazy_migrations.load(Ordering::Relaxed),
            background_migrations: self.background_migrations.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvolutionStatsSnapshot {
    pub rows_migrated: u64,
    pub migrations_applied: u64,
    pub migration_errors: u64,
    pub lazy_migrations: u64,
    pub background_migrations: u64,
}

/// Schema version registry with migration graph
pub struct SchemaRegistry {
    /// Schemas by (name, version)
    schemas: HashMap<SchemaId, SochSchema>,
    /// Current version for each schema name
    current_versions: HashMap<String, SchemaVersion>,
    /// Migrations indexed by (name, from_version)
    migrations: HashMap<(String, SchemaVersion), Migration>,
    /// Custom type converters
    custom_converters: Vec<fn(&SochValue) -> SochValue>,
    /// Statistics
    stats: Arc<EvolutionStats>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            current_versions: HashMap::new(),
            migrations: HashMap::new(),
            custom_converters: Vec::new(),
            stats: Arc::new(EvolutionStats::default()),
        }
    }

    /// Register initial schema version
    pub fn register_schema(&mut self, schema: SochSchema, version: SchemaVersion) {
        let name = schema.name.clone();
        let id = SchemaId::new(&name, version);
        self.schemas.insert(id, schema);

        // Update current version if this is newer
        let current = self.current_versions.entry(name).or_insert(0);
        if version > *current {
            *current = version;
        }
    }

    /// Register a migration between versions
    pub fn register_migration(
        &mut self,
        name: impl Into<String>,
        migration: Migration,
    ) -> Result<()> {
        let name = name.into();

        // Validate versions exist or will exist
        let key = (name.clone(), migration.from_version);

        if self.migrations.contains_key(&key) {
            return Err(SochDBError::SchemaEvolution(format!(
                "Migration from version {} already exists for {}",
                migration.from_version, name
            )));
        }

        self.migrations.insert(key, migration);
        Ok(())
    }

    /// Register a custom type converter
    pub fn register_converter(&mut self, converter: fn(&SochValue) -> SochValue) -> usize {
        let idx = self.custom_converters.len();
        self.custom_converters.push(converter);
        idx
    }

    /// Get current schema version for a name
    pub fn current_version(&self, name: &str) -> Option<SchemaVersion> {
        self.current_versions.get(name).copied()
    }

    /// Get schema by name and version
    pub fn get_schema(&self, name: &str, version: SchemaVersion) -> Option<&SochSchema> {
        self.schemas.get(&SchemaId::new(name, version))
    }

    /// Get current schema by name
    pub fn current_schema(&self, name: &str) -> Option<&SochSchema> {
        self.current_version(name)
            .and_then(|v| self.get_schema(name, v))
    }

    /// Migrate a row from old version to current version
    pub fn migrate_row(&self, name: &str, row: VersionedRow) -> Result<VersionedRow> {
        let current_version = self.current_version(name).ok_or_else(|| {
            SochDBError::SchemaEvolution(format!("No schema registered for '{}'", name))
        })?;

        if row.version == current_version {
            return Ok(row);
        }

        if row.version > current_version {
            return Err(SochDBError::SchemaEvolution(format!(
                "Row version {} is newer than current schema version {}",
                row.version, current_version
            )));
        }

        // Build migration chain
        let mut current_row = row.data;
        let mut version = row.version;

        while version < current_version {
            let migration = self
                .migrations
                .get(&(name.to_string(), version))
                .ok_or_else(|| {
                    SochDBError::SchemaEvolution(format!(
                        "No migration path from version {} for '{}'",
                        version, name
                    ))
                })?;

            current_row = self.apply_migration(&current_row, migration, name)?;
            version = migration.to_version;
            self.stats.record_migration_applied();
        }

        self.stats.record_lazy_migration();

        Ok(VersionedRow::new(current_version, current_row))
    }

    /// Apply a single migration to a row
    fn apply_migration(
        &self,
        row: &SochRow,
        migration: &Migration,
        schema_name: &str,
    ) -> Result<SochRow> {
        // Get source schema to understand column positions
        let source_schema = self
            .get_schema(schema_name, migration.from_version)
            .ok_or_else(|| {
                SochDBError::SchemaEvolution(format!(
                    "Source schema version {} not found",
                    migration.from_version
                ))
            })?;

        let mut values = row.values.clone();
        let mut column_names: Vec<String> = source_schema
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect();

        for change in &migration.changes {
            match change {
                SchemaChange::AddColumn {
                    name,
                    default,
                    position,
                    ..
                } => match position {
                    Some(pos) if *pos <= values.len() => {
                        values.insert(*pos, default.clone());
                        column_names.insert(*pos, name.clone());
                    }
                    _ => {
                        values.push(default.clone());
                        column_names.push(name.clone());
                    }
                },
                SchemaChange::DropColumn { name } => {
                    if let Some(idx) = column_names.iter().position(|n| n == name) {
                        values.remove(idx);
                        column_names.remove(idx);
                    }
                }
                SchemaChange::RenameColumn { old_name, new_name } => {
                    if let Some(idx) = column_names.iter().position(|n| n == old_name) {
                        column_names[idx] = new_name.clone();
                    }
                }
                SchemaChange::ChangeType {
                    name, converter, ..
                } => {
                    if let Some(idx) = column_names.iter().position(|n| n == name) {
                        values[idx] = converter.convert(&values[idx], &self.custom_converters)?;
                    }
                }
                SchemaChange::ReorderColumns { new_order } => {
                    let mut new_values = Vec::with_capacity(new_order.len());
                    for col_name in new_order {
                        if let Some(idx) = column_names.iter().position(|n| n == col_name) {
                            new_values.push(values[idx].clone());
                        }
                    }
                    values = new_values;
                    column_names = new_order.clone();
                }
            }
        }

        Ok(SochRow::new(values))
    }

    /// Get evolution statistics
    pub fn stats(&self) -> Arc<EvolutionStats> {
        Arc::clone(&self.stats)
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema evolution manager that wraps a registry with additional features
pub struct SchemaEvolutionManager {
    registry: SchemaRegistry,
    /// Pending background migrations (schema_name, from_version, row_count)
    pending_migrations: Vec<(String, SchemaVersion, usize)>,
}

impl SchemaEvolutionManager {
    pub fn new() -> Self {
        Self {
            registry: SchemaRegistry::new(),
            pending_migrations: Vec::new(),
        }
    }

    /// Create a new schema version with changes from current
    pub fn evolve_schema(
        &mut self,
        name: &str,
        changes: Vec<SchemaChange>,
    ) -> Result<SchemaVersion> {
        let current_version = self.registry.current_version(name).ok_or_else(|| {
            SochDBError::SchemaEvolution(format!("No schema registered for '{}'", name))
        })?;

        let current_schema = self
            .registry
            .current_schema(name)
            .ok_or_else(|| {
                SochDBError::SchemaEvolution(format!("Current schema not found for '{}'", name))
            })?
            .clone();

        // Build new schema by applying changes
        let new_version = current_version + 1;
        let new_schema = self.apply_schema_changes(&current_schema, &changes)?;

        // Register new schema and migration
        self.registry.register_schema(new_schema, new_version);

        let migration = Migration {
            from_version: current_version,
            to_version: new_version,
            changes,
        };
        self.registry.register_migration(name, migration)?;

        Ok(new_version)
    }

    /// Apply schema changes to create new schema definition
    fn apply_schema_changes(
        &self,
        schema: &SochSchema,
        changes: &[SchemaChange],
    ) -> Result<SochSchema> {
        let mut new_schema = schema.clone();

        for change in changes {
            match change {
                SchemaChange::AddColumn {
                    name,
                    column_type,
                    position,
                    ..
                } => {
                    let field = crate::soch::SochField {
                        name: name.clone(),
                        field_type: column_type.clone(),
                        nullable: true,
                        default: None,
                    };
                    match position {
                        Some(pos) if *pos <= new_schema.fields.len() => {
                            new_schema.fields.insert(*pos, field);
                        }
                        _ => {
                            new_schema.fields.push(field);
                        }
                    }
                }
                SchemaChange::DropColumn { name } => {
                    new_schema.fields.retain(|f| f.name != *name);
                }
                SchemaChange::RenameColumn { old_name, new_name } => {
                    for field in &mut new_schema.fields {
                        if field.name == *old_name {
                            field.name = new_name.clone();
                        }
                    }
                }
                SchemaChange::ChangeType { name, new_type, .. } => {
                    for field in &mut new_schema.fields {
                        if field.name == *name {
                            field.field_type = new_type.clone();
                        }
                    }
                }
                SchemaChange::ReorderColumns { new_order } => {
                    let mut new_fields = Vec::with_capacity(new_order.len());
                    for col_name in new_order {
                        if let Some(field) = new_schema.fields.iter().find(|f| &f.name == col_name)
                        {
                            new_fields.push(field.clone());
                        }
                    }
                    new_schema.fields = new_fields;
                }
            }
        }

        Ok(new_schema)
    }

    /// Migrate a versioned row to current schema
    pub fn migrate_row(&self, name: &str, row: VersionedRow) -> Result<VersionedRow> {
        self.registry.migrate_row(name, row)
    }

    /// Schedule background migration for old rows
    pub fn schedule_background_migration(
        &mut self,
        name: impl Into<String>,
        from_version: SchemaVersion,
        row_count: usize,
    ) {
        self.pending_migrations
            .push((name.into(), from_version, row_count));
    }

    /// Get pending background migrations
    pub fn pending_migrations(&self) -> &[(String, SchemaVersion, usize)] {
        &self.pending_migrations
    }

    /// Access the underlying registry
    pub fn registry(&self) -> &SchemaRegistry {
        &self.registry
    }

    /// Access the underlying registry mutably
    pub fn registry_mut(&mut self) -> &mut SchemaRegistry {
        &mut self.registry
    }

    /// Get evolution statistics
    pub fn stats(&self) -> Arc<EvolutionStats> {
        self.registry.stats()
    }
}

impl Default for SchemaEvolutionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> SochSchema {
        SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
    }

    #[test]
    fn test_schema_registration() {
        let mut registry = SchemaRegistry::new();
        let schema = create_test_schema();

        registry.register_schema(schema, 1);

        assert_eq!(registry.current_version("users"), Some(1));
        assert!(registry.get_schema("users", 1).is_some());
    }

    #[test]
    fn test_add_column_migration() {
        let mut registry = SchemaRegistry::new();

        // Register v1 schema
        let schema_v1 = create_test_schema();
        registry.register_schema(schema_v1, 1);

        // Register v2 schema with email column
        let schema_v2 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("email", SochType::Text);
        registry.register_schema(schema_v2, 2);

        // Register migration
        let migration =
            Migration::new(1, 2).add_column("email", SochType::Text, SochValue::Text("".into()));
        registry.register_migration("users", migration).unwrap();

        // Create v1 row
        let row_v1 = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("Alice".into())]),
        );

        // Migrate to v2
        let row_v2 = registry.migrate_row("users", row_v1).unwrap();

        assert_eq!(row_v2.version, 2);
        assert_eq!(row_v2.data.values.len(), 3);
        assert_eq!(row_v2.data.values[2], SochValue::Text("".into()));
    }

    #[test]
    fn test_drop_column_migration() {
        let mut registry = SchemaRegistry::new();

        // Register v1 schema with legacy field
        let schema_v1 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("legacy", SochType::Text);
        registry.register_schema(schema_v1, 1);

        // Register v2 schema without legacy
        let schema_v2 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text);
        registry.register_schema(schema_v2, 2);

        // Register migration
        let migration = Migration::new(1, 2).drop_column("legacy");
        registry.register_migration("users", migration).unwrap();

        // Create v1 row with legacy field
        let row_v1 = VersionedRow::new(
            1,
            SochRow::new(vec![
                SochValue::UInt(1),
                SochValue::Text("Alice".into()),
                SochValue::Text("old_data".into()),
            ]),
        );

        // Migrate to v2
        let row_v2 = registry.migrate_row("users", row_v1).unwrap();

        assert_eq!(row_v2.version, 2);
        assert_eq!(row_v2.data.values.len(), 2);
    }

    #[test]
    fn test_rename_column_migration() {
        let mut registry = SchemaRegistry::new();

        let schema_v1 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text);
        registry.register_schema(schema_v1, 1);

        let schema_v2 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("full_name", SochType::Text);
        registry.register_schema(schema_v2, 2);

        let migration = Migration::new(1, 2).rename_column("name", "full_name");
        registry.register_migration("users", migration).unwrap();

        let row_v1 = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("Alice".into())]),
        );

        let row_v2 = registry.migrate_row("users", row_v1).unwrap();

        assert_eq!(row_v2.version, 2);
        assert_eq!(row_v2.data.values.len(), 2);
        assert_eq!(row_v2.data.values[1], SochValue::Text("Alice".into()));
    }

    #[test]
    fn test_type_conversion_migration() {
        let mut registry = SchemaRegistry::new();

        let schema_v1 = SochSchema::new("products")
            .field("id", SochType::UInt)
            .field("price", SochType::Int);
        registry.register_schema(schema_v1, 1);

        let schema_v2 = SochSchema::new("products")
            .field("id", SochType::UInt)
            .field("price", SochType::Float);
        registry.register_schema(schema_v2, 2);

        let migration =
            Migration::new(1, 2).change_type("price", SochType::Float, TypeConverter::IntToFloat);
        registry.register_migration("products", migration).unwrap();

        let row_v1 = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Int(100)]),
        );

        let row_v2 = registry.migrate_row("products", row_v1).unwrap();

        assert_eq!(row_v2.version, 2);
        assert_eq!(row_v2.data.values[1], SochValue::Float(100.0));
    }

    #[test]
    fn test_multi_version_migration_chain() {
        let mut registry = SchemaRegistry::new();

        // v1: id, name
        let schema_v1 = create_test_schema();
        registry.register_schema(schema_v1, 1);

        // v2: id, name, email
        let schema_v2 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("email", SochType::Text);
        registry.register_schema(schema_v2, 2);

        // v3: id, full_name, email, created_at
        let schema_v3 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("full_name", SochType::Text)
            .field("email", SochType::Text)
            .field("created_at", SochType::Int);
        registry.register_schema(schema_v3, 3);

        // v1 -> v2: add email
        let migration_1_2 =
            Migration::new(1, 2).add_column("email", SochType::Text, SochValue::Text("".into()));
        registry.register_migration("users", migration_1_2).unwrap();

        // v2 -> v3: rename name -> full_name, add created_at
        let migration_2_3 = Migration::new(2, 3)
            .rename_column("name", "full_name")
            .add_column("created_at", SochType::Int, SochValue::Int(0));
        registry.register_migration("users", migration_2_3).unwrap();

        // Migrate from v1 to v3
        let row_v1 = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("Alice".into())]),
        );

        let row_v3 = registry.migrate_row("users", row_v1).unwrap();

        assert_eq!(row_v3.version, 3);
        assert_eq!(row_v3.data.values.len(), 4);
        assert_eq!(row_v3.data.values[0], SochValue::UInt(1));
        assert_eq!(row_v3.data.values[1], SochValue::Text("Alice".into()));
        assert_eq!(row_v3.data.values[2], SochValue::Text("".into()));
        assert_eq!(row_v3.data.values[3], SochValue::Int(0));
    }

    #[test]
    fn test_evolve_schema() {
        let mut manager = SchemaEvolutionManager::new();

        // Register initial schema
        let schema = create_test_schema();
        manager.registry_mut().register_schema(schema, 1);

        // Evolve to add email
        let changes = vec![SchemaChange::AddColumn {
            name: "email".to_string(),
            column_type: SochType::Text,
            default: SochValue::Text("".into()),
            position: None,
        }];

        let new_version = manager.evolve_schema("users", changes).unwrap();

        assert_eq!(new_version, 2);
        assert_eq!(manager.registry().current_version("users"), Some(2));

        let current = manager.registry().current_schema("users").unwrap();
        assert_eq!(current.fields.len(), 3);
    }

    #[test]
    fn test_no_migration_needed_for_current_version() {
        let mut registry = SchemaRegistry::new();
        let schema = create_test_schema();
        registry.register_schema(schema, 1);

        let row = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("Alice".into())]),
        );

        let result = registry.migrate_row("users", row.clone()).unwrap();

        assert_eq!(result.version, row.version);
        assert_eq!(result.data.values, row.data.values);
    }

    #[test]
    fn test_stats_tracking() {
        let mut registry = SchemaRegistry::new();

        let schema_v1 = create_test_schema();
        registry.register_schema(schema_v1, 1);

        let schema_v2 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text)
            .field("email", SochType::Text);
        registry.register_schema(schema_v2, 2);

        let migration =
            Migration::new(1, 2).add_column("email", SochType::Text, SochValue::Text("".into()));
        registry.register_migration("users", migration).unwrap();

        // Migrate a row
        let row = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("Alice".into())]),
        );
        registry.migrate_row("users", row).unwrap();

        let stats = registry.stats().snapshot();
        assert_eq!(stats.rows_migrated, 1);
        assert_eq!(stats.migrations_applied, 1);
        assert_eq!(stats.lazy_migrations, 1);
    }

    #[test]
    fn test_error_on_future_version() {
        let mut registry = SchemaRegistry::new();
        let schema = create_test_schema();
        registry.register_schema(schema, 1);

        let row = VersionedRow::new(
            99,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("Alice".into())]),
        );

        let result = registry.migrate_row("users", row);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_type_converter() {
        let mut registry = SchemaRegistry::new();

        // Register custom converter that uppercases text
        let converter_idx = registry.register_converter(|v| {
            if let SochValue::Text(s) = v {
                SochValue::Text(s.to_uppercase())
            } else {
                v.clone()
            }
        });

        let schema_v1 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text);
        registry.register_schema(schema_v1, 1);

        let schema_v2 = SochSchema::new("users")
            .field("id", SochType::UInt)
            .field("name", SochType::Text); // Same type, different format
        registry.register_schema(schema_v2, 2);

        let migration = Migration::new(1, 2).change_type(
            "name",
            SochType::Text,
            TypeConverter::Custom(converter_idx),
        );
        registry.register_migration("users", migration).unwrap();

        let row_v1 = VersionedRow::new(
            1,
            SochRow::new(vec![SochValue::UInt(1), SochValue::Text("alice".into())]),
        );

        let row_v2 = registry.migrate_row("users", row_v1).unwrap();
        assert_eq!(row_v2.data.values[1], SochValue::Text("ALICE".into()));
    }
}
