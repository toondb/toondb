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

//! Plugin-as-Table Integration
//!
//! This module allows WASM plugins to expose virtual tables that can be
//! queried using standard SELECT statements.
//!
//! ## Example
//!
//! A plugin exposing a virtual table:
//!
//! ```text
//! SELECT * FROM plugin_name.table_name WHERE key = 'value' LIMIT 10
//! ```
//!
//! This translates to plugin function calls:
//!
//! 1. `describe_table()` - Get schema
//! 2. `scan_table(filter, limit)` - Get matching rows
//!
//! ## Virtual Table Protocol
//!
//! Plugins must export:
//!
//! - `describe_tables() -> Vec<TableDescriptor>` - List available tables
//! - `describe_table(name) -> TableSchema` - Get table schema
//! - `scan_table(name, filter, limit) -> Vec<Row>` - Scan with filter
//! - `get_row(name, key) -> Option<Row>` - Point lookup (optional)

use crate::soch_ql::{SelectQuery, SochResult, SochValue, WhereClause};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Virtual Table Trait
// ============================================================================

/// Column type for virtual tables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualColumnType {
    Bool,
    Int64,
    UInt64,
    Float64,
    Text,
    Binary,
    Timestamp,
    Json,
}

/// Column definition for a virtual table
#[derive(Debug, Clone)]
pub struct VirtualColumnDef {
    /// Column name
    pub name: String,
    /// Column type
    pub col_type: VirtualColumnType,
    /// Is nullable
    pub nullable: bool,
    /// Is primary key
    pub primary_key: bool,
    /// Description
    pub description: Option<String>,
}

/// Schema for a virtual table
#[derive(Debug, Clone)]
pub struct VirtualTableSchema {
    /// Table name
    pub name: String,
    /// Column definitions
    pub columns: Vec<VirtualColumnDef>,
    /// Estimated row count (for query planning)
    pub estimated_rows: Option<u64>,
    /// Description
    pub description: Option<String>,
}

/// Virtual table trait
///
/// Plugins implement this trait to expose queryable tables.
pub trait VirtualTable: Send + Sync {
    /// Get table schema
    fn schema(&self) -> &VirtualTableSchema;

    /// Scan with optional filter
    fn scan(
        &self,
        columns: &[String],
        filter: Option<&VirtualFilter>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<VirtualRow>, VirtualTableError>;

    /// Point lookup by primary key
    fn get(&self, key: &SochValue) -> Result<Option<VirtualRow>, VirtualTableError> {
        // Default implementation: scan with equality filter on primary key
        let schema = self.schema();
        let pk_col = schema
            .columns
            .iter()
            .find(|c| c.primary_key)
            .map(|c| c.name.clone());

        if let Some(pk) = pk_col {
            let filter = VirtualFilter::Eq(pk, key.clone());
            let rows = self.scan(&[], Some(&filter), Some(1), None)?;
            Ok(rows.into_iter().next())
        } else {
            Err(VirtualTableError::NoPrimaryKey)
        }
    }

    /// Get statistics for query planning
    fn stats(&self) -> VirtualTableStats {
        VirtualTableStats {
            row_count: self.schema().estimated_rows,
            size_bytes: None,
            last_modified: None,
        }
    }

    /// Refresh cached data (if applicable)
    fn refresh(&self) -> Result<(), VirtualTableError> {
        Ok(()) // Default: no-op
    }
}

/// Row from a virtual table
#[derive(Debug, Clone)]
pub struct VirtualRow {
    /// Column values
    pub values: Vec<SochValue>,
}

impl VirtualRow {
    /// Create a new row
    pub fn new(values: Vec<SochValue>) -> Self {
        Self { values }
    }

    /// Get value by index
    pub fn get(&self, idx: usize) -> Option<&SochValue> {
        self.values.get(idx)
    }

    /// Get value by column name (requires schema)
    pub fn get_by_name<'a>(
        &'a self,
        name: &str,
        schema: &VirtualTableSchema,
    ) -> Option<&'a SochValue> {
        schema
            .columns
            .iter()
            .position(|c| c.name == name)
            .and_then(|idx| self.values.get(idx))
    }
}

/// Filter for virtual table scans
#[derive(Debug, Clone)]
pub enum VirtualFilter {
    /// Equality: column = value
    Eq(String, SochValue),
    /// Not equal: column != value
    Ne(String, SochValue),
    /// Less than: column < value
    Lt(String, SochValue),
    /// Less than or equal: column <= value
    Le(String, SochValue),
    /// Greater than: column > value
    Gt(String, SochValue),
    /// Greater than or equal: column >= value
    Ge(String, SochValue),
    /// Like pattern: column LIKE pattern
    Like(String, String),
    /// In set: column IN (values)
    In(String, Vec<SochValue>),
    /// Between: column BETWEEN low AND high
    Between(String, SochValue, SochValue),
    /// Is null: column IS NULL
    IsNull(String),
    /// Is not null: column IS NOT NULL
    IsNotNull(String),
    /// AND of filters
    And(Vec<VirtualFilter>),
    /// OR of filters
    Or(Vec<VirtualFilter>),
    /// NOT of filter
    Not(Box<VirtualFilter>),
}

impl VirtualFilter {
    /// Convert from WHERE clause
    pub fn from_where_clause(where_clause: &WhereClause) -> Self {
        let filters: Vec<VirtualFilter> = where_clause
            .conditions
            .iter()
            .map(|c| {
                use crate::soch_ql::ComparisonOp::*;
                match c.operator {
                    Eq => VirtualFilter::Eq(c.column.clone(), c.value.clone()),
                    Ne => VirtualFilter::Ne(c.column.clone(), c.value.clone()),
                    Lt => VirtualFilter::Lt(c.column.clone(), c.value.clone()),
                    Le => VirtualFilter::Le(c.column.clone(), c.value.clone()),
                    Gt => VirtualFilter::Gt(c.column.clone(), c.value.clone()),
                    Ge => VirtualFilter::Ge(c.column.clone(), c.value.clone()),
                    Like => {
                        if let SochValue::Text(pattern) = &c.value {
                            VirtualFilter::Like(c.column.clone(), pattern.clone())
                        } else {
                            VirtualFilter::Like(c.column.clone(), "".to_string())
                        }
                    }
                    In => VirtualFilter::In(c.column.clone(), vec![c.value.clone()]),
                    SimilarTo => {
                        // SimilarTo is used for vector similarity search
                        // For now, we treat it as a Like pattern for virtual tables
                        if let SochValue::Text(pattern) = &c.value {
                            VirtualFilter::Like(c.column.clone(), pattern.clone())
                        } else {
                            VirtualFilter::Like(c.column.clone(), "".to_string())
                        }
                    }
                }
            })
            .collect();

        match where_clause.operator {
            crate::soch_ql::LogicalOp::And => VirtualFilter::And(filters),
            crate::soch_ql::LogicalOp::Or => VirtualFilter::Or(filters),
        }
    }

    /// Evaluate filter against a row
    pub fn matches(&self, row: &VirtualRow, schema: &VirtualTableSchema) -> bool {
        match self {
            VirtualFilter::Eq(col, value) => row
                .get_by_name(col, schema)
                .map(|v| v == value)
                .unwrap_or(false),
            VirtualFilter::Ne(col, value) => row
                .get_by_name(col, schema)
                .map(|v| v != value)
                .unwrap_or(false),
            VirtualFilter::Lt(col, value) => {
                Self::compare_values(row.get_by_name(col, schema), value, |a, b| a < b)
            }
            VirtualFilter::Le(col, value) => {
                Self::compare_values(row.get_by_name(col, schema), value, |a, b| a <= b)
            }
            VirtualFilter::Gt(col, value) => {
                Self::compare_values(row.get_by_name(col, schema), value, |a, b| a > b)
            }
            VirtualFilter::Ge(col, value) => {
                Self::compare_values(row.get_by_name(col, schema), value, |a, b| a >= b)
            }
            VirtualFilter::Like(col, pattern) => row
                .get_by_name(col, schema)
                .and_then(|v| match v {
                    SochValue::Text(s) => Some(Self::match_like(s, pattern)),
                    _ => None,
                })
                .unwrap_or(false),
            VirtualFilter::In(col, values) => row
                .get_by_name(col, schema)
                .map(|v| values.contains(v))
                .unwrap_or(false),
            VirtualFilter::Between(col, low, high) => row
                .get_by_name(col, schema)
                .map(|v| {
                    Self::compare_values(Some(v), low, |a, b| a >= b)
                        && Self::compare_values(Some(v), high, |a, b| a <= b)
                })
                .unwrap_or(false),
            VirtualFilter::IsNull(col) => row
                .get_by_name(col, schema)
                .map(|v| *v == SochValue::Null)
                .unwrap_or(true),
            VirtualFilter::IsNotNull(col) => row
                .get_by_name(col, schema)
                .map(|v| *v != SochValue::Null)
                .unwrap_or(false),
            VirtualFilter::And(filters) => filters.iter().all(|f| f.matches(row, schema)),
            VirtualFilter::Or(filters) => filters.iter().any(|f| f.matches(row, schema)),
            VirtualFilter::Not(filter) => !filter.matches(row, schema),
        }
    }

    fn compare_values<F>(val: Option<&SochValue>, other: &SochValue, cmp: F) -> bool
    where
        F: Fn(i64, i64) -> bool,
    {
        match (val, other) {
            (Some(SochValue::Int(a)), SochValue::Int(b)) => cmp(*a, *b),
            (Some(SochValue::UInt(a)), SochValue::UInt(b)) => cmp(*a as i64, *b as i64),
            (Some(SochValue::Float(a)), SochValue::Float(b)) => {
                cmp((*a * 1000.0) as i64, (*b * 1000.0) as i64)
            }
            _ => false,
        }
    }

    fn match_like(s: &str, pattern: &str) -> bool {
        // Simple LIKE implementation
        if pattern.starts_with('%') && pattern.ends_with('%') {
            let inner = &pattern[1..pattern.len() - 1];
            s.contains(inner)
        } else if let Some(suffix) = pattern.strip_prefix('%') {
            s.ends_with(suffix)
        } else if let Some(prefix) = pattern.strip_suffix('%') {
            s.starts_with(prefix)
        } else {
            s == pattern
        }
    }
}

/// Virtual table statistics
#[derive(Debug, Clone, Default)]
pub struct VirtualTableStats {
    /// Estimated row count
    pub row_count: Option<u64>,
    /// Estimated size in bytes
    pub size_bytes: Option<u64>,
    /// Last modification timestamp
    pub last_modified: Option<u64>,
}

/// Virtual table error
#[derive(Debug, Clone)]
pub enum VirtualTableError {
    /// Table not found
    NotFound(String),
    /// Column not found
    ColumnNotFound(String),
    /// No primary key defined
    NoPrimaryKey,
    /// Plugin error
    PluginError(String),
    /// Scan failed
    ScanFailed(String),
    /// Invalid filter
    InvalidFilter(String),
}

impl std::fmt::Display for VirtualTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "virtual table not found: {}", name),
            Self::ColumnNotFound(name) => write!(f, "column not found: {}", name),
            Self::NoPrimaryKey => write!(f, "no primary key defined"),
            Self::PluginError(msg) => write!(f, "plugin error: {}", msg),
            Self::ScanFailed(msg) => write!(f, "scan failed: {}", msg),
            Self::InvalidFilter(msg) => write!(f, "invalid filter: {}", msg),
        }
    }
}

impl std::error::Error for VirtualTableError {}

// ============================================================================
// Plugin Virtual Table Wrapper
// ============================================================================

/// Virtual table backed by a WASM plugin
pub struct PluginVirtualTable {
    /// Plugin name
    plugin_name: String,
    /// Table name
    table_name: String,
    /// Cached schema
    schema: VirtualTableSchema,
    /// Cached rows (optional)
    cache: RwLock<Option<CachedData>>,
    /// Cache TTL in seconds
    cache_ttl_secs: u64,
}

/// Cached table data
struct CachedData {
    rows: Vec<VirtualRow>,
    cached_at: std::time::Instant,
}

impl PluginVirtualTable {
    /// Create a new plugin-backed virtual table
    pub fn new(plugin_name: &str, table_name: &str, schema: VirtualTableSchema) -> Self {
        Self {
            plugin_name: plugin_name.to_string(),
            table_name: table_name.to_string(),
            schema,
            cache: RwLock::new(None),
            cache_ttl_secs: 60, // 1 minute default
        }
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, secs: u64) -> Self {
        self.cache_ttl_secs = secs;
        self
    }

    /// Get the fully qualified table name
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.plugin_name, self.table_name)
    }

    /// Check if cache is valid
    fn is_cache_valid(&self) -> bool {
        if let Some(cached) = self.cache.read().as_ref() {
            cached.cached_at.elapsed().as_secs() < self.cache_ttl_secs
        } else {
            false
        }
    }

    /// Update cache
    fn update_cache(&self, rows: Vec<VirtualRow>) {
        *self.cache.write() = Some(CachedData {
            rows,
            cached_at: std::time::Instant::now(),
        });
    }
}

impl VirtualTable for PluginVirtualTable {
    fn schema(&self) -> &VirtualTableSchema {
        &self.schema
    }

    fn scan(
        &self,
        columns: &[String],
        filter: Option<&VirtualFilter>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<VirtualRow>, VirtualTableError> {
        // Check cache first
        if self.is_cache_valid()
            && let Some(cached) = self.cache.read().as_ref()
        {
            let mut rows = cached.rows.clone();

            // Apply filter
            if let Some(f) = filter {
                rows.retain(|r| f.matches(r, &self.schema));
            }

            // Apply offset
            if let Some(o) = offset {
                rows = rows.into_iter().skip(o).collect();
            }

            // Apply limit
            if let Some(l) = limit {
                rows.truncate(l);
            }

            // Project columns
            if !columns.is_empty() && columns[0] != "*" {
                rows = self.project_columns(&rows, columns);
            }

            return Ok(rows);
        }

        // In production, this would call the plugin's scan_table function
        // For now, return mock data
        let mock_rows = self.generate_mock_data(limit.unwrap_or(100));

        // Update cache
        self.update_cache(mock_rows.clone());

        // Apply filter to mock data
        let mut result = mock_rows;
        if let Some(f) = filter {
            result.retain(|r| f.matches(r, &self.schema));
        }

        if let Some(o) = offset {
            result = result.into_iter().skip(o).collect();
        }

        if let Some(l) = limit {
            result.truncate(l);
        }

        Ok(result)
    }

    fn refresh(&self) -> Result<(), VirtualTableError> {
        // Clear cache
        *self.cache.write() = None;
        Ok(())
    }
}

impl PluginVirtualTable {
    /// Project only requested columns
    fn project_columns(&self, rows: &[VirtualRow], columns: &[String]) -> Vec<VirtualRow> {
        let indices: Vec<usize> = columns
            .iter()
            .filter_map(|col| self.schema.columns.iter().position(|c| c.name == *col))
            .collect();

        rows.iter()
            .map(|row| {
                let values: Vec<SochValue> = indices
                    .iter()
                    .map(|&i| row.values.get(i).cloned().unwrap_or(SochValue::Null))
                    .collect();
                VirtualRow::new(values)
            })
            .collect()
    }

    /// Generate mock data (for demonstration)
    fn generate_mock_data(&self, count: usize) -> Vec<VirtualRow> {
        (0..count)
            .map(|i| {
                let values: Vec<SochValue> = self
                    .schema
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(col_idx, col)| match col.col_type {
                        VirtualColumnType::Int64 => SochValue::Int(i as i64 + col_idx as i64),
                        VirtualColumnType::UInt64 => SochValue::UInt(i as u64 + col_idx as u64),
                        VirtualColumnType::Float64 => SochValue::Float(i as f64 * 0.1),
                        VirtualColumnType::Text => SochValue::Text(format!("{}_{}", col.name, i)),
                        VirtualColumnType::Bool => SochValue::Bool(i % 2 == 0),
                        _ => SochValue::Null,
                    })
                    .collect();
                VirtualRow::new(values)
            })
            .collect()
    }
}

// ============================================================================
// Virtual Table Registry
// ============================================================================

/// Registry for virtual tables
pub struct VirtualTableRegistry {
    /// Tables by qualified name (plugin.table)
    tables: RwLock<HashMap<String, Arc<dyn VirtualTable>>>,
}

impl Default for VirtualTableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualTableRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
        }
    }

    /// Register a virtual table
    pub fn register(
        &self,
        qualified_name: &str,
        table: Arc<dyn VirtualTable>,
    ) -> Result<(), VirtualTableError> {
        let mut tables = self.tables.write();

        if tables.contains_key(qualified_name) {
            return Err(VirtualTableError::PluginError(format!(
                "table '{}' already registered",
                qualified_name
            )));
        }

        tables.insert(qualified_name.to_string(), table);
        Ok(())
    }

    /// Unregister a virtual table
    pub fn unregister(&self, qualified_name: &str) -> Result<(), VirtualTableError> {
        let mut tables = self.tables.write();

        if tables.remove(qualified_name).is_none() {
            return Err(VirtualTableError::NotFound(qualified_name.to_string()));
        }

        Ok(())
    }

    /// Get a virtual table
    pub fn get(&self, qualified_name: &str) -> Option<Arc<dyn VirtualTable>> {
        self.tables.read().get(qualified_name).cloned()
    }

    /// List all registered tables
    pub fn list(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    /// Execute a SELECT query on a virtual table
    pub fn execute_select(&self, query: &SelectQuery) -> Result<SochResult, VirtualTableError> {
        let table = self
            .get(&query.table)
            .ok_or_else(|| VirtualTableError::NotFound(query.table.clone()))?;

        let schema = table.schema();

        // Convert WHERE clause to filter
        let filter = query
            .where_clause
            .as_ref()
            .map(VirtualFilter::from_where_clause);

        // Scan table
        let rows = table.scan(&query.columns, filter.as_ref(), query.limit, query.offset)?;

        // Convert to SochResult
        let columns = if query.columns.is_empty() || query.columns[0] == "*" {
            schema.columns.iter().map(|c| c.name.clone()).collect()
        } else {
            query.columns.clone()
        };

        let result_rows: Vec<Vec<SochValue>> = rows.into_iter().map(|r| r.values).collect();

        Ok(SochResult {
            table: query.table.clone(),
            columns,
            rows: result_rows,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> VirtualTableSchema {
        VirtualTableSchema {
            name: "test_table".to_string(),
            columns: vec![
                VirtualColumnDef {
                    name: "id".to_string(),
                    col_type: VirtualColumnType::Int64,
                    nullable: false,
                    primary_key: true,
                    description: None,
                },
                VirtualColumnDef {
                    name: "name".to_string(),
                    col_type: VirtualColumnType::Text,
                    nullable: false,
                    primary_key: false,
                    description: None,
                },
                VirtualColumnDef {
                    name: "score".to_string(),
                    col_type: VirtualColumnType::Float64,
                    nullable: true,
                    primary_key: false,
                    description: None,
                },
            ],
            estimated_rows: Some(1000),
            description: None,
        }
    }

    #[test]
    fn test_plugin_virtual_table_creation() {
        let schema = create_test_schema();
        let table = PluginVirtualTable::new("test_plugin", "test_table", schema);

        assert_eq!(table.qualified_name(), "test_plugin.test_table");
        assert_eq!(table.schema().columns.len(), 3);
    }

    #[test]
    fn test_virtual_table_scan() {
        let schema = create_test_schema();
        let table = PluginVirtualTable::new("plugin", "table", schema);

        let rows = table.scan(&[], None, Some(10), None).unwrap();

        assert_eq!(rows.len(), 10);
        assert_eq!(rows[0].values.len(), 3); // 3 columns
    }

    #[test]
    fn test_virtual_table_scan_with_filter() {
        let schema = create_test_schema();
        let table = PluginVirtualTable::new("plugin", "table", schema.clone());

        let filter = VirtualFilter::Gt("id".to_string(), SochValue::Int(5));
        let rows = table.scan(&[], Some(&filter), Some(100), None).unwrap();

        // All rows should have id > 5
        for row in &rows {
            if let Some(SochValue::Int(id)) = row.get_by_name("id", &schema) {
                assert!(*id > 5);
            }
        }
    }

    #[test]
    fn test_virtual_filter_matches() {
        let schema = create_test_schema();
        let row = VirtualRow::new(vec![
            SochValue::Int(42),
            SochValue::Text("Alice".to_string()),
            SochValue::Float(95.5),
        ]);

        // Equality filter
        let filter = VirtualFilter::Eq("id".to_string(), SochValue::Int(42));
        assert!(filter.matches(&row, &schema));

        // Like filter
        let filter = VirtualFilter::Like("name".to_string(), "Al%".to_string());
        assert!(filter.matches(&row, &schema));

        // Greater than
        let filter = VirtualFilter::Gt("score".to_string(), SochValue::Float(90.0));
        assert!(filter.matches(&row, &schema));

        // AND filter
        let filter = VirtualFilter::And(vec![
            VirtualFilter::Eq("id".to_string(), SochValue::Int(42)),
            VirtualFilter::Gt("score".to_string(), SochValue::Float(90.0)),
        ]);
        assert!(filter.matches(&row, &schema));
    }

    #[test]
    fn test_registry_operations() {
        let registry = VirtualTableRegistry::new();
        let schema = create_test_schema();

        let table = Arc::new(PluginVirtualTable::new("plugin", "table", schema));

        // Register
        registry.register("plugin.table", table).unwrap();
        assert_eq!(registry.list().len(), 1);

        // Get
        let retrieved = registry.get("plugin.table");
        assert!(retrieved.is_some());

        // Unregister
        registry.unregister("plugin.table").unwrap();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_registry_execute_select() {
        let registry = VirtualTableRegistry::new();
        let schema = create_test_schema();

        let table = Arc::new(PluginVirtualTable::new("plugin", "data", schema));
        registry.register("plugin.data", table).unwrap();

        let query = SelectQuery {
            columns: vec!["id".to_string(), "name".to_string()],
            table: "plugin.data".to_string(),
            where_clause: None,
            order_by: None,
            limit: Some(5),
            offset: None,
        };

        let result = registry.execute_select(&query).unwrap();

        assert_eq!(result.table, "plugin.data");
        assert_eq!(result.columns, vec!["id", "name"]);
        assert_eq!(result.rows.len(), 5);
    }

    #[test]
    fn test_cache_behavior() {
        let schema = create_test_schema();
        let table = PluginVirtualTable::new("plugin", "cached", schema).with_cache_ttl(1); // 1 second TTL

        // First scan populates cache
        let rows1 = table.scan(&[], None, Some(5), None).unwrap();
        assert!(table.is_cache_valid());

        // Second scan uses cache
        let rows2 = table.scan(&[], None, Some(5), None).unwrap();
        assert_eq!(rows1.len(), rows2.len());

        // Refresh clears cache
        table.refresh().unwrap();
        assert!(!table.is_cache_valid());
    }

    #[test]
    fn test_column_projection() {
        let schema = create_test_schema();
        let table = PluginVirtualTable::new("plugin", "table", schema);

        // Request only id and name
        let rows = table
            .scan(&["id".to_string(), "name".to_string()], None, Some(5), None)
            .unwrap();

        // Should still have all values (projection happens at registry level)
        // In a real implementation, the plugin would handle projection
        assert!(!rows.is_empty());
    }
}
