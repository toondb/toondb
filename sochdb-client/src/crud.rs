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

//! CRUD Operations (DML)
//!
//! Provides insert, update, delete, upsert, and find operations.

use crate::connection::SochConnection;
use crate::error::Result;
use crate::transaction::ClientTransaction;
use std::collections::HashMap;

use sochdb_core::soch::SochValue;

/// Insert result
#[derive(Debug, Clone)]
pub struct InsertResult {
    pub rows_inserted: usize,
    pub last_id: Option<u64>,
}

/// Update result  
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub rows_updated: usize,
}

/// Delete result
#[derive(Debug, Clone)]
pub struct DeleteResult {
    pub rows_deleted: usize,
}

/// Upsert result
#[derive(Debug, Clone)]
pub struct UpsertResult {
    pub rows_upserted: usize,
    pub rows_inserted: usize,
    pub rows_updated: usize,
}

/// Row builder for inserts
pub struct RowBuilder<'a> {
    conn: &'a SochConnection,
    table: String,
    rows: Vec<HashMap<String, SochValue>>,
    current: HashMap<String, SochValue>,
}

impl<'a> RowBuilder<'a> {
    /// Create new row builder
    pub fn new(conn: &'a SochConnection, table: &str) -> Self {
        Self {
            conn,
            table: table.to_string(),
            rows: Vec::new(),
            current: HashMap::new(),
        }
    }

    /// Set field value
    pub fn set(mut self, field: &str, value: impl Into<SochValue>) -> Self {
        self.current.insert(field.to_string(), value.into());
        self
    }

    /// Add current row and start new one
    pub fn row(mut self) -> Self {
        if !self.current.is_empty() {
            self.rows.push(std::mem::take(&mut self.current));
        }
        self
    }

    /// Execute insert
    pub fn execute(mut self) -> Result<InsertResult> {
        // Add final current row if not empty
        if !self.current.is_empty() {
            self.rows.push(std::mem::take(&mut self.current));
        }

        let rows_inserted = self.rows.len();
        let mut last_id = None;

        // Insert via TCH (in-memory columnar store)
        {
            let mut tch = self.conn.tch.write();
            for row in &self.rows {
                let id = tch.insert_row(&self.table, row);
                last_id = Some(id);
                
                // Persist to storage backend for durability
                // Key format: {table}:{row_id}
                // Value: bincode serialized row data
                if let Ok(value) = bincode::serialize(row) {
                    let key = format!("{}:{}", self.table, id);
                    let _ = self.conn.storage.put(key.as_bytes(), &value);
                }
            }
        }

        Ok(InsertResult {
            rows_inserted,
            last_id,
        })
    }
}

/// Update builder
pub struct UpdateBuilder<'a> {
    conn: &'a SochConnection,
    table: String,
    updates: HashMap<String, SochValue>,
    where_clause: Option<crate::connection::WhereClause>,
}

// Re-export from connection
pub use crate::connection::{CompareOp, UpsertAction, WhereClause};

impl<'a> UpdateBuilder<'a> {
    /// Create new update builder
    pub fn new(conn: &'a SochConnection, table: &str) -> Self {
        Self {
            conn,
            table: table.to_string(),
            updates: HashMap::new(),
            where_clause: None,
        }
    }

    /// Set field to new value
    pub fn set(mut self, field: &str, value: impl Into<SochValue>) -> Self {
        self.updates.insert(field.to_string(), value.into());
        self
    }

    /// Add WHERE clause
    pub fn where_eq(mut self, field: &str, value: impl Into<SochValue>) -> Self {
        self.where_clause = Some(WhereClause::Simple {
            field: field.to_string(),
            op: CompareOp::Eq,
            value: value.into(),
        });
        self
    }

    /// Add WHERE clause with custom operator
    pub fn where_cond(mut self, field: &str, op: CompareOp, value: impl Into<SochValue>) -> Self {
        self.where_clause = Some(WhereClause::Simple {
            field: field.to_string(),
            op,
            value: value.into(),
        });
        self
    }

    /// Execute update
    /// 
    /// Note: Updates are persisted to TCH in-memory store. 
    /// The MutationResult contains affected row IDs for storage-level operations.
    pub fn execute(self) -> Result<UpdateResult> {
        let mut tch = self.conn.tch.write();
        let mutation_result = tch.update_rows(&self.table, &self.updates, self.where_clause.as_ref());

        // TODO: Wire mutation_result.affected_row_ids to storage backend for WAL/index/CDC
        Ok(UpdateResult { rows_updated: mutation_result.affected_count })
    }
}

/// Delete builder
pub struct DeleteBuilder<'a> {
    conn: &'a SochConnection,
    table: String,
    where_clause: Option<WhereClause>,
}

impl<'a> DeleteBuilder<'a> {
    /// Create new delete builder
    pub fn new(conn: &'a SochConnection, table: &str) -> Self {
        Self {
            conn,
            table: table.to_string(),
            where_clause: None,
        }
    }

    /// Add WHERE clause
    pub fn where_eq(mut self, field: &str, value: impl Into<SochValue>) -> Self {
        self.where_clause = Some(WhereClause::Simple {
            field: field.to_string(),
            op: CompareOp::Eq,
            value: value.into(),
        });
        self
    }

    /// Add WHERE clause with custom operator
    pub fn where_cond(mut self, field: &str, op: CompareOp, value: impl Into<SochValue>) -> Self {
        self.where_clause = Some(WhereClause::Simple {
            field: field.to_string(),
            op,
            value: value.into(),
        });
        self
    }

    /// Execute delete
    /// 
    /// Note: Deletes are persisted to TCH in-memory store.
    /// The MutationResult contains affected row IDs for storage-level operations.
    pub fn execute(self) -> Result<DeleteResult> {
        let mut tch = self.conn.tch.write();
        let mutation_result = tch.delete_rows(&self.table, self.where_clause.as_ref());

        // TODO: Wire mutation_result.affected_row_ids to storage backend for WAL/index/CDC
        Ok(DeleteResult { rows_deleted: mutation_result.affected_count })
    }
}

/// Find (SELECT) builder
pub struct FindBuilder<'a> {
    conn: &'a SochConnection,
    table: String,
    columns: Vec<String>,
    where_clause: Option<WhereClause>,
    order_by: Option<(String, bool)>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl<'a> FindBuilder<'a> {
    /// Create new find builder
    pub fn new(conn: &'a SochConnection, table: &str) -> Self {
        Self {
            conn,
            table: table.to_string(),
            columns: Vec::new(),
            where_clause: None,
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Select specific columns
    pub fn select(mut self, columns: &[&str]) -> Self {
        self.columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add WHERE clause
    pub fn where_eq(mut self, field: &str, value: impl Into<SochValue>) -> Self {
        self.where_clause = Some(WhereClause::Simple {
            field: field.to_string(),
            op: CompareOp::Eq,
            value: value.into(),
        });
        self
    }

    /// Add WHERE clause with custom operator
    pub fn where_cond(mut self, field: &str, op: CompareOp, value: impl Into<SochValue>) -> Self {
        self.where_clause = Some(WhereClause::Simple {
            field: field.to_string(),
            op,
            value: value.into(),
        });
        self
    }

    /// Order by column
    pub fn order_by(mut self, column: &str, ascending: bool) -> Self {
        self.order_by = Some((column.to_string(), ascending));
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Offset results
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Execute find and return rows
    pub fn execute(self) -> Result<Vec<HashMap<String, SochValue>>> {
        let tch = self.conn.tch.read();
        let mut cursor = tch.select(
            &self.table,
            &self.columns,
            self.where_clause.as_ref(),
            self.order_by.as_ref(),
            self.limit,
            self.offset,
        );
        let mut rows = Vec::new();
        while let Some(row) = cursor.next() {
            rows.push(row);
        }
        Ok(rows)
    }

    /// Execute and return first row only
    pub fn first(mut self) -> Result<Option<HashMap<String, SochValue>>> {
        self.limit = Some(1);
        let rows = self.execute()?;
        Ok(rows.into_iter().next())
    }

    /// Execute and return all rows
    pub fn all(self) -> Result<Vec<HashMap<String, SochValue>>> {
        self.execute()
    }
}

/// Upsert builder
pub struct UpsertBuilder<'a> {
    conn: &'a SochConnection,
    table: String,
    conflict_key: String,
    rows: Vec<HashMap<String, SochValue>>,
    current: HashMap<String, SochValue>,
}

impl<'a> UpsertBuilder<'a> {
    /// Create new upsert builder
    pub fn new(conn: &'a SochConnection, table: &str, conflict_key: &str) -> Self {
        Self {
            conn,
            table: table.to_string(),
            conflict_key: conflict_key.to_string(),
            rows: Vec::new(),
            current: HashMap::new(),
        }
    }

    /// Set field value
    pub fn set(mut self, field: &str, value: impl Into<SochValue>) -> Self {
        self.current.insert(field.to_string(), value.into());
        self
    }

    /// Add current row and start new one
    pub fn row(mut self) -> Self {
        if !self.current.is_empty() {
            self.rows.push(std::mem::take(&mut self.current));
        }
        self
    }

    /// Execute upsert
    pub fn execute(mut self) -> Result<UpsertResult> {
        // Add final current row if not empty
        if !self.current.is_empty() {
            self.rows.push(std::mem::take(&mut self.current));
        }

        let mut rows_inserted = 0;
        let mut rows_updated = 0;

        {
            let mut tch = self.conn.tch.write();
            for row in self.rows {
                match tch.upsert_row(&self.table, &self.conflict_key, &row) {
                    UpsertAction::Inserted => rows_inserted += 1,
                    UpsertAction::Updated => rows_updated += 1,
                }
            }
        }

        Ok(UpsertResult {
            rows_upserted: rows_inserted + rows_updated,
            rows_inserted,
            rows_updated,
        })
    }
}

/// CRUD methods on connection
impl SochConnection {
    /// Start an INSERT operation
    pub fn insert_into<'a>(&'a self, table: &str) -> RowBuilder<'a> {
        RowBuilder::new(self, table)
    }

    /// Start an UPDATE operation
    pub fn update<'a>(&'a self, table: &str) -> UpdateBuilder<'a> {
        UpdateBuilder::new(self, table)
    }

    /// Start a DELETE operation
    pub fn delete_from<'a>(&'a self, table: &str) -> DeleteBuilder<'a> {
        DeleteBuilder::new(self, table)
    }

    /// Start a SELECT operation
    pub fn find<'a>(&'a self, table: &str) -> FindBuilder<'a> {
        FindBuilder::new(self, table)
    }

    /// Start an UPSERT operation
    pub fn upsert<'a>(&'a self, table: &str, conflict_key: &str) -> UpsertBuilder<'a> {
        UpsertBuilder::new(self, table, conflict_key)
    }

    /// Quick insert single row
    pub fn insert_one(
        &self,
        table: &str,
        values: HashMap<String, SochValue>,
    ) -> Result<InsertResult> {
        let mut builder = RowBuilder::new(self, table);
        for (k, v) in values {
            builder = builder.set(&k, v);
        }
        builder.execute()
    }

    /// Quick find by ID
    pub fn find_by_id(
        &self,
        table: &str,
        id_field: &str,
        id: impl Into<SochValue>,
    ) -> Result<Option<HashMap<String, SochValue>>> {
        self.find(table).where_eq(id_field, id).first()
    }

    /// Quick delete by ID
    pub fn delete_by_id(
        &self,
        table: &str,
        id_field: &str,
        id: impl Into<SochValue>,
    ) -> Result<DeleteResult> {
        self.delete_from(table).where_eq(id_field, id).execute()
    }

    /// Count rows in table
    pub fn count(&self, table: &str) -> Result<u64> {
        let tch = self.tch.read();
        Ok(tch.count_rows(table))
    }

    /// Check if row exists
    pub fn exists(&self, table: &str, field: &str, value: impl Into<SochValue>) -> Result<bool> {
        let row = self.find(table).where_eq(field, value).first()?;
        Ok(row.is_some())
    }
}

/// CRUD methods on transaction
impl<'a> ClientTransaction<'a> {
    /// Insert row in transaction
    pub fn insert(&mut self, table: &str, values: HashMap<String, SochValue>) -> Result<u64> {
        let mut tch = self.conn.tch.write();
        let id = tch.insert_row(table, &values);
        Ok(id)
    }

    /// Update rows in transaction
    pub fn update_where(
        &mut self,
        table: &str,
        updates: HashMap<String, SochValue>,
        field: &str,
        value: impl Into<SochValue>,
    ) -> Result<usize> {
        let mut tch = self.conn.tch.write();
        let where_clause = WhereClause::Simple {
            field: field.to_string(),
            op: CompareOp::Eq,
            value: value.into(),
        };
        let mutation_result = tch.update_rows(table, &updates, Some(&where_clause));
        // TODO: Wire mutation_result.affected_row_ids to transaction WAL
        Ok(mutation_result.affected_count)
    }

    /// Delete rows in transaction
    pub fn delete_where(
        &mut self,
        table: &str,
        field: &str,
        value: impl Into<SochValue>,
    ) -> Result<usize> {
        let mut tch = self.conn.tch.write();
        let where_clause = WhereClause::Simple {
            field: field.to_string(),
            op: CompareOp::Eq,
            value: value.into(),
        };
        let mutation_result = tch.delete_rows(table, Some(&where_clause));
        // TODO: Wire mutation_result.affected_row_ids to transaction WAL
        Ok(mutation_result.affected_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_builder() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .insert_into("users")
            .set("id", SochValue::Int(1))
            .set("name", SochValue::Text("Alice".to_string()))
            .execute()
            .unwrap();

        assert_eq!(result.rows_inserted, 1);
    }

    #[test]
    fn test_multi_row_insert() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .insert_into("users")
            .set("id", SochValue::Int(1))
            .set("name", SochValue::Text("Alice".to_string()))
            .row()
            .set("id", SochValue::Int(2))
            .set("name", SochValue::Text("Bob".to_string()))
            .execute()
            .unwrap();

        assert_eq!(result.rows_inserted, 2);
    }

    #[test]
    fn test_update_builder() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .update("users")
            .set("name", SochValue::Text("New Name".to_string()))
            .where_eq("id", SochValue::Int(1))
            .execute()
            .unwrap();

        // May be 0 if no matching rows
        // rows_updated is u64, so this is just to verify the operation completed
        let _ = result.rows_updated; // Verify field exists
    }

    #[test]
    fn test_delete_builder() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .delete_from("users")
            .where_eq("id", SochValue::Int(999))
            .execute()
            .unwrap();

        // rows_deleted is u64, so this is just to verify the operation completed
        let _ = result.rows_deleted; // Verify field exists
    }

    #[test]
    fn test_find_builder() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .find("users")
            .select(&["id", "name"])
            .where_eq("active", SochValue::Bool(true))
            .order_by("name", true)
            .limit(10)
            .execute();

        // Should compile and return result
        assert!(result.is_ok());
    }

    #[test]
    fn test_upsert() {
        let conn = SochConnection::open("./test").unwrap();

        let result = conn
            .upsert("users", "id")
            .set("id", SochValue::Int(1))
            .set("name", SochValue::Text("Alice".to_string()))
            .execute()
            .unwrap();

        assert!(result.rows_upserted >= 1);
    }

    #[test]
    fn test_convenience_methods() {
        let conn = SochConnection::open("./test").unwrap();

        // Quick find
        let _user = conn.find_by_id("users", "id", SochValue::Int(1));

        // Exists check
        let _exists = conn.exists(
            "users",
            "email",
            SochValue::Text("test@test.com".to_string()),
        );

        // Count
        let _count = conn.count("users");
    }
}
