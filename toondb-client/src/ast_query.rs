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

//! # AST-Based SQL Executor
//!
//! This module provides a proper AST-based SQL executor that replaces
//! the string-heuristic approach in `query.rs`.
//!
//! ## Architecture
//!
//! All SQL goes through a single pipeline:
//!
//! ```text
//! SQL Text → Lexer → Parser → AST → Executor → Result
//! ```
//!
//! ## Benefits
//!
//! 1. **Single source of truth**: One parser handles all SQL syntax
//! 2. **Dialect support**: MySQL, PostgreSQL, SQLite variants normalize to same AST
//! 3. **Robust parsing**: Handles comments, whitespace, schema-qualified names
//! 4. **Parameterized queries**: Proper placeholder indexing and validation
//! 5. **Extensible**: Add new SQL features by extending AST, not string parsing

use crate::connection::ToonConnection;
use crate::crud::{DeleteResult, InsertResult, UpdateResult};
use crate::error::{ClientError, Result};
use crate::schema::{CreateIndexResult, CreateTableResult, DropTableResult, SchemaBuilder};
use std::collections::HashMap;

use toondb_core::toon::{ToonType, ToonValue};
use toondb_query::sql::{
    BinaryOperator, ConflictAction, CreateIndexStmt, CreateTableStmt, DataType, DeleteStmt,
    DropIndexStmt, DropTableStmt, Expr, InsertSource, InsertStmt, Literal, ObjectName,
    OnConflict, Parser, SelectItem, SelectStmt, SqlDialect, Statement, UpdateStmt,
};

/// Query execution result
#[derive(Debug)]
pub enum QueryResult {
    /// SELECT result with rows
    Select(Vec<HashMap<String, ToonValue>>),
    /// INSERT result
    Insert(InsertResult),
    /// UPDATE result
    Update(UpdateResult),
    /// DELETE result
    Delete(DeleteResult),
    /// CREATE TABLE result
    CreateTable(CreateTableResult),
    /// DROP TABLE result
    DropTable(DropTableResult),
    /// CREATE INDEX result
    CreateIndex(CreateIndexResult),
    /// Empty result (e.g., SET, BEGIN, COMMIT)
    Empty,
}

/// AST-based TOON-QL query executor
///
/// This executor uses the proper SQL parser from toondb-query instead of
/// string heuristics, providing robust handling of all SQL dialects.
pub struct AstQueryExecutor<'a> {
    conn: &'a ToonConnection,
}

impl<'a> AstQueryExecutor<'a> {
    /// Create new AST-based query executor
    pub fn new(conn: &'a ToonConnection) -> Self {
        Self { conn }
    }

    /// Execute a SQL query
    ///
    /// This method parses SQL into an AST and executes it, supporting:
    /// - Standard SQL-92 syntax
    /// - PostgreSQL dialect (ON CONFLICT)
    /// - MySQL dialect (INSERT IGNORE, ON DUPLICATE KEY)
    /// - SQLite dialect (INSERT OR IGNORE/REPLACE)
    pub fn execute(&self, sql: &str) -> Result<QueryResult> {
        self.execute_with_params(sql, &[])
    }

    /// Execute a SQL query with parameters
    ///
    /// Parameters can use either positional (`$1`, `$2`) or question mark (`?`) style.
    pub fn execute_with_params(&self, sql: &str, params: &[ToonValue]) -> Result<QueryResult> {
        // Detect dialect for better error messages
        let dialect = SqlDialect::detect(sql);

        // Parse SQL into AST
        let stmt = Parser::parse(sql).map_err(|errors| {
            let msg = errors
                .iter()
                .map(|e| format!("Line {}: {}", e.span.line, e.message))
                .collect::<Vec<_>>()
                .join("; ");
            ClientError::Parse(format!("[{}] {}", dialect, msg))
        })?;

        // Execute the parsed statement
        self.execute_statement(&stmt, params)
    }

    /// Execute a parsed statement
    fn execute_statement(&self, stmt: &Statement, params: &[ToonValue]) -> Result<QueryResult> {
        match stmt {
            Statement::Select(select) => self.execute_select(select, params),
            Statement::Insert(insert) => self.execute_insert(insert, params),
            Statement::Update(update) => self.execute_update(update, params),
            Statement::Delete(delete) => self.execute_delete(delete, params),
            Statement::CreateTable(create) => self.execute_create_table(create),
            Statement::DropTable(drop) => self.execute_drop_table(drop),
            Statement::CreateIndex(create) => self.execute_create_index(create),
            Statement::DropIndex(drop) => self.execute_drop_index(drop),
            Statement::Begin(_) | Statement::Commit | Statement::Rollback(_) => {
                Ok(QueryResult::Empty)
            }
            _ => Err(ClientError::Parse(
                "Unsupported SQL statement type".to_string(),
            )),
        }
    }

    // ===== SELECT Execution =====

    fn execute_select(
        &self,
        select: &SelectStmt,
        params: &[ToonValue],
    ) -> Result<QueryResult> {
        // Extract table name
        let from = select
            .from
            .as_ref()
            .ok_or_else(|| ClientError::Parse("SELECT requires FROM clause".to_string()))?;

        if from.tables.len() != 1 {
            return Err(ClientError::Parse(
                "Multi-table queries not yet supported".to_string(),
            ));
        }

        let table_name = self.extract_table_name(&from.tables[0])?;

        // Build query
        let mut builder = self.conn.find(&table_name);

        // Handle column selection
        let columns: Vec<String> = self.extract_select_columns(&select.columns);
        if !columns.is_empty() && columns[0] != "*" {
            builder = builder.select(
                &columns.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            );
        }

        // Handle WHERE clause
        if let Some(where_clause) = &select.where_clause {
            if let Some((field, op, value)) = self.extract_simple_condition(where_clause, params)? {
                use crate::crud::CompareOp;
                let compare_op = match op.as_str() {
                    "=" => CompareOp::Eq,
                    "!=" | "<>" => CompareOp::Ne,
                    ">" => CompareOp::Gt,
                    ">=" => CompareOp::Ge,
                    "<" => CompareOp::Lt,
                    "<=" => CompareOp::Le,
                    _ => CompareOp::Eq,
                };
                builder = builder.where_cond(&field, compare_op, value);
            }
        }

        // Handle ORDER BY
        if !select.order_by.is_empty() {
            if let Some(col_name) = self.extract_column_name(&select.order_by[0].expr) {
                builder = builder.order_by(&col_name, select.order_by[0].asc);
            }
        }

        // Handle LIMIT
        if let Some(limit_expr) = &select.limit {
            if let Some(limit) = self.extract_integer(limit_expr) {
                builder = builder.limit(limit as usize);
            }
        }

        // Handle OFFSET
        if let Some(offset_expr) = &select.offset {
            if let Some(offset) = self.extract_integer(offset_expr) {
                builder = builder.offset(offset as usize);
            }
        }

        let result = builder.execute()?;
        Ok(QueryResult::Select(result))
    }

    // ===== INSERT Execution =====

    fn execute_insert(
        &self,
        insert: &InsertStmt,
        params: &[ToonValue],
    ) -> Result<QueryResult> {
        let table_name = insert.table.name();

        // Get values from source
        let rows = match &insert.source {
            InsertSource::Values(values) => values,
            _ => {
                return Err(ClientError::Parse(
                    "Only VALUES source supported for INSERT".to_string(),
                ))
            }
        };

        if rows.is_empty() {
            return Err(ClientError::Parse("No values to insert".to_string()));
        }

        // For now, handle single-row inserts
        let first_row = &rows[0];
        let mut builder = self.conn.insert_into(table_name);

        // Get column names
        let columns = insert.columns.as_ref();

        for (idx, expr) in first_row.iter().enumerate() {
            let default_col_name = format!("col{}", idx);
            let col_name = columns
                .and_then(|cols| cols.get(idx))
                .map(|s| s.as_str())
                .unwrap_or(&default_col_name);

            let value = self.evaluate_expr(expr, params)?;
            builder = builder.set(col_name, value);
        }

        // Handle ON CONFLICT
        if let Some(on_conflict) = &insert.on_conflict {
            match &on_conflict.action {
                ConflictAction::DoNothing => {
                    // Try to insert, ignore if conflict
                    match builder.execute() {
                        Ok(result) => return Ok(QueryResult::Insert(result)),
                        Err(ClientError::Constraint(_)) => {
                            return Ok(QueryResult::Insert(InsertResult {
                                last_id: None,
                                rows_inserted: 0,
                            }))
                        }
                        Err(e) => return Err(e),
                    }
                }
                ConflictAction::DoUpdate(assignments) => {
                    // First try insert, then update on conflict
                    match builder.execute() {
                        Ok(result) => return Ok(QueryResult::Insert(result)),
                        Err(ClientError::Constraint(_)) => {
                            // Do update instead
                            let mut update_builder = self.conn.update(table_name);
                            for assign in assignments {
                                let value = self.evaluate_expr(&assign.value, params)?;
                                update_builder = update_builder.set(&assign.column, value);
                            }
                            let update_result = update_builder.execute()?;
                            return Ok(QueryResult::Update(update_result));
                        }
                        Err(e) => return Err(e),
                    }
                }
                ConflictAction::DoReplace => {
                    // Delete existing, then insert
                    // This is a simplified implementation
                    match builder.execute() {
                        Ok(result) => return Ok(QueryResult::Insert(result)),
                        Err(ClientError::Constraint(_)) => {
                            // Would need to delete + insert
                            return Err(ClientError::Parse(
                                "REPLACE conflict action not fully implemented".to_string(),
                            ))
                        }
                        Err(e) => return Err(e),
                    }
                }
                _ => {}
            }
        }

        let result = builder.execute()?;
        Ok(QueryResult::Insert(result))
    }

    // ===== UPDATE Execution =====

    fn execute_update(
        &self,
        update: &UpdateStmt,
        params: &[ToonValue],
    ) -> Result<QueryResult> {
        let table_name = update.table.name();
        let mut builder = self.conn.update(table_name);

        // Apply assignments
        for assign in &update.assignments {
            let value = self.evaluate_expr(&assign.value, params)?;
            builder = builder.set(&assign.column, value);
        }

        // Apply WHERE clause
        if let Some(where_clause) = &update.where_clause {
            if let Some((field, _op, value)) = self.extract_simple_condition(where_clause, params)?
            {
                builder = builder.where_eq(&field, value);
            }
        }

        let result = builder.execute()?;
        Ok(QueryResult::Update(result))
    }

    // ===== DELETE Execution =====

    fn execute_delete(
        &self,
        delete: &DeleteStmt,
        params: &[ToonValue],
    ) -> Result<QueryResult> {
        let table_name = delete.table.name();
        let mut builder = self.conn.delete_from(table_name);

        // Apply WHERE clause
        if let Some(where_clause) = &delete.where_clause {
            if let Some((field, _op, value)) = self.extract_simple_condition(where_clause, params)?
            {
                builder = builder.where_eq(&field, value);
            }
        }

        let result = builder.execute()?;
        Ok(QueryResult::Delete(result))
    }

    // ===== DDL Execution =====

    fn execute_create_table(&self, create: &CreateTableStmt) -> Result<QueryResult> {
        // Handle IF NOT EXISTS - we check table catalog directly
        if create.if_not_exists {
            // Check if table exists in catalog
            let catalog = self.conn.catalog.read();
            if catalog.get_table(create.name.name()).is_some() {
                return Ok(QueryResult::CreateTable(CreateTableResult {
                    table_name: create.name.name().to_string(),
                    column_count: 0,
                }));
            }
            drop(catalog);
        }

        let mut schema = SchemaBuilder::table(create.name.name());

        for col in &create.columns {
            let toon_type = self.convert_data_type(&col.data_type);
            schema = schema.field(&col.name, toon_type).not_null().builder;
        }

        // Handle primary key from column constraints
        for col in &create.columns {
            for constraint in &col.constraints {
                if let toondb_query::sql::ColumnConstraint::PrimaryKey = constraint {
                    schema = schema.primary_key(&col.name);
                    break;
                }
            }
        }

        let result = self.conn.create_table(schema.build())?;
        Ok(QueryResult::CreateTable(result))
    }

    fn execute_drop_table(&self, drop_stmt: &DropTableStmt) -> Result<QueryResult> {
        // Handle IF EXISTS - check table catalog
        if drop_stmt.if_exists {
            for name in &drop_stmt.names {
                let catalog = self.conn.catalog.read();
                if catalog.get_table(name.name()).is_none() {
                    return Ok(QueryResult::DropTable(DropTableResult {
                        table_name: name.name().to_string(),
                        rows_deleted: 0,
                    }));
                }
                std::mem::drop(catalog);
            }
        }

        // Drop first table (for now)
        if let Some(name) = drop_stmt.names.first() {
            let result = self.conn.drop_table(name.name())?;
            Ok(QueryResult::DropTable(result))
        } else {
            Err(ClientError::Parse("No table name in DROP TABLE".to_string()))
        }
    }

    fn execute_create_index(&self, create: &CreateIndexStmt) -> Result<QueryResult> {
        // Handle IF NOT EXISTS
        if create.if_not_exists {
            // Would check if index exists
            // For now, just proceed
        }

        let cols: Vec<&str> = create.columns.iter().map(|c| c.name.as_str()).collect();
        let result = self.conn.create_index(
            &create.name,
            create.table.name(),
            &cols,
            create.unique,
        )?;
        Ok(QueryResult::CreateIndex(result))
    }

    fn execute_drop_index(&self, drop: &DropIndexStmt) -> Result<QueryResult> {
        // Handle IF EXISTS
        if drop.if_exists {
            // Would check if index exists
            // For now, just proceed
        }

        self.conn.drop_index(&drop.name)?;
        Ok(QueryResult::Empty)
    }

    // ===== Helper Methods =====

    fn extract_table_name(
        &self,
        table_ref: &toondb_query::sql::TableRef,
    ) -> Result<String> {
        match table_ref {
            toondb_query::sql::TableRef::Table { name, .. } => Ok(name.name().to_string()),
            _ => Err(ClientError::Parse(
                "Complex table references not supported".to_string(),
            )),
        }
    }

    fn extract_select_columns(&self, items: &[SelectItem]) -> Vec<String> {
        items
            .iter()
            .filter_map(|item| match item {
                SelectItem::Wildcard => Some("*".to_string()),
                SelectItem::QualifiedWildcard(t) => Some(format!("{}.*", t)),
                SelectItem::Expr { expr, alias } => {
                    alias.clone().or_else(|| self.extract_column_name(expr))
                }
            })
            .collect()
    }

    fn extract_column_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Column(col) => Some(col.column.clone()),
            _ => None,
        }
    }

    fn extract_integer(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Literal(Literal::Integer(n)) => Some(*n),
            _ => None,
        }
    }

    fn extract_simple_condition(
        &self,
        expr: &Expr,
        params: &[ToonValue],
    ) -> Result<Option<(String, String, ToonValue)>> {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                let field = match left.as_ref() {
                    Expr::Column(col) => col.column.clone(),
                    _ => return Ok(None),
                };

                let op_str = match op {
                    BinaryOperator::Eq => "=",
                    BinaryOperator::Ne => "!=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Ge => ">=",
                    _ => return Ok(None),
                };

                let value = self.evaluate_expr(right, params)?;
                Ok(Some((field, op_str.to_string(), value)))
            }
            _ => Ok(None),
        }
    }

    fn evaluate_expr(&self, expr: &Expr, params: &[ToonValue]) -> Result<ToonValue> {
        match expr {
            Expr::Literal(lit) => self.literal_to_toon_value(lit),
            Expr::Placeholder(n) => {
                let idx = (*n as usize).saturating_sub(1);
                params
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| ClientError::Parse(format!("Missing parameter ${}", n)))
            }
            _ => Err(ClientError::Parse(
                "Complex expressions not yet supported".to_string(),
            )),
        }
    }

    fn literal_to_toon_value(&self, lit: &Literal) -> Result<ToonValue> {
        Ok(match lit {
            Literal::Null => ToonValue::Null,
            Literal::Boolean(b) => ToonValue::Bool(*b),
            Literal::Integer(n) => ToonValue::Int(*n),
            Literal::Float(f) => ToonValue::Float(*f),
            Literal::String(s) => ToonValue::Text(s.clone()),
            Literal::Blob(b) => ToonValue::Binary(b.clone()),
        })
    }

    fn convert_data_type(&self, dt: &DataType) -> ToonType {
        match dt {
            DataType::TinyInt | DataType::SmallInt | DataType::Int | DataType::BigInt => {
                ToonType::Int
            }
            DataType::Float | DataType::Double | DataType::Decimal { .. } => ToonType::Float,
            DataType::Char(_) | DataType::Varchar(_) | DataType::Text => ToonType::Text,
            DataType::Binary(_) | DataType::Varbinary(_) | DataType::Blob => ToonType::Binary,
            DataType::Boolean => ToonType::Bool,
            DataType::Date | DataType::Time | DataType::Timestamp | DataType::DateTime => {
                ToonType::Text
            }
            DataType::Json | DataType::Jsonb => ToonType::Text,
            // Vector types map to Array of Float
            DataType::Vector(_dims) => ToonType::Array(Box::new(ToonType::Float)),
            DataType::Embedding(_dims) => ToonType::Array(Box::new(ToonType::Float)),
            DataType::Custom(_) | DataType::Interval => ToonType::Text,
        }
    }
}

/// SQL query methods on connection using AST-based executor
impl ToonConnection {
    /// Execute TOON-QL query using AST-based parser
    ///
    /// This is the recommended way to execute SQL queries. It uses
    /// the proper SQL parser and supports all dialects.
    pub fn query_ast(&self, sql: &str) -> Result<QueryResult> {
        AstQueryExecutor::new(self).execute(sql)
    }

    /// Execute SQL query with parameters using AST-based parser
    pub fn query_ast_params(&self, sql: &str, params: &[ToonValue]) -> Result<QueryResult> {
        AstQueryExecutor::new(self).execute_with_params(sql, params)
    }

    /// Execute and get rows (for SELECT) using AST-based parser
    pub fn query_rows_ast(&self, sql: &str) -> Result<Vec<HashMap<String, ToonValue>>> {
        match self.query_ast(sql)? {
            QueryResult::Select(result) => Ok(result),
            _ => Err(ClientError::Parse("Expected SELECT query".to_string())),
        }
    }

    /// Execute non-query SQL using AST-based parser
    pub fn execute_ast(&self, sql: &str) -> Result<u64> {
        match self.query_ast(sql)? {
            QueryResult::Insert(r) => Ok(r.rows_inserted as u64),
            QueryResult::Update(r) => Ok(r.rows_updated as u64),
            QueryResult::Delete(r) => Ok(r.rows_deleted as u64),
            QueryResult::CreateTable(_) => Ok(0),
            QueryResult::DropTable(_) => Ok(0),
            QueryResult::CreateIndex(_) => Ok(0),
            QueryResult::Empty => Ok(0),
            QueryResult::Select(_) => Err(ClientError::Parse(
                "Use query_rows_ast() for SELECT".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_detection() {
        assert_eq!(SqlDialect::detect("SELECT * FROM users"), SqlDialect::Standard);
        assert_eq!(
            SqlDialect::detect("INSERT IGNORE INTO users VALUES (1)"),
            SqlDialect::MySQL
        );
        assert_eq!(
            SqlDialect::detect("INSERT OR IGNORE INTO users VALUES (1)"),
            SqlDialect::SQLite
        );
        assert_eq!(
            SqlDialect::detect("INSERT INTO users VALUES (1) ON CONFLICT DO NOTHING"),
            SqlDialect::PostgreSQL
        );
    }

    #[test]
    fn test_parse_select() {
        let stmt = Parser::parse("SELECT id, name FROM users WHERE active = true").unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_parse_insert_ignore() {
        let stmt = Parser::parse("INSERT IGNORE INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
        } else {
            panic!("Expected INSERT statement");
        }
    }
}
