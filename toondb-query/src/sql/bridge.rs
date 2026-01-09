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

//! # SQL Execution Bridge
//!
//! Unified SQL execution pipeline that routes all SQL through a single AST.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │   SQL Text  │ --> │   Lexer     │ --> │   Parser    │ --> │    AST      │
//! └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
//!                                                                    │
//!                     ┌──────────────────────────────────────────────┘
//!                     │
//!                     v
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  Executor   │ <-- │  Planner    │ <-- │  Validator  │
//! └─────────────┘     └─────────────┘     └─────────────┘
//!       │
//!       v
//! ┌─────────────┐
//! │   Result    │
//! └─────────────┘
//! ```
//!
//! ## Benefits
//!
//! 1. **Single parser**: All SQL goes through one lexer/parser
//! 2. **Type-safe AST**: Structured representation of all queries
//! 3. **Dialect normalization**: MySQL/PostgreSQL/SQLite → canonical AST
//! 4. **Extensible**: Add new features by extending AST, not string parsing

use super::ast::*;
use super::compatibility::SqlDialect;
use super::error::{SqlError, SqlResult};
use super::parser::Parser;
use std::collections::HashMap;
use toondb_core::ToonValue;

/// Execution result types
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// SELECT query result
    Rows {
        columns: Vec<String>,
        rows: Vec<HashMap<String, ToonValue>>,
    },
    /// DML result (INSERT/UPDATE/DELETE)
    RowsAffected(usize),
    /// DDL result (CREATE/DROP/ALTER)
    Ok,
    /// Transaction control result
    TransactionOk,
}

impl ExecutionResult {
    /// Get rows if this is a SELECT result
    pub fn rows(&self) -> Option<&Vec<HashMap<String, ToonValue>>> {
        match self {
            ExecutionResult::Rows { rows, .. } => Some(rows),
            _ => None,
        }
    }

    /// Get column names if this is a SELECT result
    pub fn columns(&self) -> Option<&Vec<String>> {
        match self {
            ExecutionResult::Rows { columns, .. } => Some(columns),
            _ => None,
        }
    }

    /// Get affected row count
    pub fn rows_affected(&self) -> usize {
        match self {
            ExecutionResult::RowsAffected(n) => *n,
            ExecutionResult::Rows { rows, .. } => rows.len(),
            _ => 0,
        }
    }
}

/// Storage connection trait for executing SQL against actual storage
///
/// Implementations of this trait provide the bridge between parsed SQL
/// and the underlying storage engine.
pub trait SqlConnection {
    /// Execute a SELECT query
    fn select(
        &self,
        table: &str,
        columns: &[String],
        where_clause: Option<&Expr>,
        order_by: &[OrderByItem],
        limit: Option<usize>,
        offset: Option<usize>,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult>;

    /// Execute an INSERT
    fn insert(
        &mut self,
        table: &str,
        columns: Option<&[String]>,
        rows: &[Vec<Expr>],
        on_conflict: Option<&OnConflict>,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult>;

    /// Execute an UPDATE
    fn update(
        &mut self,
        table: &str,
        assignments: &[Assignment],
        where_clause: Option<&Expr>,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult>;

    /// Execute a DELETE
    fn delete(
        &mut self,
        table: &str,
        where_clause: Option<&Expr>,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult>;

    /// Create a table
    fn create_table(&mut self, stmt: &CreateTableStmt) -> SqlResult<ExecutionResult>;

    /// Drop a table
    fn drop_table(&mut self, stmt: &DropTableStmt) -> SqlResult<ExecutionResult>;

    /// Create an index
    fn create_index(&mut self, stmt: &CreateIndexStmt) -> SqlResult<ExecutionResult>;

    /// Drop an index
    fn drop_index(&mut self, stmt: &DropIndexStmt) -> SqlResult<ExecutionResult>;

    /// Begin transaction
    fn begin(&mut self, stmt: &BeginStmt) -> SqlResult<ExecutionResult>;

    /// Commit transaction
    fn commit(&mut self) -> SqlResult<ExecutionResult>;

    /// Rollback transaction
    fn rollback(&mut self, savepoint: Option<&str>) -> SqlResult<ExecutionResult>;

    /// Check if table exists
    fn table_exists(&self, table: &str) -> SqlResult<bool>;

    /// Check if index exists
    fn index_exists(&self, index: &str) -> SqlResult<bool>;
}

/// Unified SQL executor that routes through AST
pub struct SqlBridge<C: SqlConnection> {
    conn: C,
}

impl<C: SqlConnection> SqlBridge<C> {
    /// Create a new SQL bridge with the given connection
    pub fn new(conn: C) -> Self {
        Self { conn }
    }

    /// Execute a SQL statement
    pub fn execute(&mut self, sql: &str) -> SqlResult<ExecutionResult> {
        self.execute_with_params(sql, &[])
    }

    /// Execute a SQL statement with parameters
    pub fn execute_with_params(
        &mut self,
        sql: &str,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult> {
        // Detect dialect for better error messages
        let _dialect = SqlDialect::detect(sql);

        // Parse SQL into AST
        let stmt = Parser::parse(sql).map_err(SqlError::from_parse_errors)?;

        // Validate placeholder count
        let max_placeholder = self.find_max_placeholder(&stmt);
        if max_placeholder as usize > params.len() {
            return Err(SqlError::InvalidArgument(format!(
                "Query contains {} placeholders but only {} parameters provided",
                max_placeholder,
                params.len()
            )));
        }

        // Execute statement
        self.execute_statement(&stmt, params)
    }

    /// Execute a parsed statement
    pub fn execute_statement(
        &mut self,
        stmt: &Statement,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult> {
        match stmt {
            Statement::Select(select) => self.execute_select(select, params),
            Statement::Insert(insert) => self.execute_insert(insert, params),
            Statement::Update(update) => self.execute_update(update, params),
            Statement::Delete(delete) => self.execute_delete(delete, params),
            Statement::CreateTable(create) => self.execute_create_table(create),
            Statement::DropTable(drop) => self.execute_drop_table(drop),
            Statement::CreateIndex(create) => self.execute_create_index(create),
            Statement::DropIndex(drop) => self.execute_drop_index(drop),
            Statement::AlterTable(_alter) => Err(SqlError::NotImplemented(
                "ALTER TABLE not yet implemented".into(),
            )),
            Statement::Begin(begin) => self.conn.begin(begin),
            Statement::Commit => self.conn.commit(),
            Statement::Rollback(savepoint) => self.conn.rollback(savepoint.as_deref()),
            Statement::Savepoint(_name) => Err(SqlError::NotImplemented(
                "SAVEPOINT not yet implemented".into(),
            )),
            Statement::Release(_name) => Err(SqlError::NotImplemented(
                "RELEASE SAVEPOINT not yet implemented".into(),
            )),
            Statement::Explain(_stmt) => Err(SqlError::NotImplemented(
                "EXPLAIN not yet implemented".into(),
            )),
        }
    }

    fn execute_select(
        &self,
        select: &SelectStmt,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult> {
        // Get table from FROM clause
        let from = select
            .from
            .as_ref()
            .ok_or_else(|| SqlError::InvalidArgument("SELECT requires FROM clause".into()))?;

        if from.tables.len() != 1 {
            return Err(SqlError::NotImplemented(
                "Multi-table queries not yet supported".into(),
            ));
        }

        let table_name = match &from.tables[0] {
            TableRef::Table { name, .. } => name.name().to_string(),
            TableRef::Subquery { .. } => {
                return Err(SqlError::NotImplemented(
                    "Subqueries not yet supported".into(),
                ));
            }
            TableRef::Join { .. } => {
                return Err(SqlError::NotImplemented(
                    "JOINs not yet supported".into(),
                ));
            }
            TableRef::Function { .. } => {
                return Err(SqlError::NotImplemented(
                    "Table functions not yet supported".into(),
                ));
            }
        };

        // Extract column names
        let columns = self.extract_select_columns(&select.columns)?;

        // Extract LIMIT/OFFSET
        let limit = self.extract_limit(&select.limit)?;
        let offset = self.extract_limit(&select.offset)?;

        self.conn.select(
            &table_name,
            &columns,
            select.where_clause.as_ref(),
            &select.order_by,
            limit,
            offset,
            params,
        )
    }

    fn execute_insert(
        &mut self,
        insert: &InsertStmt,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = insert.table.name();

        let rows = match &insert.source {
            InsertSource::Values(values) => values,
            InsertSource::Query(_) => {
                return Err(SqlError::NotImplemented(
                    "INSERT ... SELECT not yet supported".into(),
                ));
            }
            InsertSource::Default => {
                return Err(SqlError::NotImplemented(
                    "INSERT DEFAULT VALUES not yet supported".into(),
                ));
            }
        };

        self.conn.insert(
            table_name,
            insert.columns.as_deref(),
            rows,
            insert.on_conflict.as_ref(),
            params,
        )
    }

    fn execute_update(
        &mut self,
        update: &UpdateStmt,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = update.table.name();

        self.conn.update(
            table_name,
            &update.assignments,
            update.where_clause.as_ref(),
            params,
        )
    }

    fn execute_delete(
        &mut self,
        delete: &DeleteStmt,
        params: &[ToonValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = delete.table.name();

        self.conn.delete(
            table_name,
            delete.where_clause.as_ref(),
            params,
        )
    }

    fn execute_create_table(&mut self, stmt: &CreateTableStmt) -> SqlResult<ExecutionResult> {
        // Handle IF NOT EXISTS
        if stmt.if_not_exists {
            let table_name = stmt.name.name();
            if self.conn.table_exists(table_name)? {
                return Ok(ExecutionResult::Ok);
            }
        }

        self.conn.create_table(stmt)
    }

    fn execute_drop_table(&mut self, stmt: &DropTableStmt) -> SqlResult<ExecutionResult> {
        // Handle IF EXISTS
        if stmt.if_exists {
            for name in &stmt.names {
                if !self.conn.table_exists(name.name())? {
                    return Ok(ExecutionResult::Ok);
                }
            }
        }

        self.conn.drop_table(stmt)
    }

    fn execute_create_index(&mut self, stmt: &CreateIndexStmt) -> SqlResult<ExecutionResult> {
        // Handle IF NOT EXISTS
        if stmt.if_not_exists {
            if self.conn.index_exists(&stmt.name)? {
                return Ok(ExecutionResult::Ok);
            }
        }

        self.conn.create_index(stmt)
    }

    fn execute_drop_index(&mut self, stmt: &DropIndexStmt) -> SqlResult<ExecutionResult> {
        // Handle IF EXISTS
        if stmt.if_exists {
            if !self.conn.index_exists(&stmt.name)? {
                return Ok(ExecutionResult::Ok);
            }
        }

        self.conn.drop_index(stmt)
    }

    /// Extract column names from SELECT list
    fn extract_select_columns(&self, items: &[SelectItem]) -> SqlResult<Vec<String>> {
        let mut columns = Vec::new();

        for item in items {
            match item {
                SelectItem::Wildcard => columns.push("*".to_string()),
                SelectItem::QualifiedWildcard(table) => columns.push(format!("{}.*", table)),
                SelectItem::Expr { expr, alias } => {
                    let name = alias.clone().unwrap_or_else(|| match expr {
                        Expr::Column(col) => col.column.clone(),
                        Expr::Function(func) => format!("{}()", func.name.name()),
                        _ => "?column?".to_string(),
                    });
                    columns.push(name);
                }
            }
        }

        Ok(columns)
    }

    /// Extract LIMIT/OFFSET value
    fn extract_limit(&self, expr: &Option<Expr>) -> SqlResult<Option<usize>> {
        match expr {
            Some(Expr::Literal(Literal::Integer(n))) => Ok(Some(*n as usize)),
            Some(_) => Err(SqlError::InvalidArgument(
                "LIMIT/OFFSET must be an integer literal".into(),
            )),
            None => Ok(None),
        }
    }

    /// Find the maximum placeholder index in a statement
    fn find_max_placeholder(&self, stmt: &Statement) -> u32 {
        let mut visitor = PlaceholderVisitor::new();
        visitor.visit_statement(stmt);
        visitor.max_placeholder
    }
}

/// Visitor to find maximum placeholder index
struct PlaceholderVisitor {
    max_placeholder: u32,
}

impl PlaceholderVisitor {
    fn new() -> Self {
        Self { max_placeholder: 0 }
    }

    fn visit_statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Select(s) => self.visit_select(s),
            Statement::Insert(i) => self.visit_insert(i),
            Statement::Update(u) => self.visit_update(u),
            Statement::Delete(d) => self.visit_delete(d),
            _ => {}
        }
    }

    fn visit_select(&mut self, select: &SelectStmt) {
        for item in &select.columns {
            if let SelectItem::Expr { expr, .. } = item {
                self.visit_expr(expr);
            }
        }
        if let Some(where_clause) = &select.where_clause {
            self.visit_expr(where_clause);
        }
        if let Some(having) = &select.having {
            self.visit_expr(having);
        }
        for order in &select.order_by {
            self.visit_expr(&order.expr);
        }
        if let Some(limit) = &select.limit {
            self.visit_expr(limit);
        }
        if let Some(offset) = &select.offset {
            self.visit_expr(offset);
        }
    }

    fn visit_insert(&mut self, insert: &InsertStmt) {
        if let InsertSource::Values(rows) = &insert.source {
            for row in rows {
                for expr in row {
                    self.visit_expr(expr);
                }
            }
        }
    }

    fn visit_update(&mut self, update: &UpdateStmt) {
        for assign in &update.assignments {
            self.visit_expr(&assign.value);
        }
        if let Some(where_clause) = &update.where_clause {
            self.visit_expr(where_clause);
        }
    }

    fn visit_delete(&mut self, delete: &DeleteStmt) {
        if let Some(where_clause) = &delete.where_clause {
            self.visit_expr(where_clause);
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Placeholder(n) => {
                self.max_placeholder = self.max_placeholder.max(*n);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            Expr::UnaryOp { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Function(func) => {
                for arg in &func.args {
                    self.visit_expr(arg);
                }
            }
            Expr::Case { operand, conditions, else_result } => {
                if let Some(op) = operand {
                    self.visit_expr(op);
                }
                for (when, then) in conditions {
                    self.visit_expr(when);
                    self.visit_expr(then);
                }
                if let Some(else_expr) = else_result {
                    self.visit_expr(else_expr);
                }
            }
            Expr::InList { expr, list, .. } => {
                self.visit_expr(expr);
                for item in list {
                    self.visit_expr(item);
                }
            }
            Expr::Between { expr, low, high, .. } => {
                self.visit_expr(expr);
                self.visit_expr(low);
                self.visit_expr(high);
            }
            Expr::Cast { expr, .. } => {
                self.visit_expr(expr);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder_visitor() {
        let stmt = Parser::parse("SELECT * FROM users WHERE id = $1 AND name = $2").unwrap();
        let mut visitor = PlaceholderVisitor::new();
        visitor.visit_statement(&stmt);
        assert_eq!(visitor.max_placeholder, 2);
    }

    #[test]
    fn test_question_mark_placeholders() {
        let stmt = Parser::parse("SELECT * FROM users WHERE id = ? AND name = ?").unwrap();
        let mut visitor = PlaceholderVisitor::new();
        visitor.visit_statement(&stmt);
        assert_eq!(visitor.max_placeholder, 2);
    }

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
    }
}
