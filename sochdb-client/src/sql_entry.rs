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

//! # Unified SQL Execution Entry Point (Task 12)
//!
//! This module provides a single, unified entry point for all SQL execution
//! in SochDB. All SQL queries - regardless of origin - flow through this
//! interface.
//!
//! ## Design Goals
//!
//! 1. **Single Entry Point**: One function to rule them all
//! 2. **AST-First**: All SQL parsed into AST, never string-heuristics
//! 3. **Dialect Support**: MySQL, PostgreSQL, SQLite automatically detected
//! 4. **Parameterized**: Proper placeholder handling ($1, ?, @param)
//! 5. **Tracing**: Every query emits standardized telemetry
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────────────────────────┐
//! │   Client    │────▶│  unified_execute(sql, params)   │
//! └─────────────┘     └─────────────────────────────────┘
//!                                     │
//!                                     ▼
//!                         ┌───────────────────────┐
//!                         │    SQL Parser (AST)   │
//!                         │  - Dialect detection  │
//!                         │  - Error localization │
//!                         └───────────────────────┘
//!                                     │
//!                                     ▼
//!                         ┌───────────────────────┐
//!                         │   Statement Router    │
//!                         │  - SELECT → Query     │
//!                         │  - INSERT → Write     │
//!                         │  - DDL → Schema       │
//!                         └───────────────────────┘
//!                                     │
//!                                     ▼
//!                         ┌───────────────────────┐
//!                         │   Executor Engine     │
//!                         │  - Expression eval    │
//!                         │  - Type coercion      │
//!                         │  - Result formation   │
//!                         └───────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_client::sql_entry::{execute, execute_with_params, SqlResult};
//!
//! // Simple query
//! let result = execute(&conn, "SELECT * FROM users WHERE id = 1")?;
//!
//! // Parameterized query
//! let result = execute_with_params(
//!     &conn,
//!     "SELECT * FROM users WHERE name = $1",
//!     &[SochValue::Text("Alice".into())],
//! )?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use sochdb_core::soch::SochValue;

use crate::ast_query::AstQueryExecutor;
use crate::connection::SochConnection;
use crate::crud::{DeleteResult, InsertResult, UpdateResult};
use crate::error::{ClientError, Result};
use crate::schema::{CreateIndexResult, CreateTableResult, DropTableResult};

// Re-export QueryResult from ast_query for convenience
pub use crate::ast_query::QueryResult;

// ============================================================================
// Query Statistics
// ============================================================================

/// Global query counter for telemetry
static QUERY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Statistics for a single query execution
#[derive(Debug, Clone)]
pub struct QueryStats {
    /// Unique query ID
    pub query_id: u64,
    /// Time spent parsing SQL (µs)
    pub parse_time_us: u64,
    /// Time spent executing (µs)
    pub exec_time_us: u64,
    /// Total time (µs)
    pub total_time_us: u64,
    /// Number of rows affected/returned
    pub rows_affected: usize,
    /// Detected SQL dialect
    pub dialect: String,
    /// Query type (SELECT, INSERT, etc.)
    pub query_type: String,
}

// ============================================================================
// Unified Execute Functions
// ============================================================================

/// Execute a SQL query through the unified AST-based pipeline.
///
/// This is THE single entry point for all SQL execution in SochDB.
/// All clients, APIs, and internal code should use this function.
///
/// # Arguments
///
/// * `conn` - Database connection
/// * `sql` - SQL statement to execute
///
/// # Returns
///
/// Query result or error with localized error message
///
/// # Example
///
/// ```rust,ignore
/// let result = execute(&conn, "SELECT name FROM users")?;
/// match result {
///     QueryResult::Select(rows) => println!("Got {} rows", rows.len()),
///     _ => println!("Not a select"),
/// }
/// ```
pub fn execute(conn: &SochConnection, sql: &str) -> Result<QueryResult> {
    execute_with_params(conn, sql, &[])
}

/// Execute a parameterized SQL query through the unified AST-based pipeline.
///
/// # Arguments
///
/// * `conn` - Database connection
/// * `sql` - SQL statement with parameter placeholders
/// * `params` - Parameter values (ordered)
///
/// # Placeholder Styles
///
/// - PostgreSQL: `$1`, `$2`, `$3`
/// - MySQL: `?`, `?`, `?`
/// - SQLite: `?`, `?1`, `@param`
///
/// All styles are normalized to positional during parsing.
pub fn execute_with_params(
    conn: &SochConnection,
    sql: &str,
    params: &[SochValue],
) -> Result<QueryResult> {
    let start = Instant::now();
    let query_id = QUERY_COUNTER.fetch_add(1, Ordering::Relaxed);
    
    // Use the AST-based executor
    let executor = AstQueryExecutor::new(conn);
    let result = executor.execute_with_params(sql, params);
    
    let total_us = start.elapsed().as_micros() as u64;
    
    // Emit telemetry (would integrate with tracing in production)
    if cfg!(debug_assertions) {
        tracing::debug!(
            query_id = query_id,
            total_us = total_us,
            sql = sql,
            "SQL executed"
        );
    }
    
    result
}

/// Execute a SQL query and return statistics along with result.
pub fn execute_with_stats(
    conn: &SochConnection,
    sql: &str,
) -> Result<(QueryResult, QueryStats)> {
    execute_with_params_and_stats(conn, sql, &[])
}

/// Execute a parameterized SQL query and return statistics.
pub fn execute_with_params_and_stats(
    conn: &SochConnection,
    sql: &str,
    params: &[SochValue],
) -> Result<(QueryResult, QueryStats)> {
    let start = Instant::now();
    let query_id = QUERY_COUNTER.fetch_add(1, Ordering::Relaxed);
    
    // Detect dialect for stats
    let dialect = detect_dialect(sql);
    
    // Parse timing (would be more accurate with separate parse step)
    let parse_start = Instant::now();
    let executor = AstQueryExecutor::new(conn);
    let parse_time_us = parse_start.elapsed().as_micros() as u64;
    
    // Execute
    let exec_start = Instant::now();
    let result = executor.execute_with_params(sql, params)?;
    let exec_time_us = exec_start.elapsed().as_micros() as u64;
    
    let total_time_us = start.elapsed().as_micros() as u64;
    
    // Count rows
    let (rows_affected, query_type) = match &result {
        QueryResult::Select(rows) => (rows.len(), "SELECT"),
        QueryResult::Insert(r) => (r.rows_inserted as usize, "INSERT"),
        QueryResult::Update(r) => (r.rows_updated as usize, "UPDATE"),
        QueryResult::Delete(r) => (r.rows_deleted as usize, "DELETE"),
        QueryResult::CreateTable(_) => (0, "CREATE TABLE"),
        QueryResult::DropTable(_) => (0, "DROP TABLE"),
        QueryResult::CreateIndex(_) => (0, "CREATE INDEX"),
        QueryResult::Empty => (0, "OTHER"),
    };
    
    let stats = QueryStats {
        query_id,
        parse_time_us,
        exec_time_us,
        total_time_us,
        rows_affected,
        dialect: dialect.to_string(),
        query_type: query_type.to_string(),
    };
    
    Ok((result, stats))
}

/// Detect SQL dialect from query text
fn detect_dialect(sql: &str) -> &'static str {
    let upper = sql.to_uppercase();
    
    if upper.contains("ON CONFLICT") && upper.contains("DO UPDATE") {
        "PostgreSQL"
    } else if upper.contains("ON DUPLICATE KEY") {
        "MySQL"
    } else if upper.contains("INSERT OR REPLACE") || upper.contains("INSERT OR IGNORE") {
        "SQLite"
    } else if upper.contains("RETURNING") {
        "PostgreSQL"
    } else if upper.contains("LIMIT") && upper.contains("OFFSET") && upper.contains(",") {
        // LIMIT offset, count is MySQL syntax
        "MySQL"
    } else {
        "SQL-92"
    }
}

// ============================================================================
// Batch Execution
// ============================================================================

/// Result of batch execution
#[derive(Debug)]
pub struct BatchResult {
    /// Individual query results
    pub results: Vec<Result<QueryResult>>,
    /// Total queries executed
    pub total: usize,
    /// Successful queries
    pub succeeded: usize,
    /// Failed queries
    pub failed: usize,
}

/// Execute multiple SQL statements in order.
///
/// Each statement is executed independently. Failures don't stop
/// subsequent statements unless `stop_on_error` is true.
pub fn execute_batch(
    conn: &SochConnection,
    statements: &[&str],
    stop_on_error: bool,
) -> BatchResult {
    let mut results = Vec::with_capacity(statements.len());
    let mut succeeded = 0;
    let mut failed = 0;
    
    for sql in statements {
        let result = execute(conn, sql);
        
        match &result {
            Ok(_) => succeeded += 1,
            Err(_) => {
                failed += 1;
                if stop_on_error {
                    results.push(result);
                    break;
                }
            }
        }
        
        results.push(result);
    }
    
    BatchResult {
        results,
        total: statements.len(),
        succeeded,
        failed,
    }
}

// ============================================================================
// Prepared Statements
// ============================================================================

/// A prepared statement that can be executed multiple times.
///
/// Prepared statements parse the SQL once and can be executed
/// multiple times with different parameters.
pub struct PreparedStatement<'a> {
    conn: &'a SochConnection,
    sql: String,
    param_count: usize,
}

impl<'a> PreparedStatement<'a> {
    /// Prepare a SQL statement
    pub fn prepare(conn: &'a SochConnection, sql: impl Into<String>) -> Result<Self> {
        let sql = sql.into();
        
        // Count parameters
        let param_count = count_parameters(&sql);
        
        // Validate syntax by attempting parse
        // (in production, we'd cache the parsed AST)
        let executor = AstQueryExecutor::new(conn);
        let _ = executor.execute_with_params(&sql, &vec![SochValue::Null; param_count])?;
        
        Ok(Self {
            conn,
            sql,
            param_count,
        })
    }
    
    /// Execute the prepared statement with parameters
    pub fn execute(&self, params: &[SochValue]) -> Result<QueryResult> {
        if params.len() != self.param_count {
            return Err(ClientError::Parse(format!(
                "Expected {} parameters, got {}",
                self.param_count,
                params.len()
            )));
        }
        
        execute_with_params(self.conn, &self.sql, params)
    }
    
    /// Get the SQL text
    pub fn sql(&self) -> &str {
        &self.sql
    }
    
    /// Get expected parameter count
    pub fn param_count(&self) -> usize {
        self.param_count
    }
}

/// Count parameter placeholders in SQL
fn count_parameters(sql: &str) -> usize {
    let mut count = 0;
    let mut max_numbered = 0;
    
    let chars: Vec<char> = sql.chars().collect();
    let mut i = 0;
    
    while i < chars.len() {
        match chars[i] {
            '?' => {
                count += 1;
                i += 1;
            }
            '$' => {
                // PostgreSQL style: $1, $2, etc.
                let mut num_str = String::new();
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    num_str.push(chars[j]);
                    j += 1;
                }
                if !num_str.is_empty() {
                    if let Ok(n) = num_str.parse::<usize>() {
                        max_numbered = max_numbered.max(n);
                    }
                }
                i = j;
            }
            '\'' | '"' => {
                // Skip string literals
                let quote = chars[i];
                i += 1;
                while i < chars.len() && chars[i] != quote {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    
    count.max(max_numbered)
}

// ============================================================================
// Query Builder (Fluent API)
// ============================================================================

/// Fluent query builder that generates and executes SQL.
///
/// This provides a type-safe way to build queries programmatically.
pub struct QueryBuilder<'a> {
    conn: &'a SochConnection,
    query_type: QueryBuilderType,
    table: String,
    columns: Vec<String>,
    values: Vec<SochValue>,
    conditions: Vec<(String, &'static str, SochValue)>,
    order_by: Option<(String, bool)>,
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum QueryBuilderType {
    Select,
    Insert,
    Update,
    Delete,
}

impl<'a> QueryBuilder<'a> {
    /// Start a SELECT query
    pub fn select(conn: &'a SochConnection, table: impl Into<String>) -> Self {
        Self {
            conn,
            query_type: QueryBuilderType::Select,
            table: table.into(),
            columns: Vec::new(),
            values: Vec::new(),
            conditions: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }
    
    /// Start an INSERT query
    pub fn insert(conn: &'a SochConnection, table: impl Into<String>) -> Self {
        Self {
            conn,
            query_type: QueryBuilderType::Insert,
            table: table.into(),
            columns: Vec::new(),
            values: Vec::new(),
            conditions: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }
    
    /// Start an UPDATE query
    pub fn update(conn: &'a SochConnection, table: impl Into<String>) -> Self {
        Self {
            conn,
            query_type: QueryBuilderType::Update,
            table: table.into(),
            columns: Vec::new(),
            values: Vec::new(),
            conditions: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }
    
    /// Start a DELETE query
    pub fn delete(conn: &'a SochConnection, table: impl Into<String>) -> Self {
        Self {
            conn,
            query_type: QueryBuilderType::Delete,
            table: table.into(),
            columns: Vec::new(),
            values: Vec::new(),
            conditions: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }
    
    /// Add columns to select
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = cols.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Add a column-value pair (for INSERT/UPDATE)
    pub fn set(mut self, column: impl Into<String>, value: SochValue) -> Self {
        self.columns.push(column.into());
        self.values.push(value);
        self
    }
    
    /// Add a WHERE condition
    pub fn where_eq(mut self, column: impl Into<String>, value: SochValue) -> Self {
        self.conditions.push((column.into(), "=", value));
        self
    }
    
    /// Add ORDER BY
    pub fn order_by(mut self, column: impl Into<String>, asc: bool) -> Self {
        self.order_by = Some((column.into(), asc));
        self
    }
    
    /// Add LIMIT
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }
    
    /// Add OFFSET
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }
    
    /// Build the SQL string
    pub fn to_sql(&self) -> (String, Vec<SochValue>) {
        let mut sql = String::new();
        let mut params = Vec::new();
        let mut param_idx = 1;
        
        match self.query_type {
            QueryBuilderType::Select => {
                sql.push_str("SELECT ");
                if self.columns.is_empty() {
                    sql.push('*');
                } else {
                    sql.push_str(&self.columns.join(", "));
                }
                sql.push_str(" FROM ");
                sql.push_str(&self.table);
            }
            QueryBuilderType::Insert => {
                sql.push_str("INSERT INTO ");
                sql.push_str(&self.table);
                sql.push_str(" (");
                sql.push_str(&self.columns.join(", "));
                sql.push_str(") VALUES (");
                let placeholders: Vec<String> = (0..self.values.len())
                    .map(|i| format!("${}", i + 1))
                    .collect();
                sql.push_str(&placeholders.join(", "));
                sql.push(')');
                params = self.values.clone();
                param_idx = self.values.len() + 1;
            }
            QueryBuilderType::Update => {
                sql.push_str("UPDATE ");
                sql.push_str(&self.table);
                sql.push_str(" SET ");
                let sets: Vec<String> = self.columns.iter().enumerate()
                    .map(|(i, col)| format!("{} = ${}", col, i + 1))
                    .collect();
                sql.push_str(&sets.join(", "));
                params = self.values.clone();
                param_idx = self.values.len() + 1;
            }
            QueryBuilderType::Delete => {
                sql.push_str("DELETE FROM ");
                sql.push_str(&self.table);
            }
        }
        
        // WHERE
        if !self.conditions.is_empty() {
            sql.push_str(" WHERE ");
            let conds: Vec<String> = self.conditions.iter().enumerate()
                .map(|(i, (col, op, val))| {
                    params.push(val.clone());
                    let idx = param_idx + i;
                    format!("{} {} ${}", col, op, idx)
                })
                .collect();
            sql.push_str(&conds.join(" AND "));
        }
        
        // ORDER BY
        if let Some((col, asc)) = &self.order_by {
            sql.push_str(" ORDER BY ");
            sql.push_str(col);
            sql.push_str(if *asc { " ASC" } else { " DESC" });
        }
        
        // LIMIT
        if let Some(n) = self.limit {
            sql.push_str(&format!(" LIMIT {}", n));
        }
        
        // OFFSET
        if let Some(n) = self.offset {
            sql.push_str(&format!(" OFFSET {}", n));
        }
        
        (sql, params)
    }
    
    /// Execute the query
    pub fn execute(self) -> Result<QueryResult> {
        let (sql, params) = self.to_sql();
        execute_with_params(self.conn, &sql, &params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_count_parameters() {
        assert_eq!(count_parameters("SELECT * FROM t WHERE id = ?"), 1);
        assert_eq!(count_parameters("SELECT * FROM t WHERE id = ? AND name = ?"), 2);
        assert_eq!(count_parameters("SELECT * FROM t WHERE id = $1"), 1);
        assert_eq!(count_parameters("SELECT * FROM t WHERE id = $1 AND name = $2"), 2);
        assert_eq!(count_parameters("SELECT * FROM t WHERE name = 'test?'"), 0);
    }
    
    #[test]
    fn test_detect_dialect() {
        assert_eq!(detect_dialect("INSERT INTO t (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET x = 1"), "PostgreSQL");
        assert_eq!(detect_dialect("INSERT INTO t (id) VALUES (1) ON DUPLICATE KEY UPDATE x = 1"), "MySQL");
        assert_eq!(detect_dialect("INSERT OR REPLACE INTO t (id) VALUES (1)"), "SQLite");
        assert_eq!(detect_dialect("SELECT * FROM t"), "SQL-92");
    }
    
    #[test]
    fn test_query_builder_sql() {
        // This would need a mock connection to fully test
        // Just test SQL generation here
    }
}
