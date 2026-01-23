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

//! SOCH-QL Query Executor (Deprecated)
//!
//! This legacy executor is now a thin wrapper over the unified AST-based
//! entry point in `sql_entry`. The old string-heuristic parsing has been
//! removed to avoid duplicate logic and inconsistencies.

use crate::connection::SochConnection;
use crate::error::Result;
use crate::sql_entry;
use sochdb_core::soch::SochValue;
use std::collections::HashMap;

/// Query execution result (re-exported from AST pipeline)
pub use crate::ast_query::QueryResult;

/// SOCH-QL query executor
pub struct QueryExecutor<'a> {
    conn: &'a SochConnection,
}

impl<'a> QueryExecutor<'a> {
    /// Create new query executor
    pub fn new(conn: &'a SochConnection) -> Self {
        Self { conn }
    }

    /// Execute a SOCH-QL query via the unified AST-based pipeline.
    pub fn execute(&self, sql: &str) -> Result<QueryResult> {
        sql_entry::execute(self.conn, sql)
    }
}

/// SQL query methods on connection (compatibility wrappers)
impl SochConnection {
    /// Execute SOCH-QL query (AST-based)
    pub fn query_sql(&self, sql: &str) -> Result<QueryResult> {
        self.query_ast(sql)
    }

    /// Execute and get rows (for SELECT)
    pub fn query_rows(&self, sql: &str) -> Result<Vec<HashMap<String, SochValue>>> {
        self.query_rows_ast(sql)
    }

    /// Execute non-query SQL
    pub fn execute_sql(&self, sql: &str) -> Result<u64> {
        self.execute_ast(sql)
    }
}
