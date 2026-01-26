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
