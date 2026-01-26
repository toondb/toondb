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

//! SQL-specific error types

use thiserror::Error;

/// SQL execution errors
#[derive(Error, Debug, Clone)]
pub enum SqlError {
    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Lexer error: {0}")]
    LexError(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Transaction error: {0}")]
    TransactionError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl SqlError {
    pub fn from_parse_errors(errors: Vec<super::parser::ParseError>) -> Self {
        if let Some(first) = errors.first() {
            SqlError::ParseError {
                message: first.message.clone(),
                line: first.span.line,
                column: first.span.column,
            }
        } else {
            SqlError::ParseError {
                message: "Unknown parse error".to_string(),
                line: 0,
                column: 0,
            }
        }
    }
}

pub type SqlResult<T> = std::result::Result<T, SqlError>;
