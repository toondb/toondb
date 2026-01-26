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

//! Error types for SochDB Client SDK

use thiserror::Error;

/// Client SDK error types
#[derive(Error, Debug)]
pub enum ClientError {
    /// Path resolution failed
    #[error("Path not found: {0}")]
    PathNotFound(String),

    /// Scalar path used where array expected
    #[error("Scalar path cannot be queried: {0}")]
    ScalarPath(String),

    /// Schema-related errors
    #[error("Schema error: {0}")]
    Schema(String),

    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),

    /// Record not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Transaction errors
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Serialization failure (write skew, etc.)
    #[error("Serialization failure: txn {our_txn} conflicts with {conflicting_txn} on key")]
    SerializationFailure {
        our_txn: u64,
        conflicting_txn: u64,
        conflicting_key: Vec<u8>,
    },

    /// MVCC visibility error
    #[error("MVCC visibility error: {0}")]
    Visibility(String),

    /// WAL/durability errors
    #[error("WAL error: {0}")]
    Wal(String),

    /// Storage errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Parse errors
    #[error("Parse error: {0}")]
    Parse(String),

    /// Constraint violation (unique, foreign key, etc.)
    #[error("Constraint violation: {0}")]
    Constraint(String),

    /// Vector collection errors
    #[error("Vector error: {0}")]
    Vector(String),

    /// PQ not trained
    #[error("Product quantization codebooks not trained")]
    PqNotTrained,

    /// Type mismatch
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    /// Token budget exceeded
    #[error("Token budget exceeded: {used} > {budget}")]
    TokenBudgetExceeded { used: usize, budget: usize },

    /// Connection pool exhausted
    #[error("Connection pool exhausted")]
    PoolExhausted,

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<bincode::Error> for ClientError {
    fn from(e: bincode::Error) -> Self {
        ClientError::Serialization(e.to_string())
    }
}

/// Result type alias for client operations
pub type Result<T> = std::result::Result<T, ClientError>;
