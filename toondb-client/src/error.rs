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

//! Error types for ToonDB Client SDK

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
