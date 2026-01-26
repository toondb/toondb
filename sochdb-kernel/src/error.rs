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

//! Kernel error types
//!
//! Minimal error hierarchy for ACID operations.

use std::fmt;
use thiserror::Error;

/// Result type for kernel operations
pub type KernelResult<T> = Result<T, KernelError>;

/// Kernel error types - minimal set for ACID operations
#[derive(Error, Debug)]
pub enum KernelError {
    /// I/O error during WAL or page operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Transaction error
    #[error("Transaction error: {kind}")]
    Transaction { kind: TransactionErrorKind },

    /// WAL error
    #[error("WAL error: {kind}")]
    Wal { kind: WalErrorKind },

    /// Page manager error
    #[error("Page error: {kind}")]
    Page { kind: PageErrorKind },

    /// Catalog error
    #[error("Catalog error: {kind}")]
    Catalog { kind: CatalogErrorKind },

    /// Plugin error
    #[error("Plugin error: {message}")]
    Plugin { message: String },

    /// Corruption detected
    #[error("Data corruption detected: {details}")]
    Corruption { details: String },

    /// Recovery error
    #[error("Recovery error: {details}")]
    Recovery { details: String },
}

/// Transaction-specific error kinds
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionErrorKind {
    /// Transaction not found
    NotFound(u64),
    /// Transaction already committed
    AlreadyCommitted,
    /// Transaction already aborted
    AlreadyAborted,
    /// Write-write conflict detected
    WriteWriteConflict { row_id: u64 },
    /// Serialization failure (SSI)
    SerializationFailure,
    /// Deadlock detected
    Deadlock,
    /// Transaction timeout
    Timeout,
}

impl fmt::Display for TransactionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "transaction {} not found", id),
            Self::AlreadyCommitted => write!(f, "transaction already committed"),
            Self::AlreadyAborted => write!(f, "transaction already aborted"),
            Self::WriteWriteConflict { row_id } => {
                write!(f, "write-write conflict on row {}", row_id)
            }
            Self::SerializationFailure => write!(f, "serialization failure"),
            Self::Deadlock => write!(f, "deadlock detected"),
            Self::Timeout => write!(f, "transaction timeout"),
        }
    }
}

/// WAL-specific error kinds
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalErrorKind {
    /// Invalid LSN
    InvalidLsn(u64),
    /// Checksum mismatch
    ChecksumMismatch { expected: u32, actual: u32 },
    /// WAL corrupted
    Corrupted,
    /// WAL full
    Full,
    /// Fsync failed
    FsyncFailed,
}

impl fmt::Display for WalErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLsn(lsn) => write!(f, "invalid LSN: {}", lsn),
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "checksum mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::Corrupted => write!(f, "WAL corrupted"),
            Self::Full => write!(f, "WAL full"),
            Self::FsyncFailed => write!(f, "fsync failed"),
        }
    }
}

/// Page-specific error kinds
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageErrorKind {
    /// Page not found
    NotFound(u64),
    /// Page corrupted
    Corrupted(u64),
    /// Buffer pool full
    BufferPoolFull,
    /// Invalid page size
    InvalidSize,
}

impl fmt::Display for PageErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "page {} not found", id),
            Self::Corrupted(id) => write!(f, "page {} corrupted", id),
            Self::BufferPoolFull => write!(f, "buffer pool full"),
            Self::InvalidSize => write!(f, "invalid page size"),
        }
    }
}

/// Catalog-specific error kinds
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogErrorKind {
    /// Table not found
    TableNotFound(String),
    /// Table already exists
    TableExists(String),
    /// Column not found
    ColumnNotFound(String),
    /// Schema mismatch
    SchemaMismatch,
}

impl fmt::Display for CatalogErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TableNotFound(name) => write!(f, "table '{}' not found", name),
            Self::TableExists(name) => write!(f, "table '{}' already exists", name),
            Self::ColumnNotFound(name) => write!(f, "column '{}' not found", name),
            Self::SchemaMismatch => write!(f, "schema mismatch"),
        }
    }
}
