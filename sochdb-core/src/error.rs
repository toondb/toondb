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

//! Error types for SochDB

use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SochDBError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Data corruption detected: {details}. Location: {location}. Recommendation: {hint}")]
    DataCorruption {
        details: String,
        location: String,
        hint: String,
    },

    #[error("Corruption detected: {0}")]
    Corruption(String),

    #[error("Key not found: {0}")]
    NotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Invalid timestamp in {field}: {value} is {reason}. Valid range: {min} to {max}")]
    InvalidTimestampDetailed {
        field: String,
        value: u64,
        reason: String,
        min: u64,
        max: u64,
    },

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Compaction error: {0}")]
    Compaction(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Backup error: {0}")]
    Backup(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Invalid reference: {0}")]
    InvalidReference(String),

    #[error("Circuit breaker open: {0}")]
    CircuitOpen(String),

    #[error("Write stall timeout: {0}")]
    WriteStall(String),

    #[error("Schema evolution error: {0}")]
    SchemaEvolution(String),

    #[error("Lock error: {0}")]
    LockError(String),

    #[error("Database is locked by another process")]
    DatabaseLocked,

    #[error("WAL epoch mismatch: expected {expected}, got {actual}")]
    EpochMismatch { expected: u64, actual: u64 },

    #[error("Split-brain detected in WAL: {0}")]
    SplitBrain(String),
}

pub type Result<T> = std::result::Result<T, SochDBError>;
