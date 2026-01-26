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

//! Error types for gRPC service

use thiserror::Error;
use tonic::Status;

/// Errors that can occur in the gRPC service
#[derive(Error, Debug)]
pub enum GrpcError {
    #[error("Index not found: {0}")]
    IndexNotFound(String),
    
    #[error("Index already exists: {0}")]
    IndexAlreadyExists(String),
    
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("HNSW error: {0}")]
    HnswError(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<GrpcError> for Status {
    fn from(err: GrpcError) -> Self {
        match err {
            GrpcError::IndexNotFound(name) => {
                Status::not_found(format!("Index not found: {}", name))
            }
            GrpcError::IndexAlreadyExists(name) => {
                Status::already_exists(format!("Index already exists: {}", name))
            }
            GrpcError::DimensionMismatch { expected, actual } => {
                Status::invalid_argument(format!(
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                ))
            }
            GrpcError::InvalidRequest(msg) => Status::invalid_argument(msg),
            GrpcError::HnswError(msg) => Status::internal(format!("HNSW error: {}", msg)),
            GrpcError::Internal(msg) => Status::internal(msg),
        }
    }
}
