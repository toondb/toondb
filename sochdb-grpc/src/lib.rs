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

//! SochDB gRPC Services
//!
//! This crate provides a comprehensive gRPC interface for SochDB operations.
//! It implements a "Thick Server / Thin Client" architecture where all business
//! logic lives in the Rust server, enabling thin SDK wrappers in any language.
//!
//! ## Services
//!
//! - **VectorIndexService**: HNSW vector operations
//! - **GraphService**: Graph overlay for agent memory
//! - **PolicyService**: Policy evaluation and enforcement
//! - **ContextService**: LLM context assembly with token budgets
//! - **CollectionService**: Collection management
//! - **NamespaceService**: Multi-tenant namespace management
//! - **SemanticCacheService**: Semantic caching for LLM queries
//! - **TraceService**: Trace/span management
//! - **CheckpointService**: State checkpoint and restore
//! - **McpService**: MCP tool routing
//! - **KvService**: Basic key-value operations
//!
//! ## Usage
//!
//! ```bash
//! # Start the gRPC server
//! sochdb-grpc-server --port 50051
//!
//! # From Python client
//! import grpc
//! from sochdb.proto import sochdb_pb2, sochdb_pb2_grpc
//!
//! channel = grpc.insecure_channel('localhost:50051')
//! stub = sochdb_pb2_grpc.VectorIndexServiceStub(channel)
//! ```

pub mod proto {
    // Include generated protobuf code
    tonic::include_proto!("sochdb.v1");
}

pub mod server;
pub mod error;

// Service implementations
pub mod graph_server;
pub mod policy_server;
pub mod context_server;
pub mod collection_server;
pub mod namespace_server;
pub mod semantic_cache_server;
pub mod trace_server;
pub mod checkpoint_server;
pub mod mcp_server;
pub mod kv_server;

pub use server::VectorIndexServer;
pub use error::GrpcError;
