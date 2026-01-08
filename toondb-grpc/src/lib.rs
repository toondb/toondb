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

//! ToonDB gRPC Services
//!
//! This crate provides a comprehensive gRPC interface for ToonDB operations.
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
//! toondb-grpc-server --port 50051
//!
//! # From Python client
//! import grpc
//! from toondb.proto import toondb_pb2, toondb_pb2_grpc
//!
//! channel = grpc.insecure_channel('localhost:50051')
//! stub = toondb_pb2_grpc.VectorIndexServiceStub(channel)
//! ```

pub mod proto {
    // Include generated protobuf code
    tonic::include_proto!("toondb.v1");
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
