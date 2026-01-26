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

//! SochDB gRPC Server
//!
//! Starts a comprehensive gRPC server with all SochDB services.
//! This implements the "Thick Server / Thin Client" architecture where
//! all business logic lives in Rust, enabling thin SDK wrappers.
//!
//! ## Services
//!
//! - VectorIndexService: HNSW vector operations
//! - GraphService: Graph overlay for agent memory
//! - PolicyService: Policy evaluation
//! - ContextService: LLM context assembly
//! - CollectionService: Collection management
//! - NamespaceService: Multi-tenant namespaces
//! - SemanticCacheService: Semantic caching
//! - TraceService: Distributed tracing
//! - CheckpointService: State snapshots
//! - McpService: MCP tool routing
//! - KvService: Key-value operations
//!
//! ## Usage
//!
//! ```bash
//! # Start on default port 50051
//! sochdb-grpc-server
//!
//! # Start on custom port
//! sochdb-grpc-server --port 8080
//!
//! # Bind to specific address
//! sochdb-grpc-server --host 0.0.0.0 --port 50051
//! ```

use clap::Parser;
use tonic::transport::Server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use sochdb_grpc::{
    VectorIndexServer,
    graph_server::GraphServer,
    policy_server::PolicyServer,
    context_server::ContextServer,
    collection_server::CollectionServer,
    namespace_server::NamespaceServer,
    semantic_cache_server::SemanticCacheServer,
    trace_server::TraceServer,
    checkpoint_server::CheckpointServer,
    mcp_server::McpServer,
    kv_server::KvServer,
};

/// SochDB gRPC Server
#[derive(Parser, Debug)]
#[command(name = "sochdb-grpc-server")]
#[command(about = "SochDB gRPC server - Thick Server / Thin Client architecture")]
#[command(version)]
struct Args {
    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    
    /// Port to listen on
    #[arg(short, long, default_value = "50051")]
    port: u16,
    
    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize tracing
    let filter = if args.debug {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    let addr = format!("{}:{}", args.host, args.port).parse()?;
    
    // Create all service instances
    let vector_server = VectorIndexServer::new();
    let graph_server = GraphServer::new();
    let policy_server = PolicyServer::new();
    let context_server = ContextServer::new();
    let collection_server = CollectionServer::new();
    let namespace_server = NamespaceServer::new();
    let semantic_cache_server = SemanticCacheServer::new();
    let trace_server = TraceServer::new();
    let checkpoint_server = CheckpointServer::new();
    let mcp_server = McpServer::new();
    let kv_server = KvServer::new();
    
    tracing::info!("Starting SochDB gRPC server on {}", addr);
    tracing::info!("Server version: {}", env!("CARGO_PKG_VERSION"));
    
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║            SochDB gRPC Server (Thick Server)                 ║
╠══════════════════════════════════════════════════════════════╣
║  Server:     {}                                   
║  Version:    {}                                            
║                                                              ║
║  Services:                                                   ║
║    - VectorIndexService    Vector index operations           ║
║    - GraphService          Graph overlay                     ║
║    - PolicyService         Policy evaluation                 ║
║    - ContextService        LLM context assembly              ║
║    - CollectionService     Collection management             ║
║    - NamespaceService      Multi-tenant namespaces           ║
║    - SemanticCacheService  Semantic caching                  ║
║    - TraceService          Distributed tracing               ║
║    - CheckpointService     State snapshots                   ║
║    - McpService            MCP tool routing                  ║
║    - KvService             Key-value operations              ║
╚══════════════════════════════════════════════════════════════╝
"#,
        addr,
        env!("CARGO_PKG_VERSION")
    );
    
    Server::builder()
        .add_service(vector_server.into_service())
        .add_service(graph_server.into_service())
        .add_service(policy_server.into_service())
        .add_service(context_server.into_service())
        .add_service(collection_server.into_service())
        .add_service(namespace_server.into_service())
        .add_service(semantic_cache_server.into_service())
        .add_service(trace_server.into_service())
        .add_service(checkpoint_server.into_service())
        .add_service(mcp_server.into_service())
        .add_service(kv_server.into_service())
        .serve(addr)
        .await?;
    
    Ok(())
}
