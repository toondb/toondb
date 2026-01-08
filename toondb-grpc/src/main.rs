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

//! ToonDB gRPC Server
//!
//! Starts a comprehensive gRPC server with all ToonDB services.
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
//! toondb-grpc-server
//!
//! # Start on custom port
//! toondb-grpc-server --port 8080
//!
//! # Bind to specific address
//! toondb-grpc-server --host 0.0.0.0 --port 50051
//! ```

use clap::Parser;
use tonic::transport::Server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use toondb_grpc::{
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

/// ToonDB gRPC Server
#[derive(Parser, Debug)]
#[command(name = "toondb-grpc-server")]
#[command(about = "ToonDB gRPC server - Thick Server / Thin Client architecture")]
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
    
    tracing::info!("Starting ToonDB gRPC server on {}", addr);
    tracing::info!("Server version: {}", env!("CARGO_PKG_VERSION"));
    
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║            ToonDB gRPC Server (Thick Server)                 ║
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
