# sochdb-grpc

[![Crates.io](https://img.shields.io/crates/v/sochdb-grpc.svg)](https://crates.io/crates/sochdb-grpc)
[![Documentation](https://docs.rs/sochdb-grpc/badge.svg)](https://docs.rs/sochdb-grpc)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**SochDB gRPC Server** - Production-ready gRPC services for cross-language database access. Implements the Thick Server / Thin Client architecture where business logic resides in the server, and clients connect via gRPC or IPC.

## Features

- **10 Modular Services**: KV, Graph, Collection, Namespace, Context, Policy, Trace, Checkpoint, SemanticCache, MCP
- **Temporal Graph Support**: Time-bounded edges with time-travel queries (POINT_IN_TIME, RANGE, CURRENT)
- **Format Utilities**: LLM context optimization (TOON format: 40-66% fewer tokens than JSON)
- **Atomic Operations**: Batch writes with crash-safety guarantees
- **Zero Business Logic Duplication**: Single source of truth in Rust
- **Multi-Protocol**: Supports both gRPC (TCP) and Unix domain sockets (IPC)

## Installation

```bash
cargo install sochdb-grpc
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
sochdb-grpc = "0.3.4"
```

## Usage

### Running the Server

```bash
# Start gRPC server on default port (50051)
sochdb-grpc-server

# Custom port
sochdb-grpc-server --port 8080

# Unix domain socket
sochdb-grpc-server --socket /tmp/sochdb.sock
```

### Environment Variables

```bash
SOCHDB_LOG=debug       # Enable debug logging
SOCHDB_PORT=50051      # Server port
SOCHDB_HOST=0.0.0.0    # Bind address
```

## Services

### KvService

Key-value operations with atomic batch writes:

```protobuf
rpc Get(GetRequest) returns (GetResponse);
rpc Put(PutRequest) returns (PutResponse);
rpc BatchPut(BatchPutRequest) returns (BatchPutResponse);
rpc Delete(DeleteRequest) returns (DeleteResponse);
rpc Scan(ScanRequest) returns (ScanResponse);
```

### GraphService

Temporal graph operations with time-travel queries:

```protobuf
rpc AddEdge(AddEdgeRequest) returns (AddEdgeResponse);
rpc AddTemporalEdge(AddTemporalEdgeRequest) returns (AddTemporalEdgeResponse);
rpc QueryGraph(QueryGraphRequest) returns (QueryGraphResponse);
rpc QueryTemporalGraph(QueryTemporalGraphRequest) returns (QueryTemporalGraphResponse);
```

**Temporal Query Example:**

```rust
use sochdb_grpc::graph_server::GraphService;

// Query: "Was door_1 open 30 minutes ago?"
let request = QueryTemporalGraphRequest {
    namespace: "agent_memory".to_string(),
    edge_type: Some("state".to_string()),
    query_mode: TemporalQueryMode::PointInTime as i32,
    timestamp: (now - 30 * 60 * 1000), // 30 mins ago
};

let response = graph_service.query_temporal_graph(request).await?;
```

### CollectionService

Vector search and embedding operations:

```protobuf
rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
rpc Insert(InsertRequest) returns (InsertResponse);
rpc Search(SearchRequest) returns (SearchResponse);
rpc Delete(DeleteRequest) returns (DeleteResponse);
```

### NamespaceService

Namespace isolation for multi-tenancy:

```protobuf
rpc CreateNamespace(CreateNamespaceRequest) returns (CreateNamespaceResponse);
rpc ListNamespaces(ListNamespacesRequest) returns (ListNamespacesResponse);
rpc DeleteNamespace(DeleteNamespaceRequest) returns (DeleteNamespaceResponse);
```

### ContextService

LLM context formatting and optimization:

```protobuf
rpc FormatContext(FormatContextRequest) returns (FormatContextResponse);
rpc ConvertWireFormat(ConvertWireFormatRequest) returns (ConvertWireFormatResponse);
```

**Format Benefits:**
- **TOON Format**: 40-66% fewer tokens than JSON → Lower LLM API costs
- **Columnar Format**: Efficient for tabular data
- **Round-trip Safe**: Lossless conversions

### PolicyService, TraceService, CheckpointService, SemanticCacheService

Advanced features for production deployments:
- Access control policies
- Query tracing and debugging
- Database checkpointing
- Semantic caching for LLM queries

### McpService

Model Context Protocol integration for agent workflows.

## Architecture

**Thick Server / Thin Client:**
- ✅ Business logic in Rust (single source of truth)
- ✅ Zero code duplication across SDKs
- ✅ O(1) maintenance model
- ✅ Multi-language support (Python, Node.js, Go)

**Dual-Mode Access:**
- **Server Mode (gRPC)**: Production deployments with centralized logic
- **Embedded Mode (FFI)**: Local development and edge deployments

## Protocol Buffers

Proto definitions available at: [proto/sochdb.proto](https://github.com/sochdb/sochdb/blob/main/proto/sochdb.proto)

Generate clients:

```bash
# Python
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/sochdb.proto

# Node.js
npm install @grpc/grpc-js @grpc/proto-loader
grpc_tools_node_protoc --js_out=import_style=commonjs,binary:. --grpc_out=grpc_js:. proto/sochdb.proto

# Go
protoc --go_out=. --go-grpc_out=. proto/sochdb.proto
```

## Performance

- **Request Throughput**: ~50K ops/sec (batch_put)
- **Latency**: <1ms P50, <5ms P99 (local)
- **Memory**: ~10MB baseline + ~100 bytes per connection
- **Connections**: Supports 10K+ concurrent clients

## Development

```bash
# Build
cd sochdb-grpc
cargo build --release

# Run with debug logging
RUST_LOG=debug cargo run -- --port 50051

# Test
cargo test
```

## Contributing

See [CONTRIBUTING.md](https://github.com/sochdb/sochdb/blob/main/CONTRIBUTING.md)

## License

Apache-2.0

## Links

- **Homepage**: https://sochdb.dev
- **Documentation**: https://sochdb.dev
- **Repository**: https://github.com/sochdb/sochdb
- **SDKs**:
  - [Python SDK](https://pypi.org/project/sochdb/)
  - [Node.js SDK](https://www.npmjs.com/package/sochdb)
  - [Go SDK](https://github.com/sochdb/sochdb-go)

## Version

Current: **0.3.4**

**What's New in 0.3.4:**
- ✨ Temporal graph support with time-travel queries
- ✨ Format utilities for LLM context optimization (40-66% token savings)
- ✨ Enhanced gRPC service architecture (10 modular services)
- ✨ Atomic batch operations with crash-safety
- 🔧 Zero business logic duplication across SDKs
- 📚 Comprehensive SDK documentation (Python, Node.js, Go)
