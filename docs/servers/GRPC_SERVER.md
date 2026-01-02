# ToonDB gRPC Server Architecture

The ToonDB gRPC Server (`toondb-grpc-server`) provides a dedicated, high-performance interface for **Vector Search** operations. While the IPC server handles general data ops, this server is optimized for high-throughput embedding ingestion and k-NN path queries.

## Architecture

The server is built on `tonic` (Rust gRPC) and wraps the `toondb-index` crate's HNSW implementation.

### Capabilities

*   **Managed Indexes**: Supports creating and serving multiple vector indexes simultaneously.
*   **Streaming Inserts**: High-throughput streaming API for ingesting millions of vectors.
*   **Zero-Copy Batching**: Specialized batch APIs (`InsertBatch`) that map directly to internal memory layouts.
*   **Concurrent Search**: Thread-safe search operations allowing parallel queries.

## HNSW Configuration

When creating an index, you can tune the HNSW (Hierarchical Navigable Small World) parameters for your specific recall vs. performance trade-offs.

| Parameter | Default | Impact |
|-----------|---------|--------|
| `dimension` | Required | Vector size (e.g., 1536 for OpenAI, 768 for BERT). |
| `max_connections` (M) | 16 | Max edges per node. Higher = better recall, slower build/search. |
| `ef_construction` | 200 | Size of dynamic candidate list during build. Higher = better index quality, slower build. |
| `ef_search` | 50 | Size of candidate list during search. Higher = better recall, slower query. |
| `metric` | Cosine | Distance metric: `L2`, `Cosine`, `DotProduct`. |

## Service API Reference

The server exposes the `VectorIndexService`.

### 1. Index Management

**`CreateIndex`**
Initializes a new HNSW graph.
```protobuf
message CreateIndexRequest {
  string name = 1;
  uint32 dimension = 2;
  DistanceMetric metric = 3;
  HnswConfig config = 4;
}
```

**`DropIndex`**
Removes an index from memory.

### 2. Data Ingestion

**`InsertBatch`** (Recommended)
Atomic batch insertion. The most efficient way to load data. It uses a "flat" layout for zero-copy deserialization in Rust.
```protobuf
message InsertBatchRequest {
  string index_name = 1;
  repeated uint64 ids = 2;
  repeated float vectors = 3; // Flat array: [v1_0, v1_1... v2_0...]
}
```

**`InsertStream`**
Long-running bidirectional stream for continuous ingestion.
```protobuf
message InsertStreamRequest {
  string index_name = 1; // Only needed in first message
  uint64 id = 2;
  repeated float vector = 3;
}
```

### 3. Search

**`Search`**
Standard k-Nearest Neighbors search.
```protobuf
message SearchRequest {
  string index_name = 1;
  repeated float query = 2;
  uint32 k = 3;
}
```

**`SearchBatch`**
Execute multiple queries in parallel.

### 4. Operations

**`GetStats`**
Returns internal graph statistics useful for debugging.
*   `num_vectors`: Total count.
*   `max_layer`: Height of the HNSW graph.
*   `avg_connections`: Graph connectivity density.

**`HealthCheck`**
Standard gRPC health check.

## Client Usage Example (Python)

```python
import grpc
from toondb_pb2 import CreateIndexRequest, SearchRequest, HnswConfig
from toondb_pb2_grpc import VectorIndexServiceStub

# 1. Connect
channel = grpc.insecure_channel('localhost:50051')
stub = VectorIndexServiceStub(channel)

# 2. Create Index
stub.CreateIndex(CreateIndexRequest(
    name="prod_vectors",
    dimension=768,
    metric=1, # Cosine
    config=HnswConfig(max_connections=32, ef_construction=100)
))

# 3. Search
response = stub.Search(SearchRequest(
    index_name="prod_vectors",
    query=[0.1, 0.2, ...], # 768 floats
    k=5
))

for match in response.results:
    print(f"Doc {match.id}: Score {match.distance}")
```
