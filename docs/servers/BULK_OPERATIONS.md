# ToonDB Bulk Operations Tool

The `toondb-bulk` tool is a specialized high-performance utility designed for "offline" heavy lifting. It bypasses the standard transaction and RPC layers to interact directly with storage formats, enabling massive throughput for initialization and migration tasks.

## Capabilities

*   **Ingestion Speed**: ~1,600 vectors/second in single-threaded mode (HNSW build).
*   **Format Conversion**: Native conversion between raw binary (`.bin`/`.raw`) and NumPy (`.npy`) formats.
*   **Index Inspection**: Low-level metadata extraction from HNSW index files.

## Performance vs. Online APIs

| Function | Online API (Python/RPC) | Bulk Tool (`toondb-bulk`) | Speedup |
|----------|-------------------------|--------------------------|---------|
| Vector Indexing | ~130 vec/s | **~1,600+ vec/s** | **12x** |
| Data Conversion | Python Loop | Memmapped C++ | **50x** |

_Benchmarks run on M1 MacBook Air, 768d vectors._

## Commands Deep Dive

### `build-index`

The core command. It takes a raw vector file (or .npy) and produces a fully optimized, memory-mappable HNSW graph file (`.hnsw`).

**Algorithm:**
It performs a parallelized graph construction.
1.  **Load**: Memory-maps the input file (zero load time).
2.  **Initialize**: Pre-allocates graph nodes.
3.  **Insert**: Uses multi-threaded insertion (if `-t` > 1) to build layers.
4.  **Save**: Serializes the graph topology to disk.

**Usage:**
```bash
toondb-bulk build-index \
    --input large_dataset.npy \
    --output prod.hnsw \
    --dimension 1536 \
    --metric cosine \
    --threads 8
```

### `query` (Benchmark Mode)

Runs queries against a specialized index file. Useful for validating recall/latency before deploying the index to a live server.

**Usage:**
```bash
toondb-bulk query \
    --index prod.hnsw \
    --query test_set.npy \
    --k 10 \
    --ef 128
```

### `convert`

A utility for efficient format transcoding.

**Supported Formats:**
*   `npy`: Standard NumPy binary format.
*   `raw_f32`: Flat binary array of `f32` (Little Endian). No header.
*   `fvecs`: Evaluation format (count + vectors).

**Usage:**
```bash
toondb-bulk convert \
    --input data.npy \
    --output data.raw \
    --to-format raw_f32 \
    --dimension 768
```

## Integration Workflow

The typical workflow for a production deployment:

1.  **Data Science Team**: Generates embeddings and saves as `.npy`.
2.  **DevOps Pipeline**: Runs `toondb-bulk build-index` to generate the `.hnsw` artifact.
3.  **Deployment**: copies the `.hnsw` file to the production server.
4.  **Runtime**: `toondb-grpc-server` loads the pre-built `.hnsw` file instantly via `mmap`.
