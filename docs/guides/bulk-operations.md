# ToonDB Bulk Operations

High-performance bulk vector index operations that bypass Python FFI overhead.

> **Deep Dive:** See [Bulk Operations Reference](/servers/BULK_OPERATIONS.md) for tool internals and advanced usage.

## Why Use Bulk Operations?

Python FFI has inherent overhead for vector operations:

| Method | 768D Throughput | Overhead |
|--------|-----------------|----------|
| Python FFI | ~130 vec/s | 12× slower |
| Bulk CLI | ~1,600 vec/s | 1.0× baseline |

The overhead comes from:
- O(N·d) memcpy per batch crossing the Python/Rust boundary
- Python allocation tax (reference counting, GC pressure)
- GIL contention in multi-threaded scenarios

The Bulk API eliminates this by:
1. Writing vectors to a memory-mapped file (raw f32 or npy)
2. Spawning the `toondb-bulk` CLI as a subprocess
3. Zero FFI marshalling during the actual index build

## Quick Start

### Python API

```python
from toondb.bulk import bulk_build_index
import numpy as np

# Your embeddings (10K × 768D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index (bypasses FFI)
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

### Command-Line Interface

```bash
# Build from raw f32 file
toondb-bulk build-index \
    --input embeddings.bin \
    --output index.hnsw \
    --dimension 768

# Build from NumPy .npy file
toondb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw

# With custom HNSW parameters
toondb-bulk build-index \
    --input data.f32 \
    --output index.hnsw \
    --dimension 768 \
    --max-connections 32 \
    --ef-construction 200 \
    --threads 8

# Query an index
toondb-bulk query \
    --index index.hnsw \
    --query query.f32 \
    --k 10

# Get index info
toondb-bulk info --index index.hnsw
```

## Input Formats

### Raw float32 (Recommended)

The simplest and fastest format - just raw bytes.

**File layout:**
- `vectors.f32` - N × D × 4 bytes of row-major float32 data
- `vectors.json` (optional) - Metadata: `{"n": 10000, "dim": 768, "metric": "cosine"}`
- `ids.u64` (optional) - N × 8 bytes of uint64 IDs

**Creating raw f32 from Python:**
```python
from toondb.bulk import convert_embeddings_to_raw
import numpy as np

embeddings = np.load("embeddings.npy")
convert_embeddings_to_raw(embeddings, "embeddings.f32")
```

### NumPy .npy

Standard NumPy format. Auto-detected from extension.

Requirements:
- dtype: float32 (`<f4`)
- order: C-order (fortran_order: False)
- shape: 2D (N, D)

**Creating from Python:**
```python
import numpy as np
embeddings = np.random.randn(10000, 768).astype(np.float32)
np.save("embeddings.npy", embeddings)
```

## Python API Reference

### `bulk_build_index()`

```python
def bulk_build_index(
    embeddings: NDArray[np.float32],
    output: str | Path,
    *,
    ids: NDArray[np.uint64] | None = None,
    m: int = 16,
    ef_construction: int = 100,
    batch_size: int = 1000,
    threads: int = 0,
    quiet: bool = False,
    cleanup_temp: bool = True,
) -> BulkBuildStats:
```

**Parameters:**
- `embeddings` - 2D float32 array of shape (N, D)
- `output` - Path to save the HNSW index
- `ids` - Optional uint64 array of IDs (defaults to sequential)
- `m` - HNSW max connections per node
- `ef_construction` - HNSW construction search depth
- `batch_size` - Vectors per insertion batch
- `threads` - Number of threads (0 = auto)
- `quiet` - Suppress progress output
- `cleanup_temp` - Remove temporary files after build

**Returns:** `BulkBuildStats` with performance metrics

### `BulkBuildStats`

```python
@dataclass
class BulkBuildStats:
    vectors: int          # Number of vectors inserted
    dimension: int        # Vector dimension
    elapsed_secs: float   # Total build time
    rate: float           # Vectors per second
    output_size_mb: float # Output file size
    command: list[str]    # CLI command used
```

### `convert_embeddings_to_raw()`

```python
def convert_embeddings_to_raw(
    embeddings: NDArray[np.float32],
    output: str | Path,
    *,
    metric: str | None = None,
) -> Path:
```

Convert embeddings to ToonDB's raw f32 format for optimal bulk loading.

### `read_raw_embeddings()`

```python
def read_raw_embeddings(
    path: str | Path,
    dimension: int | None = None,
) -> NDArray[np.float32]:
```

Read embeddings from raw f32 format using memory mapping.

## CLI Reference

### `build-index`

```
toondb-bulk build-index [OPTIONS] --input <FILE> --output <FILE>

Options:
  -i, --input <FILE>         Input vector file (raw f32 or .npy)
  -o, --output <FILE>        Output index file
  -d, --dimension <DIM>      Vector dimension (auto-detected for .npy)
  -f, --format <FORMAT>      Input format: raw_f32, npy (auto-detected)
      --ids <FILE>           Optional ID file (raw u64)
  -m, --max-connections <N>  HNSW M parameter [default: 16]
  -e, --ef-construction <N>  HNSW ef_construction [default: 100]
      --batch-size <N>       Batch size for insertion [default: 1000]
  -t, --threads <N>          Number of threads (0 = auto) [default: 0]
      --quiet                Suppress progress bar
  -v, --verbose              Enable verbose logging
```

### `query`

```
toondb-bulk query [OPTIONS] --index <FILE> --query <FILE>

Options:
  -i, --index <FILE>   Index file
  -q, --query <FILE>   Query vector file (single vector, raw f32)
  -k, --k <N>          Number of neighbors [default: 10]
  -e, --ef <N>         Search ef parameter
```

### `info`

```
toondb-bulk info --index <FILE>
```

### `convert`

```
toondb-bulk convert [OPTIONS] --input <FILE> --output <FILE> --to-format <FMT>

Options:
  -i, --input <FILE>       Input file
  -o, --output <FILE>      Output file
      --from-format <FMT>  Input format (auto-detected)
      --to-format <FMT>    Output format: raw_f32
  -d, --dimension <DIM>    Dimension (required for some formats)
```

## Building from Source

```bash
# Build release binary
cargo build --release -p toondb-tools

# Binary location
./target/release/toondb-bulk --help

# Run benchmarks
cargo bench -p toondb-tools

# Install to PATH
cargo install --path toondb-tools
```

## Bundling with Python Package

The Python package can bundle the native binary:

```bash
cd toondb-python-sdk

# Build and install binary for current platform
python build_native.py

# Then build wheel
pip wheel .
```

The binary is installed to `src/toondb/_bin/<platform>/toondb-bulk`.

## Performance Tips

1. **Use raw f32 format** - Fastest to parse, memory-mappable
2. **Batch size ~1000** - Optimal for HNSW insertion
3. **Use all CPU cores** - Set `threads=0` for auto-detection
4. **Pre-normalize vectors** - If using cosine similarity
5. **SSD storage** - For large indices, use NVMe storage

## Benchmarks

Run the performance benchmark:

```bash
# Python benchmark
python benchmarks/bulk_benchmark.py --size medium

# Rust microbenchmarks
cargo bench -p toondb-tools --bench bulk_ingest
```

Expected results (Apple M1 Pro, 768D vectors):

| Test | Throughput |
|------|------------|
| bulk_build_10K | ~1,600 vec/s |
| bulk_build_100K | ~1,400 vec/s |
| ffi_insert_10K | ~130 vec/s |

## Troubleshooting

### "Could not find toondb-bulk binary"

The Python Bulk API requires the `toondb-bulk` binary. The SDK automatically
searches in this order:

1. **Bundled in wheel** (recommended):
   ```bash
   pip install toondb-client
   # Binary is at: site-packages/toondb/_bin/<platform>/toondb-bulk
   ```

2. **System PATH**:
   ```bash
   cargo install --path toondb-tools
   # Or: export PATH="$PATH:/path/to/target/release"
   ```

3. **Cargo target directory** (development):
   ```bash
   cargo build --release -p toondb-tools
   # Auto-detected if running from workspace
   ```

To debug resolution:
```python
from toondb.bulk import get_toondb_bulk_path
print(get_toondb_bulk_path())  # Shows resolved path
```

### Platform Support

The bundled binary supports:

| Platform | Wheel Tag | Notes |
|----------|-----------|-------|
| Linux x86_64 | `manylinux_2_17_x86_64` | glibc ≥ 2.17 |
| Linux aarch64 | `manylinux_2_17_aarch64` | ARM servers |
| macOS | `macosx_11_0_universal2` | Intel + Apple Silicon |
| Windows | `win_amd64` | Windows 10+ x64 |

### "Dimension mismatch"

Ensure your dimension parameter matches the data:
- For raw f32: `-d 768` or provide `meta.json`
- For npy: Dimension auto-detected from header

### "Out of memory"

For large datasets (10M+ vectors):
- Use streaming ingestion with smaller batch sizes
- Consider PQ compression before indexing
- Use 64-bit system with sufficient RAM

### "GLIBC_2.xx not found" (Linux)

Your system glibc is older than the wheel requires:
```bash
ldd --version  # Check glibc version
# Needs: 2.17 or higher
```

Solutions:
1. Use a newer distro (Ubuntu 14.04+, CentOS 7+)
2. Use a container with newer glibc
3. Build from source with your system's glibc

## Architecture

```
Python Application
       │
       ▼
┌─────────────────────┐
│  toondb.bulk.py     │  Python Bulk API
│  - Write vectors    │
│  - Spawn subprocess │
└─────────┬───────────┘
          │ subprocess.run()
          ▼
┌─────────────────────┐
│  toondb-bulk CLI    │  Rust Binary (bundled in wheel)
│  - mmap vector file │
│  - HNSW insertion   │
│  - Save index       │
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  my_index.hnsw      │  Output Index
└─────────────────────┘
```

The key insight is that subprocess overhead (process spawn) is O(1),
while FFI overhead is O(N·d) per batch. For bulk operations, the
subprocess approach wins decisively.

See [Python SDK Guide](/guides/python-sdk) for full wheel
packaging and distribution architecture.

