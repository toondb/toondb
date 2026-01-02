# ToonDB CLI Tools

> **v0.2.9** - Production-grade Python wrappers for ToonDB command-line tools

After `pip install toondb-client`, three CLI commands are globally available:

```bash
toondb-server      # IPC server for multi-process access
toondb-bulk        # High-performance vector operations
toondb-grpc-server # gRPC server for remote vector search
```

---

## toondb-server

Multi-process database access via Unix domain sockets.

### Quick Start

```bash
# Start server
toondb-server --db ./my_database

# Check status
toondb-server status --db ./my_database

# Stop server
toondb-server stop --db ./my_database
```

### Features

| Feature | Description |
|---------|-------------|
| **Stale socket detection** | Automatically cleans up orphaned socket files |
| **Health checks** | Waits for server to be ready before returning |
| **Graceful shutdown** | Handles SIGTERM/SIGINT for clean teardown |
| **PID tracking** | Writes PID file for process management |
| **Permission checks** | Validates directory is writable before starting |
| **Actionable errors** | Clear error messages with fix suggestions |

### Options

```
Usage: toondb-server [OPTIONS] [COMMAND]

Commands:
  stop     Stop a running server
  status   Check server status

Options:
  -d, --db PATH           Database directory [default: ./toondb_data]
  -s, --socket PATH       Unix socket path [default: <db>/toondb.sock]
      --max-clients N     Maximum connections [default: 100]
      --timeout-ms MS     Connection timeout [default: 30000]
      --log-level LEVEL   trace/debug/info/warn/error [default: info]
      --no-wait           Don't wait for server to be ready
      --version           Show version and exit
```

### Examples

```bash
# Development
toondb-server --db ./dev_db --log-level debug

# Production
toondb-server \
    --db /var/lib/toondb/production \
    --socket /var/run/toondb.sock \
    --max-clients 500 \
    --timeout-ms 60000 \
    --log-level info

# Check if running
toondb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)

# Graceful stop
toondb-server stop --db ./my_database
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TOONDB_SERVER_PATH` | Override bundled binary path |

### Error Handling

```bash
# Socket already in use
$ toondb-server --db ./my_db
[Server] Error: Socket already in use (PID: 12345): ./my_db/toondb.sock
         Another toondb-server instance may be running.
         Use 'toondb-server stop --socket ./my_db/toondb.sock' to stop it.

# Permission denied
$ toondb-server --db /root/db
[Server] Error: Database directory is not writable: /root/db

# Binary not found
$ toondb-server --db ./my_db
[Server] Error: toondb-server binary not found.
Searched:
  - TOONDB_SERVER_PATH environment variable
  - Bundled in package (_bin/)
  - System PATH

To fix:
  1. Reinstall: pip install --force-reinstall toondb-client
  2. Or build: cargo build --release -p toondb-server
  3. Or set: export TOONDB_SERVER_PATH=/path/to/toondb-server
```

---

## toondb-bulk

High-performance vector index building and querying.

### Quick Start

```bash
# Build HNSW index
toondb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw \
    --dimension 768

# Query index
toondb-bulk query \
    --index index.hnsw \
    --query query.raw \
    --k 10

# Get index info
toondb-bulk info --index index.hnsw
```

### Features

| Feature | Description |
|---------|-------------|
| **Input validation** | Checks file exists, readable, correct extension |
| **Output validation** | Checks directory writable, handles overwrites |
| **Progress reporting** | Shows file sizes and operation progress |
| **Error recovery** | Actionable error messages with suggestions |

### Commands

#### build-index

Build an HNSW vector index from embeddings.

```bash
toondb-bulk build-index \
    --input vectors.npy \        # .npy or .raw format
    --output index.hnsw \        # Output index path
    --dimension 768 \            # Vector dimension
    --max-connections 16 \       # HNSW M parameter
    --ef-construction 100 \      # Build-time ef
    --threads 0 \                # 0 = auto-detect
    --batch-size 1000 \          # Vectors per batch
    --metric cosine \            # cosine/l2/ip
    --overwrite                  # Overwrite existing
```

#### query

Query an HNSW index for nearest neighbors.

```bash
toondb-bulk query \
    --index index.hnsw \     # Index file
    --query query.raw \      # Query vector (.raw or .npy)
    --k 10 \                 # Number of neighbors
    --ef 64                  # Search ef parameter
```

#### info

Display index metadata.

```bash
toondb-bulk info --index index.hnsw

# Output:
# Dimension: 768
# Vectors: 100000
# Max connections: 16
# Entry point: 42
```

#### convert

Convert between vector formats.

```bash
toondb-bulk convert \
    --input vectors.npy \
    --output vectors.raw \
    --to-format raw_f32 \
    --dimension 768
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TOONDB_BULK_PATH` | Override bundled binary path |

### Error Handling

```bash
# Input not found
$ toondb-bulk build-index --input missing.npy --output out.hnsw --dimension 768
[Bulk] Error: Input file not found: /path/to/missing.npy

# Output exists
$ toondb-bulk build-index --input data.npy --output existing.hnsw --dimension 768
[Bulk] Error: Output file already exists: /path/to/existing.hnsw
       Use --overwrite to replace it

# Invalid extension
$ toondb-bulk build-index --input data.txt --output out.hnsw --dimension 768
[Bulk] Error: Invalid file extension: .txt
       Expected one of: .npy, .raw, .bin
```

---

## toondb-grpc-server

gRPC server for remote vector search operations.

### Quick Start

```bash
# Start server
toondb-grpc-server

# Custom host and port
toondb-grpc-server --host 0.0.0.0 --port 50051

# Check status
toondb-grpc-server status --port 50051
```

### Features

| Feature | Description |
|---------|-------------|
| **Port checking** | Verifies port is available before starting |
| **Process detection** | Identifies what process is using a port |
| **Privileged port check** | Warns about ports < 1024 requiring root |
| **Health checks** | Waits for gRPC endpoint to be ready |
| **Graceful shutdown** | Handles signals for clean teardown |

### Options

```
Usage: toondb-grpc-server [OPTIONS] [COMMAND]

Commands:
  status   Check if server is running

Options:
      --host HOST    Bind address [default: 127.0.0.1]
  -p, --port PORT    Listen port [default: 50051]
  -d, --debug        Enable debug logging
      --no-wait      Don't wait for server to be ready
      --version      Show version and exit
```

### Examples

```bash
# Local development
toondb-grpc-server --debug

# Production (all interfaces)
toondb-grpc-server --host 0.0.0.0 --port 50051

# Check if running
toondb-grpc-server status --port 50051
# Output: [gRPC] Running on 127.0.0.1:50051
```

### Python Client Usage

```python
import grpc
from toondb_pb2 import VectorSearchRequest
from toondb_pb2_grpc import VectorServiceStub

# Connect
channel = grpc.insecure_channel('localhost:50051')
stub = VectorServiceStub(channel)

# Search
request = VectorSearchRequest(
    index_path="index.hnsw",
    query_vector=query_vec.tobytes(),
    k=10,
    ef_search=64
)
response = stub.Search(request)

for neighbor in response.neighbors:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance}")
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TOONDB_GRPC_SERVER_PATH` | Override bundled binary path |

### Error Handling

```bash
# Port in use
$ toondb-grpc-server --port 8080
[gRPC] Error: Port 8080 is already in use by nginx (PID: 1234)
       Try a different port with --port <PORT>

# Privileged port
$ toondb-grpc-server --port 80
[gRPC] Error: Port 80 requires root privileges
       Use a port >= 1024 or run with sudo
```

---

## Exit Codes

All CLI tools use consistent exit codes:

| Code | Name | Description |
|------|------|-------------|
| 0 | SUCCESS | Operation completed successfully |
| 1 | GENERAL_ERROR | General error |
| 2 | BINARY_NOT_FOUND | Native binary not found |
| 3 | PORT/SOCKET_IN_USE | Port or socket already in use |
| 4 | PERMISSION_DENIED | Permission denied |
| 5 | STARTUP_FAILED | Server failed to start |
| 130 | INTERRUPTED | Interrupted by SIGINT (Ctrl+C) |

---

## Comparison: Before vs After

### Before (v0.2.7)

```bash
# Had to find and use absolute paths
/path/to/venv/lib/python3.11/site-packages/toondb/_bin/macos/toondb-server --db ./my_db

# No status checking
ps aux | grep toondb-server

# Manual cleanup of stale sockets
rm ./my_db/toondb.sock

# No validation
./toondb-bulk build-index --input missing.npy  # Cryptic error
```

### After (v0.2.9)

```bash
# Simple, global commands
toondb-server --db ./my_db

# Built-in status checking
toondb-server status --db ./my_db

# Automatic stale socket cleanup
toondb-server --db ./my_db  # Cleans up stale sockets automatically

# Clear, actionable errors
toondb-bulk build-index --input missing.npy
# [Bulk] Error: Input file not found: /path/to/missing.npy
```

---

## Troubleshooting

### Binary Not Found

```bash
# Check binary resolution
python -c "from toondb.cli_server import get_server_binary; print(get_server_binary())"

# Set path manually
export TOONDB_SERVER_PATH=/path/to/toondb-server
export TOONDB_BULK_PATH=/path/to/toondb-bulk
export TOONDB_GRPC_SERVER_PATH=/path/to/toondb-grpc-server
```

### Permission Issues

```bash
# Make binary executable
chmod +x /path/to/_bin/*/toondb-*

# Check directory permissions
ls -la ./my_database
```

### Port Already in Use

```bash
# Find process using port
lsof -i :50051

# Kill process
kill <PID>
```

---

*Last updated: January 2026 (v0.2.9)*
