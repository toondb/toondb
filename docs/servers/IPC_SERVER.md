# ToonDB IPC Server Architecture

The ToonDB IPC Server (`toondb-server`) provides a high-performance, multi-process interface to the ToonDB storage engine. It allows multiple applications to access a single embedded database instance simultaneously using Unix domain sockets.

## Architecture

The server uses a thread-per-client model optimized for low-latency local communication.

```text
┌────────────────────────────────────────────────────────────────┐
│                      IPC Server Process                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              Database Kernel (Arc<Database>)           │   │
│  └────────────────────────────────────────────────────────┘   │
│           ▲                    ▲                    ▲         │
│           │                    │                    │         │
│  ┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐ │
│  │ ClientHandler 1 │ │ ClientHandler 2 │ │ ClientHandler N │ │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘ │
│           │                    │                    │         │
│  ┌────────┴────────────────────┴────────────────────┴────────┐│
│  │              Unix Domain Socket Listener                  ││
│  │                  /tmp/toondb-<id>.sock                    ││
│  └────────────────────────────────────────────────────────────────┘
```

### Key Components

1.  **Listener Thread**: Accepts incoming Unix socket connections and spawns handler threads.
2.  **Client Handlers**: Dedicated thread per client that maintains transaction state and cursor isolation.
3.  **Shared Kernel**: Thread-safe `Arc<Database>` instance shared across all handlers.
4.  **Transaction Map**: Each handler maintains its own `active_txns` map (`client_txn_id` → `TxnHandle`), ensuring process isolation.

## Wire Protocol

The core communication uses a lightweight binary protocol designed for minimal overhead.

**Frame Format:**
```
┌─────────────────┬─────────────────────┬──────────────────────────┐
│ OpCode (1 byte) │ Length (4 bytes LE) │      Payload (N bytes)   │
└─────────────────┴─────────────────────┴──────────────────────────┘
```

### Protocol Operations (OpCodes)

| OpCode | Hex  | Name         | Type | Description |
|--------|------|--------------|------|-------------|
| 1      | 0x01 | PUT          | Cmd  | Auto-commit key-value write |
| 2      | 0x02 | GET          | Cmd  | Read value |
| 3      | 0x03 | DELETE       | Cmd  | Delete key |
| 4      | 0x04 | BEGIN_TXN    | Txn  | Start a new explicit transaction |
| 5      | 0x05 | COMMIT_TXN   | Txn  | Commit an active transaction |
| 6      | 0x06 | ABORT_TXN    | Txn  | Rollback/Abort transaction |
| 7      | 0x07 | QUERY        | Data | Execute SQL-like query (returns TOON) |
| 8      | 0x08 | CREATE_TABLE | DDL  | Define table schema |
| 9      | 0x09 | PUT_PATH     | Path | Write to hierarchical path |
| 10     | 0x0A | GET_PATH     | Path | Read from hierarchical path |
| 11     | 0x0B | SCAN         | Data | Range scan (prefix) |
| 12     | 0x0C | CHECKPOINT   | Sys  | Force durability flush to disk |
| 13     | 0x0D | STATS        | Sys  | detailed server Runtime metrics |
| 14     | 0x0E | PING         | Sys  | Health check |

### Response Codes

| Hex  | Name       | Description |
|------|------------|-------------|
| 0x80 | OK         | Operation succeeded (no data) |
| 0x81 | ERROR      | Operation failed (payload = error msg) |
| 0x82 | VALUE      | Data return |
| 0x83 | TXN_ID     | Transaction ID return |
| 0x86 | STATS_RESP | JSON Statistics |
| 0x87 | PONG       | Health check response |

## Server Statistics

The server maintains atomic counters for real-time monitoring. These are accessible via the `STATS` opcode.

**Returned JSON Structure:**
```json
{
  "connections_total": 150,
  "connections_active": 12,
  "requests_total": 45000,
  "requests_success": 44995,
  "requests_error": 5,
  "bytes_received": 1024000,
  "bytes_sent": 2048000,
  "uptime_secs": 3600,
  "active_transactions": 3
}
```

## Developer Guide: Implementing a Client

If you are building a client driver for a new language (e.g., Go, Ruby, Node.js), follow these steps:

1.  **Connect**: Open a Unix domain socket connection to the path (default: `./toondb_data/toondb.sock`).
2.  **Handshake**: (Currently none, connection implies readiness).
3.  **Sending Requests**:
    *   Write 1 byte OpCode.
    *   Write 4 bytes Little-Endian Length.
    *   Write `Length` bytes of Payload.
4.  **Reading Responses**:
    *   Read 1 byte OpCode.
    *   Read 4 bytes Little-Endian Length.
    *   Read `Length` bytes of Payload.
5.  **Clean Up**: Close the socket when done. The server will automatically abort any open transactions.

**Example: Raw Python Client Implementation**

```python
import socket
import struct

def simple_get(key_bytes):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect("./toondb_data/toondb.sock")
    
    # Send GET (0x02)
    payload = key_bytes
    header = struct.pack("<BI", 0x02, len(payload))
    s.sendall(header + payload)
    
    # Read Header
    resp_header = s.recv(5)
    code, length = struct.unpack("<BI", resp_header)
    
    # Read Body
    data = s.recv(length)
    
    if code == 0x82: # VALUE
        return data
    elif code == 0x81: # ERROR
        raise Exception(data.decode())
        
    s.close()
```
