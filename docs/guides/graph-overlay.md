# Graph Overlay for Agent Memory

SochDB provides a lightweight graph layer on top of its KV storage for modeling
agent memory relationships. This is NOT a full graph database - it's optimized
for typical agent memory patterns.

## Overview

The Graph Overlay enables:

- **Entity relationships**: user ↔ conversation ↔ message
- **Causal chains**: action1 → action2 → action3
- **Reference graphs**: document ← citation ← quote

## Storage Model

| Data Type | Key Pattern | Value |
|-----------|-------------|-------|
| Nodes | `_graph/{ns}/nodes/{id}` | `{type, properties}` |
| Edges | `_graph/{ns}/edges/{from}/{type}/{to}` | `{properties}` |
| Reverse Index | `_graph/{ns}/index/{type}/{to}/{from}` | `from_id` |

This enables O(1) node/edge operations and O(degree) traversals.

## Quick Start

### Python

```python
from sochdb import Database
from sochdb.graph import GraphOverlay

db = Database.open("./agent_memory")
graph = GraphOverlay(db, namespace="agent_001")

# Create nodes
graph.add_node("user_1", "User", {"name": "Alice"})
graph.add_node("conv_1", "Conversation", {"title": "Planning Session"})
graph.add_node("msg_1", "Message", {"content": "Let's start planning"})

# Create edges
graph.add_edge("user_1", "STARTED", "conv_1")
graph.add_edge("conv_1", "CONTAINS", "msg_1")
graph.add_edge("user_1", "SENT", "msg_1")

# Query relationships
for edge in graph.get_edges("user_1", "STARTED"):
    print(f"User started: {edge.to_id}")

# Traverse graph
reachable = graph.bfs("user_1", max_depth=2)
# ["user_1", "conv_1", "msg_1"]
```

### Go

```go
db, _ := sochdb.Open("./agent_memory")
graph := sochdb.NewGraphOverlay(db, "agent_001")

// Create nodes
graph.AddNode("user_1", "User", map[string]interface{}{"name": "Alice"})
graph.AddNode("conv_1", "Conversation", map[string]interface{}{"title": "Planning"})
graph.AddNode("msg_1", "Message", map[string]interface{}{"content": "Let's start"})

// Create edges
graph.AddEdge("user_1", "STARTED", "conv_1", nil)
graph.AddEdge("conv_1", "CONTAINS", "msg_1", nil)
graph.AddEdge("user_1", "SENT", "msg_1", nil)

// Traverse
reachable, _ := graph.BFS("user_1", 2, nil, nil)
// ["user_1", "conv_1", "msg_1"]

// Shortest path
path, _ := graph.ShortestPath("user_1", "msg_1", 10, nil)
// ["user_1", "conv_1", "msg_1"]
```

### TypeScript/Node.js

```typescript
import { Database, GraphOverlay, EdgeDirection } from '@sochdb/sochdb';

const db = await Database.open('./agent_memory');
const graph = new GraphOverlay(db, 'agent_001');

// Create nodes
await graph.addNode('user_1', 'User', { name: 'Alice' });
await graph.addNode('conv_1', 'Conversation', { title: 'Planning' });
await graph.addNode('msg_1', 'Message', { content: 'Let\'s start' });

// Create edges
await graph.addEdge('user_1', 'STARTED', 'conv_1');
await graph.addEdge('conv_1', 'CONTAINS', 'msg_1');
await graph.addEdge('user_1', 'SENT', 'msg_1');

// Traverse
const reachable = await graph.bfs('user_1', 2);
// ['user_1', 'conv_1', 'msg_1']

// Shortest path
const path = await graph.shortestPath('user_1', 'msg_1');
// ['user_1', 'conv_1', 'msg_1']
```

### Rust

```rust
use sochdb_client::graph::{GraphOverlay, EdgeDirection};
use std::collections::HashMap;

let conn = Connection::open("./agent_memory")?;
let graph = GraphOverlay::new(conn, "agent_001");

// Create nodes
let mut props = HashMap::new();
props.insert("name".to_string(), serde_json::json!("Alice"));
graph.add_node("user_1", "User", Some(props))?;

// Create edges
graph.add_edge("user_1", "STARTED", "conv_1", None)?;

// Traverse
let reachable = graph.bfs("user_1", 2, None, None)?;

// Shortest path
let path = graph.shortest_path("user_1", "msg_1", 10, None)?;
```

## Node Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `add_node(id, type, props)` | Create or update node | O(1) |
| `get_node(id)` | Retrieve node by ID | O(1) |
| `update_node(id, props)` | Update properties | O(1) |
| `delete_node(id, cascade)` | Delete node (optionally with edges) | O(degree) |
| `node_exists(id)` | Check if node exists | O(1) |

## Edge Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `add_edge(from, type, to, props)` | Create directed edge | O(1) |
| `get_edge(from, type, to)` | Get specific edge | O(1) |
| `get_edges(from, type?)` | Get outgoing edges | O(degree) |
| `get_incoming_edges(to, type?)` | Get incoming edges | O(degree) |
| `delete_edge(from, type, to)` | Delete edge | O(1) |

## Traversal Operations

### BFS (Breadth-First Search)

```python
# Find all reachable nodes within 3 hops
nodes = graph.bfs("user_1", max_depth=3)

# Filter by edge types
nodes = graph.bfs("user_1", max_depth=3, edge_types=["SENT", "CONTAINS"])

# Filter by node types
nodes = graph.bfs("user_1", max_depth=3, node_types=["Message"])
```

### DFS (Depth-First Search)

```python
# Depth-first traversal
nodes = graph.dfs("user_1", max_depth=5)
```

### Shortest Path

```python
# Find shortest path between two nodes
path = graph.shortest_path("user_1", "msg_10", max_depth=10)
# Returns: ["user_1", "conv_1", "msg_5", "msg_10"] or None if unreachable
```

## Query Operations

### Get Neighbors

```python
# Outgoing neighbors only
neighbors = graph.get_neighbors("user_1", direction="outgoing")

# Incoming neighbors only
neighbors = graph.get_neighbors("msg_1", direction="incoming")

# Both directions
neighbors = graph.get_neighbors("conv_1", direction="both")

# Filter by edge type
neighbors = graph.get_neighbors("user_1", edge_types=["STARTED"])
```

### Get Nodes by Type

```python
# Get all User nodes (scans, use sparingly)
users = graph.get_nodes_by_type("User", limit=100)
```

### Get Subgraph

```python
# Extract a subgraph around a node
subgraph = graph.get_subgraph("user_1", max_depth=2)
print(f"Nodes: {len(subgraph.nodes)}")
print(f"Edges: {len(subgraph.edges)}")
```

## Agent Memory Patterns

### Conversation History

```python
# Model a conversation thread
graph.add_node("conv_1", "Conversation", {"title": "Support Chat"})
graph.add_node("msg_1", "Message", {"role": "user", "content": "Help!"})
graph.add_node("msg_2", "Message", {"role": "assistant", "content": "I can help"})

graph.add_edge("conv_1", "CONTAINS", "msg_1")
graph.add_edge("conv_1", "CONTAINS", "msg_2")
graph.add_edge("msg_1", "FOLLOWED_BY", "msg_2")

# Retrieve conversation in order
messages = graph.get_edges("conv_1", "CONTAINS")
```

### Tool Call Chains

```python
# Model tool execution sequences
graph.add_node("action_1", "ToolCall", {"tool": "search", "query": "docs"})
graph.add_node("action_2", "ToolCall", {"tool": "read_file", "path": "README.md"})
graph.add_node("action_3", "ToolCall", {"tool": "summarize", "input": "..."})

graph.add_edge("action_1", "CAUSED", "action_2")
graph.add_edge("action_2", "CAUSED", "action_3")

# Find causal chain
chain = graph.bfs("action_1", max_depth=10, edge_types=["CAUSED"])
```

### Knowledge References

```python
# Model document references
graph.add_node("doc_1", "Document", {"title": "API Guide"})
graph.add_node("chunk_1", "Chunk", {"text": "Authentication uses..."})
graph.add_node("chunk_2", "Chunk", {"text": "Rate limits are..."})

graph.add_edge("doc_1", "CONTAINS", "chunk_1")
graph.add_edge("doc_1", "CONTAINS", "chunk_2")
graph.add_edge("chunk_2", "REFERENCES", "chunk_1")

# Find all chunks in document
chunks = graph.get_edges("doc_1", "CONTAINS")
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Add/Get Node | O(1) | Direct KV lookup |
| Add/Get Edge | O(1) | Direct KV lookup |
| Outgoing Edges | O(degree) | Prefix scan |
| Incoming Edges | O(degree) | Reverse index lookup |
| BFS/DFS | O(V + E) | For reachable subgraph |
| Shortest Path | O(V + E) | BFS-based |

## Best Practices

1. **Use meaningful edge types**: `SENT`, `CONTAINS`, `REFERENCES` are clearer than generic `RELATES_TO`
2. **Namespace by agent**: Use separate namespaces for each agent's memory
3. **Limit traversal depth**: Set reasonable `max_depth` to avoid runaway queries
4. **Use cascade delete carefully**: It removes all connected edges
5. **Filter early**: Use `edge_types` and `node_types` in traversals to reduce work

## See Also

- [Policy & Safety Hooks](policy-hooks.md) - Enforce access policies
- [Tool Routing](tool-routing.md) - Route tools across agents
- [Context Query](context-query.md) - Token-aware retrieval
