# HNSW Vector Search with Filtering

SochDB provides high-performance vector search using HNSW (Hierarchical Navigable Small World) graphs, with support for session and timestamp filtering. This enables efficient semantic search over agent memory systems, chat histories, and other temporal data.

## Overview

The HNSW index provides:

- **O(log n) search complexity** vs O(n) brute-force
- **~250x speedup** over linear scan for large datasets
- **Session isolation** - search only within a specific session
- **Time-window filtering** - retrieve recent memories only

## Performance Comparison

| Observations | Brute-Force P99 | HNSW P99 | Speedup |
|--------------|-----------------|----------|---------|
| 40           | 143ms           | ~30ms    | 5x      |
| 200          | 7,250ms         | ~50ms    | 145x    |
| 1,000        | ~36,000ms       | ~100ms   | 360x    |
| 10,000       | N/A (timeout)   | ~200ms   | ∞       |

## Python SDK Usage

### Basic Vector Search

```python
from sochdb import HnswIndex
import numpy as np

# Create index with 1536 dimensions (text-embedding-3-small)
index = HnswIndex(
    dimension=1536,
    m=16,                    # Connections per node (trade-off: quality vs memory)
    ef_construction=100,     # Construction quality (higher = better recall)
    metric="cosine"          # Distance metric: "cosine", "euclidean", "dot"
)

# Insert vectors
embeddings = np.random.randn(1000, 1536).astype(np.float32)
index.insert_batch(embeddings)

# Search for nearest neighbors
query = np.random.randn(1536).astype(np.float32)
ids, distances = index.search(query, k=10)

print(f"Found {len(ids)} nearest neighbors")
for id, dist in zip(ids, distances):
    print(f"  ID {id}: distance {dist:.4f}")
```

### Session-Filtered Memory Search

For agent memory systems, you often need to search within a specific session and time window:

```python
from sochdb import Database, HnswIndex
import numpy as np
import time
import json

class MemoryManager:
    """
    Manages agent memory with HNSW indexing for O(log n) search.
    """
    
    EMBEDDING_DIM = 1536
    
    def __init__(self, db_path: str):
        self.db = Database.open(db_path)
        self.hnsw_index = HnswIndex(
            dimension=self.EMBEDDING_DIM,
            m=16,
            ef_construction=100,
            metric="cosine"
        )
        self._id_to_key_map = {}
        self._next_id = 0
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Load existing embeddings into HNSW index on startup."""
        results = self.db.scan_prefix(b"session.")
        
        embeddings = []
        for key, value in results:
            key_str = key.decode()
            if ".embedding" in key_str:
                turn_key = key_str.replace(".embedding", "")
                embedding = np.frombuffer(value, dtype=np.float32)
                
                if len(embedding) == self.EMBEDDING_DIM:
                    hnsw_id = self._next_id
                    self._next_id += 1
                    self._id_to_key_map[hnsw_id] = turn_key
                    embeddings.append((hnsw_id, embedding))
        
        if embeddings:
            ids = np.array([e[0] for e in embeddings], dtype=np.uint64)
            vectors = np.vstack([e[1] for e in embeddings]).astype(np.float32)
            self.hnsw_index.insert_batch_with_ids(ids, vectors)
    
    def store(self, session_id: str, content: str, embedding: np.ndarray):
        """Store a memory with its embedding."""
        turn = int(time.time() * 1000)  # Use timestamp as turn ID
        
        path = f"session.{session_id}.observations.turn_{turn}"
        
        # Store metadata
        metadata = {
            "session_id": session_id,
            "content": content,
            "timestamp": time.time(),
        }
        self.db.put(f"{path}.metadata".encode(), json.dumps(metadata).encode())
        self.db.put(f"{path}.embedding".encode(), embedding.tobytes())
        
        # Add to HNSW index
        hnsw_id = self._next_id
        self._next_id += 1
        self._id_to_key_map[hnsw_id] = path
        
        ids = np.array([hnsw_id], dtype=np.uint64)
        vectors = embedding.reshape(1, -1).astype(np.float32)
        self.hnsw_index.insert_batch_with_ids(ids, vectors)
    
    def search(
        self,
        session_id: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        hours: int = 24
    ):
        """
        Search for similar memories with session and time filtering.
        
        Args:
            session_id: Only return results from this session
            query_embedding: Query vector
            top_k: Number of results to return
            hours: Time window in hours (default: 24)
            
        Returns:
            List of (content, similarity_score) tuples
        """
        # Over-fetch to allow for filtering
        hnsw_k = min(top_k * 3, len(self._id_to_key_map))
        
        if hnsw_k == 0:
            return []
        
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        ids, distances = self.hnsw_index.search(query, k=hnsw_k)
        
        cutoff_time = time.time() - (hours * 3600)
        results = []
        
        for hnsw_id, distance in zip(ids, distances):
            turn_key = self._id_to_key_map.get(int(hnsw_id))
            if not turn_key:
                continue
            
            # Session filter
            if not turn_key.startswith(f"session.{session_id}"):
                continue
            
            # Load and check timestamp
            metadata_bytes = self.db.get(f"{turn_key}.metadata".encode())
            if not metadata_bytes:
                continue
            
            metadata = json.loads(metadata_bytes.decode())
            
            # Time window filter
            if metadata["timestamp"] < cutoff_time:
                continue
            
            # Convert cosine distance to similarity
            similarity = 1.0 - float(distance)
            results.append((metadata["content"], similarity))
            
            if len(results) >= top_k:
                break
        
        return results
```

### Usage Example

```python
import numpy as np

# Initialize
memory = MemoryManager("./agent_memory_db")

# Generate fake embedding (in practice, use OpenAI/Azure embeddings)
def get_embedding(text: str) -> np.ndarray:
    return np.random.randn(1536).astype(np.float32)

# Store memories
memory.store("session_123", "User asked about Python", get_embedding("Python question"))
memory.store("session_123", "Explained list comprehensions", get_embedding("list comprehension"))
memory.store("session_456", "Different session topic", get_embedding("other topic"))

# Search within session
query = get_embedding("How do I use list comprehensions?")
results = memory.search("session_123", query, top_k=5, hours=24)

for content, score in results:
    print(f"[{score:.3f}] {content}")
```

## JavaScript/TypeScript SDK

```typescript
import { VectorIndex } from '@sochdb/sochdb';

// Create HNSW index
const index = new VectorIndex({
  dimension: 1536,
  m: 16,
  efConstruction: 100,
  metric: 'cosine',
});

// Insert vectors
await index.insertBatch(embeddings);

// Search
const results = await index.search(queryVector, { k: 10 });
```

## Configuration Parameters

### HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Max connections per node. Higher = better recall, more memory |
| `ef_construction` | 100 | Construction-time search depth. Higher = better index quality |
| `ef_search` | 50 | Query-time search depth. Higher = better recall, slower |
| `metric` | "cosine" | Distance metric: "cosine", "euclidean", "dot" |

### Recommended Settings

| Use Case | m | ef_construction | ef_search |
|----------|---|-----------------|-----------|
| Speed-optimized | 8 | 50 | 20 |
| Balanced | 16 | 100 | 50 |
| Quality-optimized | 32 | 200 | 100 |
| Production agent | 16 | 100 | 50 |

## Best Practices

### 1. Batch Insertions

```python
# ✅ Good: Batch insert
index.insert_batch(embeddings)  # ~15,000 vec/s

# ❌ Bad: One-by-one insert
for emb in embeddings:
    index.insert(emb)  # ~1,000 vec/s
```

### 2. Contiguous Arrays

```python
# ✅ Good: Contiguous float32 array
embeddings = np.ascontiguousarray(data, dtype=np.float32)
index.insert_batch(embeddings)

# ❌ Bad: Non-contiguous or wrong dtype
embeddings = some_list  # Requires copy
```

### 3. Warm-up Searches

```python
# First search may be slower due to memory allocation
# Warm up the index with a dummy query
_ = index.search(np.zeros(1536, dtype=np.float32), k=1)

# Now benchmark real queries
```

### 4. Session-Based Sharding

For multi-tenant systems, consider one index per tenant:

```python
class MultiTenantMemory:
    def __init__(self):
        self.indices = {}  # tenant_id -> HnswIndex
    
    def get_index(self, tenant_id: str) -> HnswIndex:
        if tenant_id not in self.indices:
            self.indices[tenant_id] = HnswIndex(dimension=1536)
        return self.indices[tenant_id]
```

## Troubleshooting

### High Latency at Scale

If P99 latency exceeds 100ms for 1000+ vectors:

1. **Check index type**: Ensure you're using `HnswIndex`, not brute-force
2. **Reduce ef_search**: Lower values = faster but less accurate
3. **Use batched queries**: `search_batch()` for multiple queries

### Low Recall

If relevant results are missing:

1. **Increase ef_search**: Higher values improve recall
2. **Check embedding quality**: Ensure embeddings are normalized
3. **Verify metric**: Use cosine for text embeddings

### Memory Usage

For 1M 1536-dim vectors:
- Full precision (f32): ~6GB
- Half precision (f16): ~3GB
- BF16 quantization: ~3GB

```python
# Enable quantization for memory savings
index = HnswIndex(dimension=1536, precision="f16")
```

## See Also

- [Vector Search Concepts](../concepts/vector-search.md)
- [Python SDK Reference](../api-reference/python-sdk.md)
- [Performance Tuning](./performance-tuning.md)
