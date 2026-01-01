# ToonDB Architecture

Deep technical documentation for ToonDB's internal architecture.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Data Model](#data-model)
3. [TOON Format Internals](#toon-format-internals)
4. [Storage Engine](#storage-engine)
5. [Transaction System](#transaction-system)
6. [Index Structures](#index-structures)
7. [Query Processing](#query-processing)
8. [Memory Management](#memory-management)
9. [Concurrency Model](#concurrency-model)
10. [Recovery & Durability](#recovery--durability)
11. [Python SDK Architecture](#python-sdk-architecture)

---

## Design Philosophy

ToonDB is built around four core principles:

### 1. Token Efficiency First

Every design decision prioritizes minimizing tokens when data is consumed by LLMs:

```
Traditional approach:
┌─────────────────────────────────────────────────────────┐
│  LLM ← JSON ← SQL Result ← Query Optimizer ← B-Tree    │
│       ~150 tokens for 3 rows                            │
└─────────────────────────────────────────────────────────┘

ToonDB approach:
┌─────────────────────────────────────────────────────────┐
│  LLM ← TOON ← Columnar Scan ← Path Resolution          │
│       ~50 tokens for 3 rows (66% reduction)            │
└─────────────────────────────────────────────────────────┘
```

### 2. Path-Based Access

O(|path|) resolution instead of O(log N) tree traversal:

```
Path: "users/42/profile/avatar"

TCH Resolution (Trie-Columnar Hybrid):
├─ users     → Table lookup (O(1) hash)
│  └─ 42     → Row index (O(1) direct)
│     └─ profile → Column group (O(1))
│        └─ avatar → Column offset (O(1))

Total: O(4) = O(|path|), regardless of table size
```

### 3. Columnar by Default

Read only what you need:

```
SELECT name, email FROM users WHERE id = 42;

Row Store (read all columns):
┌─────┬──────┬───────┬─────┬─────────────┬────────┐
│ id  │ name │ email │ age │ preferences │ avatar │
└─────┴──────┴───────┴─────┴─────────────┴────────┘
  ↑ Read 6 columns × row size

Columnar Store (read only needed):
┌─────┐  ┌──────┐  ┌───────┐
│ id  │  │ name │  │ email │
└──↑──┘  └──↑───┘  └───↑───┘
Read 3 columns only = 50% I/O reduction
```

### 4. Embeddable & Extensible

Single-file deployment with plugin architecture:

```
┌──────────────────────────────────────────────────────┐
│                    Your Application                   │
├──────────────────────────────────────────────────────┤
│  ToonDB (embedded)                                   │
│  ┌────────────┐ ┌─────────────┐ ┌─────────────────┐ │
│  │   Core     │ │   Kernel    │ │    Plugins      │ │
│  │  ~500 KB   │ │   ~1 MB     │ │   (optional)    │ │
│  └────────────┘ └─────────────┘ └─────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Data Model

### Trie-Columnar Hybrid (TCH)

TCH combines trie-based path resolution with columnar storage:

```
                    ┌─────────────────────────────────────┐
                    │           Path Trie                  │
                    │                                      │
                    │           [root]                     │
                    │          /      \                    │
                    │      users      orders               │
                    │      /    \        \                 │
                    │    [id]  [*]      [id]              │
                    │    / | \           |                 │
                    │  name email age  amount              │
                    └────────┬────────────────────────────┘
                             │
                    ┌────────▼────────────────────────────┐
                    │         Column Store                 │
                    │                                      │
                    │  users.id    [1, 2, 3, 4, 5, ...]   │
                    │  users.name  [Alice, Bob, ...]       │
                    │  users.email [a@e.com, b@e.com, ...] │
                    │  users.age   [25, 30, 28, ...]       │
                    └─────────────────────────────────────┘
```

### Path Resolution Algorithm

```rust
fn resolve(&self, path: &str) -> PathResolution {
    let parts: Vec<&str> = path.split('/').collect();
    
    // Navigate trie
    let mut node = &self.root;
    for part in &parts[..parts.len()-1] {
        node = match node.children.get(*part) {
            Some(child) => child,
            None => return PathResolution::NotFound,
        };
    }
    
    // Final part determines resolution type
    let final_part = parts.last().unwrap();
    match node.node_type {
        NodeType::Table => PathResolution::Array { ... },
        NodeType::Row => PathResolution::Value { ... },
        NodeType::Column => {
            // Direct column access
            let col_idx = self.column_index(node, final_part);
            PathResolution::Column { idx: col_idx }
        }
    }
}
```

### Schema Representation

```rust
struct TableInfo {
    schema: ArraySchema,
    columns: Vec<ColumnRef>,
    next_row_id: u64,
    indexes: HashMap<String, IndexRef>,
}

struct ColumnRef {
    id: u32,
    name: String,
    field_type: FieldType,
    compression: Compression,
    encoding: Encoding,
}

enum FieldType {
    Int64,
    UInt64,
    Float64,
    Text,
    Bytes,
    Bool,
    Vector(usize),
}
```

---

## TOON Format Internals

### Text Format Grammar

```ebnf
document     ::= table_header newline row*
table_header ::= name "[" count "]" "{" fields "}" ":"
name         ::= identifier
count        ::= integer
fields       ::= field ("," field)*
field        ::= identifier
row          ::= value ("," value)* newline
value        ::= null | bool | number | string | array | ref

null         ::= "∅"
bool         ::= "T" | "F"
number       ::= integer | float
integer      ::= "-"? digit+
float        ::= "-"? digit+ "." digit+
string       ::= raw_string | quoted_string
raw_string   ::= [^,;\n"]+
quoted_string::= '"' ([^"\\] | escape)* '"'
array        ::= "[" value ("," value)* "]"
ref          ::= "ref(" identifier "," integer ")"
```

### Binary Format

```
┌──────────────────────────────────────────────────────────────┐
│                    TOON Binary Format                         │
├──────────────────────────────────────────────────────────────┤
│ Header (16 bytes)                                            │
│ ┌──────────┬──────────┬──────────┬──────────┐               │
│ │  Magic   │ Version  │  Flags   │ Row Count│               │
│ │ (4 bytes)│ (2 bytes)│ (2 bytes)│ (8 bytes)│               │
│ └──────────┴──────────┴──────────┴──────────┘               │
├──────────────────────────────────────────────────────────────┤
│ Schema Section                                               │
│ ┌──────────┬──────────────────────────────────┐             │
│ │ Name Len │ Table Name (UTF-8)               │             │
│ ├──────────┼──────────────────────────────────┤             │
│ │ Col Count│ [Column Definitions...]           │             │
│ └──────────┴──────────────────────────────────┘             │
├──────────────────────────────────────────────────────────────┤
│ Data Section (columnar)                                      │
│ ┌──────────────────────────────────────────────┐            │
│ │ Column 0: [type_tag][values...]              │            │
│ │ Column 1: [type_tag][values...]              │            │
│ │ ...                                          │            │
│ └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### Type Tags

```rust
#[repr(u8)]
pub enum ToonTypeTag {
    Null      = 0x00,
    False     = 0x01,
    True      = 0x02,
    PosFixint = 0x10,  // 0-15 in lower nibble
    NegFixint = 0x20,  // -16 to -1 in lower nibble
    Int8      = 0x30,
    Int16     = 0x31,
    Int32     = 0x32,
    Int64     = 0x33,
    Float32   = 0x40,
    Float64   = 0x41,
    FixStr    = 0x50,  // 0-15 char length in lower nibble
    Str8      = 0x60,
    Str16     = 0x61,
    Str32     = 0x62,
    Array     = 0x70,
    Ref       = 0x80,
}
```

### Varint Encoding

```rust
fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn decode_varint(buf: &[u8]) -> (u64, usize) {
    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;
    
    for &byte in buf {
        bytes_read += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    
    (result, bytes_read)
}
```

---

## Storage Engine

### Log-Structured Columnar Store (LSCS)

```
┌────────────────────────────────────────────────────────────────┐
│                      LSCS Architecture                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Write Path:                                                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │  Write  │ →  │   WAL   │ →  │Memtable │ →  │  SST    │     │
│  │ Request │    │ (Append)│    │(In-mem) │    │ (Disk)  │     │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                                 │
│  Read Path:                                                     │
│  ┌─────────┐    ┌─────────────────────────────────────┐        │
│  │  Read   │ →  │ Memtable → L0 SSTs → L1 → L2 → ...  │        │
│  │ Request │    │     (Bloom filter + block cache)     │        │
│  └─────────┘    └─────────────────────────────────────┘        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### SST File Format

```
┌─────────────────────────────────────────────────────────────┐
│                    SST File Structure                        │
├─────────────────────────────────────────────────────────────┤
│ Data Blocks                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Block 0: [key1:val1][key2:val2]...[keyN:valN][trailer]  │ │
│ │ Block 1: [key1:val1][key2:val2]...[keyN:valN][trailer]  │ │
│ │ ...                                                      │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Meta Blocks                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Bloom Filter Block                                       │ │
│ │ Column Stats Block                                       │ │
│ │ Compression Dict Block (optional)                        │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Index Block                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [first_key_0, offset_0, size_0]                          │ │
│ │ [first_key_1, offset_1, size_1]                          │ │
│ │ ...                                                      │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Footer (48 bytes)                                            │
│ ┌────────────────┬────────────────┬────────────────────────┐│
│ │ Meta Index     │ Index Handle   │ Magic + Version        ││
│ │ BlockHandle    │ BlockHandle    │                        ││
│ └────────────────┴────────────────┴────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Block Checksums

```rust
/// CRC32C with hardware acceleration (SSE4.2)
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = !0;
    
    // Process 8 bytes at a time using CRC32Q
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        let val = u64::from_le_bytes(chunk.try_into().unwrap());
        crc = unsafe { _mm_crc32_u64(crc as u64, val) as u32 };
    }
    
    // Handle remaining bytes
    for &byte in remainder {
        crc = unsafe { _mm_crc32_u8(crc, byte) };
    }
    
    !crc
}

/// Mask CRC to avoid all-zeros problem
pub fn mask_crc(crc: u32) -> u32 {
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)
}
```

### Compaction

```
Level 0 (L0): Recent flushes, may overlap
┌────┐ ┌────┐ ┌────┐ ┌────┐
│SST1│ │SST2│ │SST3│ │SST4│  ← 4 files, overlapping key ranges
└────┘ └────┘ └────┘ └────┘
          │
          ▼ Compaction (merge sort)
Level 1 (L1): Non-overlapping, sorted
┌──────────────────────────────────────┐
│              SST (merged)             │
└──────────────────────────────────────┘
          │
          ▼ Size-triggered compaction
Level 2 (L2): 10x larger budget
┌────────────────────────────────────────────────────────────┐
│                      SST files                              │
└────────────────────────────────────────────────────────────┘
```

---

## Transaction System

### MVCC Implementation

```rust
struct VersionedValue {
    value: Option<Vec<u8>>,  // None = tombstone
    txn_id: u64,             // Transaction that wrote this
    timestamp: u64,          // Commit timestamp
    next: Option<Box<VersionedValue>>,  // Older versions
}

struct MVCCStore {
    current: HashMap<Key, VersionedValue>,
    gc_watermark: AtomicU64,
}

impl MVCCStore {
    fn get(&self, key: &Key, snapshot_ts: u64) -> Option<&[u8]> {
        let mut version = self.current.get(key)?;
        
        // Find visible version
        while version.timestamp > snapshot_ts {
            version = version.next.as_ref()?;
        }
        
        version.value.as_deref()
    }
    
    fn put(&self, key: Key, value: Vec<u8>, txn: &Transaction) {
        let new_version = VersionedValue {
            value: Some(value),
            txn_id: txn.id,
            timestamp: txn.commit_ts,
            next: self.current.remove(&key).map(Box::new),
        };
        self.current.insert(key, new_version);
    }
}
```

### Transaction Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                     Transaction Lifecycle                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  BEGIN                                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. Allocate txn_id (atomic increment)                        │ │
│  │ 2. Take snapshot_ts = current_ts                             │ │
│  │ 3. Add to active_transactions set                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│                          ▼                                        │
│  OPERATIONS (read/write)                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Reads:  Use snapshot_ts for visibility                       │ │
│  │ Writes: Buffer in transaction-local write set                │ │
│  │         Check for write-write conflicts                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│                ┌─────────┴─────────┐                             │
│                ▼                   ▼                              │
│           COMMIT              ROLLBACK                            │
│  ┌──────────────────┐   ┌──────────────────┐                     │
│  │ 1. Validate      │   │ 1. Discard write │                     │
│  │ 2. Write to WAL  │   │    set           │                     │
│  │ 3. Apply writes  │   │ 2. Remove from   │                     │
│  │ 4. Advance ts    │   │    active set    │                     │
│  │ 5. Remove from   │   │ 3. Release locks │                     │
│  │    active set    │   └──────────────────┘                     │
│  └──────────────────┘                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Group Commit

```rust
/// Optimal batch size: N* = √(2 × L_fsync × λ / C_wait)
struct GroupCommitBuffer {
    pending: VecDeque<PendingCommit>,
    batch_id: u64,
    config: GroupCommitConfig,
}

impl GroupCommitBuffer {
    fn optimal_batch_size(&self, arrival_rate: f64, wait_cost: f64) -> usize {
        let l_fsync = self.config.fsync_latency_us as f64 / 1_000_000.0;
        let n_star = (2.0 * l_fsync * arrival_rate / wait_cost).sqrt();
        (n_star as usize).clamp(1, self.config.max_batch_size)
    }
    
    fn submit_and_wait(&self, op_id: u64) -> Result<u64> {
        let mut inner = self.inner.lock();
        
        inner.pending.push_back(PendingCommit {
            id: op_id,
            batch_id: inner.batch_id,
            committed: false,
        });
        
        // Flush if batch is full
        if inner.pending.len() >= self.config.target_batch_size {
            self.flush_batch(&mut inner)?;
        }
        
        // Wait for commit (with timeout)
        self.condvar.wait_for(&mut inner, self.config.max_wait);
        
        Ok(inner.batch_id)
    }
}
```

---

## Index Structures

### HNSW (Vector Index)

```
┌────────────────────────────────────────────────────────────────┐
│                    HNSW Graph Structure                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 2:       ●───────────────────────●                      │
│                 │                       │                       │
│  Layer 1:       ●───●───────●───────────●───────●              │
│                 │   │       │           │       │               │
│  Layer 0:   ●───●───●───●───●───●───●───●───●───●───●          │
│             │   │   │   │   │   │   │   │   │   │   │          │
│            v0  v1  v2  v3  v4  v5  v6  v7  v8  v9  v10         │
│                                                                 │
│  Search: Start at top layer, greedily descend                  │
│  Insert: Random level, connect at each layer                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### HNSW Parameters

```rust
struct HNSWConfig {
    /// M: Max edges per node per layer (except layer 0)
    /// Higher M = better recall, more memory
    m: usize,  // default: 16
    
    /// M_max0: Max edges at layer 0
    /// Usually 2*M
    m_max: usize,  // default: 32
    
    /// ef_construction: Search width during build
    /// Higher = better graph quality, slower build
    ef_construction: usize,  // default: 200
    
    /// ef_search: Search width during query
    /// Higher = better recall, slower query
    ef_search: usize,  // default: 50
    
    /// Level multiplier: mL = 1/ln(M)
    /// Controls probability of higher levels
    ml: f32,  // default: 1/ln(16) ≈ 0.36
}
```

### Level Generation

```rust
/// Per HNSW paper: level = floor(-ln(uniform(0,1)) * mL)
fn random_level(&self) -> usize {
    let uniform: f32 = thread_rng().gen();
    let level = if uniform > 0.0 {
        (-uniform.ln() * self.ml).floor() as usize
    } else {
        0
    };
    level.min(MAX_LEVEL)
}

// Distribution for M=16, mL≈0.36:
// P(level=0) ≈ 70%
// P(level=1) ≈ 21%
// P(level=2) ≈ 6%
// P(level≥3) ≈ 3%
```

### Search Algorithm

```rust
fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let nodes = self.nodes.read();
    let entry = self.entry_point.load(Ordering::Acquire);
    
    if nodes.is_empty() {
        return vec![];
    }
    
    let mut current = entry;
    let mut current_dist = self.distance(query, &nodes[current].vector);
    
    // Traverse from top to layer 1
    for layer in (1..=self.max_level).rev() {
        loop {
            let mut changed = false;
            for &neighbor in &nodes[current].layers[layer] {
                let dist = self.distance(query, &nodes[neighbor].vector);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    changed = true;
                }
            }
            if !changed { break; }
        }
    }
    
    // Search at layer 0 with ef candidates
    self.search_layer(&nodes, query, current, 0, self.ef_search)
        .into_iter()
        .take(k)
        .map(|idx| (nodes[idx].edge_id, self.distance(query, &nodes[idx].vector)))
        .collect()
}
```

---

## Query Processing

### Query Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      Query Pipeline                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. PARSE                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Input: conn.query("users").where_eq("status", "active")     │ │
│  │ Output: QueryAST { table, predicates, projections, ... }    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│  2. PLAN                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Choose access method (scan vs index)                       │ │
│  │ • Push down predicates                                       │ │
│  │ • Plan column projections                                    │ │
│  │ Output: LogicalPlan                                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│  3. OPTIMIZE             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Cost-based index selection                                 │ │
│  │ • Predicate ordering (selectivity)                          │ │
│  │ • Join ordering (if applicable)                              │ │
│  │ Output: PhysicalPlan                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│  4. EXECUTE              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Open column readers                                        │ │
│  │ • Apply predicates (vectorized)                              │ │
│  │ • Materialize results                                        │ │
│  │ Output: QueryResult                                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│  5. FORMAT               ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • TOON: users[N]{cols}:row1;row2;...                        │ │
│  │ • JSON: [{"col": val}, ...]                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Predicate Pushdown

```rust
fn push_predicates(plan: &mut ScanPlan, predicates: &[Predicate]) {
    for pred in predicates {
        match pred {
            // Can push to storage layer
            Predicate::Eq(col, val) if is_indexed(col) => {
                plan.index_lookup = Some(IndexLookup {
                    column: col.clone(),
                    value: val.clone(),
                });
            }
            
            // Push to block-level filtering
            Predicate::Range(col, min, max) => {
                plan.block_filters.push(BlockFilter {
                    column: col.clone(),
                    min: min.clone(),
                    max: max.clone(),
                });
            }
            
            // Late filter (after scan)
            _ => plan.late_filters.push(pred.clone()),
        }
    }
}
```

---

## Memory Management

### Buddy Allocator

```
┌────────────────────────────────────────────────────────────────┐
│                    Buddy Allocator                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Order 10 (1KB):  [████████████████████████████████]           │
│                            │                                    │
│                   ┌────────┴────────┐                          │
│  Order 9 (512B):  [████████████████] [________________]         │
│                         │                                       │
│                   ┌─────┴─────┐                                 │
│  Order 8 (256B):  [████████] [████]  ...                       │
│                                                                 │
│  Allocation: Find smallest power-of-2 block, split if needed   │
│  Deallocation: Coalesce with buddy if both free                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Arena Allocator

```rust
struct BuddyArena {
    buddy: BuddyAllocator,
    current_block: Mutex<Option<ArenaBlock>>,
    block_size: usize,
}

impl BuddyArena {
    fn allocate(&self, size: usize, align: usize) -> Result<usize> {
        let mut current = self.current_block.lock();
        
        // Try current block
        if let Some(ref mut block) = *current {
            let aligned = (block.offset + align - 1) & !(align - 1);
            if aligned + size <= block.size {
                block.offset = aligned + size;
                return Ok(block.base + aligned);
            }
        }
        
        // Allocate new block
        let new_size = size.max(self.block_size).next_power_of_two();
        let base = self.buddy.allocate(new_size)?;
        
        *current = Some(ArenaBlock {
            base,
            offset: size,
            size: new_size,
        });
        
        Ok(base)
    }
    
    fn reset(&self) {
        // Free all blocks at once
        self.current_block.lock().take();
        for block in self.blocks.drain(..) {
            self.buddy.deallocate(block);
        }
    }
}
```

---

## Concurrency Model

### Lock Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Lock Acquisition Order                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Catalog Lock (RwLock)                                       │
│     └── Table schema changes                                     │
│                                                                  │
│  2. Table Lock (per-table RwLock)                               │
│     └── DDL operations on specific table                        │
│                                                                  │
│  3. Transaction Manager Lock (Mutex)                             │
│     └── Begin/commit/abort transactions                          │
│                                                                  │
│  4. WAL Lock (Mutex)                                             │
│     └── Append to write-ahead log                                │
│                                                                  │
│  5. Memtable Lock (RwLock)                                       │
│     └── In-memory writes                                         │
│                                                                  │
│  6. Index Lock (per-index RwLock)                                │
│     └── Index modifications                                      │
│                                                                  │
│  ALWAYS acquire in this order to prevent deadlocks              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Lock-Free Reads

```rust
// MVCC enables lock-free reads via snapshots
impl Database {
    fn read(&self, key: &[u8], snapshot: Snapshot) -> Option<Value> {
        // No locks needed - snapshot isolation
        let version = self.mvcc.get_visible(key, snapshot.timestamp);
        version.map(|v| v.value.clone())
    }
}

// Snapshot is just a timestamp
struct Snapshot {
    timestamp: u64,
    txn_id: u64,
}
```

---

## Recovery & Durability

### WAL Format

```
┌────────────────────────────────────────────────────────────────┐
│                       WAL Record Format                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐ │
│  │  CRC32   │  Length  │   Type   │  TxnID   │    Data      │ │
│  │ (4 bytes)│ (4 bytes)│ (1 byte) │ (8 bytes)│  (variable)  │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘ │
│                                                                 │
│  Record Types:                                                  │
│  • 0x01: PUT (key, value)                                      │
│  • 0x02: DELETE (key)                                          │
│  • 0x03: BEGIN_TXN (txn_id)                                    │
│  • 0x04: COMMIT_TXN (txn_id, commit_ts)                        │
│  • 0x05: ABORT_TXN (txn_id)                                    │
│  • 0x06: CHECKPOINT (LSN, active_txns)                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Recovery Process

```rust
fn recover(&self) -> Result<RecoveryStats> {
    let mut stats = RecoveryStats::default();
    
    // 1. Find latest checkpoint
    let checkpoint = self.find_latest_checkpoint()?;
    stats.checkpoint_lsn = checkpoint.lsn;
    
    // 2. Replay WAL from checkpoint
    let mut wal_reader = WalReader::open_from(checkpoint.lsn)?;
    let mut active_txns: HashSet<u64> = checkpoint.active_txns;
    
    while let Some(record) = wal_reader.next()? {
        match record.record_type {
            RecordType::BeginTxn => {
                active_txns.insert(record.txn_id);
            }
            RecordType::Put => {
                if !active_txns.contains(&record.txn_id) {
                    // Skip aborted transaction
                    continue;
                }
                self.replay_put(&record)?;
                stats.records_replayed += 1;
            }
            RecordType::CommitTxn => {
                active_txns.remove(&record.txn_id);
                stats.transactions_recovered += 1;
            }
            RecordType::AbortTxn => {
                // Roll back buffered writes
                self.rollback_txn(record.txn_id)?;
                active_txns.remove(&record.txn_id);
            }
            _ => {}
        }
    }
    
    // 3. Abort incomplete transactions
    for txn_id in active_txns {
        self.rollback_txn(txn_id)?;
        stats.transactions_aborted += 1;
    }
    
    Ok(stats)
}
```

### Checkpointing

```rust
fn checkpoint(&self) -> Result<u64> {
    // 1. Acquire checkpoint lock
    let _guard = self.checkpoint_lock.lock();
    
    // 2. Get current LSN
    let checkpoint_lsn = self.wal.current_lsn();
    
    // 3. Flush memtable to SST
    let immutable = self.memtable.freeze();
    self.flush_memtable_to_sst(&immutable)?;
    
    // 4. Write checkpoint record
    let active_txns = self.txn_manager.active_transactions();
    self.wal.append(WalRecord::Checkpoint {
        lsn: checkpoint_lsn,
        active_txns,
    })?;
    
    // 5. Sync WAL
    self.wal.sync()?;
    
    // 6. Remove old WAL segments
    self.wal.truncate_before(checkpoint_lsn)?;
    
    Ok(checkpoint_lsn)
}
```

---

## SDK Architecture

ToonDB provides official SDKs for multiple languages. All SDKs communicate with the ToonDB server via IPC (Unix domain sockets on Linux/macOS, named pipes on Windows).

### Client-Server Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Client Applications                               │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Python    │  │   JavaScript    │  │      Go      │  │    Rust    │ │
│  │  toondb-py  │  │  @sushanth/     │  │  toondb-go   │  │   toondb   │ │
│  │             │  │     toondb      │  │              │  │   crate    │ │
│  └──────┬──────┘  └───────┬─────────┘  └──────┬───────┘  └─────┬──────┘ │
└─────────┼─────────────────┼────────────────────┼─────────────────┼───────┘
          │                 │                    │                 │
          └─────────────────┼────────────────────┼─────────────────┘
                            │    IPC/Socket      │
                            │   (toondb.sock)    │
                            ▼                    ▼
                    ┌──────────────────────────────────┐
                    │         ToonDB Server            │
                    │         (toondb-mcp)             │
                    ├──────────────────────────────────┤
                    │  ┌─────────────────────────────┐ │
                    │  │      Protocol Handler       │ │
                    │  │   (Wire Protocol Parser)    │ │
                    │  └─────────────┬───────────────┘ │
                    │                │                 │
                    │  ┌─────────────▼───────────────┐ │
                    │  │       Query Engine          │ │
                    │  │    (toondb-query crate)     │ │
                    │  └─────────────┬───────────────┘ │
                    │                │                 │
                    │  ┌─────────────▼───────────────┐ │
                    │  │      Storage Engine         │ │
                    │  │   (LSM + WAL + Memtable)    │ │
                    │  └─────────────────────────────┘ │
                    └──────────────────────────────────┘
```

### SDK Features Comparison

| Feature               | Python    | JavaScript | Go         | Rust       |
|-----------------------|-----------|------------|------------|------------|
| Embedded Mode         | ✅ Auto   | ✅ Auto    | ❌ Manual  | ✅ Direct  |
| IPC Client            | ✅        | ✅         | ✅         | ✅         |
| Vector Search         | ✅        | ✅         | ✅         | ✅         |
| Transactions          | ✅        | ✅         | ✅         | ✅         |
| Query Builder         | ✅        | ✅         | ✅         | ✅         |
| Pre-built Binaries    | ✅        | ✅         | N/A        | N/A        |

### Embedded vs External Server Mode

**Embedded Mode (Python & JavaScript):**
The SDK automatically spawns and manages a `toondb-mcp` server process. The server is started on `Database.open()` and stopped on `Database.close()`.

```python
# Python - embedded mode (default)
db = ToonDB.open("./my_database")
# Server started automatically
db.close()
# Server stopped automatically
```

```javascript
// JavaScript - embedded mode (default)
const db = await Database.open('./my_database');
// Server started automatically
await db.close();
// Server stopped automatically
```

**External Server Mode (Go):**
The Go SDK requires manual server startup:

```bash
# Start server first
./toondb-mcp serve --db ./my_database
```

```go
// Then connect from Go
db, err := toondb.Open("./my_database")
```

---

## Python SDK Architecture

The Python SDK provides multiple access patterns to ToonDB:

### Access Modes

```
┌──────────────────────────────────────────────────────────────────┐
│                     Python Application                            │
├──────────────────────────────────────────────────────────────────┤
│                        toondb (PyPI)                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────────────┐ │
│  │  Embedded  │  │    IPC     │  │        Bulk API             │ │
│  │    FFI     │  │   Client   │  │  (subprocess → toondb-bulk) │ │
│  └─────┬──────┘  └─────┬──────┘  └──────────────┬──────────────┘ │
│        │               │                         │                │
└────────┼───────────────┼─────────────────────────┼────────────────┘
         │               │                         │
    ┌────▼────┐    ┌─────▼─────┐           ┌───────▼───────┐
    │  Rust   │    │    IPC    │           │  toondb-bulk  │
    │  FFI    │    │  Server   │           │    binary     │
    │  (.so)  │    │           │           │               │
    └────┬────┘    └─────┬─────┘           └───────┬───────┘
         │               │                         │
         └───────────────┴─────────────────────────┘
                         │
                   ┌─────▼─────┐
                   │  ToonDB   │
                   │   Core    │
                   └───────────┘
```

### Distribution Model (uv-style)

Wheels contain pre-built Rust binaries, eliminating compilation requirements:

```
toondb-0.2.3-py3-none-manylinux_2_17_x86_64.whl
├── toondb/
│   ├── __init__.py
│   ├── database.py      # Embedded FFI
│   ├── ipc.py           # IPC client
│   ├── bulk.py          # Bulk operations
│   └── _bin/
│       └── linux-x86_64/
│           └── toondb-bulk  # Pre-built binary
└── METADATA
```

Platform matrix:
- `manylinux_2_17_x86_64` - Linux glibc ≥ 2.17
- `manylinux_2_17_aarch64` - Linux ARM64
- `macosx_11_0_universal2` - macOS Intel + Apple Silicon
- `win_amd64` - Windows x64

### Bulk API FFI Bypass

For vector-heavy workloads, the Bulk API avoids FFI overhead:

```
Python FFI path (130 vec/s):
┌─────────┐    memcpy    ┌──────┐
│ numpy   │ ────────────→│ Rust │ → repeated N times
└─────────┘   per batch  └──────┘

Bulk API path (1,600 vec/s):
┌─────────┐   mmap    ┌──────────────┐   fork    ┌──────────────┐
│ numpy   │ ────────→ │  temp file   │ ────────→ │ toondb-bulk  │
└─────────┘  1 write  └──────────────┘  1 proc   └──────────────┘
```

See [PYTHON_DISTRIBUTION.md](PYTHON_DISTRIBUTION.md) for full distribution architecture.

---

*This document describes ToonDB v0.2.4 internals. Implementation details may change.*
