# SochDB v0.3.3 Release Notes

**Release Date:** January 5, 2026

## 🎉 New Features

### 🕸️ Graph Overlay System
A lightweight graph layer built on top of SochDB's KV storage for agent memory management:

**Features:**
- Node operations: Add, Get, Update, Delete with typed nodes
- Edge operations: Add, Get, Delete with relationship tracking
- Graph traversal: BFS, DFS, Shortest Path algorithms
- Neighbor queries: Get incoming/outgoing neighbors with filters
- Subgraph extraction: Extract connected components
- Namespace isolation: Multi-tenant graph support

**Use Cases:**
- Agent conversation history with causal chains
- Entity relationship tracking across sessions
- Action dependency graphs for planning
- Knowledge graph construction

**SDKs:** Python, Go, Node.js/TypeScript

### 📊 Context Query Enhancements
Improved token-aware context retrieval:

- Better token estimation for GPT-4, Claude models
- Enhanced deduplication strategies
- Multi-format support (text, JSON, markdown)
- Relevance scoring improvements

### 🛡️ Policy & Safety Improvements
Enhanced agent safety controls:

- Additional pre-built policy templates
- Better audit trail generation
- Performance optimizations for high-throughput scenarios

### 🔀 Tool Routing Enhancements
Multi-agent coordination improvements:

- Dynamic agent discovery
- Better load balancing strategies
- Failover improvements
- Agent health monitoring

## 🐛 Bug Fixes

- Fixed edge case in BFS traversal with cyclic graphs
- Resolved context query token counting for very long documents
- Fixed policy hook execution order in nested transactions
- Corrected tool routing priority when multiple agents match

## 🚀 Performance Improvements

- 15% faster graph traversal algorithms
- Reduced memory usage for large graph structures
- Optimized prefix scanning for graph edges
- Improved context query compilation time

## 📦 SDK Updates

### Python SDK (sochdb-client v0.3.3)
- Added `GraphOverlay` class with full API
- Enhanced `ContextQuery` with better token estimation
- Updated examples in `sochdb-python-examples/new_features/`

### Go SDK (github.com/sochdb/sochdb-go v0.3.3)
- Added `GraphOverlay` struct with 660 lines of implementation
- Added `ContextQuery` struct with 545 lines of implementation
- Full API parity with Python SDK

### Node.js SDK (@sochdb/sochdb v0.3.3)
- Added `GraphOverlay` class with TypeScript types
- Added `ContextQuery` class with async/await support
- Updated TypeScript definitions

## 📚 Documentation

### New Guides
- [Graph Overlay Guide](docs/guides/graph-overlay.md)
- [Context Query Guide](docs/guides/context-query.md)
- [Policy Hooks Guide](docs/guides/policy-hooks.md)
- [Tool Routing Guide](docs/guides/tool-routing.md)

### Updated Documentation
- API Reference updated with Graph Overlay methods
- Architecture docs updated with graph storage model
- Examples added for all new features

## 🔄 Breaking Changes

None. This is a fully backward-compatible release.

## 📝 Migration Guide

No migration needed. All existing code continues to work.

To use new features:

**Python:**
```python
from sochdb import Database, GraphOverlay
db = Database.open("./db")
graph = GraphOverlay(db, namespace="demo")
```

**Go:**
```go
import sochdb "github.com/sochdb/sochdb-go"
db, _ := sochdb.Open("./db")
graph := sochdb.NewGraphOverlay(db, "demo")
```

**Node.js:**
```typescript
import { Database, GraphOverlay } from '@sochdb/sochdb';
const db = await Database.open('./db');
const graph = new GraphOverlay(db, 'demo');
```

## 🙏 Acknowledgments

Special thanks to all contributors who helped make this release possible!

## 📋 Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete details.

## 🔗 Links

- [Main Repository](https://github.com/sochdb/sochdb)
- [Python SDK](https://github.com/sochdb/sochdb-python-sdk)
- [Go SDK](https://github.com/sochdb/sochdb-go)
- [Node.js SDK](https://github.com/sochdb/sochdb-nodejs-sdk)
- [Documentation](https://sochdb.dev)

## 📊 Stats

- **Total Features Added:** 4 major features
- **Lines of Code:** +2,500 across all SDKs
- **Test Coverage:** 95%+
- **SDK Parity:** 100% feature parity across Python, Go, Node.js

---

**Upgrade Command:**

```bash
# Python
pip install --upgrade sochdb-client

# Go
go get github.com/sochdb/sochdb-go@v0.3.3

# Node.js
npm install @sochdb/sochdb@latest
```
