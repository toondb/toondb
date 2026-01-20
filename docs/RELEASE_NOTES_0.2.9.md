# SochDB v0.2.9 Release Notes

**Release Date:** January 2, 2026  
**Focus:** Comprehensive benchmarking, SQL engine maturity, and community infrastructure

---

## 🎯 Highlights

### Real-World Performance Benchmarks
- **3× faster than ChromaDB** for vector search (0.45ms vs 1.37ms p50)
- **22× faster than LanceDB** for vector search (0.45ms vs 9.86ms p50)
- **>98% recall@10** with sub-millisecond latency using real Azure OpenAI embeddings
- End-to-end RAG analysis: **Embedding API is 333× slower than database operations**

### Full SQL Engine in Go SDK
- Complete SQL-92 support (CREATE, INSERT, SELECT, UPDATE, DELETE)
- Feature parity with Python and JavaScript SDKs
- Embedded server mode with zero external dependencies

### Community & Open Source
- Added CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- Added SECURITY.md with vulnerability reporting policy
- Created comprehensive issue templates (bug, feature, support)
- Unified release workflow for automated SDK publishing

---

## 📊 Benchmark Results Summary

### Vector Search Comparison (1,000 real documents)

| Database | Search p50 | Search p99 | Insert Rate |
|----------|------------|------------|-------------|
| **SochDB** | **0.45ms** ✅ | **0.61ms** ✅ | 7,502 vec/s |
| ChromaDB | 1.37ms | 1.73ms | 3,237 vec/s |
| LanceDB | 9.86ms | 21.63ms | 18,106 vec/s |

### Search Quality (Recall@k with Real Embeddings)

| Configuration | Recall@10 | Latency |
|---------------|-----------|---------|
| **M=8, ef_c=50** (recommended) | **99.1%** | 0.42ms |
| M=16, ef_c=100 | 98.2% | 0.47ms |
| M=16, ef_c=200 | 98.8% | 0.44ms |

**Key Finding**: SochDB achieves >98% recall with real embeddings while maintaining sub-millisecond search latency.

### End-to-End RAG Bottleneck

| Operation | Time | % of Total |
|-----------|------|------------|
| Embedding API (Azure OpenAI) | 59.5s | 99.7% |
| SochDB Insert (1K vectors) | 0.133s | 0.2% |
| SochDB Search (100 queries) | 0.046s | 0.1% |

**Insight**: The database is never the bottleneck in production RAG systems—your LLM API calls are.

---

## 🆕 New Features

### Go SDK
- Full SQL engine with DDL/DML operations
- SQL transaction support
- WHERE clause with operators (`=`, `!=`, `<`, `>`, `<=`, `>=`, `LIKE`)
- ORDER BY, LIMIT, OFFSET support
- Complete feature parity with Python/JS SDKs

### Benchmarking Suite
- Real-world embedding tests with Azure OpenAI
- Comparative benchmarks vs ChromaDB and LanceDB
- Recall@k quality measurements
- 360° performance report (quality, latency, throughput, efficiency)

### Community Infrastructure
- Contributor Covenant v2.1 Code of Conduct
- Security vulnerability reporting policy with SLAs
- YAML-based issue templates with validation
- Automated release workflow with PR-based deployment

---

## 🔧 Improvements

### Documentation
- Updated all SDK guides with v0.2.9 features
- Added comprehensive benchmark methodology
- Improved README with real-world performance data
- Added release notes documentation

### Build & Release
- Unified release workflow across all SDKs
- Protected branch support with PR-based releases
- Automated changelog generation
- Fixed YAML syntax in GitHub Actions workflows

---

## 🐛 Bug Fixes

- Fixed Rust compilation errors in storage.rs
- Fixed Go SDK test output formatting (removed redundant newlines)
- Fixed wire protocol documentation for all SDKs
- Fixed GitHub Actions YAML syntax errors

---

## 📦 Installation

### Python
```bash
pip install sochdb-client
```

### Node.js / TypeScript
```bash
npm install @sochdb/sochdb
```

### Go
```bash
go get github.com/sochdb/sochdb-go@latest
```

### Rust
```toml
[dependencies]
sochdb = "0.2.9"
```

---

## 📚 Resources

- **Documentation**: [sochdb.dev](https://sochdb.dev)
- **Benchmarks**: See [benchmarks/BENCHMARK_RESULTS_2024-12-27.md](../benchmarks/BENCHMARK_RESULTS_2024-12-27.md)
- **SDK Guides**: 
  - [Go SDK Guide](guides/go-sdk.md)
  - [Python SDK Guide](guides/python-sdk.md)
  - [Node.js SDK Guide](guides/nodejs-sdk.md)
  - [Rust SDK Guide](guides/rust-sdk.md)

---

## 🔮 What's Next (v0.3.0)

- Replication and clustering support
- Advanced query optimization
- Additional vector index types (IVF, IVFPQ)
- Enhanced monitoring and observability
- GraphQL API support

---

## 🙏 Acknowledgments

Thanks to all contributors and the open source community for feedback, testing, and support!

For questions or support:
- GitHub Issues: [sochdb/sochdb/issues](https://github.com/sochdb/sochdb/issues)
- Email: sushanth@sochdb.dev
