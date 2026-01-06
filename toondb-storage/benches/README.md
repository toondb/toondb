# ToonDB Storage Benchmark Results (Optimized)

**Date**: 2026-01-05 (2nd Run)
**System**: macOS
**Improvements**: Added `lz4_flex` and `arc-swap` optimizations.

---

## Summary

Significant performance improvements observed across Reads, Scans, and YCSB workloads (~40-50% faster).

| Metric | Baseline (Prev) | Optimized (New) | Improvement |
|--------|-----------------|-----------------|-------------|
| **Point Read** | 160 µs | **83 µs** | **+48%** |
| **Scan (10 keys)** | 170 µs | **91 µs** | **+46%** |
| **YCSB B (Read Heavy)** | 17.3 ms | **9.7 ms** | **+44%** |
| **Write (Sync=Normal)** | 436 µs | **307 µs** | **+30%** |
| **Write (Sync=Full)** | 4.16 ms | **3.48 ms** | **+16%** |

---

## Point Read Latency

| Dataset Size | Latency (median) | Improvement |
|--------------|------------------|-------------|
| 1,000 keys | **83.49 µs** | ~47% faster |
| 10,000 keys | **82.75 µs** | ~48% faster |
| 100,000 keys | **81.29 µs** | ~47% faster |

**Observation**: Read path optimization (`arc-swap` likely reducing lock contention/ref-count overhead) flattened the latency significantly.

---

## Write Throughput

| Sync Mode | Latency (median) | Change |
|-----------|------------------|--------|
| OFF (no fsync) | **155.90 µs** | -51% (Regression) |
| NORMAL (periodic) | **307.18 µs** | +30% (Faster) |
| FULL (fsync each) | **3.48 ms** | +16% (Faster) |

**Observation**: 
- Mixed results. "Sync=Off" saw a regression (possibly higher allocation overhead from new libs?).
- However, **durable writes** (Normal/Full) improved significanly.
- Batch writes remain highly efficient (0.73 µs per item for batch 100).

---

## Range Scan Performance

| Scan Size | Latency (median) | Improvement |
|-----------|------------------|-------------|
| 10 keys | **91.67 µs** | ~46% faster |
| 100 keys | **84.64 µs** | ~48% faster |
| 1,000 keys | **84.38 µs** | ~49% faster |

**Observation**: Scans are now almost as fast as point reads, suggesting excellent cursor/iterator optimization.

---

## YCSB Workloads

| Workload | Mix | Latency (median) | Improvement |
|----------|-----|------------------|-------------|
| A | 50/50 R/W | **21.12 ms** | +4% |
| B | 95/5 R/W | **9.70 ms** | **+44%** |
| C | 100% Read | **9.35 ms** | **+39%** |
| E | 95/5 Scan/Ins | **9.47 ms** | **+43%** |

**Observation**: Read/Scan-heavy workloads benefit massively from the optimizations. Update-heavy (A) sees smaller gains, limited by the write path.

---

## Concurrency Scaling

| Threads | Total Time | Per-Thread Latency |
|---------|------------|-------------------|
| 1 | **124.35 ms** | 124 µs |
| 2 | **241.61 ms** | 120 µs |
| 4 | **611.81 ms** | 152 µs |
| 8 | **1.36 s** | 170 µs |

---

## Comparison (Preliminary)

ToonDB "no-sync" write throughput in comparison benchmark:
- **ToonDB**: ~9 µs/op (amortized in 100-op loop)

---

## Running Benchmarks

```bash
cargo bench -p toondb-storage
```
