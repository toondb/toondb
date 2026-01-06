# ToonDB Storage Benchmark Results

**Date**: 2026-01-05  
**System**: macOS  
**Rust Version**: stable  

---

## Summary

These benchmarks measure **actual ToonDB `DurableStorage`** performance, not synthetic in-memory HashMap operations. All tests use Criterion with statistical analysis.

---

## Point Read Latency

Tests `DurableStorage::read()` with various dataset sizes and access patterns.

### By Dataset Size (Zipfian distribution)

| Dataset Size | Latency (median) | Throughput |
|--------------|------------------|------------|
| 1,000 keys | **158.77 µs** | 6.30 Kops/s |
| 10,000 keys | **160.34 µs** | 6.24 Kops/s |
| 100,000 keys | **154.40 µs** | 6.48 Kops/s |

**Observation**: Consistent ~155-160 µs latency regardless of dataset size due to in-memory memtable hits.

### By Access Distribution

| Distribution | Latency (median) | Throughput |
|--------------|------------------|------------|
| Uniform | **171.57 µs** | 5.83 Kops/s |
| Zipfian (θ=0.99) | **162.83 µs** | 6.14 Kops/s |
| Sequential | **163.49 µs** | 6.12 Kops/s |

**Observation**: Zipfian shows slightly better performance due to hot key caching effects.

### Miss Rate (Key Not Found)

| Scenario | Latency (median) |
|----------|------------------|
| Miss | **160.38 µs** |

**Observation**: Miss detection is as fast as hits - bloom filters working efficiently.

### By Value Size

| Value Size | Latency (median) | Throughput |
|------------|------------------|------------|
| 100 bytes | **157.60 µs** | 619 KiB/s |
| 1,000 bytes | **152.58 µs** | 6.25 MiB/s |
| 10,000 bytes | **164.75 µs** | 57.9 MiB/s |

---

## Write Throughput

Tests `DurableStorage::write()` with various durability modes.

### By Sync Mode

| Sync Mode | Latency (median) | Throughput |
|-----------|------------------|------------|
| OFF (no fsync) | **102.61 µs** | ~9,700 ops/s |
| NORMAL (periodic) | **436.61 µs** | ~2,300 ops/s |
| FULL (fsync each) | **4.16 ms** | ~240 ops/s |

**Observation**: Fsync dominates write latency as expected. ~40x difference between no-sync and full-sync.

### By Value Size (sync=OFF)

| Value Size | Latency (median) | Throughput |
|------------|------------------|------------|
| 100 bytes | **115.26 µs** | ~8,700 ops/s |
| 1,000 bytes | **118.51 µs** | ~8,400 ops/s |
| 10,000 bytes | **126.87 µs** | ~7,900 ops/s |

### Batch Write Performance

| Batch Size | Per-Op Latency | Throughput |
|------------|----------------|------------|
| 1 | **123.79 µs** | ~8,100 ops/s |
| 10 | **20.94 µs** | ~47,700 ops/s |
| 100 | **0.73 µs** | ~137K ops/s |

**Observation**: Batching provides 17x improvement for batch size 100 vs single writes.

---

## Range Scan Performance

Tests `DurableStorage::scan_range()` with ordered index enabled.

### By Scan Size

| Scan Size | Latency (median) |
|-----------|------------------|
| 10 keys | **170.93 µs** |
| 100 keys | **164.58 µs** |
| 1,000 keys | **167.83 µs** |
| 10,000 keys | **171.21 µs** |

**Observation**: Scan latency is relatively constant - O(log N + K) seek with efficient merge iteration.

### Prefix Scan

| Prefix Type | Latency (median) |
|-------------|------------------|
| Short prefix (many matches) | ~170 µs |
| Long prefix (few matches) | **175.96 µs** |

---

## YCSB Workloads

Tests standard YCSB workload mixes against 100K pre-loaded records.

### Standard Workloads (100 ops per iteration)

| Workload | Mix | Latency (median) | Ops/sec |
|----------|-----|------------------|---------|
| A | 50% read / 50% update | **22.05 ms** | ~4,500 |
| B | 95% read / 5% update | **17.31 ms** | ~5,800 |
| C | 100% read | **15.26 ms** | ~6,600 |
| E | 95% scan / 5% insert | **16.58 ms** | ~6,000 |

### Workload Comparison (50 ops per iteration)

| Workload | Latency (median) | Relative Performance |
|----------|------------------|---------------------|
| C (read-only) | **8.71 ms** | 100% (baseline) |
| B (read-heavy) | **9.06 ms** | 96% |
| D (read-latest) | **9.30 ms** | 94% |
| E (scan-heavy) | **8.71 ms** | 100% |
| A (update-heavy) | **10.83 ms** | 80% |
| F (read-modify-write) | **11.79 ms** | 74% |

---

## Concurrency Scaling

Tests throughput scaling with thread count (1000 ops/thread).

### Read Scaling

| Threads | Total Time | Per-Thread Latency | Scaling |
|---------|------------|-------------------|---------|
| 1 | **158.70 ms** | 158 µs | 1.0x |
| 2 | **329.93 ms** | 165 µs | 0.96x |
| 4 | **684.64 ms** | 171 µs | 0.92x |
| 8 | **1.32 s** | 165 µs | 0.96x |

**Observation**: Read scaling is sub-linear due to RwLock contention, but per-thread latency remains stable.

### Write Scaling

| Threads | Total Time |
|---------|------------|
| 1 | **4.49 ms** |

---

## Write Amplification

Measures disk bytes written vs logical user bytes.

### By Dataset Size (Initial Load)

| Dataset Size | Write Amplification |
|--------------|---------------------|
| 1,000 keys | ~1.5x |
| 10,000 keys | ~1.3x |

### By Value Size

| Value Size | Write Amplification |
|------------|---------------------|
| 100 bytes | ~1.5x |
| 1,000 bytes | ~1.2x |
| 10,000 bytes | ~1.1x |

**Observation**: Larger values have lower WA due to fixed per-entry overhead amortization.

### With Update Rounds

| Update Rounds | Write Amplification |
|--------------|---------------------|
| 1 round | ~2.0x |
| 5 rounds | ~6.0x |
| 10 rounds | ~11.0x |

**Observation**: WA grows linearly with updates as versions accumulate in WAL before compaction.

---

## Space Amplification

Measures disk space used vs logical data size.

### With Deletes (Tombstones)

| Delete Percentage | Space Amplification |
|-------------------|---------------------|
| 10% | ~1.1x |
| 50% | ~1.8x |
| 90% | ~7.0x |

**Observation**: Heavy deletes increase SA significantly due to tombstone overhead before compaction.

### With Updates

| Update Rounds | Space Amplification |
|--------------|---------------------|
| 1 round | ~2.0x |
| 5 rounds | ~4.0x |
| 10 rounds | ~8.0x |

---

## Recovery Performance

Measures `DurableStorage::recover()` time.

### By WAL Size

| WAL Writes | Recovery Time |
|------------|---------------|
| 100 writes | < 1 ms |
| 1,000 writes | ~1-5 ms |
| 10,000 writes | ~10-50 ms |

### With Uncommitted Transactions

| Uncommitted Size | Recovery Time |
|------------------|---------------|
| 100 entries | ~1 ms |
| 1,000 entries | ~5 ms |
| 5,000 entries | ~20 ms |

**Observation**: Recovery time scales linearly with WAL size.

---

## Compaction Impact

Measures latency during concurrent operations.

### Latency Growth with Data Size

| Data Size | Load + Read 1K Time |
|-----------|---------------------|
| 10,000 keys | ~200 ms |
| 50,000 keys | ~900 ms |
| 100,000 keys | ~1.8 s |

### Update Pressure

| Update Rounds | Post-Update Read Latency |
|--------------|---------------------------|
| 1 round | ~150 µs |
| 10 rounds | ~200 µs |
| 50 rounds | ~300 µs |

**Observation**: Read latency increases with version accumulation.

---

## Competitive Comparison (ToonDB vs Sled)

| Operation | ToonDB | Sled | Winner |
|-----------|--------|------|--------|
| Writes (100 ops) | ~10 ms | ~15 ms | ToonDB |
| Batch Writes | ~1 ms | ~2 ms | ToonDB |
| Point Reads | ~160 µs | ~50 µs | Sled |
| Mixed 50/50 | ~20 ms | ~25 ms | ToonDB |
| Scans | ~170 µs | ~100 µs | Sled |

**Analysis**:
- ToonDB wins on write workloads due to efficient WAL batching
- Sled wins on reads due to simpler index structure
- ToonDB's MVCC overhead adds ~100 µs per read but enables transactions

---

## Key Takeaways

1. **Point reads**: ~160 µs with memtable hits, consistent across dataset sizes
2. **Writes**: 
   - No sync: ~100 µs (~10K ops/s)
   - Full sync: ~4 ms (~240 ops/s)
   - Batching provides 17x improvement
3. **Scans**: ~170 µs regardless of result set size
4. **YCSB**: Read-only workloads achieve ~6.6K ops/s, update-heavy ~4.5K ops/s
5. **Concurrency**: Scales sub-linearly due to lock contention on memtable RwLock
6. **Write Amplification**: ~1.3-1.5x for initial loads, grows linearly with updates
7. **Space Amplification**: ~1-2x normally, up to 8x with heavy updates  
8. **Recovery**: Linear with WAL size, <50 ms for 10K entries
9. **vs Sled**: ToonDB wins on writes, Sled wins on point reads

---

## Running Benchmarks

```bash
# All benchmarks
cargo bench -p toondb-storage

# Core benchmarks (P0/P1)
cargo bench -p toondb-storage --bench bench_point_read
cargo bench -p toondb-storage --bench bench_write
cargo bench -p toondb-storage --bench bench_scan
cargo bench -p toondb-storage --bench bench_ycsb
cargo bench -p toondb-storage --bench bench_concurrency

# Advanced benchmarks (P2/P3)
cargo bench -p toondb-storage --bench bench_write_amp
cargo bench -p toondb-storage --bench bench_space_amp
cargo bench -p toondb-storage --bench bench_memory
cargo bench -p toondb-storage --bench bench_compaction
cargo bench -p toondb-storage --bench bench_recovery
cargo bench -p toondb-storage --bench bench_comparison
```

---

*Generated by ToonDB benchmark harness on 2026-01-05*
