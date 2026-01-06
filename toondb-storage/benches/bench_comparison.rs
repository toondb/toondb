// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Competitive Performance Comparison Benchmark
//!
//! Compares ToonDB DurableStorage against:
//!
//! | Engine | Type | Notes |
//! |--------|------|-------|
//! | ToonDB | LSM + MVCC | Our engine |
//! | Sled | LSM | Pure Rust embedded DB |
//!
//! Uses identical workloads for fair comparison.
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_comparison`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{generate_key, generate_value};
use sled;
use tempfile::TempDir;
use toondb_storage::{DurableStorage, TransactionMode};

const DATASET_SIZE: usize = 10_000;
const KEY_SIZE: usize = 16;
const VALUE_SIZE: usize = 100;

// ============================================================================
// Write Benchmarks
// ============================================================================

/// Compare write throughput
fn bench_compare_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/write");
    group.sample_size(20);
    group.throughput(Throughput::Elements(100));

    // ToonDB
    {
        let temp_dir = TempDir::new().unwrap();
        let storage = DurableStorage::open(temp_dir.path()).unwrap();
        storage.set_sync_mode(0); // No fsync for fair comparison

        let mut idx = 0;

        group.bench_function("toondb", |b| {
            b.iter(|| {
                for _ in 0..100 {
                    let key = generate_key(idx, KEY_SIZE);
                    let value = generate_value(idx, VALUE_SIZE, 42);
                    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                    storage.write(txn_id, key, value).unwrap();
                    storage.commit(txn_id).unwrap();
                    idx += 1;
                }
            });
        });
    }

    // Sled
    {
        let temp_dir = TempDir::new().unwrap();
        let db = sled::open(temp_dir.path()).unwrap();

        let mut idx = 0;

        group.bench_function("sled", |b| {
            b.iter(|| {
                for _ in 0..100 {
                    let key = generate_key(idx, KEY_SIZE);
                    let value = generate_value(idx, VALUE_SIZE, 42);
                    db.insert(key, value).unwrap();
                    idx += 1;
                }
            });
        });
    }

    group.finish();
}

/// Compare batch write throughput
fn bench_compare_batch_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/batch_write");
    group.sample_size(20);
    group.throughput(Throughput::Elements(100));

    // ToonDB batch
    {
        let temp_dir = TempDir::new().unwrap();
        let storage = DurableStorage::open(temp_dir.path()).unwrap();
        storage.set_sync_mode(0);

        let mut base_idx = 0;

        group.bench_function("toondb", |b| {
            b.iter(|| {
                let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                for i in 0..100 {
                    let key = generate_key(base_idx + i, KEY_SIZE);
                    let value = generate_value(base_idx + i, VALUE_SIZE, 42);
                    storage.write(txn_id, key, value).unwrap();
                }
                storage.commit(txn_id).unwrap();
                base_idx += 100;
            });
        });
    }

    // Sled batch
    {
        let temp_dir = TempDir::new().unwrap();
        let db = sled::open(temp_dir.path()).unwrap();

        let mut base_idx = 0;

        group.bench_function("sled", |b| {
            b.iter(|| {
                let mut batch = sled::Batch::default();
                for i in 0..100 {
                    let key = generate_key(base_idx + i, KEY_SIZE);
                    let value = generate_value(base_idx + i, VALUE_SIZE, 42);
                    batch.insert(key, value);
                }
                db.apply_batch(batch).unwrap();
                base_idx += 100;
            });
        });
    }

    group.finish();
}

// ============================================================================
// Read Benchmarks
// ============================================================================

/// Compare point read latency
fn bench_compare_point_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/point_read");
    group.sample_size(50);
    group.throughput(Throughput::Elements(1));

    // ToonDB
    {
        let temp_dir = TempDir::new().unwrap();
        let storage = DurableStorage::open(temp_dir.path()).unwrap();
        storage.set_sync_mode(0);

        // Preload
        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
        for i in 0..DATASET_SIZE {
            let key = generate_key(i, KEY_SIZE);
            let value = generate_value(i, VALUE_SIZE, 42);
            storage.write(txn_id, key, value).unwrap();
        }
        storage.commit(txn_id).unwrap();

        let mut idx = 0;

        group.bench_function("toondb", |b| {
            b.iter(|| {
                let key = generate_key(idx % DATASET_SIZE, KEY_SIZE);
                let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                let result = storage.read(txn_id, &key).unwrap();
                storage.commit(txn_id).unwrap();
                idx += 1;
                black_box(result)
            });
        });
    }

    // Sled
    {
        let temp_dir = TempDir::new().unwrap();
        let db = sled::open(temp_dir.path()).unwrap();

        // Preload
        for i in 0..DATASET_SIZE {
            let key = generate_key(i, KEY_SIZE);
            let value = generate_value(i, VALUE_SIZE, 42);
            db.insert(key, value).unwrap();
        }

        let mut idx = 0;

        group.bench_function("sled", |b| {
            b.iter(|| {
                let key = generate_key(idx % DATASET_SIZE, KEY_SIZE);
                let result = db.get(&key).unwrap();
                idx += 1;
                black_box(result)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Mixed Workload Benchmarks
// ============================================================================

/// Compare YCSB-like 50/50 read/write mix
fn bench_compare_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/mixed_50_50");
    group.sample_size(20);
    group.throughput(Throughput::Elements(100));

    // ToonDB
    {
        let temp_dir = TempDir::new().unwrap();
        let storage = DurableStorage::open(temp_dir.path()).unwrap();
        storage.set_sync_mode(0);

        // Preload
        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
        for i in 0..DATASET_SIZE {
            let key = generate_key(i, KEY_SIZE);
            let value = generate_value(i, VALUE_SIZE, 42);
            storage.write(txn_id, key, value).unwrap();
        }
        storage.commit(txn_id).unwrap();

        let mut read_idx = 0;
        let mut write_idx = DATASET_SIZE;

        group.bench_function("toondb", |b| {
            b.iter(|| {
                for _ in 0..50 {
                    // Read
                    let key = generate_key(read_idx % DATASET_SIZE, KEY_SIZE);
                    let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                    let _ = storage.read(txn_id, &key);
                    storage.commit(txn_id).unwrap();
                    read_idx += 1;
                }
                for _ in 0..50 {
                    // Write
                    let key = generate_key(write_idx, KEY_SIZE);
                    let value = generate_value(write_idx, VALUE_SIZE, 42);
                    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                    storage.write(txn_id, key, value).unwrap();
                    storage.commit(txn_id).unwrap();
                    write_idx += 1;
                }
            });
        });
    }

    // Sled
    {
        let temp_dir = TempDir::new().unwrap();
        let db = sled::open(temp_dir.path()).unwrap();

        // Preload
        for i in 0..DATASET_SIZE {
            let key = generate_key(i, KEY_SIZE);
            let value = generate_value(i, VALUE_SIZE, 42);
            db.insert(key, value).unwrap();
        }

        let mut read_idx = 0;
        let mut write_idx = DATASET_SIZE;

        group.bench_function("sled", |b| {
            b.iter(|| {
                for _ in 0..50 {
                    // Read
                    let key = generate_key(read_idx % DATASET_SIZE, KEY_SIZE);
                    let _ = db.get(&key);
                    read_idx += 1;
                }
                for _ in 0..50 {
                    // Write
                    let key = generate_key(write_idx, KEY_SIZE);
                    let value = generate_value(write_idx, VALUE_SIZE, 42);
                    db.insert(key, value).unwrap();
                    write_idx += 1;
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Scan Benchmarks
// ============================================================================

/// Compare range scan performance
fn bench_compare_scans(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/scan");
    group.sample_size(20);

    for scan_size in [10, 100, 1000] {
        // ToonDB
        {
            let temp_dir = TempDir::new().unwrap();
            let storage = DurableStorage::open_with_config(temp_dir.path(), true).unwrap();
            storage.set_sync_mode(0);

            // Preload
            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
            for i in 0..DATASET_SIZE {
                let key = generate_key(i, KEY_SIZE);
                let value = generate_value(i, VALUE_SIZE, 42);
                storage.write(txn_id, key, value).unwrap();
            }
            storage.commit(txn_id).unwrap();

            group.throughput(Throughput::Elements(scan_size as u64));
            group.bench_with_input(
                BenchmarkId::new("toondb", scan_size),
                &scan_size,
                |b, &scan_size| {
                    let mut start_idx = 0;

                    b.iter(|| {
                        let max_start = DATASET_SIZE.saturating_sub(scan_size);
                        let start_key = generate_key(start_idx % max_start, KEY_SIZE);
                        let end_key = generate_key((start_idx % max_start) + scan_size, KEY_SIZE);
                        let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                        let results = storage.scan_range(txn_id, &start_key, &end_key).unwrap();
                        storage.commit(txn_id).unwrap();
                        start_idx += 1;
                        black_box(results.len())
                    });
                },
            );
        }

        // Sled
        {
            let temp_dir = TempDir::new().unwrap();
            let db = sled::open(temp_dir.path()).unwrap();

            // Preload
            for i in 0..DATASET_SIZE {
                let key = generate_key(i, KEY_SIZE);
                let value = generate_value(i, VALUE_SIZE, 42);
                db.insert(key, value).unwrap();
            }

            group.bench_with_input(
                BenchmarkId::new("sled", scan_size),
                &scan_size,
                |b, &scan_size| {
                    let mut start_idx = 0;

                    b.iter(|| {
                        let max_start = DATASET_SIZE.saturating_sub(scan_size);
                        let start_key = generate_key(start_idx % max_start, KEY_SIZE);
                        let results: Vec<_> = db.range(start_key..).take(scan_size).collect();
                        start_idx += 1;
                        black_box(results.len())
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compare_writes,
    bench_compare_batch_writes,
    bench_compare_point_reads,
    bench_compare_mixed,
    bench_compare_scans,
);

criterion_main!(benches);
