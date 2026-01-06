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

//! Compaction Performance Impact Benchmark
//!
//! Measures how compaction affects foreground operations:
//!
//! - Read latency P99 during compaction
//! - Write latency P99 during compaction
//! - Throughput degradation
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_compaction`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{generate_key, generate_value};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use toondb_storage::{DurableStorage, TransactionMode};

/// Measure read latency while accumulating data (potential compaction trigger)
fn bench_read_during_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction/read_during_writes");
    group.sample_size(10);

    let initial_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    group.throughput(Throughput::Elements(1));
    group.bench_function("baseline", |b| {
        let temp_dir = TempDir::new().unwrap();
        let storage = DurableStorage::open(temp_dir.path()).unwrap();
        storage.set_sync_mode(0);

        // Preload
        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
        for i in 0..initial_size {
            let key = generate_key(i, key_size);
            let value = generate_value(i, value_size, 42);
            storage.write(txn_id, key, value).unwrap();
        }
        storage.commit(txn_id).unwrap();

        let mut read_idx = 0;
        let mut write_idx = initial_size;

        b.iter(|| {
            // Background write (simulating ongoing activity)
            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
            let key = generate_key(write_idx, key_size);
            let value = generate_value(write_idx, value_size, 42);
            storage.write(txn_id, key, value).unwrap();
            storage.commit(txn_id).unwrap();
            write_idx += 1;

            // Foreground read (what we're measuring)
            let key = generate_key(read_idx % initial_size, key_size);
            let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
            let result = storage.read(txn_id, &key).unwrap();
            storage.commit(txn_id).unwrap();
            read_idx += 1;

            black_box(result)
        });
    });

    group.finish();
}

/// Measure write latency during heavy read activity
fn bench_write_during_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction/write_during_reads");
    group.sample_size(10);

    let initial_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    group.throughput(Throughput::Elements(1));
    group.bench_function("baseline", |b| {
        let temp_dir = TempDir::new().unwrap();
        let storage = DurableStorage::open(temp_dir.path()).unwrap();
        storage.set_sync_mode(0);

        // Preload
        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
        for i in 0..initial_size {
            let key = generate_key(i, key_size);
            let value = generate_value(i, value_size, 42);
            storage.write(txn_id, key, value).unwrap();
        }
        storage.commit(txn_id).unwrap();

        let mut read_idx = 0;
        let mut write_idx = initial_size;

        b.iter(|| {
            // Background reads (simulating ongoing activity)
            for _ in 0..10 {
                let key = generate_key(read_idx % initial_size, key_size);
                let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                let _ = storage.read(txn_id, &key);
                storage.commit(txn_id).unwrap();
                read_idx += 1;
            }

            // Foreground write (what we're measuring)
            let key = generate_key(write_idx, key_size);
            let value = generate_value(write_idx, value_size, 42);
            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
            storage.write(txn_id, key, value).unwrap();
            storage.commit(txn_id).unwrap();
            write_idx += 1;

            black_box(())
        });
    });

    group.finish();
}

/// Measure latency percentiles with growing data (memtable flushes)
fn bench_latency_with_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction/latency_growth");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;

    for data_size in [10_000, 50_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}keys", data_size)),
            &data_size,
            |b, &data_size| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        storage.set_sync_mode(0);

                        // Load and measure
                        let start = Instant::now();

                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..data_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, 42);
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        // Read some data after load
                        for i in 0..1000 {
                            let key = generate_key(i % data_size, key_size);
                            let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                            let _ = storage.read(txn_id, &key);
                            storage.commit(txn_id).unwrap();
                        }

                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Measure steady-state performance after data accumulation
fn bench_steady_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction/steady_state");
    group.sample_size(10);

    let initial_size = 50_000;
    let key_size = 16;
    let value_size = 100;

    // Load once, then benchmark
    let temp_dir = TempDir::new().unwrap();
    let storage = DurableStorage::open(temp_dir.path()).unwrap();
    storage.set_sync_mode(0);

    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
    for i in 0..initial_size {
        let key = generate_key(i, key_size);
        let value = generate_value(i, value_size, 42);
        storage.write(txn_id, key, value).unwrap();
    }
    storage.commit(txn_id).unwrap();

    // Measure read performance in steady state
    group.throughput(Throughput::Elements(100));
    group.bench_function("read_100", |b| {
        let mut idx = 0;
        b.iter(|| {
            for _ in 0..100 {
                let key = generate_key(idx % initial_size, key_size);
                let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                let result = storage.read(txn_id, &key).unwrap();
                storage.commit(txn_id).unwrap();
                idx += 1;
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Measure impact of update workload (versioning pressure)
fn bench_update_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction/update_pressure");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    for update_rounds in [1, 10, 50] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}rounds", update_rounds)),
            &update_rounds,
            |b, &update_rounds| {
                b.iter(|| {
                    let temp_dir = TempDir::new().unwrap();
                    let storage = DurableStorage::open(temp_dir.path()).unwrap();
                    storage.set_sync_mode(0);

                    // Initial load
                    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                    for i in 0..dataset_size {
                        let key = generate_key(i, key_size);
                        let value = generate_value(i, value_size, 42);
                        storage.write(txn_id, key, value).unwrap();
                    }
                    storage.commit(txn_id).unwrap();

                    // Update rounds (accumulate versions → pressure on compaction)
                    for round in 0..update_rounds {
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, round as u64 + 100);
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();
                    }

                    // Measure read latency after updates
                    let mut total_read_time = Duration::ZERO;
                    for i in 0..1000 {
                        let key = generate_key(i % dataset_size, key_size);
                        let start = Instant::now();
                        let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                        let _ = storage.read(txn_id, &key);
                        storage.commit(txn_id).unwrap();
                        total_read_time += start.elapsed();
                    }

                    black_box(total_read_time)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_read_during_writes,
    bench_write_during_reads,
    bench_latency_with_growth,
    bench_steady_state,
    bench_update_pressure,
);

criterion_main!(benches);
