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

//! Crash Recovery Performance Benchmark
//!
//! Measures `DurableStorage::recover()` time as a function of:
//!
//! - WAL size (uncommitted transaction size)
//! - Total data volume
//! - Number of transactions to replay
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_recovery`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use measurement_harness::{generate_key, generate_value};
use std::time::Instant;
use tempfile::TempDir;
use toondb_storage::{DurableStorage, TransactionMode};

/// Measure recovery time from clean shutdown
fn bench_recovery_clean_shutdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery/clean_shutdown");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;

    for dataset_size in [1_000, 10_000, 50_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}keys", dataset_size)),
            &dataset_size,
            |b, &dataset_size| {
                // Create and populate database
                let temp_dir = TempDir::new().unwrap();
                
                {
                    let storage = DurableStorage::open(temp_dir.path()).unwrap();
                    storage.set_sync_mode(1); // Ensure data is persisted

                    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                    for i in 0..dataset_size {
                        let key = generate_key(i, key_size);
                        let value = generate_value(i, value_size, 42);
                        storage.write(txn_id, key, value).unwrap();
                    }
                    storage.commit(txn_id).unwrap();
                }
                // Storage dropped here (clean shutdown)

                b.iter(|| {
                    // Re-open and recover
                    let storage = DurableStorage::open(temp_dir.path()).unwrap();
                    let recovery_result = storage.recover();
                    black_box(recovery_result)
                });
            },
        );
    }

    group.finish();
}

/// Measure recovery time with varying WAL size
fn bench_recovery_wal_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery/wal_size");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;

    for writes_before_recover in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}writes", writes_before_recover)),
            &writes_before_recover,
            |b, &writes_before_recover| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();

                        {
                            let storage = DurableStorage::open(temp_dir.path()).unwrap();
                            storage.set_sync_mode(1);

                            // Write data that will need to be recovered
                            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                            for i in 0..writes_before_recover {
                                let key = generate_key(i, key_size);
                                let value = generate_value(i, value_size, 42);
                                storage.write(txn_id, key, value).unwrap();
                            }
                            storage.commit(txn_id).unwrap();
                        }

                        // Measure recovery time
                        let start = Instant::now();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        let _ = storage.recover();
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Measure recovery time with multiple transactions
fn bench_recovery_multiple_txns(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery/multiple_txns");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;
    let writes_per_txn = 100;

    for num_txns in [10, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}txns", num_txns)),
            &num_txns,
            |b, &num_txns| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();

                        {
                            let storage = DurableStorage::open(temp_dir.path()).unwrap();
                            storage.set_sync_mode(1);

                            for txn in 0..num_txns {
                                let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                                for i in 0..writes_per_txn {
                                    let key_idx = txn * writes_per_txn + i;
                                    let key = generate_key(key_idx, key_size);
                                    let value = generate_value(key_idx, value_size, 42);
                                    storage.write(txn_id, key, value).unwrap();
                                }
                                storage.commit(txn_id).unwrap();
                            }
                        }

                        let start = Instant::now();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        let _ = storage.recover();
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Measure recovery time with uncommitted transactions (rollback)
fn bench_recovery_with_uncommitted(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery/uncommitted");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;
    let committed_size = 5_000;

    for uncommitted_size in [100, 1_000, 5_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}uncommitted", uncommitted_size)),
            &uncommitted_size,
            |b, &uncommitted_size| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();

                        {
                            let storage = DurableStorage::open(temp_dir.path()).unwrap();
                            storage.set_sync_mode(1);

                            // Committed data
                            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                            for i in 0..committed_size {
                                let key = generate_key(i, key_size);
                                let value = generate_value(i, value_size, 42);
                                storage.write(txn_id, key, value).unwrap();
                            }
                            storage.commit(txn_id).unwrap();

                            // Uncommitted data (will be rolled back)
                            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                            for i in 0..uncommitted_size {
                                let key = generate_key(committed_size + i, key_size);
                                let value = generate_value(committed_size + i, value_size, 99);
                                storage.write(txn_id, key, value).unwrap();
                            }
                            // Don't commit - simulate crash
                        }

                        let start = Instant::now();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        let _ = storage.recover();
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Report recovery stats
fn bench_recovery_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery/report");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;
    let dataset_size = 10_000;

    group.bench_function("full_recovery", |b| {
        b.iter_custom(|iters| {
            let mut total_time = std::time::Duration::ZERO;

            for _ in 0..iters {
                let temp_dir = TempDir::new().unwrap();

                {
                    let storage = DurableStorage::open(temp_dir.path()).unwrap();
                    storage.set_sync_mode(1);

                    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                    for i in 0..dataset_size {
                        let key = generate_key(i, key_size);
                        let value = generate_value(i, value_size, 42);
                        storage.write(txn_id, key, value).unwrap();
                    }
                    storage.commit(txn_id).unwrap();
                }

                let start = Instant::now();
                let storage = DurableStorage::open(temp_dir.path()).unwrap();
                let result = storage.recover();
                total_time += start.elapsed();

                black_box(result);
            }

            total_time
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_recovery_clean_shutdown,
    bench_recovery_wal_size,
    bench_recovery_multiple_txns,
    bench_recovery_with_uncommitted,
    bench_recovery_report,
);

criterion_main!(benches);
