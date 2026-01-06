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

//! Space Amplification Measurement Benchmark
//!
//! Measures disk space used vs logical data size:
//!
//! ```text
//! Space Amplification = Disk Space Used / Logical Data Size
//! ```
//!
//! Tracks:
//! - After initial load
//! - After delete workload (tombstones)
//! - Over time with continuous updates
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_space_amp`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use measurement_harness::{generate_key, generate_value};
use std::fs;
use std::path::Path;
use tempfile::TempDir;
use toondb_storage::{DurableStorage, TransactionMode};

/// Calculate total size of files in a directory recursively
fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Ok(meta) = fs::metadata(&path) {
                    total += meta.len();
                }
            } else if path.is_dir() {
                total += dir_size(&path);
            }
        }
    }
    total
}

/// Measure space amplification for initial load
fn bench_space_amp_initial(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_amp/initial");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;

    for dataset_size in [1_000, 10_000, 50_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}keys", dataset_size)),
            &dataset_size,
            |b, &dataset_size| {
                b.iter_custom(|iters| {
                    let mut total_sa = 0.0;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        storage.set_sync_mode(0);

                        // Load data
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, 42);
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        // Calculate logical size
                        let logical_size = (dataset_size * (key_size + value_size)) as u64;
                        let disk_size = dir_size(temp_dir.path());

                        let sa = if logical_size > 0 {
                            disk_size as f64 / logical_size as f64
                        } else {
                            1.0
                        };
                        total_sa += sa;
                    }

                    std::time::Duration::from_micros((total_sa / iters as f64 * 1000.0) as u64)
                });
            },
        );
    }

    group.finish();
}

/// Measure space amplification with deletes (tombstones)
fn bench_space_amp_with_deletes(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_amp/deletes");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    for delete_pct in [10, 50, 90] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}pct", delete_pct)),
            &delete_pct,
            |b, &delete_pct| {
                b.iter_custom(|iters| {
                    let mut total_sa = 0.0;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        storage.set_sync_mode(0);

                        // Load data
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, 42);
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        // Delete some keys
                        let delete_count = (dataset_size * delete_pct) / 100;
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..delete_count {
                            let key = generate_key(i, key_size);
                            storage.delete(txn_id, key).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        // Logical size = remaining live data
                        let remaining = dataset_size - delete_count;
                        let logical_size = (remaining * (key_size + value_size)) as u64;
                        let disk_size = dir_size(temp_dir.path());

                        let sa = if logical_size > 0 {
                            disk_size as f64 / logical_size as f64
                        } else {
                            disk_size as f64 // All deleted, show raw disk use
                        };
                        total_sa += sa;
                    }

                    std::time::Duration::from_micros((total_sa / iters as f64 * 1000.0) as u64)
                });
            },
        );
    }

    group.finish();
}

/// Measure space amplification with updates (overwrites)
fn bench_space_amp_with_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_amp/updates");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    for update_rounds in [1, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}rounds", update_rounds)),
            &update_rounds,
            |b, &update_rounds| {
                b.iter_custom(|iters| {
                    let mut total_sa = 0.0;

                    for _ in 0..iters {
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

                        // Update rounds
                        for round in 0..update_rounds {
                            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                            for i in 0..dataset_size {
                                let key = generate_key(i, key_size);
                                let value = generate_value(i, value_size, round as u64 + 100);
                                storage.write(txn_id, key, value).unwrap();
                            }
                            storage.commit(txn_id).unwrap();
                        }

                        // Logical size = current live data only
                        let logical_size = (dataset_size * (key_size + value_size)) as u64;
                        let disk_size = dir_size(temp_dir.path());

                        let sa = if logical_size > 0 {
                            disk_size as f64 / logical_size as f64
                        } else {
                            1.0
                        };
                        total_sa += sa;
                    }

                    std::time::Duration::from_micros((total_sa / iters as f64 * 1000.0) as u64)
                });
            },
        );
    }

    group.finish();
}

/// Print space amplification report
fn bench_space_amp_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_amp/report");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    group.bench_function("full_report", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let storage = DurableStorage::open(temp_dir.path()).unwrap();
            storage.set_sync_mode(0);

            // Load data
            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
            for i in 0..dataset_size {
                let key = generate_key(i, key_size);
                let value = generate_value(i, value_size, 42);
                storage.write(txn_id, key, value).unwrap();
            }
            storage.commit(txn_id).unwrap();

            let logical_size = (dataset_size * (key_size + value_size)) as u64;
            let disk_size = dir_size(temp_dir.path());

            let sa = if logical_size > 0 {
                disk_size as f64 / logical_size as f64
            } else {
                1.0
            };

            black_box(sa)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_space_amp_initial,
    bench_space_amp_with_deletes,
    bench_space_amp_with_updates,
    bench_space_amp_report,
);

criterion_main!(benches);
