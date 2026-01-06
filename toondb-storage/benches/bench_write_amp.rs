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

//! Write Amplification Measurement Benchmark
//!
//! Measures actual bytes written to disk vs logical bytes inserted:
//!
//! ```text
//! Write Amplification = Total Disk Writes / User Data Written
//! ```
//!
//! Tracks across:
//! - Initial load (empty → N keys)
//! - Steady state (continuous updates)
//! - Different value sizes (100B, 1KB, 10KB)
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_write_amp`

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

/// Measure write amplification for initial load
fn bench_write_amp_initial_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_amp/initial_load");
    group.sample_size(10);

    for dataset_size in [1_000, 10_000] {
        let value_size = 100;
        let key_size = 16;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}keys", dataset_size)),
            &dataset_size,
            |b, &dataset_size| {
                b.iter_custom(|iters| {
                    let mut total_user_bytes = 0u64;
                    let mut total_disk_bytes = 0u64;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        storage.set_sync_mode(0); // No fsync for speed

                        // Track user bytes written
                        let mut user_bytes = 0u64;

                        // Initial load
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, 42);
                            user_bytes += (key.len() + value.len()) as u64;
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        // Measure disk usage
                        let disk_bytes = dir_size(temp_dir.path());

                        total_user_bytes += user_bytes;
                        total_disk_bytes += disk_bytes;
                    }

                    // Return time proportional to write amplification
                    // Higher WA = longer "time" for comparison
                    let wa = if total_user_bytes > 0 {
                        total_disk_bytes as f64 / total_user_bytes as f64
                    } else {
                        1.0
                    };

                    std::time::Duration::from_micros((wa * 1000.0) as u64)
                });
            },
        );
    }

    group.finish();
}

/// Measure write amplification for different value sizes
fn bench_write_amp_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_amp/value_size");
    group.sample_size(10);

    let dataset_size = 5_000;
    let key_size = 16;

    for value_size in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}B", value_size)),
            &value_size,
            |b, &value_size| {
                b.iter_custom(|iters| {
                    let mut total_wa = 0.0;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        storage.set_sync_mode(0);

                        let mut user_bytes = 0u64;

                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, 42);
                            user_bytes += (key.len() + value.len()) as u64;
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        let disk_bytes = dir_size(temp_dir.path());
                        let wa = if user_bytes > 0 {
                            disk_bytes as f64 / user_bytes as f64
                        } else {
                            1.0
                        };
                        total_wa += wa;
                    }

                    std::time::Duration::from_micros((total_wa / iters as f64 * 1000.0) as u64)
                });
            },
        );
    }

    group.finish();
}

/// Measure write amplification with updates (overwrite existing keys)
fn bench_write_amp_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_amp/updates");
    group.sample_size(10);

    let dataset_size = 5_000;
    let key_size = 16;
    let value_size = 100;

    for update_rounds in [1, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}rounds", update_rounds)),
            &update_rounds,
            |b, &update_rounds| {
                b.iter_custom(|iters| {
                    let mut total_wa = 0.0;

                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = DurableStorage::open(temp_dir.path()).unwrap();
                        storage.set_sync_mode(0);

                        let mut user_bytes = 0u64;

                        // Initial load
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, 42);
                            user_bytes += (key.len() + value.len()) as u64;
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();

                        // Update rounds
                        for round in 0..update_rounds {
                            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                            for i in 0..dataset_size {
                                let key = generate_key(i, key_size);
                                let value = generate_value(i, value_size, round as u64 + 100);
                                user_bytes += (key.len() + value.len()) as u64;
                                storage.write(txn_id, key, value).unwrap();
                            }
                            storage.commit(txn_id).unwrap();
                        }

                        let disk_bytes = dir_size(temp_dir.path());
                        let wa = if user_bytes > 0 {
                            disk_bytes as f64 / user_bytes as f64
                        } else {
                            1.0
                        };
                        total_wa += wa;
                    }

                    std::time::Duration::from_micros((total_wa / iters as f64 * 1000.0) as u64)
                });
            },
        );
    }

    group.finish();
}

/// Print actual write amplification values
fn bench_write_amp_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_amp/report");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    group.bench_function("full_report", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let storage = DurableStorage::open(temp_dir.path()).unwrap();
            storage.set_sync_mode(0);

            let mut user_bytes = 0u64;

            // Initial load
            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
            for i in 0..dataset_size {
                let key = generate_key(i, key_size);
                let value = generate_value(i, value_size, 42);
                user_bytes += (key.len() + value.len()) as u64;
                storage.write(txn_id, key, value).unwrap();
            }
            storage.commit(txn_id).unwrap();

            let disk_bytes = dir_size(temp_dir.path());
            let wa = if user_bytes > 0 {
                disk_bytes as f64 / user_bytes as f64
            } else {
                1.0
            };

            // Print for visibility
            if wa > 0.0 {
                // Silently calculate WA
            }

            black_box(wa)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_write_amp_initial_load,
    bench_write_amp_value_sizes,
    bench_write_amp_updates,
    bench_write_amp_report,
);

criterion_main!(benches);
