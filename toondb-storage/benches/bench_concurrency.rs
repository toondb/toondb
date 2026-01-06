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

//! Multi-threaded Scalability Benchmark
//!
//! Measures throughput scaling with thread count:
//!
//! | Threads | Read Throughput | Write Throughput | Mixed Throughput |
//! |---------|-----------------|------------------|------------------|
//! | 1 | ? | ? | ? |
//! | 2 | ? | ? | ? |
//! | 4 | ? | ? | ? |
//! | 8 | ? | ? | ? |
//! | 16 | ? | ? | ? |
//!
//! Tests lock contention on `parking_lot::RwLock` for memtable, levels, etc.
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_concurrency`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{
    BenchConfig, Distribution, DurableStorageHarness, ZipfianGenerator,
    generate_key, generate_value, next_key_index,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use toondb_storage::{DurableStorage, TransactionMode};

/// Wrapper to share DurableStorage across threads
struct SharedStorage {
    storage: Arc<DurableStorage>,
}

impl SharedStorage {
    fn new(storage: DurableStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    fn clone_storage(&self) -> Arc<DurableStorage> {
        Arc::clone(&self.storage)
    }
}

/// Run concurrent reads
fn run_concurrent_reads(
    storage: Arc<DurableStorage>,
    dataset_size: usize,
    key_size: usize,
    ops_per_thread: usize,
    thread_id: usize,
) -> usize {
    let mut rng = rand::thread_rng();
    let zipf = ZipfianGenerator::new(dataset_size, 0.99);
    let mut seq = thread_id * ops_per_thread;
    let mut completed = 0;

    for _ in 0..ops_per_thread {
        let idx = next_key_index(
            &mut rng,
            Distribution::Zipfian,
            Some(&zipf),
            dataset_size,
            &mut seq,
        );
        let key = generate_key(idx, key_size);

        let txn_id = storage
            .begin_with_mode(TransactionMode::ReadOnly)
            .unwrap();
        let _ = storage.read(txn_id, &key);
        storage.commit(txn_id).unwrap();
        completed += 1;
    }

    completed
}

/// Run concurrent writes
fn run_concurrent_writes(
    storage: Arc<DurableStorage>,
    base_key: usize,
    key_size: usize,
    value_size: usize,
    ops_per_thread: usize,
) -> usize {
    let mut completed = 0;

    for i in 0..ops_per_thread {
        let key = generate_key(base_key + i, key_size);
        let value = generate_value(base_key + i, value_size, 42);

        let txn_id = storage
            .begin_with_mode(TransactionMode::WriteOnly)
            .unwrap();
        storage.write(txn_id, key, value).unwrap();
        storage.commit(txn_id).unwrap();
        completed += 1;
    }

    completed
}

/// Run mixed read/write workload
fn run_concurrent_mixed(
    storage: Arc<DurableStorage>,
    dataset_size: usize,
    base_key: usize,
    key_size: usize,
    value_size: usize,
    ops_per_thread: usize,
    read_ratio: f64,
    thread_id: usize,
) -> usize {
    let mut rng = rand::thread_rng();
    let zipf = ZipfianGenerator::new(dataset_size, 0.99);
    let mut seq = thread_id * ops_per_thread;
    let mut completed = 0;
    let mut write_counter = base_key;

    for i in 0..ops_per_thread {
        let do_read = (i as f64 / ops_per_thread as f64) < read_ratio;

        if do_read {
            let idx = next_key_index(
                &mut rng,
                Distribution::Zipfian,
                Some(&zipf),
                dataset_size,
                &mut seq,
            );
            let key = generate_key(idx, key_size);

            let txn_id = storage
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let _ = storage.read(txn_id, &key);
            storage.commit(txn_id).unwrap();
        } else {
            let key = generate_key(write_counter, key_size);
            let value = generate_value(write_counter, value_size, 42);
            write_counter += 1;

            let txn_id = storage
                .begin_with_mode(TransactionMode::WriteOnly)
                .unwrap();
            storage.write(txn_id, key, value).unwrap();
            storage.commit(txn_id).unwrap();
        }
        completed += 1;
    }

    completed
}

/// Benchmark read scalability with thread count
fn bench_read_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency/read");
    group.sample_size(10);

    let dataset_size = 100_000;
    let ops_per_thread = 1000;
    let key_size = 16;

    // Create shared storage
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let storage = DurableStorage::open(temp_dir.path()).expect("Failed to open storage");

    // Preload data
    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
    for i in 0..dataset_size {
        let key = generate_key(i, key_size);
        let value = generate_value(i, 100, 42);
        storage.write(txn_id, key, value).unwrap();
    }
    storage.commit(txn_id).unwrap();

    let shared = SharedStorage::new(storage);

    for num_threads in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}t", num_threads)),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|t| {
                            let storage = shared.clone_storage();
                            thread::spawn(move || {
                                run_concurrent_reads(storage, dataset_size, key_size, ops_per_thread, t)
                            })
                        })
                        .collect();

                    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark write scalability with thread count
fn bench_write_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency/write");
    group.sample_size(10);

    let ops_per_thread = 500;
    let key_size = 16;
    let value_size = 100;

    for num_threads in [1, 2, 4, 8] {
        // Fresh storage for each thread count to avoid accumulation
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let storage = DurableStorage::open(temp_dir.path()).expect("Failed to open storage");
        storage.set_sync_mode(0); // No fsync for fair comparison
        let shared = SharedStorage::new(storage);
        let write_counter = Arc::new(AtomicUsize::new(0));

        group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}t", num_threads)),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let storage = shared.clone_storage();
                            let counter = Arc::clone(&write_counter);
                            let base = counter.fetch_add(ops_per_thread, Ordering::SeqCst);
                            thread::spawn(move || {
                                run_concurrent_writes(storage, base, key_size, value_size, ops_per_thread)
                            })
                        })
                        .collect();

                    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed read/write scalability
fn bench_mixed_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency/mixed");
    group.sample_size(10);

    let dataset_size = 100_000;
    let ops_per_thread = 500;
    let key_size = 16;
    let value_size = 100;

    // Create shared storage with preloaded data
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let storage = DurableStorage::open(temp_dir.path()).expect("Failed to open storage");

    // Preload
    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
    for i in 0..dataset_size {
        let key = generate_key(i, key_size);
        let value = generate_value(i, value_size, 42);
        storage.write(txn_id, key, value).unwrap();
    }
    storage.commit(txn_id).unwrap();

    let shared = SharedStorage::new(storage);
    let write_counter = Arc::new(AtomicUsize::new(dataset_size));

    for num_threads in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}t", num_threads)),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|t| {
                            let storage = shared.clone_storage();
                            let counter = Arc::clone(&write_counter);
                            let base = counter.fetch_add(ops_per_thread, Ordering::SeqCst);
                            thread::spawn(move || {
                                run_concurrent_mixed(
                                    storage,
                                    dataset_size,
                                    base,
                                    key_size,
                                    value_size,
                                    ops_per_thread,
                                    0.5, // 50% reads
                                    t,
                                )
                            })
                        })
                        .collect();

                    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

/// Direct throughput measurement (not using criterion)
fn bench_direct_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency/throughput");
    group.sample_size(5);

    let dataset_size = 50_000;
    let ops_per_thread = 2000;
    let key_size = 16;

    for num_threads in [1, 2, 4, 8, 16] {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let storage = DurableStorage::open(temp_dir.path()).expect("Failed to open storage");

        // Preload
        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
        for i in 0..dataset_size {
            let key = generate_key(i, key_size);
            let value = generate_value(i, 100, 42);
            storage.write(txn_id, key, value).unwrap();
        }
        storage.commit(txn_id).unwrap();

        let shared = SharedStorage::new(storage);

        group.bench_function(format!("{}t_reads", num_threads), |b| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let start = Instant::now();

                    let handles: Vec<_> = (0..num_threads)
                        .map(|t| {
                            let storage = shared.clone_storage();
                            thread::spawn(move || {
                                run_concurrent_reads(storage, dataset_size, key_size, ops_per_thread, t)
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    total_duration += start.elapsed();
                }

                total_duration
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_read_scalability,
    bench_write_scalability,
    bench_mixed_scalability,
    bench_direct_throughput,
);

criterion_main!(benches);
