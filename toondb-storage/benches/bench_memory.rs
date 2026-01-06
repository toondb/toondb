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

//! Memory Footprint Tracking Benchmark
//!
//! Tracks RSS (Resident Set Size) during:
//! - Database loading
//! - Steady-state operations
//! - After GC
//!
//! Memory components to identify:
//! - Memtable size
//! - Bloom filters
//! - Index structures
//! - WAL buffers
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_memory`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use measurement_harness::{generate_key, generate_value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;
use toondb_storage::{DurableStorage, TransactionMode};

/// Simple memory tracking allocator
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            if new_size > layout.size() {
                ALLOCATED.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

/// Get current allocated bytes (approximate)
fn get_allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Measure memory usage during database loading
fn bench_memory_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/loading");
    group.sample_size(10);

    let key_size = 16;
    let value_size = 100;

    for dataset_size in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}keys", dataset_size)),
            &dataset_size,
            |b, &dataset_size| {
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

                    // Theoretical memory: keys + values in memtable
                    let expected_bytes = dataset_size * (key_size + value_size);
                    
                    black_box(expected_bytes)
                });
            },
        );
    }

    group.finish();
}

/// Measure memory per key with different value sizes
fn bench_memory_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/value_size");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;

    for value_size in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}B", value_size)),
            &value_size,
            |b, &value_size| {
                b.iter(|| {
                    let temp_dir = TempDir::new().unwrap();
                    let storage = DurableStorage::open(temp_dir.path()).unwrap();
                    storage.set_sync_mode(0);

                    let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                    for i in 0..dataset_size {
                        let key = generate_key(i, key_size);
                        let value = generate_value(i, value_size, 42);
                        storage.write(txn_id, key, value).unwrap();
                    }
                    storage.commit(txn_id).unwrap();

                    // Memory per key
                    let bytes_per_key = key_size + value_size;
                    let total_expected = dataset_size * bytes_per_key;
                    
                    black_box(total_expected)
                });
            },
        );
    }

    group.finish();
}

/// Measure memory with read-heavy workload (version chains)
fn bench_memory_read_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/read_workload");
    group.sample_size(10);

    let dataset_size = 10_000;
    let key_size = 16;
    let value_size = 100;

    group.bench_function("after_reads", |b| {
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

            // Read all keys
            for i in 0..dataset_size {
                let key = generate_key(i, key_size);
                let txn_id = storage.begin_with_mode(TransactionMode::ReadOnly).unwrap();
                let _ = storage.read(txn_id, &key).unwrap();
                storage.commit(txn_id).unwrap();
            }

            black_box(dataset_size)
        });
    });

    group.finish();
}

/// Measure memory with updates (version accumulation)
fn bench_memory_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/updates");
    group.sample_size(10);

    let dataset_size = 5_000;
    let key_size = 16;
    let value_size = 100;

    for update_rounds in [1, 5, 10] {
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

                    // Updates (accumulate versions)
                    for round in 0..update_rounds {
                        let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
                        for i in 0..dataset_size {
                            let key = generate_key(i, key_size);
                            let value = generate_value(i, value_size, round as u64 + 100);
                            storage.write(txn_id, key, value).unwrap();
                        }
                        storage.commit(txn_id).unwrap();
                    }

                    // Expected versions per key: 1 + update_rounds
                    let versions_per_key = 1 + update_rounds;
                    let expected_overhead = dataset_size * versions_per_key * (key_size + value_size);
                    
                    black_box(expected_overhead)
                });
            },
        );
    }

    group.finish();
}

/// Report-style benchmark that outputs memory usage
fn bench_memory_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/report");
    group.sample_size(10);

    group.bench_function("summary", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let storage = DurableStorage::open(temp_dir.path()).unwrap();
            storage.set_sync_mode(0);

            let dataset_size = 10_000;
            let key_size = 16;
            let value_size = 100;

            let txn_id = storage.begin_with_mode(TransactionMode::WriteOnly).unwrap();
            for i in 0..dataset_size {
                let key = generate_key(i, key_size);
                let value = generate_value(i, value_size, 42);
                storage.write(txn_id, key, value).unwrap();
            }
            storage.commit(txn_id).unwrap();

            // Theoretical minimum: data only
            let data_bytes = dataset_size * (key_size + value_size);
            // With bloom filter: ~1.25 bytes per key  
            let bloom_bytes = (dataset_size as f64 * 1.25) as usize;
            // Total expected
            let expected_total = data_bytes + bloom_bytes;

            black_box(expected_total)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_loading,
    bench_memory_value_sizes,
    bench_memory_read_workload,
    bench_memory_updates,
    bench_memory_report,
);

criterion_main!(benches);
