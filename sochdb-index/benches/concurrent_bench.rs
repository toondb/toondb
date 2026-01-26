// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Concurrency and Lock Contention Benchmarks
//!
//! Measures concurrent throughput with multiple threads to identify lock bottlenecks

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use std::sync::Arc;
use std::thread;
use sochdb_index::hnsw::{HnswConfig, HnswIndex};

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

/// Benchmark concurrent inserts with different thread counts
fn bench_concurrent_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_insert");

    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let config = HnswConfig::default();
                    let index = Arc::new(HnswIndex::new(128, config));

                    let mut handles = vec![];
                    let inserts_per_thread = 1000;

                    for thread_id in 0..num_threads {
                        let index_clone = Arc::clone(&index);

                        let handle = thread::spawn(move || {
                            for i in 0..inserts_per_thread {
                                let id = (thread_id * inserts_per_thread + i) as u128;
                                let vec = generate_random_vector(128);
                                index_clone.insert(id, vec).unwrap();
                            }
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    black_box(index);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent searches (read-only workload)
fn bench_concurrent_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_search");

    // Pre-populate index
    let config = HnswConfig::default();
    let index = Arc::new(HnswIndex::new(128, config));

    for i in 0..10_000 {
        let vec = generate_random_vector(128);
        index.insert(i as u128, vec).unwrap();
    }

    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let mut handles = vec![];
                    let searches_per_thread = 100;

                    for _ in 0..num_threads {
                        let index_clone = Arc::clone(&index);

                        let handle = thread::spawn(move || {
                            for _ in 0..searches_per_thread {
                                let query = generate_random_vector(128);
                                let results = index_clone.search(&query, 10).unwrap();
                                black_box(results);
                            }
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed concurrent workload (80% reads, 20% writes)
fn bench_concurrent_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_mixed");

    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let config = HnswConfig::default();
                    let index = Arc::new(HnswIndex::new(128, config));

                    // Pre-populate
                    for i in 0..5_000 {
                        let vec = generate_random_vector(128);
                        index.insert(i as u128, vec).unwrap();
                    }

                    let mut handles = vec![];
                    let next_id = Arc::new(std::sync::atomic::AtomicU64::new(5_000));

                    // Reader threads (80% of threads, rounded up)
                    let num_readers = (num_threads * 4) / 5 + 1;
                    for _ in 0..num_readers {
                        let index_clone = Arc::clone(&index);

                        let handle = thread::spawn(move || {
                            for _ in 0..50 {
                                let query = generate_random_vector(128);
                                let results = index_clone.search(&query, 10).unwrap();
                                black_box(results);
                            }
                        });

                        handles.push(handle);
                    }

                    // Writer threads (20% of threads)
                    let num_writers = num_threads - num_readers;
                    for _ in 0..num_writers {
                        let index_clone = Arc::clone(&index);
                        let next_id_clone = Arc::clone(&next_id);

                        let handle = thread::spawn(move || {
                            for _ in 0..50 {
                                let id =
                                    next_id_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                let vec = generate_random_vector(128);
                                index_clone.insert(id as u128, vec).unwrap();
                            }
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    black_box(index);
                });
            },
        );
    }

    group.finish();
}

/// Measure throughput scaling with thread count
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.sample_size(10); // Fewer samples for faster benchmarking

    let config = HnswConfig::default();
    let index = Arc::new(HnswIndex::new(128, config));

    // Pre-populate
    for i in 0..10_000 {
        let vec = generate_random_vector(128);
        index.insert(i as u128, vec).unwrap();
    }

    for num_threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let mut handles = vec![];
                    let ops_per_thread = 1000;

                    for _ in 0..num_threads {
                        let index_clone = Arc::clone(&index);

                        let handle = thread::spawn(move || {
                            for _ in 0..ops_per_thread {
                                let query = generate_random_vector(128);
                                let results = index_clone.search(&query, 10).unwrap();
                                black_box(results);
                            }
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_concurrent_insert,
    bench_concurrent_search,
    bench_concurrent_mixed,
    bench_throughput_scaling
);
criterion_main!(benches);
