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

//! HNSW Index Benchmarks
//!
//! Measures insertion throughput, search latency, and mixed workload performance

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use sochdb_index::hnsw::{HnswConfig, HnswIndex};

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn generate_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| generate_random_vector(dim)).collect()
}

/// Benchmark insertion throughput at different scales
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");

    for size in [1_000, 10_000, 50_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_test_vectors(size, 128);

            b.iter(|| {
                let config = HnswConfig::default();
                let index = HnswIndex::new(128, config);

                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u128, vec.clone()).unwrap();
                }

                black_box(index);
            });
        });
    }

    group.finish();
}

/// Benchmark search latency with pre-populated index
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    // Pre-populate index with 100K vectors
    let config = HnswConfig::default();
    let index = HnswIndex::new(128, config);
    let vectors = generate_test_vectors(100_000, 128);

    for (i, vec) in vectors.iter().enumerate() {
        index.insert(i as u128, vec.clone()).unwrap();
    }

    // Benchmark search with different k values
    for k in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            let query = generate_random_vector(128);

            b.iter(|| {
                let results = index.search(&query, k).unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark search latency at different index sizes
fn bench_search_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_scaling");

    for size in [1_000, 10_000, 100_000] {
        let config = HnswConfig::default();
        let index = HnswIndex::new(128, config);
        let vectors = generate_test_vectors(size, 128);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(i as u128, vec.clone()).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let query = generate_random_vector(128);

            b.iter(|| {
                let results = index.search(&query, 10).unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark mixed workload (80% reads, 20% writes)
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_mixed");

    let config = HnswConfig::default();
    let index = HnswIndex::new(128, config);

    // Pre-populate with 50K vectors
    let vectors = generate_test_vectors(50_000, 128);
    for (i, vec) in vectors.iter().enumerate() {
        index.insert(i as u128, vec.clone()).unwrap();
    }

    group.bench_function("mixed_80_20", |b| {
        let mut next_id = 50_000u128;

        b.iter(|| {
            // 80% reads
            for _ in 0..80 {
                let query = generate_random_vector(128);
                let results = index.search(&query, 10).unwrap();
                black_box(results);
            }

            // 20% writes
            for _ in 0..20 {
                let vec = generate_random_vector(128);
                index.insert(next_id, vec).unwrap();
                next_id += 1;
            }
        });
    });

    group.finish();
}

/// Benchmark insert throughput for different vector dimensions
fn bench_insert_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert_dims");

    for dim in [64, 128, 768, 1536] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let vectors = generate_test_vectors(1_000, dim);

            b.iter(|| {
                let config = HnswConfig::default();
                let index = HnswIndex::new(dim, config);

                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u128, vec.clone()).unwrap();
                }

                black_box(index);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_search,
    bench_search_scaling,
    bench_mixed_workload,
    bench_insert_dimensions
);
criterion_main!(benches);
