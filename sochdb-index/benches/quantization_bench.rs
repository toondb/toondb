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

//! Quantization Performance Benchmarks
//!
//! Measures the impact of vector quantization (F16/BF16) on:
//! - Memory usage
//! - Search latency
//! - Insert throughput

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use sochdb_index::hnsw::{HnswConfig, HnswIndex};
use sochdb_index::vector_quantized::Precision;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn generate_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| generate_random_vector(dim)).collect()
}

/// Benchmark search latency with different quantization levels
fn bench_quantization_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_search");

    let vectors = generate_test_vectors(10_000, 128);

    for precision in [Precision::F32, Precision::F16, Precision::BF16] {
        group.bench_with_input(
            BenchmarkId::new("precision", format!("{:?}", precision)),
            &precision,
            |b, &precision| {
                let config = HnswConfig {
                    quantization_precision: Some(precision),
                    ..Default::default()
                };
                let index = HnswIndex::new(128, config);

                // Insert vectors
                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u128, vec.clone()).unwrap();
                }

                let query = generate_random_vector(128);

                b.iter(|| {
                    let results = index.search(&query, 10).unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark insert throughput with different quantization levels
fn bench_quantization_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_insert");

    let vectors = generate_test_vectors(5_000, 128);

    for precision in [Precision::F32, Precision::F16, Precision::BF16] {
        group.bench_with_input(
            BenchmarkId::new("precision", format!("{:?}", precision)),
            &precision,
            |b, &precision| {
                b.iter(|| {
                    let config = HnswConfig {
                        quantization_precision: Some(precision),
                        ..Default::default()
                    };
                    let index = HnswIndex::new(128, config);

                    for (i, vec) in vectors.iter().enumerate() {
                        index.insert(i as u128, vec.clone()).unwrap();
                    }

                    black_box(index);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage (reports stats, not a timing benchmark)
fn bench_quantization_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_memory");

    let vectors = generate_test_vectors(10_000, 768); // Larger vectors to see memory impact

    for precision in [Precision::F32, Precision::F16, Precision::BF16] {
        group.bench_with_input(
            BenchmarkId::new("precision", format!("{:?}", precision)),
            &precision,
            |b, &precision| {
                let config = HnswConfig {
                    quantization_precision: Some(precision),
                    ..Default::default()
                };
                let index = HnswIndex::new(768, config);

                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u128, vec.clone()).unwrap();
                }

                b.iter(|| {
                    let stats = index.memory_stats();
                    black_box(stats);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quantization_search,
    bench_quantization_insert,
    bench_quantization_memory
);
criterion_main!(benches);
