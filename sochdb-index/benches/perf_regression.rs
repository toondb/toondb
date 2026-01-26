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

//! Performance Regression Tests for HNSW Insert Operations
//!
//! This module provides criterion benchmarks that can be run in CI to detect
//! performance regressions in the HNSW insert path.
//!
//! # Usage
//!
//! ```bash
//! # Run benchmarks and save baseline
//! cargo bench --bench perf_regression -- --save-baseline main
//!
//! # Compare against baseline (in CI)
//! cargo bench --bench perf_regression -- --baseline main
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use sochdb_index::hnsw::{HnswConfig, HnswIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generate reproducible random vectors
fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Benchmark batch insert with flat API (Task 1 path)
fn bench_insert_batch_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_batch_flat");
    
    // Test configurations: (dimension, num_vectors)
    let configs = [
        (128, 1000),
        (128, 10000),
        (768, 1000),
        (768, 10000),
    ];
    
    for (dim, n) in configs {
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);
        
        let vectors = generate_vectors(n, dim, 42);
        let ids: Vec<u128> = (0..n as u128).collect();
        
        group.bench_with_input(
            BenchmarkId::new(format!("{}d", dim), n),
            &(dim, n, &vectors, &ids),
            |b, (dim, _n, vectors, ids)| {
                b.iter(|| {
                    let config = HnswConfig {
                        max_connections: 16,
                        max_connections_layer0: 32,
                        ef_construction: 48,
                        ..Default::default()
                    };
                    let index = HnswIndex::new(*dim, config);
                    let result = index.insert_batch_flat(ids, vectors, *dim);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark single-vector insert from slice (Task 1 path)
fn bench_insert_one_from_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_one_from_slice");
    
    for dim in [128, 768] {
        group.throughput(Throughput::Elements(1));
        
        let vector: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();
        
        // Pre-create index with some vectors for realistic conditions
        let config = HnswConfig {
            max_connections: 16,
            ef_construction: 48,
            ..Default::default()
        };
        let index = HnswIndex::new(dim, config);
        
        // Add some base vectors
        let base_vectors = generate_vectors(1000, dim, 42);
        let base_ids: Vec<u128> = (0..1000).collect();
        let _ = index.insert_batch_flat(&base_ids, &base_vectors, dim);
        
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &(dim, &vector),
            |b, (dim, vector)| {
                let mut id = 10000u128;
                b.iter(|| {
                    id += 1;
                    let result = index.insert_one_from_slice(id, vector);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark the old contiguous API for comparison
fn bench_insert_batch_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_batch_contiguous");
    
    let configs = [
        (128, 10000),
        (768, 10000),
    ];
    
    for (dim, n) in configs {
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);
        
        let vectors = generate_vectors(n, dim, 42);
        let ids: Vec<u128> = (0..n as u128).collect();
        
        group.bench_with_input(
            BenchmarkId::new(format!("{}d", dim), n),
            &(dim, n, &vectors, &ids),
            |b, (dim, _n, vectors, ids)| {
                b.iter(|| {
                    let config = HnswConfig {
                        max_connections: 16,
                        max_connections_layer0: 32,
                        ef_construction: 48,
                        ..Default::default()
                    };
                    let index = HnswIndex::new(*dim, config);
                    let result = index.insert_batch_contiguous(ids, vectors, *dim);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark mmap ingest path (Task 6)
fn bench_build_from_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_from_mmap");
    
    let configs = [
        (128, 10000),
        (768, 5000),  // Smaller for 768D due to time
    ];
    
    for (dim, n) in configs {
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);
        
        let vectors = generate_vectors(n, dim, 42);
        
        group.bench_with_input(
            BenchmarkId::new(format!("{}d", dim), n),
            &(dim, n, &vectors),
            |b, (dim, n, vectors)| {
                b.iter(|| {
                    let config = HnswConfig {
                        max_connections: 16,
                        max_connections_layer0: 32,
                        ef_construction: 48,
                        ..Default::default()
                    };
                    let index = HnswIndex::new(*dim, config);
                    let result = index.build_from_mmap(vectors, *dim, *n);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_batch_flat,
    bench_insert_one_from_slice,
    bench_insert_batch_contiguous,
    bench_build_from_mmap,
);

criterion_main!(benches);
