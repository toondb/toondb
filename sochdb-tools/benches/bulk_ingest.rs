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

//! Bulk Ingest Microbenchmarks
//!
//! Run with: cargo bench -p sochdb-tools --bench bulk_ingest
//!
//! These benchmarks measure:
//! - I/O throughput (npy/raw format parsing)
//! - HNSW insert throughput at various dimensions
//! - End-to-end bulk build performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::Rng;
use std::time::Duration;
use tempfile::TempDir;

use sochdb_index::hnsw::{HnswConfig, HnswIndex};

/// Generate random vectors for testing
fn generate_vectors(n: usize, d: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n * d).map(|_| rng.random::<f32>()).collect()
}

/// Benchmark HNSW batch insert at various dimensions
fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    for dimension in [128, 384, 768, 1536] {
        let n = 10_000;
        let vectors = generate_vectors(n, dimension);
        let ids: Vec<u128> = (0..n as u128).collect();
        
        group.throughput(Throughput::Elements(n as u64));
        
        group.bench_with_input(
            BenchmarkId::new("dimension", dimension),
            &(dimension, &vectors, &ids),
            |b, (d, vecs, ids)| {
                b.iter(|| {
                    let config = HnswConfig {
                        max_connections: 16,
                        max_connections_layer0: 32,
                        ef_construction: 100,
                        ..Default::default()
                    };
                    let index = HnswIndex::new(*d, config);
                    let inserted = index.insert_batch_flat(ids, vecs, *d).unwrap();
                    black_box(inserted)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark various batch sizes for 768D vectors
fn bench_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    let dimension = 768;
    let n = 10_000;
    let vectors = generate_vectors(n, dimension);
    let ids: Vec<u128> = (0..n as u128).collect();
    
    for batch_size in [100, 500, 1000, 2000, 5000] {
        group.throughput(Throughput::Elements(n as u64));
        
        group.bench_with_input(
            BenchmarkId::new("size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = HnswConfig {
                        max_connections: 16,
                        max_connections_layer0: 32,
                        ef_construction: 100,
                        ..Default::default()
                    };
                    let index = HnswIndex::new(dimension, config);
                    
                    let mut total = 0;
                    for chunk_start in (0..n).step_by(batch_size) {
                        let chunk_end = (chunk_start + batch_size).min(n);
                        let batch_ids = &ids[chunk_start..chunk_end];
                        let batch_vecs = &vectors[chunk_start * dimension..chunk_end * dimension];
                        total += index.insert_batch_flat(batch_ids, batch_vecs, dimension).unwrap();
                    }
                    black_box(total)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark raw f32 memory read (simulating mmap)
fn bench_memory_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_read");
    
    let dimension = 768;
    let n = 100_000;
    let vectors = generate_vectors(n, dimension);
    
    group.throughput(Throughput::Bytes((n * dimension * 4) as u64));
    
    group.bench_function("sequential_sum", |b| {
        b.iter(|| {
            let sum: f32 = vectors.iter().sum();
            black_box(sum)
        });
    });
    
    group.bench_function("chunked_iter", |b| {
        b.iter(|| {
            let sum: f32 = vectors
                .chunks(dimension)
                .map(|chunk| chunk.iter().sum::<f32>())
                .sum();
            black_box(sum)
        });
    });
    
    group.finish();
}

/// End-to-end bulk build benchmark
fn bench_bulk_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_build_e2e");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    
    for (n, dimension) in [(1000, 768), (5000, 768), (10000, 384)] {
        let vectors = generate_vectors(n, dimension);
        let ids: Vec<u128> = (0..n as u128).collect();
        
        let label = format!("{}x{}", n, dimension);
        group.throughput(Throughput::Elements(n as u64));
        
        group.bench_function(&label, |b| {
            let temp_dir = TempDir::new().unwrap();
            
            b.iter(|| {
                let config = HnswConfig {
                    max_connections: 16,
                    max_connections_layer0: 32,
                    ef_construction: 100,
                    ..Default::default()
                };
                let index = HnswIndex::new(dimension, config);
                
                // Insert all vectors
                let inserted = index.insert_batch_flat(&ids, &vectors, dimension).unwrap();
                
                // Save to disk
                let output = temp_dir.path().join("index.hnsw");
                index.save_to_disk_compressed(&output).unwrap();
                
                black_box(inserted)
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_insert,
    bench_batch_size,
    bench_memory_read,
    bench_bulk_build,
);

criterion_main!(benches);
