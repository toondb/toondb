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

//! Point Read Latency Benchmark
//!
//! Measures `DurableStorage::read()` performance under varying conditions:
//!
//! | Scenario | What It Tests |
//! |----------|---------------|
//! | Memtable hit | Best case - data in active memtable |
//! | Uniform random | Cache-unfriendly access pattern |
//! | Zipfian (θ=0.99) | Realistic hot/cold distribution |
//! | Miss (key not found) | Bloom filter effectiveness |
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_point_read`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{
    BenchConfig, Distribution, DurableStorageHarness, ZipfianGenerator, generate_key,
    generate_value, next_key_index,
};
use toondb_storage::TransactionMode;

/// Benchmark point reads with different dataset sizes
fn bench_point_read_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_read/size");
    group.sample_size(50);

    for size in [1_000, 10_000, 100_000] {
        let harness = DurableStorageHarness::new().expect("Failed to create harness");
        
        let config = BenchConfig {
            dataset_size: size,
            measurement_ops: 1000,
            warmup_ops: 100,
            ..Default::default()
        };

        // Preload data
        harness.preload(&config).expect("Failed to preload");

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            let mut rng = rand::thread_rng();
            let zipf = ZipfianGenerator::new(config.dataset_size, config.zipf_theta);
            let mut seq = 0;

            b.iter(|| {
                let idx = next_key_index(
                    &mut rng,
                    config.distribution,
                    Some(&zipf),
                    config.dataset_size,
                    &mut seq,
                );
                let key = generate_key(idx, config.key_size);

                let txn_id = harness
                    .storage()
                    .begin_with_mode(TransactionMode::ReadOnly)
                    .unwrap();
                let result = harness.storage().read(txn_id, &key).unwrap();
                harness.storage().commit(txn_id).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark point reads with different distributions
fn bench_point_read_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_read/distribution");
    group.sample_size(50);

    let size = 100_000;
    let harness = DurableStorageHarness::new().expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size: size,
        measurement_ops: 1000,
        warmup_ops: 100,
        ..Default::default()
    };

    // Preload data
    harness.preload(&config).expect("Failed to preload");

    // Uniform distribution
    group.throughput(Throughput::Elements(1));
    group.bench_function("uniform", |b| {
        let mut rng = rand::thread_rng();
        let mut seq = 0;

        b.iter(|| {
            let idx = next_key_index(&mut rng, Distribution::Uniform, None, size, &mut seq);
            let key = generate_key(idx, config.key_size);

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let result = harness.storage().read(txn_id, &key).unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(result)
        });
    });

    // Zipfian distribution
    group.bench_function("zipfian", |b| {
        let mut rng = rand::thread_rng();
        let zipf = ZipfianGenerator::new(size, 0.99);
        let mut seq = 0;

        b.iter(|| {
            let idx = next_key_index(&mut rng, Distribution::Zipfian, Some(&zipf), size, &mut seq);
            let key = generate_key(idx, config.key_size);

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let result = harness.storage().read(txn_id, &key).unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(result)
        });
    });

    // Sequential distribution
    group.bench_function("sequential", |b| {
        let mut rng = rand::thread_rng();
        let mut seq = 0;

        b.iter(|| {
            let idx = next_key_index(&mut rng, Distribution::Sequential, None, size, &mut seq);
            let key = generate_key(idx, config.key_size);

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let result = harness.storage().read(txn_id, &key).unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark point read misses (key not found)
fn bench_point_read_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_read/miss");
    group.sample_size(50);

    let size = 100_000;
    let harness = DurableStorageHarness::new().expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size: size,
        measurement_ops: 1000,
        warmup_ops: 100,
        ..Default::default()
    };

    // Preload data
    harness.preload(&config).expect("Failed to preload");

    // Read non-existent keys
    group.throughput(Throughput::Elements(1));
    group.bench_function("miss", |b| {
        let mut idx = size; // Start from keys that don't exist

        b.iter(|| {
            // Generate a key that doesn't exist
            let key = generate_key(idx + 1_000_000, config.key_size);
            idx += 1;

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let result = harness.storage().read(txn_id, &key).unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(result) // Should be None
        });
    });

    group.finish();
}

/// Benchmark point reads with different value sizes
fn bench_point_read_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_read/value_size");
    group.sample_size(50);

    let dataset_size = 10_000;

    for value_size in [100, 1_000, 10_000] {
        let harness = DurableStorageHarness::new().expect("Failed to create harness");

        let config = BenchConfig {
            dataset_size,
            value_size,
            measurement_ops: 1000,
            warmup_ops: 100,
            ..Default::default()
        };

        // Preload data
        harness.preload(&config).expect("Failed to preload");

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}B", value_size)),
            &value_size,
            |b, _| {
                let mut rng = rand::thread_rng();
                let zipf = ZipfianGenerator::new(dataset_size, 0.99);
                let mut seq = 0;

                b.iter(|| {
                    let idx =
                        next_key_index(&mut rng, Distribution::Zipfian, Some(&zipf), dataset_size, &mut seq);
                    let key = generate_key(idx, config.key_size);

                    let txn_id = harness
                        .storage()
                        .begin_with_mode(TransactionMode::ReadOnly)
                        .unwrap();
                    let result = harness.storage().read(txn_id, &key).unwrap();
                    harness.storage().commit(txn_id).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_point_read_sizes,
    bench_point_read_distributions,
    bench_point_read_miss,
    bench_point_read_value_sizes,
);

criterion_main!(benches);
