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

//! Write Throughput Benchmark
//!
//! Measures `DurableStorage::write()` under different durability modes:
//!
//! | Mode | Description | Expected Range |
//! |------|-------------|----------------|
//! | No fsync | Write to WAL buffer only | 100K-1M ops/sec |
//! | Batch fsync | fsync every N writes | 10K-100K ops/sec |
//! | Sync per write | fsync after each write | 200-2000 ops/sec |
//!
//! Run with: `cargo bench -p toondb-storage --bench bench_write`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{
    BenchConfig, DurableStorageHarness, SyncMode, generate_key, generate_value,
};
use toondb_storage::TransactionMode;

/// Benchmark writes with different sync modes
fn bench_write_sync_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/sync_mode");
    group.sample_size(20); // Fewer samples for write benchmarks (slower)

    for (mode_name, mode) in [
        ("off", SyncMode::Off),
        ("normal", SyncMode::Normal),
        ("full", SyncMode::Full),
    ] {
        let harness = DurableStorageHarness::new().expect("Failed to create harness");
        harness.set_sync_mode(mode);

        let config = BenchConfig {
            dataset_size: 0, // No preload needed
            value_size: 100,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(1));
        group.bench_function(mode_name, |b| {
            let mut idx = 0usize;

            b.iter(|| {
                let key = generate_key(idx, config.key_size);
                let value = generate_value(idx, config.value_size, 42);
                idx += 1;

                let txn_id = harness
                    .storage()
                    .begin_with_mode(TransactionMode::WriteOnly)
                    .unwrap();
                harness.storage().write(txn_id, key, value).unwrap();
                harness.storage().commit(txn_id).unwrap();
                black_box(())
            });
        });
    }

    group.finish();
}

/// Benchmark writes with different value sizes
fn bench_write_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/value_size");
    group.sample_size(20);

    for value_size in [100, 1_000, 10_000] {
        let harness = DurableStorageHarness::new().expect("Failed to create harness");
        harness.set_sync_mode(SyncMode::Off); // Fast writes for size comparison

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}B", value_size)),
            &value_size,
            |b, &value_size| {
                let mut idx = 0usize;
                let key_size = 16;

                b.iter(|| {
                    let key = generate_key(idx, key_size);
                    let value = generate_value(idx, value_size, 42);
                    idx += 1;

                    let txn_id = harness
                        .storage()
                        .begin_with_mode(TransactionMode::WriteOnly)
                        .unwrap();
                    harness.storage().write(txn_id, key, value).unwrap();
                    harness.storage().commit(txn_id).unwrap();
                    black_box(())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch writes
fn bench_write_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/batch");
    group.sample_size(20);

    for batch_size in [1, 10, 100, 1000] {
        let harness = DurableStorageHarness::new().expect("Failed to create harness");
        harness.set_sync_mode(SyncMode::Off);

        let key_size = 16;
        let value_size = 100;

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let mut base_idx = 0usize;

                b.iter(|| {
                    let txn_id = harness
                        .storage()
                        .begin_with_mode(TransactionMode::WriteOnly)
                        .unwrap();

                    // Write batch
                    for i in 0..batch_size {
                        let key = generate_key(base_idx + i, key_size);
                        let value = generate_value(base_idx + i, value_size, 42);
                        harness.storage().write(txn_id, key, value).unwrap();
                    }

                    harness.storage().commit(txn_id).unwrap();
                    base_idx += batch_size;
                    black_box(())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch writes using batch API
fn bench_write_batch_refs(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/batch_refs");
    group.sample_size(20);

    for batch_size in [10, 100, 1000] {
        let harness = DurableStorageHarness::new().expect("Failed to create harness");
        harness.set_sync_mode(SyncMode::Off);

        let key_size = 16;
        let value_size = 100;

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let mut base_idx = 0usize;

                // Pre-generate keys and values to isolate write performance
                let mut keys: Vec<Vec<u8>> = Vec::with_capacity(batch_size);
                let mut values: Vec<Vec<u8>> = Vec::with_capacity(batch_size);

                b.iter(|| {
                    keys.clear();
                    values.clear();

                    for i in 0..batch_size {
                        keys.push(generate_key(base_idx + i, key_size));
                        values.push(generate_value(base_idx + i, value_size, 42));
                    }

                    let writes: Vec<(&[u8], &[u8])> = keys
                        .iter()
                        .zip(values.iter())
                        .map(|(k, v)| (k.as_slice(), v.as_slice()))
                        .collect();

                    let txn_id = harness
                        .storage()
                        .begin_with_mode(TransactionMode::WriteOnly)
                        .unwrap();

                    harness.storage().write_batch_refs(txn_id, &writes).unwrap();
                    harness.storage().commit(txn_id).unwrap();

                    base_idx += batch_size;
                    black_box(())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark writes with group commit enabled
fn bench_write_group_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/group_commit");
    group.sample_size(20);

    // Without group commit
    {
        let harness =
            DurableStorageHarness::with_config(false, false).expect("Failed to create harness");
        harness.set_sync_mode(SyncMode::Normal);

        let key_size = 16;
        let value_size = 100;

        group.throughput(Throughput::Elements(1));
        group.bench_function("disabled", |b| {
            let mut idx = 0usize;

            b.iter(|| {
                let key = generate_key(idx, key_size);
                let value = generate_value(idx, value_size, 42);
                idx += 1;

                let txn_id = harness
                    .storage()
                    .begin_with_mode(TransactionMode::WriteOnly)
                    .unwrap();
                harness.storage().write(txn_id, key, value).unwrap();
                harness.storage().commit(txn_id).unwrap();
                black_box(())
            });
        });
    }

    // With group commit
    {
        let harness =
            DurableStorageHarness::with_config(false, true).expect("Failed to create harness");
        harness.set_sync_mode(SyncMode::Normal);

        let key_size = 16;
        let value_size = 100;

        group.throughput(Throughput::Elements(1));
        group.bench_function("enabled", |b| {
            let mut idx = 0usize;

            b.iter(|| {
                let key = generate_key(idx, key_size);
                let value = generate_value(idx, value_size, 42);
                idx += 1;

                let txn_id = harness
                    .storage()
                    .begin_with_mode(TransactionMode::WriteOnly)
                    .unwrap();
                harness.storage().write(txn_id, key, value).unwrap();
                harness.storage().commit(txn_id).unwrap();
                black_box(())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_write_sync_modes,
    bench_write_value_sizes,
    bench_write_batch,
    bench_write_batch_refs,
    bench_write_group_commit,
);

criterion_main!(benches);
