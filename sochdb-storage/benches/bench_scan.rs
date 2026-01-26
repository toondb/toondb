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

//! Range Scan Performance Benchmark
//!
//! Measures `DurableStorage::scan_range()` performance:
//!
//! | Scan Type | Parameters |
//! |-----------|------------|
//! | Short scan | 10 keys |
//! | Medium scan | 100 keys |
//! | Long scan | 1000 keys |
//! | Full scan | All keys |
//!
//! Measures:
//! - Time to first result (seek latency)
//! - Throughput (keys/second)
//!
//! Run with: `cargo bench -p sochdb-storage --bench bench_scan`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{BenchConfig, DurableStorageHarness, generate_key};
use sochdb_storage::TransactionMode;

/// Benchmark scans with different result set sizes
fn bench_scan_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/size");
    group.sample_size(30);

    let dataset_size = 100_000;
    let harness = DurableStorageHarness::with_config(true, false) // Enable ordered index for O(log N) scans
        .expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 100,
        ..Default::default()
    };

    // Preload data
    harness.preload(&config).expect("Failed to preload");

    for scan_size in [10, 100, 1000, 10_000] {
        group.throughput(Throughput::Elements(scan_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(scan_size),
            &scan_size,
            |b, &scan_size| {
                let key_size = config.key_size;
                let max_start = dataset_size.saturating_sub(scan_size);
                let mut start_idx = 0;

                b.iter(|| {
                    let start_key = generate_key(start_idx % max_start, key_size);
                    let end_key = generate_key((start_idx % max_start) + scan_size, key_size);
                    start_idx += 1;

                    let txn_id = harness
                        .storage()
                        .begin_with_mode(TransactionMode::ReadOnly)
                        .unwrap();
                    let results = harness
                        .storage()
                        .scan_range(txn_id, &start_key, &end_key)
                        .unwrap();
                    harness.storage().commit(txn_id).unwrap();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scan with streaming iterator vs collect
fn bench_scan_iterator_vs_collect(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/iter_vs_collect");
    group.sample_size(30);

    let dataset_size = 100_000;
    let scan_size = 1000;
    let harness = DurableStorageHarness::with_config(true, false)
        .expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 100,
        ..Default::default()
    };

    harness.preload(&config).expect("Failed to preload");

    let key_size = config.key_size;

    // Collect all results
    group.throughput(Throughput::Elements(scan_size as u64));
    group.bench_function("collect", |b| {
        let mut start_idx = 0;
        let max_start = dataset_size.saturating_sub(scan_size);

        b.iter(|| {
            let start_key = generate_key(start_idx % max_start, key_size);
            let end_key = generate_key((start_idx % max_start) + scan_size, key_size);
            start_idx += 1;

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let results = harness
                .storage()
                .scan_range(txn_id, &start_key, &end_key)
                .unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(results.len())
        });
    });

    // Use iterator (streaming)
    group.bench_function("iterator", |b| {
        let mut start_idx = 0;
        let max_start = dataset_size.saturating_sub(scan_size);

        b.iter(|| {
            let start_key = generate_key(start_idx % max_start, key_size);
            let end_key = generate_key((start_idx % max_start) + scan_size, key_size);
            start_idx += 1;

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            
            let mut count = 0;
            for (k, v) in harness.storage().scan_range_iter(txn_id, &start_key, &end_key) {
                count += k.len() + v.len();
            }
            
            harness.storage().commit(txn_id).unwrap();
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark prefix scan
fn bench_scan_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/prefix");
    group.sample_size(30);

    let dataset_size = 100_000;
    let harness = DurableStorageHarness::with_config(true, false)
        .expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 100,
        ..Default::default()
    };

    harness.preload(&config).expect("Failed to preload");

    // Prefix scans with different prefix lengths
    group.throughput(Throughput::Elements(1));
    group.bench_function("short_prefix", |b| {
        b.iter(|| {
            let prefix = b"key_000"; // Matches many keys
            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let results = harness.storage().scan(txn_id, prefix).unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(results.len())
        });
    });

    group.bench_function("long_prefix", |b| {
        b.iter(|| {
            let prefix = b"key_0000000000001"; // Matches very few keys
            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let results = harness.storage().scan(txn_id, prefix).unwrap();
            harness.storage().commit(txn_id).unwrap();
            black_box(results.len())
        });
    });

    group.finish();
}

/// Benchmark scan with different value sizes
fn bench_scan_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/value_size");
    group.sample_size(20);

    let dataset_size = 10_000;
    let scan_size = 100;

    for value_size in [100, 1_000, 10_000] {
        let harness = DurableStorageHarness::with_config(true, false)
            .expect("Failed to create harness");

        let config = BenchConfig {
            dataset_size,
            value_size,
            ..Default::default()
        };

        harness.preload(&config).expect("Failed to preload");

        let key_size = config.key_size;
        let max_start = dataset_size.saturating_sub(scan_size);

        group.throughput(Throughput::Bytes((scan_size * value_size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}B", value_size)),
            &value_size,
            |b, _| {
                let mut start_idx = 0;

                b.iter(|| {
                    let start_key = generate_key(start_idx % max_start, key_size);
                    let end_key = generate_key((start_idx % max_start) + scan_size, key_size);
                    start_idx += 1;

                    let txn_id = harness
                        .storage()
                        .begin_with_mode(TransactionMode::ReadOnly)
                        .unwrap();
                    let results = harness
                        .storage()
                        .scan_range(txn_id, &start_key, &end_key)
                        .unwrap();
                    harness.storage().commit(txn_id).unwrap();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_scan_sizes,
    bench_scan_iterator_vs_collect,
    bench_scan_prefix,
    bench_scan_value_sizes,
);

criterion_main!(benches);
