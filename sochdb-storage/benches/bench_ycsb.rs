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

//! YCSB Workload Benchmark
//!
//! Implements standard YCSB workloads against SochDB DurableStorage:
//!
//! | Workload | Read % | Update % | Scan % | Insert % | Description |
//! |----------|--------|----------|--------|----------|-------------|
//! | A | 50 | 50 | 0 | 0 | Update heavy |
//! | B | 95 | 5 | 0 | 0 | Read mostly |
//! | C | 100 | 0 | 0 | 0 | Read only |
//! | D | 95 | 0 | 0 | 5 | Read latest |
//! | E | 0 | 0 | 95 | 5 | Scan heavy |
//! | F | 50 | 0 | 0 | 50 | Read-modify-write |
//!
//! Run with: `cargo bench -p sochdb-storage --bench bench_ycsb`

mod measurement_harness;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use measurement_harness::{
    BenchConfig, Distribution, DurableStorageHarness, YcsbWorkload, ZipfianGenerator,
    generate_key, generate_value, next_key_index,
};
use rand::prelude::*;
use sochdb_storage::TransactionMode;

/// Run a YCSB workload mix
fn run_ycsb_workload(
    harness: &DurableStorageHarness,
    config: &BenchConfig,
    workload: YcsbWorkload,
    ops: usize,
) -> usize {
    let mut rng = rand::thread_rng();
    let zipf = ZipfianGenerator::new(config.dataset_size, config.zipf_theta);
    let mut seq = 0;
    let mut insert_counter = config.dataset_size;
    let mut completed = 0;

    let read_ratio = workload.read_ratio();
    let update_ratio = workload.update_ratio();
    let scan_ratio = workload.scan_ratio();
    // insert_ratio is the remainder

    for _ in 0..ops {
        let r: f64 = rng.r#gen();

        if r < read_ratio {
            // Read operation
            let idx = next_key_index(
                &mut rng,
                Distribution::Zipfian,
                Some(&zipf),
                config.dataset_size,
                &mut seq,
            );
            let key = generate_key(idx, config.key_size);

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let _ = harness.storage().read(txn_id, &key);
            harness.storage().commit(txn_id).unwrap();
            completed += 1;
        } else if r < read_ratio + update_ratio {
            // Update operation
            let idx = next_key_index(
                &mut rng,
                Distribution::Zipfian,
                Some(&zipf),
                config.dataset_size,
                &mut seq,
            );
            let key = generate_key(idx, config.key_size);
            let value = generate_value(idx, config.value_size, rng.r#gen());

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::WriteOnly)
                .unwrap();
            harness.storage().write(txn_id, key, value).unwrap();
            harness.storage().commit(txn_id).unwrap();
            completed += 1;
        } else if r < read_ratio + update_ratio + scan_ratio {
            // Scan operation
            let start_idx = rng.gen_range(0..config.dataset_size.saturating_sub(100));
            let scan_len = rng.gen_range(10..100);

            let start_key = generate_key(start_idx, config.key_size);
            let end_key = generate_key(start_idx + scan_len, config.key_size);

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::ReadOnly)
                .unwrap();
            let _ = harness.storage().scan_range(txn_id, &start_key, &end_key);
            harness.storage().commit(txn_id).unwrap();
            completed += 1;
        } else {
            // Insert operation
            let key = generate_key(insert_counter, config.key_size);
            let value = generate_value(insert_counter, config.value_size, rng.r#gen());
            insert_counter += 1;

            let txn_id = harness
                .storage()
                .begin_with_mode(TransactionMode::WriteOnly)
                .unwrap();
            harness.storage().write(txn_id, key, value).unwrap();
            harness.storage().commit(txn_id).unwrap();
            completed += 1;
        }
    }

    completed
}

/// Benchmark YCSB Workload A (50% read, 50% update)
fn bench_ycsb_a(c: &mut Criterion) {
    let mut group = c.benchmark_group("ycsb");
    group.sample_size(20);

    let dataset_size = 100_000;
    let harness = DurableStorageHarness::new().expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 1000, // 1KB values (YCSB default)
        ..Default::default()
    };

    harness.preload(&config).expect("Failed to preload");

    group.throughput(Throughput::Elements(100));
    group.bench_function("workload_a", |b| {
        b.iter(|| {
            black_box(run_ycsb_workload(&harness, &config, YcsbWorkload::A, 100))
        });
    });

    group.finish();
}

/// Benchmark YCSB Workload B (95% read, 5% update)
fn bench_ycsb_b(c: &mut Criterion) {
    let mut group = c.benchmark_group("ycsb");
    group.sample_size(20);

    let dataset_size = 100_000;
    let harness = DurableStorageHarness::new().expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 1000,
        ..Default::default()
    };

    harness.preload(&config).expect("Failed to preload");

    group.throughput(Throughput::Elements(100));
    group.bench_function("workload_b", |b| {
        b.iter(|| {
            black_box(run_ycsb_workload(&harness, &config, YcsbWorkload::B, 100))
        });
    });

    group.finish();
}

/// Benchmark YCSB Workload C (100% read)
fn bench_ycsb_c(c: &mut Criterion) {
    let mut group = c.benchmark_group("ycsb");
    group.sample_size(20);

    let dataset_size = 100_000;
    let harness = DurableStorageHarness::new().expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 1000,
        ..Default::default()
    };

    harness.preload(&config).expect("Failed to preload");

    group.throughput(Throughput::Elements(100));
    group.bench_function("workload_c", |b| {
        b.iter(|| {
            black_box(run_ycsb_workload(&harness, &config, YcsbWorkload::C, 100))
        });
    });

    group.finish();
}

/// Benchmark YCSB Workload E (95% scan, 5% insert)
fn bench_ycsb_e(c: &mut Criterion) {
    let mut group = c.benchmark_group("ycsb");
    group.sample_size(20);

    let dataset_size = 100_000;
    let harness = DurableStorageHarness::with_config(true, false)
        .expect("Failed to create harness");

    let config = BenchConfig {
        dataset_size,
        value_size: 1000,
        ..Default::default()
    };

    harness.preload(&config).expect("Failed to preload");

    group.throughput(Throughput::Elements(100));
    group.bench_function("workload_e", |b| {
        b.iter(|| {
            black_box(run_ycsb_workload(&harness, &config, YcsbWorkload::E, 100))
        });
    });

    group.finish();
}

/// Compare all YCSB workloads
fn bench_ycsb_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("ycsb/comparison");
    group.sample_size(10);

    let dataset_size = 50_000;

    for workload in [
        YcsbWorkload::A,
        YcsbWorkload::B,
        YcsbWorkload::C,
        YcsbWorkload::D,
        YcsbWorkload::E,
        YcsbWorkload::F,
    ] {
        let harness = DurableStorageHarness::with_config(true, false)
            .expect("Failed to create harness");

        let config = BenchConfig {
            dataset_size,
            value_size: 1000,
            ..Default::default()
        };

        harness.preload(&config).expect("Failed to preload");

        let workload_name = format!("{:?}", workload);
        group.throughput(Throughput::Elements(50));
        group.bench_with_input(
            BenchmarkId::from_parameter(&workload_name),
            &workload,
            |b, &workload| {
                b.iter(|| {
                    black_box(run_ycsb_workload(&harness, &config, workload, 50))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ycsb_a,
    bench_ycsb_b,
    bench_ycsb_c,
    bench_ycsb_e,
    bench_ycsb_all,
);

criterion_main!(benches);
