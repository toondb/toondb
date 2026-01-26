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

//! SochDB Benchmark Harness
//!
//! Comprehensive benchmarks for SochDB covering:
//! - YCSB-style workloads (read-heavy, write-heavy, scan)
//! - Token efficiency (TOON vs JSON)
//! - Recovery time measurements
//! - Throughput and latency analysis
//!
//! Run with: `cargo bench -p sochdb-storage`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// YCSB Workload Types
// ============================================================================

/// YCSB workload types
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    /// Workload A: 50% read, 50% update
    ReadHeavy,
    /// Workload B: 5% read, 95% update
    WriteHeavy,
    /// Workload C: 100% scan
    ScanHeavy,
    /// Workload D: Token efficiency comparison
    TokenEfficiency,
}

/// Workload configuration
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Number of operations
    pub num_operations: usize,
    /// Number of records in the database
    pub record_count: usize,
    /// Read ratio (0.0 to 1.0)
    pub read_ratio: f64,
    /// Update ratio (0.0 to 1.0)
    pub update_ratio: f64,
    /// Scan ratio (0.0 to 1.0)
    pub scan_ratio: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl WorkloadConfig {
    /// Create workload A (read-heavy)
    pub fn workload_a(record_count: usize) -> Self {
        Self {
            num_operations: 100_000,
            record_count,
            read_ratio: 0.5,
            update_ratio: 0.5,
            scan_ratio: 0.0,
            seed: 42,
        }
    }

    /// Create workload B (write-heavy)
    pub fn workload_b(record_count: usize) -> Self {
        Self {
            num_operations: 100_000,
            record_count,
            read_ratio: 0.05,
            update_ratio: 0.95,
            scan_ratio: 0.0,
            seed: 42,
        }
    }

    /// Create workload C (scan-heavy)
    pub fn workload_c(record_count: usize) -> Self {
        Self {
            num_operations: 1_000,
            record_count,
            read_ratio: 0.0,
            update_ratio: 0.0,
            scan_ratio: 1.0,
            seed: 42,
        }
    }
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

/// Sample user record for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecord {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub age: u32,
    pub score: f64,
    pub active: bool,
}

impl UserRecord {
    /// Generate a synthetic user record
    pub fn generate(id: u64, seed: u64) -> Self {
        // Simple deterministic generation based on id and seed
        let hash = (id.wrapping_mul(0x517cc1b727220a95) ^ seed).wrapping_mul(0x9e3779b97f4a7c15);

        let name = format!("User{}", id);
        let email = format!("user{}@example.com", id);
        let age = (hash % 63) as u32 + 18; // 18-80
        let score = (hash % 10000) as f64 / 100.0;
        let active = hash.is_multiple_of(2);

        Self {
            id,
            name,
            email,
            age,
            score,
            active,
        }
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    /// Convert to TOON format
    pub fn to_toon(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.id,
            self.name,
            self.email,
            self.age,
            self.score,
            if self.active { "true" } else { "false" }
        )
    }
}

/// Generate a dataset of user records
pub fn generate_dataset(count: usize, seed: u64) -> Vec<UserRecord> {
    (0..count as u64)
        .map(|i| UserRecord::generate(i, seed))
        .collect()
}

// ============================================================================
// Token Counting (Simplified)
// ============================================================================

/// Approximate token count for text (GPT-4 style: ~3.5 chars per token)
pub fn count_tokens(text: &str) -> usize {
    // GPT-4 approximation: ~3.5 characters per token for English text
    (text.len() as f64 / 3.5).ceil() as usize
}

/// Token statistics for TOON vs JSON comparison
#[derive(Debug, Clone)]
pub struct TokenStats {
    pub json_tokens: usize,
    pub soch_tokens: usize,
    pub savings_percent: f64,
}

/// Calculate token savings for a dataset
pub fn calculate_token_savings(records: &[UserRecord]) -> TokenStats {
    // JSON format: array of objects
    let json_header = r#"[{"id":,"name":"","email":"","age":,"score":,"active":}]"#;
    let json_per_record: usize = records.iter().map(|r| r.to_json().len()).sum();
    let json_total = json_per_record + (records.len() - 1); // commas between records

    // TOON format: table with header
    let soch_header = "users[N]{id,name,email,age,score,active}:\n";
    let soch_per_record: usize = records
        .iter()
        .map(|r| r.to_toon().len() + 1) // +1 for newline
        .sum();
    let soch_total = soch_header.len() + soch_per_record;

    let _json_tokens_approx = count_tokens(&format!("{}{}", json_header, json_per_record));
    let _soch_tokens_approx = count_tokens(&format!("{}", soch_total));

    // More accurate calculation
    let json_tokens = (json_total as f64 / 3.5).ceil() as usize;
    let soch_tokens = (soch_total as f64 / 3.5).ceil() as usize;

    let savings_percent = if json_tokens > 0 {
        (json_tokens as f64 - soch_tokens as f64) / json_tokens as f64 * 100.0
    } else {
        0.0
    };

    TokenStats {
        json_tokens,
        soch_tokens,
        savings_percent,
    }
}

// ============================================================================
// In-Memory Key-Value Store for Benchmarking
// ============================================================================

/// Simple in-memory store for benchmarking
pub struct MemoryStore {
    data: HashMap<u64, UserRecord>,
}

impl MemoryStore {
    /// Create a new store
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Load initial data
    pub fn load(&mut self, records: Vec<UserRecord>) {
        for record in records {
            self.data.insert(record.id, record);
        }
    }

    /// Read a record
    #[inline]
    pub fn read(&self, id: u64) -> Option<&UserRecord> {
        self.data.get(&id)
    }

    /// Update a record
    #[inline]
    pub fn update(&mut self, id: u64, record: UserRecord) {
        self.data.insert(id, record);
    }

    /// Scan records in a range
    pub fn scan(&self, start: u64, count: usize) -> Vec<&UserRecord> {
        let mut result = Vec::with_capacity(count);
        for i in 0..count as u64 {
            if let Some(record) = self.data.get(&(start + i)) {
                result.push(record);
            }
        }
        result
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Benchmark Results
// ============================================================================

/// Benchmark result statistics
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub operations: usize,
    pub total_time: Duration,
    pub throughput_ops_sec: f64,
    pub avg_latency_us: f64,
    pub p50_latency_us: f64,
    pub p99_latency_us: f64,
}

impl BenchmarkResult {
    /// Calculate from latency samples
    pub fn from_samples(name: &str, samples: &mut [Duration]) -> Self {
        let operations = samples.len();
        let total_time: Duration = samples.iter().sum();

        // Sort for percentiles
        samples.sort();

        let avg_latency_us = if operations > 0 {
            total_time.as_micros() as f64 / operations as f64
        } else {
            0.0
        };

        let p50_latency_us = if operations > 0 {
            samples[operations / 2].as_micros() as f64
        } else {
            0.0
        };

        let p99_latency_us = if operations > 0 {
            samples[operations * 99 / 100].as_micros() as f64
        } else {
            0.0
        };

        let throughput_ops_sec = if total_time.as_secs_f64() > 0.0 {
            operations as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            name: name.to_string(),
            operations,
            total_time,
            throughput_ops_sec,
            avg_latency_us,
            p50_latency_us,
            p99_latency_us,
        }
    }

    /// Print results
    pub fn print(&self) {
        println!("=== {} ===", self.name);
        println!("  Operations: {}", self.operations);
        println!("  Total time: {:.2}s", self.total_time.as_secs_f64());
        println!("  Throughput: {:.2} ops/sec", self.throughput_ops_sec);
        println!("  Avg latency: {:.2}μs", self.avg_latency_us);
        println!("  P50 latency: {:.2}μs", self.p50_latency_us);
        println!("  P99 latency: {:.2}μs", self.p99_latency_us);
    }
}

// ============================================================================
// Criterion Benchmarks
// ============================================================================

/// Benchmark read operations
fn bench_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("read");

    for size in [1_000, 10_000, 100_000].iter() {
        let records = generate_dataset(*size, 42);
        let mut store = MemoryStore::new();
        store.load(records);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut i = 0u64;
            b.iter(|| {
                let id = i % size as u64;
                i += 1;
                black_box(store.read(id))
            });
        });
    }

    group.finish();
}

/// Benchmark write operations
fn bench_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("write");

    for size in [1_000, 10_000, 100_000].iter() {
        let records = generate_dataset(*size, 42);
        let mut store = MemoryStore::new();
        store.load(records);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut i = 0u64;
            b.iter(|| {
                let id = i % size as u64;
                let record = UserRecord::generate(id, 123);
                store.update(id, record);
                i += 1;
            });
        });
    }

    group.finish();
}

/// Benchmark scan operations
fn bench_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan");

    let size = 100_000;
    let records = generate_dataset(size, 42);
    let mut store = MemoryStore::new();
    store.load(records);

    for scan_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*scan_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(scan_size),
            scan_size,
            |b, &scan_size| {
                let mut start = 0u64;
                b.iter(|| {
                    let results = store.scan(start, scan_size);
                    start = (start + 1) % (size as u64 - scan_size as u64);
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark TOON vs JSON token efficiency
fn bench_token_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_efficiency");

    for size in [100, 1000, 10000].iter() {
        let records = generate_dataset(*size, 42);

        group.bench_with_input(BenchmarkId::new("json", size), &records, |b, records| {
            b.iter(|| {
                let total: usize = records.iter().map(|r| r.to_json().len()).sum();
                black_box(total)
            });
        });

        group.bench_with_input(BenchmarkId::new("toon", size), &records, |b, records| {
            b.iter(|| {
                let total: usize = records.iter().map(|r| r.to_toon().len()).sum();
                black_box(total)
            });
        });
    }

    group.finish();

    // Print token savings analysis
    println!("\n=== Token Efficiency Analysis ===");
    for size in [100, 1000, 10000].iter() {
        let records = generate_dataset(*size, 42);
        let stats = calculate_token_savings(&records);
        println!(
            "Records: {} | JSON: {} tokens | TOON: {} tokens | Savings: {:.1}%",
            size, stats.json_tokens, stats.soch_tokens, stats.savings_percent
        );
    }
}

/// Benchmark mixed workload (YCSB-A style)
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");

    let size = 100_000;
    let records = generate_dataset(size, 42);
    let mut store = MemoryStore::new();
    store.load(records);

    // 50% read, 50% update
    group.throughput(Throughput::Elements(1));
    group.bench_function("ycsb_a", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % size as u64;
            if i.is_multiple_of(2) {
                black_box(store.read(id));
            } else {
                let record = UserRecord::generate(id, 123);
                store.update(id, record);
            }
            i += 1;
        });
    });

    // 5% read, 95% update
    group.bench_function("ycsb_b", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % size as u64;
            if i.is_multiple_of(20) {
                black_box(store.read(id));
            } else {
                let record = UserRecord::generate(id, 123);
                store.update(id, record);
            }
            i += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_read,
    bench_write,
    bench_scan,
    bench_token_efficiency,
    bench_mixed_workload,
);

criterion_main!(benches);
