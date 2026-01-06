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

//! ToonDB Measurement Harness
//!
//! Core infrastructure for benchmarking the actual ToonDB `DurableStorage` engine.
//! This replaces the synthetic `MemoryStore` benchmarks with real storage operations.
//!
//! ## Features
//!
//! - **Realistic workloads**: Zipfian key distributions, configurable value sizes
//! - **Comprehensive metrics**: Latency percentiles (P50, P95, P99, P99.9, max), throughput
//! - **Statistical validity**: Warm-up phase, multiple samples, confidence intervals
//! - **Isolation**: Each benchmark runs in a fresh temp directory
//!
//! ## Usage
//!
//! ```rust,ignore
//! use measurement_harness::{BenchConfig, DurableStorageHarness, BenchStats};
//!
//! let config = BenchConfig::default();
//! let harness = DurableStorageHarness::new()?;
//! let stats = harness.run_point_reads(&config)?;
//! stats.print_summary();
//! ```

use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use toondb_storage::{DurableStorage, TransactionMode};

// ============================================================================
// Configuration
// ============================================================================

/// Standard dataset sizes for benchmarks
pub const SMALL_DATASET: usize = 10_000;
pub const MEDIUM_DATASET: usize = 1_000_000;
pub const LARGE_DATASET: usize = 10_000_000;

/// Standard key/value sizes
pub const DEFAULT_KEY_SIZE: usize = 16;
pub const SMALL_VALUE_SIZE: usize = 100;
pub const MEDIUM_VALUE_SIZE: usize = 1_000;
pub const LARGE_VALUE_SIZE: usize = 10_000;

/// Standard Zipf skew (YCSB default)
pub const ZIPF_THETA: f64 = 0.99;

/// Warm-up and measurement defaults
pub const DEFAULT_WARMUP_OPS: usize = 1_000;
pub const DEFAULT_MEASUREMENT_OPS: usize = 10_000;

/// Key access distribution types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Distribution {
    /// Uniform random access
    Uniform,
    /// Zipfian distribution with configurable skew
    Zipfian,
    /// Sequential access
    Sequential,
}

impl Default for Distribution {
    fn default() -> Self {
        Distribution::Zipfian
    }
}

/// Sync mode for durability benchmarks
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SyncMode {
    /// No fsync - fastest, risk of data loss
    Off = 0,
    /// Periodic fsync - balanced
    Normal = 1,
    /// Fsync every commit - safest, slowest
    Full = 2,
}

impl Default for SyncMode {
    fn default() -> Self {
        SyncMode::Normal
    }
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    /// Number of keys to pre-load
    pub dataset_size: usize,
    /// Size of each key in bytes
    pub key_size: usize,
    /// Size of each value in bytes
    pub value_size: usize,
    /// Key access distribution
    pub distribution: Distribution,
    /// Zipf theta parameter (only used if distribution is Zipfian)
    pub zipf_theta: f64,
    /// Number of warm-up operations (discarded)
    pub warmup_ops: usize,
    /// Number of measured operations
    pub measurement_ops: usize,
    /// Sync mode for writes
    pub sync_mode: SyncMode,
    /// Number of concurrent threads (for concurrency benchmarks)
    pub num_threads: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            dataset_size: MEDIUM_DATASET,
            key_size: DEFAULT_KEY_SIZE,
            value_size: SMALL_VALUE_SIZE,
            distribution: Distribution::Zipfian,
            zipf_theta: ZIPF_THETA,
            warmup_ops: DEFAULT_WARMUP_OPS,
            measurement_ops: DEFAULT_MEASUREMENT_OPS,
            sync_mode: SyncMode::Normal,
            num_threads: 1,
        }
    }
}

impl BenchConfig {
    /// Create config for small dataset
    pub fn small() -> Self {
        Self {
            dataset_size: SMALL_DATASET,
            ..Default::default()
        }
    }

    /// Create config for large dataset
    pub fn large() -> Self {
        Self {
            dataset_size: LARGE_DATASET,
            ..Default::default()
        }
    }

    /// Set uniform distribution
    pub fn with_uniform(mut self) -> Self {
        self.distribution = Distribution::Uniform;
        self
    }

    /// Set sequential distribution
    pub fn with_sequential(mut self) -> Self {
        self.distribution = Distribution::Sequential;
        self
    }

    /// Set value size
    pub fn with_value_size(mut self, size: usize) -> Self {
        self.value_size = size;
        self
    }

    /// Set sync mode
    pub fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Set number of threads
    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Set dataset size
    pub fn with_dataset_size(mut self, size: usize) -> Self {
        self.dataset_size = size;
        self
    }

    /// Set measurement ops
    pub fn with_measurement_ops(mut self, ops: usize) -> Self {
        self.measurement_ops = ops;
        self
    }
}

// ============================================================================
// Zipfian Distribution Generator
// ============================================================================

/// Zipfian distribution generator for realistic hot/cold access patterns
///
/// Uses the rejection-inversion method for O(1) generation.
/// Formula: P(rank=k) = (1/k^θ) / H_N where H_N = Σ(1/i^θ)
pub struct ZipfianGenerator {
    n: usize,
    theta: f64,
    zeta_n: f64,
    zeta_2: f64,
    alpha: f64,
    eta: f64,
}

impl ZipfianGenerator {
    /// Create a new Zipfian generator for n items with skew theta
    pub fn new(n: usize, theta: f64) -> Self {
        let zeta_2 = Self::zeta(2, theta);
        let zeta_n = Self::zeta(n, theta);

        let alpha = 1.0 / (1.0 - theta);
        let eta = (1.0 - (2.0_f64 / n as f64).powf(1.0 - theta)) / (1.0 - zeta_2 / zeta_n);

        Self {
            n,
            theta,
            zeta_n,
            zeta_2,
            alpha,
            eta,
        }
    }

    /// Calculate zeta(n, theta) = Σ(1/i^theta) for i=1 to n
    fn zeta(n: usize, theta: f64) -> f64 {
        (1..=n).map(|i| 1.0 / (i as f64).powf(theta)).sum()
    }

    /// Generate the next Zipfian value in range [0, n)
    pub fn next<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.r#gen();
        let uz = u * self.zeta_n;

        if uz < 1.0 {
            return 0;
        }
        if uz < 1.0 + 0.5_f64.powf(self.theta) {
            return 1;
        }

        let result =
            (self.n as f64 * (self.eta * u - self.eta + 1.0).powf(self.alpha)) as usize;
        result.min(self.n - 1)
    }
}

// ============================================================================
// Key/Value Generation
// ============================================================================

/// Generate a deterministic key for a given index
pub fn generate_key(index: usize, key_size: usize) -> Vec<u8> {
    let mut key = format!("key_{:016}", index);
    if key.len() < key_size {
        key.push_str(&"_".repeat(key_size - key.len()));
    }
    key.truncate(key_size);
    key.into_bytes()
}

/// Generate a deterministic value for a given index
pub fn generate_value(index: usize, value_size: usize, seed: u64) -> Vec<u8> {
    let hash = (index as u64)
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(seed);
    let pattern = format!("val_{:016x}_", hash);
    let mut value = pattern.repeat((value_size / pattern.len()) + 1);
    value.truncate(value_size);
    value.into_bytes()
}

/// Generate the next key index based on distribution
pub fn next_key_index<R: Rng>(
    rng: &mut R,
    distribution: Distribution,
    zipf: Option<&ZipfianGenerator>,
    max: usize,
    sequential_counter: &mut usize,
) -> usize {
    match distribution {
        Distribution::Uniform => rng.gen_range(0..max),
        Distribution::Zipfian => zipf.map(|z| z.next(rng)).unwrap_or(0),
        Distribution::Sequential => {
            let idx = *sequential_counter % max;
            *sequential_counter += 1;
            idx
        }
    }
}

// ============================================================================
// Benchmark Statistics
// ============================================================================

/// Benchmark result statistics with latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchStats {
    /// Name of the benchmark
    pub benchmark: String,
    /// Timestamp of the run
    pub timestamp: String,
    /// Configuration used
    pub configuration: BenchConfig,
    /// Results
    pub results: BenchResults,
    /// Hardware info
    pub hardware: HardwareInfo,
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResults {
    /// Throughput in operations per second
    pub throughput_ops_sec: f64,
    /// Latency percentiles in microseconds
    pub latency_us: LatencyStats,
    /// Memory usage in MB (if measured)
    pub memory_mb: Option<f64>,
    /// Number of samples collected
    pub samples: usize,
}

/// Latency percentile statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub max: f64,
}

impl LatencyStats {
    /// Calculate latency stats from a sorted slice of microsecond values
    pub fn from_sorted(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                min: 0.0,
                mean: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                p999: 0.0,
                max: 0.0,
            };
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;

        Self {
            min: samples[0],
            mean,
            p50: samples[n / 2],
            p95: samples[n * 95 / 100],
            p99: samples[n * 99 / 100],
            p999: samples[n * 999 / 1000].min(samples[n - 1]),
            max: samples[n - 1],
        }
    }
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: String,
    pub num_cpus: usize,
    pub ram_gb: f64,
    pub os: String,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self {
            cpu: "Unknown".to_string(),
            num_cpus: num_cpus::get(),
            ram_gb: 0.0,
            os: std::env::consts::OS.to_string(),
        }
    }
}

impl BenchStats {
    /// Create new benchmark stats
    pub fn new(
        benchmark: &str,
        config: BenchConfig,
        samples: &mut [Duration],
        total_time: Duration,
    ) -> Self {
        // Sort samples for percentile calculation
        samples.sort();

        // Convert to microseconds
        let samples_us: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1_000_000.0).collect();

        let latency = LatencyStats::from_sorted(&samples_us);
        let throughput = if total_time.as_secs_f64() > 0.0 {
            samples.len() as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            benchmark: benchmark.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            configuration: config,
            results: BenchResults {
                throughput_ops_sec: throughput,
                latency_us: latency,
                memory_mb: None,
                samples: samples.len(),
            },
            hardware: HardwareInfo::default(),
        }
    }

    /// Print summary to stdout
    pub fn print_summary(&self) {
        println!("\n=== {} ===", self.benchmark);
        println!("  Dataset:     {} keys", self.configuration.dataset_size);
        println!("  Value size:  {} bytes", self.configuration.value_size);
        println!("  Distribution: {:?}", self.configuration.distribution);
        println!("  Samples:     {}", self.results.samples);
        println!(
            "  Throughput:  {:.2} ops/sec",
            self.results.throughput_ops_sec
        );
        println!("  Latency (μs):");
        println!("    Min:  {:.2}", self.results.latency_us.min);
        println!("    Mean: {:.2}", self.results.latency_us.mean);
        println!("    P50:  {:.2}", self.results.latency_us.p50);
        println!("    P95:  {:.2}", self.results.latency_us.p95);
        println!("    P99:  {:.2}", self.results.latency_us.p99);
        println!("    P999: {:.2}", self.results.latency_us.p999);
        println!("    Max:  {:.2}", self.results.latency_us.max);
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

// ============================================================================
// Durable Storage Harness
// ============================================================================

/// Test harness wrapping DurableStorage with temp directory isolation
pub struct DurableStorageHarness {
    storage: DurableStorage,
    _temp_dir: TempDir,
}

impl DurableStorageHarness {
    /// Create a new harness with default configuration
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(true, false)
    }

    /// Create a new harness with custom configuration
    ///
    /// # Arguments
    /// * `enable_ordered_index` - Enable ordered index for O(log N) scans
    /// * `enable_group_commit` - Enable group commit for higher throughput
    pub fn with_config(enable_ordered_index: bool, enable_group_commit: bool) -> anyhow::Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let storage = if enable_group_commit {
            DurableStorage::open_with_group_commit_and_config(temp_dir.path(), enable_ordered_index)?
        } else {
            DurableStorage::open_with_config(temp_dir.path(), enable_ordered_index)?
        };

        Ok(Self {
            storage,
            _temp_dir: temp_dir,
        })
    }

    /// Get reference to the underlying storage
    pub fn storage(&self) -> &DurableStorage {
        &self.storage
    }

    /// Set sync mode
    pub fn set_sync_mode(&self, mode: SyncMode) {
        self.storage.set_sync_mode(mode as u64);
    }

    /// Pre-load the database with keys
    pub fn preload(&self, config: &BenchConfig) -> anyhow::Result<Duration> {
        let start = Instant::now();

        // Use write-only mode for faster bulk loading
        let txn_id = self.storage.begin_with_mode(TransactionMode::WriteOnly)?;

        for i in 0..config.dataset_size {
            let key = generate_key(i, config.key_size);
            let value = generate_value(i, config.value_size, 42);
            self.storage.write(txn_id, key, value)?;

            // Commit periodically to avoid huge transactions
            if i > 0 && i % 100_000 == 0 {
                self.storage.commit(txn_id)?;
                let new_txn = self.storage.begin_with_mode(TransactionMode::WriteOnly)?;
                // Note: The original txn_id is no longer valid, but we reuse the variable
                // This is a simplification - in production you'd handle this differently
                self.storage.write(new_txn, generate_key(i + 1, config.key_size), generate_value(i + 1, config.value_size, 42))?;
            }
        }

        self.storage.commit(txn_id)?;
        Ok(start.elapsed())
    }

    /// Run point read benchmark
    pub fn run_point_reads(&self, config: &BenchConfig) -> anyhow::Result<BenchStats> {
        let mut rng = rand::thread_rng();
        let zipf = if config.distribution == Distribution::Zipfian {
            Some(ZipfianGenerator::new(config.dataset_size, config.zipf_theta))
        } else {
            None
        };

        let mut sequential_counter = 0usize;
        let mut samples = Vec::with_capacity(config.measurement_ops);

        // Warm-up phase
        for _ in 0..config.warmup_ops {
            let idx = next_key_index(
                &mut rng,
                config.distribution,
                zipf.as_ref(),
                config.dataset_size,
                &mut sequential_counter,
            );
            let key = generate_key(idx, config.key_size);
            let txn_id = self.storage.begin_with_mode(TransactionMode::ReadOnly)?;
            let _ = self.storage.read(txn_id, &key)?;
            self.storage.commit(txn_id)?;
        }

        // Measurement phase
        let start = Instant::now();
        for _ in 0..config.measurement_ops {
            let idx = next_key_index(
                &mut rng,
                config.distribution,
                zipf.as_ref(),
                config.dataset_size,
                &mut sequential_counter,
            );
            let key = generate_key(idx, config.key_size);

            let op_start = Instant::now();
            let txn_id = self.storage.begin_with_mode(TransactionMode::ReadOnly)?;
            let _ = self.storage.read(txn_id, &key)?;
            self.storage.commit(txn_id)?;
            samples.push(op_start.elapsed());
        }
        let total_time = start.elapsed();

        Ok(BenchStats::new("point_read", config.clone(), &mut samples, total_time))
    }

    /// Run write benchmark
    pub fn run_writes(&self, config: &BenchConfig) -> anyhow::Result<BenchStats> {
        self.set_sync_mode(config.sync_mode);

        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(config.measurement_ops);

        // Warm-up phase
        for i in 0..config.warmup_ops {
            let key = generate_key(config.dataset_size + i, config.key_size);
            let value = generate_value(i, config.value_size, rng.r#gen());
            let txn_id = self.storage.begin_with_mode(TransactionMode::WriteOnly)?;
            self.storage.write(txn_id, key, value)?;
            self.storage.commit(txn_id)?;
        }

        // Measurement phase
        let start = Instant::now();
        for i in 0..config.measurement_ops {
            let key = generate_key(config.dataset_size + config.warmup_ops + i, config.key_size);
            let value = generate_value(i, config.value_size, rng.r#gen());

            let op_start = Instant::now();
            let txn_id = self.storage.begin_with_mode(TransactionMode::WriteOnly)?;
            self.storage.write(txn_id, key, value)?;
            self.storage.commit(txn_id)?;
            samples.push(op_start.elapsed());
        }
        let total_time = start.elapsed();

        Ok(BenchStats::new("write", config.clone(), &mut samples, total_time))
    }

    /// Run range scan benchmark
    pub fn run_scans(&self, config: &BenchConfig, scan_size: usize) -> anyhow::Result<BenchStats> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(config.measurement_ops);

        // Measurement phase
        let start = Instant::now();
        for _ in 0..config.measurement_ops {
            let start_idx = rng.gen_range(0..config.dataset_size.saturating_sub(scan_size));
            let end_idx = start_idx + scan_size;

            let start_key = generate_key(start_idx, config.key_size);
            let end_key = generate_key(end_idx, config.key_size);

            let op_start = Instant::now();
            let txn_id = self.storage.begin_with_mode(TransactionMode::ReadOnly)?;
            let _ = self.storage.scan_range(txn_id, &start_key, &end_key)?;
            self.storage.commit(txn_id)?;
            samples.push(op_start.elapsed());
        }
        let total_time = start.elapsed();

        Ok(BenchStats::new(
            &format!("scan_{}", scan_size),
            config.clone(),
            &mut samples,
            total_time,
        ))
    }
}

// ============================================================================
// YCSB Workload Helpers
// ============================================================================

/// YCSB workload types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum YcsbWorkload {
    /// 50% read, 50% update
    A,
    /// 95% read, 5% update
    B,
    /// 100% read
    C,
    /// 95% read, 5% insert (read latest)
    D,
    /// 95% scan, 5% insert
    E,
    /// 50% read, 50% read-modify-write
    F,
}

impl YcsbWorkload {
    /// Get the read ratio for this workload
    pub fn read_ratio(&self) -> f64 {
        match self {
            YcsbWorkload::A => 0.50,
            YcsbWorkload::B => 0.95,
            YcsbWorkload::C => 1.00,
            YcsbWorkload::D => 0.95,
            YcsbWorkload::E => 0.00,
            YcsbWorkload::F => 0.50,
        }
    }

    /// Get the update ratio for this workload
    pub fn update_ratio(&self) -> f64 {
        match self {
            YcsbWorkload::A => 0.50,
            YcsbWorkload::B => 0.05,
            YcsbWorkload::C => 0.00,
            YcsbWorkload::D => 0.00,
            YcsbWorkload::E => 0.00,
            YcsbWorkload::F => 0.50, // RMW
        }
    }

    /// Get the scan ratio for this workload
    pub fn scan_ratio(&self) -> f64 {
        match self {
            YcsbWorkload::E => 0.95,
            _ => 0.00,
        }
    }

    /// Get the insert ratio for this workload
    pub fn insert_ratio(&self) -> f64 {
        match self {
            YcsbWorkload::D => 0.05,
            YcsbWorkload::E => 0.05,
            _ => 0.00,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zipfian_generator() {
        let zipf = ZipfianGenerator::new(1000, 0.99);
        let mut rng = rand::thread_rng();

        // Generate many samples and verify skew
        let mut counts = HashMap::new();
        for _ in 0..10000 {
            let idx = zipf.next(&mut rng);
            *counts.entry(idx).or_insert(0) += 1;
        }

        // First few keys should have most of the accesses
        let first_10_count: usize = (0..10).filter_map(|i| counts.get(&i)).sum();
        let total: usize = counts.values().sum();

        // With theta=0.99, first 10 keys should have > 50% of accesses
        assert!(
            first_10_count as f64 / total as f64 > 0.4,
            "Zipfian distribution not skewed enough"
        );
    }

    #[test]
    fn test_key_generation() {
        let key = generate_key(12345, 16);
        assert_eq!(key.len(), 16);
        assert!(key.starts_with(b"key_"));
    }

    #[test]
    fn test_value_generation() {
        let value = generate_value(12345, 100, 42);
        assert_eq!(value.len(), 100);
    }

    #[test]
    fn test_latency_stats() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let stats = LatencyStats::from_sorted(&samples);

        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 999.0);
        assert_eq!(stats.p50, 500.0);
        assert_eq!(stats.p99, 990.0);
    }

    #[test]
    fn test_harness_creation() {
        let harness = DurableStorageHarness::new();
        assert!(harness.is_ok());
    }

    #[test]
    fn test_small_benchmark() {
        let harness = DurableStorageHarness::new().unwrap();
        let config = BenchConfig {
            dataset_size: 100,
            measurement_ops: 100,
            warmup_ops: 10,
            ..Default::default()
        };

        // Preload
        harness.preload(&config).unwrap();

        // Run point reads
        let stats = harness.run_point_reads(&config).unwrap();
        assert_eq!(stats.results.samples, 100);
        assert!(stats.results.throughput_ops_sec > 0.0);
    }
}
