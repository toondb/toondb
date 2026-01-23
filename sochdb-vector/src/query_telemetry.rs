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

//! # Per-Query Telemetry (Task 10)
//!
//! Structured telemetry for every query to enable:
//! - Optimization: falsifiable hypotheses about performance
//! - Regression detection: automatic SLA monitoring
//! - Explainability: "why was this query slow?"
//!
//! ## Metrics Captured
//!
//! - Routing: time, lists considered/scanned
//! - Scan: codes evaluated, RAM bytes read
//! - Rerank: candidates, SSD ops/bytes
//! - Cache: hit ratio
//! - Error: estimated ε envelope used
//! - Stop: termination mode and reason
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::query_telemetry::{QueryTelemetry, TelemetryCollector};
//!
//! let mut telemetry = QueryTelemetry::new("search_v1");
//! telemetry.record_routing(Duration::from_micros(500), 100, 16);
//! telemetry.record_scan(1024, 16 * 1024 * 1024);
//! telemetry.set_stop_reason(StopReason::BoundSatisfied);
//! 
//! // Emit structured telemetry
//! let json = telemetry.to_json();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use crate::guarantee_ladder::{GuaranteeMode, StopReason};
use crate::cost_model::CostSummary;

// ============================================================================
// Query Telemetry
// ============================================================================

/// Comprehensive per-query telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTelemetry {
    /// Query identifier (for correlation)
    pub query_id: String,
    
    /// Query class (e.g., "low_latency", "high_recall")
    pub query_class: String,
    
    /// Timestamp when query started
    #[serde(skip)]
    pub start_time: Option<Instant>,
    
    /// Total query duration
    pub total_duration_us: u64,
    
    /// Routing phase metrics
    pub routing: RoutingMetrics,
    
    /// Scan phase metrics
    pub scan: ScanMetrics,
    
    /// Rerank phase metrics
    pub rerank: RerankMetrics,
    
    /// Cache metrics
    pub cache: CacheMetrics,
    
    /// Error envelope metrics
    pub error_envelope: ErrorEnvelopeMetrics,
    
    /// Termination metrics
    pub termination: TerminationMetrics,
    
    /// Cost summary (if budget tracking enabled)
    pub cost: Option<CostSummaryJson>,
    
    /// Custom tags for filtering/grouping
    pub tags: HashMap<String, String>,
}

/// Routing phase metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingMetrics {
    /// Time spent in routing phase
    pub duration_us: u64,
    
    /// Total lists/partitions considered
    pub lists_considered: u32,
    
    /// Lists actually scanned
    pub lists_scanned: u32,
    
    /// Centroid comparisons performed
    pub centroid_comparisons: u32,
    
    /// Whether routing used compressed centroids
    pub used_compressed_centroids: bool,
    
    /// Routing strategy used
    pub strategy: String,
}

/// Scan phase metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanMetrics {
    /// Time spent in scan phase
    pub duration_us: u64,
    
    /// Number of codes/vectors evaluated
    pub codes_evaluated: u64,
    
    /// RAM bytes read
    pub ram_bytes_read: u64,
    
    /// Number of SIMD operations
    pub simd_ops: u64,
    
    /// Vectors passing first-stage filter
    pub candidates_after_stage1: u32,
    
    /// Distance metric used
    pub distance_metric: String,
    
    /// Quantization level used
    pub quant_level: String,
}

/// Rerank phase metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankMetrics {
    /// Time spent in rerank phase
    pub duration_us: u64,
    
    /// Candidates entering rerank
    pub candidates_in: u32,
    
    /// Candidates after rerank
    pub candidates_out: u32,
    
    /// SSD random read operations
    pub ssd_random_reads: u32,
    
    /// SSD sequential bytes read
    pub ssd_sequential_bytes: u64,
    
    /// Whether IO was coalesced
    pub io_coalesced: bool,
    
    /// Number of IO ranges after coalescing
    pub coalesced_ranges: u32,
    
    /// Full-precision distance computations
    pub full_precision_distances: u32,
}

/// Cache metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Centroid cache hits
    pub centroid_cache_hits: u32,
    
    /// Centroid cache misses
    pub centroid_cache_misses: u32,
    
    /// Vector cache hits
    pub vector_cache_hits: u32,
    
    /// Vector cache misses
    pub vector_cache_misses: u32,
    
    /// Distance cache hits
    pub distance_cache_hits: u32,
    
    /// Distance cache misses
    pub distance_cache_misses: u32,
}

impl CacheMetrics {
    /// Compute overall cache hit ratio
    pub fn hit_ratio(&self) -> f32 {
        let total_hits = self.centroid_cache_hits + self.vector_cache_hits + self.distance_cache_hits;
        let total_misses = self.centroid_cache_misses + self.vector_cache_misses + self.distance_cache_misses;
        let total = total_hits + total_misses;
        if total == 0 {
            1.0
        } else {
            total_hits as f32 / total as f32
        }
    }
}

/// Error envelope metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorEnvelopeMetrics {
    /// Guarantee mode used
    pub guarantee_mode: String,
    
    /// Error quantile used (for calibrated mode)
    pub error_quantile: Option<f32>,
    
    /// Maximum error bound observed
    pub max_error_observed: f32,
    
    /// Mean error bound
    pub mean_error: f32,
    
    /// Number of candidates with tight bounds
    pub tight_bound_candidates: u32,
    
    /// Number of candidates with loose bounds
    pub loose_bound_candidates: u32,
}

/// Termination metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TerminationMetrics {
    /// Stop reason code
    pub stop_reason: String,
    
    /// Probes completed when stopped
    pub probes_at_stop: u32,
    
    /// Max probes allowed
    pub max_probes: u32,
    
    /// Whether budget was exhausted
    pub budget_exhausted: bool,
    
    /// Estimated miss probability (for calibrated mode)
    pub miss_probability: Option<f32>,
    
    /// Final result count
    pub result_count: u32,
}

/// JSON-serializable cost summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummaryJson {
    pub query_class: String,
    pub ram_bytes_used: u64,
    pub ram_bytes_limit: u64,
    pub ssd_random_reads_used: u32,
    pub ssd_random_reads_limit: u32,
    pub ssd_sequential_bytes_used: u64,
    pub ssd_sequential_bytes_limit: u64,
    pub cpu_cycles_used: u64,
    pub cpu_cycles_limit: u64,
    pub elapsed_us: u64,
    pub latency_target_us: u64,
    pub exhausted: bool,
    pub exhaustion_reason: Option<String>,
}

impl From<CostSummary> for CostSummaryJson {
    fn from(summary: CostSummary) -> Self {
        Self {
            query_class: summary.query_class,
            ram_bytes_used: summary.ram_bytes_used,
            ram_bytes_limit: summary.ram_bytes_limit,
            ssd_random_reads_used: summary.ssd_random_reads_used,
            ssd_random_reads_limit: summary.ssd_random_reads_limit,
            ssd_sequential_bytes_used: summary.ssd_sequential_bytes_used,
            ssd_sequential_bytes_limit: summary.ssd_sequential_bytes_limit,
            cpu_cycles_used: summary.cpu_cycles_used,
            cpu_cycles_limit: summary.cpu_cycles_limit,
            elapsed_us: summary.elapsed.as_micros() as u64,
            latency_target_us: summary.latency_target.as_micros() as u64,
            exhausted: summary.exhausted,
            exhaustion_reason: summary.exhaustion_reason.map(|r| format!("{:?}", r)),
        }
    }
}

impl QueryTelemetry {
    /// Create new telemetry for a query
    pub fn new(query_class: &str) -> Self {
        Self {
            query_id: uuid_v4(),
            query_class: query_class.to_string(),
            start_time: Some(Instant::now()),
            total_duration_us: 0,
            routing: RoutingMetrics::default(),
            scan: ScanMetrics::default(),
            rerank: RerankMetrics::default(),
            cache: CacheMetrics::default(),
            error_envelope: ErrorEnvelopeMetrics::default(),
            termination: TerminationMetrics::default(),
            cost: None,
            tags: HashMap::new(),
        }
    }
    
    /// Create with specific query ID
    pub fn with_id(query_id: &str, query_class: &str) -> Self {
        let mut t = Self::new(query_class);
        t.query_id = query_id.to_string();
        t
    }
    
    /// Record routing phase
    pub fn record_routing(
        &mut self,
        duration: Duration,
        lists_considered: u32,
        lists_scanned: u32,
    ) {
        self.routing.duration_us = duration.as_micros() as u64;
        self.routing.lists_considered = lists_considered;
        self.routing.lists_scanned = lists_scanned;
    }
    
    /// Record routing with full details
    pub fn record_routing_full(
        &mut self,
        duration: Duration,
        lists_considered: u32,
        lists_scanned: u32,
        centroid_comparisons: u32,
        used_compressed: bool,
        strategy: &str,
    ) {
        self.routing.duration_us = duration.as_micros() as u64;
        self.routing.lists_considered = lists_considered;
        self.routing.lists_scanned = lists_scanned;
        self.routing.centroid_comparisons = centroid_comparisons;
        self.routing.used_compressed_centroids = used_compressed;
        self.routing.strategy = strategy.to_string();
    }
    
    /// Record scan phase
    pub fn record_scan(&mut self, codes_evaluated: u64, ram_bytes: u64) {
        self.scan.codes_evaluated = codes_evaluated;
        self.scan.ram_bytes_read = ram_bytes;
    }
    
    /// Record scan with full details
    pub fn record_scan_full(
        &mut self,
        duration: Duration,
        codes_evaluated: u64,
        ram_bytes: u64,
        simd_ops: u64,
        candidates_stage1: u32,
        distance_metric: &str,
        quant_level: &str,
    ) {
        self.scan.duration_us = duration.as_micros() as u64;
        self.scan.codes_evaluated = codes_evaluated;
        self.scan.ram_bytes_read = ram_bytes;
        self.scan.simd_ops = simd_ops;
        self.scan.candidates_after_stage1 = candidates_stage1;
        self.scan.distance_metric = distance_metric.to_string();
        self.scan.quant_level = quant_level.to_string();
    }
    
    /// Record rerank phase
    pub fn record_rerank(
        &mut self,
        duration: Duration,
        candidates_in: u32,
        candidates_out: u32,
        ssd_random_reads: u32,
        ssd_sequential_bytes: u64,
    ) {
        self.rerank.duration_us = duration.as_micros() as u64;
        self.rerank.candidates_in = candidates_in;
        self.rerank.candidates_out = candidates_out;
        self.rerank.ssd_random_reads = ssd_random_reads;
        self.rerank.ssd_sequential_bytes = ssd_sequential_bytes;
    }
    
    /// Record IO coalescing details
    pub fn record_io_coalescing(&mut self, coalesced: bool, ranges: u32) {
        self.rerank.io_coalesced = coalesced;
        self.rerank.coalesced_ranges = ranges;
    }
    
    /// Record cache hits/misses
    pub fn record_cache_hit(&mut self, cache_type: CacheType) {
        match cache_type {
            CacheType::Centroid => self.cache.centroid_cache_hits += 1,
            CacheType::Vector => self.cache.vector_cache_hits += 1,
            CacheType::Distance => self.cache.distance_cache_hits += 1,
        }
    }
    
    /// Record cache miss
    pub fn record_cache_miss(&mut self, cache_type: CacheType) {
        match cache_type {
            CacheType::Centroid => self.cache.centroid_cache_misses += 1,
            CacheType::Vector => self.cache.vector_cache_misses += 1,
            CacheType::Distance => self.cache.distance_cache_misses += 1,
        }
    }
    
    /// Set guarantee mode
    pub fn set_guarantee_mode(&mut self, mode: &GuaranteeMode) {
        self.error_envelope.guarantee_mode = format!("{:?}", mode);
        self.error_envelope.error_quantile = mode.error_quantile();
    }
    
    /// Record error bounds observed
    pub fn record_error_bounds(&mut self, max_error: f32, mean_error: f32) {
        self.error_envelope.max_error_observed = max_error;
        self.error_envelope.mean_error = mean_error;
    }
    
    /// Set stop reason
    pub fn set_stop_reason(&mut self, reason: StopReason, probes: u32, max_probes: u32) {
        self.termination.stop_reason = format!("{:?}", reason);
        self.termination.probes_at_stop = probes;
        self.termination.max_probes = max_probes;
        self.termination.budget_exhausted = matches!(reason, StopReason::BudgetExhausted);
    }
    
    /// Set miss probability
    pub fn set_miss_probability(&mut self, prob: f32) {
        self.termination.miss_probability = Some(prob);
    }
    
    /// Set result count
    pub fn set_result_count(&mut self, count: u32) {
        self.termination.result_count = count;
    }
    
    /// Attach cost summary
    pub fn attach_cost(&mut self, summary: CostSummary) {
        self.cost = Some(summary.into());
    }
    
    /// Add a custom tag
    pub fn add_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }
    
    /// Finalize telemetry (compute total duration)
    pub fn finalize(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.total_duration_us = start.elapsed().as_micros() as u64;
        }
    }
    
    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
    
    /// Serialize to pretty JSON
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Cache type for hit/miss tracking
#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    Centroid,
    Vector,
    Distance,
}

// ============================================================================
// Telemetry Collector
// ============================================================================

/// Thread-safe telemetry collector with aggregation
pub struct TelemetryCollector {
    /// Collected telemetry entries
    entries: parking_lot::RwLock<Vec<QueryTelemetry>>,
    
    /// Maximum entries to keep in memory
    max_entries: usize,
    
    /// Callback for emitting telemetry
    emit_callback: parking_lot::RwLock<Option<Box<dyn Fn(&QueryTelemetry) + Send + Sync>>>,
}

impl TelemetryCollector {
    /// Create new collector
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: parking_lot::RwLock::new(Vec::with_capacity(max_entries)),
            max_entries,
            emit_callback: parking_lot::RwLock::new(None),
        }
    }
    
    /// Set callback for emitting telemetry
    pub fn set_emit_callback<F>(&self, callback: F)
    where
        F: Fn(&QueryTelemetry) + Send + Sync + 'static,
    {
        *self.emit_callback.write() = Some(Box::new(callback));
    }
    
    /// Record telemetry
    pub fn record(&self, mut telemetry: QueryTelemetry) {
        telemetry.finalize();
        
        // Emit via callback
        if let Some(callback) = &*self.emit_callback.read() {
            callback(&telemetry);
        }
        
        // Store in memory
        let mut entries = self.entries.write();
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        entries.push(telemetry);
    }
    
    /// Get recent entries
    pub fn recent(&self, count: usize) -> Vec<QueryTelemetry> {
        let entries = self.entries.read();
        let start = entries.len().saturating_sub(count);
        entries[start..].to_vec()
    }
    
    /// Compute aggregate statistics
    pub fn aggregate(&self) -> TelemetryAggregate {
        let entries = self.entries.read();
        
        if entries.is_empty() {
            return TelemetryAggregate::default();
        }
        
        let n = entries.len();
        let mut durations: Vec<u64> = entries.iter().map(|e| e.total_duration_us).collect();
        durations.sort_unstable();
        
        let total_duration: u64 = durations.iter().sum();
        let p50 = durations[n / 2];
        let p99 = durations[(n * 99) / 100];
        let max = durations[n - 1];
        
        let total_ram_bytes: u64 = entries.iter().map(|e| e.scan.ram_bytes_read).sum();
        let total_codes: u64 = entries.iter().map(|e| e.scan.codes_evaluated).sum();
        
        let budget_exhausted = entries.iter().filter(|e| e.termination.budget_exhausted).count();
        
        TelemetryAggregate {
            query_count: n,
            mean_duration_us: total_duration / n as u64,
            p50_duration_us: p50,
            p99_duration_us: p99,
            max_duration_us: max,
            total_ram_bytes_read: total_ram_bytes,
            total_codes_evaluated: total_codes,
            budget_exhausted_count: budget_exhausted,
            cache_hit_ratio: entries.iter()
                .map(|e| e.cache.hit_ratio())
                .sum::<f32>() / n as f32,
        }
    }
    
    /// Clear all entries
    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Aggregate telemetry statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetryAggregate {
    pub query_count: usize,
    pub mean_duration_us: u64,
    pub p50_duration_us: u64,
    pub p99_duration_us: u64,
    pub max_duration_us: u64,
    pub total_ram_bytes_read: u64,
    pub total_codes_evaluated: u64,
    pub budget_exhausted_count: usize,
    pub cache_hit_ratio: f32,
}

// ============================================================================
// Helpers
// ============================================================================

/// Generate a simple UUID-like string
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", now)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_telemetry_creation() {
        let mut telemetry = QueryTelemetry::new("test");
        
        telemetry.record_routing(Duration::from_micros(500), 100, 16);
        telemetry.record_scan(10000, 16 * 1024 * 1024);
        telemetry.record_rerank(Duration::from_micros(1000), 100, 10, 0, 0);
        
        telemetry.finalize();
        
        assert!(telemetry.total_duration_us > 0);
        assert_eq!(telemetry.routing.lists_considered, 100);
        assert_eq!(telemetry.scan.codes_evaluated, 10000);
    }
    
    #[test]
    fn test_telemetry_json() {
        let mut telemetry = QueryTelemetry::new("balanced");
        telemetry.record_routing(Duration::from_micros(100), 50, 8);
        telemetry.finalize();
        
        let json = telemetry.to_json();
        assert!(json.contains("balanced"));
        assert!(json.contains("lists_considered"));
    }
    
    #[test]
    fn test_collector() {
        let collector = TelemetryCollector::new(100);
        
        for i in 0..10 {
            let mut t = QueryTelemetry::new("test");
            t.total_duration_us = i * 100;
            collector.record(t);
        }
        
        let recent = collector.recent(5);
        assert_eq!(recent.len(), 5);
        
        let agg = collector.aggregate();
        assert_eq!(agg.query_count, 10);
    }
    
    #[test]
    fn test_cache_hit_ratio() {
        let mut cache = CacheMetrics::default();
        cache.centroid_cache_hits = 80;
        cache.centroid_cache_misses = 20;
        
        assert!((cache.hit_ratio() - 0.8).abs() < 0.01);
    }
}
