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

//! # Cost Model and Per-Query Budgets (Task 1)
//!
//! This module provides an explicit cost model with enforceable per-query budgets
//! to stabilize p99 latency under load while preserving recall targets.
//!
//! ## Architecture
//!
//! The cost model is "bytes-moved first" and enforces runtime limits on:
//! - RAM bytes scanned for candidate generation
//! - SSD random reads allowed in hot path (ideally 0)
//! - SSD sequential bytes allowed for rerank batching
//! - CPU cycles spent in routing/scan
//!
//! ## Math/Algorithm
//!
//! Constrained optimization: minimize E[bytes scanned] subject to:
//! - P(recall@k ≥ ρ) ≥ 1−δ
//! - p99 ≤ T
//!
//! Convert latency SLA into budgets:
//! - Bytes ≤ BW_eff · T
//! - RandomIO ≤ ⌊T / L_io⌋
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::cost_model::{QueryBudget, CostTracker, AdmissionController};
//!
//! // Define budget for query class
//! let budget = QueryBudget::new("high_recall")
//!     .ram_bytes(16 * 1024 * 1024)  // 16 MB RAM scan
//!     .ssd_random_reads(0)           // No random reads in hot path
//!     .ssd_sequential_bytes(4 * 1024 * 1024)  // 4 MB sequential for rerank
//!     .cpu_cycles(1_000_000_000);    // ~1B cycles
//!
//! // Track costs during query execution
//! let mut tracker = CostTracker::new(budget);
//! tracker.add_ram_bytes(1024);
//! if tracker.is_exhausted() {
//!     // Return best-known results under budget
//! }
//! ```

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Query Budget Definition
// ============================================================================

/// Per-query budget limits derived from SLA targets
#[derive(Debug, Clone)]
pub struct QueryBudget {
    /// Query class identifier (e.g., "low_latency", "high_recall", "balanced")
    pub query_class: String,
    
    /// Maximum RAM bytes scanned for candidate generation
    /// This bounds memory bandwidth consumption
    pub ram_bytes_limit: u64,
    
    /// Maximum SSD random reads in hot path (ideally 0)
    /// Random IO is the main source of p99 variance
    pub ssd_random_reads_limit: u32,
    
    /// Maximum SSD sequential bytes for rerank batching
    /// Sequential IO is more predictable than random
    pub ssd_sequential_bytes_limit: u64,
    
    /// Maximum CPU cycles for routing and scan
    /// Converted from latency target: cycles = latency_ns * freq_ghz
    pub cpu_cycles_limit: u64,
    
    /// Target latency SLA (e.g., p99 ≤ 10ms)
    pub latency_target: Duration,
    
    /// Target recall floor (e.g., recall@10 ≥ 0.95)
    pub recall_target: f32,
    
    /// Probability of meeting recall target (1 - δ)
    pub recall_confidence: f32,
}

impl QueryBudget {
    /// Create a new budget for a query class
    pub fn new(query_class: &str) -> Self {
        Self {
            query_class: query_class.to_string(),
            ram_bytes_limit: u64::MAX,
            ssd_random_reads_limit: u32::MAX,
            ssd_sequential_bytes_limit: u64::MAX,
            cpu_cycles_limit: u64::MAX,
            latency_target: Duration::from_millis(100),
            recall_target: 0.95,
            recall_confidence: 0.99,
        }
    }
    
    /// Set RAM bytes limit
    pub fn ram_bytes(mut self, limit: u64) -> Self {
        self.ram_bytes_limit = limit;
        self
    }
    
    /// Set SSD random reads limit
    pub fn ssd_random_reads(mut self, limit: u32) -> Self {
        self.ssd_random_reads_limit = limit;
        self
    }
    
    /// Set SSD sequential bytes limit
    pub fn ssd_sequential_bytes(mut self, limit: u64) -> Self {
        self.ssd_sequential_bytes_limit = limit;
        self
    }
    
    /// Set CPU cycles limit
    pub fn cpu_cycles(mut self, limit: u64) -> Self {
        self.cpu_cycles_limit = limit;
        self
    }
    
    /// Set latency target
    pub fn latency_target(mut self, target: Duration) -> Self {
        self.latency_target = target;
        self
    }
    
    /// Set recall target
    pub fn recall_target(mut self, target: f32, confidence: f32) -> Self {
        self.recall_target = target;
        self.recall_confidence = confidence;
        self
    }
    
    /// Create budget from SLA parameters
    ///
    /// Converts latency SLA into resource budgets:
    /// - Bytes ≤ BW_eff · T
    /// - RandomIO ≤ ⌊T / L_io⌋
    pub fn from_sla(
        query_class: &str,
        latency_target: Duration,
        recall_target: f32,
        hardware: &HardwareProfile,
    ) -> Self {
        let t_ns = latency_target.as_nanos() as u64;
        
        // Bytes ≤ BW_eff · T
        // Assume ~50% of latency budget for memory operations
        let ram_bytes = (hardware.ram_bandwidth_gbps as u64 * t_ns / 2) / 1_000_000_000;
        
        // RandomIO ≤ ⌊T / L_io⌋
        // Each random IO takes ~100μs on SSD
        let ssd_random = (t_ns / hardware.ssd_random_latency_ns) as u32;
        
        // Sequential bytes based on SSD bandwidth
        let ssd_seq = (hardware.ssd_seq_bandwidth_mbps as u64 * t_ns) / 1_000_000_000;
        
        // CPU cycles = latency_ns * freq_ghz
        let cpu_cycles = t_ns * hardware.cpu_freq_ghz as u64;
        
        Self {
            query_class: query_class.to_string(),
            ram_bytes_limit: ram_bytes,
            ssd_random_reads_limit: ssd_random,
            ssd_sequential_bytes_limit: ssd_seq,
            cpu_cycles_limit: cpu_cycles,
            latency_target,
            recall_target,
            recall_confidence: 0.99,
        }
    }
    
    /// Predefined budget for low-latency queries (p99 ≤ 5ms)
    pub fn low_latency() -> Self {
        Self::new("low_latency")
            .ram_bytes(4 * 1024 * 1024)      // 4 MB
            .ssd_random_reads(0)              // No random IO
            .ssd_sequential_bytes(0)          // No SSD access
            .cpu_cycles(500_000_000)          // ~0.5B cycles
            .latency_target(Duration::from_millis(5))
            .recall_target(0.80, 0.95)
    }
    
    /// Predefined budget for balanced queries (p99 ≤ 20ms)
    pub fn balanced() -> Self {
        Self::new("balanced")
            .ram_bytes(16 * 1024 * 1024)     // 16 MB
            .ssd_random_reads(0)              // No random IO
            .ssd_sequential_bytes(2 * 1024 * 1024)  // 2 MB sequential
            .cpu_cycles(2_000_000_000)        // ~2B cycles
            .latency_target(Duration::from_millis(20))
            .recall_target(0.90, 0.99)
    }
    
    /// Predefined budget for high-recall queries (p99 ≤ 100ms)
    pub fn high_recall() -> Self {
        Self::new("high_recall")
            .ram_bytes(64 * 1024 * 1024)     // 64 MB
            .ssd_random_reads(16)             // Limited random IO
            .ssd_sequential_bytes(8 * 1024 * 1024)  // 8 MB sequential
            .cpu_cycles(10_000_000_000)       // ~10B cycles
            .latency_target(Duration::from_millis(100))
            .recall_target(0.99, 0.999)
    }
}

// ============================================================================
// Hardware Profile
// ============================================================================

/// Hardware characteristics for SLA-to-budget conversion
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// RAM bandwidth in GB/s
    pub ram_bandwidth_gbps: f32,
    
    /// SSD random read latency in nanoseconds
    pub ssd_random_latency_ns: u64,
    
    /// SSD sequential read bandwidth in MB/s
    pub ssd_seq_bandwidth_mbps: u32,
    
    /// CPU frequency in GHz
    pub cpu_freq_ghz: f32,
    
    /// LLC (Last-Level Cache) size in bytes
    pub llc_size_bytes: usize,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            ram_bandwidth_gbps: 50.0,        // Typical DDR4
            ssd_random_latency_ns: 100_000,  // 100μs for NVMe
            ssd_seq_bandwidth_mbps: 3000,    // 3 GB/s NVMe
            cpu_freq_ghz: 3.5,               // Typical server CPU
            llc_size_bytes: 32 * 1024 * 1024, // 32 MB LLC
        }
    }
}

impl HardwareProfile {
    /// Profile for high-end server (AWS c6i.8xlarge equivalent)
    pub fn high_end_server() -> Self {
        Self {
            ram_bandwidth_gbps: 100.0,
            ssd_random_latency_ns: 50_000,
            ssd_seq_bandwidth_mbps: 5000,
            cpu_freq_ghz: 3.8,
            llc_size_bytes: 48 * 1024 * 1024,
        }
    }
    
    /// Profile for standard server (AWS c6i.2xlarge equivalent)
    pub fn standard_server() -> Self {
        Self::default()
    }
    
    /// Profile for embedded/edge deployment
    pub fn embedded() -> Self {
        Self {
            ram_bandwidth_gbps: 25.0,
            ssd_random_latency_ns: 200_000,
            ssd_seq_bandwidth_mbps: 500,
            cpu_freq_ghz: 2.0,
            llc_size_bytes: 8 * 1024 * 1024,
        }
    }
}

// ============================================================================
// Cost Tracker
// ============================================================================

/// Tracks resource consumption during query execution
#[derive(Debug)]
pub struct CostTracker {
    /// Budget being enforced
    budget: QueryBudget,
    
    /// RAM bytes consumed
    ram_bytes: AtomicU64,
    
    /// SSD random reads performed
    ssd_random_reads: AtomicU64,
    
    /// SSD sequential bytes read
    ssd_sequential_bytes: AtomicU64,
    
    /// Estimated CPU cycles consumed
    cpu_cycles: AtomicU64,
    
    /// Query start time
    start_time: Instant,
    
    /// Whether budget is exhausted
    exhausted: std::sync::atomic::AtomicBool,
    
    /// Reason for exhaustion (if any)
    exhaustion_reason: parking_lot::Mutex<Option<BudgetExhaustionReason>>,
}

/// Reason why budget was exhausted
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetExhaustionReason {
    RamBytesExceeded,
    SsdRandomReadsExceeded,
    SsdSequentialBytesExceeded,
    CpuCyclesExceeded,
    LatencyTargetExceeded,
}

impl CostTracker {
    /// Create a new cost tracker with the given budget
    pub fn new(budget: QueryBudget) -> Self {
        Self {
            budget,
            ram_bytes: AtomicU64::new(0),
            ssd_random_reads: AtomicU64::new(0),
            ssd_sequential_bytes: AtomicU64::new(0),
            cpu_cycles: AtomicU64::new(0),
            start_time: Instant::now(),
            exhausted: std::sync::atomic::AtomicBool::new(false),
            exhaustion_reason: parking_lot::Mutex::new(None),
        }
    }
    
    /// Add RAM bytes consumed
    #[inline]
    pub fn add_ram_bytes(&self, bytes: u64) -> bool {
        let new_total = self.ram_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if new_total > self.budget.ram_bytes_limit {
            self.mark_exhausted(BudgetExhaustionReason::RamBytesExceeded);
            false
        } else {
            true
        }
    }
    
    /// Add SSD random read
    #[inline]
    pub fn add_ssd_random_read(&self) -> bool {
        let new_total = self.ssd_random_reads.fetch_add(1, Ordering::Relaxed) + 1;
        if new_total > self.budget.ssd_random_reads_limit as u64 {
            self.mark_exhausted(BudgetExhaustionReason::SsdRandomReadsExceeded);
            false
        } else {
            true
        }
    }
    
    /// Add SSD sequential bytes
    #[inline]
    pub fn add_ssd_sequential_bytes(&self, bytes: u64) -> bool {
        let new_total = self.ssd_sequential_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if new_total > self.budget.ssd_sequential_bytes_limit {
            self.mark_exhausted(BudgetExhaustionReason::SsdSequentialBytesExceeded);
            false
        } else {
            true
        }
    }
    
    /// Add CPU cycles
    #[inline]
    pub fn add_cpu_cycles(&self, cycles: u64) -> bool {
        let new_total = self.cpu_cycles.fetch_add(cycles, Ordering::Relaxed) + cycles;
        if new_total > self.budget.cpu_cycles_limit {
            self.mark_exhausted(BudgetExhaustionReason::CpuCyclesExceeded);
            false
        } else {
            true
        }
    }
    
    /// Check if latency budget is exceeded
    #[inline]
    pub fn check_latency(&self) -> bool {
        if self.start_time.elapsed() > self.budget.latency_target {
            self.mark_exhausted(BudgetExhaustionReason::LatencyTargetExceeded);
            false
        } else {
            true
        }
    }
    
    /// Mark budget as exhausted
    fn mark_exhausted(&self, reason: BudgetExhaustionReason) {
        self.exhausted.store(true, Ordering::Release);
        let mut guard = self.exhaustion_reason.lock();
        if guard.is_none() {
            *guard = Some(reason);
        }
    }
    
    /// Check if budget is exhausted
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        // Also check latency on every call
        if self.start_time.elapsed() > self.budget.latency_target {
            self.mark_exhausted(BudgetExhaustionReason::LatencyTargetExceeded);
        }
        self.exhausted.load(Ordering::Acquire)
    }
    
    /// Get exhaustion reason if budget is exhausted
    pub fn exhaustion_reason(&self) -> Option<BudgetExhaustionReason> {
        *self.exhaustion_reason.lock()
    }
    
    /// Get remaining RAM bytes budget
    pub fn remaining_ram_bytes(&self) -> u64 {
        self.budget.ram_bytes_limit.saturating_sub(self.ram_bytes.load(Ordering::Relaxed))
    }
    
    /// Get remaining SSD random reads budget
    pub fn remaining_ssd_random_reads(&self) -> u32 {
        self.budget.ssd_random_reads_limit
            .saturating_sub(self.ssd_random_reads.load(Ordering::Relaxed) as u32)
    }
    
    /// Get remaining time budget
    pub fn remaining_time(&self) -> Duration {
        self.budget.latency_target
            .saturating_sub(self.start_time.elapsed())
    }
    
    /// Get utilization ratios for all resources
    pub fn utilization(&self) -> CostUtilization {
        CostUtilization {
            ram_bytes_ratio: self.ram_bytes.load(Ordering::Relaxed) as f64
                / self.budget.ram_bytes_limit.max(1) as f64,
            ssd_random_reads_ratio: self.ssd_random_reads.load(Ordering::Relaxed) as f64
                / self.budget.ssd_random_reads_limit.max(1) as f64,
            ssd_sequential_bytes_ratio: self.ssd_sequential_bytes.load(Ordering::Relaxed) as f64
                / self.budget.ssd_sequential_bytes_limit.max(1) as f64,
            cpu_cycles_ratio: self.cpu_cycles.load(Ordering::Relaxed) as f64
                / self.budget.cpu_cycles_limit.max(1) as f64,
            latency_ratio: self.start_time.elapsed().as_nanos() as f64
                / self.budget.latency_target.as_nanos().max(1) as f64,
        }
    }
    
    /// Generate a summary for telemetry
    pub fn summary(&self) -> CostSummary {
        CostSummary {
            query_class: self.budget.query_class.clone(),
            ram_bytes_used: self.ram_bytes.load(Ordering::Relaxed),
            ram_bytes_limit: self.budget.ram_bytes_limit,
            ssd_random_reads_used: self.ssd_random_reads.load(Ordering::Relaxed) as u32,
            ssd_random_reads_limit: self.budget.ssd_random_reads_limit,
            ssd_sequential_bytes_used: self.ssd_sequential_bytes.load(Ordering::Relaxed),
            ssd_sequential_bytes_limit: self.budget.ssd_sequential_bytes_limit,
            cpu_cycles_used: self.cpu_cycles.load(Ordering::Relaxed),
            cpu_cycles_limit: self.budget.cpu_cycles_limit,
            elapsed: self.start_time.elapsed(),
            latency_target: self.budget.latency_target,
            exhausted: self.is_exhausted(),
            exhaustion_reason: self.exhaustion_reason(),
        }
    }
}

/// Resource utilization ratios
#[derive(Debug, Clone)]
pub struct CostUtilization {
    pub ram_bytes_ratio: f64,
    pub ssd_random_reads_ratio: f64,
    pub ssd_sequential_bytes_ratio: f64,
    pub cpu_cycles_ratio: f64,
    pub latency_ratio: f64,
}

/// Summary of cost consumption
#[derive(Debug, Clone)]
pub struct CostSummary {
    pub query_class: String,
    pub ram_bytes_used: u64,
    pub ram_bytes_limit: u64,
    pub ssd_random_reads_used: u32,
    pub ssd_random_reads_limit: u32,
    pub ssd_sequential_bytes_used: u64,
    pub ssd_sequential_bytes_limit: u64,
    pub cpu_cycles_used: u64,
    pub cpu_cycles_limit: u64,
    pub elapsed: Duration,
    pub latency_target: Duration,
    pub exhausted: bool,
    pub exhaustion_reason: Option<BudgetExhaustionReason>,
}

// ============================================================================
// Admission Controller
// ============================================================================

/// Admission controller for backpressure under concurrency
///
/// Enforces system-wide limits to prevent individual query budgets from
/// being violated due to resource contention.
pub struct AdmissionController {
    /// Maximum concurrent queries per query class
    max_concurrent_per_class: parking_lot::RwLock<std::collections::HashMap<String, usize>>,
    
    /// Current active queries per class
    active_per_class: parking_lot::RwLock<std::collections::HashMap<String, AtomicUsize>>,
    
    /// Global memory pressure (bytes currently in-flight)
    global_memory_pressure: AtomicU64,
    
    /// Maximum global memory pressure before backpressure
    max_global_memory: u64,
    
    /// Backpressure wait time
    backpressure_wait: Duration,
}

/// Handle returned when a query is admitted
pub struct AdmissionTicket {
    query_class: String,
    estimated_memory: u64,
    controller: Arc<AdmissionController>,
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        self.controller.release(&self.query_class, self.estimated_memory);
    }
}

impl AdmissionController {
    /// Create a new admission controller
    pub fn new(max_global_memory: u64) -> Arc<Self> {
        Arc::new(Self {
            max_concurrent_per_class: parking_lot::RwLock::new(std::collections::HashMap::new()),
            active_per_class: parking_lot::RwLock::new(std::collections::HashMap::new()),
            global_memory_pressure: AtomicU64::new(0),
            max_global_memory,
            backpressure_wait: Duration::from_millis(10),
        })
    }
    
    /// Set maximum concurrent queries for a class
    pub fn set_class_limit(self: &Arc<Self>, query_class: &str, max_concurrent: usize) {
        self.max_concurrent_per_class.write()
            .insert(query_class.to_string(), max_concurrent);
    }
    
    /// Try to admit a query
    ///
    /// Returns None if admission is denied (should backpressure)
    pub fn try_admit(
        self: &Arc<Self>,
        budget: &QueryBudget,
    ) -> Option<AdmissionTicket> {
        // Check class limit
        let class_limits = self.max_concurrent_per_class.read();
        if let Some(&limit) = class_limits.get(&budget.query_class) {
            let mut active = self.active_per_class.write();
            let counter = active
                .entry(budget.query_class.clone())
                .or_insert_with(|| AtomicUsize::new(0));
            
            let current = counter.load(Ordering::Acquire);
            if current >= limit {
                return None;
            }
            counter.fetch_add(1, Ordering::AcqRel);
        }
        drop(class_limits);
        
        // Check global memory
        let estimated_memory = budget.ram_bytes_limit / 2; // Conservative estimate
        let current = self.global_memory_pressure.fetch_add(estimated_memory, Ordering::AcqRel);
        
        if current + estimated_memory > self.max_global_memory {
            // Roll back
            self.global_memory_pressure.fetch_sub(estimated_memory, Ordering::AcqRel);
            self.release_class_counter(&budget.query_class);
            return None;
        }
        
        Some(AdmissionTicket {
            query_class: budget.query_class.clone(),
            estimated_memory,
            controller: Arc::clone(self),
        })
    }
    
    /// Admit a query, waiting with backpressure if necessary
    pub fn admit_with_backpressure(
        self: &Arc<Self>,
        budget: &QueryBudget,
        max_wait: Duration,
    ) -> Option<AdmissionTicket> {
        let deadline = Instant::now() + max_wait;
        
        loop {
            if let Some(ticket) = self.try_admit(budget) {
                return Some(ticket);
            }
            
            if Instant::now() >= deadline {
                return None;
            }
            
            std::thread::sleep(self.backpressure_wait);
        }
    }
    
    /// Release resources when query completes
    fn release(&self, query_class: &str, estimated_memory: u64) {
        self.global_memory_pressure.fetch_sub(estimated_memory, Ordering::AcqRel);
        self.release_class_counter(query_class);
    }
    
    fn release_class_counter(&self, query_class: &str) {
        let active = self.active_per_class.read();
        if let Some(counter) = active.get(query_class) {
            counter.fetch_sub(1, Ordering::AcqRel);
        }
    }
    
    /// Get current system pressure metrics
    pub fn metrics(&self) -> AdmissionMetrics {
        let active = self.active_per_class.read();
        let active_per_class: std::collections::HashMap<String, usize> = active
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();
        
        AdmissionMetrics {
            global_memory_pressure: self.global_memory_pressure.load(Ordering::Relaxed),
            max_global_memory: self.max_global_memory,
            memory_utilization: self.global_memory_pressure.load(Ordering::Relaxed) as f64
                / self.max_global_memory as f64,
            active_per_class,
        }
    }
}

/// Metrics from admission controller
#[derive(Debug, Clone)]
pub struct AdmissionMetrics {
    pub global_memory_pressure: u64,
    pub max_global_memory: u64,
    pub memory_utilization: f64,
    pub active_per_class: std::collections::HashMap<String, usize>,
}

// ============================================================================
// Query Class Registry
// ============================================================================

/// Registry of query classes with their budgets
pub struct QueryClassRegistry {
    classes: parking_lot::RwLock<std::collections::HashMap<String, QueryBudget>>,
    hardware: HardwareProfile,
}

impl QueryClassRegistry {
    /// Create a new registry with default classes
    pub fn new(hardware: HardwareProfile) -> Self {
        let mut classes = std::collections::HashMap::new();
        classes.insert("low_latency".to_string(), QueryBudget::low_latency());
        classes.insert("balanced".to_string(), QueryBudget::balanced());
        classes.insert("high_recall".to_string(), QueryBudget::high_recall());
        
        Self {
            classes: parking_lot::RwLock::new(classes),
            hardware,
        }
    }
    
    /// Register a custom query class
    pub fn register(&self, budget: QueryBudget) {
        self.classes.write().insert(budget.query_class.clone(), budget);
    }
    
    /// Get budget for a query class
    pub fn get(&self, query_class: &str) -> Option<QueryBudget> {
        self.classes.read().get(query_class).cloned()
    }
    
    /// Create a custom budget from SLA parameters
    pub fn create_from_sla(
        &self,
        query_class: &str,
        latency_target: Duration,
        recall_target: f32,
    ) -> QueryBudget {
        QueryBudget::from_sla(query_class, latency_target, recall_target, &self.hardware)
    }
}

impl Default for QueryClassRegistry {
    fn default() -> Self {
        Self::new(HardwareProfile::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_budget_creation() {
        let budget = QueryBudget::new("test")
            .ram_bytes(1024 * 1024)
            .ssd_random_reads(10)
            .latency_target(Duration::from_millis(50));
        
        assert_eq!(budget.query_class, "test");
        assert_eq!(budget.ram_bytes_limit, 1024 * 1024);
        assert_eq!(budget.ssd_random_reads_limit, 10);
    }
    
    #[test]
    fn test_cost_tracker() {
        let budget = QueryBudget::new("test")
            .ram_bytes(1000)
            .ssd_random_reads(2);
        
        let tracker = CostTracker::new(budget);
        
        assert!(tracker.add_ram_bytes(500));
        assert!(!tracker.is_exhausted());
        
        assert!(tracker.add_ram_bytes(400));
        assert!(!tracker.is_exhausted());
        
        // This should exceed
        assert!(!tracker.add_ram_bytes(200));
        assert!(tracker.is_exhausted());
        assert_eq!(tracker.exhaustion_reason(), Some(BudgetExhaustionReason::RamBytesExceeded));
    }
    
    #[test]
    fn test_admission_controller() {
        let controller = AdmissionController::new(1024 * 1024);
        controller.set_class_limit("low_latency", 2);
        
        let budget = QueryBudget::low_latency();
        
        let ticket1 = controller.try_admit(&budget);
        assert!(ticket1.is_some());
        
        let ticket2 = controller.try_admit(&budget);
        assert!(ticket2.is_some());
        
        // Third should be rejected
        let ticket3 = controller.try_admit(&budget);
        assert!(ticket3.is_none());
        
        // Drop one ticket
        drop(ticket1);
        
        // Now should be admitted
        let ticket4 = controller.try_admit(&budget);
        assert!(ticket4.is_some());
    }
    
    #[test]
    fn test_budget_from_sla() {
        let hardware = HardwareProfile::default();
        let budget = QueryBudget::from_sla(
            "custom",
            Duration::from_millis(50),
            0.95,
            &hardware,
        );
        
        assert!(budget.ram_bytes_limit > 0);
        assert!(budget.cpu_cycles_limit > 0);
    }
}
