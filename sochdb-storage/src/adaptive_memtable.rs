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

//! Adaptive Memtable Sizing with Memory Pressure Feedback
//!
//! This module implements Task 10 from mm.md: dynamic memtable sizing that
//! responds to system memory pressure and write rate.
//!
//! ## Problem: Fixed Memtable Size is Suboptimal
//!
//! The current memtable uses a fixed 4MB flush threshold (`memtable_flush_size`).
//! This one-size-fits-all approach is suboptimal because:
//!
//! - **Too small:** Frequent flushes cause I/O overhead and write amplification
//! - **Too large:** Memory pressure, long recovery times, increased GC pause
//!
//! The optimal size depends on write rate, available memory, and durability
//! requirements—all of which vary at runtime.
//!
//! ## Solution: Adaptive Sizing with Feedback Control
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  Adaptive Memtable Sizer                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
//! │  │ Write Rate   │    │ Memory       │    │ Target Size  │       │
//! │  │ Estimator    │───▶│ Pressure     │───▶│ Calculator   │       │
//! │  │ (EMA)        │    │ Monitor      │    │              │       │
//! │  └──────────────┘    └──────────────┘    └──────────────┘       │
//! │                                                 │                │
//! │                                                 ▼                │
//! │                      ┌──────────────────────────────────────────┐│
//! │                      │ target = write_rate × 1.0s               ││
//! │                      │ adjusted = target × (1 - memory_pressure²)│
//! │                      │ final = clamp(adjusted, base/4, base×4)  ││
//! │                      └──────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//!
//! Goals:
//! - Optimal memory utilization across varying workloads
//! - Reduced flush frequency during low-memory conditions
//! - Faster recovery (smaller memtable = less WAL replay)
//! ```
//!
//! ## Feedback Controller
//!
//! ```text
//! target = write_rate_bytes_per_sec × 1.0  // 1 second buffer
//! adjusted = target × (1 - memory_pressure²)
//! final = clamp(adjusted, base/4, base×4)  // 1MB to 16MB
//! ```
//!
//! Memory pressure signal: Read from /proc/meminfo (Linux) or
//! mach_host_statistics (macOS). Pressure = 1 - (available / total).

use std::sync::atomic::{AtomicU64, Ordering};

/// Default base memtable size: 4MB
pub const DEFAULT_BASE_SIZE: usize = 4 * 1024 * 1024;

/// Minimum memtable size: 1MB
pub const MIN_MEMTABLE_SIZE: usize = 1 * 1024 * 1024;

/// Maximum memtable size: 16MB
pub const MAX_MEMTABLE_SIZE: usize = 16 * 1024 * 1024;

/// Target buffer duration in seconds
/// Memtable should hold approximately this much time of write throughput
pub const TARGET_BUFFER_SECONDS: f64 = 1.0;

/// EMA alpha for write rate estimation (higher = more responsive)
pub const WRITE_RATE_EMA_ALPHA: f64 = 0.1;

/// Memory pressure threshold above which we start reducing memtable size
pub const PRESSURE_THRESHOLD: f64 = 0.7;

/// Configuration for adaptive memtable sizing
#[derive(Debug, Clone)]
pub struct AdaptiveMemtableConfig {
    /// Base size in bytes (default: 4MB)
    pub base_size: usize,
    /// Minimum allowed size (default: 1MB)
    pub min_size: usize,
    /// Maximum allowed size (default: 16MB)
    pub max_size: usize,
    /// Target buffer duration in seconds (default: 1.0)
    pub target_buffer_seconds: f64,
    /// EMA alpha for write rate (default: 0.1)
    pub ema_alpha: f64,
    /// Whether to enable memory pressure feedback
    pub enable_memory_pressure: bool,
}

impl Default for AdaptiveMemtableConfig {
    fn default() -> Self {
        Self {
            base_size: DEFAULT_BASE_SIZE,
            min_size: MIN_MEMTABLE_SIZE,
            max_size: MAX_MEMTABLE_SIZE,
            target_buffer_seconds: TARGET_BUFFER_SECONDS,
            ema_alpha: WRITE_RATE_EMA_ALPHA,
            enable_memory_pressure: true,
        }
    }
}

/// Adaptive memtable sizer with memory pressure feedback
///
/// Dynamically adjusts memtable flush threshold based on:
/// 1. Write rate (to maintain target buffer duration)
/// 2. Memory pressure (to avoid OOM)
pub struct AdaptiveMemtableSizer {
    /// Configuration
    config: AdaptiveMemtableConfig,
    /// Current adaptive size (bytes)
    current_size: AtomicU64,
    /// Estimated write rate (bytes per second × 1000 for precision)
    write_rate_ema: AtomicU64,
    /// Last update timestamp (microseconds since epoch)
    last_update_us: AtomicU64,
    /// Total bytes written since last update
    bytes_since_update: AtomicU64,
    /// Last memory pressure reading (0-1000 scaled)
    memory_pressure_scaled: AtomicU64,
}

impl AdaptiveMemtableSizer {
    /// Create a new adaptive sizer with default configuration
    pub fn new() -> Self {
        Self::with_config(AdaptiveMemtableConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AdaptiveMemtableConfig) -> Self {
        let initial_rate = config.base_size as f64 / config.target_buffer_seconds;
        
        Self {
            current_size: AtomicU64::new(config.base_size as u64),
            write_rate_ema: AtomicU64::new((initial_rate * 1000.0) as u64),
            last_update_us: AtomicU64::new(now_us()),
            bytes_since_update: AtomicU64::new(0),
            memory_pressure_scaled: AtomicU64::new(0),
            config,
        }
    }

    /// Record bytes written (call on every write)
    #[inline]
    pub fn record_write(&self, bytes: usize) {
        self.bytes_since_update.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Get current recommended memtable size
    #[inline]
    pub fn current_size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed) as usize
    }

    /// Get estimated write rate in bytes per second
    #[inline]
    pub fn write_rate(&self) -> f64 {
        self.write_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Get current memory pressure (0.0 - 1.0)
    #[inline]
    pub fn memory_pressure(&self) -> f64 {
        self.memory_pressure_scaled.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Update the adaptive size (call periodically, e.g., every second)
    ///
    /// Returns the new recommended memtable size
    pub fn update(&self) -> usize {
        let now = now_us();
        let last = self.last_update_us.swap(now, Ordering::Relaxed);
        let delta_us = now.saturating_sub(last);

        if delta_us == 0 {
            return self.current_size();
        }

        // Calculate instantaneous write rate
        let bytes = self.bytes_since_update.swap(0, Ordering::Relaxed);
        let delta_secs = delta_us as f64 / 1_000_000.0;
        let instant_rate = bytes as f64 / delta_secs;

        // Update EMA of write rate
        let old_rate = self.write_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0;
        let new_rate = old_rate * (1.0 - self.config.ema_alpha) + instant_rate * self.config.ema_alpha;
        self.write_rate_ema.store((new_rate * 1000.0) as u64, Ordering::Relaxed);

        // Update memory pressure if enabled
        let pressure = if self.config.enable_memory_pressure {
            get_memory_pressure()
        } else {
            0.0
        };
        self.memory_pressure_scaled.store((pressure * 1000.0) as u64, Ordering::Relaxed);

        // Calculate target size based on write rate
        // Target = write_rate × buffer_duration
        let target_size = new_rate * self.config.target_buffer_seconds;

        // Adjust for memory pressure (quadratic dampening)
        // When pressure is high, we reduce memtable size more aggressively
        let pressure_factor = if pressure > PRESSURE_THRESHOLD {
            // Above threshold, apply quadratic reduction
            1.0 - (pressure - PRESSURE_THRESHOLD).powi(2)
        } else {
            1.0
        };

        let adjusted_size = target_size * pressure_factor;

        // Clamp to configured bounds
        let final_size = adjusted_size
            .max(self.config.min_size as f64)
            .min(self.config.max_size as f64) as usize;

        self.current_size.store(final_size as u64, Ordering::Relaxed);
        final_size
    }

    /// Check if the memtable should be flushed based on current size
    #[inline]
    pub fn should_flush(&self, current_memtable_bytes: u64) -> bool {
        current_memtable_bytes >= self.current_size.load(Ordering::Relaxed)
    }

    /// Get statistics for monitoring
    pub fn stats(&self) -> AdaptiveMemtableStats {
        AdaptiveMemtableStats {
            current_size: self.current_size(),
            write_rate_bytes_per_sec: self.write_rate(),
            memory_pressure: self.memory_pressure(),
            config: self.config.clone(),
        }
    }
}

impl Default for AdaptiveMemtableSizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for adaptive memtable sizing
#[derive(Debug, Clone)]
pub struct AdaptiveMemtableStats {
    /// Current recommended memtable size
    pub current_size: usize,
    /// Estimated write rate in bytes per second
    pub write_rate_bytes_per_sec: f64,
    /// Current memory pressure (0.0 - 1.0)
    pub memory_pressure: f64,
    /// Current configuration
    pub config: AdaptiveMemtableConfig,
}

// ============================================================================
// Platform-specific memory pressure detection
// ============================================================================

/// Get current memory pressure (0.0 = no pressure, 1.0 = critical)
#[cfg(target_os = "linux")]
fn get_memory_pressure() -> f64 {
    // Read from /proc/meminfo
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = match File::open("/proc/meminfo") {
        Ok(f) => f,
        Err(_) => return 0.0,
    };

    let reader = BufReader::new(file);
    let mut mem_total: u64 = 0;
    let mut mem_available: u64 = 0;

    for line in reader.lines().take(10).flatten() {
        if line.starts_with("MemTotal:") {
            mem_total = parse_meminfo_value(&line);
        } else if line.starts_with("MemAvailable:") {
            mem_available = parse_meminfo_value(&line);
        }
    }

    if mem_total == 0 {
        return 0.0;
    }

    // Pressure = 1 - (available / total)
    1.0 - (mem_available as f64 / mem_total as f64)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_value(line: &str) -> u64 {
    // Format: "MemTotal:       16384000 kB"
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn get_memory_pressure() -> f64 {
    // On macOS, we can use mach APIs for memory info
    // For simplicity, we use sysctl to get vm.page_free_count and related values
    use std::process::Command;

    let output = match Command::new("vm_stat").output() {
        Ok(o) => o,
        Err(_) => return 0.0,
    };

    let stdout = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(_) => return 0.0,
    };

    let mut free_pages: u64 = 0;
    let mut active_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut speculative_pages: u64 = 0;
    let mut wired_pages: u64 = 0;

    for line in stdout.lines() {
        if line.contains("Pages free:") {
            free_pages = extract_vm_stat_value(line);
        } else if line.contains("Pages active:") {
            active_pages = extract_vm_stat_value(line);
        } else if line.contains("Pages inactive:") {
            inactive_pages = extract_vm_stat_value(line);
        } else if line.contains("Pages speculative:") {
            speculative_pages = extract_vm_stat_value(line);
        } else if line.contains("Pages wired down:") {
            wired_pages = extract_vm_stat_value(line);
        }
    }

    // Total = active + inactive + speculative + free + wired
    // Available = free + inactive (approximately)
    let total = active_pages + inactive_pages + speculative_pages + free_pages + wired_pages;
    let available = free_pages + inactive_pages;

    if total == 0 {
        return 0.0;
    }

    1.0 - (available as f64 / total as f64)
}

#[cfg(target_os = "macos")]
fn extract_vm_stat_value(line: &str) -> u64 {
    // Format: "Pages free:                                3142."
    line.split(':')
        .nth(1)
        .map(|s| s.trim().trim_end_matches('.'))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn get_memory_pressure() -> f64 {
    // Default: assume no pressure on unsupported platforms
    0.0
}

// ============================================================================
// Utility functions
// ============================================================================

/// Get current time in microseconds since epoch
#[inline]
fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AdaptiveMemtableConfig::default();
        assert_eq!(config.base_size, DEFAULT_BASE_SIZE);
        assert_eq!(config.min_size, MIN_MEMTABLE_SIZE);
        assert_eq!(config.max_size, MAX_MEMTABLE_SIZE);
    }

    #[test]
    fn test_initial_size() {
        let sizer = AdaptiveMemtableSizer::new();
        assert_eq!(sizer.current_size(), DEFAULT_BASE_SIZE);
    }

    #[test]
    fn test_record_write() {
        let sizer = AdaptiveMemtableSizer::new();
        
        sizer.record_write(1000);
        sizer.record_write(2000);
        
        // Bytes accumulate until update
        assert_eq!(sizer.bytes_since_update.load(Ordering::Relaxed), 3000);
    }

    #[test]
    fn test_should_flush() {
        let sizer = AdaptiveMemtableSizer::new();
        
        // Initially at base size (4MB)
        assert!(!sizer.should_flush(1_000_000)); // 1MB < 4MB
        assert!(sizer.should_flush(5_000_000));  // 5MB >= 4MB
    }

    #[test]
    fn test_write_rate_update() {
        let sizer = AdaptiveMemtableSizer::new();
        
        // Simulate writing 1MB over ~1 second
        sizer.record_write(1_000_000);
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let new_size = sizer.update();
        
        // Size should adjust based on write rate
        assert!(new_size >= MIN_MEMTABLE_SIZE);
        assert!(new_size <= MAX_MEMTABLE_SIZE);
    }

    #[test]
    fn test_custom_config() {
        let config = AdaptiveMemtableConfig {
            base_size: 8 * 1024 * 1024,
            min_size: 2 * 1024 * 1024,
            max_size: 32 * 1024 * 1024,
            target_buffer_seconds: 2.0,
            ema_alpha: 0.2,
            enable_memory_pressure: false,
        };

        let sizer = AdaptiveMemtableSizer::with_config(config);
        assert_eq!(sizer.current_size(), 8 * 1024 * 1024);
    }

    #[test]
    fn test_memory_pressure() {
        // Just ensure it doesn't crash
        let pressure = get_memory_pressure();
        assert!(pressure >= 0.0);
        assert!(pressure <= 1.0);
    }

    #[test]
    fn test_stats() {
        let sizer = AdaptiveMemtableSizer::new();
        let stats = sizer.stats();
        
        assert_eq!(stats.current_size, DEFAULT_BASE_SIZE);
        assert!(stats.write_rate_bytes_per_sec > 0.0);
        assert!(stats.memory_pressure >= 0.0);
    }
}
