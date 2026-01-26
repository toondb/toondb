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

//! Hybrid Logical Clock (HLC) for Monotonic Commit Timestamps
//!
//! From mm.md Task 1.3: HLC-Based Transaction Ordering
//!
//! ## Problem
//!
//! Using wall-clock timestamps can violate `commit_ts >= start_ts` due to:
//! - NTP time regression
//! - Clock skew across threads
//! - Ambiguous GC boundaries
//!
//! ## Solution
//!
//! Hybrid Logical Clock provides monotonic timestamps even if physical time regresses.
//!
//! ## Algorithm
//!
//! ```text
//! HLC timestamp: ts = (physical_time << k) | logical_counter
//!
//! On event:
//!   physical = now_micros()
//!   if physical > last_physical:
//!     logical = 0
//!   else:
//!     logical = last_logical + 1
//!   ts = max(last_ts + 1, (physical << 16) | logical)
//!   last_ts = ts
//!
//! Properties:
//! - Monotonic: ts_i < ts_{i+1} always
//! - Causally consistent: if A → B then ts_A < ts_B
//! - Bounded drift: ts - real_time < max_clock_drift
//! ```
//!
//! Cost: O(1) per timestamp allocation

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bits reserved for logical counter (16 bits = 65K events per microsecond)
const LOGICAL_BITS: u32 = 16;
const LOGICAL_MASK: u64 = (1 << LOGICAL_BITS) - 1;

/// Maximum clock drift we tolerate (1 second in microseconds)
const MAX_DRIFT_US: u64 = 1_000_000;

/// Hybrid Logical Clock for monotonic, causally consistent timestamps
///
/// Thread-safe implementation using atomic operations.
///
/// ## Performance
///
/// - Allocation: O(1) amortized, single CAS operation
/// - Memory: 16 bytes (two atomic u64s)
/// - Contention: Low under typical workloads (physical time advances)
#[derive(Debug)]
pub struct HybridLogicalClock {
    /// Last allocated timestamp (physical << 16 | logical)
    last_ts: AtomicU64,
    /// Last physical time seen (microseconds since epoch)
    last_physical: AtomicU64,
}

impl Default for HybridLogicalClock {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridLogicalClock {
    /// Create a new HLC initialized to current time
    pub fn new() -> Self {
        let physical = Self::now_physical();
        let initial_ts = physical << LOGICAL_BITS;
        Self {
            last_ts: AtomicU64::new(initial_ts),
            last_physical: AtomicU64::new(physical),
        }
    }

    /// Create HLC with a specific starting timestamp (for recovery)
    pub fn with_timestamp(ts: u64) -> Self {
        let physical = ts >> LOGICAL_BITS;
        Self {
            last_ts: AtomicU64::new(ts),
            last_physical: AtomicU64::new(physical),
        }
    }

    /// Get current physical time in microseconds
    #[inline]
    fn now_physical() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before UNIX epoch")
            .as_micros() as u64
    }

    /// Allocate the next timestamp (monotonically increasing)
    ///
    /// This is the main API for transaction commit timestamps.
    ///
    /// ## Guarantees
    ///
    /// - Strictly monotonic: result > any previous result
    /// - Causally consistent: happens-before relationships preserved
    /// - Bounded drift: timestamp within MAX_DRIFT_US of real time
    #[inline]
    pub fn next(&self) -> u64 {
        loop {
            let physical = Self::now_physical();
            let last = self.last_ts.load(Ordering::Acquire);
            let last_physical = self.last_physical.load(Ordering::Acquire);

            let new_ts = if physical > last_physical {
                // Physical time advanced - reset logical counter
                physical << LOGICAL_BITS
            } else {
                // Physical time same or regressed - increment logical
                let logical = (last & LOGICAL_MASK) + 1;
                if logical > LOGICAL_MASK {
                    // Logical overflow - wait for physical time to advance
                    std::thread::yield_now();
                    continue;
                }
                (last & !LOGICAL_MASK) | logical
            };

            // Ensure monotonicity
            let new_ts = new_ts.max(last + 1);

            // CAS to update
            if self
                .last_ts
                .compare_exchange_weak(last, new_ts, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Update last physical if we advanced
                if physical > last_physical {
                    self.last_physical.store(physical, Ordering::Release);
                }
                return new_ts;
            }
            // CAS failed, retry
        }
    }

    /// Receive a timestamp from another node (for distributed scenarios)
    ///
    /// Updates local clock to be at least as recent as the received timestamp.
    pub fn receive(&self, remote_ts: u64) {
        loop {
            let last = self.last_ts.load(Ordering::Acquire);
            if remote_ts <= last {
                return; // Already ahead
            }

            let physical = Self::now_physical();
            let remote_physical = remote_ts >> LOGICAL_BITS;

            // Check drift
            if remote_physical > physical + MAX_DRIFT_US {
                // Remote clock too far ahead - could indicate attack or misconfiguration
                // We cap at our physical time + reasonable drift
                let capped = (physical + MAX_DRIFT_US) << LOGICAL_BITS;
                if self
                    .last_ts
                    .compare_exchange_weak(last, capped.max(last + 1), Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return;
                }
            } else {
                // Accept remote timestamp
                if self
                    .last_ts
                    .compare_exchange_weak(last, remote_ts, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.last_physical
                        .fetch_max(remote_physical, Ordering::Release);
                    return;
                }
            }
        }
    }

    /// Get the current timestamp without advancing
    #[inline]
    pub fn current(&self) -> u64 {
        self.last_ts.load(Ordering::Acquire)
    }

    /// Extract physical time component from a timestamp
    #[inline]
    pub fn physical_time(ts: u64) -> u64 {
        ts >> LOGICAL_BITS
    }

    /// Extract logical counter from a timestamp
    #[inline]
    pub fn logical_counter(ts: u64) -> u64 {
        ts & LOGICAL_MASK
    }

    /// Compare two timestamps
    #[inline]
    pub fn compare(a: u64, b: u64) -> std::cmp::Ordering {
        a.cmp(&b)
    }

    /// Check if timestamp a happened before timestamp b
    #[inline]
    pub fn happened_before(a: u64, b: u64) -> bool {
        a < b
    }
}

/// HLC timestamp with named components for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HlcTimestamp {
    /// Raw timestamp value
    pub raw: u64,
}

impl HlcTimestamp {
    /// Create from raw value
    pub const fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// Create from components
    pub const fn new(physical_us: u64, logical: u16) -> Self {
        Self {
            raw: (physical_us << LOGICAL_BITS) | (logical as u64),
        }
    }

    /// Get physical time in microseconds
    pub const fn physical_us(&self) -> u64 {
        self.raw >> LOGICAL_BITS
    }

    /// Get logical counter
    pub const fn logical(&self) -> u16 {
        (self.raw & LOGICAL_MASK) as u16
    }

    /// Convert to raw u64
    pub const fn as_u64(&self) -> u64 {
        self.raw
    }
}

impl std::fmt::Display for HlcTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.physical_us(), self.logical())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_monotonicity() {
        let hlc = HybridLogicalClock::new();
        let mut prev = 0u64;

        for _ in 0..10000 {
            let ts = hlc.next();
            assert!(ts > prev, "Timestamp {} should be > {}", ts, prev);
            prev = ts;
        }
    }

    #[test]
    fn test_concurrent_monotonicity() {
        let hlc = Arc::new(HybridLogicalClock::new());
        let num_threads = 8;
        let ops_per_thread = 10000;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let hlc = Arc::clone(&hlc);
                thread::spawn(move || {
                    let mut timestamps = Vec::with_capacity(ops_per_thread);
                    for _ in 0..ops_per_thread {
                        timestamps.push(hlc.next());
                    }
                    timestamps
                })
            })
            .collect();

        let mut all_timestamps = Vec::new();
        for handle in handles {
            all_timestamps.extend(handle.join().unwrap());
        }

        // All timestamps should be unique
        let unique_count = {
            let mut sorted = all_timestamps.clone();
            sorted.sort();
            sorted.dedup();
            sorted.len()
        };

        assert_eq!(
            unique_count,
            all_timestamps.len(),
            "All timestamps should be unique"
        );
    }

    #[test]
    fn test_receive_advances_clock() {
        let hlc = HybridLogicalClock::new();
        let current = hlc.current();

        // Simulate receiving a future timestamp
        let future_ts = current + 1000;
        hlc.receive(future_ts);

        assert!(hlc.current() >= future_ts);
    }

    #[test]
    fn test_hlc_timestamp_display() {
        let ts = HlcTimestamp::new(1000000, 42);
        assert_eq!(ts.physical_us(), 1000000);
        assert_eq!(ts.logical(), 42);
        assert_eq!(format!("{}", ts), "1000000:42");
    }
}
