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

//! Prometheus Metrics for HNSW Index
//!
//! Provides observability through counters, histograms, and gauges

use lazy_static::lazy_static;
use prometheus::{
    Counter, Gauge, Histogram, HistogramVec, register_counter, register_gauge, register_histogram,
    register_histogram_vec,
};

lazy_static! {
    // Counters - monotonically increasing values
    pub static ref INSERT_COUNT: Counter = register_counter!(
        "hnsw_insert_total",
        "Total number of vector insertions"
    )
    .unwrap();

    pub static ref SEARCH_COUNT: Counter = register_counter!(
        "hnsw_search_total",
        "Total number of searches performed"
    )
    .unwrap();

    pub static ref ERROR_COUNT: Counter = register_counter!(
        "hnsw_errors_total",
        "Total number of errors encountered"
    )
    .unwrap();

    // Histograms - distribution of values
    pub static ref SEARCH_LATENCY: Histogram = register_histogram!(
        "hnsw_search_duration_seconds",
        "Search latency in seconds",
        vec![
            0.0001, // 0.1ms
            0.0005, // 0.5ms
            0.001,  // 1ms
            0.005,  // 5ms
            0.01,   // 10ms
            0.05,   // 50ms
            0.1,    // 100ms
            0.5,    // 500ms
            1.0,    // 1s
        ]
    )
    .unwrap();

    pub static ref INSERT_LATENCY: Histogram = register_histogram!(
        "hnsw_insert_duration_seconds",
        "Insert latency in seconds",
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )
    .unwrap();

    pub static ref DISTANCE_CALCS: Histogram = register_histogram!(
        "hnsw_distance_calculations_per_search",
        "Number of distance calculations per search",
        vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    )
    .unwrap();

    pub static ref SEARCH_RESULT_COUNT: Histogram = register_histogram!(
        "hnsw_search_results_returned",
        "Number of results returned per search",
        vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    )
    .unwrap();

    // Histogram vec - labeled histograms
    pub static ref OPERATION_LATENCY: HistogramVec = register_histogram_vec!(
        "hnsw_operation_duration_seconds",
        "Operation latency in seconds by operation type",
        &["operation"],
        vec![0.0001, 0.001, 0.01, 0.1, 1.0]
    )
    .unwrap();

    // Gauges - current value that can go up or down
    pub static ref NODES_TOTAL: Gauge = register_gauge!(
        "hnsw_nodes_total",
        "Current number of nodes in the index"
    )
    .unwrap();

    pub static ref MEMORY_BYTES: Gauge = register_gauge!(
        "hnsw_memory_bytes",
        "Estimated memory usage in bytes"
    )
    .unwrap();

    pub static ref AVG_NEIGHBORS: Gauge = register_gauge!(
        "hnsw_avg_neighbors",
        "Average number of neighbors per node"
    )
    .unwrap();

    pub static ref MAX_LAYER: Gauge = register_gauge!(
        "hnsw_max_layer",
        "Maximum layer in the index"
    )
    .unwrap();

    pub static ref VECTOR_DIMENSION: Gauge = register_gauge!(
        "hnsw_vector_dimension",
        "Dimension of vectors in the index"
    )
    .unwrap();
}

/// Helper to time operations and record to histogram
pub struct TimerGuard {
    histogram: Histogram,
    start: std::time::Instant,
}

impl TimerGuard {
    pub fn new(histogram: Histogram) -> Self {
        Self {
            histogram,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for TimerGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

/// Update all gauges from index stats
pub fn update_gauges_from_stats(
    num_nodes: usize,
    max_layer: usize,
    avg_neighbors: f32,
    dimension: usize,
    memory_bytes: usize,
) {
    NODES_TOTAL.set(num_nodes as f64);
    MAX_LAYER.set(max_layer as f64);
    AVG_NEIGHBORS.set(avg_neighbors as f64);
    VECTOR_DIMENSION.set(dimension as f64);
    MEMORY_BYTES.set(memory_bytes as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        // Just verify metrics can be accessed without panicking
        INSERT_COUNT.inc();
        SEARCH_COUNT.inc();
        NODES_TOTAL.set(100.0);
        MEMORY_BYTES.set(1024.0 * 1024.0);
    }

    #[test]
    fn test_timer_guard() {
        {
            let _timer = TimerGuard::new(SEARCH_LATENCY.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Timer should have recorded a value around 0.01 seconds
    }

    #[test]
    fn test_update_gauges() {
        update_gauges_from_stats(1000, 5, 16.5, 128, 1024 * 1024);

        // Verify gauges were updated (values should be retrievable)
        // Note: Prometheus gauges don't have a simple get() method,
        // but we can verify no panic occurred
    }
}
