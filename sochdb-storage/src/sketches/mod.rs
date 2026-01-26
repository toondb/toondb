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

//! Probabilistic Data Structures for Streaming Analytics
//!
//! This module provides memory-efficient streaming algorithms for:
//! - DDSketch: O(1) percentile queries (P50, P90, P95, P99)
//! - HyperLogLog: Cardinality estimation (unique counts)
//! - ExponentialHistogram: Mergeable histograms for rollups
//! - CountMinSketch: Frequency estimation for top-K queries
//! - AdaptiveSketch: Memory-efficient latency tracking (sparse → dense)

pub mod adaptive_sketch;
pub mod count_min_sketch;
pub mod ddsketch;
pub mod exponential_histogram;
pub mod hyperloglog;

pub use adaptive_sketch::{AdaptiveSketch, SketchPercentiles, SparseBuffer};
pub use count_min_sketch::CountMinSketch;
pub use ddsketch::DDSketch;
pub use exponential_histogram::ExponentialHistogram;
pub use hyperloglog::HyperLogLog;
