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

//! Tiered "Hot Buffer + Background Flush" for Ultra-Fast Inserts (LSM-style)
//!
//! This module decouples "acknowledge the write" from "fully integrate into the
//! HNSW structure" using an LSM-style tiered design.
//!
//! ## Problem
//!
//! HNSW insertion is expensive (~100µs per vector) due to:
//! - Graph search to find entry point
//! - Neighbor selection at each layer
//! - Bidirectional edge updates
//!
//! For real-time ingest workloads, this latency is unacceptable.
//!
//! ## Solution: Tiered Hot Buffer
//!
//! ```text
//! Write Path:
//!   insert(id, vec) ──► Hot Buffer (O(1) append)
//!                              │
//!                              ▼ (background flush when full)
//!                        Staged HNSW Builder
//!                              │
//!                              ▼
//!                        Immutable HNSW Graph
//!
//! Read Path:
//!   search(query, k) ──► HNSW Search (O(log N))
//!                              +
//!                        Hot Buffer Scan (O(B × d))
//!                              │
//!                              ▼
//!                        Merged Results
//! ```
//!
//! ## Performance
//!
//! | Operation | Standard HNSW | Hot Buffer |
//! |-----------|---------------|------------|
//! | Insert | ~100µs | ~500ns |
//! | Search (B=1000) | 50µs | 60µs (+10µs buffer scan) |
//! | Memory overhead | 0 | O(B × d) buffer |
//!
//! ## Configuration
//!
//! - Buffer size (B): Trade-off between write speed and query overhead
//! - Flush strategy: Size-based, time-based, or manual
//! - Background threads: 0 (sync flush) or 1+ (async flush)

use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use crossbeam_channel::{bounded, Sender, Receiver};

use crate::hnsw::{HnswConfig, HnswIndex};
use crate::hnsw_staged::{StagedBuilder, StagedConfig, StagedStats};
use crate::vector_quantized::{QuantizedVector, Precision};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for hot buffer HNSW
#[derive(Debug, Clone)]
pub struct HotBufferConfig {
    /// HNSW configuration for the underlying index
    pub hnsw_config: HnswConfig,
    
    /// Maximum hot buffer size before triggering flush
    /// Default: 10,000 vectors
    pub buffer_capacity: usize,
    
    /// Number of background flush threads (0 = sync flush)
    /// Default: 1
    pub flush_threads: usize,
    
    /// Staged builder config for flush operations
    pub staged_config: StagedConfig,
    
    /// Auto-flush when buffer reaches capacity
    /// Default: true
    pub auto_flush: bool,
    
    /// Precision for distance calculations
    pub precision: Precision,
}

impl Default for HotBufferConfig {
    fn default() -> Self {
        Self {
            hnsw_config: HnswConfig::default(),
            buffer_capacity: 10_000,
            flush_threads: 1,
            staged_config: StagedConfig::default(),
            auto_flush: true,
            precision: Precision::F32,
        }
    }
}

impl HotBufferConfig {
    /// Create config for low-latency writes
    pub fn for_low_latency() -> Self {
        Self {
            buffer_capacity: 50_000,
            flush_threads: 2,
            auto_flush: true,
            ..Default::default()
        }
    }
    
    /// Create config for high-throughput batch ingest
    pub fn for_high_throughput() -> Self {
        Self {
            buffer_capacity: 100_000,
            flush_threads: 4,
            auto_flush: true,
            staged_config: StagedConfig::for_large_batch(),
            ..Default::default()
        }
    }
}

// ============================================================================
// Hot Buffer Entry
// ============================================================================

/// A vector waiting in the hot buffer
#[derive(Debug, Clone)]
struct HotBufferEntry {
    /// External ID
    id: u128,
    /// Raw vector data
    vector: Vec<f32>,
    /// Pre-computed quantized form for search
    quantized: QuantizedVector,
}

// ============================================================================
// Hot Buffer HNSW Index
// ============================================================================

/// Statistics for hot buffer operations
#[derive(Debug, Default, Clone)]
pub struct HotBufferStats {
    /// Current buffer size
    pub buffer_size: usize,
    /// Total vectors inserted
    pub total_inserts: usize,
    /// Total flush operations
    pub total_flushes: usize,
    /// Vectors flushed to HNSW
    pub vectors_flushed: usize,
    /// Background flushes in progress
    pub flushes_in_progress: usize,
    /// Total search operations
    pub total_searches: usize,
    /// Searches that hit hot buffer
    pub buffer_hits: usize,
}

/// Hot Buffer HNSW Index with tiered write path
///
/// Provides O(1) insert latency by buffering vectors and flushing
/// to HNSW in the background.
pub struct HotBufferHnsw {
    /// The underlying HNSW index
    hnsw: Arc<HnswIndex>,
    
    /// Hot buffer for pending inserts (protected by RwLock for read-heavy access)
    buffer: RwLock<Vec<HotBufferEntry>>,
    
    /// ID lookup in buffer (for dedup and contains check)
    buffer_ids: RwLock<HashMap<u128, usize>>,
    
    /// Configuration
    config: HotBufferConfig,
    
    /// Vector dimension
    dimension: usize,
    
    /// Statistics
    stats: Mutex<HotBufferStats>,
    
    /// Shutdown flag for background threads
    shutdown: Arc<AtomicBool>,
    
    /// Channel for sending flush requests to background thread
    flush_tx: Option<Sender<FlushRequest>>,
    
    /// Background flush thread handles
    flush_handles: Mutex<Vec<JoinHandle<()>>>,
    
    /// Count of in-progress flushes
    flushes_in_progress: AtomicUsize,
}

/// Request to flush a batch to HNSW
struct FlushRequest {
    entries: Vec<HotBufferEntry>,
}

impl HotBufferHnsw {
    /// Create a new hot buffer HNSW index
    pub fn new(dimension: usize, config: HotBufferConfig) -> Self {
        let hnsw = Arc::new(HnswIndex::new(dimension, config.hnsw_config.clone()));
        let shutdown = Arc::new(AtomicBool::new(false));
        
        // Create flush channel and background threads if configured
        let (flush_tx, flush_handles) = if config.flush_threads > 0 {
            let (tx, rx) = bounded::<FlushRequest>(config.flush_threads * 2);
            let mut handles = Vec::with_capacity(config.flush_threads);
            
            for _ in 0..config.flush_threads {
                let hnsw_clone = Arc::clone(&hnsw);
                let rx_clone = rx.clone();
                let shutdown_clone = Arc::clone(&shutdown);
                let staged_config = config.staged_config.clone();
                
                let handle = thread::spawn(move || {
                    flush_worker(hnsw_clone, rx_clone, shutdown_clone, staged_config);
                });
                handles.push(handle);
            }
            
            (Some(tx), handles)
        } else {
            (None, Vec::new())
        };
        
        Self {
            hnsw,
            buffer: RwLock::new(Vec::with_capacity(config.buffer_capacity)),
            buffer_ids: RwLock::new(HashMap::with_capacity(config.buffer_capacity)),
            config,
            dimension,
            stats: Mutex::new(HotBufferStats::default()),
            shutdown,
            flush_tx,
            flush_handles: Mutex::new(flush_handles),
            flushes_in_progress: AtomicUsize::new(0),
        }
    }
    
    /// Insert a vector into the hot buffer (O(1))
    ///
    /// The vector is NOT immediately added to HNSW. It lives in the hot buffer
    /// until a flush operation moves it to the index.
    pub fn insert(&self, id: u128, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension, vector.len()
            ));
        }
        
        // Check if already in HNSW or buffer
        if self.hnsw.nodes.contains_key(&id) {
            return Err(format!("Vector {} already exists in HNSW", id));
        }
        
        // Pre-compute quantized form for search
        let quantized = QuantizedVector::from_f32(
            ndarray::Array1::from_vec(vector.clone()),
            self.config.precision,
        );
        
        let entry = HotBufferEntry {
            id,
            vector,
            quantized,
        };
        
        // Insert into buffer
        let needs_flush = {
            let mut buffer = self.buffer.write();
            let mut ids = self.buffer_ids.write();
            
            if ids.contains_key(&id) {
                return Err(format!("Vector {} already exists in buffer", id));
            }
            
            let idx = buffer.len();
            buffer.push(entry);
            ids.insert(id, idx);
            
            // Update stats
            {
                let mut stats = self.stats.lock();
                stats.buffer_size = buffer.len();
                stats.total_inserts += 1;
            }
            
            self.config.auto_flush && buffer.len() >= self.config.buffer_capacity
        };
        
        if needs_flush {
            self.trigger_flush()?;
        }
        
        Ok(())
    }
    
    /// Insert multiple vectors (batch O(n) with amortized flush)
    pub fn insert_batch(&self, batch: &[(u128, Vec<f32>)]) -> Result<usize, String> {
        let mut inserted = 0;
        
        for (id, vector) in batch {
            match self.insert(*id, vector.clone()) {
                Ok(()) => inserted += 1,
                Err(_) => continue,
            }
        }
        
        Ok(inserted)
    }
    
    /// Search for k nearest neighbors
    ///
    /// Merges results from:
    /// 1. HNSW graph search (O(ef × log N))
    /// 2. Hot buffer brute-force scan (O(B × d))
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String> {
        if query.len() != self.dimension {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension, query.len()
            ));
        }
        
        let query_quantized = QuantizedVector::from_f32(
            ndarray::Array1::from_vec(query.to_vec()),
            self.config.precision,
        );
        
        // Search HNSW
        let mut hnsw_results = self.hnsw.search(query, k)?;
        
        // Scan hot buffer
        let buffer_results = {
            let buffer = self.buffer.read();
            
            // Update stats
            {
                let mut stats = self.stats.lock();
                stats.total_searches += 1;
                if !buffer.is_empty() {
                    stats.buffer_hits += 1;
                }
            }
            
            let mut results: Vec<(u128, f32)> = buffer
                .iter()
                .map(|entry| {
                    let dist = self.calculate_distance(&query_quantized, &entry.quantized);
                    (entry.id, dist)
                })
                .collect();
            
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(k);
            results
        };
        
        // Merge results
        hnsw_results.extend(buffer_results);
        hnsw_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        hnsw_results.truncate(k);
        
        Ok(hnsw_results)
    }
    
    /// Trigger a flush of the hot buffer to HNSW
    ///
    /// If background threads are configured, this is async.
    /// Otherwise, this blocks until flush completes.
    pub fn trigger_flush(&self) -> Result<(), String> {
        let entries = self.drain_buffer();
        
        if entries.is_empty() {
            return Ok(());
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.total_flushes += 1;
            stats.flushes_in_progress += 1;
        }
        self.flushes_in_progress.fetch_add(1, Ordering::Relaxed);
        
        if let Some(ref tx) = self.flush_tx {
            // Async flush via background thread
            tx.send(FlushRequest { entries }).map_err(|e| e.to_string())?;
        } else {
            // Sync flush
            self.sync_flush(entries)?;
        }
        
        Ok(())
    }
    
    /// Force a synchronous flush (blocks until complete)
    pub fn flush_sync(&self) -> Result<StagedStats, String> {
        let entries = self.drain_buffer();
        
        if entries.is_empty() {
            return Ok(StagedStats::default());
        }
        
        self.sync_flush(entries)
    }
    
    /// Wait for all background flushes to complete
    pub fn wait_for_flushes(&self) {
        while self.flushes_in_progress.load(Ordering::Relaxed) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    
    /// Get current statistics
    pub fn stats(&self) -> HotBufferStats {
        let mut stats = self.stats.lock().clone();
        stats.flushes_in_progress = self.flushes_in_progress.load(Ordering::Relaxed);
        stats.buffer_size = self.buffer.read().len();
        stats
    }
    
    /// Get current buffer size
    pub fn buffer_len(&self) -> usize {
        self.buffer.read().len()
    }
    
    /// Get total HNSW node count
    pub fn hnsw_len(&self) -> usize {
        self.hnsw.nodes.len()
    }
    
    /// Get total vector count (HNSW + buffer)
    pub fn total_len(&self) -> usize {
        self.hnsw_len() + self.buffer_len()
    }
    
    /// Get reference to underlying HNSW index
    pub fn hnsw(&self) -> &HnswIndex {
        &self.hnsw
    }
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    /// Drain the hot buffer and return entries
    fn drain_buffer(&self) -> Vec<HotBufferEntry> {
        let mut buffer = self.buffer.write();
        let mut ids = self.buffer_ids.write();
        
        let entries = std::mem::take(&mut *buffer);
        ids.clear();
        
        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.buffer_size = 0;
        }
        
        entries
    }
    
    /// Perform synchronous flush of entries to HNSW
    fn sync_flush(&self, entries: Vec<HotBufferEntry>) -> Result<StagedStats, String> {
        let batch: Vec<(u128, Vec<f32>)> = entries
            .into_iter()
            .map(|e| (e.id, e.vector))
            .collect();
        
        let builder = StagedBuilder::new(&self.hnsw, self.config.staged_config.clone());
        let (inserted, stats) = builder.insert_batch(&batch)?;
        
        // Update stats
        {
            let mut s = self.stats.lock();
            s.vectors_flushed += inserted;
            s.flushes_in_progress = s.flushes_in_progress.saturating_sub(1);
        }
        self.flushes_in_progress.fetch_sub(1, Ordering::Relaxed);
        
        Ok(stats)
    }
    
    /// Calculate distance between two quantized vectors
    fn calculate_distance(&self, a: &QuantizedVector, b: &QuantizedVector) -> f32 {
        match self.hnsw.config.metric {
            crate::hnsw::DistanceMetric::Euclidean => {
                crate::vector_quantized::euclidean_distance_quantized(a, b)
            }
            crate::hnsw::DistanceMetric::Cosine => {
                crate::vector_quantized::cosine_distance_quantized(a, b)
            }
            crate::hnsw::DistanceMetric::DotProduct => {
                crate::vector_quantized::dot_product_quantized(a, b)
            }
        }
    }
}

impl Drop for HotBufferHnsw {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Release);
        
        // Flush remaining buffer
        let _ = self.flush_sync();
        
        // Wait for background threads
        let mut handles = self.flush_handles.lock();
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
    }
}

// ============================================================================
// Background Flush Worker
// ============================================================================

/// Background worker that processes flush requests
fn flush_worker(
    hnsw: Arc<HnswIndex>,
    rx: Receiver<FlushRequest>,
    shutdown: Arc<AtomicBool>,
    staged_config: StagedConfig,
) {
    while !shutdown.load(Ordering::Acquire) {
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(request) => {
                let batch: Vec<(u128, Vec<f32>)> = request.entries
                    .into_iter()
                    .map(|e| (e.id, e.vector))
                    .collect();
                
                let builder = StagedBuilder::new(&hnsw, staged_config.clone());
                let _ = builder.insert_batch(&batch);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Check shutdown and continue
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::DistanceMetric;
    
    #[test]
    fn test_hot_buffer_basic() {
        let config = HotBufferConfig {
            hnsw_config: HnswConfig {
                max_connections: 16,
                max_connections_layer0: 32,
                ef_construction: 100,
                ef_search: 50,
                metric: DistanceMetric::Euclidean,
                quantization_precision: Some(Precision::F32),
                ..Default::default()
            },
            buffer_capacity: 100,
            flush_threads: 0, // Sync flush for testing
            auto_flush: false,
            ..Default::default()
        };
        
        let index = HotBufferHnsw::new(64, config);
        
        // Insert some vectors
        for i in 0..50 {
            let vec: Vec<f32> = (0..64).map(|d| (i * 10 + d) as f32 / 1000.0).collect();
            index.insert(i as u128, vec).unwrap();
        }
        
        assert_eq!(index.buffer_len(), 50);
        assert_eq!(index.hnsw_len(), 0);
        
        // Flush to HNSW
        let stats = index.flush_sync().unwrap();
        assert!(stats.scaffold_count + stats.wave_count == 50);
        
        assert_eq!(index.buffer_len(), 0);
        assert_eq!(index.hnsw_len(), 50);
    }
    
    #[test]
    fn test_hot_buffer_search() {
        let config = HotBufferConfig {
            hnsw_config: HnswConfig {
                max_connections: 16,
                max_connections_layer0: 32,
                ef_construction: 100,
                ef_search: 50,
                metric: DistanceMetric::Euclidean,
                quantization_precision: Some(Precision::F32),
                ..Default::default()
            },
            buffer_capacity: 200,
            flush_threads: 0,
            auto_flush: false,
            ..Default::default()
        };
        
        let index = HotBufferHnsw::new(64, config);
        
        // Insert vectors into HNSW (via flush)
        for i in 0..50 {
            let vec: Vec<f32> = (0..64).map(|d| (i * 100 + d) as f32).collect();
            index.insert(i as u128, vec).unwrap();
        }
        index.flush_sync().unwrap();
        
        // Insert vectors into buffer (no flush)
        for i in 50..100 {
            let vec: Vec<f32> = (0..64).map(|d| (i * 100 + d) as f32).collect();
            index.insert(i as u128, vec).unwrap();
        }
        
        // Query should find results from both
        let query: Vec<f32> = (0..64).map(|d| (75 * 100 + d) as f32).collect();
        let results = index.search(&query, 10).unwrap();
        
        assert!(!results.is_empty());
        // Result 75 should be in top results (from buffer)
        let ids: Vec<u128> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&75));
    }
    
    #[test]
    fn test_auto_flush() {
        let config = HotBufferConfig {
            hnsw_config: HnswConfig {
                max_connections: 16,
                max_connections_layer0: 32,
                ef_construction: 100,
                ef_search: 50,
                metric: DistanceMetric::Euclidean,
                quantization_precision: Some(Precision::F32),
                ..Default::default()
            },
            buffer_capacity: 50, // Small buffer
            flush_threads: 0,
            auto_flush: true,
            ..Default::default()
        };
        
        let index = HotBufferHnsw::new(64, config);
        
        // Insert more than buffer capacity
        for i in 0..100 {
            let vec: Vec<f32> = (0..64).map(|d| (i * 10 + d) as f32 / 1000.0).collect();
            index.insert(i as u128, vec).unwrap();
        }
        
        // Should have auto-flushed
        assert!(index.hnsw_len() > 0);
        
        // Flush remaining
        index.flush_sync().unwrap();
        
        assert_eq!(index.total_len(), 100);
    }
}
