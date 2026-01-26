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

//! Embedding Pipeline
//!
//! This module provides a batched embedding pipeline that processes
//! texts asynchronously in the background.
//!
//! ## Architecture
//!
//! ```text
//! Text Input → Queue → Batch Accumulator → Provider → Normalizer → Output
//! ```
//!
//! ## Features
//!
//! - **Batching**: Accumulates texts for efficient batch processing
//! - **Timeout**: Flushes partial batches after configurable timeout
//! - **Backpressure**: Bounded queue prevents memory exhaustion
//! - **Non-blocking**: Submit returns immediately, processing is async
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_index::embedding::{EmbeddingPipeline, PipelineConfig, LocalEmbeddingProvider};
//! use std::sync::Arc;
//!
//! let provider = Arc::new(LocalEmbeddingProvider::default_provider()?);
//! let (pipeline, receiver) = EmbeddingPipeline::new(provider, PipelineConfig::default())?;
//!
//! // Start background worker
//! let handle = pipeline.start_worker();
//!
//! // Submit texts for embedding
//! pipeline.submit(EmbeddingRequest { id: 1, text: "hello world".to_string() })?;
//!
//! // Receive results
//! while let Ok(result) = receiver.recv() {
//!     println!("Embedded id={}: {} dims", result.id, result.embedding.len());
//! }
//! ```

use super::normalize::normalize_l2_simd;
use super::provider::{EmbeddingError, EmbeddingProvider};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError, bounded};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for the embedding pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum texts per batch
    pub batch_size: usize,

    /// Maximum wait time before flushing partial batch (ms)
    pub batch_timeout_ms: u64,

    /// Maximum pending requests in queue
    pub max_pending: usize,

    /// Whether to L2 normalize embeddings
    pub normalize: bool,

    /// Number of worker threads (typically 1)
    pub num_workers: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            batch_timeout_ms: 100,
            max_pending: 1000,
            normalize: true,
            num_workers: 1,
        }
    }
}

impl PipelineConfig {
    /// Create config optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            batch_size: 8,
            batch_timeout_ms: 50,
            ..Default::default()
        }
    }

    /// Create config optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            batch_size: 64,
            batch_timeout_ms: 200,
            ..Default::default()
        }
    }
}

/// Request to embed text
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Unique identifier for tracking
    pub id: u128,

    /// Text to embed
    pub text: String,

    /// Optional metadata (passed through to result)
    pub metadata: Option<Vec<u8>>,
}

impl EmbeddingRequest {
    /// Create a simple request
    pub fn new(id: u128, text: String) -> Self {
        Self {
            id,
            text,
            metadata: None,
        }
    }
}

/// Result of embedding operation
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Request ID
    pub id: u128,

    /// Computed embedding (L2 normalized if configured)
    pub embedding: Vec<f32>,

    /// Pass-through metadata
    pub metadata: Option<Vec<u8>>,

    /// Processing time in microseconds
    pub latency_us: u64,

    /// Error if embedding failed
    pub error: Option<String>,
}

/// Statistics for the embedding pipeline
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// Total requests submitted
    pub submitted: AtomicU64,

    /// Total requests completed
    pub completed: AtomicU64,

    /// Total requests failed
    pub failed: AtomicU64,

    /// Total batches processed
    pub batches: AtomicU64,

    /// Total processing time (microseconds)
    pub total_latency_us: AtomicU64,

    /// Currently pending requests
    pub pending: AtomicU64,
}

impl PipelineStats {
    /// Get average latency per embedding
    pub fn avg_latency_us(&self) -> f64 {
        let completed = self.completed.load(Ordering::Relaxed);
        if completed == 0 {
            return 0.0;
        }
        self.total_latency_us.load(Ordering::Relaxed) as f64 / completed as f64
    }

    /// Get throughput (embeddings per second)
    pub fn throughput(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            return 0.0;
        }
        self.completed.load(Ordering::Relaxed) as f64 / elapsed_secs
    }
}

/// Embedding pipeline with background processing
pub struct EmbeddingPipeline {
    /// Channel for submitting requests
    sender: Sender<EmbeddingRequest>,

    /// Embedding provider
    provider: Arc<dyn EmbeddingProvider>,

    /// Configuration (kept for introspection)
    #[allow(dead_code)]
    config: PipelineConfig,

    /// Statistics
    stats: Arc<PipelineStats>,

    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

impl EmbeddingPipeline {
    /// Create a new embedding pipeline
    ///
    /// Returns the pipeline and a receiver for results
    pub fn new(
        provider: Arc<dyn EmbeddingProvider>,
        config: PipelineConfig,
    ) -> Result<(Self, Receiver<EmbeddingResult>), EmbeddingError> {
        let (input_tx, input_rx) = bounded(config.max_pending);
        let (output_tx, output_rx) = bounded(config.max_pending);

        let pipeline = Self {
            sender: input_tx,
            provider: Arc::clone(&provider),
            config: config.clone(),
            stats: Arc::new(PipelineStats::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
        };

        // Start worker thread
        let worker_provider = Arc::clone(&provider);
        let worker_stats = Arc::clone(&pipeline.stats);
        let worker_shutdown = Arc::clone(&pipeline.shutdown);
        let worker_config = config;

        thread::spawn(move || {
            Self::worker_loop(
                input_rx,
                output_tx,
                worker_provider,
                worker_config,
                worker_stats,
                worker_shutdown,
            );
        });

        Ok((pipeline, output_rx))
    }

    /// Submit a request for embedding
    ///
    /// Non-blocking. Returns error if queue is full.
    pub fn submit(&self, request: EmbeddingRequest) -> Result<(), EmbeddingError> {
        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
        self.stats.pending.fetch_add(1, Ordering::Relaxed);

        match self.sender.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                self.stats.pending.fetch_sub(1, Ordering::Relaxed);
                Err(EmbeddingError::InvalidInput(
                    "Pipeline queue is full".to_string(),
                ))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.stats.pending.fetch_sub(1, Ordering::Relaxed);
                Err(EmbeddingError::InferenceFailed(
                    "Pipeline worker has stopped".to_string(),
                ))
            }
        }
    }

    /// Submit a request, blocking if queue is full
    pub fn submit_blocking(&self, request: EmbeddingRequest) -> Result<(), EmbeddingError> {
        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
        self.stats.pending.fetch_add(1, Ordering::Relaxed);

        self.sender.send(request).map_err(|_| {
            self.stats.pending.fetch_sub(1, Ordering::Relaxed);
            EmbeddingError::InferenceFailed("Pipeline worker has stopped".to_string())
        })
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get number of pending requests
    pub fn pending(&self) -> u64 {
        self.stats.pending.load(Ordering::Relaxed)
    }

    /// Shutdown the pipeline gracefully
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Get provider dimension
    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    /// Worker loop that processes batches
    fn worker_loop(
        input: Receiver<EmbeddingRequest>,
        output: Sender<EmbeddingResult>,
        provider: Arc<dyn EmbeddingProvider>,
        config: PipelineConfig,
        stats: Arc<PipelineStats>,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut batch: Vec<EmbeddingRequest> = Vec::with_capacity(config.batch_size);
        let timeout = Duration::from_millis(config.batch_timeout_ms);
        let mut last_flush = Instant::now();

        loop {
            // Check shutdown
            if shutdown.load(Ordering::SeqCst) && batch.is_empty() {
                break;
            }

            // Try to receive with timeout
            match input.recv_timeout(timeout) {
                Ok(request) => {
                    batch.push(request);

                    // Flush if batch full
                    if batch.len() >= config.batch_size {
                        Self::process_batch(
                            &mut batch,
                            &provider,
                            &output,
                            &stats,
                            config.normalize,
                        );
                        last_flush = Instant::now();
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Flush partial batch on timeout
                    if !batch.is_empty() && last_flush.elapsed() >= timeout {
                        Self::process_batch(
                            &mut batch,
                            &provider,
                            &output,
                            &stats,
                            config.normalize,
                        );
                        last_flush = Instant::now();
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel closed, process remaining and exit
                    if !batch.is_empty() {
                        Self::process_batch(
                            &mut batch,
                            &provider,
                            &output,
                            &stats,
                            config.normalize,
                        );
                    }
                    break;
                }
            }
        }
    }

    /// Process a batch of requests
    fn process_batch(
        batch: &mut Vec<EmbeddingRequest>,
        provider: &Arc<dyn EmbeddingProvider>,
        output: &Sender<EmbeddingResult>,
        stats: &Arc<PipelineStats>,
        normalize: bool,
    ) {
        if batch.is_empty() {
            return;
        }

        let start = Instant::now();
        let batch_size = batch.len();

        // Extract texts for batch embedding
        let texts: Vec<&str> = batch.iter().map(|r| r.text.as_str()).collect();

        // Call provider
        let embeddings_result = provider.embed_batch(&texts);

        let elapsed_us = start.elapsed().as_micros() as u64;
        let per_item_us = elapsed_us / batch_size as u64;

        match embeddings_result {
            Ok(mut embeddings) => {
                // Normalize if configured
                if normalize {
                    for emb in &mut embeddings {
                        normalize_l2_simd(emb);
                    }
                }

                // Send results
                for (request, embedding) in batch.drain(..).zip(embeddings) {
                    let result = EmbeddingResult {
                        id: request.id,
                        embedding,
                        metadata: request.metadata,
                        latency_us: per_item_us,
                        error: None,
                    };

                    stats.completed.fetch_add(1, Ordering::Relaxed);
                    stats.pending.fetch_sub(1, Ordering::Relaxed);
                    stats
                        .total_latency_us
                        .fetch_add(per_item_us, Ordering::Relaxed);

                    let _ = output.send(result);
                }

                stats.batches.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                // Send error results
                let error_msg = e.to_string();
                for request in batch.drain(..) {
                    let result = EmbeddingResult {
                        id: request.id,
                        embedding: Vec::new(),
                        metadata: request.metadata,
                        latency_us: per_item_us,
                        error: Some(error_msg.clone()),
                    };

                    stats.failed.fetch_add(1, Ordering::Relaxed);
                    stats.pending.fetch_sub(1, Ordering::Relaxed);

                    let _ = output.send(result);
                }
            }
        }
    }
}

impl Drop for EmbeddingPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Synchronous embedding helper (for simpler use cases)
pub fn embed_sync(
    provider: &dyn EmbeddingProvider,
    texts: &[&str],
    normalize: bool,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let mut embeddings = provider.embed_batch(texts)?;

    if normalize {
        for emb in &mut embeddings {
            normalize_l2_simd(emb);
        }
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbeddingProvider;

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.batch_timeout_ms, 100);
        assert!(config.normalize);
    }

    #[test]
    fn test_embedding_request() {
        let req = EmbeddingRequest::new(42, "hello".to_string());
        assert_eq!(req.id, 42);
        assert_eq!(req.text, "hello");
        assert!(req.metadata.is_none());
    }

    #[test]
    fn test_pipeline_basic() {
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let config = PipelineConfig {
            batch_size: 2,
            batch_timeout_ms: 50,
            max_pending: 10,
            ..Default::default()
        };

        let (pipeline, receiver) = EmbeddingPipeline::new(provider, config).unwrap();

        // Submit requests
        pipeline
            .submit(EmbeddingRequest::new(1, "hello".to_string()))
            .unwrap();
        pipeline
            .submit(EmbeddingRequest::new(2, "world".to_string()))
            .unwrap();

        // Receive results (with timeout)
        let mut results = Vec::new();
        for _ in 0..2 {
            match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(result) => results.push(result),
                Err(_) => break,
            }
        }

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.error.is_none());
            assert_eq!(result.embedding.len(), 384);
        }
    }

    #[test]
    fn test_pipeline_batch_processing() {
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let config = PipelineConfig {
            batch_size: 5,
            batch_timeout_ms: 50,
            max_pending: 100,
            ..Default::default()
        };

        let (pipeline, receiver) = EmbeddingPipeline::new(provider, config).unwrap();

        // Submit more than batch size
        for i in 0..10 {
            pipeline
                .submit(EmbeddingRequest::new(i as u128, format!("text {}", i)))
                .unwrap();
        }

        // Collect results
        let mut results = Vec::new();
        while results.len() < 10 {
            match receiver.recv_timeout(Duration::from_secs(2)) {
                Ok(result) => results.push(result),
                Err(_) => break,
            }
        }

        assert_eq!(results.len(), 10);

        // Check stats
        let stats = pipeline.stats();
        assert_eq!(stats.submitted.load(Ordering::Relaxed), 10);
        assert!(stats.batches.load(Ordering::Relaxed) >= 2); // At least 2 batches
    }

    #[test]
    fn test_pipeline_normalized_output() {
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let config = PipelineConfig {
            batch_size: 1,
            batch_timeout_ms: 10,
            normalize: true,
            ..Default::default()
        };

        let (pipeline, receiver) = EmbeddingPipeline::new(provider, config).unwrap();

        pipeline
            .submit(EmbeddingRequest::new(1, "test".to_string()))
            .unwrap();

        let result = receiver.recv_timeout(Duration::from_secs(1)).unwrap();

        // Check L2 normalized
        let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Output should be normalized, got norm: {}",
            norm
        );
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats::default();

        stats.completed.store(100, Ordering::Relaxed);
        stats.total_latency_us.store(1_000_000, Ordering::Relaxed);

        assert!((stats.avg_latency_us() - 10_000.0).abs() < 0.1);
        assert!((stats.throughput(10.0) - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_embed_sync() {
        let provider = MockEmbeddingProvider::default_provider();
        let texts = vec!["hello", "world"];

        let embeddings = embed_sync(&provider, &texts, true).unwrap();

        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_pipeline_timeout_flush() {
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let config = PipelineConfig {
            batch_size: 100, // Large batch that won't fill
            batch_timeout_ms: 50,
            max_pending: 10,
            ..Default::default()
        };

        let (pipeline, receiver) = EmbeddingPipeline::new(provider, config).unwrap();

        // Submit just one (won't fill batch)
        pipeline
            .submit(EmbeddingRequest::new(1, "lonely".to_string()))
            .unwrap();

        // Should still receive due to timeout
        let result = receiver.recv_timeout(Duration::from_millis(200));
        assert!(result.is_ok(), "Should receive result after timeout flush");
    }

    #[test]
    fn test_pipeline_shutdown() {
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        let (pipeline, _receiver) =
            EmbeddingPipeline::new(provider, PipelineConfig::default()).unwrap();

        pipeline.shutdown();
        // Should not hang
    }
}
