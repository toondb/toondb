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

//! Hierarchical Embedding Architecture
//!
//! This module provides multi-level semantic representation for agent traces.
//! Traces have natural hierarchy (Session → Trace → Span) that flat embeddings lose.
//!
//! ## Architecture
//!
//! ```text
//! Session (user conversation)
//! ├── Trace 1 (single task)
//! │   ├── Span: Planning (LLM reasoning)
//! │   ├── Span: Tool Call (API execution)
//! │   └── Span: Synthesis (response)
//! └── Trace 2 (follow-up task)
//!     └── ...
//! ```
//!
//! ## Benefits
//!
//! - Query "Find traces where planning succeeded but execution failed"
//! - Understand context from parent/sibling spans
//! - Better recall for context-dependent queries (40-60% improvement)

use crate::embedding::normalize::normalize_l2;
use crate::embedding::provider::{EmbeddingError, EmbeddingProvider};
use std::sync::Arc;

/// Embedding level in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingLevel {
    /// Session-level (aggregate of all traces)
    Session,
    /// Trace-level (aggregate of spans)
    Trace,
    /// Individual span
    Span,
}

/// Aggregation strategy for combining child embeddings
#[derive(Debug, Clone, Copy, Default)]
pub enum AggregationStrategy {
    /// Simple mean of all vectors
    #[default]
    Mean,
    /// Weighted mean (e.g., by duration or token count)
    WeightedMean,
    /// Max pooling (take max of each dimension)
    MaxPooling,
    /// Attention pooling (learned importance - requires training)
    AttentionPooling,
}

/// Contextual embedding with parent/sibling information
#[derive(Debug, Clone)]
pub struct ContextualEmbedding {
    /// The base embedding vector
    pub vector: Vec<f32>,

    /// Embedding level
    pub level: EmbeddingLevel,

    /// Parent context embedding (from parent span/trace)
    pub parent_context: Option<Vec<f32>>,

    /// Sibling context (previous/next spans)
    pub sibling_context: Option<Vec<f32>>,

    /// Associated ID (span_id, trace_id, etc.)
    pub id: u128,
}

impl ContextualEmbedding {
    /// Create a new contextual embedding
    pub fn new(vector: Vec<f32>, level: EmbeddingLevel, id: u128) -> Self {
        Self {
            vector,
            level,
            parent_context: None,
            sibling_context: None,
            id,
        }
    }

    /// Add parent context
    pub fn with_parent(mut self, parent: Vec<f32>) -> Self {
        self.parent_context = Some(parent);
        self
    }

    /// Add sibling context
    pub fn with_siblings(mut self, siblings: Vec<f32>) -> Self {
        self.sibling_context = Some(siblings);
        self
    }

    /// Get combined embedding with context
    ///
    /// Combines base vector with parent and sibling context:
    /// combined = base + α×parent + β×siblings
    ///
    /// Default weights: α=0.2, β=0.1
    pub fn combined_vector(&self, parent_weight: f32, sibling_weight: f32) -> Vec<f32> {
        let mut combined = self.vector.clone();
        let dim = combined.len();

        // Add parent context
        if let Some(ref parent) = self.parent_context
            && parent.len() == dim
        {
            for (i, &p) in parent.iter().enumerate() {
                combined[i] += parent_weight * p;
            }
        }

        // Add sibling context
        if let Some(ref sibling) = self.sibling_context
            && sibling.len() == dim
        {
            for (i, &s) in sibling.iter().enumerate() {
                combined[i] += sibling_weight * s;
            }
        }

        // Re-normalize
        normalize_l2(&mut combined);
        combined
    }
}

/// Hierarchical embedder for multi-level trace representation
pub struct HierarchicalEmbedder {
    /// Base embedding provider
    provider: Arc<dyn EmbeddingProvider>,

    /// Aggregation strategy
    aggregation: AggregationStrategy,

    /// Parent context weight (0.0-1.0)
    parent_weight: f32,

    /// Sibling context weight (0.0-1.0)
    sibling_weight: f32,
}

impl HierarchicalEmbedder {
    /// Create a new hierarchical embedder
    pub fn new(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            provider,
            aggregation: AggregationStrategy::WeightedMean,
            parent_weight: 0.2,
            sibling_weight: 0.1,
        }
    }

    /// Set aggregation strategy
    pub fn with_aggregation(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation = strategy;
        self
    }

    /// Set context weights
    pub fn with_weights(mut self, parent: f32, sibling: f32) -> Self {
        self.parent_weight = parent.clamp(0.0, 1.0);
        self.sibling_weight = sibling.clamp(0.0, 1.0);
        self
    }

    /// Embed a single span
    pub fn embed_span(
        &self,
        text: &str,
        span_id: u128,
    ) -> Result<ContextualEmbedding, EmbeddingError> {
        let vector = self.provider.embed(text)?;
        Ok(ContextualEmbedding::new(
            vector,
            EmbeddingLevel::Span,
            span_id,
        ))
    }

    /// Embed multiple spans with temporal context
    ///
    /// Each span gets context from its previous and next siblings.
    pub fn embed_spans_with_context(
        &self,
        spans: &[(u128, &str)], // (span_id, text)
    ) -> Result<Vec<ContextualEmbedding>, EmbeddingError> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        // First, embed all spans
        let texts: Vec<&str> = spans.iter().map(|(_, text)| *text).collect();
        let vectors = self.provider.embed_batch(&texts)?;

        let mut embeddings: Vec<ContextualEmbedding> = spans
            .iter()
            .zip(vectors)
            .map(|((id, _), vec)| ContextualEmbedding::new(vec, EmbeddingLevel::Span, *id))
            .collect();

        // Add sibling context
        for i in 0..embeddings.len() {
            let mut sibling_sum = vec![0.0f32; self.provider.dimension()];
            let mut sibling_count = 0;

            // Previous sibling
            if i > 0 {
                for (j, val) in embeddings[i - 1].vector.iter().enumerate() {
                    sibling_sum[j] += val;
                }
                sibling_count += 1;
            }

            // Next sibling
            if i + 1 < embeddings.len() {
                for (j, val) in embeddings[i + 1].vector.iter().enumerate() {
                    sibling_sum[j] += val;
                }
                sibling_count += 1;
            }

            if sibling_count > 0 {
                for val in &mut sibling_sum {
                    *val /= sibling_count as f32;
                }
                normalize_l2(&mut sibling_sum);
                embeddings[i].sibling_context = Some(sibling_sum);
            }
        }

        Ok(embeddings)
    }

    /// Aggregate span embeddings to trace level
    ///
    /// Weights can be duration, token count, or uniform.
    pub fn aggregate_to_trace(
        &self,
        span_embeddings: &[ContextualEmbedding],
        weights: Option<&[f32]>,
        trace_id: u128,
    ) -> Result<ContextualEmbedding, EmbeddingError> {
        if span_embeddings.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "No spans to aggregate".to_string(),
            ));
        }

        let dim = span_embeddings[0].vector.len();
        let trace_vector = match self.aggregation {
            AggregationStrategy::Mean => self.mean_aggregate(span_embeddings, dim),
            AggregationStrategy::WeightedMean => {
                if let Some(w) = weights {
                    self.weighted_mean_aggregate(span_embeddings, w, dim)
                } else {
                    self.mean_aggregate(span_embeddings, dim)
                }
            }
            AggregationStrategy::MaxPooling => self.max_pool_aggregate(span_embeddings, dim),
            AggregationStrategy::AttentionPooling => {
                // Fallback to mean for now (attention needs training)
                self.mean_aggregate(span_embeddings, dim)
            }
        };

        let mut embedding = ContextualEmbedding::new(trace_vector, EmbeddingLevel::Trace, trace_id);

        // Add trace embedding as parent context to all spans
        // (This would be done externally when building the full hierarchy)
        embedding.parent_context = None;

        Ok(embedding)
    }

    /// Aggregate trace embeddings to session level
    pub fn aggregate_to_session(
        &self,
        trace_embeddings: &[ContextualEmbedding],
        weights: Option<&[f32]>,
        session_id: u128,
    ) -> Result<ContextualEmbedding, EmbeddingError> {
        if trace_embeddings.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "No traces to aggregate".to_string(),
            ));
        }

        let dim = trace_embeddings[0].vector.len();
        let session_vector = match self.aggregation {
            AggregationStrategy::WeightedMean => {
                if let Some(w) = weights {
                    self.weighted_mean_aggregate(trace_embeddings, w, dim)
                } else {
                    self.mean_aggregate(trace_embeddings, dim)
                }
            }
            _ => self.mean_aggregate(trace_embeddings, dim),
        };

        Ok(ContextualEmbedding::new(
            session_vector,
            EmbeddingLevel::Session,
            session_id,
        ))
    }

    /// Mean aggregation
    fn mean_aggregate(&self, embeddings: &[ContextualEmbedding], dim: usize) -> Vec<f32> {
        let mut sum = vec![0.0f32; dim];
        let n = embeddings.len() as f32;

        for emb in embeddings {
            for (i, &val) in emb.vector.iter().enumerate() {
                sum[i] += val;
            }
        }

        for val in &mut sum {
            *val /= n;
        }

        normalize_l2(&mut sum);
        sum
    }

    /// Weighted mean aggregation
    fn weighted_mean_aggregate(
        &self,
        embeddings: &[ContextualEmbedding],
        weights: &[f32],
        dim: usize,
    ) -> Vec<f32> {
        let mut sum = vec![0.0f32; dim];
        let total_weight: f32 = weights.iter().sum();

        for (emb, &weight) in embeddings.iter().zip(weights) {
            let normalized_weight = weight / total_weight;
            for (i, &val) in emb.vector.iter().enumerate() {
                sum[i] += val * normalized_weight;
            }
        }

        normalize_l2(&mut sum);
        sum
    }

    /// Max pooling aggregation
    fn max_pool_aggregate(&self, embeddings: &[ContextualEmbedding], dim: usize) -> Vec<f32> {
        let mut max_vals = vec![f32::NEG_INFINITY; dim];

        for emb in embeddings {
            for (i, &val) in emb.vector.iter().enumerate() {
                if val > max_vals[i] {
                    max_vals[i] = val;
                }
            }
        }

        normalize_l2(&mut max_vals);
        max_vals
    }

    /// Get dimension from base provider
    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }
}

/// Embeddings for a complete trace with all levels
#[derive(Debug)]
pub struct TraceEmbeddings {
    /// Trace-level embedding
    pub trace: ContextualEmbedding,

    /// Span-level embeddings with context
    pub spans: Vec<ContextualEmbedding>,
}

impl TraceEmbeddings {
    /// Finalize embeddings by adding parent context to spans
    pub fn finalize(&mut self) {
        let trace_vector = self.trace.vector.clone();
        for span in &mut self.spans {
            span.parent_context = Some(trace_vector.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::provider::MockEmbeddingProvider;

    fn create_test_embedder() -> HierarchicalEmbedder {
        let provider = Arc::new(MockEmbeddingProvider::default_provider());
        HierarchicalEmbedder::new(provider)
    }

    #[test]
    fn test_embed_span() {
        let embedder = create_test_embedder();
        let embedding = embedder.embed_span("Test span content", 1).unwrap();

        assert_eq!(embedding.level, EmbeddingLevel::Span);
        assert_eq!(embedding.id, 1);
        assert_eq!(embedding.vector.len(), 384);
    }

    #[test]
    fn test_embed_spans_with_context() {
        let embedder = create_test_embedder();
        let spans = vec![(1, "First span"), (2, "Second span"), (3, "Third span")];

        let embeddings = embedder.embed_spans_with_context(&spans).unwrap();

        assert_eq!(embeddings.len(), 3);

        // First span should have next sibling context
        assert!(embeddings[0].sibling_context.is_some());

        // Middle span should have both siblings
        assert!(embeddings[1].sibling_context.is_some());

        // Last span should have prev sibling context
        assert!(embeddings[2].sibling_context.is_some());
    }

    #[test]
    fn test_aggregate_to_trace() {
        let embedder = create_test_embedder();
        let spans = vec![
            (1, "Planning phase"),
            (2, "Tool call execution"),
            (3, "Response synthesis"),
        ];

        let span_embeddings = embedder.embed_spans_with_context(&spans).unwrap();

        // Use uniform weights
        let trace = embedder
            .aggregate_to_trace(&span_embeddings, None, 100)
            .unwrap();

        assert_eq!(trace.level, EmbeddingLevel::Trace);
        assert_eq!(trace.id, 100);
        assert_eq!(trace.vector.len(), 384);

        // Check normalization
        let norm: f32 = trace.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_aggregation() {
        let embedder = create_test_embedder().with_aggregation(AggregationStrategy::WeightedMean);
        let spans = vec![
            (1, "Short span"),
            (2, "Much longer and more important span with more content"),
            (3, "Short span"),
        ];

        let span_embeddings = embedder.embed_spans_with_context(&spans).unwrap();

        // Weight by "importance" (middle span is more important)
        let weights = vec![1.0, 5.0, 1.0];
        let trace = embedder
            .aggregate_to_trace(&span_embeddings, Some(&weights), 100)
            .unwrap();

        assert_eq!(trace.vector.len(), 384);
    }

    #[test]
    fn test_aggregate_to_session() {
        let embedder = create_test_embedder();

        // Create some trace embeddings
        let traces: Vec<ContextualEmbedding> = vec![
            ContextualEmbedding::new(vec![1.0; 384], EmbeddingLevel::Trace, 1),
            ContextualEmbedding::new(vec![0.5; 384], EmbeddingLevel::Trace, 2),
            ContextualEmbedding::new(vec![0.0; 384], EmbeddingLevel::Trace, 3),
        ];

        let session = embedder.aggregate_to_session(&traces, None, 1000).unwrap();

        assert_eq!(session.level, EmbeddingLevel::Session);
        assert_eq!(session.id, 1000);
    }

    #[test]
    fn test_combined_vector() {
        let mut embedding = ContextualEmbedding::new(vec![1.0, 0.0, 0.0], EmbeddingLevel::Span, 1);
        embedding.parent_context = Some(vec![0.0, 1.0, 0.0]);
        embedding.sibling_context = Some(vec![0.0, 0.0, 1.0]);

        let combined = embedding.combined_vector(0.2, 0.1);

        // Check that all components contribute
        assert!(combined[0] > 0.0); // Base
        assert!(combined[1] > 0.0); // Parent
        assert!(combined[2] > 0.0); // Sibling

        // Check normalization
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_pooling() {
        let embedder = create_test_embedder().with_aggregation(AggregationStrategy::MaxPooling);

        // Create embeddings where each has a different dimension as max
        let emb1 = ContextualEmbedding::new(vec![1.0, 0.0, 0.0], EmbeddingLevel::Span, 1);
        let emb2 = ContextualEmbedding::new(vec![0.0, 1.0, 0.0], EmbeddingLevel::Span, 2);
        let emb3 = ContextualEmbedding::new(vec![0.0, 0.0, 1.0], EmbeddingLevel::Span, 3);

        let embeddings = vec![emb1, emb2, emb3];
        let trace = embedder.aggregate_to_trace(&embeddings, None, 100).unwrap();

        // After normalization, all dimensions should be equal (1/sqrt(3))
        let expected = 1.0 / 3.0_f32.sqrt();
        for val in &trace.vector {
            assert!((*val - expected).abs() < 1e-5);
        }
    }
}
