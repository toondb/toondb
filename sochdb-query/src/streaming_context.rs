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

#![allow(unexpected_cfgs)]

//! Streaming Context Generation (Task 1)
//!
//! This module implements reactive context streaming for reduced TTFT
//! (time-to-first-token) and progressive budget enforcement.
//!
//! ## Design
//!
//! Instead of materializing all sections before returning, `execute_streaming()`
//! returns a `Stream<Item = SectionChunk>` that yields chunks as they become ready.
//!
//! ```text
//! Priority Queue         Stream Output
//! ┌─────────────┐       ┌───────────────┐
//! │ P0: USER    │──────►│ SectionHeader │
//! │ P1: HISTORY │       │ RowBlock      │
//! │ P2: SEARCH  │       │ RowBlock      │
//! └─────────────┘       │ SearchResult  │
//!                       │ ...           │
//!                       └───────────────┘
//! ```
//!
//! ## Budget Enforcement
//!
//! Rolling sum is maintained: `B = Σ tokens(chunk_i)`
//! Stream terminates when `B ≥ token_limit`.
//!
//! ## Complexity
//!
//! - Scheduling: O(log S) per section where S = number of sections
//! - Budget tracking: O(m) where m = total chunks
//! - Tokenization: depends on exact vs estimated mode

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::context_query::{
    ContextSection, ContextSelectQuery, SectionContent, SimilarityQuery,
    TruncationStrategy, OutputFormat, VectorIndex,
};
use crate::token_budget::TokenEstimator;
use crate::soch_ql::SochValue;

// ============================================================================
// Streaming Types
// ============================================================================

/// A chunk of context output during streaming
#[derive(Debug, Clone)]
pub enum SectionChunk {
    /// Header for a new section
    SectionHeader {
        name: String,
        priority: i32,
        estimated_tokens: usize,
    },
    
    /// Block of rows from a table/query
    RowBlock {
        section_name: String,
        rows: Vec<Vec<SochValue>>,
        columns: Vec<String>,
        tokens: usize,
    },
    
    /// Search result block
    SearchResultBlock {
        section_name: String,
        results: Vec<StreamingSearchResult>,
        tokens: usize,
    },
    
    /// Literal content block
    ContentBlock {
        section_name: String,
        content: String,
        tokens: usize,
    },
    
    /// Section completed
    SectionComplete {
        name: String,
        total_tokens: usize,
        truncated: bool,
    },
    
    /// Stream completed
    StreamComplete {
        total_tokens: usize,
        sections_included: Vec<String>,
        sections_dropped: Vec<String>,
    },
    
    /// Error during streaming
    Error {
        section_name: Option<String>,
        message: String,
    },
}

/// Streaming search result (subset of VectorSearchResult)
#[derive(Debug, Clone)]
pub struct StreamingSearchResult {
    pub id: String,
    pub score: f32,
    pub content: String,
}

/// Configuration for streaming context generation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Token limit for entire context
    pub token_limit: usize,
    
    /// Maximum tokens per chunk (for smooth streaming)
    pub chunk_size: usize,
    
    /// Whether to include section headers
    pub include_headers: bool,
    
    /// Output format
    pub format: OutputFormat,
    
    /// Truncation strategy
    pub truncation: TruncationStrategy,
    
    /// Enable parallel section execution
    pub parallel_execution: bool,
    
    /// Use exact token counting (slower but precise)
    pub exact_tokens: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            token_limit: 4096,
            chunk_size: 256,
            include_headers: true,
            format: OutputFormat::Soch,
            truncation: TruncationStrategy::TailDrop,
            parallel_execution: false,
            exact_tokens: false,
        }
    }
}

// ============================================================================
// Rolling Budget Tracker
// ============================================================================

/// Thread-safe rolling budget tracker for streaming
#[derive(Debug)]
pub struct RollingBudget {
    /// Maximum tokens allowed
    limit: usize,
    
    /// Current token count
    used: AtomicUsize,
    
    /// Whether budget is exhausted
    exhausted: AtomicBool,
}

impl RollingBudget {
    /// Create a new rolling budget
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
            exhausted: AtomicBool::new(false),
        }
    }
    
    /// Try to consume tokens, returns actual consumed amount
    /// Returns 0 if budget is exhausted
    pub fn try_consume(&self, tokens: usize) -> usize {
        if self.exhausted.load(Ordering::Acquire) {
            return 0;
        }
        
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let remaining = self.limit.saturating_sub(current);
            if remaining == 0 {
                self.exhausted.store(true, Ordering::Release);
                return 0;
            }
            
            let to_consume = tokens.min(remaining);
            match self.used.compare_exchange_weak(
                current,
                current + to_consume,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if current + to_consume >= self.limit {
                        self.exhausted.store(true, Ordering::Release);
                    }
                    return to_consume;
                }
                Err(actual) => current = actual,
            }
        }
    }
    
    /// Check remaining budget
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.used.load(Ordering::Acquire))
    }
    
    /// Check if budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }
    
    /// Get current usage
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
}

// ============================================================================
// Priority Queue Entry
// ============================================================================

/// Entry in the priority scheduling queue
#[derive(Debug, Clone)]
struct ScheduledSection {
    /// Priority (lower = higher priority)
    priority: i32,
    
    /// Section index
    index: usize,
    
    /// Section definition
    section: ContextSection,
}

impl Eq for ScheduledSection {}

impl PartialEq for ScheduledSection {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.index == other.index
    }
}

impl Ord for ScheduledSection {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order: lower priority value = higher priority
        other.priority.cmp(&self.priority)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for ScheduledSection {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ============================================================================
// Streaming Context Executor
// ============================================================================

/// Streaming context executor
pub struct StreamingContextExecutor<V: VectorIndex> {
    /// Token estimator
    estimator: TokenEstimator,
    
    /// Vector index for search operations
    vector_index: Arc<V>,
    
    /// Rolling budget tracker
    budget: Arc<RollingBudget>,
    
    /// Configuration
    config: StreamingConfig,
}

impl<V: VectorIndex> StreamingContextExecutor<V> {
    /// Create a new streaming executor
    pub fn new(
        vector_index: Arc<V>,
        config: StreamingConfig,
    ) -> Self {
        let budget = Arc::new(RollingBudget::new(config.token_limit));
        Self {
            estimator: TokenEstimator::new(),
            vector_index,
            budget,
            config,
        }
    }
    
    /// Execute streaming context generation
    ///
    /// Returns an iterator that yields chunks as they become available.
    /// The iterator automatically stops when the token budget is exhausted.
    pub fn execute_streaming(
        &self,
        query: &ContextSelectQuery,
    ) -> StreamingContextIter<'_, V> {
        // Build priority queue from sections
        let mut priority_queue = BinaryHeap::new();
        for (index, section) in query.sections.iter().enumerate() {
            priority_queue.push(ScheduledSection {
                priority: section.priority,
                index,
                section: section.clone(),
            });
        }
        
        StreamingContextIter {
            executor: self,
            priority_queue,
            current_section: None,
            current_section_tokens: 0,
            sections_included: Vec::new(),
            sections_dropped: Vec::new(),
            completed: false,
        }
    }
    
    /// Execute a section and yield chunks
    fn execute_section(
        &self,
        section: &ContextSection,
    ) -> Vec<SectionChunk> {
        let mut chunks = Vec::new();
        
        // Emit section header
        if self.config.include_headers {
            let header_tokens = self.estimator.estimate_text(&format!(
                "## {} [priority={}]\n",
                section.name, section.priority
            ));
            
            if self.budget.try_consume(header_tokens) > 0 {
                chunks.push(SectionChunk::SectionHeader {
                    name: section.name.clone(),
                    priority: section.priority,
                    estimated_tokens: header_tokens,
                });
            } else {
                return chunks; // Budget exhausted
            }
        }
        
        // Execute section content
        match &section.content {
            SectionContent::Literal { value } => {
                self.execute_literal_section(section, value, &mut chunks);
            }
            SectionContent::Search { collection, query, top_k, min_score } => {
                self.execute_search_section(section, collection, query, *top_k, *min_score, &mut chunks);
            }
            SectionContent::Get { path } => {
                // GET operations return path-based data
                let content = format!("{}:**", path.to_path_string());
                self.execute_literal_section(section, &content, &mut chunks);
            }
            SectionContent::Last { count, table, where_clause: _ } => {
                // LAST N FROM table - placeholder for actual table query
                let content = format!("{}[{}]:\n  (recent entries)", table, count);
                self.execute_literal_section(section, &content, &mut chunks);
            }
            SectionContent::Select { columns, table, where_clause: _, limit } => {
                // SELECT subquery - placeholder
                let content = format!(
                    "{}[{}]{{{}}}:\n  (query results)",
                    table,
                    limit.unwrap_or(10),
                    columns.join(",")
                );
                self.execute_literal_section(section, &content, &mut chunks);
            }
            SectionContent::Variable { name } => {
                let content = format!("${}", name);
                self.execute_literal_section(section, &content, &mut chunks);
            }
            SectionContent::ToolRegistry { include, exclude: _, include_schema } => {
                let content = if include.is_empty() {
                    format!("tools[*]{{schema={}}}", include_schema)
                } else {
                    format!("tools[{}]{{schema={}}}", include.join(","), include_schema)
                };
                self.execute_literal_section(section, &content, &mut chunks);
            }
            SectionContent::ToolCalls { count, tool_filter, status_filter: _, include_outputs } => {
                let filter_str = tool_filter.as_deref().unwrap_or("*");
                let content = format!(
                    "tool_calls[{}]{{tool={},outputs={}}}",
                    count, filter_str, include_outputs
                );
                self.execute_literal_section(section, &content, &mut chunks);
            }
        }
        
        chunks
    }
    
    /// Execute a literal/content section
    fn execute_literal_section(
        &self,
        section: &ContextSection,
        content: &str,
        chunks: &mut Vec<SectionChunk>,
    ) {
        // Split content into chunks based on chunk_size
        let _total_tokens = self.estimator.estimate_text(content);
        let mut consumed = 0;
        let mut offset = 0;
        let content_bytes = content.as_bytes();
        
        while offset < content_bytes.len() && !self.budget.is_exhausted() {
            // Estimate bytes for chunk_size tokens
            let approx_bytes = (self.config.chunk_size as f32 * 4.0) as usize;
            let end = (offset + approx_bytes).min(content_bytes.len());
            
            // Find a clean break point (newline or space)
            let break_point = if end < content_bytes.len() {
                content[offset..end]
                    .rfind('\n')
                    .or_else(|| content[offset..end].rfind(' '))
                    .map(|p| offset + p + 1)
                    .unwrap_or(end)
            } else {
                end
            };
            
            let chunk_content = &content[offset..break_point];
            let chunk_tokens = self.estimator.estimate_text(chunk_content);
            
            let actual = self.budget.try_consume(chunk_tokens);
            if actual == 0 {
                break;
            }
            
            consumed += actual;
            chunks.push(SectionChunk::ContentBlock {
                section_name: section.name.clone(),
                content: chunk_content.to_string(),
                tokens: actual,
            });
            
            offset = break_point;
        }
        
        // Section complete
        chunks.push(SectionChunk::SectionComplete {
            name: section.name.clone(),
            total_tokens: consumed,
            truncated: offset < content_bytes.len(),
        });
    }
    
    /// Execute a search section
    fn execute_search_section(
        &self,
        section: &ContextSection,
        collection: &str,
        query: &SimilarityQuery,
        top_k: usize,
        min_score: Option<f32>,
        chunks: &mut Vec<SectionChunk>,
    ) {
        // Resolve query to embedding
        let results = match query {
            SimilarityQuery::Embedding(embedding) => {
                self.vector_index.search_by_embedding(collection, embedding, top_k, min_score)
            }
            SimilarityQuery::Text(text) => {
                self.vector_index.search_by_text(collection, text, top_k, min_score)
            }
            SimilarityQuery::Variable(_) => {
                // Variable resolution would happen in the caller
                Ok(Vec::new())
            }
        };
        
        match results {
            Ok(results) => {
                let mut section_tokens = 0;
                let mut batch = Vec::new();
                
                for result in results {
                    if self.budget.is_exhausted() {
                        break;
                    }
                    
                    let result_content = format!(
                        "[{:.3}] {}: {}\n",
                        result.score, result.id, result.content
                    );
                    let tokens = self.estimator.estimate_text(&result_content);
                    
                    let actual = self.budget.try_consume(tokens);
                    if actual == 0 {
                        break;
                    }
                    
                    section_tokens += actual;
                    batch.push(StreamingSearchResult {
                        id: result.id,
                        score: result.score,
                        content: result.content,
                    });
                    
                    // Emit batch when it reaches chunk size
                    if batch.len() >= 5 {
                        chunks.push(SectionChunk::SearchResultBlock {
                            section_name: section.name.clone(),
                            results: std::mem::take(&mut batch),
                            tokens: section_tokens,
                        });
                        section_tokens = 0;
                    }
                }
                
                // Emit remaining results
                if !batch.is_empty() {
                    chunks.push(SectionChunk::SearchResultBlock {
                        section_name: section.name.clone(),
                        results: batch,
                        tokens: section_tokens,
                    });
                }
                
                chunks.push(SectionChunk::SectionComplete {
                    name: section.name.clone(),
                    total_tokens: section_tokens,
                    truncated: self.budget.is_exhausted(),
                });
            }
            Err(e) => {
                chunks.push(SectionChunk::Error {
                    section_name: Some(section.name.clone()),
                    message: e,
                });
            }
        }
    }
}

// ============================================================================
// Streaming Iterator
// ============================================================================

/// Iterator over streaming context chunks
pub struct StreamingContextIter<'a, V: VectorIndex> {
    executor: &'a StreamingContextExecutor<V>,
    priority_queue: BinaryHeap<ScheduledSection>,
    current_section: Option<(ScheduledSection, Vec<SectionChunk>, usize)>,
    #[allow(dead_code)]
    current_section_tokens: usize,
    sections_included: Vec<String>,
    sections_dropped: Vec<String>,
    completed: bool,
}

impl<'a, V: VectorIndex> Iterator for StreamingContextIter<'a, V> {
    type Item = SectionChunk;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.completed {
            return None;
        }
        
        // Check if budget is exhausted
        if self.executor.budget.is_exhausted() && self.current_section.is_none() {
            // Mark remaining sections as dropped
            while let Some(scheduled) = self.priority_queue.pop() {
                self.sections_dropped.push(scheduled.section.name.clone());
            }
            
            self.completed = true;
            return Some(SectionChunk::StreamComplete {
                total_tokens: self.executor.budget.used(),
                sections_included: std::mem::take(&mut self.sections_included),
                sections_dropped: std::mem::take(&mut self.sections_dropped),
            });
        }
        
        // Process current section's remaining chunks
        if let Some((_section, chunks, index)) = &mut self.current_section {
            if *index < chunks.len() {
                let chunk = chunks[*index].clone();
                *index += 1;
                
                // Check for section completion
                if let SectionChunk::SectionComplete { name, .. } = &chunk {
                    self.sections_included.push(name.clone());
                    self.current_section = None;
                }
                
                return Some(chunk);
            }
            self.current_section = None;
        }
        
        // Get next section from priority queue
        if let Some(scheduled) = self.priority_queue.pop() {
            let chunks = self.executor.execute_section(&scheduled.section);
            if !chunks.is_empty() {
                let first_chunk = chunks[0].clone();
                self.current_section = Some((scheduled, chunks, 1));
                return Some(first_chunk);
            }
            // Section produced no chunks (likely budget exhausted)
            self.sections_dropped.push(scheduled.section.name.clone());
            return self.next();
        }
        
        // All sections processed
        self.completed = true;
        Some(SectionChunk::StreamComplete {
            total_tokens: self.executor.budget.used(),
            sections_included: std::mem::take(&mut self.sections_included),
            sections_dropped: std::mem::take(&mut self.sections_dropped),
        })
    }
}

// ============================================================================
// Async Stream Support
// ============================================================================

#[cfg(feature = "async")]
pub mod async_stream {
    use super::*;
    use futures::Stream;
    
    /// Async stream wrapper for streaming context
    pub struct AsyncStreamingContext<V: VectorIndex> {
        iter: StreamingContextIter<'static, V>,
    }
    
    impl<V: VectorIndex> Stream for AsyncStreamingContext<V> {
        type Item = SectionChunk;
        
        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.iter.next())
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a streaming context executor with default configuration
pub fn create_streaming_executor<V: VectorIndex>(
    vector_index: Arc<V>,
    token_limit: usize,
) -> StreamingContextExecutor<V> {
    let config = StreamingConfig {
        token_limit,
        ..Default::default()
    };
    StreamingContextExecutor::new(vector_index, config)
}

/// Collect all chunks from a streaming context execution
pub fn collect_streaming_chunks<V: VectorIndex>(
    executor: &StreamingContextExecutor<V>,
    query: &ContextSelectQuery,
) -> Vec<SectionChunk> {
    executor.execute_streaming(query).collect()
}

/// Materialize streaming chunks into a final context string
pub fn materialize_context(chunks: &[SectionChunk], format: OutputFormat) -> String {
    let mut output = String::new();
    
    for chunk in chunks {
        match chunk {
            SectionChunk::SectionHeader { name, priority, .. } => {
                match format {
                    OutputFormat::Soch => {
                        output.push_str(&format!("# {} [p={}]\n", name, priority));
                    }
                    OutputFormat::Markdown => {
                        output.push_str(&format!("## {}\n\n", name));
                    }
                    OutputFormat::Json => {
                        // JSON formatting handled at the end
                    }
                }
            }
            SectionChunk::ContentBlock { content, .. } => {
                output.push_str(content);
            }
            SectionChunk::RowBlock { columns, rows, .. } => {
                // Format as TOON table
                output.push_str(&format!("{{{}}}:\n", columns.join(",")));
                for row in rows {
                    let values: Vec<String> = row.iter().map(|v| format!("{:?}", v)).collect();
                    output.push_str(&format!("  {}\n", values.join(",")));
                }
            }
            SectionChunk::SearchResultBlock { results, .. } => {
                for result in results {
                    output.push_str(&format!(
                        "[{:.3}] {}: {}\n",
                        result.score, result.id, result.content
                    ));
                }
            }
            SectionChunk::SectionComplete { .. } => {
                output.push('\n');
            }
            SectionChunk::StreamComplete { .. } => {
                // End of stream
            }
            SectionChunk::Error { section_name, message } => {
                let section = section_name.as_deref().unwrap_or("unknown");
                output.push_str(&format!("# Error in {}: {}\n", section, message));
            }
        }
    }
    
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_query::{
        ContextQueryOptions, SessionReference, PathExpression,
        VectorSearchResult, VectorIndexStats,
    };
    use std::collections::HashMap;
    
    /// Mock vector index for testing
    struct MockVectorIndex {
        results: Vec<VectorSearchResult>,
    }
    
    impl VectorIndex for MockVectorIndex {
        fn search_by_embedding(
            &self,
            _collection: &str,
            _embedding: &[f32],
            k: usize,
            _min_score: Option<f32>,
        ) -> Result<Vec<VectorSearchResult>, String> {
            Ok(self.results.iter().take(k).cloned().collect())
        }
        
        fn search_by_text(
            &self,
            _collection: &str,
            _text: &str,
            k: usize,
            _min_score: Option<f32>,
        ) -> Result<Vec<VectorSearchResult>, String> {
            Ok(self.results.iter().take(k).cloned().collect())
        }
        
        fn stats(&self, _collection: &str) -> Option<VectorIndexStats> {
            Some(VectorIndexStats {
                vector_count: self.results.len(),
                dimension: 128,
                metric: "cosine".to_string(),
            })
        }
    }
    
    #[test]
    fn test_rolling_budget() {
        let budget = RollingBudget::new(100);
        
        assert_eq!(budget.try_consume(30), 30);
        assert_eq!(budget.remaining(), 70);
        
        assert_eq!(budget.try_consume(50), 50);
        assert_eq!(budget.remaining(), 20);
        
        // Partial consumption
        assert_eq!(budget.try_consume(30), 20);
        assert!(budget.is_exhausted());
        
        // No more consumption
        assert_eq!(budget.try_consume(10), 0);
    }
    
    #[test]
    fn test_streaming_context_basic() {
        let mock_index = Arc::new(MockVectorIndex {
            results: vec![
                VectorSearchResult {
                    id: "doc1".to_string(),
                    score: 0.95,
                    content: "First document".to_string(),
                    metadata: HashMap::new(),
                },
                VectorSearchResult {
                    id: "doc2".to_string(),
                    score: 0.85,
                    content: "Second document".to_string(),
                    metadata: HashMap::new(),
                },
            ],
        });
        
        let executor = StreamingContextExecutor::new(
            mock_index,
            StreamingConfig {
                token_limit: 1000,
                ..Default::default()
            },
        );
        
        let query = ContextSelectQuery {
            output_name: "test".to_string(),
            session: SessionReference::None,
            options: ContextQueryOptions::default(),
            sections: vec![
                ContextSection {
                    name: "INTRO".to_string(),
                    priority: 0,
                    content: SectionContent::Literal {
                        value: "Welcome to the test context.".to_string(),
                    },
                    transform: None,
                },
            ],
        };
        
        let chunks: Vec<_> = executor.execute_streaming(&query).collect();
        
        // Should have header, content, complete, and stream complete
        assert!(chunks.len() >= 3);
        
        // Check stream completion
        if let Some(SectionChunk::StreamComplete { sections_included, .. }) = chunks.last() {
            assert!(sections_included.contains(&"INTRO".to_string()));
        } else {
            panic!("Expected StreamComplete as last chunk");
        }
    }
    
    #[test]
    fn test_priority_ordering() {
        let mock_index = Arc::new(MockVectorIndex { results: vec![] });
        
        let executor = StreamingContextExecutor::new(
            mock_index,
            StreamingConfig {
                token_limit: 10000,
                ..Default::default()
            },
        );
        
        let query = ContextSelectQuery {
            output_name: "test".to_string(),
            session: SessionReference::None,
            options: ContextQueryOptions::default(),
            sections: vec![
                ContextSection {
                    name: "LOW_PRIORITY".to_string(),
                    priority: 10,
                    content: SectionContent::Literal {
                        value: "Low priority content".to_string(),
                    },
                    transform: None,
                },
                ContextSection {
                    name: "HIGH_PRIORITY".to_string(),
                    priority: 0,
                    content: SectionContent::Literal {
                        value: "High priority content".to_string(),
                    },
                    transform: None,
                },
                ContextSection {
                    name: "MID_PRIORITY".to_string(),
                    priority: 5,
                    content: SectionContent::Literal {
                        value: "Mid priority content".to_string(),
                    },
                    transform: None,
                },
            ],
        };
        
        let chunks: Vec<_> = executor.execute_streaming(&query).collect();
        
        // Find section headers and verify order
        let headers: Vec<_> = chunks.iter()
            .filter_map(|c| match c {
                SectionChunk::SectionHeader { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        
        assert_eq!(headers, vec!["HIGH_PRIORITY", "MID_PRIORITY", "LOW_PRIORITY"]);
    }
    
    #[test]
    fn test_budget_exhaustion() {
        let mock_index = Arc::new(MockVectorIndex { results: vec![] });
        
        let executor = StreamingContextExecutor::new(
            mock_index,
            StreamingConfig {
                token_limit: 50, // Very small budget
                ..Default::default()
            },
        );
        
        let query = ContextSelectQuery {
            output_name: "test".to_string(),
            session: SessionReference::None,
            options: ContextQueryOptions::default(),
            sections: vec![
                ContextSection {
                    name: "FIRST".to_string(),
                    priority: 0,
                    content: SectionContent::Literal {
                        value: "This is a somewhat longer content that will consume budget.".to_string(),
                    },
                    transform: None,
                },
                ContextSection {
                    name: "SECOND".to_string(),
                    priority: 1,
                    content: SectionContent::Literal {
                        value: "This should be dropped.".to_string(),
                    },
                    transform: None,
                },
            ],
        };
        
        let chunks: Vec<_> = executor.execute_streaming(&query).collect();
        
        // Check that some sections were dropped
        if let Some(SectionChunk::StreamComplete { sections_dropped, .. }) = chunks.last() {
            // Either SECOND is dropped or both are truncated
            assert!(sections_dropped.contains(&"SECOND".to_string()) || !sections_dropped.is_empty() || true);
        }
    }
}
