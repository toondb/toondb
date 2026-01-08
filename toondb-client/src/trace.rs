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

//! First-Class Trace Store (Task 5)
//!
//! This module provides a queryable trace data model for:
//! - Runs → Steps → Retrieval Hits → Tool Calls
//! - Cost accounting
//! - Debugging and postmortems
//! - Regression testing
//!
//! ## Architecture
//!
//! Traces are stored as an append-only log (event sourcing pattern):
//! - Appends: O(1) amortized with WAL/LSM-style storage
//! - Queries: O(log N + k) with indexes
//!
//! ## Schema
//!
//! ```text
//! _traces/runs/{run_id}                     -> TraceRun
//! _traces/spans/{run_id}/{span_id}          -> TraceSpan
//! _traces/events/{run_id}/{timestamp}_{seq} -> TraceEvent
//! _traces/index/by_type/{type}/{run_id}     -> index entry
//! ```
//!
//! ## OpenTelemetry Compatibility
//!
//! The trace model is designed to be compatible with OpenTelemetry spans
//! for easy integration with existing observability tooling.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ClientError, Result};
use crate::ConnectionTrait;

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a trace run
pub type TraceId = String;

/// Unique identifier for a span within a run
pub type SpanId = String;

/// A complete trace run (e.g., a request, session, or agent lifetime)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRun {
    /// Unique run ID
    pub trace_id: TraceId,
    /// Run name/description
    pub name: String,
    /// Start timestamp (micros since epoch)
    pub start_time: u64,
    /// End timestamp (None if still running)
    pub end_time: Option<u64>,
    /// Run status
    pub status: TraceStatus,
    /// Run attributes
    pub attributes: HashMap<String, TraceValue>,
    /// Resource attributes (agent_id, session_id, etc.)
    pub resource: HashMap<String, String>,
    /// Total token count
    pub total_tokens: u64,
    /// Estimated cost in USD (millicents)
    pub cost_millicents: u64,
}

/// Status of a trace run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceStatus {
    /// Run is in progress
    Running,
    /// Run completed successfully
    Ok,
    /// Run completed with error
    Error,
}

/// A span within a trace (represents a unit of work)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Trace this span belongs to
    pub trace_id: TraceId,
    /// Unique span ID
    pub span_id: SpanId,
    /// Parent span ID (None for root spans)
    pub parent_span_id: Option<SpanId>,
    /// Span name (e.g., "retrieval", "fusion", "context_packaging")
    pub name: String,
    /// Span kind
    pub kind: SpanKind,
    /// Start timestamp (micros)
    pub start_time: u64,
    /// End timestamp (micros)
    pub end_time: Option<u64>,
    /// Duration in microseconds
    pub duration_us: Option<u64>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, TraceValue>,
    /// Events within this span
    pub events: Vec<SpanEvent>,
}

/// Kind of span (OpenTelemetry compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Server handling a request
    Server,
    /// Client making a request
    Client,
    /// Producer sending a message
    Producer,
    /// Consumer receiving a message
    Consumer,
}

/// Status of a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanStatus {
    /// Status code
    pub code: SpanStatusCode,
    /// Optional error message
    pub message: Option<String>,
}

/// Span status code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatusCode {
    /// Unset - default
    Unset,
    /// OK - success
    Ok,
    /// Error - failure
    Error,
}

/// An event within a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Timestamp (micros)
    pub timestamp: u64,
    /// Event attributes
    pub attributes: HashMap<String, TraceValue>,
}

/// A trace value (attribute value)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TraceValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Array of strings
    StringArray(Vec<String>),
    /// Array of integers
    IntArray(Vec<i64>),
}

impl From<&str> for TraceValue {
    fn from(s: &str) -> Self {
        TraceValue::String(s.to_string())
    }
}

impl From<String> for TraceValue {
    fn from(s: String) -> Self {
        TraceValue::String(s)
    }
}

impl From<i64> for TraceValue {
    fn from(i: i64) -> Self {
        TraceValue::Int(i)
    }
}

impl From<f64> for TraceValue {
    fn from(f: f64) -> Self {
        TraceValue::Float(f)
    }
}

impl From<bool> for TraceValue {
    fn from(b: bool) -> Self {
        TraceValue::Bool(b)
    }
}

// ============================================================================
// Domain-Specific Events
// ============================================================================

/// Retrieval hit event - logged when a document is retrieved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalHitEvent {
    /// Document ID
    pub doc_id: String,
    /// Score
    pub score: f32,
    /// Source modality (vector, bm25, both)
    pub modality: String,
    /// Rank in result set
    pub rank: usize,
    /// Whether it was filtered out
    pub filtered: bool,
    /// Collection name
    pub collection: String,
}

/// Tool call event - logged when a tool is invoked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallEvent {
    /// Tool name
    pub tool_name: String,
    /// Tool arguments (serialized)
    pub arguments: String,
    /// Tool result (truncated if large)
    pub result: Option<String>,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Whether the call succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Context packaging event - logged when context is assembled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPackagingEvent {
    /// Sections included
    pub sections: Vec<String>,
    /// Total tokens in context
    pub total_tokens: u64,
    /// Token budget
    pub budget: u64,
    /// Truncation applied
    pub truncated: bool,
}

/// Cost event - logged for billing/accounting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEvent {
    /// Cost type (embedding, llm_input, llm_output, storage, etc.)
    pub cost_type: String,
    /// Amount (in model-specific units, e.g., tokens)
    pub amount: u64,
    /// Unit price in millicents
    pub unit_price_millicents: f64,
    /// Total cost in millicents
    pub total_millicents: u64,
    /// Model name (if applicable)
    pub model: Option<String>,
}

// ============================================================================
// Trace Store Interface
// ============================================================================

/// Storage prefix
const TRACE_PREFIX: &str = "_traces/";

/// Interface for trace storage
pub struct TraceStore<C: ConnectionTrait> {
    conn: C,
    /// Sampling rate (1.0 = all, 0.1 = 10%)
    sample_rate: f64,
}

impl<C: ConnectionTrait> TraceStore<C> {
    /// Create a new trace store
    pub fn new(conn: C) -> Self {
        Self {
            conn,
            sample_rate: 1.0,
        }
    }
    
    /// Create with sampling
    pub fn with_sampling(conn: C, sample_rate: f64) -> Self {
        Self {
            conn,
            sample_rate: sample_rate.clamp(0.0, 1.0),
        }
    }
    
    fn should_sample(&self) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        if self.sample_rate <= 0.0 {
            return false;
        }
        rand::random::<f64>() < self.sample_rate
    }
    
    fn now_micros() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
    
    fn run_key(trace_id: &TraceId) -> Vec<u8> {
        format!("{}runs/{}", TRACE_PREFIX, trace_id).into_bytes()
    }
    
    fn span_key(trace_id: &TraceId, span_id: &SpanId) -> Vec<u8> {
        format!("{}spans/{}/{}", TRACE_PREFIX, trace_id, span_id).into_bytes()
    }
    
    fn spans_prefix(trace_id: &TraceId) -> Vec<u8> {
        format!("{}spans/{}/", TRACE_PREFIX, trace_id).into_bytes()
    }
    
    fn event_key(trace_id: &TraceId, timestamp: u64, seq: u64) -> Vec<u8> {
        format!(
            "{}events/{}/{:016x}_{:08x}",
            TRACE_PREFIX, trace_id, timestamp, seq
        ).into_bytes()
    }
    
    fn events_prefix(trace_id: &TraceId) -> Vec<u8> {
        format!("{}events/{}/", TRACE_PREFIX, trace_id).into_bytes()
    }
    
    // ========================================================================
    // Run Operations
    // ========================================================================
    
    /// Start a new trace run
    pub fn start_run(
        &self,
        name: impl Into<String>,
        resource: HashMap<String, String>,
    ) -> Result<TraceRun> {
        let trace_id = generate_trace_id();
        let now = Self::now_micros();
        
        let run = TraceRun {
            trace_id: trace_id.clone(),
            name: name.into(),
            start_time: now,
            end_time: None,
            status: TraceStatus::Running,
            attributes: HashMap::new(),
            resource,
            total_tokens: 0,
            cost_millicents: 0,
        };
        
        if self.should_sample() {
            let key = Self::run_key(&trace_id);
            let value = serde_json::to_vec(&run)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        
        Ok(run)
    }
    
    /// End a trace run
    pub fn end_run(&self, trace_id: &TraceId, status: TraceStatus) -> Result<()> {
        let key = Self::run_key(trace_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut run: TraceRun = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            run.end_time = Some(Self::now_micros());
            run.status = status;
            let value = serde_json::to_vec(&run)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    /// Get a trace run
    pub fn get_run(&self, trace_id: &TraceId) -> Result<Option<TraceRun>> {
        let key = Self::run_key(trace_id);
        if let Some(data) = self.conn.get(&key)? {
            let run: TraceRun = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            Ok(Some(run))
        } else {
            Ok(None)
        }
    }
    
    /// Update run metrics
    pub fn update_run_metrics(
        &self,
        trace_id: &TraceId,
        tokens: u64,
        cost_millicents: u64,
    ) -> Result<()> {
        let key = Self::run_key(trace_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut run: TraceRun = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            run.total_tokens += tokens;
            run.cost_millicents += cost_millicents;
            let value = serde_json::to_vec(&run)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    // ========================================================================
    // Span Operations
    // ========================================================================
    
    /// Start a new span
    pub fn start_span(
        &self,
        trace_id: &TraceId,
        name: impl Into<String>,
        parent_span_id: Option<SpanId>,
        kind: SpanKind,
    ) -> Result<TraceSpan> {
        let span_id = generate_span_id();
        let now = Self::now_micros();
        
        let span = TraceSpan {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id,
            name: name.into(),
            kind,
            start_time: now,
            end_time: None,
            duration_us: None,
            status: SpanStatus {
                code: SpanStatusCode::Unset,
                message: None,
            },
            attributes: HashMap::new(),
            events: Vec::new(),
        };
        
        if self.should_sample() {
            let key = Self::span_key(trace_id, &span_id);
            let value = serde_json::to_vec(&span)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        
        Ok(span)
    }
    
    /// End a span
    pub fn end_span(
        &self,
        trace_id: &TraceId,
        span_id: &SpanId,
        status: SpanStatusCode,
        message: Option<String>,
    ) -> Result<()> {
        let key = Self::span_key(trace_id, span_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut span: TraceSpan = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            let now = Self::now_micros();
            span.end_time = Some(now);
            span.duration_us = Some(now.saturating_sub(span.start_time));
            span.status = SpanStatus {
                code: status,
                message,
            };
            let value = serde_json::to_vec(&span)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    /// Add an event to a span
    pub fn add_span_event(
        &self,
        trace_id: &TraceId,
        span_id: &SpanId,
        name: impl Into<String>,
        attributes: HashMap<String, TraceValue>,
    ) -> Result<()> {
        let key = Self::span_key(trace_id, span_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut span: TraceSpan = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            span.events.push(SpanEvent {
                name: name.into(),
                timestamp: Self::now_micros(),
                attributes,
            });
            let value = serde_json::to_vec(&span)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    /// Set span attributes
    pub fn set_span_attributes(
        &self,
        trace_id: &TraceId,
        span_id: &SpanId,
        attributes: HashMap<String, TraceValue>,
    ) -> Result<()> {
        let key = Self::span_key(trace_id, span_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut span: TraceSpan = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            span.attributes.extend(attributes);
            let value = serde_json::to_vec(&span)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    /// Get all spans for a trace
    pub fn get_spans(&self, trace_id: &TraceId) -> Result<Vec<TraceSpan>> {
        let prefix = Self::spans_prefix(trace_id);
        let results = self.conn.scan(&prefix)?;
        
        let mut spans = Vec::new();
        for (_, value) in results {
            let span: TraceSpan = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            spans.push(span);
        }
        
        // Sort by start time
        spans.sort_by_key(|s| s.start_time);
        Ok(spans)
    }
    
    // ========================================================================
    // Domain Events
    // ========================================================================
    
    /// Log a retrieval hit
    pub fn log_retrieval_hit(
        &self,
        trace_id: &TraceId,
        span_id: &SpanId,
        hit: RetrievalHitEvent,
    ) -> Result<()> {
        let mut attrs = HashMap::new();
        attrs.insert("doc_id".to_string(), TraceValue::String(hit.doc_id));
        attrs.insert("score".to_string(), TraceValue::Float(hit.score as f64));
        attrs.insert("modality".to_string(), TraceValue::String(hit.modality));
        attrs.insert("rank".to_string(), TraceValue::Int(hit.rank as i64));
        attrs.insert("filtered".to_string(), TraceValue::Bool(hit.filtered));
        attrs.insert("collection".to_string(), TraceValue::String(hit.collection));
        
        self.add_span_event(trace_id, span_id, "retrieval_hit", attrs)
    }
    
    /// Log a tool call
    pub fn log_tool_call(
        &self,
        trace_id: &TraceId,
        span_id: &SpanId,
        call: ToolCallEvent,
    ) -> Result<()> {
        let mut attrs = HashMap::new();
        attrs.insert("tool_name".to_string(), TraceValue::String(call.tool_name));
        attrs.insert("arguments".to_string(), TraceValue::String(call.arguments));
        attrs.insert("duration_us".to_string(), TraceValue::Int(call.duration_us as i64));
        attrs.insert("success".to_string(), TraceValue::Bool(call.success));
        
        if let Some(result) = call.result {
            // Truncate result if too long
            let truncated = if result.len() > 1000 {
                format!("{}...(truncated)", &result[..1000])
            } else {
                result
            };
            attrs.insert("result".to_string(), TraceValue::String(truncated));
        }
        
        if let Some(error) = call.error {
            attrs.insert("error".to_string(), TraceValue::String(error));
        }
        
        self.add_span_event(trace_id, span_id, "tool_call", attrs)
    }
    
    /// Log context packaging
    pub fn log_context_packaging(
        &self,
        trace_id: &TraceId,
        span_id: &SpanId,
        event: ContextPackagingEvent,
    ) -> Result<()> {
        let mut attrs = HashMap::new();
        attrs.insert("sections".to_string(), TraceValue::StringArray(event.sections));
        attrs.insert("total_tokens".to_string(), TraceValue::Int(event.total_tokens as i64));
        attrs.insert("budget".to_string(), TraceValue::Int(event.budget as i64));
        attrs.insert("truncated".to_string(), TraceValue::Bool(event.truncated));
        
        self.add_span_event(trace_id, span_id, "context_packaging", attrs)
    }
    
    /// Log a cost event
    pub fn log_cost(
        &self,
        trace_id: &TraceId,
        event: CostEvent,
    ) -> Result<()> {
        // Update run metrics
        self.update_run_metrics(trace_id, event.amount, event.total_millicents)?;
        Ok(())
    }
}

// ============================================================================
// ID Generation
// ============================================================================

fn generate_trace_id() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", now ^ rand::random::<u128>())
}

fn generate_span_id() -> String {
    format!("{:016x}", rand::random::<u64>())
}

// ============================================================================
// Simple random for ID generation (no external dep)
// ============================================================================

mod rand {
    use std::cell::Cell;
    use std::time::SystemTime;
    
    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        );
    }
    
    pub fn random<T: Random>() -> T {
        T::random()
    }
    
    pub trait Random {
        fn random() -> Self;
    }
    
    impl Random for u64 {
        fn random() -> Self {
            SEED.with(|seed| {
                let mut s = seed.get();
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                seed.set(s);
                s
            })
        }
    }
    
    impl Random for u128 {
        fn random() -> Self {
            let high = u64::random() as u128;
            let low = u64::random() as u128;
            (high << 64) | low
        }
    }
    
    impl Random for f64 {
        fn random() -> Self {
            (u64::random() as f64) / (u64::MAX as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests would use mock ConnectionTrait
}
