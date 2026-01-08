// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Trace Service gRPC Implementation
//!
//! Provides trace/span management for observability via gRPC.

use crate::proto::{
    trace_service_server::{TraceService, TraceServiceServer},
    AddEventRequest, AddEventResponse, EndSpanRequest, EndSpanResponse, GetTraceRequest,
    GetTraceResponse, ListTracesRequest, ListTracesResponse, Span, SpanEvent, SpanStatus,
    StartSpanRequest, StartSpanResponse, StartTraceRequest, StartTraceResponse, Trace,
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// In-memory trace storage
struct TraceData {
    trace: Trace,
    spans: DashMap<String, Span>,
}

/// Trace gRPC Server
pub struct TraceServer {
    traces: DashMap<String, Arc<TraceData>>,
}

impl TraceServer {
    pub fn new() -> Self {
        Self {
            traces: DashMap::new(),
        }
    }

    pub fn into_service(self) -> TraceServiceServer<Self> {
        TraceServiceServer::new(self)
    }

    fn now_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

impl Default for TraceServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl TraceService for TraceServer {
    async fn start_trace(
        &self,
        request: Request<StartTraceRequest>,
    ) -> Result<Response<StartTraceResponse>, Status> {
        let req = request.into_inner();
        let trace_id = Uuid::new_v4().to_string();
        let root_span_id = Uuid::new_v4().to_string();
        let now = Self::now_us();

        let trace = Trace {
            trace_id: trace_id.clone(),
            name: req.name.clone(),
            start_time_us: now,
            end_time_us: 0,
            spans: vec![],
            attributes: req.attributes.clone(),
        };

        let root_span = Span {
            span_id: root_span_id.clone(),
            trace_id: trace_id.clone(),
            parent_span_id: String::new(),
            name: req.name,
            start_time_us: now,
            end_time_us: 0,
            status: SpanStatus::Unset.into(),
            events: vec![],
            attributes: req.attributes,
        };

        let data = Arc::new(TraceData {
            trace,
            spans: DashMap::new(),
        });
        data.spans.insert(root_span_id.clone(), root_span);

        self.traces.insert(trace_id.clone(), data);

        Ok(Response::new(StartTraceResponse {
            trace_id,
            root_span_id,
        }))
    }

    async fn start_span(
        &self,
        request: Request<StartSpanRequest>,
    ) -> Result<Response<StartSpanResponse>, Status> {
        let req = request.into_inner();

        match self.traces.get(&req.trace_id) {
            Some(data) => {
                let span_id = Uuid::new_v4().to_string();
                let span = Span {
                    span_id: span_id.clone(),
                    trace_id: req.trace_id,
                    parent_span_id: req.parent_span_id,
                    name: req.name,
                    start_time_us: Self::now_us(),
                    end_time_us: 0,
                    status: SpanStatus::Unset.into(),
                    events: vec![],
                    attributes: req.attributes,
                };

                data.spans.insert(span_id.clone(), span);

                Ok(Response::new(StartSpanResponse { span_id }))
            }
            None => Err(Status::not_found(format!(
                "Trace '{}' not found",
                req.trace_id
            ))),
        }
    }

    async fn end_span(
        &self,
        request: Request<EndSpanRequest>,
    ) -> Result<Response<EndSpanResponse>, Status> {
        let req = request.into_inner();

        match self.traces.get(&req.trace_id) {
            Some(data) => match data.spans.get_mut(&req.span_id) {
                Some(mut span) => {
                    let now = Self::now_us();
                    span.end_time_us = now;
                    span.status = req.status;
                    for (k, v) in req.attributes {
                        span.attributes.insert(k, v);
                    }
                    let duration_us = now - span.start_time_us;

                    Ok(Response::new(EndSpanResponse {
                        success: true,
                        duration_us,
                    }))
                }
                None => Err(Status::not_found(format!(
                    "Span '{}' not found",
                    req.span_id
                ))),
            },
            None => Err(Status::not_found(format!(
                "Trace '{}' not found",
                req.trace_id
            ))),
        }
    }

    async fn add_event(
        &self,
        request: Request<AddEventRequest>,
    ) -> Result<Response<AddEventResponse>, Status> {
        let req = request.into_inner();

        match self.traces.get(&req.trace_id) {
            Some(data) => match data.spans.get_mut(&req.span_id) {
                Some(mut span) => {
                    let event = SpanEvent {
                        name: req.event_name,
                        timestamp_us: Self::now_us(),
                        attributes: req.attributes,
                    };
                    span.events.push(event);

                    Ok(Response::new(AddEventResponse { success: true }))
                }
                None => Err(Status::not_found(format!(
                    "Span '{}' not found",
                    req.span_id
                ))),
            },
            None => Err(Status::not_found(format!(
                "Trace '{}' not found",
                req.trace_id
            ))),
        }
    }

    async fn get_trace(
        &self,
        request: Request<GetTraceRequest>,
    ) -> Result<Response<GetTraceResponse>, Status> {
        let req = request.into_inner();

        match self.traces.get(&req.trace_id) {
            Some(data) => {
                let mut trace = data.trace.clone();
                trace.spans = data.spans.iter().map(|e| e.value().clone()).collect();

                // Update end time to latest span end
                if let Some(max_end) = trace.spans.iter().map(|s| s.end_time_us).max() {
                    trace.end_time_us = max_end;
                }

                Ok(Response::new(GetTraceResponse {
                    trace: Some(trace),
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(GetTraceResponse {
                trace: None,
                error: format!("Trace '{}' not found", req.trace_id),
            })),
        }
    }

    async fn list_traces(
        &self,
        request: Request<ListTracesRequest>,
    ) -> Result<Response<ListTracesResponse>, Status> {
        let req = request.into_inner();

        let mut traces: Vec<Trace> = self
            .traces
            .iter()
            .filter(|entry| {
                let data = entry.value();
                // Filter by timestamp
                if req.since_timestamp > 0 && data.trace.start_time_us < req.since_timestamp {
                    return false;
                }
                // Filter by name
                if !req.name_filter.is_empty() && !data.trace.name.contains(&req.name_filter) {
                    return false;
                }
                true
            })
            .map(|entry| {
                let data = entry.value();
                let mut trace = data.trace.clone();
                trace.spans = data.spans.iter().map(|e| e.value().clone()).collect();
                trace
            })
            .collect();

        // Sort by start time descending
        traces.sort_by(|a, b| b.start_time_us.cmp(&a.start_time_us));

        // Apply limit
        if req.limit > 0 {
            traces.truncate(req.limit as usize);
        }

        Ok(Response::new(ListTracesResponse { traces }))
    }
}
