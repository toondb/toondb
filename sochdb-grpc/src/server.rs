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

//! gRPC Server implementation for Vector Index Service

use crate::error::GrpcError;
use crate::proto::{
    self,
    vector_index_service_server::{VectorIndexService, VectorIndexServiceServer},
    CreateIndexRequest, CreateIndexResponse, DropIndexRequest, DropIndexResponse,
    GetStatsRequest, GetStatsResponse, HealthCheckRequest, HealthCheckResponse,
    HnswConfig as ProtoHnswConfig, IndexInfo, IndexStats, InsertBatchRequest,
    InsertBatchResponse, InsertStreamRequest, InsertStreamResponse, QueryResults,
    SearchBatchRequest, SearchBatchResponse, SearchRequest, SearchResponse, SearchResult,
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};

/// Server version
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Metadata for an index
#[allow(dead_code)]
struct IndexEntry {
    index: Arc<HnswIndex>,
    name: String,
    dimension: usize,
    metric: proto::DistanceMetric,
    config: ProtoHnswConfig,
    created_at: u64,
}

/// Vector Index gRPC Server
pub struct VectorIndexServer {
    /// Map of index name -> index entry
    indexes: DashMap<String, IndexEntry>,
}

impl VectorIndexServer {
    /// Create a new server instance
    pub fn new() -> Self {
        Self {
            indexes: DashMap::new(),
        }
    }
    
    /// Create the gRPC service
    pub fn into_service(self) -> VectorIndexServiceServer<Self> {
        VectorIndexServiceServer::new(self)
    }
    
    /// Get an index by name and its dimension
    fn get_index_with_dim(&self, name: &str) -> Result<(Arc<HnswIndex>, usize), GrpcError> {
        self.indexes
            .get(name)
            .map(|entry| (entry.index.clone(), entry.dimension))
            .ok_or_else(|| GrpcError::IndexNotFound(name.to_string()))
    }
    
    /// Get an index by name
    fn get_index(&self, name: &str) -> Result<Arc<HnswIndex>, GrpcError> {
        self.indexes
            .get(name)
            .map(|entry| entry.index.clone())
            .ok_or_else(|| GrpcError::IndexNotFound(name.to_string()))
    }
    
    /// Convert proto metric to internal metric
    fn convert_metric(metric: proto::DistanceMetric) -> DistanceMetric {
        match metric {
            proto::DistanceMetric::L2 => DistanceMetric::Euclidean,
            proto::DistanceMetric::Cosine => DistanceMetric::Cosine,
            proto::DistanceMetric::DotProduct => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine, // Default
        }
    }
}

impl Default for VectorIndexServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl VectorIndexService for VectorIndexServer {
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        let req = request.into_inner();
        let name = req.name.clone();
        
        // Check if index already exists
        if self.indexes.contains_key(&name) {
            return Ok(Response::new(CreateIndexResponse {
                success: false,
                error: format!("Index '{}' already exists", name),
                info: None,
            }));
        }
        
        // Build config
        let proto_config = req.config.unwrap_or_default();
        let config = HnswConfig {
            max_connections: if proto_config.max_connections > 0 {
                proto_config.max_connections as usize
            } else {
                16
            },
            max_connections_layer0: if proto_config.max_connections_layer0 > 0 {
                proto_config.max_connections_layer0 as usize
            } else {
                32
            },
            ef_construction: if proto_config.ef_construction > 0 {
                proto_config.ef_construction as usize
            } else {
                200
            },
            ef_search: if proto_config.ef_search > 0 {
                proto_config.ef_search as usize
            } else {
                50
            },
            metric: Self::convert_metric(req.metric()),
            ..Default::default()
        };
        
        let dimension = req.dimension as usize;
        let index = HnswIndex::new(dimension, config.clone());
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let entry = IndexEntry {
            index: Arc::new(index),
            name: name.clone(),
            dimension,
            metric: req.metric(),
            config: proto_config.clone(),
            created_at,
        };
        
        self.indexes.insert(name.clone(), entry);
        
        tracing::info!("Created index '{}' with dimension {}", name, dimension);
        
        Ok(Response::new(CreateIndexResponse {
            success: true,
            error: String::new(),
            info: Some(IndexInfo {
                name,
                dimension: dimension as u32,
                metric: req.metric.into(),
                config: Some(proto_config),
                created_at,
            }),
        }))
    }
    
    async fn drop_index(
        &self,
        request: Request<DropIndexRequest>,
    ) -> Result<Response<DropIndexResponse>, Status> {
        let name = request.into_inner().name;
        
        match self.indexes.remove(&name) {
            Some(_) => {
                tracing::info!("Dropped index '{}'", name);
                Ok(Response::new(DropIndexResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(DropIndexResponse {
                success: false,
                error: format!("Index '{}' not found", name),
            })),
        }
    }
    
    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertBatchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        
        let (index, dimension) = self.get_index_with_dim(&req.index_name)?;
        
        // Validate input
        if req.vectors.len() != req.ids.len() * dimension {
            return Err(Status::invalid_argument(format!(
                "Vector data size mismatch: expected {} floats, got {}",
                req.ids.len() * dimension,
                req.vectors.len()
            )));
        }
        
        // Convert IDs to u128
        let ids: Vec<u128> = req.ids.iter().map(|&id| id as u128).collect();
        
        // Use flat batch insert for zero-copy performance
        match index.insert_batch_flat(&ids, &req.vectors, dimension) {
            Ok(count) => {
                let duration_us = start.elapsed().as_micros() as u64;
                tracing::debug!(
                    "Inserted {} vectors into '{}' in {}µs",
                    count,
                    req.index_name,
                    duration_us
                );
                Ok(Response::new(InsertBatchResponse {
                    inserted_count: count as u32,
                    error: String::new(),
                    duration_us,
                }))
            }
            Err(e) => Ok(Response::new(InsertBatchResponse {
                inserted_count: 0,
                error: e,
                duration_us: start.elapsed().as_micros() as u64,
            })),
        }
    }
    
    async fn insert_stream(
        &self,
        request: Request<Streaming<InsertStreamRequest>>,
    ) -> Result<Response<InsertStreamResponse>, Status> {
        let start = Instant::now();
        let mut stream = request.into_inner();
        
        let mut index_name: Option<String> = None;
        let mut index: Option<Arc<HnswIndex>> = None;
        let mut total_inserted = 0u32;
        let mut errors = Vec::new();
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(req) => {
                    // Get index on first message
                    if index.is_none() {
                        if req.index_name.is_empty() {
                            errors.push("First message must include index_name".to_string());
                            continue;
                        }
                        index_name = Some(req.index_name.clone());
                        match self.get_index(&req.index_name) {
                            Ok(idx) => index = Some(idx),
                            Err(e) => {
                                errors.push(e.to_string());
                                break;
                            }
                        }
                    }
                    
                    // Insert the vector
                    if let Some(ref idx) = index {
                        let vector: Vec<f32> = req.vector;
                        match idx.insert_one_from_slice(req.id as u128, &vector) {
                            Ok(()) => total_inserted += 1,
                            Err(e) => errors.push(format!("ID {}: {}", req.id, e)),
                        }
                    }
                }
                Err(e) => {
                    errors.push(format!("Stream error: {}", e));
                    break;
                }
            }
        }
        
        let duration_us = start.elapsed().as_micros() as u64;
        
        if let Some(name) = &index_name {
            tracing::debug!(
                "Stream inserted {} vectors into '{}' in {}µs",
                total_inserted,
                name,
                duration_us
            );
        }
        
        Ok(Response::new(InsertStreamResponse {
            total_inserted,
            errors,
            duration_us,
        }))
    }
    
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        
        let (index, dimension) = self.get_index_with_dim(&req.index_name)?;
        
        // Validate dimension
        if req.query.len() != dimension {
            return Err(Status::invalid_argument(format!(
                "Query dimension mismatch: expected {}, got {}",
                dimension,
                req.query.len()
            )));
        }
        
        let k = req.k.max(1) as usize;
        
        // Perform search
        let results = match index.search(&req.query, k) {
            Ok(r) => r,
            Err(e) => {
                return Ok(Response::new(SearchResponse {
                    results: vec![],
                    duration_us: start.elapsed().as_micros() as u64,
                    error: e,
                }));
            }
        };
        
        let duration_us = start.elapsed().as_micros() as u64;
        
        Ok(Response::new(SearchResponse {
            results: results
                .into_iter()
                .map(|(id, distance)| SearchResult {
                    id: id as u64,
                    distance,
                })
                .collect(),
            duration_us,
            error: String::new(),
        }))
    }
    
    async fn search_batch(
        &self,
        request: Request<SearchBatchRequest>,
    ) -> Result<Response<SearchBatchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        
        let (index, dimension) = self.get_index_with_dim(&req.index_name)?;
        let num_queries = req.num_queries as usize;
        let k = req.k.max(1) as usize;
        
        // Validate
        if req.queries.len() != num_queries * dimension {
            return Err(Status::invalid_argument(format!(
                "Query data size mismatch: expected {} floats, got {}",
                num_queries * dimension,
                req.queries.len()
            )));
        }
        
        // Perform batch search
        let mut all_results = Vec::with_capacity(num_queries);
        
        for i in 0..num_queries {
            let query = &req.queries[i * dimension..(i + 1) * dimension];
            let results = match index.search(query, k) {
                Ok(r) => r,
                Err(_) => vec![],
            };
            
            all_results.push(QueryResults {
                results: results
                    .into_iter()
                    .map(|(id, distance)| SearchResult {
                        id: id as u64,
                        distance,
                    })
                    .collect(),
            });
        }
        
        let duration_us = start.elapsed().as_micros() as u64;
        
        Ok(Response::new(SearchBatchResponse {
            results: all_results,
            duration_us,
        }))
    }
    
    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        let name = request.into_inner().index_name;
        
        match self.indexes.get(&name) {
            Some(entry) => {
                let stats = entry.index.stats();
                Ok(Response::new(GetStatsResponse {
                    stats: Some(IndexStats {
                        num_vectors: stats.num_vectors as u64,
                        dimension: entry.dimension as u32,
                        max_layer: stats.max_layer as u32,
                        memory_bytes: 0, // Memory stats available via separate call
                        avg_connections: stats.avg_connections,
                    }),
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(GetStatsResponse {
                stats: None,
                error: format!("Index '{}' not found", name),
            })),
        }
    }
    
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let indexes: Vec<String> = self.indexes.iter().map(|e| e.name.clone()).collect();
        
        Ok(Response::new(HealthCheckResponse {
            status: proto::health_check_response::Status::Serving.into(),
            version: VERSION.to_string(),
            indexes,
        }))
    }
}
