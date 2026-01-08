// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Checkpoint Service gRPC Implementation
//!
//! Provides state checkpoint and restore via gRPC.

use crate::proto::{
    checkpoint_service_server::{CheckpointService, CheckpointServiceServer},
    Checkpoint, CreateCheckpointRequest, CreateCheckpointResponse, DeleteCheckpointRequest,
    DeleteCheckpointResponse, ExportCheckpointRequest, ExportCheckpointResponse, ExportFormat,
    ImportCheckpointRequest, ImportCheckpointResponse, ListCheckpointsRequest,
    ListCheckpointsResponse, RestoreCheckpointRequest, RestoreCheckpointResponse,
};
use dashmap::DashMap;
use std::time::SystemTime;
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Stored checkpoint data
struct CheckpointData {
    info: Checkpoint,
    data: Vec<u8>,
}

/// Checkpoint gRPC Server
pub struct CheckpointServer {
    checkpoints: DashMap<String, CheckpointData>,
}

impl CheckpointServer {
    pub fn new() -> Self {
        Self {
            checkpoints: DashMap::new(),
        }
    }

    pub fn into_service(self) -> CheckpointServiceServer<Self> {
        CheckpointServiceServer::new(self)
    }
}

impl Default for CheckpointServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl CheckpointService for CheckpointServer {
    async fn create_checkpoint(
        &self,
        request: Request<CreateCheckpointRequest>,
    ) -> Result<Response<CreateCheckpointResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::new_v4().to_string();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create a mock checkpoint (in real implementation, this would snapshot actual data)
        let checkpoint_data = serde_json::json!({
            "patterns": req.include_patterns,
            "namespace": req.namespace,
            "created_at": now
        });
        let data = serde_json::to_vec(&checkpoint_data).unwrap_or_default();

        let checkpoint = Checkpoint {
            id: id.clone(),
            name: req.name,
            namespace: req.namespace,
            created_at: now,
            size_bytes: data.len() as u64,
            metadata: req.metadata,
        };

        self.checkpoints.insert(
            id,
            CheckpointData {
                info: checkpoint.clone(),
                data,
            },
        );

        Ok(Response::new(CreateCheckpointResponse {
            success: true,
            checkpoint: Some(checkpoint),
            error: String::new(),
        }))
    }

    async fn restore_checkpoint(
        &self,
        request: Request<RestoreCheckpointRequest>,
    ) -> Result<Response<RestoreCheckpointResponse>, Status> {
        let req = request.into_inner();

        match self.checkpoints.get(&req.checkpoint_id) {
            Some(data) => {
                // In real implementation, this would restore actual data
                let restored_keys = 10u64; // Mock value

                Ok(Response::new(RestoreCheckpointResponse {
                    success: true,
                    restored_keys,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(RestoreCheckpointResponse {
                success: false,
                restored_keys: 0,
                error: format!("Checkpoint '{}' not found", req.checkpoint_id),
            })),
        }
    }

    async fn list_checkpoints(
        &self,
        request: Request<ListCheckpointsRequest>,
    ) -> Result<Response<ListCheckpointsResponse>, Status> {
        let req = request.into_inner();

        let checkpoints: Vec<Checkpoint> = self
            .checkpoints
            .iter()
            .filter(|entry| {
                req.namespace.is_empty() || entry.value().info.namespace == req.namespace
            })
            .map(|entry| entry.value().info.clone())
            .collect();

        Ok(Response::new(ListCheckpointsResponse { checkpoints }))
    }

    async fn delete_checkpoint(
        &self,
        request: Request<DeleteCheckpointRequest>,
    ) -> Result<Response<DeleteCheckpointResponse>, Status> {
        let req = request.into_inner();

        match self.checkpoints.remove(&req.checkpoint_id) {
            Some(_) => Ok(Response::new(DeleteCheckpointResponse {
                success: true,
                error: String::new(),
            })),
            None => Ok(Response::new(DeleteCheckpointResponse {
                success: false,
                error: format!("Checkpoint '{}' not found", req.checkpoint_id),
            })),
        }
    }

    async fn export_checkpoint(
        &self,
        request: Request<ExportCheckpointRequest>,
    ) -> Result<Response<ExportCheckpointResponse>, Status> {
        let req = request.into_inner();

        match self.checkpoints.get(&req.checkpoint_id) {
            Some(checkpoint_data) => {
                let data = match req.format {
                    x if x == ExportFormat::Json as i32 => {
                        serde_json::to_vec(&serde_json::json!({
                            "checkpoint": {
                                "id": checkpoint_data.info.id,
                                "name": checkpoint_data.info.name,
                                "namespace": checkpoint_data.info.namespace,
                            },
                            "data": base64::Engine::encode(
                                &base64::engine::general_purpose::STANDARD,
                                &checkpoint_data.data
                            )
                        }))
                        .unwrap_or_default()
                    }
                    _ => checkpoint_data.data.clone(),
                };

                Ok(Response::new(ExportCheckpointResponse {
                    data,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(ExportCheckpointResponse {
                data: vec![],
                error: format!("Checkpoint '{}' not found", req.checkpoint_id),
            })),
        }
    }

    async fn import_checkpoint(
        &self,
        request: Request<ImportCheckpointRequest>,
    ) -> Result<Response<ImportCheckpointResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::new_v4().to_string();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let checkpoint = Checkpoint {
            id: id.clone(),
            name: req.name,
            namespace: req.namespace,
            created_at: now,
            size_bytes: req.data.len() as u64,
            metadata: std::collections::HashMap::new(),
        };

        self.checkpoints.insert(
            id,
            CheckpointData {
                info: checkpoint.clone(),
                data: req.data,
            },
        );

        Ok(Response::new(ImportCheckpointResponse {
            success: true,
            checkpoint: Some(checkpoint),
            error: String::new(),
        }))
    }
}
