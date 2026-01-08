// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Namespace Service gRPC Implementation
//!
//! Provides namespace management for multi-tenant isolation via gRPC.

use crate::proto::{
    namespace_service_server::{NamespaceService, NamespaceServiceServer},
    CreateNamespaceRequest, CreateNamespaceResponse, DeleteNamespaceRequest,
    DeleteNamespaceResponse, GetNamespaceRequest, GetNamespaceResponse, ListNamespacesRequest,
    ListNamespacesResponse, Namespace, NamespaceQuota, NamespaceStats, SetQuotaRequest,
    SetQuotaResponse,
};
use dashmap::DashMap;
use std::time::SystemTime;
use tonic::{Request, Response, Status};

/// Namespace gRPC Server
pub struct NamespaceServer {
    namespaces: DashMap<String, Namespace>,
}

impl NamespaceServer {
    pub fn new() -> Self {
        Self {
            namespaces: DashMap::new(),
        }
    }

    pub fn into_service(self) -> NamespaceServiceServer<Self> {
        NamespaceServiceServer::new(self)
    }
}

impl Default for NamespaceServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl NamespaceService for NamespaceServer {
    async fn create_namespace(
        &self,
        request: Request<CreateNamespaceRequest>,
    ) -> Result<Response<CreateNamespaceResponse>, Status> {
        let req = request.into_inner();

        if self.namespaces.contains_key(&req.name) {
            return Ok(Response::new(CreateNamespaceResponse {
                success: false,
                namespace: None,
                error: format!("Namespace '{}' already exists", req.name),
            }));
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let namespace = Namespace {
            name: req.name.clone(),
            description: req.description,
            created_at: now,
            quota: req.quota,
            stats: Some(NamespaceStats {
                storage_bytes: 0,
                vector_count: 0,
                collection_count: 0,
            }),
            metadata: req.metadata,
        };

        self.namespaces.insert(req.name, namespace.clone());

        Ok(Response::new(CreateNamespaceResponse {
            success: true,
            namespace: Some(namespace),
            error: String::new(),
        }))
    }

    async fn get_namespace(
        &self,
        request: Request<GetNamespaceRequest>,
    ) -> Result<Response<GetNamespaceResponse>, Status> {
        let req = request.into_inner();

        match self.namespaces.get(&req.name) {
            Some(ns) => Ok(Response::new(GetNamespaceResponse {
                namespace: Some(ns.clone()),
                error: String::new(),
            })),
            None => Ok(Response::new(GetNamespaceResponse {
                namespace: None,
                error: format!("Namespace '{}' not found", req.name),
            })),
        }
    }

    async fn list_namespaces(
        &self,
        _request: Request<ListNamespacesRequest>,
    ) -> Result<Response<ListNamespacesResponse>, Status> {
        let namespaces: Vec<Namespace> = self
            .namespaces
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        Ok(Response::new(ListNamespacesResponse { namespaces }))
    }

    async fn delete_namespace(
        &self,
        request: Request<DeleteNamespaceRequest>,
    ) -> Result<Response<DeleteNamespaceResponse>, Status> {
        let req = request.into_inner();

        match self.namespaces.remove(&req.name) {
            Some(_) => Ok(Response::new(DeleteNamespaceResponse {
                success: true,
                error: String::new(),
            })),
            None => Ok(Response::new(DeleteNamespaceResponse {
                success: false,
                error: format!("Namespace '{}' not found", req.name),
            })),
        }
    }

    async fn set_quota(
        &self,
        request: Request<SetQuotaRequest>,
    ) -> Result<Response<SetQuotaResponse>, Status> {
        let req = request.into_inner();

        match self.namespaces.get_mut(&req.namespace) {
            Some(mut ns) => {
                ns.quota = req.quota;
                Ok(Response::new(SetQuotaResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(SetQuotaResponse {
                success: false,
                error: format!("Namespace '{}' not found", req.namespace),
            })),
        }
    }
}
