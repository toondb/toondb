// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! KV Service gRPC Implementation
//!
//! Provides basic key-value operations via gRPC.

use crate::proto::{
    kv_service_server::{KvService, KvServiceServer},
    KvBatchGetRequest, KvBatchGetResponse, KvBatchPutRequest, KvBatchPutResponse,
    KvDeleteRequest, KvDeleteResponse, KvEntry, KvGetRequest, KvGetResponse, KvPutRequest,
    KvPutResponse, KvScanRequest, KvScanResponse,
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

/// KV entry with optional TTL
struct KvEntryData {
    value: Vec<u8>,
    expires_at: Option<Instant>,
}

/// Namespace storage
struct NamespaceKv {
    entries: DashMap<Vec<u8>, KvEntryData>,
}

impl NamespaceKv {
    fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }
}

/// KV gRPC Server
pub struct KvServer {
    namespaces: DashMap<String, Arc<NamespaceKv>>,
}

impl KvServer {
    pub fn new() -> Self {
        Self {
            namespaces: DashMap::new(),
        }
    }

    pub fn into_service(self) -> KvServiceServer<Self> {
        KvServiceServer::new(self)
    }

    fn get_or_create_namespace(&self, namespace: &str) -> Arc<NamespaceKv> {
        self.namespaces
            .entry(namespace.to_string())
            .or_insert_with(|| Arc::new(NamespaceKv::new()))
            .clone()
    }
}

impl Default for KvServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl KvService for KvServer {
    async fn get(
        &self,
        request: Request<KvGetRequest>,
    ) -> Result<Response<KvGetResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);
        let now = Instant::now();

        match ns.entries.get(&req.key) {
            Some(entry) => {
                // Check TTL
                if let Some(expires_at) = entry.expires_at {
                    if now > expires_at {
                        ns.entries.remove(&req.key);
                        return Ok(Response::new(KvGetResponse {
                            value: vec![],
                            found: false,
                            error: String::new(),
                        }));
                    }
                }

                Ok(Response::new(KvGetResponse {
                    value: entry.value.clone(),
                    found: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(KvGetResponse {
                value: vec![],
                found: false,
                error: String::new(),
            })),
        }
    }

    async fn put(
        &self,
        request: Request<KvPutRequest>,
    ) -> Result<Response<KvPutResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let expires_at = if req.ttl_seconds > 0 {
            Some(Instant::now() + Duration::from_secs(req.ttl_seconds))
        } else {
            None
        };

        ns.entries.insert(
            req.key,
            KvEntryData {
                value: req.value,
                expires_at,
            },
        );

        Ok(Response::new(KvPutResponse {
            success: true,
            error: String::new(),
        }))
    }

    async fn delete(
        &self,
        request: Request<KvDeleteRequest>,
    ) -> Result<Response<KvDeleteResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let success = ns.entries.remove(&req.key).is_some();

        Ok(Response::new(KvDeleteResponse {
            success,
            error: String::new(),
        }))
    }

    type ScanStream = ReceiverStream<Result<KvScanResponse, Status>>;

    async fn scan(
        &self,
        request: Request<KvScanRequest>,
    ) -> Result<Response<Self::ScanStream>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);
        let now = Instant::now();

        let (tx, rx) = mpsc::channel(100);

        let limit = if req.limit > 0 { req.limit as usize } else { usize::MAX };

        // Collect matching entries
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = ns
            .entries
            .iter()
            .filter(|entry| {
                // Check prefix
                if !req.prefix.is_empty() && !entry.key().starts_with(&req.prefix) {
                    return false;
                }
                // Check TTL
                if let Some(expires_at) = entry.value().expires_at {
                    if now > expires_at {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .map(|entry| (entry.key().clone(), entry.value().value.clone()))
            .collect();

        // Sort by key for consistent ordering
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Spawn task to send results
        tokio::spawn(async move {
            for (key, value) in entries {
                let response = KvScanResponse { key, value };
                if tx.send(Ok(response)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn batch_get(
        &self,
        request: Request<KvBatchGetRequest>,
    ) -> Result<Response<KvBatchGetResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);
        let now = Instant::now();

        let entries: Vec<KvEntry> = req
            .keys
            .into_iter()
            .map(|key| {
                match ns.entries.get(&key) {
                    Some(entry) => {
                        // Check TTL
                        if let Some(expires_at) = entry.expires_at {
                            if now > expires_at {
                                return KvEntry {
                                    key,
                                    value: vec![],
                                    found: false,
                                };
                            }
                        }
                        KvEntry {
                            key,
                            value: entry.value.clone(),
                            found: true,
                        }
                    }
                    None => KvEntry {
                        key,
                        value: vec![],
                        found: false,
                    },
                }
            })
            .collect();

        Ok(Response::new(KvBatchGetResponse { entries }))
    }

    async fn batch_put(
        &self,
        request: Request<KvBatchPutRequest>,
    ) -> Result<Response<KvBatchPutResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let mut success_count = 0u32;

        for entry in req.entries {
            let expires_at = if entry.ttl_seconds > 0 {
                Some(Instant::now() + Duration::from_secs(entry.ttl_seconds))
            } else {
                None
            };

            ns.entries.insert(
                entry.key,
                KvEntryData {
                    value: entry.value,
                    expires_at,
                },
            );
            success_count += 1;
        }

        Ok(Response::new(KvBatchPutResponse {
            success_count,
            error: String::new(),
        }))
    }
}
