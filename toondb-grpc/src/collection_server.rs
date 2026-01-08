// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Collection Service gRPC Implementation
//!
//! Provides collection management for vectors/documents via gRPC.

use crate::proto::{
    collection_service_server::{CollectionService, CollectionServiceServer},
    AddDocumentsRequest, AddDocumentsResponse, Collection, CreateCollectionRequest,
    CreateCollectionResponse, DeleteCollectionRequest, DeleteCollectionResponse,
    DeleteDocumentRequest, DeleteDocumentResponse, Document, DocumentResult,
    GetCollectionRequest, GetCollectionResponse, GetDocumentRequest, GetDocumentResponse,
    ListCollectionsRequest, ListCollectionsResponse, SearchCollectionRequest,
    SearchCollectionResponse,
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::{Request, Response, Status};

/// In-memory collection storage
struct CollectionData {
    info: Collection,
    documents: DashMap<String, Document>,
}

/// Collection gRPC Server
pub struct CollectionServer {
    collections: DashMap<String, Arc<CollectionData>>,
}

impl CollectionServer {
    pub fn new() -> Self {
        Self {
            collections: DashMap::new(),
        }
    }

    pub fn into_service(self) -> CollectionServiceServer<Self> {
        CollectionServiceServer::new(self)
    }

    fn collection_key(namespace: &str, name: &str) -> String {
        format!("{}:{}", namespace, name)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

impl Default for CollectionServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl CollectionService for CollectionServer {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.name);

        if self.collections.contains_key(&key) {
            return Ok(Response::new(CreateCollectionResponse {
                success: false,
                collection: None,
                error: format!("Collection '{}' already exists", req.name),
            }));
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let collection = Collection {
            name: req.name.clone(),
            namespace: req.namespace.clone(),
            dimension: req.dimension,
            metric: req.metric,
            document_count: 0,
            created_at: now,
            metadata: req.metadata,
        };

        let data = Arc::new(CollectionData {
            info: collection.clone(),
            documents: DashMap::new(),
        });

        self.collections.insert(key, data);

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            collection: Some(collection),
            error: String::new(),
        }))
    }

    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<GetCollectionResponse>, Status> {
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.name);

        match self.collections.get(&key) {
            Some(data) => {
                let mut info = data.info.clone();
                info.document_count = data.documents.len() as u64;
                Ok(Response::new(GetCollectionResponse {
                    collection: Some(info),
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(GetCollectionResponse {
                collection: None,
                error: format!("Collection '{}' not found", req.name),
            })),
        }
    }

    async fn list_collections(
        &self,
        request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let req = request.into_inner();

        let collections: Vec<Collection> = self
            .collections
            .iter()
            .filter(|entry| {
                req.namespace.is_empty() || entry.value().info.namespace == req.namespace
            })
            .map(|entry| {
                let mut info = entry.value().info.clone();
                info.document_count = entry.value().documents.len() as u64;
                info
            })
            .collect();

        Ok(Response::new(ListCollectionsResponse { collections }))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.name);

        match self.collections.remove(&key) {
            Some(_) => Ok(Response::new(DeleteCollectionResponse {
                success: true,
                error: String::new(),
            })),
            None => Ok(Response::new(DeleteCollectionResponse {
                success: false,
                error: format!("Collection '{}' not found", req.name),
            })),
        }
    }

    async fn add_documents(
        &self,
        request: Request<AddDocumentsRequest>,
    ) -> Result<Response<AddDocumentsResponse>, Status> {
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => {
                let mut ids = Vec::new();
                for doc in req.documents {
                    let id = if doc.id.is_empty() {
                        uuid::Uuid::new_v4().to_string()
                    } else {
                        doc.id.clone()
                    };
                    ids.push(id.clone());

                    let mut stored_doc = doc;
                    stored_doc.id = id.clone();
                    data.documents.insert(id, stored_doc);
                }

                Ok(Response::new(AddDocumentsResponse {
                    added_count: ids.len() as u32,
                    ids,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(AddDocumentsResponse {
                added_count: 0,
                ids: vec![],
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }

    async fn search_collection(
        &self,
        request: Request<SearchCollectionRequest>,
    ) -> Result<Response<SearchCollectionResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => {
                let mut scored: Vec<(Document, f32)> = data
                    .documents
                    .iter()
                    .filter(|entry| {
                        // Apply metadata filter
                        if req.filter.is_empty() {
                            return true;
                        }
                        for (filter_key, filter_val) in &req.filter {
                            if let Some(doc_val) = entry.value().metadata.get(filter_key) {
                                if doc_val != filter_val {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                        true
                    })
                    .map(|entry| {
                        let doc = entry.value().clone();
                        let score = Self::cosine_similarity(&req.query, &doc.embedding);
                        (doc, score)
                    })
                    .collect();

                // Sort by score descending
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k
                let results: Vec<DocumentResult> = scored
                    .into_iter()
                    .take(req.k as usize)
                    .map(|(doc, score)| DocumentResult {
                        document: Some(doc),
                        score,
                    })
                    .collect();

                Ok(Response::new(SearchCollectionResponse {
                    results,
                    duration_us: start.elapsed().as_micros() as u64,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(SearchCollectionResponse {
                results: vec![],
                duration_us: 0,
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => match data.documents.get(&req.document_id) {
                Some(doc) => Ok(Response::new(GetDocumentResponse {
                    document: Some(doc.clone()),
                    error: String::new(),
                })),
                None => Ok(Response::new(GetDocumentResponse {
                    document: None,
                    error: format!("Document '{}' not found", req.document_id),
                })),
            },
            None => Ok(Response::new(GetDocumentResponse {
                document: None,
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }

    async fn delete_document(
        &self,
        request: Request<DeleteDocumentRequest>,
    ) -> Result<Response<DeleteDocumentResponse>, Status> {
        let req = request.into_inner();
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => match data.documents.remove(&req.document_id) {
                Some(_) => Ok(Response::new(DeleteDocumentResponse {
                    success: true,
                    error: String::new(),
                })),
                None => Ok(Response::new(DeleteDocumentResponse {
                    success: false,
                    error: format!("Document '{}' not found", req.document_id),
                })),
            },
            None => Ok(Response::new(DeleteDocumentResponse {
                success: false,
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }
}
