// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Semantic Cache Service gRPC Implementation
//!
//! Provides semantic caching for LLM queries via gRPC.

use crate::proto::{
    semantic_cache_service_server::{SemanticCacheService, SemanticCacheServiceServer},
    SemanticCacheGetRequest, SemanticCacheGetResponse, SemanticCacheInvalidateRequest,
    SemanticCacheInvalidateResponse, SemanticCachePutRequest, SemanticCachePutResponse,
    SemanticCacheStatsRequest, SemanticCacheStatsResponse,
};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};

/// Cache entry with embedding and TTL
struct CacheEntry {
    key: String,
    value: String,
    embedding: Vec<f32>,
    expires_at: Option<Instant>,
}

/// Cache statistics
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

/// In-memory cache per cache name
struct CacheInstance {
    entries: DashMap<String, CacheEntry>,
    stats: CacheStats,
}

impl CacheInstance {
    fn new() -> Self {
        Self {
            entries: DashMap::new(),
            stats: CacheStats {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
            },
        }
    }
}

/// Semantic Cache gRPC Server
pub struct SemanticCacheServer {
    caches: DashMap<String, CacheInstance>,
}

impl SemanticCacheServer {
    pub fn new() -> Self {
        Self {
            caches: DashMap::new(),
        }
    }

    pub fn into_service(self) -> SemanticCacheServiceServer<Self> {
        SemanticCacheServiceServer::new(self)
    }

    fn get_or_create_cache(&self, name: &str) -> dashmap::mapref::one::Ref<'_, String, CacheInstance> {
        if !self.caches.contains_key(name) {
            self.caches.insert(name.to_string(), CacheInstance::new());
        }
        self.caches.get(name).unwrap()
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

impl Default for SemanticCacheServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl SemanticCacheService for SemanticCacheServer {
    async fn get(
        &self,
        request: Request<SemanticCacheGetRequest>,
    ) -> Result<Response<SemanticCacheGetResponse>, Status> {
        let req = request.into_inner();
        let cache = self.get_or_create_cache(&req.cache_name);
        let now = Instant::now();

        let mut best_match: Option<(String, String, f32)> = None;
        let threshold = if req.similarity_threshold > 0.0 {
            req.similarity_threshold
        } else {
            0.85 // Default threshold
        };

        // Search for semantically similar entries
        for entry in cache.entries.iter() {
            let e = entry.value();

            // Check TTL
            if let Some(expires_at) = e.expires_at {
                if now > expires_at {
                    continue;
                }
            }

            let similarity = Self::cosine_similarity(&req.query_embedding, &e.embedding);
            if similarity >= threshold {
                match &best_match {
                    Some((_, _, best_score)) if similarity > *best_score => {
                        best_match = Some((e.key.clone(), e.value.clone(), similarity));
                    }
                    None => {
                        best_match = Some((e.key.clone(), e.value.clone(), similarity));
                    }
                    _ => {}
                }
            }
        }

        match best_match {
            Some((key, value, score)) => {
                cache.stats.hits.fetch_add(1, Ordering::Relaxed);
                Ok(Response::new(SemanticCacheGetResponse {
                    hit: true,
                    cached_value: value,
                    similarity_score: score,
                    matched_key: key,
                }))
            }
            None => {
                cache.stats.misses.fetch_add(1, Ordering::Relaxed);
                Ok(Response::new(SemanticCacheGetResponse {
                    hit: false,
                    cached_value: String::new(),
                    similarity_score: 0.0,
                    matched_key: String::new(),
                }))
            }
        }
    }

    async fn put(
        &self,
        request: Request<SemanticCachePutRequest>,
    ) -> Result<Response<SemanticCachePutResponse>, Status> {
        let req = request.into_inner();

        if !self.caches.contains_key(&req.cache_name) {
            self.caches.insert(req.cache_name.clone(), CacheInstance::new());
        }

        let cache = self.caches.get(&req.cache_name).unwrap();

        let expires_at = if req.ttl_seconds > 0 {
            Some(Instant::now() + Duration::from_secs(req.ttl_seconds))
        } else {
            None
        };

        let entry = CacheEntry {
            key: req.key.clone(),
            value: req.value,
            embedding: req.key_embedding,
            expires_at,
        };

        cache.entries.insert(req.key, entry);

        Ok(Response::new(SemanticCachePutResponse {
            success: true,
            error: String::new(),
        }))
    }

    async fn invalidate(
        &self,
        request: Request<SemanticCacheInvalidateRequest>,
    ) -> Result<Response<SemanticCacheInvalidateResponse>, Status> {
        let req = request.into_inner();

        let count = if let Some(cache) = self.caches.get(&req.cache_name) {
            if req.pattern.is_empty() {
                let count = cache.entries.len();
                cache.entries.clear();
                count as u32
            } else {
                let mut count = 0u32;
                cache.entries.retain(|k, _| {
                    if k.contains(&req.pattern) {
                        count += 1;
                        false
                    } else {
                        true
                    }
                });
                count
            }
        } else {
            0
        };

        Ok(Response::new(SemanticCacheInvalidateResponse {
            invalidated_count: count,
        }))
    }

    async fn get_stats(
        &self,
        request: Request<SemanticCacheStatsRequest>,
    ) -> Result<Response<SemanticCacheStatsResponse>, Status> {
        let req = request.into_inner();

        match self.caches.get(&req.cache_name) {
            Some(cache) => {
                let hits = cache.stats.hits.load(Ordering::Relaxed);
                let misses = cache.stats.misses.load(Ordering::Relaxed);
                let total = hits + misses;
                let hit_rate = if total > 0 {
                    hits as f32 / total as f32
                } else {
                    0.0
                };

                Ok(Response::new(SemanticCacheStatsResponse {
                    hits,
                    misses,
                    entry_count: cache.entries.len() as u64,
                    hit_rate,
                }))
            }
            None => Ok(Response::new(SemanticCacheStatsResponse {
                hits: 0,
                misses: 0,
                entry_count: 0,
                hit_rate: 0.0,
            })),
        }
    }
}
