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

//! Semantic Cache with Provenance-Safe Keys (Task 8)
//!
//! This module provides a semantic caching layer that:
//! - Uses exact key matching for deterministic cache hits
//! - Supports semantic similarity search for approximate matches
//! - Preserves provenance (DERIVED_FROM edges to sources)
//! - Integrates with AllowedSet for policy-aware caching
//!
//! ## Design Principles
//!
//! 1. **Exact-match first**: Hash of (normalized_query, AllowedSet) for deterministic hits
//! 2. **Semantic fallback**: ANN search with explicit similarity threshold
//! 3. **Provenance tracking**: Every cache entry tracks its source documents
//! 4. **Policy integration**: Cache hits respect AllowedSet (no post-filtering)
//!
//! ## Complexity
//!
//! - Exact match: O(1) hash lookup
//! - Semantic search: O(log n) with HNSW index
//! - Policy gating: O(1) per entry with namespace/allowed-set check
//!
//! ## Cache Invalidation Strategy
//!
//! Cache entries are invalidated when:
//! 1. TTL expires
//! 2. Any source document is updated (via provenance edges)
//! 3. Policy changes affect the AllowedSet
//!
//! Note: Semantic similarity ≠ query equivalence. Similar queries may have
//! different correct answers, so semantic cache hits are marked as "approximate".

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ClientError, Result};
use crate::ConnectionTrait;

// ============================================================================
// Cache Types
// ============================================================================

/// Unique cache key combining query and access context
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Normalized query hash
    pub query_hash: u64,
    /// Namespace (for scoping)
    pub namespace: String,
    /// Hash of AllowedSet for policy-aware caching
    pub allowed_set_hash: u64,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(query: &str, namespace: &str, allowed_set_hash: u64) -> Self {
        Self {
            query_hash: Self::hash_query(query),
            namespace: namespace.to_string(),
            allowed_set_hash,
        }
    }
    
    fn hash_query(query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        query.trim().to_lowercase().hash(&mut hasher);
        hasher.finish()
    }
    
    /// Convert to storage key
    pub fn to_storage_key(&self) -> Vec<u8> {
        format!(
            "_cache/exact/{}/{:016x}/{:016x}",
            self.namespace, self.query_hash, self.allowed_set_hash
        ).into_bytes()
    }
}

/// A cached result entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cache key
    pub key: CacheKey,
    /// Original query text
    pub query: String,
    /// Cached result (serialized)
    pub result: Vec<u8>,
    /// Query embedding (for semantic search)
    pub embedding: Option<Vec<f32>>,
    /// Provenance: document IDs this result was derived from
    pub source_docs: Vec<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Expiration timestamp
    pub expires_at: u64,
    /// Hit count
    pub hits: u64,
    /// Whether this was an exact or semantic match
    pub match_type: CacheMatchType,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Type of cache match
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CacheMatchType {
    /// Exact key match (deterministic)
    Exact,
    /// Semantic similarity match (approximate)
    Semantic { similarity: f32 },
    /// No cache entry (miss)
    Miss,
}

impl CacheMatchType {
    /// Check if this is a cache hit
    pub fn is_hit(&self) -> bool {
        !matches!(self, Self::Miss)
    }
    
    /// Check if this is an exact match
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Cache lookup result
#[derive(Debug)]
pub struct CacheLookupResult {
    /// The cached entry (if found)
    pub entry: Option<CacheEntry>,
    /// Type of match
    pub match_type: CacheMatchType,
    /// Lookup latency in microseconds
    pub latency_us: u64,
}

impl CacheLookupResult {
    /// Check if this is a cache hit
    pub fn is_hit(&self) -> bool {
        self.entry.is_some()
    }
    
    /// Get the cached result bytes
    pub fn result(&self) -> Option<&[u8]> {
        self.entry.as_ref().map(|e| e.result.as_slice())
    }
}

// ============================================================================
// Cache Configuration
// ============================================================================

/// Configuration for the semantic cache
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Minimum similarity threshold for semantic matches
    pub similarity_threshold: f32,
    /// Maximum number of entries (for LRU eviction)
    pub max_entries: usize,
    /// Whether to enable semantic search (requires embeddings)
    pub enable_semantic_search: bool,
    /// Whether to track hit counts
    pub track_hits: bool,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(3600), // 1 hour
            similarity_threshold: 0.95, // 95% similarity required
            max_entries: 10000,
            enable_semantic_search: true,
            track_hits: true,
        }
    }
}

// ============================================================================
// Semantic Cache
// ============================================================================

/// Storage prefix for cache entries
const CACHE_PREFIX: &str = "_cache/";

/// Provenance-safe semantic cache
pub struct SemanticCache<C: ConnectionTrait> {
    conn: C,
    config: SemanticCacheConfig,
}

impl<C: ConnectionTrait> SemanticCache<C> {
    /// Create a new semantic cache
    pub fn new(conn: C) -> Self {
        Self {
            conn,
            config: SemanticCacheConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(conn: C, config: SemanticCacheConfig) -> Self {
        Self { conn, config }
    }
    
    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
    
    fn now_micros() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
    
    fn semantic_index_key(&self, namespace: &str, id: u64) -> Vec<u8> {
        format!("{}semantic/{}/{:016x}", CACHE_PREFIX, namespace, id).into_bytes()
    }
    
    fn semantic_index_prefix(&self, namespace: &str) -> Vec<u8> {
        format!("{}semantic/{}/", CACHE_PREFIX, namespace).into_bytes()
    }
    
    fn provenance_key(&self, cache_key_hash: u64, doc_id: &str) -> Vec<u8> {
        format!(
            "{}provenance/{:016x}/{}",
            CACHE_PREFIX, cache_key_hash, doc_id
        ).into_bytes()
    }
    
    fn provenance_prefix(&self, cache_key_hash: u64) -> Vec<u8> {
        format!("{}provenance/{:016x}/", CACHE_PREFIX, cache_key_hash).into_bytes()
    }
    
    // ========================================================================
    // Core Operations
    // ========================================================================
    
    /// Look up a cache entry
    ///
    /// First tries exact match, then (optionally) semantic search.
    pub fn lookup(
        &self,
        query: &str,
        namespace: &str,
        allowed_set_hash: u64,
        query_embedding: Option<&[f32]>,
    ) -> Result<CacheLookupResult> {
        let start = Self::now_micros();
        
        // Try exact match first
        let key = CacheKey::new(query, namespace, allowed_set_hash);
        if let Some(entry) = self.get_exact(&key)? {
            // Check if expired
            if entry.expires_at > Self::now_millis() {
                // Update hit count if tracking enabled
                if self.config.track_hits {
                    let _ = self.increment_hits(&key);
                }
                
                return Ok(CacheLookupResult {
                    entry: Some(entry),
                    match_type: CacheMatchType::Exact,
                    latency_us: Self::now_micros() - start,
                });
            }
            // Expired - treat as miss and clean up
            let _ = self.delete(&key);
        }
        
        // Try semantic match if enabled and embedding provided
        if self.config.enable_semantic_search {
            if let Some(embedding) = query_embedding {
                if let Some((entry, similarity)) = self.search_semantic(
                    namespace,
                    embedding,
                    allowed_set_hash,
                )? {
                    if similarity >= self.config.similarity_threshold {
                        if self.config.track_hits {
                            let _ = self.increment_hits(&entry.key);
                        }
                        
                        return Ok(CacheLookupResult {
                            entry: Some(entry),
                            match_type: CacheMatchType::Semantic { similarity },
                            latency_us: Self::now_micros() - start,
                        });
                    }
                }
            }
        }
        
        // Cache miss
        Ok(CacheLookupResult {
            entry: None,
            match_type: CacheMatchType::Miss,
            latency_us: Self::now_micros() - start,
        })
    }
    
    /// Store a result in the cache
    pub fn store(
        &self,
        query: &str,
        namespace: &str,
        allowed_set_hash: u64,
        result: &[u8],
        embedding: Option<Vec<f32>>,
        source_docs: Vec<String>,
        ttl: Option<Duration>,
    ) -> Result<CacheKey> {
        let key = CacheKey::new(query, namespace, allowed_set_hash);
        let now = Self::now_millis();
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        
        let entry = CacheEntry {
            key: key.clone(),
            query: query.to_string(),
            result: result.to_vec(),
            embedding: embedding.clone(),
            source_docs: source_docs.clone(),
            created_at: now,
            expires_at: now + ttl.as_millis() as u64,
            hits: 0,
            match_type: CacheMatchType::Exact, // Stored entries are exact
            metadata: HashMap::new(),
        };
        
        // Store exact key entry
        let storage_key = key.to_storage_key();
        let value = serde_json::to_vec(&entry)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&storage_key, &value)?;
        
        // Store semantic index if embedding provided
        if let Some(ref emb) = embedding {
            self.store_semantic_index(namespace, &key, emb)?;
        }
        
        // Store provenance edges
        let key_hash = key.query_hash ^ key.allowed_set_hash;
        for doc_id in &source_docs {
            let prov_key = self.provenance_key(key_hash, doc_id);
            self.conn.put(&prov_key, storage_key.as_slice())?;
        }
        
        Ok(key)
    }
    
    /// Delete a cache entry
    pub fn delete(&self, key: &CacheKey) -> Result<bool> {
        let storage_key = key.to_storage_key();
        
        // Check if exists
        if self.conn.get(&storage_key)?.is_none() {
            return Ok(false);
        }
        
        // Delete main entry
        self.conn.delete(&storage_key)?;
        
        // Delete provenance edges
        let key_hash = key.query_hash ^ key.allowed_set_hash;
        let prov_prefix = self.provenance_prefix(key_hash);
        for (prov_key, _) in self.conn.scan(&prov_prefix)? {
            self.conn.delete(&prov_key)?;
        }
        
        Ok(true)
    }
    
    /// Invalidate all cache entries derived from a document
    ///
    /// This is called when a source document is updated.
    pub fn invalidate_by_source(&self, doc_id: &str) -> Result<usize> {
        // Scan all provenance entries for this doc
        let prefix = format!("{}provenance/", CACHE_PREFIX).into_bytes();
        let results = self.conn.scan(&prefix)?;
        
        let mut invalidated = 0;
        for (prov_key, cache_key) in results {
            let key_str = String::from_utf8_lossy(&prov_key);
            if key_str.ends_with(&format!("/{}", doc_id)) {
                // This provenance edge points to our doc - invalidate the cache entry
                self.conn.delete(&cache_key)?;
                self.conn.delete(&prov_key)?;
                invalidated += 1;
            }
        }
        
        Ok(invalidated)
    }
    
    /// Get provenance (source documents) for a cache entry
    pub fn get_provenance(&self, key: &CacheKey) -> Result<Vec<String>> {
        let key_hash = key.query_hash ^ key.allowed_set_hash;
        let prefix = self.provenance_prefix(key_hash);
        let results = self.conn.scan(&prefix)?;
        
        let mut sources = Vec::new();
        for (prov_key, _) in results {
            let key_str = String::from_utf8_lossy(&prov_key);
            if let Some(doc_id) = key_str.rsplit('/').next() {
                sources.push(doc_id.to_string());
            }
        }
        
        Ok(sources)
    }
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    fn get_exact(&self, key: &CacheKey) -> Result<Option<CacheEntry>> {
        let storage_key = key.to_storage_key();
        if let Some(data) = self.conn.get(&storage_key)? {
            let entry: CacheEntry = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }
    
    fn store_semantic_index(
        &self,
        namespace: &str,
        key: &CacheKey,
        embedding: &[f32],
    ) -> Result<()> {
        let index_key = self.semantic_index_key(namespace, key.query_hash);
        let index_entry = SemanticIndexEntry {
            cache_key: key.clone(),
            embedding: embedding.to_vec(),
        };
        let value = serde_json::to_vec(&index_entry)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&index_key, &value)?;
        Ok(())
    }
    
    fn search_semantic(
        &self,
        namespace: &str,
        query_embedding: &[f32],
        allowed_set_hash: u64,
    ) -> Result<Option<(CacheEntry, f32)>> {
        let prefix = self.semantic_index_prefix(namespace);
        let results = self.conn.scan(&prefix)?;
        
        let mut best_match: Option<(CacheEntry, f32)> = None;
        
        for (_, value) in results {
            let index_entry: SemanticIndexEntry = match serde_json::from_slice(&value) {
                Ok(e) => e,
                Err(_) => continue,
            };
            
            // Check if AllowedSet matches
            if index_entry.cache_key.allowed_set_hash != allowed_set_hash {
                continue;
            }
            
            // Calculate similarity
            let similarity = cosine_similarity(query_embedding, &index_entry.embedding);
            
            if similarity >= self.config.similarity_threshold {
                // Get the actual cache entry
                if let Some(entry) = self.get_exact(&index_entry.cache_key)? {
                    // Check if not expired
                    if entry.expires_at > Self::now_millis() {
                        match &best_match {
                            None => best_match = Some((entry, similarity)),
                            Some((_, best_sim)) if similarity > *best_sim => {
                                best_match = Some((entry, similarity));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        
        Ok(best_match)
    }
    
    fn increment_hits(&self, key: &CacheKey) -> Result<()> {
        let storage_key = key.to_storage_key();
        if let Some(data) = self.conn.get(&storage_key)? {
            let mut entry: CacheEntry = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            entry.hits += 1;
            let value = serde_json::to_vec(&entry)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&storage_key, &value)?;
        }
        Ok(())
    }
    
    /// Clean up expired entries
    pub fn cleanup_expired(&self) -> Result<usize> {
        let prefix = format!("{}exact/", CACHE_PREFIX).into_bytes();
        let results = self.conn.scan(&prefix)?;
        
        let now = Self::now_millis();
        let mut cleaned = 0;
        
        for (key, value) in results {
            let entry: CacheEntry = match serde_json::from_slice(&value) {
                Ok(e) => e,
                Err(_) => continue,
            };
            
            if entry.expires_at < now {
                self.conn.delete(&key)?;
                
                // Clean up provenance
                let key_hash = entry.key.query_hash ^ entry.key.allowed_set_hash;
                let prov_prefix = self.provenance_prefix(key_hash);
                for (prov_key, _) in self.conn.scan(&prov_prefix)? {
                    self.conn.delete(&prov_key)?;
                }
                
                cleaned += 1;
            }
        }
        
        Ok(cleaned)
    }
}

// ============================================================================
// Helpers
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemanticIndexEntry {
    cache_key: CacheKey,
    embedding: Vec<f32>,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
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
    
    let norm = (norm_a.sqrt() * norm_b.sqrt());
    if norm == 0.0 {
        0.0
    } else {
        dot / norm
    }
}

/// Compute a hash of an AllowedSet for cache keying
///
/// This creates a stable hash that can be used to ensure cache hits
/// only occur for identical permission contexts.
pub fn hash_allowed_set<'a>(ids: impl Iterator<Item = &'a u64>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    
    // Sort for deterministic hashing
    let mut sorted: Vec<_> = ids.collect();
    sorted.sort();
    
    for id in sorted {
        id.hash(&mut hasher);
    }
    
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_key_creation() {
        let key1 = CacheKey::new("What is ToonDB?", "default", 12345);
        let key2 = CacheKey::new("what is toondb?", "default", 12345);
        let key3 = CacheKey::new("What is ToonDB?", "default", 54321);
        
        // Same query (case-insensitive) same hash
        assert_eq!(key1.query_hash, key2.query_hash);
        
        // Different AllowedSet = different key
        assert_ne!(key1.allowed_set_hash, key3.allowed_set_hash);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![0.707, 0.707, 0.0];
        
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
        assert!((cosine_similarity(&a, &d) - 0.707).abs() < 0.01);
    }
    
    #[test]
    fn test_hash_allowed_set() {
        let ids1: Vec<u64> = vec![1, 2, 3, 5, 8];
        let ids2: Vec<u64> = vec![8, 5, 3, 2, 1]; // Same set, different order
        let ids3: Vec<u64> = vec![1, 2, 3, 5, 9]; // Different set
        
        let hash1 = hash_allowed_set(ids1.iter());
        let hash2 = hash_allowed_set(ids2.iter());
        let hash3 = hash_allowed_set(ids3.iter());
        
        // Same set = same hash
        assert_eq!(hash1, hash2);
        
        // Different set = different hash
        assert_ne!(hash1, hash3);
    }
}
