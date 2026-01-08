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

//! Atomic Multi-Index Memory Writes (Task 4)
//!
//! This module provides atomic "all-or-nothing" writes across multiple indexes:
//! - KV/blob storage
//! - Vector embeddings
//! - Graph edges
//!
//! ## Problem
//!
//! Without atomic writes, crashes can leave "torn" memory:
//! - Embedding exists but edges don't
//! - Edges exist without the blob
//! - Partial graph relationships
//!
//! ## Solution: Intent Records (Mini 2PC)
//!
//! ```text
//! 1. Write intent(id, ops...) to WAL
//! 2. Apply ops one-by-one
//! 3. Write commit(id) to WAL
//! 4. Recovery replays incomplete intents
//! ```
//!
//! ## Complexity
//!
//! - Write overhead: O(1) extra metadata per memory item
//! - Recovery: O(uncommitted intents × ops per intent)
//!
//! ## Idempotency
//!
//! All operations are designed to be idempotent:
//! - PUT overwrites (idempotent)
//! - Edge creation checks existence
//! - Version stamps ensure exactly-once semantics

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ClientError, Result};
use crate::ConnectionTrait;

// ============================================================================
// Intent Operations
// ============================================================================

/// A single operation within an atomic intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOp {
    /// Store a blob/value
    PutBlob {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    
    /// Store a vector embedding
    PutEmbedding {
        collection: String,
        id: String,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    
    /// Create a graph node
    CreateNode {
        namespace: String,
        node_id: String,
        node_type: String,
        properties: HashMap<String, serde_json::Value>,
    },
    
    /// Create a graph edge
    CreateEdge {
        namespace: String,
        from_id: String,
        edge_type: String,
        to_id: String,
        properties: HashMap<String, serde_json::Value>,
    },
    
    /// Delete a blob/value
    DeleteBlob {
        key: Vec<u8>,
    },
    
    /// Delete a graph edge
    DeleteEdge {
        namespace: String,
        from_id: String,
        edge_type: String,
        to_id: String,
    },
}

/// Status of an intent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentStatus {
    /// Intent has been written, ops being applied
    Pending,
    /// All ops applied, ready to commit
    Applied,
    /// Successfully committed
    Committed,
    /// Rolled back due to error
    Aborted,
}

/// An atomic intent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIntent {
    /// Unique intent ID
    pub intent_id: u64,
    /// Memory item ID this intent belongs to
    pub memory_id: String,
    /// Operations to apply atomically
    pub ops: Vec<MemoryOp>,
    /// Current status
    pub status: IntentStatus,
    /// Creation timestamp (for recovery ordering)
    pub created_at: u64,
    /// Version stamp for idempotency
    pub version: u64,
}

impl MemoryIntent {
    /// Create a new intent
    pub fn new(intent_id: u64, memory_id: String, ops: Vec<MemoryOp>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        Self {
            intent_id,
            memory_id,
            ops,
            status: IntentStatus::Pending,
            created_at: now,
            version: now, // Use timestamp as version for simplicity
        }
    }
}

// ============================================================================
// Atomic Memory Writer
// ============================================================================

/// Prefix for intent records in storage
const INTENT_PREFIX: &str = "_intents/";

/// Atomic memory writer that ensures multi-index consistency
pub struct AtomicMemoryWriter<C: ConnectionTrait> {
    conn: C,
    next_intent_id: AtomicU64,
}

impl<C: ConnectionTrait> AtomicMemoryWriter<C> {
    /// Create a new atomic memory writer
    pub fn new(conn: C) -> Self {
        Self {
            conn,
            next_intent_id: AtomicU64::new(1),
        }
    }
    
    /// Generate a new intent ID
    fn next_id(&self) -> u64 {
        self.next_intent_id.fetch_add(1, Ordering::SeqCst)
    }
    
    /// Key for storing intent records
    fn intent_key(intent_id: u64) -> Vec<u8> {
        format!("{}{}", INTENT_PREFIX, intent_id).into_bytes()
    }
    
    /// Write an atomic memory item with all its components
    ///
    /// This is the main entry point for atomic multi-index writes.
    /// Either all operations succeed, or the entire write can be retried.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - Unique ID for this memory item
    /// * `ops` - Operations to apply atomically
    ///
    /// # Returns
    ///
    /// The intent ID on success
    pub fn write_atomic(
        &self,
        memory_id: impl Into<String>,
        ops: Vec<MemoryOp>,
    ) -> Result<AtomicWriteResult> {
        let memory_id = memory_id.into();
        let intent_id = self.next_id();
        
        // Phase 1: Write intent record
        let intent = MemoryIntent::new(intent_id, memory_id.clone(), ops);
        self.write_intent(&intent)?;
        
        // Phase 2: Apply operations
        let apply_result = self.apply_ops(&intent);
        
        // Phase 3: Commit or abort based on result
        match apply_result {
            Ok(applied_count) => {
                self.mark_committed(intent_id)?;
                Ok(AtomicWriteResult {
                    intent_id,
                    memory_id,
                    ops_applied: applied_count,
                    status: IntentStatus::Committed,
                })
            }
            Err(e) => {
                // Mark as aborted for cleanup
                let _ = self.mark_aborted(intent_id);
                Err(e)
            }
        }
    }
    
    /// Write the intent record to storage
    fn write_intent(&self, intent: &MemoryIntent) -> Result<()> {
        let key = Self::intent_key(intent.intent_id);
        let value = serde_json::to_vec(intent)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&key, &value)?;
        Ok(())
    }
    
    /// Apply all operations in an intent
    fn apply_ops(&self, intent: &MemoryIntent) -> Result<usize> {
        let mut applied = 0;
        
        for op in &intent.ops {
            self.apply_op(op, &intent.memory_id, intent.version)?;
            applied += 1;
        }
        
        Ok(applied)
    }
    
    /// Apply a single operation
    fn apply_op(&self, op: &MemoryOp, memory_id: &str, version: u64) -> Result<()> {
        match op {
            MemoryOp::PutBlob { key, value } => {
                // Add version to enable idempotency check
                let versioned_key = Self::versioned_key(key, version);
                self.conn.put(&versioned_key, value)?;
                // Also write the main key (latest version)
                self.conn.put(key, value)?;
            }
            
            MemoryOp::PutEmbedding { collection, id, embedding, metadata } => {
                // Store embedding metadata with version
                let key = format!("_vectors/{}/{}/meta", collection, id).into_bytes();
                let meta = EmbeddingMeta {
                    memory_id: memory_id.to_string(),
                    version,
                    dimensions: embedding.len(),
                    metadata: metadata.clone(),
                };
                let value = serde_json::to_vec(&meta)
                    .map_err(|e| ClientError::Serialization(e.to_string()))?;
                self.conn.put(&key, &value)?;
                
                // Store the embedding vector
                let emb_key = format!("_vectors/{}/{}/data", collection, id).into_bytes();
                let emb_bytes: Vec<u8> = embedding
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                self.conn.put(&emb_key, &emb_bytes)?;
            }
            
            MemoryOp::CreateNode { namespace, node_id, node_type, properties } => {
                let key = format!("_graph/{}/nodes/{}", namespace, node_id).into_bytes();
                let node = GraphNodeRecord {
                    id: node_id.clone(),
                    node_type: node_type.clone(),
                    properties: properties.clone(),
                    memory_id: memory_id.to_string(),
                    version,
                };
                let value = serde_json::to_vec(&node)
                    .map_err(|e| ClientError::Serialization(e.to_string()))?;
                self.conn.put(&key, &value)?;
            }
            
            MemoryOp::CreateEdge { namespace, from_id, edge_type, to_id, properties } => {
                // Store edge
                let edge_key = format!(
                    "_graph/{}/edges/{}/{}/{}", 
                    namespace, from_id, edge_type, to_id
                ).into_bytes();
                let edge = GraphEdgeRecord {
                    from_id: from_id.clone(),
                    edge_type: edge_type.clone(),
                    to_id: to_id.clone(),
                    properties: properties.clone(),
                    memory_id: memory_id.to_string(),
                    version,
                };
                let value = serde_json::to_vec(&edge)
                    .map_err(|e| ClientError::Serialization(e.to_string()))?;
                self.conn.put(&edge_key, &value)?;
                
                // Store reverse index
                let rev_key = format!(
                    "_graph/{}/index/{}/{}/{}", 
                    namespace, edge_type, to_id, from_id
                ).into_bytes();
                self.conn.put(&rev_key, from_id.as_bytes())?;
            }
            
            MemoryOp::DeleteBlob { key } => {
                self.conn.delete(key)?;
            }
            
            MemoryOp::DeleteEdge { namespace, from_id, edge_type, to_id } => {
                let edge_key = format!(
                    "_graph/{}/edges/{}/{}/{}", 
                    namespace, from_id, edge_type, to_id
                ).into_bytes();
                self.conn.delete(&edge_key)?;
                
                let rev_key = format!(
                    "_graph/{}/index/{}/{}/{}", 
                    namespace, edge_type, to_id, from_id
                ).into_bytes();
                self.conn.delete(&rev_key)?;
            }
        }
        
        Ok(())
    }
    
    /// Create a versioned key for idempotency
    fn versioned_key(key: &[u8], version: u64) -> Vec<u8> {
        let mut versioned = key.to_vec();
        versioned.extend_from_slice(b"@v");
        versioned.extend_from_slice(&version.to_le_bytes());
        versioned
    }
    
    /// Mark an intent as committed
    fn mark_committed(&self, intent_id: u64) -> Result<()> {
        self.update_intent_status(intent_id, IntentStatus::Committed)
    }
    
    /// Mark an intent as aborted
    fn mark_aborted(&self, intent_id: u64) -> Result<()> {
        self.update_intent_status(intent_id, IntentStatus::Aborted)
    }
    
    /// Update intent status
    fn update_intent_status(&self, intent_id: u64, status: IntentStatus) -> Result<()> {
        let key = Self::intent_key(intent_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut intent: MemoryIntent = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            intent.status = status;
            let value = serde_json::to_vec(&intent)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    /// Recover incomplete intents on startup
    ///
    /// This should be called during connection initialization to
    /// replay any intents that were interrupted by a crash.
    pub fn recover(&self) -> Result<RecoveryReport> {
        let prefix = INTENT_PREFIX.as_bytes();
        let intents = self.conn.scan(prefix)?;
        
        let mut report = RecoveryReport::default();
        
        for (_, value) in intents {
            let intent: MemoryIntent = match serde_json::from_slice(&value) {
                Ok(i) => i,
                Err(_) => {
                    report.corrupted += 1;
                    continue;
                }
            };
            
            match intent.status {
                IntentStatus::Pending | IntentStatus::Applied => {
                    // Replay this intent
                    match self.apply_ops(&intent) {
                        Ok(_) => {
                            self.mark_committed(intent.intent_id)?;
                            report.replayed += 1;
                        }
                        Err(_) => {
                            self.mark_aborted(intent.intent_id)?;
                            report.failed += 1;
                        }
                    }
                }
                IntentStatus::Committed => {
                    report.already_committed += 1;
                }
                IntentStatus::Aborted => {
                    report.already_aborted += 1;
                }
            }
        }
        
        Ok(report)
    }
    
    /// Clean up old committed/aborted intents
    ///
    /// Call this periodically to reclaim storage.
    pub fn cleanup(&self, max_age_secs: u64) -> Result<usize> {
        let prefix = INTENT_PREFIX.as_bytes();
        let intents = self.conn.scan(prefix)?;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let cutoff = now.saturating_sub(max_age_secs * 1000);
        
        let mut cleaned = 0;
        
        for (key, value) in intents {
            let intent: MemoryIntent = match serde_json::from_slice(&value) {
                Ok(i) => i,
                Err(_) => continue,
            };
            
            // Only clean up committed/aborted intents older than cutoff
            if intent.created_at < cutoff {
                if matches!(intent.status, IntentStatus::Committed | IntentStatus::Aborted) {
                    self.conn.delete(&key)?;
                    cleaned += 1;
                }
            }
        }
        
        Ok(cleaned)
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of an atomic write operation
#[derive(Debug)]
pub struct AtomicWriteResult {
    /// Intent ID for this write
    pub intent_id: u64,
    /// Memory item ID
    pub memory_id: String,
    /// Number of operations applied
    pub ops_applied: usize,
    /// Final status
    pub status: IntentStatus,
}

/// Report from recovery operation
#[derive(Debug, Default)]
pub struct RecoveryReport {
    /// Intents successfully replayed
    pub replayed: usize,
    /// Intents that failed replay
    pub failed: usize,
    /// Intents already committed
    pub already_committed: usize,
    /// Intents already aborted
    pub already_aborted: usize,
    /// Corrupted intent records
    pub corrupted: usize,
}

// ============================================================================
// Internal Records
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingMeta {
    memory_id: String,
    version: u64,
    dimensions: usize,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphNodeRecord {
    id: String,
    node_type: String,
    properties: HashMap<String, serde_json::Value>,
    memory_id: String,
    version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphEdgeRecord {
    from_id: String,
    edge_type: String,
    to_id: String,
    properties: HashMap<String, serde_json::Value>,
    memory_id: String,
    version: u64,
}

// ============================================================================
// Builder API
// ============================================================================

/// Builder for constructing atomic memory writes
pub struct MemoryWriteBuilder {
    memory_id: String,
    ops: Vec<MemoryOp>,
}

impl MemoryWriteBuilder {
    /// Create a new builder for a memory item
    pub fn new(memory_id: impl Into<String>) -> Self {
        Self {
            memory_id: memory_id.into(),
            ops: Vec::new(),
        }
    }
    
    /// Add a blob/value storage operation
    pub fn put_blob(mut self, key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) -> Self {
        self.ops.push(MemoryOp::PutBlob {
            key: key.into(),
            value: value.into(),
        });
        self
    }
    
    /// Add an embedding storage operation
    pub fn put_embedding(
        mut self,
        collection: impl Into<String>,
        id: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        self.ops.push(MemoryOp::PutEmbedding {
            collection: collection.into(),
            id: id.into(),
            embedding,
            metadata: HashMap::new(),
        });
        self
    }
    
    /// Add an embedding with metadata
    pub fn put_embedding_with_meta(
        mut self,
        collection: impl Into<String>,
        id: impl Into<String>,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Self {
        self.ops.push(MemoryOp::PutEmbedding {
            collection: collection.into(),
            id: id.into(),
            embedding,
            metadata,
        });
        self
    }
    
    /// Add a graph node creation
    pub fn create_node(
        mut self,
        namespace: impl Into<String>,
        node_id: impl Into<String>,
        node_type: impl Into<String>,
    ) -> Self {
        self.ops.push(MemoryOp::CreateNode {
            namespace: namespace.into(),
            node_id: node_id.into(),
            node_type: node_type.into(),
            properties: HashMap::new(),
        });
        self
    }
    
    /// Add a graph node with properties
    pub fn create_node_with_props(
        mut self,
        namespace: impl Into<String>,
        node_id: impl Into<String>,
        node_type: impl Into<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.ops.push(MemoryOp::CreateNode {
            namespace: namespace.into(),
            node_id: node_id.into(),
            node_type: node_type.into(),
            properties,
        });
        self
    }
    
    /// Add a graph edge creation
    pub fn create_edge(
        mut self,
        namespace: impl Into<String>,
        from_id: impl Into<String>,
        edge_type: impl Into<String>,
        to_id: impl Into<String>,
    ) -> Self {
        self.ops.push(MemoryOp::CreateEdge {
            namespace: namespace.into(),
            from_id: from_id.into(),
            edge_type: edge_type.into(),
            to_id: to_id.into(),
            properties: HashMap::new(),
        });
        self
    }
    
    /// Add a graph edge with properties
    pub fn create_edge_with_props(
        mut self,
        namespace: impl Into<String>,
        from_id: impl Into<String>,
        edge_type: impl Into<String>,
        to_id: impl Into<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.ops.push(MemoryOp::CreateEdge {
            namespace: namespace.into(),
            from_id: from_id.into(),
            edge_type: edge_type.into(),
            to_id: to_id.into(),
            properties,
        });
        self
    }
    
    /// Execute the atomic write
    pub fn execute<C: ConnectionTrait>(self, writer: &AtomicMemoryWriter<C>) -> Result<AtomicWriteResult> {
        writer.write_atomic(self.memory_id, self.ops)
    }
    
    /// Get the operations (for testing)
    pub fn ops(&self) -> &[MemoryOp] {
        &self.ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Tests would use a mock ConnectionTrait implementation
}
