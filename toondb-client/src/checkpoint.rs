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

//! Workflow Checkpoint Storage Primitives (Task 6)
//!
//! This module provides durable workflow state persistence as a storage primitive,
//! NOT an orchestration framework. External frameworks (Temporal, Airflow, custom)
//! can use this as a clean storage API for:
//!
//! - Checkpoint storage: `CHECKPOINT(run_id, node_id, state_blob)`
//! - Checkpoint retrieval: `RESUME(run_id)`
//! - Run metadata
//!
//! ## Architecture Decision
//!
//! ToonDB is the **source of truth** for state (runs/events/checkpoints).
//! Scheduling and execution are handled by external orchestrators.
//! This separation of concerns:
//! - Preserves clean API boundaries
//! - Avoids building a competing orchestration framework
//! - Enables easy integration with existing tools
//!
//! ## Complexity
//!
//! - Checkpoint write: O(S) where S is state size
//! - Checkpoint read: O(S)
//! - List checkpoints: O(log N + k) with index
//! - Recovery: O(S + m) where m is events since snapshot
//!
//! ## Schema
//!
//! ```text
//! _checkpoints/{run_id}/meta              -> RunMetadata
//! _checkpoints/{run_id}/nodes/{node_id}   -> CheckpointData
//! _checkpoints/{run_id}/events/{seq}      -> EventData
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ClientError, Result};
use crate::ConnectionTrait;

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a workflow run
pub type RunId = String;

/// Unique identifier for a node within a run
pub type NodeId = String;

/// Sequence number for events
pub type SeqNo = u64;

/// Metadata about a workflow run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Run ID
    pub run_id: RunId,
    /// Workflow name/type
    pub workflow: String,
    /// Run status
    pub status: RunStatus,
    /// Creation timestamp (millis since epoch)
    pub created_at: u64,
    /// Last updated timestamp
    pub updated_at: u64,
    /// Custom run parameters
    pub params: HashMap<String, serde_json::Value>,
    /// Latest checkpoint sequence
    pub latest_checkpoint_seq: u64,
    /// Latest event sequence
    pub latest_event_seq: u64,
}

/// Status of a workflow run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run is active
    Running,
    /// Run completed successfully
    Completed,
    /// Run failed
    Failed,
    /// Run was cancelled
    Cancelled,
    /// Run is paused (can be resumed)
    Paused,
}

/// A checkpoint snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Run ID
    pub run_id: RunId,
    /// Node ID within the workflow
    pub node_id: NodeId,
    /// Checkpoint sequence number
    pub seq: SeqNo,
    /// State blob (opaque to ToonDB)
    #[serde(with = "hex_serde")]
    pub state: Vec<u8>,
    /// Checkpoint timestamp
    pub timestamp: u64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Metadata about a checkpoint (without the state blob)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Run ID
    pub run_id: RunId,
    /// Node ID
    pub node_id: NodeId,
    /// Sequence number
    pub seq: SeqNo,
    /// Timestamp
    pub timestamp: u64,
    /// State size in bytes
    pub state_size: usize,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// An event in the workflow event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEvent {
    /// Run ID
    pub run_id: RunId,
    /// Event sequence number
    pub seq: SeqNo,
    /// Event type
    pub event_type: String,
    /// Event timestamp
    pub timestamp: u64,
    /// Node that emitted the event (optional)
    pub node_id: Option<NodeId>,
    /// Event payload
    pub payload: serde_json::Value,
}

// ============================================================================
// Checkpoint Store Trait
// ============================================================================

/// Trait defining the checkpoint storage interface
///
/// This is the primary API for external orchestrators to use.
pub trait CheckpointStore {
    /// Save a checkpoint for a node
    fn save_checkpoint(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
        state: &[u8],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<CheckpointMeta>;
    
    /// Load the latest checkpoint for a node
    fn load_checkpoint(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
    ) -> Result<Option<Checkpoint>>;
    
    /// Load a specific checkpoint by sequence number
    fn load_checkpoint_at(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
        seq: SeqNo,
    ) -> Result<Option<Checkpoint>>;
    
    /// List all checkpoints for a run
    fn list_checkpoints(&self, run_id: &RunId) -> Result<Vec<CheckpointMeta>>;
    
    /// List checkpoints for a specific node
    fn list_node_checkpoints(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
    ) -> Result<Vec<CheckpointMeta>>;
    
    /// Create a new run
    fn create_run(
        &self,
        run_id: &RunId,
        workflow: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<RunMetadata>;
    
    /// Get run metadata
    fn get_run(&self, run_id: &RunId) -> Result<Option<RunMetadata>>;
    
    /// Update run status
    fn update_run_status(&self, run_id: &RunId, status: RunStatus) -> Result<()>;
    
    /// Append an event to the run's event log
    fn append_event(&self, event: WorkflowEvent) -> Result<SeqNo>;
    
    /// Get events for a run (optionally since a sequence number)
    fn get_events(
        &self,
        run_id: &RunId,
        since_seq: Option<SeqNo>,
        limit: usize,
    ) -> Result<Vec<WorkflowEvent>>;
    
    /// Delete a run and all its checkpoints/events
    fn delete_run(&self, run_id: &RunId) -> Result<bool>;
}

// ============================================================================
// Default Implementation
// ============================================================================

/// Prefix for checkpoint storage
const CHECKPOINT_PREFIX: &str = "_checkpoints/";

/// Default checkpoint store implementation using ConnectionTrait
pub struct DefaultCheckpointStore<C: ConnectionTrait> {
    conn: C,
}

impl<C: ConnectionTrait> DefaultCheckpointStore<C> {
    /// Create a new checkpoint store
    pub fn new(conn: C) -> Self {
        Self { conn }
    }
    
    fn run_meta_key(run_id: &RunId) -> Vec<u8> {
        format!("{}{}/meta", CHECKPOINT_PREFIX, run_id).into_bytes()
    }
    
    fn checkpoint_key(run_id: &RunId, node_id: &NodeId, seq: SeqNo) -> Vec<u8> {
        format!(
            "{}{}/nodes/{}/{:016x}",
            CHECKPOINT_PREFIX, run_id, node_id, seq
        ).into_bytes()
    }
    
    fn checkpoint_prefix(run_id: &RunId, node_id: &NodeId) -> Vec<u8> {
        format!(
            "{}{}/nodes/{}/",
            CHECKPOINT_PREFIX, run_id, node_id
        ).into_bytes()
    }
    
    fn all_checkpoints_prefix(run_id: &RunId) -> Vec<u8> {
        format!("{}{}/nodes/", CHECKPOINT_PREFIX, run_id).into_bytes()
    }
    
    fn event_key(run_id: &RunId, seq: SeqNo) -> Vec<u8> {
        format!(
            "{}{}/events/{:016x}",
            CHECKPOINT_PREFIX, run_id, seq
        ).into_bytes()
    }
    
    fn events_prefix(run_id: &RunId) -> Vec<u8> {
        format!("{}{}/events/", CHECKPOINT_PREFIX, run_id).into_bytes()
    }
    
    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
    
    fn get_and_increment_checkpoint_seq(&self, run_id: &RunId) -> Result<SeqNo> {
        let meta = self.get_run(run_id)?
            .ok_or_else(|| ClientError::NotFound(format!("Run {} not found", run_id)))?;
        
        let new_seq = meta.latest_checkpoint_seq + 1;
        self.update_checkpoint_seq(run_id, new_seq)?;
        Ok(new_seq)
    }
    
    fn get_and_increment_event_seq(&self, run_id: &RunId) -> Result<SeqNo> {
        let meta = self.get_run(run_id)?
            .ok_or_else(|| ClientError::NotFound(format!("Run {} not found", run_id)))?;
        
        let new_seq = meta.latest_event_seq + 1;
        self.update_event_seq(run_id, new_seq)?;
        Ok(new_seq)
    }
    
    fn update_checkpoint_seq(&self, run_id: &RunId, seq: SeqNo) -> Result<()> {
        let key = Self::run_meta_key(run_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut meta: RunMetadata = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            meta.latest_checkpoint_seq = seq;
            meta.updated_at = Self::now_millis();
            let value = serde_json::to_vec(&meta)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    fn update_event_seq(&self, run_id: &RunId, seq: SeqNo) -> Result<()> {
        let key = Self::run_meta_key(run_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut meta: RunMetadata = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            meta.latest_event_seq = seq;
            meta.updated_at = Self::now_millis();
            let value = serde_json::to_vec(&meta)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
}

impl<C: ConnectionTrait> CheckpointStore for DefaultCheckpointStore<C> {
    fn save_checkpoint(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
        state: &[u8],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<CheckpointMeta> {
        let seq = self.get_and_increment_checkpoint_seq(run_id)?;
        let timestamp = Self::now_millis();
        
        let checkpoint = Checkpoint {
            run_id: run_id.clone(),
            node_id: node_id.clone(),
            seq,
            state: state.to_vec(),
            timestamp,
            metadata: metadata.unwrap_or_default(),
        };
        
        let key = Self::checkpoint_key(run_id, node_id, seq);
        let value = serde_json::to_vec(&checkpoint)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&key, &value)?;
        
        Ok(CheckpointMeta {
            run_id: run_id.clone(),
            node_id: node_id.clone(),
            seq,
            timestamp,
            state_size: state.len(),
            metadata: checkpoint.metadata,
        })
    }
    
    fn load_checkpoint(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
    ) -> Result<Option<Checkpoint>> {
        let prefix = Self::checkpoint_prefix(run_id, node_id);
        let results = self.conn.scan(&prefix)?;
        
        // Get the latest (highest seq)
        if let Some((_, value)) = results.into_iter().last() {
            let checkpoint: Checkpoint = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            Ok(Some(checkpoint))
        } else {
            Ok(None)
        }
    }
    
    fn load_checkpoint_at(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
        seq: SeqNo,
    ) -> Result<Option<Checkpoint>> {
        let key = Self::checkpoint_key(run_id, node_id, seq);
        if let Some(data) = self.conn.get(&key)? {
            let checkpoint: Checkpoint = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            Ok(Some(checkpoint))
        } else {
            Ok(None)
        }
    }
    
    fn list_checkpoints(&self, run_id: &RunId) -> Result<Vec<CheckpointMeta>> {
        let prefix = Self::all_checkpoints_prefix(run_id);
        let results = self.conn.scan(&prefix)?;
        
        let mut metas = Vec::new();
        for (_, value) in results {
            let cp: Checkpoint = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            metas.push(CheckpointMeta {
                run_id: cp.run_id,
                node_id: cp.node_id,
                seq: cp.seq,
                timestamp: cp.timestamp,
                state_size: cp.state.len(),
                metadata: cp.metadata,
            });
        }
        
        Ok(metas)
    }
    
    fn list_node_checkpoints(
        &self,
        run_id: &RunId,
        node_id: &NodeId,
    ) -> Result<Vec<CheckpointMeta>> {
        let prefix = Self::checkpoint_prefix(run_id, node_id);
        let results = self.conn.scan(&prefix)?;
        
        let mut metas = Vec::new();
        for (_, value) in results {
            let cp: Checkpoint = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            metas.push(CheckpointMeta {
                run_id: cp.run_id,
                node_id: cp.node_id,
                seq: cp.seq,
                timestamp: cp.timestamp,
                state_size: cp.state.len(),
                metadata: cp.metadata,
            });
        }
        
        Ok(metas)
    }
    
    fn create_run(
        &self,
        run_id: &RunId,
        workflow: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<RunMetadata> {
        let now = Self::now_millis();
        
        let meta = RunMetadata {
            run_id: run_id.clone(),
            workflow: workflow.to_string(),
            status: RunStatus::Running,
            created_at: now,
            updated_at: now,
            params,
            latest_checkpoint_seq: 0,
            latest_event_seq: 0,
        };
        
        let key = Self::run_meta_key(run_id);
        let value = serde_json::to_vec(&meta)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&key, &value)?;
        
        Ok(meta)
    }
    
    fn get_run(&self, run_id: &RunId) -> Result<Option<RunMetadata>> {
        let key = Self::run_meta_key(run_id);
        if let Some(data) = self.conn.get(&key)? {
            let meta: RunMetadata = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            Ok(Some(meta))
        } else {
            Ok(None)
        }
    }
    
    fn update_run_status(&self, run_id: &RunId, status: RunStatus) -> Result<()> {
        let key = Self::run_meta_key(run_id);
        if let Some(data) = self.conn.get(&key)? {
            let mut meta: RunMetadata = serde_json::from_slice(&data)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            meta.status = status;
            meta.updated_at = Self::now_millis();
            let value = serde_json::to_vec(&meta)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            self.conn.put(&key, &value)?;
        }
        Ok(())
    }
    
    fn append_event(&self, mut event: WorkflowEvent) -> Result<SeqNo> {
        let seq = self.get_and_increment_event_seq(&event.run_id)?;
        event.seq = seq;
        event.timestamp = Self::now_millis();
        
        let key = Self::event_key(&event.run_id, seq);
        let value = serde_json::to_vec(&event)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&key, &value)?;
        
        Ok(seq)
    }
    
    fn get_events(
        &self,
        run_id: &RunId,
        since_seq: Option<SeqNo>,
        limit: usize,
    ) -> Result<Vec<WorkflowEvent>> {
        let prefix = Self::events_prefix(run_id);
        let results = self.conn.scan(&prefix)?;
        
        let since = since_seq.unwrap_or(0);
        let mut events = Vec::new();
        
        for (_, value) in results {
            let event: WorkflowEvent = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            
            if event.seq > since {
                events.push(event);
                if events.len() >= limit {
                    break;
                }
            }
        }
        
        Ok(events)
    }
    
    fn delete_run(&self, run_id: &RunId) -> Result<bool> {
        // Check if run exists
        if self.get_run(run_id)?.is_none() {
            return Ok(false);
        }
        
        // Delete all checkpoints
        let cp_prefix = Self::all_checkpoints_prefix(run_id);
        for (key, _) in self.conn.scan(&cp_prefix)? {
            self.conn.delete(&key)?;
        }
        
        // Delete all events
        let ev_prefix = Self::events_prefix(run_id);
        for (key, _) in self.conn.scan(&ev_prefix)? {
            self.conn.delete(&key)?;
        }
        
        // Delete run metadata
        self.conn.delete(&Self::run_meta_key(run_id))?;
        
        Ok(true)
    }
}

// ============================================================================
// Base64 Serde Helper
// ============================================================================

mod hex_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Tests would use mock ConnectionTrait
}
