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

//! Temporal Graph Extensions (Task 7)
//!
//! This module adds temporal dimensions to the graph overlay for:
//! - Time-bounded edges ("door was open from t1 to t2")
//! - Time-travel queries ("what did the agent believe at t-2m?")
//! - Causal reasoning over evolving agent beliefs
//!
//! ## Storage Model
//!
//! Edges with validity intervals:
//! ```text
//! _graph/{namespace}/temporal/{from_id}/{edge_type}/{to_id}/{valid_from}
//!     -> TemporalEdge { valid_until, properties }
//! ```
//!
//! Time index (for efficient point-in-time queries):
//! ```text
//! _graph/{namespace}/time_index/{bucket}/{valid_from}_{edge_key}
//!     -> edge reference
//! ```
//!
//! ## Complexity
//!
//! - Point-in-time query: O(log E + k) where E is edges, k is matches
//! - Range query: O(log E + k)
//! - Without time index: O(E) scan
//!
//! The time index uses bucketed partitions (hour/day granularity) to
//! balance index size vs query efficiency.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ClientError, Result};
use crate::ConnectionTrait;

// ============================================================================
// Temporal Types
// ============================================================================

/// Timestamp in milliseconds since Unix epoch
pub type Timestamp = u64;

/// A time interval [start, end)
/// 
/// - If `end` is None, the interval extends to infinity (still valid)
/// - Both bounds are inclusive on start, exclusive on end
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeInterval {
    /// Start of validity (inclusive)
    pub start: Timestamp,
    /// End of validity (exclusive), None = still valid
    pub end: Option<Timestamp>,
}

impl TimeInterval {
    /// Create an interval from start to infinity (open-ended)
    pub fn from(start: Timestamp) -> Self {
        Self { start, end: None }
    }
    
    /// Create a closed interval [start, end)
    pub fn between(start: Timestamp, end: Timestamp) -> Self {
        Self { start, end: Some(end) }
    }
    
    /// Create an interval starting now
    pub fn now() -> Self {
        Self::from(Self::current_time())
    }
    
    /// Check if a timestamp falls within this interval
    pub fn contains(&self, t: Timestamp) -> bool {
        t >= self.start && self.end.map_or(true, |end| t < end)
    }
    
    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &TimeInterval) -> bool {
        let self_end = self.end.unwrap_or(Timestamp::MAX);
        let other_end = other.end.unwrap_or(Timestamp::MAX);
        
        self.start < other_end && other.start < self_end
    }
    
    /// Check if this interval is still active (no end)
    pub fn is_active(&self) -> bool {
        self.end.is_none()
    }
    
    /// Get current timestamp
    pub fn current_time() -> Timestamp {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as Timestamp
    }
    
    /// Close this interval at the given time
    pub fn close_at(&mut self, t: Timestamp) {
        self.end = Some(t);
    }
    
    /// Duration in milliseconds (None if open-ended)
    pub fn duration_ms(&self) -> Option<u64> {
        self.end.map(|e| e.saturating_sub(self.start))
    }
}

impl Default for TimeInterval {
    fn default() -> Self {
        Self::now()
    }
}

/// A temporal edge with validity interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    /// Source node ID
    pub from_id: String,
    /// Edge type
    pub edge_type: String,
    /// Target node ID
    pub to_id: String,
    /// Validity interval
    pub validity: TimeInterval,
    /// Edge properties
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    /// Version for conflict resolution
    #[serde(default)]
    pub version: u64,
}

impl TemporalEdge {
    /// Check if this edge is valid at a given time
    pub fn is_valid_at(&self, t: Timestamp) -> bool {
        self.validity.contains(t)
    }
    
    /// Check if this edge is currently valid
    pub fn is_active(&self) -> bool {
        self.validity.is_active() || self.validity.contains(TimeInterval::current_time())
    }
}

/// Query parameters for temporal graph queries
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    /// Point in time to query (None = now)
    pub at_time: Option<Timestamp>,
    /// Time window for range queries
    pub time_window: Option<TimeInterval>,
    /// Include invalidated edges
    pub include_history: bool,
}

impl Default for TemporalQuery {
    fn default() -> Self {
        Self {
            at_time: None,
            time_window: None,
            include_history: false,
        }
    }
}

impl TemporalQuery {
    /// Query at a specific point in time
    pub fn at(t: Timestamp) -> Self {
        Self {
            at_time: Some(t),
            time_window: None,
            include_history: false,
        }
    }
    
    /// Query within a time window
    pub fn window(start: Timestamp, end: Timestamp) -> Self {
        Self {
            at_time: None,
            time_window: Some(TimeInterval::between(start, end)),
            include_history: true,
        }
    }
    
    /// Query current state (default)
    pub fn now() -> Self {
        Self::default()
    }
    
    /// Include historical (invalidated) edges
    pub fn with_history(mut self) -> Self {
        self.include_history = true;
        self
    }
}

// ============================================================================
// Temporal Graph Overlay
// ============================================================================

/// Time bucket granularity for indexing
#[derive(Debug, Clone, Copy)]
pub enum TimeBucket {
    /// Hourly buckets
    Hour,
    /// Daily buckets  
    Day,
    /// Weekly buckets
    Week,
}

impl TimeBucket {
    /// Get bucket key for a timestamp
    pub fn bucket_key(&self, t: Timestamp) -> u64 {
        match self {
            Self::Hour => t / (3600 * 1000),
            Self::Day => t / (86400 * 1000),
            Self::Week => t / (604800 * 1000),
        }
    }
}

/// Temporal graph overlay with time-indexed edges
pub struct TemporalGraphOverlay<C: ConnectionTrait> {
    conn: C,
    namespace: String,
    prefix: String,
    bucket_granularity: TimeBucket,
}

impl<C: ConnectionTrait> TemporalGraphOverlay<C> {
    /// Create a new temporal graph overlay
    pub fn new(conn: C, namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        let prefix = format!("_graph/{}", namespace);
        Self {
            conn,
            namespace,
            prefix,
            bucket_granularity: TimeBucket::Hour,
        }
    }
    
    /// Set the time bucket granularity
    pub fn with_bucket_granularity(mut self, granularity: TimeBucket) -> Self {
        self.bucket_granularity = granularity;
        self
    }
    
    // Key helpers
    fn temporal_edge_key(&self, from_id: &str, edge_type: &str, to_id: &str, valid_from: Timestamp) -> Vec<u8> {
        format!(
            "{}/temporal/{}/{}/{}/{:016x}",
            self.prefix, from_id, edge_type, to_id, valid_from
        ).into_bytes()
    }
    
    fn temporal_edge_prefix(&self, from_id: &str, edge_type: Option<&str>, to_id: Option<&str>) -> Vec<u8> {
        match (edge_type, to_id) {
            (Some(et), Some(tid)) => {
                format!("{}/temporal/{}/{}/{}/", self.prefix, from_id, et, tid).into_bytes()
            }
            (Some(et), None) => {
                format!("{}/temporal/{}/{}/", self.prefix, from_id, et).into_bytes()
            }
            (None, _) => {
                format!("{}/temporal/{}/", self.prefix, from_id).into_bytes()
            }
        }
    }
    
    fn time_index_key(&self, bucket: u64, valid_from: Timestamp, edge_key: &str) -> Vec<u8> {
        format!(
            "{}/time_index/{:016x}/{:016x}_{}",
            self.prefix, bucket, valid_from, edge_key
        ).into_bytes()
    }
    
    fn time_index_prefix(&self, bucket: u64) -> Vec<u8> {
        format!("{}/time_index/{:016x}/", self.prefix, bucket).into_bytes()
    }
    
    // ========================================================================
    // Edge Operations
    // ========================================================================
    
    /// Add a temporal edge (valid from now)
    pub fn add_edge(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
        properties: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<TemporalEdge> {
        self.add_edge_at(from_id, edge_type, to_id, TimeInterval::now(), properties)
    }
    
    /// Add a temporal edge with explicit validity
    pub fn add_edge_at(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
        validity: TimeInterval,
        properties: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<TemporalEdge> {
        let edge = TemporalEdge {
            from_id: from_id.to_string(),
            edge_type: edge_type.to_string(),
            to_id: to_id.to_string(),
            validity,
            properties: properties.unwrap_or_default(),
            version: validity.start, // Use start time as version
        };
        
        // Store edge
        let key = self.temporal_edge_key(from_id, edge_type, to_id, validity.start);
        let value = serde_json::to_vec(&edge)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.conn.put(&key, &value)?;
        
        // Update time index
        let bucket = self.bucket_granularity.bucket_key(validity.start);
        let edge_key_str = format!("{}:{}:{}", from_id, edge_type, to_id);
        let index_key = self.time_index_key(bucket, validity.start, &edge_key_str);
        self.conn.put(&index_key, &key)?;
        
        Ok(edge)
    }
    
    /// Invalidate an edge at the current time
    pub fn invalidate_edge(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
    ) -> Result<bool> {
        self.invalidate_edge_at(from_id, edge_type, to_id, TimeInterval::current_time())
    }
    
    /// Invalidate an edge at a specific time
    pub fn invalidate_edge_at(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
        at_time: Timestamp,
    ) -> Result<bool> {
        // Find the currently active edge
        let prefix = self.temporal_edge_prefix(from_id, Some(edge_type), Some(to_id));
        let results = self.conn.scan(&prefix)?;
        
        for (key, value) in results {
            let mut edge: TemporalEdge = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            
            if edge.is_active() {
                edge.validity.close_at(at_time);
                let new_value = serde_json::to_vec(&edge)
                    .map_err(|e| ClientError::Serialization(e.to_string()))?;
                self.conn.put(&key, &new_value)?;
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Get edges valid at a specific point in time
    pub fn get_edges_at(
        &self,
        from_id: &str,
        edge_type: Option<&str>,
        at_time: Timestamp,
    ) -> Result<Vec<TemporalEdge>> {
        let prefix = self.temporal_edge_prefix(from_id, edge_type, None);
        let results = self.conn.scan(&prefix)?;
        
        let mut edges = Vec::new();
        for (_, value) in results {
            let edge: TemporalEdge = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            
            if edge.is_valid_at(at_time) {
                edges.push(edge);
            }
        }
        
        Ok(edges)
    }
    
    /// Get edges valid within a time window
    pub fn get_edges_in_window(
        &self,
        from_id: &str,
        edge_type: Option<&str>,
        window: TimeInterval,
    ) -> Result<Vec<TemporalEdge>> {
        let prefix = self.temporal_edge_prefix(from_id, edge_type, None);
        let results = self.conn.scan(&prefix)?;
        
        let mut edges = Vec::new();
        for (_, value) in results {
            let edge: TemporalEdge = serde_json::from_slice(&value)
                .map_err(|e| ClientError::Serialization(e.to_string()))?;
            
            if edge.validity.overlaps(&window) {
                edges.push(edge);
            }
        }
        
        Ok(edges)
    }
    
    /// Get neighbors at a specific point in time
    pub fn neighbors_at(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        at_time: Timestamp,
    ) -> Result<Vec<TemporalEdge>> {
        self.get_edges_at(node_id, edge_type, at_time)
    }
    
    /// Get subgraph at a specific point in time
    pub fn subgraph_at(
        &self,
        start_id: &str,
        max_depth: usize,
        at_time: Timestamp,
    ) -> Result<TemporalSubgraph> {
        let mut visited = std::collections::HashSet::new();
        let mut edges = Vec::new();
        let mut frontier = vec![(start_id.to_string(), 0)];
        
        while let Some((node_id, depth)) = frontier.pop() {
            if depth >= max_depth || visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());
            
            let node_edges = self.get_edges_at(&node_id, None, at_time)?;
            for edge in node_edges {
                if !visited.contains(&edge.to_id) {
                    frontier.push((edge.to_id.clone(), depth + 1));
                }
                edges.push(edge);
            }
        }
        
        Ok(TemporalSubgraph {
            node_ids: visited.into_iter().collect(),
            edges,
            at_time,
        })
    }
    
    /// Get the history of an edge (all versions)
    pub fn edge_history(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
    ) -> Result<Vec<TemporalEdge>> {
        let prefix = self.temporal_edge_prefix(from_id, Some(edge_type), Some(to_id));
        let results = self.conn.scan(&prefix)?;
        
        let mut edges: Vec<TemporalEdge> = results
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_slice(&value).ok())
            .collect();
        
        // Sort by validity start (oldest first)
        edges.sort_by_key(|e| e.validity.start);
        
        Ok(edges)
    }
}

/// A subgraph at a specific point in time
#[derive(Debug, Clone)]
pub struct TemporalSubgraph {
    /// Node IDs in the subgraph
    pub node_ids: Vec<String>,
    /// Edges valid at the query time
    pub edges: Vec<TemporalEdge>,
    /// The point in time this subgraph represents
    pub at_time: Timestamp,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_interval_contains() {
        let now = TimeInterval::current_time();
        let interval = TimeInterval::between(now - 1000, now + 1000);
        
        assert!(interval.contains(now));
        assert!(!interval.contains(now - 2000));
        assert!(!interval.contains(now + 2000));
    }
    
    #[test]
    fn test_time_interval_overlaps() {
        let a = TimeInterval::between(100, 200);
        let b = TimeInterval::between(150, 250);
        let c = TimeInterval::between(200, 300);
        let d = TimeInterval::between(50, 100);
        
        assert!(a.overlaps(&b)); // Overlap at 150-200
        assert!(!a.overlaps(&c)); // c starts where a ends (exclusive)
        assert!(!a.overlaps(&d)); // d ends where a starts
    }
    
    #[test]
    fn test_open_ended_interval() {
        let interval = TimeInterval::from(100);
        
        assert!(interval.is_active());
        assert!(interval.contains(100));
        assert!(interval.contains(1000000));
        assert!(!interval.contains(50));
    }
}
