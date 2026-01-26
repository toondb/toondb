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

//! Semi-GraphDB Overlay for Agent Memory.
//!
//! Provides a lightweight graph layer on top of SochDB's KV storage for modeling
//! agent memory relationships:
//!
//! - Entity-to-entity relationships (user <-> conversation <-> message)
//! - Causal chains (action1 -> action2 -> action3)
//! - Reference graphs (document <- citation <- quote)
//!
//! # Storage Model
//!
//! - Nodes: `_graph/{namespace}/nodes/{node_id}` -> `{type, properties}`
//! - Edges: `_graph/{namespace}/edges/{from_id}/{edge_type}/{to_id}` -> `{properties}`
//! - Index: `_graph/{namespace}/index/{edge_type}/{to_id}` -> `[from_ids]` (reverse lookup)
//!
//! # Example
//!
//! ```rust,no_run
//! use sochdb::graph::{GraphOverlay, GraphNode, GraphEdge};
//! use sochdb::Connection;
//! use std::collections::HashMap;
//!
//! let conn = Connection::open("./agent_memory")?;
//! let graph = GraphOverlay::new(conn, "agent_001");
//!
//! // Create nodes
//! let mut props = HashMap::new();
//! props.insert("name".to_string(), serde_json::json!("Alice"));
//! graph.add_node("user_1", "User", Some(props))?;
//!
//! // Create edges
//! graph.add_edge("user_1", "STARTED", "conv_1", None)?;
//!
//! // Traverse graph
//! let path = graph.shortest_path("user_1", "msg_1", 10, None)?;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

use crate::ConnectionTrait;
use crate::error::{ClientError, Result};

/// Graph traversal order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalOrder {
    /// Breadth-first search
    BFS,
    /// Depth-first search
    DFS,
}

/// Edge direction for neighbor queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

/// A node in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

/// An edge in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from_id: String,
    pub edge_type: String,
    pub to_id: String,
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

/// A neighboring node with its connecting edge.
#[derive(Debug, Clone)]
pub struct Neighbor {
    pub node_id: String,
    pub edge: GraphEdge,
}

/// A subgraph containing nodes and edges.
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub nodes: HashMap<String, GraphNode>,
    pub edges: Vec<GraphEdge>,
}

const PREFIX: &str = "_graph";

/// Lightweight graph overlay on SochDB.
///
/// Provides graph operations for agent memory without a full graph database.
/// Uses the underlying KV store for persistence with O(1) node/edge operations.
pub struct GraphOverlay<C: ConnectionTrait> {
    conn: C,
    namespace: String,
    prefix: String,
}

impl<C: ConnectionTrait> GraphOverlay<C> {
    /// Create a new graph overlay.
    ///
    /// # Arguments
    ///
    /// * `conn` - SochDB connection
    /// * `namespace` - Namespace for graph isolation (e.g., agent_id)
    pub fn new(conn: C, namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        let prefix = format!("{}/{}", PREFIX, namespace);
        Self {
            conn,
            namespace,
            prefix,
        }
    }

    // Key helpers
    fn node_key(&self, node_id: &str) -> Vec<u8> {
        format!("{}/nodes/{}", self.prefix, node_id).into_bytes()
    }

    fn edge_key(&self, from_id: &str, edge_type: &str, to_id: &str) -> Vec<u8> {
        format!("{}/edges/{}/{}/{}", self.prefix, from_id, edge_type, to_id).into_bytes()
    }

    fn edge_prefix(&self, from_id: &str, edge_type: Option<&str>) -> Vec<u8> {
        match edge_type {
            Some(et) => format!("{}/edges/{}/{}/", self.prefix, from_id, et).into_bytes(),
            None => format!("{}/edges/{}/", self.prefix, from_id).into_bytes(),
        }
    }

    fn reverse_index_key(&self, edge_type: &str, to_id: &str, from_id: &str) -> Vec<u8> {
        format!("{}/index/{}/{}/{}", self.prefix, edge_type, to_id, from_id).into_bytes()
    }

    fn reverse_index_prefix(&self, edge_type: &str, to_id: &str) -> Vec<u8> {
        format!("{}/index/{}/{}/", self.prefix, edge_type, to_id).into_bytes()
    }

    // =========================================================================
    // Node Operations
    // =========================================================================

    /// Add a node to the graph.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Unique node identifier
    /// * `node_type` - Node type label (e.g., "User", "Message", "Tool")
    /// * `properties` - Optional node properties
    ///
    /// # Returns
    ///
    /// The created GraphNode
    pub fn add_node(
        &self,
        node_id: &str,
        node_type: &str,
        properties: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<GraphNode> {
        let node = GraphNode {
            id: node_id.to_string(),
            node_type: node_type.to_string(),
            properties: properties.unwrap_or_default(),
        };

        let data = serde_json::to_vec(&node)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;

        self.conn.put(&self.node_key(node_id), &data)?;
        Ok(node)
    }

    /// Get a node by ID.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node identifier
    ///
    /// # Returns
    ///
    /// GraphNode if found, None otherwise
    pub fn get_node(&self, node_id: &str) -> Result<Option<GraphNode>> {
        match self.conn.get(&self.node_key(node_id))? {
            Some(data) => {
                let node: GraphNode = serde_json::from_slice(&data)
                    .map_err(|e| ClientError::Serialization(e.to_string()))?;
                Ok(Some(node))
            }
            None => Ok(None),
        }
    }

    /// Update a node's properties or type.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node identifier
    /// * `properties` - Properties to merge (None to skip)
    /// * `node_type` - New type (None to keep existing)
    ///
    /// # Returns
    ///
    /// Updated GraphNode if found, None otherwise
    pub fn update_node(
        &self,
        node_id: &str,
        properties: Option<HashMap<String, serde_json::Value>>,
        node_type: Option<&str>,
    ) -> Result<Option<GraphNode>> {
        let mut node = match self.get_node(node_id)? {
            Some(n) => n,
            None => return Ok(None),
        };

        if let Some(props) = properties {
            for (k, v) in props {
                node.properties.insert(k, v);
            }
        }
        if let Some(nt) = node_type {
            node.node_type = nt.to_string();
        }

        let data = serde_json::to_vec(&node)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;

        self.conn.put(&self.node_key(node_id), &data)?;
        Ok(Some(node))
    }

    /// Delete a node from the graph.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node identifier
    /// * `cascade` - If true, also delete all connected edges
    ///
    /// # Returns
    ///
    /// true if deleted, false if not found
    pub fn delete_node(&self, node_id: &str, cascade: bool) -> Result<bool> {
        if self.get_node(node_id)?.is_none() {
            return Ok(false);
        }

        if cascade {
            // Delete outgoing edges
            for edge in self.get_edges(node_id, None)? {
                self.delete_edge(node_id, &edge.edge_type, &edge.to_id)?;
            }

            // Delete incoming edges
            for edge in self.get_incoming_edges(node_id, None)? {
                self.delete_edge(&edge.from_id, &edge.edge_type, node_id)?;
            }
        }

        self.conn.delete(&self.node_key(node_id))?;
        Ok(true)
    }

    /// Check if a node exists.
    pub fn node_exists(&self, node_id: &str) -> Result<bool> {
        Ok(self.conn.get(&self.node_key(node_id))?.is_some())
    }

    // =========================================================================
    // Edge Operations
    // =========================================================================

    /// Add an edge between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from_id` - Source node ID
    /// * `edge_type` - Edge type label (e.g., "SENT", "REFERENCES", "CAUSED")
    /// * `to_id` - Target node ID
    /// * `properties` - Optional edge properties
    ///
    /// # Returns
    ///
    /// The created GraphEdge
    pub fn add_edge(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
        properties: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<GraphEdge> {
        let edge = GraphEdge {
            from_id: from_id.to_string(),
            edge_type: edge_type.to_string(),
            to_id: to_id.to_string(),
            properties: properties.unwrap_or_default(),
        };

        let data = serde_json::to_vec(&edge)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;

        // Store edge
        self.conn.put(&self.edge_key(from_id, edge_type, to_id), &data)?;

        // Store reverse index
        self.conn.put(
            &self.reverse_index_key(edge_type, to_id, from_id),
            from_id.as_bytes(),
        )?;

        Ok(edge)
    }

    /// Get a specific edge.
    pub fn get_edge(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
    ) -> Result<Option<GraphEdge>> {
        match self.conn.get(&self.edge_key(from_id, edge_type, to_id))? {
            Some(data) => {
                let edge: GraphEdge = serde_json::from_slice(&data)
                    .map_err(|e| ClientError::Serialization(e.to_string()))?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Get all outgoing edges from a node.
    pub fn get_edges(
        &self,
        from_id: &str,
        edge_type: Option<&str>,
    ) -> Result<Vec<GraphEdge>> {
        let prefix = self.edge_prefix(from_id, edge_type);
        let results = self.conn.scan(&prefix)?;

        let mut edges = Vec::new();
        for (_, value) in results {
            if let Ok(edge) = serde_json::from_slice::<GraphEdge>(&value) {
                edges.push(edge);
            }
        }

        Ok(edges)
    }

    /// Get all incoming edges to a node.
    pub fn get_incoming_edges(
        &self,
        to_id: &str,
        edge_type: Option<&str>,
    ) -> Result<Vec<GraphEdge>> {
        let mut edges = Vec::new();

        if let Some(et) = edge_type {
            // Query specific edge type
            let prefix = self.reverse_index_prefix(et, to_id);
            let results = self.conn.scan(&prefix)?;

            for (_, value) in results {
                let from_id = String::from_utf8_lossy(&value).to_string();
                if let Some(edge) = self.get_edge(&from_id, et, to_id)? {
                    edges.push(edge);
                }
            }
        } else {
            // Query all edge types - scan all index entries
            let index_prefix = format!("{}/index/", self.prefix).into_bytes();
            let results = self.conn.scan(&index_prefix)?;

            for (key, value) in results {
                let key_str = String::from_utf8_lossy(&key);
                let parts: Vec<&str> = key_str.split('/').collect();
                if parts.len() >= 6 && parts[4] == to_id {
                    let from_id = String::from_utf8_lossy(&value).to_string();
                    let et = parts[3];
                    if let Some(edge) = self.get_edge(&from_id, et, to_id)? {
                        edges.push(edge);
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Delete an edge.
    pub fn delete_edge(
        &self,
        from_id: &str,
        edge_type: &str,
        to_id: &str,
    ) -> Result<bool> {
        if self.get_edge(from_id, edge_type, to_id)?.is_none() {
            return Ok(false);
        }

        // Delete edge
        self.conn.delete(&self.edge_key(from_id, edge_type, to_id))?;

        // Delete reverse index
        self.conn.delete(&self.reverse_index_key(edge_type, to_id, from_id))?;

        Ok(true)
    }

    // =========================================================================
    // Traversal Operations
    // =========================================================================

    /// Breadth-first search from a starting node.
    pub fn bfs(
        &self,
        start_id: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        node_types: Option<&[&str]>,
    ) -> Result<Vec<String>> {
        self.traverse(start_id, max_depth, edge_types, node_types, TraversalOrder::BFS)
    }

    /// Depth-first search from a starting node.
    pub fn dfs(
        &self,
        start_id: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        node_types: Option<&[&str]>,
    ) -> Result<Vec<String>> {
        self.traverse(start_id, max_depth, edge_types, node_types, TraversalOrder::DFS)
    }

    fn traverse(
        &self,
        start_id: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        node_types: Option<&[&str]>,
        order: TraversalOrder,
    ) -> Result<Vec<String>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();

        let edge_type_set: HashSet<&str> = edge_types.map(|e| e.iter().copied().collect()).unwrap_or_default();
        let node_type_set: HashSet<&str> = node_types.map(|n| n.iter().copied().collect()).unwrap_or_default();

        let mut frontier: VecDeque<(String, usize)> = VecDeque::new();
        frontier.push_back((start_id.to_string(), 0));

        while let Some((node_id, depth)) = match order {
            TraversalOrder::BFS => frontier.pop_front(),
            TraversalOrder::DFS => frontier.pop_back(),
        } {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());

            // Check node type filter
            if node_types.is_some() && !node_type_set.is_empty() {
                if let Some(node) = self.get_node(&node_id)? {
                    if !node_type_set.contains(node.node_type.as_str()) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            result.push(node_id.clone());

            if depth >= max_depth {
                continue;
            }

            // Get outgoing edges
            for edge in self.get_edges(&node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }
                if !visited.contains(&edge.to_id) {
                    frontier.push_back((edge.to_id, depth + 1));
                }
            }
        }

        Ok(result)
    }

    /// Find shortest path between two nodes using BFS.
    pub fn shortest_path(
        &self,
        from_id: &str,
        to_id: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> Result<Option<Vec<String>>> {
        if from_id == to_id {
            return Ok(Some(vec![from_id.to_string()]));
        }

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(from_id.to_string());
        let mut parent: HashMap<String, String> = HashMap::new();

        let edge_type_set: HashSet<&str> = edge_types.map(|e| e.iter().copied().collect()).unwrap_or_default();

        let mut frontier: VecDeque<(String, usize)> = VecDeque::new();
        frontier.push_back((from_id.to_string(), 0));

        while let Some((node_id, depth)) = frontier.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for edge in self.get_edges(&node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }

                let next_id = edge.to_id;
                if visited.contains(&next_id) {
                    continue;
                }

                visited.insert(next_id.clone());
                parent.insert(next_id.clone(), node_id.clone());

                if next_id == to_id {
                    // Reconstruct path
                    let mut path = vec![to_id.to_string()];
                    let mut curr = to_id.to_string();
                    while let Some(p) = parent.get(&curr) {
                        path.push(p.clone());
                        curr = p.clone();
                    }
                    path.reverse();
                    return Ok(Some(path));
                }

                frontier.push_back((next_id, depth + 1));
            }
        }

        Ok(None) // No path found
    }

    // =========================================================================
    // Query Operations
    // =========================================================================

    /// Get neighboring nodes with their connecting edges.
    pub fn get_neighbors(
        &self,
        node_id: &str,
        edge_types: Option<&[&str]>,
        direction: EdgeDirection,
    ) -> Result<Vec<Neighbor>> {
        let mut neighbors = Vec::new();
        let edge_type_set: HashSet<&str> = edge_types.map(|e| e.iter().copied().collect()).unwrap_or_default();

        if matches!(direction, EdgeDirection::Outgoing | EdgeDirection::Both) {
            for edge in self.get_edges(node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }
                neighbors.push(Neighbor {
                    node_id: edge.to_id.clone(),
                    edge,
                });
            }
        }

        if matches!(direction, EdgeDirection::Incoming | EdgeDirection::Both) {
            for edge in self.get_incoming_edges(node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }
                neighbors.push(Neighbor {
                    node_id: edge.from_id.clone(),
                    edge,
                });
            }
        }

        Ok(neighbors)
    }

    /// Get all nodes of a specific type.
    ///
    /// Note: This scans all nodes, use sparingly for large graphs.
    pub fn get_nodes_by_type(
        &self,
        node_type: &str,
        limit: usize,
    ) -> Result<Vec<GraphNode>> {
        let prefix = format!("{}/nodes/", self.prefix).into_bytes();
        let results = self.conn.scan(&prefix)?;

        let mut nodes = Vec::new();
        for (_, value) in results {
            if let Ok(node) = serde_json::from_slice::<GraphNode>(&value) {
                if node.node_type == node_type {
                    nodes.push(node);
                    if limit > 0 && nodes.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(nodes)
    }

    /// Get a subgraph starting from a node.
    pub fn get_subgraph(
        &self,
        start_id: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> Result<Subgraph> {
        let node_ids = self.bfs(start_id, max_depth, edge_types, None)?;

        let mut nodes = HashMap::new();
        let mut edges = Vec::new();

        // First collect all nodes
        for node_id in &node_ids {
            if let Some(node) = self.get_node(node_id)? {
                nodes.insert(node_id.clone(), node);
            }
        }

        // Then collect edges where both endpoints are in the subgraph
        for node_id in &node_ids {
            for edge in self.get_edges(node_id, None)? {
                if nodes.contains_key(&edge.to_id) {
                    edges.push(edge);
                }
            }
        }

        Ok(Subgraph { nodes, edges })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would go here
}
