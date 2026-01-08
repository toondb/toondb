// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Graph Service gRPC Implementation
//!
//! Provides graph overlay operations for agent memory via gRPC.

use crate::proto::{
    graph_service_server::{GraphService, GraphServiceServer},
    AddEdgeRequest, AddEdgeResponse, AddNodeRequest, AddNodeResponse,
    AddTemporalEdgeRequest, AddTemporalEdgeResponse,
    DeleteEdgeRequest, DeleteEdgeResponse, DeleteNodeRequest, DeleteNodeResponse,
    GetEdgesRequest, GetEdgesResponse, GetNeighborsRequest, GetNeighborsResponse,
    GetNodeRequest, GetNodeResponse, GraphEdge, GraphNode, ShortestPathRequest,
    ShortestPathResponse, TraverseRequest, TraverseResponse,
    QueryTemporalGraphRequest, QueryTemporalGraphResponse, TemporalEdge,
};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// In-memory graph storage per namespace
struct NamespaceGraph {
    nodes: DashMap<String, GraphNode>,
    edges: DashMap<String, Vec<GraphEdge>>, // from_id -> edges
    reverse_edges: DashMap<String, Vec<GraphEdge>>, // to_id -> edges
    temporal_edges: DashMap<String, Vec<TemporalEdge>>, // from_id -> temporal edges
}

impl NamespaceGraph {
    fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            edges: DashMap::new(),
            reverse_edges: DashMap::new(),
            temporal_edges: DashMap::new(),
        }
    }
}

/// Graph gRPC Server
pub struct GraphServer {
    namespaces: DashMap<String, Arc<NamespaceGraph>>,
}

impl GraphServer {
    pub fn new() -> Self {
        Self {
            namespaces: DashMap::new(),
        }
    }

    pub fn into_service(self) -> GraphServiceServer<Self> {
        GraphServiceServer::new(self)
    }

    fn get_or_create_namespace(&self, namespace: &str) -> Arc<NamespaceGraph> {
        self.namespaces
            .entry(namespace.to_string())
            .or_insert_with(|| Arc::new(NamespaceGraph::new()))
            .clone()
    }
}

impl Default for GraphServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl GraphService for GraphServer {
    async fn add_node(
        &self,
        request: Request<AddNodeRequest>,
    ) -> Result<Response<AddNodeResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        if let Some(node) = req.node {
            ns.nodes.insert(node.id.clone(), node);
            Ok(Response::new(AddNodeResponse {
                success: true,
                error: String::new(),
            }))
        } else {
            Ok(Response::new(AddNodeResponse {
                success: false,
                error: "Node is required".to_string(),
            }))
        }
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> Result<Response<GetNodeResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        match ns.nodes.get(&req.node_id) {
            Some(node) => Ok(Response::new(GetNodeResponse {
                node: Some(node.clone()),
                error: String::new(),
            })),
            None => Ok(Response::new(GetNodeResponse {
                node: None,
                error: format!("Node '{}' not found", req.node_id),
            })),
        }
    }

    async fn delete_node(
        &self,
        request: Request<DeleteNodeRequest>,
    ) -> Result<Response<DeleteNodeResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        match ns.nodes.remove(&req.node_id) {
            Some(_) => {
                // Also remove edges
                ns.edges.remove(&req.node_id);
                ns.reverse_edges.remove(&req.node_id);
                Ok(Response::new(DeleteNodeResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(DeleteNodeResponse {
                success: false,
                error: format!("Node '{}' not found", req.node_id),
            })),
        }
    }

    async fn add_edge(
        &self,
        request: Request<AddEdgeRequest>,
    ) -> Result<Response<AddEdgeResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        if let Some(edge) = req.edge {
            let from_id = edge.from_id.clone();
            let to_id = edge.to_id.clone();

            ns.edges
                .entry(from_id)
                .or_default()
                .push(edge.clone());
            ns.reverse_edges
                .entry(to_id)
                .or_default()
                .push(edge);

            Ok(Response::new(AddEdgeResponse {
                success: true,
                error: String::new(),
            }))
        } else {
            Ok(Response::new(AddEdgeResponse {
                success: false,
                error: "Edge is required".to_string(),
            }))
        }
    }

    async fn get_edges(
        &self,
        request: Request<GetEdgesRequest>,
    ) -> Result<Response<GetEdgesResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let mut edges = Vec::new();

        // Get outgoing edges
        if let Some(out_edges) = ns.edges.get(&req.node_id) {
            for edge in out_edges.iter() {
                if req.edge_type.is_empty() || edge.edge_type == req.edge_type {
                    edges.push(edge.clone());
                }
            }
        }

        // Get incoming edges if direction is incoming or both
        if req.direction == 1 || req.direction == 2 {
            if let Some(in_edges) = ns.reverse_edges.get(&req.node_id) {
                for edge in in_edges.iter() {
                    if req.edge_type.is_empty() || edge.edge_type == req.edge_type {
                        edges.push(edge.clone());
                    }
                }
            }
        }

        Ok(Response::new(GetEdgesResponse {
            edges,
            error: String::new(),
        }))
    }

    async fn delete_edge(
        &self,
        request: Request<DeleteEdgeRequest>,
    ) -> Result<Response<DeleteEdgeResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let mut found = false;

        // Remove from forward edges
        if let Some(mut edges) = ns.edges.get_mut(&req.from_id) {
            edges.retain(|e| {
                let matches = e.edge_type == req.edge_type && e.to_id == req.to_id;
                if matches {
                    found = true;
                }
                !matches
            });
        }

        // Remove from reverse edges
        if let Some(mut edges) = ns.reverse_edges.get_mut(&req.to_id) {
            edges.retain(|e| {
                !(e.edge_type == req.edge_type && e.from_id == req.from_id)
            });
        }

        Ok(Response::new(DeleteEdgeResponse {
            success: found,
            error: if found {
                String::new()
            } else {
                "Edge not found".to_string()
            },
        }))
    }

    async fn traverse(
        &self,
        request: Request<TraverseRequest>,
    ) -> Result<Response<TraverseResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let mut visited_nodes = Vec::new();
        let mut visited_edges = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        let is_bfs = req.order == 0; // BFS = 0, DFS = 1
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((req.start_node_id.clone(), 0));

        while let Some((node_id, depth)) = if is_bfs {
            queue.pop_front()
        } else {
            queue.pop_back()
        } {
            if seen.contains(&node_id) {
                continue;
            }
            if depth > req.max_depth {
                continue;
            }

            seen.insert(node_id.clone());

            if let Some(node) = ns.nodes.get(&node_id) {
                visited_nodes.push(node.clone());
            }

            if let Some(edges) = ns.edges.get(&node_id) {
                for edge in edges.iter() {
                    let type_matches = req.edge_types.is_empty()
                        || req.edge_types.contains(&edge.edge_type);
                    if type_matches && !seen.contains(&edge.to_id) {
                        visited_edges.push(edge.clone());
                        queue.push_back((edge.to_id.clone(), depth + 1));
                    }
                }
            }
        }

        Ok(Response::new(TraverseResponse {
            nodes: visited_nodes,
            edges: visited_edges,
            error: String::new(),
        }))
    }

    async fn shortest_path(
        &self,
        request: Request<ShortestPathRequest>,
    ) -> Result<Response<ShortestPathResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        // BFS for shortest path
        let mut queue: VecDeque<(String, Vec<String>, Vec<GraphEdge>)> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();

        queue.push_back((req.from_id.clone(), vec![req.from_id.clone()], vec![]));

        while let Some((current, path, edges)) = queue.pop_front() {
            if current == req.to_id {
                return Ok(Response::new(ShortestPathResponse {
                    path,
                    edges,
                    error: String::new(),
                }));
            }

            if visited.contains(&current) {
                continue;
            }
            if path.len() as u32 > req.max_depth {
                continue;
            }

            visited.insert(current.clone());

            if let Some(node_edges) = ns.edges.get(&current) {
                for edge in node_edges.iter() {
                    let type_matches = req.edge_types.is_empty()
                        || req.edge_types.contains(&edge.edge_type);
                    if type_matches && !visited.contains(&edge.to_id) {
                        let mut new_path = path.clone();
                        new_path.push(edge.to_id.clone());
                        let mut new_edges = edges.clone();
                        new_edges.push(edge.clone());
                        queue.push_back((edge.to_id.clone(), new_path, new_edges));
                    }
                }
            }
        }

        Ok(Response::new(ShortestPathResponse {
            path: vec![],
            edges: vec![],
            error: format!("No path found from '{}' to '{}'", req.from_id, req.to_id),
        }))
    }

    async fn get_neighbors(
        &self,
        request: Request<GetNeighborsRequest>,
    ) -> Result<Response<GetNeighborsResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let mut neighbor_nodes = Vec::new();
        let mut neighbor_edges = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // Get outgoing neighbors
        if req.direction == 0 || req.direction == 2 {
            if let Some(edges) = ns.edges.get(&req.node_id) {
                for edge in edges.iter() {
                    let type_matches = req.edge_types.is_empty()
                        || req.edge_types.contains(&edge.edge_type);
                    if type_matches && !seen.contains(&edge.to_id) {
                        seen.insert(edge.to_id.clone());
                        neighbor_edges.push(edge.clone());
                        if let Some(node) = ns.nodes.get(&edge.to_id) {
                            neighbor_nodes.push(node.clone());
                        }
                    }
                }
            }
        }

        // Get incoming neighbors
        if req.direction == 1 || req.direction == 2 {
            if let Some(edges) = ns.reverse_edges.get(&req.node_id) {
                for edge in edges.iter() {
                    let type_matches = req.edge_types.is_empty()
                        || req.edge_types.contains(&edge.edge_type);
                    if type_matches && !seen.contains(&edge.from_id) {
                        seen.insert(edge.from_id.clone());
                        neighbor_edges.push(edge.clone());
                        if let Some(node) = ns.nodes.get(&edge.from_id) {
                            neighbor_nodes.push(node.clone());
                        }
                    }
                }
            }
        }

        Ok(Response::new(GetNeighborsResponse {
            nodes: neighbor_nodes,
            edges: neighbor_edges,
            error: String::new(),
        }))
    }

    async fn add_temporal_edge(
        &self,
        request: Request<AddTemporalEdgeRequest>,
    ) -> Result<Response<AddTemporalEdgeResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let temporal_edge = TemporalEdge {
            from_id: req.from_id.clone(),
            edge_type: req.edge_type.clone(),
            to_id: req.to_id.clone(),
            properties: req.properties.clone(),
            valid_from: req.valid_from,
            valid_until: req.valid_until,
        };

        ns.temporal_edges
            .entry(req.from_id)
            .or_default()
            .push(temporal_edge);

        Ok(Response::new(AddTemporalEdgeResponse {
            success: true,
            error: String::new(),
        }))
    }

    async fn query_temporal_graph(
        &self,
        request: Request<QueryTemporalGraphRequest>,
    ) -> Result<Response<QueryTemporalGraphResponse>, Status> {
        let req = request.into_inner();
        let ns = self.get_or_create_namespace(&req.namespace);

        let mut result_edges = Vec::new();

        // Get all temporal edges for the node
        if let Some(edges) = ns.temporal_edges.get(&req.node_id) {
            for edge in edges.iter() {
                // Filter by edge type if specified
                if !req.edge_types.is_empty() && !req.edge_types.contains(&edge.edge_type) {
                    continue;
                }

                // Filter by temporal mode
                let matches = match req.mode {
                    0 => {
                        // POINT_IN_TIME: check if timestamp is within [valid_from, valid_until)
                        let ts = req.timestamp;
                        ts >= edge.valid_from && (edge.valid_until == 0 || ts < edge.valid_until)
                    }
                    1 => {
                        // RANGE: check if edge overlaps with [start_time, end_time)
                        let start = req.start_time;
                        let end = req.end_time;
                        let edge_end = if edge.valid_until == 0 {
                            u64::MAX
                        } else {
                            edge.valid_until
                        };
                        edge.valid_from < end && edge_end > start
                    }
                    2 => {
                        // CURRENT: check if current time is within validity
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;
                        now >= edge.valid_from && (edge.valid_until == 0 || now < edge.valid_until)
                    }
                    _ => false,
                };

                if matches {
                    result_edges.push(edge.clone());
                }
            }
        }

        Ok(Response::new(QueryTemporalGraphResponse {
            edges: result_edges,
            error: String::new(),
        }))
    }
}
