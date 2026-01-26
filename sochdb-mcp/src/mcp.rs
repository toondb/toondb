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

//! MCP Protocol Implementation
//!
//! Implements the Model Context Protocol methods:
//! - initialize / initialized
//! - tools/list
//! - tools/call
//! - resources/list (optional)
//! - resources/read (optional)

use std::sync::Arc;

use serde_json::{Value, json};
use tracing::{debug, info, warn};

use sochdb::connection::EmbeddedConnection;
use sochdb::DurableSochClient;

use crate::jsonrpc::{RpcRequest, RpcResponse};
use crate::tools::{ToolExecutor, get_built_in_tools};

/// MCP Protocol version we support
const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server info
const SERVER_NAME: &str = "sochdb-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// MCP Server state
pub struct McpServer {
    /// SochDB connection
    conn: Arc<EmbeddedConnection>,
    /// Tool executor
    executor: ToolExecutor,
    /// Whether initialize has been called
    #[allow(dead_code)]
    initialized: bool,
}

impl McpServer {
    pub fn new(conn: Arc<EmbeddedConnection>) -> Self {
        Self {
            conn: Arc::clone(&conn),
            executor: ToolExecutor::new(DurableSochClient::from_connection(conn)),
            initialized: false,
        }
    }

    /// Get the underlying database connection
    #[allow(dead_code)]
    pub fn connection(&self) -> &Arc<EmbeddedConnection> {
        &self.conn
    }

    /// Get database-level statistics
    #[allow(dead_code)]
    pub fn db_stats(&self) -> sochdb::connection::DatabaseStats {
        self.conn.db_stats()
    }

    /// Dispatch a JSON-RPC request to the appropriate handler
    pub fn dispatch(&self, req: &RpcRequest) -> RpcResponse {
        debug!("Dispatching method: {}", req.method);

        match req.method.as_str() {
            // Lifecycle
            "initialize" => self.handle_initialize(req),
            "notifications/initialized" | "initialized" => self.handle_initialized(req),
            "shutdown" => self.handle_shutdown(req),

            // Tools
            "tools/list" => self.handle_tools_list(req),
            "tools/call" => self.handle_tools_call(req),

            // Resources (optional)
            "resources/list" => self.handle_resources_list(req),
            "resources/read" => self.handle_resources_read(req),

            // Prompts (not implemented)
            "prompts/list" => RpcResponse::success(req.id.clone(), json!({"prompts": []})),
            "prompts/get" => RpcResponse::method_not_found(req.id.clone(), &req.method),

            // Unknown
            _ => {
                warn!("Unknown method: {}", req.method);
                RpcResponse::method_not_found(req.id.clone(), &req.method)
            }
        }
    }

    // =========================================================================
    // Lifecycle Methods
    // =========================================================================

    fn handle_initialize(&self, req: &RpcRequest) -> RpcResponse {
        info!("MCP initialize called");

        // Parse client info from params (optional)
        let client_info = req.params.get("clientInfo");
        if let Some(info) = client_info {
            info!("Client: {:?}", info);
        }

        // Return server capabilities
        let result = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": false  // We don't dynamically add/remove tools
                },
                "resources": {
                    "subscribe": false,
                    "listChanged": false
                }
                // No "prompts" capability - we don't use prompts
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        });

        RpcResponse::success(req.id.clone(), result)
    }

    fn handle_initialized(&self, req: &RpcRequest) -> RpcResponse {
        info!("MCP initialized notification received");
        // This is a notification, but some clients expect a response
        if req.id.is_null() {
            // Notification - no response needed, but return empty for safety
            RpcResponse::success(Value::Null, json!({}))
        } else {
            RpcResponse::success(req.id.clone(), json!({}))
        }
    }

    fn handle_shutdown(&self, req: &RpcRequest) -> RpcResponse {
        info!("MCP shutdown requested");
        RpcResponse::success(req.id.clone(), json!({}))
    }

    // =========================================================================
    // Tools Methods
    // =========================================================================

    fn handle_tools_list(&self, req: &RpcRequest) -> RpcResponse {
        debug!("tools/list called");

        let tools = get_built_in_tools();

        // TODO: Add catalog operations as tools when catalog is exposed
        // For now, just return built-in tools

        RpcResponse::success(req.id.clone(), json!({"tools": tools}))
    }

    fn handle_tools_call(&self, req: &RpcRequest) -> RpcResponse {
        debug!("tools/call called");

        // Parse tool call params
        let name = match req.params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return RpcResponse::invalid_params(req.id.clone(), "Missing 'name' parameter"),
        };

        let arguments = req.params.get("arguments").cloned().unwrap_or(json!({}));

        info!("Calling tool: {} with args: {:?}", name, arguments);

        // Determine output format from arguments (Task 5: TOON Wire Format)
        let format = arguments
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("toon");

        let mime_type = if format == "json" {
            "application/json"
        } else {
            "text/x-toon"
        };

        // Execute the tool
        match self.executor.execute(name, arguments) {
            Ok(result) => {
                // MCP expects result in "content" array format
                // Use appropriate mime type based on format (Task 5)
                RpcResponse::success(
                    req.id.clone(),
                    json!({
                        "content": [{
                            "type": "text",
                            "mimeType": mime_type,
                            "text": result
                        }]
                    }),
                )
            }
            Err(e) => RpcResponse::sochdb_error(req.id.clone(), e),
        }
    }

    // =========================================================================
    // Resources Methods (with Semantic Metadata - Task 2)
    // =========================================================================

    fn handle_resources_list(&self, req: &RpcRequest) -> RpcResponse {
        debug!("resources/list called");

        // Scan all paths and extract unique top-level prefixes as tables
        let tables: Vec<String> = {
            self.conn.begin().ok();
            let result = match self.conn.scan("/") {
                Ok(results) => {
                    let mut table_set = std::collections::HashSet::new();
                    for (key, _) in results {
                        let parts: Vec<&str> = key.trim_start_matches('/').split('/').collect();
                        if let Some(first) = parts.first() {
                            if !first.is_empty() {
                                table_set.insert(first.to_string());
                            }
                        }
                    }
                    table_set.into_iter().collect()
                }
                Err(_) => Vec::new(),
            };
            self.conn.abort().ok();
            result
        };

        // Expose tables as resources with semantic metadata
        let resources: Vec<Value> = tables
            .iter()
            .map(|name| {
                let meta = get_resource_metadata(name);
                json!({
                    "uri": format!("sochdb://tables/{}", name),
                    "name": name,
                    "description": meta.description,
                    "mimeType": meta.mime_type,
                    // Semantic metadata annotations
                    "annotations": {
                        "tableRole": meta.role,
                        "primaryKey": meta.primary_key,
                        "clusterKey": meta.cluster_key,
                        "tsColumn": meta.ts_column,
                        "backedByVectorIndex": meta.backed_by_vector_index,
                        "embeddingDimension": meta.embedding_dimension
                    }
                })
            })
            .collect();

        // Also expose predefined views
        let views = get_predefined_views();

        RpcResponse::success(
            req.id.clone(),
            json!({
                "resources": resources,
                "views": views
            }),
        )
    }

    fn handle_resources_read(&self, req: &RpcRequest) -> RpcResponse {
        debug!("resources/read called");

        let uri = match req.params.get("uri").and_then(|v| v.as_str()) {
            Some(u) => u,
            None => return RpcResponse::invalid_params(req.id.clone(), "Missing 'uri' parameter"),
        };

        // Check format preference
        let format = req
            .params
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("toon");

        let mime_type = if format == "json" {
            "application/json"
        } else {
            "text/x-toon"
        };

        // Parse URI: sochdb://tables/<name> or sochdb://views/<name>
        if let Some(table_name) = uri.strip_prefix("sochdb://tables/") {
            self.read_table_resource(req, table_name, mime_type)
        } else if let Some(view_name) = uri.strip_prefix("sochdb://views/") {
            self.read_view_resource(req, view_name, mime_type)
        } else {
            RpcResponse::invalid_params(req.id.clone(), format!("Invalid URI: {}", uri))
        }
    }

    fn read_table_resource(
        &self,
        req: &RpcRequest,
        table_name: &str,
        mime_type: &str,
    ) -> RpcResponse {
        // Use scan to get table data
        let prefix = format!("/{}", table_name);
        self.conn.begin().ok();
        
        let scan_result = self.conn.scan(&prefix);
        self.conn.abort().ok();
        
        match scan_result {
            Ok(results) if !results.is_empty() => {
                let meta = get_resource_metadata(table_name);
                
                // Extract unique field names from paths
                let mut fields = std::collections::HashSet::new();
                for (key, _) in &results {
                    let parts: Vec<&str> = key.trim_start_matches('/').split('/').collect();
                    if parts.len() >= 3 {
                        // path format: /table/id/field
                        fields.insert(parts[2].to_string());
                    }
                }
                
                let content = serde_json::to_string_pretty(&json!({
                    "name": table_name,
                    "fields": fields.into_iter().collect::<Vec<_>>(),
                    "rowCount": results.len(),
                    // Semantic metadata
                    "tableRole": meta.role,
                    "primaryKey": meta.primary_key,
                    "clusterKey": meta.cluster_key,
                    "tsColumn": meta.ts_column,
                    "backedByVectorIndex": meta.backed_by_vector_index,
                    "embeddingDimension": meta.embedding_dimension,
                    "description": meta.description,
                    // Usage hints for LLMs
                    "usageHints": meta.usage_hints
                }))
                .unwrap_or_else(|_| "{}".to_string());

                RpcResponse::success(
                    req.id.clone(),
                    json!({
                        "contents": [{
                            "uri": format!("sochdb://tables/{}", table_name),
                            "mimeType": mime_type,
                            "text": content
                        }]
                    }),
                )
            }
            _ => RpcResponse::sochdb_error(
                req.id.clone(),
                format!("Table not found: {}", table_name),
            ),
        }
    }

    fn read_view_resource(
        &self,
        req: &RpcRequest,
        view_name: &str,
        mime_type: &str,
    ) -> RpcResponse {
        let views = get_predefined_views();

        if let Some(view) = views
            .iter()
            .find(|v| v.get("name").and_then(|n| n.as_str()) == Some(view_name))
        {
            RpcResponse::success(
                req.id.clone(),
                json!({
                    "contents": [{
                        "uri": format!("sochdb://views/{}", view_name),
                        "mimeType": mime_type,
                        "text": serde_json::to_string_pretty(view).unwrap_or_default()
                    }]
                }),
            )
        } else {
            RpcResponse::sochdb_error(req.id.clone(), format!("View not found: {}", view_name))
        }
    }
}

// ============================================================================
// Resource Metadata (Task 2 & 7: Semantic Metadata & Stable Identifiers)
// ============================================================================

struct ResourceMetadata {
    role: &'static str,
    primary_key: Vec<&'static str>,
    cluster_key: Option<Vec<&'static str>>,
    ts_column: Option<&'static str>,
    backed_by_vector_index: bool,
    embedding_dimension: Option<usize>,
    description: &'static str,
    mime_type: &'static str,
    usage_hints: Vec<&'static str>,
}

/// Get semantic metadata for known tables (O(1) lookup)
fn get_resource_metadata(table_name: &str) -> ResourceMetadata {
    match table_name {
        "episodes" => ResourceMetadata {
            role: "core_memory",
            primary_key: vec!["episode_id"],
            cluster_key: Some(vec!["ts_start"]),
            ts_column: Some("ts_start"),
            backed_by_vector_index: true,
            embedding_dimension: Some(1536),
            description: "Task/conversation episodes. Use memory.search_episodes for semantic search.",
            mime_type: "text/x-toon",
            usage_hints: vec![
                "Use memory.search_episodes(query, k) for semantic search",
                "Use memory.get_episode_timeline(episode_id) for events",
                "Clustered by ts_start for efficient time-range queries",
            ],
        },
        "events" => ResourceMetadata {
            role: "log",
            primary_key: vec!["episode_id", "seq"],
            cluster_key: Some(vec!["episode_id", "seq"]),
            ts_column: Some("ts"),
            backed_by_vector_index: false,
            embedding_dimension: None,
            description: "Event log within episodes. Use logs.tail or logs.timeline.",
            mime_type: "text/x-toon",
            usage_hints: vec![
                "Use logs.tail(table='events', limit=N) for recent events",
                "Use logs.timeline(entity_id, from_ts, to_ts) for time ranges",
                "Append-only, optimized for tail reads",
            ],
        },
        "entities" => ResourceMetadata {
            role: "dimension",
            primary_key: vec!["entity_id"],
            cluster_key: Some(vec!["kind"]),
            ts_column: Some("updated_at"),
            backed_by_vector_index: true,
            embedding_dimension: Some(1536),
            description: "Entities (users, projects, services). Use memory.search_entities.",
            mime_type: "text/x-toon",
            usage_hints: vec![
                "Use memory.search_entities(query, kind, k) for search",
                "Use memory.get_entity_facts(entity_id) for details",
                "Clustered by kind for efficient filtering",
            ],
        },
        _ => ResourceMetadata {
            role: "user_defined",
            primary_key: vec![],
            cluster_key: None,
            ts_column: None,
            backed_by_vector_index: false,
            embedding_dimension: None,
            description: "User-defined table",
            mime_type: "application/json",
            usage_hints: vec!["Use sochdb.query for custom queries"],
        },
    }
}

/// Get predefined views (Task 7: Stable Identifiers)
fn get_predefined_views() -> Vec<Value> {
    vec![
        json!({
            "name": "conversation_view",
            "uri": "sochdb://views/conversation_view",
            "description": "Conversation history with role, content, timestamp",
            "definition": "SELECT episode_id, seq, role, content, ts FROM events WHERE event_type = 'message' ORDER BY ts",
            "columns": ["episode_id", "seq", "role", "content", "ts"]
        }),
        json!({
            "name": "tool_calls_view",
            "uri": "sochdb://views/tool_calls_view",
            "description": "Tool invocations with inputs/outputs",
            "definition": "SELECT episode_id, seq, tool_name, input, output, latency_ms, ts FROM events WHERE event_type = 'tool_call' ORDER BY ts",
            "columns": ["episode_id", "seq", "tool_name", "input", "output", "latency_ms", "ts"]
        }),
        json!({
            "name": "error_view",
            "uri": "sochdb://views/error_view",
            "description": "Error events with context",
            "definition": "SELECT episode_id, seq, error_type, message, stack_trace, ts FROM events WHERE event_type = 'error' ORDER BY ts DESC",
            "columns": ["episode_id", "seq", "error_type", "message", "stack_trace", "ts"]
        }),
        json!({
            "name": "episode_summary_view",
            "uri": "sochdb://views/episode_summary_view",
            "description": "Episode summaries for quick browsing",
            "definition": "SELECT episode_id, episode_type, summary, tags, ts_start, ts_end FROM episodes ORDER BY ts_start DESC",
            "columns": ["episode_id", "episode_type", "summary", "tags", "ts_start", "ts_end"]
        }),
        json!({
            "name": "entity_directory_view",
            "uri": "sochdb://views/entity_directory_view",
            "description": "Entity directory by kind",
            "definition": "SELECT entity_id, kind, name, description, updated_at FROM entities ORDER BY kind, name",
            "columns": ["entity_id", "kind", "name", "description", "updated_at"]
        }),
    ]
}
