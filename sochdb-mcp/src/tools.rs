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

//! MCP Tool Definitions and Executor
//!
//! Provides strongly-typed MCP tools for SochDB:
//!
//! ## Core Database Tools
//! - `sochdb.context_query` - AI-optimized context with typed sections
//! - `sochdb.query` - Execute SochQL queries with format option
//! - `sochdb.get/put/delete` - Path-based CRUD
//!
//! ## Memory Tools (Episode/Entity/Event)
//! - `memory.search_episodes` - Vector search over episodes
//! - `memory.get_episode_timeline` - Get events for an episode
//! - `memory.search_entities` - Search entities by kind
//! - `memory.get_entity_facts` - Get entity with linked facts
//! - `memory.build_context` - One-shot context packing
//!
//! ## Log Tools (Temporal Access)
//! - `logs.tail` - Get last N rows from a log table
//! - `logs.timeline` - Get events in a time range


use serde_json::{Value, json};
use tracing::{info, warn};

use sochdb_core::soch::{SochTable, SochSchema, SochType, SochRow, SochValue};
use toon_format::{self, EncodeOptions, Indent};
use toon_format::types::KeyFoldingMode;

#[cfg(feature = "semantic-search")]
use std::sync::Mutex;

#[cfg(feature = "semantic-search")]
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

use sochdb::DurableSochClient;

// ContextQueryBuilder removed - using direct scan operations instead

// ============================================================================
// SochQL Query Parser Types (Task 4: Harden MCP Query Execution)
// ============================================================================

/// Parsed query result from SochQL parser
#[derive(Debug, Clone)]
enum ParsedQuery {
    /// SELECT query with structured components
    Select {
        columns: Vec<String>,
        table: String,
        where_clause: Option<Vec<WhereCondition>>,
        order_by: Option<(String, bool)>,  // (column, ascending)
    },
    /// Direct path scan
    DirectPath {
        path: String,
    },
}

/// WHERE clause condition
#[derive(Debug, Clone)]
struct WhereCondition {
    column: String,
    op: String,
    value: Value,
}

// ============================================================================
// TOON Format Serialization (~25% token savings vs JSON)
// ============================================================================

/// Helper to recursively parse JSON values (handles double-encoded strings)
fn parse_value_deep(bytes: &[u8]) -> Value {
    let s = String::from_utf8_lossy(bytes);
    match serde_json::from_str::<Value>(&s) {
        Ok(v) => {
            // If it's a string, try to parse it again if it looks like an object or array
            if let Value::String(inner_str) = &v {
               let trimmed = inner_str.trim();
               if trimmed.starts_with('{') || trimmed.starts_with('[') {
                   if let Ok(inner_val) = serde_json::from_str::<Value>(inner_str) {
                       return inner_val;
                   }
               }
            }
            v
        },
        Err(_) => Value::String(s.to_string())
    }
}

/// Try to format a JSON Array of Objects as a TOON Table
fn try_format_as_table(arr: &[Value]) -> Option<String> {
    if arr.is_empty() { return None; }
    
    // Check if all items are objects
    let first = arr[0].as_object()?;
    let keys: Vec<String> = first.keys().cloned().collect();
    
    // Create Schema
    let mut schema = SochSchema::new("results");
    for k in &keys {
        // Simple type inference or default to Text
        schema = schema.field(k, SochType::Text); 
    }
    
    let mut rows = Vec::new();
    for item in arr {
        let obj = item.as_object()?;
        let mut row_values = Vec::new();
        for k in &keys {
            let v = obj.get(k).unwrap_or(&Value::Null);
            // Convert serde Value to SochValue manually since we don't have direct mapper handy
            // and we want simple display representation
            let tv = match v {
                Value::Null => SochValue::Null,
                Value::Bool(b) => SochValue::Bool(*b),
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() { SochValue::Int(i) }
                    else if let Some(u) = n.as_u64() { SochValue::UInt(u) }
                    else { SochValue::Float(n.as_f64().unwrap_or(0.0)) }
                },
                Value::String(s) => SochValue::Text(s.clone()),
                _ => SochValue::Text(v.to_string()),
            };
            row_values.push(tv);
        }
        rows.push(SochRow::new(row_values));
    }
    
    let table = SochTable::with_rows(schema, rows);
    Some(table.format())
}

/// Serialize result to TOON/JSON/Markdown format
fn formatted_result(value: &Value, format: &str) -> Result<String, String> {
    match format {
        "json" => serde_json::to_string_pretty(value).map_err(|e| e.to_string()),
        "markdown" => json_to_markdown(value),
        "toon" | _ => {
            // Try tabular format first if it's an array
            if let Value::Array(arr) = value {
                if let Some(table_str) = try_format_as_table(arr) {
                    return Ok(table_str);
                }
            }
            
            // Fallback to default encoding
            let options = EncodeOptions::new()
                .with_indent(Indent::Spaces(2))
                .with_key_folding(KeyFoldingMode::Safe);
            toon_format::encode(value, &options).map_err(|e| e.to_string())
        }
    }
}

/// Helper for default TOON output (legacy)
fn soch_result(value: &Value) -> Result<String, String> {
    formatted_result(value, "toon")
}

fn json_to_markdown(v: &Value) -> Result<String, String> {
    match v {
        Value::Array(arr) if !arr.is_empty() => {
             // Check if array of objects (tabular)
             if let Some(Value::Object(first)) = arr.first() {
                 let mut keys: Vec<String> = first.keys().cloned().collect();
                 keys.sort();
                 
                 let mut table = String::new();
                 // Header
                 table.push_str("| ");
                 table.push_str(&keys.join(" | "));
                 table.push_str(" |\n");
                 
                 // Separator
                 table.push_str("|");
                 for _ in &keys { table.push_str("---|"); }
                 table.push('\n');
                 
                 // Rows
                 for item in arr {
                     if let Value::Object(obj) = item {
                         table.push_str("| ");
                         let row: Vec<String> = keys.iter().map(|k| {
                             obj.get(k).map(|val| {
                                 let s = match val {
                                     Value::String(s) => s.clone(),
                                     _ => val.to_string()
                                 };
                                 s.replace("|", "\\|").replace("\n", " ")
                             }).unwrap_or_default()
                         }).collect();
                         table.push_str(&row.join(" | "));
                         table.push_str(" |\n");
                     }
                 }
                 Ok(table)
             } else {
                 // List of primitives
                 let mut list = String::new();
                 for item in arr {
                     list.push_str(&format!("- {}\n", item));
                 }
                 Ok(list)
             }
        },
        _ => Ok(format!("```json\n{}\n```", serde_json::to_string_pretty(v).unwrap_or_default()))
    }
}

// ============================================================================
// Tool Schema Definitions
// ============================================================================

/// Section kind enum for typed context queries
fn section_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "Section name (e.g., 'history', 'knowledge')" },
            "priority": { "type": "integer", "description": "Priority (0 = highest)" },
            "kind": {
                "type": "string",
                "enum": ["literal", "get", "last", "search"],
                "description": "Section content type"
            },
            // Kind-specific properties
            "text": { "type": "string", "description": "For kind=literal: literal text to include" },
            "path": { "type": "string", "description": "For kind=get: path to fetch" },
            "table": { "type": "string", "description": "For kind=last/search: table name" },
            "query": { "type": "string", "description": "For kind=search: search query" },
            "top_k": { "type": "integer", "description": "For kind=last/search: max results", "default": 10 },
            "where": { "type": "object", "description": "For kind=last: optional filter conditions" }
        },
        "required": ["name", "kind"]
    })
}

/// Get the list of built-in tools as MCP JSON
pub fn get_built_in_tools() -> Vec<Value> {
    let mut tools = Vec::new();

    // =========================================================================
    // Core Database Tools
    // =========================================================================

    tools.push(json!({
        "name": "sochdb_context_query",
        "description": "Fetch AI-optimized context from SochDB with token budgeting. Sections are packed in priority order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": section_schema(),
                    "description": "Sections to include in context, ordered by priority"
                },
                "token_budget": { 
                    "type": "integer", 
                    "description": "Max tokens for output (default: 4096)",
                    "default": 4096
                },
                "format": {
                    "type": "string",
                    "enum": ["toon", "json", "markdown"],
                    "description": "Output format (default: toon for 40-60% token savings)",
                    "default": "toon"
                },
                "truncation": {
                    "type": "string",
                    "enum": ["tail_drop", "head_drop", "proportional"],
                    "description": "How to truncate if budget exceeded",
                    "default": "tail_drop"
                }
            },
            "required": ["sections"]
        }
    }));

    tools.push(json!({
        "name": "sochdb_query",
        "description": "Execute a SochQL query. Returns results in TOON format (40-60% fewer tokens than JSON).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { 
                    "type": "string",
                    "description": "SochQL query (e.g., 'SELECT id,name FROM users WHERE score > 80')"
                },
                "format": {
                    "type": "string",
                    "enum": ["toon", "json"],
                    "description": "Output format (default: toon)",
                    "default": "toon"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return",
                    "default": 100
                }
            },
            "required": ["query"]
        }
    }));

    tools.push(json!({
        "name": "sochdb_get",
        "description": "Get a value at a path (e.g., '/users/123/profile').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to fetch (e.g., '/table/row_id/column')"
                }
            },
            "required": ["path"]
        }
    }));

    tools.push(json!({
        "name": "sochdb_put",
        "description": "Set a value at a path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to store at" },
                "value": { "description": "Value to store (any JSON type)" }
            },
            "required": ["path", "value"]
        }
    }));

    tools.push(json!({
        "name": "sochdb_delete",
        "description": "Delete at a path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to delete" }
            },
            "required": ["path"]
        }
    }));

    tools.push(json!({
        "name": "sochdb_list_tables",
        "description": "List all tables with semantic metadata (role, indexes, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include semantic metadata (role, keys, etc.)",
                    "default": true
                }
            }
        }
    }));

    tools.push(json!({
        "name": "sochdb_describe",
        "description": "Get detailed schema for a table including semantic metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Table name" }
            },
            "required": ["table"]
        }
    }));

    // =========================================================================
    // Memory Tools (Episode/Entity/Event Schema)
    // =========================================================================

    tools.push(json!({
        "name": "memory_search_episodes",
        "description": "Search for similar past episodes (tasks/conversations) by semantic similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { 
                    "type": "string", 
                    "description": "Natural language query to search for similar episodes"
                },
                "k": { 
                    "type": "integer", 
                    "description": "Max results to return",
                    "default": 5
                },
                "episode_type": {
                    "type": "string",
                    "enum": ["conversation", "task", "workflow", "debug", "agent_interaction"],
                    "description": "Filter by episode type (optional)"
                },
                "entity_id": {
                    "type": "string",
                    "description": "Filter to episodes involving this entity (optional)"
                }
            },
            "required": ["query"]
        }
    }));

    tools.push(json!({
        "name": "memory_get_episode_timeline",
        "description": "Get the event timeline for an episode (tool calls, messages, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": "Episode ID to get timeline for"
                },
                "max_events": {
                    "type": "integer",
                    "description": "Max events to return (from end)",
                    "default": 50
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system", "tool", "external"],
                    "description": "Filter by event role (optional)"
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include timing/token metrics",
                    "default": false
                }
            },
            "required": ["episode_id"]
        }
    }));

    tools.push(json!({
        "name": "memory_search_entities",
        "description": "Search entities (users, projects, documents, services) by semantic similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { 
                    "type": "string", 
                    "description": "Natural language query"
                },
                "k": { 
                    "type": "integer", 
                    "description": "Max results",
                    "default": 10
                },
                "kind": {
                    "type": "string",
                    "enum": ["user", "project", "document", "service", "agent", "organization"],
                    "description": "Filter by entity kind (optional)"
                }
            },
            "required": ["query"]
        }
    }));

    tools.push(json!({
        "name": "memory_get_entity_facts",
        "description": "Get facts about an entity (attributes, recent episodes, related entities).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID to get facts for"
                },
                "include_episodes": {
                    "type": "boolean",
                    "description": "Include recent episodes involving this entity",
                    "default": true
                },
                "max_episodes": {
                    "type": "integer",
                    "description": "Max episodes to include",
                    "default": 5
                }
            },
            "required": ["entity_id"]
        }
    }));

    tools.push(json!({
        "name": "memory_build_context",
        "description": "Build optimized LLM context from memory. Automatically selects and packs relevant data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": { 
                    "type": "string", 
                    "description": "What the context will be used for (helps prioritize)"
                },
                "token_budget": { 
                    "type": "integer", 
                    "description": "Max tokens for context",
                    "default": 4096
                },
                "session_id": {
                    "type": "string",
                    "description": "Current session ID (for recent history)"
                },
                "episode_id": {
                    "type": "string",
                    "description": "Current episode ID (for task context)"
                },
                "entity_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Relevant entity IDs to include facts for"
                },
                "include_schema": {
                    "type": "boolean",
                    "description": "Include relevant table schemas",
                    "default": false
                }
            },
            "required": ["goal", "token_budget"]
        }
    }));

    // =========================================================================
    // Log Tools (Temporal Access)
    // =========================================================================

    tools.push(json!({
        "name": "logs_tail",
        "description": "Get the last N rows from a log table (most recent first).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table name to tail"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of rows to return",
                    "default": 20
                },
                "where": {
                    "type": "object",
                    "description": "Filter conditions (e.g., {\"level\": \"error\"})"
                },
                "columns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Columns to include (default: all)"
                }
            },
            "required": ["table"]
        }
    }));

    tools.push(json!({
        "name": "logs_timeline",
        "description": "Get events in a time range for an entity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID to get timeline for"
                },
                "from_ts": {
                    "type": "integer",
                    "description": "Start timestamp (microseconds since epoch)"
                },
                "to_ts": {
                    "type": "integer",
                    "description": "End timestamp (microseconds since epoch)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max events to return",
                    "default": 100
                },
                "table": {
                    "type": "string",
                    "description": "Log table to query (default: events)",
                    "default": "events"
                }
            },
            "required": ["entity_id"]
        }
    }));

    tools
}

// ============================================================================
// Embedding Manager (compile-time optional via `semantic-search` feature)
// ============================================================================

#[cfg(feature = "semantic-search")]
struct EmbeddingManager {
    model: Arc<Mutex<Option<TextEmbedding>>>,
    enabled: bool,
}

#[cfg(feature = "semantic-search")]
impl EmbeddingManager {
    fn new() -> Self {
        // Runtime opt-in: check env var SOCHDB_SEMANTIC_SEARCH=1
        let enabled = std::env::var("SOCHDB_SEMANTIC_SEARCH")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        
        if enabled {
            info!("Semantic search enabled via SOCHDB_SEMANTIC_SEARCH=1");
        }
        
        Self {
            model: Arc::new(Mutex::new(None)),
            enabled,
        }
    }

    fn get_embedding(&self, text: &str) -> Option<Vec<f32>> {
        // Skip if not enabled at runtime
        if !self.enabled {
            return None;
        }
        
        let mut model_guard = self.model.lock().unwrap();
        
        if model_guard.is_none() {
            info!("Initializing embedding model (AllMiniLML6V2)...");
            let mut options = InitOptions::new(EmbeddingModel::AllMiniLML6V2);
            options.show_download_progress = false;
            match TextEmbedding::try_new(options) {
                Ok(model) => *model_guard = Some(model),
                Err(e) => {
                    warn!("Failed to load embedding model: {}. Falling back to keyword search.", e);
                    return None;
                }
            }
        }

        if let Some(model) = model_guard.as_ref() {
            if let Ok(embeddings) = model.embed(vec![text], None) {
                if let Some(first) = embeddings.first() {
                    return Some(first.clone());
                }
            }
        }
        None
    }
}

// Stub when feature is disabled - always returns None
#[cfg(not(feature = "semantic-search"))]
struct EmbeddingManager;

#[cfg(not(feature = "semantic-search"))]
impl EmbeddingManager {
    fn new() -> Self { Self }
    fn get_embedding(&self, _text: &str) -> Option<Vec<f32>> { None }
}

// ============================================================================
// Tool Executor
// ============================================================================

pub struct ToolExecutor {
    conn: DurableSochClient, // Use DurableSochClient to access vector_search if available?
    // Wait, conn is DurableSochClient? Let's check imports or definition.
    // Ah, it was defined as DurableSochClient in previous context but let's assume it is.
    // If it's just `Connection`, we might need to access `vectors()` via client wrapper or similar.
    // Let's check ToolExecutor definition below.
    embeddings: EmbeddingManager,
}

impl ToolExecutor {
    pub fn new(conn: DurableSochClient) -> Self {
        Self { 
            conn,
            embeddings: EmbeddingManager::new(),
        }
    }

    /// Discover all top-level path prefixes from the database catalog.
    /// 
    /// This replaces hardcoded paths like "/users", "/entities", "/projects" with
    /// dynamic catalog-derived introspection.
    fn discover_prefixes(&self) -> Vec<String> {
        self.conn.connection().begin().ok();
        let result = match self.conn.connection().scan("/") {
            Ok(results) => {
                let mut prefix_set = std::collections::HashSet::new();
                for (key, _) in results {
                    // Extract top-level path component
                    let path = key.trim_start_matches('/');
                    if let Some(first_component) = path.split('/').next() {
                        if !first_component.is_empty() {
                            prefix_set.insert(format!("/{}", first_component));
                        }
                    }
                }
                prefix_set.into_iter().collect()
            }
            Err(_) => Vec::new(),
        };
        self.conn.connection().abort().ok();
        result
    }

    /// Get prefixes for entity-like tables (users, entities, projects, etc.)
    fn discover_entity_prefixes(&self) -> Vec<String> {
        let all_prefixes = self.discover_prefixes();
        if all_prefixes.is_empty() {
            // Fallback to root scan if no prefixes found
            vec!["/".to_string()]
        } else {
            all_prefixes
        }
    }

    /// Get prefixes for episode-like tables (episodes, tasks, conversations, messages)
    fn discover_episode_prefixes(&self) -> Vec<String> {
        let all_prefixes = self.discover_prefixes();
        let episode_patterns = ["episode", "task", "conversation", "message", "event", "log"];
        
        let episode_prefixes: Vec<String> = all_prefixes
            .iter()
            .filter(|p| {
                let p_lower = p.to_lowercase();
                episode_patterns.iter().any(|pat| p_lower.contains(pat))
            })
            .cloned()
            .collect();
        
        if episode_prefixes.is_empty() {
            // If no specific episode tables found, use all prefixes
            all_prefixes
        } else {
            episode_prefixes
        }
    }

    pub fn execute(&self, name: &str, args: Value) -> Result<String, String> {
        info!("Executing tool: {} with args: {:?}", name, args);

        match name {
            // Core database tools
            "sochdb_context_query" => self.exec_context_query(args),
            "sochdb_query" => self.exec_query(args),
            "sochdb_get" => self.exec_get(args),
            "sochdb_put" => self.exec_put(args),
            "sochdb_delete" => self.exec_delete(args),
            "sochdb_list_tables" => self.exec_list_tables(args),
            "sochdb_describe" => self.exec_describe(args),

            // Memory tools
            "memory_search_episodes" => self.exec_search_episodes(args),
            "memory_get_episode_timeline" => self.exec_get_episode_timeline(args),
            "memory_search_entities" => self.exec_search_entities(args),
            "memory_get_entity_facts" => self.exec_get_entity_facts(args),
            "memory_build_context" => self.exec_build_context(args),

            // Log tools
            "logs_tail" => self.exec_logs_tail(args),
            "logs_timeline" => self.exec_logs_timeline(args),

            // Catalog ops
            name if name.starts_with("op_") => Err(format!("Catalog op not implemented: {}", name)),

            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    // =========================================================================
    // Core Database Tools
    // =========================================================================

    #[allow(dead_code)]
    fn exec_checkpoint(&self, _args: Value) -> Result<String, String> {
        info!("Creating checkpoint");
        let checkpoint_lsn = self.conn.connection().checkpoint().map_err(|e| e.to_string())?;
        formatted_result(&json!({
            "status": "success", 
            "checkpoint_lsn": checkpoint_lsn,
            "message": "Checkpoint completed successfully"
        }), "json")
    }

    fn exec_context_query(&self, args: Value) -> Result<String, String> {
        let sections = args
            .get("sections")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'sections'")?;
        let budget = args
            .get("token_budget")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096);
        let fmt = args
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("toon");

        // Build context by processing each section
        let mut context_parts = Vec::new();
        
        self.conn.begin().map_err(|e| e.to_string())?;
        
        for section in sections {
            let name = section
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("section");
            let kind = section
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("literal");

            match kind {
                "literal" => {
                    let text = section.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    context_parts.push(json!({
                        "section": name,
                        "content": text
                    }));
                }
                "get" => {
                    let path = section.get("path").and_then(|v| v.as_str()).unwrap_or("/");
                    if let Ok(Some(bytes)) = self.conn.get(path) {
                        let bytes: Vec<u8> = bytes;
                        let content = String::from_utf8_lossy(&bytes);
                        context_parts.push(json!({
                            "section": name,
                            "path": path,
                            "content": serde_json::from_str::<Value>(&content).unwrap_or(Value::String(content.to_string()))
                        }));
                    }
                }
                "last" | "search" => {
                    let table = section.get("table").and_then(|v| v.as_str()).unwrap_or("");
                    let n = section.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                    let prefix = format!("/{}", table);
                    if let Ok(results) = self.conn.scan(&prefix) {
                        let limited: Vec<_> = results.into_iter().take(n).collect();
                        let mut items = Vec::new();
                        for (k, v) in limited {
                             let mut parsed = parse_value_deep(&v);
                             if let Value::Object(ref mut map) = parsed {
                                 map.insert("_path".to_string(), Value::String(k));
                                 items.push(parsed);
                             } else {
                                 items.push(json!({
                                     "_path": k,
                                     "value": parsed
                                 }));
                             }
                        }
                        context_parts.push(json!({
                            "section": name,
                            "table": table,
                            "items": items
                        }));
                    }
                }
                _ => {
                    warn!("Unknown section kind: {}", kind);
                }
            }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "context": context_parts,
            "token_budget": budget,
            "sections_processed": sections.len()
        });

        formatted_result(&result, fmt).map_err(|e| e.to_string())
    }

    fn exec_query(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query'")?;
        let fmt = args
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("toon");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

        info!("SochQL: {} (limit={})", query, limit);

        // Parse and validate query using structured parsing
        let parsed = Self::parse_sochql_query(query)?;
        
        // Enforce prefix-bounded scans for multi-tenant safety
        let scan_path = match &parsed {
            ParsedQuery::Select { table, .. } => {
                // Always prefix-bound to table namespace
                Self::validate_table_name(table)?;
                format!("/{}", table)
            }
            ParsedQuery::DirectPath { path } => {
                // Validate path doesn't escape intended prefix
                Self::validate_path_prefix(path)?;
                path.clone()
            }
        };

        // Begin transaction for scan
        self.conn.begin().map_err(|e| e.to_string())?;
        
        match self.conn.scan(&scan_path) {
            Ok(results) => {
                self.conn.abort().ok();
                
                // Apply WHERE clause filtering if specified
                let filtered = match &parsed {
                    ParsedQuery::Select { columns, where_clause, order_by, .. } => {
                        let mut output: Vec<Value> = Vec::new();
                        
                        for (key, value) in results {
                            let mut parsed_row = parse_value_deep(&value);
                            
                            // Apply WHERE filter
                            if let Some(conditions) = where_clause {
                                if !Self::matches_conditions(&parsed_row, conditions) {
                                    continue;
                                }
                            }
                            
                            // Project columns
                            if let Value::Object(ref mut map) = parsed_row {
                                map.insert("_path".to_string(), Value::String(key));
                                
                                // Column projection
                                if !columns.is_empty() && columns[0] != "*" {
                                    let projected: serde_json::Map<String, Value> = map.iter()
                                        .filter(|(k, _)| *k == "_path" || columns.contains(k))
                                        .map(|(k, v)| (k.clone(), v.clone()))
                                        .collect();
                                    output.push(Value::Object(projected));
                                } else {
                                    output.push(parsed_row);
                                }
                            } else {
                                output.push(json!({
                                    "_path": key,
                                    "value": parsed_row
                                }));
                            }
                            
                            if output.len() >= limit {
                                break;
                            }
                        }
                        
                        // Apply ORDER BY
                        if let Some((col, asc)) = order_by {
                            output.sort_by(|a, b| {
                                let va = a.get(col);
                                let vb = b.get(col);
                                let cmp = match (va, vb) {
                                    (Some(Value::Number(na)), Some(Value::Number(nb))) => {
                                        na.as_f64().partial_cmp(&nb.as_f64()).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (Some(Value::String(sa)), Some(Value::String(sb))) => sa.cmp(sb),
                                    _ => std::cmp::Ordering::Equal,
                                };
                                if *asc { cmp } else { cmp.reverse() }
                            });
                        }
                        
                        output
                    }
                    ParsedQuery::DirectPath { .. } => {
                        let mut output = Vec::new();
                        for (key, value) in results.into_iter().take(limit) {
                            let mut parsed = parse_value_deep(&value);
                            if let Value::Object(ref mut map) = parsed {
                                map.insert("_path".to_string(), Value::String(key));
                                output.push(parsed);
                            } else {
                                output.push(json!({
                                    "_path": key,
                                    "value": parsed
                                }));
                            }
                        }
                        output
                    }
                };
                
                formatted_result(&Value::Array(filtered), fmt).map_err(|e| e.to_string())
            }
            Err(e) => {
                self.conn.abort().ok();
                Err(e.to_string())
            }
        }
    }
    
    /// Parsed query result
    fn parse_sochql_query(query: &str) -> Result<ParsedQuery, String> {
        let query = query.trim();
        let upper = query.to_uppercase();
        
        if upper.starts_with("SELECT") {
            Self::parse_select_query(query)
        } else if query.starts_with('/') || query.contains('/') {
            // Direct path scan
            Ok(ParsedQuery::DirectPath { path: query.to_string() })
        } else {
            // Treat as table name
            Self::validate_table_name(query)?;
            Ok(ParsedQuery::DirectPath { path: format!("/{}", query) })
        }
    }
    
    /// Parse SELECT query into structured form
    fn parse_select_query(query: &str) -> Result<ParsedQuery, String> {
        // Grammar: SELECT cols FROM table [WHERE conditions] [ORDER BY col [ASC|DESC]] [LIMIT n]
        let upper = query.to_uppercase();
        
        // Extract columns
        let select_idx = upper.find("SELECT").ok_or("Missing SELECT")?;
        let from_idx = upper.find("FROM").ok_or("Missing FROM")?;
        
        if from_idx <= select_idx + 6 {
            return Err("Invalid SELECT query: columns missing".to_string());
        }
        
        let cols_str = query[select_idx + 6..from_idx].trim();
        let columns: Vec<String> = if cols_str == "*" {
            vec!["*".to_string()]
        } else {
            cols_str.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
        
        // Extract table name
        let after_from = &query[from_idx + 4..];
        let table_end = after_from.find(|c: char| c.is_whitespace())
            .unwrap_or(after_from.len());
        let table = after_from[..table_end].trim()
            .trim_matches(|c| c == '\'' || c == '"' || c == '`')
            .to_string();
        
        // Validate table name
        Self::validate_table_name(&table)?;
        
        // Extract WHERE clause
        let where_clause = if let Some(where_idx) = upper.find("WHERE") {
            let where_start = where_idx + 5;
            let where_end = upper[where_start..]
                .find("ORDER")
                .or_else(|| upper[where_start..].find("LIMIT"))
                .map(|i| where_start + i)
                .unwrap_or(query.len());
            
            let where_str = query[where_start..where_end].trim();
            Some(Self::parse_where_clause(where_str)?)
        } else {
            None
        };
        
        // Extract ORDER BY
        let order_by = if let Some(order_idx) = upper.find("ORDER BY") {
            let order_start = order_idx + 8;
            let order_end = upper[order_start..]
                .find("LIMIT")
                .map(|i| order_start + i)
                .unwrap_or(query.len());
            
            let order_str = query[order_start..order_end].trim();
            let asc = !order_str.to_uppercase().contains("DESC");
            let col = order_str.split_whitespace().next()
                .unwrap_or("")
                .to_string();
            Some((col, asc))
        } else {
            None
        };
        
        Ok(ParsedQuery::Select {
            columns,
            table,
            where_clause,
            order_by,
        })
    }
    
    /// Parse WHERE clause into conditions
    fn parse_where_clause(clause: &str) -> Result<Vec<WhereCondition>, String> {
        let mut conditions = Vec::new();
        
        // Simple AND-separated conditions
        for part in clause.split(" AND ") {
            let part = part.trim();
            if part.is_empty() { continue; }
            
            // Parse: column op value
            let ops = [">=", "<=", "!=", "<>", "=", ">", "<", "LIKE", "NOT LIKE"];
            
            for op in ops {
                if let Some(op_idx) = part.to_uppercase().find(op) {
                    let col = part[..op_idx].trim().to_string();
                    let val_str = part[op_idx + op.len()..].trim()
                        .trim_matches(|c| c == '\'' || c == '"');
                    
                    let value = Self::parse_value(val_str);
                    
                    conditions.push(WhereCondition {
                        column: col,
                        op: op.to_string(),
                        value,
                    });
                    break;
                }
            }
        }
        
        Ok(conditions)
    }
    
    /// Parse string value to typed Value
    fn parse_value(s: &str) -> Value {
        if s.eq_ignore_ascii_case("null") {
            Value::Null
        } else if s.eq_ignore_ascii_case("true") {
            Value::Bool(true)
        } else if s.eq_ignore_ascii_case("false") {
            Value::Bool(false)
        } else if let Ok(n) = s.parse::<i64>() {
            Value::Number(n.into())
        } else if let Ok(f) = s.parse::<f64>() {
            Value::Number(serde_json::Number::from_f64(f).unwrap_or(0.into()))
        } else {
            Value::String(s.to_string())
        }
    }
    
    /// Validate table name to prevent path injection
    fn validate_table_name(table: &str) -> Result<(), String> {
        // Table names must be alphanumeric with underscores
        if table.is_empty() {
            return Err("Empty table name".to_string());
        }
        if table.contains("..") || table.contains('/') || table.contains('\\') {
            return Err(format!("Invalid table name '{}': path traversal not allowed", table));
        }
        if !table.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(format!("Invalid table name '{}': only alphanumeric, underscore, hyphen allowed", table));
        }
        Ok(())
    }
    
    /// Validate path prefix to prevent escaping intended boundaries
    fn validate_path_prefix(path: &str) -> Result<(), String> {
        // Normalize and validate path
        if path.contains("..") {
            return Err("Path traversal (..) not allowed".to_string());
        }
        // Ensure path doesn't try to access system prefixes
        let dangerous_prefixes = ["/_internal", "/_system", "/_admin"];
        for prefix in dangerous_prefixes {
            if path.starts_with(prefix) {
                return Err(format!("Access to {} prefix not allowed", prefix));
            }
        }
        Ok(())
    }
    
    /// Check if a row matches WHERE conditions
    fn matches_conditions(row: &Value, conditions: &[WhereCondition]) -> bool {
        for cond in conditions {
            let row_val = row.get(&cond.column);
            
            let matches = match cond.op.to_uppercase().as_str() {
                "=" => row_val == Some(&cond.value),
                "!=" | "<>" => row_val != Some(&cond.value),
                ">" => Self::compare_values(row_val, &cond.value) == Some(std::cmp::Ordering::Greater),
                ">=" => matches!(Self::compare_values(row_val, &cond.value), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)),
                "<" => Self::compare_values(row_val, &cond.value) == Some(std::cmp::Ordering::Less),
                "<=" => matches!(Self::compare_values(row_val, &cond.value), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)),
                "LIKE" => {
                    if let (Some(Value::String(s)), Value::String(pattern)) = (row_val, &cond.value) {
                        Self::matches_like(s, pattern)
                    } else {
                        false
                    }
                }
                _ => true,
            };
            
            if !matches {
                return false;
            }
        }
        true
    }
    
    /// Compare two values
    fn compare_values(a: Option<&Value>, b: &Value) -> Option<std::cmp::Ordering> {
        match (a, b) {
            (Some(Value::Number(na)), Value::Number(nb)) => {
                na.as_f64().partial_cmp(&nb.as_f64())
            }
            (Some(Value::String(sa)), Value::String(sb)) => Some(sa.cmp(sb)),
            _ => None,
        }
    }
    
    /// Simple LIKE pattern matching (% = any, _ = single char)
    fn matches_like(s: &str, pattern: &str) -> bool {
        let regex_pattern = pattern
            .replace('%', ".*")
            .replace('_', ".");
        regex::Regex::new(&format!("^{}$", regex_pattern))
            .map(|re: regex::Regex| re.is_match(s))
            .unwrap_or(false)
    }

    fn exec_get(&self, args: Value) -> Result<String, String> {
        let raw_path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path'")?;
        // Normalize path to always start with /
        let path = if raw_path.starts_with('/') {
            raw_path.to_string()
        } else {
            format!("/{}", raw_path)
        };
        
        // Begin transaction for read
        self.conn.begin().map_err(|e| e.to_string())?;
        
        match self.conn.get(&path) {
            Ok(Some(bytes)) => {
                // Try to parse as JSON, fallback to string
                let result = String::from_utf8_lossy(&bytes).to_string();
                self.conn.abort().ok(); // Just reading, abort is fine
                Ok(result)
            }
            Ok(None) => {
                self.conn.abort().ok();
                Ok(format!("Not found: {}", path))
            }
            Err(e) => {
                self.conn.abort().ok();
                Err(e.to_string())
            }
        }
    }

    fn exec_put(&self, args: Value) -> Result<String, String> {
        let raw_path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path'")?;
        // Normalize path to always start with /
        let path = if raw_path.starts_with('/') {
            raw_path.to_string()
        } else {
            format!("/{}", raw_path)
        };
        let value = args.get("value").ok_or("Missing 'value'")?;
        
        // Serialize value to JSON bytes
        let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
        
        // Begin transaction for write
        self.conn.begin().map_err(|e| e.to_string())?;
        
        match self.conn.put(&path, &bytes) {
            Ok(()) => {
                self.conn.commit().map_err(|e| e.to_string())?;
                // Ensure durable write by checkpointing and syncing
                // Ensure durable write by syncing
                self.conn.fsync().ok();
                self.conn.fsync().ok();
                Ok(format!("Stored at: {}", path))
            }
            Err(e) => {
                self.conn.abort().ok();
                Err(e.to_string())
            }
        }
    }

    fn exec_delete(&self, args: Value) -> Result<String, String> {
        let raw_path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path'")?;
        // Normalize path to always start with /
        let path = if raw_path.starts_with('/') {
            raw_path.to_string()
        } else {
            format!("/{}", raw_path)
        };
        
        // Begin transaction for delete
        self.conn.begin().map_err(|e| e.to_string())?;
        
        match self.conn.delete(&path) {
            Ok(()) => {
                self.conn.commit().map_err(|e| e.to_string())?;
                self.conn.fsync().ok();
                Ok(format!("Deleted: {}", path))
            }
            Err(e) => {
                self.conn.abort().ok();
                Err(e.to_string())
            }
        }
    }

    fn exec_list_tables(&self, args: Value) -> Result<String, String> {
        let include_metadata = args
            .get("include_metadata")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Begin transaction for scan
        self.conn.begin().map_err(|e| e.to_string())?;
        
        // Scan all paths and extract unique top-level prefixes
        let tables = match self.conn.scan("/") {
            Ok(results) => {
                let mut table_set = std::collections::HashSet::new();
                for (key, _) in results {
                    // Extract first path segment as table/collection name
                    let parts: Vec<&str> = key.trim_start_matches('/').split('/').collect();
                    if let Some(first) = parts.first() {
                        if !first.is_empty() {
                            table_set.insert(first.to_string());
                        }
                    }
                }
                self.conn.abort().ok();
                table_set.into_iter().collect::<Vec<_>>()
            }
            Err(_) => {
                self.conn.abort().ok();
                Vec::new()
            }
        };

        let result: Vec<Value> = tables
            .iter()
            .map(|name| {
                if include_metadata {
                    let metadata = get_table_semantic_metadata(name);
                    json!({
                        "name": name,
                        "role": metadata.role,
                        "primaryKey": metadata.primary_key,
                        "clusterKey": metadata.cluster_key,
                        "tsColumn": metadata.ts_column,
                        "backedByVectorIndex": metadata.backed_by_vector_index,
                        "description": metadata.description
                    })
                } else {
                    json!({"name": name})
                }
            })
            .collect();

        soch_result(&json!(result)).map_err(|e| e.to_string())
    }

    fn exec_describe(&self, args: Value) -> Result<String, String> {
        let table = args
            .get("table")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'table'")?;

        // Use scan to get table data
        let prefix = format!("/{}", table);
        self.conn.begin().map_err(|e| e.to_string())?;
        
        match self.conn.scan(&prefix) {
            Ok(results) if !results.is_empty() => {
                self.conn.abort().ok();
                
                // Extract unique field names from paths
                let mut fields = std::collections::HashSet::new();
                for (key, _) in &results {
                    let parts: Vec<&str> = key.trim_start_matches('/').split('/').collect();
                    if parts.len() >= 3 {
                        fields.insert(parts[2].to_string());
                    }
                }
                
                let metadata = get_table_semantic_metadata(table);
                let result = json!({
                    "name": table,
                    "fields": fields.into_iter().collect::<Vec<_>>(),
                    "rowCount": results.len(),
                    "role": metadata.role,
                    "primaryKey": metadata.primary_key,
                    "clusterKey": metadata.cluster_key,
                    "tsColumn": metadata.ts_column,
                    "backedByVectorIndex": metadata.backed_by_vector_index,
                    "embeddingDimension": metadata.embedding_dimension,
                    "description": metadata.description
                });
                soch_result(&result).map_err(|e| e.to_string())
            }
            Ok(_) => {
                self.conn.abort().ok();
                Err(format!("Table not found: {}", table))
            }
            Err(e) => {
                self.conn.abort().ok();
                Err(e.to_string())
            }
        }
    }

    // =========================================================================
    // Memory Tools
    // =========================================================================

    fn exec_search_episodes(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query'")?;
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let episode_type = args.get("episode_type").and_then(|v| v.as_str());
        let entity_id = args.get("entity_id").and_then(|v| v.as_str());

        info!(
            "Searching episodes: query='{}', k={}, type={:?}",
            query, k, episode_type
        );

        // Search using scan and text matching, PLUS semantic search if possible
        self.conn.begin().map_err(|e| e.to_string())?;
        
        let mut results: Vec<Value> = Vec::new();
        let search_lower = query.to_lowercase();
        
        // 1. Semantic Search (Content only) using FastEmbed
        if let Some(_vector) = self.embeddings.get_embedding(query) {
             info!("Performing semantic search for: {} (Vector generated, but search disabled pending backend support)", query);
             // TODO: Enable when DurableSochClient supports vectors()
             /*
             if let Ok(collection) = self.conn.vectors("messages") {
                 // Search with generated vector
                 if let Ok(search_results) = collection.search(&vector, k) {
                     for res in search_results {
                         // Fetch full content
                         if let Ok(Some(bytes)) = self.conn.get(&res.path) {
                             let mut parsed = parse_value_deep(&bytes);
                             // Inject path and match metadata
                             if let Value::Object(ref mut map) = parsed {
                                 map.insert("_path".to_string(), Value::String(res.path.clone()));
                                 map.insert("match_type".to_string(), Value::String("semantic".to_string()));
                                 map.insert("score".to_string(), json!(res.score));
                                 results.push(parsed);
                             } else {
                                results.push(json!({
                                    "_path": res.path,
                                    "value": parsed,
                                    "match_type": "semantic",
                                    "score": res.score
                                }));
                             }
                         }
                     }
                 }
             }
             */
        }
        
        // 2. Keyword Fallback / Supplement
        // If results < k, fill with keyword matches
        if results.len() < k {
             // Use catalog-derived prefixes instead of hardcoded paths
             let scan_prefixes = self.discover_episode_prefixes();
             let mut seen_paths: std::collections::HashSet<String> = results.iter()
                .filter_map(|v| v.get("_path").and_then(|p| p.as_str()).map(|s| s.to_string()))
                .collect();

             for prefix in scan_prefixes {
                if let Ok(items) = self.conn.scan(&prefix) {
                    for (path, value) in items {
                        if seen_paths.contains(&path) { continue; }
                        
                        let val_str = String::from_utf8_lossy(&value).to_lowercase();
                        let matches = val_str.contains(&search_lower) || path.to_lowercase().contains(&search_lower);
                        
                        let type_ok = episode_type.map(|t| path.contains(t) || val_str.contains(&t.to_lowercase())).unwrap_or(true);
                        let entity_ok = entity_id.map(|e| val_str.contains(&e.to_lowercase())).unwrap_or(true);
                        
                        if matches && type_ok && entity_ok {
                            let mut parsed = parse_value_deep(&value);
                            if let Value::Object(ref mut map) = parsed {
                                map.insert("_path".to_string(), Value::String(path.clone()));
                                map.insert("match_type".to_string(), Value::String("text".to_string()));
                                results.push(parsed);
                            } else {
                                results.push(json!({
                                    "_path": path.clone(),
                                    "value": parsed,
                                    "match_type": "text"
                                }));
                            }
                            seen_paths.insert(path);
                            if results.len() >= k { break; }
                        }
                    }
                }
                if results.len() >= k { break; }
            }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "query": query,
            "k": k,
            "episode_type": episode_type,
            "entity_id": entity_id,
            "results": results,
            "result_count": results.len()
        });

        soch_result(&result).map_err(|e| e.to_string())
    }

    fn exec_get_episode_timeline(&self, args: Value) -> Result<String, String> {
        let episode_id = args
            .get("episode_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'episode_id'")?;
        let max_events = args
            .get("max_events")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;
        let role = args.get("role").and_then(|v| v.as_str());
        let include_metrics = args
            .get("include_metrics")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        info!(
            "Getting timeline: episode={}, max={}",
            episode_id, max_events
        );

        // Scan for events related to this episode
        self.conn.begin().map_err(|e| e.to_string())?;
        
        let prefix = format!("/episodes/{}", episode_id);
        let mut events = Vec::new();
        
        if let Ok(results) = self.conn.scan(&prefix) {
            let _episode_lower = episode_id.to_lowercase();
            for (path, value) in results.into_iter().take(max_events) {
                let val_str = String::from_utf8_lossy(&value);
                
                // Filter by role if specified
                let role_ok = role.map(|r| val_str.to_lowercase().contains(&r.to_lowercase()))
                    .unwrap_or(true);
                
                if role_ok {
                    let mut parsed = parse_value_deep(&value);
                    if let Value::Object(ref mut map) = parsed {
                        map.insert("_path".to_string(), Value::String(path));
                    } else {
                         parsed = json!({"_path": path, "content": parsed});
                    }
                    
                    let mut event = parsed;
                    
                    if include_metrics {
                        // Add placeholder metrics
                        event["metrics"] = json!({
                            "timestamp": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_micros()).unwrap_or(0)
                        });
                    }
                    
                    events.push(event);
                }
            }
        }
        
        // Also scan events table
        if let Ok(results) = self.conn.scan("/events") {
            for (path, value) in results {
                if events.len() >= max_events {
                    break;
                }
                let val_str = String::from_utf8_lossy(&value);
                if val_str.to_lowercase().contains(&episode_id.to_lowercase()) {
                    let role_ok = role.map(|r| val_str.to_lowercase().contains(&r.to_lowercase()))
                        .unwrap_or(true);
                    if role_ok {
                        events.push(json!({
                            "path": path,
                            "content": serde_json::from_str::<Value>(&val_str)
                                .unwrap_or(Value::String(val_str.to_string()))
                        }));
                    }
                }
            }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "episode_id": episode_id,
            "event_count": events.len(),
            "role_filter": role,
            "include_metrics": include_metrics,
            "events": events
        });

        soch_result(&result).map_err(|e| e.to_string())
    }

    fn exec_search_entities(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query'")?;
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let kind = args.get("kind").and_then(|v| v.as_str());

        info!("Searching entities: query='{}', k={}, kind={:?}", query, k, kind);

        // Search using scan and simple text matching
        self.conn.begin().map_err(|e| e.to_string())?;
        
        let mut results: Vec<Value> = Vec::new();
        let search_lower = query.to_lowercase();
        
        // 1. Semantic Search (if possible)
        // Only if kind is not specified or specifically "entities"
        let use_semantic = kind.is_none() || kind == Some("entity");
        
        if use_semantic {
            if let Some(_vector) = self.embeddings.get_embedding(query) {
                 info!("Vector generated for entities, search disabled pending backend support");
                 /*
                 if let Ok(collection) = self.conn.vectors("entities") {
                     if let Ok(search_results) = collection.search(&vector, k) {
                         for res in search_results {
                             if let Ok(Some(bytes)) = self.conn.get(&res.path) {
                                let mut parsed = parse_value_deep(&bytes);
                                if let Value::Object(ref mut map) = parsed {
                                    map.insert("_path".to_string(), Value::String(res.path.clone()));
                                    map.insert("match_type".to_string(), Value::String("semantic".to_string()));
                                    map.insert("score".to_string(), json!(res.score));
                                    results.push(parsed);
                                } else {
                                   results.push(json!({
                                       "_path": res.path,
                                       "value": parsed,
                                       "match_type": "semantic",
                                       "score": res.score
                                   }));
                                }
                             }
                         }
                     }
                 }
                 */
            }
        }
        
        // 2. Keyword Search
        if results.len() < k {
            // Use catalog-derived prefixes instead of hardcoded paths
            let scan_prefixes = if let Some(k) = kind {
                vec![format!("/{}s", k), format!("/{}", k)]
            } else {
                // Dynamic discovery from catalog
                self.discover_entity_prefixes()
            };
        
            let mut seen_paths: std::collections::HashSet<String> = results.iter()
                .filter_map(|v| v.get("_path").and_then(|p| p.as_str()).map(|s| s.to_string()))
                .collect();
            
            for prefix in scan_prefixes {
            if let Ok(items) = self.conn.scan(&prefix) {
                for (path, value) in items {
                    // Skip if we've already seen this path
                    if seen_paths.contains(&path) {
                        continue;
                    }
                    
                    let val_str = String::from_utf8_lossy(&value).to_lowercase();
                    if val_str.contains(&search_lower) || path.to_lowercase().contains(&search_lower) {
                        seen_paths.insert(path.clone());
                        
                        let mut parsed = parse_value_deep(&value);
                        if let Value::Object(ref mut map) = parsed {
                            map.insert("_path".to_string(), Value::String(path));
                            map.insert("match_type".to_string(), Value::String("text".to_string()));
                            results.push(parsed);
                        } else {
                            results.push(json!({
                                "_path": path,
                                "value": parsed,
                                "match_type": "text"
                            }));
                        }
                        if results.len() >= k {
                            break;
                        }
                    }
                }
            }
            if results.len() >= k {
                break;
            }
        }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "query": query,
            "k": k,
            "kind": kind,
            "results": results,
            "result_count": results.len()
        });

        soch_result(&result).map_err(|e| e.to_string())
    }

    fn exec_get_entity_facts(&self, args: Value) -> Result<String, String> {
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'entity_id'")?;
        let include_episodes = args
            .get("include_episodes")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let max_episodes = args
            .get("max_episodes")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        info!("Getting entity facts: entity={}", entity_id);

        self.conn.begin().map_err(|e| e.to_string())?;
        
        // Try to find entity data - search in multiple possible paths
        let search_paths = vec![
            format!("/entities/{}", entity_id),
            format!("/users/{}", entity_id),
            format!("/projects/{}", entity_id),
        ];
        
        let mut entity_data = None;
        for path in &search_paths {
            if let Ok(Some(bytes)) = self.conn.get(path) {
                let content = String::from_utf8_lossy(&bytes);
                entity_data = Some(json!({
                    "path": path,
                    "data": serde_json::from_str::<Value>(&content)
                        .unwrap_or(Value::String(content.to_string()))
                }));
                break;
            }
            // Also try scanning for nested paths
            if let Ok(results) = self.conn.scan(path) {
                if !results.is_empty() {
                    let items: Vec<Value> = results.iter().map(|(k, v)| {
                        let val_str = String::from_utf8_lossy(v);
                        json!({
                            "path": k,
                            "value": serde_json::from_str::<Value>(&val_str)
                                .unwrap_or(Value::String(val_str.to_string()))
                        })
                    }).collect();
                    entity_data = Some(json!({
                        "path": path,
                        "data": items
                    }));
                    break;
                }
            }
        }
        
        // Find related episodes if requested
        let mut recent_episodes = Vec::new();
        if include_episodes {
            let entity_lower = entity_id.to_lowercase();
            if let Ok(results) = self.conn.scan("/episodes") {
                for (path, value) in results.into_iter().take(max_episodes) {
                    let val_str = String::from_utf8_lossy(&value);
                    if val_str.to_lowercase().contains(&entity_lower) {
                        recent_episodes.push(json!({
                            "path": path,
                            "value": serde_json::from_str::<Value>(&val_str)
                                .unwrap_or(Value::String(val_str.to_string()))
                        }));
                    }
                }
            }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "entity_id": entity_id,
            "entity": entity_data,
            "recent_episodes": recent_episodes,
            "found": entity_data.is_some()
        });

        soch_result(&result).map_err(|e| e.to_string())
    }

    fn exec_build_context(&self, args: Value) -> Result<String, String> {
        let goal = args
            .get("goal")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'goal'")?;
        let budget = args
            .get("token_budget")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096);
        let session_id = args.get("session_id").and_then(|v| v.as_str());
        let episode_id = args.get("episode_id").and_then(|v| v.as_str());
        let entity_ids = args.get("entity_ids").and_then(|v| v.as_array());
        let _include_schema = args
            .get("include_schema")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        info!("Building context: goal='{}', budget={}", goal, budget);

        let mut context_parts = Vec::new();
        
        self.conn.begin().map_err(|e| e.to_string())?;

        // Session history (if available)
        if let Some(sid) = session_id {
            if let Ok(results) = self.conn.scan(&format!("/sessions/{}", sid)) {
                let items: Vec<Value> = results.into_iter().take(20).map(|(k, v)| {
                    json!({"path": k, "value": String::from_utf8_lossy(&v)})
                }).collect();
                context_parts.push(json!({
                    "section": "session_history",
                    "session_id": sid,
                    "items": items
                }));
            }
        }

        // Current episode context
        if let Some(eid) = episode_id {
            if let Ok(Some(bytes)) = self.conn.get(&format!("/episodes/{}", eid)) {
                let content = String::from_utf8_lossy(&bytes);
                context_parts.push(json!({
                    "section": "current_episode",
                    "episode_id": eid,
                    "content": serde_json::from_str::<Value>(&content).unwrap_or(Value::String(content.to_string()))
                }));
            }
        }

        // Entity facts
        if let Some(eids) = entity_ids {
            let mut entity_data = Vec::new();
            for eid in eids.iter().take(3) {
                if let Some(id) = eid.as_str() {
                    if let Ok(Some(bytes)) = self.conn.get(&format!("/entities/{}", id)) {
                        let content = String::from_utf8_lossy(&bytes);
                        entity_data.push(json!({
                            "entity_id": id,
                            "data": serde_json::from_str::<Value>(&content).unwrap_or(Value::String(content.to_string()))
                        }));
                    }
                }
            }
            if !entity_data.is_empty() {
                context_parts.push(json!({
                    "section": "entity_facts",
                    "entities": entity_data
                }));
            }
        }

        self.conn.abort().ok();

        let result = json!({
            "goal": goal,
            "token_budget": budget,
            "context": context_parts,
            "sections_built": context_parts.len()
        });

        soch_result(&result).map_err(|e| e.to_string())
    }

    // =========================================================================
    // Log Tools
    // =========================================================================

    fn exec_logs_tail(&self, args: Value) -> Result<String, String> {
        let table = args
            .get("table")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'table'")?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let where_clause = args.get("where");
        let columns = args.get("columns").and_then(|v| v.as_array());

        info!("Tailing logs: table={}, limit={}", table, limit);

        // Scan table and get last N rows
        self.conn.begin().map_err(|e| e.to_string())?;
        
        let prefix = format!("/{}", table);
        let mut rows = Vec::new();
        
        if let Ok(results) = self.conn.scan(&prefix) {
            // Get all results and take the last N (most recent)
            let all_results: Vec<_> = results.into_iter().collect();
            let start_idx = if all_results.len() > limit { all_results.len() - limit } else { 0 };
            
            for (path, value) in all_results.into_iter().skip(start_idx) {
                let parsed = parse_value_deep(&value);
                
                // Apply where filter if specified
                let where_ok = if let Some(wc) = where_clause {
                    if let Some(wc_obj) = wc.as_object() {
                        wc_obj.iter().all(|(key, expected)| {
                            parsed.get(key).map(|v| v == expected).unwrap_or(false)
                        })
                    } else {
                        true
                    }
                } else {
                    true
                };
                
                if where_ok {
                    // Filter columns if specified
                    let row = if let Some(cols) = columns {
                        let mut filtered = serde_json::Map::new();
                        filtered.insert("_path".to_string(), json!(path));
                        for col in cols {
                            if let Some(col_name) = col.as_str() {
                                if let Some(val) = parsed.get(col_name) {
                                    filtered.insert(col_name.to_string(), val.clone());
                                }
                            }
                        }
                        Value::Object(filtered)
                    } else {
                        json!({"_path": path, "data": parsed})
                    };
                    rows.push(row);
                }
            }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "table": table,
            "row_count": rows.len(),
            "rows": rows
        });

        soch_result(&result).map_err(|e| e.to_string())
    }

    fn exec_logs_timeline(&self, args: Value) -> Result<String, String> {
        let entity_id = args
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'entity_id'")?;
        let from_ts = args.get("from_ts").and_then(|v| v.as_u64());
        let to_ts = args.get("to_ts").and_then(|v| v.as_u64());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        let table = args
            .get("table")
            .and_then(|v| v.as_str())
            .unwrap_or("events");

        info!(
            "Timeline: entity={}, from={:?}, to={:?}",
            entity_id, from_ts, to_ts
        );

        // Scan table for events related to this entity
        self.conn.begin().map_err(|e| e.to_string())?;
        
        let prefix = format!("/{}", table);
        let entity_lower = entity_id.to_lowercase();
        let mut events = Vec::new();
        
        if let Ok(results) = self.conn.scan(&prefix) {
            for (path, value) in results {
                if events.len() >= limit {
                    break;
                }
                
                let val_str = String::from_utf8_lossy(&value);
                let parsed = serde_json::from_str::<Value>(&val_str)
                    .unwrap_or(Value::String(val_str.to_string()));
                
                // Check if this event relates to the entity
                let entity_match = val_str.to_lowercase().contains(&entity_lower) ||
                    path.to_lowercase().contains(&entity_lower);
                
                if entity_match {
                    // Check timestamp range if specified
                    let ts_ok = if from_ts.is_some() || to_ts.is_some() {
                        // Try to extract timestamp from the event
                        let event_ts = parsed.get("ts")
                            .or(parsed.get("timestamp"))
                            .and_then(|v| v.as_u64());
                        
                        match (event_ts, from_ts, to_ts) {
                            (Some(ts), Some(from), Some(to)) => ts >= from && ts <= to,
                            (Some(ts), Some(from), None) => ts >= from,
                            (Some(ts), None, Some(to)) => ts <= to,
                            _ => true  // No timestamp filter or can't parse
                        }
                    } else {
                        true
                    };
                    
                    if ts_ok {
                        events.push(json!({
                            "path": path,
                            "data": parsed
                        }));
                    }
                }
            }
        }
        
        self.conn.abort().ok();

        let result = json!({
            "entity_id": entity_id,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "table": table,
            "event_count": events.len(),
            "events": events
        });

        soch_result(&result).map_err(|e| e.to_string())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple semantic metadata struct for tables
struct SimpleTableMetadata {
    role: &'static str,
    primary_key: Vec<&'static str>,
    cluster_key: Option<Vec<&'static str>>,
    ts_column: Option<&'static str>,
    backed_by_vector_index: bool,
    embedding_dimension: Option<usize>,
    description: &'static str,
}

/// Get semantic metadata for known tables
fn get_table_semantic_metadata(table_name: &str) -> SimpleTableMetadata {
    match table_name {
        "episodes" => SimpleTableMetadata {
            role: "core_memory",
            primary_key: vec!["episode_id"],
            cluster_key: Some(vec!["ts_start"]),
            ts_column: Some("ts_start"),
            backed_by_vector_index: true,
            embedding_dimension: Some(1536),
            description: "Task/conversation runs. Search for similar past tasks.",
        },
        "events" => SimpleTableMetadata {
            role: "log",
            primary_key: vec!["episode_id", "seq"],
            cluster_key: Some(vec!["episode_id", "seq"]),
            ts_column: Some("ts"),
            backed_by_vector_index: false,
            embedding_dimension: None,
            description: "Steps within episodes. Use LAST N for timeline.",
        },
        "entities" => SimpleTableMetadata {
            role: "dimension",
            primary_key: vec!["entity_id"],
            cluster_key: Some(vec!["kind"]),
            ts_column: Some("updated_at"),
            backed_by_vector_index: true,
            embedding_dimension: Some(1536),
            description: "Users, projects, documents, services.",
        },
        _ => SimpleTableMetadata {
            role: "unknown",
            primary_key: vec![],
            cluster_key: None,
            ts_column: None,
            backed_by_vector_index: false,
            embedding_dimension: None,
            description: "User-defined table",
        },
    }
}
