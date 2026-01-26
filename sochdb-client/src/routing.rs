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

//! Tool Routing Primitive for Multi-Agent Scenarios
//!
//! Provides a first-class system for routing tool calls to agents based on:
//! - Agent capabilities
//! - Tool requirements
//! - Load balancing
//! - Agent availability
//!
//! # Example
//!
//! ```rust,ignore
//! use sochdb_client::routing::{
//!     AgentRegistry, ToolRouter, ToolDispatcher,
//!     Tool, ToolCategory, RoutingStrategy,
//! };
//!
//! let conn = Connection::open("./data")?;
//! let dispatcher = ToolDispatcher::new(conn);
//!
//! // Register a local agent
//! dispatcher.registry().register_agent(
//!     "code_agent",
//!     vec![ToolCategory::Code],
//!     AgentConfig::with_handler(|tool, args| {
//!         Ok(json!({"result": format!("Processed {}", tool)}))
//!     }),
//! );
//!
//! // Register a tool
//! dispatcher.router().register_tool(Tool {
//!     name: "search_code".to_string(),
//!     description: "Search codebase".to_string(),
//!     category: ToolCategory::Code,
//!     ..Default::default()
//! });
//!
//! // Invoke with automatic routing
//! let result = dispatcher.invoke("search_code", json!({"query": "auth"}))?;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::ConnectionTrait;
use crate::error::{ClientError, Result};

/// Standard tool categories for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCategory {
    Code,
    Search,
    Database,
    Web,
    File,
    Git,
    Shell,
    Email,
    Calendar,
    Memory,
    Vector,
    Graph,
    Custom,
}

/// How to select among multiple capable agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingStrategy {
    RoundRobin,
    Random,
    LeastLoaded,
    Sticky,
    #[default]
    Priority,
    Fastest,
}

/// Agent availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentStatus {
    #[default]
    Available,
    Busy,
    Offline,
    Degraded,
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    #[serde(default)]
    pub schema: Value,
    #[serde(default)]
    pub required_capabilities: Vec<ToolCategory>,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: f64,
    #[serde(default = "default_retries")]
    pub retries: u32,
    #[serde(default)]
    pub metadata: Value,
}

fn default_timeout() -> f64 {
    30.0
}

fn default_retries() -> u32 {
    1
}

impl Default for Tool {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            category: ToolCategory::Custom,
            schema: Value::Null,
            required_capabilities: Vec::new(),
            timeout_seconds: 30.0,
            retries: 1,
            metadata: Value::Null,
        }
    }
}

/// Tool handler function type.
pub type ToolHandler = Arc<dyn Fn(&str, &Value) -> std::result::Result<Value, String> + Send + Sync>;

/// Agent definition.
pub struct Agent {
    pub agent_id: String,
    pub capabilities: Vec<ToolCategory>,
    pub endpoint: Option<String>,
    pub handler: Option<ToolHandler>,
    pub priority: i32,
    pub max_concurrent: u32,
    pub metadata: Value,

    // Runtime state
    pub status: AgentStatus,
    pub current_load: Mutex<u32>,
    pub total_calls: Mutex<u64>,
    pub total_latency: Mutex<Duration>,
    pub last_success: Mutex<Option<Instant>>,
    pub last_failure: Mutex<Option<Instant>>,
}

impl Agent {
    fn new(agent_id: &str, capabilities: Vec<ToolCategory>) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            capabilities,
            endpoint: None,
            handler: None,
            priority: 100,
            max_concurrent: 10,
            metadata: Value::Null,
            status: AgentStatus::Available,
            current_load: Mutex::new(0),
            total_calls: Mutex::new(0),
            total_latency: Mutex::new(Duration::ZERO),
            last_success: Mutex::new(None),
            last_failure: Mutex::new(None),
        }
    }
}

/// Configuration for agent registration.
pub struct AgentConfig {
    pub endpoint: Option<String>,
    pub handler: Option<ToolHandler>,
    pub priority: i32,
    pub max_concurrent: u32,
    pub metadata: Value,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            handler: None,
            priority: 100,
            max_concurrent: 10,
            metadata: Value::Null,
        }
    }
}

impl AgentConfig {
    /// Create config with HTTP endpoint.
    pub fn with_endpoint(endpoint: &str) -> Self {
        Self {
            endpoint: Some(endpoint.to_string()),
            ..Default::default()
        }
    }

    /// Create config with local handler.
    pub fn with_handler<F>(handler: F) -> Self
    where
        F: Fn(&str, &Value) -> std::result::Result<Value, String> + Send + Sync + 'static,
    {
        Self {
            handler: Some(Arc::new(handler)),
            ..Default::default()
        }
    }

    /// Set priority.
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Result of a tool routing decision.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub agent_id: String,
    pub tool_name: String,
    pub result: Value,
    pub latency_ms: f64,
    pub success: bool,
    pub error: Option<String>,
    pub retries_used: u32,
}

/// Context for routing decisions.
#[derive(Debug, Clone, Default)]
pub struct RoutingContext {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub priority: i32,
    pub timeout_override: Option<f64>,
    pub preferred_agent: Option<String>,
    pub excluded_agents: Vec<String>,
    pub custom: HashMap<String, Value>,
}

impl RoutingContext {
    /// Create a new routing context.
    pub fn new() -> Self {
        Self {
            priority: 100,
            ..Default::default()
        }
    }

    /// Set session ID for sticky routing.
    pub fn with_session_id(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    /// Set preferred agent.
    pub fn with_preferred_agent(mut self, agent_id: &str) -> Self {
        self.preferred_agent = Some(agent_id.to_string());
        self
    }

    /// Add excluded agent.
    pub fn exclude_agent(mut self, agent_id: &str) -> Self {
        self.excluded_agents.push(agent_id.to_string());
        self
    }
}

const AGENT_PREFIX: &str = "/_routing/agents/";
const TOOL_PREFIX: &str = "/_routing/tools/";

/// Registry of agents and their capabilities.
pub struct AgentRegistry<C: ConnectionTrait> {
    conn: Arc<C>,
    agents: RwLock<HashMap<String, Arc<Agent>>>,
}

impl<C: ConnectionTrait> AgentRegistry<C> {
    /// Create a new agent registry.
    pub fn new(conn: Arc<C>) -> Self {
        let registry = Self {
            conn,
            agents: RwLock::new(HashMap::new()),
        };
        registry.load_agents();
        registry
    }

    fn load_agents(&self) {
        if let Ok(results) = self.conn.scan(AGENT_PREFIX.as_bytes()) {
            for (_, value) in results {
                if let Ok(data) = serde_json::from_slice::<Value>(&value) {
                    if let (Some(agent_id), Some(caps)) = (
                        data["agent_id"].as_str(),
                        data["capabilities"].as_array(),
                    ) {
                        let capabilities: Vec<ToolCategory> = caps
                            .iter()
                            .filter_map(|c| serde_json::from_value(c.clone()).ok())
                            .collect();

                        let mut agent = Agent::new(agent_id, capabilities);
                        if let Some(ep) = data["endpoint"].as_str() {
                            agent.endpoint = Some(ep.to_string());
                        }
                        if let Some(p) = data["priority"].as_i64() {
                            agent.priority = p as i32;
                        }
                        if let Some(mc) = data["max_concurrent"].as_u64() {
                            agent.max_concurrent = mc as u32;
                        }

                        let mut agents = self.agents.write().unwrap();
                        agents.insert(agent_id.to_string(), Arc::new(agent));
                    }
                }
            }
        }
    }

    /// Register an agent with capabilities.
    pub fn register_agent(
        &self,
        agent_id: &str,
        capabilities: Vec<ToolCategory>,
        config: AgentConfig,
    ) -> Arc<Agent> {
        let mut agent = Agent::new(agent_id, capabilities.clone());
        agent.endpoint = config.endpoint.clone();
        agent.handler = config.handler;
        agent.priority = config.priority;
        agent.max_concurrent = config.max_concurrent;
        agent.metadata = config.metadata.clone();

        let agent = Arc::new(agent);

        {
            let mut agents = self.agents.write().unwrap();
            agents.insert(agent_id.to_string(), Arc::clone(&agent));
        }

        // Persist to database
        let data = json!({
            "agent_id": agent_id,
            "capabilities": capabilities,
            "endpoint": config.endpoint,
            "priority": config.priority,
            "max_concurrent": config.max_concurrent,
            "metadata": config.metadata,
        });
        let key = format!("{}{}", AGENT_PREFIX, agent_id);
        let _ = self.conn.put(key.as_bytes(), data.to_string().as_bytes());

        agent
    }

    /// Remove an agent registration.
    pub fn unregister_agent(&self, agent_id: &str) -> bool {
        let mut agents = self.agents.write().unwrap();
        if agents.remove(agent_id).is_some() {
            let key = format!("{}{}", AGENT_PREFIX, agent_id);
            let _ = self.conn.delete(key.as_bytes());
            true
        } else {
            false
        }
    }

    /// Get an agent by ID.
    pub fn get_agent(&self, agent_id: &str) -> Option<Arc<Agent>> {
        let agents = self.agents.read().unwrap();
        agents.get(agent_id).cloned()
    }

    /// List all registered agents.
    pub fn list_agents(&self) -> Vec<Arc<Agent>> {
        let agents = self.agents.read().unwrap();
        agents.values().cloned().collect()
    }

    /// Find agents capable of handling the required categories.
    pub fn find_capable_agents(
        &self,
        required: &[ToolCategory],
        exclude: &[String],
    ) -> Vec<Arc<Agent>> {
        let agents = self.agents.read().unwrap();
        let exclude_set: std::collections::HashSet<_> = exclude.iter().collect();

        agents
            .values()
            .filter(|agent| {
                if exclude_set.contains(&agent.agent_id) {
                    return false;
                }
                if agent.status == AgentStatus::Offline {
                    return false;
                }
                let agent_caps: std::collections::HashSet<_> = agent.capabilities.iter().collect();
                required.iter().all(|req| agent_caps.contains(req))
            })
            .cloned()
            .collect()
    }

    /// Update an agent's status.
    pub fn update_agent_status(&self, agent_id: &str, status: AgentStatus) {
        let agents = self.agents.read().unwrap();
        if let Some(agent) = agents.get(agent_id) {
            // Note: status is not mutable through Arc, would need interior mutability
            // For now, this is a placeholder - in production, use Arc<RwLock<Agent>>
            let _ = (agent, status);
        }
    }

    /// Record a tool call result for an agent.
    pub fn record_call(&self, agent_id: &str, latency: Duration, success: bool) {
        let agents = self.agents.read().unwrap();
        if let Some(agent) = agents.get(agent_id) {
            *agent.total_calls.lock().unwrap() += 1;
            *agent.total_latency.lock().unwrap() += latency;
            if success {
                *agent.last_success.lock().unwrap() = Some(Instant::now());
            } else {
                *agent.last_failure.lock().unwrap() = Some(Instant::now());
            }
        }
    }
}

/// Routes tool calls to appropriate agents.
pub struct ToolRouter<C: ConnectionTrait> {
    registry: Arc<AgentRegistry<C>>,
    conn: Arc<C>,
    default_strategy: RoutingStrategy,
    tools: RwLock<HashMap<String, Tool>>,
    round_robin_idx: Mutex<HashMap<String, usize>>,
    session_affinity: RwLock<HashMap<String, String>>,
}

impl<C: ConnectionTrait> ToolRouter<C> {
    /// Create a new tool router.
    pub fn new(registry: Arc<AgentRegistry<C>>, conn: Arc<C>) -> Self {
        let router = Self {
            registry,
            conn,
            default_strategy: RoutingStrategy::Priority,
            tools: RwLock::new(HashMap::new()),
            round_robin_idx: Mutex::new(HashMap::new()),
            session_affinity: RwLock::new(HashMap::new()),
        };
        router.load_tools();
        router
    }

    /// Set the default routing strategy.
    pub fn with_default_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.default_strategy = strategy;
        self
    }

    fn load_tools(&self) {
        if let Ok(results) = self.conn.scan(TOOL_PREFIX.as_bytes()) {
            for (_, value) in results {
                if let Ok(tool) = serde_json::from_slice::<Tool>(&value) {
                    let mut tools = self.tools.write().unwrap();
                    tools.insert(tool.name.clone(), tool);
                }
            }
        }
    }

    /// Register a tool for routing.
    pub fn register_tool(&self, tool: Tool) -> Tool {
        let mut tools = self.tools.write().unwrap();
        tools.insert(tool.name.clone(), tool.clone());

        // Persist to database
        let key = format!("{}{}", TOOL_PREFIX, tool.name);
        if let Ok(data) = serde_json::to_vec(&tool) {
            let _ = self.conn.put(key.as_bytes(), &data);
        }

        tool
    }

    /// Remove a tool registration.
    pub fn unregister_tool(&self, name: &str) -> bool {
        let mut tools = self.tools.write().unwrap();
        if tools.remove(name).is_some() {
            let key = format!("{}{}", TOOL_PREFIX, name);
            let _ = self.conn.delete(key.as_bytes());
            true
        } else {
            false
        }
    }

    /// Get a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<Tool> {
        let tools = self.tools.read().unwrap();
        tools.get(name).cloned()
    }

    /// List all registered tools.
    pub fn list_tools(&self) -> Vec<Tool> {
        let tools = self.tools.read().unwrap();
        tools.values().cloned().collect()
    }

    /// Route a tool call to the best agent.
    pub fn route(
        &self,
        tool_name: &str,
        args: Value,
        context: Option<RoutingContext>,
        strategy: Option<RoutingStrategy>,
    ) -> RouteResult {
        let ctx = context.unwrap_or_default();

        let tool = {
            let tools = self.tools.read().unwrap();
            tools.get(tool_name).cloned()
        };

        let tool = match tool {
            Some(t) => t,
            None => {
                return RouteResult {
                    agent_id: String::new(),
                    tool_name: tool_name.to_string(),
                    result: Value::Null,
                    latency_ms: 0.0,
                    success: false,
                    error: Some(format!("Unknown tool: {}", tool_name)),
                    retries_used: 0,
                };
            }
        };

        // Determine required capabilities
        let required = if tool.required_capabilities.is_empty() {
            vec![tool.category]
        } else {
            tool.required_capabilities.clone()
        };

        // Find capable agents
        let mut capable = self.registry.find_capable_agents(&required, &ctx.excluded_agents);
        if capable.is_empty() {
            return RouteResult {
                agent_id: String::new(),
                tool_name: tool_name.to_string(),
                result: Value::Null,
                latency_ms: 0.0,
                success: false,
                error: Some(format!("No capable agents for tool '{}'", tool_name)),
                retries_used: 0,
            };
        }

        // Select agent using strategy
        let use_strategy = strategy.unwrap_or(self.default_strategy);
        let mut agent = self.select_agent(&capable, use_strategy, &ctx);

        // Execute with retries
        let timeout = ctx.timeout_override.unwrap_or(tool.timeout_seconds);
        let retries = tool.retries;
        let mut last_error = None;

        for attempt in 0..=retries {
            let start = Instant::now();
            match self.invoke_agent(&agent, &tool, &args, timeout) {
                Ok(result) => {
                    let latency = start.elapsed();
                    self.registry.record_call(&agent.agent_id, latency, true);

                    // Update session affinity
                    if let Some(ref session_id) = ctx.session_id {
                        let mut affinity = self.session_affinity.write().unwrap();
                        affinity.insert(session_id.clone(), agent.agent_id.clone());
                    }

                    return RouteResult {
                        agent_id: agent.agent_id.clone(),
                        tool_name: tool_name.to_string(),
                        result,
                        latency_ms: latency.as_secs_f64() * 1000.0,
                        success: true,
                        error: None,
                        retries_used: attempt,
                    };
                }
                Err(e) => {
                    let latency = start.elapsed();
                    self.registry.record_call(&agent.agent_id, latency, false);
                    last_error = Some(e);

                    // Try next capable agent on failure
                    capable.retain(|a| a.agent_id != agent.agent_id);
                    if !capable.is_empty() {
                        agent = self.select_agent(&capable, use_strategy, &ctx);
                    }
                }
            }
        }

        RouteResult {
            agent_id: agent.agent_id.clone(),
            tool_name: tool_name.to_string(),
            result: Value::Null,
            latency_ms: 0.0,
            success: false,
            error: last_error.or(Some("All retries exhausted".to_string())),
            retries_used: retries,
        }
    }

    fn select_agent(
        &self,
        capable: &[Arc<Agent>],
        strategy: RoutingStrategy,
        ctx: &RoutingContext,
    ) -> Arc<Agent> {
        if capable.is_empty() {
            panic!("No capable agents");
        }

        // Preferred agent override
        if let Some(ref preferred) = ctx.preferred_agent {
            if let Some(agent) = capable.iter().find(|a| &a.agent_id == preferred) {
                return Arc::clone(agent);
            }
        }

        // Session affinity (sticky routing)
        if strategy == RoutingStrategy::Sticky {
            if let Some(ref session_id) = ctx.session_id {
                let affinity = self.session_affinity.read().unwrap();
                if let Some(prev_agent) = affinity.get(session_id) {
                    if let Some(agent) = capable.iter().find(|a| &a.agent_id == prev_agent) {
                        return Arc::clone(agent);
                    }
                }
            }
        }

        match strategy {
            RoutingStrategy::RoundRobin => {
                let mut idx_map = self.round_robin_idx.lock().unwrap();
                let key: String = capable.iter().map(|a| &a.agent_id).cloned().collect();
                let idx = *idx_map.get(&key).unwrap_or(&0) % capable.len();
                idx_map.insert(key, idx + 1);
                Arc::clone(&capable[idx])
            }
            RoutingStrategy::Random => {
                use std::collections::hash_map::RandomState;
                use std::hash::{BuildHasher, Hasher};
                let hasher = RandomState::new().build_hasher();
                let idx = hasher.finish() as usize % capable.len();
                Arc::clone(&capable[idx])
            }
            RoutingStrategy::LeastLoaded => {
                capable
                    .iter()
                    .min_by_key(|a| *a.current_load.lock().unwrap())
                    .map(Arc::clone)
                    .unwrap()
            }
            RoutingStrategy::Priority => {
                capable
                    .iter()
                    .max_by(|a, b| {
                        let pa = a.priority;
                        let pb = b.priority;
                        let la = *a.current_load.lock().unwrap();
                        let lb = *b.current_load.lock().unwrap();
                        pa.cmp(&pb).then(lb.cmp(&la))
                    })
                    .map(Arc::clone)
                    .unwrap()
            }
            RoutingStrategy::Fastest => {
                capable
                    .iter()
                    .min_by(|a, b| {
                        let ca = *a.total_calls.lock().unwrap();
                        let cb = *b.total_calls.lock().unwrap();
                        let la = *a.total_latency.lock().unwrap();
                        let lb = *b.total_latency.lock().unwrap();
                        let avg_a = if ca > 0 { la / ca as u32 } else { Duration::MAX };
                        let avg_b = if cb > 0 { lb / cb as u32 } else { Duration::MAX };
                        avg_a.cmp(&avg_b)
                    })
                    .map(Arc::clone)
                    .unwrap()
            }
            RoutingStrategy::Sticky => Arc::clone(&capable[0]),
        }
    }

    fn invoke_agent(
        &self,
        agent: &Agent,
        tool: &Tool,
        args: &Value,
        _timeout: f64,
    ) -> std::result::Result<Value, String> {
        {
            let mut load = agent.current_load.lock().unwrap();
            *load += 1;
        }

        let result = if let Some(ref handler) = agent.handler {
            handler(&tool.name, args)
        } else if let Some(ref endpoint) = agent.endpoint {
            // HTTP invocation would go here
            // For now, return error indicating remote not implemented
            Err(format!(
                "Remote invocation to {} not yet implemented in Rust SDK",
                endpoint
            ))
        } else {
            Err(format!(
                "Agent {} has no handler or endpoint",
                agent.agent_id
            ))
        };

        {
            let mut load = agent.current_load.lock().unwrap();
            *load = load.saturating_sub(1);
        }

        result
    }
}

/// High-level dispatcher for multi-agent tool orchestration.
pub struct ToolDispatcher<C: ConnectionTrait> {
    conn: Arc<C>,
    registry: Arc<AgentRegistry<C>>,
    router: Arc<ToolRouter<C>>,
}

impl<C: ConnectionTrait> ToolDispatcher<C> {
    /// Create a new tool dispatcher.
    pub fn new(conn: C) -> Self {
        let conn = Arc::new(conn);
        let registry = Arc::new(AgentRegistry::new(Arc::clone(&conn)));
        let router = Arc::new(ToolRouter::new(Arc::clone(&registry), Arc::clone(&conn)));

        Self {
            conn,
            registry,
            router,
        }
    }

    /// Get the agent registry.
    pub fn registry(&self) -> &AgentRegistry<C> {
        &self.registry
    }

    /// Get the tool router.
    pub fn router(&self) -> &ToolRouter<C> {
        &self.router
    }

    /// Register a local (in-process) agent.
    pub fn register_local_agent<F>(
        &self,
        agent_id: &str,
        capabilities: Vec<ToolCategory>,
        handler: F,
        priority: i32,
    ) -> Arc<Agent>
    where
        F: Fn(&str, &Value) -> std::result::Result<Value, String> + Send + Sync + 'static,
    {
        self.registry.register_agent(
            agent_id,
            capabilities,
            AgentConfig::with_handler(handler).priority(priority),
        )
    }

    /// Register a remote (HTTP) agent.
    pub fn register_remote_agent(
        &self,
        agent_id: &str,
        capabilities: Vec<ToolCategory>,
        endpoint: &str,
        priority: i32,
    ) -> Arc<Agent> {
        self.registry.register_agent(
            agent_id,
            capabilities,
            AgentConfig::with_endpoint(endpoint).priority(priority),
        )
    }

    /// Register a tool for routing.
    pub fn register_tool(&self, tool: Tool) -> Tool {
        self.router.register_tool(tool)
    }

    /// Invoke a tool with automatic routing.
    pub fn invoke(
        &self,
        tool_name: &str,
        args: Value,
        context: Option<RoutingContext>,
    ) -> RouteResult {
        self.router.route(tool_name, args, context, None)
    }

    /// List all registered agents with their status.
    pub fn list_agents(&self) -> Vec<Value> {
        self.registry
            .list_agents()
            .iter()
            .map(|a| {
                let total_calls = *a.total_calls.lock().unwrap();
                let total_latency = *a.total_latency.lock().unwrap();
                let avg_latency = if total_calls > 0 {
                    Some(total_latency.as_secs_f64() * 1000.0 / total_calls as f64)
                } else {
                    None
                };

                json!({
                    "agent_id": a.agent_id,
                    "capabilities": a.capabilities,
                    "status": a.status,
                    "priority": a.priority,
                    "current_load": *a.current_load.lock().unwrap(),
                    "total_calls": total_calls,
                    "avg_latency_ms": avg_latency,
                    "has_endpoint": a.endpoint.is_some(),
                    "has_handler": a.handler.is_some(),
                })
            })
            .collect()
    }

    /// List all registered tools.
    pub fn list_tools(&self) -> Vec<Value> {
        self.router
            .list_tools()
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "schema": t.schema,
                    "timeout_seconds": t.timeout_seconds,
                    "retries": t.retries,
                })
            })
            .collect()
    }
}
