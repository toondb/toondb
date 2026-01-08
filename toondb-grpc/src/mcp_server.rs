// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! MCP Service gRPC Implementation
//!
//! Provides Model Context Protocol tool routing via gRPC.

use crate::proto::{
    mcp_service_server::{McpService, McpServiceServer},
    ExecuteToolRequest, ExecuteToolResponse, GetToolSchemaRequest, GetToolSchemaResponse,
    ListToolsRequest, ListToolsResponse, McpTool, RegisterToolRequest, RegisterToolResponse,
    UnregisterToolRequest, UnregisterToolResponse,
};
use dashmap::DashMap;
use std::time::Instant;
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Registered tool with handler
struct RegisteredTool {
    tool: McpTool,
    handler_endpoint: String,
}

/// MCP gRPC Server
pub struct McpServer {
    tools: DashMap<String, RegisteredTool>,
}

impl McpServer {
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
        }
    }

    pub fn into_service(self) -> McpServiceServer<Self> {
        McpServiceServer::new(self)
    }

    fn execute_built_in(&self, tool_name: &str, input: &str) -> Option<String> {
        match tool_name {
            "echo" => Some(input.to_string()),
            "uppercase" => Some(input.to_uppercase()),
            "lowercase" => Some(input.to_lowercase()),
            "reverse" => Some(input.chars().rev().collect()),
            "length" => Some(input.len().to_string()),
            _ => None,
        }
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl McpService for McpServer {
    async fn register_tool(
        &self,
        request: Request<RegisterToolRequest>,
    ) -> Result<Response<RegisterToolResponse>, Status> {
        let req = request.into_inner();

        if let Some(tool) = req.tool {
            let tool_id = if tool.name.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                tool.name.clone()
            };

            self.tools.insert(
                tool_id.clone(),
                RegisteredTool {
                    tool,
                    handler_endpoint: req.handler_endpoint,
                },
            );

            Ok(Response::new(RegisterToolResponse {
                success: true,
                tool_id,
                error: String::new(),
            }))
        } else {
            Ok(Response::new(RegisterToolResponse {
                success: false,
                tool_id: String::new(),
                error: "Tool is required".to_string(),
            }))
        }
    }

    async fn execute_tool(
        &self,
        request: Request<ExecuteToolRequest>,
    ) -> Result<Response<ExecuteToolResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        // Try built-in tools first
        if let Some(output) = self.execute_built_in(&req.tool_name, &req.input) {
            return Ok(Response::new(ExecuteToolResponse {
                success: true,
                output,
                error: String::new(),
                duration_us: start.elapsed().as_micros() as u64,
            }));
        }

        // Look up registered tool
        match self.tools.get(&req.tool_name) {
            Some(registered) => {
                // In a real implementation, this would call the handler endpoint
                // For now, we simulate execution
                let output = serde_json::json!({
                    "tool": req.tool_name,
                    "input": req.input,
                    "context": req.context,
                    "result": "Tool executed successfully",
                    "handler": registered.handler_endpoint
                })
                .to_string();

                Ok(Response::new(ExecuteToolResponse {
                    success: true,
                    output,
                    error: String::new(),
                    duration_us: start.elapsed().as_micros() as u64,
                }))
            }
            None => Ok(Response::new(ExecuteToolResponse {
                success: false,
                output: String::new(),
                error: format!("Tool '{}' not found", req.tool_name),
                duration_us: start.elapsed().as_micros() as u64,
            })),
        }
    }

    async fn list_tools(
        &self,
        request: Request<ListToolsRequest>,
    ) -> Result<Response<ListToolsResponse>, Status> {
        let req = request.into_inner();

        let tools: Vec<McpTool> = self
            .tools
            .iter()
            .filter(|entry| {
                if req.tags.is_empty() {
                    return true;
                }
                // Filter by tags
                let tool_tags = &entry.value().tool.tags;
                req.tags.iter().any(|t| tool_tags.contains(t))
            })
            .map(|entry| entry.value().tool.clone())
            .collect();

        Ok(Response::new(ListToolsResponse { tools }))
    }

    async fn unregister_tool(
        &self,
        request: Request<UnregisterToolRequest>,
    ) -> Result<Response<UnregisterToolResponse>, Status> {
        let req = request.into_inner();

        match self.tools.remove(&req.tool_name) {
            Some(_) => Ok(Response::new(UnregisterToolResponse {
                success: true,
                error: String::new(),
            })),
            None => Ok(Response::new(UnregisterToolResponse {
                success: false,
                error: format!("Tool '{}' not found", req.tool_name),
            })),
        }
    }

    async fn get_tool_schema(
        &self,
        request: Request<GetToolSchemaRequest>,
    ) -> Result<Response<GetToolSchemaResponse>, Status> {
        let req = request.into_inner();

        match self.tools.get(&req.tool_name) {
            Some(registered) => Ok(Response::new(GetToolSchemaResponse {
                tool: Some(registered.tool.clone()),
                error: String::new(),
            })),
            None => Ok(Response::new(GetToolSchemaResponse {
                tool: None,
                error: format!("Tool '{}' not found", req.tool_name),
            })),
        }
    }
}
