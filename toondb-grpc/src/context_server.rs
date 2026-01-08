// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Context Service gRPC Implementation
//!
//! Provides LLM context assembly with token budgets via gRPC.

use crate::proto::{
    context_service_server::{ContextService, ContextServiceServer},
    ContextQueryRequest, ContextQueryResponse, ContextSectionType, EstimateTokensRequest,
    EstimateTokensResponse, FormatContextRequest, FormatContextResponse, OutputFormat,
    SectionResult,
};
use tonic::{Request, Response, Status};

/// Simple token estimator (approximately 4 chars per token)
fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}

/// Context gRPC Server
pub struct ContextServer;

impl ContextServer {
    pub fn new() -> Self {
        Self
    }

    pub fn into_service(self) -> ContextServiceServer<Self> {
        ContextServiceServer::new(self)
    }
}

impl Default for ContextServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl ContextService for ContextServer {
    async fn query(
        &self,
        request: Request<ContextQueryRequest>,
    ) -> Result<Response<ContextQueryResponse>, Status> {
        let req = request.into_inner();
        let token_limit = req.token_limit as usize;

        let mut section_results = Vec::new();
        let mut total_tokens = 0u32;
        let mut context_parts = Vec::new();

        // Sort sections by priority (lower = higher priority)
        let mut sections = req.sections;
        sections.sort_by_key(|s| s.priority);

        for section in sections {
            let remaining_budget = token_limit.saturating_sub(total_tokens as usize);
            if remaining_budget == 0 {
                break;
            }

            // Generate section content based on type
            let content = match section.section_type {
                x if x == ContextSectionType::ContextSectionGet as i32 => {
                    format!("# {}\n[Data from: {}]\n", section.name, section.query)
                }
                x if x == ContextSectionType::ContextSectionLast as i32 => {
                    format!("# {} (Recent)\n[Last entries from: {}]\n", section.name, section.query)
                }
                x if x == ContextSectionType::ContextSectionSearch as i32 => {
                    format!("# {} (Search Results)\n[Search: {}]\n", section.name, section.query)
                }
                x if x == ContextSectionType::ContextSectionSelect as i32 => {
                    format!("# {} (Query)\n[SQL: {}]\n", section.name, section.query)
                }
                _ => format!("# {}\n", section.name),
            };

            let section_tokens = estimate_tokens(&content);
            let truncated = section_tokens as usize > remaining_budget;

            let final_content = if truncated {
                // Truncate to fit budget
                let char_limit = remaining_budget * 4;
                content.chars().take(char_limit).collect::<String>()
            } else {
                content
            };

            let tokens_used = estimate_tokens(&final_content);
            total_tokens += tokens_used;

            section_results.push(SectionResult {
                name: section.name,
                tokens_used,
                truncated,
                content: final_content.clone(),
            });

            context_parts.push(final_content);
        }

        // Format output
        let context = match req.format {
            x if x == OutputFormat::Json as i32 => {
                serde_json::json!({
                    "session_id": req.session_id,
                    "sections": context_parts,
                    "total_tokens": total_tokens
                })
                .to_string()
            }
            x if x == OutputFormat::Markdown as i32 => {
                context_parts.join("\n---\n")
            }
            x if x == OutputFormat::Text as i32 => {
                context_parts.join("\n\n")
            }
            _ => {
                // TOON format (default)
                format!(
                    "<context session=\"{}\">\n{}\n</context>",
                    req.session_id,
                    context_parts.join("\n")
                )
            }
        };

        Ok(Response::new(ContextQueryResponse {
            context,
            total_tokens,
            section_results,
            error: String::new(),
        }))
    }

    async fn estimate_tokens(
        &self,
        request: Request<EstimateTokensRequest>,
    ) -> Result<Response<EstimateTokensResponse>, Status> {
        let req = request.into_inner();
        let token_count = estimate_tokens(&req.content);

        Ok(Response::new(EstimateTokensResponse { token_count }))
    }

    async fn format_context(
        &self,
        request: Request<FormatContextRequest>,
    ) -> Result<Response<FormatContextResponse>, Status> {
        let req = request.into_inner();

        let formatted = match req.format {
            x if x == OutputFormat::Json as i32 => {
                serde_json::json!({ "content": req.content }).to_string()
            }
            x if x == OutputFormat::Markdown as i32 => {
                format!("```\n{}\n```", req.content)
            }
            x if x == OutputFormat::Text as i32 => req.content,
            _ => format!("<toon>{}</toon>", req.content),
        };

        Ok(Response::new(FormatContextResponse { formatted }))
    }
}
