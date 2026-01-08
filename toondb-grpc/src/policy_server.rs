// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Policy Service gRPC Implementation
//!
//! Provides policy evaluation and enforcement via gRPC.

use crate::proto::{
    policy_service_server::{PolicyService, PolicyServiceServer},
    DeletePolicyRequest, DeletePolicyResponse, EvaluatePolicyRequest, EvaluatePolicyResponse,
    ListPoliciesRequest, ListPoliciesResponse, PolicyActionType, PolicyRule, PolicyTrigger,
    RegisterPolicyRequest, RegisterPolicyResponse,
};
use dashmap::DashMap;
use regex::Regex;
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// Compiled policy with regex pattern
struct CompiledPolicy {
    rule: PolicyRule,
    pattern: Option<Regex>,
}

/// Policy gRPC Server
pub struct PolicyServer {
    policies: DashMap<String, Arc<CompiledPolicy>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl PolicyServer {
    pub fn new() -> Self {
        Self {
            policies: DashMap::new(),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub fn into_service(self) -> PolicyServiceServer<Self> {
        PolicyServiceServer::new(self)
    }

    fn compile_pattern(pattern: &str) -> Option<Regex> {
        // Convert glob pattern to regex
        let regex_pattern = pattern
            .replace(".", r"\.")
            .replace("*", ".*")
            .replace("?", ".");
        Regex::new(&format!("^{}$", regex_pattern)).ok()
    }
}

impl Default for PolicyServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl PolicyService for PolicyServer {
    async fn register_policy(
        &self,
        request: Request<RegisterPolicyRequest>,
    ) -> Result<Response<RegisterPolicyResponse>, Status> {
        let req = request.into_inner();

        if let Some(mut policy) = req.policy {
            // Generate ID if not provided
            if policy.id.is_empty() {
                let id = self
                    .next_id
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                policy.id = format!("policy_{}", id);
            }

            let id = policy.id.clone();
            let pattern = Self::compile_pattern(&policy.pattern);

            let compiled = Arc::new(CompiledPolicy {
                rule: policy,
                pattern,
            });

            self.policies.insert(id.clone(), compiled);

            Ok(Response::new(RegisterPolicyResponse {
                success: true,
                policy_id: id,
                error: String::new(),
            }))
        } else {
            Ok(Response::new(RegisterPolicyResponse {
                success: false,
                policy_id: String::new(),
                error: "Policy is required".to_string(),
            }))
        }
    }

    async fn evaluate(
        &self,
        request: Request<EvaluatePolicyRequest>,
    ) -> Result<Response<EvaluatePolicyResponse>, Status> {
        let req = request.into_inner();
        let key_str = String::from_utf8_lossy(&req.key);

        let mut matched_policies = Vec::new();
        let mut final_action = PolicyActionType::PolicyActionAllow;
        let mut reason = String::new();

        // Find matching policies
        for entry in self.policies.iter() {
            let compiled = entry.value();
            let rule = &compiled.rule;

            // Check trigger type matches operation
            let trigger_matches = match req.operation.as_str() {
                "read" => {
                    rule.trigger == PolicyTrigger::BeforeRead as i32
                        || rule.trigger == PolicyTrigger::AfterRead as i32
                }
                "write" => {
                    rule.trigger == PolicyTrigger::BeforeWrite as i32
                        || rule.trigger == PolicyTrigger::AfterWrite as i32
                }
                "delete" => {
                    rule.trigger == PolicyTrigger::BeforeDelete as i32
                        || rule.trigger == PolicyTrigger::AfterDelete as i32
                }
                _ => false,
            };

            if !trigger_matches {
                continue;
            }

            // Check pattern matches key
            let pattern_matches = match &compiled.pattern {
                Some(regex) => regex.is_match(&key_str),
                None => true, // No pattern means match all
            };

            if pattern_matches {
                matched_policies.push(rule.id.clone());

                // Apply policy action (deny takes precedence)
                if rule.default_action == PolicyActionType::PolicyActionDeny as i32 {
                    final_action = PolicyActionType::PolicyActionDeny;
                    reason = format!("Denied by policy: {}", rule.name);
                } else if rule.default_action == PolicyActionType::PolicyActionLog as i32
                    && final_action != PolicyActionType::PolicyActionDeny
                {
                    final_action = PolicyActionType::PolicyActionLog;
                    reason = format!("Logged by policy: {}", rule.name);
                }
            }
        }

        Ok(Response::new(EvaluatePolicyResponse {
            action: final_action.into(),
            modified_value: Vec::new(),
            reason,
            matched_policies,
        }))
    }

    async fn list_policies(
        &self,
        request: Request<ListPoliciesRequest>,
    ) -> Result<Response<ListPoliciesResponse>, Status> {
        let req = request.into_inner();

        let policies: Vec<PolicyRule> = self
            .policies
            .iter()
            .filter(|entry| {
                if req.pattern.is_empty() {
                    true
                } else {
                    entry.value().rule.pattern.contains(&req.pattern)
                }
            })
            .map(|entry| entry.value().rule.clone())
            .collect();

        Ok(Response::new(ListPoliciesResponse { policies }))
    }

    async fn delete_policy(
        &self,
        request: Request<DeletePolicyRequest>,
    ) -> Result<Response<DeletePolicyResponse>, Status> {
        let req = request.into_inner();

        match self.policies.remove(&req.policy_id) {
            Some(_) => Ok(Response::new(DeletePolicyResponse {
                success: true,
                error: String::new(),
            })),
            None => Ok(Response::new(DeletePolicyResponse {
                success: false,
                error: format!("Policy '{}' not found", req.policy_id),
            })),
        }
    }
}
