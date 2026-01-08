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

//! Policy & Safety Hooks for Agent Operations
//!
//! Provides a trigger system for enforcing policies on agent actions:
//!
//! - Pre-write validation (block dangerous operations)
//! - Post-read filtering (redact sensitive data)
//! - Rate limiting (prevent runaway agents)
//! - Audit logging (track all operations)
//!
//! # Example
//!
//! ```rust,ignore
//! use toondb_client::policy::{PolicyEngine, PolicyAction, PolicyContext};
//!
//! let db = Connection::open("./data")?;
//! let mut policy = PolicyEngine::new(db);
//!
//! // Block writes to system keys
//! policy.before_write("system/*", |ctx| {
//!     if ctx.agent_id.is_some() {
//!         PolicyAction::Deny
//!     } else {
//!         PolicyAction::Allow
//!     }
//! });
//!
//! // Redact sensitive data on read
//! policy.after_read("users/*/email", |ctx| {
//!     if ctx.get("redact_pii").is_some() {
//!         PolicyAction::Modify(b"[REDACTED]".to_vec())
//!     } else {
//!         PolicyAction::Allow
//!     }
//! });
//!
//! // Rate limit writes per agent
//! policy.add_rate_limit("write", 100, "agent_id");
//!
//! // Use policy-wrapped operations
//! policy.put(b"users/alice", b"data", Some(&ctx))?;
//! ```

use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::ConnectionTrait;
use crate::error::{ClientError, Result};

/// Action to take when a policy is triggered.
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyAction {
    /// Allow the operation.
    Allow,
    /// Block the operation.
    Deny,
    /// Allow with modifications.
    Modify(Vec<u8>),
    /// Allow but log the operation.
    Log,
}

/// When the policy is triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyTrigger {
    BeforeRead,
    AfterRead,
    BeforeWrite,
    AfterWrite,
    BeforeDelete,
    AfterDelete,
}

/// Context for policy evaluation.
#[derive(Debug, Clone)]
pub struct PolicyContext {
    pub operation: String,
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub timestamp: Instant,
    pub custom: HashMap<String, String>,
}

impl PolicyContext {
    /// Create a new policy context.
    pub fn new(operation: &str, key: &[u8]) -> Self {
        Self {
            operation: operation.to_string(),
            key: key.to_vec(),
            value: None,
            agent_id: None,
            session_id: None,
            timestamp: Instant::now(),
            custom: HashMap::new(),
        }
    }

    /// Get a custom context value.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.custom.get(key)
    }

    /// Set a custom context value.
    pub fn set(&mut self, key: &str, value: &str) {
        self.custom.insert(key.to_string(), value.to_string());
    }

    /// Set the agent ID.
    pub fn with_agent_id(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
        self
    }

    /// Set the session ID.
    pub fn with_session_id(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }
}

/// Policy handler function type.
pub type PolicyHandler = Arc<dyn Fn(&PolicyContext) -> PolicyAction + Send + Sync>;

/// Pattern-based policy.
struct PatternPolicy {
    pattern: String,
    trigger: PolicyTrigger,
    handler: PolicyHandler,
    regex: Regex,
}

impl PatternPolicy {
    fn new(pattern: &str, trigger: PolicyTrigger, handler: PolicyHandler) -> Self {
        // Convert glob pattern to regex
        let regex_str = pattern
            .replace(".", "\\.")
            .replace("**", ".*")
            .replace("*", "[^/]*");
        let regex_str = format!("^{}$", regex_str);
        
        Self {
            pattern: pattern.to_string(),
            trigger,
            handler,
            regex: Regex::new(&regex_str).unwrap_or_else(|_| Regex::new("^$").unwrap()),
        }
    }

    fn matches(&self, key: &[u8]) -> bool {
        if let Ok(key_str) = std::str::from_utf8(key) {
            self.regex.is_match(key_str)
        } else {
            false
        }
    }
}

/// Token bucket rate limiter.
struct RateLimiter {
    max_per_minute: u32,
    tokens: Mutex<u32>,
    last_refill: Mutex<Instant>,
}

impl RateLimiter {
    fn new(max_per_minute: u32) -> Self {
        Self {
            max_per_minute,
            tokens: Mutex::new(max_per_minute),
            last_refill: Mutex::new(Instant::now()),
        }
    }

    fn try_acquire(&self) -> bool {
        let mut tokens = self.tokens.lock().unwrap();
        let mut last_refill = self.last_refill.lock().unwrap();

        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);

        // Refill tokens based on elapsed time
        let refill = (elapsed.as_secs_f64() / 60.0 * self.max_per_minute as f64) as u32;
        if refill > 0 {
            *tokens = (*tokens + refill).min(self.max_per_minute);
            *last_refill = now;
        }

        if *tokens > 0 {
            *tokens -= 1;
            true
        } else {
            false
        }
    }
}

/// Rate limit configuration.
struct RateLimitConfig {
    operation: String,
    max_per_minute: u32,
    scope: String,
}

/// Audit log entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: Instant,
    pub operation: String,
    pub key: String,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub result: String,
}

/// Error when a policy blocks an operation.
#[derive(Debug)]
pub struct PolicyViolationError {
    pub message: String,
}

impl std::fmt::Display for PolicyViolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyViolation: {}", self.message)
    }
}

impl std::error::Error for PolicyViolationError {}

/// Policy engine for enforcing safety rules on database operations.
pub struct PolicyEngine<C: ConnectionTrait> {
    conn: C,
    policies: RwLock<HashMap<PolicyTrigger, Vec<PatternPolicy>>>,
    rate_limiters: RwLock<HashMap<String, HashMap<String, Arc<RateLimiter>>>>,
    rate_limit_configs: RwLock<Vec<RateLimitConfig>>,
    audit_log: RwLock<Vec<AuditEntry>>,
    audit_enabled: RwLock<bool>,
    max_audit_entries: usize,
}

impl<C: ConnectionTrait> PolicyEngine<C> {
    /// Create a new policy engine wrapping a connection.
    pub fn new(conn: C) -> Self {
        let mut policies = HashMap::new();
        policies.insert(PolicyTrigger::BeforeRead, Vec::new());
        policies.insert(PolicyTrigger::AfterRead, Vec::new());
        policies.insert(PolicyTrigger::BeforeWrite, Vec::new());
        policies.insert(PolicyTrigger::AfterWrite, Vec::new());
        policies.insert(PolicyTrigger::BeforeDelete, Vec::new());
        policies.insert(PolicyTrigger::AfterDelete, Vec::new());

        Self {
            conn,
            policies: RwLock::new(policies),
            rate_limiters: RwLock::new(HashMap::new()),
            rate_limit_configs: RwLock::new(Vec::new()),
            audit_log: RwLock::new(Vec::new()),
            audit_enabled: RwLock::new(false),
            max_audit_entries: 10000,
        }
    }

    /// Register a pre-write policy handler.
    pub fn before_write<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&PolicyContext) -> PolicyAction + Send + Sync + 'static,
    {
        let mut policies = self.policies.write().unwrap();
        policies
            .get_mut(&PolicyTrigger::BeforeWrite)
            .unwrap()
            .push(PatternPolicy::new(pattern, PolicyTrigger::BeforeWrite, Arc::new(handler)));
    }

    /// Register a post-write policy handler.
    pub fn after_write<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&PolicyContext) -> PolicyAction + Send + Sync + 'static,
    {
        let mut policies = self.policies.write().unwrap();
        policies
            .get_mut(&PolicyTrigger::AfterWrite)
            .unwrap()
            .push(PatternPolicy::new(pattern, PolicyTrigger::AfterWrite, Arc::new(handler)));
    }

    /// Register a pre-read policy handler.
    pub fn before_read<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&PolicyContext) -> PolicyAction + Send + Sync + 'static,
    {
        let mut policies = self.policies.write().unwrap();
        policies
            .get_mut(&PolicyTrigger::BeforeRead)
            .unwrap()
            .push(PatternPolicy::new(pattern, PolicyTrigger::BeforeRead, Arc::new(handler)));
    }

    /// Register a post-read policy handler.
    pub fn after_read<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&PolicyContext) -> PolicyAction + Send + Sync + 'static,
    {
        let mut policies = self.policies.write().unwrap();
        policies
            .get_mut(&PolicyTrigger::AfterRead)
            .unwrap()
            .push(PatternPolicy::new(pattern, PolicyTrigger::AfterRead, Arc::new(handler)));
    }

    /// Register a pre-delete policy handler.
    pub fn before_delete<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&PolicyContext) -> PolicyAction + Send + Sync + 'static,
    {
        let mut policies = self.policies.write().unwrap();
        policies
            .get_mut(&PolicyTrigger::BeforeDelete)
            .unwrap()
            .push(PatternPolicy::new(pattern, PolicyTrigger::BeforeDelete, Arc::new(handler)));
    }

    /// Add a rate limit policy.
    pub fn add_rate_limit(&self, operation: &str, max_per_minute: u32, scope: &str) {
        let mut configs = self.rate_limit_configs.write().unwrap();
        configs.push(RateLimitConfig {
            operation: operation.to_string(),
            max_per_minute,
            scope: scope.to_string(),
        });
    }

    /// Enable audit logging.
    pub fn enable_audit(&self) {
        let mut enabled = self.audit_enabled.write().unwrap();
        *enabled = true;
    }

    /// Disable audit logging.
    pub fn disable_audit(&self) {
        let mut enabled = self.audit_enabled.write().unwrap();
        *enabled = false;
    }

    /// Get recent audit log entries.
    pub fn get_audit_log(&self, limit: usize) -> Vec<AuditEntry> {
        let log = self.audit_log.read().unwrap();
        let start = log.len().saturating_sub(limit);
        log[start..].to_vec()
    }

    fn check_rate_limit(&self, operation: &str, ctx: &PolicyContext) -> bool {
        let configs = self.rate_limit_configs.read().unwrap();
        let mut limiters = self.rate_limiters.write().unwrap();

        for config in configs.iter() {
            if config.operation != operation && config.operation != "all" {
                continue;
            }

            let scope_key = match config.scope.as_str() {
                "global" => "global".to_string(),
                "agent_id" => ctx.agent_id.clone().unwrap_or_else(|| "unknown".to_string()),
                "session_id" => ctx.session_id.clone().unwrap_or_else(|| "unknown".to_string()),
                _ => ctx.get(&config.scope).cloned().unwrap_or_else(|| "unknown".to_string()),
            };

            let limiter_key = format!("{}:{}", config.operation, config.scope);
            let scope_limiters = limiters.entry(limiter_key).or_insert_with(HashMap::new);

            let limiter = scope_limiters
                .entry(scope_key)
                .or_insert_with(|| Arc::new(RateLimiter::new(config.max_per_minute)));

            if !limiter.try_acquire() {
                return false;
            }
        }
        true
    }

    fn evaluate_policies(&self, trigger: PolicyTrigger, ctx: &PolicyContext) -> PolicyAction {
        let policies = self.policies.read().unwrap();
        if let Some(trigger_policies) = policies.get(&trigger) {
            for policy in trigger_policies {
                if policy.matches(&ctx.key) {
                    let action = (policy.handler)(ctx);
                    match &action {
                        PolicyAction::Deny | PolicyAction::Modify(_) => return action,
                        _ => {}
                    }
                }
            }
        }
        PolicyAction::Allow
    }

    fn audit(&self, operation: &str, key: &[u8], ctx: &PolicyContext, result: &str) {
        let enabled = self.audit_enabled.read().unwrap();
        if !*enabled {
            return;
        }

        let mut log = self.audit_log.write().unwrap();
        log.push(AuditEntry {
            timestamp: Instant::now(),
            operation: operation.to_string(),
            key: String::from_utf8_lossy(key).to_string(),
            agent_id: ctx.agent_id.clone(),
            session_id: ctx.session_id.clone(),
            result: result.to_string(),
        });

        if log.len() > self.max_audit_entries {
            let start = log.len() - self.max_audit_entries;
            *log = log[start..].to_vec();
        }
    }

    /// Put a value with policy enforcement.
    pub fn put(
        &self,
        key: &[u8],
        value: &[u8],
        ctx: Option<&PolicyContext>,
    ) -> std::result::Result<(), PolicyViolationError> {
        let default_ctx = PolicyContext::new("write", key);
        let ctx = ctx.unwrap_or(&default_ctx);

        if !self.check_rate_limit("write", ctx) {
            self.audit("write", key, ctx, "rate_limited");
            return Err(PolicyViolationError {
                message: "Rate limit exceeded".to_string(),
            });
        }

        match self.evaluate_policies(PolicyTrigger::BeforeWrite, ctx) {
            PolicyAction::Deny => {
                self.audit("write", key, ctx, "denied");
                return Err(PolicyViolationError {
                    message: "Write blocked by policy".to_string(),
                });
            }
            PolicyAction::Modify(modified) => {
                self.conn.put(key, &modified).map_err(|_| PolicyViolationError {
                    message: "Write failed".to_string(),
                })?;
            }
            _ => {
                self.conn.put(key, value).map_err(|_| PolicyViolationError {
                    message: "Write failed".to_string(),
                })?;
            }
        }

        self.evaluate_policies(PolicyTrigger::AfterWrite, ctx);
        self.audit("write", key, ctx, "allowed");
        Ok(())
    }

    /// Get a value with policy enforcement.
    pub fn get(&self, key: &[u8], ctx: Option<&PolicyContext>) -> std::result::Result<Option<Vec<u8>>, PolicyViolationError> {
        let default_ctx = PolicyContext::new("read", key);
        let ctx = ctx.unwrap_or(&default_ctx);

        if !self.check_rate_limit("read", ctx) {
            self.audit("read", key, ctx, "rate_limited");
            return Err(PolicyViolationError {
                message: "Rate limit exceeded".to_string(),
            });
        }

        if let PolicyAction::Deny = self.evaluate_policies(PolicyTrigger::BeforeRead, ctx) {
            self.audit("read", key, ctx, "denied");
            return Err(PolicyViolationError {
                message: "Read blocked by policy".to_string(),
            });
        }

        let value = self.conn.get(key).map_err(|_| PolicyViolationError {
            message: "Read failed".to_string(),
        })?;

        if let Some(ref val) = value {
            let mut read_ctx = ctx.clone();
            read_ctx.value = Some(val.clone());

            match self.evaluate_policies(PolicyTrigger::AfterRead, &read_ctx) {
                PolicyAction::Modify(modified) => {
                    self.audit("read", key, ctx, "allowed");
                    return Ok(Some(modified));
                }
                PolicyAction::Deny => {
                    self.audit("read", key, ctx, "redacted");
                    return Ok(None);
                }
                _ => {}
            }
        }

        self.audit("read", key, ctx, "allowed");
        Ok(value)
    }

    /// Delete a value with policy enforcement.
    pub fn delete(&self, key: &[u8], ctx: Option<&PolicyContext>) -> std::result::Result<(), PolicyViolationError> {
        let default_ctx = PolicyContext::new("delete", key);
        let ctx = ctx.unwrap_or(&default_ctx);

        if !self.check_rate_limit("delete", ctx) {
            self.audit("delete", key, ctx, "rate_limited");
            return Err(PolicyViolationError {
                message: "Rate limit exceeded".to_string(),
            });
        }

        if let PolicyAction::Deny = self.evaluate_policies(PolicyTrigger::BeforeDelete, ctx) {
            self.audit("delete", key, ctx, "denied");
            return Err(PolicyViolationError {
                message: "Delete blocked by policy".to_string(),
            });
        }

        self.conn.delete(key).map_err(|_| PolicyViolationError {
            message: "Delete failed".to_string(),
        })?;

        self.audit("delete", key, ctx, "allowed");
        Ok(())
    }
}

// ============================================================================
// Built-in Policy Helpers
// ============================================================================

/// Policy that denies all matching operations.
pub fn deny_all() -> impl Fn(&PolicyContext) -> PolicyAction {
    |_| PolicyAction::Deny
}

/// Policy that allows all matching operations.
pub fn allow_all() -> impl Fn(&PolicyContext) -> PolicyAction {
    |_| PolicyAction::Allow
}

/// Policy that requires an agent_id in context.
pub fn require_agent_id() -> impl Fn(&PolicyContext) -> PolicyAction {
    |ctx| {
        if ctx.agent_id.is_some() {
            PolicyAction::Allow
        } else {
            PolicyAction::Deny
        }
    }
}

/// Policy factory that redacts values.
pub fn redact_value(replacement: Vec<u8>) -> impl Fn(&PolicyContext) -> PolicyAction {
    move |_| PolicyAction::Modify(replacement.clone())
}
// ============================================================================
// Policy Outcome Algebra (Task 2)
// ============================================================================

/// Policy outcome with deterministic precedence.
/// 
/// The precedence order forms a semilattice:
/// `Deny > Modify > Allow` with `Log` as a side-effect.
/// 
/// This ensures:
/// 1. Deterministic conflict resolution
/// 2. Associative composition
/// 3. Security-biased defaults (Deny wins)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolicyOutcome {
    /// Operation is allowed (lowest precedence)
    Allow = 0,
    /// Operation is allowed but logged
    AllowWithLog = 1,
    /// Operation is allowed with modification
    Modify = 2,
    /// Operation is blocked (highest precedence)
    Deny = 3,
}

impl PolicyOutcome {
    /// Combine two outcomes using the semilattice join (max precedence wins)
    pub fn join(self, other: PolicyOutcome) -> PolicyOutcome {
        if self >= other { self } else { other }
    }
}

impl From<&PolicyAction> for PolicyOutcome {
    fn from(action: &PolicyAction) -> Self {
        match action {
            PolicyAction::Allow => PolicyOutcome::Allow,
            PolicyAction::Log => PolicyOutcome::AllowWithLog,
            PolicyAction::Modify(_) => PolicyOutcome::Modify,
            PolicyAction::Deny => PolicyOutcome::Deny,
        }
    }
}

/// A compiled policy rule with typed hook points
#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Unique rule ID for debugging/auditing
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// When this rule triggers
    pub trigger: PolicyTrigger,
    /// Pattern to match (glob-style)
    pub pattern: String,
    /// Priority (higher = evaluated first within same trigger)
    pub priority: i32,
    /// Namespace scope (None = all namespaces)
    pub namespace: Option<String>,
    /// The outcome this rule produces
    pub outcome: PolicyOutcome,
}

/// Compiled rule set for fast evaluation
pub struct CompiledPolicySet {
    /// Rules indexed by trigger type, sorted by priority (descending)
    rules_by_trigger: HashMap<PolicyTrigger, Vec<CompiledRule>>,
}

struct CompiledRule {
    rule: PolicyRule,
    regex: Regex,
    handler: Option<PolicyHandler>,
}

impl CompiledPolicySet {
    /// Create a new compiled policy set
    pub fn new() -> Self {
        let mut rules_by_trigger = HashMap::new();
        for trigger in [
            PolicyTrigger::BeforeRead,
            PolicyTrigger::AfterRead,
            PolicyTrigger::BeforeWrite,
            PolicyTrigger::AfterWrite,
            PolicyTrigger::BeforeDelete,
            PolicyTrigger::AfterDelete,
        ] {
            rules_by_trigger.insert(trigger, Vec::new());
        }
        Self { rules_by_trigger }
    }
    
    /// Add a rule to the set
    pub fn add_rule(&mut self, rule: PolicyRule, handler: Option<PolicyHandler>) {
        let regex_str = rule.pattern
            .replace(".", "\\.")
            .replace("**", ".*")
            .replace("*", "[^/]*");
        let regex_str = format!("^{}$", regex_str);
        let regex = Regex::new(&regex_str).unwrap_or_else(|_| Regex::new("^$").unwrap());
        
        let compiled = CompiledRule {
            rule: rule.clone(),
            regex,
            handler,
        };
        
        if let Some(rules) = self.rules_by_trigger.get_mut(&rule.trigger) {
            rules.push(compiled);
            // Sort by priority descending
            rules.sort_by(|a, b| b.rule.priority.cmp(&a.rule.priority));
        }
    }
    
    /// Evaluate all applicable rules and return the final outcome
    ///
    /// Uses semilattice join for deterministic composition.
    pub fn evaluate(&self, trigger: PolicyTrigger, ctx: &PolicyContext) -> EvaluationResult {
        let mut final_outcome = PolicyOutcome::Allow;
        let mut applied_rules = Vec::new();
        let mut modifications = Vec::new();
        
        if let Some(rules) = self.rules_by_trigger.get(&trigger) {
            for compiled in rules {
                // Check namespace filter
                if let Some(ref ns) = compiled.rule.namespace {
                    if let Some(ctx_ns) = ctx.custom.get("namespace") {
                        if ns != ctx_ns.as_str() {
                            continue;
                        }
                    }
                }
                
                // Check pattern match
                let key_str = String::from_utf8_lossy(&ctx.key);
                if !compiled.regex.is_match(&key_str) {
                    continue;
                }
                
                // Evaluate handler if present, otherwise use rule's static outcome
                let outcome = if let Some(ref handler) = compiled.handler {
                    let action = handler(ctx);
                    match &action {
                        PolicyAction::Modify(data) => {
                            modifications.push(data.clone());
                        }
                        _ => {}
                    }
                    PolicyOutcome::from(&action)
                } else {
                    compiled.rule.outcome.clone()
                };
                
                applied_rules.push(compiled.rule.id.clone());
                final_outcome = final_outcome.join(outcome);
                
                // Short-circuit on Deny (highest precedence, nothing can override)
                if final_outcome == PolicyOutcome::Deny {
                    break;
                }
            }
        }
        
        EvaluationResult {
            outcome: final_outcome,
            applied_rules,
            modifications,
        }
    }
}

impl Default for CompiledPolicySet {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of policy evaluation
#[derive(Debug)]
pub struct EvaluationResult {
    /// Final outcome after semilattice join
    pub outcome: PolicyOutcome,
    /// IDs of rules that were applied
    pub applied_rules: Vec<String>,
    /// Modifications to apply (if outcome is Modify)
    pub modifications: Vec<Vec<u8>>,
}

impl EvaluationResult {
    /// Check if the operation is allowed
    pub fn is_allowed(&self) -> bool {
        !matches!(self.outcome, PolicyOutcome::Deny)
    }
    
    /// Get the modification to apply (if any)
    /// 
    /// If multiple modifications exist, they are concatenated.
    /// For more complex composition, use a custom modification handler.
    pub fn get_modification(&self) -> Option<Vec<u8>> {
        if self.modifications.is_empty() {
            None
        } else if self.modifications.len() == 1 {
            Some(self.modifications[0].clone())
        } else {
            // Multiple modifications - last one wins (or implement custom logic)
            Some(self.modifications.last().unwrap().clone())
        }
    }
}

// ============================================================================
// Policy Engine Integration with AllowedSet
// ============================================================================

/// Convert policy denials to AllowedSet exclusions
///
/// This is the integration point between the policy engine and retrieval.
/// Documents denied by policy are removed from the AllowedSet BEFORE
/// candidate generation, preserving the "no post-filtering" invariant.
impl<C: ConnectionTrait> PolicyEngine<C> {
    /// Evaluate policies and return IDs that should be EXCLUDED from results
    ///
    /// This integrates with the retrieval layer by converting policy denials
    /// into AllowedSet membership.
    pub fn get_denied_ids(
        &self,
        trigger: PolicyTrigger,
        candidate_ids: &[Vec<u8>],
        base_ctx: &PolicyContext,
    ) -> Vec<Vec<u8>> {
        let mut denied = Vec::new();
        
        for key in candidate_ids {
            let mut ctx = base_ctx.clone();
            ctx.key = key.clone();
            
            if let PolicyAction::Deny = self.evaluate_policies(trigger, &ctx) {
                denied.push(key.clone());
            }
        }
        
        denied
    }
}