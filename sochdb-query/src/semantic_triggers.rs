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

//! Semantic Trigger Engine (Task 7)
//!
//! This module implements a vector percolator for semantic trigger matching.
//! It enables proactive, event-driven agent behavior by matching incoming
//! content against stored trigger patterns.
//!
//! ## Architecture
//!
//! ```text
//! Insert/Event
//!     │
//!     ▼
//! ┌─────────────────┐
//! │    Embed Text   │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Trigger Index  │ ← Stored trigger embeddings
//! │    (ANN Search) │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Fire Callbacks │ → notify, route, escalate
//! └─────────────────┘
//! ```
//!
//! ## Complexity
//!
//! - Trigger matching: O(log T) where T = number of triggers (ANN)
//! - Alternative (LSH): O(1) bucket lookup + O(C × D) for candidates

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Trigger Types
// ============================================================================

/// A semantic trigger definition
#[derive(Debug, Clone)]
pub struct SemanticTrigger {
    /// Unique trigger identifier
    pub id: String,
    
    /// Human-readable name
    pub name: String,
    
    /// Description of what this trigger matches
    pub description: String,
    
    /// Query/pattern that defines the trigger
    pub query: String,
    
    /// Embedding of the query (for ANN matching)
    pub embedding: Option<Vec<f32>>,
    
    /// Similarity threshold (0.0 to 1.0)
    pub threshold: f32,
    
    /// Action to take when triggered
    pub action: TriggerAction,
    
    /// Whether this trigger is active
    pub enabled: bool,
    
    /// Priority (lower = higher priority)
    pub priority: i32,
    
    /// Maximum fires per time window (rate limiting)
    pub max_fires_per_window: Option<usize>,
    
    /// Time window for rate limiting (seconds)
    pub rate_limit_window_secs: Option<u64>,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Metadata
    pub metadata: HashMap<String, String>,
    
    /// Created timestamp
    pub created_at: f64,
}

/// Actions that can be taken when a trigger fires
#[derive(Debug, Clone)]
pub enum TriggerAction {
    /// Send a notification
    Notify {
        channel: String,
        template: Option<String>,
    },
    
    /// Route to a specific handler/agent
    Route {
        target: String,
        context: Option<String>,
    },
    
    /// Escalate to human review
    Escalate {
        level: EscalationLevel,
        reason: Option<String>,
    },
    
    /// Spawn a new agent/workflow
    SpawnAgent {
        agent_type: String,
        config: HashMap<String, String>,
    },
    
    /// Log the event
    Log {
        level: LogLevel,
        message: Option<String>,
    },
    
    /// Execute a webhook
    Webhook {
        url: String,
        method: String,
        headers: HashMap<String, String>,
    },
    
    /// Custom callback function name
    Callback {
        function: String,
        args: HashMap<String, String>,
    },
    
    /// Chain of actions
    Chain(Vec<TriggerAction>),
}

/// Escalation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

// ============================================================================
// Trigger Events
// ============================================================================

/// An event that can fire triggers
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    /// Event identifier
    pub id: String,
    
    /// Event content (text to match against triggers)
    pub content: String,
    
    /// Event embedding (if pre-computed)
    pub embedding: Option<Vec<f32>>,
    
    /// Event source
    pub source: EventSource,
    
    /// Event metadata
    pub metadata: HashMap<String, String>,
    
    /// Timestamp
    pub timestamp: f64,
}

/// Source of trigger events
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventSource {
    /// User message
    UserMessage,
    /// System event
    SystemEvent,
    /// Data insert
    DataInsert,
    /// Memory compaction
    MemoryCompaction,
    /// External API
    ExternalApi,
    /// Agent action
    AgentAction,
    /// Custom source
    Custom(String),
}

/// Result of a trigger match
#[derive(Debug, Clone)]
pub struct TriggerMatch {
    /// The trigger that matched
    pub trigger_id: String,
    
    /// Similarity score
    pub score: f32,
    
    /// The event that caused the match
    pub event_id: String,
    
    /// Timestamp of the match
    pub timestamp: f64,
    
    /// Whether the action was executed
    pub action_executed: bool,
    
    /// Execution result or error
    pub execution_result: Option<String>,
}

/// Statistics about trigger execution
#[derive(Debug, Clone, Default)]
pub struct TriggerStats {
    /// Total events processed
    pub events_processed: usize,
    
    /// Total triggers matched
    pub triggers_matched: usize,
    
    /// Total actions executed
    pub actions_executed: usize,
    
    /// Matches by trigger ID
    pub matches_by_trigger: HashMap<String, usize>,
    
    /// Rate-limited fires
    pub rate_limited: usize,
}

// ============================================================================
// Trigger Index
// ============================================================================

/// Index for semantic trigger matching
pub struct TriggerIndex {
    /// All registered triggers
    triggers: RwLock<HashMap<String, SemanticTrigger>>,
    
    /// Trigger embeddings for ANN search
    trigger_embeddings: RwLock<Vec<(String, Vec<f32>)>>,
    
    /// Rate limit tracking: trigger_id -> (fire_count, window_start)
    rate_limits: RwLock<HashMap<String, (usize, f64)>>,
    
    /// Recent matches (for debugging/audit)
    recent_matches: RwLock<Vec<TriggerMatch>>,
    
    /// Statistics
    stats: RwLock<TriggerStats>,
    
    /// Maximum recent matches to keep
    max_recent_matches: usize,
}

impl TriggerIndex {
    /// Create a new trigger index
    pub fn new() -> Self {
        Self {
            triggers: RwLock::new(HashMap::new()),
            trigger_embeddings: RwLock::new(Vec::new()),
            rate_limits: RwLock::new(HashMap::new()),
            recent_matches: RwLock::new(Vec::new()),
            stats: RwLock::new(TriggerStats::default()),
            max_recent_matches: 1000,
        }
    }
    
    /// Register a new trigger
    pub fn register_trigger(&self, mut trigger: SemanticTrigger) -> Result<(), TriggerError> {
        if trigger.id.is_empty() {
            return Err(TriggerError::InvalidTrigger("ID cannot be empty".to_string()));
        }
        
        // Set creation timestamp if not set
        if trigger.created_at == 0.0 {
            trigger.created_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
        }
        
        // Store trigger
        {
            let mut triggers = self.triggers.write().unwrap();
            triggers.insert(trigger.id.clone(), trigger.clone());
        }
        
        // Store embedding if present
        if let Some(embedding) = &trigger.embedding {
            let mut embeddings = self.trigger_embeddings.write().unwrap();
            embeddings.push((trigger.id.clone(), embedding.clone()));
        }
        
        Ok(())
    }
    
    /// Remove a trigger
    pub fn remove_trigger(&self, trigger_id: &str) -> Option<SemanticTrigger> {
        let removed = {
            let mut triggers = self.triggers.write().unwrap();
            triggers.remove(trigger_id)
        };
        
        if removed.is_some() {
            let mut embeddings = self.trigger_embeddings.write().unwrap();
            embeddings.retain(|(id, _)| id != trigger_id);
        }
        
        removed
    }
    
    /// Enable/disable a trigger
    pub fn set_enabled(&self, trigger_id: &str, enabled: bool) -> bool {
        let mut triggers = self.triggers.write().unwrap();
        if let Some(trigger) = triggers.get_mut(trigger_id) {
            trigger.enabled = enabled;
            true
        } else {
            false
        }
    }
    
    /// Update trigger threshold
    pub fn set_threshold(&self, trigger_id: &str, threshold: f32) -> bool {
        let mut triggers = self.triggers.write().unwrap();
        if let Some(trigger) = triggers.get_mut(trigger_id) {
            trigger.threshold = threshold.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }
    
    /// Process an event and find matching triggers
    pub fn process_event(&self, event: &TriggerEvent) -> Vec<TriggerMatch> {
        let mut matches = Vec::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.events_processed += 1;
        }
        
        // Get event embedding (required for matching)
        let event_embedding = match &event.embedding {
            Some(emb) => emb.clone(),
            None => {
                // Would generate embedding here in production
                return matches;
            }
        };
        
        // Find matching triggers via ANN search
        let candidates = self.find_candidates(&event_embedding, 10);
        
        let triggers = self.triggers.read().unwrap();
        
        for (trigger_id, score) in candidates {
            if let Some(trigger) = triggers.get(&trigger_id) {
                // Check if enabled
                if !trigger.enabled {
                    continue;
                }
                
                // Check threshold
                if score < trigger.threshold {
                    continue;
                }
                
                // Check rate limit
                if !self.check_rate_limit(&trigger_id, trigger, now) {
                    let mut stats = self.stats.write().unwrap();
                    stats.rate_limited += 1;
                    continue;
                }
                
                // Create match
                let trigger_match = TriggerMatch {
                    trigger_id: trigger_id.clone(),
                    score,
                    event_id: event.id.clone(),
                    timestamp: now,
                    action_executed: false,
                    execution_result: None,
                };
                
                matches.push(trigger_match);
                
                // Update match stats
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.triggers_matched += 1;
                    *stats.matches_by_trigger.entry(trigger_id.clone()).or_insert(0) += 1;
                }
            }
        }
        
        // Sort by priority then score
        matches.sort_by(|a, b| {
            let trigger_a = triggers.get(&a.trigger_id);
            let trigger_b = triggers.get(&b.trigger_id);
            
            match (trigger_a, trigger_b) {
                (Some(ta), Some(tb)) => {
                    ta.priority.cmp(&tb.priority)
                        .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
                }
                _ => std::cmp::Ordering::Equal,
            }
        });
        
        // Store recent matches
        {
            let mut recent = self.recent_matches.write().unwrap();
            for m in &matches {
                recent.push(m.clone());
            }
            // Trim to max size
            while recent.len() > self.max_recent_matches {
                recent.remove(0);
            }
        }
        
        matches
    }
    
    /// Find candidate triggers using ANN search
    fn find_candidates(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let embeddings = self.trigger_embeddings.read().unwrap();
        
        if embeddings.is_empty() {
            return Vec::new();
        }
        
        // Simple brute-force search (would use HNSW in production)
        let mut candidates: Vec<(String, f32)> = embeddings
            .iter()
            .map(|(id, emb)| {
                let score = cosine_similarity(query, emb);
                (id.clone(), score)
            })
            .collect();
        
        // Sort by similarity descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        candidates.truncate(k);
        candidates
    }
    
    /// Check and update rate limit for a trigger
    fn check_rate_limit(&self, trigger_id: &str, trigger: &SemanticTrigger, now: f64) -> bool {
        let max_fires = match trigger.max_fires_per_window {
            Some(max) => max,
            None => return true, // No rate limit
        };
        
        let window_secs = trigger.rate_limit_window_secs.unwrap_or(60);
        
        let mut rate_limits = self.rate_limits.write().unwrap();
        let entry = rate_limits.entry(trigger_id.to_string()).or_insert((0, now));
        
        // Check if window has expired
        if now - entry.1 > window_secs as f64 {
            entry.0 = 1;
            entry.1 = now;
            return true;
        }
        
        // Check if under limit
        if entry.0 < max_fires {
            entry.0 += 1;
            return true;
        }
        
        false
    }
    
    /// Execute action for a trigger match
    pub fn execute_action(&self, trigger_match: &mut TriggerMatch) -> Result<(), TriggerError> {
        let triggers = self.triggers.read().unwrap();
        let trigger = triggers.get(&trigger_match.trigger_id)
            .ok_or_else(|| TriggerError::TriggerNotFound(trigger_match.trigger_id.clone()))?;
        
        // Execute the action
        let result = self.execute_action_impl(&trigger.action, trigger_match)?;
        
        trigger_match.action_executed = true;
        trigger_match.execution_result = Some(result);
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.actions_executed += 1;
        }
        
        Ok(())
    }
    
    /// Execute a specific action
    fn execute_action_impl(&self, action: &TriggerAction, trigger_match: &TriggerMatch) -> Result<String, TriggerError> {
        match action {
            TriggerAction::Notify { channel, template } => {
                // Would send notification in production
                Ok(format!("Notified channel '{}' (template: {:?})", channel, template))
            }
            
            TriggerAction::Route { target, context } => {
                Ok(format!("Routed to '{}' (context: {:?})", target, context))
            }
            
            TriggerAction::Escalate { level, reason } => {
                Ok(format!("Escalated at level {:?} (reason: {:?})", level, reason))
            }
            
            TriggerAction::SpawnAgent { agent_type, config: _ } => {
                Ok(format!("Spawned agent of type '{}'", agent_type))
            }
            
            TriggerAction::Log { level, message } => {
                let msg = message.as_deref().unwrap_or(&trigger_match.trigger_id);
                Ok(format!("Logged at {:?}: {}", level, msg))
            }
            
            TriggerAction::Webhook { url, method, headers: _ } => {
                // Would make HTTP request in production
                Ok(format!("Called webhook {} {}", method, url))
            }
            
            TriggerAction::Callback { function, args: _ } => {
                Ok(format!("Called callback function '{}'", function))
            }
            
            TriggerAction::Chain(actions) => {
                let mut results = Vec::new();
                for sub_action in actions {
                    let result = self.execute_action_impl(sub_action, trigger_match)?;
                    results.push(result);
                }
                Ok(format!("Chain executed: [{}]", results.join(", ")))
            }
        }
    }
    
    /// Get all registered triggers
    pub fn list_triggers(&self) -> Vec<SemanticTrigger> {
        self.triggers.read().unwrap().values().cloned().collect()
    }
    
    /// Get trigger by ID
    pub fn get_trigger(&self, trigger_id: &str) -> Option<SemanticTrigger> {
        self.triggers.read().unwrap().get(trigger_id).cloned()
    }
    
    /// Get recent matches
    pub fn recent_matches(&self, limit: usize) -> Vec<TriggerMatch> {
        let matches = self.recent_matches.read().unwrap();
        matches.iter().rev().take(limit).cloned().collect()
    }
    
    /// Get statistics
    pub fn stats(&self) -> TriggerStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Clear statistics
    pub fn clear_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = TriggerStats::default();
    }
}

impl Default for TriggerIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Trigger Builder
// ============================================================================

/// Builder for creating semantic triggers
pub struct TriggerBuilder {
    trigger: SemanticTrigger,
}

impl TriggerBuilder {
    /// Create a new trigger builder
    pub fn new(id: &str, query: &str) -> Self {
        Self {
            trigger: SemanticTrigger {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                query: query.to_string(),
                embedding: None,
                threshold: 0.8,
                action: TriggerAction::Log {
                    level: LogLevel::Info,
                    message: None,
                },
                enabled: true,
                priority: 0,
                max_fires_per_window: None,
                rate_limit_window_secs: None,
                tags: Vec::new(),
                metadata: HashMap::new(),
                created_at: 0.0,
            },
        }
    }
    
    /// Set trigger name
    pub fn name(mut self, name: &str) -> Self {
        self.trigger.name = name.to_string();
        self
    }
    
    /// Set description
    pub fn description(mut self, description: &str) -> Self {
        self.trigger.description = description.to_string();
        self
    }
    
    /// Set embedding
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.trigger.embedding = Some(embedding);
        self
    }
    
    /// Set threshold
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.trigger.threshold = threshold.clamp(0.0, 1.0);
        self
    }
    
    /// Set action
    pub fn action(mut self, action: TriggerAction) -> Self {
        self.trigger.action = action;
        self
    }
    
    /// Set as notify action
    pub fn notify(mut self, channel: &str) -> Self {
        self.trigger.action = TriggerAction::Notify {
            channel: channel.to_string(),
            template: None,
        };
        self
    }
    
    /// Set as route action
    pub fn route(mut self, target: &str) -> Self {
        self.trigger.action = TriggerAction::Route {
            target: target.to_string(),
            context: None,
        };
        self
    }
    
    /// Set as escalate action
    pub fn escalate(mut self, level: EscalationLevel) -> Self {
        self.trigger.action = TriggerAction::Escalate {
            level,
            reason: None,
        };
        self
    }
    
    /// Set priority
    pub fn priority(mut self, priority: i32) -> Self {
        self.trigger.priority = priority;
        self
    }
    
    /// Set rate limit
    pub fn rate_limit(mut self, max_fires: usize, window_secs: u64) -> Self {
        self.trigger.max_fires_per_window = Some(max_fires);
        self.trigger.rate_limit_window_secs = Some(window_secs);
        self
    }
    
    /// Add tag
    pub fn tag(mut self, tag: &str) -> Self {
        self.trigger.tags.push(tag.to_string());
        self
    }
    
    /// Set enabled state
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.trigger.enabled = enabled;
        self
    }
    
    /// Build the trigger
    pub fn build(self) -> SemanticTrigger {
        self.trigger
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Trigger-related errors
#[derive(Debug, Clone)]
pub enum TriggerError {
    /// Invalid trigger definition
    InvalidTrigger(String),
    /// Trigger not found
    TriggerNotFound(String),
    /// Action execution failed
    ActionFailed(String),
    /// Rate limit exceeded
    RateLimitExceeded(String),
    /// Embedding error
    EmbeddingError(String),
}

impl std::fmt::Display for TriggerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTrigger(msg) => write!(f, "Invalid trigger: {}", msg),
            Self::TriggerNotFound(id) => write!(f, "Trigger not found: {}", id),
            Self::ActionFailed(msg) => write!(f, "Action failed: {}", msg),
            Self::RateLimitExceeded(id) => write!(f, "Rate limit exceeded for trigger: {}", id),
            Self::EmbeddingError(msg) => write!(f, "Embedding error: {}", msg),
        }
    }
}

impl std::error::Error for TriggerError {}

// ============================================================================
// Utilities
// ============================================================================

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a simple trigger with notify action
pub fn create_notify_trigger(
    id: &str,
    query: &str,
    channel: &str,
    embedding: Vec<f32>,
) -> SemanticTrigger {
    TriggerBuilder::new(id, query)
        .embedding(embedding)
        .notify(channel)
        .build()
}

/// Create a trigger with escalation
pub fn create_escalation_trigger(
    id: &str,
    query: &str,
    level: EscalationLevel,
    embedding: Vec<f32>,
) -> SemanticTrigger {
    TriggerBuilder::new(id, query)
        .embedding(embedding)
        .escalate(level)
        .priority(-1) // High priority
        .build()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn mock_embedding(seed: u64) -> Vec<f32> {
        (0..128)
            .map(|i| ((i as u64 + seed) % 100) as f32 / 100.0 - 0.5)
            .collect()
    }
    
    #[test]
    fn test_trigger_registration() {
        let index = TriggerIndex::new();
        
        let trigger = TriggerBuilder::new("privacy_concern", "user mentions privacy concerns")
            .embedding(mock_embedding(1))
            .threshold(0.75)
            .escalate(EscalationLevel::High)
            .build();
        
        index.register_trigger(trigger).unwrap();
        
        let triggers = index.list_triggers();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].id, "privacy_concern");
    }
    
    #[test]
    fn test_trigger_matching() {
        let index = TriggerIndex::new();
        
        let trigger = TriggerBuilder::new("security_alert", "security vulnerability")
            .embedding(mock_embedding(1))
            .threshold(0.5) // Low threshold for testing
            .notify("security-team")
            .build();
        
        index.register_trigger(trigger).unwrap();
        
        // Create event with similar embedding
        let event = TriggerEvent {
            id: "event_1".to_string(),
            content: "possible security issue detected".to_string(),
            embedding: Some(mock_embedding(1)), // Same embedding
            source: EventSource::SystemEvent,
            metadata: HashMap::new(),
            timestamp: 0.0,
        };
        
        let matches = index.process_event(&event);
        
        assert!(!matches.is_empty());
        assert_eq!(matches[0].trigger_id, "security_alert");
        assert!(matches[0].score > 0.5);
    }
    
    #[test]
    fn test_trigger_disable() {
        let index = TriggerIndex::new();
        
        let trigger = TriggerBuilder::new("test_trigger", "test")
            .embedding(mock_embedding(1))
            .threshold(0.5)
            .build();
        
        index.register_trigger(trigger).unwrap();
        
        // Disable trigger
        index.set_enabled("test_trigger", false);
        
        let event = TriggerEvent {
            id: "event_1".to_string(),
            content: "test".to_string(),
            embedding: Some(mock_embedding(1)),
            source: EventSource::UserMessage,
            metadata: HashMap::new(),
            timestamp: 0.0,
        };
        
        let matches = index.process_event(&event);
        
        // Should not match disabled trigger
        assert!(matches.is_empty());
    }
    
    #[test]
    fn test_rate_limiting() {
        let index = TriggerIndex::new();
        
        let trigger = TriggerBuilder::new("rate_limited", "test")
            .embedding(mock_embedding(1))
            .threshold(0.5)
            .rate_limit(2, 60) // Max 2 fires per 60 seconds
            .build();
        
        index.register_trigger(trigger).unwrap();
        
        let event = TriggerEvent {
            id: "event_1".to_string(),
            content: "test".to_string(),
            embedding: Some(mock_embedding(1)),
            source: EventSource::UserMessage,
            metadata: HashMap::new(),
            timestamp: 0.0,
        };
        
        // First two should match
        let m1 = index.process_event(&event);
        let m2 = index.process_event(&event);
        
        // Third should be rate limited
        let m3 = index.process_event(&event);
        
        assert!(!m1.is_empty());
        assert!(!m2.is_empty());
        assert!(m3.is_empty());
        
        // Check rate limit stats
        let stats = index.stats();
        assert!(stats.rate_limited >= 1);
    }
    
    #[test]
    fn test_action_execution() {
        let index = TriggerIndex::new();
        
        let trigger = TriggerBuilder::new("log_trigger", "test")
            .embedding(mock_embedding(1))
            .threshold(0.5)
            .action(TriggerAction::Log {
                level: LogLevel::Info,
                message: Some("Test message".to_string()),
            })
            .build();
        
        index.register_trigger(trigger).unwrap();
        
        let event = TriggerEvent {
            id: "event_1".to_string(),
            content: "test".to_string(),
            embedding: Some(mock_embedding(1)),
            source: EventSource::UserMessage,
            metadata: HashMap::new(),
            timestamp: 0.0,
        };
        
        let mut matches = index.process_event(&event);
        
        assert!(!matches.is_empty());
        
        // Execute action
        index.execute_action(&mut matches[0]).unwrap();
        
        assert!(matches[0].action_executed);
        assert!(matches[0].execution_result.is_some());
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.01);
        
        let c = vec![0.0, 1.0, 0.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!(sim2.abs() < 0.01);
    }
    
    #[test]
    fn test_trigger_builder() {
        let trigger = TriggerBuilder::new("test", "test query")
            .name("Test Trigger")
            .description("A test trigger")
            .threshold(0.85)
            .priority(5)
            .tag("test")
            .tag("example")
            .notify("test-channel")
            .rate_limit(10, 300)
            .build();
        
        assert_eq!(trigger.id, "test");
        assert_eq!(trigger.name, "Test Trigger");
        assert_eq!(trigger.threshold, 0.85);
        assert_eq!(trigger.priority, 5);
        assert_eq!(trigger.tags.len(), 2);
        assert_eq!(trigger.max_fires_per_window, Some(10));
    }
}
