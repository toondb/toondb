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

//! Canonical Memory Schema for LLM Agents
//!
//! This module defines the **core semantic schema** for SochDB as an LLM memory store:
//!
//! - `episodes` - Task/conversation runs with summaries and embeddings
//! - `events` - Steps within episodes (tool calls, messages, etc.)
//! - `entities` - Users, projects, documents, services, etc.
//!
//! ## Design Rationale
//!
//! Without a canonical core, LLMs must reason over O(T) unrelated table schemas.
//! With this fixed `(episodes, events, entities)` core, effective complexity becomes
//! O(1) + domain-specific tables.
//!
//! ## Retrieval Pattern
//!
//! 1. Vector search over episode summaries: O(log E) via HNSW/Vamana
//! 2. Range scan over events by (episode_id, seq): O(V) where V = events per episode
//!
//! ## Example Queries
//!
//! ```text
//! -- Find similar past tasks
//! SEARCH episodes BY SIMILARITY($query) TOP 5
//!
//! -- Get timeline for an episode
//! SELECT * FROM events WHERE episode_id = $id ORDER BY seq
//!
//! -- Find entities by kind
//! SEARCH entities WHERE kind = 'user' BY SIMILARITY($query) TOP 10
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::soch::{SochSchema, SochType, SochValue};

// ============================================================================
// Episode - A Task/Conversation Run
// ============================================================================

/// Episode types for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeType {
    /// Interactive conversation session
    Conversation,
    /// Autonomous task execution
    Task,
    /// Background workflow
    Workflow,
    /// Debug/testing session
    Debug,
    /// Agent-to-agent interaction
    AgentInteraction,
    /// Other/custom type
    Other,
}

impl EpisodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EpisodeType::Conversation => "conversation",
            EpisodeType::Task => "task",
            EpisodeType::Workflow => "workflow",
            EpisodeType::Debug => "debug",
            EpisodeType::AgentInteraction => "agent_interaction",
            EpisodeType::Other => "other",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "conversation" => EpisodeType::Conversation,
            "task" => EpisodeType::Task,
            "workflow" => EpisodeType::Workflow,
            "debug" => EpisodeType::Debug,
            "agent_interaction" => EpisodeType::AgentInteraction,
            _ => EpisodeType::Other,
        }
    }
}

/// An episode represents a bounded task or conversation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique episode identifier
    pub episode_id: String,
    /// Episode type
    pub episode_type: EpisodeType,
    /// Related entity IDs (users, projects, etc.)
    pub entity_ids: Vec<String>,
    /// Start timestamp (microseconds since epoch)
    pub ts_start: u64,
    /// End timestamp (0 if ongoing)
    pub ts_end: u64,
    /// Natural language summary of the episode
    pub summary: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Embedding vector for semantic search (optional)
    pub embedding: Option<Vec<f32>>,
    /// Custom metadata
    pub metadata: HashMap<String, SochValue>,
}

impl Episode {
    /// Create a new episode
    pub fn new(episode_id: impl Into<String>, episode_type: EpisodeType) -> Self {
        Self {
            episode_id: episode_id.into(),
            episode_type,
            entity_ids: Vec::new(),
            ts_start: Self::now_us(),
            ts_end: 0,
            summary: String::new(),
            tags: Vec::new(),
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    /// Get schema for episodes table
    pub fn schema() -> SochSchema {
        SochSchema::new("episodes")
            .field("episode_id", SochType::Text)
            .field("episode_type", SochType::Text)
            .field("entity_ids", SochType::Array(Box::new(SochType::Text)))
            .field("ts_start", SochType::UInt)
            .field("ts_end", SochType::UInt)
            .field("summary", SochType::Text)
            .field("tags", SochType::Array(Box::new(SochType::Text)))
            .field(
                "embedding",
                SochType::Optional(Box::new(SochType::Array(Box::new(SochType::Float)))),
            )
            .field("metadata", SochType::Object(vec![]))
            .primary_key("episode_id")
            .index("idx_episodes_type", vec!["episode_type".into()], false)
            .index("idx_episodes_ts", vec!["ts_start".into()], false)
    }

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

// ============================================================================
// Event - A Step Within an Episode
// ============================================================================

/// Event roles (who/what triggered the event)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventRole {
    /// User message/action
    User,
    /// Assistant/LLM response
    Assistant,
    /// System event
    System,
    /// Tool invocation
    Tool,
    /// External service
    External,
}

impl EventRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventRole::User => "user",
            EventRole::Assistant => "assistant",
            EventRole::System => "system",
            EventRole::Tool => "tool",
            EventRole::External => "external",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "user" => EventRole::User,
            "assistant" => EventRole::Assistant,
            "system" => EventRole::System,
            "tool" => EventRole::Tool,
            "external" => EventRole::External,
            _ => EventRole::System,
        }
    }
}

/// An event represents a single step within an episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Parent episode ID
    pub episode_id: String,
    /// Sequence number within episode (monotonically increasing)
    pub seq: u64,
    /// Timestamp (microseconds since epoch)
    pub ts: u64,
    /// Who/what triggered this event
    pub role: EventRole,
    /// Tool name (if role == Tool)
    pub tool_name: Option<String>,
    /// Input data in TOON format
    pub input_toon: String,
    /// Output data in TOON format
    pub output_toon: String,
    /// Error message if event failed
    pub error: Option<String>,
    /// Performance metrics
    pub metrics: EventMetrics,
}

/// Performance metrics for an event
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetrics {
    /// Duration in microseconds
    pub duration_us: u64,
    /// Input tokens (if applicable)
    pub input_tokens: Option<u32>,
    /// Output tokens (if applicable)
    pub output_tokens: Option<u32>,
    /// Cost in microdollars (if applicable)
    pub cost_micros: Option<u64>,
}

impl Event {
    /// Create a new event
    pub fn new(episode_id: impl Into<String>, seq: u64, role: EventRole) -> Self {
        Self {
            episode_id: episode_id.into(),
            seq,
            ts: Self::now_us(),
            role,
            tool_name: None,
            input_toon: String::new(),
            output_toon: String::new(),
            error: None,
            metrics: EventMetrics::default(),
        }
    }

    /// Get schema for events table
    pub fn schema() -> SochSchema {
        SochSchema::new("events")
            .field("episode_id", SochType::Text)
            .field("seq", SochType::UInt)
            .field("ts", SochType::UInt)
            .field("role", SochType::Text)
            .field("tool_name", SochType::Optional(Box::new(SochType::Text)))
            .field("input_toon", SochType::Text)
            .field("output_toon", SochType::Text)
            .field("error", SochType::Optional(Box::new(SochType::Text)))
            .field("duration_us", SochType::UInt)
            .field("input_tokens", SochType::Optional(Box::new(SochType::UInt)))
            .field(
                "output_tokens",
                SochType::Optional(Box::new(SochType::UInt)),
            )
            .field("cost_micros", SochType::Optional(Box::new(SochType::UInt)))
            // Composite primary key: (episode_id, seq)
            .primary_key("episode_id")
            .index(
                "idx_events_episode_seq",
                vec!["episode_id".into(), "seq".into()],
                true,
            )
            .index("idx_events_ts", vec!["ts".into()], false)
            .index("idx_events_role", vec!["role".into()], false)
    }

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

// ============================================================================
// Entity - Users, Projects, Documents, Services
// ============================================================================

/// Entity kinds for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    /// Human user
    User,
    /// Project or repository
    Project,
    /// Document or file
    Document,
    /// External service
    Service,
    /// Agent/bot identity
    Agent,
    /// Organization
    Organization,
    /// Custom entity type
    Custom,
}

impl EntityKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityKind::User => "user",
            EntityKind::Project => "project",
            EntityKind::Document => "document",
            EntityKind::Service => "service",
            EntityKind::Agent => "agent",
            EntityKind::Organization => "organization",
            EntityKind::Custom => "custom",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "user" => EntityKind::User,
            "project" => EntityKind::Project,
            "document" => EntityKind::Document,
            "service" => EntityKind::Service,
            "agent" => EntityKind::Agent,
            "organization" => EntityKind::Organization,
            _ => EntityKind::Custom,
        }
    }
}

/// An entity represents a user, project, document, service, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique entity identifier
    pub entity_id: String,
    /// Entity kind/type
    pub kind: EntityKind,
    /// Human-readable name
    pub name: String,
    /// Typed attributes
    pub attributes: HashMap<String, SochValue>,
    /// Embedding vector for semantic search
    pub embedding: Option<Vec<f32>>,
    /// Custom metadata
    pub metadata: HashMap<String, SochValue>,
    /// Created timestamp
    pub created_at: u64,
    /// Last updated timestamp
    pub updated_at: u64,
}

impl Entity {
    /// Create a new entity
    pub fn new(entity_id: impl Into<String>, kind: EntityKind, name: impl Into<String>) -> Self {
        let now = Self::now_us();
        Self {
            entity_id: entity_id.into(),
            kind,
            name: name.into(),
            attributes: HashMap::new(),
            embedding: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Get schema for entities table
    pub fn schema() -> SochSchema {
        SochSchema::new("entities")
            .field("entity_id", SochType::Text)
            .field("kind", SochType::Text)
            .field("name", SochType::Text)
            .field("attributes", SochType::Object(vec![]))
            .field(
                "embedding",
                SochType::Optional(Box::new(SochType::Array(Box::new(SochType::Float)))),
            )
            .field("metadata", SochType::Object(vec![]))
            .field("created_at", SochType::UInt)
            .field("updated_at", SochType::UInt)
            .primary_key("entity_id")
            .index("idx_entities_kind", vec!["kind".into()], false)
            .index("idx_entities_name", vec!["name".into()], false)
    }

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

// ============================================================================
// Table Metadata for MCP Resources
// ============================================================================

/// Table role for semantic metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableRole {
    /// Append-only log (events, traces)
    Log,
    /// Dimension table (entities, users)
    Dimension,
    /// Fact table (metrics, aggregates)
    Fact,
    /// Vector collection for embeddings
    VectorCollection,
    /// Lookup table (config, mappings)
    Lookup,
    /// Core memory schema table
    CoreMemory,
}

impl TableRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            TableRole::Log => "log",
            TableRole::Dimension => "dimension",
            TableRole::Fact => "fact",
            TableRole::VectorCollection => "vector_collection",
            TableRole::Lookup => "lookup",
            TableRole::CoreMemory => "core_memory",
        }
    }
}

/// Semantic metadata for a table (exposed via MCP resources)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSemanticMetadata {
    /// Table name
    pub name: String,
    /// Semantic role of this table
    pub role: TableRole,
    /// Primary key columns
    pub primary_key: Vec<String>,
    /// Clustering key (for ordered access)
    pub cluster_key: Option<Vec<String>>,
    /// Timestamp column (for temporal queries)
    pub ts_column: Option<String>,
    /// Whether backed by a vector index
    pub backed_by_vector_index: bool,
    /// Embedding dimension (if vector collection)
    pub embedding_dimension: Option<usize>,
    /// Human-readable description
    pub description: String,
}

impl TableSemanticMetadata {
    /// Get metadata for episodes table
    pub fn episodes() -> Self {
        Self {
            name: "episodes".to_string(),
            role: TableRole::CoreMemory,
            primary_key: vec!["episode_id".to_string()],
            cluster_key: Some(vec!["ts_start".to_string()]),
            ts_column: Some("ts_start".to_string()),
            backed_by_vector_index: true,
            embedding_dimension: Some(1536), // OpenAI ada-002 default
            description: "Task/conversation runs with summaries and embeddings. Search here to find similar past tasks.".to_string(),
        }
    }

    /// Get metadata for events table
    pub fn events() -> Self {
        Self {
            name: "events".to_string(),
            role: TableRole::Log,
            primary_key: vec!["episode_id".to_string(), "seq".to_string()],
            cluster_key: Some(vec!["episode_id".to_string(), "seq".to_string()]),
            ts_column: Some("ts".to_string()),
            backed_by_vector_index: false,
            embedding_dimension: None,
            description: "Steps within episodes (tool calls, messages). Use LAST N FROM events WHERE episode_id = $id for timeline.".to_string(),
        }
    }

    /// Get metadata for entities table
    pub fn entities() -> Self {
        Self {
            name: "entities".to_string(),
            role: TableRole::Dimension,
            primary_key: vec!["entity_id".to_string()],
            cluster_key: Some(vec!["kind".to_string()]),
            ts_column: Some("updated_at".to_string()),
            backed_by_vector_index: true,
            embedding_dimension: Some(1536),
            description: "Users, projects, documents, services. Search by kind and similarity."
                .to_string(),
        }
    }

    /// Get core memory table metadata
    pub fn core_tables() -> Vec<Self> {
        vec![Self::episodes(), Self::events(), Self::entities()]
    }
}

// ============================================================================
// Memory Store API
// ============================================================================

/// Result of searching episodes
#[derive(Debug, Clone)]
pub struct EpisodeSearchResult {
    pub episode: Episode,
    pub score: f32,
}

/// Result of searching entities
#[derive(Debug, Clone)]
pub struct EntitySearchResult {
    pub entity: Entity,
    pub score: f32,
}

/// Core memory operations trait
pub trait MemoryStore {
    /// Create a new episode
    fn create_episode(&self, episode: &Episode) -> crate::Result<()>;

    /// Get an episode by ID
    fn get_episode(&self, episode_id: &str) -> crate::Result<Option<Episode>>;

    /// Search episodes by vector similarity
    fn search_episodes(&self, query: &str, k: usize) -> crate::Result<Vec<EpisodeSearchResult>>;

    /// Append an event to an episode
    fn append_event(&self, event: &Event) -> crate::Result<()>;

    /// Get timeline for an episode
    fn get_timeline(&self, episode_id: &str, max_events: usize) -> crate::Result<Vec<Event>>;

    /// Create or update an entity
    fn upsert_entity(&self, entity: &Entity) -> crate::Result<()>;

    /// Get an entity by ID
    fn get_entity(&self, entity_id: &str) -> crate::Result<Option<Entity>>;

    /// Search entities by kind and vector similarity
    fn search_entities(
        &self,
        kind: Option<EntityKind>,
        query: &str,
        k: usize,
    ) -> crate::Result<Vec<EntitySearchResult>>;

    /// Get facts for an entity (attributes + linked episodes)
    fn get_entity_facts(&self, entity_id: &str) -> crate::Result<EntityFacts>;
}

/// Facts about an entity
#[derive(Debug, Clone)]
pub struct EntityFacts {
    pub entity: Entity,
    pub recent_episodes: Vec<Episode>,
    pub related_entities: Vec<Entity>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_schema() {
        let schema = Episode::schema();
        assert_eq!(schema.name, "episodes");
        assert!(schema.fields.iter().any(|f| f.name == "episode_id"));
        assert!(schema.fields.iter().any(|f| f.name == "embedding"));
    }

    #[test]
    fn test_event_schema() {
        let schema = Event::schema();
        assert_eq!(schema.name, "events");
        assert!(schema.fields.iter().any(|f| f.name == "episode_id"));
        assert!(schema.fields.iter().any(|f| f.name == "seq"));
    }

    #[test]
    fn test_entity_schema() {
        let schema = Entity::schema();
        assert_eq!(schema.name, "entities");
        assert!(schema.fields.iter().any(|f| f.name == "entity_id"));
        assert!(schema.fields.iter().any(|f| f.name == "kind"));
    }

    #[test]
    fn test_table_metadata() {
        let tables = TableSemanticMetadata::core_tables();
        assert_eq!(tables.len(), 3);
        assert!(tables.iter().any(|t| t.name == "episodes"));
        assert!(tables.iter().any(|t| t.name == "events"));
        assert!(tables.iter().any(|t| t.name == "entities"));
    }
}
