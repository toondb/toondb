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

//! Predefined SochQL Views (Task 7: Stable Identifiers)
//!
//! This module defines canonical views over the core memory schema
//! (episodes, events, entities) that provide:
//!
//! 1. **Stable naming**: LLMs and humans learn consistent patterns
//! 2. **Self-documenting**: View names encode semantics directly
//! 3. **Reduced complexity**: O(1) concepts instead of O(T) tables
//!
//! ## Views
//!
//! | View | Purpose | Common Use |
//! |------|---------|------------|
//! | `conversation_view` | Chat messages | "Show recent conversation" |
//! | `tool_calls_view` | Tool invocations | "What tools were used?" |
//! | `error_view` | Error events | "Show recent errors" |
//! | `episode_summary_view` | Episode overviews | "List recent tasks" |
//! | `entity_directory_view` | Entity catalog | "Find user/project" |

use std::collections::HashMap;

/// A predefined view definition
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// View name (e.g., "conversation_view")
    pub name: &'static str,
    /// Human-readable description
    pub description: &'static str,
    /// SochQL definition (SELECT statement)
    pub definition: &'static str,
    /// Output columns
    pub columns: &'static [&'static str],
    /// Base table(s) this view queries
    pub base_tables: &'static [&'static str],
    /// Usage hints for LLMs
    pub usage_hints: &'static [&'static str],
}

/// Get all predefined views
pub fn get_predefined_views() -> Vec<ViewDefinition> {
    vec![
        // =====================================================================
        // Conversation View - Chat history
        // =====================================================================
        ViewDefinition {
            name: "conversation_view",
            description: "Conversation history with role, content, and timestamp. \
                          Use for displaying chat transcripts or analyzing dialogue patterns.",
            definition: "SELECT episode_id, seq, role, content, ts \
                         FROM events \
                         WHERE event_type = 'message' \
                         ORDER BY ts",
            columns: &["episode_id", "seq", "role", "content", "ts"],
            base_tables: &["events"],
            usage_hints: &[
                "Filter by episode_id to get a single conversation",
                "Use LAST N to get recent messages",
                "Filter by role to get only user or assistant messages",
            ],
        },
        // =====================================================================
        // Tool Calls View - Tool invocations
        // =====================================================================
        ViewDefinition {
            name: "tool_calls_view",
            description: "Tool invocations with inputs, outputs, and timing. \
                          Use for debugging tool usage or analyzing patterns.",
            definition: "SELECT episode_id, seq, tool_name, input, output, \
                         tokens_in, tokens_out, latency_ms, ts \
                         FROM events \
                         WHERE event_type = 'tool_call' \
                         ORDER BY ts",
            columns: &[
                "episode_id",
                "seq",
                "tool_name",
                "input",
                "output",
                "tokens_in",
                "tokens_out",
                "latency_ms",
                "ts",
            ],
            base_tables: &["events"],
            usage_hints: &[
                "Filter by tool_name to analyze specific tool usage",
                "Sum tokens_in + tokens_out for cost analysis",
                "Use AVG(latency_ms) GROUP BY tool_name for performance",
            ],
        },
        // =====================================================================
        // Error View - Errors and failures
        // =====================================================================
        ViewDefinition {
            name: "error_view",
            description: "Error events with context and stack traces. \
                          Use for debugging failures or monitoring error rates.",
            definition: "SELECT episode_id, seq, error_type, message, \
                         stack_trace, ts \
                         FROM events \
                         WHERE event_type = 'error' \
                         ORDER BY ts DESC",
            columns: &[
                "episode_id",
                "seq",
                "error_type",
                "message",
                "stack_trace",
                "ts",
            ],
            base_tables: &["events"],
            usage_hints: &[
                "Use LAST N for recent errors",
                "Group by error_type for error categorization",
                "Join with episodes for context on when errors occurred",
            ],
        },
        // =====================================================================
        // Episode Summary View - Task/conversation overviews
        // =====================================================================
        ViewDefinition {
            name: "episode_summary_view",
            description: "Episode summaries for quick browsing. \
                          Use for listing recent tasks or finding similar past work.",
            definition: "SELECT episode_id, episode_type, summary, tags, \
                         ts_start, ts_end \
                         FROM episodes \
                         ORDER BY ts_start DESC",
            columns: &[
                "episode_id",
                "episode_type",
                "summary",
                "tags",
                "ts_start",
                "ts_end",
            ],
            base_tables: &["episodes"],
            usage_hints: &[
                "Use memory.search_episodes for semantic search",
                "Filter by episode_type for specific task types",
                "Calculate ts_end - ts_start for duration",
            ],
        },
        // =====================================================================
        // Entity Directory View - Entity catalog
        // =====================================================================
        ViewDefinition {
            name: "entity_directory_view",
            description: "Entity directory organized by kind. \
                          Use for finding users, projects, services, or documents.",
            definition: "SELECT entity_id, kind, name, description, updated_at \
                         FROM entities \
                         ORDER BY kind, name",
            columns: &["entity_id", "kind", "name", "description", "updated_at"],
            base_tables: &["entities"],
            usage_hints: &[
                "Use memory.search_entities for semantic search",
                "Filter by kind = 'user' or 'project' etc.",
                "Use memory.get_entity_facts for full details",
            ],
        },
        // =====================================================================
        // Session Timeline View - Full session history
        // =====================================================================
        ViewDefinition {
            name: "session_timeline_view",
            description: "Complete session timeline combining messages and tool calls. \
                          Use for full session replay or context building.",
            definition: "SELECT episode_id, seq, event_type, role, content, \
                         tool_name, ts \
                         FROM events \
                         WHERE event_type IN ('message', 'tool_call', 'tool_result') \
                         ORDER BY episode_id, seq",
            columns: &[
                "episode_id",
                "seq",
                "event_type",
                "role",
                "content",
                "tool_name",
                "ts",
            ],
            base_tables: &["events"],
            usage_hints: &[
                "Filter by episode_id for a single session",
                "Use logs.tail for recent events",
                "Interleaves messages and tool calls chronologically",
            ],
        },
        // =====================================================================
        // Metrics View - Performance and cost metrics
        // =====================================================================
        ViewDefinition {
            name: "metrics_view",
            description: "Aggregated metrics per episode. \
                          Use for cost analysis, performance monitoring.",
            definition: "SELECT episode_id, \
                         COUNT(*) as event_count, \
                         SUM(tokens_in) as total_tokens_in, \
                         SUM(tokens_out) as total_tokens_out, \
                         AVG(latency_ms) as avg_latency_ms, \
                         MIN(ts) as first_event, \
                         MAX(ts) as last_event \
                         FROM events \
                         GROUP BY episode_id",
            columns: &[
                "episode_id",
                "event_count",
                "total_tokens_in",
                "total_tokens_out",
                "avg_latency_ms",
                "first_event",
                "last_event",
            ],
            base_tables: &["events"],
            usage_hints: &[
                "Join with episodes for episode metadata",
                "Use for cost estimation and budgeting",
                "Calculate total_tokens_in + total_tokens_out for total usage",
            ],
        },
    ]
}

/// Get a view by name
pub fn get_view(name: &str) -> Option<ViewDefinition> {
    get_predefined_views().into_iter().find(|v| v.name == name)
}

/// Build a HashMap of views for O(1) lookup
pub fn build_view_map() -> HashMap<&'static str, ViewDefinition> {
    get_predefined_views()
        .into_iter()
        .map(|v| (v.name, v))
        .collect()
}

// ============================================================================
// Naming Conventions
// ============================================================================

/// Standard column naming conventions
pub mod naming {
    /// ID columns use `<entity>_id` format
    pub const ID_SUFFIX: &str = "_id";

    /// Timestamp columns use `ts_<event>` or just `ts`
    pub const TS_PREFIX: &str = "ts_";

    /// Count columns use `<thing>_count`
    pub const COUNT_SUFFIX: &str = "_count";

    /// Boolean columns use `is_<state>` or `has_<thing>`
    pub const BOOL_IS_PREFIX: &str = "is_";
    pub const BOOL_HAS_PREFIX: &str = "has_";

    /// Standard role values
    pub const ROLES: &[&str] = &["user", "assistant", "system", "tool", "external"];

    /// Standard event types
    pub const EVENT_TYPES: &[&str] = &[
        "message",
        "tool_call",
        "tool_result",
        "error",
        "start",
        "end",
        "checkpoint",
        "observation",
    ];

    /// Standard entity kinds
    pub const ENTITY_KINDS: &[&str] = &[
        "user",
        "project",
        "document",
        "service",
        "agent",
        "organization",
        "concept",
    ];

    /// Standard episode types
    pub const EPISODE_TYPES: &[&str] = &[
        "conversation",
        "task",
        "workflow",
        "debug",
        "agent_interaction",
        "batch_job",
    ];
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_predefined_views() {
        let views = get_predefined_views();
        assert!(views.len() >= 5, "Should have at least 5 predefined views");

        // Check conversation_view exists
        let conv = views.iter().find(|v| v.name == "conversation_view");
        assert!(conv.is_some());

        let conv = conv.unwrap();
        assert!(conv.columns.contains(&"role"));
        assert!(conv.columns.contains(&"content"));
    }

    #[test]
    fn test_get_view() {
        let view = get_view("tool_calls_view");
        assert!(view.is_some());

        let view = view.unwrap();
        assert!(view.columns.contains(&"tool_name"));
        assert!(view.base_tables.contains(&"events"));
    }

    #[test]
    fn test_build_view_map() {
        let map = build_view_map();
        assert!(map.contains_key("error_view"));
        assert!(map.contains_key("episode_summary_view"));
    }

    #[test]
    fn test_naming_conventions() {
        use naming::*;

        assert!(ROLES.contains(&"user"));
        assert!(ROLES.contains(&"assistant"));
        assert!(EVENT_TYPES.contains(&"tool_call"));
        assert!(ENTITY_KINDS.contains(&"project"));
    }
}
