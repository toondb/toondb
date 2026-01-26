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

//! ContextQueryBuilder Client API
//!
//! This module provides a fluent API for building and executing
//! CONTEXT SELECT queries from the SochDB client SDK.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb::context_query::ContextQueryBuilder;
//!
//! let result = client.context()
//!     .for_session("sess_123")
//!     .with_budget(4096)
//!     .section("USER", 0)
//!         .get("user.profile.{name, preferences}")
//!         .done()
//!     .section("HISTORY", 1)
//!         .last(10, "events")
//!         .where_eq("type", "tool_call")
//!         .done()
//!     .section("KNOWLEDGE", 2)
//!         .search("docs", "$query_embedding", 5)
//!         .done()
//!     .execute()?;
//!
//! println!("Context tokens: {}", result.token_count);
//! println!("Context:\n{}", result.context);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::connection::{CompareOp, SochConnection, WhereClause};
use sochdb_core::soch::SochValue;

// ============================================================================
// Context Query Builder
// ============================================================================

/// Builder for CONTEXT SELECT queries
pub struct ContextQueryBuilder {
    /// Session ID
    session_id: Option<String>,
    /// Agent ID (alternative to session)
    agent_id: Option<String>,
    /// Token budget
    token_budget: usize,
    /// Include schema in output
    include_schema: bool,
    /// Output format
    format: ContextFormat,
    /// Truncation strategy
    truncation: TruncationStrategy,
    /// Sections
    sections: Vec<ContextSection>,
    /// Session variables for query execution
    variables: HashMap<String, ContextValue>,
    /// Connection for executing DB queries (optional)
    connection: Option<Arc<SochConnection>>,
}

/// Context output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFormat {
    /// TOON format (default)
    Soch,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
    /// Plain text
    Text,
}

/// Truncation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Drop from tail (keep beginning)
    TailDrop,
    /// Drop from head (keep end)
    HeadDrop,
    /// Proportional truncation across sections
    Proportional,
    /// Fail if budget exceeded
    Strict,
}

/// Context value type
#[derive(Debug, Clone)]
pub enum ContextValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Embedding vector
    Embedding(Vec<f32>),
    /// Binary data
    Binary(Vec<u8>),
}

impl ContextQueryBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            session_id: None,
            agent_id: None,
            token_budget: 4096,
            include_schema: true,
            format: ContextFormat::Soch,
            truncation: TruncationStrategy::TailDrop,
            sections: Vec::new(),
            variables: HashMap::new(),
            connection: None,
        }
    }

    /// Create a new builder with a connection for executing DB queries
    pub fn with_connection(conn: Arc<SochConnection>) -> Self {
        Self {
            session_id: None,
            agent_id: None,
            token_budget: 4096,
            include_schema: true,
            format: ContextFormat::Soch,
            truncation: TruncationStrategy::TailDrop,
            sections: Vec::new(),
            variables: HashMap::new(),
            connection: Some(conn),
        }
    }

    /// Attach a connection for executing DB queries
    pub fn connection(mut self, conn: Arc<SochConnection>) -> Self {
        self.connection = Some(conn);
        self
    }

    /// Set session ID for context
    pub fn for_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    /// Set agent ID for context
    pub fn for_agent(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
        self
    }

    /// Set token budget
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.token_budget = budget;
        self
    }

    /// Set whether to include schema
    pub fn include_schema(mut self, include: bool) -> Self {
        self.include_schema = include;
        self
    }

    /// Set output format
    pub fn format(mut self, format: ContextFormat) -> Self {
        self.format = format;
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation = strategy;
        self
    }

    /// Set a variable for query execution
    pub fn set_var(mut self, name: &str, value: ContextValue) -> Self {
        self.variables.insert(name.to_string(), value);
        self
    }

    /// Start building a section
    pub fn section(self, name: &str, priority: i32) -> SectionBuilder {
        SectionBuilder {
            parent: self,
            name: name.to_string(),
            priority,
            content: None,
            filter: None,
            transform: None,
        }
    }

    /// Add a literal text section
    pub fn literal(mut self, name: &str, priority: i32, text: &str) -> Self {
        self.sections.push(ContextSection {
            name: name.to_string(),
            priority,
            content: SectionContent::Literal(text.to_string()),
            filter: None,
            transform: None,
        });
        self
    }

    /// Add a variable reference section
    pub fn variable(mut self, name: &str, priority: i32, var_name: &str) -> Self {
        self.sections.push(ContextSection {
            name: name.to_string(),
            priority,
            content: SectionContent::Variable(var_name.to_string()),
            filter: None,
            transform: None,
        });
        self
    }

    /// Build the context query
    pub fn build(self) -> ContextQuery {
        ContextQuery {
            session_id: self.session_id,
            agent_id: self.agent_id,
            token_budget: self.token_budget,
            include_schema: self.include_schema,
            format: self.format,
            truncation: self.truncation,
            sections: self.sections,
            variables: self.variables,
            connection: self.connection,
        }
    }

    /// Execute the query - uses connection if available for real DB queries
    pub fn execute(self) -> Result<ContextQueryResult, ContextQueryError> {
        let query = self.build();
        query.execute()
    }
}

impl Default for ContextQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Section Builder
// ============================================================================

/// Builder for a context section
pub struct SectionBuilder {
    parent: ContextQueryBuilder,
    name: String,
    priority: i32,
    content: Option<SectionContent>,
    filter: Option<FilterExpr>,
    transform: Option<TransformExpr>,
}

impl SectionBuilder {
    /// Set section to GET a path expression
    pub fn get(mut self, path: &str) -> Self {
        self.content = Some(SectionContent::Get(path.to_string()));
        self
    }

    /// Set section to LAST N rows from table
    pub fn last(mut self, count: usize, table: &str) -> Self {
        self.content = Some(SectionContent::Last {
            count,
            table: table.to_string(),
        });
        self
    }

    /// Set section to SEARCH by similarity
    pub fn search(mut self, collection: &str, query_var: &str, top_k: usize) -> Self {
        self.content = Some(SectionContent::Search {
            collection: collection.to_string(),
            query_var: query_var.to_string(),
            top_k,
            min_score: None,
        });
        self
    }

    /// Set section to SELECT query
    pub fn select(mut self, columns: &[&str], table: &str) -> Self {
        self.content = Some(SectionContent::Select {
            columns: columns.iter().map(|s| s.to_string()).collect(),
            table: table.to_string(),
            limit: None,
        });
        self
    }

    /// Add WHERE equality filter
    pub fn where_eq(mut self, column: &str, value: &str) -> Self {
        let filter = FilterExpr::Eq(column.to_string(), value.to_string());
        self.filter = match self.filter {
            None => Some(filter),
            Some(existing) => Some(FilterExpr::And(vec![existing, filter])),
        };
        self
    }

    /// Add WHERE greater-than filter
    pub fn where_gt(mut self, column: &str, value: i64) -> Self {
        let filter = FilterExpr::Gt(column.to_string(), value);
        self.filter = match self.filter {
            None => Some(filter),
            Some(existing) => Some(FilterExpr::And(vec![existing, filter])),
        };
        self
    }

    /// Add WHERE less-than filter
    pub fn where_lt(mut self, column: &str, value: i64) -> Self {
        let filter = FilterExpr::Lt(column.to_string(), value);
        self.filter = match self.filter {
            None => Some(filter),
            Some(existing) => Some(FilterExpr::And(vec![existing, filter])),
        };
        self
    }

    /// Add WHERE LIKE filter
    pub fn where_like(mut self, column: &str, pattern: &str) -> Self {
        let filter = FilterExpr::Like(column.to_string(), pattern.to_string());
        self.filter = match self.filter {
            None => Some(filter),
            Some(existing) => Some(FilterExpr::And(vec![existing, filter])),
        };
        self
    }

    /// Set LIMIT on section
    pub fn limit(mut self, limit_val: usize) -> Self {
        if let Some(SectionContent::Select { ref mut limit, .. }) = self.content {
            *limit = Some(limit_val);
        }
        self
    }

    /// Set minimum similarity score
    pub fn min_score(mut self, score: f32) -> Self {
        if let Some(SectionContent::Search {
            ref mut min_score, ..
        }) = self.content
        {
            *min_score = Some(score);
        }
        self
    }

    /// Apply summarization transform
    pub fn summarize(mut self, max_tokens: usize) -> Self {
        self.transform = Some(TransformExpr::Summarize(max_tokens));
        self
    }

    /// Apply field projection transform
    pub fn project(mut self, fields: &[&str]) -> Self {
        self.transform = Some(TransformExpr::Project(
            fields.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    /// Finish section and return to parent builder
    pub fn done(mut self) -> ContextQueryBuilder {
        let section = ContextSection {
            name: self.name,
            priority: self.priority,
            content: self
                .content
                .unwrap_or(SectionContent::Literal(String::new())),
            filter: self.filter,
            transform: self.transform,
        };

        self.parent.sections.push(section);
        self.parent
    }
}

// ============================================================================
// Section Content Types
// ============================================================================

/// Section content
#[derive(Debug, Clone)]
pub enum SectionContent {
    /// GET path expression
    Get(String),
    /// LAST N FROM table
    Last { count: usize, table: String },
    /// SEARCH by similarity
    Search {
        collection: String,
        query_var: String,
        top_k: usize,
        min_score: Option<f32>,
    },
    /// SELECT query
    Select {
        columns: Vec<String>,
        table: String,
        limit: Option<usize>,
    },
    /// Literal text
    Literal(String),
    /// Variable reference
    Variable(String),
}

/// Filter expression
#[derive(Debug, Clone)]
pub enum FilterExpr {
    Eq(String, String),
    Gt(String, i64),
    Lt(String, i64),
    Ge(String, i64),
    Le(String, i64),
    Like(String, String),
    In(String, Vec<String>),
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
}

/// Transform expression
#[derive(Debug, Clone)]
pub enum TransformExpr {
    /// Summarize to max tokens
    Summarize(usize),
    /// Project specific fields
    Project(Vec<String>),
    /// Apply template
    Template(String),
}

/// A complete section
#[derive(Debug, Clone)]
pub struct ContextSection {
    pub name: String,
    pub priority: i32,
    pub content: SectionContent,
    pub filter: Option<FilterExpr>,
    pub transform: Option<TransformExpr>,
}

// ============================================================================
// Context Query
// ============================================================================

/// A complete context query
pub struct ContextQuery {
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub token_budget: usize,
    pub include_schema: bool,
    pub format: ContextFormat,
    pub truncation: TruncationStrategy,
    pub sections: Vec<ContextSection>,
    pub variables: HashMap<String, ContextValue>,
    /// Connection for executing DB-backed queries
    connection: Option<Arc<SochConnection>>,
}

impl ContextQuery {
    /// Execute the query
    pub fn execute(&self) -> Result<ContextQueryResult, ContextQueryError> {
        // Sort sections by priority
        let mut sections = self.sections.clone();
        sections.sort_by_key(|s| s.priority);

        // Execute each section
        let mut results = Vec::new();
        let mut total_tokens = 0;

        for section in &sections {
            let content = self.execute_section(section)?;
            let tokens = estimate_tokens(&content);

            // Check budget
            if total_tokens + tokens > self.token_budget {
                match self.truncation {
                    TruncationStrategy::Strict => {
                        return Err(ContextQueryError::BudgetExceeded {
                            budget: self.token_budget,
                            required: total_tokens + tokens,
                        });
                    }
                    TruncationStrategy::TailDrop => {
                        // Truncate this section
                        let remaining = self.token_budget - total_tokens;
                        let truncated = truncate_to_tokens(&content, remaining);
                        results.push(SectionResult {
                            name: section.name.clone(),
                            content: truncated.clone(),
                            tokens: estimate_tokens(&truncated),
                            truncated: true,
                            dropped: false,
                        });
                        break;
                    }
                    TruncationStrategy::HeadDrop => {
                        // Skip this section, try next
                        results.push(SectionResult {
                            name: section.name.clone(),
                            content: String::new(),
                            tokens: 0,
                            truncated: false,
                            dropped: true,
                        });
                        continue;
                    }
                    TruncationStrategy::Proportional => {
                        // Would need two passes - simplify for now
                        let remaining = self.token_budget - total_tokens;
                        let truncated = truncate_to_tokens(&content, remaining);
                        results.push(SectionResult {
                            name: section.name.clone(),
                            content: truncated.clone(),
                            tokens: estimate_tokens(&truncated),
                            truncated: true,
                            dropped: false,
                        });
                        break;
                    }
                }
            } else {
                results.push(SectionResult {
                    name: section.name.clone(),
                    content: content.clone(),
                    tokens,
                    truncated: false,
                    dropped: false,
                });
                total_tokens += tokens;
            }
        }

        // Assemble context
        let context = self.assemble_context(&results);

        Ok(ContextQueryResult {
            context,
            token_count: total_tokens,
            token_budget: self.token_budget,
            sections: results,
            session_id: self.session_id.clone(),
        })
    }

    /// Convert this client query to the canonical ContextSelectQuery AST.
    ///
    /// This provides a unified representation that can be:
    /// - Serialized and stored as a ContextRecipe
    /// - Executed by AgentContextIntegration
    /// - Parsed by SochQL
    pub fn to_canonical(&self) -> sochdb_query::context_query::ContextSelectQuery {
        use sochdb_query::context_query as cq;

        let session = match (&self.session_id, &self.agent_id) {
            (Some(sid), _) => cq::SessionReference::Session(sid.clone()),
            (_, Some(aid)) => cq::SessionReference::Agent(aid.clone()),
            (None, None) => cq::SessionReference::None,
        };

        let truncation = match self.truncation {
            TruncationStrategy::Strict => cq::TruncationStrategy::Fail,
            TruncationStrategy::TailDrop => cq::TruncationStrategy::TailDrop,
            TruncationStrategy::HeadDrop => cq::TruncationStrategy::HeadDrop,
            TruncationStrategy::Proportional => cq::TruncationStrategy::Proportional,
        };

        let format = match self.format {
            ContextFormat::Soch => cq::OutputFormat::Soch,
            ContextFormat::Json => cq::OutputFormat::Json,
            ContextFormat::Markdown => cq::OutputFormat::Markdown,
            ContextFormat::Text => cq::OutputFormat::Soch, // Map plain text to TOON
        };

        let options = cq::ContextQueryOptions {
            token_limit: self.token_budget,
            include_schema: self.include_schema,
            format,
            truncation,
            include_headers: true,
        };

        let sections = self
            .sections
            .iter()
            .map(|s| self.convert_section(s))
            .collect();

        cq::ContextSelectQuery {
            output_name: "context".to_string(),
            session,
            options,
            sections,
        }
    }

    /// Convert a client section to canonical section
    fn convert_section(&self, section: &ContextSection) -> sochdb_query::context_query::ContextSection {
        use sochdb_query::context_query as cq;

        let content = match &section.content {
            SectionContent::Get(path) => cq::SectionContent::Get {
                path: cq::PathExpression::parse(path).unwrap_or_else(|_| cq::PathExpression {
                    segments: path.split('.').map(|s| s.to_string()).collect(),
                    fields: vec![],
                    all_fields: true,
                }),
            },
            SectionContent::Last { count, table } => cq::SectionContent::Last {
                count: *count,
                table: table.clone(),
                where_clause: section.filter.as_ref().map(|f| self.convert_filter(f)),
            },
            SectionContent::Search {
                collection,
                query_var,
                top_k,
                min_score,
            } => cq::SectionContent::Search {
                collection: collection.clone(),
                query: cq::SimilarityQuery::Variable(query_var.clone()),
                top_k: *top_k,
                min_score: *min_score,
            },
            SectionContent::Select {
                columns,
                table,
                limit,
            } => cq::SectionContent::Select {
                columns: columns.clone(),
                table: table.clone(),
                where_clause: section.filter.as_ref().map(|f| self.convert_filter(f)),
                limit: *limit,
            },
            SectionContent::Literal(text) => cq::SectionContent::Literal { value: text.clone() },
            SectionContent::Variable(name) => cq::SectionContent::Variable { name: name.clone() },
        };

        let transform = section.transform.as_ref().map(|t| match t {
            TransformExpr::Summarize(tokens) => cq::SectionTransform::Summarize { max_tokens: *tokens },
            TransformExpr::Project(fields) => cq::SectionTransform::Project { fields: fields.clone() },
            TransformExpr::Template(tpl) => cq::SectionTransform::Template { template: tpl.clone() },
        });

        cq::ContextSection {
            name: section.name.clone(),
            priority: section.priority,
            content,
            transform,
        }
    }

    /// Convert filter expression to canonical WhereClause
    fn convert_filter(&self, filter: &FilterExpr) -> sochdb_query::soch_ql::WhereClause {
        use sochdb_query::soch_ql as tq;

        let (conditions, operator) = match filter {
            FilterExpr::Eq(col, val) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::Eq,
                    value: tq::SochValue::Text(val.clone()),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::Gt(col, val) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::Gt,
                    value: tq::SochValue::Int(*val),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::Lt(col, val) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::Lt,
                    value: tq::SochValue::Int(*val),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::Ge(col, val) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::Ge,
                    value: tq::SochValue::Int(*val),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::Le(col, val) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::Le,
                    value: tq::SochValue::Int(*val),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::Like(col, pat) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::Like,
                    value: tq::SochValue::Text(pat.clone()),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::In(col, vals) => (
                vec![tq::Condition {
                    column: col.clone(),
                    operator: tq::ComparisonOp::In,
                    value: tq::SochValue::Array(vals.iter().map(|v| tq::SochValue::Text(v.clone())).collect()),
                }],
                tq::LogicalOp::And,
            ),
            FilterExpr::And(exprs) => {
                let conditions: Vec<_> = exprs
                    .iter()
                    .flat_map(|e| self.convert_filter(e).conditions)
                    .collect();
                (conditions, tq::LogicalOp::And)
            }
            FilterExpr::Or(exprs) => {
                let conditions: Vec<_> = exprs
                    .iter()
                    .flat_map(|e| self.convert_filter(e).conditions)
                    .collect();
                (conditions, tq::LogicalOp::Or)
            }
        };

        tq::WhereClause { conditions, operator }
    }

    /// Execute a single section
    fn execute_section(&self, section: &ContextSection) -> Result<String, ContextQueryError> {
        match &section.content {
            SectionContent::Literal(text) => Ok(text.clone()),
            SectionContent::Variable(name) => self
                .variables
                .get(name)
                .map(|v| match v {
                    ContextValue::String(s) => s.clone(),
                    ContextValue::Int(i) => i.to_string(),
                    ContextValue::Float(f) => format!("{:.2}", f),
                    ContextValue::Bool(b) => b.to_string(),
                    _ => format!("<{}>", name),
                })
                .ok_or_else(|| ContextQueryError::VariableNotFound(name.clone())),
            SectionContent::Get(path) => self.execute_get(section, path),
            SectionContent::Last { count, table } => self.execute_last(section, *count, table),
            SectionContent::Search {
                collection,
                query_var,
                top_k,
                min_score,
            } => self.execute_search(section, collection, query_var, *top_k, *min_score),
            SectionContent::Select {
                columns,
                table,
                limit,
            } => self.execute_select(section, columns, table, *limit),
        }
    }

    /// Execute GET path expression - resolves path and returns data
    fn execute_get(
        &self,
        section: &ContextSection,
        path: &str,
    ) -> Result<String, ContextQueryError> {
        let conn = match &self.connection {
            Some(c) => c,
            None => {
                // Fallback to placeholder if no connection
                return Ok(format!(
                    "# {}\n{}.data: <no connection - path: {}>\n",
                    section.name, path, path
                ));
            }
        };

        // Use TCH path resolution
        let tch = conn.tch.read();
        let resolution = tch.resolve(path);

        match resolution {
            crate::connection::PathResolution::Value(col_ref) => {
                // Single value - format it
                Ok(format!("# {}\n{}: {}\n", section.name, path, col_ref.name))
            }
            crate::connection::PathResolution::Array { schema, columns } => {
                // Array/table - format as TOON
                let mut output = format!("# {}\n", section.name);
                output.push_str(&format!(
                    "{}[{}]{{{}}}\n",
                    schema.name,
                    0, // Would need row count
                    columns
                        .iter()
                        .map(|c| c.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                Ok(output)
            }
            crate::connection::PathResolution::Partial { remaining } => Ok(format!(
                "# {}\n{}: <partial match, remaining: {}>\n",
                section.name, path, remaining
            )),
            crate::connection::PathResolution::NotFound => {
                Ok(format!("# {}\n{}: <not found>\n", section.name, path))
            }
        }
    }

    /// Execute LAST N FROM table - fetches most recent rows
    fn execute_last(
        &self,
        section: &ContextSection,
        count: usize,
        table: &str,
    ) -> Result<String, ContextQueryError> {
        let conn = match &self.connection {
            Some(c) => c,
            None => {
                // Fallback to placeholder if no connection
                let filter_str = section
                    .filter
                    .as_ref()
                    .map(|f| format!(" WHERE {:?}", f))
                    .unwrap_or_default();
                return Ok(format!(
                    "# {}\n{}[{}]{{...}}: (last {} rows{})\n",
                    section.name, table, count, count, filter_str
                ));
            }
        };

        // Convert filter to WhereClause
        let where_clause = section.filter.as_ref().map(|f| self.filter_to_where(f));

        // Query with ORDER BY created_at DESC (or similar timestamp field)
        // For now, we'll assume tables have a sortable field
        let tch = conn.tch.read();
        let cursor = tch.select(
            table,
            &[], // All columns
            where_clause.as_ref(),
            Some(&("created_at".to_string(), false)), // Descending order
            Some(count),
            None,
        );

        // Format results as TOON
        self.format_cursor_as_toon(&section.name, table, cursor)
    }

    /// Execute SELECT with specific columns
    fn execute_select(
        &self,
        section: &ContextSection,
        columns: &[String],
        table: &str,
        limit: Option<usize>,
    ) -> Result<String, ContextQueryError> {
        let conn = match &self.connection {
            Some(c) => c,
            None => {
                // Fallback to placeholder if no connection
                let cols = columns.join(", ");
                let limit_str = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
                return Ok(format!(
                    "# {}\nSELECT {} FROM {}{}\n",
                    section.name, cols, table, limit_str
                ));
            }
        };

        // Convert filter to WhereClause
        let where_clause = section.filter.as_ref().map(|f| self.filter_to_where(f));

        // Query the table
        let tch = conn.tch.read();
        let cursor = tch.select(
            table,
            columns,
            where_clause.as_ref(),
            None, // No ordering
            limit,
            None,
        );

        // Format results as TOON
        self.format_cursor_as_toon(&section.name, table, cursor)
    }

    /// Execute SEARCH by vector similarity
    fn execute_search(
        &self,
        section: &ContextSection,
        collection: &str,
        query_var: &str,
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<String, ContextQueryError> {
        // Get the embedding from variables
        let embedding = match self.variables.get(query_var) {
            Some(ContextValue::Embedding(v)) => v.clone(),
            Some(_) => {
                return Err(ContextQueryError::TypeMismatch {
                    expected: "embedding".to_string(),
                    found: "other".to_string(),
                });
            }
            None => {
                // Fallback to placeholder if no embedding variable
                return Ok(format!(
                    "# {}\n{}[{}]{{...}}: (top {} by similarity to ${})\n",
                    section.name, collection, top_k, top_k, query_var
                ));
            }
        };

        let _conn = match &self.connection {
            Some(c) => c,
            None => {
                return Ok(format!(
                    "# {}\n{}[{}]{{...}}: (top {} by similarity - no connection)\n",
                    section.name, collection, top_k, top_k
                ));
            }
        };

        // TODO: Use vector collection search when available
        // For now, format as placeholder with embedding info
        let _min_score = min_score.unwrap_or(0.0);
        Ok(format!(
            "# {}\n{}[{}]{{...}}: (top {} by similarity, embedding dim={})\n",
            section.name,
            collection,
            top_k,
            top_k,
            embedding.len()
        ))
    }

    /// Convert FilterExpr to WhereClause (supports full boolean expressions)
    fn filter_to_where(&self, filter: &FilterExpr) -> WhereClause {
        match filter {
            FilterExpr::Eq(col, val) => WhereClause::Simple {
                field: col.clone(),
                op: CompareOp::Eq,
                value: SochValue::Text(val.clone()),
            },
            FilterExpr::Gt(col, val) => WhereClause::Simple {
                field: col.clone(),
                op: CompareOp::Gt,
                value: SochValue::Int(*val),
            },
            FilterExpr::Lt(col, val) => WhereClause::Simple {
                field: col.clone(),
                op: CompareOp::Lt,
                value: SochValue::Int(*val),
            },
            FilterExpr::Ge(col, val) => WhereClause::Simple {
                field: col.clone(),
                op: CompareOp::Ge,
                value: SochValue::Int(*val),
            },
            FilterExpr::Le(col, val) => WhereClause::Simple {
                field: col.clone(),
                op: CompareOp::Le,
                value: SochValue::Int(*val),
            },
            FilterExpr::Like(col, val) => WhereClause::Simple {
                field: col.clone(),
                op: CompareOp::Like,
                value: SochValue::Text(val.clone()),
            },
            FilterExpr::In(col, vals) => {
                // Full IN clause support with all values
                WhereClause::In {
                    field: col.clone(),
                    values: vals.iter().map(|v| SochValue::Text(v.clone())).collect(),
                    negated: false,
                }
            }
            FilterExpr::And(filters) => {
                // Full AND support: convert all child filters
                let clauses: Vec<WhereClause> = filters
                    .iter()
                    .map(|f| self.filter_to_where(f))
                    .collect();
                if clauses.is_empty() {
                    // Empty AND is always true - use a tautology
                    WhereClause::Simple {
                        field: String::new(),
                        op: CompareOp::Eq,
                        value: SochValue::Bool(true),
                    }
                } else if clauses.len() == 1 {
                    clauses.into_iter().next().unwrap()
                } else {
                    WhereClause::And(clauses)
                }
            }
            FilterExpr::Or(filters) => {
                // Full OR support: convert all child filters
                let clauses: Vec<WhereClause> = filters
                    .iter()
                    .map(|f| self.filter_to_where(f))
                    .collect();
                if clauses.is_empty() {
                    // Empty OR is always false - use a contradiction
                    WhereClause::Simple {
                        field: String::new(),
                        op: CompareOp::Ne,
                        value: SochValue::Bool(true),
                    }
                } else if clauses.len() == 1 {
                    clauses.into_iter().next().unwrap()
                } else {
                    WhereClause::Or(clauses)
                }
            }
        }
    }

    /// Format cursor results as TOON format
    fn format_cursor_as_toon(
        &self,
        section_name: &str,
        table: &str,
        mut cursor: crate::connection::SochCursor,
    ) -> Result<String, ContextQueryError> {
        // Collect all rows from cursor
        let mut rows = Vec::new();
        while let Some(row) = cursor.next() {
            rows.push(row);
        }

        if rows.is_empty() {
            return Ok(format!("# {}\n{}[0]{{}}\n", section_name, table));
        }

        let mut output = format!("# {}\n", section_name);

        // Get field names from first row
        let fields: Vec<&String> = if let Some(first) = rows.first() {
            first.keys().collect()
        } else {
            vec![]
        };

        // Format based on output format
        match self.format {
            ContextFormat::Soch => {
                output.push_str(&format!(
                    "{}[{}]{{{}}}\n",
                    table,
                    rows.len(),
                    fields
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));

                for row in &rows {
                    let values: Vec<String> = fields
                        .iter()
                        .filter_map(|f| row.get(*f).map(format_soch_value))
                        .collect();
                    output.push_str(&format!("  [{}]\n", values.join(", ")));
                }
            }
            ContextFormat::Json => {
                output.push_str("[\n");
                for (i, row) in rows.iter().enumerate() {
                    output.push_str("  {");
                    let pairs: Vec<String> = row
                        .iter()
                        .map(|(k, v)| format!("\"{}\": {}", k, format_json_value(v)))
                        .collect();
                    output.push_str(&pairs.join(", "));
                    output.push('}');
                    if i < rows.len() - 1 {
                        output.push(',');
                    }
                    output.push('\n');
                }
                output.push_str("]\n");
            }
            ContextFormat::Markdown => {
                // Table header
                output.push_str("| ");
                output.push_str(
                    &fields
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(" | "),
                );
                output.push_str(" |\n");
                output.push_str("| ");
                output.push_str(&fields.iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
                output.push_str(" |\n");

                for row in &rows {
                    output.push_str("| ");
                    let values: Vec<String> = fields
                        .iter()
                        .filter_map(|f| row.get(*f).map(format_soch_value))
                        .collect();
                    output.push_str(&values.join(" | "));
                    output.push_str(" |\n");
                }
            }
            ContextFormat::Text => {
                for row in &rows {
                    for (k, v) in row {
                        output.push_str(&format!("{}: {}\n", k, format_soch_value(v)));
                    }
                    output.push('\n');
                }
            }
        }

        Ok(output)
    }

    /// Assemble the final context from section results
    fn assemble_context(&self, results: &[SectionResult]) -> String {
        let mut context = String::new();

        if self.include_schema {
            context.push_str("# Context\n");
            context.push_str(&format!(
                "session: {}\n",
                self.session_id.as_deref().unwrap_or("none")
            ));
            context.push_str(&format!("budget: {} tokens\n\n", self.token_budget));
        }

        for result in results {
            if !result.dropped && !result.content.is_empty() {
                context.push_str(&result.content);
                context.push('\n');
            }
        }

        context
    }
}

// ============================================================================
// Query Result
// ============================================================================

/// Result of executing a context query
#[derive(Debug, Clone)]
pub struct ContextQueryResult {
    /// Assembled context
    pub context: String,
    /// Total token count
    pub token_count: usize,
    /// Token budget
    pub token_budget: usize,
    /// Section results
    pub sections: Vec<SectionResult>,
    /// Session ID (if any)
    pub session_id: Option<String>,
}

/// Result of a single section
#[derive(Debug, Clone)]
pub struct SectionResult {
    /// Section name
    pub name: String,
    /// Section content
    pub content: String,
    /// Token count
    pub tokens: usize,
    /// Was truncated
    pub truncated: bool,
    /// Was dropped
    pub dropped: bool,
}

impl ContextQueryResult {
    /// Get token utilization percentage
    pub fn utilization(&self) -> f64 {
        (self.token_count as f64 / self.token_budget as f64) * 100.0
    }

    /// Get list of included sections
    pub fn included_sections(&self) -> Vec<&str> {
        self.sections
            .iter()
            .filter(|s| !s.dropped)
            .map(|s| s.name.as_str())
            .collect()
    }

    /// Get list of dropped sections
    pub fn dropped_sections(&self) -> Vec<&str> {
        self.sections
            .iter()
            .filter(|s| s.dropped)
            .map(|s| s.name.as_str())
            .collect()
    }

    /// Get list of truncated sections
    pub fn truncated_sections(&self) -> Vec<&str> {
        self.sections
            .iter()
            .filter(|s| s.truncated)
            .map(|s| s.name.as_str())
            .collect()
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Context query error
#[derive(Debug, Clone)]
pub enum ContextQueryError {
    /// Budget exceeded
    BudgetExceeded { budget: usize, required: usize },
    /// Variable not found
    VariableNotFound(String),
    /// Section execution failed
    SectionFailed { section: String, error: String },
    /// Invalid query
    InvalidQuery(String),
    /// Database error
    DatabaseError(String),
    /// Type mismatch
    TypeMismatch { expected: String, found: String },
}

impl std::fmt::Display for ContextQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExceeded { budget, required } => {
                write!(
                    f,
                    "token budget exceeded: {} required, {} available",
                    required, budget
                )
            }
            Self::VariableNotFound(name) => write!(f, "variable not found: ${}", name),
            Self::SectionFailed { section, error } => {
                write!(f, "section '{}' failed: {}", section, error)
            }
            Self::InvalidQuery(msg) => write!(f, "invalid query: {}", msg),
            Self::DatabaseError(msg) => write!(f, "database error: {}", msg),
            Self::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for ContextQueryError {}

// ============================================================================
// Token Estimation (delegate to sochdb-query)
// ============================================================================

use sochdb_query::token_budget::TokenEstimator;

/// Shared token estimator for consistent estimates across the client
fn get_estimator() -> TokenEstimator {
    TokenEstimator::default()
}

/// Estimate tokens for text using the canonical TokenEstimator
fn estimate_tokens(text: &str) -> usize {
    get_estimator().estimate_text(text)
}

/// Truncate text to fit within token budget using binary search
fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    get_estimator().truncate_to_tokens(text, max_tokens)
}

/// Format a SochValue for TOON output
fn format_soch_value(v: &SochValue) -> String {
    match v {
        SochValue::Null => "∅".to_string(),
        SochValue::Int(i) => i.to_string(),
        SochValue::UInt(u) => u.to_string(),
        SochValue::Float(f) => format!("{:.6}", f),
        SochValue::Text(s) => {
            if s.contains(',') || s.contains(';') || s.contains('\n') {
                format!("\"{}\"", s.replace('"', "\\\""))
            } else {
                s.clone()
            }
        }
        SochValue::Bool(b) => if *b { "T" } else { "F" }.to_string(),
        SochValue::Binary(b) => format!("b64:<{}bytes>", b.len()),
        SochValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_soch_value).collect();
            format!("[{}]", items.join(","))
        }
        SochValue::Object(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}:{}", k, format_soch_value(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
        SochValue::Ref { table, id } => format!("ref({},{})", table, id),
    }
}

/// Format a SochValue for JSON output
fn format_json_value(v: &SochValue) -> String {
    match v {
        SochValue::Null => "null".to_string(),
        SochValue::Int(i) => i.to_string(),
        SochValue::UInt(u) => u.to_string(),
        SochValue::Float(f) => format!("{}", f),
        SochValue::Text(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        SochValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        SochValue::Binary(b) => format!("\"<binary:{}>\"", b.len()),
        SochValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_json_value).collect();
            format!("[{}]", items.join(","))
        }
        SochValue::Object(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("\"{}\":{}", k, format_json_value(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
        SochValue::Ref { table, id } => format!("{{\"$ref\":\"{}\",\"id\":{}}}", table, id),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let query = ContextQueryBuilder::new()
            .for_session("sess_123")
            .with_budget(4096)
            .literal("SYSTEM", -1, "You are a helpful assistant")
            .build();

        assert_eq!(query.session_id, Some("sess_123".to_string()));
        assert_eq!(query.token_budget, 4096);
        assert_eq!(query.sections.len(), 1);
    }

    #[test]
    fn test_section_builder() {
        let query = ContextQueryBuilder::new()
            .section("USER", 0)
            .get("user.profile.{name, email}")
            .done()
            .section("HISTORY", 1)
            .last(10, "events")
            .where_eq("type", "tool_call")
            .done()
            .section("DOCS", 2)
            .search("knowledge_base", "query_embedding", 5)
            .min_score(0.7)
            .done()
            .build();

        assert_eq!(query.sections.len(), 3);

        // Check priorities
        assert_eq!(query.sections[0].priority, 0);
        assert_eq!(query.sections[1].priority, 1);
        assert_eq!(query.sections[2].priority, 2);
    }

    #[test]
    fn test_execute_with_literals() {
        let result = ContextQueryBuilder::new()
            .with_budget(1000)
            .literal("SYSTEM", 0, "You are a helpful assistant")
            .literal("USER", 1, "Hello, how are you?")
            .execute()
            .unwrap();

        assert!(result.token_count < 1000);
        assert!(result.context.contains("You are a helpful assistant"));
        assert!(result.context.contains("Hello, how are you?"));
    }

    #[test]
    fn test_variable_resolution() {
        let result = ContextQueryBuilder::new()
            .set_var("user_name", ContextValue::String("Alice".to_string()))
            .variable("GREETING", 0, "user_name")
            .execute()
            .unwrap();

        assert!(result.context.contains("Alice"));
    }

    #[test]
    fn test_budget_exceeded_strict() {
        let result = ContextQueryBuilder::new()
            .with_budget(10) // Very small budget
            .truncation(TruncationStrategy::Strict)
            .literal("LONG", 0, &"x".repeat(1000))
            .execute();

        assert!(matches!(
            result,
            Err(ContextQueryError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_budget_truncation() {
        let long_text = "x".repeat(1000);
        let result = ContextQueryBuilder::new()
            .with_budget(100)
            .truncation(TruncationStrategy::TailDrop)
            .literal("LONG", 0, &long_text)
            .execute()
            .unwrap();

        assert!(result.token_count <= 100);
        assert!(result.sections[0].truncated);
    }

    #[test]
    fn test_format_options() {
        let query = ContextQueryBuilder::new()
            .format(ContextFormat::Markdown)
            .include_schema(false)
            .build();

        assert_eq!(query.format, ContextFormat::Markdown);
        assert!(!query.include_schema);
    }

    #[test]
    fn test_complex_filters() {
        let query = ContextQueryBuilder::new()
            .section("DATA", 0)
            .select(&["id", "name", "score"], "users")
            .where_gt("score", 80)
            .where_like("name", "A%")
            .limit(10)
            .done()
            .build();

        let section = &query.sections[0];
        assert!(matches!(&section.filter, Some(FilterExpr::And(_))));
    }

    #[test]
    fn test_result_methods() {
        let result = ContextQueryBuilder::new()
            .with_budget(1000)
            .literal("A", 0, "content a")
            .literal("B", 1, "content b")
            .execute()
            .unwrap();

        let included = result.included_sections();
        assert_eq!(included.len(), 2);
        assert!(included.contains(&"A"));
        assert!(included.contains(&"B"));

        assert!(result.dropped_sections().is_empty());
        assert!(result.truncated_sections().is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn test_truncate_to_tokens() {
        let text = "This is a long text that needs truncation";
        let truncated = truncate_to_tokens(text, 5);

        assert!(truncated.len() < text.len());
        assert!(truncated.ends_with("..."));
    }
}
