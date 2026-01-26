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

//! LLM-Native CONTEXT SELECT Query
//!
//! This module implements the CONTEXT SELECT query extension for assembling
//! LLM context from multiple data sources with token budget enforcement.
//!
//! ## Grammar
//!
//! ```text
//! CONTEXT SELECT prompt_context
//! FROM session($SESSION_ID)
//! WITH (token_limit = 2048, include_schema = true)
//! SECTIONS (
//!     USER PRIORITY 0: GET user.profile.{name, preferences},
//!     HISTORY PRIORITY 1: LAST 10 FROM traces WHERE type = 'tool_call',
//!     KNOWLEDGE PRIORITY 2: SEARCH knowledge_base BY SIMILARITY($query) TOP 5
//! );
//! ```
//!
//! ## Token Budget Algorithm
//!
//! Sections are packed in priority order (lower = higher priority):
//!
//! 1. Sort sections by priority
//! 2. For each section:
//!    a. Estimate token cost
//!    b. If fits in remaining budget, include fully
//!    c. Else, try to include truncated version
//! 3. Return assembled context in TOON format
//!
//! ## Section Types
//!
//! - `GET`: Fetch data from path expression
//! - `LAST N FROM`: Fetch recent rows from table
//! - `SEARCH BY SIMILARITY`: Vector similarity search
//! - `SELECT`: Standard SQL subquery

use crate::token_budget::{BudgetSection, TokenBudgetConfig, TokenBudgetEnforcer, TokenEstimator};
use crate::soch_ql::{ComparisonOp, Condition, LogicalOp, SochValue, WhereClause};
use std::collections::HashMap;

// ============================================================================
// Context Query AST
// ============================================================================

/// A CONTEXT SELECT query
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextSelectQuery {
    /// Output variable name (e.g., "prompt_context")
    pub output_name: String,
    /// Session reference
    pub session: SessionReference,
    /// Query options
    pub options: ContextQueryOptions,
    /// Sections to include
    pub sections: Vec<ContextSection>,
}

/// Session reference (FROM clause)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SessionReference {
    /// session($SESSION_ID)
    Session(String),
    /// agent($AGENT_ID)
    Agent(String),
    /// No session binding
    None,
}

/// Query-level options (WITH clause)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextQueryOptions {
    /// Token limit for entire context
    pub token_limit: usize,
    /// Include schema in output
    pub include_schema: bool,
    /// Output format (default: TOON)
    pub format: OutputFormat,
    /// Truncation strategy
    pub truncation: TruncationStrategy,
    /// Include section headers
    pub include_headers: bool,
}

impl Default for ContextQueryOptions {
    fn default() -> Self {
        Self {
            token_limit: 4096,
            include_schema: true,
            format: OutputFormat::Soch,
            truncation: TruncationStrategy::TailDrop,
            include_headers: true,
        }
    }
}

/// Output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OutputFormat {
    /// TOON format (default)
    Soch,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
}

/// Truncation strategy when budget exceeded
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TruncationStrategy {
    /// Drop from tail (keep head)
    TailDrop,
    /// Drop from head (keep tail)
    HeadDrop,
    /// Proportional truncation
    Proportional,
    /// Fail on budget exceeded
    Fail,
}

// ============================================================================
// Context Sections
// ============================================================================

/// A section in CONTEXT SELECT
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextSection {
    /// Section name (e.g., "USER", "HISTORY", "KNOWLEDGE")
    pub name: String,
    /// Priority (lower = higher priority)
    pub priority: i32,
    /// Section content definition
    pub content: SectionContent,
    /// Optional transformer
    pub transform: Option<SectionTransform>,
}

/// Content definition for a section
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SectionContent {
    /// GET path expression
    /// Example: GET user.profile.{name, preferences}
    Get { path: PathExpression },

    /// LAST N FROM table with optional WHERE
    /// Example: LAST 10 FROM traces WHERE type = 'tool_call'
    Last {
        count: usize,
        table: String,
        where_clause: Option<WhereClause>,
    },

    /// SEARCH by similarity
    /// Example: SEARCH knowledge_base BY SIMILARITY($query) TOP 5
    Search {
        collection: String,
        query: SimilarityQuery,
        top_k: usize,
        min_score: Option<f32>,
    },

    /// Standard SELECT subquery
    Select {
        columns: Vec<String>,
        table: String,
        where_clause: Option<WhereClause>,
        limit: Option<usize>,
    },

    /// Literal value
    Literal { value: String },

    /// Variable reference
    Variable { name: String },

    /// TOOL_REGISTRY: List of available tools with schemas
    /// Example: TOOL_REGISTRY
    ToolRegistry {
        /// Include only these tools (empty = all)
        include: Vec<String>,
        /// Exclude these tools
        exclude: Vec<String>,
        /// Include full JSON schema
        include_schema: bool,
    },

    /// TOOL_CALLS: Recent tool call history
    /// Example: TOOL_CALLS LAST 10 WHERE status = 'success'
    ToolCalls {
        /// Maximum number of calls to include
        count: usize,
        /// Filter by tool name
        tool_filter: Option<String>,
        /// Filter by status (success, error, pending)
        status_filter: Option<String>,
        /// Include tool outputs in context
        include_outputs: bool,
    },
}

/// Path expression for GET
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PathExpression {
    /// Path segments (e.g., ["user", "profile"])
    pub segments: Vec<String>,
    /// Field projection (e.g., ["name", "preferences"])
    pub fields: Vec<String>,
    /// Include all fields if empty
    pub all_fields: bool,
}

impl PathExpression {
    /// Parse a path expression string
    /// Format: "path.to.node.{field1, field2}" or "path.to.node.**"
    pub fn parse(input: &str) -> Result<Self, ContextParseError> {
        let input = input.trim();

        // Check for field projection
        if let Some(brace_start) = input.find('{') {
            if !input.ends_with('}') {
                return Err(ContextParseError::InvalidPath(
                    "unclosed field projection".to_string(),
                ));
            }

            let path_part = &input[..brace_start].trim_end_matches('.');
            let fields_part = &input[brace_start + 1..input.len() - 1];

            let segments: Vec<String> = path_part.split('.').map(|s| s.to_string()).collect();
            let fields: Vec<String> = fields_part
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            Ok(PathExpression {
                segments,
                fields,
                all_fields: false,
            })
        } else if let Some(path_part) = input.strip_suffix(".**") {
            // Glob for all descendants
            let segments: Vec<String> = path_part.split('.').map(|s| s.to_string()).collect();

            Ok(PathExpression {
                segments,
                fields: vec![],
                all_fields: true,
            })
        } else {
            // Simple path
            let segments: Vec<String> = input.split('.').map(|s| s.to_string()).collect();

            Ok(PathExpression {
                segments,
                fields: vec![],
                all_fields: true,
            })
        }
    }

    /// Convert to path string
    pub fn to_path_string(&self) -> String {
        let base = self.segments.join(".");
        if self.all_fields {
            format!("{}.**", base)
        } else if !self.fields.is_empty() {
            format!("{}.{{{}}}", base, self.fields.join(", "))
        } else {
            base
        }
    }
}

/// Similarity query for SEARCH
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SimilarityQuery {
    /// Reference to session variable
    Variable(String),
    /// Inline embedding vector
    Embedding(Vec<f32>),
    /// Text to embed at query time
    Text(String),
}

/// Transform to apply to section output
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SectionTransform {
    /// Summarize content
    Summarize { max_tokens: usize },
    /// Extract specific fields
    Project { fields: Vec<String> },
    /// Apply template
    Template { template: String },
    /// Custom function
    Custom { function: String },
}

// ============================================================================
// Persisted Context Recipes
// ============================================================================

/// A reusable context recipe that can be saved, versioned, and bound to sessions.
///
/// Recipes encapsulate a CONTEXT SELECT query pattern that can be:
/// - Stored in the database with versioning
/// - Bound to sessions for consistent context assembly
/// - Shared across agents with the same context needs
/// - A/B tested by swapping recipe versions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextRecipe {
    /// Unique identifier for this recipe
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this recipe produces
    pub description: String,
    /// Version number (semantic versioning recommended)
    pub version: String,
    /// The context select query this recipe represents
    pub query: ContextSelectQuery,
    /// Recipe metadata
    pub metadata: RecipeMetadata,
    /// Session binding (optional)
    pub session_binding: Option<SessionBinding>,
}

/// Metadata for a context recipe
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RecipeMetadata {
    /// Who created this recipe
    pub author: Option<String>,
    /// When this recipe was created
    pub created_at: Option<String>,
    /// When this recipe was last modified
    pub updated_at: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Usage count for analytics
    pub usage_count: u64,
    /// Average token usage
    pub avg_tokens: Option<f32>,
}

/// How a recipe is bound to sessions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SessionBinding {
    /// Bind to a specific session ID
    Session(String),
    /// Bind to all sessions for an agent
    Agent(String),
    /// Bind to sessions matching a pattern (glob)
    Pattern(String),
    /// No binding (recipe is standalone)
    None,
}

/// Repository for storing and retrieving context recipes
pub struct ContextRecipeStore {
    /// Recipes indexed by ID
    recipes: std::sync::RwLock<HashMap<String, ContextRecipe>>,
    /// Version history for each recipe ID
    versions: std::sync::RwLock<HashMap<String, Vec<String>>>,
}

impl ContextRecipeStore {
    /// Create a new recipe store
    pub fn new() -> Self {
        Self {
            recipes: std::sync::RwLock::new(HashMap::new()),
            versions: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Save a new recipe version
    pub fn save(&self, recipe: ContextRecipe) -> Result<(), String> {
        let mut recipes = self.recipes.write().map_err(|e| e.to_string())?;
        let mut versions = self.versions.write().map_err(|e| e.to_string())?;

        let key = format!("{}:{}", recipe.id, recipe.version);
        recipes.insert(key.clone(), recipe.clone());

        versions
            .entry(recipe.id.clone())
            .or_default()
            .push(recipe.version.clone());

        Ok(())
    }

    /// Get the latest version of a recipe
    pub fn get_latest(&self, recipe_id: &str) -> Option<ContextRecipe> {
        let versions = self.versions.read().ok()?;
        let latest_version = versions.get(recipe_id)?.last()?;

        let recipes = self.recipes.read().ok()?;
        let key = format!("{}:{}", recipe_id, latest_version);
        recipes.get(&key).cloned()
    }

    /// Get a specific version of a recipe
    pub fn get_version(&self, recipe_id: &str, version: &str) -> Option<ContextRecipe> {
        let recipes = self.recipes.read().ok()?;
        let key = format!("{}:{}", recipe_id, version);
        recipes.get(&key).cloned()
    }

    /// List all versions of a recipe
    pub fn list_versions(&self, recipe_id: &str) -> Vec<String> {
        self.versions
            .read()
            .ok()
            .and_then(|v| v.get(recipe_id).cloned())
            .unwrap_or_default()
    }

    /// Find recipes matching a session binding
    pub fn find_by_session(&self, session_id: &str) -> Vec<ContextRecipe> {
        let recipes = match self.recipes.read() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        recipes
            .values()
            .filter(|r| match &r.session_binding {
                Some(SessionBinding::Session(sid)) => sid == session_id,
                Some(SessionBinding::Pattern(pattern)) => {
                    glob_match(pattern, session_id)
                }
                _ => false,
            })
            .cloned()
            .collect()
    }

    /// Find recipes for an agent
    pub fn find_by_agent(&self, agent_id: &str) -> Vec<ContextRecipe> {
        let recipes = match self.recipes.read() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        recipes
            .values()
            .filter(|r| matches!(&r.session_binding, Some(SessionBinding::Agent(aid)) if aid == agent_id))
            .cloned()
            .collect()
    }
}

impl Default for ContextRecipeStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple glob matching for session patterns
fn glob_match(pattern: &str, input: &str) -> bool {
    // Simple implementation: * matches any characters
    if pattern == "*" {
        return true;
    }
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return input.starts_with(parts[0]) && input.ends_with(parts[1]);
        }
    }
    pattern == input
}

// ============================================================================
// Vector Index Trait (Task 6 - CONTEXT SELECT SEARCH)
// ============================================================================

/// Result of a vector similarity search
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Unique identifier of the matched item
    pub id: String,
    /// Similarity score (0.0 to 1.0, higher = more similar)
    pub score: f32,
    /// The matched content/document
    pub content: String,
    /// Optional metadata
    pub metadata: HashMap<String, SochValue>,
}

/// Trait for vector index implementations
///
/// This allows plugging in different vector index backends:
/// - HNSW from sochdb-index
/// - External vector databases (Pinecone, Milvus, etc.)
/// - Simple brute-force for small collections
pub trait VectorIndex: Send + Sync {
    /// Search for k nearest neighbors to the query vector
    fn search_by_embedding(
        &self,
        collection: &str,
        embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String>;

    /// Search by text (index handles embedding generation)
    fn search_by_text(
        &self,
        collection: &str,
        text: &str,
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String>;

    /// Get index statistics
    fn stats(&self, collection: &str) -> Option<VectorIndexStats>;
}

/// Statistics about a vector index collection
#[derive(Debug, Clone)]
pub struct VectorIndexStats {
    /// Number of vectors in the collection
    pub vector_count: usize,
    /// Dimension of vectors
    pub dimension: usize,
    /// Distance metric used
    pub metric: String,
}

/// In-memory vector index for CONTEXT SELECT SEARCH
///
/// This provides a simple implementation that can be used directly
/// or wrapped around the sochdb-index HNSW implementation.
pub struct SimpleVectorIndex {
    /// Collections: name -> (vectors, metadata, dimension)
    collections: std::sync::RwLock<HashMap<String, VectorCollection>>,
}

/// A collection of vectors
struct VectorCollection {
    /// Vectors stored as (id, vector, content, metadata)
    #[allow(clippy::type_complexity)]
    vectors: Vec<(String, Vec<f32>, String, HashMap<String, SochValue>)>,
    /// Vector dimension
    dimension: usize,
}

impl SimpleVectorIndex {
    /// Create a new empty vector index
    pub fn new() -> Self {
        Self {
            collections: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Create or get a collection
    pub fn create_collection(&self, name: &str, dimension: usize) {
        let mut collections = self.collections.write().unwrap();
        collections
            .entry(name.to_string())
            .or_insert_with(|| VectorCollection {
                vectors: Vec::new(),
                dimension,
            });
    }

    /// Insert a vector into a collection
    pub fn insert(
        &self,
        collection: &str,
        id: String,
        vector: Vec<f32>,
        content: String,
        metadata: HashMap<String, SochValue>,
    ) -> Result<(), String> {
        let mut collections = self.collections.write().unwrap();
        let coll = collections
            .get_mut(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;

        if vector.len() != coll.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                coll.dimension,
                vector.len()
            ));
        }

        coll.vectors.push((id, vector, content, metadata));
        Ok(())
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl Default for SimpleVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for SimpleVectorIndex {
    fn search_by_embedding(
        &self,
        collection: &str,
        embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        let collections = self.collections.read().unwrap();
        let coll = collections
            .get(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;

        // Compute similarities and sort
        let mut scored: Vec<_> = coll
            .vectors
            .iter()
            .map(|(id, vec, content, meta)| {
                let score = Self::cosine_similarity(embedding, vec);
                (id, score, content, meta)
            })
            .filter(|(_, score, _, _)| min_score.map(|min| *score >= min).unwrap_or(true))
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(id, score, content, meta)| VectorSearchResult {
                id: id.clone(),
                score,
                content: content.clone(),
                metadata: meta.clone(),
            })
            .collect())
    }

    fn search_by_text(
        &self,
        _collection: &str,
        _text: &str,
        _k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        // Text search requires an embedding model - return error for now
        Err(
            "Text-based search requires an embedding model. Use search_by_embedding instead."
                .to_string(),
        )
    }

    fn stats(&self, collection: &str) -> Option<VectorIndexStats> {
        let collections = self.collections.read().unwrap();
        collections.get(collection).map(|coll| VectorIndexStats {
            vector_count: coll.vectors.len(),
            dimension: coll.dimension,
            metric: "cosine".to_string(),
        })
    }
}

// ============================================================================
// HNSW-backed Vector Index (Production Implementation)
// ============================================================================

/// Production-ready vector index backed by HNSW from sochdb-index.
/// 
/// This provides O(log N) search performance vs O(N) for brute-force.
/// Each collection maps to a separate HNSW index.
pub struct HnswVectorIndex {
    /// Collection name -> (HNSW index, metadata storage)
    collections: std::sync::RwLock<HashMap<String, HnswCollection>>,
}

/// A collection backed by HNSW
struct HnswCollection {
    /// The HNSW index for fast vector search
    index: sochdb_index::vector::VectorIndex,
    /// Metadata storage: edge_id -> (id, content, metadata)
    #[allow(clippy::type_complexity)]
    metadata: HashMap<u128, (String, String, HashMap<String, SochValue>)>,
    /// Next edge ID (incrementing counter)
    next_edge_id: u128,
    /// Vector dimension
    dimension: usize,
}

impl HnswVectorIndex {
    /// Create a new HNSW-backed vector index
    pub fn new() -> Self {
        Self {
            collections: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Create a collection with specified dimension and HNSW parameters
    pub fn create_collection(&self, name: &str, dimension: usize) {
        let mut collections = self.collections.write().unwrap();
        collections.entry(name.to_string()).or_insert_with(|| {
            let index = sochdb_index::vector::VectorIndex::with_dimension(
                sochdb_index::vector::DistanceMetric::Cosine,
                dimension,
            );
            HnswCollection {
                index,
                metadata: HashMap::new(),
                next_edge_id: 0,
                dimension,
            }
        });
    }

    /// Insert a vector with metadata
    pub fn insert(
        &self,
        collection: &str,
        id: String,
        vector: Vec<f32>,
        content: String,
        metadata: HashMap<String, SochValue>,
    ) -> Result<(), String> {
        let mut collections = self.collections.write().unwrap();
        let coll = collections
            .get_mut(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;

        if vector.len() != coll.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                coll.dimension,
                vector.len()
            ));
        }

        // Generate edge ID and store metadata
        let edge_id = coll.next_edge_id;
        coll.next_edge_id += 1;
        coll.metadata.insert(edge_id, (id, content, metadata));

        // Convert to ndarray embedding
        let embedding = ndarray::Array1::from_vec(vector);
        
        // Add to HNSW index
        coll.index.add(edge_id, embedding)?;

        Ok(())
    }

    /// Get collection count for a specific collection
    pub fn vector_count(&self, collection: &str) -> Option<usize> {
        let collections = self.collections.read().unwrap();
        collections.get(collection).map(|c| c.metadata.len())
    }
}

impl Default for HnswVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for HnswVectorIndex {
    fn search_by_embedding(
        &self,
        collection: &str,
        embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        let collections = self.collections.read().unwrap();
        let coll = collections
            .get(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;

        // Convert to ndarray embedding
        let query = ndarray::Array1::from_vec(embedding.to_vec());

        // Search HNSW index (returns Vec<(edge_id, distance)>)
        let results = coll.index.search(&query, k)?;

        // Convert distances to similarity scores and filter
        // HNSW returns cosine distance (0 = identical), convert to similarity (1 = identical)
        let mut search_results = Vec::with_capacity(results.len());
        for (edge_id, distance) in results {
            // Cosine similarity = 1 - cosine distance (for normalized vectors)
            let score = 1.0 - distance;
            
            // Apply min_score filter
            if let Some(min) = min_score {
                if score < min {
                    continue;
                }
            }

            // Look up metadata
            if let Some((id, content, meta)) = coll.metadata.get(&edge_id) {
                search_results.push(VectorSearchResult {
                    id: id.clone(),
                    score,
                    content: content.clone(),
                    metadata: meta.clone(),
                });
            }
        }

        Ok(search_results)
    }

    fn search_by_text(
        &self,
        _collection: &str,
        _text: &str,
        _k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        // Text search requires an embedding model - HNSW only handles vectors
        Err(
            "Text-based search requires an embedding model. Use search_by_embedding instead."
                .to_string(),
        )
    }

    fn stats(&self, collection: &str) -> Option<VectorIndexStats> {
        let collections = self.collections.read().unwrap();
        collections.get(collection).map(|coll| VectorIndexStats {
            vector_count: coll.metadata.len(),
            dimension: coll.dimension,
            metric: "cosine".to_string(),
        })
    }
}

// ============================================================================
// Context Query Result
// ============================================================================

/// Result of executing a CONTEXT SELECT query
#[derive(Debug, Clone)]
pub struct ContextResult {
    /// Assembled context in TOON format
    pub context: String,
    /// Token count
    pub token_count: usize,
    /// Token budget
    pub token_budget: usize,
    /// Sections included
    pub sections_included: Vec<SectionResult>,
    /// Sections truncated
    pub sections_truncated: Vec<String>,
    /// Sections dropped (didn't fit)
    pub sections_dropped: Vec<String>,
}

/// Result of processing a single section
#[derive(Debug, Clone)]
pub struct SectionResult {
    /// Section name
    pub name: String,
    /// Priority value
    pub priority: i32,
    /// Content in TOON format
    pub content: String,
    /// Token count for this section
    pub tokens: usize,
    /// Also available as tokens_used for compatibility
    pub tokens_used: usize,
    /// Was truncated
    pub truncated: bool,
    /// Number of rows/items
    pub row_count: usize,
}

// ============================================================================
// Parser
// ============================================================================

/// Context execution error
#[derive(Debug, Clone)]
pub enum ContextQueryError {
    /// Session mismatch
    SessionMismatch { expected: String, actual: String },
    /// Variable not found
    VariableNotFound(String),
    /// Invalid variable type
    InvalidVariableType { variable: String, expected: String },
    /// Budget exceeded
    BudgetExceeded {
        section: String,
        requested: usize,
        available: usize,
    },
    /// Budget exhausted
    BudgetExhausted(String),
    /// Permission denied
    PermissionDenied(String),
    /// Invalid path
    InvalidPath(String),
    /// Parse error
    Parse(ContextParseError),
    /// Format error (e.g., writing output)
    FormatError(String),
    /// Invalid query (e.g., missing required fields)
    InvalidQuery(String),
    /// Vector search error
    VectorSearchError(String),
}

impl std::fmt::Display for ContextQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionMismatch { expected, actual } => {
                write!(f, "session mismatch: expected {}, got {}", expected, actual)
            }
            Self::VariableNotFound(name) => write!(f, "variable not found: {}", name),
            Self::InvalidVariableType { variable, expected } => {
                write!(
                    f,
                    "variable {} has invalid type, expected {}",
                    variable, expected
                )
            }
            Self::BudgetExceeded {
                section,
                requested,
                available,
            } => {
                write!(
                    f,
                    "section {} exceeds budget: {} > {}",
                    section, requested, available
                )
            }
            Self::BudgetExhausted(msg) => write!(f, "budget exhausted: {}", msg),
            Self::PermissionDenied(msg) => write!(f, "permission denied: {}", msg),
            Self::InvalidPath(path) => write!(f, "invalid path: {}", path),
            Self::Parse(e) => write!(f, "parse error: {}", e),
            Self::FormatError(e) => write!(f, "format error: {}", e),
            Self::InvalidQuery(msg) => write!(f, "invalid query: {}", msg),
            Self::VectorSearchError(e) => write!(f, "vector search error: {}", e),
        }
    }
}

impl std::error::Error for ContextQueryError {}

/// Parse error
#[derive(Debug, Clone)]
pub enum ContextParseError {
    /// Unexpected token
    UnexpectedToken { expected: String, found: String },
    /// Missing required clause
    MissingClause(String),
    /// Invalid option
    InvalidOption(String),
    /// Invalid path expression
    InvalidPath(String),
    /// Invalid section syntax
    InvalidSection(String),
    /// General syntax error
    SyntaxError(String),
}

impl std::fmt::Display for ContextParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedToken { expected, found } => {
                write!(f, "expected {}, found '{}'", expected, found)
            }
            Self::MissingClause(clause) => write!(f, "missing {} clause", clause),
            Self::InvalidOption(opt) => write!(f, "invalid option: {}", opt),
            Self::InvalidPath(path) => write!(f, "invalid path: {}", path),
            Self::InvalidSection(sec) => write!(f, "invalid section: {}", sec),
            Self::SyntaxError(msg) => write!(f, "syntax error: {}", msg),
        }
    }
}

impl std::error::Error for ContextParseError {}

/// CONTEXT SELECT parser
pub struct ContextQueryParser {
    /// Current position
    pos: usize,
    /// Input tokens
    tokens: Vec<Token>,
}

/// Token type
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// Keyword
    Keyword(String),
    /// Identifier
    Ident(String),
    /// Number
    Number(f64),
    /// String literal
    String(String),
    /// Punctuation
    Punct(char),
    /// Variable reference ($name)
    Variable(String),
    /// End of input
    Eof,
}

impl ContextQueryParser {
    /// Create a new parser
    pub fn new(input: &str) -> Self {
        let tokens = Self::tokenize(input);
        Self { pos: 0, tokens }
    }

    /// Parse a CONTEXT SELECT query
    pub fn parse(&mut self) -> Result<ContextSelectQuery, ContextParseError> {
        // CONTEXT SELECT output_name
        self.expect_keyword("CONTEXT")?;
        self.expect_keyword("SELECT")?;
        let output_name = self.expect_ident()?;

        // FROM session($SESSION_ID) or FROM agent($AGENT_ID) - optional
        let session = if self.match_keyword("FROM") {
            self.parse_session_reference()?
        } else {
            SessionReference::None
        };

        // WITH (options) - optional
        let options = if self.match_keyword("WITH") {
            self.parse_options()?
        } else {
            ContextQueryOptions::default()
        };

        // SECTIONS (...)
        self.expect_keyword("SECTIONS")?;
        let sections = self.parse_sections()?;

        Ok(ContextSelectQuery {
            output_name,
            session,
            options,
            sections,
        })
    }

    /// Parse session reference
    fn parse_session_reference(&mut self) -> Result<SessionReference, ContextParseError> {
        if self.match_keyword("session") {
            self.expect_punct('(')?;
            let var = self.expect_variable()?;
            self.expect_punct(')')?;
            Ok(SessionReference::Session(var))
        } else if self.match_keyword("agent") {
            self.expect_punct('(')?;
            let var = self.expect_variable()?;
            self.expect_punct(')')?;
            Ok(SessionReference::Agent(var))
        } else {
            Err(ContextParseError::SyntaxError(
                "expected 'session' or 'agent'".to_string(),
            ))
        }
    }

    /// Parse WITH options
    fn parse_options(&mut self) -> Result<ContextQueryOptions, ContextParseError> {
        self.expect_punct('(')?;
        let mut options = ContextQueryOptions::default();

        loop {
            let key = self.expect_ident()?;
            self.expect_punct('=')?;

            match key.as_str() {
                "token_limit" => {
                    if let Token::Number(n) = self.current().clone() {
                        options.token_limit = n as usize;
                        self.advance();
                    }
                }
                "include_schema" => {
                    options.include_schema = self.parse_bool()?;
                }
                "format" => {
                    let format = self.expect_ident()?;
                    options.format = match format.to_lowercase().as_str() {
                        "toon" => OutputFormat::Soch,
                        "json" => OutputFormat::Json,
                        "markdown" => OutputFormat::Markdown,
                        _ => return Err(ContextParseError::InvalidOption(format)),
                    };
                }
                "truncation" => {
                    let strategy = self.expect_ident()?;
                    options.truncation = match strategy.to_lowercase().as_str() {
                        "tail_drop" | "taildrop" => TruncationStrategy::TailDrop,
                        "head_drop" | "headdrop" => TruncationStrategy::HeadDrop,
                        "proportional" => TruncationStrategy::Proportional,
                        "fail" => TruncationStrategy::Fail,
                        _ => return Err(ContextParseError::InvalidOption(strategy)),
                    };
                }
                "include_headers" => {
                    options.include_headers = self.parse_bool()?;
                }
                _ => return Err(ContextParseError::InvalidOption(key)),
            }

            if !self.match_punct(',') {
                break;
            }
        }

        self.expect_punct(')')?;
        Ok(options)
    }

    /// Parse SECTIONS block
    fn parse_sections(&mut self) -> Result<Vec<ContextSection>, ContextParseError> {
        self.expect_punct('(')?;
        let mut sections = Vec::new();

        loop {
            if self.check_punct(')') {
                break;
            }

            let section = self.parse_section()?;
            sections.push(section);

            if !self.match_punct(',') {
                break;
            }
        }

        self.expect_punct(')')?;
        Ok(sections)
    }

    /// Parse a single section
    fn parse_section(&mut self) -> Result<ContextSection, ContextParseError> {
        // SECTION_NAME PRIORITY N: content
        let name = self.expect_ident()?;

        self.expect_keyword("PRIORITY")?;
        let priority = if let Token::Number(n) = self.current().clone() {
            let val = n as i32;
            self.advance();
            val
        } else {
            0
        };

        self.expect_punct(':')?;

        let content = self.parse_section_content()?;

        Ok(ContextSection {
            name,
            priority,
            content,
            transform: None,
        })
    }

    /// Parse section content
    fn parse_section_content(&mut self) -> Result<SectionContent, ContextParseError> {
        if self.match_keyword("GET") {
            // GET path.expression.{fields}
            let path_str = self.collect_until(&[',', ')']);
            let path = PathExpression::parse(&path_str)?;
            Ok(SectionContent::Get { path })
        } else if self.match_keyword("LAST") {
            // LAST N FROM table WHERE ...
            let count = if let Token::Number(n) = self.current().clone() {
                let val = n as usize;
                self.advance();
                val
            } else {
                10 // default
            };

            self.expect_keyword("FROM")?;
            let table = self.expect_ident()?;

            let where_clause = if self.match_keyword("WHERE") {
                Some(self.parse_where_clause()?)
            } else {
                None
            };

            Ok(SectionContent::Last {
                count,
                table,
                where_clause,
            })
        } else if self.match_keyword("SEARCH") {
            // SEARCH collection BY SIMILARITY($query) TOP K
            let collection = self.expect_ident()?;
            self.expect_keyword("BY")?;
            self.expect_keyword("SIMILARITY")?;

            self.expect_punct('(')?;
            let query = if let Token::Variable(v) = self.current().clone() {
                self.advance();
                SimilarityQuery::Variable(v)
            } else if let Token::String(s) = self.current().clone() {
                self.advance();
                SimilarityQuery::Text(s)
            } else {
                return Err(ContextParseError::SyntaxError(
                    "expected variable or string for similarity query".to_string(),
                ));
            };
            self.expect_punct(')')?;

            self.expect_keyword("TOP")?;
            let top_k = if let Token::Number(n) = self.current().clone() {
                let val = n as usize;
                self.advance();
                val
            } else {
                5 // default
            };

            Ok(SectionContent::Search {
                collection,
                query,
                top_k,
                min_score: None,
            })
        } else if self.match_keyword("SELECT") {
            // Standard SELECT subquery
            let columns = self.parse_column_list()?;
            self.expect_keyword("FROM")?;
            let table = self.expect_ident()?;

            let where_clause = if self.match_keyword("WHERE") {
                Some(self.parse_where_clause()?)
            } else {
                None
            };

            let limit = if self.match_keyword("LIMIT") {
                if let Token::Number(n) = self.current().clone() {
                    let val = n as usize;
                    self.advance();
                    Some(val)
                } else {
                    None
                }
            } else {
                None
            };

            Ok(SectionContent::Select {
                columns,
                table,
                where_clause,
                limit,
            })
        } else if let Token::Variable(v) = self.current().clone() {
            self.advance();
            Ok(SectionContent::Variable { name: v })
        } else if let Token::String(s) = self.current().clone() {
            self.advance();
            Ok(SectionContent::Literal { value: s })
        } else {
            Err(ContextParseError::InvalidSection(
                "expected GET, LAST, SEARCH, SELECT, or literal".to_string(),
            ))
        }
    }

    /// Parse a WHERE clause (simplified)
    fn parse_where_clause(&mut self) -> Result<WhereClause, ContextParseError> {
        let mut conditions = Vec::new();

        loop {
            let column = self.expect_ident()?;
            let operator = self.parse_comparison_op()?;
            let value = self.parse_value()?;

            conditions.push(Condition {
                column,
                operator,
                value,
            });

            if !self.match_keyword("AND") && !self.match_keyword("OR") {
                break;
            }
        }

        Ok(WhereClause {
            conditions,
            operator: LogicalOp::And,
        })
    }

    /// Parse comparison operator
    fn parse_comparison_op(&mut self) -> Result<ComparisonOp, ContextParseError> {
        match self.current() {
            Token::Punct('=') => {
                self.advance();
                Ok(ComparisonOp::Eq)
            }
            Token::Punct('>') => {
                self.advance();
                if self.check_punct('=') {
                    self.advance();
                    Ok(ComparisonOp::Ge)
                } else {
                    Ok(ComparisonOp::Gt)
                }
            }
            Token::Punct('<') => {
                self.advance();
                if self.check_punct('=') {
                    self.advance();
                    Ok(ComparisonOp::Le)
                } else {
                    Ok(ComparisonOp::Lt)
                }
            }
            _ => {
                if self.match_keyword("LIKE") {
                    Ok(ComparisonOp::Like)
                } else if self.match_keyword("IN") {
                    Ok(ComparisonOp::In)
                } else {
                    Err(ContextParseError::SyntaxError(
                        "expected comparison operator".to_string(),
                    ))
                }
            }
        }
    }

    /// Parse a value
    fn parse_value(&mut self) -> Result<SochValue, ContextParseError> {
        match self.current().clone() {
            Token::Number(n) => {
                self.advance();
                if n.fract() == 0.0 {
                    Ok(SochValue::Int(n as i64))
                } else {
                    Ok(SochValue::Float(n))
                }
            }
            Token::String(s) => {
                self.advance();
                Ok(SochValue::Text(s))
            }
            Token::Keyword(k) if k.eq_ignore_ascii_case("null") => {
                self.advance();
                Ok(SochValue::Null)
            }
            Token::Keyword(k) if k.eq_ignore_ascii_case("true") => {
                self.advance();
                Ok(SochValue::Bool(true))
            }
            Token::Keyword(k) if k.eq_ignore_ascii_case("false") => {
                self.advance();
                Ok(SochValue::Bool(false))
            }
            Token::Variable(v) => {
                self.advance();
                // Variables are passed as text placeholders
                Ok(SochValue::Text(format!("${}", v)))
            }
            _ => Err(ContextParseError::SyntaxError("expected value".to_string())),
        }
    }

    /// Parse column list
    fn parse_column_list(&mut self) -> Result<Vec<String>, ContextParseError> {
        let mut columns = Vec::new();

        if self.check_punct('*') {
            self.advance();
            columns.push("*".to_string());
        } else {
            loop {
                columns.push(self.expect_ident()?);
                if !self.match_punct(',') {
                    break;
                }
            }
        }

        Ok(columns)
    }

    /// Parse boolean value
    fn parse_bool(&mut self) -> Result<bool, ContextParseError> {
        match self.current() {
            Token::Keyword(k) if k.eq_ignore_ascii_case("true") => {
                self.advance();
                Ok(true)
            }
            Token::Keyword(k) if k.eq_ignore_ascii_case("false") => {
                self.advance();
                Ok(false)
            }
            _ => Err(ContextParseError::SyntaxError(
                "expected boolean".to_string(),
            )),
        }
    }

    /// Tokenize input
    fn tokenize(input: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                // Whitespace
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }

                // Punctuation
                '(' | ')' | ',' | ':' | '=' | '<' | '>' | '*' | '{' | '}' | '.' => {
                    tokens.push(Token::Punct(ch));
                    chars.next();
                }

                // Variable reference
                '$' => {
                    chars.next();
                    let mut name = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            name.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::Variable(name));
                }

                // String literal
                '\'' | '"' => {
                    let quote = ch;
                    chars.next();
                    let mut s = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == quote {
                            chars.next(); // consume closing quote
                            break;
                        }
                        s.push(c);
                        chars.next();
                    }
                    tokens.push(Token::String(s));
                }

                // Number
                '0'..='9' | '-' => {
                    let mut num_str = String::new();
                    if ch == '-' {
                        num_str.push(ch);
                        chars.next();
                    }
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() || c == '.' {
                            num_str.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if let Ok(n) = num_str.parse::<f64>() {
                        tokens.push(Token::Number(n));
                    }
                }

                // Identifier or keyword
                'a'..='z' | 'A'..='Z' | '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    // Check for keywords
                    let keywords = [
                        "CONTEXT",
                        "SELECT",
                        "FROM",
                        "WITH",
                        "SECTIONS",
                        "PRIORITY",
                        "GET",
                        "LAST",
                        "SEARCH",
                        "BY",
                        "SIMILARITY",
                        "TOP",
                        "WHERE",
                        "AND",
                        "OR",
                        "LIKE",
                        "IN",
                        "LIMIT",
                        "session",
                        "agent",
                        "true",
                        "false",
                        "null",
                    ];

                    if keywords.iter().any(|k| k.eq_ignore_ascii_case(&ident)) {
                        tokens.push(Token::Keyword(ident.to_uppercase()));
                    } else {
                        tokens.push(Token::Ident(ident));
                    }
                }

                // Skip unknown
                _ => {
                    chars.next();
                }
            }
        }

        tokens.push(Token::Eof);
        tokens
    }

    // Helper methods
    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn expect_keyword(&mut self, kw: &str) -> Result<(), ContextParseError> {
        match self.current() {
            Token::Keyword(k) if k.eq_ignore_ascii_case(kw) => {
                self.advance();
                Ok(())
            }
            other => Err(ContextParseError::UnexpectedToken {
                expected: kw.to_string(),
                found: format!("{:?}", other),
            }),
        }
    }

    fn match_keyword(&mut self, kw: &str) -> bool {
        match self.current() {
            Token::Keyword(k) if k.eq_ignore_ascii_case(kw) => {
                self.advance();
                true
            }
            _ => false,
        }
    }

    fn expect_ident(&mut self) -> Result<String, ContextParseError> {
        match self.current().clone() {
            Token::Ident(s) => {
                self.advance();
                Ok(s)
            }
            Token::Keyword(s) => {
                // Allow keywords as identifiers in some contexts
                self.advance();
                Ok(s)
            }
            other => Err(ContextParseError::UnexpectedToken {
                expected: "identifier".to_string(),
                found: format!("{:?}", other),
            }),
        }
    }

    fn expect_variable(&mut self) -> Result<String, ContextParseError> {
        match self.current().clone() {
            Token::Variable(v) => {
                self.advance();
                Ok(v)
            }
            other => Err(ContextParseError::UnexpectedToken {
                expected: "variable ($name)".to_string(),
                found: format!("{:?}", other),
            }),
        }
    }

    fn expect_punct(&mut self, p: char) -> Result<(), ContextParseError> {
        match self.current() {
            Token::Punct(c) if *c == p => {
                self.advance();
                Ok(())
            }
            other => Err(ContextParseError::UnexpectedToken {
                expected: p.to_string(),
                found: format!("{:?}", other),
            }),
        }
    }

    fn match_punct(&mut self, p: char) -> bool {
        match self.current() {
            Token::Punct(c) if *c == p => {
                self.advance();
                true
            }
            _ => false,
        }
    }

    fn check_punct(&self, p: char) -> bool {
        matches!(self.current(), Token::Punct(c) if *c == p)
    }

    fn collect_until(&mut self, terminators: &[char]) -> String {
        let mut result = String::new();
        let mut depth = 0;

        loop {
            match self.current() {
                Token::Punct('{') => {
                    depth += 1;
                    result.push('{');
                    self.advance();
                }
                Token::Punct('}') => {
                    depth -= 1;
                    result.push('}');
                    self.advance();
                }
                Token::Punct(c) if depth == 0 && terminators.contains(c) => {
                    break;
                }
                Token::Punct(c) => {
                    result.push(*c);
                    self.advance();
                }
                Token::Ident(s) | Token::Keyword(s) => {
                    if !result.is_empty() && !result.ends_with(['.', '{']) {
                        result.push(' ');
                    }
                    result.push_str(s);
                    self.advance();
                }
                Token::Eof => break,
                _ => {
                    self.advance();
                }
            }
        }

        result.trim().to_string()
    }
}

// ============================================================================
// AgentContext Integration (Task 9)
// ============================================================================

use crate::agent_context::{AgentContext, AuditOperation, ContextValue};

/// Integration between CONTEXT SELECT and AgentContext
///
/// Provides:
/// - Session variable resolution
/// - Permission checking for data access
/// - Audit logging of context queries
/// - Budget integration via TokenBudgetEnforcer
/// - Vector search for SEARCH sections
/// - Embedding provider for text-to-vector search
pub struct AgentContextIntegration<'a> {
    /// The agent context
    context: &'a mut AgentContext,
    /// Token budget enforcer (used for allocation decisions)
    budget_enforcer: TokenBudgetEnforcer,
    /// Token estimator
    estimator: TokenEstimator,
    /// Vector index for SEARCH operations
    vector_index: Option<std::sync::Arc<dyn VectorIndex>>,
    /// Embedding provider for text-to-vector conversion
    embedding_provider: Option<std::sync::Arc<dyn EmbeddingProvider>>,
}

/// Trait for providing text-to-embedding conversion
///
/// Implementations can use local models (e.g., ONNX) or remote APIs.
pub trait EmbeddingProvider: Send + Sync {
    /// Convert text to embedding vector
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, String>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name/identifier
    fn model_name(&self) -> &str;
}

impl<'a> AgentContextIntegration<'a> {
    /// Create integration with an agent context
    pub fn new(context: &'a mut AgentContext) -> Self {
        let config = TokenBudgetConfig {
            total_budget: context.budget.max_tokens.unwrap_or(4096) as usize,
            ..Default::default()
        };

        Self {
            context,
            budget_enforcer: TokenBudgetEnforcer::new(config),
            estimator: TokenEstimator::default(),
            vector_index: None,
            embedding_provider: None,
        }
    }

    /// Create integration with a vector index for SEARCH operations
    pub fn with_vector_index(
        context: &'a mut AgentContext,
        vector_index: std::sync::Arc<dyn VectorIndex>,
    ) -> Self {
        let config = TokenBudgetConfig {
            total_budget: context.budget.max_tokens.unwrap_or(4096) as usize,
            ..Default::default()
        };

        Self {
            context,
            budget_enforcer: TokenBudgetEnforcer::new(config),
            estimator: TokenEstimator::default(),
            vector_index: Some(vector_index),
            embedding_provider: None,
        }
    }

    /// Create integration with both vector index and embedding provider
    pub fn with_vector_and_embedding(
        context: &'a mut AgentContext,
        vector_index: std::sync::Arc<dyn VectorIndex>,
        embedding_provider: std::sync::Arc<dyn EmbeddingProvider>,
    ) -> Self {
        let config = TokenBudgetConfig {
            total_budget: context.budget.max_tokens.unwrap_or(4096) as usize,
            ..Default::default()
        };

        Self {
            context,
            budget_enforcer: TokenBudgetEnforcer::new(config),
            estimator: TokenEstimator::default(),
            vector_index: Some(vector_index),
            embedding_provider: Some(embedding_provider),
        }
    }

    /// Set the embedding provider for text-to-vector search
    pub fn set_embedding_provider(&mut self, provider: std::sync::Arc<dyn EmbeddingProvider>) {
        self.embedding_provider = Some(provider);
    }

    /// Set the vector index for SEARCH operations
    pub fn set_vector_index(&mut self, index: std::sync::Arc<dyn VectorIndex>) {
        self.vector_index = Some(index);
    }

    /// Execute a CONTEXT SELECT query with agent context
    pub fn execute(
        &mut self,
        query: &ContextSelectQuery,
    ) -> Result<ContextQueryResult, ContextQueryError> {
        // Validate session matches
        self.validate_session(&query.session)?;

        // Audit the query start
        self.context.audit.push(crate::agent_context::AuditEntry {
            timestamp: std::time::SystemTime::now(),
            operation: AuditOperation::DbQuery,
            resource: format!("CONTEXT SELECT {}", query.output_name),
            result: crate::agent_context::AuditResult::Success,
            metadata: std::collections::HashMap::new(),
        });

        // Resolve variables in sections
        let resolved_sections = self.resolve_sections(&query.sections)?;

        // Check permissions for each section
        for section in &resolved_sections {
            self.check_section_permissions(section)?;
        }

        // Execute each section to get content and estimate tokens
        let mut section_contents: Vec<(ContextSection, String)> = Vec::new();
        for section in &resolved_sections {
            let content = self.execute_section_content(section, query.options.token_limit)?;
            section_contents.push((section.clone(), content));
        }

        // Build BudgetSection structs for the enforcer
        let budget_sections: Vec<BudgetSection> = section_contents
            .iter()
            .map(|(section, content)| {
                let estimated = self.estimator.estimate_text(content);
                // Minimum tokens: 10% of estimated or 100, whichever is smaller
                let minimum = if query.options.truncation == TruncationStrategy::Fail {
                    None
                } else {
                    Some(estimated.min(100).max(estimated / 10))
                };
                BudgetSection {
                    name: section.name.clone(),
                    estimated_tokens: estimated,
                    minimum_tokens: minimum,
                    priority: section.priority,
                    required: section.priority == 0, // Priority 0 = required
                    weight: 1.0,
                }
            })
            .collect();

        // Use the budget enforcer for allocation
        let allocation = self.budget_enforcer.allocate_sections(&budget_sections);

        // Build result based on allocation
        let mut result = ContextQueryResult::new(query.output_name.clone());
        result.format = query.options.format;
        result.allocation_explain = Some(allocation.explain.clone());

        // Process full sections
        for (section, content) in section_contents.iter() {
            if allocation.full_sections.contains(&section.name) {
                let tokens = self.estimator.estimate_text(content);
                result.sections.push(SectionResult {
                    name: section.name.clone(),
                    priority: section.priority,
                    content: content.clone(),
                    tokens,
                    tokens_used: tokens,
                    truncated: false,
                    row_count: 0,
                });
            }
        }

        // Process truncated sections
        for (section_name, _original, truncated_to) in &allocation.truncated_sections {
            if let Some((section, content)) = section_contents
                .iter()
                .find(|(s, _)| &s.name == section_name)
            {
                // Use token-aware truncation
                let truncated = self.estimator.truncate_to_tokens(content, *truncated_to);
                let actual_tokens = self.estimator.estimate_text(&truncated);
                result.sections.push(SectionResult {
                    name: section.name.clone(),
                    priority: section.priority,
                    content: truncated,
                    tokens: actual_tokens,
                    tokens_used: actual_tokens,
                    truncated: true,
                    row_count: 0,
                });
            }
        }

        // Sort result sections by priority
        result.sections.sort_by_key(|s| s.priority);

        result.total_tokens = allocation.tokens_allocated;
        result.token_limit = query.options.token_limit;

        // Record budget consumption
        self.context
            .consume_budget(result.total_tokens as u64, 0)
            .map_err(|e| ContextQueryError::BudgetExhausted(e.to_string()))?;

        Ok(result)
    }

    /// Execute a CONTEXT SELECT with EXPLAIN output
    pub fn execute_explain(
        &mut self,
        query: &ContextSelectQuery,
    ) -> Result<(ContextQueryResult, String), ContextQueryError> {
        let result = self.execute(query)?;
        let explain = result
            .allocation_explain
            .as_ref()
            .map(|decisions| {
                use crate::token_budget::BudgetAllocation;
                let allocation = BudgetAllocation {
                    full_sections: result
                        .sections
                        .iter()
                        .filter(|s| !s.truncated)
                        .map(|s| s.name.clone())
                        .collect(),
                    truncated_sections: result
                        .sections
                        .iter()
                        .filter(|s| s.truncated)
                        .map(|s| (s.name.clone(), s.tokens, s.tokens_used))
                        .collect(),
                    dropped_sections: Vec::new(),
                    tokens_allocated: result.total_tokens,
                    tokens_remaining: result.token_limit.saturating_sub(result.total_tokens),
                    explain: decisions.clone(),
                };
                allocation.explain_text()
            })
            .unwrap_or_else(|| "No allocation explain available".to_string());
        Ok((result, explain))
    }

    /// Validate session reference matches context
    fn validate_session(&self, session_ref: &SessionReference) -> Result<(), ContextQueryError> {
        match session_ref {
            SessionReference::Session(sid) => {
                // Allow variable reference
                if sid.starts_with('$') {
                    return Ok(());
                }
                // Check if matches current session
                if sid != &self.context.session_id && sid != "*" {
                    return Err(ContextQueryError::SessionMismatch {
                        expected: sid.clone(),
                        actual: self.context.session_id.clone(),
                    });
                }
            }
            SessionReference::Agent(aid) => {
                // Agent ID is in session variables
                if let Some(ContextValue::String(agent_id)) = self.context.peek_var("agent_id")
                    && aid != agent_id
                    && aid != "*"
                {
                    return Err(ContextQueryError::SessionMismatch {
                        expected: aid.clone(),
                        actual: agent_id.clone(),
                    });
                }
            }
            SessionReference::None => {}
        }
        Ok(())
    }

    /// Resolve variable references in sections
    fn resolve_sections(
        &self,
        sections: &[ContextSection],
    ) -> Result<Vec<ContextSection>, ContextQueryError> {
        let mut resolved = Vec::new();

        for section in sections {
            let mut resolved_section = section.clone();

            // Resolve variables in content
            resolved_section.content = match &section.content {
                SectionContent::Literal { value } => {
                    let resolved_value = self.resolve_variables(value);
                    SectionContent::Literal {
                        value: resolved_value,
                    }
                }
                SectionContent::Variable { name } => {
                    if let Some(value) = self.context.peek_var(name) {
                        SectionContent::Literal {
                            value: value.to_string(),
                        }
                    } else {
                        return Err(ContextQueryError::VariableNotFound(name.clone()));
                    }
                }
                SectionContent::Search {
                    collection,
                    query,
                    top_k,
                    min_score,
                } => {
                    let resolved_query = match query {
                        SimilarityQuery::Variable(var) => {
                            if let Some(value) = self.context.peek_var(var) {
                                match value {
                                    ContextValue::String(s) => SimilarityQuery::Text(s.clone()),
                                    ContextValue::List(l) => {
                                        let vec: Vec<f32> = l
                                            .iter()
                                            .filter_map(|v| match v {
                                                ContextValue::Number(n) => Some(*n as f32),
                                                _ => None,
                                            })
                                            .collect();
                                        SimilarityQuery::Embedding(vec)
                                    }
                                    _ => {
                                        return Err(ContextQueryError::InvalidVariableType {
                                            variable: var.clone(),
                                            expected: "string or vector".to_string(),
                                        });
                                    }
                                }
                            } else {
                                return Err(ContextQueryError::VariableNotFound(var.clone()));
                            }
                        }
                        other => other.clone(),
                    };
                    SectionContent::Search {
                        collection: collection.clone(),
                        query: resolved_query,
                        top_k: *top_k,
                        min_score: *min_score,
                    }
                }
                other => other.clone(),
            };

            resolved.push(resolved_section);
        }

        Ok(resolved)
    }

    /// Resolve $variable references in a string
    fn resolve_variables(&self, input: &str) -> String {
        self.context.substitute_vars(input)
    }

    /// Check permissions for section data access
    fn check_section_permissions(&self, section: &ContextSection) -> Result<(), ContextQueryError> {
        match &section.content {
            SectionContent::Get { path } => {
                // Check filesystem or database permissions based on path
                let path_str = path.to_path_string();
                if path_str.starts_with('/') {
                    self.context
                        .check_fs_permission(&path_str, AuditOperation::FsRead)
                        .map_err(|e| ContextQueryError::PermissionDenied(e.to_string()))?;
                } else {
                    // Assume it's a table path
                    let table = path
                        .segments
                        .first()
                        .ok_or_else(|| ContextQueryError::InvalidPath("empty path".to_string()))?;
                    self.context
                        .check_db_permission(table, AuditOperation::DbQuery)
                        .map_err(|e| ContextQueryError::PermissionDenied(e.to_string()))?;
                }
            }
            SectionContent::Last { table, .. } | SectionContent::Select { table, .. } => {
                self.context
                    .check_db_permission(table, AuditOperation::DbQuery)
                    .map_err(|e| ContextQueryError::PermissionDenied(e.to_string()))?;
            }
            SectionContent::Search { collection, .. } => {
                self.context
                    .check_db_permission(collection, AuditOperation::DbQuery)
                    .map_err(|e| ContextQueryError::PermissionDenied(e.to_string()))?;
            }
            SectionContent::Literal { .. } | SectionContent::Variable { .. } => {
                // No permission check needed
            }
            SectionContent::ToolRegistry { .. } | SectionContent::ToolCalls { .. } => {
                // Tool registry and calls are internal to agent context - always accessible
            }
        }
        Ok(())
    }

    /// Execute a section's content
    fn execute_section_content(
        &self,
        section: &ContextSection,
        _budget: usize,
    ) -> Result<String, ContextQueryError> {
        // In a real implementation, this would execute the query
        // For now, return placeholder based on content type
        match &section.content {
            SectionContent::Literal { value } => Ok(value.clone()),
            SectionContent::Variable { name } => self
                .context
                .peek_var(name)
                .map(|v| v.to_string())
                .ok_or_else(|| ContextQueryError::VariableNotFound(name.clone())),
            SectionContent::Get { path } => {
                // Would fetch from storage
                Ok(format!(
                    "[{}: path={}]",
                    section.name,
                    path.to_path_string()
                ))
            }
            SectionContent::Last { count, table, .. } => {
                // Would query storage
                Ok(format!("[{}: last {} from {}]", section.name, count, table))
            }
            SectionContent::Search {
                collection,
                query: similarity_query,
                top_k,
                min_score,
            } => {
                // Execute real vector search if index is available
                match &self.vector_index {
                    Some(index) => {
                        // Execute search based on query type
                        let results = match similarity_query {
                            SimilarityQuery::Embedding(emb) => {
                                index.search_by_embedding(collection, emb, *top_k, *min_score)
                            }
                            SimilarityQuery::Text(text) => {
                                // Use embedding provider if available, otherwise fall back to index
                                self.search_by_text_with_embedding(
                                    index, collection, text, *top_k, *min_score,
                                )
                            }
                            SimilarityQuery::Variable(var_name) => {
                                // Try to resolve variable as embedding or text
                                match self.context.peek_var(var_name) {
                                    Some(ContextValue::String(text)) => {
                                        self.search_by_text_with_embedding(
                                            index, collection, text, *top_k, *min_score,
                                        )
                                    }
                                    Some(ContextValue::List(list)) => {
                                        // Try to convert to f32 vector
                                        let embedding: Result<Vec<f32>, _> = list
                                            .iter()
                                            .map(|v| match v {
                                                ContextValue::Number(n) => Ok(*n as f32),
                                                ContextValue::String(s) => {
                                                    s.parse::<f32>().map_err(|_| "not a number")
                                                }
                                                _ => Err("not a number"),
                                            })
                                            .collect();

                                        match embedding {
                                            Ok(emb) => index.search_by_embedding(
                                                collection, &emb, *top_k, *min_score,
                                            ),
                                            Err(_) => {
                                                Err("Variable is not a valid embedding vector"
                                                    .to_string())
                                            }
                                        }
                                    }
                                    _ => Err(format!(
                                        "Variable '{}' not found or has wrong type",
                                        var_name
                                    )),
                                }
                            }
                        };

                        match results {
                            Ok(search_results) => {
                                // Format results as TOON array
                                self.format_search_results(&section.name, &search_results)
                            }
                            Err(e) => {
                                // Log error but don't fail query
                                Ok(format!("[{}: search error: {}]", section.name, e))
                            }
                        }
                    }
                    None => {
                        // No vector index configured, return placeholder
                        Ok(format!(
                            "[{}: search {} top {}]",
                            section.name, collection, top_k
                        ))
                    }
                }
            }
            SectionContent::Select { table, limit, .. } => {
                // Would execute SQL
                let limit_str = limit.map(|l| format!(" limit {}", l)).unwrap_or_default();
                Ok(format!(
                    "[{}: select from {}{}]",
                    section.name, table, limit_str
                ))
            }
            SectionContent::ToolRegistry {
                include,
                exclude,
                include_schema,
            } => {
                // Format tool registry as context section
                self.format_tool_registry(include, exclude, *include_schema)
            }
            SectionContent::ToolCalls {
                count,
                tool_filter,
                status_filter,
                include_outputs,
            } => {
                // Format tool call history as context section
                self.format_tool_calls(*count, tool_filter.as_deref(), status_filter.as_deref(), *include_outputs)
            }
        }
    }

    /// Format tool registry as TOON
    fn format_tool_registry(
        &self,
        include: &[String],
        exclude: &[String],
        include_schema: bool,
    ) -> Result<String, ContextQueryError> {
        use std::fmt::Write;

        // Get tools from agent context's tool registry
        let tools = &self.context.tool_registry;
        let mut output = String::new();

        writeln!(output, "[tool_registry ({} tools)]", tools.len())
            .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;

        for tool in tools {
            // Apply include/exclude filters
            if !include.is_empty() && !include.contains(&tool.name) {
                continue;
            }
            if exclude.contains(&tool.name) {
                continue;
            }

            writeln!(output, "  [{}]", tool.name)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
            writeln!(output, "    description = {:?}", tool.description)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;

            if include_schema {
                if let Some(schema) = &tool.parameters_schema {
                    writeln!(output, "    parameters = {}", schema)
                        .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
                }
            }
        }

        Ok(output)
    }

    /// Format tool call history as TOON
    fn format_tool_calls(
        &self,
        count: usize,
        tool_filter: Option<&str>,
        status_filter: Option<&str>,
        include_outputs: bool,
    ) -> Result<String, ContextQueryError> {
        use std::fmt::Write;

        // Get tool calls from agent context
        let calls = &self.context.tool_calls;
        let mut output = String::new();

        // Filter and limit calls
        let filtered: Vec<_> = calls
            .iter()
            .filter(|call| {
                tool_filter.map(|f| call.tool_name == f).unwrap_or(true)
                    && status_filter
                        .map(|s| {
                            match s {
                                "success" => call.result.is_some() && call.error.is_none(),
                                "error" => call.error.is_some(),
                                "pending" => call.result.is_none() && call.error.is_none(),
                                _ => true,
                            }
                        })
                        .unwrap_or(true)
            })
            .rev() // Most recent first
            .take(count)
            .collect();

        writeln!(output, "[tool_calls ({} calls)]", filtered.len())
            .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;

        for call in filtered {
            writeln!(output, "  [call {}]", call.call_id)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
            writeln!(output, "    tool = {:?}", call.tool_name)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
            writeln!(output, "    arguments = {:?}", call.arguments)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;

            if include_outputs {
                if let Some(result) = &call.result {
                    writeln!(output, "    result = {:?}", result)
                        .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
                }
                if let Some(error) = &call.error {
                    writeln!(output, "    error = {:?}", error)
                        .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
                }
            }
        }

        Ok(output)
    }

    /// Search by text using embedding provider if available
    ///
    /// This enables the SEARCH-by-text feature by converting text to
    /// embeddings before searching. Falls back to index.search_by_text
    /// if no embedding provider is configured.
    fn search_by_text_with_embedding(
        &self,
        index: &std::sync::Arc<dyn VectorIndex>,
        collection: &str,
        text: &str,
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        match &self.embedding_provider {
            Some(provider) => {
                // Convert text to embedding
                let embedding = provider.embed_text(text)?;
                // Search by embedding
                index.search_by_embedding(collection, &embedding, k, min_score)
            }
            None => {
                // Fall back to index's text search (may return error)
                index.search_by_text(collection, text, k, min_score)
            }
        }
    }

    /// Format vector search results as TOON
    fn format_search_results(
        &self,
        section_name: &str,
        results: &[VectorSearchResult],
    ) -> Result<String, ContextQueryError> {
        use std::fmt::Write;

        let mut output = String::new();
        writeln!(output, "[{} ({} results)]", section_name, results.len())
            .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;

        for (i, result) in results.iter().enumerate() {
            writeln!(output, "  [result {} score={:.4}]", i + 1, result.score)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
            writeln!(output, "    id = {}", result.id)
                .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;

            // Include content, properly indented
            for line in result.content.lines() {
                writeln!(output, "    {}", line)
                    .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
            }

            // Include metadata if present
            if !result.metadata.is_empty() {
                writeln!(output, "    [metadata]")
                    .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
                for (key, value) in &result.metadata {
                    writeln!(output, "      {} = {:?}", key, value)
                        .map_err(|e| ContextQueryError::FormatError(e.to_string()))?;
                }
            }
        }

        Ok(output)
    }

    /// Truncate content to fit budget
    #[allow(dead_code)]
    fn truncate_content(
        &self,
        content: &str,
        max_tokens: usize,
        strategy: TruncationStrategy,
    ) -> String {
        // Rough approximation: 4 chars per token
        let max_chars = max_tokens * 4;

        if content.len() <= max_chars {
            return content.to_string();
        }

        match strategy {
            TruncationStrategy::TailDrop => {
                let mut result: String = content.chars().take(max_chars - 3).collect();
                result.push_str("...");
                result
            }
            TruncationStrategy::HeadDrop => {
                let skip = content.len() - max_chars + 3;
                let mut result = "...".to_string();
                result.extend(content.chars().skip(skip));
                result
            }
            TruncationStrategy::Proportional => {
                // Keep first and last quarters, truncate middle
                let quarter = max_chars / 4;
                let first: String = content.chars().take(quarter).collect();
                let last: String = content
                    .chars()
                    .skip(content.len().saturating_sub(quarter))
                    .collect();
                format!("{}...{}...", first, last)
            }
            TruncationStrategy::Fail => {
                content.to_string() // Shouldn't reach here
            }
        }
    }

    /// Get session variables as context
    pub fn get_session_context(&self) -> HashMap<String, String> {
        self.context
            .variables
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect()
    }

    /// Set a session variable
    pub fn set_variable(&mut self, name: &str, value: ContextValue) {
        self.context.set_var(name, value);
    }

    /// Get remaining token budget
    pub fn remaining_budget(&self) -> u64 {
        self.context
            .budget
            .max_tokens
            .map(|max| max.saturating_sub(self.context.budget.tokens_used))
            .unwrap_or(u64::MAX)
    }
}

/// Result of executing a CONTEXT SELECT query
#[derive(Debug, Clone)]
pub struct ContextQueryResult {
    /// Output name
    pub output_name: String,
    /// Executed sections
    pub sections: Vec<SectionResult>,
    /// Total tokens used
    pub total_tokens: usize,
    /// Token limit
    pub token_limit: usize,
    /// Output format
    pub format: OutputFormat,
    /// Allocation decisions for EXPLAIN CONTEXT
    pub allocation_explain: Option<Vec<crate::token_budget::AllocationDecision>>,
}

impl ContextQueryResult {
    fn new(output_name: String) -> Self {
        Self {
            output_name,
            sections: Vec::new(),
            total_tokens: 0,
            token_limit: 0,
            format: OutputFormat::Soch,
            allocation_explain: None,
        }
    }

    /// Render the result to string
    pub fn render(&self) -> String {
        let mut output = String::new();

        match self.format {
            OutputFormat::Soch => {
                // TOON format with headers
                output.push_str(&format!("{}[{}]:\n", self.output_name, self.sections.len()));
                for section in &self.sections {
                    output.push_str(&format!(
                        "  {}[{}{}]:\n",
                        section.name,
                        section.tokens_used,
                        if section.truncated { "T" } else { "" }
                    ));
                    for line in section.content.lines() {
                        output.push_str(&format!("    {}\n", line));
                    }
                }
            }
            OutputFormat::Json => {
                output.push_str("{\n");
                output.push_str(&format!("  \"name\": \"{}\",\n", self.output_name));
                output.push_str(&format!("  \"total_tokens\": {},\n", self.total_tokens));
                output.push_str("  \"sections\": [\n");
                for (i, section) in self.sections.iter().enumerate() {
                    output.push_str(&format!("    {{\"name\": \"{}\", \"tokens\": {}, \"truncated\": {}, \"content\": \"{}\"}}",
                        section.name,
                        section.tokens_used,
                        section.truncated,
                        section.content.replace('"', "\\\"").replace('\n', "\\n")
                    ));
                    if i < self.sections.len() - 1 {
                        output.push(',');
                    }
                    output.push('\n');
                }
                output.push_str("  ]\n}");
            }
            OutputFormat::Markdown => {
                output.push_str(&format!("# {}\n\n", self.output_name));
                output.push_str(&format!(
                    "*Tokens: {}/{}*\n\n",
                    self.total_tokens, self.token_limit
                ));
                for section in &self.sections {
                    output.push_str(&format!("## {}", section.name));
                    if section.truncated {
                        output.push_str(" *(truncated)*");
                    }
                    output.push_str("\n\n");
                    output.push_str(&section.content);
                    output.push_str("\n\n");
                }
            }
        }

        output
    }

    /// Get token utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.token_limit == 0 {
            return 0.0;
        }
        (self.total_tokens as f64 / self.token_limit as f64) * 100.0
    }

    /// Check if any section was truncated
    pub fn has_truncation(&self) -> bool {
        self.sections.iter().any(|s| s.truncated)
    }
}

/// Priority-based section ordering helper
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SectionPriority(pub i32);

impl SectionPriority {
    pub const CRITICAL: SectionPriority = SectionPriority(-100);
    pub const SYSTEM: SectionPriority = SectionPriority(-1);
    pub const USER: SectionPriority = SectionPriority(0);
    pub const HISTORY: SectionPriority = SectionPriority(1);
    pub const KNOWLEDGE: SectionPriority = SectionPriority(2);
    pub const SUPPLEMENTARY: SectionPriority = SectionPriority(10);
}

// ============================================================================
// Context Query Builder
// ============================================================================

/// Builder for constructing CONTEXT SELECT queries programmatically
pub struct ContextQueryBuilder {
    output_name: String,
    session: SessionReference,
    options: ContextQueryOptions,
    sections: Vec<ContextSection>,
}

impl ContextQueryBuilder {
    /// Create a new builder
    pub fn new(output_name: &str) -> Self {
        Self {
            output_name: output_name.to_string(),
            session: SessionReference::None,
            options: ContextQueryOptions::default(),
            sections: Vec::new(),
        }
    }

    /// Set session reference
    pub fn from_session(mut self, session_id: &str) -> Self {
        self.session = SessionReference::Session(session_id.to_string());
        self
    }

    /// Set agent reference
    pub fn from_agent(mut self, agent_id: &str) -> Self {
        self.session = SessionReference::Agent(agent_id.to_string());
        self
    }

    /// Set token limit
    pub fn with_token_limit(mut self, limit: usize) -> Self {
        self.options.token_limit = limit;
        self
    }

    /// Include schema
    pub fn include_schema(mut self, include: bool) -> Self {
        self.options.include_schema = include;
        self
    }

    /// Set output format
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.options.format = format;
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.options.truncation = strategy;
        self
    }

    /// Add a GET section
    pub fn get(mut self, name: &str, priority: i32, path: &str) -> Self {
        let path_expr = PathExpression::parse(path).unwrap_or(PathExpression {
            segments: vec![path.to_string()],
            fields: vec![],
            all_fields: true,
        });

        self.sections.push(ContextSection {
            name: name.to_string(),
            priority,
            content: SectionContent::Get { path: path_expr },
            transform: None,
        });
        self
    }

    /// Add a LAST section
    pub fn last(mut self, name: &str, priority: i32, count: usize, table: &str) -> Self {
        self.sections.push(ContextSection {
            name: name.to_string(),
            priority,
            content: SectionContent::Last {
                count,
                table: table.to_string(),
                where_clause: None,
            },
            transform: None,
        });
        self
    }

    /// Add a SEARCH section
    pub fn search(
        mut self,
        name: &str,
        priority: i32,
        collection: &str,
        query_var: &str,
        top_k: usize,
    ) -> Self {
        self.sections.push(ContextSection {
            name: name.to_string(),
            priority,
            content: SectionContent::Search {
                collection: collection.to_string(),
                query: SimilarityQuery::Variable(query_var.to_string()),
                top_k,
                min_score: None,
            },
            transform: None,
        });
        self
    }

    /// Add a literal section
    pub fn literal(mut self, name: &str, priority: i32, value: &str) -> Self {
        self.sections.push(ContextSection {
            name: name.to_string(),
            priority,
            content: SectionContent::Literal {
                value: value.to_string(),
            },
            transform: None,
        });
        self
    }

    /// Build the query
    pub fn build(self) -> ContextSelectQuery {
        ContextSelectQuery {
            output_name: self.output_name,
            session: self.session,
            options: self.options,
            sections: self.sections,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_expression_simple() {
        let path = PathExpression::parse("user.profile").unwrap();
        assert_eq!(path.segments, vec!["user", "profile"]);
        assert!(path.all_fields);
    }

    #[test]
    fn test_path_expression_with_fields() {
        let path = PathExpression::parse("user.profile.{name, email}").unwrap();
        assert_eq!(path.segments, vec!["user", "profile"]);
        assert_eq!(path.fields, vec!["name", "email"]);
        assert!(!path.all_fields);
    }

    #[test]
    fn test_path_expression_glob() {
        let path = PathExpression::parse("user.**").unwrap();
        assert_eq!(path.segments, vec!["user"]);
        assert!(path.all_fields);
    }

    #[test]
    fn test_parse_simple_query() {
        let query = r#"
            CONTEXT SELECT prompt_context
            FROM session($SESSION_ID)
            WITH (token_limit = 2048, include_schema = true)
            SECTIONS (
                USER PRIORITY 0: GET user.profile.{name, preferences}
            )
        "#;

        let mut parser = ContextQueryParser::new(query);
        let result = parser.parse().unwrap();

        assert_eq!(result.output_name, "prompt_context");
        assert!(matches!(result.session, SessionReference::Session(s) if s == "SESSION_ID"));
        assert_eq!(result.options.token_limit, 2048);
        assert!(result.options.include_schema);
        assert_eq!(result.sections.len(), 1);
        assert_eq!(result.sections[0].name, "USER");
        assert_eq!(result.sections[0].priority, 0);
    }

    #[test]
    fn test_parse_multiple_sections() {
        let query = r#"
            CONTEXT SELECT context
            SECTIONS (
                A PRIORITY 0: "literal value",
                B PRIORITY 1: LAST 10 FROM logs,
                C PRIORITY 2: SEARCH docs BY SIMILARITY($query) TOP 5
            )
        "#;

        let mut parser = ContextQueryParser::new(query);
        let result = parser.parse().unwrap();

        assert_eq!(result.sections.len(), 3);

        // Check section A
        assert_eq!(result.sections[0].name, "A");
        assert!(
            matches!(&result.sections[0].content, SectionContent::Literal { value } if value == "literal value")
        );

        // Check section B
        assert_eq!(result.sections[1].name, "B");
        assert!(
            matches!(&result.sections[1].content, SectionContent::Last { count: 10, table, .. } if table == "logs")
        );

        // Check section C
        assert_eq!(result.sections[2].name, "C");
        assert!(
            matches!(&result.sections[2].content, SectionContent::Search { collection, top_k: 5, .. } if collection == "docs")
        );
    }

    #[test]
    fn test_builder() {
        let query = ContextQueryBuilder::new("prompt")
            .from_session("sess123")
            .with_token_limit(4096)
            .include_schema(false)
            .get("USER", 0, "user.profile.{name, email}")
            .last("HISTORY", 1, 20, "events")
            .search("DOCS", 2, "knowledge_base", "query_embedding", 10)
            .literal("SYSTEM", -1, "You are a helpful assistant")
            .build();

        assert_eq!(query.output_name, "prompt");
        assert_eq!(query.options.token_limit, 4096);
        assert!(!query.options.include_schema);
        assert_eq!(query.sections.len(), 4);

        // System prompt has highest priority (lowest number)
        let system = query.sections.iter().find(|s| s.name == "SYSTEM").unwrap();
        assert_eq!(system.priority, -1);
    }

    #[test]
    fn test_output_format() {
        let query = r#"
            CONTEXT SELECT ctx
            WITH (format = markdown)
            SECTIONS ()
        "#;

        let mut parser = ContextQueryParser::new(query);
        let result = parser.parse().unwrap();

        assert_eq!(result.options.format, OutputFormat::Markdown);
    }

    #[test]
    fn test_truncation_strategy() {
        let query = r#"
            CONTEXT SELECT ctx
            WITH (truncation = proportional)
            SECTIONS ()
        "#;

        let mut parser = ContextQueryParser::new(query);
        let result = parser.parse().unwrap();

        assert_eq!(result.options.truncation, TruncationStrategy::Proportional);
    }

    // ========================================================================
    // Task 6: Vector Index Tests
    // ========================================================================

    #[test]
    fn test_simple_vector_index_creation() {
        let index = SimpleVectorIndex::new();
        index.create_collection("test", 3);

        let stats = index.stats("test");
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.dimension, 3);
        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.metric, "cosine");
    }

    #[test]
    fn test_simple_vector_index_insert_and_search() {
        let index = SimpleVectorIndex::new();
        index.create_collection("docs", 3);

        // Insert some vectors
        index
            .insert(
                "docs",
                "doc1".to_string(),
                vec![1.0, 0.0, 0.0],
                "Document about cats".to_string(),
                HashMap::new(),
            )
            .unwrap();

        index
            .insert(
                "docs",
                "doc2".to_string(),
                vec![0.9, 0.1, 0.0],
                "Document about dogs".to_string(),
                HashMap::new(),
            )
            .unwrap();

        index
            .insert(
                "docs",
                "doc3".to_string(),
                vec![0.0, 0.0, 1.0],
                "Document about cars".to_string(),
                HashMap::new(),
            )
            .unwrap();

        // Search for similar to [1, 0, 0]
        let results = index
            .search_by_embedding("docs", &[1.0, 0.0, 0.0], 2, None)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1"); // Exact match
        assert!((results[0].score - 1.0).abs() < 0.001);
        assert_eq!(results[1].id, "doc2"); // Next closest
        assert!(results[1].score > 0.9); // Very similar
    }

    #[test]
    fn test_simple_vector_index_min_score_filter() {
        let index = SimpleVectorIndex::new();
        index.create_collection("docs", 3);

        index
            .insert(
                "docs",
                "a".to_string(),
                vec![1.0, 0.0, 0.0],
                "A".to_string(),
                HashMap::new(),
            )
            .unwrap();
        index
            .insert(
                "docs",
                "b".to_string(),
                vec![0.0, 1.0, 0.0],
                "B".to_string(),
                HashMap::new(),
            )
            .unwrap();
        index
            .insert(
                "docs",
                "c".to_string(),
                vec![0.0, 0.0, 1.0],
                "C".to_string(),
                HashMap::new(),
            )
            .unwrap();

        // Search with high min_score - should only return exact match
        let results = index
            .search_by_embedding("docs", &[1.0, 0.0, 0.0], 10, Some(0.9))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_simple_vector_index_dimension_mismatch() {
        let index = SimpleVectorIndex::new();
        index.create_collection("docs", 3);

        let result = index.insert(
            "docs",
            "bad".to_string(),
            vec![1.0, 0.0], // Wrong dimension
            "Content".to_string(),
            HashMap::new(),
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_simple_vector_index_nonexistent_collection() {
        let index = SimpleVectorIndex::new();

        let result = index.search_by_embedding("nonexistent", &[1.0], 1, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_vector_index_with_metadata() {
        let index = SimpleVectorIndex::new();
        index.create_collection("docs", 2);

        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), SochValue::Text("Alice".to_string()));
        metadata.insert("year".to_string(), SochValue::Int(2024));

        index
            .insert(
                "docs",
                "doc1".to_string(),
                vec![1.0, 0.0],
                "Document content".to_string(),
                metadata,
            )
            .unwrap();

        let results = index
            .search_by_embedding("docs", &[1.0, 0.0], 1, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].metadata.contains_key("author"));
        assert!(results[0].metadata.contains_key("year"));
    }

    #[test]
    fn test_vector_index_text_search_unsupported() {
        let index = SimpleVectorIndex::new();
        index.create_collection("docs", 2);

        // Text search requires an embedding model, which SimpleVectorIndex doesn't have
        let result = index.search_by_text("docs", "hello", 5, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("embedding model"));
    }
}
