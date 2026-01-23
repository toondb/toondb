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

//! SochDB Query Engine
//!
//! SOCH-QL query language for TOON-native data.
//!
//! ## SOCH-QL
//!
//! SQL-like query language that returns results in TOON format:
//!
//! ```text
//! SELECT id,name FROM users WHERE score > 80
//! → users[2]{id,name}:
//!   1,Alice
//!   3,Charlie
//! ```
//!
//! ## Query Execution Pipeline (Task 6)
//!
//! ```text
//! parse(sql) → SochQuery → validate → plan → execute → SochTable
//! ```
//!
//! Token reduction: 40-60% vs JSON (66% for typical queries)
//!
//! ## CONTEXT SELECT (LLM-Native)
//!
//! Priority-based context aggregation for LLM consumption:
//!
//! ```text
//! CONTEXT SELECT
//!   FROM session('abc123')
//!   WITH TOKEN_LIMIT 4000
//!   SECTIONS (
//!     USER {query, preferences} PRIORITY 1,
//!     HISTORY {recent} PRIORITY 2,
//!     KNOWLEDGE {docs} PRIORITY 3
//!   )
//! ```

pub mod agent_context;
pub mod bm25_filtered; // Task 6: BM25 filter pushdown via posting-set intersection
pub mod calc;
pub mod candidate_gate; // Task 4: Unified candidate gate interface
pub mod capability_token; // Task 8: Capability tokens + ACLs
pub mod context_query;
pub mod cost_optimizer; // Cost-based query optimizer (Task 6)
pub mod embedding_provider; // Task 2: Automatic embedding generation
pub mod exact_token_counter; // Task 6: BPE-accurate token counting
pub mod filter_ir; // Task 1: Canonical Filter IR (CNF/DNF)
pub mod filtered_vector_search; // Task 5: Filter-aware vector search with selectivity fallback
pub mod hybrid_retrieval; // Task 3: Vector + BM25 + RRF fusion
pub mod memory_compaction; // Task 5: Hierarchical memory compaction
pub mod metadata_index; // Task 3: Metadata index primitives (bitmap + range)
pub mod namespace; // Task 2: Namespace-scoped query API
pub mod optimizer_integration;
pub mod plugin_table;
pub mod query_optimizer;
pub mod semantic_triggers; // Task 7: Vector percolator triggers
pub mod simd_filter; // SIMD vectorized query filters (mm.md Task 5.3)
pub mod sql; // SQL-92 compatible query engine with SochDB extensions
pub mod streaming_context; // Task 1: Streaming context generation
pub mod temporal_decay; // Task 4: Recency-biased scoring
pub mod token_budget;
pub mod soch_ql;
pub mod soch_ql_executor;
pub mod topk_executor; // Streaming Top-K for ORDER BY + LIMIT (Task: Fix ORDER BY Semantics)
pub mod unified_fusion; // Task 7: Hybrid fusion that never post-filters

pub use agent_context::{
    AgentContext, AgentPermissions, AuditEntry, AuditOperation, AuditResult, ContextError,
    ContextValue, DbPermissions, FsPermissions, OperationBudget, PendingWrite, ResourceType,
    SessionId, SessionManager, TransactionScope,
};
pub use calc::{
    BinaryOp, CalcError, Evaluator, Expr, Parser as CalcParser, RowContext, UnaryOp, calculate,
    parse_expr,
};
pub use context_query::{
    ContextQueryError, ContextQueryParser, ContextQueryResult, ContextSection, ContextSelectQuery,
    HnswVectorIndex, SectionPriority, SectionResult, SimpleVectorIndex, VectorIndex,
    VectorIndexStats, VectorSearchResult,
};
pub use optimizer_integration::{
    CacheStats, ExecutionPlan, ExecutionStep, OptimizedExecutor, OptimizedQueryPlan, PlanCache,
    StorageBackend, TableStats,
};
pub use plugin_table::{
    PluginVirtualTable, VirtualColumnDef, VirtualColumnType, VirtualFilter, VirtualRow,
    VirtualTable, VirtualTableError, VirtualTableRegistry, VirtualTableSchema, VirtualTableStats,
};
pub use sql::{
    BinaryOperator, ColumnDef as SqlColumnDef, CreateTableStmt, DeleteStmt, DropTableStmt,
    Expr as SqlExpr, InsertStmt, JoinType, Lexer, OrderByItem as SqlOrderBy, Parser as SqlParser,
    SelectStmt, Span, SqlError, SqlResult, Statement, Token, TokenKind, UnaryOperator, UpdateStmt,
};
pub use token_budget::{
    BudgetAllocation, BudgetSection, TokenBudgetConfig, TokenBudgetEnforcer, TokenEstimator,
    TokenEstimatorConfig, truncate_rows, truncate_to_tokens,
};
pub use soch_ql::{
    ColumnDef, ColumnType, ComparisonOp, Condition, CreateTableQuery, InsertQuery, LogicalOp,
    OrderBy, ParseError, SelectQuery, SortDirection, SochQlParser, SochQuery, SochResult,
    SochValue, WhereClause,
};
pub use soch_ql_executor::{
    KeyRange, Predicate, PredicateCondition, QueryPlan, TokenReductionStats, SochQlExecutor,
    estimate_token_reduction, execute_sochql,
};

// Streaming Top-K for ORDER BY + LIMIT (Task: Fix ORDER BY Semantics)
pub use topk_executor::{
    ColumnRef, ExecutionStrategy as TopKExecutionStrategy, IndexAwareTopK, OrderByColumn, OrderByLimitExecutor,
    OrderByLimitStats, OrderBySpec, SingleColumnTopK, SortDirection as TopKSortDirection, TopKHeap,
};

// Task 1: Streaming context generation
pub use streaming_context::{
    RollingBudget, SectionChunk, StreamingConfig, StreamingContextExecutor, StreamingContextIter,
};

// Task 2: Automatic embedding generation
pub use embedding_provider::{
    CachedEmbeddingProvider, EmbeddingError, EmbeddingProvider, EmbeddingVectorIndex,
    MockEmbeddingProvider,
};

// Task 3: Hybrid retrieval pipeline
pub use hybrid_retrieval::{
    FusionMethod, HybridQuery, HybridQueryExecutor, LexicalIndex, MetadataFilter,
};

// Task 4: Temporal decay scoring
pub use temporal_decay::{
    DecayCurve, TemporalDecayConfig, TemporalScorer, TemporallyDecayedResult,
};

// Task 5: Memory compaction
pub use memory_compaction::{
    Abstraction, CompactionStats, Episode, ExtractiveSummarizer, HierarchicalMemory, Summary,
    Summarizer,
};

// Task 6: Exact token counting
pub use exact_token_counter::{
    ExactBudgetEnforcer, ExactTokenCounter, HeuristicTokenCounter, TokenCounter,
};

// Task 7: Semantic triggers
pub use semantic_triggers::{
    EscalationLevel, EventSource, LogLevel, SemanticTrigger, TriggerAction, TriggerBuilder,
    TriggerError, TriggerEvent, TriggerIndex, TriggerMatch, TriggerStats,
};

// ============================================================================
// Canonical Filter IR + Pushdown Contract (mm.md Tasks 1-8)
// ============================================================================

// Task 1: Canonical Filter IR (CNF/DNF with typed atoms)
pub use filter_ir::{
    AuthCapabilities, AuthScope, Disjunction, FilterAtom, FilterBuilder, FilterIR, FilterValue,
    FilteredExecutor,
};

// Task 2: Namespace-Scoped Query API (mandatory namespace)
pub use namespace::{
    Namespace, NamespaceError, NamespaceScope, QueryRequest, ScopedQuery,
};

// Task 3: Metadata Index Primitives (bitmap + range accessors)
pub use metadata_index::{
    ConcurrentMetadataIndex, EqualityIndex, MetadataIndex, PostingSet, RangeIndex,
};

// Task 4: Unified Candidate Gate Interface
pub use candidate_gate::{
    AllowedBitmap, AllowedSet, CandidateGate, ExecutionStrategy,
};

// Task 5: Filter-Aware Vector Search with selectivity-driven fallback
pub use filtered_vector_search::{
    FilterAwareSearch, FilteredSearchConfig, FilteredSearchResult, FilteredSearchStrategy,
    FilteredVectorStore, ScoredResult,
};

// Task 6: BM25 Filter Pushdown via posting-set intersection
pub use bm25_filtered::{
    Bm25Params, DisjunctiveBm25Executor, FilteredBm25Executor, FilteredPhraseExecutor,
    InvertedIndex, PositionalIndex, PositionalPosting, PostingList,
};

// Task 7: Hybrid Fusion That Never Post-Filters
pub use unified_fusion::{
    Bm25Executor, Bm25QuerySpec, FilteredCandidates, FusionConfig, FusionEngine,
    FusionMethod as UnifiedFusionMethod, FusionResult, Modality, UnifiedHybridExecutor,
    UnifiedHybridQuery, VectorExecutor, VectorQuerySpec,
};

// Task 8: Capability Tokens + ACLs
pub use capability_token::{
    AclTagIndex, CapabilityToken, TokenBuilder, TokenCapabilities, TokenError, TokenSigner,
    TokenValidator,
};
