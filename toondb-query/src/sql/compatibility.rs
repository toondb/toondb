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

//! # SQL Compatibility Matrix
//!
//! This module defines ToonDB's SQL dialect support and compatibility layer.
//!
//! ## Design Goals
//!
//! 1. **Portable Core**: SQL-92 compatible baseline that works across ecosystems
//! 2. **Dialect Sugar**: Support common dialect variants (MySQL, PostgreSQL, SQLite)
//! 3. **Single AST**: All dialects normalize to one canonical AST representation
//! 4. **Extensible**: Add new dialects without forking parsers/executors
//!
//! ## SQL Feature Matrix
//!
//! ### Guaranteed (Core SQL)
//!
//! | Category | Statement | Status | Notes |
//! |----------|-----------|--------|-------|
//! | DML | SELECT | ✅ | With WHERE, ORDER BY, LIMIT, OFFSET |
//! | DML | INSERT | ✅ | Single and multi-row |
//! | DML | UPDATE | ✅ | With WHERE clause |
//! | DML | DELETE | ✅ | With WHERE clause |
//! | DDL | CREATE TABLE | ✅ | With column types and constraints |
//! | DDL | DROP TABLE | ✅ | Basic form |
//! | DDL | ALTER TABLE | 🔄 | ADD/DROP COLUMN |
//! | DDL | CREATE INDEX | ✅ | Single and multi-column |
//! | DDL | DROP INDEX | ✅ | Basic form |
//! | Tx | BEGIN | ✅ | Start transaction |
//! | Tx | COMMIT | ✅ | Commit transaction |
//! | Tx | ROLLBACK | ✅ | Rollback transaction |
//!
//! ### Idempotent DDL
//!
//! | Statement | Status | Notes |
//! |-----------|--------|-------|
//! | CREATE TABLE IF NOT EXISTS | ✅ | No-op if exists |
//! | DROP TABLE IF EXISTS | ✅ | No-op if not exists |
//! | CREATE INDEX IF NOT EXISTS | ✅ | No-op if exists |
//! | DROP INDEX IF EXISTS | ✅ | No-op if not exists |
//!
//! ### Conflict/Upsert Family
//!
//! All of these normalize to `InsertStmt { on_conflict: Some(OnConflict { .. }) }`
//!
//! | Dialect | Syntax | Canonical AST |
//! |---------|--------|---------------|
//! | PostgreSQL | `ON CONFLICT DO NOTHING` | `OnConflict { action: DoNothing }` |
//! | PostgreSQL | `ON CONFLICT DO UPDATE SET ...` | `OnConflict { action: DoUpdate(...) }` |
//! | MySQL | `INSERT IGNORE` | `OnConflict { action: DoNothing }` |
//! | MySQL | `ON DUPLICATE KEY UPDATE` | `OnConflict { action: DoUpdate(...) }` |
//! | SQLite | `INSERT OR IGNORE` | `OnConflict { action: DoNothing }` |
//! | SQLite | `INSERT OR REPLACE` | `OnConflict { action: DoReplace }` |
//!
//! ### Out of Scope (Explicit Limitations)
//!
//! | Feature | Status | Reason |
//! |---------|--------|--------|
//! | Multi-table JOINs | ❌ | Complexity; single-table focus for v1 |
//! | Subqueries in WHERE | ❌ | Planning complexity |
//! | Window functions | ❌ | Future enhancement |
//! | CTEs (WITH clause) | ❌ | Future enhancement |
//! | Stored procedures | ❌ | Out of scope |
//!
//! ## Dialect Detection
//!
//! ToonDB auto-detects dialect from syntax:
//! - `INSERT IGNORE` → MySQL mode
//! - `INSERT OR IGNORE` → SQLite mode
//! - `ON CONFLICT` → PostgreSQL mode
//!
//! All normalize to the same internal representation.

use std::fmt;

/// SQL Dialect for parsing/normalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqlDialect {
    /// Standard SQL-92 compatible (default)
    #[default]
    Standard,
    /// PostgreSQL dialect
    PostgreSQL,
    /// MySQL dialect
    MySQL,
    /// SQLite dialect
    SQLite,
}

impl SqlDialect {
    /// Detect dialect from SQL text
    pub fn detect(sql: &str) -> Self {
        let upper = sql.to_uppercase();

        // MySQL: INSERT IGNORE
        if upper.contains("INSERT IGNORE") || upper.contains("ON DUPLICATE KEY") {
            return SqlDialect::MySQL;
        }

        // SQLite: INSERT OR IGNORE/REPLACE/ABORT
        if upper.contains("INSERT OR IGNORE")
            || upper.contains("INSERT OR REPLACE")
            || upper.contains("INSERT OR ABORT")
        {
            return SqlDialect::SQLite;
        }

        // PostgreSQL: ON CONFLICT
        if upper.contains("ON CONFLICT") {
            return SqlDialect::PostgreSQL;
        }

        SqlDialect::Standard
    }
}

impl fmt::Display for SqlDialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqlDialect::Standard => write!(f, "Standard SQL"),
            SqlDialect::PostgreSQL => write!(f, "PostgreSQL"),
            SqlDialect::MySQL => write!(f, "MySQL"),
            SqlDialect::SQLite => write!(f, "SQLite"),
        }
    }
}

/// SQL Feature support level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSupport {
    /// Fully supported
    Full,
    /// Partially supported with limitations
    Partial,
    /// Planned for future release
    Planned,
    /// Not supported and not planned
    NotSupported,
}

impl fmt::Display for FeatureSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureSupport::Full => write!(f, "✅ Full"),
            FeatureSupport::Partial => write!(f, "🔄 Partial"),
            FeatureSupport::Planned => write!(f, "📋 Planned"),
            FeatureSupport::NotSupported => write!(f, "❌ Not Supported"),
        }
    }
}

/// SQL Feature categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SqlFeature {
    // DML
    Select,
    Insert,
    Update,
    Delete,

    // DDL
    CreateTable,
    DropTable,
    AlterTable,
    CreateIndex,
    DropIndex,

    // Idempotent DDL
    CreateTableIfNotExists,
    DropTableIfExists,
    CreateIndexIfNotExists,
    DropIndexIfExists,

    // Conflict/Upsert
    OnConflictDoNothing,
    OnConflictDoUpdate,
    InsertIgnore,
    InsertOrIgnore,
    InsertOrReplace,
    OnDuplicateKeyUpdate,

    // Transactions
    Begin,
    Commit,
    Rollback,
    Savepoint,

    // Query features
    Where,
    OrderBy,
    Limit,
    Offset,
    GroupBy,
    Having,
    Distinct,

    // Joins (limited)
    InnerJoin,
    LeftJoin,
    RightJoin,
    CrossJoin,

    // Subqueries
    SubqueryInFrom,
    SubqueryInWhere,
    SubqueryInSelect,

    // Set operations
    Union,
    Intersect,
    Except,

    // Expressions
    ParameterizedQueries,
    CaseWhen,
    Cast,
    NullHandling,
    InList,
    Between,
    Like,

    // ToonDB extensions
    VectorSearch,
    EmbeddingType,
    ContextWindow,
}

/// Get feature support level
pub fn get_feature_support(feature: SqlFeature) -> FeatureSupport {
    use SqlFeature::*;

    match feature {
        // Fully supported
        Select | Insert | Update | Delete => FeatureSupport::Full,
        CreateTable | DropTable | CreateIndex | DropIndex => FeatureSupport::Full,
        CreateTableIfNotExists | DropTableIfExists => FeatureSupport::Full,
        CreateIndexIfNotExists | DropIndexIfExists => FeatureSupport::Full,
        Begin | Commit | Rollback => FeatureSupport::Full,
        Where | OrderBy | Limit | Offset | Distinct => FeatureSupport::Full,
        ParameterizedQueries | NullHandling | InList | Like => FeatureSupport::Full,
        OnConflictDoNothing | InsertIgnore | InsertOrIgnore => FeatureSupport::Full,
        VectorSearch | EmbeddingType => FeatureSupport::Full,

        // Partially supported
        AlterTable => FeatureSupport::Partial, // ADD/DROP COLUMN only
        GroupBy | Having => FeatureSupport::Partial, // Basic support
        InnerJoin => FeatureSupport::Partial, // Two-table only
        OnConflictDoUpdate | InsertOrReplace | OnDuplicateKeyUpdate => FeatureSupport::Partial,
        CaseWhen | Cast | Between => FeatureSupport::Partial,
        Union => FeatureSupport::Partial,
        SubqueryInFrom => FeatureSupport::Partial,
        Savepoint => FeatureSupport::Partial,
        ContextWindow => FeatureSupport::Partial,

        // Planned
        LeftJoin | RightJoin | CrossJoin => FeatureSupport::Planned,
        SubqueryInWhere | SubqueryInSelect => FeatureSupport::Planned,
        Intersect | Except => FeatureSupport::Planned,
    }
}

/// Compatibility matrix for different SQL dialects
pub struct CompatibilityMatrix;

impl CompatibilityMatrix {
    /// Check if a feature is supported
    pub fn is_supported(feature: SqlFeature) -> bool {
        matches!(
            get_feature_support(feature),
            FeatureSupport::Full | FeatureSupport::Partial
        )
    }

    /// Get all fully supported features
    pub fn fully_supported() -> Vec<SqlFeature> {
        use SqlFeature::*;
        vec![
            Select,
            Insert,
            Update,
            Delete,
            CreateTable,
            DropTable,
            CreateIndex,
            DropIndex,
            CreateTableIfNotExists,
            DropTableIfExists,
            CreateIndexIfNotExists,
            DropIndexIfExists,
            Begin,
            Commit,
            Rollback,
            Where,
            OrderBy,
            Limit,
            Offset,
            Distinct,
            ParameterizedQueries,
            NullHandling,
            InList,
            Like,
            OnConflictDoNothing,
            InsertIgnore,
            InsertOrIgnore,
            VectorSearch,
            EmbeddingType,
        ]
    }

    /// Print the compatibility matrix as a formatted table
    pub fn print_matrix() -> String {
        let mut output = String::new();
        output.push_str("# ToonDB SQL Compatibility Matrix\n\n");

        output.push_str("## Core DML\n\n");
        output.push_str("| Feature | Support |\n");
        output.push_str("|---------|--------|\n");
        for feature in &[
            SqlFeature::Select,
            SqlFeature::Insert,
            SqlFeature::Update,
            SqlFeature::Delete,
        ] {
            output.push_str(&format!(
                "| {:?} | {} |\n",
                feature,
                get_feature_support(*feature)
            ));
        }

        output.push_str("\n## DDL\n\n");
        output.push_str("| Feature | Support |\n");
        output.push_str("|---------|--------|\n");
        for feature in &[
            SqlFeature::CreateTable,
            SqlFeature::DropTable,
            SqlFeature::AlterTable,
            SqlFeature::CreateIndex,
            SqlFeature::DropIndex,
            SqlFeature::CreateTableIfNotExists,
            SqlFeature::DropTableIfExists,
        ] {
            output.push_str(&format!(
                "| {:?} | {} |\n",
                feature,
                get_feature_support(*feature)
            ));
        }

        output.push_str("\n## Conflict/Upsert\n\n");
        output.push_str("| Feature | Support |\n");
        output.push_str("|---------|--------|\n");
        for feature in &[
            SqlFeature::OnConflictDoNothing,
            SqlFeature::OnConflictDoUpdate,
            SqlFeature::InsertIgnore,
            SqlFeature::InsertOrIgnore,
            SqlFeature::InsertOrReplace,
            SqlFeature::OnDuplicateKeyUpdate,
        ] {
            output.push_str(&format!(
                "| {:?} | {} |\n",
                feature,
                get_feature_support(*feature)
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_detection() {
        assert_eq!(SqlDialect::detect("SELECT * FROM users"), SqlDialect::Standard);
        assert_eq!(
            SqlDialect::detect("INSERT IGNORE INTO users VALUES (1)"),
            SqlDialect::MySQL
        );
        assert_eq!(
            SqlDialect::detect("INSERT OR IGNORE INTO users VALUES (1)"),
            SqlDialect::SQLite
        );
        assert_eq!(
            SqlDialect::detect("INSERT INTO users VALUES (1) ON CONFLICT DO NOTHING"),
            SqlDialect::PostgreSQL
        );
    }

    #[test]
    fn test_feature_support() {
        assert_eq!(get_feature_support(SqlFeature::Select), FeatureSupport::Full);
        assert_eq!(
            get_feature_support(SqlFeature::AlterTable),
            FeatureSupport::Partial
        );
        assert_eq!(
            get_feature_support(SqlFeature::LeftJoin),
            FeatureSupport::Planned
        );
    }

    #[test]
    fn test_compatibility_matrix() {
        assert!(CompatibilityMatrix::is_supported(SqlFeature::Select));
        assert!(CompatibilityMatrix::is_supported(SqlFeature::AlterTable));
        assert!(!CompatibilityMatrix::is_supported(SqlFeature::LeftJoin));
    }
}
