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

//! SQL Abstract Syntax Tree
//!
//! Represents parsed SQL statements as a tree structure.

use super::token::Span;

/// Top-level SQL statement
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Statement {
    Select(SelectStmt),
    Insert(InsertStmt),
    Update(UpdateStmt),
    Delete(DeleteStmt),
    CreateTable(CreateTableStmt),
    DropTable(DropTableStmt),
    AlterTable(AlterTableStmt),
    CreateIndex(CreateIndexStmt),
    DropIndex(DropIndexStmt),
    Begin(BeginStmt),
    Commit,
    Rollback(Option<String>), // Optional savepoint name
    Savepoint(String),
    Release(String),
    Explain(Box<Statement>),
}

/// SELECT statement
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStmt {
    pub span: Span,
    pub distinct: bool,
    pub columns: Vec<SelectItem>,
    pub from: Option<FromClause>,
    pub where_clause: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<Expr>,
    pub offset: Option<Expr>,
    pub unions: Vec<(SetOp, Box<SelectStmt>)>,
}

/// Items in SELECT clause
#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// SELECT *
    Wildcard,
    /// SELECT table.*
    QualifiedWildcard(String),
    /// SELECT expr [AS alias]
    Expr { expr: Expr, alias: Option<String> },
}

/// Set operations (UNION, INTERSECT, EXCEPT)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOp {
    Union,
    UnionAll,
    Intersect,
    IntersectAll,
    Except,
    ExceptAll,
}

/// FROM clause
#[derive(Debug, Clone, PartialEq)]
pub struct FromClause {
    pub tables: Vec<TableRef>,
}

/// Table reference in FROM clause
#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    /// Simple table: table_name [AS alias]
    Table {
        name: ObjectName,
        alias: Option<String>,
    },
    /// Subquery: (SELECT ...) AS alias
    Subquery {
        query: Box<SelectStmt>,
        alias: String,
    },
    /// Join: left JOIN right ON condition
    Join {
        left: Box<TableRef>,
        join_type: JoinType,
        right: Box<TableRef>,
        condition: Option<JoinCondition>,
    },
    /// Table-valued function: func(...) AS alias
    Function {
        name: String,
        args: Vec<Expr>,
        alias: Option<String>,
    },
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Join condition
#[derive(Debug, Clone, PartialEq)]
pub enum JoinCondition {
    On(Expr),
    Using(Vec<String>),
    Natural,
}

/// ORDER BY item
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByItem {
    pub expr: Expr,
    pub asc: bool,
    pub nulls_first: Option<bool>,
}

/// INSERT statement
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStmt {
    pub span: Span,
    pub table: ObjectName,
    pub columns: Option<Vec<String>>,
    pub source: InsertSource,
    pub on_conflict: Option<OnConflict>,
    pub returning: Option<Vec<SelectItem>>,
}

/// Source of INSERT data
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// VALUES (a, b), (c, d), ...
    Values(Vec<Vec<Expr>>),
    /// SELECT ...
    Query(Box<SelectStmt>),
    /// DEFAULT VALUES
    Default,
}

/// ON CONFLICT clause
///
/// Represents conflict handling for INSERT statements across SQL dialects:
/// - PostgreSQL: `ON CONFLICT DO NOTHING/UPDATE`
/// - MySQL: `INSERT IGNORE`, `ON DUPLICATE KEY UPDATE`
/// - SQLite: `INSERT OR IGNORE/REPLACE/ABORT`
///
/// All dialects normalize to this single representation.
#[derive(Debug, Clone, PartialEq)]
pub struct OnConflict {
    pub target: Option<ConflictTarget>,
    pub action: ConflictAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConflictTarget {
    Columns(Vec<String>),
    Constraint(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConflictAction {
    /// ON CONFLICT DO NOTHING / INSERT IGNORE / INSERT OR IGNORE
    DoNothing,
    /// ON CONFLICT DO UPDATE SET ... / ON DUPLICATE KEY UPDATE ...
    DoUpdate(Vec<Assignment>),
    /// INSERT OR REPLACE (SQLite) - replaces the entire row
    DoReplace,
    /// INSERT OR ABORT (SQLite) - abort on conflict (default behavior)
    DoAbort,
    /// INSERT OR FAIL (SQLite) - fail but continue with other rows
    DoFail,
}

/// UPDATE statement
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStmt {
    pub span: Span,
    pub table: ObjectName,
    pub alias: Option<String>,
    pub assignments: Vec<Assignment>,
    pub from: Option<FromClause>,
    pub where_clause: Option<Expr>,
    pub returning: Option<Vec<SelectItem>>,
}

/// Assignment: column = expr
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub column: String,
    pub value: Expr,
}

/// DELETE statement
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStmt {
    pub span: Span,
    pub table: ObjectName,
    pub alias: Option<String>,
    pub using: Option<FromClause>,
    pub where_clause: Option<Expr>,
    pub returning: Option<Vec<SelectItem>>,
}

/// CREATE TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStmt {
    pub span: Span,
    pub if_not_exists: bool,
    pub name: ObjectName,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
    pub options: Vec<TableOption>,
}

/// Column definition
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub constraints: Vec<ColumnConstraint>,
}

/// SQL Data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    // Numeric
    TinyInt,
    SmallInt,
    Int,
    BigInt,
    Float,
    Double,
    Decimal {
        precision: Option<u32>,
        scale: Option<u32>,
    },

    // String
    Char(Option<u32>),
    Varchar(Option<u32>),
    Text,

    // Binary
    Binary(Option<u32>),
    Varbinary(Option<u32>),
    Blob,

    // Date/Time
    Date,
    Time,
    Timestamp,
    DateTime,
    Interval,

    // Boolean
    Boolean,

    // JSON
    Json,
    Jsonb,

    // ToonDB Extensions
    Vector(u32),    // VECTOR(dimensions)
    Embedding(u32), // EMBEDDING(dimensions)

    // Custom/Unknown
    Custom(String),
}

/// Column constraints
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnConstraint {
    NotNull,
    Null,
    Unique,
    PrimaryKey,
    Default(Expr),
    Check(Expr),
    References {
        table: ObjectName,
        columns: Vec<String>,
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
    },
    AutoIncrement,
    Generated {
        expr: Expr,
        stored: bool,
    },
}

/// Table-level constraints
#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    PrimaryKey {
        name: Option<String>,
        columns: Vec<String>,
    },
    Unique {
        name: Option<String>,
        columns: Vec<String>,
    },
    ForeignKey {
        name: Option<String>,
        columns: Vec<String>,
        ref_table: ObjectName,
        ref_columns: Vec<String>,
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
    },
    Check {
        name: Option<String>,
        expr: Expr,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferentialAction {
    NoAction,
    Restrict,
    Cascade,
    SetNull,
    SetDefault,
}

/// Table options (ENGINE, CHARSET, etc.)
#[derive(Debug, Clone, PartialEq)]
pub struct TableOption {
    pub name: String,
    pub value: String,
}

/// DROP TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct DropTableStmt {
    pub span: Span,
    pub if_exists: bool,
    pub names: Vec<ObjectName>,
    pub cascade: bool,
}

/// ALTER TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStmt {
    pub span: Span,
    pub name: ObjectName,
    pub operations: Vec<AlterTableOp>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterTableOp {
    AddColumn(ColumnDef),
    DropColumn {
        name: String,
        cascade: bool,
    },
    AlterColumn {
        name: String,
        operation: AlterColumnOp,
    },
    AddConstraint(TableConstraint),
    DropConstraint {
        name: String,
        cascade: bool,
    },
    RenameTable(ObjectName),
    RenameColumn {
        old_name: String,
        new_name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterColumnOp {
    SetType(DataType),
    SetNotNull,
    DropNotNull,
    SetDefault(Expr),
    DropDefault,
}

/// CREATE INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStmt {
    pub span: Span,
    pub unique: bool,
    pub if_not_exists: bool,
    pub name: String,
    pub table: ObjectName,
    pub columns: Vec<IndexColumn>,
    pub where_clause: Option<Expr>,
    pub index_type: Option<IndexType>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexColumn {
    pub name: String,
    pub asc: bool,
    pub nulls_first: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BTree,
    Hash,
    Gin,
    Gist,
    // ToonDB extensions
    Hnsw,   // For vector search
    Vamana, // For vector search
}

/// DROP INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexStmt {
    pub span: Span,
    pub if_exists: bool,
    pub name: String,
    pub table: Option<ObjectName>,
    pub cascade: bool,
}

/// BEGIN statement
#[derive(Debug, Clone, PartialEq)]
pub struct BeginStmt {
    pub read_only: bool,
    pub isolation_level: Option<IsolationLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
    Snapshot,
}

/// Object name (potentially qualified: schema.table)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectName {
    pub parts: Vec<String>,
}

impl ObjectName {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            parts: vec![name.into()],
        }
    }

    pub fn qualified(schema: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            parts: vec![schema.into(), name.into()],
        }
    }

    pub fn name(&self) -> &str {
        self.parts.last().map(|s| s.as_str()).unwrap_or("")
    }

    pub fn schema(&self) -> Option<&str> {
        if self.parts.len() > 1 {
            Some(&self.parts[self.parts.len() - 2])
        } else {
            None
        }
    }
}

impl std::fmt::Display for ObjectName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.parts.join("."))
    }
}

/// Expression
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value
    Literal(Literal),

    /// Column reference: [table.]column
    Column(ColumnRef),

    /// Binary operation: expr op expr
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },

    /// Unary operation: op expr
    UnaryOp { op: UnaryOperator, expr: Box<Expr> },

    /// Function call: func(args)
    Function(FunctionCall),

    /// CASE expression
    Case {
        operand: Option<Box<Expr>>,
        conditions: Vec<(Expr, Expr)>, // (WHEN, THEN)
        else_result: Option<Box<Expr>>,
    },

    /// Subquery: (SELECT ...)
    Subquery(Box<SelectStmt>),

    /// EXISTS (SELECT ...)
    Exists(Box<SelectStmt>),

    /// expr IN (values)
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },

    /// expr IN (SELECT ...)
    InSubquery {
        expr: Box<Expr>,
        subquery: Box<SelectStmt>,
        negated: bool,
    },

    /// expr BETWEEN low AND high
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },

    /// expr LIKE pattern [ESCAPE escape]
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        escape: Option<Box<Expr>>,
        negated: bool,
    },

    /// expr IS [NOT] NULL
    IsNull { expr: Box<Expr>, negated: bool },

    /// CAST(expr AS type)
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
    },

    /// Placeholder: $1, $2, ?
    Placeholder(u32),

    /// Array: [a, b, c] or ARRAY[a, b, c]
    Array(Vec<Expr>),

    /// Tuple/Row: (a, b, c)
    Tuple(Vec<Expr>),

    /// Array subscript: arr[index]
    Subscript { expr: Box<Expr>, index: Box<Expr> },

    // ========== ToonDB Extensions ==========
    /// Vector literal: [1.0, 2.0, 3.0]::VECTOR
    Vector(Vec<f32>),

    /// Vector search: VECTOR_SEARCH(column, query_vector, k, metric)
    VectorSearch {
        column: Box<Expr>,
        query: Box<Expr>,
        k: u32,
        metric: VectorMetric,
    },

    /// JSON path: json_col -> 'path'
    JsonAccess {
        expr: Box<Expr>,
        path: Box<Expr>,
        return_text: bool, // -> vs ->>
    },

    /// Context window for LLM: CONTEXT_WINDOW(tokens, priority_expr)
    ContextWindow {
        source: Box<Expr>,
        max_tokens: u32,
        priority: Option<Box<Expr>>,
    },
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Blob(Vec<u8>),
}

/// Column reference
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnRef {
    pub table: Option<String>,
    pub column: String,
}

impl ColumnRef {
    pub fn new(column: impl Into<String>) -> Self {
        Self {
            table: None,
            column: column.into(),
        }
    }

    pub fn qualified(table: impl Into<String>, column: impl Into<String>) -> Self {
        Self {
            table: Some(table.into()),
            column: column.into(),
        }
    }
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    // Arithmetic
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,

    // String
    Concat,
    Like,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Plus,
    Minus,
    Not,
    BitNot,
}

/// Function call
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: ObjectName,
    pub args: Vec<Expr>,
    pub distinct: bool,
    pub filter: Option<Box<Expr>>,
    pub over: Option<WindowSpec>,
}

/// Window specification for window functions
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    pub partition_by: Vec<Expr>,
    pub order_by: Vec<OrderByItem>,
    pub frame: Option<WindowFrame>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    pub kind: WindowFrameKind,
    pub start: WindowFrameBound,
    pub end: Option<WindowFrameBound>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameKind {
    Rows,
    Range,
    Groups,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WindowFrameBound {
    CurrentRow,
    Preceding(Option<Box<Expr>>), // None = UNBOUNDED
    Following(Option<Box<Expr>>), // None = UNBOUNDED
}

/// Vector distance metrics (ToonDB extension)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}
