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

//! SQL Token Types
//!
//! Comprehensive token set for SQL-92 with ToonDB extensions.

use std::fmt;
use std::hash::Hash;

/// Source location for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self {
            start,
            end,
            line,
            column,
        }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            line: self.line,
            column: self.column,
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            line: 1,
            column: 1,
        }
    }
}

/// SQL Token with location information
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub literal: String,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span, literal: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            literal: literal.into(),
        }
    }
}

/// Token classification
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Blob(Vec<u8>),
    Null,
    True,
    False,

    // Identifiers
    Identifier(String),
    QuotedIdentifier(String), // "column name" or `column name`

    // Keywords - DDL
    Create,
    Table,
    Index,
    Drop,
    Alter,
    Add,
    Column,
    Rename,
    Primary,
    Key,
    Foreign,
    References,
    Unique,
    Default,
    AutoIncrement,
    If,
    Exists,

    // Keywords - Conflict/Upsert (Dialect Support)
    Ignore,    // MySQL: INSERT IGNORE
    Replace,   // SQLite: INSERT OR REPLACE
    Conflict,  // PostgreSQL: ON CONFLICT
    Do,        // PostgreSQL: ON CONFLICT DO
    Nothing,   // PostgreSQL: DO NOTHING
    Duplicate, // MySQL: ON DUPLICATE KEY UPDATE
    Abort,     // SQLite: INSERT OR ABORT
    Fail,      // SQLite: INSERT OR FAIL
    Returning, // PostgreSQL/SQLite: RETURNING clause

    // Keywords - DML
    Select,
    Insert,
    Update,
    Delete,
    Into,
    Values,
    Set,
    From,
    Where,
    Join,
    Inner,
    Left,
    Right,
    Outer,
    Cross,
    On,
    Using,

    // Keywords - Clauses
    As,
    Distinct,
    All,
    Group,
    Having,
    Order,
    By,
    Asc,
    Desc,
    Nulls,
    First,
    Last,
    Limit,
    Offset,
    Union,
    Intersect,
    Except,

    // Keywords - Expressions
    And,
    Or,
    Not,
    Is,
    In,
    Like,
    Escape,
    Between,
    Case,
    When,
    Then,
    Else,
    End,
    Cast,
    Collate,

    // Keywords - Transactions
    Begin,
    Commit,
    Rollback,
    Transaction,
    Savepoint,
    Release,

    // Keywords - Types
    Int,
    IntegerKw,
    Bigint,
    Smallint,
    Tinyint,
    FloatKw,
    Double,
    Real,
    Decimal,
    Numeric,
    Varchar,
    Char,
    Text,
    BlobKw,
    Boolean,
    Bool,
    Date,
    Time,
    Timestamp,
    Datetime,

    // Keywords - Aggregates
    Count,
    Sum,
    Avg,
    Min,
    Max,

    // Keywords - ToonDB Extensions
    Vector,
    VectorSearch,
    JsonExtract,
    JsonSet,
    ContextWindow,
    Embedding,
    Cosine,
    Euclidean,
    DotProduct,

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Eq,         // =
    Ne,         // != or <>
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    Concat,     // ||
    BitAnd,     // &
    BitOr,      // |
    BitNot,     // ~
    LeftShift,  // <<
    RightShift, // >>

    // Punctuation
    LParen,       // (
    RParen,       // )
    LBracket,     // [
    RBracket,     // ]
    Comma,        // ,
    Semicolon,    // ;
    Dot,          // .
    Colon,        // :
    DoubleColon,  // ::
    Arrow,        // ->
    DoubleArrow,  // ->>
    QuestionMark, // ?
    At,           // @

    // Special
    Placeholder(u32), // $1, $2, ... or ?
    Comment(String),
    Whitespace,
    Eof,
    Invalid(String),
}

impl Eq for TokenKind {}

impl Hash for TokenKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            TokenKind::Integer(n) => n.hash(state),
            TokenKind::Float(f) => f.to_bits().hash(state),
            TokenKind::String(s) => s.hash(state),
            TokenKind::Blob(b) => b.hash(state),
            TokenKind::Identifier(s) => s.hash(state),
            TokenKind::QuotedIdentifier(s) => s.hash(state),
            TokenKind::Placeholder(n) => n.hash(state),
            TokenKind::Comment(s) => s.hash(state),
            TokenKind::Invalid(s) => s.hash(state),
            _ => {}
        }
    }
}

impl TokenKind {
    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Select
                | TokenKind::Insert
                | TokenKind::Update
                | TokenKind::Delete
                | TokenKind::Create
                | TokenKind::Drop
                | TokenKind::From
                | TokenKind::Where
                | TokenKind::And
                | TokenKind::Or
                | TokenKind::Not
                | TokenKind::Join
                | TokenKind::Inner
                | TokenKind::Left
                | TokenKind::Right
                | TokenKind::Outer
                | TokenKind::Cross
                | TokenKind::On
                | TokenKind::As
                | TokenKind::Distinct
                | TokenKind::All
                | TokenKind::Group
                | TokenKind::Having
                | TokenKind::Order
                | TokenKind::By
                | TokenKind::Asc
                | TokenKind::Desc
                | TokenKind::Limit
                | TokenKind::Offset
                | TokenKind::Values
                | TokenKind::Into
                | TokenKind::Set
                | TokenKind::Begin
                | TokenKind::Commit
                | TokenKind::Rollback
                | TokenKind::Table
                | TokenKind::Index
                | TokenKind::Alter
                | TokenKind::Primary
                | TokenKind::Key
                | TokenKind::Foreign
                | TokenKind::References
                | TokenKind::Unique
                | TokenKind::Default
                | TokenKind::If
                | TokenKind::Exists
                | TokenKind::Case
                | TokenKind::When
                | TokenKind::Then
                | TokenKind::Else
                | TokenKind::End
                | TokenKind::Cast
                | TokenKind::Union
                | TokenKind::Intersect
                | TokenKind::Except
                | TokenKind::Count
                | TokenKind::Sum
                | TokenKind::Avg
                | TokenKind::Min
                | TokenKind::Max
                | TokenKind::Is
                | TokenKind::In
                | TokenKind::Like
                | TokenKind::Between
                | TokenKind::Null
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Int
                | TokenKind::IntegerKw
                | TokenKind::Bigint
                | TokenKind::Smallint
                | TokenKind::FloatKw
                | TokenKind::Double
                | TokenKind::Real
                | TokenKind::Varchar
                | TokenKind::Char
                | TokenKind::Text
                | TokenKind::BlobKw
                | TokenKind::Boolean
                | TokenKind::Bool
                | TokenKind::Date
                | TokenKind::Time
                | TokenKind::Timestamp
                | TokenKind::Datetime
                | TokenKind::Vector
                | TokenKind::VectorSearch
                | TokenKind::Embedding
                | TokenKind::Cosine
                | TokenKind::Euclidean
                | TokenKind::DotProduct
                | TokenKind::ContextWindow
                | TokenKind::Using
                | TokenKind::Transaction
                | TokenKind::Savepoint
                | TokenKind::Release
                | TokenKind::Escape
                | TokenKind::Nulls
                | TokenKind::First
                | TokenKind::Last
                | TokenKind::AutoIncrement
                | TokenKind::Add
                | TokenKind::Column
                | TokenKind::Rename
                | TokenKind::Collate
                | TokenKind::Tinyint
                | TokenKind::Decimal
                | TokenKind::Numeric
                | TokenKind::JsonExtract
                | TokenKind::JsonSet
                // Conflict/Upsert keywords
                | TokenKind::Ignore
                | TokenKind::Replace
                | TokenKind::Conflict
                | TokenKind::Do
                | TokenKind::Nothing
                | TokenKind::Duplicate
                | TokenKind::Abort
                | TokenKind::Fail
                | TokenKind::Returning
        )
    }

    /// Get keyword from string (case-insensitive)
    pub fn from_keyword(s: &str) -> Option<TokenKind> {
        match s.to_uppercase().as_str() {
            "SELECT" => Some(TokenKind::Select),
            "INSERT" => Some(TokenKind::Insert),
            "UPDATE" => Some(TokenKind::Update),
            "DELETE" => Some(TokenKind::Delete),
            "CREATE" => Some(TokenKind::Create),
            "TABLE" => Some(TokenKind::Table),
            "DROP" => Some(TokenKind::Drop),
            "ALTER" => Some(TokenKind::Alter),
            "ADD" => Some(TokenKind::Add),
            "COLUMN" => Some(TokenKind::Column),
            "RENAME" => Some(TokenKind::Rename),
            "INDEX" => Some(TokenKind::Index),
            "FROM" => Some(TokenKind::From),
            "WHERE" => Some(TokenKind::Where),
            "AND" => Some(TokenKind::And),
            "OR" => Some(TokenKind::Or),
            "NOT" => Some(TokenKind::Not),
            "NULL" => Some(TokenKind::Null),
            "TRUE" => Some(TokenKind::True),
            "FALSE" => Some(TokenKind::False),
            "IS" => Some(TokenKind::Is),
            "IN" => Some(TokenKind::In),
            "LIKE" => Some(TokenKind::Like),
            "ESCAPE" => Some(TokenKind::Escape),
            "BETWEEN" => Some(TokenKind::Between),
            "JOIN" => Some(TokenKind::Join),
            "INNER" => Some(TokenKind::Inner),
            "LEFT" => Some(TokenKind::Left),
            "RIGHT" => Some(TokenKind::Right),
            "OUTER" => Some(TokenKind::Outer),
            "CROSS" => Some(TokenKind::Cross),
            "ON" => Some(TokenKind::On),
            "USING" => Some(TokenKind::Using),
            "AS" => Some(TokenKind::As),
            "DISTINCT" => Some(TokenKind::Distinct),
            "ALL" => Some(TokenKind::All),
            "GROUP" => Some(TokenKind::Group),
            "HAVING" => Some(TokenKind::Having),
            "ORDER" => Some(TokenKind::Order),
            "BY" => Some(TokenKind::By),
            "ASC" => Some(TokenKind::Asc),
            "DESC" => Some(TokenKind::Desc),
            "NULLS" => Some(TokenKind::Nulls),
            "FIRST" => Some(TokenKind::First),
            "LAST" => Some(TokenKind::Last),
            "LIMIT" => Some(TokenKind::Limit),
            "OFFSET" => Some(TokenKind::Offset),
            "VALUES" => Some(TokenKind::Values),
            "INTO" => Some(TokenKind::Into),
            "SET" => Some(TokenKind::Set),
            "BEGIN" => Some(TokenKind::Begin),
            "COMMIT" => Some(TokenKind::Commit),
            "ROLLBACK" => Some(TokenKind::Rollback),
            "TRANSACTION" => Some(TokenKind::Transaction),
            "SAVEPOINT" => Some(TokenKind::Savepoint),
            "RELEASE" => Some(TokenKind::Release),
            "PRIMARY" => Some(TokenKind::Primary),
            "KEY" => Some(TokenKind::Key),
            "FOREIGN" => Some(TokenKind::Foreign),
            "REFERENCES" => Some(TokenKind::References),
            "UNIQUE" => Some(TokenKind::Unique),
            "DEFAULT" => Some(TokenKind::Default),
            "AUTOINCREMENT" | "AUTO_INCREMENT" => Some(TokenKind::AutoIncrement),
            "IF" => Some(TokenKind::If),
            "EXISTS" => Some(TokenKind::Exists),
            "CASE" => Some(TokenKind::Case),
            "WHEN" => Some(TokenKind::When),
            "THEN" => Some(TokenKind::Then),
            "ELSE" => Some(TokenKind::Else),
            "END" => Some(TokenKind::End),
            "CAST" => Some(TokenKind::Cast),
            "COLLATE" => Some(TokenKind::Collate),
            "UNION" => Some(TokenKind::Union),
            "INTERSECT" => Some(TokenKind::Intersect),
            "EXCEPT" => Some(TokenKind::Except),
            "COUNT" => Some(TokenKind::Count),
            "SUM" => Some(TokenKind::Sum),
            "AVG" => Some(TokenKind::Avg),
            "MIN" => Some(TokenKind::Min),
            "MAX" => Some(TokenKind::Max),
            // Conflict/Upsert keywords
            "IGNORE" => Some(TokenKind::Ignore),
            "REPLACE" => Some(TokenKind::Replace),
            "CONFLICT" => Some(TokenKind::Conflict),
            "DO" => Some(TokenKind::Do),
            "NOTHING" => Some(TokenKind::Nothing),
            "DUPLICATE" => Some(TokenKind::Duplicate),
            "ABORT" => Some(TokenKind::Abort),
            "FAIL" => Some(TokenKind::Fail),
            "RETURNING" => Some(TokenKind::Returning),
            // Types
            "INT" => Some(TokenKind::Int),
            "INTEGER" => Some(TokenKind::IntegerKw),
            "BIGINT" => Some(TokenKind::Bigint),
            "SMALLINT" => Some(TokenKind::Smallint),
            "TINYINT" => Some(TokenKind::Tinyint),
            "FLOAT" => Some(TokenKind::FloatKw),
            "DOUBLE" => Some(TokenKind::Double),
            "REAL" => Some(TokenKind::Real),
            "DECIMAL" => Some(TokenKind::Decimal),
            "NUMERIC" => Some(TokenKind::Numeric),
            "VARCHAR" => Some(TokenKind::Varchar),
            "CHAR" => Some(TokenKind::Char),
            "TEXT" => Some(TokenKind::Text),
            "BLOB" => Some(TokenKind::BlobKw),
            "BOOLEAN" => Some(TokenKind::Boolean),
            "BOOL" => Some(TokenKind::Bool),
            "DATE" => Some(TokenKind::Date),
            "TIME" => Some(TokenKind::Time),
            "TIMESTAMP" => Some(TokenKind::Timestamp),
            "DATETIME" => Some(TokenKind::Datetime),
            // ToonDB Extensions
            "VECTOR" => Some(TokenKind::Vector),
            "VECTOR_SEARCH" => Some(TokenKind::VectorSearch),
            "JSON_EXTRACT" => Some(TokenKind::JsonExtract),
            "JSON_SET" => Some(TokenKind::JsonSet),
            "CONTEXT_WINDOW" => Some(TokenKind::ContextWindow),
            "EMBEDDING" => Some(TokenKind::Embedding),
            "COSINE" => Some(TokenKind::Cosine),
            "EUCLIDEAN" => Some(TokenKind::Euclidean),
            "DOT_PRODUCT" => Some(TokenKind::DotProduct),
            _ => None,
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Integer(n) => write!(f, "{}", n),
            TokenKind::Float(n) => write!(f, "{}", n),
            TokenKind::String(s) => write!(f, "'{}'", s),
            TokenKind::Identifier(s) => write!(f, "{}", s),
            TokenKind::QuotedIdentifier(s) => write!(f, "\"{}\"", s),
            TokenKind::Select => write!(f, "SELECT"),
            TokenKind::From => write!(f, "FROM"),
            TokenKind::Where => write!(f, "WHERE"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::Ne => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Le => write!(f, "<="),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::Ge => write!(f, ">="),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Eof => write!(f, "EOF"),
            TokenKind::Null => write!(f, "NULL"),
            TokenKind::True => write!(f, "TRUE"),
            TokenKind::False => write!(f, "FALSE"),
            TokenKind::And => write!(f, "AND"),
            TokenKind::Or => write!(f, "OR"),
            TokenKind::Not => write!(f, "NOT"),
            _ => write!(f, "{:?}", self),
        }
    }
}
