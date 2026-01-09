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

//! SQL Parser
//!
//! Recursive descent parser for SQL grammar.
//! Produces AST from token stream.

use super::ast::*;
use super::lexer::Lexer;
use super::token::{Span, Token, TokenKind};

/// Parser errors
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
    pub expected: Vec<String>,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            expected: Vec::new(),
        }
    }

    pub fn expected(mut self, expected: impl Into<String>) -> Self {
        self.expected.push(expected.into());
        self
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parse error at line {}, column {}: {}",
            self.span.line, self.span.column, self.message
        )?;
        if !self.expected.is_empty() {
            write!(f, " (expected: {})", self.expected.join(", "))?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// SQL Parser
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    /// Create a new parser from tokens
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse a SQL string into a statement
    pub fn parse(sql: &str) -> Result<Statement, Vec<ParseError>> {
        let tokens = Lexer::new(sql).tokenize().map_err(|lex_errors| {
            lex_errors
                .into_iter()
                .map(|e| ParseError::new(e.message, e.span))
                .collect::<Vec<_>>()
        })?;

        let mut parser = Parser::new(tokens);
        parser.parse_statement()
    }

    /// Parse multiple statements (semicolon-separated)
    pub fn parse_statements(sql: &str) -> Result<Vec<Statement>, Vec<ParseError>> {
        let tokens = Lexer::new(sql).tokenize().map_err(|lex_errors| {
            lex_errors
                .into_iter()
                .map(|e| ParseError::new(e.message, e.span))
                .collect::<Vec<_>>()
        })?;

        let mut parser = Parser::new(tokens);
        let mut statements = Vec::new();

        while !parser.is_at_end() {
            match parser.parse_statement() {
                Ok(stmt) => {
                    statements.push(stmt);
                    // Consume optional semicolon
                    parser.match_token(&TokenKind::Semicolon);
                }
                Err(errors) => return Err(errors),
            }
        }

        Ok(statements)
    }

    // ========== Helper Methods ==========

    fn is_at_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or(&self.tokens[self.tokens.len() - 1])
    }

    fn peek_nth(&self, n: usize) -> &Token {
        self.tokens
            .get(self.pos + n)
            .unwrap_or(&self.tokens[self.tokens.len() - 1])
    }

    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.pos += 1;
        }
        self.tokens.get(self.pos - 1).cloned().unwrap()
    }

    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    fn check_keyword(&self, kw: &TokenKind) -> bool {
        self.peek().kind == *kw
    }

    fn match_token(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind, message: &str) -> Result<Token, ParseError> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            Err(ParseError::new(message, self.peek().span).expected(format!("{:?}", kind)))
        }
    }

    fn expect_identifier(&mut self, message: &str) -> Result<String, ParseError> {
        match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            TokenKind::QuotedIdentifier(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(ParseError::new(message, self.peek().span).expected("identifier")),
        }
    }

    fn current_span(&self) -> Span {
        self.peek().span
    }

    // ========== Statement Parsing ==========

    fn parse_statement(&mut self) -> Result<Statement, Vec<ParseError>> {
        let result = match &self.peek().kind {
            TokenKind::Select => self.parse_select().map(Statement::Select),
            TokenKind::Insert => self.parse_insert().map(Statement::Insert),
            TokenKind::Update => self.parse_update().map(Statement::Update),
            TokenKind::Delete => self.parse_delete().map(Statement::Delete),
            TokenKind::Create => self.parse_create(),
            TokenKind::Drop => self.parse_drop(),
            TokenKind::Alter => self.parse_alter(),
            TokenKind::Begin => self.parse_begin().map(Statement::Begin),
            TokenKind::Commit => {
                self.advance();
                Ok(Statement::Commit)
            }
            TokenKind::Rollback => self.parse_rollback(),
            TokenKind::Savepoint => self.parse_savepoint(),
            TokenKind::Release => self.parse_release(),
            _ => Err(ParseError::new(
                format!("Unexpected token: {:?}", self.peek().kind),
                self.peek().span,
            )),
        };

        result.map_err(|e| vec![e])
    }

    // ========== SELECT Parsing ==========

    fn parse_select(&mut self) -> Result<SelectStmt, ParseError> {
        let start_span = self.current_span();
        self.expect(&TokenKind::Select, "Expected SELECT")?;

        // DISTINCT?
        let distinct = self.match_token(&TokenKind::Distinct);
        if !distinct {
            self.match_token(&TokenKind::All);
        }

        // Select list
        let columns = self.parse_select_list()?;

        // FROM clause (optional for SELECT 1+1 style queries)
        let from = if self.match_token(&TokenKind::From) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        // WHERE clause
        let where_clause = if self.match_token(&TokenKind::Where) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // GROUP BY clause
        let group_by = if self.check_keyword(&TokenKind::Group) {
            self.advance();
            self.expect(&TokenKind::By, "Expected BY after GROUP")?;
            self.parse_expr_list()?
        } else {
            Vec::new()
        };

        // HAVING clause
        let having = if self.match_token(&TokenKind::Having) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // ORDER BY clause
        let order_by = if self.check_keyword(&TokenKind::Order) {
            self.advance();
            self.expect(&TokenKind::By, "Expected BY after ORDER")?;
            self.parse_order_by_list()?
        } else {
            Vec::new()
        };

        // LIMIT clause
        let limit = if self.match_token(&TokenKind::Limit) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // OFFSET clause
        let offset = if self.match_token(&TokenKind::Offset) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // Set operations (UNION, INTERSECT, EXCEPT)
        let mut unions = Vec::new();
        loop {
            let set_op = if self.match_token(&TokenKind::Union) {
                if self.match_token(&TokenKind::All) {
                    SetOp::UnionAll
                } else {
                    SetOp::Union
                }
            } else if self.match_token(&TokenKind::Intersect) {
                if self.match_token(&TokenKind::All) {
                    SetOp::IntersectAll
                } else {
                    SetOp::Intersect
                }
            } else if self.match_token(&TokenKind::Except) {
                if self.match_token(&TokenKind::All) {
                    SetOp::ExceptAll
                } else {
                    SetOp::Except
                }
            } else {
                break;
            };

            let right = self.parse_select()?;
            unions.push((set_op, Box::new(right)));
        }

        Ok(SelectStmt {
            span: start_span.merge(self.current_span()),
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
            unions,
        })
    }

    fn parse_select_list(&mut self) -> Result<Vec<SelectItem>, ParseError> {
        let mut items = Vec::new();

        loop {
            items.push(self.parse_select_item()?);

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(items)
    }

    fn parse_select_item(&mut self) -> Result<SelectItem, ParseError> {
        // Check for *
        if self.match_token(&TokenKind::Star) {
            return Ok(SelectItem::Wildcard);
        }

        // Check for table.*
        if let TokenKind::Identifier(name) = &self.peek().kind
            && self.peek_nth(1).kind == TokenKind::Dot
            && self.peek_nth(2).kind == TokenKind::Star
        {
            let table = name.clone();
            self.advance(); // identifier
            self.advance(); // .
            self.advance(); // *
            return Ok(SelectItem::QualifiedWildcard(table));
        }

        // Expression with optional alias
        let expr = self.parse_expr()?;

        let alias = if self.match_token(&TokenKind::As) {
            Some(self.expect_identifier("Expected alias after AS")?)
        } else if let TokenKind::Identifier(name) = &self.peek().kind {
            // Implicit alias (without AS)
            if !self.check_keyword(&TokenKind::From)
                && !self.check_keyword(&TokenKind::Where)
                && !self.check(&TokenKind::Comma)
                && !self.check_keyword(&TokenKind::Order)
                && !self.check_keyword(&TokenKind::Group)
                && !self.check_keyword(&TokenKind::Limit)
                && !self.is_at_end()
            {
                let name = name.clone();
                self.advance();
                Some(name)
            } else {
                None
            }
        } else {
            None
        };

        Ok(SelectItem::Expr { expr, alias })
    }

    fn parse_from_clause(&mut self) -> Result<FromClause, ParseError> {
        let mut tables = vec![self.parse_table_ref()?];

        while self.match_token(&TokenKind::Comma) {
            tables.push(self.parse_table_ref()?);
        }

        Ok(FromClause { tables })
    }

    fn parse_table_ref(&mut self) -> Result<TableRef, ParseError> {
        let mut table = self.parse_table_primary()?;

        // Parse joins
        loop {
            let join_type = if self.match_token(&TokenKind::Cross) {
                self.expect(&TokenKind::Join, "Expected JOIN after CROSS")?;
                JoinType::Cross
            } else if self.match_token(&TokenKind::Inner) {
                self.expect(&TokenKind::Join, "Expected JOIN after INNER")?;
                JoinType::Inner
            } else if self.match_token(&TokenKind::Left) {
                self.match_token(&TokenKind::Outer);
                self.expect(&TokenKind::Join, "Expected JOIN after LEFT")?;
                JoinType::Left
            } else if self.match_token(&TokenKind::Right) {
                self.match_token(&TokenKind::Outer);
                self.expect(&TokenKind::Join, "Expected JOIN after RIGHT")?;
                JoinType::Right
            } else if self.match_token(&TokenKind::Join) {
                JoinType::Inner // Default join is INNER
            } else {
                break;
            };

            let right = self.parse_table_primary()?;

            let condition = if join_type == JoinType::Cross {
                None
            } else if self.match_token(&TokenKind::On) {
                Some(JoinCondition::On(self.parse_expr()?))
            } else if self.match_token(&TokenKind::Using) {
                self.expect(&TokenKind::LParen, "Expected '(' after USING")?;
                let columns = self.parse_identifier_list()?;
                self.expect(&TokenKind::RParen, "Expected ')' after USING columns")?;
                Some(JoinCondition::Using(columns))
            } else {
                return Err(ParseError::new(
                    "Expected ON or USING clause for JOIN",
                    self.current_span(),
                ));
            };

            table = TableRef::Join {
                left: Box::new(table),
                join_type,
                right: Box::new(right),
                condition,
            };
        }

        Ok(table)
    }

    fn parse_table_primary(&mut self) -> Result<TableRef, ParseError> {
        // Subquery: (SELECT ...)
        if self.match_token(&TokenKind::LParen) {
            let query = self.parse_select()?;
            self.expect(&TokenKind::RParen, "Expected ')' after subquery")?;

            self.match_token(&TokenKind::As);
            let alias = self.expect_identifier("Subquery requires an alias")?;

            return Ok(TableRef::Subquery {
                query: Box::new(query),
                alias,
            });
        }

        // Table name
        let name = self.parse_object_name()?;

        // Optional alias
        let alias = if self.match_token(&TokenKind::As) {
            Some(self.expect_identifier("Expected alias after AS")?)
        } else if let TokenKind::Identifier(id) = &self.peek().kind {
            // Check it's not a keyword
            if !self.peek().kind.is_keyword() {
                let alias = id.clone();
                self.advance();
                Some(alias)
            } else {
                None
            }
        } else {
            None
        };

        Ok(TableRef::Table { name, alias })
    }

    // ========== INSERT Parsing ==========

    fn parse_insert(&mut self) -> Result<InsertStmt, ParseError> {
        let start_span = self.current_span();
        self.expect(&TokenKind::Insert, "Expected INSERT")?;

        // Check for MySQL-style INSERT IGNORE
        let mysql_ignore = self.match_token(&TokenKind::Ignore);

        // Check for SQLite-style INSERT OR {IGNORE|REPLACE|ABORT|FAIL}
        let sqlite_conflict_action = if self.match_token(&TokenKind::Or) {
            if self.match_token(&TokenKind::Ignore) {
                Some(ConflictAction::DoNothing)
            } else if self.match_token(&TokenKind::Replace) {
                Some(ConflictAction::DoReplace)
            } else if self.match_token(&TokenKind::Abort) {
                Some(ConflictAction::DoAbort)
            } else if self.match_token(&TokenKind::Fail) {
                Some(ConflictAction::DoFail)
            } else {
                return Err(ParseError::new(
                    "Expected IGNORE, REPLACE, ABORT, or FAIL after OR",
                    self.current_span(),
                ));
            }
        } else {
            None
        };

        self.expect(&TokenKind::Into, "Expected INTO")?;

        let table = self.parse_object_name()?;

        // Optional column list
        let columns = if self.match_token(&TokenKind::LParen) {
            let cols = self.parse_identifier_list()?;
            self.expect(&TokenKind::RParen, "Expected ')' after column list")?;
            Some(cols)
        } else {
            None
        };

        // VALUES or SELECT
        let source = if self.match_token(&TokenKind::Values) {
            InsertSource::Values(self.parse_values_list()?)
        } else if self.check_keyword(&TokenKind::Select) {
            InsertSource::Query(Box::new(self.parse_select()?))
        } else if self.match_token(&TokenKind::Default) {
            self.expect(&TokenKind::Values, "Expected VALUES after DEFAULT")?;
            InsertSource::Default
        } else {
            return Err(ParseError::new(
                "Expected VALUES or SELECT",
                self.current_span(),
            ));
        };

        // Parse ON CONFLICT (PostgreSQL) or ON DUPLICATE KEY UPDATE (MySQL)
        let on_conflict = if self.match_token(&TokenKind::On) {
            if self.match_token(&TokenKind::Conflict) {
                // PostgreSQL: ON CONFLICT [target] DO {NOTHING | UPDATE SET ...}
                Some(self.parse_on_conflict()?)
            } else if self.match_token(&TokenKind::Duplicate) {
                // MySQL: ON DUPLICATE KEY UPDATE ...
                self.expect(&TokenKind::Key, "Expected KEY after DUPLICATE")?;
                self.expect(&TokenKind::Update, "Expected UPDATE after KEY")?;
                let assignments = self.parse_assignments()?;
                Some(OnConflict {
                    target: None,
                    action: ConflictAction::DoUpdate(assignments),
                })
            } else {
                return Err(ParseError::new(
                    "Expected CONFLICT or DUPLICATE after ON",
                    self.current_span(),
                ));
            }
        } else if mysql_ignore {
            // MySQL INSERT IGNORE normalizes to DoNothing
            Some(OnConflict {
                target: None,
                action: ConflictAction::DoNothing,
            })
        } else {
            // SQLite conflict action from OR clause
            sqlite_conflict_action.map(|action| OnConflict {
                target: None,
                action,
            })
        };

        // RETURNING clause (PostgreSQL/SQLite)
        let returning = if self.match_token(&TokenKind::Returning) {
            Some(self.parse_select_list()?)
        } else {
            None
        };

        Ok(InsertStmt {
            span: start_span.merge(self.current_span()),
            table,
            columns,
            source,
            on_conflict,
            returning,
        })
    }

    /// Parse ON CONFLICT clause (PostgreSQL style)
    fn parse_on_conflict(&mut self) -> Result<OnConflict, ParseError> {
        // Optional conflict target: (columns) or ON CONSTRAINT name
        let target = if self.match_token(&TokenKind::LParen) {
            let cols = self.parse_identifier_list()?;
            self.expect(&TokenKind::RParen, "Expected ')' after conflict columns")?;
            Some(ConflictTarget::Columns(cols))
        } else if self.match_token(&TokenKind::On) {
            // ON CONSTRAINT name (though this is a bit unusual syntax)
            // Actually PostgreSQL uses just ON CONFLICT ON CONSTRAINT name
            // Let's handle the standard case
            None
        } else {
            None
        };

        // DO {NOTHING | UPDATE SET ...}
        self.expect(&TokenKind::Do, "Expected DO after ON CONFLICT")?;

        let action = if self.match_token(&TokenKind::Nothing) {
            ConflictAction::DoNothing
        } else if self.match_token(&TokenKind::Update) {
            self.expect(&TokenKind::Set, "Expected SET after UPDATE")?;
            let assignments = self.parse_assignments()?;
            ConflictAction::DoUpdate(assignments)
        } else {
            return Err(ParseError::new(
                "Expected NOTHING or UPDATE after DO",
                self.current_span(),
            ));
        };

        Ok(OnConflict { target, action })
    }

    fn parse_values_list(&mut self) -> Result<Vec<Vec<Expr>>, ParseError> {
        let mut rows = Vec::new();

        loop {
            self.expect(&TokenKind::LParen, "Expected '(' for VALUES row")?;
            let row = self.parse_expr_list()?;
            self.expect(&TokenKind::RParen, "Expected ')' after VALUES row")?;
            rows.push(row);

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(rows)
    }

    // ========== UPDATE Parsing ==========

    fn parse_update(&mut self) -> Result<UpdateStmt, ParseError> {
        let start_span = self.current_span();
        self.expect(&TokenKind::Update, "Expected UPDATE")?;

        let table = self.parse_object_name()?;

        let alias = if self.match_token(&TokenKind::As) {
            Some(self.expect_identifier("Expected alias after AS")?)
        } else {
            None
        };

        self.expect(&TokenKind::Set, "Expected SET")?;

        let assignments = self.parse_assignments()?;

        let from = if self.match_token(&TokenKind::From) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        let where_clause = if self.match_token(&TokenKind::Where) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        let returning = None; // TODO

        Ok(UpdateStmt {
            span: start_span.merge(self.current_span()),
            table,
            alias,
            assignments,
            from,
            where_clause,
            returning,
        })
    }

    fn parse_assignments(&mut self) -> Result<Vec<Assignment>, ParseError> {
        let mut assignments = Vec::new();

        loop {
            let column = self.expect_identifier("Expected column name")?;
            self.expect(&TokenKind::Eq, "Expected '=' after column name")?;
            let value = self.parse_expr()?;

            assignments.push(Assignment { column, value });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(assignments)
    }

    // ========== DELETE Parsing ==========

    fn parse_delete(&mut self) -> Result<DeleteStmt, ParseError> {
        let start_span = self.current_span();
        self.expect(&TokenKind::Delete, "Expected DELETE")?;
        self.expect(&TokenKind::From, "Expected FROM")?;

        let table = self.parse_object_name()?;

        let alias = if self.match_token(&TokenKind::As) {
            Some(self.expect_identifier("Expected alias after AS")?)
        } else {
            None
        };

        let using = None; // TODO: Parse USING clause

        let where_clause = if self.match_token(&TokenKind::Where) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(DeleteStmt {
            span: start_span.merge(self.current_span()),
            table,
            alias,
            using,
            where_clause,
            returning: None,
        })
    }

    // ========== DDL Parsing ==========

    fn parse_create(&mut self) -> Result<Statement, ParseError> {
        self.expect(&TokenKind::Create, "Expected CREATE")?;

        // Check for CREATE UNIQUE INDEX
        let unique = self.match_token(&TokenKind::Unique);

        if self.match_token(&TokenKind::Table) {
            self.parse_create_table().map(Statement::CreateTable)
        } else if self.match_token(&TokenKind::Index) {
            self.parse_create_index(unique).map(Statement::CreateIndex)
        } else if unique {
            // After UNIQUE, must be INDEX
            Err(ParseError::new(
                "Expected INDEX after UNIQUE",
                self.current_span(),
            ))
        } else {
            Err(ParseError::new(
                "Expected TABLE or INDEX after CREATE",
                self.current_span(),
            ))
        }
    }

    fn parse_create_index(&mut self, unique: bool) -> Result<CreateIndexStmt, ParseError> {
        let start_span = self.current_span();

        // IF NOT EXISTS
        let if_not_exists = if self.match_token(&TokenKind::If) {
            self.expect(&TokenKind::Not, "Expected NOT after IF")?;
            self.expect(&TokenKind::Exists, "Expected EXISTS after IF NOT")?;
            true
        } else {
            false
        };

        // Index name
        let name = self.expect_identifier("Expected index name")?;

        self.expect(&TokenKind::On, "Expected ON after index name")?;

        // Table name
        let table = self.parse_object_name()?;

        // Column list
        self.expect(&TokenKind::LParen, "Expected '(' after table name")?;
        let mut columns = Vec::new();
        loop {
            let col_name = self.expect_identifier("Expected column name")?;
            
            // Optional ASC/DESC
            let asc = if self.match_token(&TokenKind::Desc) {
                false
            } else {
                self.match_token(&TokenKind::Asc);
                true
            };

            columns.push(IndexColumn {
                name: col_name,
                asc,
                nulls_first: None,
            });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "Expected ')' after column list")?;

        // Optional WHERE clause for partial indexes
        let where_clause = if self.match_token(&TokenKind::Where) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(CreateIndexStmt {
            span: start_span.merge(self.current_span()),
            unique,
            if_not_exists,
            name,
            table,
            columns,
            where_clause,
            index_type: None,
        })
    }

    fn parse_create_table(&mut self) -> Result<CreateTableStmt, ParseError> {
        let start_span = self.current_span();

        let if_not_exists = if self.match_token(&TokenKind::If) {
            self.expect(&TokenKind::Not, "Expected NOT after IF")?;
            self.expect(&TokenKind::Exists, "Expected EXISTS after IF NOT")?;
            true
        } else {
            false
        };

        let name = self.parse_object_name()?;

        self.expect(&TokenKind::LParen, "Expected '(' after table name")?;

        let mut columns = Vec::new();
        let constraints = Vec::new();

        loop {
            // Check for table constraint keywords
            if self.check_keyword(&TokenKind::Primary)
                || self.check_keyword(&TokenKind::Foreign)
                || self.check_keyword(&TokenKind::Unique)
            {
                // TODO: Parse table constraints
                break;
            }

            // Check for end of column list
            if self.check(&TokenKind::RParen) {
                break;
            }

            // Parse column definition
            columns.push(self.parse_column_def()?);

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        self.expect(&TokenKind::RParen, "Expected ')' after column definitions")?;

        Ok(CreateTableStmt {
            span: start_span.merge(self.current_span()),
            if_not_exists,
            name,
            columns,
            constraints,
            options: Vec::new(),
        })
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef, ParseError> {
        let name = self.expect_identifier("Expected column name")?;
        let data_type = self.parse_data_type()?;

        let mut constraints = Vec::new();

        // Parse column constraints
        loop {
            if self.match_token(&TokenKind::Primary) {
                self.expect(&TokenKind::Key, "Expected KEY after PRIMARY")?;
                constraints.push(ColumnConstraint::PrimaryKey);
            } else if self.match_token(&TokenKind::Not) {
                self.expect(&TokenKind::Null, "Expected NULL after NOT")?;
                constraints.push(ColumnConstraint::NotNull);
            } else if self.match_token(&TokenKind::Null) {
                constraints.push(ColumnConstraint::Null);
            } else if self.match_token(&TokenKind::Unique) {
                constraints.push(ColumnConstraint::Unique);
            } else if self.match_token(&TokenKind::Default) {
                constraints.push(ColumnConstraint::Default(self.parse_expr()?));
            } else if self.match_token(&TokenKind::AutoIncrement) {
                constraints.push(ColumnConstraint::AutoIncrement);
            } else {
                break;
            }
        }

        Ok(ColumnDef {
            name,
            data_type,
            constraints,
        })
    }

    fn parse_data_type(&mut self) -> Result<DataType, ParseError> {
        let type_name = match &self.peek().kind {
            TokenKind::Int | TokenKind::IntegerKw => {
                self.advance();
                DataType::Int
            }
            TokenKind::Bigint => {
                self.advance();
                DataType::BigInt
            }
            TokenKind::Smallint => {
                self.advance();
                DataType::SmallInt
            }
            TokenKind::Tinyint => {
                self.advance();
                DataType::TinyInt
            }
            TokenKind::FloatKw | TokenKind::Real => {
                self.advance();
                DataType::Float
            }
            TokenKind::Double => {
                self.advance();
                DataType::Double
            }
            TokenKind::Varchar => {
                self.advance();
                let len = self.parse_type_length()?;
                DataType::Varchar(len)
            }
            TokenKind::Char => {
                self.advance();
                let len = self.parse_type_length()?;
                DataType::Char(len)
            }
            TokenKind::Text => {
                self.advance();
                DataType::Text
            }
            TokenKind::BlobKw => {
                self.advance();
                DataType::Blob
            }
            TokenKind::Boolean | TokenKind::Bool => {
                self.advance();
                DataType::Boolean
            }
            TokenKind::Date => {
                self.advance();
                DataType::Date
            }
            TokenKind::Time => {
                self.advance();
                DataType::Time
            }
            TokenKind::Timestamp | TokenKind::Datetime => {
                self.advance();
                DataType::Timestamp
            }
            TokenKind::Vector => {
                self.advance();
                let dims = self.parse_type_length()?.unwrap_or(128);
                DataType::Vector(dims)
            }
            TokenKind::Embedding => {
                self.advance();
                let dims = self.parse_type_length()?.unwrap_or(1536);
                DataType::Embedding(dims)
            }
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                DataType::Custom(name)
            }
            _ => {
                return Err(ParseError::new(
                    format!("Expected data type, got {:?}", self.peek().kind),
                    self.current_span(),
                ));
            }
        };

        Ok(type_name)
    }

    fn parse_type_length(&mut self) -> Result<Option<u32>, ParseError> {
        if self.match_token(&TokenKind::LParen) {
            let len = match &self.peek().kind {
                TokenKind::Integer(n) => {
                    let n = *n as u32;
                    self.advance();
                    n
                }
                _ => return Err(ParseError::new("Expected integer", self.current_span())),
            };
            self.expect(&TokenKind::RParen, "Expected ')'")?;
            Ok(Some(len))
        } else {
            Ok(None)
        }
    }

    fn parse_drop(&mut self) -> Result<Statement, ParseError> {
        let start_span = self.current_span();
        self.expect(&TokenKind::Drop, "Expected DROP")?;

        if self.match_token(&TokenKind::Table) {
            let if_exists = if self.match_token(&TokenKind::If) {
                self.expect(&TokenKind::Exists, "Expected EXISTS after IF")?;
                true
            } else {
                false
            };

            let name = self.parse_object_name()?;
            let cascade = false; // TODO: Parse CASCADE

            Ok(Statement::DropTable(DropTableStmt {
                span: start_span.merge(self.current_span()),
                if_exists,
                names: vec![name],
                cascade,
            }))
        } else if self.match_token(&TokenKind::Index) {
            let if_exists = if self.match_token(&TokenKind::If) {
                self.expect(&TokenKind::Exists, "Expected EXISTS after IF")?;
                true
            } else {
                false
            };

            let name = self.expect_identifier("Expected index name")?;

            // Optional ON table_name (PostgreSQL style)
            let table = if self.match_token(&TokenKind::On) {
                Some(self.parse_object_name()?)
            } else {
                None
            };

            Ok(Statement::DropIndex(DropIndexStmt {
                span: start_span.merge(self.current_span()),
                if_exists,
                name,
                table,
                cascade: false,
            }))
        } else {
            Err(ParseError::new(
                "Expected TABLE or INDEX after DROP",
                self.current_span(),
            ))
        }
    }

    fn parse_alter(&mut self) -> Result<Statement, ParseError> {
        // TODO: Implement ALTER TABLE
        Err(ParseError::new(
            "ALTER not yet implemented",
            self.current_span(),
        ))
    }

    // ========== Transaction Parsing ==========

    fn parse_begin(&mut self) -> Result<BeginStmt, ParseError> {
        self.expect(&TokenKind::Begin, "Expected BEGIN")?;
        self.match_token(&TokenKind::Transaction);

        // TODO: Parse isolation level
        Ok(BeginStmt {
            read_only: false,
            isolation_level: None,
        })
    }

    fn parse_rollback(&mut self) -> Result<Statement, ParseError> {
        self.expect(&TokenKind::Rollback, "Expected ROLLBACK")?;
        self.match_token(&TokenKind::Transaction);

        // Check for ROLLBACK TO SAVEPOINT
        // TODO

        Ok(Statement::Rollback(None))
    }

    fn parse_savepoint(&mut self) -> Result<Statement, ParseError> {
        self.expect(&TokenKind::Savepoint, "Expected SAVEPOINT")?;
        let name = self.expect_identifier("Expected savepoint name")?;
        Ok(Statement::Savepoint(name))
    }

    fn parse_release(&mut self) -> Result<Statement, ParseError> {
        self.expect(&TokenKind::Release, "Expected RELEASE")?;
        self.match_token(&TokenKind::Savepoint);
        let name = self.expect_identifier("Expected savepoint name")?;
        Ok(Statement::Release(name))
    }

    // ========== Expression Parsing ==========

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and_expr()?;

        while self.match_token(&TokenKind::Or) {
            let right = self.parse_and_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_not_expr()?;

        while self.match_token(&TokenKind::And) {
            let right = self.parse_not_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<Expr, ParseError> {
        if self.match_token(&TokenKind::Not) {
            let expr = self.parse_not_expr()?;
            Ok(Expr::UnaryOp {
                op: UnaryOperator::Not,
                expr: Box::new(expr),
            })
        } else {
            self.parse_comparison_expr()
        }
    }

    fn parse_comparison_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive_expr()?;

        // IS NULL / IS NOT NULL
        if self.match_token(&TokenKind::Is) {
            let negated = self.match_token(&TokenKind::Not);
            self.expect(&TokenKind::Null, "Expected NULL after IS")?;
            return Ok(Expr::IsNull {
                expr: Box::new(left),
                negated,
            });
        }

        // IN / NOT IN
        let negated = self.match_token(&TokenKind::Not);
        if self.match_token(&TokenKind::In) {
            self.expect(&TokenKind::LParen, "Expected '(' after IN")?;

            if self.check_keyword(&TokenKind::Select) {
                let subquery = self.parse_select()?;
                self.expect(&TokenKind::RParen, "Expected ')'")?;
                return Ok(Expr::InSubquery {
                    expr: Box::new(left),
                    subquery: Box::new(subquery),
                    negated,
                });
            } else {
                let list = self.parse_expr_list()?;
                self.expect(&TokenKind::RParen, "Expected ')'")?;
                return Ok(Expr::InList {
                    expr: Box::new(left),
                    list,
                    negated,
                });
            }
        }

        // BETWEEN
        if self.match_token(&TokenKind::Between) {
            let low = self.parse_additive_expr()?;
            self.expect(&TokenKind::And, "Expected AND in BETWEEN")?;
            let high = self.parse_additive_expr()?;
            return Ok(Expr::Between {
                expr: Box::new(left),
                low: Box::new(low),
                high: Box::new(high),
                negated,
            });
        }

        // LIKE
        if self.match_token(&TokenKind::Like) {
            let pattern = self.parse_additive_expr()?;
            let escape = if self.match_token(&TokenKind::Escape) {
                Some(Box::new(self.parse_additive_expr()?))
            } else {
                None
            };
            return Ok(Expr::Like {
                expr: Box::new(left),
                pattern: Box::new(pattern),
                escape,
                negated,
            });
        }

        // If we consumed NOT but didn't find IN/BETWEEN/LIKE, error
        if negated {
            return Err(ParseError::new(
                "Expected IN, BETWEEN, or LIKE after NOT",
                self.current_span(),
            ));
        }

        // Comparison operators
        let op = match &self.peek().kind {
            TokenKind::Eq => Some(BinaryOperator::Eq),
            TokenKind::Ne => Some(BinaryOperator::Ne),
            TokenKind::Lt => Some(BinaryOperator::Lt),
            TokenKind::Le => Some(BinaryOperator::Le),
            TokenKind::Gt => Some(BinaryOperator::Gt),
            TokenKind::Ge => Some(BinaryOperator::Ge),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let right = self.parse_additive_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_additive_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative_expr()?;

        loop {
            let op = match &self.peek().kind {
                TokenKind::Plus => BinaryOperator::Plus,
                TokenKind::Minus => BinaryOperator::Minus,
                TokenKind::Concat => BinaryOperator::Concat,
                _ => break,
            };
            self.advance();

            let right = self.parse_multiplicative_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_multiplicative_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary_expr()?;

        loop {
            let op = match &self.peek().kind {
                TokenKind::Star => BinaryOperator::Multiply,
                TokenKind::Slash => BinaryOperator::Divide,
                TokenKind::Percent => BinaryOperator::Modulo,
                _ => break,
            };
            self.advance();

            let right = self.parse_unary_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        match &self.peek().kind {
            TokenKind::Minus => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: Box::new(expr),
                })
            }
            TokenKind::Plus => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Plus,
                    expr: Box::new(expr),
                })
            }
            TokenKind::BitNot => {
                self.advance();
                let expr = self.parse_unary_expr()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::BitNot,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_primary_expr(),
        }
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        let expr = match self.peek().kind.clone() {
            // Literals
            TokenKind::Integer(n) => {
                self.advance();
                Expr::Literal(Literal::Integer(n))
            }
            TokenKind::Float(n) => {
                self.advance();
                Expr::Literal(Literal::Float(n))
            }
            TokenKind::String(s) => {
                self.advance();
                Expr::Literal(Literal::String(s))
            }
            TokenKind::Blob(b) => {
                self.advance();
                Expr::Literal(Literal::Blob(b))
            }
            TokenKind::True => {
                self.advance();
                Expr::Literal(Literal::Boolean(true))
            }
            TokenKind::False => {
                self.advance();
                Expr::Literal(Literal::Boolean(false))
            }
            TokenKind::Null => {
                self.advance();
                Expr::Literal(Literal::Null)
            }

            // Placeholder
            TokenKind::Placeholder(n) => {
                self.advance();
                Expr::Placeholder(n)
            }

            // Parenthesized expression or subquery
            TokenKind::LParen => {
                self.advance();
                if self.check_keyword(&TokenKind::Select) {
                    let query = self.parse_select()?;
                    self.expect(&TokenKind::RParen, "Expected ')'")?;
                    Expr::Subquery(Box::new(query))
                } else {
                    let expr = self.parse_expr()?;

                    // Check for tuple
                    if self.match_token(&TokenKind::Comma) {
                        let mut exprs = vec![expr];
                        exprs.push(self.parse_expr()?);
                        while self.match_token(&TokenKind::Comma) {
                            exprs.push(self.parse_expr()?);
                        }
                        self.expect(&TokenKind::RParen, "Expected ')'")?;
                        Expr::Tuple(exprs)
                    } else {
                        self.expect(&TokenKind::RParen, "Expected ')'")?;
                        expr
                    }
                }
            }

            // CASE expression
            TokenKind::Case => {
                self.advance();
                self.parse_case_expr()?
            }

            // EXISTS
            TokenKind::Exists => {
                self.advance();
                self.expect(&TokenKind::LParen, "Expected '(' after EXISTS")?;
                let query = self.parse_select()?;
                self.expect(&TokenKind::RParen, "Expected ')'")?;
                Expr::Exists(Box::new(query))
            }

            // CAST
            TokenKind::Cast => {
                self.advance();
                self.expect(&TokenKind::LParen, "Expected '(' after CAST")?;
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::As, "Expected AS in CAST")?;
                let data_type = self.parse_data_type()?;
                self.expect(&TokenKind::RParen, "Expected ')'")?;
                Expr::Cast {
                    expr: Box::new(expr),
                    data_type,
                }
            }

            // ToonDB Extensions
            TokenKind::VectorSearch => {
                self.advance();
                self.parse_vector_search()?
            }
            TokenKind::ContextWindow => {
                self.advance();
                self.parse_context_window()?
            }

            // Aggregate functions
            TokenKind::Count
            | TokenKind::Sum
            | TokenKind::Avg
            | TokenKind::Min
            | TokenKind::Max => self.parse_aggregate_function()?,

            // Function call or column reference
            TokenKind::Identifier(_) | TokenKind::QuotedIdentifier(_) => {
                self.parse_identifier_or_function()?
            }

            // Type keywords used as column names
            TokenKind::Vector | TokenKind::Embedding | TokenKind::Text | TokenKind::BlobKw => {
                // Convert keyword to identifier
                let name = match &self.peek().kind {
                    TokenKind::Vector => "vector".to_string(),
                    TokenKind::Embedding => "embedding".to_string(),
                    TokenKind::Text => "text".to_string(),
                    TokenKind::BlobKw => "blob".to_string(),
                    _ => unreachable!(),
                };
                self.advance();
                Expr::Column(ColumnRef::new(name))
            }

            _ => {
                return Err(ParseError::new(
                    format!("Unexpected token in expression: {:?}", self.peek().kind),
                    self.current_span(),
                ));
            }
        };

        // Handle postfix operators
        self.parse_postfix_expr(expr)
    }

    fn parse_postfix_expr(&mut self, mut expr: Expr) -> Result<Expr, ParseError> {
        loop {
            if self.match_token(&TokenKind::LBracket) {
                // Array subscript
                let index = self.parse_expr()?;
                self.expect(&TokenKind::RBracket, "Expected ']'")?;
                expr = Expr::Subscript {
                    expr: Box::new(expr),
                    index: Box::new(index),
                };
            } else if self.match_token(&TokenKind::Arrow) {
                // JSON access: ->
                let path = self.parse_primary_expr()?;
                expr = Expr::JsonAccess {
                    expr: Box::new(expr),
                    path: Box::new(path),
                    return_text: false,
                };
            } else if self.match_token(&TokenKind::DoubleArrow) {
                // JSON access returning text: ->>
                let path = self.parse_primary_expr()?;
                expr = Expr::JsonAccess {
                    expr: Box::new(expr),
                    path: Box::new(path),
                    return_text: true,
                };
            } else if self.match_token(&TokenKind::DoubleColon) {
                // Type cast: ::type
                let data_type = self.parse_data_type()?;
                expr = Expr::Cast {
                    expr: Box::new(expr),
                    data_type,
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_case_expr(&mut self) -> Result<Expr, ParseError> {
        // Simple CASE: CASE expr WHEN val THEN result ...
        // Searched CASE: CASE WHEN cond THEN result ...

        let operand = if !self.check_keyword(&TokenKind::When) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let mut conditions = Vec::new();

        while self.match_token(&TokenKind::When) {
            let when_expr = self.parse_expr()?;
            self.expect(&TokenKind::Then, "Expected THEN")?;
            let then_expr = self.parse_expr()?;
            conditions.push((when_expr, then_expr));
        }

        let else_result = if self.match_token(&TokenKind::Else) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        self.expect(&TokenKind::End, "Expected END")?;

        Ok(Expr::Case {
            operand,
            conditions,
            else_result,
        })
    }

    fn parse_identifier_or_function(&mut self) -> Result<Expr, ParseError> {
        let name = self.parse_object_name()?;

        // Check for function call
        if self.match_token(&TokenKind::LParen) {
            let args = if self.check(&TokenKind::RParen) {
                Vec::new()
            } else {
                self.parse_expr_list()?
            };
            self.expect(&TokenKind::RParen, "Expected ')'")?;

            Ok(Expr::Function(FunctionCall {
                name,
                args,
                distinct: false,
                filter: None,
                over: None,
            }))
        } else {
            // Column reference
            let parts = name.parts;
            if parts.len() == 1 {
                Ok(Expr::Column(ColumnRef::new(
                    parts.into_iter().next().unwrap(),
                )))
            } else if parts.len() == 2 {
                let mut iter = parts.into_iter();
                let table = iter.next().unwrap();
                let column = iter.next().unwrap();
                Ok(Expr::Column(ColumnRef::qualified(table, column)))
            } else {
                Err(ParseError::new(
                    "Invalid column reference",
                    self.current_span(),
                ))
            }
        }
    }

    fn parse_aggregate_function(&mut self) -> Result<Expr, ParseError> {
        let name = match &self.peek().kind {
            TokenKind::Count => "COUNT",
            TokenKind::Sum => "SUM",
            TokenKind::Avg => "AVG",
            TokenKind::Min => "MIN",
            TokenKind::Max => "MAX",
            _ => {
                return Err(ParseError::new(
                    "Expected aggregate function",
                    self.current_span(),
                ));
            }
        };
        self.advance();

        self.expect(&TokenKind::LParen, "Expected '(' after aggregate function")?;

        let distinct = self.match_token(&TokenKind::Distinct);

        let args = if self.match_token(&TokenKind::Star) {
            vec![Expr::Column(ColumnRef::new("*"))]
        } else {
            self.parse_expr_list()?
        };

        self.expect(&TokenKind::RParen, "Expected ')'")?;

        Ok(Expr::Function(FunctionCall {
            name: ObjectName::new(name),
            args,
            distinct,
            filter: None,
            over: None,
        }))
    }

    fn parse_vector_search(&mut self) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LParen, "Expected '(' after VECTOR_SEARCH")?;

        let column = self.parse_expr()?;
        self.expect(&TokenKind::Comma, "Expected ','")?;

        let query = self.parse_expr()?;
        self.expect(&TokenKind::Comma, "Expected ','")?;

        let k = match &self.peek().kind {
            TokenKind::Integer(n) => *n as u32,
            _ => return Err(ParseError::new("Expected integer k", self.current_span())),
        };
        self.advance();

        let metric = if self.match_token(&TokenKind::Comma) {
            match &self.peek().kind {
                TokenKind::Cosine => {
                    self.advance();
                    VectorMetric::Cosine
                }
                TokenKind::Euclidean => {
                    self.advance();
                    VectorMetric::Euclidean
                }
                TokenKind::DotProduct => {
                    self.advance();
                    VectorMetric::DotProduct
                }
                _ => VectorMetric::Cosine,
            }
        } else {
            VectorMetric::Cosine
        };

        self.expect(&TokenKind::RParen, "Expected ')'")?;

        Ok(Expr::VectorSearch {
            column: Box::new(column),
            query: Box::new(query),
            k,
            metric,
        })
    }

    fn parse_context_window(&mut self) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LParen, "Expected '(' after CONTEXT_WINDOW")?;

        let source = self.parse_expr()?;
        self.expect(&TokenKind::Comma, "Expected ','")?;

        let max_tokens = match &self.peek().kind {
            TokenKind::Integer(n) => *n as u32,
            _ => {
                return Err(ParseError::new(
                    "Expected integer max_tokens",
                    self.current_span(),
                ));
            }
        };
        self.advance();

        let priority = if self.match_token(&TokenKind::Comma) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        self.expect(&TokenKind::RParen, "Expected ')'")?;

        Ok(Expr::ContextWindow {
            source: Box::new(source),
            max_tokens,
            priority,
        })
    }

    // ========== Helper Parsers ==========

    fn parse_object_name(&mut self) -> Result<ObjectName, ParseError> {
        let mut parts = Vec::new();
        parts.push(self.expect_identifier("Expected identifier")?);

        while self.match_token(&TokenKind::Dot) {
            // Check for wildcard after dot (table.*)
            if self.check(&TokenKind::Star) {
                // Don't consume star, let caller handle it
                break;
            }
            parts.push(self.expect_identifier("Expected identifier after '.'")?);
        }

        Ok(ObjectName { parts })
    }

    fn parse_identifier_list(&mut self) -> Result<Vec<String>, ParseError> {
        let mut list = vec![self.expect_identifier("Expected identifier")?];

        while self.match_token(&TokenKind::Comma) {
            list.push(self.expect_identifier("Expected identifier")?);
        }

        Ok(list)
    }

    fn parse_expr_list(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut list = vec![self.parse_expr()?];

        while self.match_token(&TokenKind::Comma) {
            list.push(self.parse_expr()?);
        }

        Ok(list)
    }

    fn parse_order_by_list(&mut self) -> Result<Vec<OrderByItem>, ParseError> {
        let mut list = Vec::new();

        loop {
            let expr = self.parse_expr()?;

            let asc = if self.match_token(&TokenKind::Desc) {
                false
            } else {
                self.match_token(&TokenKind::Asc);
                true
            };

            let nulls_first = if self.match_token(&TokenKind::Nulls) {
                if self.match_token(&TokenKind::First) {
                    Some(true)
                } else if self.match_token(&TokenKind::Last) {
                    Some(false)
                } else {
                    return Err(ParseError::new(
                        "Expected FIRST or LAST after NULLS",
                        self.current_span(),
                    ));
                }
            } else {
                None
            };

            list.push(OrderByItem {
                expr,
                asc,
                nulls_first,
            });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let stmt = Parser::parse("SELECT * FROM users").unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_select_with_where() {
        let stmt = Parser::parse("SELECT id, name FROM users WHERE id = 1").unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 2);
            assert!(select.where_clause.is_some());
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_insert() {
        let stmt = Parser::parse("INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        assert!(matches!(stmt, Statement::Insert(_)));
    }

    #[test]
    fn test_create_table() {
        let stmt = Parser::parse(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL)",
        )
        .unwrap();
        if let Statement::CreateTable(create) = stmt {
            assert_eq!(create.columns.len(), 2);
        } else {
            panic!("Expected CREATE TABLE statement");
        }
    }

    #[test]
    fn test_vector_search() {
        let stmt = Parser::parse(
            "SELECT * FROM docs WHERE VECTOR_SEARCH(embedding, $1, 10, COSINE) > 0.8",
        )
        .unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_join() {
        let stmt = Parser::parse(
            "SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id",
        )
        .unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_subquery() {
        let stmt =
            Parser::parse("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)").unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_update() {
        let stmt = Parser::parse("UPDATE users SET name = 'Bob', age = 30 WHERE id = 1").unwrap();
        assert!(matches!(stmt, Statement::Update(_)));
    }

    #[test]
    fn test_delete() {
        let stmt = Parser::parse("DELETE FROM users WHERE id = 1").unwrap();
        assert!(matches!(stmt, Statement::Delete(_)));
    }

    #[test]
    fn test_group_by() {
        let stmt = Parser::parse(
            "SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 5",
        )
        .unwrap();
        if let Statement::Select(select) = stmt {
            assert!(!select.group_by.is_empty());
            assert!(select.having.is_some());
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_order_by() {
        let stmt =
            Parser::parse("SELECT * FROM users ORDER BY name ASC, age DESC NULLS LAST").unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.order_by.len(), 2);
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_between() {
        let stmt = Parser::parse("SELECT * FROM products WHERE price BETWEEN 10 AND 100").unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_like() {
        let stmt = Parser::parse("SELECT * FROM users WHERE name LIKE '%Alice%'").unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_case() {
        let stmt =
            Parser::parse("SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END FROM t")
                .unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_transactions() {
        let stmts = Parser::parse_statements("BEGIN; COMMIT; ROLLBACK").unwrap();
        assert_eq!(stmts.len(), 3);
        assert!(matches!(stmts[0], Statement::Begin(_)));
        assert!(matches!(stmts[1], Statement::Commit));
        assert!(matches!(stmts[2], Statement::Rollback(_)));
    }

    // ===== Dialect-Specific Insert Tests =====

    #[test]
    fn test_insert_on_conflict_do_nothing() {
        let stmt = Parser::parse(
            "INSERT INTO users (id, name) VALUES (1, 'Alice') ON CONFLICT DO NOTHING",
        )
        .unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
            let on_conflict = insert.on_conflict.unwrap();
            assert!(matches!(on_conflict.action, ConflictAction::DoNothing));
        } else {
            panic!("Expected INSERT statement");
        }
    }

    #[test]
    fn test_insert_on_conflict_do_update() {
        let stmt = Parser::parse(
            "INSERT INTO users (id, name) VALUES (1, 'Alice') ON CONFLICT (id) DO UPDATE SET name = 'Bob'",
        )
        .unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
            let on_conflict = insert.on_conflict.unwrap();
            assert!(matches!(on_conflict.target, Some(ConflictTarget::Columns(_))));
            assert!(matches!(on_conflict.action, ConflictAction::DoUpdate(_)));
        } else {
            panic!("Expected INSERT statement");
        }
    }

    #[test]
    fn test_insert_ignore_mysql() {
        let stmt = Parser::parse("INSERT IGNORE INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
            let on_conflict = insert.on_conflict.unwrap();
            assert!(matches!(on_conflict.action, ConflictAction::DoNothing));
        } else {
            panic!("Expected INSERT statement");
        }
    }

    #[test]
    fn test_insert_or_ignore_sqlite() {
        let stmt =
            Parser::parse("INSERT OR IGNORE INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
            let on_conflict = insert.on_conflict.unwrap();
            assert!(matches!(on_conflict.action, ConflictAction::DoNothing));
        } else {
            panic!("Expected INSERT statement");
        }
    }

    #[test]
    fn test_insert_or_replace_sqlite() {
        let stmt =
            Parser::parse("INSERT OR REPLACE INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
            let on_conflict = insert.on_conflict.unwrap();
            assert!(matches!(on_conflict.action, ConflictAction::DoReplace));
        } else {
            panic!("Expected INSERT statement");
        }
    }

    #[test]
    fn test_on_duplicate_key_update_mysql() {
        let stmt = Parser::parse(
            "INSERT INTO users (id, name) VALUES (1, 'Alice') ON DUPLICATE KEY UPDATE name = 'Bob'",
        )
        .unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.on_conflict.is_some());
            let on_conflict = insert.on_conflict.unwrap();
            assert!(matches!(on_conflict.action, ConflictAction::DoUpdate(_)));
        } else {
            panic!("Expected INSERT statement");
        }
    }

    // ===== Idempotent DDL Tests =====

    #[test]
    fn test_create_table_if_not_exists() {
        let stmt = Parser::parse("CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY)").unwrap();
        if let Statement::CreateTable(create) = stmt {
            assert!(create.if_not_exists);
        } else {
            panic!("Expected CREATE TABLE statement");
        }
    }

    #[test]
    fn test_drop_table_if_exists() {
        let stmt = Parser::parse("DROP TABLE IF EXISTS users").unwrap();
        if let Statement::DropTable(drop) = stmt {
            assert!(drop.if_exists);
        } else {
            panic!("Expected DROP TABLE statement");
        }
    }

    #[test]
    fn test_create_index() {
        let stmt = Parser::parse("CREATE INDEX idx_users_name ON users (name)").unwrap();
        if let Statement::CreateIndex(create) = stmt {
            assert_eq!(create.name, "idx_users_name");
            assert_eq!(create.table.name(), "users");
            assert!(!create.unique);
            assert!(!create.if_not_exists);
        } else {
            panic!("Expected CREATE INDEX statement");
        }
    }

    #[test]
    fn test_create_unique_index() {
        let stmt = Parser::parse("CREATE UNIQUE INDEX idx_users_email ON users (email)").unwrap();
        if let Statement::CreateIndex(create) = stmt {
            assert!(create.unique);
        } else {
            panic!("Expected CREATE INDEX statement");
        }
    }

    #[test]
    fn test_create_index_if_not_exists() {
        let stmt =
            Parser::parse("CREATE INDEX IF NOT EXISTS idx_users_name ON users (name)").unwrap();
        if let Statement::CreateIndex(create) = stmt {
            assert!(create.if_not_exists);
        } else {
            panic!("Expected CREATE INDEX statement");
        }
    }

    #[test]
    fn test_drop_index() {
        let stmt = Parser::parse("DROP INDEX idx_users_name").unwrap();
        if let Statement::DropIndex(drop) = stmt {
            assert_eq!(drop.name, "idx_users_name");
            assert!(!drop.if_exists);
        } else {
            panic!("Expected DROP INDEX statement");
        }
    }

    #[test]
    fn test_drop_index_if_exists() {
        let stmt = Parser::parse("DROP INDEX IF EXISTS idx_users_name").unwrap();
        if let Statement::DropIndex(drop) = stmt {
            assert!(drop.if_exists);
        } else {
            panic!("Expected DROP INDEX statement");
        }
    }

    // ===== RETURNING clause tests =====

    #[test]
    fn test_insert_returning() {
        let stmt = Parser::parse(
            "INSERT INTO users (id, name) VALUES (1, 'Alice') RETURNING id, name",
        )
        .unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(insert.returning.is_some());
            let returning = insert.returning.unwrap();
            assert_eq!(returning.len(), 2);
        } else {
            panic!("Expected INSERT statement");
        }
    }
}
