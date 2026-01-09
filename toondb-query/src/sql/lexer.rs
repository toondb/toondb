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

//! SQL Lexer
//!
//! Converts SQL text into a stream of tokens.
//! Handles string literals, numbers, identifiers, keywords, and operators.

use super::token::{Span, Token, TokenKind};
use std::iter::Peekable;
use std::str::Chars;

/// SQL Lexer errors
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl LexError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Lexer error at line {}, column {}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl std::error::Error for LexError {}

/// SQL Lexer - tokenizes SQL input
pub struct Lexer<'a> {
    input: &'a str,
    chars: Peekable<Chars<'a>>,
    pos: usize,
    line: usize,
    column: usize,
    tokens: Vec<Token>,
    errors: Vec<LexError>,
    /// Counter for `?` style placeholders (auto-incrementing)
    placeholder_counter: u32,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given SQL input
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().peekable(),
            pos: 0,
            line: 1,
            column: 1,
            tokens: Vec::new(),
            errors: Vec::new(),
            placeholder_counter: 0,
        }
    }

    /// Tokenize the entire input
    pub fn tokenize(mut self) -> Result<Vec<Token>, Vec<LexError>> {
        while !self.is_at_end() {
            self.scan_token();
        }

        // Add EOF token
        self.tokens.push(Token::new(
            TokenKind::Eof,
            Span::new(self.pos, self.pos, self.line, self.column),
            "",
        ));

        if self.errors.is_empty() {
            Ok(self.tokens)
        } else {
            Err(self.errors)
        }
    }

    fn is_at_end(&mut self) -> bool {
        self.chars.peek().is_none()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn peek_next(&self) -> Option<char> {
        let mut chars = self.chars.clone();
        chars.next();
        chars.next()
    }

    fn make_span(&self, start: usize, start_line: usize, start_col: usize) -> Span {
        Span::new(start, self.pos, start_line, start_col)
    }

    fn scan_token(&mut self) {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        let c = match self.advance() {
            Some(c) => c,
            None => return,
        };

        match c {
            // Whitespace
            ' ' | '\t' | '\r' | '\n' => {
                // Skip whitespace, don't emit token
            }

            // Single-character tokens
            '(' => self.add_token(TokenKind::LParen, start, start_line, start_col),
            ')' => self.add_token(TokenKind::RParen, start, start_line, start_col),
            '[' => self.add_token(TokenKind::LBracket, start, start_line, start_col),
            ']' => self.add_token(TokenKind::RBracket, start, start_line, start_col),
            ',' => self.add_token(TokenKind::Comma, start, start_line, start_col),
            ';' => self.add_token(TokenKind::Semicolon, start, start_line, start_col),
            '+' => self.add_token(TokenKind::Plus, start, start_line, start_col),
            '*' => self.add_token(TokenKind::Star, start, start_line, start_col),
            '/' => {
                if self.peek() == Some('/') || self.peek() == Some('*') {
                    self.scan_comment(start, start_line, start_col);
                } else {
                    self.add_token(TokenKind::Slash, start, start_line, start_col);
                }
            }
            '%' => self.add_token(TokenKind::Percent, start, start_line, start_col),
            '&' => self.add_token(TokenKind::BitAnd, start, start_line, start_col),
            '~' => self.add_token(TokenKind::BitNot, start, start_line, start_col),
            '?' => {
                // Auto-incrementing placeholder for JDBC/ODBC style ?
                self.placeholder_counter += 1;
                let span = self.make_span(start, start_line, start_col);
                self.tokens.push(Token::new(
                    TokenKind::Placeholder(self.placeholder_counter),
                    span,
                    "?",
                ));
            }
            '@' => self.add_token(TokenKind::At, start, start_line, start_col),

            // Two-character tokens
            '-' => {
                if self.peek() == Some('-') {
                    // Line comment
                    self.scan_line_comment(start, start_line, start_col);
                } else if self.peek() == Some('>') {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        self.add_token(TokenKind::DoubleArrow, start, start_line, start_col);
                    } else {
                        self.add_token(TokenKind::Arrow, start, start_line, start_col);
                    }
                } else {
                    self.add_token(TokenKind::Minus, start, start_line, start_col);
                }
            }

            '=' => self.add_token(TokenKind::Eq, start, start_line, start_col),

            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.add_token(TokenKind::Ne, start, start_line, start_col);
                } else {
                    self.add_error("Unexpected character '!'", start, start_line, start_col);
                }
            }

            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.add_token(TokenKind::Le, start, start_line, start_col);
                } else if self.peek() == Some('>') {
                    self.advance();
                    self.add_token(TokenKind::Ne, start, start_line, start_col);
                } else if self.peek() == Some('<') {
                    self.advance();
                    self.add_token(TokenKind::LeftShift, start, start_line, start_col);
                } else {
                    self.add_token(TokenKind::Lt, start, start_line, start_col);
                }
            }

            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.add_token(TokenKind::Ge, start, start_line, start_col);
                } else if self.peek() == Some('>') {
                    self.advance();
                    self.add_token(TokenKind::RightShift, start, start_line, start_col);
                } else {
                    self.add_token(TokenKind::Gt, start, start_line, start_col);
                }
            }

            '|' => {
                if self.peek() == Some('|') {
                    self.advance();
                    self.add_token(TokenKind::Concat, start, start_line, start_col);
                } else {
                    self.add_token(TokenKind::BitOr, start, start_line, start_col);
                }
            }

            ':' => {
                if self.peek() == Some(':') {
                    self.advance();
                    self.add_token(TokenKind::DoubleColon, start, start_line, start_col);
                } else {
                    self.add_token(TokenKind::Colon, start, start_line, start_col);
                }
            }

            '.' => {
                if self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    self.scan_number(start, start_line, start_col, true);
                } else {
                    self.add_token(TokenKind::Dot, start, start_line, start_col);
                }
            }

            // String literals
            '\'' => self.scan_string(start, start_line, start_col, '\''),
            '"' => self.scan_quoted_identifier(start, start_line, start_col, '"'),
            '`' => self.scan_quoted_identifier(start, start_line, start_col, '`'),

            // Blob literal (X'...')
            'X' | 'x' if self.peek() == Some('\'') => {
                self.advance();
                self.scan_blob(start, start_line, start_col);
            }

            // Numbers
            '0'..='9' => self.scan_number(start, start_line, start_col, false),

            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => self.scan_identifier(start, start_line, start_col),

            // Placeholder ($1, $2, ...)
            '$' => self.scan_placeholder(start, start_line, start_col),

            _ => {
                self.add_error(
                    format!("Unexpected character '{}'", c),
                    start,
                    start_line,
                    start_col,
                );
            }
        }
    }

    fn scan_string(&mut self, start: usize, start_line: usize, start_col: usize, quote: char) {
        let mut value = String::new();

        while let Some(c) = self.peek() {
            if c == quote {
                self.advance();
                // Check for escaped quote ('')
                if self.peek() == Some(quote) {
                    self.advance();
                    value.push(quote);
                } else {
                    // End of string
                    let span = self.make_span(start, start_line, start_col);
                    self.tokens
                        .push(Token::new(TokenKind::String(value), span, ""));
                    return;
                }
            } else if c == '\\' {
                self.advance();
                // Handle escape sequences
                if let Some(escaped) = self.advance() {
                    match escaped {
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        '\\' => value.push('\\'),
                        '\'' => value.push('\''),
                        '"' => value.push('"'),
                        '0' => value.push('\0'),
                        _ => {
                            value.push('\\');
                            value.push(escaped);
                        }
                    }
                }
            } else {
                self.advance();
                value.push(c);
            }
        }

        self.add_error("Unterminated string literal", start, start_line, start_col);
    }

    fn scan_quoted_identifier(
        &mut self,
        start: usize,
        start_line: usize,
        start_col: usize,
        quote: char,
    ) {
        let mut value = String::new();

        while let Some(c) = self.peek() {
            if c == quote {
                self.advance();
                // Check for escaped quote
                if self.peek() == Some(quote) {
                    self.advance();
                    value.push(quote);
                } else {
                    let span = self.make_span(start, start_line, start_col);
                    self.tokens
                        .push(Token::new(TokenKind::QuotedIdentifier(value), span, ""));
                    return;
                }
            } else {
                self.advance();
                value.push(c);
            }
        }

        self.add_error(
            "Unterminated quoted identifier",
            start,
            start_line,
            start_col,
        );
    }

    fn scan_number(
        &mut self,
        start: usize,
        start_line: usize,
        start_col: usize,
        started_with_dot: bool,
    ) {
        let num_start = start;
        let mut has_dot = started_with_dot;
        let mut has_exp = false;

        // Consume integer part
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else if c == '.' && !has_dot && !has_exp {
                // Check it's not a range operator (..)
                if self.peek_next() == Some('.') {
                    break;
                }
                has_dot = true;
                self.advance();
            } else if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                self.advance();
                // Optional sign
                if self.peek() == Some('+') || self.peek() == Some('-') {
                    self.advance();
                }
            } else {
                break;
            }
        }

        let literal = &self.input[num_start..self.pos];
        let span = self.make_span(start, start_line, start_col);

        if has_dot || has_exp {
            match literal.parse::<f64>() {
                Ok(n) => self
                    .tokens
                    .push(Token::new(TokenKind::Float(n), span, literal)),
                Err(_) => self.add_error("Invalid float literal", start, start_line, start_col),
            }
        } else {
            match literal.parse::<i64>() {
                Ok(n) => self
                    .tokens
                    .push(Token::new(TokenKind::Integer(n), span, literal)),
                Err(_) => self.add_error("Invalid integer literal", start, start_line, start_col),
            }
        }
    }

    fn scan_identifier(&mut self, start: usize, start_line: usize, start_col: usize) {
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let literal = &self.input[start..self.pos];
        let span = self.make_span(start, start_line, start_col);

        // Check for keyword
        let kind = TokenKind::from_keyword(literal)
            .unwrap_or_else(|| TokenKind::Identifier(literal.to_string()));

        self.tokens.push(Token::new(kind, span, literal));
    }

    fn scan_placeholder(&mut self, start: usize, start_line: usize, start_col: usize) {
        let mut num = String::new();

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
                num.push(c);
            } else {
                break;
            }
        }

        let span = self.make_span(start, start_line, start_col);

        if num.is_empty() {
            self.add_error("Expected number after $", start, start_line, start_col);
        } else if let Ok(n) = num.parse::<u32>() {
            self.tokens.push(Token::new(
                TokenKind::Placeholder(n),
                span,
                &self.input[start..self.pos],
            ));
        } else {
            self.add_error("Invalid placeholder number", start, start_line, start_col);
        }
    }

    fn scan_comment(&mut self, start: usize, start_line: usize, start_col: usize) {
        self.advance(); // consume second / or *

        if self.peek() == Some('*') || self.input[start..self.pos].ends_with('*') {
            // Block comment /* ... */
            let mut depth = 1;

            while depth > 0 && !self.is_at_end() {
                let c = self.peek();
                let next = self.peek_next();

                if c == Some('*') && next == Some('/') {
                    self.advance();
                    self.advance();
                    depth -= 1;
                } else if c == Some('/') && next == Some('*') {
                    self.advance();
                    self.advance();
                    depth += 1;
                } else {
                    self.advance();
                }
            }

            if depth > 0 {
                self.add_error("Unterminated block comment", start, start_line, start_col);
            }
        } else {
            // Line comment //
            while let Some(c) = self.peek() {
                if c == '\n' {
                    break;
                }
                self.advance();
            }
        }
        // Don't emit comment tokens
    }

    fn scan_line_comment(&mut self, _start: usize, _start_line: usize, _start_col: usize) {
        self.advance(); // consume second -

        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
        // Don't emit comment tokens
    }

    fn scan_blob(&mut self, start: usize, start_line: usize, start_col: usize) {
        let mut hex = String::new();

        while let Some(c) = self.peek() {
            if c == '\'' {
                self.advance();
                break;
            } else if c.is_ascii_hexdigit() {
                self.advance();
                hex.push(c);
            } else if c.is_whitespace() {
                self.advance(); // Allow whitespace in blob
            } else {
                self.add_error(
                    "Invalid hex digit in blob literal",
                    start,
                    start_line,
                    start_col,
                );
                return;
            }
        }

        if !hex.len().is_multiple_of(2) {
            self.add_error(
                "Blob literal must have even number of hex digits",
                start,
                start_line,
                start_col,
            );
            return;
        }

        let bytes: Result<Vec<u8>, _> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
            .collect();

        match bytes {
            Ok(data) => {
                let span = self.make_span(start, start_line, start_col);
                self.tokens
                    .push(Token::new(TokenKind::Blob(data), span, ""));
            }
            Err(_) => {
                self.add_error("Invalid blob literal", start, start_line, start_col);
            }
        }
    }

    fn add_token(&mut self, kind: TokenKind, start: usize, start_line: usize, start_col: usize) {
        let span = self.make_span(start, start_line, start_col);
        let literal = &self.input[start..self.pos];
        self.tokens.push(Token::new(kind, span, literal));
    }

    fn add_error(
        &mut self,
        message: impl Into<String>,
        start: usize,
        start_line: usize,
        start_col: usize,
    ) {
        let span = self.make_span(start, start_line, start_col);
        self.errors.push(LexError::new(message, span));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let tokens = Lexer::new("SELECT * FROM users").tokenize().unwrap();
        assert_eq!(tokens.len(), 5); // SELECT, *, FROM, users, EOF
        assert_eq!(tokens[0].kind, TokenKind::Select);
        assert_eq!(tokens[1].kind, TokenKind::Star);
        assert_eq!(tokens[2].kind, TokenKind::From);
        assert!(matches!(tokens[3].kind, TokenKind::Identifier(_)));
    }

    #[test]
    fn test_string_literal() {
        let tokens = Lexer::new("SELECT 'hello''world'").tokenize().unwrap();
        assert!(matches!(&tokens[1].kind, TokenKind::String(s) if s == "hello'world"));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_numbers() {
        let tokens = Lexer::new("42 3.14 1e10 .5").tokenize().unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Integer(42)));
        assert!(matches!(tokens[1].kind, TokenKind::Float(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[2].kind, TokenKind::Float(_)));
        assert!(matches!(tokens[3].kind, TokenKind::Float(f) if (f - 0.5).abs() < 0.001));
    }

    #[test]
    fn test_operators() {
        let tokens = Lexer::new("= != <> < <= > >= || ->").tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Eq);
        assert_eq!(tokens[1].kind, TokenKind::Ne);
        assert_eq!(tokens[2].kind, TokenKind::Ne);
        assert_eq!(tokens[3].kind, TokenKind::Lt);
        assert_eq!(tokens[4].kind, TokenKind::Le);
        assert_eq!(tokens[5].kind, TokenKind::Gt);
        assert_eq!(tokens[6].kind, TokenKind::Ge);
        assert_eq!(tokens[7].kind, TokenKind::Concat);
        assert_eq!(tokens[8].kind, TokenKind::Arrow);
    }

    #[test]
    fn test_keywords() {
        let tokens = Lexer::new("SELECT INSERT UPDATE DELETE FROM WHERE")
            .tokenize()
            .unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Select);
        assert_eq!(tokens[1].kind, TokenKind::Insert);
        assert_eq!(tokens[2].kind, TokenKind::Update);
        assert_eq!(tokens[3].kind, TokenKind::Delete);
        assert_eq!(tokens[4].kind, TokenKind::From);
        assert_eq!(tokens[5].kind, TokenKind::Where);
    }

    #[test]
    fn test_placeholder() {
        let tokens = Lexer::new("$1 $2 $10").tokenize().unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Placeholder(1)));
        assert!(matches!(tokens[1].kind, TokenKind::Placeholder(2)));
        assert!(matches!(tokens[2].kind, TokenKind::Placeholder(10)));
    }

    #[test]
    fn test_line_comment() {
        let tokens = Lexer::new("SELECT -- comment\n* FROM users")
            .tokenize()
            .unwrap();
        assert_eq!(tokens.len(), 5); // SELECT, *, FROM, users, EOF
        assert_eq!(tokens[0].kind, TokenKind::Select);
        assert_eq!(tokens[1].kind, TokenKind::Star);
    }

    #[test]
    fn test_blob_literal() {
        let tokens = Lexer::new("X'48454C4C4F'").tokenize().unwrap();
        assert!(matches!(&tokens[0].kind, TokenKind::Blob(b) if b == b"HELLO"));
    }
}
