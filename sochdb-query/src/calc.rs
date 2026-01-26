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

//! Calculator Expression Evaluator (Task 13)
//!
//! Safe mathematical expression evaluator for agentic use cases:
//! - Sandboxed evaluation (no code injection)
//! - Column references for computed fields
//! - Built-in math functions (abs, sqrt, pow, etc.)
//!
//! ## Grammar (Recursive Descent)
//!
//! ```text
//! expr     → term (('+' | '-') term)*
//! term     → factor (('*' | '/' | '%') factor)*
//! factor   → unary
//! unary    → '-'? primary
//! primary  → NUMBER | COLUMN | '(' expr ')' | function
//! function → IDENT '(' (expr (',' expr)*)? ')'
//! ```
//!
//! ## Security Model
//!
//! - No variable assignment (immutable)
//! - No loops (single-pass evaluation)
//! - No function definitions (allowlist only)
//! - Timeout: 1ms max for safety

use std::collections::HashMap;
use std::fmt;
use std::iter::Peekable;
use std::str::Chars;

/// Expression AST node
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal number
    Literal(f64),
    /// Column reference
    Column(String),
    /// Binary operation
    BinaryOp {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation
    UnaryOp { op: UnaryOp, expr: Box<Expr> },
    /// Function call
    FnCall { name: String, args: Vec<Expr> },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Pow => write!(f, "^"),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

/// Token types
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

/// Tokenizer for expressions
struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().peekable(),
        }
    }

    fn next_token(&mut self) -> Result<Token, CalcError> {
        self.skip_whitespace();

        match self.chars.peek() {
            None => Ok(Token::Eof),
            Some(&c) => match c {
                '+' => {
                    self.chars.next();
                    Ok(Token::Plus)
                }
                '-' => {
                    self.chars.next();
                    Ok(Token::Minus)
                }
                '*' => {
                    self.chars.next();
                    Ok(Token::Star)
                }
                '/' => {
                    self.chars.next();
                    Ok(Token::Slash)
                }
                '%' => {
                    self.chars.next();
                    Ok(Token::Percent)
                }
                '^' => {
                    self.chars.next();
                    Ok(Token::Caret)
                }
                '(' => {
                    self.chars.next();
                    Ok(Token::LParen)
                }
                ')' => {
                    self.chars.next();
                    Ok(Token::RParen)
                }
                ',' => {
                    self.chars.next();
                    Ok(Token::Comma)
                }
                '0'..='9' | '.' => self.number(),
                'a'..='z' | 'A'..='Z' | '_' | '$' => self.ident(),
                _ => Err(CalcError::UnexpectedChar(c)),
            },
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }

    fn number(&mut self) -> Result<Token, CalcError> {
        let mut s = String::new();
        let mut has_dot = false;

        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.chars.next();
            } else if c == '.' && !has_dot {
                has_dot = true;
                s.push(c);
                self.chars.next();
            } else if c == 'e' || c == 'E' {
                // Scientific notation
                s.push(c);
                self.chars.next();
                if let Some(&sign) = self.chars.peek()
                    && (sign == '+' || sign == '-')
                {
                    s.push(sign);
                    self.chars.next();
                }
            } else {
                break;
            }
        }

        s.parse::<f64>()
            .map(Token::Number)
            .map_err(|_| CalcError::InvalidNumber(s))
    }

    fn ident(&mut self) -> Result<Token, CalcError> {
        let mut s = String::new();

        while let Some(&c) = self.chars.peek() {
            if c.is_alphanumeric() || c == '_' || c == '$' {
                s.push(c);
                self.chars.next();
            } else {
                break;
            }
        }

        Ok(Token::Ident(s))
    }
}

/// Expression parser
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    /// Create a new parser
    pub fn new(input: &'a str) -> Result<Self, CalcError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    /// Parse the expression
    pub fn parse(&mut self) -> Result<Expr, CalcError> {
        let expr = self.expression()?;
        if self.current != Token::Eof {
            return Err(CalcError::UnexpectedToken(format!("{:?}", self.current)));
        }
        Ok(expr)
    }

    fn advance(&mut self) -> Result<(), CalcError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expression(&mut self) -> Result<Expr, CalcError> {
        self.additive()
    }

    fn additive(&mut self) -> Result<Expr, CalcError> {
        let mut left = self.multiplicative()?;

        loop {
            let op = match &self.current {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance()?;
            let right = self.multiplicative()?;
            left = Expr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn multiplicative(&mut self) -> Result<Expr, CalcError> {
        let mut left = self.power()?;

        loop {
            let op = match &self.current {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                Token::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.advance()?;
            let right = self.power()?;
            left = Expr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn power(&mut self) -> Result<Expr, CalcError> {
        let left = self.unary()?;

        if self.current == Token::Caret {
            self.advance()?;
            let right = self.power()?; // Right associative
            return Ok(Expr::BinaryOp {
                op: BinaryOp::Pow,
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, CalcError> {
        if self.current == Token::Minus {
            self.advance()?;
            let expr = self.unary()?;
            return Ok(Expr::UnaryOp {
                op: UnaryOp::Neg,
                expr: Box::new(expr),
            });
        }

        self.primary()
    }

    fn primary(&mut self) -> Result<Expr, CalcError> {
        match self.current.clone() {
            Token::Number(n) => {
                self.advance()?;
                Ok(Expr::Literal(n))
            }
            Token::Ident(name) => {
                self.advance()?;
                if self.current == Token::LParen {
                    // Function call
                    self.advance()?;
                    let args = self.arguments()?;
                    if self.current != Token::RParen {
                        return Err(CalcError::ExpectedToken(")".into()));
                    }
                    self.advance()?;
                    Ok(Expr::FnCall { name, args })
                } else {
                    // Column reference
                    Ok(Expr::Column(name))
                }
            }
            Token::LParen => {
                self.advance()?;
                let expr = self.expression()?;
                if self.current != Token::RParen {
                    return Err(CalcError::ExpectedToken(")".into()));
                }
                self.advance()?;
                Ok(expr)
            }
            _ => Err(CalcError::UnexpectedToken(format!("{:?}", self.current))),
        }
    }

    fn arguments(&mut self) -> Result<Vec<Expr>, CalcError> {
        let mut args = Vec::new();

        if self.current == Token::RParen {
            return Ok(args);
        }

        args.push(self.expression()?);

        while self.current == Token::Comma {
            self.advance()?;
            args.push(self.expression()?);
        }

        Ok(args)
    }
}

/// Calculator error types
#[derive(Debug, Clone, PartialEq)]
pub enum CalcError {
    UnexpectedChar(char),
    InvalidNumber(String),
    UnexpectedToken(String),
    ExpectedToken(String),
    UndefinedColumn(String),
    UndefinedFunction(String),
    DivisionByZero,
    InvalidArgCount {
        name: String,
        expected: usize,
        got: usize,
    },
    MathError(String),
    Timeout,
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcError::UnexpectedChar(c) => write!(f, "Unexpected character: {}", c),
            CalcError::InvalidNumber(s) => write!(f, "Invalid number: {}", s),
            CalcError::UnexpectedToken(s) => write!(f, "Unexpected token: {}", s),
            CalcError::ExpectedToken(s) => write!(f, "Expected token: {}", s),
            CalcError::UndefinedColumn(s) => write!(f, "Undefined column: {}", s),
            CalcError::UndefinedFunction(s) => write!(f, "Undefined function: {}", s),
            CalcError::DivisionByZero => write!(f, "Division by zero"),
            CalcError::InvalidArgCount {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Function {} expects {} args, got {}",
                    name, expected, got
                )
            }
            CalcError::MathError(s) => write!(f, "Math error: {}", s),
            CalcError::Timeout => write!(f, "Evaluation timeout"),
        }
    }
}

impl std::error::Error for CalcError {}

/// Row context for evaluation
pub type RowContext = HashMap<String, f64>;

/// Expression evaluator
pub struct Evaluator {
    /// Maximum evaluation steps (prevent infinite loops)
    max_steps: usize,
    /// Current step count
    steps: usize,
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            max_steps: 10000,
            steps: 0,
        }
    }

    /// Create with custom step limit
    pub fn with_max_steps(max_steps: usize) -> Self {
        Self {
            max_steps,
            steps: 0,
        }
    }

    /// Evaluate expression with row context
    pub fn eval(&mut self, expr: &Expr, ctx: &RowContext) -> Result<f64, CalcError> {
        self.steps += 1;
        if self.steps > self.max_steps {
            return Err(CalcError::Timeout);
        }

        match expr {
            Expr::Literal(n) => Ok(*n),

            Expr::Column(name) => ctx
                .get(name)
                .copied()
                .ok_or_else(|| CalcError::UndefinedColumn(name.clone())),

            Expr::BinaryOp { op, left, right } => {
                let l = self.eval(left, ctx)?;
                let r = self.eval(right, ctx)?;

                match op {
                    BinaryOp::Add => Ok(l + r),
                    BinaryOp::Sub => Ok(l - r),
                    BinaryOp::Mul => Ok(l * r),
                    BinaryOp::Div => {
                        if r == 0.0 {
                            Err(CalcError::DivisionByZero)
                        } else {
                            Ok(l / r)
                        }
                    }
                    BinaryOp::Mod => {
                        if r == 0.0 {
                            Err(CalcError::DivisionByZero)
                        } else {
                            Ok(l % r)
                        }
                    }
                    BinaryOp::Pow => Ok(l.powf(r)),
                }
            }

            Expr::UnaryOp { op, expr } => {
                let v = self.eval(expr, ctx)?;
                match op {
                    UnaryOp::Neg => Ok(-v),
                }
            }

            Expr::FnCall { name, args } => self.call_function(name, args, ctx),
        }
    }

    /// Call a built-in function
    fn call_function(
        &mut self,
        name: &str,
        args: &[Expr],
        ctx: &RowContext,
    ) -> Result<f64, CalcError> {
        let evaluated: Result<Vec<f64>, CalcError> =
            args.iter().map(|a| self.eval(a, ctx)).collect();
        let args = evaluated?;

        match name.to_lowercase().as_str() {
            // Single argument functions
            "abs" => {
                check_args(name, &args, 1)?;
                Ok(args[0].abs())
            }
            "sqrt" => {
                check_args(name, &args, 1)?;
                if args[0] < 0.0 {
                    Err(CalcError::MathError("sqrt of negative number".into()))
                } else {
                    Ok(args[0].sqrt())
                }
            }
            "floor" => {
                check_args(name, &args, 1)?;
                Ok(args[0].floor())
            }
            "ceil" => {
                check_args(name, &args, 1)?;
                Ok(args[0].ceil())
            }
            "round" => {
                if args.len() == 1 {
                    Ok(args[0].round())
                } else if args.len() == 2 {
                    let factor = 10f64.powi(args[1] as i32);
                    Ok((args[0] * factor).round() / factor)
                } else {
                    Err(CalcError::InvalidArgCount {
                        name: name.into(),
                        expected: 1,
                        got: args.len(),
                    })
                }
            }
            "sin" => {
                check_args(name, &args, 1)?;
                Ok(args[0].sin())
            }
            "cos" => {
                check_args(name, &args, 1)?;
                Ok(args[0].cos())
            }
            "tan" => {
                check_args(name, &args, 1)?;
                Ok(args[0].tan())
            }
            "exp" => {
                check_args(name, &args, 1)?;
                Ok(args[0].exp())
            }
            "ln" | "log" => {
                check_args(name, &args, 1)?;
                if args[0] <= 0.0 {
                    Err(CalcError::MathError("log of non-positive number".into()))
                } else {
                    Ok(args[0].ln())
                }
            }
            "log10" => {
                check_args(name, &args, 1)?;
                if args[0] <= 0.0 {
                    Err(CalcError::MathError("log of non-positive number".into()))
                } else {
                    Ok(args[0].log10())
                }
            }
            "log2" => {
                check_args(name, &args, 1)?;
                if args[0] <= 0.0 {
                    Err(CalcError::MathError("log of non-positive number".into()))
                } else {
                    Ok(args[0].log2())
                }
            }

            // Two argument functions
            "pow" => {
                check_args(name, &args, 2)?;
                Ok(args[0].powf(args[1]))
            }
            "min" => {
                check_args(name, &args, 2)?;
                Ok(args[0].min(args[1]))
            }
            "max" => {
                check_args(name, &args, 2)?;
                Ok(args[0].max(args[1]))
            }
            "atan2" => {
                check_args(name, &args, 2)?;
                Ok(args[0].atan2(args[1]))
            }

            // Variadic functions
            "sum" => Ok(args.iter().sum()),
            "avg" => {
                if args.is_empty() {
                    Err(CalcError::InvalidArgCount {
                        name: name.into(),
                        expected: 1,
                        got: 0,
                    })
                } else {
                    Ok(args.iter().sum::<f64>() / args.len() as f64)
                }
            }

            // Conditional
            "if" => {
                check_args(name, &args, 3)?;
                if args[0] != 0.0 {
                    Ok(args[1])
                } else {
                    Ok(args[2])
                }
            }

            _ => Err(CalcError::UndefinedFunction(name.into())),
        }
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

fn check_args(name: &str, args: &[f64], expected: usize) -> Result<(), CalcError> {
    if args.len() != expected {
        Err(CalcError::InvalidArgCount {
            name: name.into(),
            expected,
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

/// Parse and evaluate an expression in one step
pub fn calculate(expr: &str, ctx: &RowContext) -> Result<f64, CalcError> {
    let mut parser = Parser::new(expr)?;
    let ast = parser.parse()?;
    let mut evaluator = Evaluator::new();
    evaluator.eval(&ast, ctx)
}

/// Parse an expression without evaluating
pub fn parse_expr(expr: &str) -> Result<Expr, CalcError> {
    let mut parser = Parser::new(expr)?;
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let ctx = RowContext::new();

        assert_eq!(calculate("2 + 3", &ctx).unwrap(), 5.0);
        assert_eq!(calculate("10 - 4", &ctx).unwrap(), 6.0);
        assert_eq!(calculate("3 * 4", &ctx).unwrap(), 12.0);
        assert_eq!(calculate("15 / 3", &ctx).unwrap(), 5.0);
        assert_eq!(calculate("7 % 4", &ctx).unwrap(), 3.0);
        assert_eq!(calculate("2 ^ 3", &ctx).unwrap(), 8.0);
    }

    #[test]
    fn test_operator_precedence() {
        let ctx = RowContext::new();

        assert_eq!(calculate("2 + 3 * 4", &ctx).unwrap(), 14.0);
        assert_eq!(calculate("(2 + 3) * 4", &ctx).unwrap(), 20.0);
        assert_eq!(calculate("2 * 3 + 4", &ctx).unwrap(), 10.0);
        assert_eq!(calculate("10 - 2 * 3", &ctx).unwrap(), 4.0);
    }

    #[test]
    fn test_unary_minus() {
        let ctx = RowContext::new();

        assert_eq!(calculate("-5", &ctx).unwrap(), -5.0);
        assert_eq!(calculate("--5", &ctx).unwrap(), 5.0);
        assert_eq!(calculate("3 + -2", &ctx).unwrap(), 1.0);
        assert_eq!(calculate("-3 * -2", &ctx).unwrap(), 6.0);
    }

    #[test]
    fn test_column_references() {
        let mut ctx = RowContext::new();
        ctx.insert("price".into(), 99.99);
        ctx.insert("quantity".into(), 5.0);
        ctx.insert("tax_rate".into(), 0.15);

        assert_eq!(calculate("price * quantity", &ctx).unwrap(), 499.95);
        assert_eq!(
            calculate("price * quantity * (1 + tax_rate)", &ctx).unwrap(),
            574.9425
        );
    }

    #[test]
    fn test_functions() {
        let ctx = RowContext::new();

        assert_eq!(calculate("abs(-5)", &ctx).unwrap(), 5.0);
        assert_eq!(calculate("sqrt(16)", &ctx).unwrap(), 4.0);
        assert_eq!(calculate("floor(3.7)", &ctx).unwrap(), 3.0);
        assert_eq!(calculate("ceil(3.2)", &ctx).unwrap(), 4.0);
        assert_eq!(calculate("round(3.5)", &ctx).unwrap(), 4.0);
        #[allow(clippy::approx_constant)]
        {
            assert_eq!(calculate("round(3.14159, 2)", &ctx).unwrap(), 3.14);
        }
        assert_eq!(calculate("min(3, 5)", &ctx).unwrap(), 3.0);
        assert_eq!(calculate("max(3, 5)", &ctx).unwrap(), 5.0);
        assert_eq!(calculate("pow(2, 10)", &ctx).unwrap(), 1024.0);
    }

    #[test]
    fn test_trig_functions() {
        let ctx = RowContext::new();
        let _pi = std::f64::consts::PI;

        assert!((calculate("sin(0)", &ctx).unwrap() - 0.0).abs() < 1e-10);
        assert!((calculate("cos(0)", &ctx).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conditional() {
        let mut ctx = RowContext::new();
        ctx.insert("score".into(), 85.0);

        // NOTE: Comparison operators not yet implemented in lexer
        // Using computed boolean (non-zero = true, zero = false)
        // if(score > 70, 1, 0) would work once we add comparison operators

        // For now, test with explicit boolean values
        assert_eq!(calculate("if(1, 10, 20)", &ctx).unwrap(), 10.0);
        assert_eq!(calculate("if(0, 10, 20)", &ctx).unwrap(), 20.0);

        // Can use score directly as condition (85 != 0 means true)
        assert_eq!(calculate("if(score, 1, 0)", &ctx).unwrap(), 1.0);
    }

    #[test]
    fn test_variadic_functions() {
        let ctx = RowContext::new();

        assert_eq!(calculate("sum(1, 2, 3, 4)", &ctx).unwrap(), 10.0);
        assert_eq!(calculate("avg(2, 4, 6)", &ctx).unwrap(), 4.0);
    }

    #[test]
    fn test_scientific_notation() {
        let ctx = RowContext::new();

        assert_eq!(calculate("1e3", &ctx).unwrap(), 1000.0);
        assert_eq!(calculate("1.5e-2", &ctx).unwrap(), 0.015);
    }

    #[test]
    fn test_division_by_zero() {
        let ctx = RowContext::new();

        assert!(matches!(
            calculate("1 / 0", &ctx),
            Err(CalcError::DivisionByZero)
        ));
        assert!(matches!(
            calculate("5 % 0", &ctx),
            Err(CalcError::DivisionByZero)
        ));
    }

    #[test]
    fn test_undefined_column() {
        let ctx = RowContext::new();

        assert!(matches!(
            calculate("undefined_col + 1", &ctx),
            Err(CalcError::UndefinedColumn(_))
        ));
    }

    #[test]
    fn test_undefined_function() {
        let ctx = RowContext::new();

        assert!(matches!(
            calculate("unknown_func(1)", &ctx),
            Err(CalcError::UndefinedFunction(_))
        ));
    }

    #[test]
    fn test_complex_expression() {
        let mut ctx = RowContext::new();
        ctx.insert("revenue".into(), 1000.0);
        ctx.insert("cost".into(), 600.0);
        ctx.insert("tax".into(), 0.15);

        // Calculate after-tax profit
        let result = calculate("(revenue - cost) * (1 - tax)", &ctx).unwrap();
        assert!((result - 340.0).abs() < 1e-10);
    }
}
