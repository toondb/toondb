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

//! SQL Module - Complete SQL support for SochDB
//!
//! Provides SQL parsing, planning, and execution.
//!
//! # Example
//!
//! ```rust,ignore
//! use sochdb_query::sql::{Parser, Statement};
//!
//! let stmt = Parser::parse("SELECT * FROM users WHERE id = 1")?;
//! ```

pub mod ast;
pub mod bridge;
pub mod compatibility;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod token;

pub use ast::*;
pub use bridge::{ExecutionResult as BridgeExecutionResult, SqlBridge, SqlConnection};
pub use compatibility::{CompatibilityMatrix, FeatureSupport, SqlDialect, SqlFeature, get_feature_support};
pub use error::{SqlError, SqlResult};
pub use lexer::{LexError, Lexer};
pub use parser::{ParseError, Parser};
pub use token::{Span, Token, TokenKind};

use std::collections::HashMap;
use sochdb_core::SochValue;

/// Result of SQL execution
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// Query returned rows
    Rows {
        columns: Vec<String>,
        rows: Vec<HashMap<String, SochValue>>,
    },
    /// Statement affected N rows
    RowsAffected(usize),
    /// Statement completed successfully
    Ok,
}

impl ExecutionResult {
    /// Get rows if this is a query result
    pub fn rows(&self) -> Option<&Vec<HashMap<String, SochValue>>> {
        match self {
            ExecutionResult::Rows { rows, .. } => Some(rows),
            _ => None,
        }
    }

    /// Get column names if this is a query result
    pub fn columns(&self) -> Option<&Vec<String>> {
        match self {
            ExecutionResult::Rows { columns, .. } => Some(columns),
            _ => None,
        }
    }

    /// Get affected row count
    pub fn rows_affected(&self) -> usize {
        match self {
            ExecutionResult::RowsAffected(n) => *n,
            ExecutionResult::Rows { rows, .. } => rows.len(),
            ExecutionResult::Ok => 0,
        }
    }
}

/// Simple SQL executor for standalone use
///
/// For full database integration, use the SqlConnection in sochdb-storage
pub struct SqlExecutor {
    /// Tables stored in memory (for testing/standalone use)
    tables: HashMap<String, TableData>,
}

/// In-memory table data
#[derive(Debug, Clone)]
pub struct TableData {
    pub columns: Vec<String>,
    pub column_types: Vec<DataType>,
    pub rows: Vec<Vec<SochValue>>,
}

impl Default for SqlExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SqlExecutor {
    /// Create a new SQL executor
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    /// Execute a SQL statement
    pub fn execute(&mut self, sql: &str) -> SqlResult<ExecutionResult> {
        self.execute_with_params(sql, &[])
    }

    /// Execute a SQL statement with parameters
    pub fn execute_with_params(
        &mut self,
        sql: &str,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let stmt = Parser::parse(sql).map_err(SqlError::from_parse_errors)?;
        self.execute_statement(&stmt, params)
    }

    /// Execute a parsed statement
    pub fn execute_statement(
        &mut self,
        stmt: &Statement,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        match stmt {
            Statement::Select(select) => self.execute_select(select, params),
            Statement::Insert(insert) => self.execute_insert(insert, params),
            Statement::Update(update) => self.execute_update(update, params),
            Statement::Delete(delete) => self.execute_delete(delete, params),
            Statement::CreateTable(create) => self.execute_create_table(create),
            Statement::DropTable(drop) => self.execute_drop_table(drop),
            Statement::Begin(_) => Ok(ExecutionResult::Ok),
            Statement::Commit => Ok(ExecutionResult::Ok),
            Statement::Rollback(_) => Ok(ExecutionResult::Ok),
            _ => Err(SqlError::NotImplemented(
                "Statement type not yet supported".into(),
            )),
        }
    }

    fn execute_select(
        &self,
        select: &SelectStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        // Get the table
        let from = select
            .from
            .as_ref()
            .ok_or_else(|| SqlError::InvalidArgument("SELECT requires FROM clause".into()))?;

        if from.tables.len() != 1 {
            return Err(SqlError::NotImplemented(
                "Multi-table queries not yet supported".into(),
            ));
        }

        let table_name = match &from.tables[0] {
            TableRef::Table { name, .. } => name.name().to_string(),
            _ => {
                return Err(SqlError::NotImplemented(
                    "Complex table references not yet supported".into(),
                ));
            }
        };

        let table = self
            .tables
            .get(&table_name)
            .ok_or_else(|| SqlError::TableNotFound(table_name.clone()))?;

        // Collect matching source rows
        let mut source_rows = Vec::new();

        for row in &table.rows {
            // Build row as HashMap
            let row_map: HashMap<String, SochValue> = table
                .columns
                .iter()
                .zip(row.iter())
                .map(|(col, val)| (col.clone(), val.clone()))
                .collect();

            // Apply WHERE filter
            if let Some(where_clause) = &select.where_clause
                && !self.evaluate_where(where_clause, &row_map, params)?
            {
                continue;
            }

            source_rows.push(row_map);
        }

        // Apply ORDER BY
        if !select.order_by.is_empty() {
            source_rows.sort_by(|a, b| {
                for order_item in &select.order_by {
                    if let Expr::Column(col_ref) = &order_item.expr {
                        let a_val = a.get(&col_ref.column);
                        let b_val = b.get(&col_ref.column);

                        let cmp = self.compare_values(a_val, b_val);
                        if cmp != std::cmp::Ordering::Equal {
                            return if order_item.asc { cmp } else { cmp.reverse() };
                        }
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // Apply OFFSET
        if let Some(Expr::Literal(Literal::Integer(n))) = &select.offset {
            let n = *n as usize;
            if n < source_rows.len() {
                source_rows = source_rows.into_iter().skip(n).collect();
            } else {
                source_rows.clear();
            }
        }

        // Apply LIMIT
        if let Some(Expr::Literal(Literal::Integer(n))) = &select.limit {
            source_rows.truncate(*n as usize);
        }

        // Determine output columns and evaluate SELECT expressions
        let mut output_columns: Vec<String> = Vec::new();
        let mut result_rows: Vec<HashMap<String, SochValue>> = Vec::new();

        // Check for SELECT *
        let is_wildcard = matches!(&select.columns[..], [SelectItem::Wildcard]);

        if is_wildcard {
            output_columns = table.columns.clone();
            result_rows = source_rows;
        } else {
            // Determine column names first
            for item in &select.columns {
                match item {
                    SelectItem::Wildcard => output_columns.push("*".to_string()),
                    SelectItem::QualifiedWildcard(t) => output_columns.push(format!("{}.*", t)),
                    SelectItem::Expr { expr, alias } => {
                        let col_name = alias.clone().unwrap_or_else(|| match expr {
                            Expr::Column(col) => col.column.clone(),
                            Expr::Function(func) => format!("{}()", func.name.name()),
                            _ => "?column?".to_string(),
                        });
                        output_columns.push(col_name);
                    }
                }
            }

            // Now evaluate expressions for each row
            for source_row in &source_rows {
                let mut result_row = HashMap::new();

                for (idx, item) in select.columns.iter().enumerate() {
                    let col_name = &output_columns[idx];

                    match item {
                        SelectItem::Wildcard => {
                            // Add all columns from source row
                            for (k, v) in source_row {
                                result_row.insert(k.clone(), v.clone());
                            }
                        }
                        SelectItem::QualifiedWildcard(_) => {
                            // For now, treat same as wildcard
                            for (k, v) in source_row {
                                result_row.insert(k.clone(), v.clone());
                            }
                        }
                        SelectItem::Expr { expr, .. } => {
                            let value = self.evaluate_expr(expr, source_row, params)?;
                            result_row.insert(col_name.clone(), value);
                        }
                    }
                }

                result_rows.push(result_row);
            }
        }

        Ok(ExecutionResult::Rows {
            columns: output_columns,
            rows: result_rows,
        })
    }

    fn execute_insert(
        &mut self,
        insert: &InsertStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = insert.table.name().to_string();

        // First check if table exists and get column info
        let table_columns = {
            let table = self
                .tables
                .get(&table_name)
                .ok_or_else(|| SqlError::TableNotFound(table_name.clone()))?;
            table.columns.clone()
        };

        let mut rows_affected = 0;
        let mut new_rows = Vec::new();

        match &insert.source {
            InsertSource::Values(rows) => {
                for value_exprs in rows {
                    let mut row_values = Vec::new();

                    if let Some(columns) = &insert.columns {
                        if columns.len() != value_exprs.len() {
                            return Err(SqlError::InvalidArgument(format!(
                                "Column count ({}) doesn't match value count ({})",
                                columns.len(),
                                value_exprs.len()
                            )));
                        }

                        // Map columns to table column order
                        for table_col in &table_columns {
                            if let Some(pos) = columns.iter().position(|c| c == table_col) {
                                let value =
                                    self.evaluate_expr(&value_exprs[pos], &HashMap::new(), params)?;
                                row_values.push(value);
                            } else {
                                row_values.push(SochValue::Null);
                            }
                        }
                    } else {
                        // Insert in column order
                        for expr in value_exprs {
                            let value = self.evaluate_expr(expr, &HashMap::new(), params)?;
                            row_values.push(value);
                        }
                    }

                    new_rows.push(row_values);
                    rows_affected += 1;
                }
            }
            InsertSource::Query(_) => {
                return Err(SqlError::NotImplemented(
                    "INSERT ... SELECT not yet supported".into(),
                ));
            }
            InsertSource::Default => {
                return Err(SqlError::NotImplemented(
                    "INSERT DEFAULT VALUES not yet supported".into(),
                ));
            }
        }

        // Now add the rows to the table
        let table = self.tables.get_mut(&table_name).unwrap();
        for row in new_rows {
            table.rows.push(row);
        }

        Ok(ExecutionResult::RowsAffected(rows_affected))
    }

    fn execute_update(
        &mut self,
        update: &UpdateStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = update.table.name().to_string();

        // First, collect table info and rows that need updating
        let (_table_columns, updates_to_apply) = {
            let table = self
                .tables
                .get(&table_name)
                .ok_or_else(|| SqlError::TableNotFound(table_name.clone()))?;

            let mut updates = Vec::new();

            for row_idx in 0..table.rows.len() {
                // Build row as HashMap for WHERE evaluation
                let row_map: HashMap<String, SochValue> = table
                    .columns
                    .iter()
                    .zip(table.rows[row_idx].iter())
                    .map(|(col, val)| (col.clone(), val.clone()))
                    .collect();

                // Check WHERE condition
                let matches = if let Some(where_clause) = &update.where_clause {
                    self.evaluate_where(where_clause, &row_map, params)?
                } else {
                    true
                };

                if matches {
                    // Collect the updates for this row
                    let mut row_updates = Vec::new();
                    for assignment in &update.assignments {
                        if let Some(col_idx) =
                            table.columns.iter().position(|c| c == &assignment.column)
                        {
                            let value = self.evaluate_expr(&assignment.value, &row_map, params)?;
                            row_updates.push((col_idx, value));
                        }
                    }
                    updates.push((row_idx, row_updates));
                }
            }

            (table.columns.clone(), updates)
        };

        let rows_affected = updates_to_apply.len();

        // Now apply the updates
        let table = self.tables.get_mut(&table_name).unwrap();
        for (row_idx, row_updates) in updates_to_apply {
            for (col_idx, value) in row_updates {
                table.rows[row_idx][col_idx] = value;
            }
        }

        Ok(ExecutionResult::RowsAffected(rows_affected))
    }

    fn execute_delete(
        &mut self,
        delete: &DeleteStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = delete.table.name().to_string();

        // First, determine which rows to delete
        let indices_to_remove = {
            let table = self
                .tables
                .get(&table_name)
                .ok_or_else(|| SqlError::TableNotFound(table_name.clone()))?;

            let mut indices = Vec::new();

            for (row_idx, row) in table.rows.iter().enumerate() {
                // Build row as HashMap for WHERE evaluation
                let row_map: HashMap<String, SochValue> = table
                    .columns
                    .iter()
                    .zip(row.iter())
                    .map(|(col, val)| (col.clone(), val.clone()))
                    .collect();

                // Check WHERE condition
                let matches = if let Some(where_clause) = &delete.where_clause {
                    self.evaluate_where(where_clause, &row_map, params)?
                } else {
                    true
                };

                if matches {
                    indices.push(row_idx);
                }
            }

            indices
        };

        let rows_affected = indices_to_remove.len();

        // Now remove the rows
        let table = self.tables.get_mut(&table_name).unwrap();
        // Remove in reverse order to preserve indices
        for idx in indices_to_remove.into_iter().rev() {
            table.rows.remove(idx);
        }

        Ok(ExecutionResult::RowsAffected(rows_affected))
    }

    fn execute_create_table(&mut self, create: &CreateTableStmt) -> SqlResult<ExecutionResult> {
        let table_name = create.name.name().to_string();

        if self.tables.contains_key(&table_name) {
            if create.if_not_exists {
                return Ok(ExecutionResult::Ok);
            }
            return Err(SqlError::ConstraintViolation(format!(
                "Table '{}' already exists",
                table_name
            )));
        }

        let columns: Vec<String> = create.columns.iter().map(|c| c.name.clone()).collect();
        let column_types: Vec<DataType> =
            create.columns.iter().map(|c| c.data_type.clone()).collect();

        self.tables.insert(
            table_name,
            TableData {
                columns,
                column_types,
                rows: Vec::new(),
            },
        );

        Ok(ExecutionResult::Ok)
    }

    fn execute_drop_table(&mut self, drop: &DropTableStmt) -> SqlResult<ExecutionResult> {
        for name in &drop.names {
            let table_name = name.name().to_string();
            if self.tables.remove(&table_name).is_none() && !drop.if_exists {
                return Err(SqlError::TableNotFound(table_name));
            }
        }

        Ok(ExecutionResult::Ok)
    }

    // ========== Expression Evaluation ==========

    fn evaluate_where(
        &self,
        expr: &Expr,
        row: &HashMap<String, SochValue>,
        params: &[SochValue],
    ) -> SqlResult<bool> {
        let value = self.evaluate_expr(expr, row, params)?;
        match value {
            SochValue::Bool(b) => Ok(b),
            SochValue::Null => Ok(false),
            _ => Err(SqlError::TypeError(
                "WHERE clause must evaluate to boolean".into(),
            )),
        }
    }

    fn evaluate_expr(
        &self,
        expr: &Expr,
        row: &HashMap<String, SochValue>,
        params: &[SochValue],
    ) -> SqlResult<SochValue> {
        match expr {
            Expr::Literal(lit) => Ok(self.literal_to_value(lit)),

            Expr::Column(col_ref) => row
                .get(&col_ref.column)
                .cloned()
                .ok_or_else(|| SqlError::ColumnNotFound(col_ref.column.clone())),

            Expr::Placeholder(n) => params
                .get((*n as usize).saturating_sub(1))
                .cloned()
                .ok_or_else(|| SqlError::InvalidArgument(format!("Parameter ${} not provided", n))),

            Expr::BinaryOp { left, op, right } => {
                let left_val = self.evaluate_expr(left, row, params)?;
                let right_val = self.evaluate_expr(right, row, params)?;
                self.evaluate_binary_op(&left_val, op, &right_val)
            }

            Expr::UnaryOp { op, expr } => {
                let val = self.evaluate_expr(expr, row, params)?;
                self.evaluate_unary_op(op, &val)
            }

            Expr::IsNull { expr, negated } => {
                let val = self.evaluate_expr(expr, row, params)?;
                let is_null = matches!(val, SochValue::Null);
                Ok(SochValue::Bool(if *negated { !is_null } else { is_null }))
            }

            Expr::InList {
                expr,
                list,
                negated,
            } => {
                let val = self.evaluate_expr(expr, row, params)?;
                let mut found = false;
                for item in list {
                    let item_val = self.evaluate_expr(item, row, params)?;
                    if self.values_equal(&val, &item_val) {
                        found = true;
                        break;
                    }
                }
                Ok(SochValue::Bool(if *negated { !found } else { found }))
            }

            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let val = self.evaluate_expr(expr, row, params)?;
                let low_val = self.evaluate_expr(low, row, params)?;
                let high_val = self.evaluate_expr(high, row, params)?;

                let cmp_low = self.compare_values(Some(&val), Some(&low_val));
                let cmp_high = self.compare_values(Some(&val), Some(&high_val));

                let in_range =
                    cmp_low != std::cmp::Ordering::Less && cmp_high != std::cmp::Ordering::Greater;

                Ok(SochValue::Bool(if *negated { !in_range } else { in_range }))
            }

            Expr::Like {
                expr,
                pattern,
                negated,
                ..
            } => {
                let val = self.evaluate_expr(expr, row, params)?;
                let pattern_val = self.evaluate_expr(pattern, row, params)?;

                match (&val, &pattern_val) {
                    (SochValue::Text(s), SochValue::Text(p)) => {
                        let regex_pattern = p.replace('%', ".*").replace('_', ".");
                        let matches = regex::Regex::new(&format!("^{}$", regex_pattern))
                            .map(|re| re.is_match(s))
                            .unwrap_or(false);
                        Ok(SochValue::Bool(if *negated { !matches } else { matches }))
                    }
                    _ => Ok(SochValue::Bool(false)),
                }
            }

            Expr::Function(func) => self.evaluate_function(func, row, params),

            Expr::Case {
                operand,
                conditions,
                else_result,
            } => {
                if let Some(op) = operand {
                    // Simple CASE
                    let op_val = self.evaluate_expr(op, row, params)?;
                    for (when_expr, then_expr) in conditions {
                        let when_val = self.evaluate_expr(when_expr, row, params)?;
                        if self.values_equal(&op_val, &when_val) {
                            return self.evaluate_expr(then_expr, row, params);
                        }
                    }
                } else {
                    // Searched CASE
                    for (when_expr, then_expr) in conditions {
                        let when_val = self.evaluate_expr(when_expr, row, params)?;
                        if matches!(when_val, SochValue::Bool(true)) {
                            return self.evaluate_expr(then_expr, row, params);
                        }
                    }
                }

                if let Some(else_expr) = else_result {
                    self.evaluate_expr(else_expr, row, params)
                } else {
                    Ok(SochValue::Null)
                }
            }

            _ => Err(SqlError::NotImplemented(format!(
                "Expression type {:?} not yet supported",
                expr
            ))),
        }
    }

    fn literal_to_value(&self, lit: &Literal) -> SochValue {
        match lit {
            Literal::Null => SochValue::Null,
            Literal::Boolean(b) => SochValue::Bool(*b),
            Literal::Integer(n) => SochValue::Int(*n),
            Literal::Float(f) => SochValue::Float(*f),
            Literal::String(s) => SochValue::Text(s.clone()),
            Literal::Blob(b) => SochValue::Binary(b.clone()),
        }
    }

    fn evaluate_binary_op(
        &self,
        left: &SochValue,
        op: &BinaryOperator,
        right: &SochValue,
    ) -> SqlResult<SochValue> {
        match op {
            BinaryOperator::Eq => Ok(SochValue::Bool(self.values_equal(left, right))),
            BinaryOperator::Ne => Ok(SochValue::Bool(!self.values_equal(left, right))),
            BinaryOperator::Lt => Ok(SochValue::Bool(
                self.compare_values(Some(left), Some(right)) == std::cmp::Ordering::Less,
            )),
            BinaryOperator::Le => Ok(SochValue::Bool(
                self.compare_values(Some(left), Some(right)) != std::cmp::Ordering::Greater,
            )),
            BinaryOperator::Gt => Ok(SochValue::Bool(
                self.compare_values(Some(left), Some(right)) == std::cmp::Ordering::Greater,
            )),
            BinaryOperator::Ge => Ok(SochValue::Bool(
                self.compare_values(Some(left), Some(right)) != std::cmp::Ordering::Less,
            )),

            BinaryOperator::And => match (left, right) {
                (SochValue::Bool(l), SochValue::Bool(r)) => Ok(SochValue::Bool(*l && *r)),
                (SochValue::Null, _) | (_, SochValue::Null) => Ok(SochValue::Null),
                _ => Err(SqlError::TypeError("AND requires boolean operands".into())),
            },

            BinaryOperator::Or => match (left, right) {
                (SochValue::Bool(l), SochValue::Bool(r)) => Ok(SochValue::Bool(*l || *r)),
                (SochValue::Bool(true), _) | (_, SochValue::Bool(true)) => {
                    Ok(SochValue::Bool(true))
                }
                (SochValue::Null, _) | (_, SochValue::Null) => Ok(SochValue::Null),
                _ => Err(SqlError::TypeError("OR requires boolean operands".into())),
            },

            BinaryOperator::Plus => self.arithmetic_op(left, right, |a, b| a + b, |a, b| a + b),
            BinaryOperator::Minus => self.arithmetic_op(left, right, |a, b| a - b, |a, b| a - b),
            BinaryOperator::Multiply => self.arithmetic_op(left, right, |a, b| a * b, |a, b| a * b),
            BinaryOperator::Divide => self.arithmetic_op(
                left,
                right,
                |a, b| if b != 0 { a / b } else { 0 },
                |a, b| a / b,
            ),
            BinaryOperator::Modulo => self.arithmetic_op(
                left,
                right,
                |a, b| if b != 0 { a % b } else { 0 },
                |a, b| a % b,
            ),

            BinaryOperator::Concat => match (left, right) {
                (SochValue::Text(l), SochValue::Text(r)) => {
                    Ok(SochValue::Text(format!("{}{}", l, r)))
                }
                (SochValue::Null, _) | (_, SochValue::Null) => Ok(SochValue::Null),
                _ => Err(SqlError::TypeError("|| requires string operands".into())),
            },

            _ => Err(SqlError::NotImplemented(format!(
                "Operator {:?} not implemented",
                op
            ))),
        }
    }

    fn evaluate_unary_op(&self, op: &UnaryOperator, val: &SochValue) -> SqlResult<SochValue> {
        match op {
            UnaryOperator::Not => match val {
                SochValue::Bool(b) => Ok(SochValue::Bool(!b)),
                SochValue::Null => Ok(SochValue::Null),
                _ => Err(SqlError::TypeError("NOT requires boolean operand".into())),
            },
            UnaryOperator::Minus => match val {
                SochValue::Int(n) => Ok(SochValue::Int(-n)),
                SochValue::Float(f) => Ok(SochValue::Float(-f)),
                SochValue::Null => Ok(SochValue::Null),
                _ => Err(SqlError::TypeError(
                    "Unary minus requires numeric operand".into(),
                )),
            },
            UnaryOperator::Plus => Ok(val.clone()),
            UnaryOperator::BitNot => match val {
                SochValue::Int(n) => Ok(SochValue::Int(!n)),
                _ => Err(SqlError::TypeError("~ requires integer operand".into())),
            },
        }
    }

    fn evaluate_function(
        &self,
        func: &FunctionCall,
        row: &HashMap<String, SochValue>,
        params: &[SochValue],
    ) -> SqlResult<SochValue> {
        let func_name = func.name.name().to_uppercase();

        match func_name.as_str() {
            "COALESCE" => {
                for arg in &func.args {
                    let val = self.evaluate_expr(arg, row, params)?;
                    if !matches!(val, SochValue::Null) {
                        return Ok(val);
                    }
                }
                Ok(SochValue::Null)
            }

            "NULLIF" => {
                if func.args.len() != 2 {
                    return Err(SqlError::InvalidArgument(
                        "NULLIF requires 2 arguments".into(),
                    ));
                }
                let val1 = self.evaluate_expr(&func.args[0], row, params)?;
                let val2 = self.evaluate_expr(&func.args[1], row, params)?;
                if self.values_equal(&val1, &val2) {
                    Ok(SochValue::Null)
                } else {
                    Ok(val1)
                }
            }

            "ABS" => {
                if func.args.len() != 1 {
                    return Err(SqlError::InvalidArgument("ABS requires 1 argument".into()));
                }
                let val = self.evaluate_expr(&func.args[0], row, params)?;
                match val {
                    SochValue::Int(n) => Ok(SochValue::Int(n.abs())),
                    SochValue::Float(f) => Ok(SochValue::Float(f.abs())),
                    SochValue::Null => Ok(SochValue::Null),
                    _ => Err(SqlError::TypeError("ABS requires numeric argument".into())),
                }
            }

            "LENGTH" | "LEN" => {
                if func.args.len() != 1 {
                    return Err(SqlError::InvalidArgument(
                        "LENGTH requires 1 argument".into(),
                    ));
                }
                let val = self.evaluate_expr(&func.args[0], row, params)?;
                match val {
                    SochValue::Text(s) => Ok(SochValue::Int(s.len() as i64)),
                    SochValue::Binary(b) => Ok(SochValue::Int(b.len() as i64)),
                    SochValue::Null => Ok(SochValue::Null),
                    _ => Err(SqlError::TypeError(
                        "LENGTH requires string argument".into(),
                    )),
                }
            }

            "UPPER" => {
                if func.args.len() != 1 {
                    return Err(SqlError::InvalidArgument(
                        "UPPER requires 1 argument".into(),
                    ));
                }
                let val = self.evaluate_expr(&func.args[0], row, params)?;
                match val {
                    SochValue::Text(s) => Ok(SochValue::Text(s.to_uppercase())),
                    SochValue::Null => Ok(SochValue::Null),
                    _ => Err(SqlError::TypeError("UPPER requires string argument".into())),
                }
            }

            "LOWER" => {
                if func.args.len() != 1 {
                    return Err(SqlError::InvalidArgument(
                        "LOWER requires 1 argument".into(),
                    ));
                }
                let val = self.evaluate_expr(&func.args[0], row, params)?;
                match val {
                    SochValue::Text(s) => Ok(SochValue::Text(s.to_lowercase())),
                    SochValue::Null => Ok(SochValue::Null),
                    _ => Err(SqlError::TypeError("LOWER requires string argument".into())),
                }
            }

            "TRIM" => {
                if func.args.len() != 1 {
                    return Err(SqlError::InvalidArgument("TRIM requires 1 argument".into()));
                }
                let val = self.evaluate_expr(&func.args[0], row, params)?;
                match val {
                    SochValue::Text(s) => Ok(SochValue::Text(s.trim().to_string())),
                    SochValue::Null => Ok(SochValue::Null),
                    _ => Err(SqlError::TypeError("TRIM requires string argument".into())),
                }
            }

            "SUBSTR" | "SUBSTRING" => {
                if func.args.len() < 2 || func.args.len() > 3 {
                    return Err(SqlError::InvalidArgument(
                        "SUBSTR requires 2 or 3 arguments".into(),
                    ));
                }
                let val = self.evaluate_expr(&func.args[0], row, params)?;
                let start = self.evaluate_expr(&func.args[1], row, params)?;
                let len = if func.args.len() == 3 {
                    Some(self.evaluate_expr(&func.args[2], row, params)?)
                } else {
                    None
                };

                match (val, start) {
                    (SochValue::Text(s), SochValue::Int(start)) => {
                        let start = (start.max(1) - 1) as usize;
                        if start >= s.len() {
                            return Ok(SochValue::Text(String::new()));
                        }
                        let result = if let Some(SochValue::Int(len)) = len {
                            s.chars().skip(start).take(len as usize).collect()
                        } else {
                            s.chars().skip(start).collect()
                        };
                        Ok(SochValue::Text(result))
                    }
                    (SochValue::Null, _) | (_, SochValue::Null) => Ok(SochValue::Null),
                    _ => Err(SqlError::TypeError(
                        "SUBSTR requires string and integer arguments".into(),
                    )),
                }
            }

            _ => Err(SqlError::NotImplemented(format!(
                "Function {} not implemented",
                func_name
            ))),
        }
    }

    // ========== Helper Methods ==========

    fn values_equal(&self, left: &SochValue, right: &SochValue) -> bool {
        match (left, right) {
            (SochValue::Null, _) | (_, SochValue::Null) => false,
            (SochValue::Int(l), SochValue::Int(r)) => l == r,
            (SochValue::Float(l), SochValue::Float(r)) => (l - r).abs() < f64::EPSILON,
            (SochValue::Int(l), SochValue::Float(r)) => (*l as f64 - r).abs() < f64::EPSILON,
            (SochValue::Float(l), SochValue::Int(r)) => (l - *r as f64).abs() < f64::EPSILON,
            (SochValue::Text(l), SochValue::Text(r)) => l == r,
            (SochValue::Bool(l), SochValue::Bool(r)) => l == r,
            (SochValue::Binary(l), SochValue::Binary(r)) => l == r,
            (SochValue::UInt(l), SochValue::UInt(r)) => l == r,
            (SochValue::Int(l), SochValue::UInt(r)) => *l >= 0 && (*l as u64) == *r,
            (SochValue::UInt(l), SochValue::Int(r)) => *r >= 0 && *l == (*r as u64),
            _ => false,
        }
    }

    fn compare_values(
        &self,
        left: Option<&SochValue>,
        right: Option<&SochValue>,
    ) -> std::cmp::Ordering {
        match (left, right) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, _) => std::cmp::Ordering::Less,
            (_, None) => std::cmp::Ordering::Greater,
            (Some(SochValue::Null), _) | (_, Some(SochValue::Null)) => std::cmp::Ordering::Equal,
            (Some(SochValue::Int(l)), Some(SochValue::Int(r))) => l.cmp(r),
            (Some(SochValue::Float(l)), Some(SochValue::Float(r))) => {
                l.partial_cmp(r).unwrap_or(std::cmp::Ordering::Equal)
            }
            (Some(SochValue::Int(l)), Some(SochValue::Float(r))) => (*l as f64)
                .partial_cmp(r)
                .unwrap_or(std::cmp::Ordering::Equal),
            (Some(SochValue::Float(l)), Some(SochValue::Int(r))) => l
                .partial_cmp(&(*r as f64))
                .unwrap_or(std::cmp::Ordering::Equal),
            (Some(SochValue::Text(l)), Some(SochValue::Text(r))) => l.cmp(r),
            (Some(SochValue::UInt(l)), Some(SochValue::UInt(r))) => l.cmp(r),
            _ => std::cmp::Ordering::Equal,
        }
    }

    fn arithmetic_op<FI, FF>(
        &self,
        left: &SochValue,
        right: &SochValue,
        int_op: FI,
        float_op: FF,
    ) -> SqlResult<SochValue>
    where
        FI: Fn(i64, i64) -> i64,
        FF: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (SochValue::Null, _) | (_, SochValue::Null) => Ok(SochValue::Null),
            (SochValue::Int(l), SochValue::Int(r)) => Ok(SochValue::Int(int_op(*l, *r))),
            (SochValue::Float(l), SochValue::Float(r)) => Ok(SochValue::Float(float_op(*l, *r))),
            (SochValue::Int(l), SochValue::Float(r)) => {
                Ok(SochValue::Float(float_op(*l as f64, *r)))
            }
            (SochValue::Float(l), SochValue::Int(r)) => {
                Ok(SochValue::Float(float_op(*l, *r as f64)))
            }
            (SochValue::UInt(l), SochValue::UInt(r)) => {
                Ok(SochValue::Int(int_op(*l as i64, *r as i64)))
            }
            (SochValue::Int(l), SochValue::UInt(r)) => Ok(SochValue::Int(int_op(*l, *r as i64))),
            (SochValue::UInt(l), SochValue::Int(r)) => Ok(SochValue::Int(int_op(*l as i64, *r))),
            _ => Err(SqlError::TypeError(
                "Arithmetic requires numeric operands".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_and_insert() {
        let mut executor = SqlExecutor::new();

        // Create table
        let result = executor
            .execute("CREATE TABLE users (id INTEGER, name VARCHAR(100))")
            .unwrap();
        assert!(matches!(result, ExecutionResult::Ok));

        // Insert rows
        let result = executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);

        let result = executor
            .execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);

        // Select
        let result = executor.execute("SELECT * FROM users").unwrap();
        assert_eq!(result.rows_affected(), 2);
    }

    #[test]
    fn test_select_with_where() {
        let mut executor = SqlExecutor::new();

        executor
            .execute("CREATE TABLE products (id INTEGER, name TEXT, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, name, price) VALUES (1, 'Apple', 1.50)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, name, price) VALUES (2, 'Banana', 0.75)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, name, price) VALUES (3, 'Orange', 2.00)")
            .unwrap();

        let result = executor
            .execute("SELECT * FROM products WHERE price > 1.0")
            .unwrap();
        assert_eq!(result.rows_affected(), 2);
    }

    #[test]
    fn test_update() {
        let mut executor = SqlExecutor::new();

        executor
            .execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();

        let result = executor
            .execute("UPDATE users SET name = 'Alicia' WHERE id = 1")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);

        let result = executor
            .execute("SELECT * FROM users WHERE name = 'Alicia'")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);
    }

    #[test]
    fn test_delete() {
        let mut executor = SqlExecutor::new();

        executor
            .execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        executor
            .execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .unwrap();

        let result = executor.execute("DELETE FROM users WHERE id = 1").unwrap();
        assert_eq!(result.rows_affected(), 1);

        let result = executor.execute("SELECT * FROM users").unwrap();
        assert_eq!(result.rows_affected(), 1);
    }

    #[test]
    fn test_functions() {
        let mut executor = SqlExecutor::new();

        executor.execute("CREATE TABLE t (s TEXT)").unwrap();
        executor
            .execute("INSERT INTO t (s) VALUES ('hello')")
            .unwrap();

        let result = executor.execute("SELECT UPPER(s) FROM t").unwrap();
        if let ExecutionResult::Rows { rows, .. } = result {
            let row = &rows[0];
            // The column name might be UPPER(s) or similar
            assert!(
                row.values()
                    .any(|v| matches!(v, SochValue::Text(s) if s == "HELLO"))
            );
        } else {
            panic!("Expected rows");
        }
    }

    #[test]
    fn test_order_by() {
        let mut executor = SqlExecutor::new();

        executor.execute("CREATE TABLE nums (n INTEGER)").unwrap();
        executor.execute("INSERT INTO nums (n) VALUES (3)").unwrap();
        executor.execute("INSERT INTO nums (n) VALUES (1)").unwrap();
        executor.execute("INSERT INTO nums (n) VALUES (2)").unwrap();

        let result = executor
            .execute("SELECT * FROM nums ORDER BY n ASC")
            .unwrap();
        if let ExecutionResult::Rows { rows, .. } = result {
            let values: Vec<i64> = rows
                .iter()
                .filter_map(|r| r.get("n"))
                .filter_map(|v| {
                    if let SochValue::Int(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(values, vec![1, 2, 3]);
        } else {
            panic!("Expected rows");
        }
    }

    #[test]
    fn test_limit_offset() {
        let mut executor = SqlExecutor::new();

        executor.execute("CREATE TABLE nums (n INTEGER)").unwrap();
        for i in 1..=10 {
            executor
                .execute(&format!("INSERT INTO nums (n) VALUES ({})", i))
                .unwrap();
        }

        let result = executor
            .execute("SELECT * FROM nums LIMIT 3 OFFSET 2")
            .unwrap();
        assert_eq!(result.rows_affected(), 3);
    }

    #[test]
    fn test_between() {
        let mut executor = SqlExecutor::new();

        executor.execute("CREATE TABLE nums (n INTEGER)").unwrap();
        for i in 1..=10 {
            executor
                .execute(&format!("INSERT INTO nums (n) VALUES ({})", i))
                .unwrap();
        }

        let result = executor
            .execute("SELECT * FROM nums WHERE n BETWEEN 3 AND 7")
            .unwrap();
        assert_eq!(result.rows_affected(), 5);
    }

    #[test]
    fn test_in_list() {
        let mut executor = SqlExecutor::new();

        executor.execute("CREATE TABLE nums (n INTEGER)").unwrap();
        for i in 1..=5 {
            executor
                .execute(&format!("INSERT INTO nums (n) VALUES ({})", i))
                .unwrap();
        }

        let result = executor
            .execute("SELECT * FROM nums WHERE n IN (1, 3, 5)")
            .unwrap();
        assert_eq!(result.rows_affected(), 3);
    }
}
