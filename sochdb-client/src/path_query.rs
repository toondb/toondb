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

//! Path-Native Query Builder
//!
//! Leverages TCH's O(|path|) resolution for efficient queries.
//!
//! ## Complexity
//!
//! - Path resolution: O(|path|) via radix-compressed trie traversal
//! - Column scan: O(N/B) where N = rows, B = block size
//! - Predicate pushdown: Filter applied during scan, reducing to O(k) where k = matching rows
//! - Total: O(|path|) + O(N/B) for scan, or O(|path|) + O(log N) with learned index

use crate::connection::{PathResolution, SochConnection};
use crate::error::{ClientError, Result};
use crate::result::SochResult;

use sochdb_core::soch::SochValue;

#[cfg(test)]
use crate::connection::FieldType;

/// Comparison operators for predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Like,
    In,
    IsNull,
    IsNotNull,
}

/// Predicate that operates on ColumnStore directly
#[derive(Debug, Clone)]
pub struct ColumnPredicate {
    pub field: String,
    pub op: CompareOp,
    pub value: SochValue,
}

impl ColumnPredicate {
    pub fn new(field: &str, op: CompareOp, value: SochValue) -> Self {
        Self {
            field: field.to_string(),
            op,
            value,
        }
    }

    /// Evaluate predicate against a value
    pub fn evaluate(&self, value: &SochValue) -> bool {
        match self.op {
            CompareOp::Eq => value == &self.value,
            CompareOp::Ne => value != &self.value,
            CompareOp::Lt => self.compare_ord(value) == Some(std::cmp::Ordering::Less),
            CompareOp::Le => matches!(
                self.compare_ord(value),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            CompareOp::Gt => self.compare_ord(value) == Some(std::cmp::Ordering::Greater),
            CompareOp::Ge => matches!(
                self.compare_ord(value),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
            CompareOp::Like => self.match_like(value),
            CompareOp::In => self.match_in(value),
            CompareOp::IsNull => matches!(value, SochValue::Null),
            CompareOp::IsNotNull => !matches!(value, SochValue::Null),
        }
    }

    fn compare_ord(&self, value: &SochValue) -> Option<std::cmp::Ordering> {
        match (value, &self.value) {
            (SochValue::Int(a), SochValue::Int(b)) => Some(a.cmp(b)),
            (SochValue::Float(a), SochValue::Float(b)) => a.partial_cmp(b),
            (SochValue::Text(a), SochValue::Text(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    fn match_like(&self, value: &SochValue) -> bool {
        match (value, &self.value) {
            (SochValue::Text(text), SochValue::Text(pattern)) => {
                // Simple LIKE implementation: % = any, _ = single char
                let pattern = pattern.replace('%', ".*").replace('_', ".");
                regex_lite_match(&pattern, text)
            }
            _ => false,
        }
    }

    fn match_in(&self, value: &SochValue) -> bool {
        match &self.value {
            SochValue::Array(values) => values.contains(value),
            _ => false,
        }
    }
}

/// Simple regex-like matching (without full regex dependency)
fn regex_lite_match(pattern: &str, text: &str) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }

    // Handle .* (any chars)
    if let Some(rest) = pattern.strip_prefix(".*") {
        for i in 0..=text.len() {
            if regex_lite_match(rest, &text[i..]) {
                return true;
            }
        }
        return false;
    }

    // Handle . (single char)
    if pattern.starts_with('.') && !text.is_empty() {
        return regex_lite_match(&pattern[1..], &text[1..]);
    }

    // Handle literal
    if !text.is_empty() && pattern.starts_with(text.chars().next().unwrap()) {
        return regex_lite_match(&pattern[1..], &text[1..]);
    }

    false
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Path-native query builder for TCH
pub struct PathQuery<'a> {
    conn: &'a SochConnection,
    /// Root path being queried
    path: String,
    /// Predicates to push down to ColumnStore
    predicates: Vec<ColumnPredicate>,
    /// Column projection (None = all)
    projection: Option<Vec<String>>,
    /// Order by
    order_by: Option<(String, SortDirection)>,
    /// Limit
    limit: Option<usize>,
    /// Offset
    offset: Option<usize>,
}

impl<'a> PathQuery<'a> {
    /// Start query from path - O(|path|) resolution
    pub fn from_path(conn: &'a SochConnection, path: &str) -> Self {
        Self {
            conn,
            path: path.to_string(),
            predicates: vec![],
            projection: None,
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Navigate to nested path: O(|subpath|) additional
    pub fn nested(mut self, subpath: &str) -> Self {
        self.path = format!("{}.{}", self.path, subpath);
        self
    }

    /// Add predicate (pushed to ColumnStore scan)
    pub fn filter(mut self, field: &str, op: CompareOp, value: impl Into<SochValue>) -> Self {
        self.predicates
            .push(ColumnPredicate::new(field, op, value.into()));
        self
    }

    /// Shorthand for equality filter
    pub fn where_eq(self, field: &str, value: impl Into<SochValue>) -> Self {
        self.filter(field, CompareOp::Eq, value)
    }

    /// Shorthand for greater-than filter
    pub fn where_gt(self, field: &str, value: impl Into<SochValue>) -> Self {
        self.filter(field, CompareOp::Gt, value)
    }

    /// Shorthand for less-than filter
    pub fn where_lt(self, field: &str, value: impl Into<SochValue>) -> Self {
        self.filter(field, CompareOp::Lt, value)
    }

    /// Shorthand for LIKE filter
    pub fn where_like(self, field: &str, pattern: &str) -> Self {
        self.filter(field, CompareOp::Like, SochValue::Text(pattern.to_string()))
    }

    /// Project specific columns
    pub fn project(mut self, columns: &[impl AsRef<str>]) -> Self {
        self.projection = Some(columns.iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Select specific columns (alias for project)
    pub fn select(self, columns: &[impl AsRef<str>]) -> Self {
        self.project(columns)
    }

    /// Order by column
    pub fn order_by(mut self, column: &str, ascending: bool) -> Self {
        self.order_by = Some((
            column.to_string(),
            if ascending {
                SortDirection::Asc
            } else {
                SortDirection::Desc
            },
        ));
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Skip first n results
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Execute and return SochResult for streaming output
    pub fn execute(self) -> Result<SochResult<'a>> {
        self.conn.record_query();

        // 1. Resolve path via TCH - O(|path|)
        let resolution = self.conn.resolve(&self.path)?;

        match resolution {
            PathResolution::Array { schema, columns } => {
                // 2. Apply projection if specified
                let projected_columns = if let Some(ref proj) = self.projection {
                    columns
                        .into_iter()
                        .filter(|c| proj.contains(&c.name))
                        .collect()
                } else {
                    columns
                };

                // 3. Create result with metadata
                Ok(SochResult::new(
                    self.path.clone(),
                    schema,
                    projected_columns,
                    self.predicates,
                    self.order_by,
                    self.limit,
                    self.offset,
                ))
            }
            PathResolution::Value(_col_ref) => {
                // Scalar value access
                Err(ClientError::ScalarPath(self.path))
            }
            PathResolution::Partial { remaining } => Err(ClientError::PathNotFound(format!(
                "{} (partial match, remaining: {})",
                self.path, remaining
            ))),
            PathResolution::NotFound => Err(ClientError::PathNotFound(self.path)),
        }
    }

    /// Execute and collect all results
    pub fn collect(self) -> Result<Vec<String>> {
        let result = self.execute()?;
        Ok(result.collect())
    }

    /// Execute and get TOON string
    pub fn to_toon(self) -> Result<String> {
        let mut result = self.execute()?;
        Ok(result.to_soch_string())
    }

    /// Get the path being queried
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Predicate builder helpers
#[derive(Debug, Clone)]
pub struct Predicate {
    pub column: String,
    pub op: CompareOp,
    pub value: SochValue,
}

impl Predicate {
    pub fn eq(column: &str, value: SochValue) -> Self {
        Self {
            column: column.to_string(),
            op: CompareOp::Eq,
            value,
        }
    }

    pub fn gt(column: &str, value: SochValue) -> Self {
        Self {
            column: column.to_string(),
            op: CompareOp::Gt,
            value,
        }
    }

    pub fn lt(column: &str, value: SochValue) -> Self {
        Self {
            column: column.to_string(),
            op: CompareOp::Lt,
            value,
        }
    }

    pub fn like(column: &str, pattern: &str) -> Self {
        Self {
            column: column.to_string(),
            op: CompareOp::Like,
            value: SochValue::Text(pattern.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_query_basic() {
        let conn = SochConnection::open("./test").unwrap();

        conn.register_table(
            "users",
            &[
                ("id".to_string(), FieldType::UInt64),
                ("name".to_string(), FieldType::Text),
            ],
        )
        .unwrap();

        let query = PathQuery::from_path(&conn, "users");
        assert_eq!(query.path(), "users");
    }

    #[test]
    fn test_path_query_nested() {
        let conn = SochConnection::open("./test").unwrap();

        let query = PathQuery::from_path(&conn, "users")
            .nested("profile")
            .nested("settings");

        assert_eq!(query.path(), "users.profile.settings");
    }

    #[test]
    fn test_path_query_filter() {
        let conn = SochConnection::open("./test").unwrap();

        conn.register_table(
            "users",
            &[
                ("id".to_string(), FieldType::UInt64),
                ("score".to_string(), FieldType::Float64),
            ],
        )
        .unwrap();

        let query = PathQuery::from_path(&conn, "users")
            .where_gt("score", SochValue::Float(80.0))
            .limit(10);

        assert_eq!(query.predicates.len(), 1);
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_predicate_evaluate() {
        let pred = ColumnPredicate::new("score", CompareOp::Gt, SochValue::Int(80));

        assert!(pred.evaluate(&SochValue::Int(90)));
        assert!(!pred.evaluate(&SochValue::Int(70)));
        assert!(!pred.evaluate(&SochValue::Int(80)));
    }

    #[test]
    fn test_predicate_like() {
        let pred = ColumnPredicate::new(
            "name",
            CompareOp::Like,
            SochValue::Text("John%".to_string()),
        );

        assert!(pred.evaluate(&SochValue::Text("John".to_string())));
        assert!(pred.evaluate(&SochValue::Text("Johnny".to_string())));
        assert!(!pred.evaluate(&SochValue::Text("Jane".to_string())));
    }

    #[test]
    fn test_regex_lite_match() {
        assert!(regex_lite_match("hello", "hello"));
        assert!(!regex_lite_match("hello", "world"));
        assert!(regex_lite_match(".*world", "hello world"));
        assert!(regex_lite_match("h.llo", "hello"));
    }
}
