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

//! Result Streaming with Token Metrics
//!
//! Wraps SochCursor with real-time token efficiency tracking.
//!
//! ## Token Model
//!
//! - TOON: `T_TOON = H + N × (Σ|vᵢ| + C - 1)`
//! - JSON: `T_JSON = N × (C × (|fᵢ| + |vᵢ| + 4) + 2)`
//! - Savings: `1 - T_TOON/T_JSON ≈ 0.4 to 0.66`

use std::sync::Arc;

use crate::connection::{ArraySchema, ColumnRef};
use crate::path_query::{ColumnPredicate, SortDirection};

/// Result metrics for token efficiency tracking
#[derive(Debug, Clone, Default)]
pub struct ResultMetrics {
    /// Tokens emitted so far (TOON format)
    pub soch_tokens: usize,
    /// Equivalent JSON tokens (for comparison)
    pub json_tokens_equivalent: usize,
    /// Rows emitted
    pub rows_emitted: usize,
    /// Bytes emitted
    pub bytes_emitted: usize,
}

impl ResultMetrics {
    /// Token savings vs JSON (percentage)
    pub fn token_savings_percent(&self) -> f64 {
        if self.json_tokens_equivalent == 0 {
            return 0.0;
        }
        (1.0 - (self.soch_tokens as f64 / self.json_tokens_equivalent as f64)) * 100.0
    }

    /// Token reduction ratio
    pub fn reduction_ratio(&self) -> f64 {
        if self.json_tokens_equivalent == 0 {
            return 1.0;
        }
        self.soch_tokens as f64 / self.json_tokens_equivalent as f64
    }
}

/// SDK result set with token metrics
#[allow(dead_code)]
pub struct SochResult<'a> {
    /// Table/path name
    path: String,
    /// Schema of the result
    schema: Arc<ArraySchema>,
    /// Projected columns
    columns: Vec<ColumnRef>,
    /// Predicates for filtering
    predicates: Vec<ColumnPredicate>,
    /// Order by
    order_by: Option<(String, SortDirection)>,
    /// Limit
    limit: Option<usize>,
    /// Offset
    offset: Option<usize>,
    /// Token metrics
    metrics: ResultMetrics,
    /// Rows emitted
    rows: Vec<String>,
    /// Header emitted
    header_emitted: bool,
    /// Token budget
    token_budget: Option<usize>,
    /// Lifetime marker
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> SochResult<'a> {
    /// Create a new result set
    pub fn new(
        path: String,
        schema: Arc<ArraySchema>,
        columns: Vec<ColumnRef>,
        predicates: Vec<ColumnPredicate>,
        order_by: Option<(String, SortDirection)>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Self {
        Self {
            path,
            schema,
            columns,
            predicates,
            order_by,
            limit,
            offset,
            metrics: ResultMetrics::default(),
            rows: Vec::new(),
            header_emitted: false,
            token_budget: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            path: String::new(),
            schema: Arc::new(ArraySchema {
                name: String::new(),
                fields: vec![],
                types: vec![],
            }),
            columns: vec![],
            predicates: vec![],
            order_by: None,
            limit: None,
            offset: None,
            metrics: ResultMetrics::default(),
            rows: Vec::new(),
            header_emitted: false,
            token_budget: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set token budget limit
    pub fn with_token_limit(mut self, budget: usize) -> Self {
        self.token_budget = Some(budget);
        self
    }

    /// Get current token metrics
    pub fn metrics(&self) -> &ResultMetrics {
        &self.metrics
    }

    /// Get the path being queried
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Collect all output as TOON string
    pub fn to_soch_string(&mut self) -> String {
        let mut output = String::new();

        // Emit header
        if !self.header_emitted {
            let header = self.emit_header();
            self.update_metrics(&header);
            output.push_str(&header);
            output.push('\n');
            self.header_emitted = true;
        }

        // Emit rows (simulated - in real impl, would iterate over actual data)
        // For now, just show the structure

        output
    }

    /// Stream to writer
    pub fn stream_to<W: std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        let output = self.to_soch_string();
        writer.write_all(output.as_bytes())
    }

    /// Collect all lines
    pub fn collect(self) -> Vec<String> {
        let mut result = vec![];

        // Header
        let col_names: Vec<_> = self.columns.iter().map(|c| c.name.as_str()).collect();
        let header = format!(
            "{}[{}]{{{}}}:",
            self.path,
            0, // row count - would be real in actual impl
            col_names.join(",")
        );
        result.push(header);

        result
    }

    fn emit_header(&self) -> String {
        let col_names: Vec<_> = self.columns.iter().map(|c| c.name.as_str()).collect();
        format!(
            "{}[{}]{{{}}}:",
            self.path,
            0, // Would be actual row count in real impl
            col_names.join(",")
        )
    }

    fn update_metrics(&mut self, line: &str) {
        let soch_tokens = estimate_tokens(line);
        let json_tokens = estimate_json_equivalent_tokens(line, &self.column_names());

        self.metrics.soch_tokens += soch_tokens;
        self.metrics.json_tokens_equivalent += json_tokens;
        self.metrics.bytes_emitted += line.len();

        if self.metrics.rows_emitted > 0 || !line.contains('{') {
            self.metrics.rows_emitted += 1;
        }

        // Check token budget
        if let Some(budget) = self.token_budget
            && self.metrics.soch_tokens > budget
        {
            // Would truncate in real impl
        }
    }
}

/// Rough token estimation (≈ bytes/4 for English text)
fn estimate_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// Estimate JSON equivalent tokens
fn estimate_json_equivalent_tokens(line: &str, fields: &[&str]) -> usize {
    // JSON: {"field1": "val1", "field2": "val2"}
    // Overhead per field: field name + quotes + colon + space + comma
    let field_overhead: usize = fields.iter().map(|f| f.len() + 5).sum();
    let base_tokens = estimate_tokens(line);
    base_tokens + field_overhead.div_ceil(4) // +2 for {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::FieldType;

    #[test]
    fn test_result_metrics() {
        let metrics = ResultMetrics {
            soch_tokens: 100,
            json_tokens_equivalent: 200,
            ..Default::default()
        };

        assert!((metrics.token_savings_percent() - 50.0).abs() < 0.1);
        assert!((metrics.reduction_ratio() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_empty_result() {
        let result = SochResult::empty();
        assert!(result.path().is_empty());
        assert!(result.column_names().is_empty());
    }

    #[test]
    fn test_token_estimation() {
        // ~4 chars per token: (len + 3) / 4
        assert_eq!(estimate_tokens("hello"), 2); // (5+3)/4 = 2
        assert_eq!(estimate_tokens("hello world test"), 4); // (16+3)/4 = 4
    }

    #[test]
    fn test_result_with_columns() {
        let schema = Arc::new(ArraySchema {
            name: "users".to_string(),
            fields: vec!["id".to_string(), "name".to_string()],
            types: vec![FieldType::UInt64, FieldType::Text],
        });

        let columns = vec![
            ColumnRef {
                id: 0,
                name: "id".to_string(),
                field_type: FieldType::UInt64,
            },
            ColumnRef {
                id: 1,
                name: "name".to_string(),
                field_type: FieldType::Text,
            },
        ];

        let result = SochResult::new(
            "users".to_string(),
            schema,
            columns,
            vec![],
            None,
            None,
            None,
        );

        assert_eq!(result.column_names(), vec!["id", "name"]);
    }
}
