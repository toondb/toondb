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

//! JSON Structured Logging Plugin for SochDB
//!
//! This is an example plugin demonstrating the SochDB plugin architecture.
//! It provides structured JSON logging for observability.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_kernel::PluginManager;
//! use sochdb_plugin_logging::JsonLoggingPlugin;
//! use std::sync::Arc;
//!
//! let plugins = PluginManager::new();
//! let logging = JsonLoggingPlugin::new();
//! plugins.register_observability(Arc::new(logging))?;
//! ```

use serde::Serialize;
use std::any::Any;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use sochdb_kernel::{Extension, ExtensionCapability, ExtensionInfo, ObservabilityExtension};

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Structured log entry
#[derive(Debug, Serialize)]
struct LogEntry<'a> {
    timestamp: String,
    level: LogLevel,
    message: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    fields: Vec<(&'a str, &'a str)>,
}

/// Metric entry
#[derive(Debug, Serialize)]
struct MetricEntry<'a> {
    timestamp: String,
    metric_type: &'static str,
    name: &'a str,
    value: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    labels: Vec<(&'a str, &'a str)>,
}

/// Span entry
#[derive(Debug, Serialize)]
struct SpanEntry<'a> {
    timestamp: String,
    span_id: u64,
    parent_id: Option<u64>,
    event: &'a str,
    name: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    attributes: Vec<(&'a str, &'a str)>,
}

/// JSON Structured Logging Plugin
///
/// Outputs structured JSON logs to stdout (can be redirected to files or log aggregators).
pub struct JsonLoggingPlugin {
    /// Next span ID
    next_span_id: AtomicU64,
    /// Minimum log level
    min_level: LogLevel,
    /// Include metrics in output
    include_metrics: bool,
    /// Include traces in output
    include_traces: bool,
}

impl Default for JsonLoggingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLoggingPlugin {
    /// Create a new JSON logging plugin
    pub fn new() -> Self {
        Self {
            next_span_id: AtomicU64::new(1),
            min_level: LogLevel::Info,
            include_metrics: true,
            include_traces: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(min_level: LogLevel, include_metrics: bool, include_traces: bool) -> Self {
        Self {
            next_span_id: AtomicU64::new(1),
            min_level,
            include_metrics,
            include_traces,
        }
    }

    /// Get current timestamp in ISO 8601 format
    fn timestamp() -> String {
        chrono::Utc::now().to_rfc3339()
    }

    /// Write a log entry
    fn write_log(&self, level: LogLevel, message: &str, fields: &[(&str, &str)]) {
        if level as u8 >= self.min_level as u8 {
            let entry = LogEntry {
                timestamp: Self::timestamp(),
                level,
                message,
                fields: fields.to_vec(),
            };

            if let Ok(json) = serde_json::to_string(&entry) {
                let _ = writeln!(io::stdout(), "{}", json);
            }
        }
    }

    /// Write a metric entry
    fn write_metric(
        &self,
        metric_type: &'static str,
        name: &str,
        value: f64,
        labels: &[(&str, &str)],
    ) {
        if self.include_metrics {
            let entry = MetricEntry {
                timestamp: Self::timestamp(),
                metric_type,
                name,
                value,
                labels: labels.to_vec(),
            };

            if let Ok(json) = serde_json::to_string(&entry) {
                let _ = writeln!(io::stdout(), "{}", json);
            }
        }
    }

    /// Write a span entry
    fn write_span(
        &self,
        event: &str,
        span_id: u64,
        parent_id: Option<u64>,
        name: &str,
        attributes: &[(&str, &str)],
    ) {
        if self.include_traces {
            let entry = SpanEntry {
                timestamp: Self::timestamp(),
                span_id,
                parent_id,
                event,
                name,
                attributes: attributes.to_vec(),
            };

            if let Ok(json) = serde_json::to_string(&entry) {
                let _ = writeln!(io::stdout(), "{}", json);
            }
        }
    }
}

impl Extension for JsonLoggingPlugin {
    fn info(&self) -> ExtensionInfo {
        ExtensionInfo {
            name: "json-logging".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            description: "JSON structured logging for SochDB".into(),
            author: "SochDB Contributors".into(),
            capabilities: vec![ExtensionCapability::Observability],
        }
    }

    fn init(&mut self) -> sochdb_kernel::KernelResult<()> {
        self.write_log(LogLevel::Info, "JSON logging plugin initialized", &[]);
        Ok(())
    }

    fn shutdown(&mut self) -> sochdb_kernel::KernelResult<()> {
        self.write_log(LogLevel::Info, "JSON logging plugin shutting down", &[]);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ObservabilityExtension for JsonLoggingPlugin {
    // -------------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------------

    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        self.write_metric("counter", name, value as f64, labels);
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        self.write_metric("gauge", name, value, labels);
    }

    fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        self.write_metric("histogram", name, value, labels);
    }

    // -------------------------------------------------------------------------
    // Tracing
    // -------------------------------------------------------------------------

    fn span_start(&self, name: &str, parent: Option<u64>) -> u64 {
        let span_id = self.next_span_id.fetch_add(1, Ordering::SeqCst);
        self.write_span("start", span_id, parent, name, &[]);
        span_id
    }

    fn span_end(&self, span_id: u64) {
        self.write_span("end", span_id, None, "", &[]);
    }

    fn span_event(&self, span_id: u64, name: &str, attributes: &[(&str, &str)]) {
        self.write_span("event", span_id, None, name, attributes);
    }

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    fn log_debug(&self, message: &str, fields: &[(&str, &str)]) {
        self.write_log(LogLevel::Debug, message, fields);
    }

    fn log_info(&self, message: &str, fields: &[(&str, &str)]) {
        self.write_log(LogLevel::Info, message, fields);
    }

    fn log_warn(&self, message: &str, fields: &[(&str, &str)]) {
        self.write_log(LogLevel::Warn, message, fields);
    }

    fn log_error(&self, message: &str, fields: &[(&str, &str)]) {
        self.write_log(LogLevel::Error, message, fields);
    }

    // -------------------------------------------------------------------------
    // Health
    // -------------------------------------------------------------------------

    fn report_health(&self, health: &sochdb_kernel::kernel_api::HealthInfo) {
        let fields = [(
            "operational",
            if health.operational { "true" } else { "false" },
        )];
        self.write_log(LogLevel::Debug, "health check", &fields);
        self.gauge_set("sochdb_wal_size_bytes", health.wal_size_bytes as f64, &[]);
        self.gauge_set("sochdb_active_txns", health.active_txns as f64, &[]);
        self.gauge_set("sochdb_buffer_pool_usage", health.buffer_pool_usage, &[]);
        self.gauge_set(
            "sochdb_checkpoint_age_secs",
            health.checkpoint_age_secs as f64,
            &[],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_creation() {
        let plugin = JsonLoggingPlugin::new();
        let info = plugin.info();
        assert_eq!(info.name, "json-logging");
    }

    #[test]
    fn test_metrics() {
        let plugin = JsonLoggingPlugin::new();

        // These should not panic
        plugin.counter_inc("test_counter", 1, &[("env", "test")]);
        plugin.gauge_set("test_gauge", 42.0, &[]);
        plugin.histogram_observe("test_histogram", 0.5, &[("bucket", "p99")]);
    }

    #[test]
    fn test_spans() {
        let plugin = JsonLoggingPlugin::new();

        let span1 = plugin.span_start("parent_op", None);
        let span2 = plugin.span_start("child_op", Some(span1));

        plugin.span_event(span2, "processing", &[("items", "100")]);
        plugin.span_end(span2);
        plugin.span_end(span1);
    }

    #[test]
    fn test_logging() {
        let plugin = JsonLoggingPlugin::new();

        plugin.log_debug("debug message", &[]);
        plugin.log_info("info message", &[("key", "value")]);
        plugin.log_warn("warning", &[]);
        plugin.log_error("error occurred", &[("code", "500")]);
    }
}
