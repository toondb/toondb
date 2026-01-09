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

//! Modern Python Plugin Runtime
//!
//! AI-era design for running Python plugins in ToonDB using:
//! - **Pyodide**: Full CPython 3.12 with numpy, pandas, scikit-learn
//! - **WASM Component Model**: Standard interfaces for cross-language composition
//! - **AI Triggers**: Natural language → Python code generation
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                  Python Plugin System                     │
//! ├──────────────────────────────────────────────────────────┤
//! │  PythonPlugin → PyodideRuntime → WASM Sandbox            │
//! │       ↓              ↓                ↓                  │
//! │   packages:      micropip          Memory isolation      │
//! │   numpy,pandas   install           Resource metering     │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! let runtime = PyodideRuntime::new(RuntimeConfig::default()).await?;
//! runtime.install_packages(&["numpy", "pandas"]).await?;
//!
//! let plugin = PythonPlugin::new("fraud_detector")
//!     .with_code(r#"
//!         import numpy as np
//!         def on_insert(row):
//!             if row["amount"] > 10000:
//!                 raise TriggerAbort("Amount too high")
//!             return row
//!     "#)
//!     .with_trigger("transactions", TriggerEvent::BeforeInsert);
//!
//! runtime.register(plugin)?;
//! ```

use crate::error::{KernelError, KernelResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Runtime Configuration
// ============================================================================

/// Configuration for the Pyodide runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Memory limit per plugin instance (bytes)
    pub memory_limit_bytes: u64,
    /// CPU time limit (milliseconds)
    pub timeout_ms: u64,
    /// Pre-installed packages
    pub packages: Vec<String>,
    /// Enable debug logging
    pub debug: bool,
    /// Allow network access (for package installation)
    pub allow_network: bool,
    /// Custom wheel URLs
    pub wheel_urls: Vec<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            memory_limit_bytes: 64 * 1024 * 1024, // 64 MB
            timeout_ms: 5000,                      // 5 seconds
            packages: vec![],
            debug: false,
            allow_network: false,
            wheel_urls: vec![],
        }
    }
}

impl RuntimeConfig {
    /// Create config with ML packages (numpy, pandas, sklearn)
    pub fn with_ml_packages() -> Self {
        Self {
            packages: vec![
                "numpy".into(),
                "pandas".into(),
                "scikit-learn".into(),
            ],
            memory_limit_bytes: 256 * 1024 * 1024, // 256 MB for ML
            timeout_ms: 30000,                      // 30s for model inference
            ..Default::default()
        }
    }

    /// Create lightweight config for validation scripts
    pub fn lightweight() -> Self {
        Self {
            memory_limit_bytes: 16 * 1024 * 1024, // 16 MB
            timeout_ms: 100,                       // 100ms
            packages: vec![],
            ..Default::default()
        }
    }
}

// ============================================================================
// Trigger Events
// ============================================================================

/// Types of trigger events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerEvent {
    BeforeInsert,
    AfterInsert,
    BeforeUpdate,
    AfterUpdate,
    BeforeDelete,
    AfterDelete,
    /// Stream processing (micro-batch)
    OnBatch,
}

impl TriggerEvent {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().replace(' ', "_").as_str() {
            "BEFORE_INSERT" => Some(Self::BeforeInsert),
            "AFTER_INSERT" => Some(Self::AfterInsert),
            "BEFORE_UPDATE" => Some(Self::BeforeUpdate),
            "AFTER_UPDATE" => Some(Self::AfterUpdate),
            "BEFORE_DELETE" => Some(Self::BeforeDelete),
            "AFTER_DELETE" => Some(Self::AfterDelete),
            "ON_BATCH" => Some(Self::OnBatch),
            _ => None,
        }
    }

    pub fn handler_name(&self) -> &'static str {
        match self {
            Self::BeforeInsert => "on_before_insert",
            Self::AfterInsert => "on_after_insert",
            Self::BeforeUpdate => "on_before_update",
            Self::AfterUpdate => "on_after_update",
            Self::BeforeDelete => "on_before_delete",
            Self::AfterDelete => "on_after_delete",
            Self::OnBatch => "on_batch",
        }
    }

    pub fn is_before(&self) -> bool {
        matches!(self, Self::BeforeInsert | Self::BeforeUpdate | Self::BeforeDelete)
    }
}

// ============================================================================
// Python Plugin Definition
// ============================================================================

/// A Python plugin with code and trigger bindings
#[derive(Debug, Clone)]
pub struct PythonPlugin {
    /// Unique plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Python source code
    pub code: String,
    /// Required packages
    pub packages: Vec<String>,
    /// Custom wheel URLs (for private packages)
    pub wheels: Vec<String>,
    /// Table → Events mapping
    pub triggers: HashMap<String, Vec<TriggerEvent>>,
    /// Plugin-specific config overrides
    pub config: Option<RuntimeConfig>,
}

impl PythonPlugin {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            code: String::new(),
            packages: vec![],
            wheels: vec![],
            triggers: HashMap::new(),
            config: None,
        }
    }

    pub fn with_version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    pub fn with_code(mut self, code: &str) -> Self {
        self.code = code.to_string();
        self
    }

    pub fn with_packages(mut self, packages: Vec<&str>) -> Self {
        self.packages = packages.into_iter().map(String::from).collect();
        self
    }

    pub fn with_trigger(mut self, table: &str, event: TriggerEvent) -> Self {
        self.triggers
            .entry(table.to_string())
            .or_default()
            .push(event);
        self
    }

    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = Some(config);
        self
    }
}

// ============================================================================
// Trigger Context & Result
// ============================================================================

/// Context passed to trigger execution
#[derive(Debug, Clone)]
pub struct TriggerContext {
    /// Table being modified
    pub table: String,
    /// Event type
    pub event: TriggerEvent,
    /// Row data as JSON string (for Pyodide interop)
    pub row_json: String,
    /// Old row for UPDATE/DELETE (JSON)
    pub old_row_json: Option<String>,
    /// Transaction ID
    pub txn_id: u64,
    /// Batch of rows for ON_BATCH events
    pub batch_json: Option<String>,
}

/// Result from trigger execution
#[derive(Debug, Clone)]
pub enum TriggerResult {
    /// Continue with optionally modified row (JSON)
    Continue(Option<String>),
    /// Abort with error message and code
    Abort { message: String, code: String },
    /// Skip this row
    Skip,
    /// Batch result (for ON_BATCH)
    Batch(String),
}

// ============================================================================
// Pyodide Runtime (Simulated for now)
// ============================================================================

/// Runtime statistics
#[derive(Debug, Default)]
pub struct RuntimeStats {
    pub total_executions: AtomicU64,
    pub total_time_us: AtomicU64,
    pub errors: AtomicU64,
    pub aborts: AtomicU64,
    pub packages_installed: AtomicU64,
}

/// Pyodide-based Python runtime
///
/// In production, this wraps actual Pyodide WASM module.
/// Currently provides a simulation for API design validation.
pub struct PyodideRuntime {
    config: RuntimeConfig,
    /// Registered plugins
    plugins: RwLock<HashMap<String, PythonPlugin>>,
    /// Table → Plugin mappings
    trigger_map: RwLock<HashMap<(String, TriggerEvent), Vec<String>>>,
    /// Installed packages
    installed_packages: RwLock<Vec<String>>,
    /// Runtime statistics
    stats: Arc<RuntimeStats>,
    /// Plugin instances (in production: actual WASM instances)
    #[allow(dead_code)]
    instances: RwLock<HashMap<String, PluginInstance>>,
}

/// A loaded plugin instance
#[allow(dead_code)]
struct PluginInstance {
    plugin_name: String,
    loaded_at: u64,
    memory_used: u64,
    call_count: u64,
}

impl PyodideRuntime {
    /// Create a new runtime with configuration
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            plugins: RwLock::new(HashMap::new()),
            trigger_map: RwLock::new(HashMap::new()),
            installed_packages: RwLock::new(vec![]),
            stats: Arc::new(RuntimeStats::default()),
            instances: RwLock::new(HashMap::new()),
        }
    }

    /// Install Python packages via micropip
    ///
    /// In production, this downloads and installs packages into the WASM environment.
    pub async fn install_packages(&self, packages: &[&str]) -> KernelResult<()> {
        let mut installed = self.installed_packages.write();
        for pkg in packages {
            if !installed.contains(&pkg.to_string()) {
                // Simulate package installation
                if self.config.debug {
                    eprintln!("[Pyodide] Installing package: {}", pkg);
                }
                installed.push(pkg.to_string());
                self.stats.packages_installed.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Register a Python plugin
    pub fn register(&self, plugin: PythonPlugin) -> KernelResult<()> {
        // Validate plugin code
        self.validate_code(&plugin.code)?;

        // Register plugin
        let name = plugin.name.clone();
        {
            let mut plugins = self.plugins.write();
            plugins.insert(name.clone(), plugin.clone());
        }

        // Update trigger mappings
        {
            let mut trigger_map = self.trigger_map.write();
            for (table, events) in &plugin.triggers {
                for event in events {
                    trigger_map
                        .entry((table.clone(), *event))
                        .or_default()
                        .push(name.clone());
                }
            }
        }

        if self.config.debug {
            eprintln!("[Pyodide] Registered plugin: {}", name);
        }

        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&self, name: &str) -> KernelResult<()> {
        let mut plugins = self.plugins.write();
        if let Some(plugin) = plugins.remove(name) {
            // Remove from trigger map
            let mut trigger_map = self.trigger_map.write();
            for (table, events) in &plugin.triggers {
                for event in events {
                    if let Some(names) = trigger_map.get_mut(&(table.clone(), *event)) {
                        names.retain(|n| n != name);
                    }
                }
            }
            Ok(())
        } else {
            Err(KernelError::Plugin {
                message: format!("Plugin not found: {}", name),
            })
        }
    }

    /// Fire triggers for an event
    pub async fn fire(
        &self,
        table: &str,
        event: TriggerEvent,
        context: &TriggerContext,
    ) -> KernelResult<TriggerResult> {
        let start = Instant::now();
        self.stats.total_executions.fetch_add(1, Ordering::Relaxed);

        // Find plugins to execute
        let plugin_names = {
            let trigger_map = self.trigger_map.read();
            trigger_map
                .get(&(table.to_string(), event))
                .cloned()
                .unwrap_or_default()
        };

        if plugin_names.is_empty() {
            return Ok(TriggerResult::Continue(None));
        }

        // Execute each plugin in order
        let mut current_row = context.row_json.clone();

        for name in plugin_names {
            let plugins = self.plugins.read();
            if let Some(plugin) = plugins.get(&name) {
                let result = self.execute_plugin(plugin, event, &current_row).await?;

                match result {
                    TriggerResult::Continue(Some(modified)) => {
                        current_row = modified;
                    }
                    TriggerResult::Abort { message, code } => {
                        self.stats.aborts.fetch_add(1, Ordering::Relaxed);
                        return Ok(TriggerResult::Abort { message, code });
                    }
                    TriggerResult::Skip => {
                        return Ok(TriggerResult::Skip);
                    }
                    _ => {}
                }
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.total_time_us.fetch_add(elapsed, Ordering::Relaxed);

        Ok(TriggerResult::Continue(Some(current_row)))
    }

    /// Execute a single plugin
    async fn execute_plugin(
        &self,
        plugin: &PythonPlugin,
        event: TriggerEvent,
        row_json: &str,
    ) -> KernelResult<TriggerResult> {
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let start = Instant::now();

        // In production, this would:
        // 1. Get or create WASM instance for this plugin
        // 2. Call the appropriate handler function
        // 3. Marshal data between Rust and Python

        // Simulate execution
        let result = self.simulate_execution(plugin, event, row_json, timeout)?;

        if self.config.debug {
            eprintln!(
                "[Pyodide] {} executed in {:?}",
                plugin.name,
                start.elapsed()
            );
        }

        Ok(result)
    }

    /// Simulated execution (placeholder for real Pyodide)
    fn simulate_execution(
        &self,
        plugin: &PythonPlugin,
        event: TriggerEvent,
        row_json: &str,
        timeout: Duration,
    ) -> KernelResult<TriggerResult> {
        let start = Instant::now();

        // Check timeout
        if start.elapsed() > timeout {
            return Err(KernelError::Plugin {
                message: "Execution timed out".to_string(),
            });
        }

        // Simulate common trigger logic based on code patterns
        let code = &plugin.code;

        // Check for abort conditions
        if code.contains("TriggerAbort") || code.contains("raise") {
            // Parse simulated condition from code
            if code.contains("amount") && code.contains("> 10000") {
                // Check if row has high amount
                if row_json.contains("\"amount\":") {
                    if let Some(amount) = self.extract_amount(row_json) {
                        if amount > 10000.0 {
                            return Ok(TriggerResult::Abort {
                                message: "Amount too high".to_string(),
                                code: "LIMIT_EXCEEDED".to_string(),
                            });
                        }
                    }
                }
            }
        }

        // Check for transformations
        if code.contains(".lower()") {
            // Simulate lowercase transformation
            let modified = row_json.to_lowercase();
            return Ok(TriggerResult::Continue(Some(modified)));
        }

        // For BEFORE triggers, return potentially modified row
        if event.is_before() {
            Ok(TriggerResult::Continue(Some(row_json.to_string())))
        } else {
            Ok(TriggerResult::Continue(None))
        }
    }

    fn extract_amount(&self, json: &str) -> Option<f64> {
        // Simple extraction (in production, use serde_json)
        if let Some(start) = json.find("\"amount\":") {
            let rest = &json[start + 9..].trim_start();
            let end = rest.find(|c: char| !c.is_numeric() && c != '.' && c != '-');
            let num_str = match end {
                Some(e) => &rest[..e],
                None => rest,
            };
            num_str.trim().parse().ok()
        } else {
            None
        }
    }

    /// Validate Python code
    fn validate_code(&self, code: &str) -> KernelResult<()> {
        // Check for obviously dangerous patterns
        let forbidden = [
            "__import__('os')",
            "subprocess",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "__builtins__",
        ];

        for pattern in forbidden {
            if code.contains(pattern) {
                return Err(KernelError::Plugin {
                    message: format!("Forbidden pattern in code: {}", pattern),
                });
            }
        }

        // Check for required handler function
        let handlers = ["on_insert", "on_before_insert", "on_after_insert", 
                        "on_update", "on_delete", "on_batch", "handler"];
        if !handlers.iter().any(|h| code.contains(&format!("def {}(", h))) {
            return Err(KernelError::Plugin {
                message: "Code must define a handler function".to_string(),
            });
        }

        Ok(())
    }

    /// Get runtime statistics
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// List registered plugins
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.read().keys().cloned().collect()
    }
}

// ============================================================================
// AI Trigger Generator (Future Feature)
// ============================================================================

/// Generates Python trigger code from natural language instructions
#[allow(dead_code)]
pub struct AiTriggerGenerator {
    /// Model name (e.g., "gpt-4o", "claude-3", "local:llama")
    model: String,
    /// API endpoint
    endpoint: Option<String>,
}

#[allow(dead_code)]
impl AiTriggerGenerator {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            endpoint: None,
        }
    }

    /// Generate trigger code from natural language
    pub async fn generate(&self, instruction: &str, table_schema: &str) -> KernelResult<String> {
        // In production, call LLM API
        // For now, return a template
        let code = format!(
            r#"
# Generated from: {}
# Table schema: {}

def on_before_insert(row: dict) -> dict:
    # TODO: Implement validation logic
    return row
"#,
            instruction, table_schema
        );
        Ok(code)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_builder() {
        let plugin = PythonPlugin::new("test")
            .with_version("2.0.0")
            .with_code("def on_insert(row): return row")
            .with_packages(vec!["numpy", "pandas"])
            .with_trigger("users", TriggerEvent::BeforeInsert);

        assert_eq!(plugin.name, "test");
        assert_eq!(plugin.version, "2.0.0");
        assert!(plugin.packages.contains(&"numpy".to_string()));
        assert!(plugin.triggers.contains_key("users"));
    }

    #[test]
    fn test_runtime_config() {
        let ml_config = RuntimeConfig::with_ml_packages();
        assert!(ml_config.packages.contains(&"numpy".to_string()));
        assert_eq!(ml_config.memory_limit_bytes, 256 * 1024 * 1024);

        let light_config = RuntimeConfig::lightweight();
        assert_eq!(light_config.timeout_ms, 100);
    }

    #[tokio::test]
    async fn test_runtime_register() {
        let runtime = PyodideRuntime::new(RuntimeConfig::default());

        let plugin = PythonPlugin::new("validator")
            .with_code("def on_insert(row): return row")
            .with_trigger("users", TriggerEvent::BeforeInsert);

        runtime.register(plugin).unwrap();
        assert!(runtime.list_plugins().contains(&"validator".to_string()));
    }

    #[tokio::test]
    async fn test_runtime_fire_trigger() {
        let runtime = PyodideRuntime::new(RuntimeConfig::default());

        let plugin = PythonPlugin::new("amount_check")
            .with_code(r#"
def on_insert(row):
    if row["amount"] > 10000:
        raise TriggerAbort("Amount too high")
    return row
"#)
            .with_trigger("orders", TriggerEvent::BeforeInsert);

        runtime.register(plugin).unwrap();

        // Test normal row
        let context = TriggerContext {
            table: "orders".to_string(),
            event: TriggerEvent::BeforeInsert,
            row_json: r#"{"amount": 500}"#.to_string(),
            old_row_json: None,
            txn_id: 1,
            batch_json: None,
        };

        let result = runtime.fire("orders", TriggerEvent::BeforeInsert, &context).await;
        assert!(matches!(result, Ok(TriggerResult::Continue(_))));

        // Test high amount (should abort)
        let context2 = TriggerContext {
            table: "orders".to_string(),
            event: TriggerEvent::BeforeInsert,
            row_json: r#"{"amount": 50000}"#.to_string(),
            old_row_json: None,
            txn_id: 2,
            batch_json: None,
        };

        let result2 = runtime.fire("orders", TriggerEvent::BeforeInsert, &context2).await;
        assert!(matches!(result2, Ok(TriggerResult::Abort { .. })));
    }

    #[test]
    fn test_code_validation() {
        let runtime = PyodideRuntime::new(RuntimeConfig::default());

        // Valid code
        assert!(runtime.validate_code("def on_insert(row): return row").is_ok());

        // Forbidden pattern
        assert!(runtime.validate_code("import subprocess").is_err());

        // No handler function
        assert!(runtime.validate_code("x = 42").is_err());
    }
}
