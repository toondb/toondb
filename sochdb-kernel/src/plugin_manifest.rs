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

//! Plugin Manifest Schema
//!
//! Defines the TOML-based manifest format for WASM plugins.
//!
//! ## Manifest Format
//!
//! ```toml
//! [plugin]
//! name = "my-analytics-plugin"
//! version = "1.0.0"
//! description = "Analytics plugin for aggregation"
//! author = "SochDB Team"
//! license = "MIT"
//!
//! [capabilities]
//! can_read_table = ["analytics_*", "metrics"]
//! can_write_table = ["analytics_results"]
//! can_vector_search = false
//! can_index_search = true
//! can_call_plugin = ["logging-plugin"]
//!
//! [resources]
//! memory_limit_mb = 64
//! fuel_limit = 10000000
//! timeout_ms = 1000
//!
//! [exports]
//! functions = ["on_insert", "on_update", "aggregate"]
//!
//! [hooks]
//! before_insert = ["validate_row"]
//! after_insert = ["index_row", "emit_metric"]
//! ```

use crate::error::{KernelError, KernelResult};
use crate::wasm_runtime::WasmPluginCapabilities;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Plugin Manifest
// ============================================================================

/// Plugin manifest defining metadata and capabilities
#[derive(Debug, Clone)]
pub struct PluginManifest {
    /// Plugin metadata
    pub plugin: PluginMetadata,
    /// Granted capabilities
    pub capabilities: ManifestCapabilities,
    /// Resource limits
    pub resources: ResourceLimits,
    /// Exported functions
    pub exports: ExportedFunctions,
    /// Table hooks
    pub hooks: TableHooks,
    /// Optional configuration schema
    pub config_schema: Option<ConfigSchema>,
}

/// Plugin metadata section
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Plugin name (unique identifier)
    pub name: String,
    /// Semantic version
    pub version: String,
    /// Human-readable description
    pub description: String,
    /// Author name or organization
    pub author: String,
    /// License identifier (SPDX)
    pub license: Option<String>,
    /// Homepage URL
    pub homepage: Option<String>,
    /// Repository URL
    pub repository: Option<String>,
    /// Minimum SochDB kernel version required
    pub min_kernel_version: Option<String>,
}

/// Capability declarations
#[derive(Debug, Clone, Default)]
pub struct ManifestCapabilities {
    /// Tables the plugin can read (glob patterns)
    pub can_read_table: Vec<String>,
    /// Tables the plugin can write (glob patterns)
    pub can_write_table: Vec<String>,
    /// Can perform vector similarity search
    pub can_vector_search: bool,
    /// Can perform index lookups
    pub can_index_search: bool,
    /// Other plugins this plugin can call
    pub can_call_plugin: Vec<String>,
}

/// Resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Memory limit in megabytes
    pub memory_limit_mb: u64,
    /// Fuel limit (instruction count)
    pub fuel_limit: u64,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum concurrent instances
    pub max_instances: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_limit_mb: 16,
            fuel_limit: 1_000_000,
            timeout_ms: 100,
            max_instances: 4,
        }
    }
}

/// Exported functions from the plugin
#[derive(Debug, Clone, Default)]
pub struct ExportedFunctions {
    /// List of exported function names
    pub functions: Vec<String>,
    /// Function signatures (optional, for validation)
    pub signatures: HashMap<String, FunctionSignature>,
}

/// Function signature
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Parameter types
    pub params: Vec<WasmType>,
    /// Return types
    pub returns: Vec<WasmType>,
}

/// WASM type for signature validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmType {
    I32,
    I64,
    F32,
    F64,
    ExternRef,
}

/// Table hook bindings
#[derive(Debug, Clone, Default)]
pub struct TableHooks {
    /// Functions to call before INSERT
    pub before_insert: Vec<String>,
    /// Functions to call after INSERT
    pub after_insert: Vec<String>,
    /// Functions to call before UPDATE
    pub before_update: Vec<String>,
    /// Functions to call after UPDATE
    pub after_update: Vec<String>,
    /// Functions to call before DELETE
    pub before_delete: Vec<String>,
    /// Functions to call after DELETE
    pub after_delete: Vec<String>,
}

/// Configuration schema for plugin settings
#[derive(Debug, Clone, Default)]
pub struct ConfigSchema {
    /// Configuration fields
    pub fields: Vec<ConfigField>,
}

/// A configuration field
#[derive(Debug, Clone)]
pub struct ConfigField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: ConfigFieldType,
    /// Whether the field is required
    pub required: bool,
    /// Default value (as string)
    pub default: Option<String>,
    /// Description
    pub description: Option<String>,
}

/// Configuration field types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFieldType {
    String,
    Integer,
    Float,
    Boolean,
    StringArray,
}

// ============================================================================
// Manifest Parsing
// ============================================================================

impl PluginManifest {
    /// Parse a manifest from TOML content
    pub fn from_toml(content: &str) -> KernelResult<Self> {
        // Simple TOML-like parser (in production, use the toml crate)
        let mut manifest = Self::default();

        let mut current_section = "";
        let mut _current_subsection = "";

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Section headers
            if line.starts_with('[') && line.ends_with(']') {
                let section = &line[1..line.len() - 1];
                if section.contains('.') {
                    let parts: Vec<&str> = section.split('.').collect();
                    current_section = parts[0];
                    _current_subsection = parts[1];
                } else {
                    current_section = section;
                    _current_subsection = "";
                }
                continue;
            }

            // Key-value pairs
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                let value = value.trim_matches('"');

                match current_section {
                    "plugin" => Self::parse_plugin_field(&mut manifest.plugin, key, value),
                    "capabilities" => {
                        Self::parse_capabilities_field(&mut manifest.capabilities, key, value)
                    }
                    "resources" => Self::parse_resources_field(&mut manifest.resources, key, value),
                    "exports" => Self::parse_exports_field(&mut manifest.exports, key, value),
                    "hooks" => Self::parse_hooks_field(&mut manifest.hooks, key, value),
                    _ => {}
                }
            }
        }

        // Validate required fields
        manifest.validate()?;

        Ok(manifest)
    }

    /// Parse from a file
    pub fn from_file(path: &Path) -> KernelResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| KernelError::Plugin {
            message: format!("failed to read manifest: {}", e),
        })?;
        Self::from_toml(&content)
    }

    /// Convert to WasmPluginCapabilities
    pub fn to_capabilities(&self) -> WasmPluginCapabilities {
        WasmPluginCapabilities {
            can_read_table: self.capabilities.can_read_table.clone(),
            can_write_table: self.capabilities.can_write_table.clone(),
            can_vector_search: self.capabilities.can_vector_search,
            can_index_search: self.capabilities.can_index_search,
            can_call_plugin: self.capabilities.can_call_plugin.clone(),
            memory_limit_bytes: self.resources.memory_limit_mb * 1024 * 1024,
            fuel_limit: self.resources.fuel_limit,
            timeout_ms: self.resources.timeout_ms,
        }
    }

    /// Validate the manifest
    pub fn validate(&self) -> KernelResult<()> {
        // Name is required
        if self.plugin.name.is_empty() {
            return Err(KernelError::Plugin {
                message: "plugin name is required".to_string(),
            });
        }

        // Name must be valid identifier
        if !self
            .plugin
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(KernelError::Plugin {
                message: format!("invalid plugin name: {}", self.plugin.name),
            });
        }

        // Version is required
        if self.plugin.version.is_empty() {
            return Err(KernelError::Plugin {
                message: "plugin version is required".to_string(),
            });
        }

        // Resource limits must be reasonable
        if self.resources.memory_limit_mb > 1024 {
            return Err(KernelError::Plugin {
                message: "memory limit exceeds 1GB maximum".to_string(),
            });
        }

        if self.resources.timeout_ms > 60_000 {
            return Err(KernelError::Plugin {
                message: "timeout exceeds 60s maximum".to_string(),
            });
        }

        // All hook functions must be in exports
        let exported: std::collections::HashSet<_> = self.exports.functions.iter().collect();
        for hook in self.all_hooks() {
            if !exported.contains(&hook) {
                return Err(KernelError::Plugin {
                    message: format!("hook function '{}' not in exports", hook),
                });
            }
        }

        Ok(())
    }

    /// Get all hook function names
    fn all_hooks(&self) -> Vec<String> {
        let mut hooks = Vec::new();
        hooks.extend(self.hooks.before_insert.clone());
        hooks.extend(self.hooks.after_insert.clone());
        hooks.extend(self.hooks.before_update.clone());
        hooks.extend(self.hooks.after_update.clone());
        hooks.extend(self.hooks.before_delete.clone());
        hooks.extend(self.hooks.after_delete.clone());
        hooks
    }

    // Parsing helpers
    fn parse_plugin_field(plugin: &mut PluginMetadata, key: &str, value: &str) {
        match key {
            "name" => plugin.name = value.to_string(),
            "version" => plugin.version = value.to_string(),
            "description" => plugin.description = value.to_string(),
            "author" => plugin.author = value.to_string(),
            "license" => plugin.license = Some(value.to_string()),
            "homepage" => plugin.homepage = Some(value.to_string()),
            "repository" => plugin.repository = Some(value.to_string()),
            "min_kernel_version" => plugin.min_kernel_version = Some(value.to_string()),
            _ => {}
        }
    }

    fn parse_capabilities_field(caps: &mut ManifestCapabilities, key: &str, value: &str) {
        match key {
            "can_read_table" => caps.can_read_table = Self::parse_string_array(value),
            "can_write_table" => caps.can_write_table = Self::parse_string_array(value),
            "can_vector_search" => caps.can_vector_search = value == "true",
            "can_index_search" => caps.can_index_search = value == "true",
            "can_call_plugin" => caps.can_call_plugin = Self::parse_string_array(value),
            _ => {}
        }
    }

    fn parse_resources_field(res: &mut ResourceLimits, key: &str, value: &str) {
        match key {
            "memory_limit_mb" => res.memory_limit_mb = value.parse().unwrap_or(16),
            "fuel_limit" => res.fuel_limit = value.parse().unwrap_or(1_000_000),
            "timeout_ms" => res.timeout_ms = value.parse().unwrap_or(100),
            "max_instances" => res.max_instances = value.parse().unwrap_or(4),
            _ => {}
        }
    }

    fn parse_exports_field(exports: &mut ExportedFunctions, key: &str, value: &str) {
        if key == "functions" {
            exports.functions = Self::parse_string_array(value);
        }
    }

    fn parse_hooks_field(hooks: &mut TableHooks, key: &str, value: &str) {
        let funcs = Self::parse_string_array(value);
        match key {
            "before_insert" => hooks.before_insert = funcs,
            "after_insert" => hooks.after_insert = funcs,
            "before_update" => hooks.before_update = funcs,
            "after_update" => hooks.after_update = funcs,
            "before_delete" => hooks.before_delete = funcs,
            "after_delete" => hooks.after_delete = funcs,
            _ => {}
        }
    }

    fn parse_string_array(value: &str) -> Vec<String> {
        // Parse [a, b, c] format
        let value = value.trim();
        if value.starts_with('[') && value.ends_with(']') {
            let inner = &value[1..value.len() - 1];
            inner
                .split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                .filter(|s| !s.is_empty())
                .collect()
        } else {
            vec![value.to_string()]
        }
    }
}

impl Default for PluginManifest {
    fn default() -> Self {
        Self {
            plugin: PluginMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                author: String::new(),
                license: None,
                homepage: None,
                repository: None,
                min_kernel_version: None,
            },
            capabilities: ManifestCapabilities::default(),
            resources: ResourceLimits::default(),
            exports: ExportedFunctions::default(),
            hooks: TableHooks::default(),
            config_schema: None,
        }
    }
}

// ============================================================================
// Manifest Builder
// ============================================================================

/// Builder for creating plugin manifests programmatically
pub struct ManifestBuilder {
    manifest: PluginManifest,
}

impl ManifestBuilder {
    /// Create a new builder with required fields
    pub fn new(name: &str, version: &str) -> Self {
        let mut manifest = PluginManifest::default();
        manifest.plugin.name = name.to_string();
        manifest.plugin.version = version.to_string();
        Self { manifest }
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.manifest.plugin.description = desc.to_string();
        self
    }

    /// Set author
    pub fn author(mut self, author: &str) -> Self {
        self.manifest.plugin.author = author.to_string();
        self
    }

    /// Set license
    pub fn license(mut self, license: &str) -> Self {
        self.manifest.plugin.license = Some(license.to_string());
        self
    }

    /// Add readable table pattern
    pub fn can_read(mut self, pattern: &str) -> Self {
        self.manifest
            .capabilities
            .can_read_table
            .push(pattern.to_string());
        self
    }

    /// Add writable table pattern
    pub fn can_write(mut self, pattern: &str) -> Self {
        self.manifest
            .capabilities
            .can_write_table
            .push(pattern.to_string());
        self
    }

    /// Enable vector search
    pub fn with_vector_search(mut self) -> Self {
        self.manifest.capabilities.can_vector_search = true;
        self
    }

    /// Enable index search
    pub fn with_index_search(mut self) -> Self {
        self.manifest.capabilities.can_index_search = true;
        self
    }

    /// Set memory limit
    pub fn memory_limit_mb(mut self, mb: u64) -> Self {
        self.manifest.resources.memory_limit_mb = mb;
        self
    }

    /// Set fuel limit
    pub fn fuel_limit(mut self, fuel: u64) -> Self {
        self.manifest.resources.fuel_limit = fuel;
        self
    }

    /// Set timeout
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.manifest.resources.timeout_ms = ms;
        self
    }

    /// Add exported function
    pub fn export(mut self, func: &str) -> Self {
        self.manifest.exports.functions.push(func.to_string());
        self
    }

    /// Add before_insert hook
    pub fn before_insert(mut self, func: &str) -> Self {
        self.manifest.hooks.before_insert.push(func.to_string());
        self
    }

    /// Add after_insert hook
    pub fn after_insert(mut self, func: &str) -> Self {
        self.manifest.hooks.after_insert.push(func.to_string());
        self
    }

    /// Build the manifest
    pub fn build(self) -> KernelResult<PluginManifest> {
        self.manifest.validate()?;
        Ok(self.manifest)
    }
}

// ============================================================================
// Manifest Serialization
// ============================================================================

impl PluginManifest {
    /// Serialize to TOML format
    pub fn to_toml(&self) -> String {
        let mut out = String::new();

        // [plugin]
        out.push_str("[plugin]\n");
        out.push_str(&format!("name = \"{}\"\n", self.plugin.name));
        out.push_str(&format!("version = \"{}\"\n", self.plugin.version));
        if !self.plugin.description.is_empty() {
            out.push_str(&format!("description = \"{}\"\n", self.plugin.description));
        }
        if !self.plugin.author.is_empty() {
            out.push_str(&format!("author = \"{}\"\n", self.plugin.author));
        }
        if let Some(license) = &self.plugin.license {
            out.push_str(&format!("license = \"{}\"\n", license));
        }
        out.push('\n');

        // [capabilities]
        out.push_str("[capabilities]\n");
        if !self.capabilities.can_read_table.is_empty() {
            out.push_str(&format!(
                "can_read_table = {:?}\n",
                self.capabilities.can_read_table
            ));
        }
        if !self.capabilities.can_write_table.is_empty() {
            out.push_str(&format!(
                "can_write_table = {:?}\n",
                self.capabilities.can_write_table
            ));
        }
        out.push_str(&format!(
            "can_vector_search = {}\n",
            self.capabilities.can_vector_search
        ));
        out.push_str(&format!(
            "can_index_search = {}\n",
            self.capabilities.can_index_search
        ));
        out.push('\n');

        // [resources]
        out.push_str("[resources]\n");
        out.push_str(&format!(
            "memory_limit_mb = {}\n",
            self.resources.memory_limit_mb
        ));
        out.push_str(&format!("fuel_limit = {}\n", self.resources.fuel_limit));
        out.push_str(&format!("timeout_ms = {}\n", self.resources.timeout_ms));
        out.push('\n');

        // [exports]
        if !self.exports.functions.is_empty() {
            out.push_str("[exports]\n");
            out.push_str(&format!("functions = {:?}\n", self.exports.functions));
            out.push('\n');
        }

        // [hooks]
        if !self.hooks.before_insert.is_empty() || !self.hooks.after_insert.is_empty() {
            out.push_str("[hooks]\n");
            if !self.hooks.before_insert.is_empty() {
                out.push_str(&format!("before_insert = {:?}\n", self.hooks.before_insert));
            }
            if !self.hooks.after_insert.is_empty() {
                out.push_str(&format!("after_insert = {:?}\n", self.hooks.after_insert));
            }
            if !self.hooks.before_update.is_empty() {
                out.push_str(&format!("before_update = {:?}\n", self.hooks.before_update));
            }
            if !self.hooks.after_update.is_empty() {
                out.push_str(&format!("after_update = {:?}\n", self.hooks.after_update));
            }
        }

        out
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MANIFEST: &str = r#"
[plugin]
name = "my-analytics-plugin"
version = "1.0.0"
description = "Analytics plugin for aggregation"
author = "SochDB Team"
license = "MIT"

[capabilities]
can_read_table = ["analytics_*", "metrics"]
can_write_table = ["analytics_results"]
can_vector_search = false
can_index_search = true

[resources]
memory_limit_mb = 64
fuel_limit = 10000000
timeout_ms = 1000

[exports]
functions = ["on_insert", "aggregate"]

[hooks]
before_insert = []
after_insert = ["on_insert"]
"#;

    #[test]
    fn test_parse_manifest() {
        let manifest = PluginManifest::from_toml(SAMPLE_MANIFEST).unwrap();

        assert_eq!(manifest.plugin.name, "my-analytics-plugin");
        assert_eq!(manifest.plugin.version, "1.0.0");
        assert_eq!(manifest.plugin.author, "SochDB Team");
        assert_eq!(manifest.plugin.license, Some("MIT".to_string()));

        assert_eq!(
            manifest.capabilities.can_read_table,
            vec!["analytics_*", "metrics"]
        );
        assert_eq!(
            manifest.capabilities.can_write_table,
            vec!["analytics_results"]
        );
        assert!(!manifest.capabilities.can_vector_search);
        assert!(manifest.capabilities.can_index_search);

        assert_eq!(manifest.resources.memory_limit_mb, 64);
        assert_eq!(manifest.resources.fuel_limit, 10_000_000);
        assert_eq!(manifest.resources.timeout_ms, 1000);

        assert!(
            manifest
                .exports
                .functions
                .contains(&"on_insert".to_string())
        );
        assert!(
            manifest
                .exports
                .functions
                .contains(&"aggregate".to_string())
        );

        assert!(
            manifest
                .hooks
                .after_insert
                .contains(&"on_insert".to_string())
        );
    }

    #[test]
    fn test_manifest_validation() {
        // Missing name
        let manifest = PluginManifest::default();
        assert!(manifest.validate().is_err());

        // Invalid name
        let mut manifest = PluginManifest::default();
        manifest.plugin.name = "invalid name!".to_string();
        manifest.plugin.version = "1.0.0".to_string();
        assert!(manifest.validate().is_err());

        // Valid minimal manifest
        let mut manifest = PluginManifest::default();
        manifest.plugin.name = "valid-plugin".to_string();
        manifest.plugin.version = "1.0.0".to_string();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_manifest_builder() {
        let manifest = ManifestBuilder::new("test-plugin", "1.0.0")
            .description("A test plugin")
            .author("Test Author")
            .license("MIT")
            .can_read("users")
            .can_read("logs_*")
            .can_write("results")
            .with_vector_search()
            .memory_limit_mb(32)
            .fuel_limit(500_000)
            .export("handler")
            .build()
            .unwrap();

        assert_eq!(manifest.plugin.name, "test-plugin");
        assert!(
            manifest
                .capabilities
                .can_read_table
                .contains(&"users".to_string())
        );
        assert!(
            manifest
                .capabilities
                .can_read_table
                .contains(&"logs_*".to_string())
        );
        assert!(manifest.capabilities.can_vector_search);
        assert_eq!(manifest.resources.memory_limit_mb, 32);
    }

    #[test]
    fn test_to_capabilities() {
        let manifest = ManifestBuilder::new("test", "1.0.0")
            .can_read("table1")
            .memory_limit_mb(32)
            .fuel_limit(500_000)
            .timeout_ms(200)
            .build()
            .unwrap();

        let caps = manifest.to_capabilities();

        assert!(caps.can_read("table1"));
        assert!(!caps.can_read("other"));
        assert_eq!(caps.memory_limit_bytes, 32 * 1024 * 1024);
        assert_eq!(caps.fuel_limit, 500_000);
        assert_eq!(caps.timeout_ms, 200);
    }

    #[test]
    fn test_to_toml() {
        let manifest = ManifestBuilder::new("roundtrip-test", "2.0.0")
            .description("Test roundtrip")
            .author("Test")
            .can_read("data")
            .memory_limit_mb(16)
            .export("init")
            .build()
            .unwrap();

        let toml = manifest.to_toml();

        // Parse it back
        let parsed = PluginManifest::from_toml(&toml).unwrap();

        assert_eq!(parsed.plugin.name, "roundtrip-test");
        assert_eq!(parsed.plugin.version, "2.0.0");
        assert!(
            parsed
                .capabilities
                .can_read_table
                .contains(&"data".to_string())
        );
    }

    #[test]
    fn test_resource_limits_validation() {
        // Memory too high
        let mut manifest = PluginManifest::default();
        manifest.plugin.name = "test".to_string();
        manifest.plugin.version = "1.0.0".to_string();
        manifest.resources.memory_limit_mb = 2048;
        assert!(manifest.validate().is_err());

        // Timeout too high
        let mut manifest = PluginManifest::default();
        manifest.plugin.name = "test".to_string();
        manifest.plugin.version = "1.0.0".to_string();
        manifest.resources.timeout_ms = 120_000;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_hook_validation() {
        // Hook function not in exports
        let mut manifest = PluginManifest::default();
        manifest.plugin.name = "test".to_string();
        manifest.plugin.version = "1.0.0".to_string();
        manifest
            .hooks
            .before_insert
            .push("missing_function".to_string());
        // This should fail because missing_function is not exported
        assert!(manifest.validate().is_err());

        // Add the function to exports - now should pass
        manifest
            .exports
            .functions
            .push("missing_function".to_string());
        assert!(manifest.validate().is_ok());
    }
}
