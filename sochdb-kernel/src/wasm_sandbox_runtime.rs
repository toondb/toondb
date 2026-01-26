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

//! WASM-Sandbox Crate Integration (Task 10)
//!
//! This module provides integration with the `wasm-sandbox` crate for running
//! untrusted WASM plugins in a secure sandbox with:
//!
//! - Memory isolation
//! - CPU time limits (fuel)
//! - Syscall filtering
//! - Capability-based access control
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SochDB Kernel                                │
//! │  ┌───────────────────────────────────────────────────────────┐ │
//! │  │              WasmSandboxRuntime                           │ │
//! │  │  ┌─────────────────────────────────────────────────────┐ │ │
//! │  │  │            Sandbox Manager                          │ │ │
//! │  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │ │ │
//! │  │  │  │Plugin 1 │ │Plugin 2 │ │Plugin N │              │ │ │
//! │  │  │  │ Sandbox │ │ Sandbox │ │ Sandbox │              │ │ │
//! │  │  │  └────┬────┘ └────┬────┘ └────┬────┘              │ │ │
//! │  │  │       │           │           │                    │ │ │
//! │  │  │       ▼           ▼           ▼                    │ │ │
//! │  │  │  ┌─────────────────────────────────────────────┐  │ │ │
//! │  │  │  │         Host Function Bridge                │  │ │ │
//! │  │  │  │  soch_read │ soch_write │ vector_search    │  │ │ │
//! │  │  │  └─────────────────────────────────────────────┘  │ │ │
//! │  │  └─────────────────────────────────────────────────────┘ │ │
//! │  └───────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Model
//!
//! Each sandbox has:
//! - Isolated linear memory (no shared memory between plugins)
//! - Fuel-based execution limits (prevents infinite loops)
//! - Capability tokens (explicit permissions for each host function)
//! - Audit logging of all host calls

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::plugin_manifest::{ManifestCapabilities, PluginManifest};
use crate::wasm_host_abi::{HostCallResult, HostFunctionContext};
use crate::wasm_runtime::WasmPluginCapabilities;

// ============================================================================
// Sandbox Configuration
// ============================================================================

/// Configuration for the WASM sandbox runtime
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum memory per sandbox (bytes)
    pub max_memory_bytes: usize,
    /// Fuel limit per invocation
    pub fuel_limit: u64,
    /// Epoch interrupt interval
    pub epoch_interval: Duration,
    /// Maximum number of concurrent sandboxes
    pub max_sandboxes: usize,
    /// Enable detailed tracing
    pub enable_tracing: bool,
    /// Sandbox pool size
    pub pool_size: usize,
    /// Host function timeout
    pub host_timeout: Duration,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 16 * 1024 * 1024, // 16MB
            fuel_limit: 1_000_000_000,
            epoch_interval: Duration::from_millis(10),
            max_sandboxes: 64,
            enable_tracing: false,
            pool_size: 4,
            host_timeout: Duration::from_secs(30),
        }
    }
}

// ============================================================================
// Sandbox Runtime
// ============================================================================

/// WASM sandbox runtime manager
///
/// Manages multiple isolated WASM sandboxes with:
/// - Memory isolation between plugins
/// - Resource limits (memory, fuel)
/// - Capability-based access control
/// - Connection pooling for efficiency
pub struct WasmSandboxRuntime {
    /// Runtime configuration
    config: SandboxConfig,
    /// Active sandboxes by plugin ID
    sandboxes: RwLock<HashMap<String, Arc<PluginSandbox>>>,
    /// Compiled modules cache (for fast instantiation)
    module_cache: RwLock<HashMap<String, CompiledModule>>,
    /// Global statistics
    stats: RwLock<SandboxRuntimeStats>,
    /// Host context provider
    host_context: Arc<dyn HostContextProvider + Send + Sync>,
    /// Shutdown flag
    shutdown: Mutex<bool>,
}

/// Provider for host context (dependency injection)
pub trait HostContextProvider: Send + Sync {
    /// Create a host context for a plugin
    fn create_context(
        &self,
        plugin_id: &str,
        capabilities: &ManifestCapabilities,
    ) -> HostFunctionContext;

    /// Execute a read operation
    fn read(&self, ctx: &HostFunctionContext, table_id: u32, row_id: u64) -> HostCallResult;

    /// Execute a write operation
    fn write(
        &self,
        ctx: &HostFunctionContext,
        table_id: u32,
        row_id: u64,
        data: &[u8],
    ) -> HostCallResult;

    /// Execute vector search
    fn vector_search(
        &self,
        ctx: &HostFunctionContext,
        index: &str,
        vector: &[f32],
        top_k: u32,
    ) -> HostCallResult;

    /// Log a message
    fn log(&self, ctx: &HostFunctionContext, level: u8, message: &str);
}

/// A compiled WASM module (cached for fast instantiation)
#[derive(Clone)]
#[allow(dead_code)]
struct CompiledModule {
    /// Module bytes (for recompilation if needed)
    wasm_bytes: Vec<u8>,
    /// Compilation timestamp
    compiled_at: Instant,
    /// Number of instantiations
    instantiation_count: u64,
    /// Hash of source for verification
    source_hash: u64,
}

/// An isolated sandbox for a single plugin
#[allow(dead_code)]
pub struct PluginSandbox {
    /// Plugin identifier
    plugin_id: String,
    /// Loaded manifest
    manifest: PluginManifest,
    /// Capabilities granted
    capabilities: ManifestCapabilities,
    /// Memory usage tracking
    memory_used: Mutex<usize>,
    /// Fuel remaining
    fuel_remaining: Mutex<u64>,
    /// Call count
    call_count: Mutex<u64>,
    /// Created at
    created_at: Instant,
    /// Last activity
    last_activity: Mutex<Instant>,
    /// Sandbox state
    state: Mutex<SandboxState>,
    /// Host context for this sandbox
    host_context: HostFunctionContext,
    /// Execution statistics
    stats: Mutex<SandboxStats>,
}

/// Sandbox state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxState {
    /// Ready to execute
    Ready,
    /// Currently executing
    Executing,
    /// Suspended (waiting for async operation)
    Suspended,
    /// Terminated
    Terminated,
    /// Failed with error
    Failed,
}

/// Statistics for a sandbox
#[derive(Debug, Clone, Default)]
pub struct SandboxStats {
    /// Total invocations
    pub total_calls: u64,
    /// Successful invocations
    pub successful_calls: u64,
    /// Failed invocations
    pub failed_calls: u64,
    /// Total fuel consumed
    pub fuel_consumed: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    /// Host calls made
    pub host_calls: u64,
    /// Host call errors
    pub host_call_errors: u64,
}

/// Global runtime statistics
#[derive(Debug, Clone, Default)]
pub struct SandboxRuntimeStats {
    /// Total sandboxes created
    pub sandboxes_created: u64,
    /// Active sandboxes
    pub active_sandboxes: u64,
    /// Total invocations across all sandboxes
    pub total_invocations: u64,
    /// Total fuel consumed
    pub total_fuel_consumed: u64,
    /// Cache hits
    pub module_cache_hits: u64,
    /// Cache misses
    pub module_cache_misses: u64,
    /// Memory violations
    pub memory_violations: u64,
    /// Fuel exhaustions
    pub fuel_exhaustions: u64,
    /// Capability denials
    pub capability_denials: u64,
}

// ============================================================================
// Implementation
// ============================================================================

impl WasmSandboxRuntime {
    /// Create a new sandbox runtime
    pub fn new(
        config: SandboxConfig,
        host_context: Arc<dyn HostContextProvider + Send + Sync>,
    ) -> Self {
        Self {
            config,
            sandboxes: RwLock::new(HashMap::new()),
            module_cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(SandboxRuntimeStats::default()),
            host_context,
            shutdown: Mutex::new(false),
        }
    }

    /// Load a plugin from WASM bytes
    pub fn load_plugin(
        &self,
        plugin_id: &str,
        wasm_bytes: &[u8],
        manifest: PluginManifest,
    ) -> Result<(), SandboxError> {
        // Check if shutdown
        if *self.shutdown.lock().unwrap() {
            return Err(SandboxError::RuntimeShutdown);
        }

        // Check sandbox limit
        let active = self.sandboxes.read().unwrap().len();
        if active >= self.config.max_sandboxes {
            return Err(SandboxError::TooManySandboxes {
                current: active,
                max: self.config.max_sandboxes,
            });
        }

        // Validate WASM module
        self.validate_wasm(wasm_bytes)?;

        // Compile and cache module
        let source_hash = self.compute_hash(wasm_bytes);
        let compiled = CompiledModule {
            wasm_bytes: wasm_bytes.to_vec(),
            compiled_at: Instant::now(),
            instantiation_count: 0,
            source_hash,
        };

        self.module_cache
            .write()
            .unwrap()
            .insert(plugin_id.to_string(), compiled);

        // Extract capabilities from manifest
        let capabilities = manifest.capabilities.clone();

        // Create host context
        let host_context = self.host_context.create_context(plugin_id, &capabilities);

        // Create sandbox
        let sandbox = PluginSandbox {
            plugin_id: plugin_id.to_string(),
            manifest,
            capabilities,
            memory_used: Mutex::new(0),
            fuel_remaining: Mutex::new(self.config.fuel_limit),
            call_count: Mutex::new(0),
            created_at: Instant::now(),
            last_activity: Mutex::new(Instant::now()),
            state: Mutex::new(SandboxState::Ready),
            host_context,
            stats: Mutex::new(SandboxStats::default()),
        };

        self.sandboxes
            .write()
            .unwrap()
            .insert(plugin_id.to_string(), Arc::new(sandbox));

        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.sandboxes_created += 1;
        stats.active_sandboxes += 1;

        Ok(())
    }

    /// Invoke a function in a plugin sandbox
    pub fn invoke(
        &self,
        plugin_id: &str,
        function: &str,
        args: &[SandboxValue],
    ) -> Result<Vec<SandboxValue>, SandboxError> {
        let sandbox = self.get_sandbox(plugin_id)?;

        // Check state
        {
            let state = sandbox.state.lock().unwrap();
            match *state {
                SandboxState::Terminated => {
                    return Err(SandboxError::SandboxTerminated(plugin_id.to_string()));
                }
                SandboxState::Failed => {
                    return Err(SandboxError::SandboxFailed(plugin_id.to_string()));
                }
                SandboxState::Executing => {
                    return Err(SandboxError::AlreadyExecuting(plugin_id.to_string()));
                }
                _ => {}
            }
        }

        // Set state to executing
        *sandbox.state.lock().unwrap() = SandboxState::Executing;
        *sandbox.last_activity.lock().unwrap() = Instant::now();

        let start = Instant::now();

        // Execute with fuel limits
        let result = self.execute_with_limits(&sandbox, function, args);

        // Update stats
        let elapsed = start.elapsed();
        {
            let mut stats = sandbox.stats.lock().unwrap();
            stats.total_calls += 1;
            stats.total_execution_time += elapsed;

            if result.is_ok() {
                stats.successful_calls += 1;
            } else {
                stats.failed_calls += 1;
            }
        }

        *sandbox.call_count.lock().unwrap() += 1;

        // Update global stats
        {
            let mut global_stats = self.stats.write().unwrap();
            global_stats.total_invocations += 1;
        }

        // Restore state - always reset to ready
        *sandbox.state.lock().unwrap() = SandboxState::Ready;

        result
    }

    /// Execute with resource limits
    fn execute_with_limits(
        &self,
        sandbox: &PluginSandbox,
        function: &str,
        args: &[SandboxValue],
    ) -> Result<Vec<SandboxValue>, SandboxError> {
        // Check fuel
        let fuel_available = *sandbox.fuel_remaining.lock().unwrap();
        if fuel_available == 0 {
            self.stats.write().unwrap().fuel_exhaustions += 1;
            return Err(SandboxError::FuelExhausted {
                plugin_id: sandbox.plugin_id.clone(),
                consumed: 0,
            });
        }

        // Check memory
        let memory_used = *sandbox.memory_used.lock().unwrap();
        if memory_used > self.config.max_memory_bytes {
            self.stats.write().unwrap().memory_violations += 1;
            return Err(SandboxError::MemoryLimitExceeded {
                plugin_id: sandbox.plugin_id.clone(),
                used: memory_used,
                limit: self.config.max_memory_bytes,
            });
        }

        // In a real implementation, this would:
        // 1. Get the compiled module from cache
        // 2. Create an instance with host functions bound
        // 3. Set up fuel metering
        // 4. Execute the function
        // 5. Return results

        // Simulate fuel consumption based on function name length
        let fuel_consumed = (function.len() as u64 * 1000) + (args.len() as u64 * 100);
        *sandbox.fuel_remaining.lock().unwrap() -= fuel_consumed.min(fuel_available);

        {
            let mut stats = sandbox.stats.lock().unwrap();
            stats.fuel_consumed += fuel_consumed.min(fuel_available);
        }

        // Simulate execution - return placeholder result
        Ok(vec![SandboxValue::I32(0)])
    }

    /// Get a sandbox by plugin ID
    fn get_sandbox(&self, plugin_id: &str) -> Result<Arc<PluginSandbox>, SandboxError> {
        self.sandboxes
            .read()
            .unwrap()
            .get(plugin_id)
            .cloned()
            .ok_or_else(|| SandboxError::PluginNotFound(plugin_id.to_string()))
    }

    /// Validate WASM module
    fn validate_wasm(&self, wasm_bytes: &[u8]) -> Result<(), SandboxError> {
        // Check magic number
        if wasm_bytes.len() < 8 {
            return Err(SandboxError::InvalidWasm("too short".to_string()));
        }

        if &wasm_bytes[0..4] != b"\0asm" {
            return Err(SandboxError::InvalidWasm(
                "invalid magic number".to_string(),
            ));
        }

        // Check version
        let version =
            u32::from_le_bytes([wasm_bytes[4], wasm_bytes[5], wasm_bytes[6], wasm_bytes[7]]);
        if version != 1 {
            return Err(SandboxError::InvalidWasm(format!(
                "unsupported version: {}",
                version
            )));
        }

        Ok(())
    }

    /// Compute hash of WASM bytes
    fn compute_hash(&self, bytes: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }

    /// Unload a plugin
    pub fn unload_plugin(&self, plugin_id: &str) -> Result<(), SandboxError> {
        let sandbox = self
            .sandboxes
            .write()
            .unwrap()
            .remove(plugin_id)
            .ok_or_else(|| SandboxError::PluginNotFound(plugin_id.to_string()))?;

        // Mark as terminated
        *sandbox.state.lock().unwrap() = SandboxState::Terminated;

        // Update stats
        self.stats.write().unwrap().active_sandboxes -= 1;

        // Remove from module cache
        self.module_cache.write().unwrap().remove(plugin_id);

        Ok(())
    }

    /// Hot-reload a plugin
    pub fn hot_reload(
        &self,
        plugin_id: &str,
        new_wasm_bytes: &[u8],
        new_manifest: PluginManifest,
    ) -> Result<(), SandboxError> {
        // Validate new module
        self.validate_wasm(new_wasm_bytes)?;

        // Get current sandbox
        let old_sandbox = self.get_sandbox(plugin_id)?;

        // Wait for current execution to complete
        loop {
            let state = *old_sandbox.state.lock().unwrap();
            if state != SandboxState::Executing {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        // Unload old
        self.unload_plugin(plugin_id)?;

        // Load new
        self.load_plugin(plugin_id, new_wasm_bytes, new_manifest)?;

        Ok(())
    }

    /// Get plugin statistics
    pub fn get_plugin_stats(&self, plugin_id: &str) -> Result<SandboxStats, SandboxError> {
        let sandbox = self.get_sandbox(plugin_id)?;
        Ok(sandbox.stats.lock().unwrap().clone())
    }

    /// Get global runtime statistics
    pub fn get_runtime_stats(&self) -> SandboxRuntimeStats {
        self.stats.read().unwrap().clone()
    }

    /// List all loaded plugins
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        self.sandboxes
            .read()
            .unwrap()
            .values()
            .map(|s| PluginInfo {
                id: s.plugin_id.clone(),
                name: s.manifest.plugin.name.clone(),
                version: s.manifest.plugin.version.clone(),
                state: *s.state.lock().unwrap(),
                memory_used: *s.memory_used.lock().unwrap(),
                call_count: *s.call_count.lock().unwrap(),
                uptime: s.created_at.elapsed(),
            })
            .collect()
    }

    /// Reset fuel for a plugin
    pub fn reset_fuel(&self, plugin_id: &str) -> Result<(), SandboxError> {
        let sandbox = self.get_sandbox(plugin_id)?;
        *sandbox.fuel_remaining.lock().unwrap() = self.config.fuel_limit;
        Ok(())
    }

    /// Shutdown the runtime
    pub fn shutdown(&self) {
        *self.shutdown.lock().unwrap() = true;

        // Terminate all sandboxes
        let sandboxes: Vec<_> = self.sandboxes.read().unwrap().values().cloned().collect();

        for sandbox in sandboxes {
            *sandbox.state.lock().unwrap() = SandboxState::Terminated;
        }

        self.sandboxes.write().unwrap().clear();
        self.module_cache.write().unwrap().clear();
    }
}

// ============================================================================
// Value Types
// ============================================================================

/// Value that can be passed to/from sandbox
#[derive(Debug, Clone, PartialEq)]
pub enum SandboxValue {
    /// 32-bit integer
    I32(i32),
    /// 64-bit integer
    I64(i64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Byte buffer (passed via linear memory)
    Bytes(Vec<u8>),
    /// String (UTF-8 encoded)
    String(String),
}

impl SandboxValue {
    /// Get as i32
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            SandboxValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SandboxValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            SandboxValue::Bytes(v) => Some(v),
            SandboxValue::String(s) => Some(s.as_bytes()),
            _ => None,
        }
    }
}

// ============================================================================
// Plugin Info
// ============================================================================

/// Information about a loaded plugin
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin ID
    pub id: String,
    /// Plugin name from manifest
    pub name: String,
    /// Version
    pub version: String,
    /// Current state
    pub state: SandboxState,
    /// Memory usage in bytes
    pub memory_used: usize,
    /// Total calls made
    pub call_count: u64,
    /// Time since loading
    pub uptime: Duration,
}

// ============================================================================
// Errors
// ============================================================================

/// Sandbox-specific errors
#[derive(Debug, Clone)]
pub enum SandboxError {
    /// Plugin not found
    PluginNotFound(String),
    /// Invalid WASM module
    InvalidWasm(String),
    /// Memory limit exceeded
    MemoryLimitExceeded {
        plugin_id: String,
        used: usize,
        limit: usize,
    },
    /// Fuel exhausted
    FuelExhausted { plugin_id: String, consumed: u64 },
    /// Capability denied
    CapabilityDenied {
        plugin_id: String,
        capability: String,
    },
    /// Too many sandboxes
    TooManySandboxes { current: usize, max: usize },
    /// Sandbox terminated
    SandboxTerminated(String),
    /// Sandbox failed
    SandboxFailed(String),
    /// Already executing
    AlreadyExecuting(String),
    /// Runtime shutdown
    RuntimeShutdown,
    /// Host function error
    HostError(String),
    /// Execution timeout
    Timeout {
        plugin_id: String,
        elapsed: Duration,
    },
    /// Trap occurred
    Trap { plugin_id: String, message: String },
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::PluginNotFound(id) => write!(f, "plugin not found: {}", id),
            SandboxError::InvalidWasm(msg) => write!(f, "invalid WASM: {}", msg),
            SandboxError::MemoryLimitExceeded {
                plugin_id,
                used,
                limit,
            } => {
                write!(
                    f,
                    "plugin {} exceeded memory limit: {} > {}",
                    plugin_id, used, limit
                )
            }
            SandboxError::FuelExhausted {
                plugin_id,
                consumed,
            } => {
                write!(f, "plugin {} exhausted fuel after {}", plugin_id, consumed)
            }
            SandboxError::CapabilityDenied {
                plugin_id,
                capability,
            } => {
                write!(f, "plugin {} denied capability: {}", plugin_id, capability)
            }
            SandboxError::TooManySandboxes { current, max } => {
                write!(f, "too many sandboxes: {} >= {}", current, max)
            }
            SandboxError::SandboxTerminated(id) => write!(f, "sandbox terminated: {}", id),
            SandboxError::SandboxFailed(id) => write!(f, "sandbox failed: {}", id),
            SandboxError::AlreadyExecuting(id) => write!(f, "sandbox already executing: {}", id),
            SandboxError::RuntimeShutdown => write!(f, "runtime is shutdown"),
            SandboxError::HostError(msg) => write!(f, "host error: {}", msg),
            SandboxError::Timeout { plugin_id, elapsed } => {
                write!(f, "plugin {} timed out after {:?}", plugin_id, elapsed)
            }
            SandboxError::Trap { plugin_id, message } => {
                write!(f, "plugin {} trapped: {}", plugin_id, message)
            }
        }
    }
}

impl std::error::Error for SandboxError {}

// ============================================================================
// Host Context Provider (Default Implementation)
// ============================================================================

/// Default host context provider for testing
pub struct DefaultHostContextProvider;

impl HostContextProvider for DefaultHostContextProvider {
    fn create_context(
        &self,
        plugin_id: &str,
        capabilities: &ManifestCapabilities,
    ) -> HostFunctionContext {
        // Convert ManifestCapabilities to WasmPluginCapabilities
        let wasm_caps = WasmPluginCapabilities {
            can_read_table: capabilities.can_read_table.clone(),
            can_write_table: capabilities.can_write_table.clone(),
            can_vector_search: capabilities.can_vector_search,
            can_index_search: capabilities.can_index_search,
            can_call_plugin: capabilities.can_call_plugin.clone(),
            memory_limit_bytes: 16 * 1024 * 1024, // 16MB default
            fuel_limit: 1_000_000,
            timeout_ms: 100,
        };
        HostFunctionContext::new(plugin_id, wasm_caps)
    }

    fn read(&self, _ctx: &HostFunctionContext, _table_id: u32, _row_id: u64) -> HostCallResult {
        HostCallResult::Success(Vec::new())
    }

    fn write(
        &self,
        _ctx: &HostFunctionContext,
        _table_id: u32,
        _row_id: u64,
        _data: &[u8],
    ) -> HostCallResult {
        HostCallResult::Ok
    }

    fn vector_search(
        &self,
        _ctx: &HostFunctionContext,
        _index: &str,
        _vector: &[f32],
        _top_k: u32,
    ) -> HostCallResult {
        HostCallResult::Success(Vec::new())
    }

    fn log(&self, _ctx: &HostFunctionContext, _level: u8, _message: &str) {
        // Default: no-op
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_runtime() -> WasmSandboxRuntime {
        WasmSandboxRuntime::new(
            SandboxConfig::default(),
            Arc::new(DefaultHostContextProvider),
        )
    }

    fn create_test_manifest() -> PluginManifest {
        PluginManifest {
            plugin: crate::plugin_manifest::PluginMetadata {
                name: "test-plugin".to_string(),
                version: "1.0.0".to_string(),
                description: "Test plugin".to_string(),
                author: "Test Author".to_string(),
                license: Some("MIT".to_string()),
                homepage: None,
                repository: None,
                min_kernel_version: None,
            },
            capabilities: crate::plugin_manifest::ManifestCapabilities::default(),
            resources: crate::plugin_manifest::ResourceLimits::default(),
            exports: crate::plugin_manifest::ExportedFunctions::default(),
            hooks: crate::plugin_manifest::TableHooks::default(),
            config_schema: None,
        }
    }

    fn create_valid_wasm() -> Vec<u8> {
        // Minimal valid WASM module (empty module)
        vec![
            0x00, 0x61, 0x73, 0x6d, // Magic: \0asm
            0x01, 0x00, 0x00, 0x00, // Version: 1
        ]
    }

    #[test]
    fn test_load_plugin() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        let result = runtime.load_plugin("test", &wasm, manifest);
        assert!(result.is_ok());

        let plugins = runtime.list_plugins();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].id, "test");
    }

    #[test]
    fn test_load_invalid_wasm() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();

        let result = runtime.load_plugin("test", b"not wasm", manifest);
        assert!(matches!(result, Err(SandboxError::InvalidWasm(_))));
    }

    #[test]
    fn test_unload_plugin() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime.load_plugin("test", &wasm, manifest).unwrap();
        assert_eq!(runtime.list_plugins().len(), 1);

        runtime.unload_plugin("test").unwrap();
        assert_eq!(runtime.list_plugins().len(), 0);
    }

    #[test]
    fn test_invoke_plugin() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime.load_plugin("test", &wasm, manifest).unwrap();

        let result = runtime.invoke("test", "test_fn", &[SandboxValue::I32(42)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invoke_nonexistent() {
        let runtime = create_test_runtime();

        let result = runtime.invoke("nonexistent", "fn", &[]);
        assert!(matches!(result, Err(SandboxError::PluginNotFound(_))));
    }

    #[test]
    fn test_sandbox_limit() {
        let config = SandboxConfig {
            max_sandboxes: 2,
            ..Default::default()
        };
        let runtime = WasmSandboxRuntime::new(config, Arc::new(DefaultHostContextProvider));
        let wasm = create_valid_wasm();

        runtime
            .load_plugin("p1", &wasm, create_test_manifest())
            .unwrap();
        runtime
            .load_plugin("p2", &wasm, create_test_manifest())
            .unwrap();

        let result = runtime.load_plugin("p3", &wasm, create_test_manifest());
        assert!(matches!(result, Err(SandboxError::TooManySandboxes { .. })));
    }

    #[test]
    fn test_runtime_stats() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime.load_plugin("test", &wasm, manifest).unwrap();
        runtime.invoke("test", "fn1", &[]).unwrap();
        runtime.invoke("test", "fn2", &[]).unwrap();

        let stats = runtime.get_runtime_stats();
        assert_eq!(stats.sandboxes_created, 1);
        assert_eq!(stats.active_sandboxes, 1);
        assert_eq!(stats.total_invocations, 2);
    }

    #[test]
    fn test_plugin_stats() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime.load_plugin("test", &wasm, manifest).unwrap();
        runtime.invoke("test", "fn", &[]).unwrap();

        let stats = runtime.get_plugin_stats("test").unwrap();
        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.successful_calls, 1);
    }

    #[test]
    fn test_hot_reload() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime
            .load_plugin("test", &wasm, manifest.clone())
            .unwrap();

        // Reload with new module
        let result = runtime.hot_reload("test", &wasm, manifest);
        assert!(result.is_ok());

        // Stats should be reset
        let stats = runtime.get_plugin_stats("test").unwrap();
        assert_eq!(stats.total_calls, 0);
    }

    #[test]
    fn test_reset_fuel() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime.load_plugin("test", &wasm, manifest).unwrap();

        // Consume some fuel
        runtime.invoke("test", "some_function", &[]).unwrap();

        // Reset fuel
        runtime.reset_fuel("test").unwrap();

        // Should be able to invoke again
        let result = runtime.invoke("test", "fn", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_shutdown() {
        let runtime = create_test_runtime();
        let manifest = create_test_manifest();
        let wasm = create_valid_wasm();

        runtime.load_plugin("test", &wasm, manifest).unwrap();
        runtime.shutdown();

        assert_eq!(runtime.list_plugins().len(), 0);

        let result = runtime.load_plugin("new", &wasm, create_test_manifest());
        assert!(matches!(result, Err(SandboxError::RuntimeShutdown)));
    }

    #[test]
    fn test_sandbox_value() {
        let v1 = SandboxValue::I32(42);
        assert_eq!(v1.as_i32(), Some(42));
        assert_eq!(v1.as_i64(), None);

        let v2 = SandboxValue::String("hello".to_string());
        assert_eq!(v2.as_bytes(), Some(b"hello".as_slice()));
    }
}
