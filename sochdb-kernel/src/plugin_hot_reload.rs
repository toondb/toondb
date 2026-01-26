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

//! Hot-Reload Without Restart
//!
//! This module implements zero-downtime plugin upgrades using atomic
//! swapping and epoch-based draining.
//!
//! ## Design
//!
//! ```text
//!                      ┌───────────────────────────┐
//!                      │    HotReloadablePlugin    │
//!                      │                           │
//!                      │  ┌─────────────────────┐  │
//!                      │  │  Arc<Current Plugin> │  │
//!                      │  └──────────┬──────────┘  │
//!                      │             │             │
//!   New Version ──────►│  ┌──────────▼──────────┐  │
//!                      │  │  prepare_upgrade()   │  │
//!                      │  └──────────┬──────────┘  │
//!                      │             │             │
//!                      │  ┌──────────▼──────────┐  │
//!                      │  │  drain_in_flight()   │  │
//!                      │  └──────────┬──────────┘  │
//!                      │             │             │
//!                      │  ┌──────────▼──────────┐  │
//!                      │  │  atomic_swap()       │  │
//!                      │  └──────────┬──────────┘  │
//!                      │             │             │
//!                      │  ┌──────────▼──────────┐  │
//!                      │  │  cleanup_old()       │  │
//!                      │  └─────────────────────┘  │
//!                      └───────────────────────────┘
//! ```
//!
//! ## Safety Properties
//!
//! 1. **No Request Drops**: In-flight calls complete on old version
//! 2. **Atomic Transition**: New calls immediately use new version
//! 3. **Memory Safety**: Old version freed only when refs drop to zero
//! 4. **Rollback**: If new version fails, old version remains active

use crate::error::{KernelError, KernelResult};
use crate::plugin_manifest::PluginManifest;
use crate::wasm_runtime::{WasmInstanceConfig, WasmPluginInstance, WasmValue};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// Epoch-Based Draining
// ============================================================================

/// Epoch counter for tracking in-flight operations
pub struct EpochTracker {
    /// Current epoch number
    epoch: AtomicU64,
    /// Reference counts per epoch (circular buffer of 8 epochs)
    epoch_refs: [AtomicUsize; 8],
}

impl Default for EpochTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochTracker {
    /// Create a new epoch tracker
    pub fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            epoch_refs: Default::default(),
        }
    }

    /// Get current epoch
    pub fn current(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Enter an epoch (increment ref count)
    pub fn enter(&self) -> EpochGuard<'_> {
        let epoch = self.current();
        let idx = (epoch % 8) as usize;
        self.epoch_refs[idx].fetch_add(1, Ordering::AcqRel);
        EpochGuard {
            tracker: self,
            epoch,
        }
    }

    /// Advance to next epoch
    pub fn advance(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Wait for an epoch to drain (all refs released)
    pub fn wait_drain(&self, target_epoch: u64, timeout: Duration) -> bool {
        let idx = (target_epoch % 8) as usize;
        let start = Instant::now();

        while start.elapsed() < timeout {
            if self.epoch_refs[idx].load(Ordering::Acquire) == 0 {
                return true;
            }
            std::thread::sleep(Duration::from_micros(100));
        }

        false
    }

    /// Get reference count for an epoch
    pub fn refs_for_epoch(&self, epoch: u64) -> usize {
        let idx = (epoch % 8) as usize;
        self.epoch_refs[idx].load(Ordering::Acquire)
    }
}

/// Guard that releases epoch reference on drop
pub struct EpochGuard<'a> {
    tracker: &'a EpochTracker,
    epoch: u64,
}

impl Drop for EpochGuard<'_> {
    fn drop(&mut self) {
        let idx = (self.epoch % 8) as usize;
        self.tracker.epoch_refs[idx].fetch_sub(1, Ordering::AcqRel);
    }
}

// ============================================================================
// Hot-Reloadable Plugin
// ============================================================================

/// State of a hot-reloadable plugin
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotReloadState {
    /// Normal operation
    Active,
    /// Preparing for upgrade
    PreparingUpgrade,
    /// Draining in-flight calls
    Draining,
    /// Performing atomic swap
    Swapping,
    /// Upgrade complete
    UpgradeComplete,
    /// Upgrade failed, rolled back
    RolledBack,
}

/// Statistics for hot-reload operations
#[derive(Debug, Clone, Default)]
pub struct HotReloadStats {
    /// Number of successful upgrades
    pub successful_upgrades: u64,
    /// Number of failed upgrades
    pub failed_upgrades: u64,
    /// Total drain time in microseconds
    pub total_drain_time_us: u64,
    /// Longest drain time
    pub max_drain_time_us: u64,
    /// Current version number
    pub version: u64,
}

/// A hot-reloadable plugin wrapper
pub struct HotReloadablePlugin {
    /// Plugin name
    name: String,
    /// Current active instance (wrapped in Arc for atomic swap)
    current: RwLock<Arc<WasmPluginInstance>>,
    /// Pending new instance (during upgrade)
    pending: Mutex<Option<Arc<WasmPluginInstance>>>,
    /// Epoch tracker for draining
    epochs: EpochTracker,
    /// Current state
    state: RwLock<HotReloadState>,
    /// Statistics
    stats: RwLock<HotReloadStats>,
    /// Current manifest
    manifest: RwLock<PluginManifest>,
    /// Upgrade timeout
    drain_timeout: Duration,
}

impl HotReloadablePlugin {
    /// Create a new hot-reloadable plugin
    pub fn new(name: &str, instance: Arc<WasmPluginInstance>, manifest: PluginManifest) -> Self {
        Self {
            name: name.to_string(),
            current: RwLock::new(instance),
            pending: Mutex::new(None),
            epochs: EpochTracker::new(),
            state: RwLock::new(HotReloadState::Active),
            stats: RwLock::new(HotReloadStats::default()),
            manifest: RwLock::new(manifest),
            drain_timeout: Duration::from_secs(5),
        }
    }

    /// Get the current active instance
    pub fn current(&self) -> Arc<WasmPluginInstance> {
        self.current.read().clone()
    }

    /// Call a function, tracking the epoch
    pub fn call(&self, func_name: &str, args: &[WasmValue]) -> KernelResult<Vec<WasmValue>> {
        // Enter epoch
        let _guard = self.epochs.enter();

        // Get current instance
        let instance = self.current();

        // Execute call
        instance.call(func_name, args)
    }

    /// Prepare an upgrade with new WASM bytes
    pub fn prepare_upgrade(
        &self,
        wasm_bytes: &[u8],
        new_manifest: PluginManifest,
    ) -> KernelResult<()> {
        // Check current state
        {
            let state = self.state.read();
            if *state != HotReloadState::Active {
                return Err(KernelError::Plugin {
                    message: format!("cannot upgrade in state {:?}, must be Active", *state),
                });
            }
        }

        // Set preparing state
        *self.state.write() = HotReloadState::PreparingUpgrade;

        // Validate new manifest against current
        self.validate_upgrade(&new_manifest)?;

        // Create new instance
        let config = WasmInstanceConfig {
            capabilities: new_manifest.to_capabilities(),
            ..Default::default()
        };

        let new_instance = WasmPluginInstance::new(&self.name, wasm_bytes, config)?;
        new_instance.init()?;

        // Store pending
        *self.pending.lock() = Some(Arc::new(new_instance));
        *self.manifest.write() = new_manifest;

        Ok(())
    }

    /// Execute the upgrade
    pub fn execute_upgrade(&self) -> KernelResult<()> {
        // Check state
        {
            let state = self.state.read();
            if *state != HotReloadState::PreparingUpgrade {
                return Err(KernelError::Plugin {
                    message: "must call prepare_upgrade first".to_string(),
                });
            }
        }

        let drain_start = Instant::now();

        // Enter draining state
        *self.state.write() = HotReloadState::Draining;

        // Advance epoch
        let old_epoch = self.epochs.current();
        self.epochs.advance();

        // Wait for old epoch to drain
        if !self.epochs.wait_drain(old_epoch, self.drain_timeout) {
            // Rollback
            *self.state.write() = HotReloadState::RolledBack;
            *self.pending.lock() = None;

            let mut stats = self.stats.write();
            stats.failed_upgrades += 1;

            return Err(KernelError::Plugin {
                message: format!(
                    "drain timeout: {} refs still held after {:?}",
                    self.epochs.refs_for_epoch(old_epoch),
                    self.drain_timeout
                ),
            });
        }

        let drain_time = drain_start.elapsed();

        // Enter swapping state
        *self.state.write() = HotReloadState::Swapping;

        // Get pending instance
        let new_instance = self
            .pending
            .lock()
            .take()
            .ok_or_else(|| KernelError::Plugin {
                message: "pending instance missing during upgrade".to_string(),
            })?;

        // Atomic swap
        *self.current.write() = new_instance;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.successful_upgrades += 1;
            stats.version += 1;
            let drain_us = drain_time.as_micros() as u64;
            stats.total_drain_time_us += drain_us;
            stats.max_drain_time_us = stats.max_drain_time_us.max(drain_us);
        }

        // Mark complete
        *self.state.write() = HotReloadState::UpgradeComplete;

        // Reset to active
        *self.state.write() = HotReloadState::Active;

        Ok(())
    }

    /// Perform a full upgrade (prepare + execute)
    pub fn upgrade(&self, wasm_bytes: &[u8], new_manifest: PluginManifest) -> KernelResult<()> {
        self.prepare_upgrade(wasm_bytes, new_manifest)?;
        self.execute_upgrade()
    }

    /// Cancel a pending upgrade
    pub fn cancel_upgrade(&self) -> KernelResult<()> {
        let state = *self.state.read();

        match state {
            HotReloadState::PreparingUpgrade => {
                *self.pending.lock() = None;
                *self.state.write() = HotReloadState::Active;
                Ok(())
            }
            HotReloadState::Active => {
                // Nothing to cancel
                Ok(())
            }
            _ => Err(KernelError::Plugin {
                message: format!("cannot cancel in state {:?}", state),
            }),
        }
    }

    /// Get current state
    pub fn state(&self) -> HotReloadState {
        *self.state.read()
    }

    /// Get statistics
    pub fn stats(&self) -> HotReloadStats {
        self.stats.read().clone()
    }

    /// Get plugin name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get current manifest
    pub fn manifest(&self) -> PluginManifest {
        self.manifest.read().clone()
    }

    /// Set drain timeout
    pub fn set_drain_timeout(&mut self, timeout: Duration) {
        self.drain_timeout = timeout;
    }

    /// Validate that an upgrade is compatible
    fn validate_upgrade(&self, new_manifest: &PluginManifest) -> KernelResult<()> {
        let current = self.manifest.read();

        // Name must match
        if current.plugin.name != new_manifest.plugin.name {
            return Err(KernelError::Plugin {
                message: format!(
                    "plugin name mismatch: {} vs {}",
                    current.plugin.name, new_manifest.plugin.name
                ),
            });
        }

        // Version should be different (warning, not error)
        if current.plugin.version == new_manifest.plugin.version {
            // Just a warning in production
        }

        // All existing hooks must still be present
        for hook in &current.hooks.before_insert {
            if !new_manifest.exports.functions.contains(hook) {
                return Err(KernelError::Plugin {
                    message: format!("new version missing hook function: {}", hook),
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Hot-Reload Manager
// ============================================================================

/// Manager for all hot-reloadable plugins
pub struct HotReloadManager {
    /// Plugins by name
    plugins: RwLock<std::collections::HashMap<String, Arc<HotReloadablePlugin>>>,
}

impl Default for HotReloadManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HotReloadManager {
    /// Create a new manager
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Register a new hot-reloadable plugin
    pub fn register(
        &self,
        name: &str,
        instance: Arc<WasmPluginInstance>,
        manifest: PluginManifest,
    ) -> KernelResult<()> {
        let mut plugins = self.plugins.write();

        if plugins.contains_key(name) {
            return Err(KernelError::Plugin {
                message: format!("plugin '{}' already registered", name),
            });
        }

        let plugin = Arc::new(HotReloadablePlugin::new(name, instance, manifest));
        plugins.insert(name.to_string(), plugin);

        Ok(())
    }

    /// Get a plugin by name
    pub fn get(&self, name: &str) -> Option<Arc<HotReloadablePlugin>> {
        self.plugins.read().get(name).cloned()
    }

    /// Upgrade a plugin
    pub fn upgrade(
        &self,
        name: &str,
        wasm_bytes: &[u8],
        new_manifest: PluginManifest,
    ) -> KernelResult<()> {
        let plugin = self.get(name).ok_or_else(|| KernelError::Plugin {
            message: format!("plugin '{}' not found", name),
        })?;

        plugin.upgrade(wasm_bytes, new_manifest)
    }

    /// Unregister a plugin
    pub fn unregister(&self, name: &str) -> KernelResult<()> {
        let mut plugins = self.plugins.write();

        if plugins.remove(name).is_none() {
            return Err(KernelError::Plugin {
                message: format!("plugin '{}' not found", name),
            });
        }

        Ok(())
    }

    /// List all plugins
    pub fn list(&self) -> Vec<String> {
        self.plugins.read().keys().cloned().collect()
    }

    /// Get stats for all plugins
    pub fn all_stats(&self) -> Vec<(String, HotReloadStats)> {
        self.plugins
            .read()
            .iter()
            .map(|(name, plugin)| (name.clone(), plugin.stats()))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin_manifest::ManifestBuilder;

    fn create_test_instance(name: &str) -> Arc<WasmPluginInstance> {
        let config = WasmInstanceConfig::default();
        let instance = WasmPluginInstance::new(name, b"test wasm", config).unwrap();
        instance.init().unwrap();
        Arc::new(instance)
    }

    fn create_test_manifest(name: &str, version: &str) -> PluginManifest {
        ManifestBuilder::new(name, version)
            .export("on_insert")
            .build()
            .unwrap()
    }

    #[test]
    fn test_epoch_tracker() {
        let tracker = EpochTracker::new();

        assert_eq!(tracker.current(), 0);

        // Enter epoch 0
        let guard = tracker.enter();
        assert_eq!(tracker.refs_for_epoch(0), 1);

        // Advance to epoch 1
        tracker.advance();
        assert_eq!(tracker.current(), 1);

        // Old epoch still has ref
        assert_eq!(tracker.refs_for_epoch(0), 1);

        // Drop guard
        drop(guard);
        assert_eq!(tracker.refs_for_epoch(0), 0);
    }

    #[test]
    fn test_epoch_drain() {
        let tracker = EpochTracker::new();

        // No refs, drain should succeed immediately
        assert!(tracker.wait_drain(0, Duration::from_millis(10)));

        // With a ref, drain should timeout
        let _guard = tracker.enter();
        assert!(!tracker.wait_drain(0, Duration::from_millis(10)));
    }

    #[test]
    fn test_hot_reload_creation() {
        let instance = create_test_instance("test");
        let manifest = create_test_manifest("test", "1.0.0");

        let plugin = HotReloadablePlugin::new("test", instance, manifest);

        assert_eq!(plugin.name(), "test");
        assert_eq!(plugin.state(), HotReloadState::Active);
    }

    #[test]
    fn test_hot_reload_call() {
        let instance = create_test_instance("test");
        let manifest = create_test_manifest("test", "1.0.0");

        let plugin = HotReloadablePlugin::new("test", instance, manifest);

        // Call should work
        let result = plugin.call("on_insert", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hot_reload_prepare() {
        let instance = create_test_instance("test");
        let manifest = create_test_manifest("test", "1.0.0");

        let plugin = HotReloadablePlugin::new("test", instance, manifest);

        let new_manifest = create_test_manifest("test", "2.0.0");
        plugin.prepare_upgrade(b"new wasm", new_manifest).unwrap();

        assert_eq!(plugin.state(), HotReloadState::PreparingUpgrade);
    }

    #[test]
    fn test_hot_reload_full_upgrade() {
        let instance = create_test_instance("test");
        let manifest = create_test_manifest("test", "1.0.0");

        let plugin = HotReloadablePlugin::new("test", instance, manifest);

        let new_manifest = create_test_manifest("test", "2.0.0");
        plugin.upgrade(b"new wasm", new_manifest).unwrap();

        assert_eq!(plugin.state(), HotReloadState::Active);

        let stats = plugin.stats();
        assert_eq!(stats.successful_upgrades, 1);
        assert_eq!(stats.version, 1);
    }

    #[test]
    fn test_hot_reload_cancel() {
        let instance = create_test_instance("test");
        let manifest = create_test_manifest("test", "1.0.0");

        let plugin = HotReloadablePlugin::new("test", instance, manifest);

        let new_manifest = create_test_manifest("test", "2.0.0");
        plugin.prepare_upgrade(b"new wasm", new_manifest).unwrap();

        plugin.cancel_upgrade().unwrap();
        assert_eq!(plugin.state(), HotReloadState::Active);
    }

    #[test]
    fn test_hot_reload_name_mismatch() {
        let instance = create_test_instance("test");
        let manifest = create_test_manifest("test", "1.0.0");

        let plugin = HotReloadablePlugin::new("test", instance, manifest);

        // Different name should fail
        let new_manifest = create_test_manifest("different", "2.0.0");
        let result = plugin.prepare_upgrade(b"new wasm", new_manifest);
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_operations() {
        let manager = HotReloadManager::new();

        let instance = create_test_instance("plugin1");
        let manifest = create_test_manifest("plugin1", "1.0.0");

        // Register
        manager.register("plugin1", instance, manifest).unwrap();
        assert_eq!(manager.list().len(), 1);

        // Get
        let plugin = manager.get("plugin1").unwrap();
        assert_eq!(plugin.name(), "plugin1");

        // Upgrade
        let new_manifest = create_test_manifest("plugin1", "2.0.0");
        manager
            .upgrade("plugin1", b"new wasm", new_manifest)
            .unwrap();

        // Stats
        let stats = manager.all_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.successful_upgrades, 1);

        // Unregister
        manager.unregister("plugin1").unwrap();
        assert!(manager.list().is_empty());
    }

    #[test]
    fn test_manager_duplicate() {
        let manager = HotReloadManager::new();

        let instance1 = create_test_instance("dup");
        let manifest1 = create_test_manifest("dup", "1.0.0");
        manager.register("dup", instance1, manifest1).unwrap();

        let instance2 = create_test_instance("dup");
        let manifest2 = create_test_manifest("dup", "1.0.0");
        let result = manager.register("dup", instance2, manifest2);
        assert!(result.is_err());
    }

    #[test]
    fn test_concurrent_calls_during_upgrade() {
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let instance = create_test_instance("concurrent");
        let manifest = create_test_manifest("concurrent", "1.0.0");

        let plugin = Arc::new(HotReloadablePlugin::new("concurrent", instance, manifest));

        // Flag to stop workers
        let stop = Arc::new(AtomicBool::new(false));

        // Spawn worker threads making calls
        let mut handles = vec![];
        for _ in 0..4 {
            let p = plugin.clone();
            let s = stop.clone();
            handles.push(thread::spawn(move || {
                let mut calls = 0;
                while !s.load(Ordering::Relaxed) {
                    let _ = p.call("on_insert", &[]);
                    calls += 1;
                    if calls > 100 {
                        break;
                    }
                }
            }));
        }

        // Perform upgrade while calls are happening
        thread::sleep(Duration::from_millis(5));
        let new_manifest = create_test_manifest("concurrent", "2.0.0");
        let result = plugin.upgrade(b"new wasm", new_manifest);

        // Stop workers
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }

        // Upgrade should succeed
        assert!(result.is_ok());
        assert_eq!(plugin.stats().successful_upgrades, 1);
    }
}
