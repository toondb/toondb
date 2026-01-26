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

//! MVCC Version Management for LSCS
//!
//! Implements Multi-Version Concurrency Control for the Log-Structured Column Store.
//! Provides lock-free reads during compaction by maintaining version snapshots.
//!
//! ## Architecture
//!
//! ```text
//! VersionSet
//! ├── current: Arc<Snapshot>     (latest version for new reads)
//! ├── active_snapshots: Vec<Weak<Snapshot>>  (snapshots still in use)
//! └── cleanup_threshold: usize   (trigger cleanup when exceeded)
//!
//! Read Path:
//!   snapshot = version_set.acquire();  // Arc clone + ref count
//!   // ... perform reads using snapshot ...
//!   // Drop snapshot -> decrements ref count
//!
//! Write Path:
//!   new_snapshot = create_new_snapshot(...);
//!   version_set.install(new_snapshot);  // Atomic swap
//!   // Old snapshot cleaned when all readers finish
//! ```

use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// A read-only snapshot of the storage state.
///
/// Readers acquire this via `VersionSet::acquire()`. The snapshot remains
/// valid for the lifetime of the Arc, even if newer snapshots are installed.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Snapshot/version number (monotonically increasing)
    pub version: u64,
    /// Timestamp when this snapshot was created
    pub timestamp_us: u64,
    /// Column groups visible in this snapshot (by level)
    pub column_groups: Vec<Vec<ColumnGroupRef>>,
    /// Minimum visible transaction ID (for MVCC reads)
    pub min_visible_txn: u64,
    /// Maximum visible transaction ID
    pub max_visible_txn: u64,
}

/// Reference to a column group file
#[derive(Debug, Clone)]
pub struct ColumnGroupRef {
    /// Unique identifier
    pub id: u64,
    /// Path to column group directory
    pub path: String,
    /// Level in LSM tree
    pub level: u32,
    /// Sequence number
    pub sequence: u64,
    /// Row count
    pub row_count: u64,
    /// Minimum timestamp in this group
    pub min_timestamp: u64,
    /// Maximum timestamp in this group
    pub max_timestamp: u64,
}

impl Snapshot {
    /// Create a new empty snapshot
    pub fn empty(version: u64) -> Self {
        Self {
            version,
            timestamp_us: Self::now_us(),
            column_groups: Vec::new(),
            min_visible_txn: 0,
            max_visible_txn: u64::MAX,
        }
    }

    /// Create a snapshot with column groups
    pub fn new(
        version: u64,
        column_groups: Vec<Vec<ColumnGroupRef>>,
        min_visible_txn: u64,
        max_visible_txn: u64,
    ) -> Self {
        Self {
            version,
            timestamp_us: Self::now_us(),
            column_groups,
            min_visible_txn,
            max_visible_txn,
        }
    }

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Get column groups at a specific level
    pub fn level_groups(&self, level: usize) -> &[ColumnGroupRef] {
        self.column_groups
            .get(level)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get total number of column groups
    pub fn total_groups(&self) -> usize {
        self.column_groups.iter().map(|l| l.len()).sum()
    }

    /// Check if a transaction is visible in this snapshot
    pub fn is_visible(&self, txn_id: u64) -> bool {
        txn_id >= self.min_visible_txn && txn_id <= self.max_visible_txn
    }
}

/// RAII guard for a snapshot.
///
/// Holds an Arc to the Snapshot, ensuring it remains valid
/// for the lifetime of this guard.
#[derive(Debug)]
pub struct SnapshotGuard {
    snapshot: Arc<Snapshot>,
}

impl SnapshotGuard {
    /// Get the version number
    pub fn version(&self) -> u64 {
        self.snapshot.version
    }

    /// Access the underlying Snapshot
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Get column groups at a level
    pub fn level_groups(&self, level: usize) -> &[ColumnGroupRef] {
        self.snapshot.level_groups(level)
    }

    /// Check if a transaction is visible
    pub fn is_visible(&self, txn_id: u64) -> bool {
        self.snapshot.is_visible(txn_id)
    }
}

impl std::ops::Deref for SnapshotGuard {
    type Target = Snapshot;

    fn deref(&self) -> &Self::Target {
        &self.snapshot
    }
}

/// Manages multiple versions/snapshots of storage state.
///
/// Provides lock-free reads by allowing readers to hold references to
/// older snapshots while compaction creates new versions.
pub struct VersionSet {
    /// Current snapshot (latest) - readers acquire this
    current: RwLock<Arc<Snapshot>>,
    /// Active snapshots that may still have readers
    active_snapshots: Mutex<Vec<Weak<Snapshot>>>,
    /// Version number counter
    next_version: AtomicU64,
    /// Statistics
    stats: VersionSetStats,
}

impl VersionSet {
    /// Create a new version set with an empty initial snapshot
    pub fn new() -> Self {
        let initial = Arc::new(Snapshot::empty(0));
        Self {
            current: RwLock::new(initial),
            active_snapshots: Mutex::new(Vec::new()),
            next_version: AtomicU64::new(1),
            stats: VersionSetStats::default(),
        }
    }

    /// Acquire a snapshot for reading
    ///
    /// Returns a guard that keeps the snapshot alive until dropped.
    pub fn acquire(&self) -> SnapshotGuard {
        self.stats.acquires.fetch_add(1, Ordering::Relaxed);
        let current = self.current.read();
        SnapshotGuard {
            snapshot: Arc::clone(&current),
        }
    }

    /// Install a new snapshot
    ///
    /// The old snapshot remains valid for existing readers.
    pub fn install(&self, snapshot: Snapshot) {
        self.stats.installs.fetch_add(1, Ordering::Relaxed);

        let new_snapshot = Arc::new(snapshot);

        // Track the old snapshot
        let old = {
            let mut current = self.current.write();
            let old = Arc::clone(&current);
            *current = new_snapshot;
            old
        };

        // Add old snapshot to active list for cleanup tracking
        let mut active = self.active_snapshots.lock();
        active.push(Arc::downgrade(&old));

        // Update peak versions
        let current_count = active.len() as u64;
        self.stats
            .peak_versions
            .fetch_max(current_count, Ordering::Relaxed);
    }

    /// Create and install a new snapshot with column groups
    pub fn create_snapshot(&self, column_groups: Vec<Vec<ColumnGroupRef>>) -> u64 {
        let version = self.next_version.fetch_add(1, Ordering::SeqCst);
        let snapshot = Snapshot::new(version, column_groups, 0, u64::MAX);
        self.install(snapshot);
        version
    }

    /// Clean up old snapshots that have no readers
    pub fn cleanup(&self) {
        let mut active = self.active_snapshots.lock();
        let before_len = active.len();
        active.retain(|weak| weak.strong_count() > 0);
        let cleaned = before_len - active.len();
        self.stats
            .cleanups
            .fetch_add(cleaned as u64, Ordering::Relaxed);
    }

    /// Get current version number
    pub fn current_version(&self) -> u64 {
        self.current.read().version
    }

    /// Get number of active snapshots (includes current)
    pub fn active_count(&self) -> usize {
        let active = self.active_snapshots.lock();
        active.iter().filter(|w| w.strong_count() > 0).count() + 1
    }

    /// Get statistics snapshot
    pub fn stats(&self) -> VersionSetStatsSnapshot {
        VersionSetStatsSnapshot {
            installs: self.stats.installs.load(Ordering::Relaxed),
            acquires: self.stats.acquires.load(Ordering::Relaxed),
            cleanups: self.stats.cleanups.load(Ordering::Relaxed),
            peak_versions: self.stats.peak_versions.load(Ordering::Relaxed),
            active_versions: self.active_count(),
        }
    }
}

impl Default for VersionSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the version set
#[derive(Debug, Default)]
pub struct VersionSetStats {
    /// Number of snapshot installs
    pub installs: AtomicU64,
    /// Number of snapshot acquires
    pub acquires: AtomicU64,
    /// Number of old snapshots cleaned up
    pub cleanups: AtomicU64,
    /// Peak number of concurrent snapshots
    pub peak_versions: AtomicU64,
}

/// Snapshot of version set statistics
#[derive(Debug, Clone, Default)]
pub struct VersionSetStatsSnapshot {
    pub installs: u64,
    pub acquires: u64,
    pub cleanups: u64,
    pub peak_versions: u64,
    pub active_versions: usize,
}

// Re-export for backwards compatibility with lib.rs
pub use Snapshot as ReadVersion;
pub use SnapshotGuard as VersionGuard;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_version_set() {
        let vs = VersionSet::new();
        assert_eq!(vs.current_version(), 0);
        assert_eq!(vs.active_count(), 1);
    }

    #[test]
    fn test_acquire_snapshot() {
        let vs = VersionSet::new();
        let guard = vs.acquire();
        assert_eq!(guard.version(), 0);
    }

    #[test]
    fn test_install_snapshot() {
        let vs = VersionSet::new();

        let snapshot = Snapshot::new(1, vec![], 0, 100);
        vs.install(snapshot);

        assert_eq!(vs.current_version(), 1);

        let guard = vs.acquire();
        assert_eq!(guard.version(), 1);
        assert!(guard.is_visible(50));
        assert!(!guard.is_visible(101));
    }

    #[test]
    fn test_old_snapshot_stays_valid() {
        let vs = VersionSet::new();

        // Acquire before install
        let old_guard = vs.acquire();
        assert_eq!(old_guard.version(), 0);

        // Install new version
        let snapshot = Snapshot::new(1, vec![], 0, u64::MAX);
        vs.install(snapshot);

        // Old guard still valid
        assert_eq!(old_guard.version(), 0);

        // New acquire gets new version
        let new_guard = vs.acquire();
        assert_eq!(new_guard.version(), 1);
    }

    #[test]
    fn test_cleanup() {
        let vs = VersionSet::new();

        // Install several versions
        for i in 1..=5 {
            let snapshot = Snapshot::new(i, vec![], 0, u64::MAX);
            vs.install(snapshot);
        }

        // Without any guards held, cleanup should remove all old versions
        vs.cleanup();

        // Only current should remain
        assert_eq!(vs.active_count(), 1);
    }

    #[test]
    fn test_cleanup_with_active_readers() {
        let vs = VersionSet::new();

        // Acquire v0
        let guard = vs.acquire();

        // Install v1
        vs.install(Snapshot::new(1, vec![], 0, u64::MAX));

        // Install v2
        vs.install(Snapshot::new(2, vec![], 0, u64::MAX));

        // v0 is still held by guard, so cleanup won't remove it
        vs.cleanup();

        // v0 (held), v1 (released), v2 (current) -> 2 active
        // Actually, v0 is held and v2 is current = at least 2
        assert!(vs.active_count() >= 2);

        // Drop guard
        drop(guard);
        vs.cleanup();

        // Now only current should remain
        assert_eq!(vs.active_count(), 1);
    }

    #[test]
    fn test_stats() {
        let vs = VersionSet::new();

        let _g1 = vs.acquire();
        let _g2 = vs.acquire();

        vs.install(Snapshot::new(1, vec![], 0, u64::MAX));

        let stats = vs.stats();
        assert_eq!(stats.acquires, 2);
        assert_eq!(stats.installs, 1);
    }
}
