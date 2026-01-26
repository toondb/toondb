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

//! Unified MVCC Version Chain Interface
//!
//! This module defines the canonical interface for MVCC version chains across SochDB.
//! Multiple implementations exist for different subsystems, but they share these traits.
//!
//! ## Implementations
//!
//! | Implementation | Location | Use Case |
//! |---------------|----------|----------|
//! | `VersionChain` | `sochdb_core::epoch_gc` | Epoch-based GC with VecDeque |
//! | `VersionChain` | `sochdb_storage::mvcc_snapshot` | Snapshot-based visibility |
//! | `VersionChain` | `sochdb_storage::version_store` | Generic key-value MVCC |
//! | `VersionChain` | `sochdb_storage::durable_storage` | Binary-search optimized |
//!
//! ## Visibility Semantics
//!
//! All implementations follow these MVCC visibility rules:
//!
//! 1. **Read Committed**: A version is visible if its creating transaction has committed
//!    before the reader's start timestamp.
//!
//! 2. **Snapshot Isolation**: A version is visible if:
//!    - It was committed before the reader's snapshot timestamp
//!    - It was not deleted, or deleted after the snapshot timestamp
//!
//! 3. **Serializable (SSI)**: Adds read-write conflict detection on top of SI.

/// Transaction identifier type
pub type TxnId = u64;

/// Logical timestamp type
pub type Timestamp = u64;

/// Version visibility context
/// 
/// Provides the information needed to determine if a version is visible
/// to a particular reader/transaction.
#[derive(Debug, Clone)]
pub struct VisibilityContext {
    /// Reader's transaction ID
    pub reader_txn_id: TxnId,
    /// Reader's snapshot timestamp
    pub snapshot_ts: Timestamp,
    /// Set of transaction IDs that are still active (not committed)
    pub active_txn_ids: std::collections::HashSet<TxnId>,
}

impl VisibilityContext {
    /// Create a new visibility context
    pub fn new(reader_txn_id: TxnId, snapshot_ts: Timestamp) -> Self {
        Self {
            reader_txn_id,
            snapshot_ts,
            active_txn_ids: std::collections::HashSet::new(),
        }
    }

    /// Create with active transaction set
    pub fn with_active_txns(
        reader_txn_id: TxnId,
        snapshot_ts: Timestamp,
        active_txn_ids: std::collections::HashSet<TxnId>,
    ) -> Self {
        Self {
            reader_txn_id,
            snapshot_ts,
            active_txn_ids,
        }
    }

    /// Check if a transaction was committed before this snapshot
    pub fn is_committed_before(&self, txn_id: TxnId, commit_ts: Option<Timestamp>) -> bool {
        match commit_ts {
            Some(ts) => ts < self.snapshot_ts && !self.active_txn_ids.contains(&txn_id),
            None => false,
        }
    }
}

/// Version metadata
/// 
/// Common metadata for all version chain implementations.
#[derive(Debug, Clone)]
pub struct VersionMeta {
    /// Transaction that created this version
    pub created_by: TxnId,
    /// Timestamp when this version was created
    pub created_ts: Timestamp,
    /// Transaction that deleted this version (0 = not deleted)
    pub deleted_by: TxnId,
    /// Timestamp when this version was deleted (0 = not deleted)
    pub deleted_ts: Timestamp,
    /// Commit timestamp (0 = not yet committed)
    pub commit_ts: Timestamp,
}

impl VersionMeta {
    /// Create metadata for a new uncommitted version
    pub fn new_uncommitted(created_by: TxnId, created_ts: Timestamp) -> Self {
        Self {
            created_by,
            created_ts,
            deleted_by: 0,
            deleted_ts: 0,
            commit_ts: 0,
        }
    }

    /// Check if this version is visible according to the context
    pub fn is_visible(&self, ctx: &VisibilityContext) -> bool {
        // Must be committed before snapshot
        if self.commit_ts == 0 {
            // Uncommitted - only visible to creating transaction
            return self.created_by == ctx.reader_txn_id;
        }

        if self.commit_ts >= ctx.snapshot_ts {
            return false;
        }

        // Must not be deleted, or deleted after snapshot
        if self.deleted_by != 0 && self.deleted_ts < ctx.snapshot_ts {
            return false;
        }

        true
    }

    /// Mark as committed
    pub fn commit(&mut self, commit_ts: Timestamp) {
        self.commit_ts = commit_ts;
    }

    /// Mark as deleted
    pub fn delete(&mut self, deleted_by: TxnId, deleted_ts: Timestamp) {
        self.deleted_by = deleted_by;
        self.deleted_ts = deleted_ts;
    }

    /// Check if version is committed
    pub fn is_committed(&self) -> bool {
        self.commit_ts != 0
    }

    /// Check if version is deleted
    pub fn is_deleted(&self) -> bool {
        self.deleted_by != 0
    }
}

/// Trait for MVCC version chain implementations
/// 
/// Implementors store multiple versions of a value and provide
/// visibility-based access according to MVCC semantics.
pub trait MvccVersionChain {
    /// The value type stored in versions
    type Value;

    /// Get the visible version for the given context
    fn get_visible(&self, ctx: &VisibilityContext) -> Option<&Self::Value>;

    /// Get the latest version (regardless of visibility)
    fn get_latest(&self) -> Option<&Self::Value>;

    /// Number of versions in the chain
    fn version_count(&self) -> usize;

    /// Check if the chain is empty
    fn is_empty(&self) -> bool {
        self.version_count() == 0
    }
}

/// Trait for mutable version chain operations
pub trait MvccVersionChainMut: MvccVersionChain {
    /// Add a new uncommitted version
    fn add_uncommitted(&mut self, value: Self::Value, txn_id: TxnId);

    /// Commit a version
    fn commit_version(&mut self, txn_id: TxnId, commit_ts: Timestamp) -> bool;

    /// Mark the latest visible version as deleted
    fn delete_version(&mut self, txn_id: TxnId, delete_ts: Timestamp) -> bool;

    /// Garbage collect versions older than the given timestamp
    /// Returns (versions_removed, bytes_freed)
    fn gc(&mut self, min_visible_ts: Timestamp) -> (usize, usize);
}

/// Trait for detecting write conflicts
pub trait WriteConflictDetection {
    /// Check if there's a write-write conflict with another transaction
    fn has_write_conflict(&self, txn_id: TxnId) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_meta_visibility() {
        let mut meta = VersionMeta::new_uncommitted(1, 100);
        
        // Uncommitted - only visible to creator
        let ctx = VisibilityContext::new(1, 200);
        assert!(meta.is_visible(&ctx));
        
        let ctx2 = VisibilityContext::new(2, 200);
        assert!(!meta.is_visible(&ctx2));
        
        // After commit - visible to later snapshots
        meta.commit(150);
        assert!(meta.is_visible(&ctx2));
        
        // Not visible to earlier snapshots
        let ctx3 = VisibilityContext::new(3, 100);
        assert!(!meta.is_visible(&ctx3));
    }

    #[test]
    fn test_version_meta_deletion() {
        let mut meta = VersionMeta::new_uncommitted(1, 100);
        meta.commit(150);
        meta.delete(2, 200);
        
        // Visible before deletion
        let ctx = VisibilityContext::new(3, 180);
        assert!(meta.is_visible(&ctx));
        
        // Not visible after deletion
        let ctx2 = VisibilityContext::new(3, 250);
        assert!(!meta.is_visible(&ctx2));
    }

    #[test]
    fn test_visibility_context_committed_before() {
        let mut active = std::collections::HashSet::new();
        active.insert(5);
        
        let ctx = VisibilityContext::with_active_txns(1, 200, active);
        
        // Committed before snapshot
        assert!(ctx.is_committed_before(2, Some(100)));
        
        // Committed after snapshot
        assert!(!ctx.is_committed_before(3, Some(250)));
        
        // Active transaction - not committed
        assert!(!ctx.is_committed_before(5, Some(100)));
        
        // No commit timestamp
        assert!(!ctx.is_committed_before(6, None));
    }
}
