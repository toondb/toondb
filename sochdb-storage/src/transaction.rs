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

//! Unified Transaction Coordinator
//!
//! This module defines the canonical transaction management interface for SochDB.
//! It consolidates the functionality of multiple transaction managers:
//!
//! ## Implementation Guide
//!
//! There are THREE transaction manager implementations in SochDB. Here's when to use each:
//!
//! ### 1. `MvccTransactionManager` (wal_integration.rs) - **RECOMMENDED FOR PRODUCTION**
//!
//! Full-featured transaction manager with:
//! - ✅ WAL-based durability (fsync on commit)
//! - ✅ MVCC snapshot isolation
//! - ✅ Serializable Snapshot Isolation (SSI)
//! - ✅ Group commit for high throughput
//! - ✅ Version chains with garbage collection
//!
//! ```ignore
//! use sochdb_storage::wal_integration::MvccTransactionManager;
//!
//! let txm = MvccTransactionManager::new("wal.log", |key, value| {
//!     // Apply callback
//!     Ok(())
//! })?;
//!
//! let txn_id = txm.begin(IsolationLevel::Serializable)?;
//! txm.write(txn_id, b"key", b"value")?;
//! txm.commit(txn_id)?; // fsync guarantee
//! ```
//!
//! ### 2. `MvccManager` (durable_storage.rs) - SSI VALIDATION ONLY
//!
//! Provides SSI conflict detection but delegates durability to storage layer.
//! Used internally by `DurableStorage` for transaction tracking.
//!
//! - ✅ SSI rw-antidependency detection
//! - ✅ Bloom filter optimization for read/write sets
//! - ❌ No WAL (relies on caller for durability)
//!
//! ### 3. `TransactionManager` (mvcc_snapshot.rs) - **DEPRECATED FOR NEW CODE**
//!
//! Minimal snapshot isolation without durability. Only suitable for:
//! - Unit testing
//! - Ephemeral in-memory operations
//!
//! **Do not use for production workloads requiring crash recovery.**
//!
//! ## Transaction Coordinator Trait
//!
//! The `TransactionCoordinator` trait unifies the common interface:

use sochdb_core::Result;

// Re-export TransactionMode from durable_storage (canonical definition)
pub use crate::durable_storage::TransactionMode;

/// Isolation level for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Read committed - see all committed data
    ReadCommitted,
    /// Snapshot isolation (default) - consistent point-in-time view
    #[default]
    SnapshotIsolation,
    /// Serializable via SSI - full serializability guarantee
    Serializable,
}

/// Transaction handle returned by begin operations
#[derive(Debug, Clone)]
pub struct TransactionHandle {
    /// Unique transaction ID
    pub txn_id: u64,
    /// Snapshot timestamp for MVCC visibility
    pub snapshot_ts: u64,
    /// Transaction mode
    pub mode: TransactionMode,
    /// Isolation level
    pub isolation_level: IsolationLevel,
}

/// Unified transaction coordinator interface
///
/// This trait defines the canonical API for transaction management.
/// All new transaction manager implementations should implement this trait.
///
/// ## Durability Contract
///
/// Implementations MUST document their durability guarantees:
/// - `Durable`: Committed transactions survive crash (WAL + fsync)
/// - `Ephemeral`: No durability guarantee (in-memory only)
pub trait TransactionCoordinator: Send + Sync {
    /// Begin a new transaction with default isolation level
    fn begin(&self) -> Result<TransactionHandle>;

    /// Begin a transaction with specified isolation level
    fn begin_with_isolation(&self, isolation: IsolationLevel) -> Result<TransactionHandle>;

    /// Begin a transaction with specified mode (for optimization)
    fn begin_with_mode(&self, mode: TransactionMode) -> Result<TransactionHandle>;

    /// Commit a transaction
    ///
    /// For durable implementations, this includes fsync.
    /// Returns the commit timestamp on success.
    fn commit(&self, txn_id: u64) -> Result<u64>;

    /// Abort a transaction
    ///
    /// Discards all changes made by the transaction.
    fn abort(&self, txn_id: u64) -> Result<()>;

    /// Get the current snapshot timestamp for a transaction
    fn get_snapshot_ts(&self, txn_id: u64) -> Option<u64>;

    /// Check if a transaction is still active
    fn is_active(&self, txn_id: u64) -> bool;

    /// Record a read for SSI tracking
    fn record_read(&self, txn_id: u64, key: &[u8]);

    /// Record a write for SSI tracking
    fn record_write(&self, txn_id: u64, key: &[u8]);

    /// Get durability guarantee of this implementation
    fn durability(&self) -> DurabilityLevel;
}

/// Durability guarantee level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityLevel {
    /// Fully durable with WAL + fsync
    Durable,
    /// Group commit (durable after batch fsync)
    GroupCommit,
    /// No durability (in-memory only)
    Ephemeral,
}

/// Recovery statistics returned after crash recovery
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of transactions recovered
    pub transactions_recovered: usize,
    /// Number of individual writes recovered
    pub writes_recovered: usize,
    /// The commit timestamp after recovery
    pub commit_ts: u64,
}

// =============================================================================
// Re-exports for convenience
// =============================================================================

// Note: The canonical implementation is MvccTransactionManager from wal_integration
// pub use crate::wal_integration::MvccTransactionManager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_mode_tracking() {
        assert!(TransactionMode::ReadWrite.tracks_reads());
        assert!(TransactionMode::ReadWrite.tracks_writes());
        
        assert!(!TransactionMode::ReadOnly.tracks_reads());
        assert!(!TransactionMode::ReadOnly.tracks_writes());
        
        assert!(!TransactionMode::WriteOnly.tracks_reads());
        assert!(TransactionMode::WriteOnly.tracks_writes());
    }

    #[test]
    fn test_isolation_level_default() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::SnapshotIsolation);
    }
}
