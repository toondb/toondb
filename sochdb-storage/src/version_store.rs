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

//! Version Store for MVCC
//!
//! Manages version chains for multi-version concurrency control.
//! This module provides the storage layer for transaction isolation.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Version identifier
pub type VersionId = u64;

/// Transaction ID for MVCC visibility
pub type TxnId = u64;

/// A versioned value with transaction metadata
#[derive(Debug, Clone)]
pub struct VersionedValue<T> {
    /// The value
    pub value: T,
    /// Transaction that created this version
    pub created_by: TxnId,
    /// Transaction that deleted this version (None if active)
    pub deleted_by: Option<TxnId>,
    /// Timestamp of creation
    pub created_at: u64,
}

/// Version store for a single key's version chain
#[derive(Debug)]
pub struct VersionChain<T> {
    /// Versions ordered by creation time (newest first)
    versions: RwLock<Vec<VersionedValue<T>>>,
}

impl<T: Clone> VersionChain<T> {
    /// Create a new empty version chain
    pub fn new() -> Self {
        Self {
            versions: RwLock::new(Vec::new()),
        }
    }

    /// Add a new version
    pub fn add_version(&self, value: T, txn_id: TxnId, timestamp: u64) {
        let version = VersionedValue {
            value,
            created_by: txn_id,
            deleted_by: None,
            created_at: timestamp,
        };
        self.versions.write().insert(0, version);
    }

    /// Get the visible version for a transaction
    pub fn get_visible(&self, read_txn: TxnId) -> Option<T> {
        let versions = self.versions.read();
        for version in versions.iter() {
            // Skip if created by a future transaction
            if version.created_by > read_txn {
                continue;
            }
            // Skip if deleted by a committed transaction <= read_txn
            if let Some(deleted_by) = version.deleted_by
                && deleted_by <= read_txn
            {
                continue;
            }
            return Some(version.value.clone());
        }
        None
    }

    /// Mark the current version as deleted
    pub fn mark_deleted(&self, txn_id: TxnId) -> bool {
        let mut versions = self.versions.write();
        if let Some(version) = versions.first_mut()
            && version.deleted_by.is_none()
        {
            version.deleted_by = Some(txn_id);
            return true;
        }
        false
    }

    /// Garbage collect old versions not visible to any active transaction
    pub fn gc(&self, oldest_active_txn: TxnId) {
        let mut versions = self.versions.write();
        // Keep at least one version, remove those not visible to any active txn
        if versions.len() <= 1 {
            return;
        }

        versions.retain(|v| {
            // Keep if created by a transaction >= oldest active
            // or if it's the most recent version
            v.created_by >= oldest_active_txn || v.deleted_by.is_none()
        });
    }
}

impl<T: Clone> Default for VersionChain<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Version store managing version chains for all keys
pub struct VersionStore<K, V> {
    /// Version chains by key
    chains: RwLock<HashMap<K, VersionChain<V>>>,
    /// Next transaction ID
    next_txn_id: AtomicU64,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> VersionStore<K, V> {
    /// Create a new version store
    pub fn new() -> Self {
        Self {
            chains: RwLock::new(HashMap::new()),
            next_txn_id: AtomicU64::new(1),
        }
    }

    /// Get the next transaction ID
    pub fn next_txn_id(&self) -> TxnId {
        self.next_txn_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Insert or update a value
    pub fn put(&self, key: K, value: V, txn_id: TxnId) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let chains = self.chains.read();
        if let Some(chain) = chains.get(&key) {
            chain.add_version(value, txn_id, timestamp);
            return;
        }
        drop(chains);

        // Need to create new chain
        let mut chains = self.chains.write();
        let chain = chains.entry(key).or_default();
        chain.add_version(value, txn_id, timestamp);
    }

    /// Get the visible value for a transaction
    pub fn get(&self, key: &K, read_txn: TxnId) -> Option<V> {
        let chains = self.chains.read();
        chains
            .get(key)
            .and_then(|chain| chain.get_visible(read_txn))
    }

    /// Delete a key
    pub fn delete(&self, key: &K, txn_id: TxnId) -> bool {
        let chains = self.chains.read();
        if let Some(chain) = chains.get(key) {
            return chain.mark_deleted(txn_id);
        }
        false
    }

    /// Garbage collect old versions
    pub fn gc(&self, oldest_active_txn: TxnId) {
        let chains = self.chains.read();
        for chain in chains.values() {
            chain.gc(oldest_active_txn);
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> Default for VersionStore<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_chain_basic() {
        let chain = VersionChain::<i32>::new();

        chain.add_version(1, 1, 100);
        chain.add_version(2, 2, 200);
        chain.add_version(3, 3, 300);

        // Transaction 2 should see version 2
        assert_eq!(chain.get_visible(2), Some(2));

        // Transaction 4 should see version 3
        assert_eq!(chain.get_visible(4), Some(3));

        // Transaction 1 should see version 1
        assert_eq!(chain.get_visible(1), Some(1));
    }

    #[test]
    fn test_version_store_mvcc() {
        let store = VersionStore::<String, i32>::new();

        // Insert with transaction 1
        store.put("key1".to_string(), 100, 1);

        // Update with transaction 2
        store.put("key1".to_string(), 200, 2);

        // Transaction 1 should see 100
        assert_eq!(store.get(&"key1".to_string(), 1), Some(100));

        // Transaction 3 should see 200
        assert_eq!(store.get(&"key1".to_string(), 3), Some(200));
    }

    #[test]
    fn test_delete() {
        let store = VersionStore::<String, i32>::new();

        store.put("key1".to_string(), 100, 1);

        // Delete with transaction 2
        assert!(store.delete(&"key1".to_string(), 2));

        // Transaction 1 should still see 100
        assert_eq!(store.get(&"key1".to_string(), 1), Some(100));

        // Transaction 3 should see None (deleted)
        assert_eq!(store.get(&"key1".to_string(), 3), None);
    }
}
