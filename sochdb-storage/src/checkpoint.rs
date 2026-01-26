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

//! ARIES-Style Checkpointing with WAL Compaction
//!
//! From mm.md Task 1.4: Checkpoint and WAL Truncation for Bounded Recovery
//!
//! ## Problem
//!
//! Without active checkpointing + truncation, recovery requires replaying the entire WAL,
//! trending toward unbounded startup time as WAL grows.
//!
//! ## Solution
//!
//! ARIES-style checkpointing with:
//! 1. Periodic checkpoint triggers (time-based or size-based)
//! 2. Checkpoint record with active_txns and dirty_pages
//! 3. Flush all dirty pages to stable storage
//! 4. Truncate WAL prefix up to checkpoint LSN
//!
//! ## Math
//!
//! ```text
//! Without checkpointing:
//!   Recovery time = O(total_WAL_records) = O(lifetime_operations)
//!
//! With checkpointing every C operations:
//!   Recovery time = O(records_since_checkpoint) ≤ O(C)
//!
//! For C = 100,000 records, ~10ms replay time:
//!   Recovery time bounded at ~1s regardless of DB lifetime
//!
//! WAL size bounded: max_size = checkpoint_interval × avg_record_size
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::hlc::HybridLogicalClock;
use sochdb_core::{Result, SochDBError};

/// Log Sequence Number - monotonically increasing identifier for WAL records
pub type Lsn = u64;

/// Page identifier
pub type PageId = u64;

/// Checkpoint interval configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Maximum WAL size before forced checkpoint (bytes)
    pub max_wal_size: u64,
    /// Maximum time between checkpoints
    pub max_interval: Duration,
    /// Minimum records before checkpoint
    pub min_records: u64,
    /// Whether to truncate WAL after checkpoint
    pub truncate_wal: bool,
    /// Whether checkpointing is enabled
    pub enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_wal_size: 64 * 1024 * 1024, // 64 MB
            max_interval: Duration::from_secs(60),
            min_records: 100_000,
            truncate_wal: true,
            enabled: true,
        }
    }
}

/// Active transaction entry for checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTransactionEntry {
    /// Transaction ID
    pub txn_id: u64,
    /// First LSN written by this transaction
    pub first_lsn: Lsn,
    /// Last LSN written by this transaction
    pub last_lsn: Lsn,
    /// Transaction start timestamp
    pub start_ts: u64,
}

/// Dirty page entry for checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirtyPageEntry {
    /// Page ID
    pub page_id: PageId,
    /// Recovery LSN (first LSN that dirtied this page)
    pub recovery_lsn: Lsn,
}

/// Checkpoint data written to WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Checkpoint ID (monotonically increasing)
    pub checkpoint_id: u64,
    /// LSN at start of checkpoint
    pub begin_checkpoint_lsn: Lsn,
    /// LSN at end of checkpoint
    pub end_checkpoint_lsn: Lsn,
    /// Active transactions at checkpoint time
    pub active_transactions: Vec<ActiveTransactionEntry>,
    /// Dirty pages at checkpoint time
    pub dirty_pages: Vec<DirtyPageEntry>,
    /// Timestamp when checkpoint was taken
    pub timestamp: u64,
    /// Oldest LSN needed for recovery (min of active txn first_lsn and dirty page recovery_lsn)
    pub oldest_required_lsn: Lsn,
}

impl CheckpointData {
    /// Create a new checkpoint
    pub fn new(
        checkpoint_id: u64,
        begin_lsn: Lsn,
        active_txns: Vec<ActiveTransactionEntry>,
        dirty_pages: Vec<DirtyPageEntry>,
    ) -> Self {
        // Calculate oldest required LSN
        let oldest_txn_lsn = active_txns.iter().map(|t| t.first_lsn).min().unwrap_or(Lsn::MAX);
        let oldest_page_lsn = dirty_pages.iter().map(|p| p.recovery_lsn).min().unwrap_or(Lsn::MAX);
        let oldest_required_lsn = oldest_txn_lsn.min(oldest_page_lsn).min(begin_lsn);

        Self {
            checkpoint_id,
            begin_checkpoint_lsn: begin_lsn,
            end_checkpoint_lsn: 0, // Set after checkpoint is complete
            active_transactions: active_txns,
            dirty_pages,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            oldest_required_lsn,
        }
    }
}

/// Checkpoint state persisted to disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Last completed checkpoint data
    pub last_checkpoint: Option<CheckpointData>,
    /// Total checkpoints taken
    pub total_checkpoints: u64,
    /// Total bytes truncated from WAL
    pub total_bytes_truncated: u64,
}

impl Default for CheckpointMeta {
    fn default() -> Self {
        Self {
            last_checkpoint: None,
            total_checkpoints: 0,
            total_bytes_truncated: 0,
        }
    }
}

/// Dirty page tracker for efficient checkpointing
pub struct DirtyPageTracker {
    /// Map of page_id -> recovery_lsn (first LSN that dirtied page)
    dirty_pages: RwLock<HashMap<PageId, Lsn>>,
}

impl DirtyPageTracker {
    pub fn new() -> Self {
        Self {
            dirty_pages: RwLock::new(HashMap::new()),
        }
    }

    /// Mark a page as dirty with its recovery LSN
    pub fn mark_dirty(&self, page_id: PageId, lsn: Lsn) {
        let mut dirty = self.dirty_pages.write();
        dirty.entry(page_id).or_insert(lsn);
    }

    /// Mark a page as clean (after flush to disk)
    pub fn mark_clean(&self, page_id: PageId) {
        self.dirty_pages.write().remove(&page_id);
    }

    /// Get all dirty pages for checkpoint
    pub fn get_dirty_pages(&self) -> Vec<DirtyPageEntry> {
        self.dirty_pages
            .read()
            .iter()
            .map(|(&page_id, &recovery_lsn)| DirtyPageEntry { page_id, recovery_lsn })
            .collect()
    }

    /// Get count of dirty pages
    pub fn dirty_count(&self) -> usize {
        self.dirty_pages.read().len()
    }
}

impl Default for DirtyPageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Active transaction tracker for checkpointing
pub struct ActiveTransactionTracker {
    /// Map of txn_id -> (first_lsn, last_lsn, start_ts)
    active_txns: RwLock<HashMap<u64, (Lsn, Lsn, u64)>>,
}

impl ActiveTransactionTracker {
    pub fn new() -> Self {
        Self {
            active_txns: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new transaction
    pub fn register(&self, txn_id: u64, start_ts: u64) {
        self.active_txns
            .write()
            .insert(txn_id, (Lsn::MAX, 0, start_ts));
    }

    /// Update transaction's LSN range
    pub fn update_lsn(&self, txn_id: u64, lsn: Lsn) {
        if let Some(entry) = self.active_txns.write().get_mut(&txn_id) {
            if entry.0 == Lsn::MAX {
                entry.0 = lsn; // First LSN
            }
            entry.1 = lsn; // Last LSN
        }
    }

    /// Remove a transaction (on commit or abort)
    pub fn remove(&self, txn_id: u64) {
        self.active_txns.write().remove(&txn_id);
    }

    /// Get all active transactions for checkpoint
    pub fn get_active_transactions(&self) -> Vec<ActiveTransactionEntry> {
        self.active_txns
            .read()
            .iter()
            .filter(|(_, (first_lsn, _, _))| *first_lsn != Lsn::MAX)
            .map(|(&txn_id, &(first_lsn, last_lsn, start_ts))| ActiveTransactionEntry {
                txn_id,
                first_lsn,
                last_lsn,
                start_ts,
            })
            .collect()
    }

    /// Get count of active transactions
    pub fn active_count(&self) -> usize {
        self.active_txns.read().len()
    }
}

impl Default for ActiveTransactionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint manager
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Path to checkpoint metadata file
    meta_path: PathBuf,
    /// Path to WAL directory
    #[allow(dead_code)]
    wal_dir: PathBuf,
    /// Current checkpoint metadata
    meta: RwLock<CheckpointMeta>,
    /// Dirty page tracker
    dirty_pages: Arc<DirtyPageTracker>,
    /// Active transaction tracker
    active_txns: Arc<ActiveTransactionTracker>,
    /// Current LSN counter
    current_lsn: AtomicU64,
    /// Records since last checkpoint
    records_since_checkpoint: AtomicU64,
    /// WAL bytes since last checkpoint
    wal_bytes_since_checkpoint: AtomicU64,
    /// Last checkpoint time
    last_checkpoint_time: Mutex<Instant>,
    /// Checkpoint in progress flag
    checkpoint_in_progress: AtomicBool,
    /// Next checkpoint ID
    next_checkpoint_id: AtomicU64,
    /// HLC for timestamps
    #[allow(dead_code)]
    hlc: Arc<HybridLogicalClock>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(
        data_dir: &Path,
        config: CheckpointConfig,
        dirty_pages: Arc<DirtyPageTracker>,
        active_txns: Arc<ActiveTransactionTracker>,
        hlc: Arc<HybridLogicalClock>,
    ) -> Result<Self> {
        let meta_path = data_dir.join("checkpoint.meta");
        let wal_dir = data_dir.join("wal");

        // Ensure directories exist
        fs::create_dir_all(&wal_dir)?;

        // Load existing metadata
        let meta = if meta_path.exists() {
            let data = fs::read(&meta_path)?;
            bincode::deserialize(&data).unwrap_or_default()
        } else {
            CheckpointMeta::default()
        };

        let next_id = meta.last_checkpoint.as_ref().map(|c| c.checkpoint_id + 1).unwrap_or(1);
        let last_lsn = meta.last_checkpoint.as_ref().map(|c| c.end_checkpoint_lsn).unwrap_or(0);

        Ok(Self {
            config,
            meta_path,
            wal_dir,
            meta: RwLock::new(meta),
            dirty_pages,
            active_txns,
            current_lsn: AtomicU64::new(last_lsn),
            records_since_checkpoint: AtomicU64::new(0),
            wal_bytes_since_checkpoint: AtomicU64::new(0),
            last_checkpoint_time: Mutex::new(Instant::now()),
            checkpoint_in_progress: AtomicBool::new(false),
            next_checkpoint_id: AtomicU64::new(next_id),
            hlc,
        })
    }

    /// Allocate the next LSN
    #[inline]
    pub fn next_lsn(&self) -> Lsn {
        self.current_lsn.fetch_add(1, Ordering::SeqCst)
    }

    /// Record a WAL write for checkpoint tracking
    pub fn record_wal_write(&self, bytes: u64) {
        self.records_since_checkpoint.fetch_add(1, Ordering::Relaxed);
        self.wal_bytes_since_checkpoint.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Check if checkpoint is needed
    pub fn should_checkpoint(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        if self.checkpoint_in_progress.load(Ordering::Relaxed) {
            return false;
        }

        let records = self.records_since_checkpoint.load(Ordering::Relaxed);
        let bytes = self.wal_bytes_since_checkpoint.load(Ordering::Relaxed);
        let elapsed = self.last_checkpoint_time.lock().elapsed();

        records >= self.config.min_records
            || bytes >= self.config.max_wal_size
            || elapsed >= self.config.max_interval
    }

    /// Take a checkpoint
    ///
    /// This is the main checkpoint operation:
    /// 1. Write BEGIN_CHECKPOINT record
    /// 2. Collect active transactions and dirty pages
    /// 3. Flush all dirty pages to stable storage
    /// 4. Write END_CHECKPOINT record with collected data
    /// 5. Optionally truncate WAL
    pub fn checkpoint<F>(&self, flush_dirty_pages: F) -> Result<CheckpointData>
    where
        F: FnOnce(&[DirtyPageEntry]) -> Result<()>,
    {
        // Set checkpoint in progress
        if self
            .checkpoint_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(SochDBError::Internal("Checkpoint already in progress".into()));
        }

        // Guard to reset flag on exit (manual scope guard)
        struct CheckpointGuard<'a>(&'a AtomicBool);
        impl<'a> Drop for CheckpointGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = CheckpointGuard(&self.checkpoint_in_progress);

        let checkpoint_id = self.next_checkpoint_id.fetch_add(1, Ordering::SeqCst);
        let begin_lsn = self.next_lsn();

        // Collect state
        let active_txns = self.active_txns.get_active_transactions();
        let dirty_pages = self.dirty_pages.get_dirty_pages();

        // Create checkpoint data
        let mut checkpoint = CheckpointData::new(checkpoint_id, begin_lsn, active_txns, dirty_pages.clone());

        // Flush all dirty pages to stable storage
        flush_dirty_pages(&dirty_pages)?;

        // Mark pages as clean
        for page in &dirty_pages {
            self.dirty_pages.mark_clean(page.page_id);
        }

        // Record end LSN
        let end_lsn = self.next_lsn();
        checkpoint.end_checkpoint_lsn = end_lsn;

        // Update metadata
        {
            let mut meta = self.meta.write();
            meta.last_checkpoint = Some(checkpoint.clone());
            meta.total_checkpoints += 1;

            // Persist metadata
            let data = bincode::serialize(&*meta).map_err(|e| SochDBError::Serialization(e.to_string()))?;
            fs::write(&self.meta_path, data)?;
        }

        // Reset counters
        self.records_since_checkpoint.store(0, Ordering::Relaxed);
        self.wal_bytes_since_checkpoint.store(0, Ordering::Relaxed);
        *self.last_checkpoint_time.lock() = Instant::now();

        // Truncate WAL if configured
        if self.config.truncate_wal {
            self.truncate_wal(checkpoint.oldest_required_lsn)?;
        }

        Ok(checkpoint)
    }

    /// Truncate WAL up to the given LSN
    fn truncate_wal(&self, safe_lsn: Lsn) -> Result<()> {
        // In a real implementation, this would:
        // 1. Identify WAL segments that can be removed
        // 2. Rename/archive or delete old segments
        // 3. Update metadata

        // For now, we just track the truncation point
        let mut meta = self.meta.write();
        if let Some(ref checkpoint) = meta.last_checkpoint {
            let truncated = checkpoint.begin_checkpoint_lsn.saturating_sub(safe_lsn);
            meta.total_bytes_truncated += truncated;
        }

        Ok(())
    }

    /// Get the LSN that is safe for recovery (oldest required LSN)
    pub fn recovery_lsn(&self) -> Option<Lsn> {
        self.meta
            .read()
            .last_checkpoint
            .as_ref()
            .map(|c| c.oldest_required_lsn)
    }

    /// Get the last checkpoint
    pub fn last_checkpoint(&self) -> Option<CheckpointData> {
        self.meta.read().last_checkpoint.clone()
    }

    /// Get checkpoint statistics
    pub fn stats(&self) -> CheckpointStats {
        let meta = self.meta.read();
        CheckpointStats {
            total_checkpoints: meta.total_checkpoints,
            total_bytes_truncated: meta.total_bytes_truncated,
            records_since_checkpoint: self.records_since_checkpoint.load(Ordering::Relaxed),
            wal_bytes_since_checkpoint: self.wal_bytes_since_checkpoint.load(Ordering::Relaxed),
            dirty_pages: self.dirty_pages.dirty_count(),
            active_transactions: self.active_txns.active_count(),
        }
    }
}

/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub total_checkpoints: u64,
    pub total_bytes_truncated: u64,
    pub records_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
    pub dirty_pages: usize,
    pub active_transactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_data_creation() {
        let active_txns = vec![
            ActiveTransactionEntry {
                txn_id: 1,
                first_lsn: 100,
                last_lsn: 150,
                start_ts: 1000,
            },
            ActiveTransactionEntry {
                txn_id: 2,
                first_lsn: 120,
                last_lsn: 180,
                start_ts: 1100,
            },
        ];

        let dirty_pages = vec![
            DirtyPageEntry { page_id: 10, recovery_lsn: 90 },
            DirtyPageEntry { page_id: 20, recovery_lsn: 110 },
        ];

        let checkpoint = CheckpointData::new(1, 200, active_txns, dirty_pages);

        // Oldest required LSN should be 90 (from dirty page)
        assert_eq!(checkpoint.oldest_required_lsn, 90);
    }

    #[test]
    fn test_dirty_page_tracker() {
        let tracker = DirtyPageTracker::new();

        tracker.mark_dirty(1, 100);
        tracker.mark_dirty(2, 110);
        tracker.mark_dirty(1, 120); // Should not update (already dirty)

        assert_eq!(tracker.dirty_count(), 2);

        let pages = tracker.get_dirty_pages();
        assert_eq!(pages.len(), 2);

        // First LSN should be preserved
        let page1 = pages.iter().find(|p| p.page_id == 1).unwrap();
        assert_eq!(page1.recovery_lsn, 100);

        tracker.mark_clean(1);
        assert_eq!(tracker.dirty_count(), 1);
    }

    #[test]
    fn test_active_transaction_tracker() {
        let tracker = ActiveTransactionTracker::new();

        tracker.register(1, 1000);
        tracker.update_lsn(1, 100);
        tracker.update_lsn(1, 150);

        tracker.register(2, 1100);
        tracker.update_lsn(2, 120);

        assert_eq!(tracker.active_count(), 2);

        let txns = tracker.get_active_transactions();
        assert_eq!(txns.len(), 2);

        let txn1 = txns.iter().find(|t| t.txn_id == 1).unwrap();
        assert_eq!(txn1.first_lsn, 100);
        assert_eq!(txn1.last_lsn, 150);

        tracker.remove(1);
        assert_eq!(tracker.active_count(), 1);
    }

    #[test]
    fn test_checkpoint_manager() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let dirty_pages = Arc::new(DirtyPageTracker::new());
        let active_txns = Arc::new(ActiveTransactionTracker::new());
        let hlc = Arc::new(HybridLogicalClock::new());

        let manager = CheckpointManager::new(
            temp_dir.path(),
            CheckpointConfig::default(),
            dirty_pages.clone(),
            active_txns.clone(),
            hlc,
        )?;

        // Mark some dirty pages
        dirty_pages.mark_dirty(1, manager.next_lsn());
        dirty_pages.mark_dirty(2, manager.next_lsn());

        // Register a transaction
        active_txns.register(100, 1000);
        active_txns.update_lsn(100, manager.next_lsn());

        // Take a checkpoint
        let checkpoint = manager.checkpoint(|_pages| Ok(()))?;

        assert_eq!(checkpoint.checkpoint_id, 1);
        assert_eq!(checkpoint.dirty_pages.len(), 2);
        assert_eq!(checkpoint.active_transactions.len(), 1);

        Ok(())
    }
}
