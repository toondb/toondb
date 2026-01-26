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

//! ARIES-Style Crash Recovery Implementation
//!
//! This module implements the Algorithm for Recovery and Isolation Exploiting Semantics (ARIES),
//! a widely-used database recovery algorithm that provides:
//!
//! - **Steal/No-Force Policy**: Uncommitted changes can be written to disk, committed changes
//!   don't need to be force-written before commit.
//! - **WAL-Based Recovery**: All changes logged before being applied.
//! - **Idempotent Replay**: Replay(Replay(S₀, L), L) = Replay(S₀, L)
//! - **Three-Phase Recovery**: Analysis → Redo → Undo
//!
//! ## ARIES Recovery Algorithm
//!
//! ```text
//! 1. Analysis Phase:
//!    - Scan WAL from last checkpoint to end
//!    - Build: active_txns = {txid | TxnBegin seen, no Commit/Abort}
//!    - Build: dirty_pages = {page_id | Updated after checkpoint}
//!
//! 2. Redo Phase:
//!    - For each WAL record r in [checkpoint_lsn, end_lsn]:
//!        if r.page_lsn > page[r.page_id].lsn:
//!            apply(r)  // Only redo if page hasn't seen this update
//!            page[r.page_id].lsn = r.page_lsn
//!
//! 3. Undo Phase:
//!    - For each T ∈ active_txns (losers):
//!        curr_lsn = T.last_lsn
//!        while curr_lsn > 0:
//!            r = WAL[curr_lsn]
//!            if r is Update:
//!                write_CLR(undo(r))  // Compensation log record
//!                undo_on_page(r)
//!            curr_lsn = r.prev_lsn
//! ```
//!
//! ## Page LSN Invariant
//!
//! For each page P: `P.lsn ≤ durable_lsn`
//!
//! This ensures Write-Ahead Logging: before page P is written to disk,
//! all WAL records with lsn ≤ P.lsn must be durable.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use sochdb_core::{
    AriesCheckpointData, AriesDirtyPageEntry, AriesTransactionEntry, Lsn, PageId, Result, TxnId,
    TxnState, WalRecordType,
};

use crate::txn_wal::{TxnWal, TxnWalEntry};

/// LSN for pages to track which updates have been applied
pub type PageLsn = Lsn;

/// Page header with LSN for ARIES recovery
///
/// This structure is stored at the beginning of each database page
/// to enable idempotent recovery.
#[derive(Debug, Clone, Default)]
pub struct PageHeader {
    /// LSN of the last update applied to this page
    pub page_lsn: PageLsn,
    /// Page checksum for corruption detection
    pub checksum: u32,
    /// Page type identifier
    pub page_type: u8,
    /// Reserved for future use
    pub _reserved: [u8; 7],
}

impl PageHeader {
    /// Size of page header in bytes
    pub const SIZE: usize = 20;

    /// Create a new page header with the given LSN
    pub fn new(page_lsn: PageLsn) -> Self {
        Self {
            page_lsn,
            checksum: 0,
            page_type: 0,
            _reserved: [0; 7],
        }
    }

    /// Serialize page header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.page_lsn.to_le_bytes());
        buf[8..12].copy_from_slice(&self.checksum.to_le_bytes());
        buf[12] = self.page_type;
        // Reserved bytes remain zero
        buf
    }

    /// Deserialize page header from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            page_lsn: u64::from_le_bytes(data[0..8].try_into().ok()?),
            checksum: u32::from_le_bytes(data[8..12].try_into().ok()?),
            page_type: data[12],
            _reserved: [0; 7],
        })
    }
}

/// Transaction table entry for ARIES analysis phase
#[derive(Debug, Clone)]
pub struct TransactionTableEntry {
    /// Transaction ID
    pub txn_id: TxnId,
    /// Transaction state
    pub state: TxnState,
    /// LSN of last log record for this transaction
    pub last_lsn: Lsn,
    /// LSN to undo next (follows prev_lsn chain)
    pub undo_next_lsn: Option<Lsn>,
}

/// Dirty page table entry for ARIES analysis phase
#[derive(Debug, Clone)]
pub struct DirtyPageTableEntry {
    /// Page ID
    pub page_id: PageId,
    /// Recovery LSN - first LSN that might have dirtied this page
    /// after the last time it was flushed
    pub rec_lsn: Lsn,
}

/// ARIES recovery manager
///
/// Coordinates crash recovery using the ARIES algorithm.
pub struct AriesRecoveryManager {
    /// Write-ahead log
    wal: Arc<TxnWal>,
    /// Current LSN counter (monotonically increasing)
    current_lsn: AtomicU64,
    /// Transaction table (txn_id -> entry)
    transaction_table: RwLock<HashMap<TxnId, TransactionTableEntry>>,
    /// Dirty page table (page_id -> entry)
    dirty_page_table: RwLock<HashMap<PageId, DirtyPageTableEntry>>,
    /// Last checkpoint LSN
    last_checkpoint_lsn: AtomicU64,
    /// Page LSN cache (for idempotent redo)
    page_lsn_cache: RwLock<HashMap<PageId, PageLsn>>,
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct AriesRecoveryStats {
    /// Total WAL records scanned in analysis phase
    pub analysis_records_scanned: u64,
    /// Number of active transactions found
    pub active_transactions: u64,
    /// Number of dirty pages found
    pub dirty_pages: u64,
    /// Records processed in redo phase
    pub redo_records_processed: u64,
    /// Records skipped in redo (already applied)
    pub redo_records_skipped: u64,
    /// Transactions undone
    pub undo_transactions: u64,
    /// CLRs written during undo
    pub clrs_written: u64,
    /// Total recovery time in microseconds
    pub recovery_time_us: u64,
    /// LSN where analysis started (from checkpoint)
    pub analysis_start_lsn: Lsn,
    /// LSN at end of log
    pub log_end_lsn: Lsn,
}

/// Result of ARIES three-phase recovery
#[derive(Debug)]
pub struct AriesRecoveryResult {
    /// Recovery statistics
    pub stats: AriesRecoveryStats,
    /// Committed writes to apply to storage
    pub committed_writes: Vec<(Vec<u8>, Vec<u8>)>,
    /// Highest LSN seen (for continuing WAL writes)
    pub max_lsn: Lsn,
    /// Highest transaction ID seen (for allocating new txn IDs)
    pub max_txn_id: TxnId,
}

/// Undo action to be performed
#[derive(Debug, Clone)]
pub struct UndoAction {
    /// Transaction ID being undone
    pub txn_id: TxnId,
    /// Key to undo
    pub key: Vec<u8>,
    /// Before-image (value to restore)
    pub before_image: Option<Vec<u8>>,
    /// Page affected
    pub page_id: PageId,
    /// LSN of the original operation being undone
    pub original_lsn: Lsn,
    /// Next LSN to undo after this one
    pub next_undo_lsn: Option<Lsn>,
}

impl AriesRecoveryManager {
    /// Create a new ARIES recovery manager
    pub fn new(wal: Arc<TxnWal>) -> Self {
        Self {
            wal,
            current_lsn: AtomicU64::new(1),
            transaction_table: RwLock::new(HashMap::new()),
            dirty_page_table: RwLock::new(HashMap::new()),
            last_checkpoint_lsn: AtomicU64::new(0),
            page_lsn_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Allocate a new LSN (thread-safe, monotonically increasing)
    pub fn allocate_lsn(&self) -> Lsn {
        self.current_lsn.fetch_add(1, Ordering::SeqCst)
    }

    /// Get current LSN without incrementing
    pub fn current_lsn(&self) -> Lsn {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Set current LSN (used during recovery)
    pub fn set_current_lsn(&self, lsn: Lsn) {
        self.current_lsn.store(lsn, Ordering::SeqCst);
    }

    /// Get page LSN from cache (or 0 if not cached)
    pub fn get_page_lsn(&self, page_id: PageId) -> PageLsn {
        self.page_lsn_cache
            .read()
            .get(&page_id)
            .copied()
            .unwrap_or(0)
    }

    /// Set page LSN in cache
    pub fn set_page_lsn(&self, page_id: PageId, lsn: PageLsn) {
        self.page_lsn_cache.write().insert(page_id, lsn);
    }

    /// Track transaction in transaction table
    pub fn track_transaction(&self, txn_id: TxnId, last_lsn: Lsn) {
        let mut table = self.transaction_table.write();
        let entry = table
            .entry(txn_id)
            .or_insert_with(|| TransactionTableEntry {
                txn_id,
                state: TxnState::Active,
                last_lsn: 0,
                undo_next_lsn: None,
            });
        entry.last_lsn = last_lsn;
    }

    /// Mark transaction as committed
    pub fn commit_transaction(&self, txn_id: TxnId) {
        let mut table = self.transaction_table.write();
        if let Some(entry) = table.get_mut(&txn_id) {
            entry.state = TxnState::Committed;
        }
    }

    /// Mark transaction as aborted
    pub fn abort_transaction(&self, txn_id: TxnId) {
        let mut table = self.transaction_table.write();
        if let Some(entry) = table.get_mut(&txn_id) {
            entry.state = TxnState::Aborted;
        }
    }

    /// Track dirty page
    pub fn track_dirty_page(&self, page_id: PageId, lsn: Lsn) {
        let mut table = self.dirty_page_table.write();
        table.entry(page_id).or_insert_with(|| DirtyPageTableEntry {
            page_id,
            rec_lsn: lsn,
        });
    }

    /// Mark page as clean (after flush)
    pub fn mark_page_clean(&self, page_id: PageId) {
        self.dirty_page_table.write().remove(&page_id);
    }

    /// Perform full ARIES three-phase recovery
    ///
    /// 1. Analysis: Determine transaction and dirty page state
    /// 2. Redo: Reapply all logged actions (idempotent via LSN check)
    /// 3. Undo: Roll back uncommitted transactions with CLRs
    pub fn recover(&self) -> Result<AriesRecoveryResult> {
        let start_time = std::time::Instant::now();
        let mut stats = AriesRecoveryStats::default();

        // Phase 1: Analysis
        let (wal_records, analysis_result) = self.analysis_phase(&mut stats)?;

        // Phase 2: Redo
        let redo_result = self.redo_phase(&wal_records, &analysis_result, &mut stats)?;

        // Phase 3: Undo
        let _undo_result = self.undo_phase(&wal_records, &analysis_result, &mut stats)?;

        stats.recovery_time_us = start_time.elapsed().as_micros() as u64;

        Ok(AriesRecoveryResult {
            stats,
            committed_writes: redo_result.committed_writes,
            max_lsn: redo_result.max_lsn,
            max_txn_id: redo_result.max_txn_id,
        })
    }

    /// Analysis Phase: Determine state at crash
    ///
    /// Scans WAL from last checkpoint to build:
    /// - Transaction table: active transactions and their last LSN
    /// - Dirty page table: pages that may need redo
    fn analysis_phase(
        &self,
        stats: &mut AriesRecoveryStats,
    ) -> Result<(Vec<TxnWalEntry>, AnalysisResult)> {
        let checkpoint_lsn = self.last_checkpoint_lsn.load(Ordering::SeqCst);
        stats.analysis_start_lsn = checkpoint_lsn;

        let mut transaction_table: HashMap<TxnId, TransactionTableEntry> = HashMap::new();
        let mut dirty_pages: HashMap<PageId, DirtyPageTableEntry> = HashMap::new();
        let mut wal_records = Vec::new();
        let mut max_lsn: Lsn = 0;
        let mut max_txn_id: TxnId = 0;

        // Read all WAL records
        let (_writes, _) = self.wal.replay_for_recovery()?;

        // For now, we simulate WAL records from the recovery data
        // In a full implementation, we'd read raw WAL records with LSNs
        let mut lsn: Lsn = checkpoint_lsn.max(1);

        // Re-read WAL to build proper record list
        self.wal.replay(|entry| {
            let current_lsn = lsn;
            lsn += 1;
            stats.analysis_records_scanned += 1;

            if current_lsn > max_lsn {
                max_lsn = current_lsn;
            }
            if entry.txn_id > max_txn_id {
                max_txn_id = entry.txn_id;
            }

            match entry.record_type {
                WalRecordType::TxnBegin => {
                    transaction_table.insert(
                        entry.txn_id,
                        TransactionTableEntry {
                            txn_id: entry.txn_id,
                            state: TxnState::Active,
                            last_lsn: current_lsn,
                            undo_next_lsn: None,
                        },
                    );
                }
                WalRecordType::Data | WalRecordType::PageUpdate => {
                    // Update transaction's last LSN
                    if let Some(txn_entry) = transaction_table.get_mut(&entry.txn_id) {
                        txn_entry.last_lsn = current_lsn;
                        txn_entry.undo_next_lsn = Some(current_lsn);
                    }
                    // Track dirty page (using hash of key as page ID for simplicity)
                    let page_id = self.key_to_page_id(&entry.key);
                    dirty_pages
                        .entry(page_id)
                        .or_insert_with(|| DirtyPageTableEntry {
                            page_id,
                            rec_lsn: current_lsn,
                        });
                }
                WalRecordType::TxnCommit => {
                    if let Some(txn_entry) = transaction_table.get_mut(&entry.txn_id) {
                        txn_entry.state = TxnState::Committed;
                    }
                }
                WalRecordType::TxnAbort => {
                    if let Some(txn_entry) = transaction_table.get_mut(&entry.txn_id) {
                        txn_entry.state = TxnState::Aborted;
                    }
                }
                WalRecordType::Checkpoint => {
                    // Fuzzy checkpoint - just a marker
                }
                WalRecordType::CheckpointEnd => {
                    // Process checkpoint data if available
                    // (would parse checkpoint_data from entry.value)
                }
                WalRecordType::CompensationLogRecord => {
                    // CLRs are redo-only, update undo_next_lsn
                    if let Some(txn_entry) = transaction_table.get_mut(&entry.txn_id) {
                        txn_entry.last_lsn = current_lsn;
                        // CLR's undo_next_lsn would skip past the compensated operation
                    }
                }
                WalRecordType::SchemaChange => {
                    // Schema changes treated like data for recovery
                }
            }

            wal_records.push(entry);
            Ok(())
        })?;

        stats.log_end_lsn = max_lsn;
        stats.active_transactions = transaction_table
            .values()
            .filter(|t| t.state == TxnState::Active)
            .count() as u64;
        stats.dirty_pages = dirty_pages.len() as u64;

        Ok((
            wal_records,
            AnalysisResult {
                transaction_table,
                dirty_pages,
                max_lsn,
                max_txn_id,
            },
        ))
    }

    /// Redo Phase: Reapply logged actions (idempotent)
    ///
    /// For each WAL record from analysis start to end:
    /// - If record's LSN > page's LSN, apply the update
    /// - This ensures idempotent recovery: applying twice has same effect as once
    fn redo_phase(
        &self,
        wal_records: &[TxnWalEntry],
        analysis: &AnalysisResult,
        stats: &mut AriesRecoveryStats,
    ) -> Result<RedoResult> {
        let mut committed_writes = Vec::new();
        let mut lsn = stats.analysis_start_lsn.max(1);

        for entry in wal_records {
            let record_lsn = lsn;
            lsn += 1;

            match entry.record_type {
                WalRecordType::Data
                | WalRecordType::PageUpdate
                | WalRecordType::CompensationLogRecord => {
                    let page_id = self.key_to_page_id(&entry.key);
                    let page_lsn = self.get_page_lsn(page_id);

                    // Idempotent redo: only apply if record LSN > page LSN
                    if record_lsn > page_lsn {
                        // Check if transaction was committed
                        let is_committed = analysis
                            .transaction_table
                            .get(&entry.txn_id)
                            .map(|t| t.state == TxnState::Committed)
                            .unwrap_or(false);

                        if is_committed {
                            committed_writes.push((entry.key.clone(), entry.value.clone()));
                        }

                        // Update page LSN
                        self.set_page_lsn(page_id, record_lsn);
                        stats.redo_records_processed += 1;
                    } else {
                        stats.redo_records_skipped += 1;
                    }
                }
                _ => {
                    // Non-data records don't need redo
                }
            }
        }

        Ok(RedoResult {
            committed_writes,
            max_lsn: analysis.max_lsn,
            max_txn_id: analysis.max_txn_id,
        })
    }

    /// Undo Phase: Roll back uncommitted transactions
    ///
    /// For each transaction still active at crash:
    /// - Follow prev_lsn chain to undo operations
    /// - Write CLR for each undo operation
    fn undo_phase(
        &self,
        wal_records: &[TxnWalEntry],
        analysis: &AnalysisResult,
        stats: &mut AriesRecoveryStats,
    ) -> Result<UndoResult> {
        let mut undo_actions = Vec::new();

        // Find all "loser" transactions (active at crash)
        let loser_txns: Vec<_> = analysis
            .transaction_table
            .values()
            .filter(|t| t.state == TxnState::Active)
            .cloned()
            .collect();

        stats.undo_transactions = loser_txns.len() as u64;

        // For each loser transaction, traverse prev_lsn chain
        for txn_entry in &loser_txns {
            let mut current_lsn = txn_entry.undo_next_lsn;

            while let Some(undo_lsn) = current_lsn {
                // Find the WAL record at this LSN
                // In a real implementation, we'd have an LSN -> offset index
                let record_idx = (undo_lsn as usize).saturating_sub(1);
                if record_idx < wal_records.len() {
                    let record = &wal_records[record_idx];

                    if record.txn_id == txn_entry.txn_id {
                        match record.record_type {
                            WalRecordType::Data | WalRecordType::PageUpdate => {
                                // Create undo action
                                let page_id = self.key_to_page_id(&record.key);
                                let prev_lsn = if undo_lsn > 1 {
                                    Some(undo_lsn - 1)
                                } else {
                                    None
                                };

                                undo_actions.push(UndoAction {
                                    txn_id: record.txn_id,
                                    key: record.key.clone(),
                                    before_image: None, // Would come from undo_info in record
                                    page_id,
                                    original_lsn: undo_lsn,
                                    next_undo_lsn: prev_lsn,
                                });

                                // Write CLR to WAL for the undo operation
                                // The CLR contains: original LSN, undo_next_lsn (prev_lsn)
                                let clr_lsn = self.wal.append_clr(
                                    record.txn_id,
                                    undo_lsn,
                                    prev_lsn,
                                    &record.key, // Undo data - key being undone
                                )?;
                                stats.clrs_written += 1;

                                // Update page LSN after CLR
                                self.set_page_lsn(page_id, clr_lsn);
                            }
                            WalRecordType::CompensationLogRecord => {
                                // CLRs are redo-only, skip to their undo_next_lsn
                                // Extract undo_next_lsn from the CLR's key field
                                if record.key.len() >= 8 {
                                    let undo_next = u64::from_le_bytes(
                                        record.key[0..8].try_into().unwrap_or([0; 8]),
                                    );
                                    current_lsn =
                                        if undo_next > 0 { Some(undo_next) } else { None };
                                    continue;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Move to previous LSN in chain
                current_lsn = if undo_lsn > 1 {
                    Some(undo_lsn - 1)
                } else {
                    None
                };
            }

            // Write abort record for the loser transaction
            self.wal.abort_transaction(txn_entry.txn_id)?;
        }

        // Final fsync to ensure CLRs and aborts are durable
        self.wal.sync()?;

        Ok(UndoResult { undo_actions })
    }

    /// Map key to page ID (simple hash for demo)
    fn key_to_page_id(&self, key: &[u8]) -> PageId {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Write a checkpoint
    ///
    /// Checkpoints speed up recovery by limiting how far back we need to scan.
    pub fn write_checkpoint(&self) -> Result<Lsn> {
        let checkpoint_lsn = self.allocate_lsn();

        // Build checkpoint data
        let active_txns: Vec<_> = self
            .transaction_table
            .read()
            .values()
            .filter(|t| t.state == TxnState::Active)
            .map(|t| AriesTransactionEntry {
                txn_id: t.txn_id,
                state: t.state,
                last_lsn: t.last_lsn,
                undo_next_lsn: t.undo_next_lsn,
            })
            .collect();

        let dirty_pages: Vec<_> = self
            .dirty_page_table
            .read()
            .values()
            .map(|p| AriesDirtyPageEntry {
                page_id: p.page_id,
                rec_lsn: p.rec_lsn,
            })
            .collect();

        let checkpoint_data = AriesCheckpointData {
            active_transactions: active_txns,
            dirty_pages,
            begin_checkpoint_lsn: checkpoint_lsn,
        };

        // Write checkpoint begin
        self.wal.write_checkpoint()?;

        // Serialize and write checkpoint data
        let serialized = self.serialize_checkpoint_data(&checkpoint_data);
        self.wal.write_checkpoint_end(&serialized)?;

        self.last_checkpoint_lsn
            .store(checkpoint_lsn, Ordering::SeqCst);

        Ok(checkpoint_lsn)
    }

    /// Write a fuzzy checkpoint (non-blocking)
    ///
    /// Fuzzy checkpoints allow concurrent operations while capturing state:
    /// 1. Write BEGIN_CHECKPOINT at current LSN
    /// 2. Capture active transaction table (snapshot)
    /// 3. Capture dirty page table (snapshot)
    /// 4. Write END_CHECKPOINT with captured data
    ///
    /// Recovery uses these snapshots to minimize WAL scan range.
    pub fn write_fuzzy_checkpoint(&self) -> Result<Lsn> {
        let checkpoint_begin_lsn = self.allocate_lsn();

        // 1. Write BEGIN_CHECKPOINT (marks start of checkpoint)
        self.wal.write_checkpoint()?;

        // 2. Capture active transaction table (atomic snapshot)
        let active_txns: Vec<AriesTransactionEntry> = self
            .transaction_table
            .read()
            .values()
            .filter(|t| t.state == TxnState::Active)
            .map(|t| AriesTransactionEntry {
                txn_id: t.txn_id,
                state: t.state,
                last_lsn: t.last_lsn,
                undo_next_lsn: t.undo_next_lsn,
            })
            .collect();

        // 3. Capture dirty page table (atomic snapshot)
        let dirty_pages: Vec<AriesDirtyPageEntry> = self
            .dirty_page_table
            .read()
            .values()
            .map(|p| AriesDirtyPageEntry {
                page_id: p.page_id,
                rec_lsn: p.rec_lsn,
            })
            .collect();

        // 4. Build and serialize checkpoint data
        let checkpoint_data = AriesCheckpointData {
            active_transactions: active_txns,
            dirty_pages,
            begin_checkpoint_lsn: checkpoint_begin_lsn,
        };

        let serialized = self.serialize_checkpoint_data(&checkpoint_data);

        // 5. Write END_CHECKPOINT with captured data
        self.wal.write_checkpoint_end(&serialized)?;

        // 6. fsync to ensure checkpoint is durable
        self.wal.sync()?;

        self.last_checkpoint_lsn
            .store(checkpoint_begin_lsn, Ordering::SeqCst);

        Ok(checkpoint_begin_lsn)
    }

    /// Serialize checkpoint data to bytes
    fn serialize_checkpoint_data(&self, data: &AriesCheckpointData) -> Vec<u8> {
        // Simple binary serialization:
        // [num_txns: u32][txn_entries...][num_pages: u32][page_entries...][begin_lsn: u64]
        let mut buf = Vec::new();

        // Serialize transaction entries
        buf.extend_from_slice(&(data.active_transactions.len() as u32).to_le_bytes());
        for txn in &data.active_transactions {
            buf.extend_from_slice(&txn.txn_id.to_le_bytes());
            buf.push(match txn.state {
                TxnState::Active => 0,
                TxnState::Committed => 1,
                TxnState::Aborted => 2,
            });
            buf.extend_from_slice(&txn.last_lsn.to_le_bytes());
            buf.extend_from_slice(&txn.undo_next_lsn.unwrap_or(0).to_le_bytes());
        }

        // Serialize dirty page entries
        buf.extend_from_slice(&(data.dirty_pages.len() as u32).to_le_bytes());
        for page in &data.dirty_pages {
            buf.extend_from_slice(&page.page_id.to_le_bytes());
            buf.extend_from_slice(&page.rec_lsn.to_le_bytes());
        }

        // Serialize begin checkpoint LSN
        buf.extend_from_slice(&data.begin_checkpoint_lsn.to_le_bytes());

        buf
    }

    /// Deserialize checkpoint data from bytes
    #[allow(dead_code)]
    fn deserialize_checkpoint_data(&self, data: &[u8]) -> Option<AriesCheckpointData> {
        if data.len() < 4 {
            return None;
        }

        let mut offset = 0;

        // Read transaction entries
        let num_txns = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        offset += 4;

        let mut active_transactions = Vec::with_capacity(num_txns);
        for _ in 0..num_txns {
            if offset + 25 > data.len() {
                return None;
            }

            let txn_id = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;

            let state = match data[offset] {
                0 => TxnState::Active,
                1 => TxnState::Committed,
                2 => TxnState::Aborted,
                _ => return None,
            };
            offset += 1;

            let last_lsn = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;

            let undo_next = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;

            active_transactions.push(AriesTransactionEntry {
                txn_id,
                state,
                last_lsn,
                undo_next_lsn: if undo_next > 0 { Some(undo_next) } else { None },
            });
        }

        // Read dirty page entries
        if offset + 4 > data.len() {
            return None;
        }
        let num_pages = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        offset += 4;

        let mut dirty_pages = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            if offset + 16 > data.len() {
                return None;
            }

            let page_id = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;

            let rec_lsn = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;

            dirty_pages.push(AriesDirtyPageEntry { page_id, rec_lsn });
        }

        // Read begin checkpoint LSN
        if offset + 8 > data.len() {
            return None;
        }
        let begin_checkpoint_lsn = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);

        Some(AriesCheckpointData {
            active_transactions,
            dirty_pages,
            begin_checkpoint_lsn,
        })
    }
}

/// Result of analysis phase
#[derive(Debug)]
#[allow(dead_code)]
struct AnalysisResult {
    transaction_table: HashMap<TxnId, TransactionTableEntry>,
    dirty_pages: HashMap<PageId, DirtyPageTableEntry>,
    max_lsn: Lsn,
    max_txn_id: TxnId,
}

/// Result of redo phase
#[derive(Debug)]
struct RedoResult {
    committed_writes: Vec<(Vec<u8>, Vec<u8>)>,
    max_lsn: Lsn,
    max_txn_id: TxnId,
}

/// Result of undo phase
#[derive(Debug)]
#[allow(dead_code)]
struct UndoResult {
    undo_actions: Vec<UndoAction>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_page_header_roundtrip() {
        let header = PageHeader::new(12345);
        let bytes = header.to_bytes();
        let recovered = PageHeader::from_bytes(&bytes).unwrap();
        assert_eq!(recovered.page_lsn, 12345);
    }

    #[test]
    fn test_lsn_allocation() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(TxnWal::new(dir.path().join("test.wal")).unwrap());
        let recovery = AriesRecoveryManager::new(wal);

        let lsn1 = recovery.allocate_lsn();
        let lsn2 = recovery.allocate_lsn();
        let lsn3 = recovery.allocate_lsn();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(lsn3, 3);
    }

    #[test]
    fn test_page_lsn_tracking() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(TxnWal::new(dir.path().join("test.wal")).unwrap());
        let recovery = AriesRecoveryManager::new(wal);

        assert_eq!(recovery.get_page_lsn(100), 0);

        recovery.set_page_lsn(100, 42);
        assert_eq!(recovery.get_page_lsn(100), 42);

        recovery.set_page_lsn(100, 50);
        assert_eq!(recovery.get_page_lsn(100), 50);
    }

    #[test]
    fn test_transaction_tracking() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(TxnWal::new(dir.path().join("test.wal")).unwrap());
        let recovery = AriesRecoveryManager::new(wal);

        recovery.track_transaction(1, 10);
        recovery.track_transaction(1, 20);

        {
            let table = recovery.transaction_table.read();
            let entry = table.get(&1).unwrap();
            assert_eq!(entry.last_lsn, 20);
            assert_eq!(entry.state, TxnState::Active);
        }

        recovery.commit_transaction(1);
        {
            let table = recovery.transaction_table.read();
            let entry = table.get(&1).unwrap();
            assert_eq!(entry.state, TxnState::Committed);
        }
    }

    #[test]
    fn test_recovery_with_simple_workload() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Create WAL with some transactions
        {
            let wal = TxnWal::new(&wal_path).unwrap();

            // Committed transaction
            let txn1 = wal.begin_transaction().unwrap();
            wal.write(txn1, b"key1".to_vec(), b"value1".to_vec())
                .unwrap();
            wal.write(txn1, b"key2".to_vec(), b"value2".to_vec())
                .unwrap();
            wal.commit_transaction(txn1).unwrap();

            // Uncommitted transaction (simulates crash)
            let txn2 = wal.begin_transaction().unwrap();
            wal.write(txn2, b"key3".to_vec(), b"value3".to_vec())
                .unwrap();
            // No commit!
        }

        // Recover
        {
            let wal = Arc::new(TxnWal::new(&wal_path).unwrap());
            let recovery = AriesRecoveryManager::new(wal);
            let result = recovery.recover().unwrap();

            // Should have 2 committed writes from txn1
            assert_eq!(result.committed_writes.len(), 2);
            assert_eq!(result.stats.active_transactions, 1); // txn2 was uncommitted
            assert_eq!(result.stats.undo_transactions, 1);
        }
    }

    #[test]
    fn test_fuzzy_checkpoint() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("checkpoint_test.wal");

        let wal = Arc::new(TxnWal::new(&wal_path).unwrap());
        let recovery = AriesRecoveryManager::new(wal);

        // Track some transactions
        recovery.track_transaction(1, 10);
        recovery.track_transaction(2, 20);

        // Track some dirty pages
        recovery.track_dirty_page(100, 15);
        recovery.track_dirty_page(200, 25);

        // Write fuzzy checkpoint
        let checkpoint_lsn = recovery.write_fuzzy_checkpoint().unwrap();
        assert!(checkpoint_lsn > 0);

        // Verify checkpoint LSN was recorded
        assert_eq!(
            recovery.last_checkpoint_lsn.load(Ordering::SeqCst),
            checkpoint_lsn
        );
    }

    #[test]
    fn test_checkpoint_serialization() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("serialize_test.wal");

        let wal = Arc::new(TxnWal::new(&wal_path).unwrap());
        let recovery = AriesRecoveryManager::new(wal);

        // Create checkpoint data
        let checkpoint_data = AriesCheckpointData {
            active_transactions: vec![
                AriesTransactionEntry {
                    txn_id: 1,
                    state: TxnState::Active,
                    last_lsn: 100,
                    undo_next_lsn: Some(50),
                },
                AriesTransactionEntry {
                    txn_id: 2,
                    state: TxnState::Committed,
                    last_lsn: 200,
                    undo_next_lsn: None,
                },
            ],
            dirty_pages: vec![
                AriesDirtyPageEntry {
                    page_id: 10,
                    rec_lsn: 50,
                },
                AriesDirtyPageEntry {
                    page_id: 20,
                    rec_lsn: 75,
                },
            ],
            begin_checkpoint_lsn: 1000,
        };

        // Serialize
        let serialized = recovery.serialize_checkpoint_data(&checkpoint_data);
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized = recovery.deserialize_checkpoint_data(&serialized).unwrap();

        // Verify
        assert_eq!(deserialized.active_transactions.len(), 2);
        assert_eq!(deserialized.dirty_pages.len(), 2);
        assert_eq!(deserialized.begin_checkpoint_lsn, 1000);

        assert_eq!(deserialized.active_transactions[0].txn_id, 1);
        assert_eq!(deserialized.active_transactions[0].state, TxnState::Active);
        assert_eq!(
            deserialized.active_transactions[1].state,
            TxnState::Committed
        );

        assert_eq!(deserialized.dirty_pages[0].page_id, 10);
        assert_eq!(deserialized.dirty_pages[1].rec_lsn, 75);
    }

    #[test]
    fn test_clr_append() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("clr_test.wal");

        let wal = TxnWal::new(&wal_path).unwrap();

        // Begin a transaction
        let txn_id = wal.begin_transaction().unwrap();

        // Write some data
        wal.write(txn_id, b"key1".to_vec(), b"value1".to_vec())
            .unwrap();

        // Write a CLR (compensation log record)
        let clr_lsn = wal
            .append_clr(
                txn_id,
                1,       // original LSN
                Some(0), // undo_next_lsn (no more to undo)
                b"undo_data",
            )
            .unwrap();

        assert!(clr_lsn > 0);

        // Abort the transaction
        wal.abort_transaction(txn_id).unwrap();
    }

    #[test]
    fn test_recovery_with_clrs() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("clr_recovery_test.wal");

        // Phase 1: Create WAL with transaction and CLRs
        {
            let wal = TxnWal::new(&wal_path).unwrap();

            // Committed transaction
            let txn1 = wal.begin_transaction().unwrap();
            wal.write(txn1, b"key1".to_vec(), b"value1".to_vec())
                .unwrap();
            wal.commit_transaction(txn1).unwrap();

            // Transaction with CLR (simulates partial undo)
            let txn2 = wal.begin_transaction().unwrap();
            wal.write(txn2, b"key2".to_vec(), b"value2".to_vec())
                .unwrap();
            // CLR indicates this was undone
            wal.append_clr(txn2, 2, Some(0), b"key2").unwrap();
            wal.abort_transaction(txn2).unwrap();
        }

        // Phase 2: Recover
        {
            let wal = Arc::new(TxnWal::new(&wal_path).unwrap());
            let recovery = AriesRecoveryManager::new(wal);
            let result = recovery.recover().unwrap();

            // Only committed transaction's writes should be recovered
            assert_eq!(result.committed_writes.len(), 1);
            assert_eq!(result.committed_writes[0].0, b"key1".to_vec());
        }
    }
}
