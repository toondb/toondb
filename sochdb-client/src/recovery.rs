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

//! Crash Recovery API
//!
//! Provides WAL verification and recovery operations.

use std::time::Instant;

use crate::connection::SochConnection;
use crate::error::{ClientError, Result};

/// Recovery status
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStatus {
    /// No recovery needed
    Clean,
    /// Recovery completed successfully
    Recovered { replayed_entries: u64 },
    /// Recovery failed
    Failed { reason: String },
    /// Corruption detected
    Corrupted { details: String },
}

/// WAL verification result
#[derive(Debug, Clone)]
pub struct WalVerificationResult {
    pub is_valid: bool,
    pub total_entries: u64,
    pub valid_entries: u64,
    pub corrupted_entries: u64,
    pub last_valid_lsn: u64,
    pub checksum_errors: Vec<ChecksumError>,
}

/// Checksum error detail
#[derive(Debug, Clone)]
pub struct ChecksumError {
    pub lsn: u64,
    pub expected: u64,
    pub actual: u64,
    pub entry_type: String,
}

/// Recovery manager
pub struct RecoveryManager<'a> {
    conn: &'a SochConnection,
}

impl<'a> RecoveryManager<'a> {
    /// Create new recovery manager
    pub fn new(conn: &'a SochConnection) -> Self {
        Self { conn }
    }

    /// Check if recovery is needed
    pub fn needs_recovery(&self) -> bool {
        self.conn.storage.needs_recovery()
    }

    /// Get last checkpoint LSN
    pub fn last_checkpoint_lsn(&self) -> u64 {
        self.conn.storage.last_checkpoint_lsn()
    }

    /// Get current WAL LSN
    pub fn current_lsn(&self) -> u64 {
        self.conn.storage.current_lsn()
    }

    /// Verify WAL integrity
    pub fn verify_wal(&self) -> Result<WalVerificationResult> {
        let result = self.conn.storage.verify_wal()?;

        Ok(WalVerificationResult {
            is_valid: result.corrupted_entries == 0,
            total_entries: result.total_entries,
            valid_entries: result.valid_entries,
            corrupted_entries: result.corrupted_entries,
            last_valid_lsn: result.last_valid_lsn,
            checksum_errors: result
                .checksum_errors
                .into_iter()
                .map(|e| ChecksumError {
                    lsn: e.lsn,
                    expected: e.expected,
                    actual: e.actual,
                    entry_type: e.entry_type,
                })
                .collect(),
        })
    }

    /// Perform recovery
    pub fn recover(&self) -> Result<RecoveryStatus> {
        if !self.needs_recovery() {
            return Ok(RecoveryStatus::Clean);
        }

        // Verify WAL first
        let verification = self.verify_wal()?;
        if !verification.is_valid {
            return Ok(RecoveryStatus::Corrupted {
                details: format!(
                    "{} corrupted entries found, last valid LSN: {}",
                    verification.corrupted_entries, verification.last_valid_lsn,
                ),
            });
        }

        // Replay WAL from last checkpoint
        let replayed = self.conn.storage.replay_wal_from_checkpoint()?;

        Ok(RecoveryStatus::Recovered {
            replayed_entries: replayed,
        })
    }

    /// Force checkpoint
    pub fn checkpoint(&self) -> Result<CheckpointResult> {
        let start = Instant::now();

        let lsn = self.conn.storage.force_checkpoint()?;

        Ok(CheckpointResult {
            checkpoint_lsn: lsn,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Truncate WAL up to LSN (after checkpoint)
    pub fn truncate_wal(&self, up_to_lsn: u64) -> Result<TruncateResult> {
        let bytes_freed = self.conn.storage.truncate_wal(up_to_lsn)?;

        Ok(TruncateResult {
            up_to_lsn,
            bytes_freed,
        })
    }

    /// Get WAL statistics
    pub fn wal_stats(&self) -> WalStats {
        let stats = self.conn.storage.wal_stats();
        WalStats {
            total_size_bytes: stats.total_size_bytes,
            active_size_bytes: stats.active_size_bytes,
            archived_size_bytes: stats.archived_size_bytes,
            oldest_entry_lsn: stats.oldest_entry_lsn,
            newest_entry_lsn: stats.newest_entry_lsn,
            entry_count: stats.entry_count,
        }
    }
}

/// Checkpoint result
#[derive(Debug, Clone)]
pub struct CheckpointResult {
    pub checkpoint_lsn: u64,
    pub duration_ms: u64,
}

/// WAL truncate result
#[derive(Debug, Clone)]
pub struct TruncateResult {
    pub up_to_lsn: u64,
    pub bytes_freed: u64,
}

/// WAL statistics
#[derive(Debug, Clone)]
pub struct WalStats {
    pub total_size_bytes: u64,
    pub active_size_bytes: u64,
    pub archived_size_bytes: u64,
    pub oldest_entry_lsn: u64,
    pub newest_entry_lsn: u64,
    pub entry_count: u64,
}

/// Recovery methods on connection
impl SochConnection {
    /// Create recovery manager
    pub fn recovery(&self) -> RecoveryManager<'_> {
        RecoveryManager::new(self)
    }

    /// Quick check if recovery needed
    pub fn needs_recovery(&self) -> bool {
        self.storage.needs_recovery()
    }

    /// Quick recover
    pub fn recover(&self) -> Result<RecoveryStatus> {
        self.recovery().recover()
    }

    /// Force checkpoint
    pub fn checkpoint(&self) -> Result<CheckpointResult> {
        self.recovery().checkpoint()
    }
}

/// Open database with automatic recovery
pub fn open_with_recovery(path: &str) -> Result<SochConnection> {
    let conn = SochConnection::open(path)?;

    // Automatic recovery if needed
    match conn.recover()? {
        RecoveryStatus::Clean => {
            // No recovery needed
        }
        RecoveryStatus::Recovered {
            replayed_entries: _,
        } => {
            // Recovery completed
        }
        RecoveryStatus::Failed { reason } => {
            return Err(ClientError::Storage(format!("Recovery failed: {}", reason)));
        }
        RecoveryStatus::Corrupted { details } => {
            return Err(ClientError::Storage(format!(
                "Corruption detected: {}",
                details
            )));
        }
    }

    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_status() {
        let clean = RecoveryStatus::Clean;
        assert_eq!(clean, RecoveryStatus::Clean);

        let recovered = RecoveryStatus::Recovered {
            replayed_entries: 100,
        };
        match recovered {
            RecoveryStatus::Recovered { replayed_entries } => {
                assert_eq!(replayed_entries, 100);
            }
            _ => panic!("Expected Recovered status"),
        }
    }

    #[test]
    fn test_recovery_manager() {
        let conn = SochConnection::open("./test").unwrap();
        let recovery = conn.recovery();

        // Should not need recovery on fresh db
        assert!(!recovery.needs_recovery());
    }

    #[test]
    fn test_checkpoint() {
        let conn = SochConnection::open("./test").unwrap();
        let result = conn.checkpoint().unwrap();

        // Fields are u64, just verify they exist
        let _ = result.checkpoint_lsn;
        let _ = result.duration_ms;
    }

    #[test]
    fn test_wal_verification() {
        let conn = SochConnection::open("./test").unwrap();
        let result = conn.recovery().verify_wal().unwrap();

        assert!(result.is_valid);
        assert_eq!(result.corrupted_entries, 0);
    }

    #[test]
    fn test_wal_stats() {
        let conn = SochConnection::open("./test").unwrap();
        let stats = conn.recovery().wal_stats();

        // Fields are u64, just verify they exist
        let _ = stats.total_size_bytes;
        let _ = stats.entry_count;
    }
}
