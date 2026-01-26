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

//! StorageEngine Trait Abstraction (Task 1)
//!
//! Decouples the query engine from concrete storage implementations.
//! Enables pluggable backends: LSMTree for compatibility, Lscs for columnar TOON workloads.
//!
//! ## I/O Reduction Model
//!
//! ```text
//! Traditional Row I/O: O(N × K) where N=rows, K=total columns
//! Columnar I/O:        O(N × k) where k=selected columns
//!
//! For k/K = 0.2 (typical TOON projection):
//! I/O_reduction = 1 - (k/K) = 80%
//! ```

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use sochdb_core::{Result, SochDBError, SochRow, SochValue};

/// Transaction handle for ACID operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TxnHandle {
    /// Transaction ID
    pub txn_id: u64,
    /// Snapshot version for MVCC reads
    pub snapshot_version: u64,
    /// Transaction start timestamp
    pub start_ts: u64,
}

impl TxnHandle {
    /// Create a new transaction handle
    pub fn new(txn_id: u64, snapshot_version: u64) -> Self {
        let start_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        Self {
            txn_id,
            snapshot_version,
            start_ts,
        }
    }
}

/// Column identifier
pub type ColumnId = u32;

/// Row identifier
pub type RowId = u64;

/// A row of data with column values
#[derive(Debug, Clone)]
pub struct Row {
    /// Row ID
    pub id: RowId,
    /// Column values (indexed by column position)
    pub values: Vec<Option<Vec<u8>>>,
    /// Transaction start timestamp (MVCC)
    pub txn_start: u64,
    /// Transaction end timestamp (MVCC, 0 = active)
    pub txn_end: u64,
}

impl Row {
    /// Create a new row
    pub fn new(id: RowId, values: Vec<Option<Vec<u8>>>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        Self {
            id,
            values,
            txn_start: now,
            txn_end: 0,
        }
    }

    /// Check if row is visible at given snapshot
    pub fn is_visible(&self, snapshot_version: u64) -> bool {
        self.txn_start <= snapshot_version && (self.txn_end == 0 || self.txn_end > snapshot_version)
    }

    /// Convert to SochRow
    pub fn to_soch_row(&self, _schema: &[String]) -> SochRow {
        let values: Vec<SochValue> = self
            .values
            .iter()
            .map(|v| match v {
                Some(bytes) => {
                    // Try to interpret as string first
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        SochValue::Text(s.to_string())
                    } else {
                        SochValue::Binary(bytes.clone())
                    }
                }
                None => SochValue::Null,
            })
            .collect();
        SochRow::new(values)
    }
}

/// Iterator over columns
pub struct ColumnIterator {
    /// Current position
    position: usize,
    /// Rows with projected columns
    rows: Vec<Row>,
    /// Column IDs being iterated
    column_ids: Vec<ColumnId>,
}

impl ColumnIterator {
    /// Create a new column iterator
    pub fn new(rows: Vec<Row>, column_ids: Vec<ColumnId>) -> Self {
        Self {
            position: 0,
            rows,
            column_ids,
        }
    }

    /// Get column IDs
    pub fn column_ids(&self) -> &[ColumnId] {
        &self.column_ids
    }
}

impl Iterator for ColumnIterator {
    type Item = Row;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.rows.len() {
            let row = self.rows[self.position].clone();
            self.position += 1;
            Some(row)
        } else {
            None
        }
    }
}

/// Storage engine statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total rows stored
    pub total_rows: u64,
    /// Total bytes on disk
    pub disk_bytes: u64,
    /// Bytes in memory (memtables)
    pub memory_bytes: u64,
    /// Number of levels
    pub num_levels: u32,
    /// Files per level
    pub files_per_level: Vec<u32>,
    /// Read amplification
    pub read_amplification: f64,
    /// Write amplification
    pub write_amplification: f64,
}

/// StorageEngine trait - the core abstraction for pluggable storage backends
///
/// Implementations:
/// - `Lscs`: Columnar storage for TOON workloads (80% I/O reduction for projections)
/// - `LegacyLsmTree`: Row-oriented storage for compatibility
pub trait StorageEngine: Send + Sync {
    /// Begin a new transaction
    fn begin_txn(&self) -> Result<TxnHandle>;

    /// Get a single row by key
    fn get(&self, txn: &TxnHandle, key: &[u8]) -> Result<Option<Row>>;

    /// Put a row (insert or update)
    fn put(&self, txn: &TxnHandle, key: &[u8], row: Row) -> Result<()>;

    /// Delete a row
    fn delete(&self, txn: &TxnHandle, key: &[u8]) -> Result<()>;

    /// Scan a range of rows
    fn scan(&self, txn: &TxnHandle, range: Range<Vec<u8>>) -> Result<Vec<Row>>;

    /// Scan columns selectively (columnar optimization)
    ///
    /// This is the key optimization for TOON workloads:
    /// - Traditional: Read all columns O(N × K)
    /// - Columnar: Read only selected columns O(N × k)
    ///
    /// For k/K = 0.2, this is 80% I/O reduction
    fn scan_columns(
        &self,
        txn: &TxnHandle,
        range: Range<Vec<u8>>,
        cols: &[ColumnId],
    ) -> Result<ColumnIterator>;

    /// Commit a transaction
    fn commit(&self, txn: TxnHandle) -> Result<()>;

    /// Abort a transaction
    fn abort(&self, txn: TxnHandle) -> Result<()>;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;

    /// Force flush memtables to disk
    fn flush(&self) -> Result<()>;

    /// Trigger compaction
    fn compact(&self) -> Result<()>;

    /// Close the storage engine
    fn close(&self) -> Result<()>;
}

/// Open a storage engine from a path
pub fn open_storage_engine<P: AsRef<Path>>(
    path: P,
    engine_type: StorageEngineType,
) -> Result<Arc<dyn StorageEngine>> {
    match engine_type {
        StorageEngineType::Lscs => {
            use crate::lscs::{ColumnDef, ColumnType, Lscs, LscsConfig, TableSchema};

            // Create default schema for general-purpose storage
            let schema = TableSchema::new(
                "default".to_string(),
                vec![
                    ColumnDef {
                        name: "key".to_string(),
                        col_type: ColumnType::Binary,
                        nullable: false,
                    },
                    ColumnDef {
                        name: "value".to_string(),
                        col_type: ColumnType::Binary,
                        nullable: true,
                    },
                ],
            )
            .with_mvcc();

            let lscs = Lscs::new(path.as_ref().to_path_buf(), schema, LscsConfig::default())?;
            Ok(Arc::new(LscsAdapter::new(lscs)))
        }
        StorageEngineType::Legacy => Err(SochDBError::InvalidArgument(
            "Legacy LSMTree has been removed; use Lscs".to_string(),
        )),
    }
}

/// Storage engine types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageEngineType {
    /// LSCS columnar storage (default)
    #[default]
    Lscs,
    /// Legacy row-oriented storage (deprecated)
    Legacy,
}

/// Adapter to make Lscs implement StorageEngine
pub struct LscsAdapter {
    inner: crate::lscs::Lscs,
    next_txn_id: std::sync::atomic::AtomicU64,
    version_counter: std::sync::atomic::AtomicU64,
}

impl LscsAdapter {
    /// Create a new adapter
    pub fn new(lscs: crate::lscs::Lscs) -> Self {
        Self {
            inner: lscs,
            next_txn_id: std::sync::atomic::AtomicU64::new(1),
            version_counter: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

impl StorageEngine for LscsAdapter {
    fn begin_txn(&self) -> Result<TxnHandle> {
        let txn_id = self
            .next_txn_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let snapshot = self
            .version_counter
            .load(std::sync::atomic::Ordering::Acquire);
        Ok(TxnHandle::new(txn_id, snapshot))
    }

    fn get(&self, txn: &TxnHandle, key: &[u8]) -> Result<Option<Row>> {
        // Use learned index for O(1) lookup
        let row_id = u64::from_le_bytes(
            key.try_into()
                .map_err(|_| SochDBError::InvalidArgument("Key must be 8 bytes".to_string()))?,
        );

        // Use LSCS get() with MVCC filtering
        if let Some(values) = self.inner.get(row_id)? {
            // Extract MVCC timestamps from the last two columns (__txn_start, __txn_end)
            let num_cols = values.len();
            let (txn_start, txn_end) = if num_cols >= 2 {
                let start = values[num_cols - 2]
                    .as_ref()
                    .and_then(|v| v.get(..8))
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .unwrap_or(0);
                let end = values[num_cols - 1]
                    .as_ref()
                    .and_then(|v| v.get(..8))
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .unwrap_or(0);
                (start, end)
            } else {
                (0, 0)
            };

            let row = Row {
                id: row_id,
                values,
                txn_start,
                txn_end,
            };

            // Apply MVCC visibility check
            if row.is_visible(txn.snapshot_version) {
                return Ok(Some(row));
            }
        }

        Ok(None)
    }

    fn put(&self, _txn: &TxnHandle, key: &[u8], row: Row) -> Result<()> {
        let values: Vec<Option<&[u8]>> = row.values.iter().map(|v| v.as_deref()).collect();
        let _ = key; // Key is derived from row ID
        self.inner.insert(&values)?;
        Ok(())
    }

    fn delete(&self, txn: &TxnHandle, key: &[u8]) -> Result<()> {
        // Mark row as deleted by setting txn_end
        // In MVCC, delete is a "tombstone" - we mark the row with an end timestamp
        let row_id = u64::from_le_bytes(
            key.try_into()
                .map_err(|_| SochDBError::InvalidArgument("Key must be 8 bytes".to_string()))?,
        );

        // Write tombstone by updating __txn_end column to current transaction timestamp
        self.inner
            .mark_deleted(row_id, txn.txn_id, txn.snapshot_version)?;
        Ok(())
    }

    fn scan(&self, txn: &TxnHandle, range: Range<Vec<u8>>) -> Result<Vec<Row>> {
        // Parse range as row IDs
        let start = if range.start.len() >= 8 {
            u64::from_le_bytes(range.start[..8].try_into().unwrap())
        } else {
            0
        };
        let end = if range.end.len() >= 8 {
            u64::from_le_bytes(range.end[..8].try_into().unwrap())
        } else {
            u64::MAX
        };

        // Use LSCS scan_range
        let scan_results = self.inner.scan_range(start, end)?;
        let rows: Vec<Row> = scan_results
            .into_iter()
            .filter_map(|(row_id, values)| {
                let row = Row {
                    id: row_id,
                    values,
                    txn_start: 0,
                    txn_end: 0,
                };
                if row.is_visible(txn.snapshot_version) {
                    Some(row)
                } else {
                    None
                }
            })
            .collect();

        Ok(rows)
    }

    fn scan_columns(
        &self,
        txn: &TxnHandle,
        range: Range<Vec<u8>>,
        cols: &[ColumnId],
    ) -> Result<ColumnIterator> {
        // Parse range as row IDs
        let start = if range.start.len() >= 8 {
            u64::from_le_bytes(range.start[..8].try_into().unwrap())
        } else {
            0
        };
        let end = if range.end.len() >= 8 {
            u64::from_le_bytes(range.end[..8].try_into().unwrap())
        } else {
            u64::MAX
        };

        // Columnar selective read - only read requested columns
        // This achieves 80% I/O reduction when reading 20% of columns
        let col_indices: Vec<usize> = cols.iter().map(|&c| c as usize).collect();

        let scan_results = self.inner.scan_columns_range(start, end, &col_indices)?;

        let rows: Vec<Row> = scan_results
            .into_iter()
            .filter_map(|(row_id, values)| {
                let row = Row {
                    id: row_id,
                    values,
                    txn_start: 0,
                    txn_end: 0,
                };
                if row.is_visible(txn.snapshot_version) {
                    Some(row)
                } else {
                    None
                }
            })
            .collect();

        Ok(ColumnIterator::new(rows, cols.to_vec()))
    }

    fn commit(&self, txn: TxnHandle) -> Result<()> {
        // Ensure durability by calling fsync
        self.inner.fsync()?;
        self.version_counter
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        let _ = txn;
        Ok(())
    }

    fn abort(&self, _txn: TxnHandle) -> Result<()> {
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        let lscs_stats = self.inner.stats();
        StorageStats {
            total_rows: lscs_stats.next_row_id,
            disk_bytes: lscs_stats.disk_bytes,
            memory_bytes: lscs_stats.active_memtable_bytes as u64,
            num_levels: lscs_stats.level_row_counts.len() as u32,
            files_per_level: vec![0; lscs_stats.level_row_counts.len()],
            read_amplification: 1.0,
            write_amplification: 1.0,
        }
    }

    fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    fn compact(&self) -> Result<()> {
        self.inner.compact()
    }

    fn close(&self) -> Result<()> {
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_txn_handle() {
        let handle = TxnHandle::new(1, 100);
        assert_eq!(handle.txn_id, 1);
        assert_eq!(handle.snapshot_version, 100);
        assert!(handle.start_ts > 0);
    }

    #[test]
    fn test_row_visibility() {
        let mut row = Row::new(1, vec![Some(b"test".to_vec())]);
        row.txn_start = 100;
        row.txn_end = 0;

        // Active row visible at any version >= 100
        assert!(row.is_visible(100));
        assert!(row.is_visible(200));
        assert!(!row.is_visible(99));

        // Deleted row
        row.txn_end = 150;
        assert!(row.is_visible(120)); // Between start and end
        assert!(!row.is_visible(150)); // At deletion
        assert!(!row.is_visible(200)); // After deletion
    }

    #[test]
    fn test_column_iterator() {
        let rows = vec![
            Row::new(1, vec![Some(b"a".to_vec()), Some(b"b".to_vec())]),
            Row::new(2, vec![Some(b"c".to_vec()), Some(b"d".to_vec())]),
        ];
        let mut iter = ColumnIterator::new(rows, vec![0, 1]);

        assert_eq!(iter.column_ids(), &[0, 1]);
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }
}
