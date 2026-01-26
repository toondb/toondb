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

//! Tombstones for Logical Deletion (Task 6)
//!
//! This module implements logical deletion via tombstones for vector indexes.
//! Deleted vectors remain in the index but are filtered out during retrieval.
//!
//! ## Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     TombstoneManager                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  In-Memory:                                                      │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  Bitmap / HashSet for O(1) deletion checks                │   │
//! │  │  RoaringBitmap for space efficiency                       │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! │                                                                  │
//! │  On-Disk:                                                        │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  tombstones.bin: Append-only log of deleted IDs           │   │
//! │  │  Format: [header][entry][entry]...                        │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Filtering During Retrieval
//!
//! During vector search:
//! 1. Get top-K candidates from HNSW
//! 2. Filter out deleted IDs (O(1) per candidate)
//! 3. Fetch more candidates if needed (over-fetch strategy)
//!
//! ## Future: Compaction
//!
//! Tombstones can be compacted during index rebuilds:
//! 1. Build new index without deleted vectors
//! 2. Swap indexes atomically
//! 3. Discard tombstone file

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

// ============================================================================
// Tombstone ID Types
// ============================================================================

/// Vector ID type (matches the storage layer)
pub type VectorId = u128;

/// Internal ID type for compact storage
pub type InternalId = u64;

// ============================================================================
// Tombstone Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum TombstoneError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    
    #[error("corrupted tombstone file: {0}")]
    Corrupted(String),
    
    #[error("tombstone file version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
}

pub type Result<T> = std::result::Result<T, TombstoneError>;

// ============================================================================
// Tombstone Manager
// ============================================================================

/// Manager for tombstone-based logical deletion
///
/// Provides O(1) deletion checks during vector retrieval.
pub struct TombstoneManager {
    /// Path to tombstone file
    path: PathBuf,
    
    /// In-memory set of deleted IDs
    deleted: RwLock<HashSet<InternalId>>,
    
    /// Count of deleted IDs
    count: AtomicU64,
    
    /// File handle for appending (if writable)
    writer: Option<RwLock<BufWriter<File>>>,
}

/// Tombstone file header
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct TombstoneHeader {
    /// Magic bytes: "TOMB"
    magic: [u8; 4],
    /// Version
    version: u32,
    /// Number of entries
    count: u64,
    /// Reserved for future use
    _reserved: [u8; 16],
}

impl TombstoneHeader {
    const MAGIC: [u8; 4] = *b"TOMB";
    const VERSION: u32 = 1;
    const SIZE: usize = std::mem::size_of::<Self>();
    
    fn new(count: u64) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            count,
            _reserved: [0u8; 16],
        }
    }
    
    fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(TombstoneError::Corrupted(
                "invalid magic bytes".to_string(),
            ));
        }
        if self.version != Self::VERSION {
            return Err(TombstoneError::VersionMismatch {
                expected: Self::VERSION,
                actual: self.version,
            });
        }
        Ok(())
    }
    
    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.count.to_le_bytes());
        bytes
    }
    
    fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        Self {
            magic: bytes[0..4].try_into().unwrap(),
            version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            count: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            _reserved: bytes[16..32].try_into().unwrap(),
        }
    }
}

impl TombstoneManager {
    /// Create a new tombstone manager
    ///
    /// If the file exists, loads existing tombstones.
    /// If not, creates a new empty tombstone file.
    pub fn new(path: impl AsRef<Path>, writable: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        let (deleted, count, writer) = if path.exists() {
            // Load existing tombstones
            let (deleted, count) = Self::load_from_file(&path)?;
            
            let writer = if writable {
                let file = OpenOptions::new()
                    .append(true)
                    .open(&path)?;
                Some(RwLock::new(BufWriter::new(file)))
            } else {
                None
            };
            
            (deleted, count, writer)
        } else if writable {
            // Create new file
            let file = File::create(&path)?;
            let mut writer = BufWriter::new(file);
            
            // Write header
            let header = TombstoneHeader::new(0);
            writer.write_all(&header.to_bytes())?;
            writer.flush()?;
            
            // Reopen for append
            drop(writer);
            let file = OpenOptions::new().append(true).open(&path)?;
            
            (HashSet::new(), 0, Some(RwLock::new(BufWriter::new(file))))
        } else {
            (HashSet::new(), 0, None)
        };
        
        Ok(Self {
            path,
            deleted: RwLock::new(deleted),
            count: AtomicU64::new(count),
            writer,
        })
    }
    
    /// Load tombstones from file
    fn load_from_file(path: &Path) -> Result<(HashSet<InternalId>, u64)> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read header
        let mut header_bytes = [0u8; TombstoneHeader::SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header = TombstoneHeader::from_bytes(&header_bytes);
        header.validate()?;
        
        // Read entries
        let mut deleted = HashSet::with_capacity(header.count as usize);
        let mut id_bytes = [0u8; 8];
        let mut count = 0u64;
        
        while reader.read_exact(&mut id_bytes).is_ok() {
            let id = u64::from_le_bytes(id_bytes);
            deleted.insert(id);
            count += 1;
        }
        
        Ok((deleted, count))
    }
    
    /// Mark an ID as deleted
    pub fn delete(&self, id: InternalId) -> Result<bool> {
        // Check if already deleted
        {
            let deleted = self.deleted.read();
            if deleted.contains(&id) {
                return Ok(false); // Already deleted
            }
        }
        
        // Add to in-memory set
        {
            let mut deleted = self.deleted.write();
            if !deleted.insert(id) {
                return Ok(false); // Already deleted
            }
        }
        
        // Append to file
        if let Some(ref writer) = self.writer {
            let mut writer = writer.write();
            writer.write_all(&id.to_le_bytes())?;
            writer.flush()?;
        }
        
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }
    
    /// Mark multiple IDs as deleted
    pub fn delete_batch(&self, ids: &[InternalId]) -> Result<usize> {
        let mut new_deletions = Vec::new();
        
        // Add to in-memory set
        {
            let mut deleted = self.deleted.write();
            for &id in ids {
                if deleted.insert(id) {
                    new_deletions.push(id);
                }
            }
        }
        
        if new_deletions.is_empty() {
            return Ok(0);
        }
        
        // Append to file
        if let Some(ref writer) = self.writer {
            let mut writer = writer.write();
            for id in &new_deletions {
                writer.write_all(&id.to_le_bytes())?;
            }
            writer.flush()?;
        }
        
        let count = new_deletions.len();
        self.count.fetch_add(count as u64, Ordering::Relaxed);
        Ok(count)
    }
    
    /// Check if an ID is deleted
    #[inline]
    pub fn is_deleted(&self, id: InternalId) -> bool {
        self.deleted.read().contains(&id)
    }
    
    /// Check multiple IDs for deletion
    pub fn filter_deleted(&self, ids: &[InternalId]) -> Vec<InternalId> {
        let deleted = self.deleted.read();
        ids.iter()
            .copied()
            .filter(|id| !deleted.contains(id))
            .collect()
    }
    
    /// Get the count of deleted IDs
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
    
    /// Get all deleted IDs (for compaction)
    pub fn all_deleted(&self) -> Vec<InternalId> {
        self.deleted.read().iter().copied().collect()
    }
    
    /// Sync to disk
    pub fn sync(&self) -> Result<()> {
        if let Some(ref writer) = self.writer {
            writer.write().flush()?;
        }
        Ok(())
    }
    
    /// Compact the tombstone file (rewrites without duplicates)
    pub fn compact(&self) -> Result<()> {
        let deleted: Vec<_> = self.deleted.read().iter().copied().collect();
        
        // Write to temporary file
        let temp_path = self.path.with_extension("tmp");
        {
            let file = File::create(&temp_path)?;
            let mut writer = BufWriter::new(file);
            
            // Write header
            let header = TombstoneHeader::new(deleted.len() as u64);
            writer.write_all(&header.to_bytes())?;
            
            // Write entries
            for id in &deleted {
                writer.write_all(&id.to_le_bytes())?;
            }
            writer.flush()?;
        }
        
        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;
        
        Ok(())
    }
}

// ============================================================================
// Tombstone Filter for Vector Search
// ============================================================================

/// Filter for vector search results that excludes deleted IDs
pub struct TombstoneFilter {
    manager: Arc<TombstoneManager>,
    /// Over-fetch factor: fetch this many extra candidates to account for deletions
    overfetch_factor: f32,
}

impl TombstoneFilter {
    /// Create a new tombstone filter
    pub fn new(manager: Arc<TombstoneManager>) -> Self {
        Self {
            manager,
            overfetch_factor: 1.2, // 20% over-fetch by default
        }
    }
    
    /// Set the over-fetch factor
    pub fn with_overfetch(mut self, factor: f32) -> Self {
        self.overfetch_factor = factor.max(1.0);
        self
    }
    
    /// Calculate how many candidates to fetch given the target K
    pub fn effective_k(&self, k: usize) -> usize {
        let deletion_rate = self.deletion_rate();
        if deletion_rate < 0.01 {
            // Very few deletions, minimal over-fetch
            (k as f32 * 1.05).ceil() as usize
        } else {
            // Estimate how many extra we need
            let factor = 1.0 / (1.0 - deletion_rate);
            (k as f32 * factor * self.overfetch_factor).ceil() as usize
        }
    }
    
    /// Get the current deletion rate (for adaptive over-fetch)
    fn deletion_rate(&self) -> f32 {
        // This would ideally use total_vectors / deleted_count
        // For now, use a conservative estimate
        0.1
    }
    
    /// Filter search results, removing deleted IDs
    pub fn filter<T>(&self, results: Vec<(InternalId, T)>, limit: usize) -> Vec<(InternalId, T)> {
        results
            .into_iter()
            .filter(|(id, _)| !self.manager.is_deleted(*id))
            .take(limit)
            .collect()
    }
    
    /// Filter and check if we need more candidates
    pub fn filter_with_continuation<T>(
        &self,
        results: Vec<(InternalId, T)>,
        limit: usize,
    ) -> (Vec<(InternalId, T)>, bool) {
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|(id, _)| !self.manager.is_deleted(*id))
            .take(limit)
            .collect();
        
        let need_more = filtered.len() < limit;
        (filtered, need_more)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    fn temp_tombstone() -> (TempDir, TombstoneManager) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("tombstones.bin");
        let manager = TombstoneManager::new(&path, true).unwrap();
        (temp_dir, manager)
    }
    
    #[test]
    fn test_delete_single() {
        let (_temp, manager) = temp_tombstone();
        
        assert!(!manager.is_deleted(1));
        
        assert!(manager.delete(1).unwrap());
        assert!(manager.is_deleted(1));
        
        // Double delete returns false
        assert!(!manager.delete(1).unwrap());
    }
    
    #[test]
    fn test_delete_batch() {
        let (_temp, manager) = temp_tombstone();
        
        let count = manager.delete_batch(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(count, 5);
        
        for id in 1..=5 {
            assert!(manager.is_deleted(id));
        }
        assert!(!manager.is_deleted(6));
        
        // Partial overlap
        let count = manager.delete_batch(&[4, 5, 6, 7]).unwrap();
        assert_eq!(count, 2); // Only 6 and 7 are new
    }
    
    #[test]
    fn test_filter_deleted() {
        let (_temp, manager) = temp_tombstone();
        
        manager.delete_batch(&[2, 4, 6, 8]).unwrap();
        
        let ids: Vec<_> = (1..=10).collect();
        let filtered = manager.filter_deleted(&ids);
        
        assert_eq!(filtered, vec![1, 3, 5, 7, 9, 10]);
    }
    
    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("tombstones.bin");
        
        // Create and delete some IDs
        {
            let manager = TombstoneManager::new(&path, true).unwrap();
            manager.delete_batch(&[1, 2, 3]).unwrap();
            manager.sync().unwrap();
        }
        
        // Reload and verify
        {
            let manager = TombstoneManager::new(&path, false).unwrap();
            assert!(manager.is_deleted(1));
            assert!(manager.is_deleted(2));
            assert!(manager.is_deleted(3));
            assert!(!manager.is_deleted(4));
            assert_eq!(manager.count(), 3);
        }
    }
    
    #[test]
    fn test_compact() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("tombstones.bin");
        
        {
            let manager = TombstoneManager::new(&path, true).unwrap();
            
            // Delete same IDs multiple times (simulating append-only)
            for _ in 0..5 {
                manager.delete_batch(&[1, 2, 3]).unwrap();
            }
            
            manager.compact().unwrap();
        }
        
        // Verify compacted file
        let manager = TombstoneManager::new(&path, false).unwrap();
        assert_eq!(manager.count(), 3);
    }
    
    #[test]
    fn test_tombstone_filter() {
        let (_temp, manager) = temp_tombstone();
        let manager = Arc::new(manager);
        
        // Delete IDs 2 and 4
        manager.delete_batch(&[2, 4]).unwrap();
        
        let filter = TombstoneFilter::new(manager);
        
        // Search results with deleted items
        let results: Vec<(InternalId, f32)> = vec![
            (1, 0.9),
            (2, 0.8), // Deleted
            (3, 0.7),
            (4, 0.6), // Deleted
            (5, 0.5),
        ];
        
        let filtered = filter.filter(results, 3);
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].0, 1);
        assert_eq!(filtered[1].0, 3);
        assert_eq!(filtered[2].0, 5);
    }
    
    #[test]
    fn test_effective_k() {
        let (_temp, manager) = temp_tombstone();
        let manager = Arc::new(manager);
        
        let filter = TombstoneFilter::new(manager);
        
        // With default settings, should over-fetch
        let k = filter.effective_k(10);
        assert!(k > 10);
    }
}
