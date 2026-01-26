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

//! SochFS Metadata Layer (Task 11)
//!
//! Persists filesystem metadata via LSCS system tables for ACID guarantees:
//! - _sys_fs_inodes: Inode storage (id, type, size, blocks, permissions, timestamps)
//! - _sys_fs_dirs: Directory entries (parent_id, name, child_inode)
//! - _sys_fs_superblock: Filesystem metadata (root, next_inode, next_block)
//!
//! ## System Table Schema
//!
//! ```text
//! _sys_fs_inodes:
//! ┌──────────┬───────────┬────────┬────────────┬─────────────┐
//! │ inode_id │ file_type │ size   │ blocks     │ permissions │
//! │ u64 PK   │ u8        │ u64    │ blob       │ u16         │
//! └──────────┴───────────┴────────┴────────────┴─────────────┘
//!
//! _sys_fs_dirs:
//! ┌────────────┬────────────┬──────────────┐
//! │ parent_id  │ name       │ child_inode  │
//! │ u64        │ text       │ u64          │
//! └────────────┴────────────┴──────────────┘
//! ```
//!
//! ## Path Resolution: O(d) where d = depth
//!
//! ```text
//! resolve("/docs/report.toon"):
//!   1. lookup(_sys_fs_dirs, parent=1, name="docs") → inode=2
//!   2. lookup(_sys_fs_dirs, parent=2, name="report.toon") → inode=7
//!   3. lookup(_sys_fs_inodes, inode=7) → Inode{...}
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::vfs::{DirEntry, FileType, Inode, InodeId, Permissions, Superblock};
use crate::{Result, SochDBError};

/// System table names
pub const SYSTEM_TABLE_INODES: &str = "_sys_fs_inodes";
pub const SYSTEM_TABLE_DIRS: &str = "_sys_fs_dirs";
pub const SYSTEM_TABLE_SUPERBLOCK: &str = "_sys_fs_superblock";

/// Root inode ID
pub const ROOT_INODE: InodeId = 1;

/// Serialized inode for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InodeRow {
    pub inode_id: u64,
    pub file_type: u8,
    pub size: u64,
    pub blocks: Vec<u64>,
    pub permissions: u16,
    pub created_us: u64,
    pub modified_us: u64,
    pub accessed_us: u64,
    pub nlink: u32,
    pub symlink_target: Option<String>,
    pub soch_schema: Option<String>,
}

impl From<&Inode> for InodeRow {
    fn from(inode: &Inode) -> Self {
        Self {
            inode_id: inode.id,
            file_type: inode.file_type as u8,
            size: inode.size,
            blocks: inode.blocks.clone(),
            permissions: inode.permissions.to_mode() as u16,
            created_us: inode.created_us,
            modified_us: inode.modified_us,
            accessed_us: inode.accessed_us,
            nlink: inode.nlink,
            symlink_target: inode.symlink_target.clone(),
            soch_schema: inode.soch_schema.clone(),
        }
    }
}

impl InodeRow {
    pub fn to_inode(&self) -> Inode {
        Inode {
            id: self.inode_id,
            file_type: match self.file_type {
                1 => FileType::Regular,
                2 => FileType::Directory,
                3 => FileType::Symlink,
                4 => FileType::SochDocument,
                _ => FileType::Regular,
            },
            size: self.size,
            blocks: self.blocks.clone(),
            permissions: Permissions::from_mode(self.permissions as u8),
            created_us: self.created_us,
            modified_us: self.modified_us,
            accessed_us: self.accessed_us,
            nlink: self.nlink,
            symlink_target: self.symlink_target.clone(),
            soch_schema: self.soch_schema.clone(),
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| SochDBError::Serialization(e.to_string()))
    }
}

/// Serialized directory entry for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntryRow {
    pub parent_id: u64,
    pub name: String,
    pub child_inode: u64,
    pub file_type: u8,
}

impl DirEntryRow {
    pub fn new(
        parent_id: InodeId,
        name: String,
        child_inode: InodeId,
        file_type: FileType,
    ) -> Self {
        Self {
            parent_id,
            name,
            child_inode,
            file_type: file_type as u8,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| SochDBError::Serialization(e.to_string()))
    }

    /// Convert to key for lookup (parent_id + name)
    pub fn to_key(&self) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + self.name.len());
        key.extend_from_slice(&self.parent_id.to_le_bytes());
        key.extend_from_slice(self.name.as_bytes());
        key
    }

    /// Create key from parent and name
    pub fn make_key(parent_id: InodeId, name: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + name.len());
        key.extend_from_slice(&parent_id.to_le_bytes());
        key.extend_from_slice(name.as_bytes());
        key
    }
}

/// WAL operation for filesystem changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FsWalOp {
    /// Create inode
    CreateInode(InodeRow),
    /// Update inode
    UpdateInode(InodeRow),
    /// Delete inode
    DeleteInode(u64),
    /// Add directory entry
    AddDirEntry(DirEntryRow),
    /// Remove directory entry
    RemoveDirEntry { parent_id: u64, name: String },
    /// Update superblock
    UpdateSuperblock(Superblock),
}

impl FsWalOp {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| SochDBError::Serialization(e.to_string()))
    }
}

/// SochFS Metadata Store
///
/// Manages filesystem metadata with ACID guarantees via LSCS system tables.
#[allow(clippy::type_complexity)]
pub struct FsMetadataStore {
    /// Inode cache
    inodes: RwLock<HashMap<InodeId, Inode>>,
    /// Directory cache (parent_id -> entries)
    directories: RwLock<HashMap<InodeId, Vec<DirEntryRow>>>,
    /// Superblock
    superblock: RwLock<Superblock>,
    /// Write callback for persistence
    write_fn: Box<dyn Fn(&[u8], &[u8]) -> Result<()> + Send + Sync>,
    /// WAL callback
    wal_fn: Box<dyn Fn(&FsWalOp) -> Result<()> + Send + Sync>,
    /// Dirty inodes (need flush)
    #[allow(dead_code)]
    dirty_inodes: RwLock<Vec<InodeId>>,
}

impl FsMetadataStore {
    /// Create a new metadata store
    pub fn new<W, L>(write_fn: W, wal_fn: L) -> Self
    where
        W: Fn(&[u8], &[u8]) -> Result<()> + Send + Sync + 'static,
        L: Fn(&FsWalOp) -> Result<()> + Send + Sync + 'static,
    {
        let superblock = Superblock::new("toonfs");
        let root_inode = Inode::new_directory(ROOT_INODE);

        let mut inodes = HashMap::new();
        inodes.insert(ROOT_INODE, root_inode);

        Self {
            inodes: RwLock::new(inodes),
            directories: RwLock::new(HashMap::new()),
            superblock: RwLock::new(superblock),
            write_fn: Box::new(write_fn),
            wal_fn: Box::new(wal_fn),
            dirty_inodes: RwLock::new(Vec::new()),
        }
    }

    /// Initialize filesystem (create root if needed)
    pub fn init(&self) -> Result<()> {
        let sb = self.superblock.read();
        let root = Inode::new_directory(sb.root_inode);
        drop(sb);

        // Create root inode
        let row = InodeRow::from(&root);
        (self.wal_fn)(&FsWalOp::CreateInode(row.clone()))?;

        let key = root.id.to_le_bytes();
        (self.write_fn)(&key, &row.to_bytes())?;

        self.inodes.write().insert(root.id, root);
        Ok(())
    }

    /// Get inode by ID
    pub fn get_inode(&self, id: InodeId) -> Option<Inode> {
        self.inodes.read().get(&id).cloned()
    }

    /// Create a new inode
    pub fn create_inode(&self, file_type: FileType) -> Result<Inode> {
        let id = {
            let mut sb = self.superblock.write();
            sb.alloc_inode()
        };

        let inode = match file_type {
            FileType::Regular => Inode::new_file(id),
            FileType::Directory => Inode::new_directory(id),
            FileType::Symlink => Inode::new_symlink(id, String::new()),
            FileType::SochDocument => Inode::new_toon(id, String::new()),
        };

        // WAL first
        let row = InodeRow::from(&inode);
        (self.wal_fn)(&FsWalOp::CreateInode(row.clone()))?;

        // Then persist
        let key = inode.id.to_le_bytes();
        (self.write_fn)(&key, &row.to_bytes())?;

        // Cache
        self.inodes.write().insert(id, inode.clone());

        Ok(inode)
    }

    /// Update an inode
    pub fn update_inode(&self, inode: &Inode) -> Result<()> {
        let row = InodeRow::from(inode);

        // WAL first
        (self.wal_fn)(&FsWalOp::UpdateInode(row.clone()))?;

        // Then persist
        let key = inode.id.to_le_bytes();
        (self.write_fn)(&key, &row.to_bytes())?;

        // Update cache
        self.inodes.write().insert(inode.id, inode.clone());

        Ok(())
    }

    /// Delete an inode
    pub fn delete_inode(&self, id: InodeId) -> Result<()> {
        // WAL first
        (self.wal_fn)(&FsWalOp::DeleteInode(id))?;

        // Remove from cache
        self.inodes.write().remove(&id);

        Ok(())
    }

    /// Add directory entry
    pub fn add_dir_entry(
        &self,
        parent_id: InodeId,
        name: &str,
        child_id: InodeId,
        file_type: FileType,
    ) -> Result<()> {
        let entry = DirEntryRow::new(parent_id, name.to_string(), child_id, file_type);

        // WAL first
        (self.wal_fn)(&FsWalOp::AddDirEntry(entry.clone()))?;

        // Then persist
        let key = entry.to_key();
        (self.write_fn)(&key, &entry.to_bytes())?;

        // Update cache
        self.directories
            .write()
            .entry(parent_id)
            .or_default()
            .push(entry);

        Ok(())
    }

    /// Remove directory entry
    pub fn remove_dir_entry(&self, parent_id: InodeId, name: &str) -> Result<()> {
        // WAL first
        (self.wal_fn)(&FsWalOp::RemoveDirEntry {
            parent_id,
            name: name.to_string(),
        })?;

        // Update cache
        let mut dirs = self.directories.write();
        if let Some(entries) = dirs.get_mut(&parent_id) {
            entries.retain(|e| e.name != name);
        }

        Ok(())
    }

    /// List directory entries
    pub fn list_dir(&self, parent_id: InodeId) -> Vec<DirEntryRow> {
        self.directories
            .read()
            .get(&parent_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Lookup entry in directory
    pub fn lookup(&self, parent_id: InodeId, name: &str) -> Option<InodeId> {
        self.directories
            .read()
            .get(&parent_id)
            .and_then(|entries| entries.iter().find(|e| e.name == name))
            .map(|e| e.child_inode)
    }

    /// Resolve path to inode
    ///
    /// O(d) where d = path depth
    pub fn resolve_path(&self, path: &str) -> Result<InodeId> {
        let path = path.trim_start_matches('/');
        if path.is_empty() {
            return Ok(ROOT_INODE);
        }

        let mut current = ROOT_INODE;
        for component in path.split('/') {
            if component.is_empty() || component == "." {
                continue;
            }
            if component == ".." {
                // Get parent from directory
                if let Some(inode) = self.get_inode(current)
                    && inode.is_dir()
                {
                    // Look up parent in _sys_fs_dirs would go here
                    // For now, just stay at root if we can't go up
                }
                continue;
            }

            current = self.lookup(current, component).ok_or_else(|| {
                SochDBError::NotFound(format!("Path component not found: {}", component))
            })?;
        }

        Ok(current)
    }

    /// Create file in directory
    pub fn create_file(&self, parent_id: InodeId, name: &str) -> Result<Inode> {
        // Check parent is directory
        let parent = self
            .get_inode(parent_id)
            .ok_or_else(|| SochDBError::NotFound("Parent not found".into()))?;

        if !parent.is_dir() {
            return Err(SochDBError::InvalidArgument(
                "Parent is not a directory".into(),
            ));
        }

        // Check name doesn't exist
        if self.lookup(parent_id, name).is_some() {
            return Err(SochDBError::InvalidArgument("File already exists".into()));
        }

        // Create inode
        let inode = self.create_inode(FileType::Regular)?;

        // Add directory entry
        self.add_dir_entry(parent_id, name, inode.id, FileType::Regular)?;

        Ok(inode)
    }

    /// Create directory in parent
    pub fn create_dir(&self, parent_id: InodeId, name: &str) -> Result<Inode> {
        // Check parent is directory
        let parent = self
            .get_inode(parent_id)
            .ok_or_else(|| SochDBError::NotFound("Parent not found".into()))?;

        if !parent.is_dir() {
            return Err(SochDBError::InvalidArgument(
                "Parent is not a directory".into(),
            ));
        }

        // Check name doesn't exist
        if self.lookup(parent_id, name).is_some() {
            return Err(SochDBError::InvalidArgument(
                "Directory already exists".into(),
            ));
        }

        // Create inode
        let inode = self.create_inode(FileType::Directory)?;

        // Add directory entry
        self.add_dir_entry(parent_id, name, inode.id, FileType::Directory)?;

        Ok(inode)
    }

    /// Delete file or empty directory
    pub fn delete(&self, parent_id: InodeId, name: &str) -> Result<()> {
        let child_id = self
            .lookup(parent_id, name)
            .ok_or_else(|| SochDBError::NotFound("Entry not found".into()))?;

        let child = self
            .get_inode(child_id)
            .ok_or_else(|| SochDBError::NotFound("Inode not found".into()))?;

        // If directory, must be empty
        if child.is_dir() {
            let entries = self.list_dir(child_id);
            if !entries.is_empty() {
                return Err(SochDBError::InvalidArgument("Directory not empty".into()));
            }
        }

        // Remove directory entry
        self.remove_dir_entry(parent_id, name)?;

        // Delete inode
        self.delete_inode(child_id)?;

        Ok(())
    }

    /// Get superblock
    pub fn superblock(&self) -> Superblock {
        self.superblock.read().clone()
    }

    /// Update superblock
    pub fn update_superblock(&self, sb: &Superblock) -> Result<()> {
        (self.wal_fn)(&FsWalOp::UpdateSuperblock(sb.clone()))?;
        *self.superblock.write() = sb.clone();
        Ok(())
    }

    /// Recover from WAL operations
    pub fn replay_wal_op(&self, op: &FsWalOp) -> Result<()> {
        match op {
            FsWalOp::CreateInode(row) => {
                self.inodes.write().insert(row.inode_id, row.to_inode());
            }
            FsWalOp::UpdateInode(row) => {
                self.inodes.write().insert(row.inode_id, row.to_inode());
            }
            FsWalOp::DeleteInode(id) => {
                self.inodes.write().remove(id);
            }
            FsWalOp::AddDirEntry(entry) => {
                self.directories
                    .write()
                    .entry(entry.parent_id)
                    .or_default()
                    .push(entry.clone());
            }
            FsWalOp::RemoveDirEntry { parent_id, name } => {
                let mut dirs = self.directories.write();
                if let Some(entries) = dirs.get_mut(parent_id) {
                    entries.retain(|e| &e.name != name);
                }
            }
            FsWalOp::UpdateSuperblock(sb) => {
                *self.superblock.write() = sb.clone();
            }
        }
        Ok(())
    }
}

/// SochFS - Complete filesystem layer
#[allow(clippy::type_complexity)]
pub struct SochFS {
    /// Metadata store
    metadata: FsMetadataStore,
    /// Block storage callback
    block_write_fn: Box<dyn Fn(u64, &[u8]) -> Result<u64> + Send + Sync>,
    /// Block read callback
    block_read_fn: Box<dyn Fn(u64, usize) -> Result<Vec<u8>> + Send + Sync>,
}

impl SochFS {
    /// Create new SochFS instance
    pub fn new<W, L, BW, BR>(write_fn: W, wal_fn: L, block_write_fn: BW, block_read_fn: BR) -> Self
    where
        W: Fn(&[u8], &[u8]) -> Result<()> + Send + Sync + 'static,
        L: Fn(&FsWalOp) -> Result<()> + Send + Sync + 'static,
        BW: Fn(u64, &[u8]) -> Result<u64> + Send + Sync + 'static,
        BR: Fn(u64, usize) -> Result<Vec<u8>> + Send + Sync + 'static,
    {
        Self {
            metadata: FsMetadataStore::new(write_fn, wal_fn),
            block_write_fn: Box::new(block_write_fn),
            block_read_fn: Box::new(block_read_fn),
        }
    }

    /// Initialize filesystem
    pub fn init(&self) -> Result<()> {
        self.metadata.init()
    }

    /// Resolve path to inode
    pub fn resolve(&self, path: &str) -> Result<InodeId> {
        self.metadata.resolve_path(path)
    }

    /// Get inode
    pub fn get_inode(&self, id: InodeId) -> Option<Inode> {
        self.metadata.get_inode(id)
    }

    /// Create file
    pub fn create_file(&self, path: &str) -> Result<Inode> {
        let (parent_path, name) = split_path(path);
        let parent_id = self.metadata.resolve_path(&parent_path)?;
        self.metadata.create_file(parent_id, &name)
    }

    /// Create directory
    pub fn mkdir(&self, path: &str) -> Result<Inode> {
        let (parent_path, name) = split_path(path);
        let parent_id = self.metadata.resolve_path(&parent_path)?;
        self.metadata.create_dir(parent_id, &name)
    }

    /// Delete file or empty directory
    pub fn delete(&self, path: &str) -> Result<()> {
        let (parent_path, name) = split_path(path);
        let parent_id = self.metadata.resolve_path(&parent_path)?;
        self.metadata.delete(parent_id, &name)
    }

    /// List directory
    pub fn readdir(&self, path: &str) -> Result<Vec<DirEntry>> {
        let inode_id = self.metadata.resolve_path(path)?;
        let inode = self
            .metadata
            .get_inode(inode_id)
            .ok_or_else(|| SochDBError::NotFound("Directory not found".into()))?;

        if !inode.is_dir() {
            return Err(SochDBError::InvalidArgument("Not a directory".into()));
        }

        let entries = self.metadata.list_dir(inode_id);
        Ok(entries
            .into_iter()
            .map(|e| DirEntry {
                name: e.name,
                inode: e.child_inode,
                file_type: match e.file_type {
                    1 => FileType::Regular,
                    2 => FileType::Directory,
                    3 => FileType::Symlink,
                    4 => FileType::SochDocument,
                    _ => FileType::Regular,
                },
            })
            .collect())
    }

    /// Write file data
    pub fn write_file(&self, path: &str, data: &[u8]) -> Result<usize> {
        let inode_id = self.metadata.resolve_path(path)?;
        let mut inode = self
            .metadata
            .get_inode(inode_id)
            .ok_or_else(|| SochDBError::NotFound("File not found".into()))?;

        if !inode.is_file() && !inode.is_toon() {
            return Err(SochDBError::InvalidArgument("Not a regular file".into()));
        }

        // Write data block
        let block_id = (self.block_write_fn)(inode_id, data)?;

        // Update inode
        inode.blocks = vec![block_id];
        inode.size = data.len() as u64;
        inode.touch();

        self.metadata.update_inode(&inode)?;

        Ok(data.len())
    }

    /// Read file data
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let inode_id = self.metadata.resolve_path(path)?;
        let inode = self
            .metadata
            .get_inode(inode_id)
            .ok_or_else(|| SochDBError::NotFound("File not found".into()))?;

        if !inode.is_file() && !inode.is_toon() {
            return Err(SochDBError::InvalidArgument("Not a regular file".into()));
        }

        if inode.blocks.is_empty() {
            return Ok(Vec::new());
        }

        // Read data from blocks
        let mut data = Vec::new();
        for &block_id in &inode.blocks {
            let block_data = (self.block_read_fn)(block_id, inode.size as usize)?;
            data.extend(block_data);
        }

        Ok(data)
    }

    /// Get file stat
    pub fn stat(&self, path: &str) -> Result<Inode> {
        let inode_id = self.metadata.resolve_path(path)?;
        self.metadata
            .get_inode(inode_id)
            .ok_or_else(|| SochDBError::NotFound("File not found".into()))
    }
}

/// Split path into parent and name
fn split_path(path: &str) -> (String, String) {
    let path = path.trim_end_matches('/');
    if let Some(pos) = path.rfind('/') {
        let parent = if pos == 0 { "/" } else { &path[..pos] };
        let name = &path[pos + 1..];
        (parent.to_string(), name.to_string())
    } else {
        ("/".to_string(), path.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn test_inode_row_serialization() {
        let inode = Inode::new_file(42);
        let row = InodeRow::from(&inode);
        let bytes = row.to_bytes();
        let recovered = InodeRow::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.inode_id, 42);
        assert_eq!(recovered.file_type, FileType::Regular as u8);
    }

    #[test]
    fn test_dir_entry_key() {
        let entry = DirEntryRow::new(1, "test.txt".to_string(), 42, FileType::Regular);
        let key = entry.to_key();
        let expected_key = DirEntryRow::make_key(1, "test.txt");
        assert_eq!(key, expected_key);
    }

    #[test]
    fn test_path_split() {
        assert_eq!(
            split_path("/foo/bar"),
            ("/foo".to_string(), "bar".to_string())
        );
        assert_eq!(split_path("/foo"), ("/".to_string(), "foo".to_string()));
        assert_eq!(split_path("foo"), ("/".to_string(), "foo".to_string()));
    }

    #[test]
    fn test_metadata_store() {
        let store = FsMetadataStore::new(|_, _| Ok(()), |_| Ok(()));
        store.init().unwrap();

        // Create file
        let file = store.create_file(ROOT_INODE, "test.txt").unwrap();
        assert!(file.is_file());

        // Lookup
        let found = store.lookup(ROOT_INODE, "test.txt");
        assert_eq!(found, Some(file.id));

        // List dir
        let entries = store.list_dir(ROOT_INODE);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test.txt");
    }

    #[test]
    fn test_path_resolution() {
        let store = FsMetadataStore::new(|_, _| Ok(()), |_| Ok(()));
        store.init().unwrap();

        // Create nested structure: /docs/reports/summary.txt
        let docs = store.create_dir(ROOT_INODE, "docs").unwrap();
        let reports = store.create_dir(docs.id, "reports").unwrap();
        let _summary = store.create_file(reports.id, "summary.txt").unwrap();

        // Resolve paths
        assert_eq!(store.resolve_path("/").unwrap(), ROOT_INODE);
        assert_eq!(store.resolve_path("/docs").unwrap(), docs.id);
        assert_eq!(store.resolve_path("/docs/reports").unwrap(), reports.id);
    }

    #[test]
    fn test_toonfs() {
        let block_counter = AtomicU64::new(0);
        let blocks: std::sync::Arc<RwLock<HashMap<u64, Vec<u8>>>> =
            std::sync::Arc::new(RwLock::new(HashMap::new()));
        let blocks_write = blocks.clone();
        let blocks_read = blocks.clone();

        let fs = SochFS::new(
            |_, _| Ok(()),
            |_| Ok(()),
            move |_inode, data: &[u8]| {
                let id = block_counter.fetch_add(1, Ordering::SeqCst);
                blocks_write.write().insert(id, data.to_vec());
                Ok(id)
            },
            move |id, _size| {
                blocks_read
                    .read()
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| SochDBError::NotFound("Block not found".into()))
            },
        );

        fs.init().unwrap();

        // Create and write file
        fs.create_file("/test.txt").unwrap();
        fs.write_file("/test.txt", b"Hello, SochFS!").unwrap();

        // Read back
        let data = fs.read_file("/test.txt").unwrap();
        assert_eq!(data, b"Hello, SochFS!");

        // Stat
        let stat = fs.stat("/test.txt").unwrap();
        assert_eq!(stat.size, 14);
    }
}
