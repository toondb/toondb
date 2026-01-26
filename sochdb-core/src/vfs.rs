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

//! Virtual Filesystem Types for SochDB
//!
//! POSIX-like filesystem interface backed by WAL for ACID guarantees.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Inode number (unique file identifier)
pub type InodeId = u64;

/// Block number for data storage
pub type BlockId = u64;

/// File types in the VFS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FileType {
    /// Regular file
    Regular = 1,
    /// Directory
    Directory = 2,
    /// Symbolic link
    Symlink = 3,
    /// TOON document (special type for native format)
    SochDocument = 4,
}

/// File permissions (Unix-style)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Permissions {
    pub fn new(read: bool, write: bool, execute: bool) -> Self {
        Self {
            read,
            write,
            execute,
        }
    }

    pub fn all() -> Self {
        Self {
            read: true,
            write: true,
            execute: true,
        }
    }

    pub fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            execute: false,
        }
    }

    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            execute: false,
        }
    }

    pub fn to_mode(&self) -> u8 {
        let mut mode = 0u8;
        if self.read {
            mode |= 0b100;
        }
        if self.write {
            mode |= 0b010;
        }
        if self.execute {
            mode |= 0b001;
        }
        mode
    }

    pub fn from_mode(mode: u8) -> Self {
        Self {
            read: mode & 0b100 != 0,
            write: mode & 0b010 != 0,
            execute: mode & 0b001 != 0,
        }
    }
}

impl Default for Permissions {
    fn default() -> Self {
        Self::read_write()
    }
}

/// Inode structure (file/directory metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inode {
    /// Unique inode number
    pub id: InodeId,
    /// File type
    pub file_type: FileType,
    /// File size in bytes
    pub size: u64,
    /// Data blocks (for regular files)
    pub blocks: Vec<BlockId>,
    /// Permissions
    pub permissions: Permissions,
    /// Creation time (microseconds since epoch)
    pub created_us: u64,
    /// Modification time (microseconds since epoch)
    pub modified_us: u64,
    /// Access time (microseconds since epoch)
    pub accessed_us: u64,
    /// Number of hard links
    pub nlink: u32,
    /// For symlinks: target path
    pub symlink_target: Option<String>,
    /// For TOON documents: schema name
    pub soch_schema: Option<String>,
}

impl Inode {
    pub fn new_file(id: InodeId) -> Self {
        let now = now_micros();
        Self {
            id,
            file_type: FileType::Regular,
            size: 0,
            blocks: Vec::new(),
            permissions: Permissions::read_write(),
            created_us: now,
            modified_us: now,
            accessed_us: now,
            nlink: 1,
            symlink_target: None,
            soch_schema: None,
        }
    }

    pub fn new_directory(id: InodeId) -> Self {
        let now = now_micros();
        Self {
            id,
            file_type: FileType::Directory,
            size: 0,
            blocks: Vec::new(),
            permissions: Permissions::all(),
            created_us: now,
            modified_us: now,
            accessed_us: now,
            nlink: 2, // . and ..
            symlink_target: None,
            soch_schema: None,
        }
    }

    pub fn new_symlink(id: InodeId, target: String) -> Self {
        let now = now_micros();
        Self {
            id,
            file_type: FileType::Symlink,
            size: target.len() as u64,
            blocks: Vec::new(),
            permissions: Permissions::all(),
            created_us: now,
            modified_us: now,
            accessed_us: now,
            nlink: 1,
            symlink_target: Some(target),
            soch_schema: None,
        }
    }

    pub fn new_toon(id: InodeId, schema: String) -> Self {
        let now = now_micros();
        Self {
            id,
            file_type: FileType::SochDocument,
            size: 0,
            blocks: Vec::new(),
            permissions: Permissions::read_write(),
            created_us: now,
            modified_us: now,
            accessed_us: now,
            nlink: 1,
            symlink_target: None,
            soch_schema: Some(schema),
        }
    }

    pub fn is_dir(&self) -> bool {
        self.file_type == FileType::Directory
    }

    pub fn is_file(&self) -> bool {
        self.file_type == FileType::Regular
    }

    pub fn is_symlink(&self) -> bool {
        self.file_type == FileType::Symlink
    }

    pub fn is_toon(&self) -> bool {
        self.file_type == FileType::SochDocument
    }

    pub fn touch(&mut self) {
        self.modified_us = now_micros();
        self.accessed_us = self.modified_us;
    }

    pub fn update_access_time(&mut self) {
        self.accessed_us = now_micros();
    }

    /// Serialize inode to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Deserialize inode from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }
}

/// Directory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    /// Entry name (file/subdirectory name)
    pub name: String,
    /// Inode number
    pub inode: InodeId,
    /// File type (cached for readdir efficiency)
    pub file_type: FileType,
}

impl DirEntry {
    pub fn new(name: impl Into<String>, inode: InodeId, file_type: FileType) -> Self {
        Self {
            name: name.into(),
            inode,
            file_type,
        }
    }
}

/// Directory contents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Directory {
    /// Parent inode (0 for root)
    pub parent: InodeId,
    /// Directory entries
    pub entries: Vec<DirEntry>,
}

impl Directory {
    pub fn new(parent: InodeId) -> Self {
        Self {
            parent,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: DirEntry) {
        self.entries.push(entry);
    }

    pub fn remove_entry(&mut self, name: &str) -> Option<DirEntry> {
        if let Some(pos) = self.entries.iter().position(|e| e.name == name) {
            Some(self.entries.remove(pos))
        } else {
            None
        }
    }

    pub fn find_entry(&self, name: &str) -> Option<&DirEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }
}

/// Superblock - filesystem metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Superblock {
    /// Magic number for identification
    pub magic: [u8; 4],
    /// Filesystem version
    pub version: u32,
    /// Root inode number
    pub root_inode: InodeId,
    /// Next free inode number
    pub next_inode: InodeId,
    /// Next free block number
    pub next_block: BlockId,
    /// Total inodes allocated
    pub total_inodes: u64,
    /// Total blocks used
    pub total_blocks: u64,
    /// Block size in bytes
    pub block_size: u32,
    /// Creation time
    pub created_us: u64,
    /// Last mount time
    pub mounted_us: u64,
    /// Filesystem label
    pub label: String,
}

impl Superblock {
    pub const MAGIC: [u8; 4] = *b"TOON";
    pub const VERSION: u32 = 1;
    pub const DEFAULT_BLOCK_SIZE: u32 = 4096;

    pub fn new(label: impl Into<String>) -> Self {
        let now = now_micros();
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            root_inode: 1, // Root directory is always inode 1
            next_inode: 2,
            next_block: 0,
            total_inodes: 1,
            total_blocks: 0,
            block_size: Self::DEFAULT_BLOCK_SIZE,
            created_us: now,
            mounted_us: now,
            label: label.into(),
        }
    }

    /// Allocate a new inode number
    pub fn alloc_inode(&mut self) -> InodeId {
        let id = self.next_inode;
        self.next_inode += 1;
        self.total_inodes += 1;
        id
    }

    /// Allocate block numbers
    pub fn alloc_blocks(&mut self, count: u64) -> Vec<BlockId> {
        let start = self.next_block;
        self.next_block += count;
        self.total_blocks += count;
        (start..start + count).collect()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let sb: Self = bincode::deserialize(data).map_err(|e| e.to_string())?;
        if sb.magic != Self::MAGIC {
            return Err("Invalid magic number".into());
        }
        Ok(sb)
    }
}

/// VFS operation types for WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VfsOp {
    /// Create file
    CreateFile {
        parent: InodeId,
        name: String,
        inode: Inode,
    },
    /// Create directory
    CreateDir {
        parent: InodeId,
        name: String,
        inode: Inode,
    },
    /// Delete entry
    Delete {
        parent: InodeId,
        name: String,
        inode: InodeId,
    },
    /// Rename/move entry
    Rename {
        old_parent: InodeId,
        old_name: String,
        new_parent: InodeId,
        new_name: String,
        inode: InodeId,
    },
    /// Write data block
    WriteBlock {
        inode: InodeId,
        block: BlockId,
        data: Vec<u8>,
    },
    /// Truncate file
    Truncate { inode: InodeId, new_size: u64 },
    /// Update inode metadata
    UpdateInode { inode: Inode },
    /// Update superblock
    UpdateSuperblock { superblock: Superblock },
    /// Create symlink
    CreateSymlink {
        parent: InodeId,
        name: String,
        inode: Inode,
        target: String,
    },
}

/// File stat information (like POSIX stat)
#[derive(Debug, Clone)]
pub struct FileStat {
    pub inode: InodeId,
    pub file_type: FileType,
    pub size: u64,
    pub blocks: u64,
    pub block_size: u32,
    pub nlink: u32,
    pub permissions: Permissions,
    pub created: SystemTime,
    pub modified: SystemTime,
    pub accessed: SystemTime,
}

impl From<&Inode> for FileStat {
    fn from(inode: &Inode) -> Self {
        Self {
            inode: inode.id,
            file_type: inode.file_type,
            size: inode.size,
            blocks: inode.blocks.len() as u64,
            block_size: Superblock::DEFAULT_BLOCK_SIZE,
            nlink: inode.nlink,
            permissions: inode.permissions,
            created: micros_to_system_time(inode.created_us),
            modified: micros_to_system_time(inode.modified_us),
            accessed: micros_to_system_time(inode.accessed_us),
        }
    }
}

/// Get current time in microseconds since Unix epoch
fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Convert microseconds to SystemTime
fn micros_to_system_time(micros: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + std::time::Duration::from_micros(micros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inode_serialization() {
        let inode = Inode::new_file(42);
        let bytes = inode.to_bytes().expect("Failed to serialize inode");
        let parsed = Inode::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.id, 42);
        assert!(parsed.is_file());
    }

    #[test]
    fn test_directory_operations() {
        let mut dir = Directory::new(1);
        dir.add_entry(DirEntry::new("file1.txt", 10, FileType::Regular));
        dir.add_entry(DirEntry::new("subdir", 11, FileType::Directory));

        assert!(dir.contains("file1.txt"));
        assert!(dir.contains("subdir"));
        assert!(!dir.contains("nonexistent"));

        let entry = dir.find_entry("file1.txt").unwrap();
        assert_eq!(entry.inode, 10);

        let removed = dir.remove_entry("file1.txt").unwrap();
        assert_eq!(removed.inode, 10);
        assert!(!dir.contains("file1.txt"));
    }

    #[test]
    fn test_superblock() {
        let mut sb = Superblock::new("test-fs");
        assert_eq!(sb.magic, Superblock::MAGIC);
        assert_eq!(sb.root_inode, 1);

        let inode1 = sb.alloc_inode();
        let inode2 = sb.alloc_inode();
        assert_eq!(inode1, 2);
        assert_eq!(inode2, 3);

        let blocks = sb.alloc_blocks(5);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0], 0);
        assert_eq!(blocks[4], 4);
    }

    #[test]
    fn test_permissions() {
        let perms = Permissions::new(true, true, false);
        assert_eq!(perms.to_mode(), 0b110);

        let from_mode = Permissions::from_mode(0b101);
        assert!(from_mode.read);
        assert!(!from_mode.write);
        assert!(from_mode.execute);
    }
}
