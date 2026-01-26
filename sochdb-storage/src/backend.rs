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

//! Storage backend abstraction
//!
//! Defines traits for abstracting storage operations, allowing
//! SochDB to work with different storage backends (local filesystem,
//! S3, GCS, Azure Blob, etc.)

use std::path::Path;
use sochdb_core::Result;

/// Object metadata
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub key: String,
    pub size: u64,
    pub last_modified: u64, // Unix timestamp in seconds
}

/// Storage backend trait
///
/// Abstracts storage operations to support multiple backends:
/// - LocalFsBackend: Local filesystem (default)
/// - S3Backend: AWS S3 (planned)
/// - GcsBackend: Google Cloud Storage (planned)
/// - AzureBlobBackend: Azure Blob Storage (planned)
///
/// **Usage:**
/// ```ignore
/// let backend = LocalFsBackend::new("/data")?;
/// backend.put("wal.log", &data)?;
/// let data = backend.get("wal.log")?;
/// ```
pub trait StorageBackend: Send + Sync {
    /// Write data to a key
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;

    /// Read data from a key
    fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Delete a key
    fn delete(&self, key: &str) -> Result<()>;

    /// Check if a key exists
    fn exists(&self, key: &str) -> Result<bool>;

    /// List all keys with a prefix
    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>>;

    /// Sync/flush data to durable storage
    fn sync(&self) -> Result<()>;

    /// Get the base path for this backend (if applicable)
    fn base_path(&self) -> Option<&Path>;
}

/// Local filesystem backend
///
/// Default implementation using local filesystem.
/// All operations are thread-safe.
pub struct LocalFsBackend {
    base_dir: std::path::PathBuf,
}

impl LocalFsBackend {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    fn resolve_path(&self, key: &str) -> std::path::PathBuf {
        self.base_dir.join(key)
    }
}

impl StorageBackend for LocalFsBackend {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = self.resolve_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, data)?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.resolve_path(key);
        let data = std::fs::read(path)?;
        Ok(data)
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.resolve_path(key);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let path = self.resolve_path(key);
        Ok(path.exists())
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        let prefix_path = self.resolve_path(prefix);
        let search_dir = if prefix_path.is_dir() {
            prefix_path
        } else {
            prefix_path.parent().unwrap_or(&self.base_dir).to_path_buf()
        };

        let mut results = Vec::new();
        if search_dir.exists() {
            for entry in std::fs::read_dir(search_dir)? {
                let entry = entry?;
                let path = entry.path();
                let metadata = entry.metadata()?;

                // Get key relative to base_dir
                let key = path
                    .strip_prefix(&self.base_dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();

                // Only include if it matches the prefix
                if key.starts_with(prefix) || prefix.is_empty() {
                    results.push(ObjectMetadata {
                        key,
                        size: metadata.len(),
                        last_modified: metadata
                            .modified()?
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    });
                }
            }
        }

        Ok(results)
    }

    fn sync(&self) -> Result<()> {
        // For local filesystem, we rely on OS page cache
        // Could add explicit fsync here if needed
        Ok(())
    }

    fn base_path(&self) -> Option<&Path> {
        Some(&self.base_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_local_fs_backend() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let backend = LocalFsBackend::new(temp_dir.path())?;

        // Put
        backend.put("test.txt", b"hello world")?;

        // Exists
        assert!(backend.exists("test.txt")?);
        assert!(!backend.exists("nonexistent.txt")?);

        // Get
        let data = backend.get("test.txt")?;
        assert_eq!(data, b"hello world");

        // List
        backend.put("dir/file1.txt", b"data1")?;
        backend.put("dir/file2.txt", b"data2")?;
        let objects = backend.list("dir/")?;
        assert!(objects.len() >= 2);

        // Delete
        backend.delete("test.txt")?;
        assert!(!backend.exists("test.txt")?);

        Ok(())
    }
}
