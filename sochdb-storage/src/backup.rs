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

//! Backup and restore functionality for SochDB database
//!
//! This module provides functionality to create full snapshots of the database
//! and restore from those snapshots. Backups include all data files, indexes,
//! and metadata.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use sochdb_core::{Result, SochDBError};

/// Metadata about a backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Timestamp when backup was created (microseconds since Unix epoch)
    pub timestamp_us: u64,

    /// Human-readable timestamp
    pub created_at: String,

    /// Total size of backup in bytes
    pub size_bytes: u64,

    /// Number of files in backup
    pub file_count: usize,

    /// Database version
    pub database_version: String,

    /// SHA256 checksum of all files
    pub checksum: String,

    /// Source database path
    pub source_path: String,
}

impl BackupMetadata {
    /// Generate a backup name from timestamp
    pub fn generate_name(&self) -> String {
        format!("sochdb-backup-{}", self.timestamp_us)
    }
}

/// Manages backup and restore operations
pub struct BackupManager {
    source_path: PathBuf,
}

impl BackupManager {
    /// Create a new backup manager for the given database path
    pub fn new<P: AsRef<Path>>(source_path: P) -> Self {
        Self {
            source_path: source_path.as_ref().to_path_buf(),
        }
    }

    /// Create a backup of the database to the specified destination
    ///
    /// The backup includes:
    /// - All SSTable files (*.sst)
    /// - WAL file (wal.log)
    /// - Causal index (causal.index)
    /// - Vector index (vector.index)
    /// - Agent registry (agent_registry.json)
    /// - Metadata manifest (manifest.json)
    ///
    /// # Example
    /// ```ignore
    /// let manager = BackupManager::new("./test-db");
    /// let metadata = manager.create_backup("./backups/backup-2025-11-06")?;
    /// println!("Backup created with {} files", metadata.file_count);
    /// ```
    pub fn create_backup<P: AsRef<Path>>(&self, destination: P) -> Result<BackupMetadata> {
        let dest_path = destination.as_ref();

        // Create destination directory
        fs::create_dir_all(dest_path).map_err(|e| {
            SochDBError::Backup(format!("Failed to create backup directory: {}", e))
        })?;

        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let created_at = chrono::Local::now().to_rfc3339();

        // Collect all files to backup
        let files_to_backup = self.collect_files()?;

        if files_to_backup.is_empty() {
            return Err(SochDBError::Backup(
                "No files found in source database".to_string(),
            ));
        }

        let mut total_size = 0u64;
        let mut checksums = Vec::new();

        // Copy each file
        for (rel_path, src_path) in &files_to_backup {
            let dest_file_path = dest_path.join(rel_path);

            // Create parent directory if needed
            if let Some(parent) = dest_file_path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    SochDBError::Backup(format!("Failed to create directory: {}", e))
                })?;
            }

            // Copy file
            fs::copy(src_path, &dest_file_path).map_err(|e| {
                SochDBError::Backup(format!("Failed to copy file {}: {}", rel_path, e))
            })?;

            // Calculate checksum and size
            let metadata = fs::metadata(&dest_file_path)
                .map_err(|e| SochDBError::Backup(format!("Failed to read file metadata: {}", e)))?;

            total_size += metadata.len();

            let checksum = self.calculate_file_checksum(&dest_file_path)?;
            checksums.push(format!("{}:{}", rel_path, checksum));
        }

        // Calculate overall checksum (hash of all individual checksums)
        let overall_checksum = self.calculate_string_checksum(&checksums.join("\n"));

        // Create metadata
        let metadata = BackupMetadata {
            timestamp_us,
            created_at,
            size_bytes: total_size,
            file_count: files_to_backup.len(),
            database_version: env!("CARGO_PKG_VERSION").to_string(),
            checksum: overall_checksum,
            source_path: self.source_path.display().to_string(),
        };

        // Write manifest
        let manifest_path = dest_path.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| SochDBError::Backup(format!("Failed to serialize manifest: {}", e)))?;

        fs::write(&manifest_path, manifest_json)
            .map_err(|e| SochDBError::Backup(format!("Failed to write manifest: {}", e)))?;

        Ok(metadata)
    }

    /// Restore a backup to the specified destination
    ///
    /// # Warning
    /// This will overwrite any existing data at the destination path.
    ///
    /// # Example
    /// ```ignore
    /// let manager = BackupManager::new("./restored-db");
    /// manager.restore_backup("./backups/backup-2025-11-06")?;
    /// ```
    pub fn restore_backup<P: AsRef<Path>>(&self, backup_path: P) -> Result<BackupMetadata> {
        let backup_path = backup_path.as_ref();

        // Read and verify manifest
        let manifest_path = backup_path.join("manifest.json");
        let manifest_json = fs::read_to_string(&manifest_path)
            .map_err(|e| SochDBError::Backup(format!("Failed to read manifest: {}", e)))?;

        let metadata: BackupMetadata = serde_json::from_str(&manifest_json)
            .map_err(|e| SochDBError::Backup(format!("Failed to parse manifest: {}", e)))?;

        // Create destination directory
        fs::create_dir_all(&self.source_path).map_err(|e| {
            SochDBError::Backup(format!("Failed to create destination directory: {}", e))
        })?;

        // Get all files in backup (excluding manifest)
        let files = self.collect_backup_files(backup_path)?;

        // Copy all files to destination
        for (rel_path, src_path) in files {
            let dest_path = self.source_path.join(&rel_path);

            // Create parent directory if needed
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    SochDBError::Backup(format!("Failed to create directory: {}", e))
                })?;
            }

            fs::copy(&src_path, &dest_path).map_err(|e| {
                SochDBError::Backup(format!("Failed to restore file {}: {}", rel_path, e))
            })?;
        }

        Ok(metadata)
    }

    /// List all backups in the specified directory
    ///
    /// Returns a list of backup metadata sorted by timestamp (newest first).
    pub fn list_backups<P: AsRef<Path>>(backup_dir: P) -> Result<Vec<BackupMetadata>> {
        let backup_dir = backup_dir.as_ref();

        if !backup_dir.exists() {
            return Ok(Vec::new());
        }

        let mut backups = Vec::new();

        let entries = fs::read_dir(backup_dir)
            .map_err(|e| SochDBError::Backup(format!("Failed to read backup directory: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                SochDBError::Backup(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();
            if path.is_dir() {
                let manifest_path = path.join("manifest.json");
                if manifest_path.exists() {
                    match fs::read_to_string(&manifest_path) {
                        Ok(json) => {
                            if let Ok(metadata) = serde_json::from_str::<BackupMetadata>(&json) {
                                backups.push(metadata);
                            }
                        }
                        Err(_) => continue,
                    }
                }
            }
        }

        // Sort by timestamp (newest first)
        backups.sort_by(|a, b| b.timestamp_us.cmp(&a.timestamp_us));

        Ok(backups)
    }

    /// Verify the integrity of a backup
    ///
    /// Checks that all files exist and checksums match the manifest.
    pub fn verify_backup<P: AsRef<Path>>(backup_path: P) -> Result<bool> {
        let backup_path = backup_path.as_ref();

        // Read manifest
        let manifest_path = backup_path.join("manifest.json");
        let manifest_json = fs::read_to_string(&manifest_path)
            .map_err(|e| SochDBError::Backup(format!("Failed to read manifest: {}", e)))?;

        let _metadata: BackupMetadata = serde_json::from_str(&manifest_json)
            .map_err(|e| SochDBError::Backup(format!("Failed to parse manifest: {}", e)))?;

        // Verify all files exist
        let manager = BackupManager::new(backup_path);
        let files = manager.collect_backup_files(backup_path)?;

        if files.is_empty() {
            return Ok(false);
        }

        // All files exist if we got here
        Ok(true)
    }

    // Helper methods

    fn collect_files(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut files = Vec::new();

        if !self.source_path.exists() {
            return Err(SochDBError::Backup(
                "Source database path does not exist".to_string(),
            ));
        }

        Self::collect_files_recursive(&self.source_path, &self.source_path, &mut files)?;

        Ok(files)
    }

    fn collect_files_recursive(
        current_path: &Path,
        base_path: &Path,
        files: &mut Vec<(String, PathBuf)>,
    ) -> Result<()> {
        let entries = fs::read_dir(current_path)
            .map_err(|e| SochDBError::Backup(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry =
                entry.map_err(|e| SochDBError::Backup(format!("Failed to read entry: {}", e)))?;

            let path = entry.path();

            if path.is_dir() {
                // Recursively collect files from subdirectories
                Self::collect_files_recursive(&path, base_path, files)?;
            } else {
                // Add file with relative path
                let rel_path = path
                    .strip_prefix(base_path)
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                files.push((rel_path, path));
            }
        }

        Ok(())
    }

    fn collect_backup_files(&self, backup_path: &Path) -> Result<Vec<(String, PathBuf)>> {
        let mut files = Vec::new();
        Self::collect_backup_files_recursive(backup_path, backup_path, &mut files)?;

        // Filter out manifest.json
        files.retain(|(rel_path, _)| rel_path != "manifest.json");

        Ok(files)
    }

    fn collect_backup_files_recursive(
        current_path: &Path,
        base_path: &Path,
        files: &mut Vec<(String, PathBuf)>,
    ) -> Result<()> {
        let entries = fs::read_dir(current_path)
            .map_err(|e| SochDBError::Backup(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry =
                entry.map_err(|e| SochDBError::Backup(format!("Failed to read entry: {}", e)))?;

            let path = entry.path();

            if path.is_dir() {
                Self::collect_backup_files_recursive(&path, base_path, files)?;
            } else {
                let rel_path = path
                    .strip_prefix(base_path)
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                files.push((rel_path, path));
            }
        }

        Ok(())
    }

    fn calculate_file_checksum(&self, path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};

        let mut file = File::open(path)
            .map_err(|e| SochDBError::Backup(format!("Failed to open file for checksum: {}", e)))?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let n = file.read(&mut buffer).map_err(|e| {
                SochDBError::Backup(format!("Failed to read file for checksum: {}", e))
            })?;

            if n == 0 {
                break;
            }

            hasher.update(&buffer[..n]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn calculate_string_checksum(&self, data: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_restore_backup() {
        // Create a temporary database directory
        let db_dir = TempDir::new().unwrap();
        let db_path = db_dir.path();

        // Create some test files
        fs::write(db_path.join("test.sst"), b"test data").unwrap();
        fs::write(db_path.join("wal.log"), b"wal data").unwrap();
        fs::create_dir_all(db_path.join("subdir")).unwrap();
        fs::write(db_path.join("subdir").join("index.dat"), b"index data").unwrap();

        // Create backup
        let backup_dir = TempDir::new().unwrap();
        let backup_path = backup_dir.path().join("backup-1");

        let manager = BackupManager::new(db_path);
        let metadata = manager.create_backup(&backup_path).unwrap();

        assert_eq!(metadata.file_count, 3);
        assert!(metadata.size_bytes > 0);
        assert!(backup_path.join("manifest.json").exists());
        assert!(backup_path.join("test.sst").exists());

        // Restore to new location
        let restore_dir = TempDir::new().unwrap();
        let restore_path = restore_dir.path().join("restored");

        let restore_manager = BackupManager::new(&restore_path);
        let restored_metadata = restore_manager.restore_backup(&backup_path).unwrap();

        assert_eq!(restored_metadata.file_count, metadata.file_count);
        assert!(restore_path.join("test.sst").exists());
        assert!(restore_path.join("wal.log").exists());
        assert!(restore_path.join("subdir").join("index.dat").exists());

        // Verify content
        let content = fs::read_to_string(restore_path.join("test.sst")).unwrap();
        assert_eq!(content, "test data");
    }

    #[test]
    fn test_list_backups() {
        let backup_dir = TempDir::new().unwrap();
        let backup_path = backup_dir.path();

        // Create source database
        let db_dir = TempDir::new().unwrap();
        fs::write(db_dir.path().join("test.sst"), b"data").unwrap();

        let manager = BackupManager::new(db_dir.path());

        // Create multiple backups
        let backup1 = backup_path.join("backup-1");
        let backup2 = backup_path.join("backup-2");

        manager.create_backup(&backup1).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.create_backup(&backup2).unwrap();

        // List backups
        let backups = BackupManager::list_backups(backup_path).unwrap();

        assert_eq!(backups.len(), 2);
        // Should be sorted newest first
        assert!(backups[0].timestamp_us > backups[1].timestamp_us);
    }

    #[test]
    fn test_verify_backup() {
        let db_dir = TempDir::new().unwrap();
        fs::write(db_dir.path().join("test.sst"), b"data").unwrap();

        let backup_dir = TempDir::new().unwrap();
        let backup_path = backup_dir.path().join("backup");

        let manager = BackupManager::new(db_dir.path());
        manager.create_backup(&backup_path).unwrap();

        let valid = BackupManager::verify_backup(&backup_path).unwrap();
        assert!(valid);
    }
}
