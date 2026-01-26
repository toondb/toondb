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

//! LSM MANIFEST File Implementation (Gap #13 Fix)
//!
//! Provides atomic tracking of SSTable state for crash-safe recovery.
//! Inspired by LevelDB/RocksDB MANIFEST file design.
//!
//! The MANIFEST tracks:
//! - Current version number
//! - List of active SSTables per level
//! - Compaction state
//!
//! On startup, the LSM tree reads the MANIFEST to determine which SSTables
//! are valid, rather than blindly scanning the filesystem.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use sochdb_core::Result;
use tracing::{debug, info, warn};

/// Name of the MANIFEST file
#[allow(dead_code)]
const MANIFEST_FILENAME: &str = "MANIFEST";

/// Name of the CURRENT file (pointer to active MANIFEST)
const CURRENT_FILENAME: &str = "CURRENT";

/// A version edit represents a change to the LSM state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEdit {
    /// Version number (monotonically increasing)
    pub version: u64,
    /// SSTables added in this edit
    pub added_files: Vec<FileMetadata>,
    /// SSTables removed in this edit
    pub removed_files: Vec<FileMetadata>,
    /// Log sequence number
    pub log_number: Option<u64>,
    /// Next file ID
    pub next_file_id: Option<u64>,
}

/// Metadata about an SSTable file
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileMetadata {
    /// Level number (0-6)
    pub level: usize,
    /// File ID (used for filename generation)
    pub file_id: u64,
    /// File size in bytes
    pub file_size: u64,
    /// Smallest key in this SSTable (hex-encoded edge_id)
    pub smallest_key: Option<String>,
    /// Largest key in this SSTable (hex-encoded edge_id)
    pub largest_key: Option<String>,
}

/// Current LSM state derived from MANIFEST
#[derive(Debug, Default)]
pub struct LsmState {
    /// Current version number
    pub version: u64,
    /// Active files per level (level -> set of file_ids)
    pub active_files: Vec<HashSet<u64>>,
    /// Log sequence number
    pub log_number: u64,
    /// Next file ID to use
    pub next_file_id: u64,
}

impl LsmState {
    /// Create a new empty state
    pub fn new() -> Self {
        Self {
            version: 0,
            active_files: vec![HashSet::new(); 7], // 7 levels
            log_number: 0,
            next_file_id: 1,
        }
    }

    /// Apply a version edit to the state
    pub fn apply(&mut self, edit: &VersionEdit) {
        self.version = edit.version;

        // Add new files
        for file in &edit.added_files {
            if file.level < self.active_files.len() {
                self.active_files[file.level].insert(file.file_id);
            }
        }

        // Remove deleted files
        for file in &edit.removed_files {
            if file.level < self.active_files.len() {
                self.active_files[file.level].remove(&file.file_id);
            }
        }

        // Update metadata
        if let Some(log_num) = edit.log_number {
            self.log_number = log_num;
        }
        if let Some(next_id) = edit.next_file_id {
            self.next_file_id = next_id;
        }
    }

    /// Check if a file is active
    pub fn is_file_active(&self, level: usize, file_id: u64) -> bool {
        self.active_files
            .get(level)
            .map(|files| files.contains(&file_id))
            .unwrap_or(false)
    }
}

/// MANIFEST file manager
pub struct Manifest {
    /// Directory containing the MANIFEST
    data_dir: PathBuf,
    /// Current MANIFEST file number
    manifest_number: u64,
    /// Writer for appending edits
    writer: Option<BufWriter<File>>,
    /// Current LSM state
    state: LsmState,
}

impl Manifest {
    /// Open or create a MANIFEST in the given directory
    pub fn open<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        fs::create_dir_all(&data_dir)?;

        // Check for CURRENT file
        let current_path = data_dir.join(CURRENT_FILENAME);
        let (manifest_number, state) = if current_path.exists() {
            // Read existing MANIFEST
            let current_content = fs::read_to_string(&current_path)?;
            let manifest_name = current_content.trim();
            let manifest_number = parse_manifest_number(manifest_name).unwrap_or(1);
            let manifest_path = data_dir.join(manifest_name);

            if manifest_path.exists() {
                let state = Self::read_manifest(&manifest_path)?;
                info!(
                    "Loaded MANIFEST-{:06} with version {}",
                    manifest_number, state.version
                );
                (manifest_number, state)
            } else {
                warn!("MANIFEST file {} not found, starting fresh", manifest_name);
                (manifest_number + 1, LsmState::new())
            }
        } else {
            // No CURRENT file, start fresh
            debug!("No CURRENT file found, creating new MANIFEST");
            (1, LsmState::new())
        };

        let mut manifest = Self {
            data_dir,
            manifest_number,
            writer: None,
            state,
        };

        // Open writer for appending
        manifest.open_writer()?;

        Ok(manifest)
    }

    /// Read all edits from a MANIFEST file
    fn read_manifest(path: &Path) -> Result<LsmState> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut state = LsmState::new();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            match serde_json::from_str::<VersionEdit>(&line) {
                Ok(edit) => state.apply(&edit),
                Err(e) => {
                    warn!("Failed to parse MANIFEST line: {}", e);
                    // Continue reading - partial corruption shouldn't fail entire load
                }
            }
        }

        Ok(state)
    }

    /// Open writer for the current MANIFEST
    fn open_writer(&mut self) -> Result<()> {
        let manifest_name = format!("MANIFEST-{:06}", self.manifest_number);
        let manifest_path = self.data_dir.join(&manifest_name);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&manifest_path)?;

        self.writer = Some(BufWriter::new(file));

        // Update CURRENT file atomically
        let current_path = self.data_dir.join(CURRENT_FILENAME);
        let temp_path = self.data_dir.join("CURRENT.tmp");
        fs::write(&temp_path, &manifest_name)?;
        fs::rename(&temp_path, &current_path)?;

        Ok(())
    }

    /// Log a version edit to the MANIFEST
    pub fn log_edit(&mut self, edit: &VersionEdit) -> Result<()> {
        // Apply to in-memory state
        self.state.apply(edit);

        // Write to MANIFEST file
        if let Some(ref mut writer) = self.writer {
            let line = serde_json::to_string(edit)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writeln!(writer, "{}", line)?;
            writer.flush()?;
            debug!("Logged version edit: version={}", edit.version);
        }

        Ok(())
    }

    /// Record adding new SSTables
    pub fn add_files(&mut self, files: Vec<FileMetadata>) -> Result<u64> {
        let version = self.state.version + 1;
        let edit = VersionEdit {
            version,
            added_files: files,
            removed_files: vec![],
            log_number: None,
            next_file_id: None,
        };
        self.log_edit(&edit)?;
        Ok(version)
    }

    /// Record removing SSTables (after compaction)
    pub fn remove_files(&mut self, files: Vec<FileMetadata>) -> Result<u64> {
        let version = self.state.version + 1;
        let edit = VersionEdit {
            version,
            added_files: vec![],
            removed_files: files,
            log_number: None,
            next_file_id: None,
        };
        self.log_edit(&edit)?;
        Ok(version)
    }

    /// Record a compaction: add new files, remove old files
    pub fn log_compaction(
        &mut self,
        added: Vec<FileMetadata>,
        removed: Vec<FileMetadata>,
    ) -> Result<u64> {
        let version = self.state.version + 1;
        let edit = VersionEdit {
            version,
            added_files: added,
            removed_files: removed,
            log_number: None,
            next_file_id: None,
        };
        self.log_edit(&edit)?;
        Ok(version)
    }

    /// Update next file ID
    pub fn set_next_file_id(&mut self, next_id: u64) -> Result<()> {
        let version = self.state.version + 1;
        let edit = VersionEdit {
            version,
            added_files: vec![],
            removed_files: vec![],
            log_number: None,
            next_file_id: Some(next_id),
        };
        self.log_edit(&edit)
    }

    /// Get the current LSM state
    pub fn state(&self) -> &LsmState {
        &self.state
    }

    /// Get the current version number
    pub fn version(&self) -> u64 {
        self.state.version
    }

    /// Get active files for a level
    pub fn active_files(&self, level: usize) -> Option<&HashSet<u64>> {
        self.state.active_files.get(level)
    }

    /// Sync to disk
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.writer {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        Ok(())
    }

    /// Create a new MANIFEST file (for periodic compaction of the MANIFEST itself)
    pub fn rotate(&mut self) -> Result<()> {
        // Sync current MANIFEST
        self.sync()?;

        // Create new MANIFEST
        self.manifest_number += 1;
        self.open_writer()?;

        // Write a snapshot of current state
        let snapshot = VersionEdit {
            version: self.state.version,
            added_files: self.collect_all_files(),
            removed_files: vec![],
            log_number: Some(self.state.log_number),
            next_file_id: Some(self.state.next_file_id),
        };
        self.log_edit(&snapshot)?;

        Ok(())
    }

    /// Collect all active files as FileMetadata
    fn collect_all_files(&self) -> Vec<FileMetadata> {
        let mut files = Vec::new();
        for (level, file_ids) in self.state.active_files.iter().enumerate() {
            for &file_id in file_ids {
                files.push(FileMetadata {
                    level,
                    file_id,
                    file_size: 0, // Size not tracked in state
                    smallest_key: None,
                    largest_key: None,
                });
            }
        }
        files
    }
}

/// Parse manifest number from filename like "MANIFEST-000001"
fn parse_manifest_number(name: &str) -> Option<u64> {
    name.strip_prefix("MANIFEST-").and_then(|n| n.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_manifest_basic() {
        let temp_dir = TempDir::new().unwrap();
        let mut manifest = Manifest::open(temp_dir.path()).unwrap();

        // Add some files
        let files = vec![
            FileMetadata {
                level: 0,
                file_id: 1,
                file_size: 1000,
                smallest_key: None,
                largest_key: None,
            },
            FileMetadata {
                level: 0,
                file_id: 2,
                file_size: 2000,
                smallest_key: None,
                largest_key: None,
            },
        ];
        manifest.add_files(files).unwrap();

        assert_eq!(manifest.version(), 1);
        assert!(manifest.state.is_file_active(0, 1));
        assert!(manifest.state.is_file_active(0, 2));
        assert!(!manifest.state.is_file_active(0, 3));
    }

    #[test]
    fn test_manifest_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Create and populate manifest
        {
            let mut manifest = Manifest::open(temp_dir.path()).unwrap();
            manifest
                .add_files(vec![FileMetadata {
                    level: 0,
                    file_id: 1,
                    file_size: 1000,
                    smallest_key: None,
                    largest_key: None,
                }])
                .unwrap();
            manifest.sync().unwrap();
        }

        // Reopen and verify state
        {
            let manifest = Manifest::open(temp_dir.path()).unwrap();
            assert!(manifest.state.is_file_active(0, 1));
        }
    }

    #[test]
    fn test_compaction_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let mut manifest = Manifest::open(temp_dir.path()).unwrap();

        // Add L0 files
        manifest
            .add_files(vec![
                FileMetadata {
                    level: 0,
                    file_id: 1,
                    file_size: 1000,
                    smallest_key: None,
                    largest_key: None,
                },
                FileMetadata {
                    level: 0,
                    file_id: 2,
                    file_size: 1000,
                    smallest_key: None,
                    largest_key: None,
                },
            ])
            .unwrap();

        // Simulate compaction: merge L0 -> L1
        manifest
            .log_compaction(
                vec![FileMetadata {
                    level: 1,
                    file_id: 3,
                    file_size: 2000,
                    smallest_key: None,
                    largest_key: None,
                }],
                vec![
                    FileMetadata {
                        level: 0,
                        file_id: 1,
                        file_size: 1000,
                        smallest_key: None,
                        largest_key: None,
                    },
                    FileMetadata {
                        level: 0,
                        file_id: 2,
                        file_size: 1000,
                        smallest_key: None,
                        largest_key: None,
                    },
                ],
            )
            .unwrap();

        // Verify state
        assert!(!manifest.state.is_file_active(0, 1));
        assert!(!manifest.state.is_file_active(0, 2));
        assert!(manifest.state.is_file_active(1, 3));
    }
}
