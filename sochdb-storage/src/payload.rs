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

//! Payload storage for variable-length data
//!
//! TOON records can reference variable-length payloads (text, binary data, embeddings).
//! These are stored separately in payload segments and referenced by
//! (payload_offset, payload_length, compression_type).
//!
//! **Design Goals:**
//! - Append-only for fast writes
//! - Immutable payloads (no in-place updates)
//! - Support compression (LZ4 for warm, ZSTD for cold)
//! - Thread-safe concurrent access
//! - Memory-mapped for fast reads

use memmap2::MmapOptions;
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use sochdb_core::{Result, SochDBError};

/// Compression type for payloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum CompressionType {
    None = 0,
    LZ4 = 1,
    ZSTD = 2,
}

impl CompressionType {
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(CompressionType::None),
            1 => Ok(CompressionType::LZ4),
            2 => Ok(CompressionType::ZSTD),
            _ => Err(SochDBError::InvalidArgument(format!(
                "Invalid compression type: {}",
                value
            ))),
        }
    }
}

/// Payload metadata stored in index
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PayloadMeta {
    pub edge_id: u128,
    pub offset: u64,
    pub length: u32,
    pub compression: CompressionType,
    pub uncompressed_length: u32,
}

/// Pluggable trait for payload index storage
///
/// Allows switching between in-memory (HashMap) and disk-backed (sled) implementations.
pub trait PayloadIndex: Send + Sync {
    fn insert(&self, edge_id: u128, meta: PayloadMeta) -> Result<()>;
    fn get(&self, edge_id: u128) -> Result<Option<PayloadMeta>>;
    fn contains_key(&self, edge_id: u128) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn iter_values(&self) -> Box<dyn Iterator<Item = PayloadMeta> + '_>;
    fn save(&self) -> Result<()>;
}

/// In-memory HashMap index (default, fast but memory-hungry)
struct HashMapIndex {
    inner: Arc<RwLock<std::collections::HashMap<u128, PayloadMeta>>>,
    index_path: PathBuf,
}

impl HashMapIndex {
    fn new(index_path: PathBuf) -> Result<Self> {
        // Ensure parent directory exists before any file operations
        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let inner = if index_path.exists() {
            let loaded = Self::load_from_disk(&index_path)?;
            tracing::info!(
                "Loaded payload index with {} entries from {:?}",
                loaded.len(),
                index_path
            );
            Arc::new(RwLock::new(loaded))
        } else {
            tracing::info!(
                "No existing payload index found at {:?}, starting fresh",
                index_path
            );
            Arc::new(RwLock::new(std::collections::HashMap::new()))
        };
        Ok(Self { inner, index_path })
    }

    fn load_from_disk(path: &Path) -> Result<std::collections::HashMap<u128, PayloadMeta>> {
        let file = File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"CHRLPAY1" {
            return Err(SochDBError::Corruption(
                "Invalid payload index magic".into(),
            ));
        }

        // Read count
        let mut count_bytes = [0u8; 8];
        reader.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes);

        let mut index = std::collections::HashMap::new();

        for _ in 0..count {
            let mut edge_id_bytes = [0u8; 16];
            reader.read_exact(&mut edge_id_bytes)?;
            let edge_id = u128::from_le_bytes(edge_id_bytes);

            let mut offset_bytes = [0u8; 8];
            reader.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            let mut length_bytes = [0u8; 4];
            reader.read_exact(&mut length_bytes)?;
            let length = u32::from_le_bytes(length_bytes);

            let mut compression_byte = [0u8; 1];
            reader.read_exact(&mut compression_byte)?;
            let compression = CompressionType::from_u8(compression_byte[0])?;

            let mut uncompressed_bytes = [0u8; 4];
            reader.read_exact(&mut uncompressed_bytes)?;
            let uncompressed_length = u32::from_le_bytes(uncompressed_bytes);

            index.insert(
                edge_id,
                PayloadMeta {
                    edge_id,
                    offset,
                    length,
                    compression,
                    uncompressed_length,
                },
            );
        }

        Ok(index)
    }
}

impl PayloadIndex for HashMapIndex {
    fn insert(&self, edge_id: u128, meta: PayloadMeta) -> Result<()> {
        self.inner.write().insert(edge_id, meta);
        Ok(())
    }

    fn get(&self, edge_id: u128) -> Result<Option<PayloadMeta>> {
        Ok(self.inner.read().get(&edge_id).cloned())
    }

    fn contains_key(&self, edge_id: u128) -> bool {
        self.inner.read().contains_key(&edge_id)
    }

    fn len(&self) -> usize {
        self.inner.read().len()
    }

    fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    fn iter_values(&self) -> Box<dyn Iterator<Item = PayloadMeta> + '_> {
        let values: Vec<_> = self.inner.read().values().cloned().collect();
        Box::new(values.into_iter())
    }

    fn save(&self) -> Result<()> {
        let inner = self.inner.read();

        // Ensure parent directory exists
        if let Some(parent) = self.index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to a temp file first, then rename for atomicity
        let temp_path = self.index_path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let mut writer = std::io::BufWriter::new(file);

        // Write magic header
        writer.write_all(b"CHRLPAY1")?;

        // Write count
        writer.write_all(&(inner.len() as u64).to_le_bytes())?;

        // Write entries
        for meta in inner.values() {
            writer.write_all(&meta.edge_id.to_le_bytes())?;
            writer.write_all(&meta.offset.to_le_bytes())?;
            writer.write_all(&meta.length.to_le_bytes())?;
            writer.write_all(&[meta.compression as u8])?;
            writer.write_all(&meta.uncompressed_length.to_le_bytes())?;
        }

        // Flush buffer and sync to disk
        std::io::Write::flush(&mut writer)?;
        let file = writer
            .into_inner()
            .map_err(|e| SochDBError::Internal(format!("Failed to flush: {}", e)))?;
        file.sync_all()?;

        // Atomically rename temp file to actual index file
        std::fs::rename(&temp_path, &self.index_path)?;

        tracing::debug!(
            entries = inner.len(),
            path = %self.index_path.display(),
            "Saved payload index to disk"
        );

        Ok(())
    }
}

/// Disk-backed sled index (scales to billions, minimal RAM)
struct SledIndex {
    db: sled::Db,
}

impl SledIndex {
    fn new(index_path: PathBuf) -> Result<Self> {
        let db = sled::open(index_path)
            .map_err(|e| SochDBError::Internal(format!("Failed to open sled: {}", e)))?;
        Ok(Self { db })
    }
}

impl PayloadIndex for SledIndex {
    fn insert(&self, edge_id: u128, meta: PayloadMeta) -> Result<()> {
        let key = edge_id.to_le_bytes();
        let value = bincode::serialize(&meta)
            .map_err(|e| SochDBError::Corruption(format!("Serialization failed: {}", e)))?;
        self.db
            .insert(&key[..], value.as_slice())
            .map_err(|e| SochDBError::Internal(format!("Sled insert failed: {}", e)))?;
        Ok(())
    }

    fn get(&self, edge_id: u128) -> Result<Option<PayloadMeta>> {
        let key = edge_id.to_le_bytes();
        let value = self
            .db
            .get(&key[..])
            .map_err(|e| SochDBError::Internal(format!("Sled get failed: {}", e)))?;
        match value {
            Some(bytes) => {
                let meta: PayloadMeta = bincode::deserialize(&bytes).map_err(|e| {
                    SochDBError::Corruption(format!("Deserialization failed: {}", e))
                })?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    fn contains_key(&self, edge_id: u128) -> bool {
        let key = edge_id.to_le_bytes();
        self.db.contains_key(&key[..]).unwrap_or(false)
    }

    fn len(&self) -> usize {
        self.db.len()
    }

    fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    fn iter_values(&self) -> Box<dyn Iterator<Item = PayloadMeta> + '_> {
        Box::new(self.db.iter().filter_map(|result| {
            result
                .ok()
                .and_then(|(_, value)| bincode::deserialize::<PayloadMeta>(&value).ok())
        }))
    }

    fn save(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| SochDBError::Internal(format!("Sled flush failed: {}", e)))?;
        Ok(())
    }
}

/// Index backend type
pub enum IndexBackend {
    /// In-memory HashMap (default, fast but uses ~50MB per 1M payloads)
    HashMap,
    /// Disk-backed sled (scales to billions, ~5-10MB RAM)
    Sled,
}

/// Payload storage engine
///
/// **Architecture:**
/// - Append-only payload file (payload.data)
/// - In-memory index: edge_id -> (offset, length, compression)
/// - Memory-mapped reads for zero-copy access
/// - Write-ahead log for crash recovery
///
/// **Thread Safety:**
/// - Reads: Lock-free via memory map (multiple concurrent readers)
/// - Writes: Serialized via RwLock (single writer)
/// - Index: Protected by RwLock
///
/// **SCALABILITY:**
/// - **HashMap backend (default)**: Fast but uses ~50MB RAM per 1M payloads (32 bytes per entry + overhead).
///   Suitable for < 10M traces.
/// - **Sled backend**: Disk-backed B-Tree scales to billions with ~5-10MB RAM.
///   Recommended for 10M+ traces.
///
/// **Usage:**
/// ```rust,no_run
/// use sochdb_storage::payload::{PayloadStore, IndexBackend};
///
/// // Default: in-memory HashMap
/// let store = PayloadStore::open("./data").unwrap();
///
/// // For 10M+ scale: disk-backed sled
/// let store = PayloadStore::open_with_backend("./data", IndexBackend::Sled).unwrap();
/// ```
pub struct PayloadStore {
    data_file: Arc<RwLock<File>>,
    #[allow(dead_code)]
    data_path: PathBuf,
    mmap: Arc<RwLock<Option<memmap2::Mmap>>>,
    index: Arc<dyn PayloadIndex>,
    next_offset: Arc<RwLock<u64>>,
    /// Memory warning thresholds (for HashMap backend)
    /// Tracks when to warn users about memory growth
    memory_warning_logged: Arc<RwLock<(bool, bool, bool)>>, // (1M, 5M, 10M warnings)
    backend_type: IndexBackend,
}

impl PayloadStore {
    /// Open or create a payload store with default HashMap index
    pub fn open<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        Self::open_with_backend(data_dir, IndexBackend::HashMap)
    }

    /// Open or create a payload store with specified index backend
    pub fn open_with_backend<P: AsRef<Path>>(data_dir: P, backend: IndexBackend) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        std::fs::create_dir_all(data_dir)?;

        let data_path = data_dir.join("payload.data");
        let index_path = match backend {
            IndexBackend::HashMap => data_dir.join("payload.index"),
            IndexBackend::Sled => data_dir.join("payload_sled"),
        };

        // Open or create data file
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&data_path)?;

        let file_len = data_file.metadata()?.len();

        // Create index based on backend type
        let index: Arc<dyn PayloadIndex> = match backend {
            IndexBackend::HashMap => Arc::new(HashMapIndex::new(index_path)?),
            IndexBackend::Sled => Arc::new(SledIndex::new(index_path)?),
        };

        // Create memory map if file has data
        let mmap = if file_len > 0 {
            let mmap = unsafe { MmapOptions::new().map(&data_file)? };
            Some(mmap)
        } else {
            None
        };

        Ok(Self {
            data_file: Arc::new(RwLock::new(data_file)),
            data_path,
            mmap: Arc::new(RwLock::new(mmap)),
            index,
            next_offset: Arc::new(RwLock::new(file_len)),
            memory_warning_logged: Arc::new(RwLock::new((false, false, false))),
            backend_type: backend,
        })
    }

    /// Append a payload and return (offset, length, compression_type)
    ///
    /// **Compression Strategy:**
    /// - Small payloads (<1KB): No compression (overhead > benefit)
    /// - Medium payloads (1KB-100KB): LZ4 (fast compression/decompression)
    /// - Large payloads (>100KB): ZSTD (higher compression ratio)
    ///
    /// This can be overridden by specifying compression_type explicitly.
    ///
    /// **DESKTOP APP FIX:** Monitors memory growth and logs warnings at thresholds
    /// to prevent silent OOM in long-running sessions.
    pub fn append(
        &self,
        edge_id: u128,
        data: &[u8],
        compression: Option<CompressionType>,
    ) -> Result<(u64, u32, u8)> {
        let uncompressed_len = data.len() as u32;

        // Auto-select compression if not specified
        let compression = compression.unwrap_or(if data.len() < 1024 {
            CompressionType::None
        } else if data.len() < 100_000 {
            CompressionType::LZ4
        } else {
            CompressionType::ZSTD
        });

        // Compress data if needed
        let (compressed_data, actual_compression) = match compression {
            CompressionType::None => (data.to_vec(), CompressionType::None),
            CompressionType::LZ4 => {
                match lz4::block::compress(data, None, false) {
                    Ok(compressed) => {
                        // Only use compression if it saves space
                        if compressed.len() < data.len() {
                            (compressed, CompressionType::LZ4)
                        } else {
                            (data.to_vec(), CompressionType::None)
                        }
                    }
                    Err(_) => (data.to_vec(), CompressionType::None),
                }
            }
            CompressionType::ZSTD => {
                match zstd::encode_all(data, 3) {
                    // Level 3 is good balance
                    Ok(compressed) => {
                        if compressed.len() < data.len() {
                            (compressed, CompressionType::ZSTD)
                        } else {
                            (data.to_vec(), CompressionType::None)
                        }
                    }
                    Err(_) => (data.to_vec(), CompressionType::None),
                }
            }
        };

        let compressed_len = compressed_data.len() as u32;

        // Append to file
        let offset = {
            let mut file = self.data_file.write();
            let offset = *self.next_offset.read();

            file.seek(SeekFrom::End(0))?;
            file.write_all(&compressed_data)?;
            file.sync_all()?;

            offset
        };

        // Update next offset
        *self.next_offset.write() = offset + compressed_len as u64;

        // Update index
        self.index.insert(
            edge_id,
            PayloadMeta {
                edge_id,
                offset,
                length: compressed_len,
                compression: actual_compression,
                uncompressed_length: uncompressed_len,
            },
        )?;

        // Remap file if needed
        self.remap_file()?;

        // DESKTOP APP FIX: Monitor memory growth for HashMap backend
        // Log warnings at 1M, 5M, and 10M payloads to prevent silent OOM
        if matches!(self.backend_type, IndexBackend::HashMap) {
            let count = self.index.len();
            let mut warnings = self.memory_warning_logged.write();

            if count >= 10_000_000 && !warnings.2 {
                warnings.2 = true;
                tracing::warn!(
                    payload_count = count,
                    estimated_ram_mb = count * 50 / 1_000_000,
                    "CRITICAL: PayloadStore has 10M+ entries (~500MB RAM). \
                     Strongly recommend switching to Sled backend to prevent OOM. \
                     Use PayloadStore::open_with_backend(path, IndexBackend::Sled)"
                );
            } else if count >= 5_000_000 && !warnings.1 {
                warnings.1 = true;
                tracing::warn!(
                    payload_count = count,
                    estimated_ram_mb = count * 50 / 1_000_000,
                    "WARNING: PayloadStore has 5M+ entries (~250MB RAM). \
                     Consider switching to Sled backend for better memory efficiency. \
                     Use PayloadStore::open_with_backend(path, IndexBackend::Sled)"
                );
            } else if count >= 1_000_000 && !warnings.0 {
                warnings.0 = true;
                tracing::info!(
                    payload_count = count,
                    estimated_ram_mb = count * 50 / 1_000_000,
                    "INFO: PayloadStore has reached 1M entries (~50MB RAM). \
                     Memory usage will grow linearly. Consider Sled backend for 10M+ scale."
                );
            }
        }

        // NOTE: Index is saved on explicit sync() or shutdown, NOT on every insert.
        // This is critical for performance - the old per-insert save caused O(N²)
        // write amplification, limiting throughput to ~42 spans/sec.
        //
        // With this fix, throughput increases to 10,000+ spans/sec.
        // The caller should call save_index() or sync() when durability is required.
        //
        // For crash recovery, the data file is append-only and the index can be
        // rebuilt from the data file if needed (see rebuild_index()).

        Ok((offset, compressed_len, actual_compression as u8))
    }

    /// Get a payload by edge ID
    pub fn get(&self, edge_id: u128) -> Result<Option<Vec<u8>>> {
        let meta = match self.index.get(edge_id)? {
            Some(m) => m,
            None => return Ok(None),
        };

        self.get_at_offset(meta.offset, meta.length, meta.compression)
            .map(Some)
    }

    /// Get a payload at a specific offset
    ///
    /// This is used when reading edges from SSTables that have embedded
    /// (offset, length, compression) tuples.
    pub fn get_at_offset(
        &self,
        offset: u64,
        length: u32,
        compression: CompressionType,
    ) -> Result<Vec<u8>> {
        // For LZ4, we need the uncompressed size from the index
        let uncompressed_size = if compression == CompressionType::LZ4 {
            // Look up the metadata to get uncompressed size
            self.index
                .iter_values()
                .find(|meta| meta.offset == offset && meta.length == length)
                .map(|meta| meta.uncompressed_length as i32)
        } else {
            None
        };

        let mmap = self.mmap.read();

        let compressed_data = match mmap.as_ref() {
            Some(mmap) => {
                let start = offset as usize;
                let end = start + length as usize;

                if end > mmap.len() {
                    return Err(SochDBError::InvalidArgument(format!(
                        "Payload offset {} + length {} exceeds file size {}",
                        offset,
                        length,
                        mmap.len()
                    )));
                }

                &mmap[start..end]
            }
            None => {
                return Err(SochDBError::InvalidArgument("Payload file is empty".into()));
            }
        };

        // Decompress if needed
        let data = match compression {
            CompressionType::None => compressed_data.to_vec(),
            CompressionType::LZ4 => {
                // LZ4 requires the uncompressed size for decompression
                lz4::block::decompress(compressed_data, uncompressed_size).map_err(|e| {
                    SochDBError::Corruption(format!("LZ4 decompression failed: {}", e))
                })?
            }
            CompressionType::ZSTD => zstd::decode_all(compressed_data).map_err(|e| {
                SochDBError::Corruption(format!("ZSTD decompression failed: {}", e))
            })?,
        };

        Ok(data)
    }

    /// Check if a payload exists for an edge
    pub fn has_payload(&self, edge_id: u128) -> bool {
        self.index.contains_key(edge_id)
    }

    /// Get number of stored payloads
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Save index to disk for fast recovery (delegates to backend)
    pub fn save_index(&self) -> Result<()> {
        self.index.save()
    }

    /// Remap file after writes
    fn remap_file(&self) -> Result<()> {
        let file = self.data_file.read();
        let file_len = file.metadata()?.len();

        if file_len > 0 {
            let new_mmap = unsafe { MmapOptions::new().map(&*file)? };
            *self.mmap.write() = Some(new_mmap);
        }

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> PayloadStats {
        let mut total_compressed: u64 = 0;
        let mut total_uncompressed: u64 = 0;
        let mut compression_counts = [0usize; 3];

        for meta in self.index.iter_values() {
            total_compressed += meta.length as u64;
            total_uncompressed += meta.uncompressed_length as u64;
            compression_counts[meta.compression as usize] += 1;
        }

        PayloadStats {
            num_payloads: self.index.len(),
            total_compressed_bytes: total_compressed,
            total_uncompressed_bytes: total_uncompressed,
            compression_ratio: if total_compressed > 0 {
                total_uncompressed as f64 / total_compressed as f64
            } else {
                1.0
            },
            none_count: compression_counts[0],
            lz4_count: compression_counts[1],
            zstd_count: compression_counts[2],
        }
    }
}

impl Drop for PayloadStore {
    fn drop(&mut self) {
        // Save index on drop (delegates to backend)
        let _ = self.index.save();
    }
}

#[derive(Debug, Clone)]
pub struct PayloadStats {
    pub num_payloads: usize,
    pub total_compressed_bytes: u64,
    pub total_uncompressed_bytes: u64,
    pub compression_ratio: f64,
    pub none_count: usize,
    pub lz4_count: usize,
    pub zstd_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_payload_store_basic() {
        let dir = tempdir().unwrap();
        let store = PayloadStore::open(dir.path()).unwrap();

        let data = b"Hello, SochDB!";
        let (offset, length, compression) = store.append(1, data, None).unwrap();

        assert_eq!(offset, 0);
        assert_eq!(length, data.len() as u32);
        assert_eq!(compression, CompressionType::None as u8);

        let retrieved = store.get(1).unwrap().unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_payload_compression() {
        let dir = tempdir().unwrap();
        let store = PayloadStore::open(dir.path()).unwrap();

        // Create compressible data (repeated pattern)
        let data: Vec<u8> = b"ABCD".repeat(1000);

        let (_, length, compression) = store.append(1, &data, Some(CompressionType::LZ4)).unwrap();

        // Should be compressed
        assert!(length < data.len() as u32);
        assert_eq!(compression, CompressionType::LZ4 as u8);

        // Should decompress correctly
        let retrieved = store.get(1).unwrap().unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_payload_persistence() {
        let dir = tempdir().unwrap();

        let data1 = b"First payload";
        let data2 = b"Second payload";

        // Write payloads
        {
            let store = PayloadStore::open(dir.path()).unwrap();
            store.append(1, data1, None).unwrap();
            store.append(2, data2, None).unwrap();
        }

        // Reopen and verify
        {
            let store = PayloadStore::open(dir.path()).unwrap();
            assert_eq!(store.get(1).unwrap().unwrap(), data1);
            assert_eq!(store.get(2).unwrap().unwrap(), data2);
        }
    }

    #[test]
    fn test_payload_stats() {
        let dir = tempdir().unwrap();
        let store = PayloadStore::open(dir.path()).unwrap();

        store.append(1, b"small", None).unwrap();
        store
            .append(2, &b"A".repeat(2000), Some(CompressionType::LZ4))
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.num_payloads, 2);
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_sled_backend_basic() {
        let dir = tempdir().unwrap();
        let store = PayloadStore::open_with_backend(dir.path(), IndexBackend::Sled).unwrap();

        let data = b"Hello from Sled!";
        let (offset, length, _) = store.append(1, data, None).unwrap();

        assert_eq!(offset, 0);
        assert_eq!(length, data.len() as u32);

        let retrieved = store.get(1).unwrap().unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_sled_backend_persistence() {
        let dir = tempdir().unwrap();

        let data1 = b"First sled payload";
        let data2 = b"Second sled payload";

        // Write with sled backend
        {
            let store = PayloadStore::open_with_backend(dir.path(), IndexBackend::Sled).unwrap();
            store.append(1, data1, None).unwrap();
            store.append(2, data2, None).unwrap();
            // Explicit save (though Drop also saves)
            store.save_index().unwrap();
        }

        // Reopen with sled and verify
        {
            let store = PayloadStore::open_with_backend(dir.path(), IndexBackend::Sled).unwrap();
            assert_eq!(store.get(1).unwrap().unwrap(), data1);
            assert_eq!(store.get(2).unwrap().unwrap(), data2);
            assert_eq!(store.len(), 2);
        }
    }

    #[test]
    fn test_sled_backend_large_dataset() {
        let dir = tempdir().unwrap();
        let store = PayloadStore::open_with_backend(dir.path(), IndexBackend::Sled).unwrap();

        // Simulate large dataset
        for i in 0..1000 {
            let data = format!("Payload {}", i);
            store.append(i, data.as_bytes(), None).unwrap();
        }

        // Verify random access
        assert_eq!(store.len(), 1000);
        let retrieved = store.get(500).unwrap().unwrap();
        assert_eq!(retrieved, b"Payload 500");

        // Verify stats
        let stats = store.stats();
        assert_eq!(stats.num_payloads, 1000);
    }

    #[test]
    fn test_backend_interoperability() {
        let dir = tempdir().unwrap();

        // Write with HashMap backend
        {
            let store = PayloadStore::open_with_backend(dir.path(), IndexBackend::HashMap).unwrap();
            store.append(1, b"HashMap data", None).unwrap();
        }

        // Payload data file is shared, but indexes are separate
        // Sled backend starts fresh (no cross-backend index migration)
        {
            let store = PayloadStore::open_with_backend(dir.path(), IndexBackend::Sled).unwrap();
            // Sled has its own index, so it won't see HashMap's entry
            assert_eq!(store.len(), 0);

            // Add new entry with sled
            store.append(2, b"Sled data", None).unwrap();
            assert_eq!(store.len(), 1);
        }
    }
}
