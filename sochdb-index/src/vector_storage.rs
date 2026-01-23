// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use memmap2::Mmap;
use ndarray::Array1;
use parking_lot::RwLock;
use std::any::Any;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Trait for abstracting vector storage
pub trait VectorStorage: Send + Sync {
    /// Downcast support
    fn as_any(&self) -> &dyn Any;

    /// Append a vector to storage and return its ID
    fn append(&self, vector: &Array1<f32>) -> io::Result<u64>;

    /// Get a vector by ID
    fn get(&self, id: u64) -> io::Result<Array1<f32>>;

    /// Get a vector into a provided buffer (avoids allocation)
    fn get_into(&self, id: u64, out: &mut [f32]) -> io::Result<()>;

    /// Get vector dimension
    fn dim(&self) -> usize;

    /// Get number of vectors stored
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Memory-mapped file storage for vectors
///
/// Layout:
/// - Header: [u32: magic, u32: version, u32: dim, u64: count] (handled by wrapper or separate metadata)
/// - Data: [f32; dim] * count
///
/// For simplicity in this iteration, we assume the file only contains raw vector data (f32s).
/// Metadata (dim, count) is managed by the HNSW index or a separate header file.
pub struct MmapVectorStorage {
    file: RwLock<File>,
    mmap: RwLock<Option<Mmap>>,
    dim: usize,
    count: RwLock<usize>,
    #[allow(dead_code)]
    path: std::path::PathBuf,
}

impl MmapVectorStorage {
    pub fn new<P: AsRef<Path>>(path: P, dim: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let len = file.metadata()?.len();
        let count = if dim > 0 {
            (len as usize) / (dim * 4)
        } else {
            0
        };

        let mmap = if len > 0 {
            unsafe { Some(Mmap::map(&file)?) }
        } else {
            None
        };

        Ok(Self {
            file: RwLock::new(file),
            mmap: RwLock::new(mmap),
            dim,
            count: RwLock::new(count),
            path: path.as_ref().to_path_buf(),
        })
    }

    fn ensure_mmap(&self) -> io::Result<()> {
        // If mmap is missing or stale (file grew), remap
        // Note: This is a simple implementation. In production, we might want to remap only when necessary
        // or use a paged approach to avoid remapping the whole file constantly.
        // For append-only, we can just remap when we need to read newly added data.
        // However, standard mmap behavior on some OSs might not reflect file growth immediately without remap.

        // Check if file size matches mmap size
        let file_len = self.file.read().metadata()?.len();
        let mmap_len = self.mmap.read().as_ref().map(|m| m.len()).unwrap_or(0) as u64;

        if file_len != mmap_len {
            let mut mmap_guard = self.mmap.write();
            if file_len > 0 {
                let file = self.file.read();
                *mmap_guard = unsafe { Some(Mmap::map(&*file)?) };
            }
        }
        Ok(())
    }
}

impl VectorStorage for MmapVectorStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn append(&self, vector: &Array1<f32>) -> io::Result<u64> {
        if vector.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dim,
                    vector.len()
                ),
            ));
        }

        let mut file = self.file.write();

        // Write raw bytes
        for &val in vector.iter() {
            file.write_all(&val.to_le_bytes())?;
        }
        file.flush()?;

        let mut count = self.count.write();
        let id = *count as u64;
        *count += 1;

        // Invalidate mmap so next read remaps
        // Optimization: Don't remap immediately, only on read if needed.
        // But for simplicity, we'll let ensure_mmap handle it on read.

        Ok(id)
    }

    fn get(&self, id: u64) -> io::Result<Array1<f32>> {
        self.ensure_mmap()?;

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard
            .as_ref()
            .ok_or_else(|| io::Error::other("Mmap not initialized"))?;

        let idx = id as usize;
        let start = idx * self.dim * 4;
        let end = start + self.dim * 4;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector ID {} out of bounds", id),
            ));
        }

        let bytes = &mmap[start..end];
        let mut vector_data = Vec::with_capacity(self.dim);

        for chunk in bytes.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            vector_data.push(val);
        }

        Ok(Array1::from_vec(vector_data))
    }

    fn get_into(&self, id: u64, out: &mut [f32]) -> io::Result<()> {
        self.ensure_mmap()?;

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard
            .as_ref()
            .ok_or_else(|| io::Error::other("Mmap not initialized"))?;

        let idx = id as usize;
        let start = idx * self.dim * 4;
        let end = start + self.dim * 4;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector ID {} out of bounds", id),
            ));
        }

        let bytes = &mmap[start..end];
        if out.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.dim,
                    out.len()
                ),
            ));
        }

        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        *self.count.read()
    }
}

/// Region-clustered mmap storage with ID remapping
pub struct RegionMmapVectorStorage {
    file: RwLock<File>,
    mmap: RwLock<Option<Mmap>>,
    dim: usize,
    count: RwLock<usize>,
    path: std::path::PathBuf,
    id_to_offset: RwLock<Vec<u64>>,
}

impl RegionMmapVectorStorage {
    pub fn new<P: AsRef<Path>>(path: P, dim: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let len = file.metadata()?.len();
        let count = if dim > 0 {
            (len as usize) / (dim * 4)
        } else {
            0
        };

        let mmap = if len > 0 {
            unsafe { Some(Mmap::map(&file)?) }
        } else {
            None
        };

        let mut id_to_offset = Vec::with_capacity(count);
        let map_path = Self::mapping_path(path.as_ref());
        if map_path.exists() {
            let bytes = std::fs::read(&map_path)?;
            for chunk in bytes.chunks_exact(8) {
                id_to_offset.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }

        if id_to_offset.len() != count {
            id_to_offset = (0..count as u64).collect();
        }

        Ok(Self {
            file: RwLock::new(file),
            mmap: RwLock::new(mmap),
            dim,
            count: RwLock::new(count),
            path: path.as_ref().to_path_buf(),
            id_to_offset: RwLock::new(id_to_offset),
        })
    }

    fn mapping_path(path: &Path) -> std::path::PathBuf {
        path.with_extension("region.map")
    }

    fn ensure_mmap(&self) -> io::Result<()> {
        let file_len = self.file.read().metadata()?.len();
        let mmap_len = self.mmap.read().as_ref().map(|m| m.len()).unwrap_or(0) as u64;

        if file_len != mmap_len {
            let mut mmap_guard = self.mmap.write();
            if file_len > 0 {
                let file = self.file.read();
                *mmap_guard = unsafe { Some(Mmap::map(&*file)?) };
            }
        }
        Ok(())
    }

    fn save_mapping(&self) -> io::Result<()> {
        let map_path = Self::mapping_path(&self.path);
        let mapping = self.id_to_offset.read();
        let mut bytes = Vec::with_capacity(mapping.len() * 8);
        for &val in mapping.iter() {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        std::fs::write(map_path, bytes)
    }

    /// Reorder storage into region layout using an ordered list of IDs
    pub fn reorder_by_regions(&self, order: &[u64], _region_size: usize) -> io::Result<()> {
        let count = *self.count.read();
        if count == 0 {
            return Ok(());
        }

        self.ensure_mmap()?;

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard
            .as_ref()
            .ok_or_else(|| io::Error::other("Mmap not initialized"))?;

        let mut seen = vec![false; count];
        let mut ordered_ids = Vec::with_capacity(count);
        for &id in order {
            let idx = id as usize;
            if idx >= count || seen[idx] {
                continue;
            }
            seen[idx] = true;
            ordered_ids.push(id);
        }
        for id in 0..count as u64 {
            if !seen[id as usize] {
                ordered_ids.push(id);
            }
        }

        let id_to_offset = self.id_to_offset.read();
        let tmp_path = self.path.with_extension("region.tmp");
        let mut tmp_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;

        for (new_index, &id) in ordered_ids.iter().enumerate() {
            let offset_id = *id_to_offset.get(id as usize).unwrap_or(&id);
            let start = offset_id as usize * self.dim * 4;
            let end = start + self.dim * 4;
            if end > mmap.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Vector ID {} out of bounds", id),
                ));
            }
            tmp_file.write_all(&mmap[start..end])?;
            if new_index % 1024 == 0 {
                tmp_file.flush()?;
            }
        }
        tmp_file.flush()?;

        drop(mmap_guard);

        std::fs::rename(&tmp_path, &self.path)?;

        let mut mapping = self.id_to_offset.write();
        mapping.clear();
        mapping.resize(count, 0);
        for (new_index, &id) in ordered_ids.iter().enumerate() {
            mapping[id as usize] = new_index as u64;
        }
        drop(mapping);
        self.save_mapping()?;

        let mut file_guard = self.file.write();
        *file_guard = OpenOptions::new().read(true).write(true).open(&self.path)?;
        let mut mmap_guard = self.mmap.write();
        *mmap_guard = unsafe { Some(Mmap::map(&*file_guard)?) };

        Ok(())
    }
}

impl VectorStorage for RegionMmapVectorStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn append(&self, vector: &Array1<f32>) -> io::Result<u64> {
        if vector.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dim,
                    vector.len()
                ),
            ));
        }

        let mut file = self.file.write();

        for &val in vector.iter() {
            file.write_all(&val.to_le_bytes())?;
        }
        file.flush()?;

        let mut count = self.count.write();
        let id = *count as u64;
        *count += 1;

        let mut mapping = self.id_to_offset.write();
        mapping.push(id);
        drop(mapping);
        self.save_mapping()?;

        Ok(id)
    }

    fn get(&self, id: u64) -> io::Result<Array1<f32>> {
        self.ensure_mmap()?;

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard
            .as_ref()
            .ok_or_else(|| io::Error::other("Mmap not initialized"))?;

        let mapping = self.id_to_offset.read();
        let offset_id = *mapping
            .get(id as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ID out of bounds"))?;

        let start = offset_id as usize * self.dim * 4;
        let end = start + self.dim * 4;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector ID {} out of bounds", id),
            ));
        }

        let bytes = &mmap[start..end];
        let mut vector_data = Vec::with_capacity(self.dim);
        for chunk in bytes.chunks_exact(4) {
            vector_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        Ok(Array1::from_vec(vector_data))
    }

    fn get_into(&self, id: u64, out: &mut [f32]) -> io::Result<()> {
        self.ensure_mmap()?;

        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard
            .as_ref()
            .ok_or_else(|| io::Error::other("Mmap not initialized"))?;

        if out.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.dim,
                    out.len()
                ),
            ));
        }

        let mapping = self.id_to_offset.read();
        let offset_id = *mapping
            .get(id as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ID out of bounds"))?;

        let start = offset_id as usize * self.dim * 4;
        let end = start + self.dim * 4;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector ID {} out of bounds", id),
            ));
        }

        let bytes = &mmap[start..end];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        *self.count.read()
    }
}

/// In-memory vector storage for tests and small indices
pub struct MemoryVectorStorage {
    vectors: RwLock<Vec<Array1<f32>>>,
    dim: AtomicUsize,
}

impl MemoryVectorStorage {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: RwLock::new(Vec::new()),
            dim: AtomicUsize::new(dim),
        }
    }
}

impl VectorStorage for MemoryVectorStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn append(&self, vector: &Array1<f32>) -> io::Result<u64> {
        let dim = self.dim.load(Ordering::Relaxed);
        if dim > 0 && vector.len() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    dim,
                    vector.len()
                ),
            ));
        }

        if dim == 0 {
            self.dim.store(vector.len(), Ordering::Relaxed);
        }

        let mut vectors = self.vectors.write();
        let id = vectors.len() as u64;
        vectors.push(vector.clone());
        Ok(id)
    }

    fn get(&self, id: u64) -> io::Result<Array1<f32>> {
        let vectors = self.vectors.read();
        if (id as usize) < vectors.len() {
            Ok(vectors[id as usize].clone())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector ID {} out of bounds", id),
            ))
        }
    }

    fn get_into(&self, id: u64, out: &mut [f32]) -> io::Result<()> {
        let vectors = self.vectors.read();
        if (id as usize) < vectors.len() {
            let vec = &vectors[id as usize];
            if out.len() != vec.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Buffer size mismatch: expected {}, got {}",
                        vec.len(),
                        out.len()
                    ),
                ));
            }
            out.copy_from_slice(vec.as_slice().unwrap());
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector ID {} out of bounds", id),
            ))
        }
    }

    fn dim(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    fn len(&self) -> usize {
        self.vectors.read().len()
    }
}
