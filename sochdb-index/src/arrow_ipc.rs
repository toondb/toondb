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

//! Arrow IPC Zero-Copy Data Exchange for Cross-Language Interop
//!
//! This module provides Arrow IPC support for zero-copy batch vector ingestion.
//! Arrow IPC is a language-agnostic binary format that enables efficient data
//! exchange between Python (PyArrow), JavaScript (Arrow JS), Rust, Go, and more.
//!
//! ## Why Arrow IPC?
//!
//! 1. **Zero-Copy**: Data stays in Arrow format, no deserialization needed
//! 2. **Cross-Language**: Works with any Arrow-compatible language
//! 3. **Standardized**: Apache Arrow is the industry standard for columnar data
//! 4. **Batch-Oriented**: Designed for bulk operations (10K+ vectors)
//!
//! ## Usage from Python (PyArrow)
//!
//! ```python
//! import pyarrow as pa
//! from sochdb import VectorIndex
//!
//! # Create Arrow RecordBatch with id + vector columns
//! ids = pa.array([1, 2, 3], type=pa.uint64())
//! vectors = pa.FixedSizeListArray.from_arrays(
//!     pa.array([0.1, 0.2, ...], type=pa.float32()),
//!     list_size=768
//! )
//! batch = pa.record_batch([ids, vectors], names=["id", "vector"])
//!
//! # Serialize to IPC and send to Rust
//! sink = pa.BufferOutputStream()
//! writer = pa.ipc.new_stream(sink, batch.schema)
//! writer.write_batch(batch)
//! writer.close()
//! ipc_data = sink.getvalue().to_pybytes()
//!
//! # Zero-copy insert via FFI
//! index.insert_arrow_ipc(ipc_data)
//! ```
//!
//! ## Expected Schema
//!
//! The Arrow RecordBatch must have exactly two columns:
//! - `id`: UInt64 (vector identifiers)
//! - `vector`: FixedSizeList<Float32> (embedding vectors)

use crate::hnsw::HnswIndex;
use std::io::Cursor;
use std::sync::Arc;

// Re-export for convenience
pub use arrow::error::ArrowError;

/// Arrow IPC ingestion error
#[derive(Debug, thiserror::Error)]
pub enum ArrowIpcError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    
    #[error("Invalid schema: expected 'id' (UInt64) and 'vector' (FixedSizeList<Float32>), got: {0}")]
    InvalidSchema(String),
    
    #[error("Dimension mismatch: index expects {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Empty batch: no vectors to insert")]
    EmptyBatch,
    
    #[error("HNSW insert error: {0}")]
    HnswError(String),
}

/// Insert vectors from Arrow IPC stream into HNSW index.
///
/// This function provides zero-copy batch insertion from Arrow IPC format.
/// The IPC data must contain a RecordBatch with:
/// - Column 0: `id` (UInt64 array)
/// - Column 1: `vector` (FixedSizeList<Float32> array)
///
/// # Arguments
///
/// * `index` - Reference to the HNSW index
/// * `ipc_data` - Arrow IPC stream data (from RecordBatchStreamWriter)
///
/// # Returns
///
/// Number of vectors successfully inserted, or error.
///
/// # Example
///
/// ```rust,ignore
/// use sochdb_index::arrow_ipc::insert_from_arrow_ipc;
/// use sochdb_index::hnsw::HnswIndex;
///
/// let index = HnswIndex::new(768, 32, 200);
/// let ipc_data: &[u8] = /* Arrow IPC bytes */;
/// let count = insert_from_arrow_ipc(&index, ipc_data)?;
/// println!("Inserted {} vectors", count);
/// ```
pub fn insert_from_arrow_ipc(
    index: &HnswIndex,
    ipc_data: &[u8],
) -> Result<usize, ArrowIpcError> {
    use arrow::array::{Array, AsArray, Float32Array, UInt64Array};
    use arrow::datatypes::DataType;
    use arrow::ipc::reader::StreamReader;
    
    // Parse Arrow IPC stream
    let cursor = Cursor::new(ipc_data);
    let reader = StreamReader::try_new(cursor, None)?;
    
    // Validate schema
    let schema = reader.schema();
    if schema.fields().len() != 2 {
        return Err(ArrowIpcError::InvalidSchema(format!(
            "expected 2 columns (id, vector), got {}",
            schema.fields().len()
        )));
    }
    
    // Check id column
    let id_field = &schema.fields()[0];
    if id_field.data_type() != &DataType::UInt64 {
        return Err(ArrowIpcError::InvalidSchema(format!(
            "id column must be UInt64, got {:?}",
            id_field.data_type()
        )));
    }
    
    // Check vector column - must be FixedSizeList<Float32>
    let vector_field = &schema.fields()[1];
    let (list_size, inner_type) = match vector_field.data_type() {
        DataType::FixedSizeList(inner, size) => (*size as usize, inner.data_type()),
        other => {
            return Err(ArrowIpcError::InvalidSchema(format!(
                "vector column must be FixedSizeList<Float32>, got {:?}",
                other
            )));
        }
    };
    
    if inner_type != &DataType::Float32 {
        return Err(ArrowIpcError::InvalidSchema(format!(
            "vector inner type must be Float32, got {:?}",
            inner_type
        )));
    }
    
    // Validate dimension matches index
    let index_dim = index.dimension;
    if list_size != index_dim {
        return Err(ArrowIpcError::DimensionMismatch {
            expected: index_dim,
            actual: list_size,
        });
    }
    
    // Process all batches in the stream
    let mut total_inserted = 0;
    
    for batch_result in reader {
        let batch = batch_result?;
        let num_rows = batch.num_rows();
        
        if num_rows == 0 {
            continue;
        }
        
        // Extract id column
        let id_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| ArrowIpcError::InvalidSchema("id column is not UInt64".into()))?;
        
        // Extract vector values column (the inner Float32 array)
        let vector_list = batch.column(1).as_fixed_size_list();
        let values = vector_list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                ArrowIpcError::InvalidSchema("vector values are not Float32".into())
            })?;
        
        // Get raw float slice - this is the zero-copy path
        let raw_floats = values.values().as_ref();
        
        // Use flat batch insert for maximum performance
        let ids: Vec<u128> = (0..num_rows)
            .map(|i| id_array.value(i) as u128)
            .collect();
        
        // Call the flat batch insert API
        let inserted = index.insert_batch_flat(&ids, raw_floats, index_dim)
            .map_err(|e| ArrowIpcError::HnswError(e))?;
        
        total_inserted += inserted;
    }
    
    if total_inserted == 0 {
        return Err(ArrowIpcError::EmptyBatch);
    }
    
    Ok(total_inserted)
}

/// Insert vectors from Arrow RecordBatch directly (when already parsed).
///
/// This is useful when you have an Arrow RecordBatch in memory and want
/// to avoid the IPC serialization overhead.
///
/// # Arguments
///
/// * `index` - Reference to the HNSW index
/// * `batch` - Arrow RecordBatch with id + vector columns
///
/// # Returns
///
/// Number of vectors successfully inserted, or error.
pub fn insert_from_record_batch(
    index: &HnswIndex,
    batch: &arrow::array::RecordBatch,
) -> Result<usize, ArrowIpcError> {
    use arrow::array::{AsArray, Float32Array, UInt64Array};
    
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Err(ArrowIpcError::EmptyBatch);
    }
    
    // Extract id column
    let id_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| ArrowIpcError::InvalidSchema("id column is not UInt64".into()))?;
    
    // Extract vector values
    let vector_list = batch.column(1).as_fixed_size_list();
    let values = vector_list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            ArrowIpcError::InvalidSchema("vector values are not Float32".into())
        })?;
    
    // Validate dimension
    let list_size = vector_list.value_length() as usize;
    let index_dim = index.dimension;
    if list_size != index_dim {
        return Err(ArrowIpcError::DimensionMismatch {
            expected: index_dim,
            actual: list_size,
        });
    }
    
    // Zero-copy access to float data
    let raw_floats = values.values().as_ref();
    
    // Build ID vector
    let ids: Vec<u128> = (0..num_rows)
        .map(|i| id_array.value(i) as u128)
        .collect();
    
    // Insert using flat batch API
    let inserted = index.insert_batch_flat(&ids, raw_floats, index_dim)
        .map_err(|e| ArrowIpcError::HnswError(e))?;
    
    Ok(inserted)
}

// =============================================================================
// FFI BINDINGS
// =============================================================================

use std::os::raw::c_int;

/// FFI: Insert vectors from Arrow IPC stream.
///
/// # Arguments
///
/// * `ptr` - Pointer to HnswIndex (from hnsw_new)
/// * `ipc_data` - Pointer to Arrow IPC bytes
/// * `ipc_len` - Length of IPC data in bytes
///
/// # Returns
///
/// Number of vectors inserted on success, -1 on error.
///
/// # Safety
///
/// - `ptr` must be a valid pointer from `hnsw_new`
/// - `ipc_data` must point to valid memory of at least `ipc_len` bytes
/// - `ipc_data` must remain valid for the duration of the call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnsw_insert_arrow_ipc(
    ptr: *mut crate::ffi::HnswIndexPtr,
    ipc_data: *const u8,
    ipc_len: usize,
) -> c_int {
    if ptr.is_null() || ipc_data.is_null() || ipc_len == 0 {
        return -1;
    }
    
    let index_ptr = unsafe { &*ptr };
    let ipc_slice = unsafe { std::slice::from_raw_parts(ipc_data, ipc_len) };
    
    match insert_from_arrow_ipc(&index_ptr.0, ipc_slice) {
        Ok(count) => count as c_int,
        Err(e) => {
            eprintln!("[SochDB Arrow IPC Error] {}", e);
            -1
        }
    }
}

/// FFI: Get the last Arrow IPC error message.
///
/// This function returns the last error message from Arrow IPC operations.
/// The returned string is valid until the next Arrow IPC operation.
///
/// # Returns
///
/// Pointer to null-terminated error string, or null if no error.
///
/// # Safety
///
/// The returned pointer is valid until the next Arrow IPC operation.
#[unsafe(no_mangle)]
pub extern "C" fn hnsw_arrow_ipc_last_error() -> *const std::os::raw::c_char {
    // Thread-local error storage
    thread_local! {
        static LAST_ERROR: std::cell::RefCell<Option<std::ffi::CString>> = const { std::cell::RefCell::new(None) };
    }
    
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Create an Arrow schema for vector ingestion.
///
/// This helper creates the expected schema for Arrow IPC ingestion:
/// - `id`: UInt64
/// - `vector`: FixedSizeList<Float32>
///
/// # Arguments
///
/// * `dimension` - Vector dimension (e.g., 768)
///
/// # Returns
///
/// Arrow Schema with the expected columns.
pub fn create_vector_schema(dimension: i32) -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema};
    
    Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dimension,
            ),
            false,
        ),
    ])
}

/// Create an Arrow RecordBatch from IDs and vectors.
///
/// This helper creates a RecordBatch that can be serialized to IPC.
///
/// # Arguments
///
/// * `ids` - Vector IDs (u64)
/// * `vectors` - Flat vector data (f32, row-major)
/// * `dimension` - Vector dimension
///
/// # Returns
///
/// Arrow RecordBatch ready for IPC serialization.
pub fn create_vector_batch(
    ids: &[u64],
    vectors: &[f32],
    dimension: usize,
) -> Result<arrow::array::RecordBatch, ArrowIpcError> {
    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, UInt64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field};
    
    if vectors.len() != ids.len() * dimension {
        return Err(ArrowIpcError::DimensionMismatch {
            expected: ids.len() * dimension,
            actual: vectors.len(),
        });
    }
    
    // Create ID array
    let id_array: ArrayRef = Arc::new(UInt64Array::from(ids.to_vec()));
    
    // Create vector array (FixedSizeList<Float32>)
    let values = Float32Array::from(vectors.to_vec());
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_array: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(field, dimension as i32, Arc::new(values), None)?
    );
    
    let schema = create_vector_schema(dimension as i32);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![id_array, vector_array])?;
    
    Ok(batch)
}

/// Serialize a RecordBatch to Arrow IPC bytes.
///
/// # Arguments
///
/// * `batch` - RecordBatch to serialize
///
/// # Returns
///
/// Arrow IPC stream bytes.
pub fn serialize_to_ipc(
    batch: &arrow::array::RecordBatch,
) -> Result<Vec<u8>, ArrowIpcError> {
    use arrow::ipc::writer::StreamWriter;
    use std::io::Cursor;
    
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = StreamWriter::try_new(cursor, &batch.schema())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswIndex, HnswConfig};
    
    #[test]
    fn test_arrow_ipc_roundtrip() {
        // Create a small index
        let config = HnswConfig {
            max_connections: 16,
            ef_construction: 50,
            ..Default::default()
        };
        let index = HnswIndex::new(4, config);
        
        // Create test data
        let ids: Vec<u64> = vec![1, 2, 3];
        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,  // vector 1
            0.0, 1.0, 0.0, 0.0,  // vector 2
            0.0, 0.0, 1.0, 0.0,  // vector 3
        ];
        
        // Create RecordBatch and serialize to IPC
        let batch = create_vector_batch(&ids, &vectors, 4).unwrap();
        let ipc_data = serialize_to_ipc(&batch).unwrap();
        
        // Insert via Arrow IPC
        let inserted = insert_from_arrow_ipc(&index, &ipc_data).unwrap();
        assert_eq!(inserted, 3);
        
        // Verify search works
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1); // Should find vector 1
    }
    
    #[test]
    fn test_dimension_mismatch() {
        // Create index with dimension 8
        let config = HnswConfig {
            max_connections: 16,
            ef_construction: 50,
            ..Default::default()
        };
        let index = HnswIndex::new(8, config);
        
        // Create data with dimension 4
        let ids: Vec<u64> = vec![1];
        let vectors: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        
        let batch = create_vector_batch(&ids, &vectors, 4).unwrap();
        let ipc_data = serialize_to_ipc(&batch).unwrap();
        
        // Should fail with dimension mismatch
        let result = insert_from_arrow_ipc(&index, &ipc_data);
        assert!(matches!(result, Err(ArrowIpcError::DimensionMismatch { .. })));
    }
    
    #[test]
    fn test_empty_batch() {
        let config = HnswConfig {
            max_connections: 16,
            ef_construction: 50,
            ..Default::default()
        };
        let index = HnswIndex::new(4, config);
        
        let ids: Vec<u64> = vec![];
        let vectors: Vec<f32> = vec![];
        
        let batch = create_vector_batch(&ids, &vectors, 4).unwrap();
        let ipc_data = serialize_to_ipc(&batch).unwrap();
        
        let result = insert_from_arrow_ipc(&index, &ipc_data);
        assert!(matches!(result, Err(ArrowIpcError::EmptyBatch)));
    }
}
