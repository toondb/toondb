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

use crate::database::{Database, TxnHandle};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::slice;
use std::sync::Arc;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};

/// Opaque pointer to Database
pub struct DatabasePtr(Arc<Database>);

// =========================================================================
// Collection Index Registry (in-memory)
// =========================================================================

struct CollectionIndex {
    index: Arc<HnswIndex>,
    dimension: usize,
    metric: DistanceMetric,
}

static COLLECTION_INDEXES: OnceLock<Mutex<HashMap<String, Arc<CollectionIndex>>>> = OnceLock::new();

fn collection_key(namespace: &str, collection: &str) -> String {
    format!("{}/{}", namespace, collection)
}

fn vector_bin_key(namespace: &str, collection: &str, id_hash: u128) -> String {
    format!("{}/collections/{}/vectors_bin/{:032x}", namespace, collection, id_hash)
}

fn metadata_key(namespace: &str, collection: &str, id_hash: u128) -> String {
    format!("{}/collections/{}/meta/{:032x}", namespace, collection, id_hash)
}

fn hash_id_to_u128(id: &str) -> u128 {
    let hash = blake3::hash(id.as_bytes());
    let bytes = hash.as_bytes();
    u128::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ])
}

fn ensure_collection_index(
    db: &Database,
    namespace: &str,
    collection: &str,
    dimension: usize,
    metric: DistanceMetric,
) -> Arc<CollectionIndex> {
    let registry = COLLECTION_INDEXES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = collection_key(namespace, collection);

    let mut registry_guard = registry.lock().unwrap();
    if let Some(existing) = registry_guard.get(&key) {
        return existing.clone();
    }

    let mut config = HnswConfig::default();
    config.metric = metric;
    let index = Arc::new(HnswIndex::new(dimension, config));

    let entry = Arc::new(CollectionIndex {
        index,
        dimension,
        metric,
    });
    registry_guard.insert(key, entry.clone());

    entry
}

fn resolve_collection_config(
    db: &Database,
    namespace: &str,
    collection: &str,
) -> Option<(usize, DistanceMetric)> {
    let key = format!("{}/_collections/{}", namespace, collection);
    let txn = db.begin_transaction().ok()?;
    let value = db.get(txn, key.as_bytes()).ok().flatten();
    let _ = db.commit(txn);
    let value = value?;

    let parsed: serde_json::Value = serde_json::from_slice(&value).ok()?;
    let dimension = parsed.get("dimension")?.as_u64()? as usize;
    let metric = match parsed.get("metric").and_then(|v| v.as_u64()).unwrap_or(0) {
        1 => DistanceMetric::Euclidean,
        2 => DistanceMetric::DotProduct,
        _ => DistanceMetric::Cosine,
    };
    Some((dimension, metric))
}

fn serialize_vector_binary(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + vector.len() * 4);
    let len = vector.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    for value in vector {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decode_score(metric: DistanceMetric, distance: f32) -> f32 {
    match metric {
        DistanceMetric::Cosine => 1.0 - distance,
        DistanceMetric::DotProduct => -distance,
        DistanceMetric::Euclidean => -distance,
    }
}

/// C-compatible Transaction Handle
#[repr(C)]
pub struct C_TxnHandle {
    pub txn_id: u64,
    pub snapshot_ts: u64,
}

/// C-compatible Commit Result
/// Returns commit_ts on success, or 0 with error_code on failure
#[repr(C)]
pub struct C_CommitResult {
    /// Commit timestamp (HLC-backed, monotonically increasing)
    /// This is 0 if the commit failed.
    pub commit_ts: u64,
    /// Error code: 0 = success, -1 = error, -2 = SSI conflict
    pub error_code: i32,
}

/// C-compatible Database Configuration
/// 
/// All fields have sensible defaults when set to 0/false.
/// This allows clients to only set the fields they care about.
#[repr(C)]
pub struct C_DatabaseConfig {
    /// Enable WAL for durability (default: true if wal_enabled_set is false)
    pub wal_enabled: bool,
    /// Whether wal_enabled was explicitly set
    pub wal_enabled_set: bool,
    /// Sync mode: 0=OFF, 1=NORMAL (default), 2=FULL
    pub sync_mode: u8,
    /// Whether sync_mode was explicitly set
    pub sync_mode_set: bool,
    /// Memtable size in bytes (0 = default 64MB)
    pub memtable_size_bytes: u64,
    /// Enable group commit for throughput (default: true if group_commit_set is false)
    pub group_commit: bool,
    /// Whether group_commit was explicitly set  
    pub group_commit_set: bool,
    /// Default index policy: 0=WriteOptimized, 1=Balanced (default), 2=ScanOptimized, 3=AppendOnly
    pub default_index_policy: u8,
    /// Whether default_index_policy was explicitly set
    pub default_index_policy_set: bool,
}

/// Open the database with configuration.
/// Returns a pointer to the database instance, or null on error.
/// # Safety
/// The path must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_open_with_config(
    path: *const c_char, 
    config: C_DatabaseConfig
) -> *mut DatabasePtr {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    // Build config from C struct, using defaults for unset fields
    let mut db_config = crate::database::DatabaseConfig::default();
    
    if config.wal_enabled_set {
        db_config.wal_enabled = config.wal_enabled;
    }
    
    if config.sync_mode_set {
        db_config.sync_mode = match config.sync_mode {
            0 => crate::database::SyncMode::Off,
            1 => crate::database::SyncMode::Normal,
            _ => crate::database::SyncMode::Full,
        };
    }
    
    if config.memtable_size_bytes > 0 {
        db_config.memtable_size_limit = config.memtable_size_bytes as usize;
    }
    
    if config.group_commit_set {
        db_config.group_commit = config.group_commit;
    }
    
    if config.default_index_policy_set {
        db_config.default_index_policy = match config.default_index_policy {
            0 => crate::index_policy::IndexPolicy::WriteOptimized,
            1 => crate::index_policy::IndexPolicy::Balanced,
            2 => crate::index_policy::IndexPolicy::ScanOptimized,
            _ => crate::index_policy::IndexPolicy::AppendOnly,
        };
    }

    match Database::open_with_config(path_str, db_config) {
        Ok(db) => {
            let ptr = Box::new(DatabasePtr(db));
            Box::into_raw(ptr)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Open the database.
/// Returns a pointer to the database instance, or null on error.
/// # Safety
/// The path must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_open(path: *const c_char) -> *mut DatabasePtr {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    // Use default config for now
    let config = crate::database::DatabaseConfig::default();

    // Database::open returns Result<Arc<Database>>
    match Database::open_with_config(path_str, config) {
        Ok(db) => {
            let ptr = Box::new(DatabasePtr(db));
            Box::into_raw(ptr)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Close the database and free the pointer.
/// # Safety
/// The ptr must be a valid pointer returned by sochdb_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_close(ptr: *mut DatabasePtr) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Begin a transaction.
/// Returns C_TxnHandle. On error, txn_id will be 0.
/// # Safety
/// The ptr must be a valid pointer returned by sochdb_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_begin_txn(ptr: *mut DatabasePtr) -> C_TxnHandle {
    if ptr.is_null() {
        return C_TxnHandle {
            txn_id: 0,
            snapshot_ts: 0,
        };
    }
    let db = unsafe { &(*ptr).0 };
    match db.begin_transaction() {
        Ok(txn) => C_TxnHandle {
            txn_id: txn.txn_id,
            snapshot_ts: txn.snapshot_ts,
        },
        Err(_) => C_TxnHandle {
            txn_id: 0,
            snapshot_ts: 0,
        },
    }
}

/// Commit a transaction.
/// Returns C_CommitResult with commit_ts on success.
/// The commit_ts is HLC-backed and monotonically increasing, suitable for 
/// MVCC observability, replication, and audit trails.
/// # Safety
/// The ptr must be a valid pointer returned by sochdb_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_commit(ptr: *mut DatabasePtr, handle: C_TxnHandle) -> C_CommitResult {
    if ptr.is_null() {
        return C_CommitResult {
            commit_ts: 0,
            error_code: -1,
        };
    }
    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };
    match db.commit(txn) {
        Ok(commit_ts) => C_CommitResult {
            commit_ts,
            error_code: 0,
        },
        Err(_) => C_CommitResult {
            commit_ts: 0,
            error_code: -1,
        },
    }
}

/// Abort a transaction.
/// Returns 0 on success, -1 on error.
/// # Safety
/// The ptr must be a valid pointer returned by sochdb_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_abort(ptr: *mut DatabasePtr, handle: C_TxnHandle) -> c_int {
    if ptr.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };
    match db.abort(txn) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Put a key-value pair.
/// Returns 0 on success, -1 on error.
/// # Safety
/// The ptr must be a valid pointer returned by sochdb_open.
/// key_ptr and val_ptr must be valid pointers with the specified lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_put(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    key_ptr: *const u8,
    key_len: usize,
    val_ptr: *const u8,
    val_len: usize,
) -> c_int {
    if ptr.is_null() || key_ptr.is_null() || val_ptr.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    let key = unsafe { slice::from_raw_parts(key_ptr, key_len) };
    let val = unsafe { slice::from_raw_parts(val_ptr, val_len) };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    match db.put(txn, key, val) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get a value.
/// Writes pointer to val_out and length to len_out.
/// The caller must free the returned bytes using sochdb_free_bytes.
/// Returns 0 on success (found), 1 on not found, -1 on error.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_get(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    key_ptr: *const u8,
    key_len: usize,
    val_out: *mut *mut u8,
    len_out: *mut usize,
) -> c_int {
    if ptr.is_null() || key_ptr.is_null() || val_out.is_null() || len_out.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    let key = unsafe { slice::from_raw_parts(key_ptr, key_len) };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    match db.get(txn, key) {
        Ok(Some(val)) => {
            // Copy value to heap to pass to C
            let mut buf = val.into_boxed_slice();
            unsafe {
                *val_out = buf.as_mut_ptr();
                *len_out = buf.len();
            }
            let _ = Box::into_raw(buf); // Leak memory, caller must free
            0
        }
        Ok(None) => 1, // Not found
        Err(_) => -1,
    }
}

/// Free bytes allocated by sochdb_get.
/// # Safety
/// ptr must be a valid pointer returned by sochdb_get.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_free_bytes(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

/// Delete a key.
/// Returns 0 on success, -1 on error.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_delete(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    key_ptr: *const u8,
    key_len: usize,
) -> c_int {
    if ptr.is_null() || key_ptr.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    let key = unsafe { slice::from_raw_parts(key_ptr, key_len) };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    match db.delete(txn, key) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Put path.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_put_path(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    path_ptr: *const c_char,
    val_ptr: *const u8,
    val_len: usize,
) -> c_int {
    if ptr.is_null() || path_ptr.is_null() || val_ptr.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    let c_str = unsafe { CStr::from_ptr(path_ptr) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let val = unsafe { slice::from_raw_parts(val_ptr, val_len) };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    match db.put_path(txn, path_str, val) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get path.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_get_path(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    path_ptr: *const c_char,
    val_out: *mut *mut u8,
    len_out: *mut usize,
) -> c_int {
    if ptr.is_null() || path_ptr.is_null() || val_out.is_null() || len_out.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    let c_str = unsafe { CStr::from_ptr(path_ptr) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    match db.get_path(txn, path_str) {
        Ok(Some(val)) => {
            let mut buf = val.into_boxed_slice();
            unsafe {
                *val_out = buf.as_mut_ptr();
                *len_out = buf.len();
            }
            let _ = Box::into_raw(buf);
            0
        }
        Ok(None) => 1,
        Err(_) => -1,
    }
}

/// Opaque pointer to Scan Iterator
#[allow(clippy::type_complexity)]
pub struct ScanIteratorPtr(
    Box<dyn Iterator<Item = Result<(Vec<u8>, Vec<u8>), sochdb_core::SochDBError>>>,
);

/// Start a scan.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_scan(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    start_ptr: *const u8,
    start_len: usize,
    end_ptr: *const u8,
    end_len: usize,
) -> *mut ScanIteratorPtr {
    if ptr.is_null() {
        return ptr::null_mut();
    }
    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    let start = if !start_ptr.is_null() && start_len > 0 {
        unsafe { slice::from_raw_parts(start_ptr, start_len).to_vec() }
    } else {
        vec![]
    };

    let end = if !end_ptr.is_null() && end_len > 0 {
        unsafe { slice::from_raw_parts(end_ptr, end_len).to_vec() }
    } else {
        vec![] // Empty end means unbounded in `scan` usually, or we need to handle it
    };

    // Note: The underlying `scan` method expects `Range<Vec<u8>>`.
    // We need to handle empty start/end correctly.
    // For now, let's assume the caller provides valid bounds or we use defaults.
    // Ideally, we'd pass optionals.

    // Using a simplified approach: if start is empty, use empty vec (start of db).
    // If end is empty, use a "max" key or handle in `scan` impl.
    // The `StorageEngine::scan` takes `Range<Vec<u8>>`.

    // Using a simplified approach: if start is empty, use empty vec (start of db).
    // If end is empty, use empty vec (unbounded).

    match db.scan_range(txn, &start, &end) {
        Ok(rows) => {
            // Convert rows to an iterator of (key, value)
            // scan_range returns Vec<(Vec<u8>, Vec<u8>)>
            let iter = Box::new(rows.into_iter().map(Ok));

            let ptr = Box::new(ScanIteratorPtr(iter));
            Box::into_raw(ptr)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Start a prefix scan - returns only keys that start with the given prefix.
/// This is the safe method for multi-tenant isolation.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_scan_prefix(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    prefix_ptr: *const u8,
    prefix_len: usize,
) -> *mut ScanIteratorPtr {
    if ptr.is_null() {
        return ptr::null_mut();
    }
    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    let prefix = if !prefix_ptr.is_null() && prefix_len > 0 {
        unsafe { slice::from_raw_parts(prefix_ptr, prefix_len).to_vec() }
    } else {
        vec![]
    };

    // Use the proper scan method that filters by prefix
    match db.scan(txn, &prefix) {
        Ok(rows) => {
            // The underlying scan already filters by prefix, but double-check
            // to ensure no data leakage
            let prefix_owned = prefix.clone();
            let filtered: Vec<(Vec<u8>, Vec<u8>)> = rows
                .into_iter()
                .filter(|(k, _)| k.starts_with(&prefix_owned))
                .collect();
            
            let iter = Box::new(filtered.into_iter().map(Ok));
            let ptr = Box::new(ScanIteratorPtr(iter));
            Box::into_raw(ptr)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Get next item from scan iterator.
/// Returns 0 on success, 1 on done, -1 on error.
/// # Safety
/// All pointer arguments must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_scan_next(
    iter_ptr: *mut ScanIteratorPtr,
    key_out: *mut *mut u8,
    key_len_out: *mut usize,
    val_out: *mut *mut u8,
    val_len_out: *mut usize,
) -> c_int {
    if iter_ptr.is_null() || key_out.is_null() || val_out.is_null() {
        return -1;
    }
    let iter = unsafe { &mut (*iter_ptr).0 };

    match iter.next() {
        Some(Ok((key, val))) => {
            let mut key_buf = key.into_boxed_slice();
            let mut val_buf = val.into_boxed_slice();
            unsafe {
                *key_out = key_buf.as_mut_ptr();
                *key_len_out = key_buf.len();
                *val_out = val_buf.as_mut_ptr();
                *val_len_out = val_buf.len();
            }
            let _ = Box::into_raw(key_buf);
            let _ = Box::into_raw(val_buf);
            0
        }
        Some(Err(_)) => -1,
        None => 1, // Done
    }
}

/// Free scan iterator.
/// # Safety
/// ptr must be a valid pointer returned by sochdb_scan.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_scan_free(ptr: *mut ScanIteratorPtr) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Checkpoint the database.
/// # Safety
/// ptr must be a valid pointer returned by sochdb_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_checkpoint(ptr: *mut DatabasePtr) -> c_int {
    if ptr.is_null() {
        return -1;
    }
    let db = unsafe { &(*ptr).0 };
    match db.flush() {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Storage statistics
#[repr(C)]
pub struct CStorageStats {
    pub memtable_size_bytes: u64,
    pub wal_size_bytes: u64,
    pub active_transactions: usize,
    pub min_active_snapshot: u64,
    pub last_checkpoint_lsn: u64,
}

/// Get storage statistics.
/// # Safety
/// ptr must be a valid pointer returned by sochdb_open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_stats(ptr: *mut DatabasePtr) -> CStorageStats {
    if ptr.is_null() {
        return CStorageStats {
            memtable_size_bytes: 0,
            wal_size_bytes: 0,
            active_transactions: 0,
            min_active_snapshot: 0,
            last_checkpoint_lsn: 0,
        };
    }
    let db = unsafe { &(*ptr).0 };
    let stats = db.storage_stats();

    CStorageStats {
        memtable_size_bytes: stats.memtable_size_bytes,
        wal_size_bytes: stats.wal_size_bytes,
        active_transactions: stats.active_transactions,
        min_active_snapshot: stats.min_active_snapshot,
        last_checkpoint_lsn: stats.last_checkpoint_lsn,
    }
}

// ============================================================================
// Batched Operations - Minimize FFI Call Overhead
// ============================================================================

/// Batch descriptor for put_many operation
///
/// Memory layout for batch:
/// ```text
/// [num_entries: u32]
/// For each entry:
///   [key_len: u32][value_len: u32][key_bytes: ...][value_bytes: ...]
/// ```
///
/// This packed format minimizes FFI crossing overhead:
/// - One call instead of N calls
/// - No per-entry pointer chasing
/// - Contiguous memory for CPU cache efficiency
#[repr(C)]
pub struct CBatchPut {
    /// Pointer to packed batch data
    pub data: *const u8,
    /// Total length of packed data
    pub len: usize,
}

/// Put multiple key-value pairs in a single FFI call.
///
/// This is the high-performance path for Python and other FFI users.
/// Instead of N individual sochdb_put calls (each with FFI overhead),
/// the caller packs all writes into a single buffer and makes one call.
///
/// ## Performance
///
/// For N writes with per-call overhead c:
/// - Individual puts: N × c (e.g., 100 × 500ns = 50µs overhead)
/// - put_many: 1 × c (e.g., 1 × 500ns = 0.5µs overhead)
/// - Speedup: 100× for FFI overhead alone
///
/// ## Buffer Format
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────┐
/// │  num_entries (4 bytes, little-endian u32)                      │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 1:                                                      │
/// │    key_len (4 bytes, u32) | val_len (4 bytes, u32)             │
/// │    key_bytes (key_len bytes)                                   │
/// │    value_bytes (val_len bytes)                                 │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 2: ...                                                  │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry N: ...                                                  │
/// └────────────────────────────────────────────────────────────────┘
/// ```
///
/// ## Returns
///
/// - 0: All entries written successfully
/// - -1: Error (null pointer, invalid format, write failure)
/// - >0: Number of entries successfully written before error
///
/// # Safety
///
/// - `ptr` must be a valid DatabasePtr from sochdb_open
/// - `batch` must point to a valid CBatchPut with correct format
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_put_many(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    batch: CBatchPut,
) -> c_int {
    if ptr.is_null() || batch.data.is_null() || batch.len < 4 {
        return -1;
    }

    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    // Parse batch
    let data = unsafe { slice::from_raw_parts(batch.data, batch.len) };
    
    // Read number of entries
    if data.len() < 4 {
        return -1;
    }
    let num_entries = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    
    let mut offset = 4;
    let mut success_count = 0;

    for _ in 0..num_entries {
        // Read key_len and value_len
        if offset + 8 > data.len() {
            return success_count;
        }
        let key_len = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        ]) as usize;
        let val_len = u32::from_le_bytes([
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
        ]) as usize;
        offset += 8;

        // Read key and value
        if offset + key_len + val_len > data.len() {
            return success_count;
        }
        let key = &data[offset..offset + key_len];
        offset += key_len;
        let value = &data[offset..offset + val_len];
        offset += val_len;

        // Write to database
        match db.put(txn, key, value) {
            Ok(_) => success_count += 1,
            Err(_) => return success_count,
        }
    }

    success_count
}

/// Delete multiple keys in a single FFI call.
///
/// ## Buffer Format
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────┐
/// │  num_entries (4 bytes, little-endian u32)                      │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 1:                                                      │
/// │    key_len (4 bytes, u32)                                      │
/// │    key_bytes (key_len bytes)                                   │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 2: ...                                                  │
/// └────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Safety
///
/// Same as sochdb_put_many.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_delete_many(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    keys_data: *const u8,
    keys_len: usize,
) -> c_int {
    if ptr.is_null() || keys_data.is_null() || keys_len < 4 {
        return -1;
    }

    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    let data = unsafe { slice::from_raw_parts(keys_data, keys_len) };
    
    let num_entries = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    
    let mut offset = 4;
    let mut success_count = 0;

    for _ in 0..num_entries {
        if offset + 4 > data.len() {
            return success_count;
        }
        let key_len = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        ]) as usize;
        offset += 4;

        if offset + key_len > data.len() {
            return success_count;
        }
        let key = &data[offset..offset + key_len];
        offset += key_len;

        match db.delete(txn, key) {
            Ok(_) => success_count += 1,
            Err(_) => return success_count,
        }
    }

    success_count
}

/// Get multiple values in a single FFI call.
///
/// ## Input Format
///
/// Same as delete_many: packed keys.
///
/// ## Output Format
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────┐
/// │  num_results (4 bytes, u32)                                    │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 1:                                                      │
/// │    status (1 byte): 0=found, 1=not_found, 2=error              │
/// │    if found: val_len (4 bytes, u32), value_bytes               │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 2: ...                                                  │
/// └────────────────────────────────────────────────────────────────┘
/// ```
///
/// ## Returns
///
/// Pointer to allocated result buffer. Caller must free with sochdb_free_bytes.
///
/// # Safety
///
/// Same as sochdb_put_many.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_get_many(
    ptr: *mut DatabasePtr,
    handle: C_TxnHandle,
    keys_data: *const u8,
    keys_len: usize,
    result_out: *mut *mut u8,
    result_len_out: *mut usize,
) -> c_int {
    if ptr.is_null() || keys_data.is_null() || keys_len < 4 
        || result_out.is_null() || result_len_out.is_null() {
        return -1;
    }

    let db = unsafe { &(*ptr).0 };
    let txn = TxnHandle {
        txn_id: handle.txn_id,
        snapshot_ts: handle.snapshot_ts,
    };

    let data = unsafe { slice::from_raw_parts(keys_data, keys_len) };
    
    let num_entries = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    
    // Build result buffer
    let mut result = Vec::with_capacity(4 + num_entries * 10); // Estimate
    result.extend_from_slice(&(num_entries as u32).to_le_bytes());
    
    let mut offset = 4;

    for _ in 0..num_entries {
        if offset + 4 > data.len() {
            result.push(2); // Error
            continue;
        }
        let key_len = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        ]) as usize;
        offset += 4;

        if offset + key_len > data.len() {
            result.push(2); // Error
            continue;
        }
        let key = &data[offset..offset + key_len];
        offset += key_len;

        match db.get(txn, key) {
            Ok(Some(value)) => {
                result.push(0); // Found
                result.extend_from_slice(&(value.len() as u32).to_le_bytes());
                result.extend_from_slice(&value);
            }
            Ok(None) => {
                result.push(1); // Not found
            }
            Err(_) => {
                result.push(2); // Error
            }
        }
    }

    // Return result
    let mut boxed = result.into_boxed_slice();
    unsafe {
        *result_out = boxed.as_mut_ptr();
        *result_len_out = boxed.len();
    }
    let _ = Box::into_raw(boxed); // Leak for caller to free
    
    0
}

// ============================================================================
// Batched Scan - Minimize FFI Call Overhead for Iterations
// ============================================================================

/// Fetch a batch of results from scan iterator.
///
/// This dramatically reduces FFI overhead for scan operations.
/// Instead of N calls to `sochdb_scan_next` (each with FFI overhead),
/// fetch up to `batch_size` results in a single call.
///
/// ## Performance
///
/// For N results with per-call overhead c:
/// - Individual next calls: N × c (e.g., 10000 × 500ns = 5ms overhead)
/// - Batched (size=1000): 10 × c (e.g., 10 × 500ns = 5µs overhead)
/// - Speedup: 1000× for FFI overhead
///
/// ## Output Format
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────┐
/// │  num_results (4 bytes, little-endian u32)                      │
/// │  is_done (1 byte): 0=more results available, 1=scan complete   │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 1:                                                      │
/// │    key_len (4 bytes, u32)                                      │
/// │    val_len (4 bytes, u32)                                      │
/// │    key_bytes (key_len bytes)                                   │
/// │    value_bytes (val_len bytes)                                 │
/// ├────────────────────────────────────────────────────────────────┤
/// │  Entry 2: ...                                                  │
/// └────────────────────────────────────────────────────────────────┘
/// ```
///
/// ## Returns
///
/// - 0: Batch fetched successfully (check is_done flag for completion)
/// - 1: Scan complete (no more results)
/// - -1: Error
///
/// ## Usage from Python
///
/// ```python
/// iter_ptr = lib.sochdb_scan(...)
/// while True:
///     result = lib.sochdb_scan_batch(iter_ptr, 1000, ...)
///     if result == 1:  # Done
///         break
///     # Parse batch buffer for up to 1000 results
/// lib.sochdb_scan_free(iter_ptr)
/// ```
///
/// # Safety
///
/// - `iter_ptr` must be a valid ScanIteratorPtr from sochdb_scan
/// - Output pointers must be valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_scan_batch(
    iter_ptr: *mut ScanIteratorPtr,
    batch_size: usize,
    result_out: *mut *mut u8,
    result_len_out: *mut usize,
) -> c_int {
    if iter_ptr.is_null() || result_out.is_null() || result_len_out.is_null() || batch_size == 0 {
        return -1;
    }

    let iter = unsafe { &mut (*iter_ptr).0 };
    
    // Pre-allocate result buffer
    // Estimate: header (5 bytes) + batch_size * (8 bytes header + ~100 bytes avg data)
    let estimated_size = 5 + batch_size * 108;
    let mut result = Vec::with_capacity(estimated_size);
    
    // Reserve space for header (will fill in at end)
    result.extend_from_slice(&[0u8; 5]); // 4 bytes count + 1 byte is_done
    
    let mut count = 0u32;
    let mut is_done = false;
    
    for _ in 0..batch_size {
        match iter.next() {
            Some(Ok((key, val))) => {
                // Write key_len, val_len, key, value
                result.extend_from_slice(&(key.len() as u32).to_le_bytes());
                result.extend_from_slice(&(val.len() as u32).to_le_bytes());
                result.extend_from_slice(&key);
                result.extend_from_slice(&val);
                count += 1;
            }
            Some(Err(_)) => {
                // Write header with current count and return error
                result[0..4].copy_from_slice(&count.to_le_bytes());
                result[4] = 0; // Not done (error case)
                
                let mut boxed = result.into_boxed_slice();
                unsafe {
                    *result_out = boxed.as_mut_ptr();
                    *result_len_out = boxed.len();
                }
                let _ = Box::into_raw(boxed);
                return -1;
            }
            None => {
                is_done = true;
                break;
            }
        }
    }
    
    // Fill in header
    result[0..4].copy_from_slice(&count.to_le_bytes());
    result[4] = if is_done { 1 } else { 0 };
    
    // If no results and done, signal completion
    if count == 0 && is_done {
        // Still allocate minimal buffer so caller can free consistently
        let mut boxed = result.into_boxed_slice();
        unsafe {
            *result_out = boxed.as_mut_ptr();
            *result_len_out = boxed.len();
        }
        let _ = Box::into_raw(boxed);
        return 1; // Done
    }
    
    // Return buffer
    let mut boxed = result.into_boxed_slice();
    unsafe {
        *result_out = boxed.as_mut_ptr();
        *result_len_out = boxed.len();
    }
    let _ = Box::into_raw(boxed);
    
    0 // Success, check is_done flag for completion
}

// ============================================================================
// Per-Table Index Policy API
// ============================================================================

/// Set index policy for a table.
///
/// # Policy Values
/// - 0: WriteOptimized - O(1) writes, O(N) scans. For write-heavy tables.
/// - 1: Balanced (default) - O(1) amortized writes, O(output + log K) scans.
/// - 2: ScanOptimized - O(log N) writes, O(log N + K) scans. For analytics.
/// - 3: AppendOnly - O(1) writes, O(N) forward-only scans. For time-series.
///
/// # Returns
/// - 0: Success
/// - -1: Invalid pointer or table name
/// - -2: Invalid policy value
///
/// # Safety
/// ptr must be a valid DatabasePtr, table_name must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_set_table_index_policy(
    ptr: *mut DatabasePtr,
    table_name: *const c_char,
    policy: u8,
) -> c_int {
    if ptr.is_null() || table_name.is_null() {
        return -1;
    }
    
    let c_str = unsafe { CStr::from_ptr(table_name) };
    let table = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    let index_policy = match policy {
        0 => crate::index_policy::IndexPolicy::WriteOptimized,
        1 => crate::index_policy::IndexPolicy::Balanced,
        2 => crate::index_policy::IndexPolicy::ScanOptimized,
        3 => crate::index_policy::IndexPolicy::AppendOnly,
        _ => return -2,
    };
    
    let db = unsafe { &(*ptr).0 };
    
    // Configure the table's index policy through the database registry
    let config = crate::index_policy::TableIndexConfig::new(table, index_policy);
    db.index_registry().configure_table(config);
    
    0
}

/// Get index policy for a table.
///
/// # Returns
/// - 0: WriteOptimized
/// - 1: Balanced  
/// - 2: ScanOptimized
/// - 3: AppendOnly
/// - 255: Error (invalid pointer)
///
/// # Safety
/// ptr must be a valid DatabasePtr, table_name must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_get_table_index_policy(
    ptr: *mut DatabasePtr,
    table_name: *const c_char,
) -> u8 {
    if ptr.is_null() || table_name.is_null() {
        return 255;
    }
    
    let c_str = unsafe { CStr::from_ptr(table_name) };
    let table = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return 255,
    };
    
    let db = unsafe { &(*ptr).0 };
    let config = db.index_registry().get_config(table);
    
    match config.policy {
        crate::index_policy::IndexPolicy::WriteOptimized => 0,
        crate::index_policy::IndexPolicy::Balanced => 1,
        crate::index_policy::IndexPolicy::ScanOptimized => 2,
        crate::index_policy::IndexPolicy::AppendOnly => 3,
    }
}

/// C-compatible Temporal Edge
#[repr(C)]
pub struct C_TemporalEdge {
    pub from_id: *const c_char,
    pub edge_type: *const c_char,
    pub to_id: *const c_char,
    pub valid_from: u64,
    pub valid_until: u64,
    pub properties_json: *const c_char,  // JSON string of properties
}

/// Add a temporal edge with validity interval.
/// # Safety
/// All pointers must be valid C strings. properties_json can be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_add_temporal_edge(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    edge: C_TemporalEdge,
) -> c_int {
    if ptr.is_null() || namespace.is_null() || edge.from_id.is_null() 
        || edge.edge_type.is_null() || edge.to_id.is_null() {
        return -1;
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let from = match unsafe { CStr::from_ptr(edge.from_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let etype = match unsafe { CStr::from_ptr(edge.edge_type) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let to = match unsafe { CStr::from_ptr(edge.to_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let db = unsafe { &(*ptr).0 };
    
    // Begin transaction for atomic write
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    // Store temporal edge: _graph/{ns}/temporal/{from}/{type}/{to}/{valid_from}
    let key = format!(
        "_graph/{}/temporal/{}/{}/{}/{:016x}",
        ns, from, etype, to, edge.valid_from
    );
    
    let props_str = if edge.properties_json.is_null() {
        "{}".to_string()
    } else {
        match unsafe { CStr::from_ptr(edge.properties_json) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };
    
    let value = format!(
        r#"{{"from_id":"{}","edge_type":"{}","to_id":"{}","valid_from":{},"valid_until":{},"properties":{}}}"#,
        from, etype, to, edge.valid_from, edge.valid_until, props_str
    );
    
    if let Err(_) = db.put(txn, key.as_bytes(), value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    match db.commit(txn) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Query temporal graph edges. Returns a JSON array of matching edges.
/// Caller must free the returned string with sochdb_free_string.
/// 
/// query_mode: 0=POINT_IN_TIME, 1=RANGE, 2=CURRENT
/// # Safety
/// All pointers must be valid C strings. edge_type can be null for no filter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_query_temporal_graph(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    node_id: *const c_char,
    query_mode: u8,
    timestamp: u64,      // For POINT_IN_TIME
    start_time: u64,     // For RANGE
    end_time: u64,       // For RANGE
    edge_type: *const c_char,  // Optional filter (null = all types)
    out_len: *mut usize,
) -> *mut c_char {
    if ptr.is_null() || namespace.is_null() || node_id.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let node = match unsafe { CStr::from_ptr(node_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    
    let edge_filter = if edge_type.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(edge_type) }.to_str() {
            Ok(s) => Some(s),
            Err(_) => return ptr::null_mut(),
        }
    };

    let db = unsafe { &(*ptr).0 };
    
    // Begin transaction for scan
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };
    
    // Scan prefix: _graph/{ns}/temporal/{node}/
    let prefix = format!("_graph/{}/temporal/{}/", ns, node);
    let pairs = match db.scan(txn, prefix.as_bytes()) {
        Ok(p) => p,
        Err(_) => {
            let _ = db.abort(txn);
            return ptr::null_mut();
        }
    };
    
    // Commit read transaction
    if let Err(_) = db.commit(txn) {
        return ptr::null_mut();
    }
    
    let mut results = Vec::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    
    for (_key, value) in pairs {
        // Parse the JSON value
        let value_str = match std::str::from_utf8(&value) {
            Ok(s) => s,
            Err(_) => continue,
        };
        
        // Simple JSON parsing (in production, use serde_json)
        if let Some(valid_from_pos) = value_str.find(r#""valid_from":"#) {
            if let Some(valid_until_pos) = value_str.find(r#""valid_until":"#) {
                let vf_start = valid_from_pos + r#""valid_from":"#.len();
                let vf_end = value_str[vf_start..].find(',').unwrap_or(0) + vf_start;
                let vu_start = valid_until_pos + r#""valid_until":"#.len();
                let vu_end = value_str[vu_start..].find(',').unwrap_or(0) + vu_start;
                
                let valid_from: u64 = value_str[vf_start..vf_end].parse().unwrap_or(0);
                let valid_until: u64 = value_str[vu_start..vu_end].parse().unwrap_or(0);
                
                // Filter by edge_type if specified
                if let Some(filter) = edge_filter {
                    if !value_str.contains(&format!(r#""edge_type":"{}""#, filter)) {
                        continue;
                    }
                }
                
                // Filter by query mode
                let matches = match query_mode {
                    0 => timestamp >= valid_from && (valid_until == 0 || timestamp < valid_until),
                    1 => {
                        let edge_end = if valid_until == 0 { u64::MAX } else { valid_until };
                        valid_from < end_time && edge_end > start_time
                    }
                    2 => now >= valid_from && (valid_until == 0 || now < valid_until),
                    _ => false,
                };
                
                if matches {
                    results.push(value_str.to_string());
                }
            }
        }
    }
    
    // Build JSON array
    let json = format!("[{}]", results.join(","));
    let c_string = match std::ffi::CString::new(json) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    
    unsafe { *out_len = c_string.as_bytes().len() };
    c_string.into_raw()
}

/// Free a string returned by sochdb_query_temporal_graph.
/// # Safety
/// The ptr must be a valid pointer returned by sochdb_query_temporal_graph.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}

// ============================================================================
// Graph Overlay FFI - Nodes, Edges, Traversal
// ============================================================================

/// Add a node to the graph overlay.
/// 
/// Stores node as: _graph/{namespace}/nodes/{node_id}
/// 
/// # Returns
/// - 0: Success
/// - -1: Error
/// 
/// # Safety
/// All pointers must be valid C strings. properties_json can be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_graph_add_node(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    node_id: *const c_char,
    node_type: *const c_char,
    properties_json: *const c_char,
) -> c_int {
    if ptr.is_null() || namespace.is_null() || node_id.is_null() || node_type.is_null() {
        return -1;
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let id = match unsafe { CStr::from_ptr(node_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let ntype = match unsafe { CStr::from_ptr(node_type) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let props = if properties_json.is_null() {
        "{}".to_string()
    } else {
        match unsafe { CStr::from_ptr(properties_json) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let db = unsafe { &(*ptr).0 };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    let key = format!("_graph/{}/nodes/{}", ns, id);
    let value = format!(
        r#"{{"id":"{}","node_type":"{}","properties":{}}}"#,
        id, ntype, props
    );
    
    if let Err(_) = db.put(txn, key.as_bytes(), value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    match db.commit(txn) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Add an edge between nodes in the graph overlay.
/// 
/// Stores edge as: _graph/{namespace}/edges/{from_id}/{edge_type}/{to_id}
/// 
/// # Returns
/// - 0: Success
/// - -1: Error
/// 
/// # Safety
/// All pointers must be valid C strings. properties_json can be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_graph_add_edge(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    from_id: *const c_char,
    edge_type: *const c_char,
    to_id: *const c_char,
    properties_json: *const c_char,
) -> c_int {
    if ptr.is_null() || namespace.is_null() || from_id.is_null() 
        || edge_type.is_null() || to_id.is_null() {
        return -1;
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let from = match unsafe { CStr::from_ptr(from_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let etype = match unsafe { CStr::from_ptr(edge_type) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let to = match unsafe { CStr::from_ptr(to_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let props = if properties_json.is_null() {
        "{}".to_string()
    } else {
        match unsafe { CStr::from_ptr(properties_json) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let db = unsafe { &(*ptr).0 };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    let key = format!("_graph/{}/edges/{}/{}/{}", ns, from, etype, to);
    let value = format!(
        r#"{{"from_id":"{}","edge_type":"{}","to_id":"{}","properties":{}}}"#,
        from, etype, to, props
    );
    
    if let Err(_) = db.put(txn, key.as_bytes(), value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    match db.commit(txn) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Traverse the graph from a starting node.
/// 
/// Returns JSON: {"nodes": [...], "edges": [...]}
/// Caller must free the returned string with sochdb_free_string.
/// 
/// order: 0=BFS, 1=DFS
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_graph_traverse(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    start_node: *const c_char,
    max_depth: usize,
    order: u8,  // 0=BFS, 1=DFS
    out_len: *mut usize,
) -> *mut c_char {
    if ptr.is_null() || namespace.is_null() || start_node.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let start = match unsafe { CStr::from_ptr(start_node) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let db = unsafe { &(*ptr).0 };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };
    
    // Collect nodes and edges through traversal
    let mut visited_nodes = std::collections::HashSet::new();
    let mut nodes_json = Vec::new();
    let mut edges_json = Vec::new();
    
    // Use queue for BFS, stack for DFS
    let mut frontier: Vec<(String, usize)> = vec![(start.to_string(), 0)];
    
    while let Some((current_node, depth)) = if order == 0 {
        // BFS: remove from front
        if frontier.is_empty() { None } else { Some(frontier.remove(0)) }
    } else {
        // DFS: remove from back
        frontier.pop()
    } {
        if depth > max_depth || visited_nodes.contains(&current_node) {
            continue;
        }
        visited_nodes.insert(current_node.clone());
        
        // Get node data
        let node_key = format!("_graph/{}/nodes/{}", ns, current_node);
        if let Ok(Some(node_data)) = db.get(txn, node_key.as_bytes()) {
            if let Ok(s) = std::str::from_utf8(&node_data) {
                nodes_json.push(s.to_string());
            }
        }
        
        // Get outgoing edges
        let edge_prefix = format!("_graph/{}/edges/{}/", ns, current_node);
        if let Ok(edges) = db.scan(txn, edge_prefix.as_bytes()) {
            for (_key, value) in edges {
                if let Ok(edge_str) = std::str::from_utf8(&value) {
                    edges_json.push(edge_str.to_string());
                    
                    // Extract to_id for traversal
                    if let Some(to_pos) = edge_str.find(r#""to_id":""#) {
                        let start_idx = to_pos + r#""to_id":""#.len();
                        if let Some(end_idx) = edge_str[start_idx..].find('"') {
                            let to_id = &edge_str[start_idx..start_idx + end_idx];
                            if !visited_nodes.contains(to_id) {
                                frontier.push((to_id.to_string(), depth + 1));
                            }
                        }
                    }
                }
            }
        }
    }
    
    if let Err(_) = db.commit(txn) {
        return ptr::null_mut();
    }
    
    let result = format!(
        r#"{{"nodes":[{}],"edges":[{}]}}"#,
        nodes_json.join(","),
        edges_json.join(",")
    );
    
    let c_string = match std::ffi::CString::new(result) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    
    unsafe { *out_len = c_string.as_bytes().len() };
    c_string.into_raw()
}

// ============================================================================
// Semantic Cache FFI
// ============================================================================

/// Store a value in the semantic cache with its embedding.
/// 
/// # Returns
/// - 0: Success
/// - -1: Error
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_cache_put(
    ptr: *mut DatabasePtr,
    cache_name: *const c_char,
    key: *const c_char,
    value: *const c_char,
    embedding_ptr: *const f32,
    embedding_len: usize,
    ttl_seconds: u64,
) -> c_int {
    if ptr.is_null() || cache_name.is_null() || key.is_null() 
        || value.is_null() || embedding_ptr.is_null() {
        return -1;
    }

    let cache = match unsafe { CStr::from_ptr(cache_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let k = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let v = match unsafe { CStr::from_ptr(value) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let embedding = unsafe { slice::from_raw_parts(embedding_ptr, embedding_len) };

    let db = unsafe { &(*ptr).0 };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    // Compute expiry timestamp
    let expires_at = if ttl_seconds > 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() + ttl_seconds
    } else {
        0 // No expiry
    };
    
    // Store cache entry: _cache/{cache_name}/{key_hash}
    let key_hash = format!("{:016x}", twox_hash::xxh3::hash64(k.as_bytes()));
    let cache_key = format!("_cache/{}/{}", cache, key_hash);
    
    // Serialize embedding as JSON array
    let embedding_json: Vec<String> = embedding.iter().map(|f| f.to_string()).collect();
    
    let cache_value = format!(
        r#"{{"key":"{}","value":"{}","embedding":[{}],"expires_at":{}}}"#,
        k, v, embedding_json.join(","), expires_at
    );
    
    if let Err(_) = db.put(txn, cache_key.as_bytes(), cache_value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    match db.commit(txn) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Look up a value in the semantic cache by embedding similarity.
/// 
/// Returns the cached value if similarity >= threshold, null otherwise.
/// Caller must free the returned string with sochdb_free_string.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_cache_get(
    ptr: *mut DatabasePtr,
    cache_name: *const c_char,
    query_embedding_ptr: *const f32,
    embedding_len: usize,
    threshold: f32,
    out_len: *mut usize,
) -> *mut c_char {
    if ptr.is_null() || cache_name.is_null() || query_embedding_ptr.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }

    let cache = match unsafe { CStr::from_ptr(cache_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let query = unsafe { slice::from_raw_parts(query_embedding_ptr, embedding_len) };

    let db = unsafe { &(*ptr).0 };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };
    
    let prefix = format!("_cache/{}/", cache);
    let entries = match db.scan(txn, prefix.as_bytes()) {
        Ok(e) => e,
        Err(_) => {
            let _ = db.abort(txn);
            return ptr::null_mut();
        }
    };
    
    let _ = db.commit(txn);
    
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let mut best_match: Option<(f32, String)> = None;
    
    for (_key, value) in entries {
        let value_str = match std::str::from_utf8(&value) {
            Ok(s) => s,
            Err(_) => continue,
        };
        
        // Parse expires_at
        if let Some(exp_pos) = value_str.find(r#""expires_at":"#) {
            let exp_start = exp_pos + r#""expires_at":"#.len();
            if let Some(exp_end) = value_str[exp_start..].find('}') {
                let expires_at: u64 = value_str[exp_start..exp_start + exp_end]
                    .parse()
                    .unwrap_or(0);
                if expires_at > 0 && now > expires_at {
                    continue; // Expired
                }
            }
        }
        
        // Parse embedding and compute cosine similarity
        if let Some(emb_pos) = value_str.find(r#""embedding":["#) {
            let emb_start = emb_pos + r#""embedding":["#.len();
            if let Some(emb_end) = value_str[emb_start..].find(']') {
                let emb_str = &value_str[emb_start..emb_start + emb_end];
                let cached_embedding: Vec<f32> = emb_str
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                
                if cached_embedding.len() == query.len() {
                    let similarity = cosine_similarity(query, &cached_embedding);
                    if similarity >= threshold {
                        if best_match.is_none() || similarity > best_match.as_ref().unwrap().0 {
                            // Extract value field
                            if let Some(val_pos) = value_str.find(r#""value":""#) {
                                let val_start = val_pos + r#""value":""#.len();
                                if let Some(val_end) = value_str[val_start..].find('"') {
                                    let cached_value = &value_str[val_start..val_start + val_end];
                                    best_match = Some((similarity, cached_value.to_string()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    match best_match {
        Some((_, value)) => {
            let c_string = match std::ffi::CString::new(value) {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            };
            unsafe { *out_len = c_string.as_bytes().len() };
            c_string.into_raw()
        }
        None => ptr::null_mut(),
    }
}

/// Compute cosine similarity between two vectors
/// Returns normalized similarity in [0, 1] range for threshold comparisons
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        let similarity = dot / (norm_a * norm_b);
        // Normalize from [-1, 1] to [0, 1] for threshold comparisons
        // This ensures consistent scoring across all SDKs (Python/Node.js/Go)
        (similarity + 1.0) / 2.0
    }
}

// ============================================================================
// Trace Service FFI
// ============================================================================

/// Start a new trace. Returns trace_id and root_span_id.
/// 
/// Caller must free the returned strings with sochdb_free_string.
/// 
/// # Returns
/// - 0: Success
/// - -1: Error
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_trace_start(
    ptr: *mut DatabasePtr,
    name: *const c_char,
    trace_id_out: *mut *mut c_char,
    span_id_out: *mut *mut c_char,
) -> c_int {
    if ptr.is_null() || name.is_null() || trace_id_out.is_null() || span_id_out.is_null() {
        return -1;
    }

    let trace_name = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let db = unsafe { &(*ptr).0 };
    
    // Generate unique IDs
    let trace_id = format!("trace_{:016x}", rand_u64());
    let span_id = format!("span_{:016x}", rand_u64());
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    
    // Store trace: _traces/{trace_id}
    let trace_key = format!("_traces/{}", trace_id);
    let trace_value = format!(
        r#"{{"trace_id":"{}","name":"{}","start_us":{},"root_span_id":"{}"}}"#,
        trace_id, trace_name, now, span_id
    );
    
    if let Err(_) = db.put(txn, trace_key.as_bytes(), trace_value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    // Store root span: _traces/{trace_id}/spans/{span_id}
    let span_key = format!("_traces/{}/spans/{}", trace_id, span_id);
    let span_value = format!(
        r#"{{"span_id":"{}","name":"{}","start_us":{},"parent_span_id":null,"status":"active"}}"#,
        span_id, trace_name, now
    );
    
    if let Err(_) = db.put(txn, span_key.as_bytes(), span_value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    if let Err(_) = db.commit(txn) {
        return -1;
    }
    
    // Return trace_id and span_id
    let trace_c = match std::ffi::CString::new(trace_id) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let span_c = match std::ffi::CString::new(span_id) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    unsafe {
        *trace_id_out = trace_c.into_raw();
        *span_id_out = span_c.into_raw();
    }
    
    0
}

/// Start a child span within a trace.
/// 
/// Caller must free the returned span_id with sochdb_free_string.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_trace_span_start(
    ptr: *mut DatabasePtr,
    trace_id: *const c_char,
    parent_span_id: *const c_char,
    name: *const c_char,
    span_id_out: *mut *mut c_char,
) -> c_int {
    if ptr.is_null() || trace_id.is_null() || parent_span_id.is_null() 
        || name.is_null() || span_id_out.is_null() {
        return -1;
    }

    let tid = match unsafe { CStr::from_ptr(trace_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let pid = match unsafe { CStr::from_ptr(parent_span_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let span_name = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let db = unsafe { &(*ptr).0 };
    let span_id = format!("span_{:016x}", rand_u64());
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    
    let span_key = format!("_traces/{}/spans/{}", tid, span_id);
    let span_value = format!(
        r#"{{"span_id":"{}","name":"{}","start_us":{},"parent_span_id":"{}","status":"active"}}"#,
        span_id, span_name, now, pid
    );
    
    if let Err(_) = db.put(txn, span_key.as_bytes(), span_value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    if let Err(_) = db.commit(txn) {
        return -1;
    }
    
    let span_c = match std::ffi::CString::new(span_id) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    unsafe { *span_id_out = span_c.into_raw() };
    0
}

/// End a span and record its duration.
/// 
/// status: 0=unset, 1=ok, 2=error
/// 
/// # Returns
/// Duration in microseconds on success, -1 on error.
/// 
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_trace_span_end(
    ptr: *mut DatabasePtr,
    trace_id: *const c_char,
    span_id: *const c_char,
    status: u8,
) -> i64 {
    if ptr.is_null() || trace_id.is_null() || span_id.is_null() {
        return -1;
    }

    let tid = match unsafe { CStr::from_ptr(trace_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let sid = match unsafe { CStr::from_ptr(span_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let db = unsafe { &(*ptr).0 };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    let span_key = format!("_traces/{}/spans/{}", tid, sid);
    
    // Read current span
    let span_data = match db.get(txn, span_key.as_bytes()) {
        Ok(Some(data)) => data,
        _ => {
            let _ = db.abort(txn);
            return -1;
        }
    };
    
    let span_str = match std::str::from_utf8(&span_data) {
        Ok(s) => s,
        Err(_) => {
            let _ = db.abort(txn);
            return -1;
        }
    };
    
    // Parse start_us
    let start_us = if let Some(pos) = span_str.find(r#""start_us":"#) {
        let start = pos + r#""start_us":"#.len();
        if let Some(end) = span_str[start..].find(',') {
            span_str[start..start + end].parse().unwrap_or(0u64)
        } else {
            0u64
        }
    } else {
        0u64
    };
    
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    
    let duration_us = now.saturating_sub(start_us);
    let status_str = match status {
        1 => "ok",
        2 => "error",
        _ => "unset",
    };
    
    // Update span with end time and duration
    let new_span = span_str
        .replace(r#""status":"active""#, &format!(r#""status":"{}","end_us":{},"duration_us":{}"#, status_str, now, duration_us));
    
    if let Err(_) = db.put(txn, span_key.as_bytes(), new_span.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    if let Err(_) = db.commit(txn) {
        return -1;
    }
    
    duration_us as i64
}

/// Generate a pseudo-random u64 (simple XorShift for trace IDs)
fn rand_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);
    
    let mut s = STATE.load(Ordering::Relaxed);
    if s == 0 {
        s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    STATE.store(s, Ordering::Relaxed);
    s.wrapping_mul(0x2545F4914F6CDD1D)
}

// =========================================================================
// Vector Index Operations (KV-based, native Rust performance)
// =========================================================================

/// Create a vector collection for storing embeddings
/// 
/// # Returns
/// - 0: Success (or already exists)
/// - -1: Error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_create(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    collection: *const c_char,
    dimension: usize,
    dist_type: u8, // 0=Cosine, 1=Euclidean, 2=Dot
) -> c_int {
    if ptr.is_null() || namespace.is_null() || collection.is_null() {
        return -1;
    }
    
    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let col = match unsafe { CStr::from_ptr(collection) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    let db = unsafe { &(*ptr).0 };
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    // Store collection config
    let config_key = format!("{}/_collections/{}", ns, col);
    let config_value = format!(
        r#"{{"dimension":{},"metric":{}}}"#,
        dimension, dist_type
    );
    
    if let Err(_) = db.put(txn, config_key.as_bytes(), config_value.as_bytes()) {
        let _ = db.abort(txn);
        return -1;
    }
    
    let result = match db.commit(txn) {
        Ok(_) => 0,
        Err(_) => -1,
    };

    if result == 0 {
        let metric = match dist_type {
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };
        let _ = ensure_collection_index(db, ns, col, dimension, metric);
    }

    result
}

/// Insert a vector into a collection
/// 
/// # Returns
/// - 0: Success
/// - -1: Error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_insert(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    collection: *const c_char,
    id: *const c_char,
    vector_ptr: *const f32,
    vector_len: usize,
    metadata_json: *const c_char, // Optional JSON metadata
) -> c_int {
    if ptr.is_null() || namespace.is_null() || collection.is_null() 
        || id.is_null() || vector_ptr.is_null() {
        return -1;
    }
    
    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let col = match unsafe { CStr::from_ptr(collection) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let doc_id = match unsafe { CStr::from_ptr(id) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let vector = unsafe { slice::from_raw_parts(vector_ptr, vector_len) };
    let db = unsafe { &(*ptr).0 };

    let (dimension, metric) = match resolve_collection_config(db, ns, col) {
        Some(config) => config,
        None => (vector_len, DistanceMetric::Cosine),
    };
    if vector_len != dimension {
        return -1;
    }
    
    let metadata = if !metadata_json.is_null() {
        match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => "{}".to_string(),
        }
    } else {
        "{}".to_string()
    };
    
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    let id_hash = hash_id_to_u128(doc_id);
    let vec_key = vector_bin_key(ns, col, id_hash);
    let vec_value = serialize_vector_binary(vector);

    if let Err(_) = db.put(txn, vec_key.as_bytes(), &vec_value) {
        let _ = db.abort(txn);
        return -1;
    }

    let metadata_value = match serde_json::from_str::<serde_json::Value>(&metadata) {
        Ok(value) => serde_json::json!({"id": doc_id, "metadata": value}),
        Err(_) => serde_json::json!({"id": doc_id, "metadata": serde_json::json!({})}),
    };
    let meta_key = metadata_key(ns, col, id_hash);
    if let Ok(meta_bytes) = serde_json::to_vec(&metadata_value) {
        if let Err(_) = db.put(txn, meta_key.as_bytes(), &meta_bytes) {
            let _ = db.abort(txn);
            return -1;
        }
    }
    
    if let Err(_) = db.commit(txn) {
        return -1;
    }

    let index = ensure_collection_index(db, ns, col, dimension, metric);
    let _ = index.index.insert(id_hash, vector.to_vec());

    0
}

/// C-compatible search result
#[repr(C)]
pub struct CSearchResult {
    pub id_ptr: *mut c_char,
    pub score: f32,
    pub metadata_ptr: *mut c_char,
}

/// Search a collection for nearest vectors
/// 
/// # Returns
/// - >= 0: Number of results
/// - -1: Error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_search(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    collection: *const c_char,
    query_ptr: *const f32,
    query_len: usize,
    k: usize,
    results_out: *mut CSearchResult,
) -> c_int {
    if ptr.is_null() || namespace.is_null() || collection.is_null() 
        || query_ptr.is_null() || results_out.is_null() {
        return -1;
    }
    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let col = match unsafe { CStr::from_ptr(collection) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let query = unsafe { slice::from_raw_parts(query_ptr, query_len) };
    let db = unsafe { &(*ptr).0 };
    let (dimension, metric) = match resolve_collection_config(db, ns, col) {
        Some(config) => config,
        None => return 0,
    };

    if query_len != dimension {
        return -1;
    }

    let index = ensure_collection_index(db, ns, col, dimension, metric);
    let mut scored = match index.index.search(query, k) {
        Ok(results) => results,
        Err(_) => return -1,
    };

    let result_count = scored.len().min(k);
    for (i, (id_hash, distance)) in scored.drain(..result_count).enumerate() {
        let meta_key = metadata_key(ns, col, id_hash);
        let txn = match db.begin_transaction() {
            Ok(t) => t,
            Err(_) => return -1,
        };
        let meta_value = db.get(txn, meta_key.as_bytes()).ok().flatten();
        let _ = db.commit(txn);

        let mut id_value = String::new();
        let mut metadata_json = serde_json::json!({});
        if let Some(bytes) = meta_value.as_deref() {
            if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(bytes) {
                id_value = parsed.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                metadata_json = parsed.get("metadata").cloned().unwrap_or(serde_json::json!({}));
            }
        }
        let metadata = serde_json::to_string(&metadata_json).unwrap_or_else(|_| "{}".to_string());

        let c_id = match std::ffi::CString::new(id_value) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        };
        let c_meta = match std::ffi::CString::new(metadata) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        };

        unsafe {
            (*results_out.add(i)).id_ptr = c_id;
            (*results_out.add(i)).score = decode_score(metric, distance);
            (*results_out.add(i)).metadata_ptr = c_meta;
        }
    }

    result_count as c_int
}

/// Search a collection and return results as struct-of-arrays (ids + scores)
///
/// - ids_out: pointer to u64 array (allocated by Rust)
/// - scores_out: pointer to f32 array (allocated by Rust)
/// - len_out: number of results
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_search_soa(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    collection: *const c_char,
    query_ptr: *const f32,
    query_len: usize,
    k: usize,
    min_score: f32,
    filter_json: *const c_char,
    ids_hi_out: *mut *mut u64,
    ids_lo_out: *mut *mut u64,
    scores_out: *mut *mut f32,
    len_out: *mut usize,
) -> c_int {
    if ptr.is_null() || namespace.is_null() || collection.is_null()
        || query_ptr.is_null() || ids_hi_out.is_null() || ids_lo_out.is_null()
        || scores_out.is_null() || len_out.is_null() {
        return -1;
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let col = match unsafe { CStr::from_ptr(collection) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let query = unsafe { slice::from_raw_parts(query_ptr, query_len) };
    let db = unsafe { &(*ptr).0 };

    let (dimension, metric) = match resolve_collection_config(db, ns, col) {
        Some(config) => config,
        None => return 0,
    };
    if query_len != dimension {
        return -1;
    }

    let filter = if !filter_json.is_null() {
        match unsafe { CStr::from_ptr(filter_json) }.to_str() {
            Ok(s) => serde_json::from_str::<serde_json::Value>(s).ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    let index = ensure_collection_index(db, ns, col, dimension, metric);
    let results = match index.index.search(query, k) {
        Ok(results) => results,
        Err(_) => return -1,
    };

    let mut ids_hi: Vec<u64> = Vec::with_capacity(results.len());
    let mut ids_lo: Vec<u64> = Vec::with_capacity(results.len());
    let mut scores: Vec<f32> = Vec::with_capacity(results.len());

    for (id_hash, distance) in results {
        let score = decode_score(metric, distance);
        if min_score > 0.0 && score < min_score {
            continue;
        }

        if let Some(filter_value) = &filter {
            let meta_key = metadata_key(ns, col, id_hash);
            let txn = match db.begin_transaction() {
                Ok(t) => t,
                Err(_) => return -1,
            };
            let meta_value = db.get(txn, meta_key.as_bytes()).ok().flatten();
            let _ = db.commit(txn);
            let meta_value = match meta_value {
                Some(value) => value,
                None => continue,
            };
            let parsed = match serde_json::from_slice::<serde_json::Value>(&meta_value) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let metadata = parsed.get("metadata").cloned().unwrap_or(Value::Null);

            if !metadata_matches_filter(&metadata, filter_value) {
                continue;
            }
        }

        ids_hi.push((id_hash >> 64) as u64);
        ids_lo.push((id_hash & u128::from(u64::MAX)) as u64);
        scores.push(score);
        if ids_hi.len() >= k {
            break;
        }
    }

    let len = ids_hi.len();
    let mut ids_hi_box = ids_hi.into_boxed_slice();
    let mut ids_lo_box = ids_lo.into_boxed_slice();
    let mut scores_box = scores.into_boxed_slice();

    unsafe {
        *len_out = len;
        *ids_hi_out = ids_hi_box.as_mut_ptr();
        *ids_lo_out = ids_lo_box.as_mut_ptr();
        *scores_out = scores_box.as_mut_ptr();
    }

    std::mem::forget(ids_hi_box);
    std::mem::forget(ids_lo_box);
    std::mem::forget(scores_box);

    len as c_int
}

/// Fetch metadata JSON for a list of ids (u64 hashes)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_fetch_metadata_json(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    collection: *const c_char,
    ids_hi_ptr: *const u64,
    ids_lo_ptr: *const u64,
    ids_len: usize,
) -> *mut c_char {
    if ptr.is_null() || namespace.is_null() || collection.is_null()
        || ids_hi_ptr.is_null() || ids_lo_ptr.is_null() {
        return ptr::null_mut();
    }

    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let col = match unsafe { CStr::from_ptr(collection) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let ids_hi = unsafe { slice::from_raw_parts(ids_hi_ptr, ids_len) };
    let ids_lo = unsafe { slice::from_raw_parts(ids_lo_ptr, ids_len) };
    let db = unsafe { &(*ptr).0 };

    let mut results = Vec::with_capacity(ids_len);
    for i in 0..ids_len {
        let id_hash = ((ids_hi[i] as u128) << 64) | (ids_lo[i] as u128);
        let meta_key = metadata_key(ns, col, id_hash);
        let txn = match db.begin_transaction() {
            Ok(t) => t,
            Err(_) => return ptr::null_mut(),
        };
        let meta_value = db.get(txn, meta_key.as_bytes()).ok().flatten();
        let _ = db.commit(txn);
        if let Some(bytes) = meta_value {
            if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                results.push(parsed);
                continue;
            }
        }
        results.push(serde_json::json!({"id": "", "metadata": {}}));
    }

    match serde_json::to_string(&results) {
        Ok(json) => match std::ffi::CString::new(json) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Free arrays returned by sochdb_collection_search_soa
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_free_u64(ptr: *mut u64, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_free_f32(ptr: *mut f32, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

fn metadata_matches_filter(metadata: &Value, filter: &Value) -> bool {
    let filter_obj = match filter.as_object() {
        Some(obj) => obj,
        None => return true,
    };
    let metadata_obj = match metadata.as_object() {
        Some(obj) => obj,
        None => return false,
    };

    for (key, expected) in filter_obj.iter() {
        match metadata_obj.get(key) {
            Some(actual) if actual == expected => {}
            _ => return false,
        }
    }

    true
}

/// Search a collection for keywords (simple term match)
/// 
/// # Returns
/// - >= 0: Number of results
/// - -1: Error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_collection_keyword_search(
    ptr: *mut DatabasePtr,
    namespace: *const c_char,
    collection: *const c_char,
    query_ptr: *const c_char,
    k: usize,
    results_out: *mut CSearchResult,
) -> c_int {
    if ptr.is_null() || namespace.is_null() || collection.is_null() 
        || query_ptr.is_null() || results_out.is_null() {
        return -1;
    }
    
    let ns = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let col = match unsafe { CStr::from_ptr(collection) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let query_str = match unsafe { CStr::from_ptr(query_ptr) }.to_str() {
        Ok(s) => s.to_lowercase(),
        Err(_) => return -1,
    };
    let terms: Vec<&str> = query_str.split_whitespace().collect();
    if terms.is_empty() {
        return 0;
    }
    
    let db = unsafe { &(*ptr).0 };
    let txn = match db.begin_transaction() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    
    // Scan all vectors in collection (we assume vectors & metadata are stored together)
    let prefix = format!("{}/collections/{}/vectors/", ns, col);
    let entries = match db.scan(txn, prefix.as_bytes()) {
        Ok(e) => e,
        Err(_) => {
            let _ = db.abort(txn);
            return -1;
        }
    };
    let _ = db.commit(txn);
    
    // Score documents based on term frequency
    let mut scored: Vec<(f32, String, String)> = Vec::new();
    
    for (_key, value) in entries {
        // Parse whole JSON (robust)
        let doc: Value = match serde_json::from_slice(&value) {
            Ok(v) => v,
            Err(_) => continue,
        };
        
        // Search in metadata string (includes values)
        let metadata_val = doc.get("metadata");
        let metadata_str = metadata_val.map(|v| v.to_string()).unwrap_or("{}".to_string());
        
        // Also check "content" field if present (fallback compat)
        let content_str = doc.get("content").and_then(|v| v.as_str()).unwrap_or("");
        
        // Combine text to search
        let search_text = format!("{} {}", metadata_str, content_str).to_lowercase();
         
        let mut score = 0.0;
        for term in &terms {
            score += search_text.matches(term).count() as f32;
        }
        
        if score > 0.0 {
            let id = doc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if id.is_empty() { continue; }
            
            scored.push((score, id, metadata_str));
        }
    }
    
    // Sort by score descending
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Return top k
    let result_count = scored.len().min(k);
    for (i, (score, id, metadata)) in scored.into_iter().take(k).enumerate() {
        let c_id = match std::ffi::CString::new(id) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        };
        let c_meta = match std::ffi::CString::new(metadata) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        };
        
        unsafe {
            (*results_out.add(i)).id_ptr = c_id;
            (*results_out.add(i)).score = score;
            (*results_out.add(i)).metadata_ptr = c_meta;
        }
    }
    
    result_count as c_int
}

/// Free a search result
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sochdb_search_result_free(result: *mut CSearchResult, count: usize) {
    if result.is_null() {
        return;
    }
    
    for i in 0..count {
        let r = unsafe { &mut *result.add(i) };
        if !r.id_ptr.is_null() {
            let _ = unsafe { std::ffi::CString::from_raw(r.id_ptr) };
        }
        if !r.metadata_ptr.is_null() {
            let _ = unsafe { std::ffi::CString::from_raw(r.metadata_ptr) };
        }
    }
}

