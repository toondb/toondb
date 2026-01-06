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

//! Type-State Transaction API
//!
//! This module provides a compile-time safe transaction API that makes
//! transaction misuse impossible. Using Rust's type system, we encode
//! the transaction lifecycle as types, ensuring that:
//!
//! - Committed transactions cannot be used again
//! - Aborted transactions cannot be committed
//! - Read operations are only allowed on active transactions
//! - Write operations are only allowed on active transactions
//! - Double-commit is impossible
//! - Use-after-commit is impossible
//!
//! ## Type-State Pattern
//!
//! ```text
//! Transaction<Active>
//!     │
//!     ├── commit() ──────► Transaction<Committed> (consumed, no methods)
//!     │
//!     └── abort() ───────► Transaction<Aborted> (consumed, no methods)
//! ```
//!
//! The key insight is that `commit()` and `abort()` **consume** the
//! transaction by taking `self` (not `&mut self`). This means the
//! original transaction variable can no longer be used after the call.
//!
//! ## Usage Example
//!
//! ```ignore
//! let txn = db.begin_transaction();  // Transaction<Active>
//!
//! txn.put(b"key", b"value")?;        // OK - Active allows writes
//! let val = txn.get(b"key")?;        // OK - Active allows reads
//!
//! let committed = txn.commit()?;     // Transaction<Committed> - txn consumed
//!
//! // txn.get(b"key");                // COMPILE ERROR: txn was moved
//! // txn.commit();                   // COMPILE ERROR: txn was moved
//!
//! // committed.get(b"key");          // COMPILE ERROR: no get() on Committed
//! ```
//!
//! ## Advantages
//!
//! 1. **Zero runtime overhead**: All checks happen at compile time
//! 2. **Self-documenting API**: The type signatures show valid operations
//! 3. **No runtime errors**: Invalid sequences are impossible to write
//! 4. **Better IDE support**: Autocomplete only shows valid methods

use std::marker::PhantomData;
use std::sync::Arc;
use crate::error::Result;

// =============================================================================
// Transaction States (Marker Types)
// =============================================================================

/// Marker trait for all transaction states
pub trait TransactionState: private::Sealed {}

/// State: Transaction is active and can perform operations
#[derive(Debug)]
pub struct Active;
impl TransactionState for Active {}

/// State: Transaction has been committed successfully
#[derive(Debug)]
pub struct Committed;
impl TransactionState for Committed {}

/// State: Transaction has been aborted/rolled back
#[derive(Debug)]
pub struct Aborted;
impl TransactionState for Aborted {}

/// State: Transaction is preparing for 2PC commit
#[derive(Debug)]
pub struct Preparing;
impl TransactionState for Preparing {}

/// State: Transaction is prepared and waiting for final commit (2PC)
#[derive(Debug)]
pub struct Prepared;
impl TransactionState for Prepared {}

// Seal the TransactionState trait to prevent external implementations
mod private {
    pub trait Sealed {}
    impl Sealed for super::Active {}
    impl Sealed for super::Committed {}
    impl Sealed for super::Aborted {}
    impl Sealed for super::Preparing {}
    impl Sealed for super::Prepared {}
}

// =============================================================================
// Transaction Mode Markers
// =============================================================================

/// Marker trait for transaction modes
pub trait TransactionMode: private2::Sealed {}

/// Read-only transaction mode
#[derive(Debug)]
pub struct ReadOnly;
impl TransactionMode for ReadOnly {}

/// Read-write transaction mode
#[derive(Debug)]
pub struct ReadWrite;
impl TransactionMode for ReadWrite {}

/// Write-only transaction mode
#[derive(Debug)]
pub struct WriteOnly;
impl TransactionMode for WriteOnly {}

mod private2 {
    pub trait Sealed {}
    impl Sealed for super::ReadOnly {}
    impl Sealed for super::ReadWrite {}
    impl Sealed for super::WriteOnly {}
}

// =============================================================================
// Core Transaction Type
// =============================================================================

/// Type-safe transaction with compile-time state tracking
///
/// The `State` type parameter tracks the transaction lifecycle:
/// - `Active`: Can perform reads, writes, commit, or abort
/// - `Committed`: Final state after successful commit (no operations allowed)
/// - `Aborted`: Final state after abort (no operations allowed)
/// - `Preparing`: In 2PC prepare phase
/// - `Prepared`: Ready for 2PC final commit
///
/// The `Mode` type parameter tracks the transaction mode:
/// - `ReadOnly`: Only read operations allowed
/// - `ReadWrite`: Both read and write operations allowed
/// - `WriteOnly`: Only write operations allowed (skips read tracking)
pub struct Transaction<State: TransactionState, Mode: TransactionMode = ReadWrite> {
    /// Internal transaction handle
    inner: TransactionInner,
    /// State marker (zero-sized)
    _state: PhantomData<State>,
    /// Mode marker (zero-sized)
    _mode: PhantomData<Mode>,
}

/// Internal transaction data (shared across states)
struct TransactionInner {
    /// Transaction ID
    txn_id: u64,
    /// Snapshot timestamp for reads
    snapshot_ts: u64,
    /// Commit timestamp (set on commit)
    commit_ts: Option<u64>,
    /// Write buffer for pending writes
    write_buffer: Vec<WriteEntry>,
    /// Read set for SSI validation
    read_set: Vec<Vec<u8>>,
    /// Backend storage reference
    storage: Arc<dyn TransactionStorage>,
}

/// A buffered write entry
struct WriteEntry {
    key: Vec<u8>,
    value: Option<Vec<u8>>, // None = delete
}

// =============================================================================
// Transaction Storage Trait (Backend Interface)
// =============================================================================

/// Backend storage interface for transactions
///
/// This trait is implemented by the actual storage layer (DurableStorage, etc.)
pub trait TransactionStorage: Send + Sync {
    /// Get a value at the given snapshot timestamp
    fn get(&self, key: &[u8], snapshot_ts: u64, txn_id: u64) -> Result<Option<Vec<u8>>>;
    
    /// Apply writes and commit the transaction
    fn commit(
        &self,
        txn_id: u64,
        writes: Vec<(Vec<u8>, Option<Vec<u8>>)>,
        read_set: Vec<Vec<u8>>,
    ) -> Result<u64>;
    
    /// Abort the transaction
    fn abort(&self, txn_id: u64) -> Result<()>;
    
    /// Prepare for 2PC (returns prepare result)
    fn prepare(&self, txn_id: u64) -> Result<()>;
    
    /// Finalize 2PC commit
    fn finalize_commit(&self, txn_id: u64) -> Result<u64>;
    
    /// Allocate a new transaction ID
    fn allocate_txn_id(&self) -> u64;
    
    /// Get current snapshot timestamp
    fn snapshot_ts(&self) -> u64;
}

// =============================================================================
// Transaction Builder
// =============================================================================

/// Builder for creating transactions with specific options
pub struct TransactionBuilder {
    storage: Arc<dyn TransactionStorage>,
    snapshot_ts: Option<u64>,
}

impl TransactionBuilder {
    /// Create a new transaction builder
    pub fn new(storage: Arc<dyn TransactionStorage>) -> Self {
        Self {
            storage,
            snapshot_ts: None,
        }
    }

    /// Set a specific snapshot timestamp
    pub fn with_snapshot_ts(mut self, ts: u64) -> Self {
        self.snapshot_ts = Some(ts);
        self
    }

    /// Build a read-only transaction
    pub fn read_only(self) -> Transaction<Active, ReadOnly> {
        let txn_id = self.storage.allocate_txn_id();
        let snapshot_ts = self.snapshot_ts.unwrap_or_else(|| self.storage.snapshot_ts());
        
        Transaction {
            inner: TransactionInner {
                txn_id,
                snapshot_ts,
                commit_ts: None,
                write_buffer: Vec::new(),
                read_set: Vec::new(),
                storage: self.storage,
            },
            _state: PhantomData,
            _mode: PhantomData,
        }
    }

    /// Build a read-write transaction
    pub fn read_write(self) -> Transaction<Active, ReadWrite> {
        let txn_id = self.storage.allocate_txn_id();
        let snapshot_ts = self.snapshot_ts.unwrap_or_else(|| self.storage.snapshot_ts());
        
        Transaction {
            inner: TransactionInner {
                txn_id,
                snapshot_ts,
                commit_ts: None,
                write_buffer: Vec::with_capacity(16),
                read_set: Vec::with_capacity(32),
                storage: self.storage,
            },
            _state: PhantomData,
            _mode: PhantomData,
        }
    }

    /// Build a write-only transaction
    pub fn write_only(self) -> Transaction<Active, WriteOnly> {
        let txn_id = self.storage.allocate_txn_id();
        let snapshot_ts = self.snapshot_ts.unwrap_or_else(|| self.storage.snapshot_ts());
        
        Transaction {
            inner: TransactionInner {
                txn_id,
                snapshot_ts,
                commit_ts: None,
                write_buffer: Vec::with_capacity(16),
                read_set: Vec::new(), // Not tracked for WriteOnly
                storage: self.storage,
            },
            _state: PhantomData,
            _mode: PhantomData,
        }
    }
}

// =============================================================================
// Active Transaction Operations
// =============================================================================

impl<Mode: TransactionMode> Transaction<Active, Mode> {
    /// Get the transaction ID
    pub fn txn_id(&self) -> u64 {
        self.inner.txn_id
    }

    /// Get the snapshot timestamp
    pub fn snapshot_ts(&self) -> u64 {
        self.inner.snapshot_ts
    }

    /// Abort the transaction
    ///
    /// Consumes the transaction and returns an Aborted transaction.
    /// No further operations are possible after abort.
    pub fn abort(self) -> Result<Transaction<Aborted, Mode>> {
        self.inner.storage.abort(self.inner.txn_id)?;
        
        Ok(Transaction {
            inner: self.inner,
            _state: PhantomData,
            _mode: PhantomData,
        })
    }
}

// Read operations for ReadOnly and ReadWrite modes
impl Transaction<Active, ReadOnly> {
    /// Get a value by key (read-only transaction)
    ///
    /// Note: Read-only transactions don't track reads for SSI
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.storage.get(key, self.inner.snapshot_ts, self.inner.txn_id)
    }

    /// Commit the read-only transaction
    ///
    /// For read-only transactions, this is essentially a no-op,
    /// but it properly transitions the state.
    pub fn commit(self) -> Result<Transaction<Committed, ReadOnly>> {
        // Read-only transactions don't need to go through full commit
        Ok(Transaction {
            inner: TransactionInner {
                commit_ts: Some(self.inner.snapshot_ts),
                ..self.inner
            },
            _state: PhantomData,
            _mode: PhantomData,
        })
    }
}

impl Transaction<Active, ReadWrite> {
    /// Get a value by key (read-write transaction)
    ///
    /// Tracks the read for SSI validation
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Track read for SSI
        self.inner.read_set.push(key.to_vec());
        
        // Check write buffer first (read-your-writes)
        for entry in self.inner.write_buffer.iter().rev() {
            if entry.key == key {
                return Ok(entry.value.clone());
            }
        }
        
        // Fall through to storage
        self.inner.storage.get(key, self.inner.snapshot_ts, self.inner.txn_id)
    }

    /// Put a key-value pair
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.write_buffer.push(WriteEntry {
            key: key.to_vec(),
            value: Some(value.to_vec()),
        });
        Ok(())
    }

    /// Delete a key
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.inner.write_buffer.push(WriteEntry {
            key: key.to_vec(),
            value: None,
        });
        Ok(())
    }

    /// Commit the transaction
    ///
    /// Consumes the transaction and returns a Committed transaction.
    /// The committed transaction contains the commit timestamp.
    pub fn commit(self) -> Result<Transaction<Committed, ReadWrite>> {
        let writes: Vec<_> = self.inner.write_buffer
            .into_iter()
            .map(|e| (e.key, e.value))
            .collect();
        
        let commit_ts = self.inner.storage.commit(
            self.inner.txn_id,
            writes,
            self.inner.read_set.clone(),
        )?;
        
        Ok(Transaction {
            inner: TransactionInner {
                commit_ts: Some(commit_ts),
                txn_id: self.inner.txn_id,
                snapshot_ts: self.inner.snapshot_ts,
                write_buffer: Vec::new(),
                read_set: self.inner.read_set,
                storage: self.inner.storage,
            },
            _state: PhantomData,
            _mode: PhantomData,
        })
    }

    /// Prepare for two-phase commit
    ///
    /// Transitions to Preparing state.
    pub fn prepare(self) -> Result<Transaction<Prepared, ReadWrite>> {
        self.inner.storage.prepare(self.inner.txn_id)?;
        
        Ok(Transaction {
            inner: self.inner,
            _state: PhantomData,
            _mode: PhantomData,
        })
    }
}

impl Transaction<Active, WriteOnly> {
    /// Put a key-value pair (write-only transaction)
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.write_buffer.push(WriteEntry {
            key: key.to_vec(),
            value: Some(value.to_vec()),
        });
        Ok(())
    }

    /// Delete a key
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.inner.write_buffer.push(WriteEntry {
            key: key.to_vec(),
            value: None,
        });
        Ok(())
    }

    /// Commit the transaction
    pub fn commit(self) -> Result<Transaction<Committed, WriteOnly>> {
        let writes: Vec<_> = self.inner.write_buffer
            .into_iter()
            .map(|e| (e.key, e.value))
            .collect();
        
        // WriteOnly doesn't pass read_set (empty)
        let commit_ts = self.inner.storage.commit(
            self.inner.txn_id,
            writes,
            Vec::new(),
        )?;
        
        Ok(Transaction {
            inner: TransactionInner {
                commit_ts: Some(commit_ts),
                txn_id: self.inner.txn_id,
                snapshot_ts: self.inner.snapshot_ts,
                write_buffer: Vec::new(),
                read_set: Vec::new(),
                storage: self.inner.storage,
            },
            _state: PhantomData,
            _mode: PhantomData,
        })
    }
}

// =============================================================================
// Prepared Transaction (2PC)
// =============================================================================

impl Transaction<Prepared, ReadWrite> {
    /// Finalize the 2PC commit
    pub fn finalize(self) -> Result<Transaction<Committed, ReadWrite>> {
        let commit_ts = self.inner.storage.finalize_commit(self.inner.txn_id)?;
        
        Ok(Transaction {
            inner: TransactionInner {
                commit_ts: Some(commit_ts),
                ..self.inner
            },
            _state: PhantomData,
            _mode: PhantomData,
        })
    }

    /// Abort the prepared transaction
    pub fn abort(self) -> Result<Transaction<Aborted, ReadWrite>> {
        self.inner.storage.abort(self.inner.txn_id)?;
        
        Ok(Transaction {
            inner: self.inner,
            _state: PhantomData,
            _mode: PhantomData,
        })
    }
}

// =============================================================================
// Committed Transaction
// =============================================================================

impl<Mode: TransactionMode> Transaction<Committed, Mode> {
    /// Get the commit timestamp
    pub fn commit_ts(&self) -> Option<u64> {
        self.inner.commit_ts
    }

    /// Get the transaction ID
    pub fn txn_id(&self) -> u64 {
        self.inner.txn_id
    }
}

// =============================================================================
// Aborted Transaction
// =============================================================================

impl<Mode: TransactionMode> Transaction<Aborted, Mode> {
    /// Get the transaction ID
    pub fn txn_id(&self) -> u64 {
        self.inner.txn_id
    }
}

// =============================================================================
// Debug Implementations
// =============================================================================

impl<State: TransactionState, Mode: TransactionMode> std::fmt::Debug for Transaction<State, Mode> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transaction")
            .field("txn_id", &self.inner.txn_id)
            .field("snapshot_ts", &self.inner.snapshot_ts)
            .field("commit_ts", &self.inner.commit_ts)
            .field("write_count", &self.inner.write_buffer.len())
            .field("read_count", &self.inner.read_set.len())
            .finish()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Begin a new read-write transaction
pub fn begin_transaction(storage: Arc<dyn TransactionStorage>) -> Transaction<Active, ReadWrite> {
    TransactionBuilder::new(storage).read_write()
}

/// Begin a new read-only transaction
pub fn begin_read_only(storage: Arc<dyn TransactionStorage>) -> Transaction<Active, ReadOnly> {
    TransactionBuilder::new(storage).read_only()
}

/// Begin a new write-only transaction
pub fn begin_write_only(storage: Arc<dyn TransactionStorage>) -> Transaction<Active, WriteOnly> {
    TransactionBuilder::new(storage).write_only()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use parking_lot::RwLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Mock storage implementation for testing
    struct MockStorage {
        data: RwLock<HashMap<Vec<u8>, Vec<u8>>>,
        txn_counter: AtomicU64,
        ts_counter: AtomicU64,
    }

    impl MockStorage {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                data: RwLock::new(HashMap::new()),
                txn_counter: AtomicU64::new(1),
                ts_counter: AtomicU64::new(1),
            })
        }
    }

    impl TransactionStorage for MockStorage {
        fn get(&self, key: &[u8], _snapshot_ts: u64, _txn_id: u64) -> Result<Option<Vec<u8>>> {
            Ok(self.data.read().get(key).cloned())
        }

        fn commit(
            &self,
            _txn_id: u64,
            writes: Vec<(Vec<u8>, Option<Vec<u8>>)>,
            _read_set: Vec<Vec<u8>>,
        ) -> Result<u64> {
            let mut data = self.data.write();
            for (key, value) in writes {
                match value {
                    Some(v) => data.insert(key, v),
                    None => data.remove(&key),
                };
            }
            Ok(self.ts_counter.fetch_add(1, Ordering::SeqCst))
        }

        fn abort(&self, _txn_id: u64) -> Result<()> {
            Ok(())
        }

        fn prepare(&self, _txn_id: u64) -> Result<()> {
            Ok(())
        }

        fn finalize_commit(&self, _txn_id: u64) -> Result<u64> {
            Ok(self.ts_counter.fetch_add(1, Ordering::SeqCst))
        }

        fn allocate_txn_id(&self) -> u64 {
            self.txn_counter.fetch_add(1, Ordering::SeqCst)
        }

        fn snapshot_ts(&self) -> u64 {
            self.ts_counter.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn test_read_write_transaction() {
        let storage = MockStorage::new();
        let mut txn = begin_transaction(storage);

        txn.put(b"key1", b"value1").unwrap();
        
        // Read-your-writes
        let val = txn.get(b"key1").unwrap();
        assert_eq!(val, Some(b"value1".to_vec()));

        let committed = txn.commit().unwrap();
        assert!(committed.commit_ts().is_some());
    }

    #[test]
    fn test_read_only_transaction() {
        let storage = MockStorage::new();
        
        // First, write some data
        {
            let mut write_txn = begin_transaction(storage.clone());
            write_txn.put(b"key", b"value").unwrap();
            write_txn.commit().unwrap();
        }
        
        // Now read it with read-only transaction
        let read_txn = begin_read_only(storage);
        let val = read_txn.get(b"key").unwrap();
        assert_eq!(val, Some(b"value".to_vec()));
        
        let committed = read_txn.commit().unwrap();
        assert!(committed.commit_ts().is_some());
    }

    #[test]
    fn test_abort_transaction() {
        let storage = MockStorage::new();
        let mut txn = begin_transaction(storage.clone());

        txn.put(b"key", b"value").unwrap();
        
        let aborted = txn.abort().unwrap();
        assert!(aborted.txn_id() > 0);

        // Verify data wasn't committed
        let read_txn = begin_read_only(storage);
        assert_eq!(read_txn.get(b"key").unwrap(), None);
    }

    #[test]
    fn test_write_only_transaction() {
        let storage = MockStorage::new();
        let mut txn = begin_write_only(storage.clone());

        txn.put(b"key1", b"value1").unwrap();
        txn.put(b"key2", b"value2").unwrap();
        txn.delete(b"key1").unwrap();
        
        let committed = txn.commit().unwrap();
        assert!(committed.commit_ts().is_some());

        // Verify: key1 deleted, key2 exists
        let read_txn = begin_read_only(storage);
        assert_eq!(read_txn.get(b"key1").unwrap(), None);
        assert_eq!(read_txn.get(b"key2").unwrap(), Some(b"value2".to_vec()));
    }

    // This test demonstrates that the following code would NOT compile:
    // ```
    // let txn = begin_transaction(storage);
    // let committed = txn.commit().unwrap();
    // txn.get(b"key"); // COMPILE ERROR: txn was moved
    // committed.put(b"key", b"value"); // COMPILE ERROR: no put() on Committed
    // ```
}
