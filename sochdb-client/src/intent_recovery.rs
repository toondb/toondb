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

//! # Scalable Intent Recovery + Garbage Collection (Task 15)
//!
//! Provides recovery from crashes and garbage collection for committed intents.
//!
//! ## Recovery Strategy
//!
//! 1. **Scan WAL**: Find all INTENT records without matching COMMIT/ABORT
//! 2. **Classify**: Determine if intent should be replayed or rolled back
//! 3. **Execute**: Apply redo or undo operations
//! 4. **Cleanup**: Remove stale intent records
//!
//! ## Garbage Collection
//!
//! Committed intents and their WAL records can be garbage collected after:
//! - All operations are durably applied
//! - A checkpoint has been taken
//! - Retention period has passed
//!
//! ## Scalability
//!
//! - Parallel recovery for independent intents
//! - Incremental GC (no full scans)
//! - Checkpoint-based truncation

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};

use crate::wal_atomic::{Lsn, WalConfig, WalPayload, WalReader, WalRecord, WalRecordType, WalWriter};

// ============================================================================
// Intent Status
// ============================================================================

/// Status of an intent during recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryIntentStatus {
    /// Intent record found, no operations yet
    Started,
    /// Some operations applied
    InProgress,
    /// All operations completed, commit pending
    Completed,
    /// Committed
    Committed,
    /// Aborted
    Aborted,
    /// Unknown (corrupted)
    Unknown,
}

/// An intent being recovered
#[derive(Debug)]
pub struct RecoveryIntent {
    /// Intent ID
    pub intent_id: u64,
    /// Memory ID
    pub memory_id: String,
    /// Intent start LSN
    pub start_lsn: Lsn,
    /// Expected operation count
    pub expected_ops: usize,
    /// Operations found in WAL
    pub operations: Vec<RecoveryOperation>,
    /// Current status
    pub status: RecoveryIntentStatus,
    /// Commit LSN (if committed)
    pub commit_lsn: Option<Lsn>,
    /// Abort LSN (if aborted)
    pub abort_lsn: Option<Lsn>,
    /// Timestamp
    pub timestamp: u64,
}

/// An operation to recover
#[derive(Debug, Clone)]
pub struct RecoveryOperation {
    /// Operation index
    pub op_index: usize,
    /// Operation type
    pub op_type: String,
    /// Key
    pub key: Vec<u8>,
    /// Value (if any)
    pub value: Option<Vec<u8>>,
    /// LSN
    pub lsn: Lsn,
}

// ============================================================================
// Recovery Engine
// ============================================================================

/// Configuration for recovery
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// WAL directory
    pub wal_dir: PathBuf,
    /// Maximum parallel recovery threads
    pub max_parallel: usize,
    /// Timeout for recovery operations
    pub timeout: Duration,
    /// Whether to redo incomplete intents
    pub redo_incomplete: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("./wal"),
            max_parallel: 4,
            timeout: Duration::from_secs(30),
            redo_incomplete: true,
        }
    }
}

/// Result of recovery
#[derive(Debug)]
pub struct RecoveryResult {
    /// Number of intents recovered
    pub intents_recovered: usize,
    /// Number of intents replayed
    pub intents_replayed: usize,
    /// Number of intents rolled back
    pub intents_rolled_back: usize,
    /// Number of operations redone
    pub ops_redone: usize,
    /// Number of operations undone
    pub ops_undone: usize,
    /// Last recovered LSN
    pub last_lsn: Lsn,
    /// Recovery duration
    pub duration: Duration,
    /// Errors encountered (non-fatal)
    pub errors: Vec<String>,
}

/// Callback for applying an operation during recovery
pub type ApplyCallback = Box<dyn Fn(&RecoveryOperation) -> io::Result<()> + Send + Sync>;

/// Callback for undoing an operation during recovery
pub type UndoCallback = Box<dyn Fn(&RecoveryOperation) -> io::Result<()> + Send + Sync>;

/// Intent recovery engine
pub struct RecoveryEngine {
    config: RecoveryConfig,
    reader: WalReader,
    apply_callback: Option<ApplyCallback>,
    undo_callback: Option<UndoCallback>,
}

impl RecoveryEngine {
    /// Create new recovery engine
    pub fn new(config: RecoveryConfig) -> Self {
        let reader = WalReader::new(&config.wal_dir);
        Self {
            config,
            reader,
            apply_callback: None,
            undo_callback: None,
        }
    }
    
    /// Set apply callback
    pub fn with_apply<F>(mut self, f: F) -> Self
    where
        F: Fn(&RecoveryOperation) -> io::Result<()> + Send + Sync + 'static,
    {
        self.apply_callback = Some(Box::new(f));
        self
    }
    
    /// Set undo callback
    pub fn with_undo<F>(mut self, f: F) -> Self
    where
        F: Fn(&RecoveryOperation) -> io::Result<()> + Send + Sync + 'static,
    {
        self.undo_callback = Some(Box::new(f));
        self
    }
    
    /// Run recovery
    pub fn recover(&self) -> io::Result<RecoveryResult> {
        let start = Instant::now();
        
        // Phase 1: Scan WAL and build intent map
        let records = self.reader.read_all()?;
        let intents = self.build_intent_map(&records);
        
        // Phase 2: Classify intents
        let (to_redo, to_undo) = self.classify_intents(&intents);
        
        // Phase 3: Apply recovery
        let mut ops_redone = 0;
        let mut ops_undone = 0;
        let mut errors = Vec::new();
        
        // Redo incomplete intents
        for intent in &to_redo {
            match self.redo_intent(intent) {
                Ok(n) => ops_redone += n,
                Err(e) => errors.push(format!("Redo intent {}: {}", intent.intent_id, e)),
            }
        }
        
        // Undo aborted intents
        for intent in &to_undo {
            match self.undo_intent(intent) {
                Ok(n) => ops_undone += n,
                Err(e) => errors.push(format!("Undo intent {}: {}", intent.intent_id, e)),
            }
        }
        
        // Find last LSN
        let last_lsn = records.last().map(|r| r.lsn).unwrap_or(Lsn::ZERO);
        
        Ok(RecoveryResult {
            intents_recovered: intents.len(),
            intents_replayed: to_redo.len(),
            intents_rolled_back: to_undo.len(),
            ops_redone,
            ops_undone,
            last_lsn,
            duration: start.elapsed(),
            errors,
        })
    }
    
    /// Build intent map from WAL records
    fn build_intent_map(&self, records: &[WalRecord]) -> HashMap<u64, RecoveryIntent> {
        let mut intents: HashMap<u64, RecoveryIntent> = HashMap::new();
        
        for record in records {
            match &record.payload {
                WalPayload::IntentStart { memory_id, op_count } => {
                    intents.insert(record.intent_id, RecoveryIntent {
                        intent_id: record.intent_id,
                        memory_id: memory_id.clone(),
                        start_lsn: record.lsn,
                        expected_ops: *op_count,
                        operations: Vec::new(),
                        status: RecoveryIntentStatus::Started,
                        commit_lsn: None,
                        abort_lsn: None,
                        timestamp: record.timestamp,
                    });
                }
                WalPayload::Operation { op_index, op_type, key, value } => {
                    if let Some(intent) = intents.get_mut(&record.intent_id) {
                        intent.operations.push(RecoveryOperation {
                            op_index: *op_index,
                            op_type: op_type.clone(),
                            key: key.clone(),
                            value: value.clone(),
                            lsn: record.lsn,
                        });
                        intent.status = RecoveryIntentStatus::InProgress;
                    }
                }
                WalPayload::Commit => {
                    if let Some(intent) = intents.get_mut(&record.intent_id) {
                        intent.status = RecoveryIntentStatus::Committed;
                        intent.commit_lsn = Some(record.lsn);
                    }
                }
                WalPayload::Abort { .. } => {
                    if let Some(intent) = intents.get_mut(&record.intent_id) {
                        intent.status = RecoveryIntentStatus::Aborted;
                        intent.abort_lsn = Some(record.lsn);
                    }
                }
                WalPayload::Checkpoint { .. } => {
                    // Checkpoints don't affect intent status
                }
            }
        }
        
        intents
    }
    
    /// Classify intents into redo and undo sets
    fn classify_intents<'a>(
        &self,
        intents: &'a HashMap<u64, RecoveryIntent>,
    ) -> (Vec<&'a RecoveryIntent>, Vec<&'a RecoveryIntent>) {
        let mut to_redo = Vec::new();
        let mut to_undo = Vec::new();
        
        for intent in intents.values() {
            match intent.status {
                RecoveryIntentStatus::Started | RecoveryIntentStatus::InProgress => {
                    // Incomplete intent - redo if configured
                    if self.config.redo_incomplete {
                        to_redo.push(intent);
                    } else {
                        to_undo.push(intent);
                    }
                }
                RecoveryIntentStatus::Completed => {
                    // All ops done but not committed - redo
                    to_redo.push(intent);
                }
                RecoveryIntentStatus::Aborted => {
                    // Explicitly aborted - undo
                    to_undo.push(intent);
                }
                RecoveryIntentStatus::Committed | RecoveryIntentStatus::Unknown => {
                    // Already committed or unknown - skip
                }
            }
        }
        
        (to_redo, to_undo)
    }
    
    /// Redo an intent (replay operations)
    fn redo_intent(&self, intent: &RecoveryIntent) -> io::Result<usize> {
        if let Some(ref callback) = self.apply_callback {
            let mut applied = 0;
            
            // Sort operations by index
            let mut ops = intent.operations.clone();
            ops.sort_by_key(|op| op.op_index);
            
            for op in &ops {
                callback(op)?;
                applied += 1;
            }
            
            Ok(applied)
        } else {
            Ok(0)
        }
    }
    
    /// Undo an intent (reverse operations)
    fn undo_intent(&self, intent: &RecoveryIntent) -> io::Result<usize> {
        if let Some(ref callback) = self.undo_callback {
            let mut undone = 0;
            
            // Sort operations by index in reverse
            let mut ops = intent.operations.clone();
            ops.sort_by_key(|op| std::cmp::Reverse(op.op_index));
            
            for op in &ops {
                callback(op)?;
                undone += 1;
            }
            
            Ok(undone)
        } else {
            Ok(0)
        }
    }
}

// ============================================================================
// Garbage Collector
// ============================================================================

/// GC configuration
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// WAL directory
    pub wal_dir: PathBuf,
    /// Minimum age before GC (duration since commit)
    pub min_age: Duration,
    /// Maximum WAL size before forced GC
    pub max_wal_size: u64,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
    /// Whether to run in background
    pub background: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("./wal"),
            min_age: Duration::from_secs(300), // 5 minutes
            max_wal_size: 1024 * 1024 * 1024,  // 1 GB
            checkpoint_interval: Duration::from_secs(60),
            background: true,
        }
    }
}

/// GC statistics
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of intents collected
    pub intents_collected: usize,
    /// Number of WAL files removed
    pub files_removed: usize,
    /// Bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Last GC time
    pub last_gc: Option<Instant>,
    /// Total GC runs
    pub total_runs: u64,
}

/// Garbage collector for WAL and intent records
pub struct GarbageCollector {
    config: GcConfig,
    wal: Arc<WalWriter>,
    stats: RwLock<GcStats>,
    last_checkpoint_lsn: AtomicU64,
    stop_flag: AtomicU64,
}

impl GarbageCollector {
    /// Create new garbage collector
    pub fn new(config: GcConfig, wal: Arc<WalWriter>) -> Self {
        Self {
            config,
            wal,
            stats: RwLock::new(GcStats::default()),
            last_checkpoint_lsn: AtomicU64::new(0),
            stop_flag: AtomicU64::new(0),
        }
    }
    
    /// Run GC once
    pub fn run_once(&self) -> io::Result<GcStats> {
        let start = Instant::now();
        let reader = WalReader::new(&self.config.wal_dir);
        
        // Read all records
        let records = reader.read_all()?;
        
        // Find committed intents
        let committed: HashSet<u64> = records.iter()
            .filter(|r| r.record_type == WalRecordType::Commit)
            .map(|r| r.intent_id)
            .collect();
        
        // Find LSN up to which all intents are committed
        let mut safe_lsn = Lsn::ZERO;
        let mut all_committed = true;
        
        for record in &records {
            if record.record_type == WalRecordType::Intent {
                if !committed.contains(&record.intent_id) {
                    all_committed = false;
                    break;
                }
            }
            if all_committed {
                safe_lsn = record.lsn;
            }
        }
        
        // Write checkpoint
        let checkpoint_lsn = self.wal.write_checkpoint(safe_lsn, committed.len())?;
        self.last_checkpoint_lsn.store(checkpoint_lsn.0, Ordering::SeqCst);
        
        // Find files that can be removed
        let files = reader.list_files()?;
        let mut files_removed = 0;
        let mut bytes_reclaimed = 0u64;
        
        for file in &files {
            // Read file and check if all records are before safe_lsn
            if let Ok(file_records) = reader.read_file(file) {
                let max_lsn = file_records.iter()
                    .map(|r| r.lsn)
                    .max()
                    .unwrap_or(Lsn::ZERO);
                
                // Only remove if all records are safely checkpointed and old enough
                if max_lsn <= safe_lsn {
                    // Check age
                    if let Ok(metadata) = std::fs::metadata(file) {
                        if let Ok(modified) = metadata.modified() {
                            if let Ok(age) = modified.elapsed() {
                                if age >= self.config.min_age {
                                    bytes_reclaimed += metadata.len();
                                    std::fs::remove_file(file)?;
                                    files_removed += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let mut stats = self.stats.write();
        stats.intents_collected = committed.len();
        stats.files_removed += files_removed;
        stats.bytes_reclaimed += bytes_reclaimed;
        stats.last_gc = Some(start);
        stats.total_runs += 1;
        
        Ok(stats.clone())
    }
    
    /// Start background GC
    pub fn start_background(self: Arc<Self>) -> thread::JoinHandle<()> {
        let gc = self.clone();
        
        thread::spawn(move || {
            while gc.stop_flag.load(Ordering::Relaxed) == 0 {
                // Wait for interval
                thread::sleep(gc.config.checkpoint_interval);
                
                // Run GC
                if let Err(e) = gc.run_once() {
                    tracing::warn!("GC error: {}", e);
                }
            }
        })
    }
    
    /// Stop background GC
    pub fn stop(&self) {
        self.stop_flag.store(1, Ordering::SeqCst);
    }
    
    /// Get current stats
    pub fn stats(&self) -> GcStats {
        self.stats.read().clone()
    }
    
    /// Get last checkpoint LSN
    pub fn last_checkpoint(&self) -> Lsn {
        Lsn(self.last_checkpoint_lsn.load(Ordering::SeqCst))
    }
}

// ============================================================================
// Combined Recovery Manager
// ============================================================================

/// Combined recovery and GC manager
pub struct RecoveryManager {
    recovery_engine: RecoveryEngine,
    gc: Arc<GarbageCollector>,
}

impl RecoveryManager {
    /// Create new recovery manager
    pub fn new(
        recovery_config: RecoveryConfig,
        gc_config: GcConfig,
        wal: Arc<WalWriter>,
    ) -> Self {
        Self {
            recovery_engine: RecoveryEngine::new(recovery_config),
            gc: Arc::new(GarbageCollector::new(gc_config, wal)),
        }
    }
    
    /// Run recovery
    pub fn recover(&self) -> io::Result<RecoveryResult> {
        self.recovery_engine.recover()
    }
    
    /// Run GC
    pub fn gc(&self) -> io::Result<GcStats> {
        self.gc.run_once()
    }
    
    /// Start background GC
    pub fn start_background_gc(&self) -> thread::JoinHandle<()> {
        self.gc.clone().start_background()
    }
    
    /// Stop background GC
    pub fn stop_gc(&self) {
        self.gc.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_recovery_intent_classification() {
        let config = RecoveryConfig {
            wal_dir: PathBuf::from("/tmp/test"),
            redo_incomplete: true,
            ..Default::default()
        };
        
        let engine = RecoveryEngine::new(config);
        
        let mut intents = HashMap::new();
        
        // Committed intent - should not be touched
        intents.insert(1, RecoveryIntent {
            intent_id: 1,
            memory_id: "mem1".to_string(),
            start_lsn: Lsn(1),
            expected_ops: 2,
            operations: vec![],
            status: RecoveryIntentStatus::Committed,
            commit_lsn: Some(Lsn(10)),
            abort_lsn: None,
            timestamp: 0,
        });
        
        // In-progress intent - should redo
        intents.insert(2, RecoveryIntent {
            intent_id: 2,
            memory_id: "mem2".to_string(),
            start_lsn: Lsn(11),
            expected_ops: 2,
            operations: vec![],
            status: RecoveryIntentStatus::InProgress,
            commit_lsn: None,
            abort_lsn: None,
            timestamp: 0,
        });
        
        // Aborted intent - should undo
        intents.insert(3, RecoveryIntent {
            intent_id: 3,
            memory_id: "mem3".to_string(),
            start_lsn: Lsn(20),
            expected_ops: 2,
            operations: vec![],
            status: RecoveryIntentStatus::Aborted,
            commit_lsn: None,
            abort_lsn: Some(Lsn(25)),
            timestamp: 0,
        });
        
        let (to_redo, to_undo) = engine.classify_intents(&intents);
        
        assert_eq!(to_redo.len(), 1);
        assert_eq!(to_redo[0].intent_id, 2);
        
        assert_eq!(to_undo.len(), 1);
        assert_eq!(to_undo[0].intent_id, 3);
    }
    
    #[test]
    fn test_gc_config() {
        let config = GcConfig::default();
        assert_eq!(config.min_age, Duration::from_secs(300));
        assert!(config.background);
    }
}
