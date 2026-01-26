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

//! Slot-Based Columnar Row Storage (PAX/Hybrid Format) - Recommendation 1
//!
//! ## Problem
//!
//! Current implementation stores each row as HashMap<String, SochValue>, which incurs:
//! - 48 bytes minimum HashMap overhead per row
//! - 24 bytes per String key (pointer + length + capacity)
//! - 16-32 bytes per SochValue (enum tag + union)
//! - Heap fragmentation from individual allocations
//!
//! For a 4-column row: 48 + 4×(24 + 24) = 240 bytes vs SQLite's ~50 bytes
//!
//! ## Solution
//!
//! Slot-based row format: fixed header + variable payload
//! - Row ID (8 bytes)
//! - Null bitmap (2 bytes for 16 columns)
//! - Slot count (1 byte)
//! - Flags (1 byte for deleted, MVCC markers)
//! - MVCC timestamps (16 bytes)
//! - Slot array: [offset: u16, len: u16] per column
//! - Data follows slots contiguously
//!
//! ## Performance Analysis
//!
//! | Component    | Current   | Proposed       | Savings |
//! |--------------|-----------|----------------|---------|
//! | Header       | 48 (HashMap)| 28 (SlotRow)  | 42%     |
//! | 4 String keys| 96        | 0 (schema-indexed)| 100% |
//! | 4 SochValues | 128       | 32 (raw bytes) | 75%     |
//! | Total        | 272 bytes | 60 bytes       | 78%     |
//!
//! Cache analysis: With 64KB pages and 60-byte rows, we get 1,092 rows per page
//! vs current ~240 rows per page with HashMap approach.

/// Minimum slot row header size (without slots)
pub const SLOT_ROW_HEADER_SIZE: usize = 28;

/// Maximum columns supported by null bitmap (u16)
pub const MAX_SLOT_COLUMNS: usize = 16;

/// Slot row flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SlotRowFlags {
    /// Normal row
    Normal = 0,
    /// Deleted (tombstone)
    Deleted = 1,
    /// Uncommitted (MVCC)
    Uncommitted = 2,
    /// Compressed
    Compressed = 4,
}

impl SlotRowFlags {
    pub fn is_deleted(&self) -> bool {
        (*self as u8) & 1 != 0
    }

    pub fn is_uncommitted(&self) -> bool {
        (*self as u8) & 2 != 0
    }

    pub fn is_compressed(&self) -> bool {
        (*self as u8) & 4 != 0
    }
}

/// Slot entry: offset and length for a column value
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct Slot {
    /// Offset from data start (u16 allows 64KB max row size)
    pub offset: u16,
    /// Length of value in bytes
    pub len: u16,
}

impl Slot {
    pub const SIZE: usize = 4;

    #[inline]
    pub fn new(offset: u16, len: u16) -> Self {
        Self { offset, len }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        self.len == 0 && self.offset == 0xFFFF
    }

    #[inline]
    pub fn null() -> Self {
        Self { offset: 0xFFFF, len: 0 }
    }
}

/// Slot-based row format with O(1) field access
///
/// ## Memory Layout
///
/// ```text
/// ┌───────────────────────────────────────────────────────┐
/// │ Header (28 bytes)                                      │
/// │  row_id: u64           (8 bytes)                       │
/// │  null_bitmap: u16      (2 bytes)                       │
/// │  slot_count: u8        (1 byte)                        │
/// │  flags: u8             (1 byte)                        │
/// │  txn_start: u64        (8 bytes) MVCC                  │
/// │  txn_end: u64          (8 bytes) MVCC                  │
/// ├───────────────────────────────────────────────────────┤
/// │ Slots: [offset: u16, len: u16] × slot_count           │
/// ├───────────────────────────────────────────────────────┤
/// │ Data: contiguous column values                         │
/// └───────────────────────────────────────────────────────┘
/// ```
#[repr(C)]
pub struct SlotRow {
    /// Row identifier
    row_id: u64,
    /// Null bitmap (bit i = 1 means column i is NULL)
    null_bitmap: u16,
    /// Number of columns (slots)
    slot_count: u8,
    /// Row flags (deleted, uncommitted, etc.)
    flags: u8,
    /// MVCC: Transaction that created this version
    txn_start: u64,
    /// MVCC: Transaction that deleted this version (u64::MAX = not deleted)
    txn_end: u64,
    /// Raw storage for slots + data
    /// Layout: [Slot; slot_count][data bytes...]
    storage: Vec<u8>,
}

impl SlotRow {
    /// Create a new slot row with specified column count
    pub fn new(row_id: u64, slot_count: u8) -> Self {
        assert!((slot_count as usize) <= MAX_SLOT_COLUMNS);
        let slots_size = (slot_count as usize) * Slot::SIZE;
        Self {
            row_id,
            null_bitmap: 0,
            slot_count,
            flags: SlotRowFlags::Normal as u8,
            txn_start: 0,
            txn_end: u64::MAX,
            storage: vec![0u8; slots_size],
        }
    }

    /// Create from raw values
    pub fn from_values(row_id: u64, values: &[Option<&[u8]>]) -> Self {
        let slot_count = values.len().min(MAX_SLOT_COLUMNS) as u8;
        let slots_size = (slot_count as usize) * Slot::SIZE;
        
        // Calculate total data size
        let data_size: usize = values.iter()
            .map(|v| v.map(|b| b.len()).unwrap_or(0))
            .sum();
        
        let mut storage = vec![0u8; slots_size + data_size];
        let mut null_bitmap = 0u16;
        let mut data_offset = 0u16;
        
        // Write slots and data
        for (i, value) in values.iter().enumerate() {
            let slot = match value {
                Some(data) => {
                    let slot = Slot::new(data_offset, data.len() as u16);
                    // Write data
                    let data_start = slots_size + data_offset as usize;
                    storage[data_start..data_start + data.len()].copy_from_slice(data);
                    data_offset += data.len() as u16;
                    slot
                }
                None => {
                    null_bitmap |= 1 << i;
                    Slot::null()
                }
            };
            
            // Write slot
            let slot_start = i * Slot::SIZE;
            storage[slot_start..slot_start + 2].copy_from_slice(&slot.offset.to_le_bytes());
            storage[slot_start + 2..slot_start + 4].copy_from_slice(&slot.len.to_le_bytes());
        }
        
        Self {
            row_id,
            null_bitmap,
            slot_count,
            flags: SlotRowFlags::Normal as u8,
            txn_start: 0,
            txn_end: u64::MAX,
            storage,
        }
    }

    /// Get row ID
    #[inline]
    pub fn row_id(&self) -> u64 {
        self.row_id
    }

    /// Get column count
    #[inline]
    pub fn column_count(&self) -> usize {
        self.slot_count as usize
    }

    /// Check if column is NULL - O(1)
    #[inline]
    pub fn is_null(&self, column_idx: usize) -> bool {
        if column_idx >= self.slot_count as usize {
            return true;
        }
        (self.null_bitmap & (1 << column_idx)) != 0
    }

    /// Get slot for a column - O(1)
    #[inline]
    fn get_slot(&self, column_idx: usize) -> Option<Slot> {
        if column_idx >= self.slot_count as usize {
            return None;
        }
        let slot_start = column_idx * Slot::SIZE;
        if slot_start + Slot::SIZE > self.storage.len() {
            return None;
        }
        
        let offset = u16::from_le_bytes([
            self.storage[slot_start],
            self.storage[slot_start + 1],
        ]);
        let len = u16::from_le_bytes([
            self.storage[slot_start + 2],
            self.storage[slot_start + 3],
        ]);
        
        Some(Slot { offset, len })
    }

    /// Get column value as bytes - O(1)
    ///
    /// This is the key performance advantage: direct offset arithmetic
    /// instead of HashMap lookup (~50ns → ~2ns)
    #[inline]
    pub fn get_bytes(&self, column_idx: usize) -> Option<&[u8]> {
        if self.is_null(column_idx) {
            return None;
        }
        
        let slot = self.get_slot(column_idx)?;
        if slot.is_null() {
            return None;
        }
        
        let slots_size = (self.slot_count as usize) * Slot::SIZE;
        let data_start = slots_size + slot.offset as usize;
        let data_end = data_start + slot.len as usize;
        
        if data_end > self.storage.len() {
            return None;
        }
        
        Some(&self.storage[data_start..data_end])
    }

    /// Get column as i64 - O(1)
    #[inline]
    pub fn get_i64(&self, column_idx: usize) -> Option<i64> {
        let bytes = self.get_bytes(column_idx)?;
        if bytes.len() != 8 {
            return None;
        }
        Some(i64::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Get column as u64 - O(1)
    #[inline]
    pub fn get_u64(&self, column_idx: usize) -> Option<u64> {
        let bytes = self.get_bytes(column_idx)?;
        if bytes.len() != 8 {
            return None;
        }
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Get column as f64 - O(1)
    #[inline]
    pub fn get_f64(&self, column_idx: usize) -> Option<f64> {
        let bytes = self.get_bytes(column_idx)?;
        if bytes.len() != 8 {
            return None;
        }
        Some(f64::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Get column as bool - O(1)
    #[inline]
    pub fn get_bool(&self, column_idx: usize) -> Option<bool> {
        let bytes = self.get_bytes(column_idx)?;
        if bytes.is_empty() {
            return None;
        }
        Some(bytes[0] != 0)
    }

    /// Get column as string - O(1)
    #[inline]
    pub fn get_str(&self, column_idx: usize) -> Option<&str> {
        let bytes = self.get_bytes(column_idx)?;
        std::str::from_utf8(bytes).ok()
    }

    /// Set MVCC timestamps
    pub fn set_mvcc(&mut self, txn_start: u64, txn_end: u64) {
        self.txn_start = txn_start;
        self.txn_end = txn_end;
    }

    /// Get MVCC start timestamp
    #[inline]
    pub fn txn_start(&self) -> u64 {
        self.txn_start
    }

    /// Get MVCC end timestamp
    #[inline]
    pub fn txn_end(&self) -> u64 {
        self.txn_end
    }

    /// Check if visible at snapshot
    #[inline]
    pub fn is_visible_at(&self, snapshot_ts: u64) -> bool {
        self.txn_start < snapshot_ts && snapshot_ts <= self.txn_end
    }

    /// Set deleted flag
    pub fn set_deleted(&mut self) {
        self.flags |= SlotRowFlags::Deleted as u8;
    }

    /// Check if deleted
    #[inline]
    pub fn is_deleted(&self) -> bool {
        (self.flags & SlotRowFlags::Deleted as u8) != 0
    }

    /// Get total memory size
    pub fn memory_size(&self) -> usize {
        SLOT_ROW_HEADER_SIZE + self.storage.len()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_size = SLOT_ROW_HEADER_SIZE + self.storage.len();
        let mut buf = Vec::with_capacity(total_size);
        
        buf.extend_from_slice(&self.row_id.to_le_bytes());
        buf.extend_from_slice(&self.null_bitmap.to_le_bytes());
        buf.push(self.slot_count);
        buf.push(self.flags);
        buf.extend_from_slice(&self.txn_start.to_le_bytes());
        buf.extend_from_slice(&self.txn_end.to_le_bytes());
        buf.extend_from_slice(&self.storage);
        
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < SLOT_ROW_HEADER_SIZE {
            return None;
        }
        
        let row_id = u64::from_le_bytes(data[0..8].try_into().ok()?);
        let null_bitmap = u16::from_le_bytes(data[8..10].try_into().ok()?);
        let slot_count = data[10];
        let flags = data[11];
        let txn_start = u64::from_le_bytes(data[12..20].try_into().ok()?);
        let txn_end = u64::from_le_bytes(data[20..28].try_into().ok()?);
        let storage = data[28..].to_vec();
        
        Some(Self {
            row_id,
            null_bitmap,
            slot_count,
            flags,
            txn_start,
            txn_end,
            storage,
        })
    }
}

impl std::fmt::Debug for SlotRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlotRow")
            .field("row_id", &self.row_id)
            .field("slot_count", &self.slot_count)
            .field("null_bitmap", &format!("{:016b}", self.null_bitmap))
            .field("flags", &self.flags)
            .field("txn_start", &self.txn_start)
            .field("txn_end", &self.txn_end)
            .field("storage_len", &self.storage.len())
            .finish()
    }
}

// =============================================================================
// Arena-Based Slot Row Allocation
// =============================================================================

/// Arena allocator for SlotRow storage
///
/// Allocates rows from contiguous memory blocks to:
/// - Reduce heap fragmentation
/// - Improve cache locality
/// - Enable efficient bulk deallocation
pub struct SlotRowArena {
    /// Memory blocks
    blocks: Vec<Vec<u8>>,
    /// Current block index
    current_block: usize,
    /// Current offset in current block
    current_offset: usize,
    /// Block size (default 64KB)
    block_size: usize,
    /// Total bytes allocated
    total_allocated: usize,
}

impl SlotRowArena {
    /// Default block size (64KB - L2 cache friendly)
    pub const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

    pub fn new() -> Self {
        Self::with_block_size(Self::DEFAULT_BLOCK_SIZE)
    }

    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            blocks: vec![vec![0u8; block_size]],
            current_block: 0,
            current_offset: 0,
            block_size,
            total_allocated: 0,
        }
    }

    /// Allocate space for a row
    pub fn allocate(&mut self, size: usize) -> &mut [u8] {
        if self.current_offset + size > self.block_size {
            // Need new block
            self.blocks.push(vec![0u8; self.block_size.max(size)]);
            self.current_block = self.blocks.len() - 1;
            self.current_offset = 0;
        }
        
        let start = self.current_offset;
        self.current_offset += size;
        self.total_allocated += size;
        
        &mut self.blocks[self.current_block][start..start + size]
    }

    /// Store a SlotRow and return handle
    pub fn store(&mut self, row: &SlotRow) -> SlotRowHandle {
        let bytes = row.to_bytes();
        let slot = self.allocate(bytes.len());
        slot.copy_from_slice(&bytes);
        
        SlotRowHandle {
            block_idx: self.current_block,
            offset: self.current_offset - bytes.len(),
            len: bytes.len(),
        }
    }

    /// Get row from handle
    pub fn get(&self, handle: &SlotRowHandle) -> Option<SlotRow> {
        let block = self.blocks.get(handle.block_idx)?;
        let data = block.get(handle.offset..handle.offset + handle.len)?;
        SlotRow::from_bytes(data)
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get number of blocks
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Reset arena (keeps allocated memory)
    pub fn reset(&mut self) {
        self.current_block = 0;
        self.current_offset = 0;
        self.total_allocated = 0;
    }
}

impl Default for SlotRowArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to a SlotRow stored in arena
#[derive(Debug, Clone, Copy)]
pub struct SlotRowHandle {
    block_idx: usize,
    offset: usize,
    len: usize,
}

impl SlotRowHandle {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_row_basic() {
        let row = SlotRow::from_values(1, &[
            Some(b"hello"),
            Some(&42i64.to_le_bytes()),
            None,
            Some(b"world"),
        ]);
        
        assert_eq!(row.row_id(), 1);
        assert_eq!(row.column_count(), 4);
        assert!(!row.is_null(0));
        assert!(!row.is_null(1));
        assert!(row.is_null(2));
        assert!(!row.is_null(3));
    }

    #[test]
    fn test_slot_row_get_bytes() {
        let row = SlotRow::from_values(1, &[
            Some(b"hello"),
            Some(b"world"),
        ]);
        
        assert_eq!(row.get_bytes(0), Some(b"hello".as_slice()));
        assert_eq!(row.get_bytes(1), Some(b"world".as_slice()));
        assert_eq!(row.get_bytes(2), None);
    }

    #[test]
    fn test_slot_row_get_typed() {
        let row = SlotRow::from_values(1, &[
            Some(&42i64.to_le_bytes()),
            Some(&3.14f64.to_le_bytes()),
            Some(&1u8.to_le_bytes()),
        ]);
        
        assert_eq!(row.get_i64(0), Some(42));
        assert_eq!(row.get_f64(1), Some(3.14));
        assert_eq!(row.get_bool(2), Some(true));
    }

    #[test]
    fn test_slot_row_get_str() {
        let row = SlotRow::from_values(1, &[
            Some(b"hello world"),
        ]);
        
        assert_eq!(row.get_str(0), Some("hello world"));
    }

    #[test]
    fn test_slot_row_mvcc() {
        let mut row = SlotRow::from_values(1, &[Some(b"test")]);
        row.set_mvcc(100, 200);
        
        assert_eq!(row.txn_start(), 100);
        assert_eq!(row.txn_end(), 200);
        assert!(row.is_visible_at(150));
        assert!(!row.is_visible_at(50));
        assert!(!row.is_visible_at(250));
    }

    #[test]
    fn test_slot_row_serialize() {
        let row = SlotRow::from_values(42, &[
            Some(b"hello"),
            Some(&123i64.to_le_bytes()),
            None,
        ]);
        
        let bytes = row.to_bytes();
        let restored = SlotRow::from_bytes(&bytes).unwrap();
        
        assert_eq!(restored.row_id(), 42);
        assert_eq!(restored.get_bytes(0), Some(b"hello".as_slice()));
        assert_eq!(restored.get_i64(1), Some(123));
        assert!(restored.is_null(2));
    }

    #[test]
    fn test_slot_row_memory_size() {
        let row = SlotRow::from_values(1, &[
            Some(b"hello"),
            Some(b"world"),
        ]);
        
        // Header (28) + 2 slots (8) + data (10) = 46 bytes
        // Much smaller than HashMap equivalent (~150+ bytes)
        let size = row.memory_size();
        assert!(size < 100, "SlotRow size {} should be < 100 bytes", size);
    }

    #[test]
    fn test_slot_row_arena() {
        let mut arena = SlotRowArena::new();
        
        let row1 = SlotRow::from_values(1, &[Some(b"hello")]);
        let row2 = SlotRow::from_values(2, &[Some(b"world")]);
        
        let h1 = arena.store(&row1);
        let h2 = arena.store(&row2);
        
        let r1 = arena.get(&h1).unwrap();
        let r2 = arena.get(&h2).unwrap();
        
        assert_eq!(r1.row_id(), 1);
        assert_eq!(r2.row_id(), 2);
        assert_eq!(r1.get_str(0), Some("hello"));
        assert_eq!(r2.get_str(0), Some("world"));
    }

    #[test]
    fn test_slot_row_arena_many_rows() {
        // Use small block size to force multiple blocks
        let mut arena = SlotRowArena::with_block_size(1024);
        let mut handles = Vec::new();
        
        for i in 0..1000 {
            let row = SlotRow::from_values(i, &[
                Some(&i.to_le_bytes()),
                Some(format!("row_{}", i).as_bytes()),
            ]);
            handles.push(arena.store(&row));
        }
        
        // Verify all rows
        for (i, handle) in handles.iter().enumerate() {
            let row = arena.get(handle).unwrap();
            assert_eq!(row.row_id(), i as u64);
            assert_eq!(row.get_u64(0), Some(i as u64));
        }
        
        // Should use multiple blocks (1000 rows * ~50 bytes = ~50KB > 1024)
        assert!(arena.block_count() > 1);
    }
}
