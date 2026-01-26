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

//! String Interning for Path Segments
//!
//! This module implements string interning to reduce memory usage when
//! storing repeated path segments in the PathTrie and other data structures.
//!
//! ## Memory Savings
//!
//! For paths like "users.profile.settings.theme", the segment "users" may
//! appear thousands of times across different paths. String interning
//! ensures each unique segment is stored only once.
//!
//! ## Example
//!
//! ```rust,ignore
//! use sochdb_core::string_interner::StringInterner;
//!
//! let mut interner = StringInterner::new();
//!
//! // Intern a string - returns a small integer symbol
//! let sym1 = interner.get_or_intern("users");
//! let sym2 = interner.get_or_intern("users"); // Same symbol returned
//!
//! assert_eq!(sym1, sym2); // Same string = same symbol
//!
//! // Resolve back to string
//! let s = interner.resolve(sym1);
//! assert_eq!(s, Some("users"));
//! ```
//!
//! ## Thread Safety
//!
//! The `ConcurrentStringInterner` variant is thread-safe and uses a
//! sharded lock design for concurrent access.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// A symbol representing an interned string
///
/// This is a small integer that can be used to look up the original string.
/// Using symbols instead of strings reduces memory usage when the same
/// string appears multiple times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(u32);

impl Symbol {
    /// Create a symbol from a raw u32 (for deserialization)
    pub fn from_raw(value: u32) -> Self {
        Symbol(value)
    }

    /// Get the raw u32 value (for serialization)
    pub fn as_raw(&self) -> u32 {
        self.0
    }
}

impl From<u32> for Symbol {
    fn from(value: u32) -> Self {
        Symbol(value)
    }
}

impl From<Symbol> for u32 {
    fn from(symbol: Symbol) -> Self {
        symbol.0
    }
}

/// A string interner that stores unique strings and returns symbols
///
/// This is the single-threaded version. For concurrent access, use
/// `ConcurrentStringInterner`.
#[derive(Debug)]
pub struct StringInterner {
    /// Map from string to symbol
    string_to_symbol: HashMap<String, Symbol>,
    /// Map from symbol to string (for resolve)
    symbol_to_string: Vec<String>,
    /// Next symbol to assign
    next_symbol: u32,
}

impl StringInterner {
    /// Create a new empty string interner
    pub fn new() -> Self {
        Self {
            string_to_symbol: HashMap::new(),
            symbol_to_string: Vec::new(),
            next_symbol: 0,
        }
    }

    /// Create a new string interner with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            string_to_symbol: HashMap::with_capacity(capacity),
            symbol_to_string: Vec::with_capacity(capacity),
            next_symbol: 0,
        }
    }

    /// Intern a string and return its symbol
    ///
    /// If the string is already interned, returns the existing symbol.
    /// Otherwise, creates a new symbol for this string.
    pub fn get_or_intern(&mut self, s: &str) -> Symbol {
        if let Some(&symbol) = self.string_to_symbol.get(s) {
            return symbol;
        }

        let symbol = Symbol(self.next_symbol);
        self.next_symbol += 1;

        self.string_to_symbol.insert(s.to_string(), symbol);
        self.symbol_to_string.push(s.to_string());

        symbol
    }

    /// Intern an owned string (avoids clone if string is new)
    pub fn get_or_intern_owned(&mut self, s: String) -> Symbol {
        if let Some(&symbol) = self.string_to_symbol.get(&s) {
            return symbol;
        }

        let symbol = Symbol(self.next_symbol);
        self.next_symbol += 1;

        self.symbol_to_string.push(s.clone());
        self.string_to_symbol.insert(s, symbol);

        symbol
    }

    /// Look up an already-interned string
    ///
    /// Returns None if the string has not been interned
    pub fn get(&self, s: &str) -> Option<Symbol> {
        self.string_to_symbol.get(s).copied()
    }

    /// Resolve a symbol to its string
    ///
    /// Returns None if the symbol is invalid
    pub fn resolve(&self, symbol: Symbol) -> Option<&str> {
        self.symbol_to_string
            .get(symbol.0 as usize)
            .map(|s| s.as_str())
    }

    /// Get the number of interned strings
    pub fn len(&self) -> usize {
        self.symbol_to_string.len()
    }

    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.symbol_to_string.is_empty()
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        // HashMap overhead (rough estimate: 3 words per entry)
        let map_overhead = self.string_to_symbol.len() * (3 * std::mem::size_of::<usize>());

        // String storage
        let string_storage: usize = self
            .symbol_to_string
            .iter()
            .map(|s| s.len() + std::mem::size_of::<String>())
            .sum();

        // Vec overhead
        let vec_overhead = std::mem::size_of::<Vec<String>>()
            + self.symbol_to_string.capacity() * std::mem::size_of::<String>();

        map_overhead + string_storage + vec_overhead
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Number of shards for the concurrent interner
const SHARD_COUNT: usize = 16;

/// Thread-safe string interner using sharded locks
///
/// Uses a simple hash-based sharding to reduce lock contention
/// when multiple threads are interning strings concurrently.
#[derive(Debug)]
pub struct ConcurrentStringInterner {
    /// Sharded map from string to symbol
    shards: [RwLock<HashMap<String, Symbol>>; SHARD_COUNT],
    /// Global symbol table (append-only)
    symbols: RwLock<Vec<String>>,
    /// Next symbol to assign (atomic for lock-free reads)
    next_symbol: AtomicU32,
}

impl ConcurrentStringInterner {
    /// Create a new empty concurrent string interner
    pub fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| RwLock::new(HashMap::new())),
            symbols: RwLock::new(Vec::new()),
            next_symbol: AtomicU32::new(0),
        }
    }

    /// Create a new concurrent string interner with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let per_shard = capacity / SHARD_COUNT + 1;
        Self {
            shards: std::array::from_fn(|_| RwLock::new(HashMap::with_capacity(per_shard))),
            symbols: RwLock::new(Vec::with_capacity(capacity)),
            next_symbol: AtomicU32::new(0),
        }
    }

    /// Get the shard index for a string
    #[inline]
    fn shard_for(&self, s: &str) -> usize {
        // Simple FNV-1a hash for shard selection
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in s.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        (hash as usize) % SHARD_COUNT
    }

    /// Intern a string and return its symbol
    pub fn get_or_intern(&self, s: &str) -> Symbol {
        let shard_idx = self.shard_for(s);

        // Try read-only lookup first (fast path)
        {
            let shard = self.shards[shard_idx].read();
            if let Some(&symbol) = shard.get(s) {
                return symbol;
            }
        }

        // Need to insert - acquire write lock
        let mut shard = self.shards[shard_idx].write();

        // Double-check after acquiring write lock (another thread may have inserted)
        if let Some(&symbol) = shard.get(s) {
            return symbol;
        }

        // Allocate new symbol
        let symbol = Symbol(self.next_symbol.fetch_add(1, Ordering::SeqCst));

        // Insert into symbol table (under separate lock)
        {
            let mut symbols = self.symbols.write();
            // Ensure symbols vector has space
            while symbols.len() <= symbol.0 as usize {
                symbols.push(String::new());
            }
            symbols[symbol.0 as usize] = s.to_string();
        }

        // Insert into shard
        shard.insert(s.to_string(), symbol);

        symbol
    }

    /// Look up an already-interned string
    pub fn get(&self, s: &str) -> Option<Symbol> {
        let shard_idx = self.shard_for(s);
        let shard = self.shards[shard_idx].read();
        shard.get(s).copied()
    }

    /// Resolve a symbol to its string
    pub fn resolve(&self, symbol: Symbol) -> Option<String> {
        let symbols = self.symbols.read();
        symbols.get(symbol.0 as usize).cloned()
    }

    /// Get the number of interned strings
    pub fn len(&self) -> usize {
        self.next_symbol.load(Ordering::Relaxed) as usize
    }

    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let symbols = self.symbols.read();

        // String storage
        let string_storage: usize = symbols
            .iter()
            .map(|s| s.len() + std::mem::size_of::<String>())
            .sum();

        // Shard overhead (rough estimate)
        let shard_overhead =
            SHARD_COUNT * 3 * std::mem::size_of::<usize>() * self.len() / SHARD_COUNT;

        string_storage + shard_overhead
    }
}

impl Default for ConcurrentStringInterner {
    fn default() -> Self {
        Self::new()
    }
}

// Make ConcurrentStringInterner Send + Sync
unsafe impl Send for ConcurrentStringInterner {}
unsafe impl Sync for ConcurrentStringInterner {}

/// Global string interner for path segments
///
/// This is a convenience wrapper around ConcurrentStringInterner
/// that provides a global instance for interning path segments.
pub mod global {
    use super::*;
    use std::sync::OnceLock;

    static GLOBAL_INTERNER: OnceLock<ConcurrentStringInterner> = OnceLock::new();

    /// Get the global string interner
    pub fn interner() -> &'static ConcurrentStringInterner {
        GLOBAL_INTERNER.get_or_init(ConcurrentStringInterner::new)
    }

    /// Intern a string in the global interner
    pub fn intern(s: &str) -> Symbol {
        interner().get_or_intern(s)
    }

    /// Resolve a symbol from the global interner
    pub fn resolve(symbol: Symbol) -> Option<String> {
        interner().resolve(symbol)
    }

    /// Get memory usage of the global interner
    pub fn memory_usage() -> usize {
        interner().memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_string_interner_basic() {
        let mut interner = StringInterner::new();

        let sym1 = interner.get_or_intern("hello");
        let sym2 = interner.get_or_intern("world");
        let sym3 = interner.get_or_intern("hello"); // Same as sym1

        assert_eq!(sym1, sym3);
        assert_ne!(sym1, sym2);

        assert_eq!(interner.resolve(sym1), Some("hello"));
        assert_eq!(interner.resolve(sym2), Some("world"));
    }

    #[test]
    fn test_string_interner_get() {
        let mut interner = StringInterner::new();

        assert_eq!(interner.get("hello"), None);

        let sym = interner.get_or_intern("hello");
        assert_eq!(interner.get("hello"), Some(sym));
    }

    #[test]
    fn test_string_interner_owned() {
        let mut interner = StringInterner::new();

        let sym1 = interner.get_or_intern_owned("hello".to_string());
        let sym2 = interner.get_or_intern_owned("hello".to_string());

        assert_eq!(sym1, sym2);
    }

    #[test]
    fn test_string_interner_len() {
        let mut interner = StringInterner::new();

        assert_eq!(interner.len(), 0);
        assert!(interner.is_empty());

        interner.get_or_intern("hello");
        assert_eq!(interner.len(), 1);

        interner.get_or_intern("world");
        assert_eq!(interner.len(), 2);

        interner.get_or_intern("hello"); // Duplicate
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_concurrent_interner_basic() {
        let interner = ConcurrentStringInterner::new();

        let sym1 = interner.get_or_intern("hello");
        let sym2 = interner.get_or_intern("world");
        let sym3 = interner.get_or_intern("hello");

        assert_eq!(sym1, sym3);
        assert_ne!(sym1, sym2);

        assert_eq!(interner.resolve(sym1), Some("hello".to_string()));
        assert_eq!(interner.resolve(sym2), Some("world".to_string()));
    }

    #[test]
    fn test_concurrent_interner_threaded() {
        use std::sync::Arc;

        let interner = Arc::new(ConcurrentStringInterner::new());
        let mut handles = vec![];

        // Spawn multiple threads interning overlapping strings
        for _i in 0..8 {
            let interner = Arc::clone(&interner);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let s = format!("string_{}", j % 50);
                    interner.get_or_intern(&s);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 50 unique strings
        assert_eq!(interner.len(), 50);
    }

    #[test]
    fn test_path_segment_interning() {
        let mut interner = StringInterner::new();

        // Simulate interning path segments
        let paths = [
            "users.profile.settings.theme",
            "users.profile.settings.language",
            "users.profile.name",
            "products.inventory.count",
            "products.inventory.location",
        ];

        for path in &paths {
            for segment in path.split('.') {
                interner.get_or_intern(segment);
            }
        }

        // Count unique segments
        // users, profile, settings, theme, language, name, products, inventory, count, location = 10
        assert_eq!(interner.len(), 10);
    }

    #[test]
    fn test_symbol_serialization() {
        let mut interner = StringInterner::new();

        let sym = interner.get_or_intern("test");
        let raw = sym.as_raw();
        let sym2 = Symbol::from_raw(raw);

        assert_eq!(sym, sym2);
    }

    #[test]
    fn test_global_interner() {
        let sym1 = global::intern("global_test");
        let sym2 = global::intern("global_test");

        assert_eq!(sym1, sym2);
        assert_eq!(global::resolve(sym1), Some("global_test".to_string()));
    }

    #[test]
    fn test_memory_usage() {
        let mut interner = StringInterner::new();

        for i in 0..1000 {
            interner.get_or_intern(&format!("string_{}", i));
        }

        let usage = interner.memory_usage();
        // Should be reasonable - less than 100KB for 1000 short strings
        assert!(usage < 100_000, "Memory usage too high: {}", usage);
    }

    #[test]
    fn test_empty_string() {
        let mut interner = StringInterner::new();

        let sym = interner.get_or_intern("");
        assert_eq!(interner.resolve(sym), Some(""));
    }

    #[test]
    fn test_unicode_strings() {
        let mut interner = StringInterner::new();

        let sym1 = interner.get_or_intern("こんにちは");
        let sym2 = interner.get_or_intern("世界");
        let sym3 = interner.get_or_intern("こんにちは");

        assert_eq!(sym1, sym3);
        assert_ne!(sym1, sym2);

        assert_eq!(interner.resolve(sym1), Some("こんにちは"));
        assert_eq!(interner.resolve(sym2), Some("世界"));
    }
}
