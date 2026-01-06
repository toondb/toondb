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

//! Block-Oriented SSTable Format
//!
//! This module implements a RocksDB-style block-based SSTable format
//! optimized for efficient point lookups and range scans.
//!
//! ## Module Structure
//!
//! - `block.rs`: Block encoding/decoding with restart points and hash index
//! - `table.rs`: SSTable reader with lazy block loading
//! - `builder.rs`: SSTable builder with configurable block size
//! - `filter.rs`: Pluggable filter policy (Bloom, Ribbon, Xor)
//! - `format.rs`: Forward-compatible table container format

pub mod block;
pub mod builder;
pub mod filter;
pub mod format;
pub mod table;

pub use block::{BlockBuilder, BlockHandle, BlockIterator, BlockType};
pub use builder::{SSTableBuilder, SSTableBuilderOptions, SSTableBuilderResult};
pub use filter::{FilterPolicy, BloomFilterPolicy, RibbonFilterPolicy, XorFilterPolicy, FilterReader};
pub use format::{SSTableFormat, Section, SectionType, TableMagic, Header, Footer, HEADER_SIZE};
pub use table::{SSTable, SSTableIterator, TableMetadata, ReadOptions, BlockCache, CachedBlock};
