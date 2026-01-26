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
