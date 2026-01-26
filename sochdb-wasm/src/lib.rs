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

//! SochDB WASM - Vector Index for Browser JavaScript
//!
//! This crate provides a WebAssembly build of SochDB's HNSW vector index
//! for use in browser JavaScript applications.
//!
//! ## Features
//!
//! - Pure Rust implementation compiled to WASM
//! - Zero network dependencies - runs entirely in browser
//! - Efficient typed array interop with JavaScript
//! - Suitable for client-side embedding search, recommendations, etc.
//!
//! ## Usage (JavaScript/TypeScript)
//!
//! ```javascript
//! import init, { WasmVectorIndex } from 'sochdb-wasm';
//!
//! async function main() {
//!   // Initialize WASM module
//!   await init();
//!
//!   // Create index with dimension=768, M=16, ef_construction=100
//!   const index = new WasmVectorIndex(768, 16, 100);
//!
//!   // Insert vectors (Float32Array)
//!   const ids = BigUint64Array.from([1n, 2n, 3n]);
//!   const vectors = new Float32Array(768 * 3);
//!   // ... fill vectors ...
//!   
//!   const inserted = index.insertBatch(ids, vectors);
//!   console.log(`Inserted ${inserted} vectors`);
//!
//!   // Search
//!   const query = new Float32Array(768);
//!   const results = index.search(query, 10);
//!   console.log('Results:', results);
//! }
//! ```
//!
//! ## Build Instructions
//!
//! ```bash
//! # Install wasm-pack
//! cargo install wasm-pack
//!
//! # Build for web
//! cd sochdb-wasm
//! wasm-pack build --target web --release
//!
//! # Build for bundler (webpack, rollup, etc.)
//! wasm-pack build --target bundler --release
//! ```

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

mod hnsw_core;

pub use hnsw_core::WasmVectorIndex;

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Search result returned from vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct SearchResult {
    /// Vector ID
    pub id: u64,
    /// Distance to query vector
    pub distance: f32,
}

#[wasm_bindgen]
impl SearchResult {
    #[wasm_bindgen(constructor)]
    pub fn new(id: u64, distance: f32) -> Self {
        Self { id, distance }
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct IndexStats {
    /// Number of vectors in index
    pub num_vectors: u32,
    /// Vector dimension
    pub dimension: u32,
    /// Maximum layer in HNSW graph
    pub max_layer: u32,
    /// Average connections per node
    pub avg_connections: f32,
}

#[wasm_bindgen]
impl IndexStats {
    #[wasm_bindgen(getter)]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn console_log(s: &str) {
    web_sys::console::log_1(&JsValue::from_str(s));
}

/// Get current performance timestamp (high resolution)
#[wasm_bindgen]
pub fn performance_now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}
