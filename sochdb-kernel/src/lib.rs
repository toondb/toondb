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

//! # SochDB Kernel
//!
//! The minimal ACID core of SochDB with a plugin architecture.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Extension Layer                          │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
//! │  │ LSCS Plugin │ │Vector Plugin│ │ Observability Plugin│   │
//! │  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
//! │         │               │                    │              │
//! │         ▼               ▼                    ▼              │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              Plugin Manager (Registry)               │   │
//! │  └─────────────────────────┬───────────────────────────┘   │
//! └────────────────────────────┼────────────────────────────────┘
//!                              │
//! ┌────────────────────────────┼────────────────────────────────┐
//! │                            ▼                                │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │                 Kernel API (Traits)                  │   │
//! │  │   KernelStorage, KernelTransaction, KernelCatalog    │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                                                             │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
//! │  │   WAL    │ │   MVCC   │ │  Pager   │ │   Catalog    │   │
//! │  │ Recovery │ │   Txn    │ │  Buffer  │ │   Schema     │   │
//! │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
//! │                                                             │
//! │                     KERNEL (~5K LOC)                        │
//! │              Auditable • Stable API • ACID                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design Principles
//!
//! 1. **Minimal Core**: Only ACID-critical code in kernel (<5K LOC)
//! 2. **Plugin Everything**: Storage backends, indices, observability are plugins
//! 3. **No Dependency Bloat**: Core has minimal deps, plugins bring their own
//! 4. **Stable API**: Kernel API is versioned, plugins can evolve independently
//! 5. **Auditable**: Small enough for formal verification
//!
//! ## WASM Plugin System
//!
//! The kernel supports secure WASM-sandboxed plugins with:
//! - Memory isolation (linear memory per plugin)
//! - Fuel limits (instruction counting)
//! - Capability-based access control
//! - Hot-reload without restart

pub mod atomic_claim; // Atomic claim protocol for queue operations (Task: Linearizable Dequeue)
pub mod error;
pub mod kernel_api;
pub mod page;
pub mod plugin;
pub mod plugin_hot_reload;
pub mod plugin_manifest;
pub mod python_sandbox;
pub mod transaction;
pub mod wal;
pub mod wasm_host_abi;
pub mod wasm_runtime;
pub mod wasm_sandbox_runtime;

// Re-exports for convenience
pub use error::{KernelError, KernelResult};
pub use kernel_api::{KernelCatalog, KernelStorage, KernelTransaction};
pub use plugin::{
    Extension, ExtensionCapability, ExtensionInfo, IndexExtension, ObservabilityExtension,
    PluginManager, StorageExtension,
};
pub use transaction::{IsolationLevel, TransactionId, TransactionState, TxnManager};
pub use wal::{LogSequenceNumber, WalManager, WalRecord, WalRecordType};

// Atomic claim protocol for queue operations
pub use atomic_claim::{
    AtomicClaimManager, ClaimResult, ClaimStats, ClaimToken, CompareAndSwap, LeaseConfig,
    LeaseManager,
};

/// Kernel version for API stability tracking
pub const KERNEL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Kernel API version - bump when breaking changes occur
pub const KERNEL_API_VERSION: u32 = 1;

/// Maximum recommended kernel code size for auditability
pub const MAX_KERNEL_LOC: usize = 5000;
