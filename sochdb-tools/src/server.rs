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

//! SochDB IPC Server
//!
//! Standalone server process that provides IPC access to a SochDB database.
//! This is used by the JavaScript and Python SDKs in embedded mode.
//!
//! # Usage
//!
//! ```bash
//! # Start server for a database directory
//! sochdb-server --db ./my_database
//!
//! # Specify custom socket path
//! sochdb-server --db ./my_database --socket /tmp/custom.sock
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing::{Level, info, error};
use tracing_subscriber::FmtSubscriber;

use sochdb_storage::database::Database;
use sochdb_storage::ipc_server::{IpcServer, IpcServerConfig};

/// SochDB IPC Server - provides multi-process access to SochDB databases
#[derive(Parser, Debug)]
#[command(name = "sochdb-server")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the database directory
    #[arg(short, long, default_value = "./sochdb_data")]
    db: PathBuf,

    /// Path to the Unix socket (default: <db>/sochdb.sock)
    #[arg(short, long)]
    socket: Option<PathBuf>,

    /// Maximum number of client connections
    #[arg(long, default_value = "100")]
    max_clients: usize,

    /// Connection timeout in milliseconds
    #[arg(long, default_value = "30000")]
    timeout_ms: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() {
    let args = Args::parse();

    // Initialize tracing
    let level = match args.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber");

    // Determine socket path
    let socket_path = args.socket.unwrap_or_else(|| {
        args.db.join("sochdb.sock")
    });

    // Ensure database directory exists
    if !args.db.exists() {
        std::fs::create_dir_all(&args.db).expect("Failed to create database directory");
    }

    info!(
        "Starting SochDB server: db={:?}, socket={:?}",
        args.db, socket_path
    );

    // Open database
    let db = match Database::open(&args.db) {
        Ok(db) => Arc::new(db),
        Err(e) => {
            error!("Failed to open database: {}", e);
            std::process::exit(1);
        }
    };

    // Configure and start IPC server
    let config = IpcServerConfig {
        socket_path: socket_path.clone(),
        max_connections: args.max_clients,
        ..Default::default()
    };

    let server = IpcServer::new(Arc::clone(&db), config);

    // Handle shutdown signals
    ctrlc::set_handler(move || {
        info!("Received shutdown signal, cleaning up...");
        // Clean up socket file
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    // Start the server (blocks) - use run() not start() which is non-blocking
    info!("SochDB server ready, accepting connections");
    if let Err(e) = server.run() {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}
