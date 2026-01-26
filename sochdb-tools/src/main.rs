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

//! SochDB Bulk CLI
//!
//! High-performance command-line tool for bulk vector index operations.
//! Bypasses Python FFI overhead for maximum throughput.
//!
//! ## Usage
//!
//! ```bash
//! # Build HNSW index from raw vectors
//! sochdb-bulk build-index --input vectors.f32 --output index.hnsw -d 768
//!
//! # Build from NumPy .npy file
//! sochdb-bulk build-index --input embeddings.npy --output index.hnsw
//!
//! # Query an existing index
//! sochdb-bulk query --index index.hnsw --query query.f32 --k 10
//!
//! # Get index info
//! sochdb-bulk info --index index.hnsw
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use sochdb_index::hnsw::{HnswConfig, HnswIndex};
use sochdb_tools::io::{open_vectors, ids_from_mmap, open_ids, VectorFormat};
use sochdb_tools::io::{ensure_resident_for_hnsw, FaultTelemetry, load_bulk};
use sochdb_tools::progress::ProgressReporter;
use sochdb_tools::guardrails::{check_safe_mode, log_insert_path, print_perf_summary};
use sochdb_tools::ordering::{compute_ordering, OrderingStrategy};

/// SochDB Bulk CLI - High-performance vector index operations
#[derive(Parser)]
#[command(name = "sochdb-bulk")]
#[command(about = "High-performance bulk operations for SochDB vector indices")]
#[command(version)]
#[command(propagate_version = true)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build an HNSW index from vector data
    BuildIndex {
        /// Input vector file (raw f32 or .npy)
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output index file
        #[arg(short, long)]
        output: PathBuf,
        
        /// Vector dimension (auto-detected for .npy)
        #[arg(short, long)]
        dimension: Option<usize>,
        
        /// Input format: raw_f32, npy (auto-detected from extension)
        #[arg(short, long)]
        format: Option<String>,
        
        /// Optional ID file (raw u64)
        #[arg(long)]
        ids: Option<PathBuf>,
        
        /// HNSW M parameter (max connections per node)
        #[arg(short = 'm', long, default_value = "16")]
        max_connections: usize,
        
        /// HNSW ef_construction parameter
        #[arg(short = 'e', long, default_value = "100")]
        ef_construction: usize,
        
        /// Batch size for insertion
        #[arg(long, default_value = "1000")]
        batch_size: usize,
        
        /// Number of threads (0 = auto)
        #[arg(short = 't', long, default_value = "0")]
        threads: usize,
        
        /// Skip progress bar
        #[arg(long)]
        quiet: bool,
        
        /// Use direct read instead of mmap (Task 4: avoids page faults)
        #[arg(long)]
        direct_read: bool,
        
        /// Enable prefaulting for mmap (Task 1: ensures memory residency)
        #[arg(long)]
        prefault: bool,
        
        /// Enable page fault telemetry (Task 2: validates hypothesis)
        #[arg(long)]
        telemetry: bool,
        
        /// Reorder vectors for cache locality (Task 5: random_projection, kmeans, none)
        #[arg(long, default_value = "none")]
        ordering: String,
    },
    
    /// Query an existing index
    Query {
        /// Index file
        #[arg(short, long)]
        index: PathBuf,
        
        /// Query vector file (single vector, raw f32)
        #[arg(short, long)]
        query: PathBuf,
        
        /// Number of neighbors to return
        #[arg(short, long, default_value = "10")]
        k: usize,
        
        /// Search ef parameter
        #[arg(short = 'e', long)]
        ef: Option<usize>,
    },
    
    /// Get information about an index
    Info {
        /// Index file
        #[arg(short, long)]
        index: PathBuf,
    },
    
    /// Convert vector format
    Convert {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
        
        /// Input format
        #[arg(long)]
        from_format: Option<String>,
        
        /// Output format
        #[arg(long)]
        to_format: String,
        
        /// Dimension (required for some formats)
        #[arg(short, long)]
        dimension: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let filter = if cli.verbose {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();
    
    match cli.command {
        Commands::BuildIndex {
            input,
            output,
            dimension,
            format,
            ids,
            max_connections,
            ef_construction,
            batch_size,
            threads,
            quiet,
            direct_read,
            prefault,
            telemetry,
            ordering,
        } => {
            build_index(
                input,
                output,
                dimension,
                format,
                ids,
                max_connections,
                ef_construction,
                batch_size,
                threads,
                quiet,
                direct_read,
                prefault,
                telemetry,
                ordering,
            )
        }
        Commands::Query { index, query, k, ef } => {
            query_index(index, query, k, ef)
        }
        Commands::Info { index } => {
            show_info(index)
        }
        Commands::Convert {
            input,
            output,
            from_format,
            to_format,
            dimension,
        } => {
            convert_format(input, output, from_format, to_format, dimension)
        }
    }
}

fn build_index(
    input: PathBuf,
    output: PathBuf,
    dimension: Option<usize>,
    format: Option<String>,
    ids_path: Option<PathBuf>,
    max_connections: usize,
    ef_construction: usize,
    batch_size: usize,
    threads: usize,
    quiet: bool,
    direct_read: bool,
    prefault: bool,
    telemetry_enabled: bool,
    ordering_str: String,
) -> Result<()> {
    // Banner
    if !quiet {
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║          SochDB Bulk Index Builder                           ║");
        eprintln!("║          High-Performance HNSW Construction                  ║");
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
        eprintln!();
    }
    
    // Configure thread pool
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }
    
    // Parse ordering strategy (Task 5)
    let ordering_strategy = match ordering_str.to_lowercase().as_str() {
        "random_projection" | "rp" => OrderingStrategy::RandomProjection,
        "kmeans" | "coarse_kmeans" => {
            // k = sqrt(N), but we don't know N yet; use placeholder
            OrderingStrategy::CoarseKMeans { k: 100 }
        }
        "block" => OrderingStrategy::BlockShuffle { block_size: 1024 },
        _ => OrderingStrategy::None,
    };
    
    // ==========================================================================
    // TASK 4: Direct Read vs Mmap
    // ==========================================================================
    // For bulk builds, direct read() avoids page-fault issues entirely
    
    let (n, d, vectors, _owned_data): (usize, usize, &[f32], Option<sochdb_tools::OwnedVectors>) = if direct_read {
        eprintln!("Loading vectors via direct read (--direct-read)...");
        let owned = load_bulk(&input, dimension)
            .context("Failed to load vectors")?;
        let n = owned.num_vectors;
        let d = owned.dimension;
        eprintln!("  Loaded {} vectors × {} dimensions ({:.1} MB)", 
            n, d, (n * d * 4) as f64 / 1024.0 / 1024.0);
        
        // We need to return a reference, so we box and leak
        // This is fine for CLI - process exits anyway
        let data: &'static [f32] = Box::leak(owned.data.into_boxed_slice());
        (n, d, data, None)
    } else {
        // Parse format
        let format = format
            .and_then(|s| VectorFormat::from_str(&s))
            .or_else(|| VectorFormat::from_extension(&input));
        
        eprintln!("Loading vectors from {:?}...", input);
        let reader = open_vectors(&input, format, dimension)
            .context("Failed to open vector file")?;
        
        let n = reader.num_vectors();
        let d = reader.dimension();
        
        // ==========================================================================
        // TASK 1: Prefault mmap to eliminate page-fault storms
        // ==========================================================================
        if prefault {
            eprintln!("Prefaulting memory region (--prefault)...");
            let vectors = reader.vectors();
            let ptr = vectors.as_ptr() as *mut u8;
            let len = vectors.len() * 4;
            let stats = ensure_resident_for_hnsw(ptr, len, !quiet);
            if !quiet {
                stats.print_summary("Prefault");
            }
        }
        
        // Leak the reader to get a static reference
        // Fine for CLI - process exits anyway
        let reader_box = Box::new(reader);
        let reader_ref: &'static sochdb_tools::VectorReader = Box::leak(reader_box);
        
        (n, d, reader_ref.vectors(), None)
    };
    
    eprintln!("  Vectors:   {}", n);
    eprintln!("  Dimension: {}", d);
    eprintln!("  Size:      {:.1} MB", (n * d * 4) as f64 / 1024.0 / 1024.0);
    eprintln!();
    
    // ==========================================================================
    // TASK 5: Locality-aware insertion ordering
    // ==========================================================================
    let (vectors, ids): (&[f32], Vec<u128>) = if ordering_strategy != OrderingStrategy::None {
        eprintln!("Computing locality-aware ordering...");
        
        // Adjust k for kmeans based on actual N
        let strategy = match ordering_strategy {
            OrderingStrategy::CoarseKMeans { .. } => {
                OrderingStrategy::CoarseKMeans { k: (n as f64).sqrt() as usize }
            }
            other => other,
        };
        
        let reorder_result = compute_ordering(vectors, d, strategy);
        eprintln!("  Strategy: {:?}", reorder_result.strategy);
        eprintln!("  Compute time: {:.2}s", reorder_result.compute_time.as_secs_f64());
        
        // Apply reordering
        let reordered_vectors = reorder_result.apply_to_vectors(vectors, d);
        
        // Generate or reorder IDs
        let ids: Vec<u128> = if let Some(ids_file) = &ids_path {
            eprintln!("Loading IDs from {:?}...", ids_file);
            let (mmap, num_ids) = open_ids(ids_file)?;
            let u64_ids = ids_from_mmap(&mmap);
            if num_ids != n {
                anyhow::bail!("ID count mismatch: {} IDs vs {} vectors", num_ids, n);
            }
            reorder_result.apply_to_ids(u64_ids).iter().map(|&id| id as u128).collect()
        } else {
            reorder_result.apply_to_ids(&(0..n as u128).collect::<Vec<_>>())
        };
        
        // Leak reordered vectors for static lifetime
        let reordered: &'static [f32] = Box::leak(reordered_vectors.into_boxed_slice());
        (reordered, ids)
    } else {
        // Load optional IDs without reordering
        let ids: Vec<u128> = if let Some(ids_file) = &ids_path {
            eprintln!("Loading IDs from {:?}...", ids_file);
            let (mmap, num_ids) = open_ids(ids_file)?;
            let u64_ids = ids_from_mmap(&mmap);
            if num_ids != n {
                anyhow::bail!("ID count mismatch: {} IDs vs {} vectors", num_ids, n);
            }
            u64_ids.iter().map(|&id| id as u128).collect()
        } else {
            (0..n as u128).collect()
        };
        (vectors, ids)
    };
    
    // Create index
    eprintln!("Building HNSW index...");
    eprintln!("  M:              {}", max_connections);
    eprintln!("  ef_construction: {}", ef_construction);
    eprintln!("  Batch size:     {}", batch_size);
    if threads > 0 {
        eprintln!("  Threads:        {}", threads);
    } else {
        eprintln!("  Threads:        auto ({})", rayon::current_num_threads());
    }
    eprintln!();
    
    let config = HnswConfig {
        max_connections,
        max_connections_layer0: max_connections * 2,
        ef_construction,
        ..Default::default()
    };
    
    let index = HnswIndex::new(d, config);
    
    // Check safe mode and log insert path
    if check_safe_mode() {
        eprintln!("  Safe mode: ENABLED (slower but guaranteed correct)");
    }
    log_insert_path("insert_batch_contiguous", batch_size, d);
    
    // ==========================================================================
    // TASK 2: Start telemetry if enabled
    // ==========================================================================
    let mut telemetry = if telemetry_enabled {
        eprintln!("Telemetry enabled (--telemetry): capturing page fault stats...");
        Some(FaultTelemetry::capture_start_labeled("HNSW Build"))
    } else {
        None
    };
    
    // Build with progress
    let start = Instant::now();
    let mut progress = if quiet {
        None
    } else {
        Some(ProgressReporter::new(n as u64, "Inserting"))
    };
    
    // ==========================================================================
    // OPTIMIZED INSERT PATH (Task 3: Unify with profiler baseline)
    // ==========================================================================
    // Use insert_batch_contiguous directly (same as insert_profile.rs) instead
    // of insert_batch_flat to ensure we're using the exact same code path.
    // The vectors are already contiguous from mmap, so we get zero-copy.
    // ==========================================================================
    
    let mut total_inserted = 0usize;
    for (_batch_idx, chunk_start) in (0..n).step_by(batch_size).enumerate() {
        let chunk_end = (chunk_start + batch_size).min(n);
        let batch_ids = &ids[chunk_start..chunk_end];
        let batch_vectors = &vectors[chunk_start * d..chunk_end * d];
        
        // Use insert_batch_contiguous directly (same as profiler)
        let inserted = index.insert_batch_contiguous(batch_ids, batch_vectors, d)
            .map_err(|e| anyhow::anyhow!("Insert failed: {}", e))?;
        
        total_inserted += inserted;
        
        if let Some(ref mut p) = progress {
            p.set(total_inserted as u64);
        }
    }
    
    if let Some(ref p) = progress {
        p.finish("Done");
    }
    
    let elapsed = start.elapsed();
    
    // ==========================================================================
    // TASK 2: Capture telemetry end and print summary
    // ==========================================================================
    if let Some(ref mut t) = telemetry {
        t.capture_end();
        t.print_summary("HNSW Build");
        
        // Check for fault problems
        if t.has_fault_problem() {
            eprintln!();
            eprintln!("💡 SUGGESTION: High page fault count detected.");
            eprintln!("   Try: --direct-read or --prefault to eliminate page faults");
        }
    }
    
    // Use guardrails for summary
    let file_size = {
        // Save index first
        eprintln!();
        eprintln!("Saving index to {:?}...", output);
        let save_start = Instant::now();
        
        index.save_to_disk_compressed(&output)
            .map_err(|e| anyhow::anyhow!("Failed to save: {}", e))?;
        
        let save_elapsed = save_start.elapsed();
        let size = std::fs::metadata(&output)?.len();
        
        eprintln!("Saved {:.1} MB in {:.2}s ({:.0} MB/s)",
            size as f64 / 1024.0 / 1024.0,
            save_elapsed.as_secs_f64(),
            size as f64 / 1024.0 / 1024.0 / save_elapsed.as_secs_f64()
        );
        
        size
    };
    
    // Print performance summary with guardrails
    print_perf_summary(total_inserted, d, elapsed, Some(file_size));
    
    Ok(())
}

/// Load an HNSW index, trying both compressed and uncompressed formats
fn load_index(path: &PathBuf) -> Result<HnswIndex> {
    // Try compressed first (default for save_to_disk_compressed)
    if let Ok(index) = HnswIndex::load_from_disk_compressed(path) {
        return Ok(index);
    }
    // Fall back to uncompressed
    HnswIndex::load_from_disk(path)
        .map_err(|e| anyhow::anyhow!("Failed to load index: {}", e))
}

fn query_index(index_path: PathBuf, query_path: PathBuf, k: usize, _ef: Option<usize>) -> Result<()> {
    eprintln!("Loading index from {:?}...", index_path);
    let index = load_index(&index_path)?;
    
    let stats = index.stats();
    eprintln!("Loaded: {} vectors, dimension {}", stats.num_vectors, stats.dimension);
    
    // Load query
    let query_data = std::fs::read(&query_path)?;
    if query_data.len() != stats.dimension * 4 {
        anyhow::bail!(
            "Query size mismatch: expected {} bytes ({} dim), got {}",
            stats.dimension * 4, stats.dimension, query_data.len()
        );
    }
    
    let query: &[f32] = unsafe {
        std::slice::from_raw_parts(query_data.as_ptr() as *const f32, stats.dimension)
    };
    
    // Search
    eprintln!("Searching for {} neighbors...", k);
    let start = Instant::now();
    
    let results = index.search(query, k)
        .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;
    
    let elapsed = start.elapsed();
    
    eprintln!();
    eprintln!("Results (in {:.2}ms):", elapsed.as_secs_f64() * 1000.0);
    for (i, (id, distance)) in results.iter().enumerate() {
        eprintln!("  {:2}. ID: {:>12}  Distance: {:.6}", i + 1, id, distance);
    }
    
    Ok(())
}

fn show_info(index_path: PathBuf) -> Result<()> {
    eprintln!("Loading index from {:?}...", index_path);
    let index = load_index(&index_path)?;
    
    let stats = index.stats();
    let file_size = std::fs::metadata(&index_path)?.len();
    
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                     INDEX INFORMATION                        ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  File:          {:?}", index_path);
    eprintln!("║  Vectors:       {:>12}", stats.num_vectors);
    eprintln!("║  Dimension:     {:>12}", stats.dimension);
    eprintln!("║  Max Layer:     {:>12}", stats.max_layer);
    eprintln!("║  Avg Connections: {:>10.1}", stats.avg_connections);
    eprintln!("║  File Size:     {:>10.1} MB", file_size as f64 / 1024.0 / 1024.0);
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    
    Ok(())
}

fn convert_format(
    input: PathBuf,
    output: PathBuf,
    from_format: Option<String>,
    to_format: String,
    dimension: Option<usize>,
) -> Result<()> {
    let format = from_format
        .and_then(|s| VectorFormat::from_str(&s))
        .or_else(|| VectorFormat::from_extension(&input));
    
    eprintln!("Reading {:?}...", input);
    let reader = open_vectors(&input, format, dimension)?;
    
    let n = reader.num_vectors();
    let d = reader.dimension();
    let vectors = reader.vectors();
    
    eprintln!("  {} vectors × {} dimensions", n, d);
    
    match to_format.to_lowercase().as_str() {
        "raw" | "raw_f32" | "f32" => {
            use sochdb_tools::io::raw::write_raw_f32;
            write_raw_f32(&output, vectors, d, None)?;
            eprintln!("Wrote raw f32 to {:?}", output);
        }
        _ => {
            anyhow::bail!("Unsupported output format: {}", to_format);
        }
    }
    
    Ok(())
}
