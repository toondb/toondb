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

//! SochDB Bulk Load Utility
//! 
//! High-performance HNSW index construction from raw vector data.
//! 
//! This binary achieves 100% of pure Rust throughput by bypassing
//! the Python FFI layer entirely. Use for:
//! - ETL pipelines
//! - CI/CD index builds
//! - Scheduled index rebuilds
//! 
//! # Usage
//! 
//! ```bash
//! # Build from raw f32 file (N × D × 4 bytes)
//! sochdb-bulk-load --input embeddings.bin --output index.hnsw --dimension 768
//! 
//! # Build from NumPy .npy file
//! sochdb-bulk-load --input embeddings.npy --output index.hnsw --dimension 768 --format npy
//! 
//! # Custom HNSW parameters
//! sochdb-bulk-load --input data.bin --output index.hnsw --dimension 768 \
//!     --max-connections 16 --ef-construction 100
//! ```
//! 
//! # Performance
//! 
//! | Dimension | Throughput | 
//! |-----------|------------|
//! | 128D      | ~9,500 vec/s |
//! | 768D      | ~1,600 vec/s |

use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use sochdb_index::hnsw::{HnswConfig, HnswIndex};

/// Command-line arguments
struct Args {
    /// Input file (NumPy .npy or raw f32)
    input: PathBuf,
    /// Output index file
    output: PathBuf,
    /// Vector dimension
    dimension: usize,
    /// HNSW M parameter (max connections)
    max_connections: usize,
    /// HNSW ef_construction parameter
    ef_construction: usize,
    /// Input format: npy, raw_f32
    format: String,
    /// Number of vectors (required for raw_f32 format)
    num_vectors: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut dimension = None;
    let mut max_connections = 16usize;
    let mut ef_construction = 100usize;
    let mut format = "raw_f32".to_string();
    let mut num_vectors = None;
    
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" | "-i" => {
                input = args.next().map(PathBuf::from);
            }
            "--output" | "-o" => {
                output = args.next().map(PathBuf::from);
            }
            "--dimension" | "-d" => {
                dimension = args.next().and_then(|s| s.parse().ok());
            }
            "--max-connections" | "-m" => {
                max_connections = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
            }
            "--ef-construction" | "-e" => {
                ef_construction = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
            }
            "--format" | "-f" => {
                format = args.next().unwrap_or_else(|| "raw_f32".to_string());
            }
            "--num-vectors" | "-n" => {
                num_vectors = args.next().and_then(|s| s.parse().ok());
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
    }
    
    Ok(Args {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
        dimension: dimension.ok_or("--dimension is required")?,
        max_connections,
        ef_construction,
        format,
        num_vectors,
    })
}

fn print_usage() {
    eprintln!(r#"
SochDB Bulk Load - High-performance HNSW index builder

USAGE:
    bulk-load --input <FILE> --output <FILE> --dimension <DIM> [OPTIONS]

REQUIRED:
    -i, --input <FILE>        Input vector file (raw f32 or .npy)
    -o, --output <FILE>       Output index file
    -d, --dimension <DIM>     Vector dimension

OPTIONS:
    -m, --max-connections <M>     HNSW M parameter [default: 16]
    -e, --ef-construction <EF>    Build-time ef [default: 100]
    -f, --format <FORMAT>         Input format: raw_f32, npy [default: raw_f32]
    -n, --num-vectors <N>         Number of vectors (required for raw_f32)
    -h, --help                    Print help

EXAMPLES:
    # Build from raw f32 binary file
    bulk-load -i embeddings.bin -o index.hnsw -d 768 -n 10000

    # Build from NumPy .npy file (auto-detects count)
    bulk-load -i embeddings.npy -o index.hnsw -d 768 -f npy

    # Custom HNSW parameters for higher recall
    bulk-load -i data.bin -o index.hnsw -d 768 -n 10000 -m 32 -e 200
"#);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  SochDB Bulk Load - High-Performance HNSW Builder            ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Loading vectors from {:?}...", args.input);
    
    // Memory-map input file for zero-copy access
    let file = File::open(&args.input)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    // Parse format and extract vectors
    let (n_vectors, vectors) = match args.format.as_str() {
        "npy" => parse_npy(&mmap, args.dimension)?,
        "raw_f32" => parse_raw_f32(&mmap, args.dimension, args.num_vectors)?,
        _ => return Err(format!("Unknown format: {}. Use 'npy' or 'raw_f32'", args.format).into()),
    };
    
    eprintln!("Loaded {} vectors × {} dimensions", n_vectors, args.dimension);
    eprintln!("Config: M={}, ef_construction={}", args.max_connections, args.ef_construction);
    eprintln!();
    
    // Create index with optimal config
    let config = HnswConfig {
        max_connections: args.max_connections,
        max_connections_layer0: args.max_connections * 2,
        ef_construction: args.ef_construction,
        ..Default::default()
    };
    
    let index = HnswIndex::new(args.dimension, config);
    
    // Generate sequential IDs
    let ids: Vec<u128> = (0..n_vectors as u128).collect();
    
    // Bulk insert with progress reporting
    eprintln!("Building HNSW index...");
    let start = Instant::now();
    
    let inserted = index.insert_batch_flat(&ids, vectors, args.dimension)?;
    
    let elapsed = start.elapsed();
    let rate = inserted as f64 / elapsed.as_secs_f64();
    eprintln!(
        "Inserted {} vectors in {:.2}s ({:.0} vec/s)",
        inserted, elapsed.as_secs_f64(), rate
    );
    
    // Serialize to disk
    eprintln!("Saving index to {:?}...", args.output);
    
    // Use the persistence module to save
    let save_start = Instant::now();
    index.save_to_disk_compressed(&args.output)
        .map_err(|e| format!("Failed to save index: {}", e))?;
    let save_elapsed = save_start.elapsed();
    
    let file_size = std::fs::metadata(&args.output)?.len();
    eprintln!(
        "Saved {:.1} MB in {:.2}s ({:.0} MB/s)",
        file_size as f64 / 1024.0 / 1024.0,
        save_elapsed.as_secs_f64(),
        file_size as f64 / 1024.0 / 1024.0 / save_elapsed.as_secs_f64()
    );
    
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Done! Index ready at {:?}", args.output);
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    
    Ok(())
}

/// Parse NumPy .npy file format
/// 
/// NPY format:
/// - 6-byte magic: \x93NUMPY
/// - 2-byte version
/// - 2-byte header length (v1) or 4-byte (v2+)
/// - Header dict (ASCII): {'descr': '<f4', 'fortran_order': False, 'shape': (N, D)}
/// - Data: N × D × 4 bytes
fn parse_npy(mmap: &Mmap, dimension: usize) -> Result<(usize, &[f32]), Box<dyn std::error::Error>> {
    // Check magic bytes
    if mmap.len() < 10 || &mmap[0..6] != b"\x93NUMPY" {
        return Err("Invalid NPY file: missing magic bytes".into());
    }
    
    let version_major = mmap[6];
    let version_minor = mmap[7];
    
    // Parse header length
    let (header_len, header_start) = if version_major == 1 {
        let len = u16::from_le_bytes([mmap[8], mmap[9]]) as usize;
        (len, 10)
    } else if version_major >= 2 {
        let len = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
        (len, 12)
    } else {
        return Err(format!("Unsupported NPY version: {}.{}", version_major, version_minor).into());
    };
    
    let data_start = header_start + header_len;
    
    // Parse header to extract shape
    let header = std::str::from_utf8(&mmap[header_start..data_start])?;
    
    // Simple shape extraction: find 'shape': (N, D)
    let shape_start = header.find("'shape':").or_else(|| header.find("\"shape\":"))
        .ok_or("Could not find shape in NPY header")?;
    
    let paren_start = header[shape_start..].find('(').ok_or("Could not find shape tuple")? + shape_start;
    let paren_end = header[paren_start..].find(')').ok_or("Could not find shape tuple end")? + paren_start;
    
    let shape_str = &header[paren_start + 1..paren_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    if dims.len() != 2 {
        return Err(format!("Expected 2D array, got shape: {:?}", dims).into());
    }
    
    let n_vectors = dims[0];
    let file_dim = dims[1];
    
    if file_dim != dimension {
        return Err(format!(
            "Dimension mismatch: file has {}, expected {}",
            file_dim, dimension
        ).into());
    }
    
    // Verify data size
    let expected_bytes = n_vectors * dimension * 4;
    let actual_bytes = mmap.len() - data_start;
    if actual_bytes < expected_bytes {
        return Err(format!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_bytes, actual_bytes
        ).into());
    }
    
    // Create slice from data
    let data = &mmap[data_start..];
    let vectors: &[f32] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const f32,
            n_vectors * dimension,
        )
    };
    
    Ok((n_vectors, vectors))
}

/// Parse raw f32 binary file
fn parse_raw_f32(
    mmap: &Mmap,
    dimension: usize,
    num_vectors: Option<usize>,
) -> Result<(usize, &[f32]), Box<dyn std::error::Error>> {
    let n_floats = mmap.len() / 4;
    
    let n_vectors = if let Some(n) = num_vectors {
        if n * dimension > n_floats {
            return Err(format!(
                "File too small: need {} floats for {} vectors × {} dim, but file has {}",
                n * dimension, n, dimension, n_floats
            ).into());
        }
        n
    } else {
        let n = n_floats / dimension;
        if n == 0 {
            return Err("File is empty or dimension is too large".into());
        }
        eprintln!("Auto-detected {} vectors from file size", n);
        n
    };
    
    let vectors: &[f32] = unsafe {
        std::slice::from_raw_parts(
            mmap.as_ptr() as *const f32,
            n_vectors * dimension,
        )
    };
    
    Ok((n_vectors, vectors))
}
