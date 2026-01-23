//! Hot path profiler for HNSW operations
//! 
//! Run with: cargo run --release -p sochdb-index --bin profile_hotpath
//!
//! This profiles the actual time spent in each phase of insert and search.

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use sochdb_index::hnsw::{HnswConfig, HnswIndex, DistanceMetric};
use rand::Rng;

const NUM_VECTORS: usize = 2000;
const DIMENSION: usize = 384;
const NUM_SEARCHES: usize = 500;
const K: usize = 10;

// Global counters for detailed breakdown
static DISTANCE_CALLS: AtomicU64 = AtomicU64::new(0);
static HEAP_PUSHES: AtomicU64 = AtomicU64::new(0);
static VISITED_CHECKS: AtomicU64 = AtomicU64::new(0);

fn main() {
    println!("=== HNSW Hot Path Profiler (Detailed) ===\n");
    println!("Config: {} vectors, {} dim, {} searches, k={}\n", NUM_VECTORS, DIMENSION, NUM_SEARCHES, K);
    
    // Create index
    let config = HnswConfig {
        max_connections: 16,
        max_connections_layer0: 32,
        ef_construction: 100,
        ef_search: 100,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    
    let index = HnswIndex::new(DIMENSION, config);
    
    // Generate random vectors
    let mut rng = rand::thread_rng();
    
    let vectors: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|_| (0..DIMENSION).map(|_| rng.r#gen::<f32>()).collect())
        .collect();
    
    let query_vectors: Vec<Vec<f32>> = (0..NUM_SEARCHES)
        .map(|_| (0..DIMENSION).map(|_| rng.r#gen::<f32>()).collect())
        .collect();
    
    // ========================================
    // PHASE 1: Profile Single Inserts
    // ========================================
    println!("--- Phase 1: Single Inserts ({} vectors) ---", NUM_VECTORS);
    
    let mut insert_times: Vec<Duration> = Vec::with_capacity(NUM_VECTORS);
    
    for (i, vec) in vectors.iter().enumerate() {
        let start = Instant::now();
        let _ = index.insert(i as u128, vec.clone());
        insert_times.push(start.elapsed());
    }
    
    let total_insert: Duration = insert_times.iter().sum();
    let avg_insert = total_insert / NUM_VECTORS as u32;
    let p50_insert = percentile(&mut insert_times.clone(), 50);
    let p99_insert = percentile(&mut insert_times.clone(), 99);
    
    println!("  Total:  {:?}", total_insert);
    println!("  Avg:    {:?}", avg_insert);
    println!("  P50:    {:?}", p50_insert);
    println!("  P99:    {:?}", p99_insert);
    println!();
    
    // ========================================
    // PHASE 2: Profile Searches
    // ========================================
    println!("--- Phase 2: Searches ({} queries, k={}) ---", NUM_SEARCHES, K);
    
    let mut search_times: Vec<Duration> = Vec::with_capacity(NUM_SEARCHES);
    
    for query in &query_vectors {
        let start = Instant::now();
        let _ = index.search(query, K);
        search_times.push(start.elapsed());
    }
    
    let total_search: Duration = search_times.iter().sum();
    let avg_search = total_search / NUM_SEARCHES as u32;
    let p50_search = percentile(&mut search_times.clone(), 50);
    let p99_search = percentile(&mut search_times.clone(), 99);
    
    println!("  Total:  {:?}", total_search);
    println!("  Avg:    {:?}", avg_search);
    println!("  P50:    {:?}", p50_search);
    println!("  P99:    {:?}", p99_search);
    println!();
    
    // ========================================
    // PHASE 3: Profile Batch Insert
    // ========================================
    println!("--- Phase 3: Batch Insert (1000 vectors) ---");
    
    let batch_vectors: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..DIMENSION).map(|_| rng.r#gen::<f32>()).collect())
        .collect();
    
    let batch_ids: Vec<u128> = (NUM_VECTORS as u128..(NUM_VECTORS + 1000) as u128).collect();
    let flat_vectors: Vec<f32> = batch_vectors.iter().flatten().copied().collect();
    
    let start = Instant::now();
    let _ = index.insert_batch_contiguous(&batch_ids, &flat_vectors, DIMENSION);
    let batch_time = start.elapsed();
    
    println!("  Total:  {:?}", batch_time);
    println!("  Per-vector: {:?}", batch_time / 1000);
    println!();
    
    // ========================================
    // PHASE 4: Detailed Search Breakdown
    // ========================================
    println!("--- Phase 4: Search Component Analysis ---");
    
    // Warm up
    for query in &query_vectors[0..10] {
        let _ = index.search(query, K);
    }
    
    // Detailed timing
    let mut times_layer_search: Vec<Duration> = Vec::new();
    let mut times_distance: Vec<Duration> = Vec::new();
    
    for query in &query_vectors {
        // Time full search
        let start = Instant::now();
        let _ = index.search(query, K);
        let full_time = start.elapsed();
        times_layer_search.push(full_time);
    }
    
    let avg_full = times_layer_search.iter().sum::<Duration>() / NUM_SEARCHES as u32;
    println!("  Avg full search: {:?}", avg_full);
    
    // Stats
    println!("\n--- Index Stats ---");
    println!("  Nodes: {}", index.len());
    println!("  Dimension: {}", DIMENSION);
    
    println!("\n=== Profiling Complete ===");
}

fn percentile(times: &mut [Duration], p: usize) -> Duration {
    times.sort();
    let idx = (times.len() * p) / 100;
    times[idx.min(times.len() - 1)]
}
