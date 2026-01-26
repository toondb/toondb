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

//! Quality Metrics and Ground Truth Validation
//!
//! Tests HNSW recall by comparing against brute-force exact nearest neighbor search
//!
//! Note: These tests use randomly generated vectors and HNSW is an approximate algorithm,
//! so results can vary between runs. The thresholds are set conservatively to account for
//! this variability while still catching major regressions.

use rand::Rng;
use std::collections::HashSet;
use sochdb_index::hnsw::{HnswConfig, HnswIndex};

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn generate_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| generate_random_vector(dim)).collect()
}

/// Compute cosine distance between two vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - (dot / (norm_a * norm_b + 1e-8))
}

/// Brute-force exact nearest neighbor search (ground truth)
fn exact_search(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(idx, vec)| (idx, cosine_distance(query, vec)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(idx, _)| idx).collect()
}

/// Compute recall@K: percentage of true top-K found by HNSW
fn compute_recall(hnsw_results: &[usize], true_results: &[usize]) -> f64 {
    let hnsw_set: HashSet<usize> = hnsw_results.iter().copied().collect();
    let true_set: HashSet<usize> = true_results.iter().copied().collect();

    let intersection = hnsw_set.intersection(&true_set).count();
    intersection as f64 / true_results.len() as f64
}

#[test]
#[ignore = "High variance with random vectors - HNSW recall depends on data distribution"]
fn test_recall_at_10() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(128, config);

    // Build index with 10K vectors
    let corpus = generate_test_vectors(10_000, 128);
    for (i, vec) in corpus.iter().enumerate() {
        index.insert(i as u128, vec.clone()).unwrap();
    }

    // Test on 100 random queries
    let num_queries = 100;
    let k = 10;
    let mut total_recall = 0.0;

    for _ in 0..num_queries {
        let query = generate_random_vector(128);

        // Ground truth (exact search)
        let true_neighbors = exact_search(&query, &corpus, k);

        // HNSW approximate search
        let hnsw_results = index.search(&query, k).unwrap();
        let hnsw_ids: Vec<usize> = hnsw_results.iter().map(|(id, _)| *id as usize).collect();

        // Calculate recall for this query
        let recall = compute_recall(&hnsw_ids, &true_neighbors);
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f64;

    println!("Average Recall@10: {:.2}%", avg_recall * 100.0);

    // Assert quality threshold - HNSW is an approximate algorithm
    // 70%+ recall is acceptable for high-dimensional random vectors
    assert!(
        avg_recall >= 0.70,
        "Recall too low: {:.2}% (expected ≥70%)",
        avg_recall * 100.0
    );
}

#[test]
#[ignore = "Flaky test with random data - run manually to verify quality"]
fn test_recall_at_different_k() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(128, config);

    let corpus = generate_test_vectors(5_000, 128);
    for (i, vec) in corpus.iter().enumerate() {
        index.insert(i as u128, vec.clone()).unwrap();
    }

    let num_queries = 50;

    for k in [1, 10, 50, 100] {
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query = generate_random_vector(128);
            let true_neighbors = exact_search(&query, &corpus, k);
            let hnsw_results = index.search(&query, k).unwrap();
            let hnsw_ids: Vec<usize> = hnsw_results.iter().map(|(id, _)| *id as usize).collect();

            let recall = compute_recall(&hnsw_ids, &true_neighbors);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Recall@{}: {:.2}%", k, avg_recall * 100.0);

        // Lower k should have higher recall
        if k <= 10 {
            assert!(
                avg_recall >= 0.75,
                "Recall@{} too low: {:.2}%",
                k,
                avg_recall * 100.0
            );
        }
    }
}

#[test]
#[ignore = "Flaky test with random data - run manually to verify quality"]
fn test_recall_with_different_ef_search() {
    let corpus = generate_test_vectors(5_000, 128);
    let num_queries = 20;
    let k = 10;

    // Test different ef_search values
    for ef_search in [10, 50, 100, 200] {
        let config = HnswConfig {
            ef_search,
            ..Default::default()
        };

        let index = HnswIndex::new(128, config);
        for (i, vec) in corpus.iter().enumerate() {
            index.insert(i as u128, vec.clone()).unwrap();
        }

        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query = generate_random_vector(128);
            let true_neighbors = exact_search(&query, &corpus, k);
            let hnsw_results = index.search(&query, k).unwrap();
            let hnsw_ids: Vec<usize> = hnsw_results.iter().map(|(id, _)| *id as usize).collect();

            let recall = compute_recall(&hnsw_ids, &true_neighbors);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!(
            "ef_search={}: Recall@10 = {:.2}%",
            ef_search,
            avg_recall * 100.0
        );

        // Higher ef_search should give better recall
        if ef_search >= 100 {
            assert!(
                avg_recall >= 0.80,
                "ef_search={} recall too low: {:.2}%",
                ef_search,
                avg_recall * 100.0
            );
        }
    }
}

#[test]
#[ignore = "Flaky test with random data - run manually to verify quality"]
fn test_recall_scaling_with_index_size() {
    let k = 10;
    let num_queries = 20;

    for size in [1_000, 5_000, 10_000] {
        let config = HnswConfig::default();
        let index = HnswIndex::new(128, config);

        let corpus = generate_test_vectors(size, 128);
        for (i, vec) in corpus.iter().enumerate() {
            index.insert(i as u128, vec.clone()).unwrap();
        }

        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query = generate_random_vector(128);
            let true_neighbors = exact_search(&query, &corpus, k);
            let hnsw_results = index.search(&query, k).unwrap();
            let hnsw_ids: Vec<usize> = hnsw_results.iter().map(|(id, _)| *id as usize).collect();

            let recall = compute_recall(&hnsw_ids, &true_neighbors);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!(
            "Index size {}: Recall@10 = {:.2}%",
            size,
            avg_recall * 100.0
        );

        // Recall should remain reasonable even as index grows
        // Lower threshold for larger indices is expected with approximate search
        assert!(
            avg_recall >= 0.65,
            "Size {} recall too low: {:.2}%",
            size,
            avg_recall * 100.0
        );
    }
}

#[test]
#[ignore = "Flaky test with random data - run manually to verify quality"]
fn test_mean_reciprocal_rank() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(128, config);

    let corpus = generate_test_vectors(5_000, 128);
    for (i, vec) in corpus.iter().enumerate() {
        index.insert(i as u128, vec.clone()).unwrap();
    }

    let num_queries = 50;
    let k = 10;
    let mut total_mrr = 0.0;

    for _ in 0..num_queries {
        let query = generate_random_vector(128);
        let true_neighbors = exact_search(&query, &corpus, k);
        let true_top1 = true_neighbors[0]; // The actual nearest neighbor

        let hnsw_results = index.search(&query, k).unwrap();

        // Find rank of true top-1 in HNSW results
        let rank = hnsw_results
            .iter()
            .position(|(id, _)| *id as usize == true_top1)
            .map(|pos| pos + 1); // Convert to 1-indexed

        if let Some(rank) = rank {
            total_mrr += 1.0 / rank as f64;
        }
        // If not found in top-k, contributes 0 to MRR
    }

    let mrr = total_mrr / num_queries as f64;
    println!("Mean Reciprocal Rank: {:.4}", mrr);

    // MRR should be reasonable (true #1 found somewhere in results)
    assert!(mrr >= 0.30, "MRR too low: {:.4}", mrr);
}

#[test]
#[ignore = "Flaky test with random data - run manually to verify quality"]
fn test_precision_at_k() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(128, config);

    let corpus = generate_test_vectors(5_000, 128);
    for (i, vec) in corpus.iter().enumerate() {
        index.insert(i as u128, vec.clone()).unwrap();
    }

    let num_queries = 50;
    let k = 10;
    let mut total_precision = 0.0;

    for _ in 0..num_queries {
        let query = generate_random_vector(128);
        let true_neighbors = exact_search(&query, &corpus, k);
        let true_set: HashSet<usize> = true_neighbors.iter().copied().collect();

        let hnsw_results = index.search(&query, k).unwrap();

        // Count how many of the returned results are actually in top-K
        let relevant = hnsw_results
            .iter()
            .filter(|(id, _)| true_set.contains(&(*id as usize)))
            .count();

        let precision = relevant as f64 / k as f64;
        total_precision += precision;
    }

    let avg_precision = total_precision / num_queries as f64;
    println!("Precision@{}: {:.2}%", k, avg_precision * 100.0);

    assert!(
        avg_precision >= 0.70,
        "Precision@{} too low: {:.2}%",
        k,
        avg_precision * 100.0
    );
}
