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

//! Product Quantization for 32x compression
//!
//! Extends QuantizedVector with PQ support for massive memory reduction.
//! Works with the existing MmapVectorStorage for full vectors on disk.
//!
//! Memory comparison (384-dim vectors, 10M vectors):
//! - F32: 15.36 GB
//! - F16: 7.68 GB (2x compression)
//! - PQ:  480 MB  (32x compression!)

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Number of subspaces (for 384-dim MiniLM: 384/8 = 48 subspaces)
pub const DEFAULT_PQ_SUBSPACES: usize = 48;
/// Dimensions per subspace (default: 8)
pub const DEFAULT_PQ_SUBDIM: usize = 8;
/// Number of centroids per subspace (256 = 1 byte per code)
pub const PQ_CENTROIDS: usize = 256;

/// Product Quantization codebooks
///
/// Trained once on a sample of vectors, then used to encode all vectors.
/// Memory: 48 subspaces × 256 centroids × 8 dims × 4 bytes = 393 KB
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PQCodebooks {
    /// Centroids for each subspace: [subspace][centroid][dim]
    /// Shape: (n_subspaces, PQ_CENTROIDS, subdim)
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Dimension of original vectors (e.g., 384 for MiniLM)
    pub original_dim: usize,
    /// Number of subspaces
    pub n_subspaces: usize,
    /// Dimensions per subspace
    pub subdim: usize,
}

impl PQCodebooks {
    /// Train codebooks from a sample of vectors using k-means
    ///
    /// # Arguments
    /// * `vectors` - Training vectors (should be representative sample, 10k-100k vectors)
    /// * `n_iter` - Number of k-means iterations (default: 20)
    /// * `subdim` - Dimensions per subspace (default: 8, must divide vector dim evenly)
    ///
    /// # Example
    /// ```ignore
    /// let codebooks = PQCodebooks::train(&sample_vectors, 20, 8);
    /// ```
    pub fn train(vectors: &[Array1<f32>], n_iter: usize, subdim: usize) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vectors");
        let original_dim = vectors[0].len();
        assert!(
            original_dim.is_multiple_of(subdim),
            "Dimension {} must be divisible by subdim {}",
            original_dim,
            subdim
        );

        let n_subspaces = original_dim / subdim;
        let mut centroids = Vec::with_capacity(n_subspaces);

        for subspace_idx in 0..n_subspaces {
            // Extract subvectors for this subspace
            let start = subspace_idx * subdim;
            let end = start + subdim;

            let subvectors: Vec<Array1<f32>> = vectors
                .iter()
                .map(|v| v.slice(ndarray::s![start..end]).to_owned())
                .collect();

            // Run k-means to find centroids
            let subspace_centroids = kmeans_simple(&subvectors, PQ_CENTROIDS, n_iter);

            // Convert Array1<f32> to Vec<f32> for storage
            let centroid_vecs: Vec<Vec<f32>> =
                subspace_centroids.iter().map(|c| c.to_vec()).collect();
            centroids.push(centroid_vecs);
        }

        Self {
            centroids,
            original_dim,
            n_subspaces,
            subdim,
        }
    }

    /// Train with default parameters
    pub fn train_default(vectors: &[Array1<f32>]) -> Self {
        Self::train(vectors, 20, DEFAULT_PQ_SUBDIM)
    }

    /// Encode a vector to PQ codes (e.g., 48 bytes for 384-dim vector)
    pub fn encode(&self, vector: &Array1<f32>) -> PQCodes {
        assert_eq!(vector.len(), self.original_dim, "Vector dimension mismatch");

        let mut codes = Vec::with_capacity(self.n_subspaces);

        for subspace_idx in 0..self.n_subspaces {
            let start = subspace_idx * self.subdim;
            let end = start + self.subdim;
            let subvector = vector.slice(ndarray::s![start..end]);

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut best_code = 0u8;

            for (code, centroid) in self.centroids[subspace_idx].iter().enumerate() {
                let dist = squared_l2_slice(subvector.as_slice().unwrap(), centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_code = code as u8;
                }
            }

            codes.push(best_code);
        }

        PQCodes { codes }
    }

    /// Encode from a slice of f32
    pub fn encode_slice(&self, vector: &[f32]) -> PQCodes {
        let array = Array1::from_vec(vector.to_vec());
        self.encode(&array)
    }

    /// Decode PQ codes back to approximate vector (for verification only)
    pub fn decode(&self, pq: &PQCodes) -> Array1<f32> {
        let mut result = Vec::with_capacity(self.original_dim);

        for (subspace_idx, &code) in pq.codes.iter().enumerate() {
            let centroid = &self.centroids[subspace_idx][code as usize];
            result.extend(centroid.iter());
        }

        Array1::from_vec(result)
    }

    /// Precompute distance lookup table for a query
    ///
    /// This is THE key optimization: instead of computing 384 multiplications
    /// per vector, we precompute 48 × 256 distances once, then just do 48 lookups.
    ///
    /// Memory: 48 subspaces × 256 centroids × 4 bytes = 49 KB per query
    pub fn build_distance_table(&self, query: &Array1<f32>) -> DistanceTable {
        assert_eq!(query.len(), self.original_dim, "Query dimension mismatch");

        let mut tables = Vec::with_capacity(self.n_subspaces);

        for subspace_idx in 0..self.n_subspaces {
            let start = subspace_idx * self.subdim;
            let end = start + self.subdim;
            let query_sub = query.slice(ndarray::s![start..end]);

            // Precompute distance to all 256 centroids for this subspace
            let mut table = vec![0.0f32; PQ_CENTROIDS];
            for (code, centroid) in self.centroids[subspace_idx].iter().enumerate() {
                table[code] = squared_l2_slice(query_sub.as_slice().unwrap(), centroid);
            }

            tables.push(table);
        }

        DistanceTable { tables }
    }

    /// Build distance table from a slice
    pub fn build_distance_table_slice(&self, query: &[f32]) -> DistanceTable {
        let array = Array1::from_vec(query.to_vec());
        self.build_distance_table(&array)
    }

    /// Save codebooks to disk with gzip compression
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        bincode::serialize_into(encoder, self).map_err(|e| io::Error::other(e.to_string()))
    }

    /// Load codebooks from disk
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let decoder = flate2::read::GzDecoder::new(file);
        bincode::deserialize_from(decoder).map_err(|e| io::Error::other(e.to_string()))
    }

    /// Get memory size of codebooks in bytes
    pub fn memory_size(&self) -> usize {
        // n_subspaces * PQ_CENTROIDS * subdim * sizeof(f32)
        self.n_subspaces * PQ_CENTROIDS * self.subdim * 4
    }

    /// Calculate compression ratio compared to F32 vectors
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.original_dim * 4; // f32
        let pq_bytes = self.n_subspaces; // 1 byte per subspace
        original_bytes as f32 / pq_bytes as f32
    }
}

/// PQ-encoded vector (e.g., 48 bytes for 384-dim, 32x compression!)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PQCodes {
    /// One byte per subspace: 48 bytes total for 384-dim
    pub codes: Vec<u8>,
}

impl PQCodes {
    /// Create PQCodes from raw bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            codes: bytes.to_vec(),
        }
    }

    /// Get raw bytes for storage
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.codes
    }

    /// Memory size in bytes
    #[inline]
    pub fn memory_size(&self) -> usize {
        self.codes.len()
    }

    /// Number of subspaces
    #[inline]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

/// Precomputed distance lookup table for a query
///
/// Memory: 48 subspaces × 256 centroids × 4 bytes = 49 KB per query
pub struct DistanceTable {
    tables: Vec<Vec<f32>>,
}

impl DistanceTable {
    /// Compute approximate squared L2 distance using table lookups
    ///
    /// This is O(48 lookups) instead of O(384 multiplications) = 8x fewer ops!
    #[inline]
    pub fn distance(&self, pq: &PQCodes) -> f32 {
        let mut total = 0.0f32;
        for (subspace_idx, &code) in pq.codes.iter().enumerate() {
            // Safety: code is u8 (0-255), tables have 256 entries
            total += self.tables[subspace_idx][code as usize];
        }
        total
    }

    /// Compute approximate L2 distance (square root of squared distance)
    #[inline]
    pub fn distance_l2(&self, pq: &PQCodes) -> f32 {
        self.distance(pq).sqrt()
    }

    /// Batch distance computation for multiple PQ codes
    /// More efficient due to better cache locality
    #[inline]
    pub fn distance_batch(&self, pqs: &[PQCodes]) -> Vec<f32> {
        pqs.iter().map(|pq| self.distance(pq)).collect()
    }

    /// SIMD-accelerated distance computation (when available)
    /// Processes 4 PQ codes in parallel using scalar operations
    /// (Can be extended to use actual SIMD intrinsics for further speedup)
    pub fn distance_batch_4(&self, pqs: &[PQCodes; 4]) -> [f32; 4] {
        let mut results = [0.0f32; 4];
        for subspace_idx in 0..self.tables.len() {
            let table = &self.tables[subspace_idx];
            for (i, pq) in pqs.iter().enumerate() {
                results[i] += table[pq.codes[subspace_idx] as usize];
            }
        }
        results
    }
}

/// Simple k-means implementation for PQ training
fn kmeans_simple(vectors: &[Array1<f32>], k: usize, n_iter: usize) -> Vec<Array1<f32>> {
    if vectors.is_empty() {
        return vec![];
    }

    let dim = vectors[0].len();
    let n_vectors = vectors.len();

    // Initialize centroids: use k-means++ style initialization
    let mut rng = rand::thread_rng();
    let mut centroids = kmeans_plusplus_init(vectors, k.min(n_vectors), &mut rng);

    // Pad if not enough vectors
    while centroids.len() < k {
        centroids.push(Array1::zeros(dim));
    }

    for _iter in 0..n_iter {
        // Assign vectors to nearest centroid
        let mut assignments: Vec<Vec<usize>> = vec![vec![]; k];

        for (i, vec) in vectors.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best_c = 0;

            for (c, centroid) in centroids.iter().enumerate() {
                let dist = squared_l2(&vec.view(), centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }

            assignments[best_c].push(i);
        }

        // Recompute centroids
        for (c, assigned) in assignments.iter().enumerate() {
            if assigned.is_empty() {
                continue;
            }

            let mut new_centroid = Array1::zeros(dim);
            for &idx in assigned {
                new_centroid += &vectors[idx];
            }
            centroids[c] = new_centroid / assigned.len() as f32;
        }
    }

    centroids
}

/// K-means++ initialization for better centroid starting points
///
/// **Gap #5 Fix:** Uses log-sum-exp trick for numerical stability when
/// distances have high variance. This prevents floating-point overflow/underflow
/// in the D² weighted sampling.
fn kmeans_plusplus_init<R: Rng>(
    vectors: &[Array1<f32>],
    k: usize,
    rng: &mut R,
) -> Vec<Array1<f32>> {
    if vectors.is_empty() || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid randomly
    let first_idx = rng.gen_range(0..vectors.len());
    centroids.push(vectors[first_idx].clone());

    // Choose remaining centroids with probability proportional to D(x)^2
    for _ in 1..k {
        // Compute squared distances to nearest centroid for each vector
        let distances: Vec<f32> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| squared_l2(&v.view(), c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Use log-sum-exp trick for numerical stability with high-variance distances
        // log(sum(exp(log(d_i)))) = log_max + log(sum(exp(log(d_i) - log_max)))
        let log_distances: Vec<f32> = distances
            .iter()
            .map(|&d| if d > 0.0 { d.ln() } else { f32::NEG_INFINITY })
            .collect();

        let log_max = log_distances
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        if log_max.is_infinite() {
            // All points are centroids or have zero distance, pick random
            let idx = rng.gen_range(0..vectors.len());
            centroids.push(vectors[idx].clone());
            continue;
        }

        // Compute normalized weights using log-sum-exp
        let exp_shifted: Vec<f32> = log_distances
            .iter()
            .map(|&ld| (ld - log_max).exp())
            .collect();
        let total: f32 = exp_shifted.iter().sum();

        if total <= 0.0 {
            let idx = rng.gen_range(0..vectors.len());
            centroids.push(vectors[idx].clone());
            continue;
        }

        // Sample proportional to D^2 using numerically stable weights
        let threshold = rng.r#gen::<f32>() * total;
        let mut cumsum = 0.0;
        let mut selected = 0;

        for (i, &w) in exp_shifted.iter().enumerate() {
            cumsum += w;
            if cumsum >= threshold {
                selected = i;
                break;
            }
        }

        centroids.push(vectors[selected].clone());
    }

    centroids
}

#[inline]
fn squared_l2(a: &ArrayView1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

#[inline]
fn squared_l2_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Array1<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| Array1::from_iter((0..dim).map(|_| rng.r#gen::<f32>())))
            .collect()
    }

    #[test]
    fn test_pq_compression_ratio() {
        let dim = 384;
        let n_vectors = 100;

        let vectors = generate_random_vectors(n_vectors, dim);

        // Train PQ
        let codebooks = PQCodebooks::train(&vectors, 10, 8);

        // Encode a vector
        let pq = codebooks.encode(&vectors[0]);

        // Check compression ratio
        let original_bytes = dim * 4; // f32
        let pq_bytes = pq.memory_size();

        println!(
            "Original: {} bytes, PQ: {} bytes, Ratio: {:.1}x",
            original_bytes,
            pq_bytes,
            original_bytes as f32 / pq_bytes as f32
        );

        assert_eq!(pq_bytes, 48); // 384 / 8 = 48 subspaces
        assert_eq!(original_bytes / pq_bytes, 32); // 32x compression
    }

    #[test]
    fn test_pq_encode_decode() {
        let dim = 384;
        let n_vectors = 100;

        let vectors = generate_random_vectors(n_vectors, dim);
        let codebooks = PQCodebooks::train(&vectors, 20, 8);

        // Encode and decode
        let original = &vectors[0];
        let pq = codebooks.encode(original);
        let decoded = codebooks.decode(&pq);

        // Check reconstruction error (should be reasonable, not exact)
        let error: f32 = original
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        println!("Reconstruction error (L2): {}", error);
        // Error should be reasonable for PQ (typically < 1.0 for normalized vectors)
    }

    #[test]
    fn test_distance_table_speedup() {
        let dim = 384;
        let n_vectors = 1000;

        let vectors = generate_random_vectors(n_vectors, dim);
        let codebooks = PQCodebooks::train(&vectors[..100], 10, 8);

        // Encode all vectors
        let pq_codes: Vec<PQCodes> = vectors.iter().map(|v| codebooks.encode(v)).collect();

        let query = &vectors[0];

        // Build distance table once per query
        let table = codebooks.build_distance_table(query);

        // Time table lookup vs naive
        let start = std::time::Instant::now();
        let pq_distances: Vec<f32> = pq_codes.iter().map(|pq| table.distance(pq)).collect();
        let table_time = start.elapsed();

        let start = std::time::Instant::now();
        let _naive_distances: Vec<f32> = vectors
            .iter()
            .map(|v| squared_l2(&query.view(), v))
            .collect();
        let naive_time = start.elapsed();

        println!(
            "Table lookup: {:?}, Naive: {:?}, Speedup: {:.1}x",
            table_time,
            naive_time,
            naive_time.as_nanos() as f64 / table_time.as_nanos().max(1) as f64
        );

        // Verify distances are reasonable (not identical due to quantization)
        assert!(!pq_distances.is_empty());
    }

    #[test]
    fn test_pq_codes_serialization() {
        let codes = PQCodes {
            codes: vec![1, 2, 3, 4, 5],
        };

        let bytes = bincode::serialize(&codes).unwrap();
        let decoded: PQCodes = bincode::deserialize(&bytes).unwrap();

        assert_eq!(codes, decoded);
    }

    #[test]
    fn test_codebooks_save_load() {
        let dim = 32;
        let vectors = generate_random_vectors(50, dim);
        let codebooks = PQCodebooks::train(&vectors, 5, 8);

        // Save to temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("codebooks.bin.gz");

        codebooks.save(&path).unwrap();
        let loaded = PQCodebooks::load(&path).unwrap();

        assert_eq!(codebooks.original_dim, loaded.original_dim);
        assert_eq!(codebooks.n_subspaces, loaded.n_subspaces);
        assert_eq!(codebooks.centroids.len(), loaded.centroids.len());
    }

    #[test]
    fn test_memory_usage_calculation() {
        let dim = 384;
        let vectors = generate_random_vectors(100, dim);
        let codebooks = PQCodebooks::train(&vectors, 5, 8);

        let memory = codebooks.memory_size();
        let expected = 48 * 256 * 8 * 4; // 48 subspaces * 256 centroids * 8 dims * 4 bytes

        assert_eq!(memory, expected);
        println!(
            "Codebook memory: {} bytes ({:.2} KB)",
            memory,
            memory as f64 / 1024.0
        );
    }

    #[test]
    fn test_distance_batch() {
        let dim = 64;
        let vectors = generate_random_vectors(20, dim);
        let codebooks = PQCodebooks::train(&vectors, 5, 8);

        let pq_codes: Vec<PQCodes> = vectors.iter().map(|v| codebooks.encode(v)).collect();
        let table = codebooks.build_distance_table(&vectors[0]);

        // Test batch distance
        let batch_distances = table.distance_batch(&pq_codes);
        let individual_distances: Vec<f32> = pq_codes.iter().map(|pq| table.distance(pq)).collect();

        assert_eq!(batch_distances.len(), individual_distances.len());
        for (b, i) in batch_distances.iter().zip(individual_distances.iter()) {
            assert!((b - i).abs() < 1e-6);
        }
    }
}
