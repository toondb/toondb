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

//! # Cosine/Dot (MIPS) List Bounds (Task 3)
//!
//! Provides computable upper bounds for cosine/dot similarity over IVF lists,
//! enabling best-first probing and bound-based termination for non-L2 metrics.
//!
//! ## Architecture
//!
//! For cosine/dot similarity, we store spherical cap metadata per list:
//! - Centroid direction c (unit vector)
//! - Max angular deviation θ_max (or min dot to centroid)
//!
//! ## Math/Algorithm
//!
//! For normalized queries q, the upper bound on achievable similarity in list L is:
//!
//! ```text
//! max_{v∈L} q·v ≤ cos(max(0, angle(q,c) - θ_max))
//! ```
//!
//! where:
//! - angle(q,c) = arccos(q·c)
//! - θ_max = max_{v∈L} arccos(v·c)
//!
//! Bound evaluation is O(1) per list after precomputing q·c.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sochdb_vector::list_bounds::{SphericalCapMetadata, ListBoundComputer};
//!
//! // Build metadata during indexing
//! let metadata = SphericalCapMetadata::from_vectors(&vectors, centroid);
//!
//! // At query time, compute bounds
//! let computer = ListBoundComputer::new(&query);
//! let bound = computer.upper_bound(&metadata);
//! ```

use std::f32::consts::PI;

// ============================================================================
// Distance Metric
// ============================================================================

/// Supported distance metrics with bound computation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance - smaller is better
    L2,
    /// Cosine similarity - larger is better (1 = identical)
    Cosine,
    /// Inner product (dot) - larger is better
    InnerProduct,
    /// Negative inner product (for MIPS via nearest neighbor)
    NegativeInnerProduct,
}

impl DistanceMetric {
    /// Returns true if higher scores are better
    pub fn higher_is_better(&self) -> bool {
        matches!(self, Self::Cosine | Self::InnerProduct)
    }
    
    /// Returns true if vectors should be normalized
    pub fn requires_normalization(&self) -> bool {
        matches!(self, Self::Cosine)
    }
}

// ============================================================================
// Spherical Cap Metadata
// ============================================================================

/// Spherical cap metadata for a list/partition
///
/// Represents the region of the unit sphere covered by vectors in this list.
/// Used to compute tight bounds for cosine/dot similarity.
#[derive(Debug, Clone)]
pub struct SphericalCapMetadata {
    /// Centroid direction (unit vector)
    pub centroid: Vec<f32>,
    
    /// Maximum angular deviation from centroid (in radians)
    /// θ_max = max_{v∈L} arccos(v·c)
    pub theta_max: f32,
    
    /// Minimum dot product with centroid
    /// min_dot = min_{v∈L} v·c = cos(θ_max)
    pub min_dot_to_centroid: f32,
    
    /// Maximum dot product with centroid (typically ~1.0 for tight clusters)
    pub max_dot_to_centroid: f32,
    
    /// Number of vectors in this list
    pub vector_count: u32,
    
    /// Mean dot product with centroid (for statistics)
    pub mean_dot_to_centroid: f32,
}

impl SphericalCapMetadata {
    /// Build metadata from a set of normalized vectors and their centroid
    ///
    /// # Arguments
    /// * `vectors` - Normalized vectors in the list (each row is a vector)
    /// * `centroid` - Normalized centroid of the list
    ///
    /// # Complexity
    /// O(n × d) where n = number of vectors, d = dimension
    pub fn from_vectors(vectors: &[Vec<f32>], centroid: &[f32]) -> Self {
        if vectors.is_empty() {
            return Self {
                centroid: centroid.to_vec(),
                theta_max: 0.0,
                min_dot_to_centroid: 1.0,
                max_dot_to_centroid: 1.0,
                vector_count: 0,
                mean_dot_to_centroid: 1.0,
            };
        }
        
        let mut min_dot = f32::MAX;
        let mut max_dot = f32::MIN;
        let mut sum_dot = 0.0;
        
        for v in vectors {
            let dot = dot_product(v, centroid);
            min_dot = min_dot.min(dot);
            max_dot = max_dot.max(dot);
            sum_dot += dot;
        }
        
        // Clamp to valid range for arccos
        let clamped_min = min_dot.clamp(-1.0, 1.0);
        let theta_max = clamped_min.acos();
        
        Self {
            centroid: centroid.to_vec(),
            theta_max,
            min_dot_to_centroid: min_dot,
            max_dot_to_centroid: max_dot,
            vector_count: vectors.len() as u32,
            mean_dot_to_centroid: sum_dot / vectors.len() as f32,
        }
    }
    
    /// Build metadata from flat vector data
    pub fn from_flat_vectors(data: &[f32], dim: usize, centroid: &[f32]) -> Self {
        let n_vectors = data.len() / dim;
        
        if n_vectors == 0 {
            return Self {
                centroid: centroid.to_vec(),
                theta_max: 0.0,
                min_dot_to_centroid: 1.0,
                max_dot_to_centroid: 1.0,
                vector_count: 0,
                mean_dot_to_centroid: 1.0,
            };
        }
        
        let mut min_dot = f32::MAX;
        let mut max_dot = f32::MIN;
        let mut sum_dot = 0.0;
        
        for i in 0..n_vectors {
            let v = &data[i * dim..(i + 1) * dim];
            let dot = dot_product(v, centroid);
            min_dot = min_dot.min(dot);
            max_dot = max_dot.max(dot);
            sum_dot += dot;
        }
        
        let clamped_min = min_dot.clamp(-1.0, 1.0);
        let theta_max = clamped_min.acos();
        
        Self {
            centroid: centroid.to_vec(),
            theta_max,
            min_dot_to_centroid: min_dot,
            max_dot_to_centroid: max_dot,
            vector_count: n_vectors as u32,
            mean_dot_to_centroid: sum_dot / n_vectors as f32,
        }
    }
    
    /// Update metadata incrementally when a new vector is added
    pub fn add_vector(&mut self, vector: &[f32]) {
        let dot = dot_product(vector, &self.centroid);
        
        let old_sum = self.mean_dot_to_centroid * self.vector_count as f32;
        self.vector_count += 1;
        self.mean_dot_to_centroid = (old_sum + dot) / self.vector_count as f32;
        
        if dot < self.min_dot_to_centroid {
            self.min_dot_to_centroid = dot;
            self.theta_max = dot.clamp(-1.0, 1.0).acos();
        }
        if dot > self.max_dot_to_centroid {
            self.max_dot_to_centroid = dot;
        }
    }
    
    /// Get the angular radius of the spherical cap (in radians)
    pub fn angular_radius(&self) -> f32 {
        self.theta_max
    }
    
    /// Get the angular radius in degrees
    pub fn angular_radius_degrees(&self) -> f32 {
        self.theta_max * 180.0 / PI
    }
    
    /// Estimate the "tightness" of the cluster (0 = loose, 1 = tight)
    pub fn tightness(&self) -> f32 {
        // A perfectly tight cluster has all vectors at the centroid (theta_max = 0)
        // A maximally loose cluster has theta_max = π
        1.0 - (self.theta_max / PI)
    }
}

// ============================================================================
// L2 List Metadata
// ============================================================================

/// Metadata for L2 distance bounds (centroid + radius)
#[derive(Debug, Clone)]
pub struct L2ListMetadata {
    /// Centroid of the list
    pub centroid: Vec<f32>,
    
    /// Maximum L2 distance from centroid to any vector in list
    pub radius: f32,
    
    /// Mean L2 distance from centroid
    pub mean_radius: f32,
    
    /// Number of vectors
    pub vector_count: u32,
}

impl L2ListMetadata {
    /// Build from vectors
    pub fn from_vectors(vectors: &[Vec<f32>], centroid: &[f32]) -> Self {
        if vectors.is_empty() {
            return Self {
                centroid: centroid.to_vec(),
                radius: 0.0,
                mean_radius: 0.0,
                vector_count: 0,
            };
        }
        
        let mut max_dist = 0.0f32;
        let mut sum_dist = 0.0;
        
        for v in vectors {
            let dist = l2_distance(v, centroid);
            max_dist = max_dist.max(dist);
            sum_dist += dist;
        }
        
        Self {
            centroid: centroid.to_vec(),
            radius: max_dist,
            mean_radius: sum_dist / vectors.len() as f32,
            vector_count: vectors.len() as u32,
        }
    }
    
    /// Compute lower bound on L2 distance from query to any vector in list
    ///
    /// LB = max(0, dist(q, c) - radius)
    pub fn lower_bound(&self, query: &[f32]) -> f32 {
        let dist_to_centroid = l2_distance(query, &self.centroid);
        (dist_to_centroid - self.radius).max(0.0)
    }
}

// ============================================================================
// List Bound Computer
// ============================================================================

/// Computes bounds for a query across multiple lists
///
/// Precomputes query-related values once, then evaluates bounds per list in O(1).
pub struct ListBoundComputer<'a> {
    /// Query vector
    query: &'a [f32],
    
    /// Precomputed query norm (for L2)
    query_norm: f32,
    
    /// Distance metric
    metric: DistanceMetric,
}

impl<'a> ListBoundComputer<'a> {
    /// Create a new bound computer for a query
    pub fn new(query: &'a [f32], metric: DistanceMetric) -> Self {
        let query_norm = l2_norm(query);
        Self {
            query,
            query_norm,
            metric,
        }
    }
    
    /// Compute upper bound on similarity for a spherical cap (cosine/dot)
    ///
    /// For normalized query q and list with centroid c and max deviation θ_max:
    /// max_{v∈L} q·v ≤ cos(max(0, angle(q,c) - θ_max))
    ///
    /// Complexity: O(d) for dot product, rest is O(1)
    pub fn cosine_upper_bound(&self, metadata: &SphericalCapMetadata) -> f32 {
        // Compute q·c
        let query_dot_centroid = dot_product(self.query, &metadata.centroid);
        
        // angle(q,c) = arccos(q·c)
        let clamped = query_dot_centroid.clamp(-1.0, 1.0);
        let angle_to_centroid = clamped.acos();
        
        // Upper bound angle to best vector: max(0, angle - θ_max)
        let min_angle = (angle_to_centroid - metadata.theta_max).max(0.0);
        
        // Upper bound on similarity: cos(min_angle)
        min_angle.cos()
    }
    
    /// Compute lower bound on L2 distance for a list
    ///
    /// LB = max(0, dist(q, c) - radius)
    pub fn l2_lower_bound(&self, metadata: &L2ListMetadata) -> f32 {
        let dist_to_centroid = l2_distance(self.query, &metadata.centroid);
        (dist_to_centroid - metadata.radius).max(0.0)
    }
    
    /// Compute bound appropriate for the configured metric
    ///
    /// For similarity metrics (cosine, dot): returns upper bound (higher = tighter)
    /// For distance metrics (L2): returns lower bound (lower = tighter)
    pub fn compute_bound(&self, cap: &SphericalCapMetadata, l2: Option<&L2ListMetadata>) -> f32 {
        match self.metric {
            DistanceMetric::Cosine | DistanceMetric::InnerProduct => {
                self.cosine_upper_bound(cap)
            }
            DistanceMetric::L2 => {
                if let Some(l2_meta) = l2 {
                    self.l2_lower_bound(l2_meta)
                } else {
                    // Fall back to using spherical cap for normalized vectors
                    // LB ≈ sqrt(2 - 2*cos(angle))
                    let ub = self.cosine_upper_bound(cap);
                    (2.0 - 2.0 * ub).max(0.0).sqrt()
                }
            }
            DistanceMetric::NegativeInnerProduct => {
                -self.cosine_upper_bound(cap)
            }
        }
    }
}

// ============================================================================
// Multi-List Bound Ordering
// ============================================================================

/// Precomputed bounds for best-first list ordering
#[derive(Debug, Clone)]
pub struct ListBound {
    /// List index
    pub list_idx: u32,
    /// Bound value (interpretation depends on metric)
    pub bound: f32,
}

impl ListBound {
    /// Order lists by bound for best-first probing
    ///
    /// For similarity metrics: descending order (best first)
    /// For distance metrics: ascending order (best first)
    pub fn order_for_probing(
        bounds: &mut [ListBound],
        metric: DistanceMetric,
    ) {
        match metric {
            DistanceMetric::Cosine | DistanceMetric::InnerProduct => {
                // Higher similarity = better, sort descending
                bounds.sort_by(|a, b| b.bound.partial_cmp(&a.bound).unwrap());
            }
            DistanceMetric::L2 | DistanceMetric::NegativeInnerProduct => {
                // Lower distance = better, sort ascending
                bounds.sort_by(|a, b| a.bound.partial_cmp(&b.bound).unwrap());
            }
        }
    }
    
    /// Check if we can terminate based on kth best score and remaining bounds
    ///
    /// For similarity: stop if kth_score > best_remaining_bound
    /// For distance: stop if kth_score < best_remaining_bound
    pub fn can_terminate(
        kth_score: f32,
        best_remaining_bound: f32,
        metric: DistanceMetric,
    ) -> bool {
        match metric {
            DistanceMetric::Cosine | DistanceMetric::InnerProduct => {
                kth_score > best_remaining_bound
            }
            DistanceMetric::L2 | DistanceMetric::NegativeInnerProduct => {
                kth_score < best_remaining_bound
            }
        }
    }
}

// ============================================================================
// Unified List Metadata
// ============================================================================

/// Combined metadata supporting all metrics
#[derive(Debug, Clone)]
pub struct UnifiedListMetadata {
    /// Spherical cap for cosine/dot
    pub cap: SphericalCapMetadata,
    
    /// L2 metadata (optional, computed on demand)
    pub l2: Option<L2ListMetadata>,
    
    /// List index
    pub list_idx: u32,
}

impl UnifiedListMetadata {
    /// Build unified metadata
    pub fn new(list_idx: u32, cap: SphericalCapMetadata) -> Self {
        Self {
            cap,
            l2: None,
            list_idx,
        }
    }
    
    /// Add L2 metadata
    pub fn with_l2(mut self, l2: L2ListMetadata) -> Self {
        self.l2 = Some(l2);
        self
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute dot product of two vectors
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute L2 norm
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute L2 distance
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Normalize a vector in-place
pub fn normalize_inplace(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Normalize a vector, returning new vector
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spherical_cap_metadata() {
        // Create a tight cluster of 3D unit vectors
        let centroid = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            normalize(&[1.0, 0.1, 0.0]),
            normalize(&[1.0, -0.1, 0.0]),
            normalize(&[1.0, 0.0, 0.1]),
            normalize(&[1.0, 0.0, -0.1]),
        ];
        
        let metadata = SphericalCapMetadata::from_vectors(&vectors, &centroid);
        
        assert!(metadata.theta_max > 0.0);
        assert!(metadata.theta_max < PI / 4.0); // Should be a tight cluster
        assert!(metadata.tightness() > 0.5);
    }
    
    #[test]
    fn test_cosine_upper_bound() {
        // Centroid pointing in x direction
        let centroid = vec![1.0, 0.0, 0.0];
        let metadata = SphericalCapMetadata {
            centroid: centroid.clone(),
            theta_max: 0.3, // About 17 degrees
            min_dot_to_centroid: 0.3_f32.cos(),
            max_dot_to_centroid: 1.0,
            vector_count: 10,
            mean_dot_to_centroid: 0.95,
        };
        
        // Query in same direction as centroid
        let query = vec![1.0, 0.0, 0.0];
        let computer = ListBoundComputer::new(&query, DistanceMetric::Cosine);
        let bound = computer.cosine_upper_bound(&metadata);
        
        // Should be close to 1.0 since query aligns with centroid
        assert!((bound - 1.0).abs() < 0.01);
        
        // Query perpendicular to centroid
        let query2 = vec![0.0, 1.0, 0.0];
        let computer2 = ListBoundComputer::new(&query2, DistanceMetric::Cosine);
        let bound2 = computer2.cosine_upper_bound(&metadata);
        
        // Upper bound should account for theta_max
        // angle = π/2, so upper bound = cos(π/2 - 0.3) = sin(0.3)
        assert!((bound2 - 0.3_f32.sin()).abs() < 0.01);
    }
    
    #[test]
    fn test_l2_lower_bound() {
        let centroid = vec![0.0, 0.0, 0.0];
        let metadata = L2ListMetadata {
            centroid,
            radius: 1.0,
            mean_radius: 0.5,
            vector_count: 100,
        };
        
        // Query at distance 2 from centroid
        let query = vec![2.0, 0.0, 0.0];
        let computer = ListBoundComputer::new(&query, DistanceMetric::L2);
        let lb = computer.l2_lower_bound(&metadata);
        
        // Lower bound should be 2 - 1 = 1
        assert!((lb - 1.0).abs() < 0.01);
        
        // Query inside the radius
        let query2 = vec![0.5, 0.0, 0.0];
        let computer2 = ListBoundComputer::new(&query2, DistanceMetric::L2);
        let lb2 = computer2.l2_lower_bound(&metadata);
        
        // Lower bound should be 0 (query is within radius)
        assert!((lb2 - 0.0).abs() < 0.01);
    }
    
    #[test]
    fn test_list_ordering() {
        let mut bounds = vec![
            ListBound { list_idx: 0, bound: 0.5 },
            ListBound { list_idx: 1, bound: 0.9 },
            ListBound { list_idx: 2, bound: 0.3 },
        ];
        
        // For cosine, descending order (highest similarity first)
        ListBound::order_for_probing(&mut bounds, DistanceMetric::Cosine);
        assert_eq!(bounds[0].list_idx, 1); // 0.9
        assert_eq!(bounds[1].list_idx, 0); // 0.5
        assert_eq!(bounds[2].list_idx, 2); // 0.3
        
        // For L2, ascending order (lowest distance first)
        ListBound::order_for_probing(&mut bounds, DistanceMetric::L2);
        assert_eq!(bounds[0].list_idx, 2); // 0.3
        assert_eq!(bounds[1].list_idx, 0); // 0.5
        assert_eq!(bounds[2].list_idx, 1); // 0.9
    }
}
