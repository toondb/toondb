//! Micro-benchmark for HNSW hot path operations
//! 
//! This directly measures the key operations without going through the full API.

use std::time::Instant;
use std::collections::{HashSet, BinaryHeap};
use std::cmp::Reverse;

const ITERATIONS: usize = 100_000;
const VEC_SIZE: usize = 384;

fn main() {
    println!("=== HNSW Micro-Benchmarks ===\n");
    
    // ========================================
    // 1. HashSet<u32> visited check
    // ========================================
    {
        let mut visited: HashSet<u32> = HashSet::with_capacity(1024);
        
        // Warmup
        for i in 0..1000u32 {
            visited.insert(i);
        }
        visited.clear();
        
        let start = Instant::now();
        for i in 0..ITERATIONS as u32 {
            visited.insert(i % 1000);
        }
        let elapsed = start.elapsed();
        println!("HashSet<u32> insert: {:?} / {} = {:?}/op", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32);
    }
    
    // ========================================
    // 2. BinaryHeap push/pop
    // ========================================
    {
        #[derive(Clone, Copy, PartialEq)]
        struct Candidate { distance: f32, id: u128 }
        impl Eq for Candidate {}
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.distance.partial_cmp(&self.distance).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(256);
        
        let start = Instant::now();
        for i in 0..ITERATIONS {
            let c = Candidate { distance: i as f32, id: i as u128 };
            heap.push(c);
            if heap.len() > 100 {
                heap.pop();
            }
        }
        let elapsed = start.elapsed();
        println!("BinaryHeap push/pop: {:?} / {} = {:?}/op", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32);
    }
    
    // ========================================
    // 3. Distance computation (L2 squared)
    // ========================================
    {
        let v1: Vec<f32> = (0..VEC_SIZE).map(|i| i as f32 * 0.001).collect();
        let v2: Vec<f32> = (0..VEC_SIZE).map(|i| (i + 100) as f32 * 0.001).collect();
        
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..ITERATIONS {
            sum += l2_squared(&v1, &v2);
        }
        let elapsed = start.elapsed();
        println!("L2 squared ({} dim): {:?} / {} = {:?}/op (sum={})", 
            VEC_SIZE, elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    // ========================================
    // 4. Dot product
    // ========================================
    {
        let v1: Vec<f32> = (0..VEC_SIZE).map(|i| i as f32 * 0.001).collect();
        let v2: Vec<f32> = (0..VEC_SIZE).map(|i| (i + 100) as f32 * 0.001).collect();
        
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..ITERATIONS {
            sum += dot_product(&v1, &v2);
        }
        let elapsed = start.elapsed();
        println!("Dot product ({} dim): {:?} / {} = {:?}/op (sum={})", 
            VEC_SIZE, elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    // ========================================
    // 5. RwLock read acquisition
    // ========================================
    {
        use parking_lot::RwLock;
        let lock: RwLock<Vec<u32>> = RwLock::new(vec![1, 2, 3, 4, 5]);
        
        let start = Instant::now();
        let mut sum = 0u64;
        for _ in 0..ITERATIONS {
            let guard = lock.read();
            sum += guard[0] as u64;
        }
        let elapsed = start.elapsed();
        println!("RwLock read: {:?} / {} = {:?}/op (sum={})", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    // ========================================
    // 6. DashMap get
    // ========================================
    {
        use dashmap::DashMap;
        let map: DashMap<u128, u32> = DashMap::new();
        for i in 0..1000u32 {
            map.insert(i as u128, i);
        }
        
        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..ITERATIONS {
            if let Some(v) = map.get(&((i % 1000) as u128)) {
                sum += *v as u64;
            }
        }
        let elapsed = start.elapsed();
        println!("DashMap get: {:?} / {} = {:?}/op (sum={})", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    // ========================================
    // 7. Vec index access (baseline)
    // ========================================
    {
        let vec: Vec<u32> = (0..1000).collect();
        
        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..ITERATIONS {
            sum += vec[i % 1000] as u64;
        }
        let elapsed = start.elapsed();
        println!("Vec index: {:?} / {} = {:?}/op (sum={})", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    // ========================================
    // 8. Arc clone
    // ========================================
    {
        use std::sync::Arc;
        let arc: Arc<Vec<f32>> = Arc::new(vec![1.0; VEC_SIZE]);
        
        let start = Instant::now();
        let mut sum = 0usize;
        for _ in 0..ITERATIONS {
            let cloned = Arc::clone(&arc);
            sum += cloned.len();
        }
        let elapsed = start.elapsed();
        println!("Arc clone: {:?} / {} = {:?}/op (sum={})", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    // ========================================
    // 9. thread_local access
    // ========================================
    {
        use std::cell::RefCell;
        thread_local! {
            static TLS: RefCell<u64> = RefCell::new(0);
        }
        
        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..ITERATIONS {
            TLS.with(|v| {
                *v.borrow_mut() += i as u64;
                sum = *v.borrow();
            });
        }
        let elapsed = start.elapsed();
        println!("thread_local access: {:?} / {} = {:?}/op (sum={})", 
            elapsed, ITERATIONS, elapsed / ITERATIONS as u32, sum);
    }
    
    println!("\n=== Benchmark Complete ===");
}

#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}
