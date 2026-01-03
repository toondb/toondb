#!/usr/bin/env python3
"""
ToonDB Recall Benchmark: Measures search quality with real embeddings

This benchmark computes recall@k by comparing ToonDB's HNSW approximate search
against brute-force exact search (ground truth).

Usage:
    source .venv312/bin/activate
    TOONDB_LIB_PATH=target/release python3 benchmarks/recall_benchmark.py
"""

import os
import sys
import time
import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Embedding Client
# =============================================================================

class EmbeddingClient:
    """Azure OpenAI embedding client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        self.dimension = 1536
    
    def embed_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed texts in batches."""
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/embeddings?api-version={self.version}"
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json={"input": batch, "model": self.deployment},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"   Error on batch {i//batch_size}: {e}")
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Embedded {i + len(batch)}/{len(texts)} texts...")
        
        return np.array(all_embeddings, dtype=np.float32)


# =============================================================================
# Ground Truth Computation (Brute Force)
# =============================================================================

def compute_ground_truth(
    corpus_embeddings: np.ndarray, 
    query_embeddings: np.ndarray, 
    k: int
) -> Dict[int, List[int]]:
    """
    Compute exact nearest neighbors using brute-force cosine similarity.
    Returns {query_id: [doc_id1, doc_id2, ...]} for top-k.
    """
    # Normalize for cosine similarity
    corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    ground_truth = {}
    
    for q_idx, query in enumerate(query_norm):
        # Compute cosine similarities
        similarities = np.dot(corpus_norm, query)
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        ground_truth[q_idx] = top_k_indices.tolist()
    
    return ground_truth


def compute_recall(retrieved: List[int], ground_truth: List[int], k: int) -> float:
    """Compute recall@k: fraction of ground truth in top-k retrieved."""
    gt_set = set(ground_truth[:k])
    retrieved_set = set(retrieved[:k])
    if len(gt_set) == 0:
        return 1.0
    return len(gt_set & retrieved_set) / len(gt_set)


# =============================================================================
# Document Generator
# =============================================================================

TOPICS = [
    "machine learning", "database optimization", "cloud computing", 
    "API design", "security best practices", "performance tuning",
    "microservices architecture", "data pipelines", "monitoring",
    "error handling", "testing strategies", "deployment automation",
    "vector search", "nearest neighbors", "embedding models",
    "transformer architectures", "attention mechanisms", "fine-tuning"
]

TEMPLATES = [
    "How to implement {topic} in production systems with best practices",
    "Troubleshooting {topic} issues and common error patterns",
    "{topic} comparison: approaches and trade-offs for enterprise",
    "Getting started with {topic}: a comprehensive guide",
    "Advanced {topic} techniques for high-scale applications",
    "Best practices for {topic} in distributed systems",
    "{topic} optimization strategies for reducing latency",
    "Migrating legacy systems to modern {topic} patterns",
]

def generate_documents(count: int) -> List[str]:
    """Generate realistic document texts."""
    docs = []
    for i in range(count):
        topic = random.choice(TOPICS)
        template = random.choice(TEMPLATES)
        base = template.format(topic=topic)
        suffix = f" - Document {i} covering {random.choice(['beginner', 'intermediate', 'advanced'])} level content."
        docs.append(base + suffix)
    return docs


# =============================================================================
# ToonDB Adapter
# =============================================================================

class ToonDBRecallBenchmark:
    """ToonDB benchmark with different HNSW configurations."""
    
    def __init__(self, dimension: int, m: int = 16, ef_construction: int = 100):
        from toondb import VectorIndex
        self.m = m
        self.ef_construction = ef_construction
        self.index = VectorIndex(
            dimension=dimension, 
            max_connections=m, 
            ef_construction=ef_construction
        )
        self.count = 0
    
    def insert(self, embeddings: np.ndarray):
        ids = np.arange(self.count, self.count + len(embeddings), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        self.count += len(embeddings)
    
    def search(self, query: np.ndarray, k: int) -> List[int]:
        # Note: ef_search is set internally based on ef_construction
        results = list(self.index.search(query, k=k))
        return [int(doc_id) for doc_id, _ in results]
    
    def config_str(self) -> str:
        return f"M={self.m}, ef_c={self.ef_construction}"


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("="*70)
    print("  TOONDB RECALL BENCHMARK")
    print("  Measuring Search Quality with Real Embeddings")
    print("="*70)
    
    NUM_DOCS = 1000  # Corpus size
    NUM_QUERIES = 100  # Number of queries
    K_VALUES = [1, 5, 10, 20, 50]  # recall@k values to compute
    
    # HNSW configurations to test (varying M and ef_construction)
    CONFIGS = [
        {"m": 8,  "ef_construction": 50},   # Fast, lower quality
        {"m": 16, "ef_construction": 100},  # Balanced
        {"m": 16, "ef_construction": 200},  # Higher quality
        {"m": 32, "ef_construction": 200},  # Maximum quality
        {"m": 32, "ef_construction": 400},  # Ultra quality
    ]
    
    # Initialize embedding client
    embed_client = EmbeddingClient()
    
    # Generate documents
    print(f"\n1. Generating {NUM_DOCS} documents and {NUM_QUERIES} queries...")
    docs = generate_documents(NUM_DOCS)
    queries = generate_documents(NUM_QUERIES)
    print(f"   ✓ Generated {len(docs)} docs and {len(queries)} queries")
    
    # Embed documents
    print(f"\n2. Embedding {NUM_DOCS} documents...")
    doc_embeddings = embed_client.embed_batch(docs)
    print(f"   ✓ Embedded {len(doc_embeddings)} documents")
    
    # Embed queries
    print(f"\n3. Embedding {NUM_QUERIES} queries...")
    query_embeddings = embed_client.embed_batch(queries)
    print(f"   ✓ Embedded {len(query_embeddings)} queries")
    
    # Compute ground truth
    print(f"\n4. Computing ground truth (brute-force exact search)...")
    max_k = max(K_VALUES)
    ground_truth = compute_ground_truth(doc_embeddings, query_embeddings, max_k)
    print(f"   ✓ Computed top-{max_k} ground truth for {len(queries)} queries")
    
    # Benchmark each configuration
    print(f"\n5. Benchmarking ToonDB with different HNSW configurations...")
    print("-"*70)
    
    all_results = []
    
    for config in CONFIGS:
        print(f"\n   Configuration: M={config['m']}, ef_construction={config['ef_construction']}")
        
        # Create index
        index = ToonDBRecallBenchmark(
            dimension=embed_client.dimension,
            m=config["m"],
            ef_construction=config["ef_construction"]
        )
        
        # Insert documents
        start = time.perf_counter()
        index.insert(doc_embeddings)
        insert_time = (time.perf_counter() - start) * 1000
        print(f"   Insert time: {insert_time:.1f}ms")
        
        # Search and compute recall
        recalls = {k: [] for k in K_VALUES}
        search_times = []
        
        for q_idx, query in enumerate(query_embeddings):
            start = time.perf_counter()
            retrieved = index.search(query, max_k)
            search_times.append((time.perf_counter() - start) * 1000)
            
            for k in K_VALUES:
                recall = compute_recall(retrieved, ground_truth[q_idx], k)
                recalls[k].append(recall)
        
        # Aggregate metrics
        avg_search_time = np.mean(search_times)
        
        result = {
            "config": config,
            "insert_time_ms": insert_time,
            "avg_search_time_ms": avg_search_time,
            "recalls": {}
        }
        
        print(f"   Avg search time: {avg_search_time:.3f}ms")
        print(f"   Recall metrics:")
        
        for k in K_VALUES:
            mean_recall = np.mean(recalls[k])
            min_recall = np.min(recalls[k])
            max_recall = np.max(recalls[k])
            result["recalls"][f"recall@{k}"] = {
                "mean": mean_recall,
                "min": min_recall,
                "max": max_recall
            }
            print(f"     recall@{k}: {mean_recall:.4f} (min={min_recall:.4f}, max={max_recall:.4f})")
        
        all_results.append(result)
    
    # Final summary table
    print(f"\n" + "="*70)
    print("  RECALL BENCHMARK SUMMARY")
    print("="*70)
    
    header = f"{'Configuration':<35} {'Search (ms)':<12} " + " ".join([f"R@{k:<3}" for k in K_VALUES])
    print(f"\n  {header}")
    print(f"  {'-'*(len(header)+5)}")
    
    for result in all_results:
        config = result["config"]
        config_str = f"M={config['m']:2}, ef_c={config['ef_construction']:3}"
        search_time = f"{result['avg_search_time_ms']:.2f}"
        recalls = " ".join([f"{result['recalls'][f'recall@{k}']['mean']:.3f}" for k in K_VALUES])
        print(f"  {config_str:<35} {search_time:<12} {recalls}")
    
    # Best configs
    print(f"\n  📊 ANALYSIS")
    print(f"  {'-'*50}")
    
    # Find best recall@10 config
    best_r10 = max(all_results, key=lambda x: x["recalls"]["recall@10"]["mean"])
    print(f"  Best recall@10: {best_r10['recalls']['recall@10']['mean']:.4f} (M={best_r10['config']['m']}, ef_c={best_r10['config']['ef_construction']})")
    
    # Find fastest config with recall@10 > 0.95
    fast_high_quality = [r for r in all_results if r["recalls"]["recall@10"]["mean"] >= 0.95]
    if fast_high_quality:
        fastest = min(fast_high_quality, key=lambda x: x["avg_search_time_ms"])
        print(f"  Fastest with R@10≥95%: {fastest['avg_search_time_ms']:.2f}ms (M={fastest['config']['m']}, ef_c={fastest['config']['ef_construction']})")
    
    # Speed vs quality tradeoff
    print(f"\n  🎯 KEY INSIGHTS")
    print(f"  {'-'*50}")
    print(f"  • Higher ef_construction improves recall but increases build time")
    print(f"  • M=16 with ef_c=200 typically gives >95% recall@10")
    print(f"  • For real-time apps, M=16/ef_c=100 balances speed & quality")
    
    print("="*70)
    
    # Save results
    output_file = "benchmarks/reports/recall_benchmark_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "corpus_size": NUM_DOCS,
            "num_queries": NUM_QUERIES,
            "dimension": embed_client.dimension,
            "results": all_results
        }, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
