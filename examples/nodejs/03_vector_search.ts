/**
 * Vector Search Example
 * 
 * Demonstrates vector similarity search:
 * - Creating a vector index with HNSW
 * - Bulk loading embeddings
 * - Finding nearest neighbors
 * - Distance utilities
 */

import { VectorIndex, VectorIndexConfig, DistanceMetric } from '@sochdb/sochdb';

async function main() {
  // Configuration for the vector index
  const config: VectorIndexConfig = {
    dimension: 128,                      // Embedding dimension
    metric: DistanceMetric.Cosine,       // Cosine similarity
    m: 16,                               // HNSW connections per node
    efConstruction: 100,                 // Construction quality factor
  };

  const index = new VectorIndex('./vector_index', config);
  console.log('✓ Vector index created');

  // Generate sample embeddings (in practice, use a real embedding model)
  const vectors: number[][] = [];
  const labels: string[] = [];

  for (let i = 0; i < 100; i++) {
    // Create a simple pattern-based embedding for demo
    const vec: number[] = new Array(128);
    for (let j = 0; j < 128; j++) {
      vec[j] = ((i * j) % 256) / 255.0;
    }
    vectors.push(vec);
    labels.push(`document_${i}`);
  }

  // Bulk build the index
  await index.bulkBuild(vectors, labels);
  console.log(`✓ Indexed ${vectors.length} vectors`);

  // Create a query vector (similar to document_42)
  const query: number[] = new Array(128);
  for (let j = 0; j < 128; j++) {
    query[j] = ((42 * j) % 256) / 255.0;
  }

  // Search for nearest neighbors
  const results = await index.query(query, 5);  // k=5

  console.log('\n✓ Top 5 nearest neighbors:');
  results.forEach((result, i) => {
    console.log(`  ${i + 1}. ${result.label ?? '<no label>'} (distance: ${result.distance.toFixed(4)})`);
  });

  // Distance utility functions
  const a = [1.0, 0.0, 0.0];
  const b = [0.707, 0.707, 0.0];

  const cosineDist = VectorIndex.computeCosineDistance(a, b);
  const euclideanDist = VectorIndex.computeEuclideanDistance(a, b);

  console.log('\n✓ Distance calculations:');
  console.log(`  Cosine distance: ${cosineDist.toFixed(4)}`);
  console.log(`  Euclidean distance: ${euclideanDist.toFixed(4)}`);

  // Normalize a vector
  const v = [3.0, 4.0];
  const normalized = VectorIndex.normalizeVector(v);
  console.log(`  Normalized [3, 4]: [${normalized.map(x => x.toFixed(2)).join(', ')}]`);
}

main().catch(console.error);
