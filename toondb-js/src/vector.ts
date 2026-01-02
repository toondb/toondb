/**
 * ToonDB Vector Index
 *
 * HNSW vector search support for ToonDB.
 *
 * @packageDocumentation
 */

// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { DatabaseError } from './errors';

/**
 * Vector search result.
 */
export interface VectorSearchResult {
  /** ID of the vector */
  id: number;
  /** Distance from query vector */
  distance: number;
  /** Optional label/metadata associated with the vector */
  label?: string;
}

/**
 * Configuration for vector index.
 */
export interface VectorIndexConfig {
  /** Number of dimensions */
  dimension: number;
  /** Maximum connections per node (default: 16) */
  m?: number;
  /** Construction beam width (default: 100) */
  efConstruction?: number;
  /** Distance metric: 'cosine' | 'euclidean' | 'dot' (default: 'cosine') */
  metric?: 'cosine' | 'euclidean' | 'dot';
}

/**
 * Bulk build statistics.
 */
export interface BulkBuildStats {
  /** Number of vectors indexed */
  vectors: number;
  /** Build time in seconds */
  buildTimeSeconds: number;
  /** Throughput in vectors/second */
  rate: number;
}

/**
 * Get the path to the toondb-bulk binary.
 */
function findBulkBinary(): string {
  const platform = process.platform;
  const arch = process.arch;

  let target: string;
  if (platform === 'darwin') {
    target = arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
  } else if (platform === 'win32') {
    target = 'x86_64-pc-windows-msvc';
  } else {
    target = arch === 'arm64' ? 'aarch64-unknown-linux-gnu' : 'x86_64-unknown-linux-gnu';
  }

  const binaryName = platform === 'win32' ? 'toondb-bulk.exe' : 'toondb-bulk';

  // Search paths - check multiple levels for different build outputs (cjs, esm)
  const searchPaths = [
    // Bundled in package - from dist/cjs or dist/esm
    path.join(__dirname, '..', '..', '_bin', target, binaryName),
    path.join(__dirname, '..', '_bin', target, binaryName),
    path.join(__dirname, '_bin', target, binaryName),
    // From package root
    path.resolve(__dirname, '..', '..', '..', '_bin', target, binaryName),
    // Development paths
    path.join(__dirname, '..', '..', '..', 'target', 'release', binaryName),
    path.join(__dirname, '..', '..', '..', 'target', 'debug', binaryName),
    path.resolve(process.cwd(), '_bin', target, binaryName),
    path.resolve(process.cwd(), 'target', 'release', binaryName),
  ];

  for (const p of searchPaths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }

  // Try PATH
  const pathDirs = (process.env.PATH || '').split(path.delimiter);
  for (const dir of pathDirs) {
    const p = path.join(dir, binaryName);
    if (fs.existsSync(p)) {
      return p;
    }
  }

  throw new DatabaseError(
    `Could not find ${binaryName}. Install via: cargo build --release -p toondb-tools`
  );
}

/**
 * HNSW Vector Index for high-performance similarity search.
 *
 * @example
 * ```typescript
 * import { VectorIndex } from '@sushanth/toondb';
 *
 * // Build an index
 * const stats = await VectorIndex.bulkBuild(embeddings, {
 *   output: 'my_index.hnsw',
 *   dimension: 768,
 *   m: 16,
 *   efConstruction: 100,
 * });
 *
 * // Query the index
 * const results = await VectorIndex.query('my_index.hnsw', queryVector, {
 *   k: 10,
 *   efSearch: 64,
 * });
 * ```
 */
export class VectorIndex {
  /**
   * Build an HNSW index from vectors.
   *
   * @param vectors - Float32Array of vectors (dimension × count)
   * @param options - Build options
   * @param labels - Optional array of string labels for each vector
   * @returns Build statistics
   */
  static async bulkBuild(
    vectors: Float32Array,
    options: {
      output: string;
      dimension: number;
      m?: number;
      efConstruction?: number;
      metric?: 'cosine' | 'euclidean' | 'dot';
    },
    labels?: string[]
  ): Promise<BulkBuildStats> {
    // Validate inputs
    if (!vectors || vectors.length === 0) {
      throw new DatabaseError('vectors array cannot be empty');
    }
    
    if (options.dimension <= 0) {
      throw new DatabaseError(`dimension must be positive, got ${options.dimension}`);
    }
    
    if (vectors.length % options.dimension !== 0) {
      throw new DatabaseError(
        `vectors length (${vectors.length}) must be divisible by dimension (${options.dimension})`
      );
    }
    
    const bulkPath = findBulkBinary();
    const vectorCount = vectors.length / options.dimension;

    // Write vectors to temp file
    const tempFile = path.join(
      require('os').tmpdir(),
      `toondb_vectors_${Date.now()}.bin`
    );
    fs.writeFileSync(tempFile, Buffer.from(vectors.buffer));

    // Write labels to temp file if provided
    let labelsFile: string | undefined;
    if (labels && labels.length > 0) {
      if (labels.length !== vectorCount) {
        throw new DatabaseError(`Labels count (${labels.length}) must match vector count (${vectorCount})`);
      }
      labelsFile = path.join(
        require('os').tmpdir(),
        `toondb_labels_${Date.now()}.txt`
      );
      fs.writeFileSync(labelsFile, labels.join('\n'));
    }

    try {
      const args = [
        'build-index',
        '--input', tempFile,
        '--output', options.output,
        '--dimension', options.dimension.toString(),
        '--max-connections', (options.m || 16).toString(),
        '--ef-construction', (options.efConstruction || 100).toString(),
      ];

      // Add labels file argument if provided
      if (labelsFile) {
        args.push('--labels', labelsFile);
      }

      const startTime = Date.now();

      const result = await new Promise<BulkBuildStats>((resolve, reject) => {
        const child = spawn(bulkPath, args);
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        child.on('close', (code) => {
          const buildTimeSeconds = (Date.now() - startTime) / 1000;

          if (code !== 0) {
            reject(new DatabaseError(`Bulk build failed: ${stderr}`));
            return;
          }

          resolve({
            vectors: vectorCount,
            buildTimeSeconds,
            rate: vectorCount / buildTimeSeconds,
          });
        });

        child.on('error', (err) => {
          reject(new DatabaseError(`Failed to spawn bulk process: ${err.message}`));
        });
      });
      
      return result;
    } finally {
      // Clean up temp files
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
      if (labelsFile && fs.existsSync(labelsFile)) {
        fs.unlinkSync(labelsFile);
      }
    }
  }

  /**
   * Query an HNSW index.
   *
   * @param indexPath - Path to the index file
   * @param query - Query vector (Float32Array)
   * @param options - Query options
   * @returns Array of search results
   */
  static async query(
    indexPath: string,
    query: Float32Array,
    options?: {
      k?: number;
      efSearch?: number;
    }
  ): Promise<VectorSearchResult[]> {
    // Validate inputs
    if (!indexPath || indexPath.trim() === '') {
      throw new DatabaseError('indexPath cannot be empty');
    }
    
    if (!fs.existsSync(indexPath)) {
      throw new DatabaseError(`Index file not found: ${indexPath}`);
    }
    
    if (!query || query.length === 0) {
      throw new DatabaseError('query vector cannot be empty');
    }
    
    const bulkPath = findBulkBinary();
    const k = options?.k || 10;
    const efSearch = options?.efSearch || 64;

    // Write query to temp file
    const tempFile = path.join(
      require('os').tmpdir(),
      `toondb_query_${Date.now()}.bin`
    );
    fs.writeFileSync(tempFile, Buffer.from(query.buffer));

    try {
      const args = [
        'query',
        '--index', indexPath,
        '--query', tempFile,
        '--k', k.toString(),
        '--ef', efSearch.toString(),
      ];

      const result = await new Promise<VectorSearchResult[]>((resolve, reject) => {
        const child = spawn(bulkPath, args);
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        child.on('close', (code) => {
          if (code !== 0) {
            reject(new DatabaseError(`Query failed: ${stderr}`));
            return;
          }

          try {
            const results: VectorSearchResult[] = JSON.parse(stdout);
            if (results.length === 0) {
              console.warn(`Warning: Vector search returned 0 results. This may indicate:
  - Index is empty (no vectors were indexed)
  - Query vector dimension mismatch
  - efSearch parameter too low (try increasing from ${efSearch})
  - Distance threshold filtering out all results`);
            }
            resolve(results);
          } catch {
            // Parse line format: id,distance[,label]
            const results: VectorSearchResult[] = stdout
              .trim()
              .split('\n')
              .filter((line) => line.length > 0)
              .map((line) => {
                const parts = line.split(',');
                const result: VectorSearchResult = {
                  id: parseInt(parts[0], 10),
                  distance: parseFloat(parts[1]),
                };
                // Include label if present (third column)
                if (parts.length >= 3 && parts[2].trim()) {
                  result.label = parts[2].trim();
                }
                return result;
              });
            if (results.length === 0) {
              console.warn(`Warning: Vector search returned 0 results. This may indicate:
  - Index is empty (no vectors were indexed)
  - Query vector dimension mismatch
  - efSearch parameter too low (try increasing from ${efSearch})
  - Distance threshold filtering out all results`);
            }
            resolve(results);
          }
        });

        child.on('error', (err) => {
          reject(new DatabaseError(`Failed to spawn query process: ${err.message}`));
        });
      });
      
      return result;
    } finally {
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
    }
  }

  /**
   * Get index metadata.
   *
   * @param indexPath - Path to the index file
   * @returns Index metadata
   */
  static async info(indexPath: string): Promise<{
    vectors: number;
    dimension: number;
    metric: string;
  }> {
    const bulkPath = findBulkBinary();

    return new Promise((resolve, reject) => {
      const child = spawn(bulkPath, ['info', '--index', indexPath]);
      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        if (code !== 0) {
          reject(new DatabaseError(`Info failed: ${stderr}`));
          return;
        }

        try {
          const info = JSON.parse(stdout);
          resolve({
            vectors: info.vectors || info.count || 0,
            dimension: info.dimension || 0,
            metric: info.metric || 'cosine',
          });
        } catch {
          reject(new DatabaseError(`Failed to parse info: ${stdout}`));
        }
      });

      child.on('error', (err) => {
        reject(new DatabaseError(`Failed to spawn info process: ${err.message}`));
      });
    });
  }

  // Instance properties for non-static usage
  private _path: string;
  private _config: VectorIndexConfig;

  /**
   * Create a VectorIndex instance.
   * 
   * @param indexPath - Path where the index will be stored
   * @param config - Optional configuration
   */
  constructor(indexPath: string, config?: Partial<VectorIndexConfig>) {
    this._path = indexPath;
    this._config = {
      dimension: config?.dimension || 0,
      m: config?.m || 16,
      efConstruction: config?.efConstruction || 100,
      metric: config?.metric || 'cosine',
    };
  }

  /**
   * Build an index from vectors (instance method).
   * 
   * @param vectors - Array of number arrays
   * @param labels - Optional labels for each vector
   */
  async bulkBuild(vectors: number[][], labels?: string[]): Promise<BulkBuildStats> {
    if (vectors.length === 0) {
      return { vectors: 0, buildTimeSeconds: 0, rate: 0 };
    }

    // Validate dimensions
    const dim = vectors[0].length;
    for (let i = 1; i < vectors.length; i++) {
      if (vectors[i].length !== dim) {
        throw new DatabaseError(
          `Vector ${i} has dimension ${vectors[i].length}, expected ${dim}`
        );
      }
    }

    // Flatten to Float32Array
    const flat = new Float32Array(vectors.length * dim);
    for (let i = 0; i < vectors.length; i++) {
      for (let j = 0; j < dim; j++) {
        flat[i * dim + j] = vectors[i][j];
      }
    }

    return VectorIndex.bulkBuild(flat, {
      output: this._path,
      dimension: dim,
      m: this._config.m,
      efConstruction: this._config.efConstruction,
      metric: this._config.metric,
    });
  }

  /**
   * Query the index (instance method).
   * 
   * @param query - Query vector as number array
   * @param k - Number of nearest neighbors to return
   * @param efSearch - Search quality parameter
   * @returns Array of search results
   */
  async query(query: number[], k?: number, efSearch?: number): Promise<VectorSearchResult[]> {
    const queryFloat = new Float32Array(query);
    return VectorIndex.query(this._path, queryFloat, { k, efSearch });
  }

  /**
   * Compute cosine distance between two vectors.
   * 
   * @param a - First vector
   * @param b - Second vector
   * @returns Cosine distance (0 = identical, 2 = opposite)
   */
  static computeCosineDistance(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) {
      return 1;
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return 1 - similarity;
  }

  /**
   * Compute Euclidean (L2) distance between two vectors.
   * 
   * @param a - First vector
   * @param b - Second vector
   * @returns Euclidean distance
   */
  static computeEuclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Compute dot product of two vectors.
   * 
   * @param a - First vector
   * @param b - Second vector
   * @returns Dot product
   */
  static computeDotProduct(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /**
   * Normalize a vector to unit length.
   * 
   * @param v - Input vector
   * @returns Normalized vector (new array)
   */
  static normalizeVector(v: number[]): number[] {
    let norm = 0;
    for (const x of v) {
      norm += x * x;
    }
    norm = Math.sqrt(norm);

    if (norm === 0) {
      return [...v];
    }

    return v.map(x => x / norm);
  }
}
