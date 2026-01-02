// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// DistanceMetric represents the distance function for vector similarity.
type DistanceMetric string

const (
	// Cosine distance (1 - cosine_similarity).
	Cosine DistanceMetric = "cosine"
	// Euclidean (L2) distance.
	Euclidean DistanceMetric = "euclidean"
	// InnerProduct (dot product) - higher is more similar.
	InnerProduct DistanceMetric = "innerproduct"
)

// VectorIndexConfig holds configuration for building vector indexes.
type VectorIndexConfig struct {
	// Dimension of the vectors.
	Dimension int

	// Metric is the distance metric to use.
	// Default: Cosine
	Metric DistanceMetric

	// M is the HNSW connectivity parameter.
	// Higher values = more accuracy, more memory.
	// Default: 16
	M int

	// EfConstruction is the HNSW construction parameter.
	// Higher values = better index quality, slower construction.
	// Default: 100
	EfConstruction int

	// EfSearch is the HNSW search parameter.
	// Higher values = more accuracy, slower search.
	// Default: 50
	EfSearch int
}

// DefaultVectorIndexConfig returns default configuration.
func DefaultVectorIndexConfig(dimension int) *VectorIndexConfig {
	return &VectorIndexConfig{
		Dimension:      dimension,
		Metric:         Cosine,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
	}
}

// VectorIndex represents an HNSW vector search index.
type VectorIndex struct {
	path      string
	config    *VectorIndexConfig
	indexPath string
}

// VectorSearchResult represents a single search result.
type VectorSearchResult struct {
	// ID is the vector ID.
	ID uint64
	// Distance is the distance from the query vector.
	Distance float32
	// Label is an optional label associated with the vector.
	Label string
}

// VectorIndexInfo contains index metadata.
type VectorIndexInfo struct {
	Dimension      int
	NumVectors     uint64
	Metric         DistanceMetric
	M              int
	EfConstruction int
	IndexSizeBytes int64
}

// NewVectorIndex creates a new vector index at the given path.
func NewVectorIndex(path string, config *VectorIndexConfig) *VectorIndex {
	if config == nil {
		config = DefaultVectorIndexConfig(0)
	}
	return &VectorIndex{
		path:      path,
		config:    config,
		indexPath: filepath.Join(path, "vectors.hnsw"),
	}
}

// BulkBuild builds an HNSW index from a dataset.
//
// The vectors parameter should be a slice of float32 slices,
// where each inner slice has length equal to the dimension.
//
// Labels are currently ignored in raw format mode.
func (vi *VectorIndex) BulkBuild(vectors [][]float32, labels []string) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate dimensions
	dim := len(vectors[0])
	for i, v := range vectors {
		if len(v) != dim {
			return &ToonDBError{
				Op:      "bulk_build",
				Message: fmt.Sprintf("vector %d has dimension %d, expected %d", i, len(v), dim),
				Err:     ErrVectorDimension,
			}
		}
	}

	vi.config.Dimension = dim

	// Create output directory
	if err := os.MkdirAll(vi.path, 0755); err != nil {
		return err
	}

	// Write vectors to temporary RAW format file
	rawPath := filepath.Join(vi.path, "vectors.raw")
	if err := vi.writeVectorsToRaw(rawPath, vectors); err != nil {
		return err
	}
	defer os.Remove(rawPath)

	// Find toondb-bulk binary
	bulkBin, err := vi.findBulkBinary()
	if err != nil {
		return err
	}

	// Run bulk build
	// Note: We removed --m, --metric, --ef-construction as requested by user fixes.
	args := []string{
		"build-index",
		"--input", rawPath,
		"--output", vi.indexPath,
		"--dimension", strconv.Itoa(dim),
	}

	cmd := exec.Command(bulkBin, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return &ToonDBError{
			Op:      "bulk_build",
			Message: fmt.Sprintf("bulk build failed: %s", string(output)),
			Err:     err,
		}
	}

	return nil
}

// Query performs a k-nearest neighbors search.
//
// Example:
//
//	results, err := index.Query(queryVector, 10, 50)
//	for _, r := range results {
//	    fmt.Printf("ID: %d, Distance: %f, Label: %s\n", r.ID, r.Distance, r.Label)
//	}
func (vi *VectorIndex) Query(vector []float32, k int, efSearch int) ([]VectorSearchResult, error) {
	if efSearch <= 0 {
		efSearch = vi.config.EfSearch
	}

	// Find toondb-bulk binary
	bulkBin, err := vi.findBulkBinary()
	if err != nil {
		return nil, err
	}

	// Write query vector to temp file
	queryPath := filepath.Join(vi.path, ".query_tmp")
	if err := vi.writeQueryVector(queryPath, vector); err != nil {
		return nil, err
	}
	defer os.Remove(queryPath)

	// Run query
	// Note: Removed --ef-search as requested.
	args := []string{
		"query",
		"--index", vi.indexPath,
		"--query", queryPath,
		"--k", strconv.Itoa(k),
	}

	cmd := exec.Command(bulkBin, args...)
	output, err := cmd.Output()
	if err != nil {
		return nil, &ToonDBError{
			Op:      "query",
			Message: fmt.Sprintf("query failed: %v", err),
			Err:     err,
		}
	}

	// Parse results
	return vi.parseQueryResults(string(output))
}

// Info returns index metadata.
func (vi *VectorIndex) Info() (*VectorIndexInfo, error) {
	info, err := os.Stat(vi.indexPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, &ToonDBError{
				Op:      "info",
				Message: "index does not exist",
				Err:     err,
			}
		}
		return nil, err
	}

	return &VectorIndexInfo{
		Dimension:      vi.config.Dimension,
		Metric:         vi.config.Metric,
		M:              vi.config.M,
		EfConstruction: vi.config.EfConstruction,
		IndexSizeBytes: info.Size(),
	}, nil
}

// Helper methods

func (vi *VectorIndex) findBulkBinary() (string, error) {
	// Check common locations
	locations := []string{
		"toondb-bulk",
		"./toondb-bulk",
		"../target/release/toondb-bulk",
		"/usr/local/bin/toondb-bulk",
	}

	// Add platform-specific extension on Windows
	if runtime.GOOS == "windows" {
		for i, loc := range locations {
			if !strings.HasSuffix(loc, ".exe") {
				locations[i] = loc + ".exe"
			}
		}
	}

	for _, loc := range locations {
		if _, err := exec.LookPath(loc); err == nil {
			return loc, nil
		}
		if _, err := os.Stat(loc); err == nil {
			return loc, nil
		}
	}

	return "", &ToonDBError{
		Op:      "find_binary",
		Message: "toondb-bulk binary not found in PATH or common locations",
	}
}

func (vi *VectorIndex) writeVectorsToRaw(path string, vectors [][]float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)

	// Write vectors flat
	for _, vec := range vectors {
		for _, v := range vec {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}

	return w.Flush()
}

func (vi *VectorIndex) writeQueryVector(path string, vector []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)

	// No dimension header for raw format

	// Write vector components
	for _, v := range vector {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}

	return w.Flush()
}

func (vi *VectorIndex) parseQueryResults(output string) ([]VectorSearchResult, error) {
	var results []VectorSearchResult

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Format: ID,Distance,Label
		parts := strings.SplitN(line, ",", 3)
		if len(parts) < 2 {
			continue
		}

		id, err := strconv.ParseUint(parts[0], 10, 64)
		if err != nil {
			continue
		}

		dist, err := strconv.ParseFloat(parts[1], 32)
		if err != nil {
			continue
		}

		label := ""
		if len(parts) > 2 {
			label = parts[2]
		}

		results = append(results, VectorSearchResult{
			ID:       id,
			Distance: float32(dist),
			Label:    label,
		})
	}

	return results, nil
}

// ComputeCosineDistance computes cosine distance between two vectors.
func ComputeCosineDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return math.MaxFloat32
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - similarity
}

// ComputeEuclideanDistance computes Euclidean (L2) distance between two vectors.
func ComputeEuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return math.MaxFloat32
	}

	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return float32(math.Sqrt(float64(sum)))
}

// NormalizeVector normalizes a vector to unit length.
func NormalizeVector(v []float32) []float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm == 0 {
		return v
	}

	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = x / norm
	}
	return result
}

// ReadVectorsFromFVECS reads vectors from an FVECS format file.
func ReadVectorsFromFVECS(path string) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var vectors [][]float32

	for {
		// Read dimension
		var dim uint32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		// Read vector
		vec := make([]float32, dim)
		for i := range vec {
			if err := binary.Read(r, binary.LittleEndian, &vec[i]); err != nil {
				return nil, err
			}
		}

		vectors = append(vectors, vec)
	}

	return vectors, nil
}
