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

//! Integration tests for CONTEXT SELECT
//!
//! These tests verify the end-to-end functionality of CONTEXT SELECT queries,
//! including parsing, vector search integration, and result formatting.

use std::collections::HashMap;
use std::sync::Arc;

use sochdb_query::context_query::{
    ContextQueryBuilder, ContextQueryParser, OutputFormat, SectionContent, SessionReference,
    SimpleVectorIndex, TruncationStrategy, VectorIndex, VectorIndexStats, VectorSearchResult,
};
use sochdb_query::soch_ql::SochValue;

// ============================================================================
// Parser Integration Tests
// ============================================================================

#[test]
fn test_full_query_parse_and_validate() {
    let query = r#"
        CONTEXT SELECT agent_context
        FROM session($current_session)
        WITH (
            token_limit = 8192,
            include_schema = true,
            format = toon,
            truncation = proportional
        )
        SECTIONS (
            SYSTEM PRIORITY -1: "You are a helpful database assistant.",
            SCHEMA PRIORITY 0: GET schema.tables.**,
            HISTORY PRIORITY 1: LAST 20 FROM query_history WHERE success = true,
            DOCS PRIORITY 2: SEARCH knowledge_base BY SIMILARITY($user_query) TOP 10
        )
    "#;

    let mut parser = ContextQueryParser::new(query);
    let result = parser.parse().expect("Failed to parse query");

    // Validate output name
    assert_eq!(result.output_name, "agent_context");

    // Validate session
    match &result.session {
        SessionReference::Session(s) => assert_eq!(s, "current_session"),
        _ => panic!("Expected Session reference"),
    }

    // Validate options
    assert_eq!(result.options.token_limit, 8192);
    assert!(result.options.include_schema);
    assert_eq!(result.options.format, OutputFormat::Soch);
    assert_eq!(result.options.truncation, TruncationStrategy::Proportional);

    // Validate sections
    assert_eq!(result.sections.len(), 4);

    // SYSTEM section
    assert_eq!(result.sections[0].name, "SYSTEM");
    assert_eq!(result.sections[0].priority, -1);
    match &result.sections[0].content {
        SectionContent::Literal { value } => {
            assert_eq!(value, "You are a helpful database assistant.");
        }
        _ => panic!("Expected Literal content for SYSTEM section"),
    }

    // SCHEMA section
    assert_eq!(result.sections[1].name, "SCHEMA");
    assert_eq!(result.sections[1].priority, 0);

    // HISTORY section
    assert_eq!(result.sections[2].name, "HISTORY");
    assert_eq!(result.sections[2].priority, 1);
    match &result.sections[2].content {
        SectionContent::Last { count, table, .. } => {
            assert_eq!(*count, 20);
            assert_eq!(table, "query_history");
        }
        _ => panic!("Expected Last content for HISTORY section"),
    }

    // DOCS section
    assert_eq!(result.sections[3].name, "DOCS");
    assert_eq!(result.sections[3].priority, 2);
    match &result.sections[3].content {
        SectionContent::Search {
            collection, top_k, ..
        } => {
            assert_eq!(collection, "knowledge_base");
            assert_eq!(*top_k, 10);
        }
        _ => panic!("Expected Search content for DOCS section"),
    }
}

#[test]
fn test_minimal_query() {
    let query = r#"
        CONTEXT SELECT ctx
        SECTIONS ()
    "#;

    let mut parser = ContextQueryParser::new(query);
    let result = parser.parse().expect("Failed to parse minimal query");

    assert_eq!(result.output_name, "ctx");
    assert!(matches!(result.session, SessionReference::None));
    assert_eq!(result.sections.len(), 0);
}

#[test]
fn test_query_builder() {
    let query = ContextQueryBuilder::new("test_context")
        .from_session("test_session")
        .with_token_limit(4096)
        .format(OutputFormat::Markdown)
        .truncation(TruncationStrategy::TailDrop)
        .include_schema(true)
        .literal("SYSTEM", -1, "System prompt here")
        .get("DATA", 0, "user.profile.**")
        .last("HISTORY", 1, 10, "events")
        .search("SIMILAR", 2, "docs", "embedding_var", 5)
        .build();

    assert_eq!(query.output_name, "test_context");
    assert_eq!(query.options.token_limit, 4096);
    assert_eq!(query.options.format, OutputFormat::Markdown);
    assert_eq!(query.sections.len(), 4);
}

// ============================================================================
// Vector Index Integration Tests
// ============================================================================

#[test]
fn test_vector_index_end_to_end() {
    let index = SimpleVectorIndex::new();
    index.create_collection("documents", 4);

    // Insert test documents with embeddings
    let docs = vec![
        (
            "doc1",
            vec![1.0, 0.0, 0.0, 0.0],
            "Rust programming language guide",
        ),
        (
            "doc2",
            vec![0.9, 0.1, 0.0, 0.0],
            "Systems programming with Rust",
        ),
        (
            "doc3",
            vec![0.0, 1.0, 0.0, 0.0],
            "Python data science tutorial",
        ),
        (
            "doc4",
            vec![0.0, 0.9, 0.1, 0.0],
            "Machine learning with Python",
        ),
        (
            "doc5",
            vec![0.0, 0.0, 1.0, 0.0],
            "JavaScript web development",
        ),
        ("doc6", vec![0.0, 0.0, 0.0, 1.0], "Database design patterns"),
    ];

    for (id, embedding, content) in docs {
        index
            .insert(
                "documents",
                id.to_string(),
                embedding,
                content.to_string(),
                HashMap::new(),
            )
            .expect("Failed to insert document");
    }

    // Search for Rust-related documents
    let results = index
        .search_by_embedding("documents", &[1.0, 0.0, 0.0, 0.0], 3, None)
        .expect("Search failed");

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, "doc1"); // Exact match
    assert_eq!(results[1].id, "doc2"); // Similar

    // Verify stats
    let stats = index.stats("documents").expect("Stats should exist");
    assert_eq!(stats.vector_count, 6);
    assert_eq!(stats.dimension, 4);
    assert_eq!(stats.metric, "cosine");
}

#[test]
fn test_vector_index_with_min_score() {
    let index = SimpleVectorIndex::new();
    index.create_collection("test", 3);

    // Insert orthogonal vectors
    index
        .insert(
            "test",
            "a".to_string(),
            vec![1.0, 0.0, 0.0],
            "Doc A".to_string(),
            HashMap::new(),
        )
        .unwrap();
    index
        .insert(
            "test",
            "b".to_string(),
            vec![0.0, 1.0, 0.0],
            "Doc B".to_string(),
            HashMap::new(),
        )
        .unwrap();
    index
        .insert(
            "test",
            "c".to_string(),
            vec![0.0, 0.0, 1.0],
            "Doc C".to_string(),
            HashMap::new(),
        )
        .unwrap();

    // Search with high min_score - only exact match should return
    let results = index
        .search_by_embedding("test", &[1.0, 0.0, 0.0], 10, Some(0.99))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "a");
    assert!(results[0].score >= 0.99);

    // Search with low min_score - all should return
    let results = index
        .search_by_embedding("test", &[1.0, 0.0, 0.0], 10, Some(0.0))
        .unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_vector_index_metadata_preservation() {
    let index = SimpleVectorIndex::new();
    index.create_collection("with_meta", 2);

    let mut metadata = HashMap::new();
    metadata.insert(
        "source".to_string(),
        SochValue::Text("wikipedia".to_string()),
    );
    metadata.insert("page".to_string(), SochValue::Int(42));
    metadata.insert("confidence".to_string(), SochValue::Float(0.95));

    index
        .insert(
            "with_meta",
            "doc_with_meta".to_string(),
            vec![1.0, 0.0],
            "Document content".to_string(),
            metadata,
        )
        .unwrap();

    let results = index
        .search_by_embedding("with_meta", &[1.0, 0.0], 1, None)
        .unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    // Verify metadata preserved
    assert!(result.metadata.contains_key("source"));
    assert!(result.metadata.contains_key("page"));
    assert!(result.metadata.contains_key("confidence"));

    match result.metadata.get("source") {
        Some(SochValue::Text(s)) => assert_eq!(s, "wikipedia"),
        _ => panic!("Expected Text value for source"),
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_parser_error_recovery() {
    // Missing SECTIONS keyword
    let query = "CONTEXT SELECT ctx WITH (token_limit = 100)";
    let mut _parser = ContextQueryParser::new(query);
    let result = _parser.parse();
    assert!(result.is_err());

    // Invalid option
    let query = "CONTEXT SELECT ctx WITH (invalid_option = true) SECTIONS ()";
    let mut _parser = ContextQueryParser::new(query);
    // This should parse but ignore unknown option
    // (depending on implementation - may fail)
}

#[test]
fn test_vector_index_error_cases() {
    let index = SimpleVectorIndex::new();

    // Collection doesn't exist
    let result = index.search_by_embedding("nonexistent", &[1.0], 5, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));

    // Create collection and try wrong dimension
    index.create_collection("test", 3);
    let result = index.insert(
        "test",
        "doc".to_string(),
        vec![1.0, 0.0], // Wrong dimension
        "content".to_string(),
        HashMap::new(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("dimension"));
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[test]
fn test_vector_index_concurrent_reads() {
    use std::thread;

    let index = Arc::new(SimpleVectorIndex::new());
    index.create_collection("concurrent", 2);

    // Insert some data
    for i in 0..100 {
        index
            .insert(
                "concurrent",
                format!("doc{}", i),
                vec![(i as f32).cos(), (i as f32).sin()],
                format!("Document {}", i),
                HashMap::new(),
            )
            .unwrap();
    }

    // Spawn multiple readers
    let mut handles = vec![];
    for _ in 0..10 {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let _ = idx.search_by_embedding("concurrent", &[1.0, 0.0], 5, None);
            }
        }));
    }

    // All should complete without panic
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Custom Vector Index Implementation Test
// ============================================================================

/// A mock vector index for testing that always returns predefined results
struct MockVectorIndex {
    results: Vec<VectorSearchResult>,
}

impl MockVectorIndex {
    fn new(results: Vec<VectorSearchResult>) -> Self {
        Self { results }
    }
}

impl VectorIndex for MockVectorIndex {
    fn search_by_embedding(
        &self,
        _collection: &str,
        _embedding: &[f32],
        k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Ok(self.results.iter().take(k).cloned().collect())
    }

    fn search_by_text(
        &self,
        _collection: &str,
        _text: &str,
        k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Ok(self.results.iter().take(k).cloned().collect())
    }

    fn stats(&self, _collection: &str) -> Option<VectorIndexStats> {
        Some(VectorIndexStats {
            vector_count: self.results.len(),
            dimension: 128,
            metric: "mock".to_string(),
        })
    }
}

#[test]
fn test_custom_vector_index() {
    let mock_results = vec![
        VectorSearchResult {
            id: "mock1".to_string(),
            score: 0.99,
            content: "Mock result 1".to_string(),
            metadata: HashMap::new(),
        },
        VectorSearchResult {
            id: "mock2".to_string(),
            score: 0.85,
            content: "Mock result 2".to_string(),
            metadata: HashMap::new(),
        },
    ];

    let index = MockVectorIndex::new(mock_results);

    // Test that custom implementation works
    let results = index.search_by_embedding("any", &[0.0], 10, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "mock1");

    let stats = index.stats("any").unwrap();
    assert_eq!(stats.metric, "mock");
}

// ============================================================================
// Query Result Formatting Tests
// ============================================================================

#[test]
fn test_section_priority_ordering() {
    let query = ContextQueryBuilder::new("ordered")
        .literal("LOW", 10, "Low priority")
        .literal("MEDIUM", 5, "Medium priority")
        .literal("HIGH", 0, "High priority")
        .literal("CRITICAL", -1, "Critical priority")
        .build();

    // Verify sections are in insertion order (sorting happens at execution)
    assert_eq!(query.sections.len(), 4);

    // When sorted by priority, order should be: CRITICAL, HIGH, MEDIUM, LOW
    let mut sorted = query.sections.clone();
    sorted.sort_by_key(|s| s.priority);

    assert_eq!(sorted[0].name, "CRITICAL");
    assert_eq!(sorted[1].name, "HIGH");
    assert_eq!(sorted[2].name, "MEDIUM");
    assert_eq!(sorted[3].name, "LOW");
}

#[test]
fn test_multiple_collections() {
    let index = SimpleVectorIndex::new();

    // Create multiple collections
    index.create_collection("code", 4);
    index.create_collection("docs", 4);
    index.create_collection("logs", 4);

    // Insert into each
    index
        .insert(
            "code",
            "fn1".to_string(),
            vec![1.0, 0.0, 0.0, 0.0],
            "function one".to_string(),
            HashMap::new(),
        )
        .unwrap();
    index
        .insert(
            "docs",
            "doc1".to_string(),
            vec![0.0, 1.0, 0.0, 0.0],
            "documentation".to_string(),
            HashMap::new(),
        )
        .unwrap();
    index
        .insert(
            "logs",
            "log1".to_string(),
            vec![0.0, 0.0, 1.0, 0.0],
            "log entry".to_string(),
            HashMap::new(),
        )
        .unwrap();

    // Verify isolation
    let code_stats = index.stats("code").unwrap();
    assert_eq!(code_stats.vector_count, 1);

    let docs_stats = index.stats("docs").unwrap();
    assert_eq!(docs_stats.vector_count, 1);

    // Search in specific collection
    let results = index
        .search_by_embedding("code", &[1.0, 0.0, 0.0, 0.0], 10, None)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "fn1");
}
