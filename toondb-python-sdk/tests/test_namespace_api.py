# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for ToonDB Python SDK - Namespace, Collection, and Search APIs

These tests cover:
- Task 8: Namespace Handle API
- Task 9: Collection Builder
- Task 10: One Search Surface
- Task 11: Error Taxonomy
- Task 12: ContextQuery Builder
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# Test imports
from toondb import (
    # Namespace
    Namespace,
    NamespaceConfig,
    # Collection
    Collection,
    CollectionConfig,
    DistanceMetric,
    QuantizationType,
    # Search
    SearchRequest,
    SearchResult,
    SearchResults,
    # Context
    ContextQuery,
    ContextResult,
    ContextChunk,
    TokenEstimator,
    DeduplicationStrategy,
    estimate_tokens,
    split_by_tokens,
    # Errors
    ToonDBError,
    ErrorCode,
    NamespaceError,
    NamespaceNotFoundError,
    NamespaceExistsError,
    CollectionError,
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionConfigError,
    ValidationError,
    DimensionMismatchError,
    QueryError,
)


# ============================================================================
# Task 11: Error Taxonomy Tests
# ============================================================================

class TestErrorTaxonomy:
    """Test the error taxonomy (Task 11)."""
    
    def test_error_code_enum(self):
        """Test ErrorCode enum values."""
        assert ErrorCode.INTERNAL_ERROR == 9001
        assert ErrorCode.NAMESPACE_NOT_FOUND == 3001
        assert ErrorCode.COLLECTION_ALREADY_EXISTS == 4002
        assert ErrorCode.INVALID_VECTOR_DIMENSION == 6001
    
    def test_namespace_not_found_error(self):
        """Test NamespaceNotFoundError."""
        err = NamespaceNotFoundError("tenant_123")
        assert err.code == ErrorCode.NAMESPACE_NOT_FOUND
        assert "tenant_123" in str(err)
        assert err.namespace == "tenant_123"
        assert err.remediation is not None
    
    def test_namespace_exists_error(self):
        """Test NamespaceExistsError."""
        err = NamespaceExistsError("tenant_123")
        assert err.code == ErrorCode.NAMESPACE_EXISTS
        assert "tenant_123" in str(err)
    
    def test_collection_not_found_error(self):
        """Test CollectionNotFoundError."""
        err = CollectionNotFoundError("documents", "tenant_123")
        assert err.code == ErrorCode.COLLECTION_NOT_FOUND
        assert "documents" in str(err)
        assert err.collection == "documents"
        assert err.namespace == "tenant_123"
    
    def test_collection_exists_error(self):
        """Test CollectionExistsError."""
        err = CollectionExistsError("documents", "tenant_123")
        assert err.code == ErrorCode.COLLECTION_EXISTS
    
    def test_dimension_mismatch_error(self):
        """Test DimensionMismatchError."""
        err = DimensionMismatchError(expected=384, actual=768)
        assert err.code == ErrorCode.DIMENSION_MISMATCH
        assert "384" in str(err)
        assert "768" in str(err)
    
    def test_validation_error(self):
        """Test ValidationError."""
        err = ValidationError("Invalid parameter")
        assert err.code == ErrorCode.VALIDATION_ERROR
        assert "Invalid parameter" in str(err)
    
    def test_error_inheritance(self):
        """Test error class hierarchy."""
        assert issubclass(NamespaceNotFoundError, NamespaceError)
        assert issubclass(NamespaceError, ToonDBError)
        assert issubclass(CollectionNotFoundError, CollectionError)
        assert issubclass(DimensionMismatchError, ValidationError)


# ============================================================================
# Task 9: Collection Config Tests
# ============================================================================

class TestCollectionConfig:
    """Test CollectionConfig (Task 9)."""
    
    def test_basic_config(self):
        """Test basic collection configuration."""
        config = CollectionConfig(name="documents", dimension=384)
        assert config.name == "documents"
        assert config.dimension == 384
        assert config.metric == DistanceMetric.COSINE
        assert config.m == 16
        assert config.ef_construction == 100
    
    def test_custom_config(self):
        """Test custom collection configuration."""
        config = CollectionConfig(
            name="embeddings",
            dimension=768,
            metric=DistanceMetric.EUCLIDEAN,
            m=32,
            ef_construction=200,
            enable_hybrid_search=True,
            content_field="text",
        )
        assert config.dimension == 768
        assert config.metric == DistanceMetric.EUCLIDEAN
        assert config.m == 32
        assert config.enable_hybrid_search is True
        assert config.content_field == "text"
    
    def test_config_immutability(self):
        """Test that config is immutable (frozen dataclass)."""
        config = CollectionConfig(name="test", dimension=384)
        with pytest.raises(Exception):  # FrozenInstanceError
            config.dimension = 512
    
    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValidationError):
            CollectionConfig(name="test", dimension=0)
        
        with pytest.raises(ValidationError):
            CollectionConfig(name="test", dimension=-1)
        
        with pytest.raises(ValidationError):
            CollectionConfig(name="test", dimension=384, m=0)
    
    def test_config_serialization(self):
        """Test config to/from dict."""
        config = CollectionConfig(
            name="test",
            dimension=384,
            metric=DistanceMetric.DOT_PRODUCT,
            enable_hybrid_search=True,
        )
        
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["dimension"] == 384
        assert data["metric"] == "dot_product"
        assert data["enable_hybrid_search"] is True
        
        restored = CollectionConfig.from_dict(data)
        assert restored.name == config.name
        assert restored.dimension == config.dimension
        assert restored.metric == config.metric


# ============================================================================
# Task 10: Search Request Tests
# ============================================================================

class TestSearchRequest:
    """Test SearchRequest (Task 10)."""
    
    def test_vector_search_request(self):
        """Test vector search request."""
        vector = [0.1] * 384
        request = SearchRequest(vector=vector, k=10)
        request.validate(expected_dimension=384)
        assert request.vector == vector
        assert request.k == 10
    
    def test_keyword_search_request(self):
        """Test keyword search request."""
        request = SearchRequest(text_query="machine learning", k=10)
        request.validate()
        assert request.text_query == "machine learning"
        assert request.alpha == 0.5
    
    def test_hybrid_search_request(self):
        """Test hybrid search request."""
        vector = [0.1] * 384
        request = SearchRequest(
            vector=vector,
            text_query="python programming",
            alpha=0.7,
            k=20,
        )
        request.validate(expected_dimension=384)
        assert request.vector == vector
        assert request.text_query == "python programming"
        assert request.alpha == 0.7
    
    def test_search_request_validation(self):
        """Test search request validation."""
        # Missing both vector and text
        request = SearchRequest(k=10)
        with pytest.raises(ValidationError):
            request.validate()
        
        # Dimension mismatch
        request = SearchRequest(vector=[0.1] * 384)
        with pytest.raises(DimensionMismatchError):
            request.validate(expected_dimension=768)
        
        # Invalid k
        request = SearchRequest(vector=[0.1] * 384, k=0)
        with pytest.raises(ValidationError):
            request.validate()
        
        # Invalid alpha
        request = SearchRequest(vector=[0.1] * 384, alpha=1.5)
        with pytest.raises(ValidationError):
            request.validate()


# ============================================================================
# Task 8: Namespace Config Tests
# ============================================================================

class TestNamespaceConfig:
    """Test NamespaceConfig (Task 8)."""
    
    def test_basic_config(self):
        """Test basic namespace config."""
        config = NamespaceConfig(name="tenant_123")
        assert config.name == "tenant_123"
        assert config.display_name is None
        assert config.labels == {}
        assert config.read_only is False
    
    def test_full_config(self):
        """Test full namespace config."""
        config = NamespaceConfig(
            name="tenant_123",
            display_name="Acme Corporation",
            labels={"tier": "enterprise", "region": "us-west"},
            read_only=False,
        )
        assert config.display_name == "Acme Corporation"
        assert config.labels["tier"] == "enterprise"
    
    def test_config_serialization(self):
        """Test config to/from dict."""
        config = NamespaceConfig(
            name="tenant_123",
            display_name="Test",
            labels={"key": "value"},
        )
        
        data = config.to_dict()
        assert data["name"] == "tenant_123"
        assert data["display_name"] == "Test"
        
        restored = NamespaceConfig.from_dict(data)
        assert restored.name == config.name
        assert restored.display_name == config.display_name


# ============================================================================
# Task 12: ContextQuery Tests
# ============================================================================

class TestTokenEstimator:
    """Test TokenEstimator."""
    
    def test_heuristic_estimator(self):
        """Test heuristic token estimation."""
        estimator = TokenEstimator()
        
        # ~4 chars per token
        text = "a" * 100  # 100 chars ≈ 25 tokens
        tokens = estimator.count(text)
        assert 20 <= tokens <= 30
    
    def test_custom_tokenizer(self):
        """Test custom tokenizer function."""
        def mock_tokenizer(text: str) -> int:
            return len(text.split())
        
        estimator = TokenEstimator(tokenizer=mock_tokenizer)
        assert estimator.count("hello world foo bar") == 4


class TestContextChunk:
    """Test ContextChunk."""
    
    def test_basic_chunk(self):
        """Test basic context chunk."""
        chunk = ContextChunk(
            id=1,
            text="Sample text content",
            score=0.85,
            tokens=5,
        )
        assert chunk.id == 1
        assert chunk.text == "Sample text content"
        assert chunk.score == 0.85
        assert chunk.tokens == 5
    
    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = ContextChunk(
            id="doc_123",
            text="Content",
            score=0.9,
            tokens=10,
            source="file.txt",
            metadata={"page": 5},
        )
        assert chunk.source == "file.txt"
        assert chunk.metadata["page"] == 5
    
    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = ContextChunk(
            id=1,
            text="Text",
            score=0.5,
            tokens=2,
            source="src",
        )
        data = chunk.to_dict()
        assert data["id"] == 1
        assert data["text"] == "Text"
        assert data["source"] == "src"


class TestContextResult:
    """Test ContextResult."""
    
    def test_empty_result(self):
        """Test empty context result."""
        result = ContextResult(
            chunks=[],
            total_tokens=0,
            budget_tokens=4000,
        )
        assert len(result) == 0
        assert result.as_text() == ""
    
    def test_result_with_chunks(self):
        """Test result with chunks."""
        chunks = [
            ContextChunk(id=1, text="First chunk", score=0.9, tokens=3),
            ContextChunk(id=2, text="Second chunk", score=0.8, tokens=3),
        ]
        result = ContextResult(
            chunks=chunks,
            total_tokens=6,
            budget_tokens=100,
        )
        
        assert len(result) == 2
        assert result[0].text == "First chunk"
        
        text = result.as_text(separator="|")
        assert "First chunk" in text
        assert "Second chunk" in text
        assert "|" in text
    
    def test_result_as_markdown(self):
        """Test markdown formatting."""
        chunks = [
            ContextChunk(
                id=1, 
                text="Content here", 
                score=0.9, 
                tokens=3,
                source="doc.txt",
            ),
        ]
        result = ContextResult(chunks=chunks, total_tokens=3, budget_tokens=100)
        
        md = result.as_markdown(include_scores=True)
        assert "### Context 1" in md
        assert "doc.txt" in md
        assert "0.9" in md
        assert "Content here" in md
    
    def test_result_as_json(self):
        """Test JSON formatting."""
        chunks = [ContextChunk(id=1, text="Text", score=0.5, tokens=2)]
        result = ContextResult(chunks=chunks, total_tokens=2, budget_tokens=100)
        
        import json
        data = json.loads(result.as_json())
        assert len(data["chunks"]) == 1
        assert data["total_tokens"] == 2


class TestContextQueryBuilder:
    """Test ContextQuery builder."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock collection."""
        collection = Mock()
        collection._config = CollectionConfig(name="test", dimension=384)
        
        # Mock search to return empty results
        collection.search = Mock(return_value=SearchResults(
            results=[],
            total_count=0,
            query_time_ms=1.0,
        ))
        
        return collection
    
    def test_builder_fluent_api(self, mock_collection):
        """Test fluent API returns self."""
        query = ContextQuery(mock_collection)
        
        result = query.add_vector_query([0.1] * 384)
        assert result is query
        
        result = query.add_keyword_query("test")
        assert result is query
        
        result = query.with_token_budget(2000)
        assert result is query
        
        result = query.with_min_relevance(0.5)
        assert result is query
    
    def test_builder_requires_query(self, mock_collection):
        """Test builder requires at least one query component."""
        query = ContextQuery(mock_collection)
        
        with pytest.raises(ValueError):
            query.execute()
    
    def test_vector_query_only(self, mock_collection):
        """Test vector-only query."""
        query = (
            ContextQuery(mock_collection)
            .add_vector_query([0.1] * 384, weight=1.0)
            .with_token_budget(1000)
        )
        
        result = query.execute()
        assert isinstance(result, ContextResult)
        assert result.budget_tokens == 1000
    
    def test_keyword_query_only(self, mock_collection):
        """Test keyword-only query."""
        query = (
            ContextQuery(mock_collection)
            .add_keyword_query("machine learning")
            .with_token_budget(1000)
        )
        
        result = query.execute()
        assert isinstance(result, ContextResult)
    
    def test_hybrid_query(self, mock_collection):
        """Test hybrid query with both vector and keyword."""
        query = (
            ContextQuery(mock_collection)
            .add_vector_query([0.1] * 384, weight=0.7)
            .add_keyword_query("test", weight=0.3)
            .with_token_budget(2000)
        )
        
        result = query.execute()
        assert isinstance(result, ContextResult)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_estimate_tokens(self):
        """Test estimate_tokens function."""
        # Uses heuristic when tiktoken not available
        tokens = estimate_tokens("Hello world, this is a test.")
        assert tokens > 0
    
    def test_split_by_tokens(self):
        """Test split_by_tokens function."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = split_by_tokens(text, max_tokens=10, overlap_tokens=2)
        
        assert len(chunks) > 0
        # All chunks should contain text
        for chunk in chunks:
            assert len(chunk) > 0


class TestDeduplication:
    """Test deduplication strategies."""
    
    def test_deduplication_enum(self):
        """Test DeduplicationStrategy enum."""
        assert DeduplicationStrategy.NONE == "none"
        assert DeduplicationStrategy.EXACT == "exact"
        assert DeduplicationStrategy.SEMANTIC == "semantic"


# ============================================================================
# Integration Tests (Mock-based)
# ============================================================================

class TestNamespaceIntegration:
    """Integration tests for namespace API."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db._closed = False
        db._namespaces = {}
        db.put = Mock()
        db.get = Mock(return_value=None)
        db.scan_prefix = Mock(return_value=iter([]))
        db.transaction = Mock()
        return db
    
    def test_namespace_context_manager(self, mock_db):
        """Test namespace as context manager."""
        config = NamespaceConfig(name="test")
        ns = Namespace(mock_db, "test", config)
        
        with ns as active_ns:
            assert active_ns is ns
            assert active_ns.name == "test"
    
    def test_collection_crud(self, mock_db):
        """Test collection CRUD operations."""
        ns = Namespace(mock_db, "test")
        
        # Create collection
        collection = ns.create_collection("documents", dimension=384)
        assert collection.name == "documents"
        assert collection.config.dimension == 384
        
        # Get collection
        same_collection = ns.collection("documents")
        assert same_collection is collection
        
        # List collections
        names = ns.list_collections()
        assert "documents" in names
        
        # Create duplicate should fail
        with pytest.raises(CollectionExistsError):
            ns.create_collection("documents", dimension=384)
    
    def test_collection_search_methods(self, mock_db):
        """Test collection search convenience methods."""
        ns = Namespace(mock_db, "test")
        config = CollectionConfig(
            name="docs",
            dimension=384,
            enable_hybrid_search=True,
        )
        collection = Collection(ns, config)
        
        # Vector search
        results = collection.vector_search([0.1] * 384, k=10)
        assert isinstance(results, SearchResults)
        
        # Keyword search (requires hybrid enabled)
        results = collection.keyword_search("test", k=10)
        assert isinstance(results, SearchResults)
        
        # Hybrid search
        results = collection.hybrid_search([0.1] * 384, "test", k=10)
        assert isinstance(results, SearchResults)
    
    def test_keyword_search_requires_hybrid(self, mock_db):
        """Test keyword search fails without hybrid enabled."""
        ns = Namespace(mock_db, "test")
        config = CollectionConfig(
            name="docs",
            dimension=384,
            enable_hybrid_search=False,  # Disabled
        )
        collection = Collection(ns, config)
        
        with pytest.raises(CollectionConfigError):
            collection.keyword_search("test")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
