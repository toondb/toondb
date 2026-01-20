# Token-Aware Context Query

SochDB provides a `ContextQuery` builder for retrieving context optimized for
LLM prompts. It handles token budgeting, relevance scoring, deduplication, and
structured output.

## Overview

The ContextQuery builder provides:

1. **Token Budgeting** - Fit context within model limits
2. **Hybrid Search** - Combine vector + keyword queries
3. **Relevance Scoring** - Prioritize most relevant chunks
4. **Deduplication** - Avoid repeating similar content
5. **Structured Output** - Ready for LLM prompts

## Quick Start

### Python

```python
from sochdb import Database
from sochdb.context import ContextQuery, DeduplicationStrategy

db = Database.open("./my_data")

# Build context query
result = (
    ContextQuery(collection)
    .add_vector_query(query_embedding, weight=0.7)
    .add_keyword_query("machine learning", weight=0.3)
    .with_token_budget(4000)
    .with_min_relevance(0.5)
    .with_deduplication(DeduplicationStrategy.SEMANTIC, threshold=0.9)
    .execute()
)

# Use in prompt
prompt = f"{result.as_text()}\n\nQuestion: {question}"

# Check metrics
print(f"Tokens: {result.total_tokens}/{result.budget_tokens}")
print(f"Chunks: {len(result.chunks)}, Dropped: {result.dropped_count}")
```

### Go

```go
query := sochdb.NewContextQuery(db, "documents").
    AddVectorQuery(embedding, 0.7).
    AddKeywordQuery("machine learning", 0.3).
    WithTokenBudget(4000).
    WithMinRelevance(0.5).
    WithDeduplication(sochdb.DeduplicationSemantic, 0.9)

result, err := query.Execute()
if err != nil {
    log.Fatal(err)
}

// Use in prompt
prompt := result.AsText("\n\n---\n\n") + "\n\nQuestion: " + question

// Check metrics
fmt.Printf("Tokens: %d/%d\n", result.TotalTokens, result.BudgetTokens)
```

### TypeScript/Node.js

```typescript
import { ContextQuery, DeduplicationStrategy } from '@sochdb/sochdb';

const result = await new ContextQuery(db, 'documents')
  .addVectorQuery(embedding, 0.7)
  .addKeywordQuery('machine learning', 0.3)
  .withTokenBudget(4000)
  .withMinRelevance(0.5)
  .withDeduplication(DeduplicationStrategy.SEMANTIC, 0.9)
  .execute();

// Use in prompt
const prompt = `${result.asText()}\n\nQuestion: ${question}`;

// Check metrics
console.log(`Tokens: ${result.totalTokens}/${result.budgetTokens}`);
```

### Rust

```rust
use sochdb_client::context_query::{ContextQueryBuilder, TruncationStrategy};

let result = ContextQueryBuilder::with_connection(conn.clone())
    .for_session("session_123")
    .with_budget(4000)
    .section("DOCS", 0)
        .search("documents", embedding, 10)
        .done()
    .execute()?;

println!("Tokens: {}", result.token_count);
```

## Token Estimation

By default, tokens are estimated using a heuristic (4 chars ≈ 1 token).
For accuracy, provide a tokenizer:

### Python (with tiktoken)

```python
from sochdb.context import TokenEstimator

# Use tiktoken for accurate counting
estimator = TokenEstimator.tiktoken(model="gpt-4")

result = (
    ContextQuery(collection)
    .with_tokenizer(estimator)
    .with_token_budget(4000)
    .execute()
)
```

### Node.js (with tiktoken)

```typescript
import { TokenEstimator } from '@sochdb/sochdb';

const estimator = TokenEstimator.tiktoken('gpt-4');

const result = await new ContextQuery(db, 'docs')
  .withTokenizer((text) => estimator.count(text))
  .withTokenBudget(4000)
  .execute();
```

## Query Types

### Vector Query

Semantic similarity search using embeddings:

```python
# Single vector query
query.add_vector_query(embedding, weight=1.0)

# With custom top-k
query.add_vector_query_with_k(embedding, weight=0.7, top_k=100)
```

### Keyword Query

BM25-style text matching:

```python
# Single keyword query
query.add_keyword_query("neural networks", weight=1.0)

# With custom top-k
query.add_keyword_query_with_k("transformer", weight=0.3, top_k=50)
```

### Hybrid Queries

Combine multiple queries with RRF fusion:

```python
result = (
    ContextQuery(collection)
    .add_vector_query(query_embedding, weight=0.7)
    .add_keyword_query("deep learning", weight=0.2)
    .add_keyword_query("neural network", weight=0.1)
    .with_fusion_k(60)  # RRF k parameter
    .execute()
)
```

## Deduplication Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `NONE` | No deduplication | Maximum recall |
| `EXACT` | Exact text match | Remove identical chunks |
| `SEMANTIC` | Similarity threshold | Remove near-duplicates |

```python
# Exact deduplication
query.with_deduplication(DeduplicationStrategy.EXACT)

# Semantic deduplication (0.9 = 90% similarity threshold)
query.with_deduplication(DeduplicationStrategy.SEMANTIC, threshold=0.9)
```

## Truncation Strategies

When token budget is exceeded:

| Strategy | Description |
|----------|-------------|
| `TAIL_DROP` | Drop lowest-scored chunks (default) |
| `HEAD_DROP` | Drop highest-scored chunks |
| `PROPORTIONAL` | Truncate text proportionally |
| `STRICT` | Fail if budget exceeded |

```python
from sochdb.context import TruncationStrategy

query.with_truncation(TruncationStrategy.PROPORTIONAL)
```

## Output Formatting

### As Text

```python
# Default separator
context = result.as_text()

# Custom separator
context = result.as_text(separator="\n\n===\n\n")
```

### As Markdown

```python
# Without scores
markdown = result.as_markdown()

# With relevance scores
markdown = result.as_markdown(include_scores=True)
```

Output:
```markdown
## Chunk 1 (score: 0.95)

Deep learning models use neural networks...

## Chunk 2 (score: 0.87)

Transformer architectures have revolutionized...
```

## Context Chunks

Each chunk contains:

```python
for chunk in result.chunks:
    print(f"ID: {chunk.id}")
    print(f"Text: {chunk.text}")
    print(f"Score: {chunk.score}")
    print(f"Tokens: {chunk.tokens}")
    print(f"Source: {chunk.source}")
    print(f"Metadata: {chunk.metadata}")
```

## Result Metrics

```python
result = query.execute()

# Token usage
print(f"Tokens used: {result.total_tokens}")
print(f"Token budget: {result.budget_tokens}")

# Chunk counts
print(f"Chunks returned: {len(result.chunks)}")
print(f"Chunks dropped: {result.dropped_count}")

# Score ranges
if result.vector_score_range:
    print(f"Vector scores: {result.vector_score_range[0]:.3f} - {result.vector_score_range[1]:.3f}")
if result.keyword_score_range:
    print(f"Keyword scores: {result.keyword_score_range[0]:.3f} - {result.keyword_score_range[1]:.3f}")
```

## Advanced Options

### Recency Weighting

Boost recent documents:

```python
query.with_recency_weight(0.2)  # 20% weight on recency
```

### Diversity Weighting

Encourage diverse results:

```python
query.with_diversity_weight(0.3)  # 30% weight on diversity
```

### Custom Top-K per Query

```python
query.add_vector_query_with_k(embedding, weight=0.7, top_k=100)
query.add_keyword_query_with_k("search term", weight=0.3, top_k=50)
```

### Max Chunks Limit

```python
query.with_max_chunks(20)  # Maximum 20 chunks regardless of budget
```

## Helper Functions

### Estimate Tokens

```python
from sochdb.context import estimate_tokens

tokens = estimate_tokens("Hello, world!")
# ~3 tokens
```

### Split by Tokens

```python
from sochdb.context import split_by_tokens

chunks = split_by_tokens(long_text, max_tokens_per_chunk=500, overlap=50)
```

## Best Practices

1. **Set realistic budgets**: Leave room for instructions (e.g., 4000 for 8000 context)
2. **Use hybrid search**: Combine vector + keyword for best results
3. **Deduplicate for diversity**: Use semantic deduplication to avoid redundancy
4. **Use accurate tokenizers**: tiktoken for OpenAI, transformers for others
5. **Monitor dropped chunks**: High drop rates indicate tight budgets

## See Also

- [Graph Overlay](graph-overlay.md) - Agent memory relationships
- [Policy Hooks](policy-hooks.md) - Safety policies
- [Vector Search](vector-search.md) - HNSW indexing
