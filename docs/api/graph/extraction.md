# Extraction

This page covers the advanced extraction layer that turns raw text into a graph-shaped intermediate representation before persistence. Most users should use extraction through [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] rather than constructing these pieces directly.

For the public ingestion flow and stepwise examples, start with [Flows](../../flows.md) and [How to ingest a document](../../how-to/ingest-a-document.md).

## Structured output with Instructor

`KnowledgeGraphExtractor` relies on [Instructor](https://python.useinstructor.com/) for structured LLM output. When `extract(...)` is called, the chunk text is sent to the configured chat model together with a system prompt that defines the desired node and relationship schema. Instructor requests the model to return JSON matching the `ExtractedKnowledgeGraph` Pydantic model, validates the response, and surfaces any schema violations as early as possible. This removes the need for manual JSON parsing or ad-hoc regex extraction.

```python
from grawiki.graph.extraction import KnowledgeGraphExtractor

extractor = KnowledgeGraphExtractor(
    model="openai:gpt-4.1-mini",
    embedding=embedder,
)
graph = await extractor.extract("Alan Turing was a pioneering computer scientist.")
```

The resulting `graph` is a `KnowledgeGraph` whose nodes already carry embeddings and durable UUIDs, ready for persistence.

::: grawiki.graph.extraction
