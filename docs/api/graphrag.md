# GraphRAG

[`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] is the main public facade. It combines document ingestion, chunk-level graph extraction, retrieval, agent-memory persistence, and duplicate-entity inspection in one class.

`GraphRAG` always keeps the generic text chunker ready and can optionally add a markdown-aware pipeline path:

- the generic text `Chunker` for plain-text content,
- an explicit markdown pipeline adapter for markdown content when `markdown_pipeline=` is provided.

`read_document(...)` is the only file-format detection step. It marks `.md` and `.markdown` files as markdown content, converts `.pdf` files to markdown in memory, and leaves other files as plain text. `ingest_text(...)` does not auto-detect content; callers choose `format="text"` or `format="markdown"` explicitly. Both `ingest(...)` and `ingest_text(...)` then run the same private ingestion flow for chunking, optional `process_chunks(...)`, embedding, persistence, and extraction.

When you pass `chunk_processors=` to `GraphRAG(...)`, those processors run after chunking and before chunk embeddings and chunk-level graph extraction.

In the default ingestion flow, vector embeddings are created for chunks, entities, memories, and queries, while document nodes are persisted without document-level vectors.

The main entry points are:

- `ingest(...)` and `ingest_text(...)` for document ingestion.
- `search(...)` and `recall(...)` for query-time retrieval.
- `remember(...)` for writing memory nodes.
- `find_entity_duplicate_candidates(...)` and `dedupe_entities(...)` for duplicate inspection and merge execution.

When `resolve_entities_on_ingest=True`, the same similarity infrastructure used for duplicate inspection is also applied during ingestion. Extracted entities can then be matched to persisted entities before new nodes are written.

For end-to-end examples, see [Flows](../flows.md) and the task-oriented guides in [How to](../how-to/index.md).

::: grawiki.rag.graph_rag.GraphRAG
