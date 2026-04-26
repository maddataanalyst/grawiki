# GraphRAG

[`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] is the main public facade. It combines document ingestion, chunk-level graph extraction, retrieval, agent-memory persistence, and duplicate-entity inspection in one class.

The main entry points are:

- `ingest(...)` and `ingest_text(...)` for document ingestion.
- `search(...)` and `recall(...)` for query-time retrieval.
- `remember(...)` for writing memory nodes.
- `find_entity_duplicate_candidates(...)` and `dedupe_entities(...)` for duplicate inspection and merge execution.

When `resolve_entities_on_ingest=True`, the same similarity infrastructure used for duplicate inspection is also applied during ingestion. Extracted entities can then be matched to persisted entities before new nodes are written.

For end-to-end examples, see [Flows](../flows.md) and the task-oriented guides in [How to](../how-to/index.md).

::: grawiki.rag.graph_rag.GraphRAG
