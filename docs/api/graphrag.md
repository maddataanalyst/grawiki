# GraphRAG

[`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] is the main public facade. It combines document ingestion, chunk-level graph extraction, retrieval, agent-memory persistence, and duplicate-entity inspection behind one class.

The most important flows are:

- `ingest(...)` and `ingest_text(...)` for document ingestion.
- `search(...)` and `recall(...)` for query-time retrieval.
- `remember(...)` for writing memory nodes.
- `find_entity_duplicate_candidates(...)` and `dedupe_entities(...)` for duplicate inspection and merge execution.

When `resolve_entities_on_ingest=True`, the same similarity infrastructure used for duplicate inspection is also applied during ingestion so extracted entities can be matched to persisted entities before new nodes are written.

::: grawiki.rag.graph_rag.GraphRAG
