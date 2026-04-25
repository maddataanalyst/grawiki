# API Overview

GraWiki exposes a layered API, but most users should start with [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG]. It provides the end-to-end ingestion, search, memory, and entity-deduplication workflows in one facade.

Use this section in the following order:

1. [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] for the high-level application surface.
2. Retrieval pages for query-time search behavior.
3. Graph model pages for persisted node and relationship shapes.
4. Database abstractions when implementing or debugging a backend.
5. Similarity and deduplication pages when inspecting duplicate entities or running merges.

Stable-first navigation:

- Start with the facade and the abstractions pages for normal application use.
- Treat the extraction and FalkorDB adapter pages as advanced reference material.
- Treat helper modules with leading underscores as internal and intentionally undocumented here.

The generated API sections are backed by docstrings from `src/`, so the reference stays aligned with the code that ships.

::: grawiki
