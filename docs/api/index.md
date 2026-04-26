# API Overview

GraWiki exposes a layered API. Most users should start with [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG], which provides ingestion, search, memory, and entity-deduplication workflows through one facade.

Use this section in the following order:

1. [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] for the high-level application surface.
2. Retrieval pages for query-time search behavior.
3. Graph model pages for persisted node and relationship shapes.
4. Database abstractions when implementing or debugging a backend.
5. Similarity and deduplication pages when inspecting duplicate entities or running merges.

At a high level, the API is split into facade-level entry points and lower-level implementation layers:

- `GraphRAG` is the normal application surface.
- Retrieval, graph, database, and similarity pages document the subsystems that `GraphRAG` composes.
- The extraction and FalkorDB adapter pages are advanced reference material.
- Helper modules with leading underscores are internal and intentionally undocumented here.

For task-oriented examples, use the [How to](../how-to/index.md) section alongside this reference. The generated API sections are backed by docstrings from `src/`, so the reference stays aligned with the code that ships.

::: grawiki
