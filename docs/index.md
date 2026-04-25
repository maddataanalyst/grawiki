# GraWiki

![GraWiki](assets/images/grawiki_text_logo.png){ width="420" }

GraWiki is an early-stage open source Python project that combines two ideas:

1. GraphRAG-style knowledge extraction and retrieval.
2. LLM Wiki style memory for agents.

The core premise is simple: use an LLM to extract structured knowledge from
documents, persist that knowledge in a graph, and treat the same graph as part
of an LLM system's long-lived memory.

GraWiki is not positioned as a finished framework. It is a work in progress and
an experimental repository for building practical graph-backed memory and
retrieval workflows.

## What the project covers

GraWiki is built around a small end-to-end surface:

- ingest source documents into a typed property graph,
- retrieve graph-backed context from documents, chunks, memories, and entities,
- persist agent memory as first-class graph nodes,
- inspect and merge duplicate entities through similarity-based workflows.

## Current capabilities

- Read source documents and split them into chunks.
- Extract entities and relationships from chunk text.
- Persist documents, chunks, entities, and edges in a graph database.
- Retrieve graph-backed context with full-text and vector search.
- Store and recall agent memories as dedicated graph nodes.
- Inspect semantic-key collisions and broader duplicate candidates.
- Merge duplicate entities into canonical masters.

## Why this project exists

Most RAG systems treat retrieved context as an ephemeral document slice. GraWiki
pushes toward a more persistent knowledge organization model: documents can be
transformed into graph structure, and agent interactions can leave behind memory
nodes that can later be searched and expanded through connected context.

That framing is closer to a lightweight, graph-backed "wiki" for an LLM system
than to a narrow document-search pipeline alone [@edge2024fromlocal; @karpathy2026llmwiki].

## Repository structure

The repository is organized around a few major areas:

- `src/grawiki/` contains the reusable project code.
- `tests/` contains pytest coverage for the facade, retrieval, graph models, extraction, and the FalkorDB adapter.
- `docs/` contains the public MkDocs site, including generated API pages under `docs/api/`.
- `agent_tools/` contains internal contributor and agent-facing guidance such as `CODEMAP.md`.
- `notebooks/` contains exploratory and debugging material.

At the package level:

- `grawiki.core` holds shared source-data types and the embedding protocol.
- `grawiki.doc_processing` handles document loading and chunking.
- `grawiki.graph` defines the graph schema and extraction logic.
- `grawiki.db` defines the database abstraction layer and FalkorDB backend.
- `grawiki.retrieval` owns query-time retrieval strategies.
- `grawiki.similarity` covers duplicate inspection, similarity matchers, and deduplication helpers.
- `grawiki.rag` exposes the `GraphRAG` facade.

For a fuller map of the repository, see the dedicated project structure page.
