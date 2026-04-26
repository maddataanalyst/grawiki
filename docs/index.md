# GraWiki

![GraWiki](assets/images/grawiki_text_logo.png){ width="420" }

GraWiki is an early-stage open source Python project with two main concerns:

1. GraphRAG-style knowledge extraction and retrieval.
2. LLM Wiki style memory for agents.

It uses an LLM to extract structured knowledge from documents, persists that knowledge in a graph, and uses the same graph as part of an LLM system's long-lived memory.

The project is still experimental. It is a working repository for graph-backed memory and retrieval workflows rather than a finished framework.

## What the project covers

GraWiki focuses on a small end-to-end surface:

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

Many RAG systems treat retrieved context as a temporary document slice. GraWiki instead stores document structure and agent memory in a persistent graph that can be searched later and expanded through connected context.

That design is closer to a lightweight graph-backed "wiki" for an LLM system than to a document-search pipeline alone [@edge2024fromlocal; @karpathy2026llmwiki].

## How to use the docs

- Start with [Flows](flows.md) for the main ingestion and retrieval paths.
- Use [How to](how-to/index.md) for task-oriented guides derived from the maintained notebooks.
- Use [API Overview](api/index.md) when you need the facade and lower-level references.

## Repository structure

The repository is organized around a few major areas:

- `src/grawiki/` contains the reusable project code.
- `tests/` contains pytest coverage for the facade, retrieval, graph models, extraction, and the FalkorDB adapter.
- `docs/` contains the public MkDocs site, including generated API pages under `docs/api/`.
- `agent_tools/` contains internal contributor and agent-facing guidance such as `CODEMAP.md`.
- `notebooks/` contains focused tutorial notebooks, supporting debug scripts, and sample text inputs.

At the package level:

- `grawiki.core` holds shared source-data types and the embedding protocol.
- `grawiki.doc_processing` handles document loading and chunking.
- `grawiki.graph` defines the graph schema and extraction logic.
- `grawiki.db` defines the database abstraction layer and FalkorDB backend.
- `grawiki.retrieval` owns query-time retrieval strategies.
- `grawiki.similarity` covers duplicate inspection, similarity matchers, and deduplication helpers.
- `grawiki.rag` exposes the `GraphRAG` facade.

For a fuller map of the repository, see [Project Structure](structure.md).

## Maintained notebooks

The maintained notebook flow lives in three numbered notebooks under `notebooks/`:

- `01_ingest_and_deduplicate.ipynb`
- `02_agent_memory_and_recall.ipynb`
- `03_visualize_graph.ipynb`

Run notebook 1 first to build the local FalkorDB graph. Notebook 2 reuses that graph for agent memory examples, and notebook 3 visualizes the resulting graph.

To install the tutorial dependencies, choose one:

- For file-based (FalkorDBLite):
  ```bash
  uv sync --group falkordblite --extra notebooks --extra viz
  ```

- For Docker-based (full FalkorDB):
  ```bash
  uv sync --group falkordb --extra notebooks --extra viz
  ```

The sample texts used there are Medium articles by Filip Wojcik. They are available from his public Medium profile and are accessible without a subscription.
