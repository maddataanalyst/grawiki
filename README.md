# GraWiki

GraWiki is an early-stage open source Python project for graph-backed knowledge
extraction, retrieval, and agent memory.

It combines two closely related ideas:

1. GraphRAG-style ingestion and retrieval over extracted entities and relationships.
2. Andrej Karpathy's "LLM Wiki" style memory, where prior agent work is stored as durable graph state instead of only transient prompt context.

The project uses an LLM to turn text into structured graph data, persists that
data in a graph database, and then reuses the same graph for search, memory
recall, similarity inspection, and duplicate-entity cleanup.

## What GraWiki does

GraWiki currently focuses on two main workflows.

1. Document-to-graph ingestion.
   Source documents are read, chunked, embedded, processed by an LLM extractor,
   and persisted as document nodes, chunk nodes, entity nodes, and typed
   relationships.
2. Graph-backed agent memory.
   Agent outputs can be stored as dedicated `__memory__` nodes, linked back into
   the graph, and later recalled together with connected context.

Current capabilities include:

- reading source documents and splitting them into chunks,
- extracting entities and relationships from chunk text,
- persisting documents, chunks, entities, relationships, and memories in a graph database,
- retrieving graph-backed context with vector and full-text search,
- expanding graph context around matched entities and memories,
- inspecting duplicate candidates through semantic-key collision checks and pluggable similarity matchers,
- merging duplicate entities through the facade-level deduplication workflow.

## Installation

Install the base package:

```bash
pip install grawiki
```

Install the local file-backed FalkorDBLite backend:

```bash
pip install 'grawiki[falkordblite]'
```

Install the full FalkorDB server backend:

```bash
pip install 'grawiki[falkordb]'
```

Useful optional extras:

- `grawiki[notebooks]` for the maintained notebooks.
- `grawiki[viz]` for `networkx` and `matplotlib` graph visualization.
- `grawiki[docs]` for local MkDocs builds.
- `grawiki[all]` for the full optional dependency set.

## Package layout

The public repository is organized around a small number of major areas.

- `src/grawiki/`: main application package.
- `tests/`: pytest coverage for the facade, retrieval layer, graph models, extraction, query generation, and FalkorDB adapter.
- `docs/`: public MkDocs documentation, including narrative pages and generated API reference pages under `docs/api/`.
- `notebooks/`: maintained tutorial notebooks plus sample input data.

### Main package structure

- `grawiki.core`: shared source-data types and the embedding protocol.
- `grawiki.doc_processing`: document loading and chunking.
- `grawiki.graph`: graph schema, extraction, and prompts.
- `grawiki.db`: backend-agnostic database interfaces plus the FalkorDB implementation.
- `grawiki.retrieval`: query-time retrieval strategies.
- `grawiki.similarity`: duplicate-candidate inspection, similarity matchers, and deduplication helpers.
- `grawiki.rag`: the `GraphRAG` facade that ties ingestion, retrieval, memory, and deduplication together.

### Core entrypoints

- `grawiki.GraphRAG`: the main public facade.
- `src/grawiki/rag/graph_rag.py`: end-to-end ingestion, search, recall, memory, and deduplication flows.
- `src/grawiki/graph/models.py`: the canonical graph schema.
- `src/grawiki/db/base.py`: the backend contract.
- `src/grawiki/similarity/`: the duplicate-inspection and merge-support surface.

## Runtime flow

At a high level, GraWiki works like this:

1. A source document is loaded and split into chunks.
2. Documents and chunks are embedded and persisted.
3. Each chunk is sent to an LLM extractor to produce nodes and relationships.
4. Extracted entities can optionally be resolved against existing persisted entities during ingest.
5. The resulting graph is stored and becomes available for retrieval, memory linking, and later deduplication.
6. Queries are handled through configured retrievers that combine text search, vector search, and graph-context expansion.
7. Memory recall searches `__memory__` nodes first, then expands linked graph context around them.

## Documentation

Public documentation lives in `docs/` and is built with MkDocs Material. It includes:

- conceptual background,
- flow documentation,
- a project structure page,
- generated API reference pages centered on `GraphRAG`.

## Tutorial notebooks

The repository ships three tutorial notebooks under `notebooks/`:

- `01_ingest_and_deduplicate.ipynb`: step-by-step ingestion, entity inspection, duplicate finding, deduplication, and final querying.
- `02_agent_memory_and_recall.ipynb`: a `pydantic_ai.Agent` wired to `GraphRAG.search(...)`, `GraphRAG.remember(...)`, and `GraphRAG.recall(...)`.
- `03_visualize_graph.ipynb`: a lightweight graph view built with optional `networkx` and `matplotlib`.

Run notebook 1 first. Notebook 2 reuses the same FalkorDB graph, and notebook 3 visualizes that populated graph.

The sample texts used by the notebooks live in `notebooks/experimental_data/`. They are Medium.com articles by Filip Wojcik, sourced from `https://medium.com/@filip.igor.wojcik`, and are fully accessible without a subscription.

## Development

Install development tooling:

```bash
uv sync --group dev
```

Install development tooling with the FalkorDBLite notebook stack:

```bash
uv sync --group dev --extra falkordblite --extra notebooks --extra viz
```

Install development tooling with the Docker-backed FalkorDB stack:

```bash
uv sync --group dev --extra falkordb --extra notebooks --extra viz
```

Install the documentation toolchain:

```bash
uv sync --group dev --extra docs
```

Install git hooks:

```bash
uv run pre-commit install
```

Run all configured checks manually:

```bash
uv run pre-commit run --all-files
```

Run the test suite:

```bash
uv run pytest
```

Build the public documentation site locally:

```bash
uv run mkdocs build --strict
```
