# AGENTS.md

## Project Overview

GraWiki is an early-stage open source Python project that combines GraphRAG-style knowledge extraction with Andrej Karpathy's "LLM Wiki" concept for agentic memory.

The repository currently focuses on two core capabilities:

1. Extracting structured knowledge from text and storing it in a graph representation.
2. Storing and retrieving agent memories built from prior queries, retrieved context, reasoning, and answers.

## Repository Layout

- `src/grawiki/`: main application package.
- `src/grawiki/core/`: shared source-data models and the `Embedding` protocol.
- `src/grawiki/doc_processing/`: document loading and chunking.
- `src/grawiki/db/`: graph database abstractions, Cypher builders, and FalkorDB adapter.
- `src/grawiki/graph/`: graph models, prompts, and extraction logic.
- `src/grawiki/retrieval/`: retrieval strategy layer and query-time embedding.
- `src/grawiki/rag/`: high-level `GraphRAG` facade.
- `tests/`: pytest coverage for the facade, retrieval primitives, and FalkorDB adapter.
- `main.py`: minimal entrypoint.
- `notebooks/`: exploratory research and debugging notebooks.

## Current API Shape

- `GraphRAG` is the main public facade, but it now also exposes modular public step methods for notebooks and debugging: `read_document`, `chunk_document`, `embed_document`, `embed_chunks`, `build_document_node`, `build_chunk_nodes`, `persist_document_and_chunks`, `extract_kg_per_chunk`, and `persist_entities_and_relationships`.
- Shared embedding is defined by `src/grawiki/core/embedding.py:Embedding`. The concrete default is `pydantic_ai.Embedder`, instantiated directly by callers rather than wrapped in a local `DefaultEmbedder` class.
- Retrieval owns query-time embedding. `Retriever.vector(...)` embeds the query first, then calls `GraphDB.vector_search(...)`; DB adapters should not embed queries themselves.
- `GraphDB` now has a clearer primitive layer: `ensure_indexes`, `upsert_nodes`, `upsert_relationships`, `fulltext_search`, `vector_search`, and `neighbors`. Legacy helpers such as `save_docs_and_chunks_to_db`, `save_entities_and_rels`, and grouped `search` still exist as convenience wrappers.
- `FalkorGraphDB` requires explicit `close()` in tests and short-lived scripts because it manages an embedded FalkorDBLite/Redis process.

## Coding Rules

1. Always use `uv` for package management.
   - Add and update dependencies with `uv add`, `uv remove`, and `uv sync`.
   - Do not use `pip install`, `poetry`, or other package managers for this repository.

2. Write docstrings in NumPy format everywhere.
   - Apply NumPy-style docstrings to modules, classes, functions, and methods.
   - Keep docstrings accurate and update them when behavior or signatures change.

3. Use `pytest` for testing.
   - Add tests for new behavior and regressions.
   - Prefer deterministic unit tests over notebook-only validation.

4. Run linting before commits.
   - Use the configured `pre-commit` hooks before committing changes.
   - Ensure `ruff` checks pass before opening or updating a pull request.

## Standard Developer Commands

Install development dependencies:

```bash
uv sync --group dev
```

Install git hooks:

```bash
uv run pre-commit install
```

Run all lint checks:

```bash
uv run pre-commit run --all-files
```

Run tests:

```bash
uv run pytest
```
