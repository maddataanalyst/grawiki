# CODEMAP

This file is a quick orientation guide for agents and contributors working in the GraWiki repository.

## Project Purpose

GraWiki is a Python project for:

1. Reading source documents.
2. Splitting them into chunks.
3. Extracting a structured knowledge graph from those chunks with an LLM.
4. Persisting documents, chunks, entities, and relationships in a graph database.
5. Searching the stored graph with full-text and vector retrieval.
6. Inspecting potentially duplicate entities through semantic-key collision checks and pluggable similarity matchers.

In the future: part of the agentic memory. Memories will be added as a separate __entity__, indexed by time.
Agent will have an option to "recall" -- run dedicated graph search on memories and linked entities.

In short - this project tries to be a lightweight, simplified version of Llama Index graph RAG + graph memory.

## Top-Level Folders

### `src/grawiki/`

Main application package. This is where the reusable project code lives.

### `tests/`

Pytest suite covering `GraphRAG` behavior, the modular ingestion-step API,
the retrieval layer, graph models and extraction, Cypher query generation,
and the FalkorDB adapter.

### `notebooks/`

Exploratory and debugging material. This includes debug scripts, local database files, and experimental input data. Treat this area as research/support code rather than the primary application surface.

### `docs/`

Documentation for humans and agents. `CODEMAP.md` lives here.

## Top-Level Files

### `main.py`

Minimal placeholder entrypoint. It currently just prints a message and is not the main integration surface for ingestion.

### `pyproject.toml`

Project metadata, dependency declarations, and development tool configuration.

### `README.md`

Short project overview and developer setup commands.

### `AGENTS.md`

Repository-specific operating instructions for coding agents.

## Package Map: `src/grawiki`

### Namespace: `grawiki`

Top-level Python package for the project. `src/grawiki/__init__.py`
currently re-exports `GraphRAG`, the main public facade.

### Namespace: `grawiki.core`

Core shared types and embedding abstractions.

#### `src/grawiki/core/commons.py`

Defines the lightweight source-data models used before persistence:

- `Document`: raw input document with `id`, `title`, `content`, and metadata.
- `Chunk`: chunked segment of a document with `document_id`, `content`, and metadata.

These models are the bridge between file reading/chunking and graph persistence.

#### `src/grawiki/core/embedding.py`

Shared embedding contract.

Key responsibilities:

- Defines the `Embedding` protocol used across ingestion, extraction, and retrieval.
- Documents the async `embed_documents(...)` and `embed_query(...)` methods expected from an embedding implementation.
- Lets `GraphRAG`, `KnowledgeGraphExtractor`, and `Retriever` share one embedding client instance.

The default concrete implementation is now used directly from `pydantic_ai.Embedder` at call sites rather than wrapped in this module.

### Namespace: `grawiki.doc_processing`

Document loading and chunking utilities.

#### `src/grawiki/doc_processing/document_processing.py`

Simple source-document helpers:

- `read_document(file_path)`: reads a text file from disk and wraps it as a `Document`.
- `chunk_document(document, chunker)`: delegates chunk creation to a configured `Chunker`.

This module stays intentionally thin and acts as the boundary between filesystem input and in-memory document objects.

#### `src/grawiki/doc_processing/chunkers.py`

Chunking strategy wrapper around the `chonkie` library.

Key responsibilities:

- Defines `Chunker`.
- Selects one of several chunking backends: `fast`, `recursive`, `semantic`, `sentence`, or `token`.
- Converts external chunking results into project `Chunk` objects with generated UUIDs.

Use this module when changing how text is segmented before extraction and embedding.

### Namespace: `grawiki.graph`

Knowledge-graph-specific models, prompts, and extraction logic.

#### `src/grawiki/graph/models.py`

Core graph schema definitions used throughout the repository.

Main model groups:

- Base graph model:
  - `GraphModel`: shared strict Pydantic base with extra fields forbidden.
- Persisted graph objects:
  - `Node`: generic durable graph node with `id`, `label`, `semantic_key`, properties, and optional embedding.
  - `Relationship`: durable edge using node IDs as endpoints.
  - `KnowledgeGraph`: container for persisted nodes and relationships.
- System/internal node types:
  - `DocumentNode`: persisted representation of a source document.
  - `ChunkNode`: persisted representation of a document chunk.
  - `MemoryNode`: reserved model for agent memory storage.

This is the canonical schema module for the project. The transient
extractor-facing types (``ExtractedNode``, ``ExtractedRelationship``,
``ExtractedKnowledgeGraph``) live in ``graph/extraction.py`` because
they are an implementation detail of the LLM extractor and not part of
the persisted domain.

#### `src/grawiki/graph/prompts.py`

Stores the `KG_EXTRACTION_PROMPT` string used to instruct the LLM extractor.

The prompt defines:

- what counts as a node or relationship,
- output shape expectations,
- naming rules,
- forbidden internal labels,
- quality constraints for extraction.

Change this module when tuning extraction behavior without changing extraction code.

#### `src/grawiki/graph/extraction.py`

LLM-driven graph extraction implementation.

Key responsibilities:

- Defines `KnowledgeGraphExtractor`.
- Defines `KnowledgeGraphExtractorProtocol` for injecting alternate extractors in tests and notebooks.
- Defines the transient LLM-facing types: `ExtractedNode`, `ExtractedRelationship`, `ExtractedKnowledgeGraph`.
- Runs a `pydantic_ai.Agent` against chunk text.
- Produces structured `ExtractedKnowledgeGraph` output.
- Repairs missing node references when relationships point to node names not present in the output.
- Converts extractor output into durable `KnowledgeGraph` objects via the module-level `_node_from_extracted` helper.
- Adds embeddings to extracted entity nodes.

This module is the main bridge between raw chunk text and a persisted graph-ready structure.

### Namespace: `grawiki.db`

Database abstraction layer and FalkorDB implementation.

#### `src/grawiki/db/base.py`

Defines the backend-agnostic graph database contract.

Key contents:

- `GraphDB`: abstract base class for persistence and raw search primitives.
- `NodeHit`: flat hit shape used by the new retrieval layer.
- `NeighborRelationship`: one-hop graph-context expansion row used by retrieval.
- `SearchMethod`: literal type for supported retrieval modes (`fulltext` or `vector`).
- `SearchResults`: common grouped search result shape.

The `GraphDB` interface covers:

- database/index setup,
- explicit index management through `ensure_indexes(...)`,
- raw full-text and vector search,
- graph-neighbor expansion through `neighbor_relationships(...)`,
- entity enumeration through `list_entities(...)`,
- generic node and relationship upserts via `upsert_nodes(...)` and `upsert_relationships(...)`,
- legacy wrapper methods kept for compatibility during the migration.

Use this module when adding a new database backend or changing the shared persistence/search contract.

#### `src/grawiki/db/cypher.py`

Shared Cypher query builders.

Key responsibilities:

- Builds label-parameterized node upsert queries.
- Builds generic entity-to-entity relationship upserts.
- Builds explicit system-link queries such as `__has_chunk__` and `__mentions__`.
- Sanitizes labels and relationship types into backend-safe Cypher identifiers.
- Injects optional embedding assignments into query strings.

This module keeps Cypher string construction separate from adapter control flow.

#### `src/grawiki/db/falkordb.py`

Concrete `GraphDB` implementation backed by FalkorDBLite.

Key responsibilities:

- Defines `FalkorGraphDB`.
- Opens/selects a FalkorDB graph stored on disk.
- Creates and tracks full-text and vector indexes, including per-label vector dimensions.
- Persists nodes and relationships through `upsert_nodes(...)` and `upsert_relationships(...)`.
- Exposes raw full-text search, vector search, and neighbor expansion primitives.
- Serializes embeddings and metadata into backend-compatible forms.
- Exposes debugging helpers such as `query(...)`, `ro_query(...)`, `explain(...)`, `query_fulltext_nodes(...)`, and `query_similar_nodes(...)`.
- Requires explicit `close()` during teardown in tests and scripts.
- Provides lower-level query helpers for debugging and experimentation.

This is currently the main persistence backend used by the repository.

#### `src/grawiki/db/__init__.py`

Small export module that re-exports:

- `GraphDB`
- `FalkorGraphDB`

### Namespace: `grawiki.retrieval`

Retrieval strategy layer that owns query-side embeddings.

#### `src/grawiki/retrieval/base.py`

Defines the `Retriever` protocol used by the `GraphRAG` facade.

#### `src/grawiki/retrieval/text.py`

Key responsibilities:

- Defines `TextRetriever`.
- Embeds vector queries with the shared `Embedding` implementation.
- Calls `GraphDB.fulltext_search(...)` and `GraphDB.vector_search(...)`.
- Deduplicates flat `NodeHit` lists returned by the DB layer.
- Exposes `fulltext(...)` and `vector(...)` as the query-side strategy surface.

This module is where query strategy lives; the DB should stay a storage engine.

#### `src/grawiki/retrieval/keywords.py`

Keyword-path retriever that:

- extracts keyword phrases from the raw query,
- embeds the phrases,
- searches entity vectors,
- expands one-hop graph context with `GraphDB.neighbor_relationships(...)`,
- returns enriched `NodeHit` objects suitable for downstream RAG context.

### Namespace: `grawiki.similarity`

Entity similarity inspection and duplicate-candidate finding.

#### `src/grawiki/similarity/base.py`

Defines `EntitySimilarityMatcher`, the protocol implemented by concrete
entity-matching algorithms.

#### `src/grawiki/similarity/fuzzy.py`

Defines `RapidFuzzEntitySimilarityMatcher`, which:

- loads persisted entities via `GraphDB.list_entities(...)`,
- scores name similarity with `rapidfuzz.fuzz.WRatio`,
- filters by threshold and optionally by ontology label,
- returns ranked `NodeHit` candidate matches.

#### `src/grawiki/similarity/vector.py`

Defines `VectorEntitySimilarityMatcher`, which:

- loads persisted entities and embeddings via `GraphDB.list_entities(...)`,
- computes exact in-memory cosine similarity,
- filters by threshold and optionally by ontology label,
- returns ranked `NodeHit` candidate matches.

#### `src/grawiki/similarity/similarity_finder.py`

Defines `EntitySimilarityFinder`, which:

- groups persisted entities by `semantic_key`,
- reports collision groups with more than one entity,
- ranks candidates inside exact semantic-key collision groups,
- runs broader matcher-based duplicate scans across all persisted entities,
- delegates candidate generation to an injected `EntitySimilarityMatcher`,
- defaults to the vector matcher when no matcher is injected,
- exposes a combined two-step duplicate report that first inspects exact semantic-key collisions and then runs a broader matcher-based scan while optionally skipping those exact collisions.

This module is the orchestration layer for duplicate-candidate inspection. It
does not merge nodes; it only finds potential candidates.

### Namespace: `grawiki.rag`

High-level RAG facade.

#### `src/grawiki/rag/graph_rag.py`

Key responsibilities:

- Defines `GraphRAG`, the main end-to-end ingestion and search entrypoint.
- Exposes stepwise public helpers for notebooks and debugging: `read_document(...)`, `chunk_document(...)`, `embed_document(...)`, `embed_chunks(...)`, `build_document_node(...)`, `build_chunk_nodes(...)`, `persist_document_and_chunks(...)`, `extract_kg_per_chunk(...)`, and `persist_entities_and_relationships(...)`.
- Reads and chunks documents.
- Embeds documents and chunks with the shared embedding client.
- Persists documents, chunks, entities, and relationships through `GraphDB`.
- Delegates retrieval search to configured `Retriever` implementations.
- Exposes entity similarity helpers through `find_similar_entities(...)` and `find_entity_collision_candidates(...)`.
- Exposes entity duplicate-finding helpers through `find_similar_entities(...)`, `find_entity_collision_candidates(...)`, and `find_entity_duplicate_candidates(...)`.
- Supports dependency injection through the `embedding=`, `kg_extractor=`, `retrievers=`, `similarity_finder=`, `resolve_entities_on_ingest=`, and `entity_resolution_threshold=` constructor arguments.

Important public methods:

- `GraphRAG.ingest(path)`: main file-ingestion entrypoint.
- `GraphRAG.ingest_text(text, title)`: ingest text already available in memory.
- `GraphRAG.search(query)`: retrieve flat `NodeHit` results.
- `GraphRAG.find_similar_entities(entity)`: inspect candidates for one entity using the configured similarity finder.
- `GraphRAG.find_entity_collision_candidates()`: inspect semantic-key collision groups and their candidate matches.
- `GraphRAG.find_entity_duplicate_candidates()`: run the two-step heuristic combining exact semantic-key collisions and broader matcher-based duplicate scanning.
- `GraphRAG.ingest(path)` and `GraphRAG.ingest_text(text, title)`: when `resolve_entities_on_ingest=True`, extracted entities are matched against persisted ones (via vector cosine similarity) immediately after extraction and before persistence; any extracted node whose similarity to a persisted node exceeds `entity_resolution_threshold` is replaced by the persisted node, and all relationship endpoints are rewritten to use the persisted node's id.

## Data Flow Overview

The main runtime path currently looks like this:

1. `grawiki.GraphRAG.ingest()` starts ingestion.
2. `GraphRAG.read_document()` loads a file into a `Document`.
3. `GraphRAG.chunk_document()` uses `grawiki.doc_processing.chunkers.Chunker` to split it into `Chunk` objects.
4. `GraphRAG.embed_document()` and `GraphRAG.embed_chunks()` compute embeddings with the shared embedding client.
5. `GraphRAG.build_document_node()` and `GraphRAG.build_chunk_nodes()` attach embeddings to persisted node models.
6. `GraphRAG.persist_document_and_chunks()` ensures indexes and persists documents, chunks, and `__has_chunk__` relationships.
7. `GraphRAG.extract_kg_per_chunk()` runs `KnowledgeGraphExtractor.extract(...)` concurrently across chunks.
8. When `resolve_entities_on_ingest=True`, `GraphRAG._resolve_extracted_entities()` matches freshly-extracted entities against persisted ones via the configured `EntitySimilarityFinder`; hits above `entity_resolution_threshold` cause the extracted node and its relationship endpoints to be rewritten to the persisted node's id so the persistence step reuses existing entities instead of creating duplicates. Skipped entirely when the flag is `False` (the default).
9. `GraphRAG.persist_entities_and_relationships()` ensures entity indexes and persists extracted entities, `__mentions__` links, and entity relationships.
10. `Retriever.fulltext(...)` or `Retriever.vector(...)` executes query-time retrieval; vector queries are embedded in the retrieval layer, not in the DB adapter.
11. `grawiki.GraphRAG.search()` returns flat `NodeHit` results.
12. `EntitySimilarityFinder.find_semantic_key_collisions()` detects exact semantic-key duplicates.
13. `EntitySimilarityFinder.find_collision_candidates()` ranks candidates inside those exact collision groups.
14. `EntitySimilarityFinder.find_similarity_candidates()` performs the broader matcher-based duplicate scan across persisted entities.
15. `EntitySimilarityFinder.find_duplicate_candidates()` combines both stages into one duplicate-inspection report.

## Notes For New Agents

### Main integration points

- For end-to-end ingestion and search work, start in `src/grawiki/rag/graph_rag.py`.
- For stepwise ingestion debugging or notebook workflows, use the public helper methods on `GraphRAG` before dropping into lower-level modules.
- For schema changes, start in `src/grawiki/graph/models.py`.
- For extraction behavior changes, inspect `src/grawiki/graph/extraction.py` and `src/grawiki/graph/prompts.py` together.
- For persistence changes, inspect `src/grawiki/db/base.py`, `src/grawiki/db/cypher.py`, and `src/grawiki/db/falkordb.py` together.
- For query strategy changes, inspect `src/grawiki/retrieval/text.py` and `src/grawiki/retrieval/keywords.py`.
- For entity deduplication and duplicate-candidate work, inspect `src/grawiki/similarity/`.
- For text loading or chunking changes, inspect `src/grawiki/doc_processing/`.

### Internal labels in the graph

The project uses reserved system labels for infrastructure nodes:

- `__document__`
- `__chunk__`
- `__memory__`
- `__entity__` is also used by the database layer as the common base label for extracted entities.

Avoid reusing these labels in extraction prompts or ad hoc schema changes unless you intend to modify the persistence model as well.
