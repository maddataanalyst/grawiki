# CODEMAP

This file is a quick orientation guide for agents and contributors working in the GraWiki repository.

## Project Purpose

GraWiki is a Python project for:

1. Reading source documents.
2. Splitting them into chunks.
3. Extracting a structured knowledge graph from those chunks with an LLM.
4. Persisting documents, chunks, entities, and relationships in a graph database.
5. Searching the stored graph with full-text and vector retrieval.

In the future: part of the agentic memory. Memories will be added as a separate __entity__, indexed by time.
Agent will have an option to "recall" -- run dedicated graph search on memories and linked entities.

In short - this project tries to be a lightweight, simplified version of Llama Index graph RAG + graph memory.

## Top-Level Folders

### `src/grawiki/`

Main application package. This is where the reusable project code lives.

### `tests/`

Pytest suite covering pipeline behavior, graph models and extraction, Cypher query generation, and the FalkorDB adapter.

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

Top-level Python package for the project. `src/grawiki/__init__.py` is currently empty and mainly marks the package boundary.

### Namespace: `grawiki.core`

Core shared types and the high-level ingestion/search pipeline.

#### `src/grawiki/core/commons.py`

Defines the lightweight source-data models used before persistence:

- `Document`: raw input document with `id`, `title`, `content`, and metadata.
- `Chunk`: chunked segment of a document with `document_id`, `content`, and metadata.

These models are the bridge between file reading/chunking and graph persistence.

#### `src/grawiki/core/pipeline.py`

Main orchestration layer.

Key responsibilities:

- Defines small protocols for pluggable embedders and graph extractors.
- Exposes `GrawikiPipeline`, the main end-to-end application workflow.
- Reads a file from disk.
- Chunks the document.
- Computes embeddings for documents and chunks.
- Converts raw `Document`/`Chunk` objects into graph node models.
- Persists documents and chunks in the graph database.
- Runs chunk-level knowledge graph extraction.
- Persists extracted entities and relationships.
- Exposes a `search()` method that delegates to the graph database and adds query embeddings when vector search is used.

Important public method:

- `GrawikiPipeline.ingest_file(path)`: the main end-to-end ingestion entrypoint.

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

- `GraphDB`: abstract base class for persistence and search.
- `SearchMethod`: literal type for supported retrieval modes (`fulltext` or `vector`).
- `SearchResults`: common grouped search result shape.

The `GraphDB` interface covers:

- database/index setup,
- saving documents and chunks,
- saving extracted entities and relationships,
- running full-text or vector search.

Use this module when adding a new database backend or changing the shared persistence/search contract.

#### `src/grawiki/db/cypher_queries.py`

Shared Cypher query builders.

Key responsibilities:

- Builds upsert queries for document nodes.
- Builds upsert queries for chunk nodes and document-to-chunk links.
- Builds upsert queries for entity nodes and chunk-to-entity mention links.
- Builds upsert queries for entity-to-entity relationships.
- Sanitizes labels and relationship types into backend-safe Cypher identifiers.
- Injects optional embedding assignments into query strings.

This module keeps Cypher string construction separate from adapter control flow.

#### `src/grawiki/db/falkordb.py`

Concrete `GraphDB` implementation backed by FalkorDBLite.

Key responsibilities:

- Defines `FalkorGraphDB`.
- Opens/selects a FalkorDB graph stored on disk.
- Creates and tracks full-text and vector indexes.
- Persists document and chunk nodes.
- Persists extracted entities and relationships.
- Exposes grouped full-text and vector search across documents, chunks, and entities.
- Serializes embeddings and metadata into backend-compatible forms.
- Provides lower-level query helpers for debugging and experimentation.

This is currently the main persistence backend used by the repository.

#### `src/grawiki/db/__init__.py`

Small export module that re-exports:

- `GraphDB`
- `FalkorGraphDB`

## Data Flow Overview

The main runtime path currently looks like this:

1. `grawiki.core.pipeline.GrawikiPipeline.ingest_file()` starts ingestion.
2. `grawiki.doc_processing.document_processing.read_document()` loads a file into a `Document`.
3. `grawiki.doc_processing.chunkers.Chunker` splits the document into `Chunk` objects.
4. `grawiki.core.pipeline` embeds the document and chunks.
5. `grawiki.graph.models.DocumentNode` and `ChunkNode` are created for persistence.
6. `grawiki.db.falkordb.FalkorGraphDB` persists documents and chunks.
7. `grawiki.graph.extraction.KnowledgeGraphExtractor` extracts entity/relationship graphs from each chunk.
8. `grawiki.db.falkordb.FalkorGraphDB` persists extracted entities and relationships.
9. `grawiki.core.pipeline.GrawikiPipeline.search()` delegates full-text or vector retrieval to the DB layer.

## Notes For New Agents

### Main integration points

- For end-to-end ingestion work, start in `src/grawiki/core/pipeline.py`.
- For schema changes, start in `src/grawiki/graph/models.py`.
- For extraction behavior changes, inspect `src/grawiki/graph/extraction.py` and `src/grawiki/graph/prompts.py` together.
- For persistence or retrieval changes, inspect `src/grawiki/db/base.py`, `src/grawiki/db/cypher_queries.py`, and `src/grawiki/db/falkordb.py` together.
- For text loading or chunking changes, inspect `src/grawiki/doc_processing/`.

### Internal labels in the graph

The project uses reserved system labels for infrastructure nodes:

- `__document__`
- `__chunk__`
- `__memory__`
- `__entity__` is also used by the database layer as the common base label for extracted entities.

Avoid reusing these labels in extraction prompts or ad hoc schema changes unless you intend to modify the persistence model as well.
