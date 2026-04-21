"""Step-by-step extraction and FalkorDB persistence debug notebook."""

# ruff: noqa: F704

from __future__ import annotations

# %% Imports
import autoroot  # noqa: F401
from importlib import reload
from pathlib import Path

from dotenv import load_dotenv
from rich import print as pprint

import src.grawiki.core.pipeline as pipeline_module
import src.grawiki.doc_processing.chunkers as chunkers
import src.grawiki.doc_processing.document_processing as doc_proc
from src.grawiki.db.falkordb import FalkorGraphDB

pipeline_module = reload(pipeline_module)
doc_proc = reload(doc_proc)
chunkers = reload(chunkers)


# %% Environment
load_dotenv(override=True)

DOC_PATH = Path("experimental_data/agent_architectures.txt")
DB_PATH = Path("tmp/extraction_debug.db")
GRAPH_NAME = "extraction_debug"
MODEL_NAME = "openai:gpt-5-mini"
EMBEDDING_MODEL = "openai:text-embedding-3-small"
CHUNKING_STRATEGY = "sentence"
MAX_WORKERS = 4
MAX_CHUNKS = 2

pprint(
    {
        "doc_path": str(DOC_PATH),
        "db_path": str(DB_PATH),
        "graph_name": GRAPH_NAME,
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunking_strategy": CHUNKING_STRATEGY,
        "max_workers": MAX_WORKERS,
        "max_chunks": MAX_CHUNKS,
    }
)


# %% Init adapter and pipeline
adapter = FalkorGraphDB(DB_PATH, GRAPH_NAME)
pipeline = pipeline_module.GrawikiPipeline(
    model=MODEL_NAME,
    embedding_model=EMBEDDING_MODEL,
    graph_db=adapter,
    chunking_strategy=CHUNKING_STRATEGY,
    max_workers=MAX_WORKERS,
)

pprint({"adapter_db_path": adapter.db_path, "graph_name": adapter.graph_name})


# %% Optional cleanup for reruns
adapter._graph.delete()
adapter = FalkorGraphDB(DB_PATH, GRAPH_NAME)
pipeline = pipeline_module.GrawikiPipeline(
    model=MODEL_NAME,
    embedding_model=EMBEDDING_MODEL,
    graph_db=adapter,
    chunking_strategy=CHUNKING_STRATEGY,
    max_workers=MAX_WORKERS,
)


# %% Setup indexes
await pipeline.setup_db()
pprint(adapter.query("CALL db.indexes()").result_set)


# %% Read document
document = pipeline.read_document(DOC_PATH)
pprint(
    {
        "document_id": document.id,
        "title": document.title,
        "preview": document.content[:300],
    }
)


# %% Chunk document and keep a small sample
all_chunks = pipeline.chunk_document(document)
selected_chunks = all_chunks[:MAX_CHUNKS]

pprint(
    [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "preview": chunk.content[:220],
        }
        for chunk in selected_chunks
    ]
)


# %% Embed document and selected chunks
document_embedding = await pipeline.embed_document(document)
selected_chunk_embeddings = await pipeline.embed_chunks(selected_chunks)

pprint(
    {
        "document_embedding_dimension": len(document_embedding),
        "selected_chunk_count": len(selected_chunk_embeddings),
        "chunk_embedding_dimension": (
            len(selected_chunk_embeddings[0]) if selected_chunk_embeddings else 0
        ),
    }
)


# %% Build nodes
document_node = pipeline.build_document_node(document, document_embedding)
selected_chunk_nodes = pipeline.build_chunk_nodes(
    selected_chunks,
    selected_chunk_embeddings,
)

pprint(
    {
        "document_node": document_node.model_dump(),
        "first_chunk_node": (
            selected_chunk_nodes[0].model_dump() if selected_chunk_nodes else None
        ),
    }
)


# %% Persist document and selected chunks
await pipeline.persist_documents_and_chunks(document_node, selected_chunk_nodes)

pprint(
    {
        "documents": adapter.ro_query(
            "MATCH (d:__document__) RETURN d.id, d.name ORDER BY d.name"
        ).result_set,
        "chunks": adapter.ro_query(
            "MATCH (d:__document__)-[:__has_chunk__]->(c:__chunk__) "
            "RETURN d.name, c.id, c.name ORDER BY c.id"
        ).result_set,
    }
)


# %% Extract chunk graphs
chunk_graphs = await pipeline.extract_chunk_graphs(selected_chunks)

for chunk_id, graph in chunk_graphs.items():
    pprint(
        {
            "chunk_id": chunk_id,
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "semantic_key": node.semantic_key,
                    "name": node.name,
                    "properties": node.properties,
                    "embedding_dimension": len(node.embedding),
                }
                for node in graph.nodes
            ],
            "relationships": [
                {
                    "id": rel.id,
                    "label": rel.label,
                    "source": rel.source,
                    "target": rel.target,
                    "properties": rel.properties,
                }
                for rel in graph.relationships
            ],
        }
    )


# %% Persist extracted entities and relationships
await pipeline.persist_entities_and_relationships(selected_chunks, chunk_graphs)

pprint(
    {
        "entities": adapter.ro_query(
            "MATCH (e:__entity__) "
            "RETURN e.id, e.name, e.label, e.semantic_key, e.properties ORDER BY e.name"
        ).result_set,
        "mentions": adapter.ro_query(
            "MATCH (c:__chunk__)-[:__mentions__]->(e:__entity__) "
            "RETURN c.id, e.name, e.label, e.semantic_key ORDER BY c.id, e.name"
        ).result_set,
        "relationships": adapter.ro_query(
            "MATCH (source:__entity__)-[r]->(target:__entity__) "
            "RETURN source.name, type(r), target.name, r.properties "
            "ORDER BY source.name, target.name"
        ).result_set,
    }
)


# %% Inspect indexes
pprint(adapter.query("CALL db.indexes()").result_set)


# %% Full-text search example
pprint(await pipeline.search("Turing", method="fulltext", limit=5))


# %% Vector search example
pprint(await pipeline.search("machine intelligence", method="vector", limit=5))


# %% Explain vector query
entity_query_embedding = (
    await pipeline.embedding.embed_query("machine intelligence")
).embeddings[0]
pprint(adapter.explain_vector_query("__entity__", entity_query_embedding, 5))


# %% One-shot ingestion alternative
# Uncomment to run the full flow in one call instead of step by step.
# await pipeline.ingest_file(DOC_PATH)
