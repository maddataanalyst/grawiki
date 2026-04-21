"""Tests for the main ingestion pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.grawiki.core.commons import Chunk
from src.grawiki.core.pipeline import GrawikiPipeline
from src.grawiki.db.base import GraphDB, SearchMethod, SearchResults
from src.grawiki.graph.models import KnowledgeGraph, Node


class FakeEmbeddingResult:
    """Minimal embedding result object for tests."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeEmbedder:
    """Deterministic embedder used for pipeline tests."""

    async def embed_documents(self, documents: str | list[str]) -> FakeEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        embeddings = [
            [float(len(text)), float(len(text.split())), 1.0] for text in texts
        ]
        return FakeEmbeddingResult(embeddings)

    async def embed_query(self, query: str | list[str]) -> FakeEmbeddingResult:
        texts = [query] if isinstance(query, str) else list(query)
        embeddings = [[42.0, float(len(text)), 7.0] for text in texts]
        return FakeEmbeddingResult(embeddings)


class FakeGraphDB(GraphDB):
    """In-memory graph DB stub that records pipeline interactions."""

    def __init__(self) -> None:
        self.setup_calls: list[dict[str, int] | None] = []
        self.saved_doc_nodes = []
        self.saved_chunk_nodes = []
        self.saved_entities: list[tuple[list[Chunk], dict[str, KnowledgeGraph]]] = []
        self.search_calls: list[tuple[str, SearchMethod, int, list[float] | None]] = []

    async def setup(self, embedding_dimensions: dict[str, int] | None = None) -> None:
        self.setup_calls.append(embedding_dimensions)

    async def save_docs_and_chunks_to_db(self, doc_nodes, chunk_nodes) -> None:
        self.saved_doc_nodes.extend(doc_nodes)
        self.saved_chunk_nodes.extend(chunk_nodes)

    async def save_entities_and_rels(
        self,
        chunks: list[Chunk],
        chunk_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        self.saved_entities.append((chunks, chunk_graphs))

    async def search(
        self,
        query: str,
        method: SearchMethod,
        *,
        limit: int = 10,
        query_embedding: list[float] | None = None,
    ) -> SearchResults:
        self.search_calls.append((query, method, limit, query_embedding))
        return {"__document__": [], "__chunk__": [], "__entity__": []}


class ConcurrencyTrackingExtractor:
    """Extractor stub that tracks maximum concurrent executions."""

    def __init__(self) -> None:
        self.current = 0
        self.maximum = 0

    async def extract(self, chunk: Chunk) -> KnowledgeGraph:
        self.current += 1
        self.maximum = max(self.maximum, self.current)
        await asyncio.sleep(0.01)
        self.current -= 1
        return KnowledgeGraph(
            nodes=[
                Node(
                    id=f"node_{chunk.id}",
                    label="Concept",
                    semantic_key=f"concept_{chunk.id}",
                    name=f"Concept {chunk.id}",
                    embedding=[1.0, 0.0, 0.0],
                )
            ]
        )


def test_pipeline_step_methods_and_ingest_file(tmp_path: Path) -> None:
    """The pipeline should expose modular public steps and end-to-end ingestion."""

    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "Alan Turing inspired graph memory.\nGraphs connect chunks.\n"
    )

    graph_db = FakeGraphDB()
    pipeline = GrawikiPipeline(
        model="test-model",
        embedding_model="test-embedding",
        graph_db=graph_db,
        max_workers=2,
        embedder=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
    )

    document = pipeline.read_document(input_path)
    chunks = pipeline.chunk_document(document)
    document_embedding = asyncio.run(pipeline.embed_document(document))
    chunk_embeddings = asyncio.run(pipeline.embed_chunks(chunks))
    document_node = pipeline.build_document_node(document, document_embedding)
    chunk_nodes = pipeline.build_chunk_nodes(chunks, chunk_embeddings)
    asyncio.run(pipeline.persist_documents_and_chunks(document_node, chunk_nodes))
    chunk_graphs = asyncio.run(pipeline.extract_chunk_graphs(chunks))
    asyncio.run(pipeline.persist_entities_and_relationships(chunks, chunk_graphs))

    assert document_node.embedding == document_embedding
    assert len(chunk_nodes) == len(chunks)
    assert graph_db.saved_doc_nodes[0].id == document.id
    assert len(graph_db.saved_chunk_nodes) == len(chunks)
    assert graph_db.saved_entities[0][1].keys() == chunk_graphs.keys()

    graph_db = FakeGraphDB()
    extractor = ConcurrencyTrackingExtractor()
    pipeline = GrawikiPipeline(
        model="test-model",
        embedding_model="test-embedding",
        graph_db=graph_db,
        max_workers=2,
        embedder=FakeEmbedder(),
        kg_extractor=extractor,
    )
    asyncio.run(pipeline.ingest_file(input_path))

    assert len(graph_db.saved_doc_nodes) == 1
    assert len(graph_db.saved_chunk_nodes) > 0
    assert len(graph_db.saved_entities) == 1
    assert graph_db.setup_calls[0] is None
    assert {"__document__", "__chunk__"} == set(graph_db.setup_calls[1])
    assert graph_db.setup_calls[2] == {"__entity__": 3}
    assert extractor.maximum <= 2


def test_pipeline_search_uses_db_and_query_embeddings() -> None:
    """The pipeline should delegate fulltext search and embed vector queries."""

    graph_db = FakeGraphDB()
    pipeline = GrawikiPipeline(
        model="test-model",
        embedding_model="test-embedding",
        graph_db=graph_db,
        embedder=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
    )

    asyncio.run(pipeline.search("Turing", method="fulltext", limit=3))
    asyncio.run(pipeline.search("machine intelligence", method="vector", limit=5))

    assert graph_db.search_calls[0] == ("Turing", "fulltext", 3, None)
    assert graph_db.search_calls[1] == (
        "machine intelligence",
        "vector",
        5,
        [42.0, float(len("machine intelligence")), 7.0],
    )
