"""Tests for the GraphRAG facade."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.grawiki.core.commons import Chunk
from src.grawiki.db.base import GraphDB, NodeHit
from src.grawiki.graph.models import KnowledgeGraph, Node, Relationship
from src.grawiki.rag.graph_rag import GraphRAG


class FakeEmbeddingResult:
    """Minimal embedding result object for tests."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeEmbedder:
    """Deterministic embedder used for GraphRAG tests."""

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
    """In-memory graph DB stub that records facade interactions."""

    def __init__(self) -> None:
        self.setup_calls: list[dict[str, int] | None] = []
        self.saved_doc_nodes = []
        self.saved_chunk_nodes = []
        self.saved_entities: list[tuple[list[Chunk], dict[str, KnowledgeGraph]]] = []
        self.fulltext_calls: list[tuple[list[str], str, int]] = []
        self.vector_calls: list[tuple[list[str], list[float], int]] = []

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

    async def ensure_indexes(
        self, *, labels: Iterable[str], vector_dims: Mapping[str, int] | None = None
    ) -> None:
        pass

    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        pass

    async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
        pass

    async def fulltext_search(
        self, *, labels: Sequence[str], query_text: str, limit: int = 10
    ) -> list[NodeHit]:
        self.fulltext_calls.append((list(labels), query_text, limit))
        return []

    async def vector_search(
        self, *, labels: Sequence[str], query_embedding: list[float], limit: int = 10
    ) -> list[NodeHit]:
        self.vector_calls.append((list(labels), list(query_embedding), limit))
        return []

    async def neighbors(
        self,
        *,
        node_ids: Sequence[str],
        rel_types: Sequence[str] | None = None,
        depth: int = 1,
    ) -> list[Node]:
        return []


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


def test_graph_rag_step_methods_and_ingest(tmp_path: Path) -> None:
    """GraphRAG should expose modular public steps and end-to-end ingestion."""

    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "Alan Turing inspired graph memory.\nGraphs connect chunks.\n"
    )

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=2,
        embedder=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
    )

    document = rag.read_document(input_path)
    chunks = rag.chunk_document(document)
    document_embedding = asyncio.run(rag.embed_document(document))
    chunk_embeddings = asyncio.run(rag.embed_chunks(chunks))
    document_node = rag.build_document_node(document, document_embedding)
    chunk_nodes = rag.build_chunk_nodes(chunks, chunk_embeddings)
    asyncio.run(rag.persist_documents_and_chunks(document_node, chunk_nodes))
    chunk_graphs = asyncio.run(rag.extract_chunk_graphs(chunks))
    asyncio.run(rag.persist_entities_and_relationships(chunks, chunk_graphs))

    assert document_node.embedding == document_embedding
    assert len(chunk_nodes) == len(chunks)
    assert graph_db.saved_doc_nodes[0].id == document.id
    assert len(graph_db.saved_chunk_nodes) == len(chunks)
    assert graph_db.saved_entities[0][1].keys() == chunk_graphs.keys()

    # End-to-end via ingest()
    graph_db = FakeGraphDB()
    extractor = ConcurrencyTrackingExtractor()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=2,
        embedder=FakeEmbedder(),
        kg_extractor=extractor,
    )
    asyncio.run(rag.ingest(input_path))

    assert len(graph_db.saved_doc_nodes) == 1
    assert len(graph_db.saved_chunk_nodes) > 0
    assert len(graph_db.saved_entities) == 1
    assert graph_db.setup_calls[0] is None
    assert {"__document__", "__chunk__"} == set(graph_db.setup_calls[1])
    assert graph_db.setup_calls[2] == {"__entity__": 3}
    assert extractor.maximum <= 2


def test_graph_rag_search_uses_retriever_primitives() -> None:
    """GraphRAG.search should delegate to db.fulltext_search / db.vector_search."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedder=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
    )

    asyncio.run(rag.search("Turing", method="fulltext", limit=3))
    asyncio.run(rag.search("machine intelligence", method="vector", limit=5))

    labels = ["__document__", "__chunk__", "__entity__"]
    assert graph_db.fulltext_calls[0] == (labels, "Turing", 3)
    assert graph_db.vector_calls[0] == (
        labels,
        [42.0, float(len("machine intelligence")), 7.0],
        5,
    )
