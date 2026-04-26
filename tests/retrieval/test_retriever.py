"""Unit tests for the Retriever strategy layer."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Sequence

import pytest

from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.graph.models import Node, Relationship
from grawiki.retrieval.text import TextRetriever, _deduplicate_hits


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbeddingResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeEmbedder:
    """Deterministic embedder that echoes text length as vector components."""

    async def embed_query(self, query: str | list[str]) -> FakeEmbeddingResult:
        texts = [query] if isinstance(query, str) else list(query)
        return FakeEmbeddingResult([[float(len(t)), 0.0, 1.0] for t in texts])

    async def embed_documents(self, documents: str | list[str]) -> FakeEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        return FakeEmbeddingResult([[float(len(t)), 0.0, 1.0] for t in texts])


def _make_node(node_id: str, label: str = "__chunk__") -> Node:
    return Node(id=node_id, label=label, semantic_key=node_id, name=node_id)


class FakeGraphDB(GraphDB):
    """In-memory stub that records which primitives were called."""

    def __init__(
        self,
        fulltext_hits: list[NodeHit] | None = None,
        vector_hits: list[NodeHit] | None = None,
    ) -> None:
        self.fulltext_calls: list[tuple[list[str], str, int]] = []
        self.vector_calls: list[tuple[list[str], list[float], int]] = []
        self._fulltext_hits = fulltext_hits or []
        self._vector_hits = vector_hits or []

    async def setup(self, embedding_dimensions=None) -> None:
        pass

    async def ensure_indexes(
        self, *, labels: Iterable[str], vector_dims: Mapping[str, int] | None = None
    ) -> None:
        pass

    async def fulltext_search(
        self, *, labels: Sequence[str], query_text: str, limit: int = 10
    ) -> list[NodeHit]:
        self.fulltext_calls.append((list(labels), query_text, limit))
        return list(self._fulltext_hits)

    async def vector_search(
        self, *, labels: Sequence[str], query_embedding: list[float], limit: int = 10
    ) -> list[NodeHit]:
        self.vector_calls.append((list(labels), list(query_embedding), limit))
        return list(self._vector_hits)

    async def neighbor_relationships(
        self,
        *,
        node_ids: Sequence[str],
        limit_per_node: int = 5,
    ) -> dict[str, list[NeighborRelationship]]:
        return {node_id: [] for node_id in node_ids}

    async def recall_subgraph(
        self,
        *,
        memory_ids: Sequence[str],
        hops: int = 1,
        limit_per_memory: int = 20,
    ) -> dict[str, list[NeighborRelationship]]:
        return {memory_id: [] for memory_id in memory_ids}

    async def list_entities(self, *, include_embeddings: bool = False) -> list[Node]:
        return []

    async def entity_relationship_counts(
        self, node_ids: Sequence[str]
    ) -> dict[str, int]:
        return {node_id: 0 for node_id in node_ids}

    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        pass

    async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
        pass

    async def merge_entity_nodes(
        self,
        *,
        master: Node,
        duplicate_ids: Sequence[str],
    ) -> None:
        pass

    async def delete_memory(self, memory_id: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests: fulltext
# ---------------------------------------------------------------------------


def test_fulltext_delegates_to_db_and_deduplicates() -> None:
    """fulltext() should call db.fulltext_search and deduplicate results."""

    node = _make_node("n1")
    db = FakeGraphDB(fulltext_hits=[NodeHit(node=node), NodeHit(node=node)])
    retriever = TextRetriever(db=db, embedding=FakeEmbedder())

    hits = asyncio.run(retriever.fulltext("hello", labels=["__chunk__"], limit=5))

    assert db.fulltext_calls == [(["__chunk__"], "hello", 5)]
    assert len(hits) == 1
    assert hits[0].node.id == "n1"


def test_fulltext_passes_labels_and_limit() -> None:
    """fulltext() should forward labels and limit unchanged."""

    db = FakeGraphDB()
    retriever = TextRetriever(db=db, embedding=FakeEmbedder())

    asyncio.run(retriever.fulltext("q", labels=["__document__", "__entity__"], limit=3))

    assert db.fulltext_calls[0][0] == ["__document__", "__entity__"]
    assert db.fulltext_calls[0][2] == 3


# ---------------------------------------------------------------------------
# Tests: vector
# ---------------------------------------------------------------------------


def test_vector_embeds_query_before_calling_db() -> None:
    """vector() should embed the query and pass the vector to db.vector_search."""

    db = FakeGraphDB()
    retriever = TextRetriever(db=db, embedding=FakeEmbedder())

    asyncio.run(retriever.vector("hi", labels=["__chunk__"], limit=7))

    assert len(db.vector_calls) == 1
    labels, embedding, limit = db.vector_calls[0]
    assert labels == ["__chunk__"]
    assert embedding == [float(len("hi")), 0.0, 1.0]
    assert limit == 7


def test_vector_deduplicates_results() -> None:
    """vector() should deduplicate hits returned by the DB."""

    node = _make_node("n2")
    db = FakeGraphDB(
        vector_hits=[NodeHit(node=node, score=0.9), NodeHit(node=node, score=0.8)]
    )
    retriever = TextRetriever(db=db, embedding=FakeEmbedder())

    hits = asyncio.run(retriever.vector("q", labels=["__chunk__"]))

    assert len(hits) == 1
    assert hits[0].node.id == "n2"


def test_retrieve_dispatches_to_vector_search() -> None:
    """retrieve() should call vector() when configured for vector search."""

    db = FakeGraphDB()
    retriever = TextRetriever(
        db=db,
        embedding=FakeEmbedder(),
        search_method="vector",
        search_labels=["__entity__"],
    )

    asyncio.run(retriever.retrieve("query", limit=4))

    assert db.vector_calls == [(["__entity__"], [5.0, 0.0, 1.0], 4)]


def test_retrieve_dispatches_to_fulltext_search() -> None:
    """retrieve() should call fulltext() when configured for full-text search."""

    db = FakeGraphDB()
    retriever = TextRetriever(
        db=db,
        embedding=FakeEmbedder(),
        search_method="fulltext",
        search_labels=["__document__"],
    )

    asyncio.run(retriever.retrieve("query", limit=2))

    assert db.fulltext_calls == [(["__document__"], "query", 2)]


def test_retrieve_rejects_unknown_search_method() -> None:
    """retrieve() should fail fast for unsupported search methods."""

    db = FakeGraphDB()
    retriever = TextRetriever(
        db=db,
        embedding=FakeEmbedder(),
        search_method="vector",
    )
    retriever.search_method = "invalid"

    with pytest.raises(ValueError, match="Unsupported search method"):
        asyncio.run(retriever.retrieve("query"))


# ---------------------------------------------------------------------------
# Tests: _deduplicate_hits (module-level utility)
# ---------------------------------------------------------------------------


def test_deduplicate_hits_preserves_first_occurrence() -> None:
    """_deduplicate_hits should keep the first NodeHit and drop later duplicates."""

    a = NodeHit(node=_make_node("x"), score=0.9)
    b = NodeHit(node=_make_node("x"), score=0.5)
    c = NodeHit(node=_make_node("y"), score=0.7)

    result = _deduplicate_hits([a, b, c])

    assert len(result) == 2
    assert result[0] is a
    assert result[1] is c
