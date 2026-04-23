"""Unit tests for the Retriever strategy layer."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Sequence

from src.grawiki.db.base import GraphDB, NodeHit
from src.grawiki.graph.models import KnowledgeGraph, Node, Relationship
from src.grawiki.retrieval.retriever import Retriever, _deduplicate_hits


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
        neighbor_nodes: list[Node] | None = None,
    ) -> None:
        self.fulltext_calls: list[tuple[list[str], str, int]] = []
        self.vector_calls: list[tuple[list[str], list[float], int]] = []
        self.neighbor_calls: list[tuple[list[str], list[str] | None, int]] = []
        self._fulltext_hits = fulltext_hits or []
        self._vector_hits = vector_hits or []
        self._neighbor_nodes = neighbor_nodes or []

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

    async def neighbors(
        self,
        *,
        node_ids: Sequence[str],
        rel_types: Sequence[str] | None = None,
        depth: int = 1,
    ) -> list[Node]:
        self.neighbor_calls.append(
            (list(node_ids), list(rel_types) if rel_types else None, depth)
        )
        return list(self._neighbor_nodes)

    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        pass

    async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests: fulltext
# ---------------------------------------------------------------------------


def test_fulltext_delegates_to_db_and_deduplicates() -> None:
    """fulltext() should call db.fulltext_search and deduplicate results."""

    node = _make_node("n1")
    db = FakeGraphDB(fulltext_hits=[NodeHit(node=node), NodeHit(node=node)])
    retriever = Retriever(db=db, embedding=FakeEmbedder())

    hits = asyncio.run(retriever.fulltext("hello", labels=["__chunk__"], limit=5))

    assert db.fulltext_calls == [(["__chunk__"], "hello", 5)]
    assert len(hits) == 1
    assert hits[0].node.id == "n1"


def test_fulltext_passes_labels_and_limit() -> None:
    """fulltext() should forward labels and limit unchanged."""

    db = FakeGraphDB()
    retriever = Retriever(db=db, embedding=FakeEmbedder())

    asyncio.run(retriever.fulltext("q", labels=["__document__", "__entity__"], limit=3))

    assert db.fulltext_calls[0][0] == ["__document__", "__entity__"]
    assert db.fulltext_calls[0][2] == 3


# ---------------------------------------------------------------------------
# Tests: vector
# ---------------------------------------------------------------------------


def test_vector_embeds_query_before_calling_db() -> None:
    """vector() should embed the query and pass the vector to db.vector_search."""

    db = FakeGraphDB()
    retriever = Retriever(db=db, embedding=FakeEmbedder())

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
    retriever = Retriever(db=db, embedding=FakeEmbedder())

    hits = asyncio.run(retriever.vector("q", labels=["__chunk__"]))

    assert len(hits) == 1
    assert hits[0].node.id == "n2"


# ---------------------------------------------------------------------------
# Tests: expand
# ---------------------------------------------------------------------------


def test_expand_calls_neighbors_with_seed_ids() -> None:
    """expand() should call db.neighbors with the ids of the seed hits."""

    neighbor = _make_node("neighbor1")
    db = FakeGraphDB(neighbor_nodes=[neighbor])
    seed_hits = [
        NodeHit(node=_make_node("s1")),
        NodeHit(node=_make_node("s2")),
    ]
    retriever = Retriever(db=db, embedding=FakeEmbedder())

    nodes = asyncio.run(
        retriever.expand(seed_hits, rel_types=["__mentions__"], depth=2)
    )

    assert len(db.neighbor_calls) == 1
    ids, rel_types, depth = db.neighbor_calls[0]
    assert set(ids) == {"s1", "s2"}
    assert rel_types == ["__mentions__"]
    assert depth == 2
    assert nodes == [neighbor]


def test_expand_returns_empty_for_no_seeds() -> None:
    """expand() should short-circuit and return [] when seeds is empty."""

    db = FakeGraphDB()
    retriever = Retriever(db=db, embedding=FakeEmbedder())

    nodes = asyncio.run(retriever.expand([]))

    assert nodes == []
    assert db.neighbor_calls == []


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
