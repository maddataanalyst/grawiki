"""Unit tests for the keyword-path retriever."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Sequence

from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.graph.models import ChunkNode, Node, Relationship
from grawiki.retrieval.keywords import KeywordsPathRetriever


class FakeEmbeddingResult:
    """Minimal embedding result object for tests."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeEmbedder:
    """Deterministic embedder that records keyword queries."""

    def __init__(self, empty_queries: set[str] | None = None) -> None:
        self.empty_queries = empty_queries or set()
        self.queries: list[str] = []

    async def embed_query(self, query: str | list[str]) -> FakeEmbeddingResult:
        if not isinstance(query, str):
            raise TypeError("FakeEmbedder expects a single string query.")
        self.queries.append(query)
        if query in self.empty_queries:
            return FakeEmbeddingResult([])
        return FakeEmbeddingResult([[float(len(query)), 1.0, 0.0]])

    async def embed_documents(self, documents: str | list[str]) -> FakeEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        return FakeEmbeddingResult([[float(len(text)), 1.0, 0.0] for text in texts])


class FakeKeywordExtractor:
    """Stub keyword extractor returning a fixed keyword list."""

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords

    async def extract(self, query: str) -> list[str]:
        return list(self.keywords)


class FakeGraphDB(GraphDB):
    """GraphDB stub supporting the primitives used by the retriever."""

    def __init__(
        self,
        *,
        vector_results: dict[float, list[NodeHit]] | None = None,
        relationship_results: dict[str, list[NeighborRelationship]] | None = None,
    ) -> None:
        self.vector_results = vector_results or {}
        self.relationship_results = relationship_results or {}
        self.vector_calls: list[tuple[list[str], list[float], int]] = []
        self.relationship_calls: list[tuple[list[str], int]] = []

    async def setup(self, embedding_dimensions=None) -> None:
        pass

    async def ensure_indexes(
        self, *, labels: Iterable[str], vector_dims: Mapping[str, int] | None = None
    ) -> None:
        pass

    async def fulltext_search(
        self, *, labels: Sequence[str], query_text: str, limit: int = 10
    ) -> list[NodeHit]:
        return []

    async def vector_search(
        self, *, labels: Sequence[str], query_embedding: list[float], limit: int = 10
    ) -> list[NodeHit]:
        self.vector_calls.append((list(labels), list(query_embedding), limit))
        return list(self.vector_results.get(query_embedding[0], []))

    async def neighbor_relationships(
        self,
        *,
        node_ids: Sequence[str],
        limit_per_node: int = 5,
    ) -> dict[str, list[NeighborRelationship]]:
        self.relationship_calls.append((list(node_ids), limit_per_node))
        return {
            node_id: list(self.relationship_results.get(node_id, []))
            for node_id in node_ids
        }

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


def _entity(node_id: str, name: str, *, label: str = "Person") -> Node:
    return Node(
        id=node_id,
        label=label,
        semantic_key=f"{label.casefold()}_{node_id}",
        name=name,
    )


def test_keywords_path_retriever_returns_empty_without_keywords() -> None:
    """Retriever should short-circuit when keyword extraction returns nothing."""

    db = FakeGraphDB()
    retriever = KeywordsPathRetriever(
        model="test-model",
        db=db,
        embedding=FakeEmbedder(),
        keyword_extractor=FakeKeywordExtractor([]),
    )

    hits = asyncio.run(retriever.retrieve("who studied computability?"))

    assert hits == []
    assert db.vector_calls == []
    assert db.relationship_calls == []


def test_keywords_path_retriever_builds_context_and_keeps_best_scores() -> None:
    """Retriever should deduplicate keywords and keep the best hit per entity."""

    alan = _entity("entity_turing", "Alan Turing")
    lambda_calc = _entity("entity_lambda", "Lambda Calculus", label="Concept")
    computability = _entity("entity_comp", "Computability", label="Concept")
    db = FakeGraphDB(
        vector_results={
            11.0: [
                NodeHit(node=alan, score=0.6, matched_on="vector"),
                NodeHit(node=lambda_calc, score=0.5, matched_on="vector"),
            ],
            13.0: [
                NodeHit(node=alan, score=0.9, matched_on="vector"),
                NodeHit(node=computability, score=0.7, matched_on="vector"),
            ],
        },
        relationship_results={
            "entity_turing": [
                NeighborRelationship(
                    source_id="entity_turing",
                    source_name="Alan Turing",
                    relationship_label="studied",
                    target=ChunkNode(
                        id="chunk_1",
                        semantic_key="chunk_chunk_1",
                        name="Chunk chunk_1",
                        document_id="doc_1",
                        content="Alan Turing studied computability.",
                    ),
                )
            ]
        },
    )
    embedder = FakeEmbedder()
    retriever = KeywordsPathRetriever(
        model="test-model",
        db=db,
        embedding=embedder,
        keyword_extractor=FakeKeywordExtractor(
            ["Alan Turing", " alan turing ", "", "Computability"]
        ),
        path_limit=3,
    )

    hits = asyncio.run(retriever.retrieve("who studied computability?", limit=2))

    assert embedder.queries == ["Alan Turing", "Computability"]
    assert db.relationship_calls == [(["entity_turing", "entity_comp"], 3)]
    assert [(hit.node.id, hit.score) for hit in hits] == [
        ("entity_turing", 0.9),
        ("entity_comp", 0.7),
    ]
    assert hits[0].matched_on == "keyword_path"
    assert (
        hits[0]
        .node.properties["content"]
        .startswith("Source Node: Alan Turing (id: entity_turing), similarity: 0.9000")
    )
    assert (
        "-[studied]-> NAME: Chunk chunk_1; LABELS: __chunk__"
        in hits[0].node.properties["content"]
    )
    assert "No connected graph context found." in hits[1].node.properties["content"]


def test_keywords_path_retriever_skips_empty_embedding_results() -> None:
    """Retriever should skip keywords whose embeddings could not be produced."""

    db = FakeGraphDB()
    retriever = KeywordsPathRetriever(
        model="test-model",
        db=db,
        embedding=FakeEmbedder(empty_queries={"Alan Turing"}),
        keyword_extractor=FakeKeywordExtractor(["Alan Turing"]),
    )

    hits = asyncio.run(retriever.retrieve("who studied computability?"))

    assert hits == []
    assert db.vector_calls == []
    assert db.relationship_calls == []
