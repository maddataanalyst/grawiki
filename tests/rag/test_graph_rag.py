"""Tests for the GraphRAG facade."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from grawiki.core.commons import Chunk
import pytest

from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.graph.models import KnowledgeGraph, Node, Relationship
from grawiki.rag.graph_rag import GraphRAG
from grawiki.retrieval.text import TextRetriever
from grawiki.similarity.fuzzy import RapidFuzzEntitySimilarityMatcher
from grawiki.similarity.similarity_finder import EntitySimilarityFinder


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
        self.entities: list[Node] = []

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

    async def neighbor_relationships(
        self,
        *,
        node_ids: Sequence[str],
        limit_per_node: int = 5,
    ) -> dict[str, list[NeighborRelationship]]:
        return {node_id: [] for node_id in node_ids}

    async def list_entities(self, *, include_embeddings: bool = False) -> list[Node]:
        if include_embeddings:
            return [node.model_copy(deep=True) for node in self.entities]
        return [
            node.model_copy(update={"embedding": []}, deep=True)
            for node in self.entities
        ]


class StaticRetriever:
    """Retriever stub returning a predefined hit list."""

    def __init__(self, hits: list[NodeHit]) -> None:
        self.hits = hits

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        return list(self.hits)


class FailingRetriever:
    """Retriever stub that always raises to exercise fail-open search."""

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        raise RuntimeError("retriever failed")


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
    search_labels = ["__chunk__", "__entity__"]

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=2,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=search_labels,
            ),
        ),
    )

    document = rag.read_document(input_path)
    chunks = rag.chunk_document(document)
    document_embedding = asyncio.run(rag.embed_document(document))
    chunk_embeddings = asyncio.run(rag.embed_chunks(chunks))
    document_node = rag.build_document_node(document, document_embedding)
    chunk_nodes = rag.build_chunk_nodes(chunks, chunk_embeddings)
    asyncio.run(rag.persist_document_and_chunks(document_node, chunk_nodes))
    chunk_graphs = asyncio.run(rag.extract_kg_per_chunk(chunks))
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
        embedding=FakeEmbedder(),
        kg_extractor=extractor,
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=search_labels,
            ),
        ),
    )
    asyncio.run(rag.ingest(input_path))

    assert len(graph_db.saved_doc_nodes) == 1
    assert len(graph_db.saved_chunk_nodes) > 0
    assert len(graph_db.saved_entities) == 1
    assert graph_db.setup_calls[0] is None
    assert {"__document__", "__chunk__"} == set(graph_db.setup_calls[1])
    assert graph_db.setup_calls[2] == {"__entity__": 3}
    assert extractor.maximum <= 2


def test_graph_rag_search_uses_configured_retrievers() -> None:
    """GraphRAG.search should delegate through the configured retrievers."""

    graph_db = FakeGraphDB()
    labels = ["__chunk__", "__entity__"]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="fulltext",
                search_labels=labels,
            ),
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=labels,
            ),
        ),
    )

    asyncio.run(rag.search("machine intelligence", limit=5))

    assert graph_db.fulltext_calls[0] == (labels, "machine intelligence", 5)
    assert graph_db.vector_calls[0] == (
        labels,
        [42.0, float(len("machine intelligence")), 7.0],
        5,
    )


def test_graph_rag_search_keeps_highest_score_and_distinct_labels() -> None:
    """GraphRAG should merge retriever results by label-plus-id and max score."""

    graph_db = FakeGraphDB()
    shared_chunk_low = NodeHit(
        node=Node(
            id="shared",
            label="__chunk__",
            semantic_key="chunk_shared",
            name="Chunk shared",
        ),
        score=0.4,
        matched_on="fulltext",
    )
    shared_chunk_high = NodeHit(
        node=Node(
            id="shared",
            label="__chunk__",
            semantic_key="chunk_shared",
            name="Chunk shared",
        ),
        score=0.9,
        matched_on="vector",
    )
    same_id_entity = NodeHit(
        node=Node(
            id="shared",
            label="Person",
            semantic_key="person_shared",
            name="Shared Person",
        ),
        score=0.7,
        matched_on="keyword_path",
    )
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            StaticRetriever([shared_chunk_low]),
            StaticRetriever([shared_chunk_high, same_id_entity]),
        ),
    )

    hits = asyncio.run(rag.search("machine intelligence", limit=5))

    assert [(hit.node.label, hit.node.id, hit.score) for hit in hits] == [
        ("__chunk__", "shared", 0.9),
        ("Person", "shared", 0.7),
    ]


def test_graph_rag_search_falls_back_to_working_retrievers() -> None:
    """GraphRAG should return partial results when one retriever fails."""

    graph_db = FakeGraphDB()
    surviving_hit = NodeHit(
        node=Node(
            id="entity_1",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan Turing",
        ),
        score=0.8,
        matched_on="keyword_path",
    )
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(FailingRetriever(), StaticRetriever([surviving_hit])),
    )

    hits = asyncio.run(rag.search("machine intelligence", limit=5))

    assert hits == [surviving_hit]


def test_graph_rag_search_raises_when_all_retrievers_fail() -> None:
    """GraphRAG should fail when every configured retriever errors."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(FailingRetriever(),),
    )

    with pytest.raises(RuntimeError, match="All retrievers failed"):
        asyncio.run(rag.search("machine intelligence", limit=5))


def test_graph_rag_similarity_helpers_use_injected_similarity_finder() -> None:
    """GraphRAG should expose entity similarity helpers via the injected finder."""

    graph_db = FakeGraphDB()
    graph_db.entities = [
        Node(
            id="entity_1",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan Turing",
            embedding=[1.0, 0.0, 0.0],
        ),
        Node(
            id="entity_2",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan M. Turing",
            embedding=[0.9, 0.1, 0.0],
        ),
    ]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
        similarity_finder=EntitySimilarityFinder(
            graph_db,
            matcher=RapidFuzzEntitySimilarityMatcher(db=graph_db),
        ),
    )

    hits = asyncio.run(
        rag.find_similar_entities(
            graph_db.entities[0],
            threshold=85.0,
        )
    )
    collisions = asyncio.run(rag.find_entity_collision_candidates(threshold=85.0))

    assert [hit.node.id for hit in hits] == ["entity_2"]
    assert [group.semantic_key for group in collisions] == ["person_alan-turing"]


class StaticExtractor:
    """Extractor stub that always returns a pre-built KnowledgeGraph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    async def extract(self, chunk: Chunk) -> KnowledgeGraph:
        return self._graph


class _CallCountingFinder:
    """Test-only proxy that records every search() call made against an EntitySimilarityFinder."""

    def __init__(self, inner: EntitySimilarityFinder) -> None:
        self._inner = inner
        self.search_calls: list[Node] = []

    async def search(
        self,
        entity: Node,
        *,
        limit: int = 10,
        threshold: float | None = None,
        same_label_only: bool = True,
        candidates: list[Node] | None = None,
    ) -> list[NodeHit]:
        self.search_calls.append(entity)
        return await self._inner.search(
            entity,
            limit=limit,
            threshold=threshold,
            same_label_only=same_label_only,
            candidates=candidates,
        )

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def test_graph_rag_resolves_entities_on_ingest_when_enabled() -> None:
    """When resolve_entities_on_ingest=True, extracted nodes matching a persisted
    node above the threshold should be swapped for the persisted node before
    persistence, and relationship endpoints should be rewritten accordingly.
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    extracted_node = Node(
        id="extracted-1",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )
    # Cosine([0.1,0.2,0.31], [0.1,0.2,0.3]) ≈ 0.99998, well above 0.9 threshold.

    extracted_kg = KnowledgeGraph(
        nodes=[extracted_node],
        relationships=[
            Relationship(
                id="rel-1",
                source="extracted-1",
                target="extracted-1",
                label="RELATED_TO",
                properties={},
            )
        ],
    )

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(extracted_kg),
        retrievers=(StaticRetriever([]),),
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    asyncio.run(rag.ingest_text("ReAct agents are a kind of ReAct.", title="t"))

    # Inspect the chunk_graphs that were passed to persist_entities_and_relationships.
    persisted_chunk_graphs = fake_db.saved_entities[-1][1]
    all_node_ids = {
        node.id
        for graph in persisted_chunk_graphs.values()
        for node in graph.nodes
    }
    all_rel_sources = {
        rel.source
        for graph in persisted_chunk_graphs.values()
        for rel in graph.relationships
    }
    all_rel_targets = {
        rel.target
        for graph in persisted_chunk_graphs.values()
        for rel in graph.relationships
    }

    assert "extracted-1" not in all_node_ids, (
        "extracted-1 should have been replaced by persisted-1"
    )
    assert "persisted-1" in all_node_ids, (
        "persisted-1 should appear in the persisted chunk graphs"
    )
    assert all_rel_sources == {"persisted-1"}, (
        "relationship source should have been rewritten to exactly persisted-1"
    )
    assert all_rel_targets == {"persisted-1"}, (
        "relationship target should have been rewritten to exactly persisted-1"
    )


def test_graph_rag_rejects_out_of_range_entity_resolution_threshold() -> None:
    """GraphRAG should reject thresholds outside the cosine [-1, 1] range."""

    graph_db = FakeGraphDB()
    with pytest.raises(ValueError, match="entity_resolution_threshold"):
        GraphRAG(
            model="test-model",
            embedding_model="test-embedding",
            db=graph_db,
            embedding=FakeEmbedder(),
            kg_extractor=ConcurrencyTrackingExtractor(),
            retrievers=(StaticRetriever([]),),
            entity_resolution_threshold=1.5,
        )


def test_graph_rag_does_not_resolve_entities_when_disabled() -> None:
    """When resolve_entities_on_ingest=False (the default), extracted nodes are
    persisted as-is regardless of similarity to any persisted node.
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    extracted_node = Node(
        id="extracted-1",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )

    extracted_kg = KnowledgeGraph(nodes=[extracted_node], relationships=[])

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(extracted_kg),
        retrievers=(StaticRetriever([]),),
        # resolve_entities_on_ingest defaults to False
    )

    asyncio.run(rag.ingest_text("ReAct agents are a kind of ReAct.", title="t"))

    persisted_chunk_graphs = fake_db.saved_entities[-1][1]
    all_node_ids = {
        node.id
        for graph in persisted_chunk_graphs.values()
        for node in graph.nodes
    }

    assert "extracted-1" in all_node_ids, (
        "extracted-1 should be persisted as-is when resolution is disabled"
    )


def test_graph_rag_exposes_combined_duplicate_candidate_report() -> None:
    """GraphRAG should expose the two-step duplicate-finding heuristic."""

    graph_db = FakeGraphDB()
    graph_db.entities = [
        Node(
            id="entity_1",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan Turing",
            embedding=[1.0, 0.0, 0.0],
        ),
        Node(
            id="entity_2",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan M. Turing",
            embedding=[0.95, 0.05, 0.0],
        ),
        Node(
            id="entity_3",
            label="Person",
            semantic_key="person_grace-hopper",
            name="Grace Hopper",
            embedding=[0.0, 1.0, 0.0],
        ),
        Node(
            id="entity_4",
            label="Person",
            semantic_key="person_grace-m-hopper",
            name="Grace M. Hopper",
            embedding=[0.0, 0.95, 0.05],
        ),
    ]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
    )

    report = asyncio.run(
        rag.find_entity_duplicate_candidates(
            threshold=0.9,
            skip_semantic_key_collisions_in_similarity_scan=True,
        )
    )

    assert list(report.semantic_key_collisions) == ["person_alan-turing"]
    assert [
        group.semantic_key for group in report.semantic_key_collision_candidates
    ] == ["person_alan-turing"]
    assert [result.source.id for result in report.similarity_candidates] == ["entity_3"]
    assert [hit.node.id for hit in report.similarity_candidates[0].hits] == ["entity_4"]


def test_resolve_extracted_entities_searches_each_unique_id_once() -> None:
    """When the same extracted entity appears in two chunks, the similarity
    search should be issued exactly once (de-duplication by entity id).
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    # Extracted node is the same object in both chunks — same id, nearly
    # identical embedding so cosine similarity ≈ 0.99998 > 0.9 threshold.
    extracted = Node(
        id="extracted-1",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    inner_finder = EntitySimilarityFinder(db=fake_db)
    counting_finder = _CallCountingFinder(inner_finder)

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph(nodes=[extracted], relationships=[])),
        retrievers=(StaticRetriever([]),),
        similarity_finder=counting_finder,
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    # Directly call the private method with two chunks that share the same
    # extracted entity.  This sidesteps the need to produce exactly two chunks
    # from ingest_text and lets the test focus on the de-duplication logic.
    chunk_graphs = {
        "chunk-a": KnowledgeGraph(nodes=[extracted], relationships=[]),
        "chunk-b": KnowledgeGraph(nodes=[extracted], relationships=[]),
    }
    rewritten = asyncio.run(rag._resolve_extracted_entities(chunk_graphs))

    # One search call, not two — the implementation deduplicates by entity id.
    assert len(counting_finder.search_calls) == 1, (
        f"Expected exactly 1 similarity search, got {len(counting_finder.search_calls)}"
    )

    # Both chunks should have the extracted id replaced by the persisted id.
    for chunk_id, graph in rewritten.items():
        node_ids = {n.id for n in graph.nodes}
        assert "extracted-1" not in node_ids, (
            f"chunk '{chunk_id}': extracted-1 should have been replaced"
        )
        assert "persisted-1" in node_ids, (
            f"chunk '{chunk_id}': persisted-1 should appear after rewrite"
        )


def test_resolve_extracted_entities_rewrites_relationship_endpoints() -> None:
    """Relationship source/target pointing at a resolved entity must be rewritten
    to the persisted id.  Endpoints pointing at an unresolved entity stay as-is.
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    # This extracted node is close enough to be resolved (cosine ≈ 0.99998).
    resolved_extracted = Node(
        id="extracted-resolved",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )
    # This extracted node has an orthogonal embedding — it will not match any
    # persisted node so it should be left untouched.
    unresolved_extracted = Node(
        id="extracted-unresolved",
        label="Concept",
        semantic_key="concept_other",
        name="Something else entirely",
        embedding=[0.0, 0.0, 1.0],
    )

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(
            KnowledgeGraph(nodes=[resolved_extracted], relationships=[])
        ),
        retrievers=(StaticRetriever([]),),
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    # chunk-a: has a relationship whose source is the resolved entity (self-loop)
    # and a mixed relationship whose source is resolved but target is unresolved.
    # chunk-b: contains only the unresolved entity, no relationships.
    chunk_graphs = {
        "chunk-a": KnowledgeGraph(
            nodes=[resolved_extracted],
            relationships=[
                Relationship(
                    id="rel-1",
                    source="extracted-resolved",
                    target="extracted-resolved",
                    label="RELATED_TO",
                    properties={},
                ),
                Relationship(
                    id="rel-3",
                    source="extracted-resolved",
                    target="extracted-unresolved",
                    label="INFLUENCES",
                    properties={},
                ),
            ],
        ),
        "chunk-b": KnowledgeGraph(
            nodes=[unresolved_extracted],
            relationships=[
                Relationship(
                    id="rel-2",
                    source="extracted-unresolved",
                    target="extracted-unresolved",
                    label="RELATED_TO",
                    properties={},
                )
            ],
        ),
    }
    rewritten = asyncio.run(rag._resolve_extracted_entities(chunk_graphs))

    # chunk-a: resolved entity's id must appear in nodes and in both endpoints of rel-1.
    chunk_a = rewritten["chunk-a"]
    chunk_a_node_ids = {n.id for n in chunk_a.nodes}
    assert "persisted-1" in chunk_a_node_ids, (
        "chunk-a: resolved node should carry persisted-1"
    )
    assert "extracted-resolved" not in chunk_a_node_ids, (
        "chunk-a: extracted-resolved should be gone after rewrite"
    )
    rel_1 = next(r for r in chunk_a.relationships if r.id == "rel-1")
    assert rel_1.source == "persisted-1", (
        "chunk-a rel-1 source should be rewritten to persisted-1"
    )
    assert rel_1.target == "persisted-1", (
        "chunk-a rel-1 target should be rewritten to persisted-1"
    )

    # rel-3 (mixed): source is resolved → rewritten; target is unresolved → stays.
    rel_3 = next(r for r in chunk_a.relationships if r.id == "rel-3")
    assert rel_3.source == "persisted-1", (
        "chunk-a rel-3 source should be rewritten to persisted-1"
    )
    assert rel_3.target == "extracted-unresolved", (
        "chunk-a rel-3 target should remain extracted-unresolved"
    )

    # chunk-b: unresolved entity and its relationship endpoints must be untouched.
    chunk_b = rewritten["chunk-b"]
    chunk_b_node_ids = {n.id for n in chunk_b.nodes}
    assert "extracted-unresolved" in chunk_b_node_ids, (
        "chunk-b: unresolved node should be preserved as-is"
    )
    assert chunk_b.relationships[0].source == "extracted-unresolved", (
        "chunk-b relationship source should remain extracted-unresolved"
    )
    assert chunk_b.relationships[0].target == "extracted-unresolved", (
        "chunk-b relationship target should remain extracted-unresolved"
    )
