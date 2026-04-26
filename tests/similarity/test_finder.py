"""Unit tests for entity similarity finder."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Sequence

import pytest

from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.graph.models import Node, Relationship
from grawiki.similarity.fuzzy import RapidFuzzEntitySimilarityMatcher
from grawiki.similarity.similarity_finder import EntitySimilarityFinder
from grawiki.similarity.vector import VectorEntitySimilarityMatcher


class FakeGraphDB(GraphDB):
    """GraphDB stub exposing only the primitives used by similarity tests."""

    def __init__(self, entities: list[Node]) -> None:
        self.entities = entities
        self.list_entities_calls: list[bool] = []

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
        return []

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
        self.list_entities_calls.append(include_embeddings)
        if include_embeddings:
            return [entity.model_copy(deep=True) for entity in self.entities]
        return [
            entity.model_copy(update={"embedding": []}, deep=True)
            for entity in self.entities
        ]

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


def _entity(
    node_id: str,
    semantic_key: str,
    name: str,
    *,
    label: str = "Person",
    embedding: list[float] | None = None,
) -> Node:
    return Node(
        id=node_id,
        label=label,
        semantic_key=semantic_key,
        name=name,
        embedding=embedding or [],
    )


def test_find_semantic_key_collisions_returns_only_duplicate_keys() -> None:
    """Collision inspection should return only semantic keys with duplicates."""

    db = FakeGraphDB(
        [
            _entity("a", "person_alan-turing", "Alan Turing"),
            _entity("b", "person_alan-turing", "A. Turing"),
            _entity("c", "concept_lambda", "Lambda Calculus", label="Concept"),
        ]
    )
    finder = EntitySimilarityFinder(db)

    collisions = asyncio.run(finder.find_semantic_key_collisions())

    assert list(collisions) == ["person_alan-turing"]
    assert [entity.id for entity in collisions["person_alan-turing"]] == ["a", "b"]
    assert all(not entity.embedding for entity in collisions["person_alan-turing"])


def test_rapidfuzz_search_excludes_self_and_keeps_cross_label_hits() -> None:
    """RapidFuzz search should consider duplicate names across label sets."""

    source = _entity("a", "person_alan-turing", "Alan Turing")
    db = FakeGraphDB(
        [
            source,
            _entity("b", "person_alan-turing-2", "Alan M. Turing"),
            _entity("c", "concept_alan-turing", "Alan Turing", label="Concept"),
            _entity("d", "person_grace-hopper", "Grace Hopper"),
        ]
    )
    finder = EntitySimilarityFinder(
        db,
        matcher=RapidFuzzEntitySimilarityMatcher(db=db),
    )

    hits = asyncio.run(finder.search(source, threshold=85.0, limit=5))

    assert [hit.node.id for hit in hits] == ["c", "b"]
    assert hits[0].matched_on == "rapidfuzz"
    assert hits[0].score >= 85.0


def test_vector_search_uses_cosine_similarity_and_threshold() -> None:
    """Vector search should rank entities by cosine similarity across labels."""

    source = _entity(
        "a",
        "person_alan-turing",
        "Alan Turing",
        embedding=[1.0, 0.0, 0.0],
    )
    db = FakeGraphDB(
        [
            source,
            _entity(
                "b",
                "person_alan-m-turing",
                "Alan M. Turing",
                embedding=[0.9, 0.1, 0.0],
            ),
            _entity(
                "c",
                "person_grace-hopper",
                "Grace Hopper",
                embedding=[0.6, 0.8, 0.0],
            ),
            _entity(
                "d",
                "concept_turing",
                "Turing Machine",
                label="Concept",
                embedding=[1.0, 0.0, 0.0],
            ),
        ]
    )
    finder = EntitySimilarityFinder(
        db,
        matcher=VectorEntitySimilarityMatcher(db=db),
    )

    hits = asyncio.run(finder.search(source, threshold=0.8, limit=5))

    assert [hit.node.id for hit in hits] == ["d", "b"]
    assert hits[0].matched_on == "vector"
    assert hits[0].score == pytest.approx(1.0)


def test_find_collision_candidates_searches_within_each_collision_group() -> None:
    """Collision candidate search should compare only members of the same group."""

    db = FakeGraphDB(
        [
            _entity(
                "a",
                "person_alan-turing",
                "Alan Turing",
                embedding=[1.0, 0.0, 0.0],
            ),
            _entity(
                "b",
                "person_alan-turing",
                "Alan M Turing",
                embedding=[0.95, 0.05, 0.0],
            ),
            _entity(
                "c",
                "person_grace-hopper",
                "Grace Hopper",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )
    finder = EntitySimilarityFinder(db)

    groups = asyncio.run(finder.find_collision_candidates(threshold=0.9, limit=2))

    assert len(groups) == 1
    assert groups[0].semantic_key == "person_alan-turing"
    assert [result.source.id for result in groups[0].results] == ["a", "b"]
    assert [hit.node.id for hit in groups[0].results[0].hits] == ["b"]
    assert [hit.node.id for hit in groups[0].results[1].hits] == ["a"]


def test_find_similarity_candidates_scans_beyond_semantic_key_collisions() -> None:
    """Broader similarity scan should skip exact collisions and find near-duplicates."""

    db = FakeGraphDB(
        [
            _entity(
                "a",
                "person_alan-turing",
                "Alan Turing",
                embedding=[1.0, 0.0, 0.0],
            ),
            _entity(
                "b",
                "person_alan-turing",
                "Alan M Turing",
                embedding=[0.95, 0.05, 0.0],
            ),
            _entity(
                "c",
                "person_grace-hopper",
                "Grace Hopper",
                embedding=[0.0, 1.0, 0.0],
            ),
            _entity(
                "d",
                "person_grace-m-hopper",
                "Grace M Hopper",
                embedding=[0.0, 0.95, 0.05],
            ),
        ]
    )
    finder = EntitySimilarityFinder(db)

    results = asyncio.run(
        finder.find_similarity_candidates(
            threshold=0.9,
            limit=2,
            skip_semantic_key_collisions=True,
        )
    )

    assert [result.source.id for result in results] == ["c"]
    assert [hit.node.id for hit in results[0].hits] == ["d"]


def test_find_duplicate_candidates_returns_exact_and_broad_results() -> None:
    """Combined duplicate report should include both heuristic stages."""

    db = FakeGraphDB(
        [
            _entity(
                "a",
                "person_alan-turing",
                "Alan Turing",
                embedding=[1.0, 0.0, 0.0],
            ),
            _entity(
                "b",
                "person_alan-turing",
                "Alan M Turing",
                embedding=[0.95, 0.05, 0.0],
            ),
            _entity(
                "c",
                "person_grace-hopper",
                "Grace Hopper",
                embedding=[0.0, 1.0, 0.0],
            ),
            _entity(
                "d",
                "person_grace-m-hopper",
                "Grace M Hopper",
                embedding=[0.0, 0.95, 0.05],
            ),
        ]
    )
    finder = EntitySimilarityFinder(db)

    report = asyncio.run(
        finder.find_duplicate_candidates(
            threshold=0.9,
            limit=2,
            skip_semantic_key_collisions_in_similarity_scan=True,
        )
    )

    assert list(report.semantic_key_collisions) == ["person_alan-turing"]
    assert [
        group.semantic_key for group in report.semantic_key_collision_candidates
    ] == ["person_alan-turing"]
    assert [result.source.id for result in report.similarity_candidates] == ["c"]
    assert [hit.node.id for hit in report.similarity_candidates[0].hits] == ["d"]
    assert db.list_entities_calls == [True, True]


def test_find_collision_candidates_reuses_precomputed_collisions() -> None:
    """Collision candidate search should reuse supplied collision groups."""

    db = FakeGraphDB(
        [
            _entity(
                "a",
                "person_alan-turing",
                "Alan Turing",
                embedding=[1.0, 0.0, 0.0],
            ),
            _entity(
                "b",
                "person_alan-turing",
                "Alan M Turing",
                embedding=[0.95, 0.05, 0.0],
            ),
        ]
    )
    finder = EntitySimilarityFinder(db)
    collisions = {
        "person_alan-turing": [entity.model_copy(deep=True) for entity in db.entities]
    }

    groups = asyncio.run(
        finder.find_collision_candidates(
            collisions=collisions,
            threshold=0.9,
            limit=2,
        )
    )

    assert [group.semantic_key for group in groups] == ["person_alan-turing"]
    assert db.list_entities_calls == []
