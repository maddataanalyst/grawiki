"""Tests for graph extraction helpers."""

import uuid

from src.grawiki.graph.graph_extraction import KnowledgeGraphExtractor
from src.grawiki.graph.models import (
    ExtractedKnowledgeGraph,
    ExtractedNode,
    ExtractedRelationship,
)


def test_fix_missing_nodes_adds_unknown_placeholders() -> None:
    """Missing relationship endpoints should create placeholder extracted nodes."""

    extractor = KnowledgeGraphExtractor.__new__(KnowledgeGraphExtractor)
    graph = ExtractedKnowledgeGraph(
        relationships=[
            ExtractedRelationship(
                source="Alan Turing",
                target="Computability",
                label="studied",
            )
        ]
    )

    fixed_graph = extractor._fix_missing_nodes(graph)

    assert [node.name for node in fixed_graph.nodes] == ["Alan Turing", "Computability"]
    assert [node.label for node in fixed_graph.nodes] == [
        "__unknown__",
        "__unknown__",
    ]


def test_build_knowledge_graph_assigns_uuids_and_rewrites_relationship_endpoints() -> (
    None
):
    """Name-based extracted graphs should become UUID-based internal graphs."""

    extractor = KnowledgeGraphExtractor.__new__(KnowledgeGraphExtractor)
    graph = ExtractedKnowledgeGraph(
        nodes=[
            ExtractedNode(label="Person", name="Alan Turing"),
            ExtractedNode(label="Concept", name="Computability"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Alan Turing",
                target="Computability",
                label="studied",
            )
        ],
    )

    updated_graph = extractor._build_knowledge_graph(graph)

    assert [node.name for node in updated_graph.nodes] == [
        "Alan Turing",
        "Computability",
    ]
    for node in updated_graph.nodes:
        assert str(uuid.UUID(node.id)) == node.id
    assert len(updated_graph.relationships) == 1
    assert (
        str(uuid.UUID(updated_graph.relationships[0].id))
        == updated_graph.relationships[0].id
    )
    for rel in updated_graph.relationships:
        assert rel.source in {node.id for node in updated_graph.nodes}
        assert rel.target in {node.id for node in updated_graph.nodes}


def test_build_knowledge_graph_handles_placeholder_nodes() -> None:
    """Placeholder extracted nodes should convert to UUID-based internal nodes."""

    extractor = KnowledgeGraphExtractor.__new__(KnowledgeGraphExtractor)
    graph = ExtractedKnowledgeGraph(
        relationships=[
            ExtractedRelationship(
                source="Alan Turing",
                target="Computability",
                label="studied",
            )
        ]
    )

    fixed_graph = extractor._fix_missing_nodes(graph)
    updated_graph = extractor._build_knowledge_graph(fixed_graph)

    assert len(updated_graph.nodes) == 2
    for node in updated_graph.nodes:
        assert str(uuid.UUID(node.id)) == node.id
    assert (
        str(uuid.UUID(updated_graph.relationships[0].id))
        == updated_graph.relationships[0].id
    )
    assert updated_graph.relationships[0].source in {
        node.id for node in updated_graph.nodes
    }
    assert updated_graph.relationships[0].target in {
        node.id for node in updated_graph.nodes
    }


def test_build_knowledge_graph_deduplicates_duplicate_names() -> None:
    """Duplicate extracted names should map to one internal node."""

    extractor = KnowledgeGraphExtractor.__new__(KnowledgeGraphExtractor)
    graph = ExtractedKnowledgeGraph(
        nodes=[
            ExtractedNode(label="Person", name="Alan Turing"),
            ExtractedNode(
                label="Person", name="Alan Turing", properties={"field": "AI"}
            ),
        ]
    )

    updated_graph = extractor._build_knowledge_graph(graph)

    assert len(updated_graph.nodes) == 1
    assert updated_graph.nodes[0].name == "Alan Turing"
