"""Tests for deduplication helpers."""

from __future__ import annotations

from grawiki.graph.models import Node
from grawiki.similarity.deduplication import (
    MergeReport,
    build_merged_master,
    pick_master,
)


def _node(
    node_id: str,
    name: str,
    *,
    labels: frozenset[str] | None = None,
    properties: dict[str, str] | None = None,
) -> Node:
    return Node(
        id=node_id,
        labels=labels or frozenset({"Person"}),
        semantic_key=f"entity_{node_id}",
        name=name,
        properties=properties or {},
    )


def test_pick_master_prefers_properties_then_relationships_then_name_then_id() -> None:
    """Master selection should follow the configured deterministic priority order."""

    sparse = _node("b", "Longer Name")
    rich = _node("c", "Short", properties={"role": "scientist", "era": "modern"})
    tied = _node("a", "Short", properties={"role": "scientist", "era": "modern"})

    # rich and tied have the same property count; relationship count breaks that tie.
    relation_counts = {"a": 5, "b": 20, "c": 1}

    assert pick_master([sparse, rich, tied], relation_counts).id == "a"


def test_build_merged_master_unions_labels_and_keeps_master_semantic_key() -> None:
    """Merged masters should preserve the master's identity and union label sets."""

    master = _node(
        "m",
        "ReAct agent",
        labels=frozenset({"Concept"}),
        properties={"kind": "concept", "source": "paper"},
    )
    duplicate = _node(
        "d",
        "ReAct agent",
        labels=frozenset({"AgentType"}),
        properties={"kind": "agent", "framework": "LangGraph"},
    )

    merged, conflicts = build_merged_master(master, [duplicate])

    assert merged.id == "m"
    assert merged.semantic_key == master.semantic_key
    assert merged.labels == frozenset({"Concept", "AgentType"})
    assert merged.properties == {
        "kind": "concept",
        "source": "paper",
        "framework": "LangGraph",
    }
    assert conflicts == ("kind",)


def test_merge_report_exposes_decision_shape() -> None:
    """MergeReport should retain the merge metadata needed for dry-run inspection."""

    report = MergeReport(
        master_id="m",
        duplicate_ids=("a", "b"),
        source="similarity",
        merged_labels=("AgentType", "Concept"),
        property_conflicts=("kind",),
    )

    assert report.master_id == "m"
    assert report.duplicate_ids == ("a", "b")
    assert report.merged_labels == ("AgentType", "Concept")
    assert report.property_conflicts == ("kind",)
