"""Helpers for post-persistence entity deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from grawiki.graph.models import Node


@dataclass
class MergeReport:
    """Summary of one merge decision.

    Parameters
    ----------
    master_id : str
        Identifier of the surviving entity node.
    duplicate_ids : tuple[str, ...]
        Identifiers of duplicate nodes merged into the master.
    source : str
        Candidate source category, e.g. ``"collision"`` or ``"similarity"``.
    merged_labels : tuple[str, ...]
        Alphabetically sorted label set that will remain on the master.
    property_conflicts : tuple[str, ...]
        Property keys for which multiple duplicate values were observed and the
        master's value was kept.
    """

    master_id: str
    duplicate_ids: tuple[str, ...]
    source: str
    merged_labels: tuple[str, ...]
    property_conflicts: tuple[str, ...]


def pick_master(nodes: Sequence[Node], relation_counts: Mapping[str, int]) -> Node:
    """Return the preferred master node from a duplicate group."""

    if not nodes:
        raise ValueError("pick_master requires at least one node.")
    return sorted(
        nodes,
        key=lambda node: (
            -len(node.properties),
            -int(relation_counts.get(node.id, 0)),
            -len(node.name),
            node.id,
        ),
    )[0]


def merge_node_properties(
    master: Node,
    duplicates: Sequence[Node],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Merge node properties while preserving the master's values on conflicts."""

    merged = dict(master.properties)
    conflicts: set[str] = set()
    for node in duplicates:
        for key, value in node.properties.items():
            existing = merged.get(key)
            if existing is None:
                merged[key] = value
                continue
            if existing != value:
                conflicts.add(key)
    return merged, tuple(sorted(conflicts))


def build_merged_master(
    master: Node,
    duplicates: Sequence[Node],
) -> tuple[Node, tuple[str, ...]]:
    """Return the final master node state for a merge group."""

    merged_properties, conflicts = merge_node_properties(master, duplicates)
    merged_labels = frozenset().union(
        master.labels, *(node.labels for node in duplicates)
    )
    return (
        master.model_copy(
            update={
                "labels": merged_labels,
                "properties": merged_properties,
            },
            deep=True,
        ),
        conflicts,
    )
