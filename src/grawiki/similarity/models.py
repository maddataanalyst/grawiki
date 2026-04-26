"""Models for entity similarity and duplicate inspection."""

from __future__ import annotations

from dataclasses import dataclass

from grawiki.db.base import NodeHit
from grawiki.graph.models import Node


@dataclass
class EntitySimilarityResult:
    """Similarity result set for one source entity.

    Parameters
    ----------
    source : Node
        Entity node that was used as the similarity query.
    hits : list[NodeHit]
        Ranked candidate matches for the source entity.
    """

    source: Node
    hits: list[NodeHit]


@dataclass
class SemanticKeyCollisionCandidates:
    """Similarity candidates generated for a duplicated semantic key group.

    Parameters
    ----------
    semantic_key : str
        Semantic key shared by more than one persisted entity.
    results : list[EntitySimilarityResult]
        Per-entity similarity results restricted to the collision group.
    """

    semantic_key: str
    results: list[EntitySimilarityResult]


@dataclass
class EntityDuplicateCandidates:
    """Two-stage duplicate-candidate report for persisted entities.

    Parameters
    ----------
    semantic_key_collisions : dict[str, list[Node]]
        Exact collision groups keyed by semantic key.
    semantic_key_collision_candidates : list[SemanticKeyCollisionCandidates]
        Matcher-ranked candidates restricted to exact semantic-key collision
        groups.
    similarity_candidates : list[EntitySimilarityResult]
        Matcher-ranked candidates found by the broader similarity scan.
    """

    semantic_key_collisions: dict[str, list[Node]]
    semantic_key_collision_candidates: list[SemanticKeyCollisionCandidates]
    similarity_candidates: list[EntitySimilarityResult]
