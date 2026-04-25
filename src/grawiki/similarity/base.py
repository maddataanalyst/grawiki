"""Protocols for entity similarity matching."""

from __future__ import annotations

from typing import Protocol, Sequence

from grawiki.db.base import NodeHit
from grawiki.graph.models import Node


class EntitySimilarityMatcher(Protocol):
    """Protocol for entity-to-entity similarity matcher implementations."""

    async def search(
        self,
        *,
        entity: Node,
        limit: int = 10,
        threshold: float | None = None,
        candidates: Sequence[Node] | None = None,
    ) -> list[NodeHit]:
        """Return ranked entity candidates for ``entity``.

        Parameters
        ----------
        entity : Node
            Source entity used as the similarity query.
        limit : int, optional
            Maximum number of candidate hits to return.
        threshold : float | None, optional
            Optional matcher-specific minimum score.
        candidates : Sequence[Node] | None, optional
            Optional pre-filtered candidate pool.

        Returns
        -------
        list[NodeHit]
            Ranked candidate hits for the source entity.
        """
