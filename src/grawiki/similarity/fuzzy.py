"""RapidFuzz-based entity similarity matching."""

from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz

from grawiki.db.base import GraphDB, NodeHit
from grawiki.graph.models import Node


class RapidFuzzEntitySimilarityMatcher:
    """Match similar entity names using RapidFuzz.

    Parameters
    ----------
    db : GraphDB
        Graph database adapter used to enumerate persisted entities.
    default_threshold : float, optional
        Minimum similarity score required when ``threshold`` is not provided to
        :meth:`search`.
    """

    def __init__(self, db: GraphDB, *, default_threshold: float = 90.0) -> None:
        """Initialize the RapidFuzz matcher.

        Parameters
        ----------
        db : GraphDB
            Graph database adapter used to enumerate persisted entities.
        default_threshold : float, optional
            Minimum similarity score required when ``threshold`` is not provided
            to :meth:`search`.
        """

        self.db = db
        self.default_threshold = default_threshold

    async def search(
        self,
        *,
        entity: Node,
        limit: int = 10,
        threshold: float | None = None,
        candidates: Sequence[Node] | None = None,
    ) -> list[NodeHit]:
        """Return RapidFuzz similarity hits for one entity.

        Parameters
        ----------
        entity : Node
            Source entity used as the similarity query.
        limit : int, optional
            Maximum number of candidate hits to return.
        threshold : float | None, optional
            Minimum score required to keep a candidate. Defaults to
            :attr:`default_threshold`.
        candidates : Sequence[Node] | None, optional
            Optional candidate pool. When omitted, all persisted entities are
            loaded from the database.

        Returns
        -------
        list[NodeHit]
            Ranked candidate hits.

        Notes
        -----
        Candidate scores use :func:`rapidfuzz.fuzz.WRatio` and therefore fall
        in the range ``[0, 100]``.
        """

        if limit < 1:
            return []
        source_name = entity.name.strip()
        if not source_name:
            return []

        score_cutoff = self.default_threshold if threshold is None else threshold
        available = (
            list(candidates)
            if candidates is not None
            else await self.db.list_entities()
        )

        hits: list[NodeHit] = []
        for candidate in available:
            if candidate.id == entity.id:
                continue
            candidate_name = candidate.name.strip()
            if not candidate_name:
                continue
            score = float(fuzz.WRatio(source_name, candidate_name))
            if score < score_cutoff:
                continue
            hits.append(
                NodeHit(
                    node=candidate,
                    score=score,
                    matched_on="rapidfuzz",
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.node.name, hit.node.id))
        return hits[:limit]
