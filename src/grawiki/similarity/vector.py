"""Embedding-based entity similarity matching."""

from __future__ import annotations

import math
from collections.abc import Sequence

from grawiki.db.base import GraphDB, NodeHit
from grawiki.graph.models import Node


class VectorEntitySimilarityMatcher:
    """Match similar entities using cosine similarity over embeddings.

    Parameters
    ----------
    db : GraphDB
        Graph database adapter used to enumerate persisted entities.
    default_threshold : float, optional
        Minimum cosine similarity required when ``threshold`` is not provided to
        :meth:`search`.
    """

    def __init__(self, db: GraphDB, *, default_threshold: float = 0.8) -> None:
        """Initialize the vector matcher.

        Parameters
        ----------
        db : GraphDB
            Graph database adapter used to enumerate persisted entities.
        default_threshold : float, optional
            Minimum cosine similarity required when ``threshold`` is not
            provided to :meth:`search`.
        """

        self.db = db
        self.default_threshold = default_threshold

    async def search(
        self,
        *,
        entity: Node,
        limit: int = 10,
        threshold: float | None = None,
        same_label_only: bool = True,
        candidates: Sequence[Node] | None = None,
    ) -> list[NodeHit]:
        """Return vector similarity hits for one entity.

        Parameters
        ----------
        entity : Node
            Source entity used as the similarity query.
        limit : int, optional
            Maximum number of candidate hits to return.
        threshold : float | None, optional
            Minimum cosine similarity required to keep a candidate. Defaults to
            :attr:`default_threshold`.
        same_label_only : bool, optional
            Whether candidates must share the same ontology label.
        candidates : Sequence[Node] | None, optional
            Optional candidate pool. When omitted, all persisted entities with
            embeddings are loaded from the database.

        Returns
        -------
        list[NodeHit]
            Ranked candidate hits.

        Notes
        -----
        Candidate scores use cosine similarity and therefore usually fall in
        the range ``[-1, 1]``.
        """

        if limit < 1 or not entity.embedding:
            return []

        score_cutoff = self.default_threshold if threshold is None else threshold
        available = (
            list(candidates)
            if candidates is not None
            else await self.db.list_entities(include_embeddings=True)
        )
        hits: list[NodeHit] = []
        for candidate in available:
            if candidate.id == entity.id:
                continue
            if same_label_only and candidate.label != entity.label:
                continue
            score = _cosine_similarity(entity.embedding, candidate.embedding)
            if score is None or score < score_cutoff:
                continue
            hits.append(
                NodeHit(
                    node=candidate,
                    score=score,
                    matched_on="vector",
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.node.name, hit.node.id))
        return hits[:limit]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return cosine similarity between two same-length embeddings.

    Parameters
    ----------
    left : Sequence[float]
        Left embedding.
    right : Sequence[float]
        Right embedding.

    Returns
    -------
    float | None
        Cosine similarity in the range ``[-1, 1]`` or ``None`` when the inputs
        are empty, have different lengths, or contain a zero vector.

    Notes
    -----
    This helper performs an exact in-memory cosine calculation and does not use
    the database vector index.
    """

    if not left or not right or len(left) != len(right):
        return None
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)
