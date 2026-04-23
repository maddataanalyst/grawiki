"""Retrieval strategy layer — owns the embedder on the query path."""

from __future__ import annotations

from src.grawiki.core.embedding import Embedder
from src.grawiki.db.base import GraphDB, NodeHit
from src.grawiki.graph.models import Node


class Retriever:
    """Embed queries, call DB primitives, and compose results.

    The retriever is the single place where query-side embedding happens.
    The DB layer receives only pre-computed vectors and raw text; ranking,
    deduplication, and expansion strategy are concerns of this class.

    Parameters
    ----------
    db : GraphDB
        Storage engine adapter used for raw search primitives.
    embedder : Embedder
        Shared embedder instance. The same instance should be reused by
        the ingestion path to avoid loading the model twice.
    """

    def __init__(self, db: GraphDB, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder

    async def fulltext(
        self,
        query: str,
        *,
        labels: list[str],
        limit: int = 10,
    ) -> list[NodeHit]:
        """Run a full-text search across the given node labels.

        Parameters
        ----------
        query : str
            Raw query text forwarded to the DB full-text index.
        labels : list[str]
            Node labels to search.
        limit : int, optional
            Maximum hits per label.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated hit list across all requested labels.
        """

        hits = await self.db.fulltext_search(
            labels=labels, query_text=query, limit=limit
        )
        return _deduplicate_hits(hits)

    async def vector(
        self,
        query: str,
        *,
        labels: list[str],
        limit: int = 10,
    ) -> list[NodeHit]:
        """Embed a query and run a vector similarity search.

        Parameters
        ----------
        query : str
            Raw query text. The retriever embeds it using :attr:`embedder`.
        labels : list[str]
            Node labels to search.
        limit : int, optional
            Maximum hits per label.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated hit list sorted by score ascending.

        Raises
        ------
        ValueError
            Raised when the embedder returns an empty result.
        """

        result = await self.embedder.embed_query(query)
        if not result.embeddings:
            raise ValueError("Embedder returned an empty result for query.")
        embedding = list(result.embeddings[0])
        hits = await self.db.vector_search(
            labels=labels, query_embedding=embedding, limit=limit
        )
        return _deduplicate_hits(hits)

    async def expand(
        self,
        seeds: list[NodeHit],
        *,
        rel_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[Node]:
        """Fetch graph neighbors of the given seed hits.

        Parameters
        ----------
        seeds : list[NodeHit]
            Starting nodes. Their ``node.id`` values are used as seeds.
        rel_types : list[str] | None, optional
            Restrict traversal to these relationship types. ``None`` follows
            any relationship.
        depth : int, optional
            Maximum traversal depth.

        Returns
        -------
        list[Node]
            Distinct neighbor nodes reachable from the seeds.
        """

        if not seeds:
            return []
        return await self.db.neighbors(
            node_ids=[h.node.id for h in seeds],
            rel_types=rel_types,
            depth=depth,
        )


def _deduplicate_hits(hits: list[NodeHit]) -> list[NodeHit]:
    """Return a deduplicated copy of ``hits``, preserving order by first occurrence."""

    seen: set[str] = set()
    result: list[NodeHit] = []
    for hit in hits:
        node_id = hit.node.id
        if node_id in seen:
            continue
        seen.add(node_id)
        result.append(hit)
    return result
