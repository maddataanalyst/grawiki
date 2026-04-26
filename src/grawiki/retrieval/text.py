"""Text retrieval strategy layer that owns query-side embedding."""

from __future__ import annotations

from typing import Literal

from grawiki.core.embedding import Embedding
from grawiki.db.base import GraphDB, NodeHit
from grawiki.retrieval.base import Retriever


class TextRetriever(Retriever):
    """Embed queries, call DB primitives, and compose results.

    The retriever is the single place where query-side embedding happens.
    The DB layer receives only pre-computed vectors and raw text; ranking
    and deduplication are concerns of this class.

    Parameters
    ----------
    db : GraphDB
        Storage engine adapter used for raw search primitives.
    embedding : Embedding
        Shared embedding client instance. The same instance should be reused by
        the ingestion path to avoid loading the model twice.
    """

    def __init__(
        self,
        db: GraphDB,
        embedding: Embedding,
        search_method: Literal["vector", "fulltext"] = "vector",
        search_labels: list[str] | None = None,
    ) -> None:
        self.db = db
        self.embedding = embedding
        self.search_method = search_method
        self.search_labels = search_labels or ["__chunk__"]

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        """Run a retrieval query and return a list of hits.

        Parameters
        ----------
        query : str
            Raw query text.
        limit : int | None, optional
            Maximum hits per label. Overrides the default limit if provided.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated hit list across all requested labels.
        """

        if self.search_method == "vector":
            return await self.vector(
                query=query, labels=self.search_labels, limit=limit
            )
        elif self.search_method == "fulltext":
            return await self.fulltext(
                query=query, labels=self.search_labels, limit=limit
            )
        else:
            raise ValueError(f"Unsupported search method: {self.search_method}")

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
            Raw query text. The retriever embeds it using :attr:`embedding`.
        labels : list[str]
            Node labels to search.
        limit : int, optional
            Maximum hits per label.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated hit list with higher scores representing more
            relevant matches.

        Raises
        ------
        ValueError
            Raised when the embedding returns an empty result.
        """

        result = await self.embedding.embed_query(query)
        if not result.embeddings:
            raise ValueError("Embedding returned an empty result for query.")
        embedding = list(result.embeddings[0])
        hits = await self.db.vector_search(
            labels=labels, query_embedding=embedding, limit=limit
        )
        return _deduplicate_hits(hits)


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
