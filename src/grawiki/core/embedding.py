"""Shared embedder protocol and default wrapper.

This module defines the single embedder contract used across GraWiki. Both
the :class:`~src.grawiki.graph.extraction.KnowledgeGraphExtractor` and the
ingestion/retrieval layers accept an :class:`Embedder` instance so that
exactly one embedding model is constructed per pipeline.
"""

from __future__ import annotations

from typing import Protocol


class Embedding(Protocol):
    """Structural protocol for embedder implementations used by GraWiki.

    Notes
    -----
    Any object exposing ``embed_documents`` and ``embed_query`` coroutine
    methods that return an object with an ``embeddings`` attribute
    (``list[list[float]]``) satisfies this protocol.
    """

    async def embed_documents(self, documents: str | list[str]):
        """Embed one or more document-like strings.

        Parameters
        ----------
        documents : str or list[str]
            Text(s) to embed.
        """

    async def embed_query(self, query: str | list[str]):
        """Embed one or more query strings.

        Parameters
        ----------
        query : str or list[str]
            Query text(s) to embed.
        """
