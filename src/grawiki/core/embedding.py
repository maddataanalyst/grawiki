"""Shared embedder protocol and default wrapper.

This module defines the single embedder contract used across GraWiki. Both
the :class:`~src.grawiki.graph.extraction.KnowledgeGraphExtractor` and the
ingestion/retrieval layers accept an :class:`Embedder` instance so that
exactly one embedding model is constructed per pipeline.
"""

from __future__ import annotations

from typing import Protocol

from pydantic_ai import Embedder as PydanticAIEmbedder


class Embedder(Protocol):
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


class DefaultEmbedder:
    """Thin wrapper around :class:`pydantic_ai.Embedder`.

    Wrapping the ``pydantic_ai`` embedder keeps a single injection point
    for the rest of the library: callers depend on :class:`Embedder` and
    swap implementations freely (e.g. with a fake in tests).

    Parameters
    ----------
    model : str
        Name of the embedding model to load.
    """

    def __init__(self, model: str) -> None:
        self._impl = PydanticAIEmbedder(model)

    async def embed_documents(self, documents: str | list[str]):
        """Embed one or more document-like strings.

        Parameters
        ----------
        documents : str or list[str]
            Text(s) to embed.

        Returns
        -------
        Any
            Embedding result object provided by the underlying backend.
        """

        return await self._impl.embed_documents(documents)

    async def embed_query(self, query: str | list[str]):
        """Embed one or more query strings.

        Parameters
        ----------
        query : str or list[str]
            Query text(s) to embed.

        Returns
        -------
        Any
            Embedding result object provided by the underlying backend.
        """

        return await self._impl.embed_query(query)
