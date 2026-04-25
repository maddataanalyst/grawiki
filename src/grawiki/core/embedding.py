"""Shared embedding protocol used across GraWiki.

This module defines the structural contract that embedding clients must satisfy
when they are passed into GraWiki components. The project does not wrap a
default embedding implementation here; callers typically construct
``pydantic_ai.Embedder`` directly and share that instance across ingestion,
extraction, retrieval, and similarity workflows.
"""

from __future__ import annotations

from typing import Protocol


class Embedding(Protocol):
    """Structural protocol for embedding implementations used by GraWiki.

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
