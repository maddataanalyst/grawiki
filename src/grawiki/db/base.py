"""Backend-agnostic graph database interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from src.grawiki.core.commons import Chunk, Document
from src.grawiki.graph.models import ChunkNode, DocumentNode, KnowledgeGraph


SearchMethod = Literal["fulltext", "vector"]
SearchResults = dict[str, list[dict[str, Any]]]


class GraphDB(ABC):
    """Abstract interface for graph database adapters.

    Notes
    -----
    Concrete adapters are responsible for mapping these semantic operations
    onto backend-specific Cypher implementations.
    """

    @abstractmethod
    async def setup(self, embedding_dimensions: dict[str, int] | None = None) -> None:
        """Prepare backend indexes and other database structures.

        Parameters
        ----------
        embedding_dimensions : dict[str, int] | None, optional
            Mapping from node label to embedding dimensionality for vector
            indexes that require the dimension to be known ahead of time.
        """

    async def save_documents_and_chunks(
        self,
        documents: list[Document],
        chunks: list[Chunk],
    ) -> None:
        """Persist source documents and their chunks.

        Parameters
        ----------
        documents : list[Document]
            Source documents to persist.
        chunks : list[Chunk]
            Source chunks to persist and connect to their parent documents.
        """

        document_nodes = [DocumentNode.from_document(doc) for doc in documents]
        chunk_nodes = [ChunkNode.from_chunk(chunk) for chunk in chunks]
        await self.save_docs_and_chunks_to_db(document_nodes, chunk_nodes)

    @abstractmethod
    async def save_docs_and_chunks_to_db(
        self,
        doc_nodes: list[DocumentNode],
        chunk_nodes: list[ChunkNode],
    ) -> None:
        """Persist prepared document and chunk nodes.

        Parameters
        ----------
        doc_nodes : list[DocumentNode]
            Prepared document nodes ready for persistence.
        chunk_nodes : list[ChunkNode]
            Prepared chunk nodes ready for persistence.
        """

    @abstractmethod
    async def save_entities_and_rels(
        self,
        chunks: list[Chunk],
        chunk_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        """Persist extracted chunk entities and relationships.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks that own the extracted graphs.
        chunk_graphs : dict[str, KnowledgeGraph]
            Extracted graphs keyed by chunk identifier.

        Raises
        ------
        ValueError
            Raised when a graph references a chunk identifier that is not
            present in ``chunks``.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        method: SearchMethod,
        *,
        limit: int = 10,
        query_embedding: list[float] | None = None,
    ) -> SearchResults:
        """Search documents, chunks, and entities.

        Parameters
        ----------
        query : str
            Raw user query text.
        method : {"fulltext", "vector"}
            Search strategy to execute.
        limit : int, optional
            Maximum number of results to return per node family.
        query_embedding : list[float] | None, optional
            Embedded query vector required for vector search.

        Returns
        -------
        SearchResults
            Search hits grouped by node family.
        """
