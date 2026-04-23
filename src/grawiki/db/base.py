"""Backend-agnostic graph database interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from src.grawiki.core.commons import Chunk, Document
from src.grawiki.graph.models import ChunkNode, DocumentNode, KnowledgeGraph, Node


SearchMethod = Literal["fulltext", "vector"]
SearchResults = dict[str, list[dict[str, Any]]]


@dataclass
class NodeHit:
    """Search result pairing a node with scoring metadata.

    Parameters
    ----------
    node : Node
    Node returned by the backend. May be a concrete subclass such as
        :class:`~src.grawiki.graph.models.DocumentNode`,
        :class:`~src.grawiki.graph.models.ChunkNode`, or
        :class:`~src.grawiki.graph.models.MemoryNode` depending on the
        node's label.
    score : float, optional
        Backend-reported similarity score. Defaults to ``0.0`` when the
        backend does not provide one (for example full-text hits).
    matched_on : str, optional
        Short descriptor of how the hit was matched (for example
        ``"fulltext:content"`` or ``"vector"``). Empty when not
        reported.
    """

    node: Node
    score: float = 0.0
    matched_on: str = ""


class GraphDB(ABC):
    """Abstract interface for graph database adapters.

    Notes
    -----
    The contract has two layers. Storage-engine primitives
    (:meth:`upsert_nodes`, :meth:`upsert_relationships`,
    :meth:`fulltext_search`, :meth:`vector_search`, :meth:`neighbors`,
    :meth:`ensure_indexes`) are the foundational operations every backend
    must implement. Higher-level convenience methods
    (:meth:`save_documents_and_chunks`, :meth:`save_docs_and_chunks_to_db`,
    :meth:`save_entities_and_rels`, :meth:`search`) are currently parallel
    abstract methods; they will be collapsed into thin wrappers over the
    primitives in a follow-on migration step.
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

    @abstractmethod
    async def ensure_indexes(
        self,
        *,
        labels: Iterable[str],
        vector_dims: Mapping[str, int] | None = None,
    ) -> None:
        """Ensure full-text and vector indexes exist for ``labels``.

        Parameters
        ----------
        labels : Iterable[str]
            Node labels whose indexes should be created.
        vector_dims : Mapping[str, int] | None, optional
            Per-label embedding dimensionality. Labels omitted from the
            mapping do not get a vector index.
        """

    @abstractmethod
    async def fulltext_search(
        self,
        *,
        labels: Sequence[str],
        query_text: str,
        limit: int = 10,
    ) -> list[NodeHit]:
        """Run a full-text search across one or more node labels.

        Parameters
        ----------
        labels : Sequence[str]
            Labels whose full-text indexes should be queried.
        query_text : str
            Raw full-text query string.
        limit : int, optional
            Maximum number of hits to return per label.

        Returns
        -------
        list[NodeHit]
            Flat list of hits across the requested labels. Callers group
            by ``hit.node.label`` when a grouped view is needed.
        """

    @abstractmethod
    async def vector_search(
        self,
        *,
        labels: Sequence[str],
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[NodeHit]:
        """Run a vector similarity search across one or more node labels.

        Parameters
        ----------
        labels : Sequence[str]
            Labels whose vector indexes should be queried.
        query_embedding : list[float]
            Pre-computed query embedding. The DB does not embed queries;
            that concern lives in the retrieval layer.
        limit : int, optional
            Maximum number of hits to return per label.

        Returns
        -------
        list[NodeHit]
            Flat list of hits across the requested labels.
        """

    @abstractmethod
    async def neighbors(
        self,
        *,
        node_ids: Sequence[str],
        rel_types: Sequence[str] | None = None,
        depth: int = 1,
    ) -> list[Node]:
        """Fetch neighbors of the given seed nodes up to ``depth`` hops.

        Parameters
        ----------
        node_ids : Sequence[str]
            Seed node identifiers.
        rel_types : Sequence[str] | None, optional
            Restrict traversal to these relationship types. ``None`` means
            follow any relationship.
        depth : int, optional
            Maximum traversal depth. Defaults to one hop.

        Returns
        -------
        list[Node]
            Distinct neighbor nodes reachable from the seeds, excluding the
            seeds themselves.
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
