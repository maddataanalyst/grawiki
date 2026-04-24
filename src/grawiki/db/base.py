"""Backend-agnostic graph database interfaces."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from grawiki.core.commons import Chunk, Document
from grawiki.graph.models import (
    ChunkNode,
    DocumentNode,
    KnowledgeGraph,
    Node,
    Relationship,
)


SearchMethod = Literal["fulltext", "vector"]
SearchResults = dict[str, list[dict[str, Any]]]


@dataclass
class NodeHit:
    """Search result pairing a node with scoring metadata.

    Parameters
    ----------
    node : Node
        Node returned by the backend. May be a concrete subclass such as
        :class:`~grawiki.graph.models.DocumentNode`,
        :class:`~grawiki.graph.models.ChunkNode`, or
        :class:`~grawiki.graph.models.MemoryNode` depending on the
        node's label.
    score : float, optional
        Adapter-reported relevance score. Adapters may normalize
        backend-specific distance values into higher-is-better scores.
        Defaults to ``0.0`` when the backend does not provide one (for
        example full-text hits).
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
    :meth:`save_entities_and_rels`, :meth:`search`) are thin wrappers over
    the primitives that preserve the legacy API during the migration.
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

    @abstractmethod
    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        """Upsert nodes. Dispatches on label for persistence semantics.

        Parameters
        ----------
        nodes : Sequence[Node]
            Nodes to create or update. Each node's ``label`` determines which
            concrete storage path is used.
        """

    @abstractmethod
    async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
        """Upsert relationships between existing nodes (matched by id).

        Parameters
        ----------
        rels : Sequence[Relationship]
            Relationships to create or update. Both endpoints must already
            exist in the graph and are matched by their ``id`` field.
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

        await self.upsert_nodes([*doc_nodes, *chunk_nodes])
        has_chunk_rels = [
            Relationship(
                id=str(uuid.uuid4()),
                source=doc_node.id,
                target=chunk_node.id,
                label="__has_chunk__",
            )
            for doc_node in doc_nodes
            for chunk_node in chunk_nodes
            if chunk_node.document_id == doc_node.id
        ]
        await self.upsert_relationships(has_chunk_rels)

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

        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        unknown_chunk_ids = sorted(set(chunk_graphs) - set(chunks_by_id))
        if unknown_chunk_ids:
            missing = ", ".join(unknown_chunk_ids)
            raise ValueError(f"Unknown chunk ids in chunk_graphs: {missing}")

        all_entity_nodes = [
            node for graph in chunk_graphs.values() for node in graph.nodes
        ]
        await self.upsert_nodes(all_entity_nodes)

        mentions_rels: list[Relationship] = []
        entity_rels: list[Relationship] = []
        for chunk_id, graph in chunk_graphs.items():
            nodes_by_id = {node.id: node for node in graph.nodes}
            for node in graph.nodes:
                mentions_rels.append(
                    Relationship(
                        id=str(uuid.uuid4()),
                        source=chunk_id,
                        target=node.semantic_key,
                        label="__mentions__",
                    )
                )
            for rel in graph.relationships:
                if rel.source not in nodes_by_id or rel.target not in nodes_by_id:
                    raise ValueError(
                        "Relationship references a node missing from the "
                        f"chunk graph for chunk '{chunk_id}'."
                    )
                entity_rels.append(rel)

        await self.upsert_relationships(mentions_rels)
        await self.upsert_relationships(entity_rels)

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

        labels = ["__document__", "__chunk__", "__entity__"]
        if method == "fulltext":
            hits = await self.fulltext_search(
                labels=labels, query_text=query, limit=limit
            )
        else:
            if query_embedding is None:
                raise ValueError("Vector search requires a query embedding.")
            hits = await self.vector_search(
                labels=labels, query_embedding=query_embedding, limit=limit
            )
        return _group_hits_by_label(hits, limit=limit)


def _group_hits_by_label(hits: list[NodeHit], *, limit: int) -> SearchResults:
    """Group a flat list of NodeHit objects by node label, preserving dict shape.

    Parameters
    ----------
    hits : list[NodeHit]
        Flat list of search hits from the DB primitives.
    limit : int
        Maximum results per label group.

    Returns
    -------
    SearchResults
        Hits grouped by label, capped at ``limit`` per group.
    """

    groups: SearchResults = {"__document__": [], "__chunk__": [], "__entity__": []}
    for hit in hits:
        label = hit.node.label
        bucket = groups.get(label)
        if bucket is None:
            # Ontology labels (e.g. "Person") are entities; system labels that
            # don't match a group (e.g. "__memory__") are dropped.
            if not label.startswith("__"):
                bucket = groups.get("__entity__")
        if bucket is None:
            continue
        if len(bucket) >= limit:
            continue
        row: dict[str, Any] = {
            "id": hit.node.id,
            "name": hit.node.name,
        }
        from grawiki.graph.models import ChunkNode, DocumentNode

        if isinstance(hit.node, DocumentNode):
            row["content"] = hit.node.content
        elif isinstance(hit.node, ChunkNode):
            row["content"] = hit.node.content
            row["document_id"] = hit.node.document_id
        else:
            row["label"] = hit.node.label
            row["semantic_key"] = hit.node.semantic_key
        if hit.matched_on:
            row["matched_on"] = hit.matched_on
        if hit.score:
            row["score"] = hit.score
        bucket.append(row)
    return groups
