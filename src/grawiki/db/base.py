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
        Matched node. May be a concrete subclass such as
        :class:`~grawiki.graph.models.DocumentNode`,
        :class:`~grawiki.graph.models.ChunkNode`, or
        :class:`~grawiki.graph.models.MemoryNode` depending on the
        node's label.
    score : float, optional
        Relevance or similarity score. Adapters and higher-level services may
        normalize backend-specific distance values into higher-is-better
        scores. Defaults to ``0.0`` when no score is reported.
    matched_on : str, optional
        Short descriptor of how the hit was matched (for example
        ``"fulltext:content"``, ``"vector"``, or ``"rapidfuzz"``). Empty
        when not reported.
    """

    node: Node
    score: float = 0.0
    matched_on: str = ""


@dataclass
class NeighborRelationship:
    """One-hop relationship context around a seed node.

    Parameters
    ----------
    source_id : str
        Identifier of the seed node that was expanded.
    source_name : str
        Human-readable name of the seed node.
    relationship_label : str
        Label of the relationship connecting the seed to the target.
    target : Node
        Neighbor node connected to the seed.
    """

    source_id: str
    source_name: str
    relationship_label: str
    target: Node


class GraphDB(ABC):
    """Abstract interface for graph database adapters.

    Notes
    -----
    The contract has two layers. Storage-engine primitives
    (:meth:`upsert_nodes`, :meth:`upsert_relationships`,
    :meth:`fulltext_search`, :meth:`vector_search`,
    :meth:`neighbor_relationships`, :meth:`list_entities`,
    :meth:`ensure_indexes`) are the
    foundational operations every backend
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
            Flat list of hits across the requested labels. Callers group by the
            node family / label set when a grouped view is needed.
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
    async def neighbor_relationships(
        self,
        *,
        node_ids: Sequence[str],
        limit_per_node: int = 5,
    ) -> dict[str, list[NeighborRelationship]]:
        """Fetch one-hop relationship context for the given seed nodes.

        Parameters
        ----------
        node_ids : Sequence[str]
            Seed node identifiers.
        limit_per_node : int, optional
            Maximum number of one-hop relationships returned for each seed.

        Returns
        -------
        dict[str, list[NeighborRelationship]]
            Relationship context keyed by seed node identifier. Seed ids with
            no matching context should still be present with an empty list.
        """

    @abstractmethod
    async def recall_subgraph(
        self,
        *,
        memory_ids: Sequence[str],
        hops: int = 1,
        limit_per_memory: int = 20,
    ) -> dict[str, list[NeighborRelationship]]:
        """Fetch a flattened k-hop recall subgraph for memory seeds.

        Parameters
        ----------
        memory_ids : Sequence[str]
            Memory node identifiers used as traversal seeds.
        hops : int, optional
            Maximum traversal depth in hops. Must be at least ``1``.
        limit_per_memory : int, optional
            Maximum number of distinct paths expanded per memory seed before
            flattening them into relationship rows.

        Returns
        -------
        dict[str, list[NeighborRelationship]]
            Flattened relationship rows keyed by memory id. Traversal is
            undirected for discovery, but each row preserves the stored
            relationship direction.
        """

    @abstractmethod
    async def list_entities(self, *, include_embeddings: bool = False) -> list[Node]:
        """Return persisted entity nodes.

        Parameters
        ----------
        include_embeddings : bool, optional
            Whether entity embeddings should be loaded when available. Callers
            that only need identifiers and names should keep this disabled to
            avoid transferring large vectors unnecessarily.

        Returns
        -------
        list[Node]
            Persisted entity nodes ordered by backend-defined stable ordering.
        """

    @abstractmethod
    async def entity_relationship_counts(
        self, node_ids: Sequence[str]
    ) -> dict[str, int]:
        """Return incident relationship counts for entity nodes.

        Parameters
        ----------
        node_ids : Sequence[str]
            Entity identifiers whose incident edge counts should be returned.

        Returns
        -------
        dict[str, int]
            Mapping from entity id to total incoming-plus-outgoing relationship
            count. Missing ids should still appear with ``0``.
        """

    @abstractmethod
    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        """Upsert nodes. Dispatches on labels for persistence semantics.

        Parameters
        ----------
        nodes : Sequence[Node]
            Nodes to create or update. Each node's label set determines which
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

    @abstractmethod
    async def merge_entity_nodes(
        self,
        *,
        master: Node,
        duplicate_ids: Sequence[str],
    ) -> None:
        """Merge duplicate entity nodes into ``master``.

        Parameters
        ----------
        master : Node
            Final persisted state for the surviving master node. The master is
            matched by ``master.id`` and updated before duplicate nodes are
            deleted.
        duplicate_ids : Sequence[str]
            Entity identifiers to merge into ``master`` and then delete.

        Raises
        ------
        ValueError
            Raised when ``duplicate_ids`` contains ``master.id``.
        """

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> None:
        """Delete one memory and prune now-orphaned directly mentioned entities.

        Parameters
        ----------
        memory_id : str
            Identifier of the memory node to remove.
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
        owner_ids: Sequence[str],
        owner_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        """Persist extracted owner-linked entities and relationships.

        Parameters
        ----------
        owner_ids : Sequence[str]
            Node identifiers that own the extracted graphs, such as chunk or
            memory ids.
        owner_graphs : dict[str, KnowledgeGraph]
            Extracted graphs keyed by owner identifier.

        Raises
        ------
        ValueError
            Raised when a graph references a chunk identifier that is not
            present in ``owner_ids``.
        """

        known_owner_ids = set(owner_ids)
        unknown_owner_ids = sorted(set(owner_graphs) - known_owner_ids)
        if unknown_owner_ids:
            missing = ", ".join(unknown_owner_ids)
            raise ValueError(f"Unknown owner ids in owner_graphs: {missing}")

        all_entity_nodes = [
            node for graph in owner_graphs.values() for node in graph.nodes
        ]
        await self.upsert_nodes(all_entity_nodes)

        canonical_ids_by_node_id: dict[str, str] = {}
        canonical_ids_by_semantic_key: dict[str, str] = {}
        for node in all_entity_nodes:
            canonical_id = canonical_ids_by_semantic_key.setdefault(
                node.semantic_key,
                node.id,
            )
            canonical_ids_by_node_id[node.id] = canonical_id

        mentions_rels: list[Relationship] = []
        entity_rels: list[Relationship] = []
        for owner_id, graph in owner_graphs.items():
            nodes_by_id = {node.id: node for node in graph.nodes}
            for node in graph.nodes:
                mentions_rels.append(
                    Relationship(
                        id=str(uuid.uuid4()),
                        source=owner_id,
                        target=canonical_ids_by_node_id[node.id],
                        label="__mentions__",
                    )
                )
            for rel in graph.relationships:
                if rel.source not in nodes_by_id or rel.target not in nodes_by_id:
                    raise ValueError(
                        "Relationship references a node missing from the "
                        f"owner graph for owner '{owner_id}'."
                    )
                entity_rels.append(
                    rel.model_copy(
                        update={
                            "source": canonical_ids_by_node_id[rel.source],
                            "target": canonical_ids_by_node_id[rel.target],
                        }
                    )
                )

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

        labels = ["__document__", "__chunk__", "__memory__", "__entity__"]
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

    from grawiki.graph.models import ChunkNode, DocumentNode, MemoryNode

    groups: SearchResults = {
        "__document__": [],
        "__chunk__": [],
        "__memory__": [],
        "__entity__": [],
    }
    for hit in hits:
        labels = hit.node.labels
        bucket = None
        if DocumentNode.system_label in labels:
            bucket = groups["__document__"]
        elif ChunkNode.system_label in labels:
            bucket = groups["__chunk__"]
        elif MemoryNode.system_label in labels:
            bucket = groups["__memory__"]
        elif not any(label.startswith("__") for label in labels):
            bucket = groups["__entity__"]
        if bucket is None:
            continue
        if len(bucket) >= limit:
            continue
        row: dict[str, Any] = {
            "id": hit.node.id,
            "name": hit.node.name,
        }

        if isinstance(hit.node, DocumentNode):
            row["content"] = hit.node.content
        elif isinstance(hit.node, ChunkNode):
            row["content"] = hit.node.content
            row["document_id"] = hit.node.document_id
        elif isinstance(hit.node, MemoryNode):
            row["content"] = hit.node.content
            row["creation_date"] = hit.node.creation_date
            row["metadata"] = dict(hit.node.metadata)
        else:
            row["labels"] = sorted(hit.node.labels)
            row["semantic_key"] = hit.node.semantic_key
        if hit.matched_on:
            row["matched_on"] = hit.matched_on
        if hit.score:
            row["score"] = hit.score
        bucket.append(row)
    return groups
