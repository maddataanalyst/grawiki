"""FalkorDBLite-backed graph database adapter."""

from __future__ import annotations

import math
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from redis.exceptions import ResponseError
from redislite.falkordb_client import FalkorDB

from src.grawiki.core.commons import Chunk
from src.grawiki.db.base import GraphDB, NodeHit, SearchMethod, SearchResults
from src.grawiki.db.cypher_queries import (
    build_chunk_upsert_query,
    build_document_upsert_query,
    build_entity_upsert_query,
    build_relationship_upsert_query,
)
from src.grawiki.graph.models import (
    ChunkNode,
    DocumentNode,
    KnowledgeGraph,
    MemoryNode,
    Node,
)


logger = logging.getLogger(__name__)


_FULLTEXT_INDEX_FIELDS: dict[str, tuple[str, ...]] = {
    "__document__": ("name", "content"),
    "__chunk__": ("name", "content"),
    "__memory__": ("name", "content"),
    "__entity__": ("name",),
}
_FALLBACK_FULLTEXT_FIELDS: tuple[str, ...] = ("name",)
_VECTOR_INDEX_LABELS = ("__document__", "__chunk__", "__memory__", "__entity__")

# Column order used by the generic node-row return expression. Keeping this
# in one place lets the parsing helper index into result rows safely.
_NODE_COLUMNS: tuple[str, ...] = (
    "id",
    "label",
    "semantic_key",
    "name",
    "properties",
    "content",
    "document_id",
    "creation_date",
    "metadata",
)

# Stored relationship types must match this shape to be safe to interpolate
# into Cypher. This accepts the project's ``__system__``-style reserved
# types without the lossy normalization that ``sanitize_cypher_identifier``
# applies to entity labels.
_REL_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class FalkorGraphDB(GraphDB):
    """Graph adapter implemented on top of FalkorDBLite.

    Parameters
    ----------
    db_path : str | Path
        Filesystem path used by FalkorDBLite for persistence.
    graph_name : str
        Logical graph name within the database file.
    vector_similarity_function : Literal["cosine", "euclidean"], optional
        Similarity function used for vector indexes.
    vector_index_m : int, optional
        HNSW graph connectivity parameter used for vector indexes.
    vector_index_ef_construction : int, optional
        HNSW construction candidate count used for vector indexes.
    vector_index_ef_runtime : int, optional
        HNSW query-time candidate count used for vector indexes.
    """

    def __init__(
        self,
        db_path: str | Path,
        graph_name: str,
        *,
        vector_similarity_function: Literal["cosine", "euclidean"] = "cosine",
        vector_index_m: int = 16,
        vector_index_ef_construction: int = 200,
        vector_index_ef_runtime: int = 10,
    ) -> None:
        self.db_path = str(Path(db_path))
        self.graph_name = graph_name
        self.vector_similarity_function = vector_similarity_function
        self.vector_index_m = vector_index_m
        self.vector_index_ef_construction = vector_index_ef_construction
        self.vector_index_ef_runtime = vector_index_ef_runtime
        self._db = FalkorDB(self.db_path)
        self._graph = self._db.select_graph(graph_name)
        self._fulltext_indexes_ready = False
        self._vector_index_dimensions: dict[str, int] = {}

    async def setup(self, embedding_dimensions: dict[str, int] | None = None) -> None:
        """Prepare FalkorDB indexes used by the application.

        Parameters
        ----------
        embedding_dimensions : dict[str, int] | None, optional
            Mapping from node label to embedding dimensionality used for vector
            index creation.
        """

        if not self._fulltext_indexes_ready:
            self._ensure_fulltext_indexes()
            self._fulltext_indexes_ready = True

        if embedding_dimensions:
            self._ensure_vector_indexes(embedding_dimensions)

    async def save_docs_and_chunks_to_db(
        self,
        doc_nodes: list[DocumentNode],
        chunk_nodes: list[ChunkNode],
    ) -> None:
        """Persist prepared document and chunk nodes to FalkorDBLite.

        Parameters
        ----------
        doc_nodes : list[DocumentNode]
            Document nodes to upsert.
        chunk_nodes : list[ChunkNode]
            Chunk nodes to upsert.
        """

        await self.setup(self._collect_embedding_dimensions([*doc_nodes, *chunk_nodes]))

        for doc_node in doc_nodes:
            payload = doc_node.model_dump()
            payload["metadata"] = self._serialize_mapping(doc_node.metadata)
            embedding_literal = self._serialize_embedding(doc_node.embedding)
            self._query(build_document_upsert_query(embedding_literal), payload)

        for chunk_node in chunk_nodes:
            payload = chunk_node.model_dump()
            payload["metadata"] = self._serialize_mapping(chunk_node.metadata)
            embedding_literal = self._serialize_embedding(chunk_node.embedding)
            self._query(build_chunk_upsert_query(embedding_literal), payload)

    async def save_entities_and_rels(
        self,
        chunks: list[Chunk],
        chunk_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        """Persist extracted entities and relationships for the provided chunks.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks that own the extracted graphs.
        chunk_graphs : dict[str, KnowledgeGraph]
            Extracted graphs keyed by chunk identifier.

        Raises
        ------
        ValueError
            Raised when a graph references an unknown chunk identifier.
        """

        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        unknown_chunk_ids = sorted(set(chunk_graphs) - set(chunks_by_id))
        if unknown_chunk_ids:
            missing = ", ".join(unknown_chunk_ids)
            raise ValueError(f"Unknown chunk ids in chunk_graphs: {missing}")

        entity_nodes = [node for graph in chunk_graphs.values() for node in graph.nodes]
        await self.setup(self._collect_embedding_dimensions(entity_nodes))

        for chunk_id, graph in chunk_graphs.items():
            nodes_by_id = {node.id: node for node in graph.nodes}

            for node in graph.nodes:
                self._save_entity(chunk_id=chunk_id, node=node)

            for relationship in graph.relationships:
                try:
                    source_node = nodes_by_id[relationship.source]
                    target_node = nodes_by_id[relationship.target]
                except KeyError as exc:
                    raise ValueError(
                        "Relationship references a node missing from the "
                        f"chunk graph for chunk '{chunk_id}'."
                    ) from exc

                self._query(
                    build_relationship_upsert_query(relationship.label),
                    {
                        "id": relationship.id,
                        "label": relationship.label,
                        "source_semantic_key": source_node.semantic_key,
                        "target_semantic_key": target_node.semantic_key,
                        "properties": self._serialize_mapping(relationship.properties),
                    },
                )

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a write-capable query.

        Parameters
        ----------
        query : str
            Cypher query to execute.
        params : dict[str, Any] | None, optional
            Query parameters.

        Returns
        -------
        Any
            Backend-native query result.
        """

        return self._query(query, params)

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
            Query vector used for vector search.

        Returns
        -------
        SearchResults
            Search hits grouped by node family.
        """

        if method == "fulltext":
            return self._search_fulltext(query, limit=limit)
        if query_embedding is None:
            raise ValueError("Vector search requires a query embedding.")
        return self._search_vector(query_embedding, limit=limit)

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

        labels_list = list(labels)
        existing = self._list_indexes()

        for label in labels_list:
            fields = _FULLTEXT_INDEX_FIELDS.get(label, _FALLBACK_FULLTEXT_FIELDS)
            for property_name in fields:
                if self._has_index(existing, label, property_name, "FULLTEXT"):
                    continue
                logger.info(
                    "Creating FalkorDB full-text index for %s.%s",
                    label,
                    property_name,
                )
                self._query(
                    f"CREATE FULLTEXT INDEX FOR (n:{label}) ON (n.{property_name})"
                )
                existing = self._list_indexes()

        if not vector_dims:
            return

        for label in labels_list:
            dimension = vector_dims.get(label)
            if dimension is None:
                continue
            known_dimension = self._vector_index_dimensions.get(label)
            if known_dimension is not None and known_dimension != dimension:
                raise ValueError(
                    "Embedding dimension mismatch for "
                    f"{label}: expected {known_dimension}, got {dimension}."
                )
            if self._has_index(existing, label, "embedding", "VECTOR"):
                self._vector_index_dimensions[label] = dimension
                continue

            logger.info(
                "Creating FalkorDB vector index for %s.embedding with dimension %s",
                label,
                dimension,
            )
            self._query(
                "CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
                "OPTIONS {{dimension:{dimension}, similarityFunction:'{similarity}', "
                "M:{m}, efConstruction:{ef_construction}, efRuntime:{ef_runtime}}}".format(
                    label=label,
                    dimension=dimension,
                    similarity=self.vector_similarity_function,
                    m=self.vector_index_m,
                    ef_construction=self.vector_index_ef_construction,
                    ef_runtime=self.vector_index_ef_runtime,
                )
            )
            self._vector_index_dimensions[label] = dimension
            existing = self._list_indexes()

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
            Flat list of hits across the requested labels.
        """

        return_expression = f"{self._node_return_expression()} LIMIT {int(limit)}"
        hits: list[NodeHit] = []
        for label in labels:
            result = self.query_fulltext_nodes(
                label,
                query_text,
                return_expression=return_expression,
            )
            for row in result.result_set:
                node = self._node_from_row(row, system_label=label)
                hits.append(NodeHit(node=node, matched_on="fulltext"))
        return hits

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
            Pre-computed query embedding.
        limit : int, optional
            Maximum number of hits to return per label.

        Returns
        -------
        list[NodeHit]
            Flat list of hits across the requested labels.
        """

        return_expression = f"{self._node_return_expression()}, score"
        hits: list[NodeHit] = []
        for label in labels:
            result = self.query_similar_nodes(
                label,
                query_embedding,
                limit,
                return_expression=return_expression,
            )
            for row in result.result_set:
                score_value = row[len(_NODE_COLUMNS)]
                node = self._node_from_row(row, system_label=label)
                hits.append(
                    NodeHit(
                        node=node,
                        score=float(score_value),
                        matched_on="vector",
                    )
                )
        return hits

    async def neighbors(
        self,
        *,
        node_ids: Sequence[str],
        rel_types: Sequence[str] | None = None,
        depth: int = 1,
    ) -> list[Node]:
        """Fetch distinct outgoing neighbors of the given seed nodes.

        Parameters
        ----------
        node_ids : Sequence[str]
            Seed node identifiers.
        rel_types : Sequence[str] | None, optional
            Restrict traversal to these relationship types. ``None`` follows
            any relationship.
        depth : int, optional
            Maximum traversal depth. Defaults to one hop.

        Returns
        -------
        list[Node]
            Distinct neighbor nodes, excluding the seeds themselves.
        """

        if depth < 1:
            raise ValueError("neighbors depth must be at least 1.")
        if not node_ids:
            return []

        if rel_types:
            for rel_type in rel_types:
                if not _REL_TYPE_PATTERN.match(rel_type):
                    raise ValueError(
                        f"Invalid relationship type for neighbors traversal: {rel_type!r}"
                    )
            rel_pattern = f"[:{'|'.join(rel_types)}*1..{int(depth)}]"
        else:
            rel_pattern = f"[*1..{int(depth)}]"

        query = (
            f"MATCH (seed)-{rel_pattern}->(neighbor) "
            f"WHERE seed.id IN $ids AND NOT neighbor.id IN $ids "
            f"WITH DISTINCT neighbor "
            f"RETURN {self._node_return_expression(variable='neighbor')}, "
            f"labels(neighbor) AS node_labels"
        )
        result = self.ro_query(query, {"ids": list(node_ids)})

        nodes: list[Node] = []
        seen_ids: set[str] = set()
        for row in result.result_set:
            cypher_labels = row[len(_NODE_COLUMNS)] or []
            system_label = self._canonical_system_label(cypher_labels)
            node = self._node_from_row(row, system_label=system_label)
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            nodes.append(node)
        return nodes

    def ro_query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a read-only query.

        Parameters
        ----------
        query : str
            Cypher query to execute.
        params : dict[str, Any] | None, optional
            Query parameters.

        Returns
        -------
        Any
            Backend-native query result.
        """

        return self._graph.ro_query(query, params or {})

    def explain(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Explain a Cypher query plan.

        Parameters
        ----------
        query : str
            Cypher query to explain.
        params : dict[str, Any] | None, optional
            Query parameters.

        Returns
        -------
        Any
            Backend-native explain result.
        """

        return self._graph.explain(query, params or {})

    def query_fulltext_nodes(
        self,
        label: str,
        search_term: str,
        *,
        return_expression: str = "node",
    ) -> Any:
        """Run a full-text node query.

        Parameters
        ----------
        label : str
            Node label targeted by the full-text index.
        search_term : str
            Full-text search string.
        return_expression : str, optional
            Cypher expression returned for each matched node.

        Returns
        -------
        Any
            Backend-native query result.
        """

        label_literal = self._serialize_cypher_string(label)
        term_literal = self._serialize_cypher_string(search_term)
        return self.ro_query(
            "CALL db.idx.fulltext.queryNodes({label}, {term}) "
            "YIELD node RETURN {return_expression}".format(
                label=label_literal,
                term=term_literal,
                return_expression=return_expression,
            )
        )

    def query_similar_nodes(
        self,
        label: str,
        embedding: list[float],
        k: int,
        *,
        attribute: str = "embedding",
        return_expression: str = "node, score",
        order_by: str | None = "score ASC",
    ) -> Any:
        """Run a vector similarity search against node embeddings.

        Parameters
        ----------
        label : str
            Node label targeted by the vector index.
        embedding : list[float]
            Query embedding vector.
        k : int
            Maximum number of nearest neighbors requested.
        attribute : str, optional
            Indexed vector property name.
        return_expression : str, optional
            Cypher expression returned from the procedure output.
        order_by : str | None, optional
            Optional ``ORDER BY`` clause content.

        Returns
        -------
        Any
            Backend-native query result.
        """

        vector_literal = self._require_embedding_literal(embedding)
        label_literal = self._serialize_cypher_string(label)
        attribute_literal = self._serialize_cypher_string(attribute)
        order_clause = f" ORDER BY {order_by}" if order_by else ""
        return self.ro_query(
            "CALL db.idx.vector.queryNodes({label}, {attribute}, {k}, {embedding}) "
            "YIELD node, score RETURN {return_expression}{order_clause}".format(
                label=label_literal,
                attribute=attribute_literal,
                k=int(k),
                embedding=vector_literal,
                return_expression=return_expression,
                order_clause=order_clause,
            )
        )

    def explain_vector_query(
        self,
        label: str,
        embedding: list[float],
        k: int,
        *,
        attribute: str = "embedding",
        return_expression: str = "node",
    ) -> Any:
        """Explain a vector node query plan.

        Parameters
        ----------
        label : str
            Node label targeted by the vector index.
        embedding : list[float]
            Query embedding vector.
        k : int
            Maximum number of nearest neighbors requested.
        attribute : str, optional
            Indexed vector property name.
        return_expression : str, optional
            Cypher expression returned from the procedure output.

        Returns
        -------
        Any
            Backend-native explain result.
        """

        vector_literal = self._require_embedding_literal(embedding)
        label_literal = self._serialize_cypher_string(label)
        attribute_literal = self._serialize_cypher_string(attribute)
        return self.explain(
            "CALL db.idx.vector.queryNodes({label}, {attribute}, {k}, {embedding}) "
            "YIELD node, score RETURN {return_expression}".format(
                label=label_literal,
                attribute=attribute_literal,
                k=int(k),
                embedding=vector_literal,
                return_expression=return_expression,
            )
        )

    def _query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a backend write query with optional parameters."""

        return self._graph.query(query, params or {})

    def _save_entity(self, chunk_id: str, node: Node) -> None:
        """Upsert one entity node and link it to its source chunk."""

        embedding_literal = self._serialize_embedding(node.embedding)
        self._query(
            build_entity_upsert_query(node.label, embedding_literal),
            {
                "chunk_id": chunk_id,
                "id": node.id,
                "label": node.label,
                "name": node.name,
                "semantic_key": node.semantic_key,
                "properties": self._serialize_mapping(node.properties),
            },
        )

    @staticmethod
    def _serialize_mapping(mapping: dict[str, str]) -> str:
        """Serialize backend-unsupported map properties to JSON strings."""

        return json.dumps(mapping, sort_keys=True)

    def _search_fulltext(self, query: str, *, limit: int) -> SearchResults:
        """Run grouped full-text search across searchable node families."""

        return {
            "__document__": self._deduplicate_hits(
                [
                    *self._fulltext_hits_for_documents(query, "name", limit),
                    *self._fulltext_hits_for_documents(query, "content", limit),
                ]
            )[:limit],
            "__chunk__": self._deduplicate_hits(
                [
                    *self._fulltext_hits_for_chunks(query, "name", limit),
                    *self._fulltext_hits_for_chunks(query, "content", limit),
                ]
            )[:limit],
            "__entity__": self._deduplicate_hits(
                self._fulltext_hits_for_entities(query, limit)
            )[:limit],
        }

    def _search_vector(
        self, query_embedding: list[float], *, limit: int
    ) -> SearchResults:
        """Run grouped vector search across searchable node families."""

        return {
            "__document__": self._vector_hits_for_documents(query_embedding, limit),
            "__chunk__": self._vector_hits_for_chunks(query_embedding, limit),
            "__entity__": self._vector_hits_for_entities(query_embedding, limit),
        }

    def _fulltext_hits_for_documents(
        self,
        query: str,
        attribute: Literal["name", "content"],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return document full-text hits for one indexed attribute."""

        result = self.query_fulltext_nodes(
            "__document__",
            query,
            return_expression=(
                "node.id AS id, node.name AS name, node.content AS content, "
                f"'{attribute}' AS matched_on LIMIT {int(limit)}"
            ),
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "content": row[2],
                "matched_on": row[3],
            }
            for row in result.result_set
        ]

    def _fulltext_hits_for_chunks(
        self,
        query: str,
        attribute: Literal["name", "content"],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return chunk full-text hits for one indexed attribute."""

        result = self.query_fulltext_nodes(
            "__chunk__",
            query,
            return_expression=(
                "node.id AS id, node.name AS name, node.content AS content, "
                "node.document_id AS document_id, "
                f"'{attribute}' AS matched_on LIMIT {int(limit)}"
            ),
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "content": row[2],
                "document_id": row[3],
                "matched_on": row[4],
            }
            for row in result.result_set
        ]

    def _fulltext_hits_for_entities(
        self, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Return entity full-text hits."""

        result = self.query_fulltext_nodes(
            "__entity__",
            query,
            return_expression=(
                "node.id AS id, node.name AS name, node.label AS label, "
                "node.semantic_key AS semantic_key, 'name' AS matched_on "
                f"LIMIT {int(limit)}"
            ),
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "label": row[2],
                "semantic_key": row[3],
                "matched_on": row[4],
            }
            for row in result.result_set
        ]

    def _vector_hits_for_documents(
        self, query_embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """Return document vector hits."""

        result = self.query_similar_nodes(
            "__document__",
            query_embedding,
            limit,
            return_expression="node.id, node.name, node.content, score",
        )
        return [
            {"id": row[0], "name": row[1], "content": row[2], "score": row[3]}
            for row in result.result_set
        ]

    def _vector_hits_for_chunks(
        self, query_embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """Return chunk vector hits."""

        result = self.query_similar_nodes(
            "__chunk__",
            query_embedding,
            limit,
            return_expression="node.id, node.name, node.content, node.document_id, score",
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "content": row[2],
                "document_id": row[3],
                "score": row[4],
            }
            for row in result.result_set
        ]

    def _vector_hits_for_entities(
        self, query_embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """Return entity vector hits."""

        result = self.query_similar_nodes(
            "__entity__",
            query_embedding,
            limit,
            return_expression="node.id, node.name, node.label, node.semantic_key, score",
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "label": row[2],
                "semantic_key": row[3],
                "score": row[4],
            }
            for row in result.result_set
        ]

    @staticmethod
    def _deduplicate_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate search hits by node id while preserving order."""

        deduplicated: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hit in hits:
            hit_id = hit.get("id")
            if hit_id in seen_ids:
                continue
            if isinstance(hit_id, str):
                seen_ids.add(hit_id)
            deduplicated.append(hit)
        return deduplicated

    def _ensure_fulltext_indexes(self) -> None:
        """Create required full-text indexes if they do not already exist."""

        existing = self._list_indexes()
        for label, properties in _FULLTEXT_INDEX_FIELDS.items():
            for property_name in properties:
                if self._has_index(existing, label, property_name, "FULLTEXT"):
                    continue
                logger.info(
                    "Creating FalkorDB full-text index for %s.%s",
                    label,
                    property_name,
                )
                self._query(
                    f"CREATE FULLTEXT INDEX FOR (n:{label}) ON (n.{property_name})"
                )
                existing = self._list_indexes()

    def _ensure_vector_indexes(self, embedding_dimensions: dict[str, int]) -> None:
        """Create required vector indexes for known embedding dimensions."""

        existing = self._list_indexes()
        for label in _VECTOR_INDEX_LABELS:
            dimension = embedding_dimensions.get(label)
            if dimension is None:
                continue
            known_dimension = self._vector_index_dimensions.get(label)
            if known_dimension is not None and known_dimension != dimension:
                raise ValueError(
                    "Embedding dimension mismatch for "
                    f"{label}: expected {known_dimension}, got {dimension}."
                )
            if self._has_index(existing, label, "embedding", "VECTOR"):
                self._vector_index_dimensions[label] = dimension
                continue

            logger.info(
                "Creating FalkorDB vector index for %s.embedding with dimension %s",
                label,
                dimension,
            )
            self._query(
                "CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
                "OPTIONS {{dimension:{dimension}, similarityFunction:'{similarity}', "
                "M:{m}, efConstruction:{ef_construction}, efRuntime:{ef_runtime}}}".format(
                    label=label,
                    dimension=dimension,
                    similarity=self.vector_similarity_function,
                    m=self.vector_index_m,
                    ef_construction=self.vector_index_ef_construction,
                    ef_runtime=self.vector_index_ef_runtime,
                )
            )
            self._vector_index_dimensions[label] = dimension
            existing = self._list_indexes()

    def _list_indexes(self) -> list[list[Any]]:
        """Return FalkorDB index metadata rows."""

        try:
            return self.query("CALL db.indexes()").result_set
        except ResponseError as exc:
            if "empty key" in str(exc).lower():
                return []
            raise

    @staticmethod
    def _has_index(
        indexes: list[list[Any]],
        label: str,
        property_name: str,
        expected_type: str,
    ) -> bool:
        """Return whether an index already exists for one label/property/type."""

        for row in indexes:
            if not row or row[0] != label:
                continue
            indexed_properties = row[1] if len(row) > 1 else []
            property_types = row[2] if len(row) > 2 else {}
            if property_name not in indexed_properties:
                continue
            types_for_property = property_types.get(property_name, [])
            if expected_type in types_for_property:
                return True
        return False

    @staticmethod
    def _collect_embedding_dimensions(nodes: list[Node]) -> dict[str, int]:
        """Collect vector dimensions from nodes that already carry embeddings."""

        dimensions: dict[str, int] = {}
        for node in nodes:
            if not node.embedding:
                continue
            dimension = len(node.embedding)
            label = node.label if node.label in _VECTOR_INDEX_LABELS else "__entity__"
            known_dimension = dimensions.get(label)
            if known_dimension is not None and known_dimension != dimension:
                raise ValueError(
                    "Embedding dimension mismatch within nodes for "
                    f"{label}: expected {known_dimension}, got {dimension}."
                )
            dimensions[label] = dimension
        return dimensions

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> str | None:
        """Convert an embedding to a FalkorDB vector literal."""

        if not embedding:
            return None
        normalized = [float(value) for value in embedding]
        if not all(math.isfinite(value) for value in normalized):
            raise ValueError("Embeddings must contain only finite float values.")
        return f"vecf32({normalized})"

    @classmethod
    def _require_embedding_literal(cls, embedding: list[float]) -> str:
        """Return a non-empty FalkorDB vector literal."""

        literal = cls._serialize_embedding(embedding)
        if literal is None:
            raise ValueError("Embedding cannot be empty for vector search queries.")
        return literal

    @staticmethod
    def _serialize_cypher_string(value: str) -> str:
        """Return a safely quoted Cypher string literal."""

        return json.dumps(value)

    @staticmethod
    def _node_return_expression(variable: str = "node") -> str:
        """Build a return expression fetching every field needed to rebuild a Node.

        Parameters
        ----------
        variable : str, optional
            Cypher variable name holding the node being projected.

        Returns
        -------
        str
            Comma-separated projection using ``_NODE_COLUMNS`` for ordering.
            Fields absent on a given node type yield ``NULL`` and are ignored
            when the corresponding subclass is constructed.
        """

        return ", ".join(f"{variable}.{column}" for column in _NODE_COLUMNS)

    @staticmethod
    def _canonical_system_label(cypher_labels: Sequence[str]) -> str:
        """Pick the ``__system__``-style label from a multi-label node.

        Entity nodes are stored with both ``__entity__`` and their ontology
        label (e.g. ``Person``). The system-style label (``__entity__``) is
        what :meth:`_node_from_row` dispatches on.
        """

        for label in cypher_labels:
            if label.startswith("__") and label.endswith("__"):
                return label
        return cypher_labels[0] if cypher_labels else ""

    @staticmethod
    def _node_from_row(row: Sequence[Any], *, system_label: str) -> Node:
        """Rebuild a :class:`Node` (or subclass) from a Cypher result row.

        Parameters
        ----------
        row : Sequence[Any]
            Row produced by a projection built with
            :meth:`_node_return_expression`. The first ``len(_NODE_COLUMNS)``
            positions correspond to ``_NODE_COLUMNS``.
        system_label : str
            ``__system__``-style label used to select the concrete subclass.
            Nodes whose label does not match a known system family are
            returned as the generic :class:`Node`.
        """

        (
            node_id,
            stored_label,
            semantic_key,
            name,
            properties_json,
            content,
            document_id,
            creation_date,
            metadata_json,
        ) = row[: len(_NODE_COLUMNS)]

        properties = (
            json.loads(properties_json) if isinstance(properties_json, str) else {}
        )
        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else {}

        if system_label == "__document__":
            return DocumentNode(
                id=node_id,
                semantic_key=semantic_key,
                name=name,
                properties=properties,
                content=content or "",
                metadata=metadata,
            )
        if system_label == "__chunk__":
            return ChunkNode(
                id=node_id,
                semantic_key=semantic_key,
                name=name,
                properties=properties,
                content=content or "",
                document_id=document_id or "",
                metadata=metadata,
            )
        if system_label == "__memory__":
            return MemoryNode(
                id=node_id,
                semantic_key=semantic_key,
                name=name,
                properties=properties,
                content=content or "",
                creation_date=creation_date or "",
                metadata=metadata,
            )
        return Node(
            id=node_id,
            label=stored_label or system_label,
            semantic_key=semantic_key,
            name=name,
            properties=properties,
        )
