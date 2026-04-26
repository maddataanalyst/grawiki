"""Dual-mode FalkorDB graph database adapter (Lite + Server)."""

from __future__ import annotations

import math
import json
import logging
from pathlib import Path
from typing import cast
from typing import Any, Iterable, Literal, Mapping, Sequence

from redis.exceptions import ResponseError

from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit, SearchResults
from grawiki.db.cypher import (
    link_nodes_cypher,
    sanitize_cypher_identifier,
    upsert_node_cypher,
    upsert_rel_by_id_cypher,
    upsert_rel_cypher,
)
from grawiki.graph.models import (
    ChunkNode,
    DocumentNode,
    MemoryNode,
    Node,
    Relationship,
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
    "labels",
    "semantic_key",
    "name",
    "properties",
    "content",
    "document_id",
    "creation_date",
    "metadata",
)


class FalkorGraphDB(GraphDB):
    """Graph adapter supporting both FalkorDBLite and full FalkorDB.

    Parameters
    ----------
    graph_name : str
        Logical graph name within the database.
    db_path : str | Path | None, optional
        Filesystem path for FalkorDBLite persistence. Use this **or**
        ``host``/``port``, not both.
    host : str | None, optional
        Hostname of a running FalkorDB server. When provided the adapter
        connects via TCP instead of using an embedded database.
    port : int, optional
        Port number for the FalkorDB server. Defaults to ``6379``.
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
        graph_name: str,
        *,
        db_path: str | Path | None = None,
        host: str | None = None,
        port: int = 6379,
        vector_similarity_function: Literal["cosine", "euclidean"] = "cosine",
        vector_index_m: int = 16,
        vector_index_ef_construction: int = 200,
        vector_index_ef_runtime: int = 10,
    ) -> None:
        if db_path is not None and host is not None:
            raise ValueError(
                "Provide either db_path (FalkorDBLite) or host (FalkorDB), not both."
            )
        if db_path is None and host is None:
            raise ValueError(
                "Either db_path (FalkorDBLite) or host (FalkorDB) is required."
            )

        self.graph_name = graph_name
        self.vector_similarity_function = vector_similarity_function
        self.vector_index_m = vector_index_m
        self.vector_index_ef_construction = vector_index_ef_construction
        self.vector_index_ef_runtime = vector_index_ef_runtime

        if host is not None:
            from falkordb import FalkorDB as FalkorDBClient

            self._db = FalkorDBClient(host=host, port=port)
            self.db_path = None
        else:
            from redislite.falkordb_client import (
                FalkorDB as FalkorDBClient,
            )

            self.db_path = str(Path(db_path))
            self._db = FalkorDBClient(self.db_path)

        self._graph = self._db.select_graph(graph_name)
        self._fulltext_indexes_ready = False
        self._vector_index_dimensions: dict[str, int] = {}

    def close(self) -> None:
        """Close the database connection.

        Notes
        -----
        FalkorDBLite runs an embedded Redis process underneath the adapter.
        Tests should call this explicitly during teardown instead of relying on
        redislite's ``atexit`` cleanup hook. Server-mode connections (via
        ``host``/``port``) may not require explicit close, but calling this
        method is safe in both modes.
        """

        if hasattr(self._db, "close"):
            self._db.close()

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

    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        """Upsert nodes, creating indexes on first use per label.

        Parameters
        ----------
        nodes : Sequence[Node]
            Nodes to create or update. Dispatches on concrete type and label.
        """

        labels: set[str] = set()
        dims: dict[str, int] = {}
        for node in nodes:
            if isinstance(node, (DocumentNode, ChunkNode, MemoryNode)):
                labels.update(node.labels)
            else:
                labels.add("__entity__")
            if node.embedding:
                key = (
                    next(iter(node.labels))
                    if next(iter(node.labels)) in _VECTOR_INDEX_LABELS
                    else "__entity__"
                )
                dims[key] = len(node.embedding)

        await self.ensure_indexes(labels=labels, vector_dims=dims or None)

        for node in nodes:
            self._upsert_single_node(node)

    async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
        """Upsert relationships, dispatching on label for match semantics.

        Parameters
        ----------
        rels : Sequence[Relationship]
            Relationships to create or update.

        Notes
        -----
        ``__has_chunk__`` matches both endpoints by node ``id``.
        ``__mentions__`` matches the entity target by ``id``.
        All other labels match entity endpoints by ``id``.
        Every relationship persists the same ``id``, ``label``, and
        serialized ``properties`` fields.
        """

        for rel in rels:
            params = {
                "source": rel.source,
                "target": rel.target,
                "id": rel.id,
                "label": rel.label,
                "properties": self._serialize_mapping(rel.properties),
            }
            if rel.label == "__has_chunk__":
                self._query(
                    link_nodes_cypher(
                        "__has_chunk__",
                        source_label="__document__",
                        target_label="__chunk__",
                    ),
                    params,
                )
            elif rel.label.startswith("__"):
                self._query(
                    upsert_rel_by_id_cypher(rel.label),
                    params,
                )
            else:
                self._query(
                    upsert_rel_cypher(rel.label),
                    params,
                )

    def _upsert_single_node(self, node: Node) -> None:
        """Dispatch one node upsert to the correct Cypher builder."""

        embedding_literal = self._serialize_embedding(node.embedding)
        if isinstance(node, DocumentNode):
            payload = node.model_dump(exclude={"labels"})
            payload["metadata"] = self._serialize_mapping(node.metadata)
            self._query(
                upsert_node_cypher(
                    ["__document__"],
                    ["name", "semantic_key", "content", "metadata"],
                    embedding_literal=embedding_literal,
                ),
                payload,
            )
        elif isinstance(node, ChunkNode):
            payload = node.model_dump(exclude={"labels"})
            payload["metadata"] = self._serialize_mapping(node.metadata)
            self._query(
                upsert_node_cypher(
                    ["__chunk__"],
                    [
                        "name",
                        "semantic_key",
                        "document_id",
                        "content",
                        "metadata",
                    ],
                    embedding_literal=embedding_literal,
                ),
                payload,
            )
        elif isinstance(node, MemoryNode):
            payload = node.model_dump(exclude={"labels"})
            payload["metadata"] = self._serialize_mapping(node.metadata)
            self._query(
                upsert_node_cypher(
                    ["__memory__"],
                    [
                        "name",
                        "semantic_key",
                        "content",
                        "creation_date",
                        "metadata",
                    ],
                    embedding_literal=embedding_literal,
                ),
                payload,
            )
        else:
            stored_labels = sorted(node.labels)
            self._query(
                upsert_node_cypher(
                    [
                        "__entity__",
                        *[sanitize_cypher_identifier(label) for label in stored_labels],
                    ],
                    ["name", "semantic_key", "properties", "labels"],
                    merge_field="semantic_key",
                    on_create_set_id=True,
                    embedding_literal=embedding_literal,
                ),
                {
                    "id": node.id,
                    "name": node.name,
                    "semantic_key": node.semantic_key,
                    "labels": stored_labels,
                    "properties": self._serialize_mapping(node.properties),
                },
            )

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
        existing_indexes = self._list_indexes()
        for label in labels:
            if not self._has_index(existing_indexes, label, "embedding", "VECTOR"):
                continue
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

    async def neighbor_relationships(
        self,
        *,
        node_ids: Sequence[str],
        limit_per_node: int = 5,
    ) -> dict[str, list[NeighborRelationship]]:
        """Fetch one-hop relationship context for each seed node.

        Parameters
        ----------
        node_ids : Sequence[str]
            Seed node identifiers.
        limit_per_node : int, optional
            Maximum number of relationship rows returned for each seed.

        Returns
        -------
        dict[str, list[NeighborRelationship]]
            Relationship context keyed by seed id.
        """

        if limit_per_node < 1:
            raise ValueError(
                "neighbor_relationships limit_per_node must be at least 1."
            )
        unique_ids = list(dict.fromkeys(node_ids))
        contexts: dict[str, list[NeighborRelationship]] = {
            node_id: [] for node_id in unique_ids
        }
        if not unique_ids:
            return contexts

        query = (
            "MATCH (source) WHERE source.id IN $ids "
            "MATCH (source)-[rel]-(neighbor) "
            "WITH source, rel, neighbor, labels(neighbor) AS node_labels "
            "ORDER BY source.id, neighbor.name, neighbor.id, rel.label "
            "WITH source, collect([rel.label, neighbor, node_labels])[..$limit] AS rows "
            "UNWIND rows AS row "
            "WITH source, row[0] AS rel_label, row[1] AS neighbor, row[2] AS node_labels "
            f"RETURN source.id AS source_id, source.name AS source_name, rel_label, {self._node_return_expression(variable='neighbor')}, node_labels"
        )
        result = self.ro_query(query, {"ids": unique_ids, "limit": int(limit_per_node)})

        for row in result.result_set:
            source_id = row[0]
            source_name = row[1] or ""
            relationship_label = row[2] or ""
            node_offset = 3
            node_row = row[node_offset : node_offset + len(_NODE_COLUMNS)]
            cypher_labels = row[node_offset + len(_NODE_COLUMNS)] or []
            system_label = self._canonical_system_label(cypher_labels)
            target = self._node_from_row(node_row, system_label=system_label)
            contexts.setdefault(source_id, []).append(
                NeighborRelationship(
                    source_id=source_id,
                    source_name=source_name,
                    relationship_label=relationship_label,
                    target=target,
                )
            )
        return contexts

    async def recall_subgraph(
        self,
        *,
        memory_ids: Sequence[str],
        hops: int = 1,
        limit_per_memory: int = 20,
    ) -> dict[str, list[NeighborRelationship]]:
        """Fetch a flattened k-hop recall subgraph for memory seeds."""

        if hops < 1:
            raise ValueError("recall_subgraph hops must be at least 1.")
        if limit_per_memory < 1:
            raise ValueError("recall_subgraph limit_per_memory must be at least 1.")

        unique_ids = list(dict.fromkeys(memory_ids))
        contexts: dict[str, list[NeighborRelationship]] = {
            memory_id: [] for memory_id in unique_ids
        }
        if not unique_ids:
            return contexts

        query = (
            "MATCH (seed:__memory__) WHERE seed.id IN $ids "
            f"MATCH path=(seed)-[*1..{int(hops)}]-(node) "
            "WHERE all(path_node IN nodes(path) "
            "WHERE path_node.id = seed.id OR NOT '__memory__' IN labels(path_node)) "
            "WITH seed, path, node "
            "ORDER BY seed.id, length(path), node.name, node.id "
            f"WITH seed, collect(DISTINCT path)[..{int(limit_per_memory)}] AS paths "
            "UNWIND paths AS path "
            "UNWIND relationships(path) AS rel "
            "WITH DISTINCT seed, startNode(rel) AS source, endNode(rel) AS target, "
            "type(rel) AS rel_label "
            f"RETURN seed.id AS seed_id, source.id AS source_id, source.name AS source_name, rel_label, {self._node_return_expression(variable='target')}, labels(target) AS node_labels"
        )
        result = self.ro_query(query, {"ids": unique_ids})

        for row in result.result_set:
            seed_id = row[0]
            source_id = row[1]
            source_name = row[2] or ""
            relationship_label = row[3] or ""
            node_offset = 4
            node_row = row[node_offset : node_offset + len(_NODE_COLUMNS)]
            cypher_labels = row[node_offset + len(_NODE_COLUMNS)] or []
            system_label = self._canonical_system_label(cypher_labels)
            target = self._node_from_row(node_row, system_label=system_label)
            contexts.setdefault(seed_id, []).append(
                NeighborRelationship(
                    source_id=source_id,
                    source_name=source_name,
                    relationship_label=relationship_label,
                    target=target,
                )
            )
        return contexts

    async def list_entities(self, *, include_embeddings: bool = False) -> list[Node]:
        """Return persisted entity nodes ordered by semantic key then id.

        Parameters
        ----------
        include_embeddings : bool, optional
            Whether to include entity embeddings in the result.

        Returns
        -------
        list[Node]
            Persisted entity nodes.
        """

        embedding_projection = ", n.embedding" if include_embeddings else ""
        result = self.ro_query(
            "MATCH (n:__entity__) "
            "RETURN n.id, n.labels, n.semantic_key, n.name, n.properties"
            f"{embedding_projection} ORDER BY n.semantic_key, n.id"
        )
        entities: list[Node] = []
        for row in result.result_set:
            properties_json = row[4]
            properties = (
                json.loads(properties_json) if isinstance(properties_json, str) else {}
            )
            embedding: list[float] = []
            if include_embeddings and len(row) > 5 and row[5] is not None:
                raw_embedding = cast(Sequence[Any], row[5])
                embedding = [float(value) for value in raw_embedding]
            entities.append(
                Node(
                    id=row[0],
                    labels=frozenset(self._deserialize_labels(row[1])),
                    semantic_key=row[2],
                    name=row[3],
                    properties=properties,
                    embedding=embedding,
                )
            )
        return entities

    async def entity_relationship_counts(
        self, node_ids: Sequence[str]
    ) -> dict[str, int]:
        """Return total incident relationship counts for entity ids."""

        unique_ids = list(dict.fromkeys(node_ids))
        counts = {node_id: 0 for node_id in unique_ids}
        if not unique_ids:
            return counts

        result = self.ro_query(
            "MATCH (n:__entity__) WHERE n.id IN $ids "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN n.id, count(r)",
            {"ids": unique_ids},
        )
        for node_id, count in result.result_set:
            counts[str(node_id)] = int(count)
        return counts

    async def merge_entity_nodes(
        self,
        *,
        master: Node,
        duplicate_ids: Sequence[str],
    ) -> None:
        """Merge duplicate entity nodes into ``master``."""

        dup_ids = list(dict.fromkeys(duplicate_ids))
        if master.id in dup_ids:
            raise ValueError("master.id must not appear in duplicate_ids")
        if not dup_ids:
            return

        touched_ids = [master.id, *dup_ids]
        dup_set = set(dup_ids)
        self._update_entity_node(master)

        incident_rows = self.ro_query(
            "MATCH (s)-[r]->(t) "
            "WHERE s.id IN $ids OR t.id IN $ids "
            "RETURN s.id, t.id, type(r), r.id, r.properties",
            {"ids": touched_ids},
        ).result_set

        canonical: dict[tuple[str, str, str], dict[str, Any]] = {}
        rel_ids_to_delete: list[str] = []
        sorted_rows = sorted(
            incident_rows,
            key=lambda row: (
                int((row[0] in dup_set) or (row[1] in dup_set)),
                str(row[2]),
                str(row[0]),
                str(row[1]),
                str(row[3]),
            ),
        )
        for source_id, target_id, rel_type, rel_id, raw_properties in sorted_rows:
            rel_ids_to_delete.append(str(rel_id))
            new_source = master.id if source_id in dup_set else source_id
            new_target = master.id if target_id in dup_set else target_id
            if new_source == new_target:
                continue

            key = (str(new_source), str(rel_type), str(new_target))
            properties = (
                json.loads(raw_properties) if isinstance(raw_properties, str) else {}
            )
            existing = canonical.get(key)
            if existing is None:
                canonical[key] = {
                    "id": str(rel_id),
                    "source": str(new_source),
                    "target": str(new_target),
                    "label": str(rel_type),
                    "properties": dict(properties),
                }
                continue

            existing["properties"] = self._merge_string_mappings(
                existing["properties"],
                properties,
            )

        if rel_ids_to_delete:
            self._query(
                "MATCH ()-[r]->() WHERE r.id IN $ids DELETE r",
                {"ids": rel_ids_to_delete},
            )

        for rel in canonical.values():
            self._query(
                upsert_rel_by_id_cypher(rel["label"]),
                {
                    "source": rel["source"],
                    "target": rel["target"],
                    "id": rel["id"],
                    "label": rel["label"],
                    "properties": self._serialize_mapping(rel["properties"]),
                },
            )

        self._query(
            "MATCH (n:__entity__) WHERE n.id IN $ids DELETE n",
            {"ids": dup_ids},
        )

    async def delete_memory(self, memory_id: str) -> None:
        """Delete one memory and prune directly-mentioned orphan entities."""

        candidate_rows = self.ro_query(
            "MATCH (:__memory__ {id: $memory_id})-[:__mentions__]->(e:__entity__) "
            "RETURN DISTINCT e.id",
            {"memory_id": memory_id},
        ).result_set
        candidate_ids = [str(row[0]) for row in candidate_rows]

        self._query(
            "MATCH (m:__memory__ {id: $memory_id}) DETACH DELETE m",
            {"memory_id": memory_id},
        )

        if not candidate_ids:
            return

        self._query(
            "MATCH (e:__entity__) WHERE e.id IN $ids "
            "OPTIONAL MATCH (e)-[r]-() "
            "WITH e, count(r) AS rel_count "
            "WHERE rel_count = 0 DELETE e",
            {"ids": candidate_ids},
        )

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
        order_by: str | None = None,
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
            Optional ``ORDER BY`` clause content. When omitted, cosine indexes
            are normalized to similarity and sorted by descending score, while
            euclidean indexes keep their backend distance ordering.

        Returns
        -------
        Any
            Backend-native query result.
        """

        vector_literal = self._require_embedding_literal(embedding)
        label_literal = self._serialize_cypher_string(label)
        attribute_literal = self._serialize_cypher_string(attribute)
        if self.vector_similarity_function == "cosine":
            effective_order_by = "score DESC" if order_by is None else order_by
            order_clause = (
                f" ORDER BY {effective_order_by}" if effective_order_by else ""
            )
            return self.ro_query(
                "CALL db.idx.vector.queryNodes({label}, {attribute}, {k}, {embedding}) "
                "YIELD node, score WITH node, (1 - score) AS score "
                "RETURN {return_expression}{order_clause}".format(
                    label=label_literal,
                    attribute=attribute_literal,
                    k=int(k),
                    embedding=vector_literal,
                    return_expression=return_expression,
                    order_clause=order_clause,
                )
            )

        effective_order_by = "score ASC" if order_by is None else order_by
        order_clause = f" ORDER BY {effective_order_by}" if effective_order_by else ""
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

    def _update_entity_node(self, node: Node) -> None:
        """Update an existing entity node by id and add any missing labels."""

        sorted_labels = sorted(node.labels)
        embedding_literal = self._serialize_embedding(node.embedding)
        label_clause = ""
        if sorted_labels:
            cypher_labels = ":".join(
                sanitize_cypher_identifier(label) for label in sorted_labels
            )
            label_clause = f"\nSET n:{cypher_labels}"
        embedding_clause = ""
        if embedding_literal is not None:
            embedding_clause = f",\n    n.embedding = {embedding_literal}"
        self._query(
            (
                "MATCH (n:__entity__ {id: $id})\n"
                "SET n.name = $name,\n"
                "    n.semantic_key = $semantic_key,\n"
                "    n.properties = $properties,\n"
                f"    n.labels = $labels{embedding_clause}"
                f"{label_clause}\n"
                "RETURN n"
            ),
            {
                "id": node.id,
                "name": node.name,
                "semantic_key": node.semantic_key,
                "labels": sorted_labels,
                "properties": self._serialize_mapping(node.properties),
            },
        )

    @staticmethod
    def _merge_string_mappings(
        left: Mapping[str, str],
        right: Mapping[str, str],
    ) -> dict[str, str]:
        """Merge two string mappings, preferring existing values in ``left``."""

        merged = dict(left)
        for key, value in right.items():
            merged.setdefault(str(key), str(value))
        return merged

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
                "node.id AS id, node.name AS name, node.labels AS labels, "
                "node.semantic_key AS semantic_key, 'name' AS matched_on "
                f"LIMIT {int(limit)}"
            ),
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "labels": self._deserialize_labels(row[2]),
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
            return_expression="node.id, node.name, node.labels, node.semantic_key, score",
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "labels": self._deserialize_labels(row[2]),
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
            label = (
                next(iter(node.labels))
                if next(iter(node.labels)) in _VECTOR_INDEX_LABELS
                else "__entity__"
            )
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
    def _deserialize_labels(raw_labels: Any) -> list[str]:
        """Return a sorted list of ontology labels from a Falkor row value."""

        if raw_labels is None:
            return []
        if isinstance(raw_labels, str):
            return [raw_labels]
        if isinstance(raw_labels, Sequence):
            return sorted(str(value) for value in raw_labels)
        return [str(raw_labels)]

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
            stored_labels,
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
                labels=frozenset({DocumentNode.system_label}),
                content=content or "",
                metadata=metadata,
            )
        if system_label == "__chunk__":
            return ChunkNode(
                id=node_id,
                semantic_key=semantic_key,
                name=name,
                properties=properties,
                labels=frozenset({ChunkNode.system_label}),
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
                labels=frozenset({MemoryNode.system_label}),
                content=content or "",
                creation_date=creation_date or "",
                metadata=metadata,
            )
        return Node(
            id=node_id,
            labels=frozenset(FalkorGraphDB._deserialize_labels(stored_labels)),
            semantic_key=semantic_key,
            name=name,
            properties=properties,
        )
