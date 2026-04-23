"""Cypher query helpers shared by graph database adapters."""

from __future__ import annotations

import re


_IDENTIFIER_PATTERN = re.compile(r"[^0-9A-Za-z_]")


def _build_embedding_assignment(alias: str, embedding_literal: str | None) -> str:
    """Build an optional Cypher assignment for a vector embedding.

    Parameters
    ----------
    alias : str
        Variable name used for the node in the enclosing Cypher query.
    embedding_literal : str | None
        FalkorDB vector literal such as ``vecf32([0.1, 0.2])``.

    Returns
    -------
    str
        Cypher snippet that can be appended to a ``SET`` clause.
    """

    if embedding_literal is None:
        return ""
    return f",\n    {alias}.embedding = {embedding_literal}"


def build_document_upsert_query(embedding_literal: str | None = None) -> str:
    """Build a document upsert query.

    Parameters
    ----------
    embedding_literal : str | None, optional
        FalkorDB vector literal such as ``vecf32([0.1, 0.2])``.

    Returns
    -------
    str
        Cypher query string for document upsert.
    """

    return f"""
MERGE (d:__document__ {{id: $id}})
SET d.label = $label,
    d.name = $name,
    d.semantic_key = $semantic_key,
    d.content = $content,
    d.metadata = $metadata{_build_embedding_assignment("d", embedding_literal)}
RETURN d
""".strip()


def build_chunk_upsert_query(embedding_literal: str | None = None) -> str:
    """Build a chunk upsert query.

    Parameters
    ----------
    embedding_literal : str | None, optional
        FalkorDB vector literal such as ``vecf32([0.1, 0.2])``.

    Returns
    -------
    str
        Cypher query string for chunk upsert and document linkage.
    """

    return f"""
MATCH (d:__document__ {{id: $document_id}})
MERGE (c:__chunk__ {{id: $id}})
SET c.label = $label,
    c.name = $name,
    c.semantic_key = $semantic_key,
    c.document_id = $document_id,
    c.content = $content,
    c.metadata = $metadata{_build_embedding_assignment("c", embedding_literal)}
MERGE (d)-[:__has_chunk__]->(c)
RETURN c
""".strip()


def sanitize_cypher_identifier(identifier: str) -> str:
    """Normalize a user-provided label or relation type for Cypher.

    Parameters
    ----------
    identifier : str
        Raw label or relationship type.

    Returns
    -------
    str
        Safe Cypher identifier containing only letters, digits, and
        underscores, and never starting with a digit.

    Raises
    ------
    ValueError
        Raised when the identifier becomes empty after sanitization.
    """

    sanitized = _IDENTIFIER_PATTERN.sub("_", identifier.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        raise ValueError("Cypher identifier cannot be empty.")
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def build_entity_upsert_query(label: str, embedding_literal: str | None = None) -> str:
    """Build an entity upsert query with a backend-safe label.

    Parameters
    ----------
    label : str
        Ontology label for the entity node.

    Returns
    -------
    str
        Cypher query string for entity upsert and chunk mention creation.
    """

    safe_label = sanitize_cypher_identifier(label)
    return f"""
MATCH (c:__chunk__ {{id: $chunk_id}})
MERGE (e:__entity__:{safe_label} {{semantic_key: $semantic_key}})
ON CREATE SET e.id = $id
SET e.label = $label,
    e.name = $name,
    e.semantic_key = $semantic_key,
    e.properties = $properties{_build_embedding_assignment("e", embedding_literal)}
MERGE (c)-[:__mentions__]->(e)
RETURN e
""".strip()


def build_chunk_node_upsert_query(embedding_literal: str | None = None) -> str:
    """Build a chunk node-only upsert query (no document linkage).

    Parameters
    ----------
    embedding_literal : str | None, optional
        FalkorDB vector literal such as ``vecf32([0.1, 0.2])``.

    Returns
    -------
    str
        Cypher query string for chunk node upsert without doc→chunk edge.
    """

    return f"""
MERGE (c:__chunk__ {{id: $id}})
SET c.label = $label,
    c.name = $name,
    c.semantic_key = $semantic_key,
    c.document_id = $document_id,
    c.content = $content,
    c.metadata = $metadata{_build_embedding_assignment("c", embedding_literal)}
RETURN c
""".strip()


def build_memory_node_upsert_query(embedding_literal: str | None = None) -> str:
    """Build a memory node upsert query.

    Parameters
    ----------
    embedding_literal : str | None, optional
        FalkorDB vector literal such as ``vecf32([0.1, 0.2])``.

    Returns
    -------
    str
        Cypher query string for memory node upsert.
    """

    return f"""
MERGE (m:__memory__ {{id: $id}})
SET m.label = $label,
    m.name = $name,
    m.semantic_key = $semantic_key,
    m.content = $content,
    m.creation_date = $creation_date,
    m.metadata = $metadata{_build_embedding_assignment("m", embedding_literal)}
RETURN m
""".strip()


def build_entity_node_upsert_query(
    label: str, embedding_literal: str | None = None
) -> str:
    """Build an entity node-only upsert query (no chunk linkage).

    Parameters
    ----------
    label : str
        Ontology label for the entity node.
    embedding_literal : str | None, optional
        FalkorDB vector literal such as ``vecf32([0.1, 0.2])``.

    Returns
    -------
    str
        Cypher query string for entity node upsert without chunk→entity edge.
    """

    safe_label = sanitize_cypher_identifier(label)
    return f"""
MERGE (e:__entity__:{safe_label} {{semantic_key: $semantic_key}})
ON CREATE SET e.id = $id
SET e.label = $label,
    e.name = $name,
    e.semantic_key = $semantic_key,
    e.properties = $properties{_build_embedding_assignment("e", embedding_literal)}
RETURN e
""".strip()


def build_has_chunk_rel_query() -> str:
    """Build a query to create a document-to-chunk relationship.

    Returns
    -------
    str
        Cypher query string matching by id and merging the edge.
    """

    return """
MATCH (s:__document__ {id: $source})
MATCH (t:__chunk__ {id: $target})
MERGE (s)-[:__has_chunk__]->(t)
""".strip()


def build_mentions_rel_query() -> str:
    """Build a query to create a chunk-to-entity relationship.

    The entity is matched by ``semantic_key`` (via ``$target``) rather than
    ``id`` so that entity merging is handled transparently.

    Returns
    -------
    str
        Cypher query string matching chunk by id and entity by semantic_key.
    """

    return """
MATCH (s:__chunk__ {id: $source})
MATCH (t:__entity__ {semantic_key: $target})
MERGE (s)-[:__mentions__]->(t)
""".strip()


def build_entity_rel_upsert_query(label: str) -> str:
    """Build an entity-to-entity relationship upsert query matched by node id.

    Parameters
    ----------
    label : str
        Raw relationship type.

    Returns
    -------
    str
        Cypher query string for entity-to-entity relationship upsert.
    """

    safe_label = sanitize_cypher_identifier(label)
    return f"""
MATCH (s:__entity__ {{id: $source}})
MATCH (t:__entity__ {{id: $target}})
MERGE (s)-[r:{safe_label}]->(t)
ON CREATE SET r.id = $id
SET r.label = $label,
    r.properties = $properties
RETURN r
""".strip()


def build_relationship_upsert_query(label: str) -> str:
    """Build a relationship upsert query with a backend-safe type.

    Parameters
    ----------
    label : str
        Raw relationship type.

    Returns
    -------
    str
        Cypher query string for relationship upsert.
    """

    safe_label = sanitize_cypher_identifier(label)
    return f"""
MATCH (source:__entity__ {{semantic_key: $source_semantic_key}})
MATCH (target:__entity__ {{semantic_key: $target_semantic_key}})
MERGE (source)-[r:{safe_label}]->(target)
ON CREATE SET r.id = $id
SET r.label = $label,
    r.properties = $properties
RETURN r
""".strip()
