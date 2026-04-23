"""Tests for parametrized Cypher query builders."""

import pytest

from src.grawiki.db.cypher import (
    link_nodes_cypher,
    sanitize_cypher_identifier,
    upsert_node_cypher,
    upsert_rel_cypher,
)


def test_sanitize_cypher_identifier_normalizes_invalid_characters() -> None:
    """Raw labels should become Cypher-safe identifiers."""

    assert sanitize_cypher_identifier("Person Name") == "Person_Name"
    assert sanitize_cypher_identifier("works-with") == "works_with"
    assert sanitize_cypher_identifier("9Lives") == "_9Lives"


def test_sanitize_cypher_identifier_rejects_empty_output() -> None:
    """Identifiers that sanitize to nothing should fail fast."""

    with pytest.raises(ValueError, match="cannot be empty"):
        sanitize_cypher_identifier("---")


def test_upsert_node_cypher_document_shape() -> None:
    """Document node upsert should MERGE by id and SET the supplied fields."""

    query = upsert_node_cypher(
        ["__document__"],
        ["label", "name", "semantic_key", "content", "metadata"],
    )

    assert "MERGE (n:__document__ {id: $id})" in query
    assert "n.label = $label" in query
    assert "n.content = $content" in query
    assert "ON CREATE SET" not in query
    assert "RETURN n" in query


def test_upsert_node_cypher_entity_shape() -> None:
    """Entity node upsert should MERGE by semantic_key and carry ON CREATE SET id."""

    query = upsert_node_cypher(
        ["__entity__", "Research_Concept"],
        ["label", "name", "semantic_key", "properties"],
        merge_field="semantic_key",
        on_create_set_id=True,
    )

    assert ":__entity__:Research_Concept" in query
    assert "{semantic_key: $semantic_key}" in query
    assert "ON CREATE SET n.id = $id" in query
    assert "n.properties = $properties" in query
    assert "RETURN n" in query


def test_upsert_node_cypher_embedding_appended() -> None:
    """Embedding literal should be appended to the SET clause when provided."""

    query = upsert_node_cypher(
        ["__chunk__"],
        ["label", "name"],
        embedding_literal="vecf32([0.1, 0.2])",
    )

    assert "n.embedding = vecf32([0.1, 0.2])" in query


def test_upsert_rel_cypher_inlines_sanitized_type() -> None:
    """Relationship query should inline the sanitized type and match both endpoints by id."""

    query = upsert_rel_cypher("depends-on")

    assert "MERGE (s)-[r:depends_on]->(t)" in query
    assert "s:__entity__ {id: $source}" in query
    assert "t:__entity__ {id: $target}" in query
    assert "ON CREATE SET r.id = $id" in query
    assert "r.properties = $properties" in query


def test_link_nodes_cypher_has_chunk() -> None:
    """System __has_chunk__ link should match source by id and target by id."""

    query = link_nodes_cypher(
        "__has_chunk__",
        source_label="__document__",
        target_label="__chunk__",
    )

    assert "MATCH (s:__document__ {id: $source})" in query
    assert "MATCH (t:__chunk__ {id: $target})" in query
    assert "MERGE (s)-[:__has_chunk__]->(t)" in query
    assert "SET" not in query


def test_link_nodes_cypher_mentions() -> None:
    """System __mentions__ link should match target entity by semantic_key."""

    query = link_nodes_cypher(
        "__mentions__",
        source_label="__chunk__",
        target_label="__entity__",
        target_match_field="semantic_key",
    )

    assert "MATCH (s:__chunk__ {id: $source})" in query
    assert "MATCH (t:__entity__ {semantic_key: $target})" in query
    assert "MERGE (s)-[:__mentions__]->(t)" in query
    assert "SET" not in query
