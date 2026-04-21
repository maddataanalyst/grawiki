"""Tests for Cypher query helpers."""

import pytest

from src.grawiki.db.cypher_queries import (
    build_entity_upsert_query,
    build_relationship_upsert_query,
    sanitize_cypher_identifier,
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


def test_build_entity_upsert_query_inlines_sanitized_label() -> None:
    """Entity queries should inline only the sanitized ontology label."""

    query = build_entity_upsert_query("Research Concept")

    assert ":__entity__:Research_Concept" in query
    assert "{semantic_key: $semantic_key}" in query
    assert "MERGE (c)-[:__mentions__]->(e)" in query


def test_build_relationship_upsert_query_inlines_sanitized_type() -> None:
    """Relationship queries should inline only the sanitized type."""

    query = build_relationship_upsert_query("depends-on")

    assert "MERGE (source)-[r:depends_on]->(target)" in query
    assert "source_semantic_key" in query
    assert "target_semantic_key" in query
