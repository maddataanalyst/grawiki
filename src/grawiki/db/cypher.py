"""Label-parameterized Cypher query builders shared by graph database adapters."""

from __future__ import annotations

import re


_IDENTIFIER_PATTERN = re.compile(r"[^0-9A-Za-z_]")


def _build_embedding_assignment(alias: str, embedding_literal: str | None) -> str:
    if embedding_literal is None:
        return ""
    return f",\n    {alias}.embedding = {embedding_literal}"


def _build_relationship_set_clause(alias: str = "r") -> str:
    return (
        f"ON CREATE SET {alias}.id = $id\n"
        f"SET {alias}.label = $label,\n"
        f"    {alias}.properties = $properties"
    )


def sanitize_cypher_identifier(identifier: str) -> str:
    """Normalize a user-provided label or relation type for Cypher.

    Parameters
    ----------
    identifier : str
        Raw label or relationship type (typically from LLM output).

    Returns
    -------
    str
        Safe Cypher identifier containing only letters, digits, and
        underscores, never starting with a digit.

    Raises
    ------
    ValueError
        Raised when the identifier becomes empty after sanitization.

    Notes
    -----
    Do not call this on reserved system types such as ``__mentions__``;
    it strips leading/trailing underscores and mangles their meaning.
    Use regex validation for system-controlled identifiers instead.
    """

    sanitized = _IDENTIFIER_PATTERN.sub("_", identifier.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        raise ValueError("Cypher identifier cannot be empty.")
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def upsert_node_cypher(
    labels: list[str],
    set_fields: list[str],
    *,
    merge_field: str = "id",
    on_create_set_id: bool = False,
    embedding_literal: str | None = None,
) -> str:
    """Build a node upsert query for any label combination.

    Parameters
    ----------
    labels : list[str]
        Cypher node labels. Caller is responsible for sanitizing ontology
        labels (e.g. via :func:`sanitize_cypher_identifier`) before passing them
        here. System labels (``__document__``, ``__entity__``, etc.) are safe
        to pass directly.
    set_fields : list[str]
        Property names to assign in the SET clause, each bound to a
        query parameter of the same name.
    merge_field : str, optional
        Property used in the MERGE match predicate. Defaults to ``"id"``.
    on_create_set_id : bool, optional
        When ``True`` adds ``ON CREATE SET n.id = $id`` after the MERGE.
        Used for entity nodes that merge by ``semantic_key`` but need a
        stable UUID assigned on first creation.
    embedding_literal : str | None, optional
        Pre-rendered backend-specific vector literal (e.g. ``vecf32([...])``)
        to append to the SET clause. Pass ``None`` when no embedding is stored.

    Returns
    -------
    str
        Cypher query string.
    """

    labels_str = ":".join(labels)
    set_clause = ",\n    ".join(f"n.{f} = ${f}" for f in set_fields)
    on_create = "\nON CREATE SET n.id = $id" if on_create_set_id else ""
    embedding = _build_embedding_assignment("n", embedding_literal)

    return f"""
MERGE (n:{labels_str} {{{merge_field}: ${merge_field}}}){on_create}
SET {set_clause}{embedding}
RETURN n
""".strip()


def upsert_rel_cypher(rel_type: str) -> str:
    """Build an entity-to-entity relationship upsert matched by node id.

    Both endpoints are ``__entity__`` nodes matched on their ``id`` field.
    The relationship type is sanitized via :func:`sanitize_cypher_identifier`
    to guard against unexpected characters in LLM-extracted labels.

    Parameters
    ----------
    rel_type : str
        Raw relationship type. Sanitized internally; may contain spaces or
        punctuation that the LLM produced.

    Returns
    -------
    str
        Cypher query string.
    """

    safe_type = sanitize_cypher_identifier(rel_type)
    rel_set_clause = _build_relationship_set_clause()
    return f"""
MATCH (s:__entity__ {{id: $source}})
MATCH (t:__entity__ {{id: $target}})
MERGE (s)-[r:{safe_type}]->(t)
{rel_set_clause}
RETURN r
""".strip()


def upsert_rel_by_id_cypher(rel_type: str) -> str:
    """Build a relationship upsert matched by endpoint ids only."""

    safe_type = (
        rel_type if rel_type.startswith("__") else sanitize_cypher_identifier(rel_type)
    )
    rel_set_clause = _build_relationship_set_clause()
    return f"""
MATCH (s {{id: $source}})
MATCH (t {{id: $target}})
MERGE (s)-[r:{safe_type}]->(t)
{rel_set_clause}
RETURN r
""".strip()


def link_nodes_cypher(
    rel_type: str,
    *,
    source_label: str,
    source_match_field: str = "id",
    target_label: str,
    target_match_field: str = "id",
) -> str:
    """Build a system-relationship upsert query with stored metadata.

    This builder is intended for system-controlled relationship types such as
    ``__has_chunk__`` and ``__mentions__``. Unlike :func:`upsert_rel_cypher`, it
    does **not** sanitize the relationship type because reserved system types
    such as ``__mentions__`` must be preserved exactly. Callers must validate
    system relationship types before passing them here.

    Parameters
    ----------
    rel_type : str
        System relationship type. Must already be a valid Cypher identifier.
    source_label : str
        Cypher label for the source node used in the MATCH clause.
    source_match_field : str, optional
        Property used to match the source node. Defaults to ``"id"``.
    target_label : str
        Cypher label for the target node used in the MATCH clause.
    target_match_field : str, optional
        Property used to match the target node. Defaults to ``"id"``.

    Returns
    -------
    str
        Cypher query string matching both endpoints and persisting the
        relationship ``id``, ``label``, and ``properties`` fields.
    """

    rel_set_clause = _build_relationship_set_clause()
    return f"""
MATCH (s:{source_label} {{{source_match_field}: $source}})
MATCH (t:{target_label} {{{target_match_field}: $target}})
MERGE (s)-[r:{rel_type}]->(t)
{rel_set_clause}
RETURN r
""".strip()
