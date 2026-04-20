"""Tests for graph extraction prompt semantics."""

from src.grawiki.graph.graph_prompts import KG_EXTRACTION_PROMPT


def test_prompt_defines_node_identity_fields() -> None:
    """Prompt should define node label and name semantics without ids."""

    assert "Node label: node TYPE or ontology category" in KG_EXTRACTION_PROMPT
    assert (
        "Node name: canonical human-readable name of that specific node instance"
        in KG_EXTRACTION_PROMPT
    )
    assert "temporary reference used by relationships" in KG_EXTRACTION_PROMPT


def test_prompt_requires_relationship_endpoints_to_use_node_names() -> None:
    """Prompt should require source and target to use node names."""

    assert "Relationship source and target" in KG_EXTRACTION_PROMPT
    assert "must match node names exactly, never labels" in KG_EXTRACTION_PROMPT


def test_prompt_reserves_system_labels_for_application_code() -> None:
    """Prompt should prevent the semantic extractor from emitting system nodes."""

    assert "__chunk__" in KG_EXTRACTION_PROMPT
    assert "__document__" in KG_EXTRACTION_PROMPT
    assert "__memory__" in KG_EXTRACTION_PROMPT
    assert "internal system nodes created by application code" in KG_EXTRACTION_PROMPT


def test_prompt_contains_good_and_bad_name_examples() -> None:
    """Prompt should include concrete examples that disambiguate the fields."""

    assert "label='Person', name='Marie Curie'" in KG_EXTRACTION_PROMPT
    assert "label='Marie Curie', name='Person'" in KG_EXTRACTION_PROMPT


def test_prompt_requires_unique_names_and_name_based_relationships() -> None:
    """Prompt should use names as temporary references within one extraction."""

    assert (
        "Node names must be unique within one extraction result" in KG_EXTRACTION_PROMPT
    )
    assert "Relationships should use node names as endpoints" in (KG_EXTRACTION_PROMPT)
