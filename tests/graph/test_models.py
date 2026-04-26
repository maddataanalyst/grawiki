"""Tests for graph model semantics."""

from grawiki.core.commons import Chunk, Document
from grawiki.graph.models import ChunkNode, DocumentNode


def test_document_node_from_document_uses_title_as_name() -> None:
    """Document nodes should keep document identity semantics."""

    document = Document(
        id="document_computing_machinery",
        title="Computing Machinery and Intelligence",
        content="Can machines think?",
        metadata={"author": "Alan Turing"},
    )

    node = DocumentNode.from_document(document)

    assert node.id == "document_computing_machinery"
    assert node.labels == frozenset({DocumentNode.system_label})
    assert node.name == "Computing Machinery and Intelligence"
    assert node.content == "Can machines think?"
    assert node.metadata == {"author": "Alan Turing"}


def test_chunk_node_from_chunk_uses_deterministic_name() -> None:
    """Chunk nodes should inherit node semantics and assign a stable name."""

    chunk = Chunk(
        id="chunk_001",
        document_id="document_computing_machinery",
        content="Turing discusses machine intelligence.",
        metadata={"order": "1"},
    )

    node = ChunkNode.from_chunk(chunk)

    assert node.id == "chunk_001"
    assert node.labels == frozenset({ChunkNode.system_label})
    assert node.name == "Chunk chunk_001"
    assert node.document_id == "document_computing_machinery"
    assert node.metadata == {"order": "1"}
