"""Pydantic models used to represent the knowledge graph."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from grawiki.core.commons import Chunk, Document


class GraphModel(BaseModel):
    """Base class for graph models.

    Notes
    -----
    Graph outputs are kept strict so extractors cannot silently add
    unsupported fields.
    """

    model_config = ConfigDict(extra="forbid")


class Node(GraphModel):
    """Graph node representing one concrete entity or concept.

    Field-level descriptions document the persisted schema in detail. The
    `labels` field supports multi-label entities, while `semantic_key` provides
    a stable deduplication-oriented identifier derived from the node type and
    name.
    """

    id: str = Field(
        description=(
            "Machine-generated identifier unique within the graph. "
            "This value is assigned by application code, typically as a UUID."
        )
    )
    labels: frozenset[str] = Field(
        description=(
            "Ontology labels or categories assigned to the node, such as "
            "{'Person'} or {'Concept', 'Theory'}."
        )
    )
    semantic_key: str = Field(
        description=(
            "A key constructed as the concatenation of node label and shortened name, "
            "used to identify nodes with the same label and name across the graph. "
            "Should be very short and brief. Example: person_alan-turing, code-snippet_llm-implementation."
        )
    )
    name: str = Field(
        description=(
            "Human-readable canonical name of the specific node instance, "
            "such as 'Alan Turing'."
        )
    )
    properties: dict[str, str] = Field(
        default_factory=dict,
        description="Short factual properties associated with the node.",
    )
    embedding: list[float] = Field(
        default_factory=list,
        description=(
            "Optional vector embedding associated with the node, for example for retrieval purposes. "
            "The specific embedding model and dimensions are determined by application code."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_input_labels(cls, data: object) -> object:
        """Normalize legacy ``label`` input into the ``labels`` field."""

        if not isinstance(data, Mapping):
            return data

        normalized = dict(data)
        raw_label = normalized.pop("label", None)
        raw_labels = normalized.get("labels")

        if raw_labels is None:
            if raw_label is not None:
                normalized["labels"] = [raw_label]
            return normalized

        if isinstance(raw_labels, str):
            values = {raw_labels}
        else:
            values = {str(value) for value in raw_labels}
        if raw_label is not None:
            values.add(str(raw_label))
        normalized["labels"] = list(values)
        return normalized

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: frozenset[str]) -> frozenset[str]:
        """Require a non-empty, non-blank label set."""

        cleaned = frozenset(label.strip() for label in labels if label.strip())
        if not cleaned:
            raise ValueError("Node.labels must contain at least one non-empty label.")
        return cleaned


class ChunkNode(Node):
    """Graph node representing one source chunk.

    Chunk nodes are system-owned nodes linked back to a parent
    :class:`DocumentNode` and typically carry both raw chunk text and an
    embedding for retrieval.
    """

    system_label: ClassVar[str] = "__chunk__"
    labels: frozenset[str] = Field(
        default_factory=lambda: frozenset({ChunkNode.system_label})
    )
    document_id: str = Field(
        description="Identifier of the document this chunk belongs to."
    )
    content: str = Field(description="Raw text content of the chunk.")
    embedding: list[float] = Field(
        default_factory=list,
        description="Optional vector embedding associated with the chunk, for example for retrieval purposes.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata associated with the chunk.",
    )
    doc_position: int = Field(
        default=0,
        description=(
            "Position of the chunk within the original document, starting from 0. "
            "This can be used to reconstruct the original document order or for positional retrieval."
        ),
    )

    @classmethod
    def from_chunk(cls, chunk: Chunk, *, name: str | None = None) -> "ChunkNode":
        """Create a chunk node from a source chunk.

        Parameters
        ----------
        chunk : Chunk
            Source chunk to represent in the graph.
        name : str | None, optional
            Optional human-readable name. Defaults to ``Chunk <chunk.id>``.

        Returns
        -------
        ChunkNode
            Chunk represented as a graph node.
        """

        return cls(
            id=chunk.id,
            name=name or f"Chunk {chunk.id}",
            semantic_key="chunk_" + chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=dict(chunk.metadata),
            doc_position=chunk.doc_position,
        )


class DocumentNode(Node):
    """Graph node representing one source document.

    Document nodes store the source text and metadata for an ingested document
    and can also carry a document-level embedding.
    """

    system_label: ClassVar[str] = "__document__"
    labels: frozenset[str] = Field(
        default_factory=lambda: frozenset({DocumentNode.system_label})
    )
    content: str = Field(description="Raw text content of the document.")
    embedding: list[float] = Field(
        default_factory=list,
        description="Optional vector embedding associated with the document, for example for retrieval purposes.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata associated with the document.",
    )

    @classmethod
    def from_document(cls, document: Document) -> "DocumentNode":
        """Create a document node from a source document.

        Parameters
        ----------
        document : Document
            Source document to represent in the graph.

        Returns
        -------
        DocumentNode
            Document represented as a graph node.
        """

        return cls(
            id=document.id,
            name=document.title,
            semantic_key="document_" + document.id,
            content=document.content,
            metadata=dict(document.metadata),
        )


class MemoryNode(Node):
    """Graph node representing one stored memory.

    Memory nodes capture persisted agent-facing notes or summaries and can be
    searched and expanded through connected graph context.
    """

    system_label: ClassVar[str] = "__memory__"
    labels: frozenset[str] = Field(
        default_factory=lambda: frozenset({MemoryNode.system_label})
    )
    content: str = Field(description="Stored memory content.")
    creation_date: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat(),
        description="ISO 8601 timestamp describing when the memory was created.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata associated with the memory.",
    )


class Relationship(GraphModel):
    """Directed relationship between two graph nodes.

    Relationships are persisted with explicit source and target node ids plus a
    short type label and optional factual properties.
    """

    id: str = Field(
        description=(
            "Machine-generated identifier unique within the graph. "
            "This value is assigned by application code, typically as a UUID."
        )
    )
    source: str = Field(description="Machine-generated identifier of the source node.")
    target: str = Field(description="Machine-generated identifier of the target node.")
    label: str = Field(
        description=(
            "Relationship type expressed as a short verb-like connector "
            "such as 'invented' or 'located_in'."
        )
    )
    properties: dict[str, str] = Field(
        default_factory=dict,
        description="Short factual properties associated with the relationship.",
    )


class KnowledgeGraph(GraphModel):
    """Knowledge graph consisting of nodes and relationships.

    This is the durable in-memory graph container passed between extraction,
    persistence, and higher-level workflows.
    """

    nodes: list[Node] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
