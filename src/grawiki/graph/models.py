"""Pydantic models used to represent the knowledge graph."""

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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

    Parameters
    ----------
    id : str
        Machine-generated identifier unique within the graph.
    label : str
        Ontology type of the node, for example ``Person`` or ``Concept``.
    semantic_key: str
        A key constructed as the concatenation of node label and shortened name, used to identify nodes with the
        same label and name across the graph. Should be very short and brief.
        Example: person_alan-turing, code-snippet_llm-implementation.
    name : str
        Human-readable canonical name of the node instance.
    properties : dict[str, str], optional
        Short factual properties associated with the node.
    embedding: list[float]
        Optional vector embedding associated with the node, for example for retrieval purposes.
    """

    id: str = Field(
        description=(
            "Machine-generated identifier unique within the graph. "
            "This value is assigned by application code, typically as a UUID."
        )
    )
    label: str = Field(
        description=(
            "Ontology type or category of the node, such as 'Person', "
            "'Organization', or 'Concept'."
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


class ChunkNode(Node):
    """Graph node representing one source chunk.

    Parameters
    ----------
    id : str
        Stable identifier of the chunk node.
    name : str
        Human-readable chunk name.
    document_id : str
        Identifier of the parent document.
    content : str
        Raw chunk text.
    embedding: list[float], optional
        Optional vector embedding associated with the chunk, for example for retrieval purposes.
    metadata : dict[str, str], optional
        Additional chunk metadata.
    """

    label: Literal["__chunk__"] = "__chunk__"
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
        )


class DocumentNode(Node):
    """Graph node representing one source document.

    Parameters
    ----------
    id : str
        Stable identifier of the document node.
    name : str
        Human-readable document name, typically the document title.
    content : str
        Raw document text.
    embedding: list[float], optional
        Optional vector embedding associated with the document, for example for retrieval purposes.
    metadata : dict[str, str], optional
        Additional document metadata.
    """

    label: Literal["__document__"] = "__document__"
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

    Parameters
    ----------
    id : str
        Stable identifier of the memory node.
    name : str
        Human-readable memory title or summary.
    content : str
        Stored memory content.
    creation_date : str, optional
        ISO 8601 timestamp describing when the memory was created.
    metadata : dict[str, str], optional
        Additional memory metadata.
    """

    label: Literal["__memory__"] = "__memory__"
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

    Parameters
    ----------
    id : str
        Machine-generated identifier unique within the graph.
    source : str
        Machine-generated identifier of the source node.
    target : str
        Machine-generated identifier of the target node.
    label : str
        Relationship type expressed as a short verb-like connector.
    properties : dict[str, str], optional
        Short factual properties associated with the relationship.
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

    Parameters
    ----------
    nodes : list[Node], optional
        Nodes present in the graph.
    relationships : list[Relationship], optional
        Relationships present in the graph. Every relationship endpoint
        should reference an existing node identifier.
    """

    nodes: list[Node] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
