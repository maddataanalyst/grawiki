"""Knowledge graph extraction helpers.

This module also defines the LLM-facing transient types
(:class:`ExtractedNode`, :class:`ExtractedRelationship`,
:class:`ExtractedKnowledgeGraph`). They live here rather than in
:mod:`grawiki.graph.models` because they are an implementation detail
of extraction — the persisted domain model (``Node`` / ``Relationship`` /
``KnowledgeGraph``) does not reference them.
"""

import uuid

from pydantic import Field
from pydantic_ai import Agent
from typing import Protocol

from grawiki.core.embedding import Embedding
from grawiki.graph.prompts import KG_EXTRACTION_PROMPT
from grawiki.graph.models import (
    GraphModel,
    KnowledgeGraph,
    Node,
    Relationship,
)


class KnowledgeGraphExtractorProtocol(Protocol):
    """Protocol for chunk-level knowledge graph extractors."""

    async def extract(self, text: str) -> KnowledgeGraph:
        """Extract a graph for one text input."""


class ExtractedNode(GraphModel):
    """Extractor-facing node without a machine-generated identifier.

    This transient shape is produced by the LLM extractor before the
    application assigns durable UUIDs and converts the result into persisted
    :class:`~grawiki.graph.models.Node` objects.
    """

    label: str = Field(
        description=(
            "Ontology type or category of the node, such as 'Person', "
            "'Organization', or 'Concept'."
        )
    )
    name: str = Field(
        description=(
            "Human-readable node name used as the temporary reference key "
            "within one extraction result."
        )
    )
    semantic_key: str = Field(
        description=(
            "A key constructed as the concatenation of node label and shortened name, "
            "used to identify nodes with the same label and name across the graph. "
            "Should be very short and brief. Example: person_alan-turing, code-snippet_llm-implementation."
        )
    )
    properties: dict[str, str] = Field(
        default_factory=dict,
        description="Short factual properties associated with the node.",
    )


class ExtractedRelationship(GraphModel):
    """Extractor-facing relationship using node names as endpoints.

    Relationship endpoints reference extracted node names within one extraction
    result and are later rewritten to durable node identifiers during
    persistence.
    """

    source: str = Field(
        description="Source node name matching one extracted node name exactly."
    )
    target: str = Field(
        description="Target node name matching one extracted node name exactly."
    )
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


class ExtractedKnowledgeGraph(GraphModel):
    """Extractor-facing graph before machine identifiers are assigned.

    Node names act as temporary reference keys within one extraction result and
    are later promoted into a persisted
    :class:`~grawiki.graph.models.KnowledgeGraph`.
    """

    nodes: list[ExtractedNode] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


def _node_from_extracted(extracted_node: ExtractedNode) -> Node:
    """Promote an :class:`ExtractedNode` to a persisted :class:`Node`.

    The extractor emits nodes referenced by name. Before persistence the
    app assigns each node a durable UUID; this helper performs that
    promotion in one place so the extractor is the only module that
    knows how to cross the pre-id / post-id boundary.

    Parameters
    ----------
    extracted_node : ExtractedNode
        Extractor output node.

    Returns
    -------
    Node
        Persisted-shape node with a freshly assigned UUID.
    """

    return Node(
        id=str(uuid.uuid4()),
        labels=frozenset({extracted_node.label}),
        semantic_key=extracted_node.semantic_key,
        name=extracted_node.name,
        properties=dict(extracted_node.properties),
    )


class KnowledgeGraphExtractor:
    """Extract chunk-level knowledge graphs and attach entity embeddings.

    Parameters
    ----------
    model : str
        Chat model used for structured knowledge extraction.
    embedding : Embedding
        Embedding client used for entity node vectors. Injected so callers share
        one embedding model across the pipeline instead of each component
        constructing its own.
    prompt : str, optional
        Extraction prompt template.
    max_triplets : int, optional
        Maximum number of triplets requested from the model.
    allowed_entity_types : list[str] | None, optional
        Optional entity label allow-list.
    allowed_relation_types : list[str] | None, optional
        Optional relationship label allow-list.
    fix_missing_nodes : bool, optional
        Whether to inject placeholder nodes for relationships that reference
        missing node names.
    *args
        Forwarded to :class:`pydantic_ai.Agent`.
    **kwargs
        Forwarded to :class:`pydantic_ai.Agent`.
    """

    def __init__(
        self,
        model: str,
        embedding: Embedding,
        prompt: str = KG_EXTRACTION_PROMPT,
        max_triplets: int = 5,
        allowed_entity_types: list[str] | None = None,
        allowed_relation_types: list[str] | None = None,
        fix_missing_nodes: bool = True,
        *args,
        **kwargs,
    ):
        self.fix_missing_nodes = fix_missing_nodes
        formatted_prompt = prompt.format(
            max_triplets=max_triplets,
            allowed_entity_types=", ".join(allowed_entity_types)
            if allowed_entity_types
            else "",
            allowed_relation_types=", ".join(allowed_relation_types)
            if allowed_relation_types
            else "",
        )
        self.embedding = embedding
        self.agent = Agent(
            model=model,
            system_prompt=formatted_prompt,
            output_type=ExtractedKnowledgeGraph,
            *args,
            **kwargs,
        )

    async def extract(self, text: str) -> KnowledgeGraph:
        """Extract a knowledge graph for one text input.

        Parameters
        ----------
        text : str
            Source text to analyze.

        Returns
        -------
        KnowledgeGraph
            Extracted graph with embedded entity nodes.
        """

        graph = await self.agent.run(text)
        output_graph = graph.output

        if self.fix_missing_nodes:
            output_graph = self._fix_missing_nodes(output_graph)

        return await self._build_knowledge_graph(output_graph)

    def _fix_missing_nodes(
        self, graph: ExtractedKnowledgeGraph
    ) -> ExtractedKnowledgeGraph:
        """Ensure relationship endpoints exist as nodes.

        Parameters
        ----------
        graph : ExtractedKnowledgeGraph
            Extracted graph that may reference missing node names.

        Returns
        -------
        ExtractedKnowledgeGraph
            Graph with placeholder nodes added when needed.
        """

        node_names = {node.name for node in graph.nodes}
        for rel in graph.relationships:
            if rel.source not in node_names:
                graph.nodes.append(
                    ExtractedNode(
                        label="__unknown__",
                        name=rel.source,
                        semantic_key="__unknown__:" + rel.source,
                    )
                )
                node_names.add(rel.source)
            if rel.target not in node_names:
                graph.nodes.append(
                    ExtractedNode(
                        label="__unknown__",
                        name=rel.target,
                        semantic_key="__unknown__:" + rel.target,
                    )
                )
                node_names.add(rel.target)
        return graph

    async def _build_knowledge_graph(
        self, graph: ExtractedKnowledgeGraph
    ) -> KnowledgeGraph:
        """Convert extractor output to persisted graph models.

        Parameters
        ----------
        graph : ExtractedKnowledgeGraph
            Extracted graph emitted by the agent.

        Returns
        -------
        KnowledgeGraph
            Graph with durable node identifiers and entity embeddings.
        """

        nodes_by_name: dict[str, Node] = {}

        for extracted_node in graph.nodes:
            if extracted_node.name in nodes_by_name:
                continue
            nodes_by_name[extracted_node.name] = _node_from_extracted(extracted_node)

        node_names = list(nodes_by_name)
        if node_names:
            embedding_result = await self.embedding.embed_documents(node_names)
            for node_name, embedding in zip(
                node_names, embedding_result.embeddings, strict=True
            ):
                nodes_by_name[node_name].embedding = list(embedding)

        relationships = [
            Relationship(
                id=str(uuid.uuid4()),
                source=nodes_by_name[rel.source].id,
                target=nodes_by_name[rel.target].id,
                label=rel.label,
                properties=dict(rel.properties),
            )
            for rel in graph.relationships
        ]

        return KnowledgeGraph(
            nodes=list(nodes_by_name.values()),
            relationships=relationships,
        )
