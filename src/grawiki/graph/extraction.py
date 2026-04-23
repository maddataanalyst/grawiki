"""Knowledge graph extraction helpers."""

import uuid

from pydantic_ai import Agent, Embedder
from src.grawiki.core.commons import Chunk
from src.grawiki.graph.graph_prompts import KG_EXTRACTION_PROMPT
from src.grawiki.graph.models import (
    ExtractedKnowledgeGraph,
    ExtractedNode,
    KnowledgeGraph,
    Node,
    Relationship,
)


class KnowledgeGraphExtractor:
    """Extract chunk-level knowledge graphs and attach entity embeddings.

    Parameters
    ----------
    model : str
        Chat model used for structured knowledge extraction.
    embedding : str
        Embedding model used for entity node vectors.
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
        embedding: str,
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
        self.embedding = Embedder(embedding)
        self.agent = Agent(
            model=model,
            system_prompt=formatted_prompt,
            output_type=ExtractedKnowledgeGraph,
            *args,
            **kwargs,
        )

    async def extract(self, chunk: Chunk) -> KnowledgeGraph:
        """Extract a knowledge graph for one chunk.

        Parameters
        ----------
        chunk : Chunk
            Source chunk to analyze.

        Returns
        -------
        KnowledgeGraph
            Extracted graph with embedded entity nodes.
        """

        graph = await self.agent.run(chunk.content)
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
            nodes_by_name[extracted_node.name] = Node.from_extracted_node(
                extracted_node
            )

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
