import uuid

from pydantic_ai import Agent
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
    def __init__(
        self,
        model: str,
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
        self.agent = Agent(
            model=model,
            system_prompt=formatted_prompt,
            output_type=ExtractedKnowledgeGraph,
            *args,
            **kwargs,
        )

    async def extract(self, chunk: Chunk) -> KnowledgeGraph:
        graph = await self.agent.run(chunk.content)
        output_graph = graph.output

        if self.fix_missing_nodes:
            output_graph = self._fix_missing_nodes(output_graph)

        return self._build_knowledge_graph(output_graph)

    def _fix_missing_nodes(
        self, graph: ExtractedKnowledgeGraph
    ) -> ExtractedKnowledgeGraph:
        node_names = {node.name for node in graph.nodes}
        for rel in graph.relationships:
            if rel.source not in node_names:
                graph.nodes.append(ExtractedNode(label="__unknown__", name=rel.source))
                node_names.add(rel.source)
            if rel.target not in node_names:
                graph.nodes.append(ExtractedNode(label="__unknown__", name=rel.target))
                node_names.add(rel.target)
        return graph

    def _build_knowledge_graph(self, graph: ExtractedKnowledgeGraph) -> KnowledgeGraph:
        nodes_by_name: dict[str, Node] = {}

        for extracted_node in graph.nodes:
            if extracted_node.name in nodes_by_name:
                continue

            nodes_by_name[extracted_node.name] = Node(
                id=str(uuid.uuid4()),
                label=extracted_node.label,
                name=extracted_node.name,
                properties=dict(extracted_node.properties),
            )

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
