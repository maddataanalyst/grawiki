"""Tests for knowledge graph extraction helpers."""

from __future__ import annotations

import asyncio

from src.grawiki.graph.graph_extraction import KnowledgeGraphExtractor
from src.grawiki.graph.models import ExtractedKnowledgeGraph, ExtractedNode


class FakeEmbeddingResult:
    """Minimal embedding result object for extractor tests."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeEmbedder:
    """Deterministic embedder stub."""

    async def embed_documents(self, documents: str | list[str]) -> FakeEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        return FakeEmbeddingResult(
            [[float(index), 1.0, 2.0] for index, _ in enumerate(texts, start=1)]
        )


def test_build_knowledge_graph_embeds_entities() -> None:
    """Entity nodes should receive concrete list embeddings, not coroutine objects."""

    extractor = object.__new__(KnowledgeGraphExtractor)
    extractor.embedding = FakeEmbedder()

    graph = ExtractedKnowledgeGraph(
        nodes=[
            ExtractedNode(
                label="Person",
                name="Alan Turing",
                semantic_key="person_alan-turing",
            )
        ],
        relationships=[],
    )

    built_graph = asyncio.run(extractor._build_knowledge_graph(graph))

    assert built_graph.nodes[0].embedding == [1.0, 1.0, 2.0]
    assert isinstance(built_graph.nodes[0].embedding, list)
