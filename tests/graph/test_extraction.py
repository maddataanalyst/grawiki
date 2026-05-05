"""Tests for knowledge graph extraction helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from grawiki.graph.extraction import (
    ExtractedKnowledgeGraph,
    ExtractedNode,
    KnowledgeGraphExtractor,
)


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


class FakeExtractorClient:
    """Structured-output client stub that records extractor requests."""

    def __init__(self, result: ExtractedKnowledgeGraph) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> ExtractedKnowledgeGraph:
        self.calls.append(kwargs)
        return self.result


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
    assert built_graph.nodes[0].labels == frozenset({"Person"})
    assert isinstance(built_graph.nodes[0].embedding, list)


def test_extractor_defaults_to_english_output_language() -> None:
    """The default extraction prompt should request English graph strings."""

    extractor = KnowledgeGraphExtractor(model="test-model", embedding=FakeEmbedder())

    assert extractor.output_language == "English"
    assert "in English" in extractor.extraction_prompt


def test_extract_uses_configured_output_language_and_extract_kwargs() -> None:
    """extract() should send the configured language prompt and API kwargs."""

    extractor = KnowledgeGraphExtractor(
        model="test-model",
        embedding=FakeEmbedder(),
        output_language="Polish",
        extract_kwargs={"reasoning_effort": "minimal"},
    )
    client = FakeExtractorClient(ExtractedKnowledgeGraph())
    extractor._extractor_client = client

    built_graph = asyncio.run(
        extractor.extract("Marie Curie received the Nobel Prize in Chemistry.")
    )

    assert built_graph.nodes == []
    assert built_graph.relationships == []
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["response_model"] is ExtractedKnowledgeGraph
    assert call["reasoning_effort"] == "minimal"
    assert call["messages"][0]["role"] == "system"
    assert "in Polish" in call["messages"][0]["content"]
    assert call["messages"][1]["content"].endswith(
        "Marie Curie received the Nobel Prize in Chemistry."
    )
