"""Tests for chunk post-processors."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from grawiki.core.commons import Chunk
from grawiki.doc_processing import chunk_processors
from grawiki.doc_processing.chunk_processors import (
    HypotheticalQuestions,
    HypotheticalQuestionsChunkProcessor,
)


def test_hypothetical_questions_chunk_processor_formats_output_and_copies_chunk(
    monkeypatch,
) -> None:
    """Generated questions should be prepended without mutating the source chunk."""

    init_calls: list[dict[str, object]] = []
    run_inputs: list[str] = []
    outputs = [
        HypotheticalQuestions(
            questions=[
                "What problem does the chunk describe?",
                "Which graph concept is introduced?",
            ]
        )
    ]

    class FakeAgent:
        def __init__(self, *args, **kwargs) -> None:
            init_calls.append(kwargs)

        async def run(self, prompt: str) -> SimpleNamespace:
            run_inputs.append(prompt)
            return SimpleNamespace(output=outputs.pop(0))

    monkeypatch.setattr(chunk_processors, "Agent", FakeAgent)
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Original chunk text.",
        doc_position=4,
        metadata={"section": "intro"},
    )

    processor = HypotheticalQuestionsChunkProcessor(
        model="test-model",
        num_question=2,
        language="Polish",
    )
    processed = asyncio.run(processor(chunk))

    assert run_inputs == ["Original chunk text."]
    assert init_calls[0]["model"] == "test-model"
    assert init_calls[0]["output_type"] is HypotheticalQuestions
    assert "Generate max 2 questions in Polish language" in init_calls[0]["system_prompt"]
    assert "What problem does the chunk describe?" in processed.content
    assert "Which graph concept is introduced?" in processed.content
    assert processed.content.endswith("Original chunk text.")
    assert processed.id == chunk.id
    assert processed.document_id == chunk.document_id
    assert processed.doc_position == chunk.doc_position
    assert processed.metadata == chunk.metadata
    assert processed.metadata is not chunk.metadata

    processed.metadata["section"] = "changed"

    assert chunk.metadata == {"section": "intro"}


def test_hypothetical_questions_chunk_processor_keeps_content_when_no_questions(
    monkeypatch,
) -> None:
    """Empty question output should leave chunk content unchanged."""

    outputs = [HypotheticalQuestions()]

    class FakeAgent:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def run(self, prompt: str) -> SimpleNamespace:
            del prompt
            return SimpleNamespace(output=outputs.pop(0))

    monkeypatch.setattr(chunk_processors, "Agent", FakeAgent)
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Original chunk text.",
        metadata={"section": "intro"},
    )

    processor = HypotheticalQuestionsChunkProcessor(model="test-model")
    processed = asyncio.run(processor(chunk))

    assert processed is not chunk
    assert processed.content == chunk.content
    assert processed.metadata == chunk.metadata
    assert processed.metadata is not chunk.metadata
