"""Chunking adapters for source documents."""

from __future__ import annotations

import uuid
from typing import Literal

from chonkie import (
    FastChunker,
    Pipeline,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)

from grawiki.core.commons import Chunk, Document


def build_default_markdown_pipeline() -> Pipeline:
    """Return the default markdown chunking pipeline.

    Returns
    -------
    Pipeline
        Markdown-aware pipeline that segments prose with a token chunker using
        the character tokenizer and a conservative chunk size.
    """

    return (
        Pipeline()
        .process_with("markdown")
        .chunk_with("token", tokenizer="character", chunk_size=4096)
    )


class Chunker:
    """Adapter over Chonkie chunkers for generic source text.

    Parameters
    ----------
    strategy : {"fast", "recursive", "semantic", "sentence", "token"}, optional
        Chonkie chunking strategy to use.
    *args
        Positional arguments forwarded to the selected Chonkie chunker.
    **kwargs
        Keyword arguments forwarded to the selected Chonkie chunker.
    """

    def __init__(
        self,
        strategy: Literal[
            "fast", "recursive", "semantic", "sentence", "token"
        ] = "sentence",
        *args,
        **kwargs,
    ):
        self.chunker = self._get_chunker(strategy, *args, **kwargs)

    def _get_chunker(self, strategy: str, *args, **kwargs):
        """Return the concrete Chonkie chunker for ``strategy``."""

        if strategy == "fast":
            return FastChunker(*args, **kwargs)
        elif strategy == "recursive":
            return RecursiveChunker(*args, **kwargs)
        elif strategy == "semantic":
            return SemanticChunker(*args, **kwargs)
        elif strategy == "sentence":
            return SentenceChunker(*args, **kwargs)
        elif strategy == "token":
            return TokenChunker(*args, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into generic text chunks.

        Parameters
        ----------
        document : Document
            Source document to split.

        Returns
        -------
        list[Chunk]
            Chunks in source order.
        """

        chunks = self.chunker.chunk(document.content)
        results = []
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            results.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk.text,
                    doc_position=idx,
                    metadata={},
                )
            )
        return results


class MarkdownPipelineChunker:
    """Pipeline-backed chunking adapter for markdown or text content.

    Parameters
    ----------
    pipeline : Pipeline | None, optional
        Preconfigured pipeline. When omitted, the built-in markdown-aware
        pipeline returned by :func:`build_default_markdown_pipeline` is used.

    Notes
    -----
    The configured pipeline must include at least one chunking stage. When the
    processed result also exposes markdown-specific ``code`` or ``tables``
    collections, those blocks are merged into the ordered chunk stream without
    additional re-chunking.
    """

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline or build_default_markdown_pipeline()
        self._validate_pipeline(self.pipeline)

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document with the configured pipeline.

        Parameters
        ----------
        document : Document
            Source document to split.

        Returns
        -------
        list[Chunk]
            Chunks ordered by their original position when start indexes are
            available from the pipeline output.
        """

        processed_document = self.pipeline.run(texts=document.content)
        all_candidates = self._collect_candidates(processed_document)
        if all(start_index is not None for _, start_index, _, _ in all_candidates):
            all_candidates.sort(key=lambda item: (item[1], item[3]))

        source_path = document.metadata.get("filepath")
        results = []
        for idx, (chunk_text, start_index, chunk_type, _) in enumerate(all_candidates):
            metadata: dict[str, str] = {}
            if chunk_type is not None:
                metadata["type"] = chunk_type
            if start_index is not None:
                metadata["start_index"] = str(start_index)
            if source_path:
                metadata["source_path"] = source_path
            results.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    content=chunk_text,
                    doc_position=idx,
                    metadata=metadata,
                )
            )
        return results

    def _collect_candidates(
        self, processed_document
    ) -> list[tuple[str, int | None, str | None, int]]:
        """Return normalized chunk candidates from a processed document."""

        has_structured_markdown = any(
            hasattr(processed_document, attribute) for attribute in ("code", "tables")
        )
        content_type = "text" if has_structured_markdown else None
        candidates: list[tuple[str, int | None, str | None, int]] = []
        order = 0
        for items, chunk_type in (
            (getattr(processed_document, "chunks", []), content_type),
            (getattr(processed_document, "code", []), "code"),
            (getattr(processed_document, "tables", []), "table"),
        ):
            normalized_items = self._collect_processed_items(
                items,
                chunk_type=chunk_type,
                start_order=order,
            )
            candidates.extend(normalized_items)
            order += len(normalized_items)
        return candidates

    def _collect_processed_items(
        self,
        items,
        *,
        chunk_type: Literal["text", "code", "table"] | None,
        start_order: int,
    ) -> list[tuple[str, int | None, str | None, int]]:
        """Return normalized chunk candidates for one processed item list."""

        candidates: list[tuple[str, int | None, str | None, int]] = []
        order = start_order
        for item in items or []:
            item_text = self._get_processed_item_text(item)
            if not item_text:
                continue
            candidates.append(
                (
                    item_text,
                    self._get_processed_item_start_index(item),
                    chunk_type or getattr(item, "type", None),
                    order,
                )
            )
            order += 1
        return candidates

    def _get_processed_item_text(self, item) -> str:
        """Return normalized text content for one processed item."""

        return getattr(item, "text", None) or getattr(item, "content", "")

    def _get_processed_item_start_index(self, item) -> int | None:
        """Return the normalized start index for one processed item."""

        start_index = getattr(item, "start_index", None)
        if start_index is None:
            return None
        return int(start_index)

    def _validate_pipeline(self, pipeline: Pipeline) -> None:
        """Validate that ``pipeline`` contains at least one chunking stage."""

        config = pipeline.to_config()
        has_chunk = any(step["type"] == "chunk" for step in config)
        if not has_chunk:
            raise ValueError(
                "MarkdownPipelineChunker requires a pipeline with at least one "
                "chunking step."
            )


# Compatibility alias for callers that still import the old adapter name.
MarkdownChunker = MarkdownPipelineChunker

__all__ = [
    "Chunker",
    "MarkdownChunker",
    "MarkdownPipelineChunker",
    "build_default_markdown_pipeline",
]
