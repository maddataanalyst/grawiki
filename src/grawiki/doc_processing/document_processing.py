"""Source-document loading and generic chunking helpers."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

import pymupdf4llm

from grawiki.core.commons import Chunk, Document
from grawiki.doc_processing.chunkers import Chunker

SourceFormat = Literal["text", "markdown", "pdf"]
ContentFormat = Literal["text", "markdown"]


def read_document(file_path: Path) -> Document:
    """Read one source document and normalize its format metadata.

    Parameters
    ----------
    file_path : Path
        Source file to load.

    Returns
    -------
    Document
        Loaded document with normalized ``filepath``, ``source_format``, and
        ``content_format`` metadata.
    """

    source_format, content_format = _detect_formats(file_path)
    if source_format == "pdf":
        text = pymupdf4llm.to_markdown(file_path)
    else:
        text = file_path.read_text()
    return Document(
        id=str(uuid.uuid4()),
        title=file_path.stem,
        content=text,
        metadata={
            "filepath": str(file_path),
            "source_format": source_format,
            "content_format": content_format,
        },
    )


def chunk_document(document: Document, chunker: Chunker) -> list[Chunk]:
    """Delegate generic text chunking to the configured chunker.

    Parameters
    ----------
    document : Document
        Source document to split.
    chunker : Chunker
        Generic text chunker to apply.

    Returns
    -------
    list[Chunk]
        Chunks in source order.
    """

    return chunker.chunk(document)


def _detect_formats(file_path: Path) -> tuple[SourceFormat, ContentFormat]:
    """Return normalized source and content formats for ``file_path``."""

    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown", "markdown"
    if suffix == ".pdf":
        return "pdf", "markdown"
    return "text", "text"
