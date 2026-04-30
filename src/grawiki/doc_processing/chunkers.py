"""Chunking adapters for source documents."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from chonkie import (
    FastChunker,
    MarkdownChef,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)

from grawiki.core.commons import Chunk, Document


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


class MarkdownChunker:
    """Markdown-aware chunking adapter backed by Chonkie MarkdownChef.

    Parameters
    ----------
    tokenizer : str, optional
        Tokenizer identifier forwarded to :class:`chonkie.MarkdownChef`.
    """

    def __init__(self, tokenizer: str = "gpt2"):
        self.markdown_chef = MarkdownChef(tokenizer=tokenizer)

    def chunk(
        self,
        document: Document,
        source_path: Path | None = None,
    ) -> list[Chunk]:
        """Split a markdown document into text, code, and table chunks.

        Parameters
        ----------
        document : Document
            Source document to split.
        source_path : Path | None, optional
            Filesystem path for file-backed markdown. When omitted,
            ``document.content`` is parsed directly.

        Returns
        -------
        list[Chunk]
            MarkdownChef chunks ordered by their original position.
        """

        if source_path is None:
            doc = self.markdown_chef.parse(document.content)
        else:
            doc = self.markdown_chef.process(source_path)

        all_chef_chunks_and_indices = []
        for chef_chunk in doc.chunks:
            all_chef_chunks_and_indices.append(
                (chef_chunk.text, chef_chunk.start_index, {"type": "text"})
            )
        for chef_code in doc.code:
            all_chef_chunks_and_indices.append(
                (chef_code.content, chef_code.start_index, {"type": "code"})
            )
        for chef_table in doc.tables:
            all_chef_chunks_and_indices.append(
                (chef_table.content, chef_table.start_index, {"type": "table"})
            )

        # Sort chunks by their original position in the document
        all_chef_chunks_and_indices.sort(key=lambda x: x[1])
        results = []
        for idx, (chunk_text, start_index, metadata) in enumerate(
            all_chef_chunks_and_indices
        ):
            chunk_id = str(uuid.uuid4())
            enriched_metadata = {
                **metadata,
                "start_index": str(start_index),
            }
            if source_path is not None:
                enriched_metadata["source_path"] = str(source_path)
            results.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    doc_position=idx,
                    metadata=enriched_metadata,
                )
            )
        return results
