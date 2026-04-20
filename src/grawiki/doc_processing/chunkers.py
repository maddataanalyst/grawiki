import uuid
from typing import Literal

from chonkie import (
    FastChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)

from src.grawiki.core.commons import Chunk, Document

# TODO: later add support for markdown processing via Chonkie MarkdownChef


class Chunker:
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
        chunks = self.chunker.chunk(document.content)
        results = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            results.append(
                Chunk(id=chunk_id, document_id=document.id, content=chunk.text)
            )
        return results
