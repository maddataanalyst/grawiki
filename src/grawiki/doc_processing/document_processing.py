import uuid
from pathlib import Path
from grawiki.core.commons import Document, Chunk
from grawiki.doc_processing.chunkers import Chunker


def read_document(file_path: Path) -> Document:
    txt = file_path.read_text()
    return Document(
        id=str(uuid.uuid4()),
        title=file_path.stem,
        content=txt,
        metadata={"filepath": str(file_path)},
    )


def chunk_document(document: Document, chunker: Chunker) -> list[Chunk]:
    chunks = chunker.chunk(document)
    return chunks
