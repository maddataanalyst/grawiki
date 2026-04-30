"""Source data models used before graph persistence."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Raw source document before graph persistence.

    Parameters
    ----------
    id : str
        Stable document identifier.
    title : str
        Human-readable document title.
    content : str
        Raw document text.
    metadata : dict[str, str], optional
        Additional source metadata.
    """

    id: str
    title: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Raw source chunk before graph persistence.

    Parameters
    ----------
    id : str
        Stable chunk identifier.
    document_id : str
        Identifier of the parent document.
    content : str
        Raw chunk text.
    doc_position : int, optional
        Zero-based chunk position within the parent document.
    metadata : dict[str, str], optional
        Additional chunk metadata.
    """

    id: str
    document_id: str
    content: str
    doc_position: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)
