from pydantic import BaseModel


class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: dict[str, str] = {}


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: dict[str, str] = {}
