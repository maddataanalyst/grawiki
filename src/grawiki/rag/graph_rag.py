"""GraphRAG facade — document ingestion and retrieval-augmented search."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Literal

from pydantic_ai import Embedder

from src.grawiki.core.commons import Chunk, Document
from src.grawiki.core.embedding import Embedding
from src.grawiki.db.base import GraphDB, NodeHit
from src.grawiki.doc_processing.chunkers import Chunker
from src.grawiki.doc_processing.document_processing import chunk_document, read_document
from src.grawiki.graph.extraction import (
    KnowledgeGraphExtractor,
    KnowledgeGraphExtractorProtocol,
)
from src.grawiki.graph.models import ChunkNode, DocumentNode, KnowledgeGraph
from src.grawiki.retrieval.retriever import Retriever


logger = logging.getLogger(__name__)


_DEFAULT_SEARCH_LABELS = ["__chunk__", "__entity__"]


class GraphRAG:
    """Orchestrate document ingestion and retrieval-augmented search.

    Parameters
    ----------
    model : str
        Chat model used by the knowledge graph extractor.
    embedding_model : str
        Embedding model used for documents, chunks, entities, and queries.
    db : GraphDB
        Graph database adapter used for persistence and search.
    chunking_strategy : str, optional
        Chunking strategy passed to :class:`~src.grawiki.doc_processing.chunkers.Chunker`.
    max_workers : int, optional
        Maximum number of concurrent chunk-level extraction coroutines.
    embedder : Embedding | None, optional
        Embedding override for tests or debugging.
    kg_extractor : KnowledgeGraphExtractorProtocol | None, optional
        Knowledge graph extractor override for tests or debugging.
    """

    def __init__(
        self,
        *,
        model: str,
        embedding_model: str,
        db: GraphDB,
        chunking_strategy: Literal[
            "fast", "recursive", "semantic", "sentence", "token"
        ] = "sentence",
        max_workers: int = 4,
        embedding: Embedding | None = None,
        kg_extractor: KnowledgeGraphExtractorProtocol | None = None,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.chunking_strategy = chunking_strategy
        self.max_workers = max_workers
        self._db = db
        self._chunker = Chunker(strategy=chunking_strategy)
        self._max_workers = max_workers
        self._embedding = embedding or Embedder(embedding_model)
        self._extractor = kg_extractor or KnowledgeGraphExtractor(
            model=model,
            embedding=self._embedding,
        )
        self._retriever = Retriever(db=db, embedding=self._embedding)

    # ------------------------------------------------------------------
    # Public step methods (useful for notebooks and debugging)
    # ------------------------------------------------------------------

    def read_document(self, path: Path) -> Document:
        """Load one source document from disk."""

        logger.info("Reading document from %s", path)
        return read_document(path)

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""

        logger.info("Chunking document %s", document.id)
        chunks = chunk_document(document, self._chunker)
        logger.info("Created %s chunks for document %s", len(chunks), document.id)
        return chunks

    async def embed_document(self, document: Document) -> list[float]:
        """Embed one document's content."""

        logger.info("Embedding document %s", document.id)
        result = await self._embedding.embed_documents([document.content])
        return list(result.embeddings[0])

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed chunk contents in one batch."""

        if not chunks:
            return []
        logger.info("Embedding %s chunks", len(chunks))
        result = await self._embedding.embed_documents(
            [chunk.content for chunk in chunks]
        )
        return [list(e) for e in result.embeddings]

    def build_document_node(
        self, document: Document, embedding: list[float]
    ) -> DocumentNode:
        """Build a document node with its embedding attached."""

        node = DocumentNode.from_document(document)
        node.embedding = embedding
        return node

    def build_chunk_nodes(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> list[ChunkNode]:
        """Build chunk nodes with embeddings attached."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        chunk_nodes = [ChunkNode.from_chunk(chunk) for chunk in chunks]
        for cn, emb in zip(chunk_nodes, embeddings, strict=True):
            cn.embedding = emb
        return chunk_nodes

    async def persist_document_and_chunks(
        self,
        document_node: DocumentNode,
        chunk_nodes: list[ChunkNode],
    ) -> None:
        """Persist one document node and its chunk nodes with indexes."""

        dims: dict[str, int] = {}
        if document_node.embedding:
            dims[document_node.label] = len(document_node.embedding)
        if chunk_nodes and chunk_nodes[0].embedding:
            dims[chunk_nodes[0].label] = len(chunk_nodes[0].embedding)

        logger.info(
            "Persisting document %s and %s chunks", document_node.id, len(chunk_nodes)
        )
        await self._db.setup(embedding_dimensions=dims or None)
        await self._db.save_docs_and_chunks_to_db([document_node], chunk_nodes)

    async def extract_kg_per_chunk(
        self, chunks: list[Chunk]
    ) -> dict[str, KnowledgeGraph]:
        """Extract knowledge graphs for chunks with bounded concurrency."""

        if not chunks:
            return {}
        logger.info(
            "Extracting knowledge graphs for %s chunks with max_workers=%s",
            len(chunks),
            self._max_workers,
        )
        semaphore = asyncio.Semaphore(self._max_workers)

        async def extract_one(chunk: Chunk) -> tuple[str, KnowledgeGraph]:
            async with semaphore:
                graph = await self._extractor.extract(chunk)
                return chunk.id, graph

        results = await asyncio.gather(*(extract_one(c) for c in chunks))
        return dict(results)

    async def persist_entities_and_relationships(
        self,
        chunks: list[Chunk],
        chunk_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        """Persist extracted entities and relationships."""

        entity_dim: int | None = None
        for graph in chunk_graphs.values():
            for node in graph.nodes:
                if node.embedding:
                    entity_dim = len(node.embedding)
                    break
            if entity_dim is not None:
                break

        logger.info("Persisting entities for %s chunk graphs", len(chunk_graphs))
        await self._db.setup(
            embedding_dimensions={"__entity__": entity_dim}
            if entity_dim is not None
            else None
        )
        await self._db.save_entities_and_rels(chunks, chunk_graphs)

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    async def ingest(self, path: Path) -> None:
        """Run the full ingestion flow for one file.

        Parameters
        ----------
        path : Path
            Source file to ingest.
        """

        logger.info("Starting ingestion for %s", path)
        await self._db.setup()
        document = self.read_document(path)
        chunks = self.chunk_document(document)
        document_embedding = await self.embed_document(document)
        chunk_embeddings = await self.embed_chunks(chunks)
        document_node = self.build_document_node(document, document_embedding)
        chunk_nodes = self.build_chunk_nodes(chunks, chunk_embeddings)
        await self.persist_document_and_chunks(document_node, chunk_nodes)
        chunk_graphs = await self.extract_kg_per_chunk(chunks)
        await self.persist_entities_and_relationships(chunks, chunk_graphs)
        logger.info("Completed ingestion for %s", path)

    async def ingest_text(self, text: str, title: str) -> None:
        """Ingest a document supplied as a string.

        Parameters
        ----------
        text : str
            Document content to ingest.
        title : str
            Human-readable document title used as the document name.
        """

        document = Document(id=str(uuid.uuid4()), title=title, content=text)
        logger.info("Starting ingestion for text document %r", title)
        await self._db.setup()
        chunks = self.chunk_document(document)
        document_embedding = await self.embed_document(document)
        chunk_embeddings = await self.embed_chunks(chunks)
        document_node = self.build_document_node(document, document_embedding)
        chunk_nodes = self.build_chunk_nodes(chunks, chunk_embeddings)
        await self.persist_document_and_chunks(document_node, chunk_nodes)
        chunk_graphs = await self.extract_kg_per_chunk(chunks)
        await self.persist_entities_and_relationships(chunks, chunk_graphs)
        logger.info("Completed ingestion for text document %r", title)

    async def search(
        self,
        query: str,
        *,
        method: Literal["fulltext", "vector"] = "vector",
        limit: int = 10,
    ) -> list[NodeHit]:
        """Search documents, chunks, and entities.

        Parameters
        ----------
        query : str
            Raw user query text.
        method : {"fulltext", "vector"}, optional
            Search strategy. Defaults to ``"vector"``.
        limit : int, optional
            Maximum number of results per node family.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated search hits across documents, chunks, and entities.
        """

        logger.info("Running %s search for query %r", method, query)
        if method == "fulltext":
            return await self._retriever.fulltext(
                query, labels=_DEFAULT_SEARCH_LABELS, limit=limit
            )
        return await self._retriever.vector(
            query, labels=_DEFAULT_SEARCH_LABELS, limit=limit
        )
