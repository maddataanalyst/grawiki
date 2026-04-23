"""Main ingestion pipeline for document, chunk, and graph persistence."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal, Protocol

from pydantic_ai import Embedder

from src.grawiki.core.commons import Chunk, Document
from src.grawiki.db.base import GraphDB, SearchMethod, SearchResults
from src.grawiki.doc_processing.chunkers import Chunker
from src.grawiki.doc_processing.document_processing import chunk_document, read_document
from src.grawiki.graph.extraction import KnowledgeGraphExtractor
from src.grawiki.graph.models import ChunkNode, DocumentNode, KnowledgeGraph


logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """Protocol for embedding providers used by the pipeline."""

    async def embed_documents(self, documents: str | list[str]):
        """Embed one or more document-like strings."""

    async def embed_query(self, query: str | list[str]):
        """Embed one or more query strings."""


class KnowledgeGraphExtractorProtocol(Protocol):
    """Protocol for chunk-level knowledge graph extractors."""

    async def extract(self, chunk: Chunk) -> KnowledgeGraph:
        """Extract a graph for a single chunk."""


class GrawikiPipeline:
    """Orchestrate document ingestion into the graph database.

    The pipeline is the main high-level entrypoint for the repository's
    ingestion flow. It exposes public step methods for debugging and notebook
    usage, while :meth:`ingest_file` composes those steps into one end-to-end
    operation.

    Parameters
    ----------
    model : str
        Chat model used by the knowledge graph extractor.
    embedding_model : str
        Embedding model used for documents, chunks, and entities.
    graph_db : GraphDB
        Graph database adapter used for persistence.
    chunking_strategy : Literal["fast", "recursive", "semantic", "sentence", "token"], optional
        Chunking strategy passed to :class:`src.grawiki.doc_processing.chunkers.Chunker`.
    max_workers : int, optional
        Maximum number of chunk-level extraction coroutines to run in parallel.
    embedder : EmbedderProtocol | None, optional
        Optional embedder override used for tests or debugging.
    kg_extractor : KnowledgeGraphExtractorProtocol | None, optional
        Optional knowledge graph extractor override used for tests or debugging.
    """

    def __init__(
        self,
        model: str,
        embedding_model: str,
        graph_db: GraphDB,
        chunking_strategy: Literal[
            "fast", "recursive", "semantic", "sentence", "token"
        ] = "sentence",
        max_workers: int = 4,
        embedder: EmbedderProtocol | None = None,
        kg_extractor: KnowledgeGraphExtractorProtocol | None = None,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.graph_db = graph_db
        self.chunking_strategy = chunking_strategy
        self.max_workers = max_workers
        self.chunker = Chunker(strategy=chunking_strategy)
        self.embedding = embedder or Embedder(embedding_model)
        self.kg_extractor = kg_extractor or KnowledgeGraphExtractor(
            model=model,
            embedding=embedding_model,
        )

    async def setup_db(
        self, embedding_dimensions: dict[str, int] | None = None
    ) -> None:
        """Prepare database indexes needed by the pipeline.

        Parameters
        ----------
        embedding_dimensions : dict[str, int] | None, optional
            Mapping from node label to embedding dimension for vector indexes.
        """

        logger.info("Setting up graph database indexes")
        await self.graph_db.setup(embedding_dimensions=embedding_dimensions)

    def read_document(self, path: Path) -> Document:
        """Load one source document from disk.

        Parameters
        ----------
        path : Path
            Filesystem path to the source document.

        Returns
        -------
        Document
            Parsed document object.
        """

        logger.info("Reading document from %s", path)
        document = read_document(path)
        logger.debug(
            "Loaded document %s (%s chars)", document.id, len(document.content)
        )
        return document

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Parameters
        ----------
        document : Document
            Source document to chunk.

        Returns
        -------
        list[Chunk]
            Produced chunks.
        """

        logger.info(
            "Chunking document %s with strategy %s", document.id, self.chunking_strategy
        )
        chunks = chunk_document(document, self.chunker)
        logger.info("Created %s chunks for document %s", len(chunks), document.id)
        return chunks

    async def embed_document(self, document: Document) -> list[float]:
        """Embed one document's content.

        Parameters
        ----------
        document : Document
            Document whose content should be embedded.

        Returns
        -------
        list[float]
            Document embedding.
        """

        logger.info("Embedding document %s", document.id)
        result = await self.embedding.embed_documents([document.content])
        embedding = list(result.embeddings[0])
        logger.debug("Document %s embedding dimension: %s", document.id, len(embedding))
        return embedding

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed chunk contents in one batch.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks whose content should be embedded.

        Returns
        -------
        list[list[float]]
            Embedding per chunk, in the same order as ``chunks``.
        """

        if not chunks:
            logger.info("No chunks to embed")
            return []

        logger.info("Embedding %s chunks", len(chunks))
        result = await self.embedding.embed_documents(
            [chunk.content for chunk in chunks]
        )
        embeddings = [list(embedding) for embedding in result.embeddings]
        logger.debug(
            "Chunk embedding dimension: %s",
            len(embeddings[0]) if embeddings else 0,
        )
        return embeddings

    def build_document_node(
        self,
        document: Document,
        embedding: list[float],
    ) -> DocumentNode:
        """Build a persisted document node.

        Parameters
        ----------
        document : Document
            Source document to convert.
        embedding : list[float]
            Vector embedding for the document content.

        Returns
        -------
        DocumentNode
            Prepared document node.
        """

        document_node = DocumentNode.from_document(document)
        document_node.embedding = embedding
        return document_node

    def build_chunk_nodes(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[ChunkNode]:
        """Build persisted chunk nodes.

        Parameters
        ----------
        chunks : list[Chunk]
            Source chunks to convert.
        embeddings : list[list[float]]
            Chunk embeddings in the same order as ``chunks``.

        Returns
        -------
        list[ChunkNode]
            Prepared chunk nodes.

        Raises
        ------
        ValueError
            Raised when the chunk and embedding counts differ.
        """

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        chunk_nodes = [ChunkNode.from_chunk(chunk) for chunk in chunks]
        for chunk_node, embedding in zip(chunk_nodes, embeddings, strict=True):
            chunk_node.embedding = embedding
        return chunk_nodes

    async def persist_documents_and_chunks(
        self,
        document_node: DocumentNode,
        chunk_nodes: list[ChunkNode],
    ) -> None:
        """Persist one document node and its chunk nodes.

        Parameters
        ----------
        document_node : DocumentNode
            Prepared document node to upsert.
        chunk_nodes : list[ChunkNode]
            Prepared chunk nodes to upsert.
        """

        embedding_dimensions: dict[str, int] = {}
        if document_node.embedding:
            embedding_dimensions[document_node.label] = len(document_node.embedding)
        if chunk_nodes and chunk_nodes[0].embedding:
            embedding_dimensions[chunk_nodes[0].label] = len(chunk_nodes[0].embedding)

        logger.info(
            "Persisting document %s and %s chunks",
            document_node.id,
            len(chunk_nodes),
        )
        await self.setup_db(embedding_dimensions=embedding_dimensions or None)
        await self.graph_db.save_docs_and_chunks_to_db([document_node], chunk_nodes)

    async def extract_chunk_graphs(
        self,
        chunks: list[Chunk],
    ) -> dict[str, KnowledgeGraph]:
        """Extract knowledge graphs for chunks with bounded concurrency.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks to analyze.

        Returns
        -------
        dict[str, KnowledgeGraph]
            Extracted graphs keyed by chunk identifier.
        """

        if not chunks:
            logger.info("No chunks to extract")
            return {}

        logger.info(
            "Extracting knowledge graphs for %s chunks with max_workers=%s",
            len(chunks),
            self.max_workers,
        )
        semaphore = asyncio.Semaphore(self.max_workers)

        async def extract_one(chunk: Chunk) -> tuple[str, KnowledgeGraph]:
            async with semaphore:
                graph = await self.kg_extractor.extract(chunk)
                logger.debug(
                    "Extracted graph for chunk %s with %s nodes and %s relationships",
                    chunk.id,
                    len(graph.nodes),
                    len(graph.relationships),
                )
                return chunk.id, graph

        results = await asyncio.gather(*(extract_one(chunk) for chunk in chunks))
        return dict(results)

    async def persist_entities_and_relationships(
        self,
        chunks: list[Chunk],
        chunk_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        """Persist extracted entities and relationships.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks that own the extracted graphs.
        chunk_graphs : dict[str, KnowledgeGraph]
            Extracted graphs keyed by chunk identifier.
        """

        entity_dimension: int | None = None
        for graph in chunk_graphs.values():
            for node in graph.nodes:
                if node.embedding:
                    entity_dimension = len(node.embedding)
                    break
            if entity_dimension is not None:
                break

        logger.info(
            "Persisting entities and relationships for %s chunk graphs",
            len(chunk_graphs),
        )
        await self.setup_db(
            embedding_dimensions={"__entity__": entity_dimension}
            if entity_dimension is not None
            else None
        )
        await self.graph_db.save_entities_and_rels(chunks, chunk_graphs)

    async def ingest_file(self, path: Path) -> None:
        """Run the full ingestion flow for one file.

        Parameters
        ----------
        path : Path
            Source file to ingest.
        """

        logger.info("Starting ingestion for %s", path)
        await self.setup_db()
        document = self.read_document(path)
        chunks = self.chunk_document(document)
        document_embedding = await self.embed_document(document)
        chunk_embeddings = await self.embed_chunks(chunks)
        document_node = self.build_document_node(document, document_embedding)
        chunk_nodes = self.build_chunk_nodes(chunks, chunk_embeddings)
        await self.persist_documents_and_chunks(document_node, chunk_nodes)
        chunk_graphs = await self.extract_chunk_graphs(chunks)
        await self.persist_entities_and_relationships(chunks, chunk_graphs)
        logger.info("Completed ingestion for %s", path)

    async def search(
        self,
        query: str,
        method: SearchMethod,
        *,
        limit: int = 10,
    ) -> SearchResults:
        """Search documents, chunks, and entities through the graph DB.

        Parameters
        ----------
        query : str
            Raw user query text.
        method : {"fulltext", "vector"}
            Search strategy to execute.
        limit : int, optional
            Maximum number of results to return per node family.

        Returns
        -------
        SearchResults
            Search hits grouped by node family.
        """

        logger.info("Running %s search for query %r", method, query)
        if method == "fulltext":
            return await self.graph_db.search(query, method=method, limit=limit)

        embedding_result = await self.embedding.embed_query(query)
        if not embedding_result.embeddings:
            raise ValueError("Query embedding result is empty.")

        query_embedding = list(embedding_result.embeddings[0])
        return await self.graph_db.search(
            query,
            method=method,
            limit=limit,
            query_embedding=query_embedding,
        )
