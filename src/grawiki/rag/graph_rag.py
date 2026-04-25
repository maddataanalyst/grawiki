"""GraphRAG facade — document ingestion and retrieval-augmented search."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Literal

from pydantic_ai import Embedder

from grawiki.core.commons import Chunk, Document
from grawiki.core.embedding import Embedding
from grawiki.db.base import GraphDB, NodeHit
from grawiki.doc_processing.chunkers import Chunker
from grawiki.doc_processing.document_processing import chunk_document, read_document
from grawiki.graph.extraction import (
    KnowledgeGraphExtractor,
    KnowledgeGraphExtractorProtocol,
)
from grawiki.graph.models import (
    ChunkNode,
    DocumentNode,
    KnowledgeGraph,
    Node,
    Relationship,
)
from grawiki.retrieval.base import Retriever
from grawiki.retrieval.keywords import KeywordsPathRetriever
from grawiki.retrieval.text import TextRetriever
from grawiki.similarity.deduplication import (
    MergeReport,
    build_merged_master,
    pick_master,
)
from grawiki.similarity.models import (
    EntityDuplicateCandidates,
    SemanticKeyCollisionCandidates,
)
from grawiki.similarity.similarity_finder import EntitySimilarityFinder

logger = logging.getLogger(__name__)


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
        Chunking strategy passed to :class:`~grawiki.doc_processing.chunkers.Chunker`.
    max_workers : int, optional
        Maximum number of concurrent chunk-level extraction coroutines.
    embedding : Embedding | None, optional
        Embedding override for tests or debugging.
    kg_extractor : KnowledgeGraphExtractorProtocol | None, optional
        Knowledge graph extractor override for tests or debugging.
    similarity_finder : EntitySimilarityFinder | None, optional
        Entity similarity finder used for collision inspection and candidate
        lookup. Defaults to a finder backed by the vector similarity matcher.
    resolve_entities_on_ingest : bool, optional
        When ``True``, each freshly-extracted entity is compared against
        persisted entities before persistence. If a persisted entity is found
        whose cosine similarity exceeds ``entity_resolution_threshold``, the
        extracted node is replaced by the persisted node and all relationship
        endpoints are rewritten accordingly. Defaults to ``False``.
    entity_resolution_threshold : float, optional
        Minimum cosine-similarity score for two entities to be considered the
        same during ingest-time resolution. Only used when
        ``resolve_entities_on_ingest=True``. Defaults to ``0.92``.
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
        retrievers: tuple[Retriever, ...] | None = None,
        similarity_finder: EntitySimilarityFinder | None = None,
        resolve_entities_on_ingest: bool = False,
        entity_resolution_threshold: float = 0.92,
    ) -> None:
        """Initialize the GraphRAG facade.

        Parameters
        ----------
        model : str
            Chat model used by the knowledge graph extractor.
        embedding_model : str
            Embedding model used for documents, chunks, entities, and queries.
        db : GraphDB
            Graph database adapter used for persistence and search.
        chunking_strategy : {"fast", "recursive", "semantic", "sentence", "token"}, optional
            Chunking strategy used by the internal :class:`~grawiki.doc_processing.chunkers.Chunker`.
        max_workers : int, optional
            Maximum number of concurrent chunk-level extraction coroutines.
        embedding : Embedding | None, optional
            Embedding override for tests, notebooks, or alternate deployments.
        kg_extractor : KnowledgeGraphExtractorProtocol | None, optional
            Knowledge graph extractor override for tests or debugging.
        retrievers : tuple[Retriever, ...] | None, optional
            Retrieval strategies used by :meth:`search`.
        similarity_finder : EntitySimilarityFinder | None, optional
            Entity similarity finder used by the similarity helper methods.
        resolve_entities_on_ingest : bool, optional
            When ``True``, each freshly-extracted entity is compared against
            persisted entities before persistence. Matched nodes are replaced
            with the persisted node and relationship endpoints are rewritten.
            Defaults to ``False``.
        entity_resolution_threshold : float, optional
            Minimum cosine-similarity score for ingest-time entity resolution.
            Only used when ``resolve_entities_on_ingest=True``.
            Defaults to ``0.92``.

        Raises
        ------
        ValueError
            Raised when ``entity_resolution_threshold`` is outside ``[-1.0, 1.0]``,
            which is the valid range for cosine similarity scores.
        """

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
        self._entity_similarity = similarity_finder or EntitySimilarityFinder(db=db)
        self.resolve_entities_on_ingest = resolve_entities_on_ingest
        self.entity_resolution_threshold = entity_resolution_threshold
        if not -1.0 <= entity_resolution_threshold <= 1.0:
            raise ValueError(
                "entity_resolution_threshold must be within [-1.0, 1.0] "
                f"(got {entity_resolution_threshold!r})."
            )
        self._retrievers = retrievers or (
            TextRetriever(db=db, embedding=self._embedding),
            KeywordsPathRetriever(model=model, db=db, embedding=self._embedding),
        )

    # ------------------------------------------------------------------
    # Public step methods (useful for notebooks and debugging)
    # ------------------------------------------------------------------

    def read_document(self, path: Path) -> Document:
        """Load one source document from disk.

        Parameters
        ----------
        path : Path
            Filesystem path to the source document.

        Returns
        -------
        Document
            Loaded source document.
        """

        logger.info("Reading document from %s", path)
        return read_document(path)

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Parameters
        ----------
        document : Document
            Source document to segment.

        Returns
        -------
        list[Chunk]
            Chunk sequence produced by the configured chunker.
        """

        logger.info("Chunking document %s", document.id)
        chunks = chunk_document(document, self._chunker)
        logger.info("Created %s chunks for document %s", len(chunks), document.id)
        return chunks

    async def embed_document(self, document: Document) -> list[float]:
        """Embed one document's content.

        Parameters
        ----------
        document : Document
            Source document whose content should be embedded.

        Returns
        -------
        list[float]
            Embedding vector for the document content.
        """

        logger.info("Embedding document %s", document.id)
        result = await self._embedding.embed_documents([document.content])
        return list(result.embeddings[0])

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed chunk contents in one batch.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks whose content should be embedded.

        Returns
        -------
        list[list[float]]
            Embedding vectors aligned with the input chunk order.
        """

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
        """Build a document node with its embedding attached.

        Parameters
        ----------
        document : Document
            Source document to convert into a persisted node model.
        embedding : list[float]
            Embedding vector for the document.

        Returns
        -------
        DocumentNode
            Prepared document node ready for persistence.
        """

        node = DocumentNode.from_document(document)
        node.embedding = embedding
        return node

    def build_chunk_nodes(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> list[ChunkNode]:
        """Build chunk nodes with embeddings attached.

        Parameters
        ----------
        chunks : list[Chunk]
            Source chunks to convert into persisted node models.
        embeddings : list[list[float]]
            Embedding vectors aligned with ``chunks``.

        Returns
        -------
        list[ChunkNode]
            Prepared chunk nodes ready for persistence.

        Raises
        ------
        ValueError
            Raised when the number of chunks and embeddings does not match.
        """

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
        """Persist one document node and its chunk nodes with indexes.

        Parameters
        ----------
        document_node : DocumentNode
            Prepared document node.
        chunk_nodes : list[ChunkNode]
            Prepared chunk nodes associated with the document.
        """

        dims: dict[str, int] = {}
        if document_node.embedding:
            dims[DocumentNode.system_label] = len(document_node.embedding)
        if chunk_nodes and chunk_nodes[0].embedding:
            dims[ChunkNode.system_label] = len(chunk_nodes[0].embedding)

        logger.info(
            "Persisting document %s and %s chunks", document_node.id, len(chunk_nodes)
        )
        await self._db.setup(embedding_dimensions=dims or None)
        await self._db.save_docs_and_chunks_to_db([document_node], chunk_nodes)

    async def extract_kg_per_chunk(
        self, chunks: list[Chunk]
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
        """Persist extracted entities and relationships.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks that own the extracted graphs.
        chunk_graphs : dict[str, KnowledgeGraph]
            Extracted graphs keyed by chunk identifier.
        """

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

    async def find_similar_entities(
        self,
        entity: Node,
        *,
        limit: int = 10,
        threshold: float | None = None,
        candidates: list[Node] | None = None,
    ) -> list[NodeHit]:
        """Return candidate entities similar to ``entity``.

        Parameters
        ----------
        entity : Node
            Source entity used as the similarity query.
        limit : int, optional
            Maximum number of candidate hits to return.
        threshold : float | None, optional
            Optional strategy-specific minimum score.
        candidates : list[Node] | None, optional
            Optional candidate pool. When omitted, persisted entities are
            loaded from the graph database.

        Returns
        -------
        list[NodeHit]
            Ranked similarity candidates.

        Notes
        -----
        The configured :class:`~grawiki.similarity.similarity_finder.EntitySimilarityFinder`
        decides which concrete matcher implementation is used.
        """

        return await self._entity_similarity.search(
            entity,
            limit=limit,
            threshold=threshold,
            candidates=candidates,
        )

    async def find_entity_collision_candidates(
        self,
        *,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[SemanticKeyCollisionCandidates]:
        """Return semantic-key collision groups annotated with merge candidates.

        Parameters
        ----------
        limit : int, optional
            Maximum number of candidate hits returned per source entity.
        threshold : float | None, optional
            Optional strategy-specific minimum score.

        Returns
        -------
        list[SemanticKeyCollisionCandidates]
            Collision groups with per-entity candidate matches.

        Notes
        -----
        Candidate generation uses the similarity matcher configured on the
        injected entity similarity finder.
        """

        return await self._entity_similarity.find_collision_candidates(
            limit=limit,
            threshold=threshold,
        )

    async def find_entity_duplicate_candidates(
        self,
        *,
        limit: int = 10,
        threshold: float | None = None,
        skip_semantic_key_collisions_in_similarity_scan: bool = True,
    ) -> EntityDuplicateCandidates:
        """Run the two-step duplicate-finding heuristic across entities.

        Parameters
        ----------
        limit : int, optional
            Maximum number of candidate hits returned per source entity.
        threshold : float | None, optional
            Optional matcher-specific minimum score.
        skip_semantic_key_collisions_in_similarity_scan : bool, optional
            Whether the broader similarity scan should exclude entities already
            involved in exact semantic-key collisions.

        Returns
        -------
        EntityDuplicateCandidates
            Combined duplicate-candidate report produced by the injected entity
            similarity finder.
        """

        return await self._entity_similarity.find_duplicate_candidates(
            limit=limit,
            threshold=threshold,
            skip_semantic_key_collisions_in_similarity_scan=skip_semantic_key_collisions_in_similarity_scan,
        )

    async def dedupe_entities(
        self,
        *,
        limit: int = 10,
        threshold: float | None = None,
        min_merge_score: float = 0.95,
        dry_run: bool = False,
    ) -> list[MergeReport]:
        """Find duplicate entities and merge them into canonical masters.

        Parameters
        ----------
        limit : int, optional
            Maximum candidate hits returned per source entity during duplicate
            inspection.
        threshold : float | None, optional
            Optional similarity threshold forwarded to the duplicate finder.
        min_merge_score : float, optional
            Minimum candidate score required for inclusion in a merge group.
        dry_run : bool, optional
            When ``True``, reports are produced without applying destructive DB
            changes.

        Returns
        -------
        list[MergeReport]
            Reports describing the merge decisions that were made.
        """

        candidates = await self.find_entity_duplicate_candidates(
            limit=limit,
            threshold=threshold,
        )
        reports: list[MergeReport] = []
        seen: set[str] = set()

        for collision_group in candidates.semantic_key_collision_candidates:
            group = [
                node
                for node in _flatten_collision_group(
                    collision_group,
                    min_score=min_merge_score,
                )
                if node.id not in seen
            ]
            if len(group) < 2:
                continue
            merged_master, report = await self._build_merge_candidate(
                group,
                source="collision",
            )
            reports.append(report)
            if not dry_run:
                await self._db.merge_entity_nodes(
                    master=merged_master,
                    duplicate_ids=list(report.duplicate_ids),
                )
            seen.update(node.id for node in group)

        for result in candidates.similarity_candidates:
            group = [result.source] + [
                hit.node for hit in result.hits if hit.score >= min_merge_score
            ]
            group = [node for node in group if node.id not in seen]
            if len(group) < 2:
                continue
            merged_master, report = await self._build_merge_candidate(
                group,
                source="similarity",
            )
            reports.append(report)
            if not dry_run:
                await self._db.merge_entity_nodes(
                    master=merged_master,
                    duplicate_ids=list(report.duplicate_ids),
                )
            seen.update(node.id for node in group)

        return reports

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _resolve_extracted_entities(
        self, chunk_graphs: dict[str, KnowledgeGraph]
    ) -> dict[str, KnowledgeGraph]:
        """Replace extracted entities with persisted duplicates when similar.

        Parameters
        ----------
        chunk_graphs : dict[str, KnowledgeGraph]
            Freshly-extracted chunk graphs. Not mutated.

        Returns
        -------
        dict[str, KnowledgeGraph]
            New dict with matched nodes replaced by the persisted node and all
            relationships rewritten accordingly. When no matches are found the
            input is returned unchanged.

        Notes
        -----
        Uses ``self._entity_similarity`` so the same similarity strategy that
        powers the duplicate-candidate helpers is applied during ingestion.
        Real data shows duplicates routinely cross ontology labels.
        """

        # Collect unique extracted entities across all chunks.
        unique: dict[str, Node] = {}
        for graph in chunk_graphs.values():
            for node in graph.nodes:
                unique.setdefault(node.id, node)

        # One similarity search per unique extracted entity.
        resolved: dict[str, Node] = {}
        for ext_id, ext_node in unique.items():
            hits = await self._entity_similarity.search(
                ext_node,
                limit=1,
                threshold=self.entity_resolution_threshold,
            )
            if hits:
                resolved[ext_id] = hits[0].node

        if not resolved:
            return chunk_graphs

        rewritten: dict[str, KnowledgeGraph] = {}
        for cid, graph in chunk_graphs.items():
            new_nodes = [resolved.get(n.id, n) for n in graph.nodes]
            new_rels = [
                Relationship(
                    id=r.id,
                    source=resolved[r.source].id if r.source in resolved else r.source,
                    target=resolved[r.target].id if r.target in resolved else r.target,
                    label=r.label,
                    properties=dict(r.properties),
                )
                for r in graph.relationships
            ]
            rewritten[cid] = KnowledgeGraph(nodes=new_nodes, relationships=new_rels)
        return rewritten

    async def _build_merge_candidate(
        self,
        group: list[Node],
        *,
        source: str,
    ) -> tuple[Node, MergeReport]:
        """Build the merged master payload and report for one duplicate group."""

        relation_counts = await self._db.entity_relationship_counts(
            [node.id for node in group]
        )
        master = pick_master(group, relation_counts)
        duplicates = [node for node in group if node.id != master.id]
        merged_master, conflicts = build_merged_master(master, duplicates)
        return merged_master, MergeReport(
            master_id=merged_master.id,
            duplicate_ids=tuple(node.id for node in duplicates),
            source=source,
            merged_labels=tuple(sorted(merged_master.labels)),
            property_conflicts=conflicts,
        )

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    async def ingest(self, path: Path) -> None:
        """Run the full ingestion flow for one file.

        Parameters
        ----------
        path : Path
            Source file to ingest.

        Returns
        -------
        None
            This method persists the resulting graph side effects to the
            configured database.
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
        if self.resolve_entities_on_ingest:
            chunk_graphs = await self._resolve_extracted_entities(chunk_graphs)
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

        Returns
        -------
        None
            This method persists the resulting graph side effects to the
            configured database.
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
        if self.resolve_entities_on_ingest:
            chunk_graphs = await self._resolve_extracted_entities(chunk_graphs)
        await self.persist_entities_and_relationships(chunks, chunk_graphs)
        logger.info("Completed ingestion for text document %r", title)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[NodeHit]:
        """Search documents, chunks, and entities.

        Parameters
        ----------
        query : str
            Raw user query text.
        limit : int, optional
            Maximum number of results per node family.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated search hits across documents, chunks, and
            entities.

        Raises
        ------
        RuntimeError
            Raised when every configured retriever fails for the query.
        """

        logger.info("Running search for query %r", query)
        node_hit_scores: dict[str, float] = {}
        node_hits: dict[str, NodeHit] = {}
        errors: list[Exception] = []
        for retriever in self._retrievers:
            try:
                hits = await retriever.retrieve(query, limit=limit)
            except Exception as exc:  # pragma: no cover - defensive integration path
                errors.append(exc)
                logger.warning(
                    "Retriever %s failed for query %r: %s",
                    type(retriever).__name__,
                    query,
                    exc,
                )
                continue
            for hit in hits:
                hit_key = hit.node.id
                score = hit.score
                if hit_key in node_hit_scores:
                    if score > node_hit_scores[hit_key]:
                        node_hit_scores[hit_key] = score
                        node_hits[hit_key] = hit
                else:
                    node_hit_scores[hit_key] = score
                    node_hits[hit_key] = hit
        if errors and not node_hits:
            raise RuntimeError("All retrievers failed for the query.") from errors[0]
        sorted_hits = sorted(
            node_hits.values(),
            key=lambda h: node_hit_scores[h.node.id],
            reverse=True,
        )[:limit]
        return sorted_hits


def _flatten_collision_group(
    group: SemanticKeyCollisionCandidates,
    *,
    min_score: float,
) -> list[Node]:
    """Return unique nodes participating in a semantic-key collision group."""

    flattened: dict[str, Node] = {}
    for result in group.results:
        flattened.setdefault(result.source.id, result.source)
        for hit in result.hits:
            if hit.score >= min_score:
                flattened.setdefault(hit.node.id, hit.node)
    return list(flattened.values())
