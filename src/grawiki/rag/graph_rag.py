"""GraphRAG facade — document ingestion and retrieval-augmented search."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic_ai import Embedder

from grawiki.core.commons import Chunk, Document
from grawiki.core.embedding import Embedding
from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.doc_processing.chunkers import Chunker, MarkdownChunker
from grawiki.doc_processing.document_processing import chunk_document, read_document
from grawiki.graph.extraction import (
    KnowledgeGraphExtractor,
    KnowledgeGraphExtractorProtocol,
)
from grawiki.graph.models import (
    ChunkNode,
    DocumentNode,
    KnowledgeGraph,
    MemoryNode,
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
MEMORY_RELATED_LABEL = "__related__"


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
    markdown_chunker : MarkdownChunker | None, optional
        Dedicated markdown-aware chunker used for ``.md`` / ``.markdown`` files
        and in-memory text ingestion when configured.
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
        markdown_chunker: MarkdownChunker | None = None,
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
        markdown_chunker : MarkdownChunker | None, optional
            Dedicated markdown-aware chunker used when the document appears to be markdown.
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
        self._markdown_chunker = markdown_chunker
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
            TextRetriever(
                db=db,
                embedding=self._embedding,
                search_labels=[ChunkNode.system_label, MemoryNode.system_label],
            ),
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

    def chunk_document(
        self,
        document: Document,
        path: Path | None = None,
    ) -> list[Chunk]:
        """Split a document into chunks.

        Parameters
        ----------
        document : Document
            Source document to segment.
        path : Path | None, optional
            Filesystem path to the source document. MarkdownChef is used when a
            markdown chunker is configured and the path suffix is ``.md`` or
            ``.markdown``. When no path is available, configured markdown
            chunking is used for in-memory text.

        Returns
        -------
        list[Chunk]
            Chunk sequence produced by the configured chunker.
        """

        logger.info("Chunking document %s", document.id)
        is_markdown_path = path is not None and path.suffix.lower() in {
            ".md",
            ".markdown",
        }
        if self._markdown_chunker and (path is None or is_markdown_path):
            logger.info("Using dedicated markdown chunker for document %s", document.id)
            chunks = self._markdown_chunker.chunk(document, source_path=path)
        else:
            chunks = chunk_document(document, self._chunker)
        logger.info("Created %s chunks for document %s", len(chunks), document.id)
        return chunks

    async def embed_document(self, document: Document) -> list[float]:
        """Return no document-level embedding for ingestion.

        Parameters
        ----------
        document : Document
            Source document. Kept in the signature for step-method API
            compatibility.

        Returns
        -------
        list[float]
            Empty list. Document content is persisted without a vector; chunk,
            entity, memory, and query embeddings remain the retrieval path.
        """

        logger.info("Skipping document-level embedding for document %s", document.id)
        return []

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
        """Build a document node with an optional embedding attached.

        Parameters
        ----------
        document : Document
            Source document to convert into a persisted node model.
        embedding : list[float]
            Optional embedding vector for the document. The ingestion path now
            passes an empty list, but the parameter is retained for API
            compatibility.

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
                graph = await self._extractor.extract(chunk.content)
                return chunk.id, graph

        results = await asyncio.gather(*(extract_one(c) for c in chunks))
        return dict(results)

    async def persist_entities_and_relationships(
        self,
        owner_ids: Sequence[str],
        owner_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        """Persist extracted entities and relationships.

        Parameters
        ----------
        owner_ids : Sequence[str]
            Node identifiers that own the extracted graphs.
        owner_graphs : dict[str, KnowledgeGraph]
            Extracted graphs keyed by owner identifier.
        """

        entity_dim: int | None = None
        for graph in owner_graphs.values():
            for node in graph.nodes:
                if node.embedding:
                    entity_dim = len(node.embedding)
                    break
            if entity_dim is not None:
                break

        logger.info("Persisting entities for %s owner graphs", len(owner_graphs))
        await self._db.setup(
            embedding_dimensions={"__entity__": entity_dim}
            if entity_dim is not None
            else None
        )
        await self._db.save_entities_and_rels(owner_ids, owner_graphs)

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
        chunks = self.chunk_document(document, path)
        chunk_embeddings = await self.embed_chunks(chunks)
        document_node = self.build_document_node(document, [])
        chunk_nodes = self.build_chunk_nodes(chunks, chunk_embeddings)
        await self.persist_document_and_chunks(document_node, chunk_nodes)
        chunk_graphs = await self.extract_kg_per_chunk(chunks)
        if self.resolve_entities_on_ingest:
            chunk_graphs = await self._resolve_extracted_entities(chunk_graphs)
        await self.persist_entities_and_relationships(
            [chunk.id for chunk in chunks],
            chunk_graphs,
        )
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
        chunk_embeddings = await self.embed_chunks(chunks)
        document_node = self.build_document_node(document, [])
        chunk_nodes = self.build_chunk_nodes(chunks, chunk_embeddings)
        await self.persist_document_and_chunks(document_node, chunk_nodes)
        chunk_graphs = await self.extract_kg_per_chunk(chunks)
        if self.resolve_entities_on_ingest:
            chunk_graphs = await self._resolve_extracted_entities(chunk_graphs)
        await self.persist_entities_and_relationships(
            [chunk.id for chunk in chunks],
            chunk_graphs,
        )
        logger.info("Completed ingestion for text document %r", title)

    async def remember(
        self,
        memory: MemoryNode | str,
        *,
        memory_id: str | None = None,
        name: str | None = None,
        semantic_key: str | None = None,
        metadata: dict[str, str] | None = None,
        related_node_ids: Sequence[str] = (),
    ) -> MemoryNode:
        """Persist one memory, replacing an existing memory when requested.

        Parameters
        ----------
        memory : MemoryNode | str
            Memory payload to persist. Raw strings are normalized into a new
            :class:`~grawiki.graph.models.MemoryNode`.
        memory_id : str | None, optional
            Existing memory identifier to replace. When omitted, ``memory.id`` is
            used as-is.
        name : str | None, optional
            Optional memory name override. Primarily useful when ``memory`` is a
            raw string.
        semantic_key : str | None, optional
            Optional semantic key override. Defaults to the final memory id.
        metadata : dict[str, str] | None, optional
            Optional metadata merged into the memory metadata.
        related_node_ids : Sequence[str], optional
            Existing node ids that should be explicitly linked from the memory.

        Returns
        -------
        MemoryNode
            Persisted memory payload including its final id.
        """

        persisted_memory = self._memory_for_persistence(
            memory,
            memory_id=memory_id,
            name=name,
            semantic_key=semantic_key,
            metadata=metadata,
        )
        logger.info("Remembering memory %s", persisted_memory.id)
        await self._db.setup()
        if memory_id is not None:
            await self._db.delete_memory(memory_id)

        persisted_memory.embedding = await self._embed_text(persisted_memory.content)
        await self._db.setup(
            embedding_dimensions={
                MemoryNode.system_label: len(persisted_memory.embedding)
            }
            if persisted_memory.embedding
            else None
        )
        await self._db.upsert_nodes([persisted_memory])
        await self._persist_memory_relationships(
            persisted_memory.id,
            related_node_ids=related_node_ids,
        )

        memory_graph = await self._extractor.extract(persisted_memory.content)
        owner_graphs = {persisted_memory.id: memory_graph}
        if self.resolve_entities_on_ingest:
            owner_graphs = await self._resolve_extracted_entities(owner_graphs)
        await self.persist_entities_and_relationships(
            [persisted_memory.id], owner_graphs
        )
        logger.info("Completed remember flow for memory %s", persisted_memory.id)
        return persisted_memory

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[NodeHit]:
        """Aggregate results from the configured retrievers.

        Parameters
        ----------
        query : str
            Raw user query text.
        limit : int, optional
            Maximum number of final hits returned after combining retriever
            outputs.

        Returns
        -------
        list[NodeHit]
            Flat, deduplicated search hits across the configured retrievers.
            With the default retriever set this typically includes chunk,
            memory, and keyword-expanded entity results.

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

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
        hops: int = 1,
        limit_per_hop: int = 5,
    ) -> list[NodeHit]:
        """Search memories and attach connected graph context.

        Parameters
        ----------
        query : str
            Raw user query text.
        user_id : str | None, optional
            Optional memory-owner filter applied after memory retrieval.
        limit : int, optional
            Maximum number of memories returned.
        hops : int, optional
            Number of graph-expansion hops to include.
        limit_per_hop : int, optional
            Maximum recall paths expanded per memory seed.
        """

        if limit < 1 or hops < 1:
            return []

        search_limit = limit if user_id is None else max(limit * 5, limit)
        memory_hits = await self._search_memories(query, limit=search_limit)
        if user_id is not None:
            memory_hits = [
                hit for hit in memory_hits if _memory_user_id(hit.node) == user_id
            ]
        memory_hits = memory_hits[:limit]
        if not memory_hits:
            return []

        contexts_by_seed = await self._db.recall_subgraph(
            memory_ids=[hit.node.id for hit in memory_hits],
            hops=hops,
            limit_per_memory=limit_per_hop,
        )
        return [
            NodeHit(
                node=_memory_node_with_context(
                    hit.node, contexts_by_seed.get(hit.node.id, [])
                ),
                score=hit.score,
                matched_on=hit.matched_on,
            )
            for hit in memory_hits
        ]

    async def _embed_text(self, text: str) -> list[float]:
        """Embed one arbitrary text string."""

        result = await self._embedding.embed_documents([text])
        return list(result.embeddings[0])

    def _memory_for_persistence(
        self,
        memory: MemoryNode | str,
        *,
        memory_id: str | None,
        name: str | None = None,
        semantic_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> MemoryNode:
        """Return a persisted memory payload with the canonical identifier."""

        if isinstance(memory, str):
            final_id = memory_id or str(uuid.uuid4())
            return MemoryNode(
                id=final_id,
                semantic_key=semantic_key or final_id,
                name=name or _default_memory_name(memory, final_id),
                content=memory,
                metadata=dict(metadata or {}),
            )

        final_id = memory_id or memory.id
        merged_metadata = dict(memory.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return memory.model_copy(
            update={
                "id": final_id,
                "name": name or memory.name,
                "semantic_key": semantic_key
                or _default_memory_semantic_key(memory, final_id),
                "metadata": merged_metadata,
            },
            deep=True,
        )

    async def _persist_memory_relationships(
        self,
        memory_id: str,
        *,
        related_node_ids: Sequence[str],
    ) -> None:
        """Persist explicit memory links to existing nodes."""

        unique_related_ids = [
            node_id
            for node_id in dict.fromkeys(related_node_ids)
            if node_id and node_id != memory_id
        ]
        if not unique_related_ids:
            return
        await self._db.upsert_relationships(
            [
                Relationship(
                    id=str(uuid.uuid4()),
                    source=memory_id,
                    target=node_id,
                    label=MEMORY_RELATED_LABEL,
                )
                for node_id in unique_related_ids
            ]
        )

    async def _search_memories(self, query: str, *, limit: int) -> list[NodeHit]:
        """Return memory-only hits merged from vector and full-text search."""

        vector_hits: list[NodeHit] = []
        query_result = await self._embedding.embed_query(query)
        if query_result.embeddings:
            vector_hits = await self._db.vector_search(
                labels=[MemoryNode.system_label],
                query_embedding=list(query_result.embeddings[0]),
                limit=limit,
            )

        fulltext_hits = await self._db.fulltext_search(
            labels=[MemoryNode.system_label],
            query_text=query,
            limit=limit,
        )

        ordered_ids: list[str] = []
        best_hits: dict[str, NodeHit] = {}
        for hit in [*vector_hits, *fulltext_hits]:
            node_id = hit.node.id
            if node_id not in best_hits:
                ordered_ids.append(node_id)
                best_hits[node_id] = hit
                continue
            if hit.score > best_hits[node_id].score:
                best_hits[node_id] = hit
        return [best_hits[node_id] for node_id in ordered_ids[:limit]]


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


def _memory_user_id(node: Node) -> str | None:
    """Return the memory owner id stored in metadata when present."""

    metadata = getattr(node, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("user_id")
    return value if isinstance(value, str) else None


def _default_memory_semantic_key(memory: MemoryNode, final_id: str) -> str:
    """Return the default semantic key for a persisted memory."""

    if memory.semantic_key == memory.id:
        return final_id
    return memory.semantic_key


def _default_memory_name(content: str, memory_id: str) -> str:
    """Return a short generated name for a raw-text memory."""

    cleaned_lines = [" ".join(line.split()) for line in content.splitlines()]
    first_non_empty = next((line for line in cleaned_lines if line), "")
    if first_non_empty:
        return first_non_empty[:80]
    return f"Memory {memory_id}"


def _memory_node_with_context(
    node: Node,
    relationships: Sequence[NeighborRelationship],
) -> Node:
    """Attach rendered recall context to one memory node."""

    properties = dict(node.properties)
    properties["recall_context"] = _build_recall_text(relationships)
    return node.model_copy(update={"properties": properties}, deep=True)


def _build_recall_text(relationships: Sequence[NeighborRelationship]) -> str:
    """Render recall graph context as readable text."""

    if not relationships:
        return "No connected graph context found."
    return "\n".join(
        (
            f"{relationship.source_name or relationship.source_id} "
            f"-[{relationship.relationship_label}]-> "
            f"{relationship.target.name or relationship.target.id}"
        )
        for relationship in relationships
    )
