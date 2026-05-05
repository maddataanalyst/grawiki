"""Tests for the GraphRAG facade."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest
from chonkie import Pipeline

from grawiki.core.commons import Chunk, Document
from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.doc_processing import document_processing
from grawiki.doc_processing.chunkers import MarkdownPipelineChunker
from grawiki.doc_processing.document_processing import read_document
from grawiki.graph.extraction import KnowledgeGraphExtractor
from grawiki.graph.models import KnowledgeGraph, MemoryNode, Node, Relationship
from grawiki.rag.graph_rag import GraphRAG
from grawiki.retrieval.text import TextRetriever
from grawiki.similarity.deduplication import MergeReport
from grawiki.similarity.fuzzy import RapidFuzzEntitySimilarityMatcher
from grawiki.similarity.similarity_finder import EntitySimilarityFinder


def build_markdown_pipeline(*, chunk_size: int = 4096) -> Pipeline:
    """Return a deterministic markdown pipeline for tests."""

    return (
        Pipeline()
        .process_with("markdown")
        .chunk_with("token", tokenizer="character", chunk_size=chunk_size)
    )


class FakeEmbeddingResult:
    """Minimal embedding result object for tests."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeEmbedder:
    """Deterministic embedder used for GraphRAG tests."""

    def __init__(self) -> None:
        self.document_inputs: list[list[str]] = []
        self.query_inputs: list[str | list[str]] = []

    async def embed_documents(self, documents: str | list[str]) -> FakeEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        self.document_inputs.append(texts)
        embeddings = [
            [float(len(text)), float(len(text.split())), 1.0] for text in texts
        ]
        return FakeEmbeddingResult(embeddings)

    async def embed_query(self, query: str | list[str]) -> FakeEmbeddingResult:
        self.query_inputs.append(query)
        texts = [query] if isinstance(query, str) else list(query)
        embeddings = [[42.0, float(len(text)), 7.0] for text in texts]
        return FakeEmbeddingResult(embeddings)


class FakeGraphDB(GraphDB):
    """In-memory graph DB stub that records facade interactions."""

    def __init__(self) -> None:
        self.setup_calls: list[dict[str, int] | None] = []
        self.saved_doc_nodes = []
        self.saved_chunk_nodes = []
        self.saved_entities: list[tuple[list[str], dict[str, KnowledgeGraph]]] = []
        self.upserted_nodes: list[Node] = []
        self.upserted_relationships: list[Relationship] = []
        self.deleted_memory_ids: list[str] = []
        self.fulltext_calls: list[tuple[list[str], str, int]] = []
        self.vector_calls: list[tuple[list[str], list[float], int]] = []
        self.recall_calls: list[tuple[list[str], int, int]] = []
        self.fulltext_results: list[NodeHit] = []
        self.vector_results: list[NodeHit] = []
        self.neighbor_map: dict[str, list[NeighborRelationship]] = {}
        self.recall_results: dict[str, list[NeighborRelationship]] = {}
        self.entities: list[Node] = []
        self.relationship_counts: dict[str, int] = {}
        self.merges: list[tuple[Node, tuple[str, ...]]] = []

    async def setup(self, embedding_dimensions: dict[str, int] | None = None) -> None:
        self.setup_calls.append(embedding_dimensions)

    async def save_docs_and_chunks_to_db(self, doc_nodes, chunk_nodes) -> None:
        self.saved_doc_nodes.extend(doc_nodes)
        self.saved_chunk_nodes.extend(chunk_nodes)

    async def save_entities_and_rels(
        self,
        owner_ids: Sequence[str],
        owner_graphs: dict[str, KnowledgeGraph],
    ) -> None:
        self.saved_entities.append((list(owner_ids), owner_graphs))

    async def ensure_indexes(
        self, *, labels: Iterable[str], vector_dims: Mapping[str, int] | None = None
    ) -> None:
        pass

    async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        self.upserted_nodes.extend(node.model_copy(deep=True) for node in nodes)

    async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
        self.upserted_relationships.extend(rel.model_copy(deep=True) for rel in rels)

    async def fulltext_search(
        self, *, labels: Sequence[str], query_text: str, limit: int = 10
    ) -> list[NodeHit]:
        self.fulltext_calls.append((list(labels), query_text, limit))
        return [
            NodeHit(
                node=hit.node.model_copy(deep=True),
                score=hit.score,
                matched_on=hit.matched_on,
            )
            for hit in self.fulltext_results[:limit]
        ]

    async def vector_search(
        self, *, labels: Sequence[str], query_embedding: list[float], limit: int = 10
    ) -> list[NodeHit]:
        self.vector_calls.append((list(labels), list(query_embedding), limit))
        return [
            NodeHit(
                node=hit.node.model_copy(deep=True),
                score=hit.score,
                matched_on=hit.matched_on,
            )
            for hit in self.vector_results[:limit]
        ]

    async def neighbor_relationships(
        self,
        *,
        node_ids: Sequence[str],
        limit_per_node: int = 5,
    ) -> dict[str, list[NeighborRelationship]]:
        return {
            node_id: [
                NeighborRelationship(
                    source_id=relationship.source_id,
                    source_name=relationship.source_name,
                    relationship_label=relationship.relationship_label,
                    target=relationship.target.model_copy(deep=True),
                )
                for relationship in self.neighbor_map.get(node_id, [])[:limit_per_node]
            ]
            for node_id in node_ids
        }

    async def recall_subgraph(
        self,
        *,
        memory_ids: Sequence[str],
        hops: int = 1,
        limit_per_memory: int = 20,
    ) -> dict[str, list[NeighborRelationship]]:
        self.recall_calls.append((list(memory_ids), hops, limit_per_memory))
        return {
            memory_id: [
                NeighborRelationship(
                    source_id=relationship.source_id,
                    source_name=relationship.source_name,
                    relationship_label=relationship.relationship_label,
                    target=relationship.target.model_copy(deep=True),
                )
                for relationship in self.recall_results.get(memory_id, [])
            ]
            for memory_id in memory_ids
        }

    async def list_entities(self, *, include_embeddings: bool = False) -> list[Node]:
        if include_embeddings:
            return [node.model_copy(deep=True) for node in self.entities]
        return [
            node.model_copy(update={"embedding": []}, deep=True)
            for node in self.entities
        ]

    async def entity_relationship_counts(
        self, node_ids: Sequence[str]
    ) -> dict[str, int]:
        return {
            node_id: self.relationship_counts.get(node_id, 0) for node_id in node_ids
        }

    async def merge_entity_nodes(
        self,
        *,
        master: Node,
        duplicate_ids: Sequence[str],
    ) -> None:
        self.merges.append((master.model_copy(deep=True), tuple(duplicate_ids)))

    async def delete_memory(self, memory_id: str) -> None:
        self.deleted_memory_ids.append(memory_id)


class StaticRetriever:
    """Retriever stub returning a predefined hit list."""

    def __init__(self, hits: list[NodeHit]) -> None:
        self.hits = hits

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        return list(self.hits)


class FailingRetriever:
    """Retriever stub that always raises to exercise fail-open search."""

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        raise RuntimeError("retriever failed")


class StaticKeywordExtractor:
    """Keyword extractor stub returning predefined keywords."""

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords

    async def extract(self, query: str) -> list[str]:
        return list(self.keywords)


class ConcurrencyTrackingExtractor:
    """Extractor stub that tracks maximum concurrent executions."""

    def __init__(self) -> None:
        self.current = 0
        self.maximum = 0

    async def extract(self, text: str) -> KnowledgeGraph:
        self.current += 1
        self.maximum = max(self.maximum, self.current)
        await asyncio.sleep(0.01)
        self.current -= 1
        key = str(len(text))
        return KnowledgeGraph(
            nodes=[
                Node(
                    id=f"node_{key}",
                    label="Concept",
                    semantic_key=f"concept_{key}",
                    name=f"Concept {key}",
                    embedding=[1.0, 0.0, 0.0],
                )
            ]
        )


class ProgressTrackingExtractor:
    """Extractor stub that completes chunks out of input order."""

    async def extract(self, text: str) -> KnowledgeGraph:
        await asyncio.sleep(float(text))
        return KnowledgeGraph(
            nodes=[
                Node(
                    id=f"node_{text}",
                    label="Concept",
                    semantic_key=f"concept_{text}",
                    name=f"Concept {text}",
                )
            ]
        )


class CancelTrackingFailingExtractor:
    """Extractor stub that fails once and records cancellation of other work."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancelled: list[str] = []

    async def extract(self, text: str) -> KnowledgeGraph:
        self.started.append(text)
        if text == "fail":
            await asyncio.sleep(0)
            raise RuntimeError("chunk extraction failed")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise


class RecordingExtractor:
    """Extractor stub that records every chunk text it receives."""

    def __init__(self, graph: KnowledgeGraph | None = None) -> None:
        self._graph = graph or KnowledgeGraph()
        self.texts: list[str] = []

    async def extract(self, text: str) -> KnowledgeGraph:
        self.texts.append(text)
        return self._graph


class PrefixingProcessor:
    """Chunk processor stub that prepends a deterministic marker."""

    def __init__(
        self, prefix: str, call_log: list[tuple[str, str]] | None = None
    ) -> None:
        self.prefix = prefix
        self.call_log = call_log

    async def __call__(self, chunk: Chunk) -> Chunk:
        if self.call_log is not None:
            self.call_log.append((self.prefix, chunk.id))
        return chunk.model_copy(
            update={
                "content": f"{self.prefix}{chunk.content}",
                "metadata": dict(chunk.metadata),
            }
        )


def build_chunks(*contents: str) -> list[Chunk]:
    """Create deterministic test chunks from raw text contents."""

    return [
        Chunk(
            id=f"chunk_{index}",
            document_id="document_1",
            content=content,
            doc_position=index,
        )
        for index, content in enumerate(contents, start=1)
    ]


def test_graph_rag_default_extractor_accepts_language_configuration() -> None:
    """GraphRAG should pass language and API kwargs into the default extractor."""

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=FakeGraphDB(),
        embedding=FakeEmbedder(),
        kg_output_language="Polish",
        kg_extractor_kwargs={"reasoning_effort": "minimal"},
        retrievers=(StaticRetriever([]),),
    )

    assert isinstance(rag._extractor, KnowledgeGraphExtractor)
    assert rag._extractor.output_language == "Polish"
    assert rag._extractor._extract_kwargs == {"reasoning_effort": "minimal"}


def test_graph_rag_step_methods_and_ingest(tmp_path: Path) -> None:
    """GraphRAG should expose modular public steps and end-to-end ingestion."""

    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "Alan Turing inspired graph memory.\nGraphs connect chunks.\n"
    )
    search_labels = ["__chunk__", "__entity__"]

    graph_db = FakeGraphDB()
    embedder = FakeEmbedder()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=2,
        embedding=embedder,
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=search_labels,
            ),
        ),
    )

    document = rag.read_document(input_path)
    chunks = rag.chunk_document(document)
    chunks = asyncio.run(rag.process_chunks(chunks))
    document_embedding = asyncio.run(rag.embed_document(document))
    assert document_embedding == []
    assert embedder.document_inputs == []
    chunk_embeddings = asyncio.run(rag.embed_chunks(chunks))
    document_node = rag.build_document_node(document, document_embedding)
    chunk_nodes = rag.build_chunk_nodes(chunks, chunk_embeddings)
    asyncio.run(rag.persist_document_and_chunks(document_node, chunk_nodes))
    chunk_graphs = asyncio.run(rag.extract_kg_per_chunk(chunks))
    asyncio.run(
        rag.persist_entities_and_relationships(
            [chunk.id for chunk in chunks], chunk_graphs
        )
    )

    assert len(embedder.document_inputs) == 1
    assert document_node.embedding == document_embedding
    assert len(chunk_nodes) == len(chunks)
    assert graph_db.saved_doc_nodes[0].id == document.id
    assert len(graph_db.saved_chunk_nodes) == len(chunks)
    assert graph_db.saved_entities[0][1].keys() == chunk_graphs.keys()

    # End-to-end via ingest()
    graph_db = FakeGraphDB()
    extractor = ConcurrencyTrackingExtractor()
    embedder = FakeEmbedder()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=2,
        embedding=embedder,
        kg_extractor=extractor,
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=search_labels,
            ),
        ),
    )
    asyncio.run(rag.ingest(input_path))

    assert len(graph_db.saved_doc_nodes) == 1
    assert len(graph_db.saved_chunk_nodes) > 0
    assert len(graph_db.saved_entities) == 1
    assert graph_db.setup_calls[0] is None
    assert graph_db.setup_calls[1] == {"__chunk__": 3}
    assert graph_db.setup_calls[2] == {"__entity__": 3}
    assert len(embedder.document_inputs) == 1
    assert extractor.maximum <= 2


def test_graph_rag_process_chunks_executes_processors_in_order() -> None:
    """Configured chunk processors should run sequentially and preserve order."""

    graph_db = FakeGraphDB()
    call_log: list[tuple[str, str]] = []
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
        chunk_processors=[
            PrefixingProcessor("first:", call_log),
            PrefixingProcessor("second:", call_log),
        ],
    )

    processed = asyncio.run(rag.process_chunks(build_chunks("alpha", "beta")))

    assert [chunk.content for chunk in processed] == [
        "second:first:alpha",
        "second:first:beta",
    ]
    assert call_log == [
        ("first:", "chunk_1"),
        ("first:", "chunk_2"),
        ("second:", "chunk_1"),
        ("second:", "chunk_2"),
    ]


def test_graph_rag_ingest_text_uses_processed_chunk_content_everywhere() -> None:
    """Chunk processor output should drive embedding, persistence, and extraction."""

    graph_db = FakeGraphDB()
    embedder = FakeEmbedder()
    extractor = RecordingExtractor()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=embedder,
        kg_extractor=extractor,
        retrievers=(StaticRetriever([]),),
        chunk_processors=[PrefixingProcessor("processed: ")],
    )

    asyncio.run(rag.ingest_text("Alpha.", title="Processed"))

    assert len(embedder.document_inputs) == 1
    assert embedder.document_inputs[0] == extractor.texts
    assert [node.content for node in graph_db.saved_chunk_nodes] == extractor.texts
    assert extractor.texts == ["processed: Alpha."]


def test_graph_rag_stepwise_process_chunks_matches_one_shot_ingestion() -> None:
    """Explicit stepwise ingestion should match the one-shot processing flow."""

    one_shot_db = FakeGraphDB()
    one_shot_embedder = FakeEmbedder()
    one_shot_extractor = RecordingExtractor()
    one_shot_rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=one_shot_db,
        embedding=one_shot_embedder,
        kg_extractor=one_shot_extractor,
        retrievers=(StaticRetriever([]),),
        chunk_processors=[PrefixingProcessor("processed: ")],
    )

    stepwise_db = FakeGraphDB()
    stepwise_embedder = FakeEmbedder()
    stepwise_extractor = RecordingExtractor()
    stepwise_rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=stepwise_db,
        embedding=stepwise_embedder,
        kg_extractor=stepwise_extractor,
        retrievers=(StaticRetriever([]),),
        chunk_processors=[PrefixingProcessor("processed: ")],
    )

    asyncio.run(one_shot_rag.ingest_text("Alpha.", title="Processed"))

    document = Document(
        id="doc-1",
        title="Processed",
        content="Alpha.",
        metadata={"source_format": "text", "content_format": "text"},
    )
    asyncio.run(stepwise_db.setup())
    chunks = stepwise_rag.chunk_document(document)
    chunks = asyncio.run(stepwise_rag.process_chunks(chunks))
    document_embedding = asyncio.run(stepwise_rag.embed_document(document))
    chunk_embeddings = asyncio.run(stepwise_rag.embed_chunks(chunks))
    document_node = stepwise_rag.build_document_node(document, document_embedding)
    chunk_nodes = stepwise_rag.build_chunk_nodes(chunks, chunk_embeddings)
    asyncio.run(stepwise_rag.persist_document_and_chunks(document_node, chunk_nodes))
    chunk_graphs = asyncio.run(stepwise_rag.extract_kg_per_chunk(chunks))
    asyncio.run(
        stepwise_rag.persist_entities_and_relationships(
            [chunk.id for chunk in chunks],
            chunk_graphs,
        )
    )

    assert one_shot_db.setup_calls == stepwise_db.setup_calls
    assert one_shot_embedder.document_inputs == stepwise_embedder.document_inputs
    assert one_shot_extractor.texts == stepwise_extractor.texts
    assert [node.content for node in one_shot_db.saved_chunk_nodes] == [
        node.content for node in stepwise_db.saved_chunk_nodes
    ]


def test_graph_rag_extract_kg_per_chunk_without_progress_preserves_return_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """extract_kg_per_chunk() should keep its existing shape by default."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
    )
    chunks = build_chunks("0.01", "0.01", "0.01")

    with caplog.at_level(logging.INFO, logger="grawiki.rag.graph_rag"):
        chunk_graphs = asyncio.run(rag.extract_kg_per_chunk(chunks))

    assert set(chunk_graphs) == {chunk.id for chunk in chunks}
    assert all(isinstance(graph, KnowledgeGraph) for graph in chunk_graphs.values())
    assert "Extracting knowledge graphs for 3 chunks with max_workers=4" in caplog.text
    assert "Extracted KG for chunk" not in caplog.text
    assert "Completed KG extraction" not in caplog.text


def test_graph_rag_extract_kg_per_chunk_with_progress_logs_each_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """extract_kg_per_chunk(show_progress=True) should log start, progress, finish."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=2,
        embedding=FakeEmbedder(),
        kg_extractor=ProgressTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
    )
    chunks = build_chunks("0.03", "0.01", "0.02")

    with caplog.at_level(logging.INFO, logger="grawiki.rag.graph_rag"):
        chunk_graphs = asyncio.run(rag.extract_kg_per_chunk(chunks, show_progress=True))

    progress_messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("Extracted KG for chunk ")
    ]

    assert set(chunk_graphs) == {chunk.id for chunk in chunks}
    assert "Extracting knowledge graphs for 3 chunks with max_workers=2" in caplog.text
    assert progress_messages == [
        "Extracted KG for chunk 1/3",
        "Extracted KG for chunk 2/3",
        "Extracted KG for chunk 3/3",
    ]
    assert "Completed KG extraction for 3 chunks" in caplog.text


def test_graph_rag_extract_kg_per_chunk_failure_cancels_remaining_tasks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Chunk extraction should cancel remaining work and re-raise the failure."""

    graph_db = FakeGraphDB()
    extractor = CancelTrackingFailingExtractor()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        max_workers=3,
        embedding=FakeEmbedder(),
        kg_extractor=extractor,
        retrievers=(StaticRetriever([]),),
    )
    chunks = build_chunks("slow-1", "fail", "slow-2")

    with caplog.at_level(logging.INFO, logger="grawiki.rag.graph_rag"):
        with pytest.raises(RuntimeError, match="chunk extraction failed"):
            asyncio.run(rag.extract_kg_per_chunk(chunks, show_progress=True))

    assert set(extractor.started) == {"slow-1", "fail", "slow-2"}
    assert sorted(extractor.cancelled) == ["slow-1", "slow-2"]
    assert "Extracting knowledge graphs for 3 chunks with max_workers=3" in caplog.text
    assert "Completed KG extraction for 3 chunks" not in caplog.text


def test_graph_rag_ingest_passes_show_progress_to_chunk_extraction(
    tmp_path: Path,
) -> None:
    """ingest(show_progress=True) should forward the flag to chunk extraction."""

    input_path = tmp_path / "input.txt"
    input_path.write_text("Alpha.\nBeta.\n")

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )
    extract_calls: list[bool] = []

    async def fake_extract_kg_per_chunk(
        chunks: list[Chunk],
        *,
        show_progress: bool = False,
    ) -> dict[str, KnowledgeGraph]:
        extract_calls.append(show_progress)
        return {chunk.id: KnowledgeGraph() for chunk in chunks}

    rag.extract_kg_per_chunk = fake_extract_kg_per_chunk  # type: ignore[method-assign]

    asyncio.run(rag.ingest(input_path, show_progress=True))

    assert extract_calls == [True]


def test_graph_rag_ingest_text_passes_show_progress_to_chunk_extraction() -> None:
    """ingest_text(show_progress=True) should forward the flag to chunk extraction."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )
    extract_calls: list[bool] = []

    async def fake_extract_kg_per_chunk(
        chunks: list[Chunk],
        *,
        show_progress: bool = False,
    ) -> dict[str, KnowledgeGraph]:
        extract_calls.append(show_progress)
        return {chunk.id: KnowledgeGraph() for chunk in chunks}

    rag.extract_kg_per_chunk = fake_extract_kg_per_chunk  # type: ignore[method-assign]

    asyncio.run(
        rag.ingest_text(
            "Alpha.\nBeta.\n",
            title="In-memory text",
            show_progress=True,
        )
    )

    assert extract_calls == [True]


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_read_document_marks_markdown_files(tmp_path: Path, suffix: str) -> None:
    """read_document() should mark markdown sources at load time."""

    input_path = tmp_path / f"input{suffix}"
    input_path.write_text("# Heading\n\nMarkdown body.\n")

    document = read_document(input_path)

    assert document.content == "# Heading\n\nMarkdown body.\n"
    assert document.metadata == {
        "filepath": str(input_path),
        "source_format": "markdown",
        "content_format": "markdown",
    }


def test_read_document_converts_pdf_to_markdown_in_memory(
    tmp_path: Path, monkeypatch
) -> None:
    """read_document() should convert PDFs to markdown without sidecar files."""

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4 placeholder")
    calls: list[Path] = []

    def fake_to_markdown(path: Path) -> str:
        calls.append(path)
        return "# PDF\n\nConverted markdown.\n"

    monkeypatch.setattr(
        document_processing.pymupdf4llm, "to_markdown", fake_to_markdown
    )

    document = read_document(input_path)

    assert calls == [input_path]
    assert document.content == "# PDF\n\nConverted markdown.\n"
    assert document.metadata == {
        "filepath": str(input_path),
        "source_format": "pdf",
        "content_format": "markdown",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.pdf"]


def test_markdown_pipeline_chunker_preserves_order_and_metadata(tmp_path: Path) -> None:
    """Markdown chunking should preserve prose, code, and table ordering."""

    input_path = tmp_path / "input.md"
    document = Document(
        id="doc-1",
        title="Markdown",
        content=(
            "# Heading\n\n"
            "Alpha beta gamma delta epsilon zeta eta theta.\n\n"
            "```python\n"
            "print(1)\n"
            "print(2)\n"
            "```\n\n"
            "| a | b |\n"
            "| - | - |\n"
            "| 1 | 2 |\n"
        ),
        metadata={"filepath": str(input_path), "content_format": "markdown"},
    )
    chunker = MarkdownPipelineChunker(build_markdown_pipeline(chunk_size=30))

    chunks = chunker.chunk(document)

    assert [chunk.metadata["type"] for chunk in chunks] == [
        "text",
        "text",
        "code",
        "table",
    ]
    assert [int(chunk.metadata["start_index"]) for chunk in chunks] == sorted(
        int(chunk.metadata["start_index"]) for chunk in chunks
    )
    assert all(chunk.metadata["source_path"] == str(input_path) for chunk in chunks)
    assert chunks[0].content.startswith("# Heading")
    assert chunks[2].content.startswith("print(1)")
    assert chunks[3].content.startswith("| a | b |")


def test_markdown_pipeline_chunker_keeps_large_code_and_tables_intact() -> None:
    """Structured markdown blocks should stay intact in explicit markdown mode."""

    document = Document(
        id="doc-1",
        title="Markdown",
        content=(
            "# Heading\n\n"
            "Intro paragraph.\n\n"
            "```python\n"
            "print(1234567890)\n"
            "print(abcdefghij)\n"
            "```\n\n"
            "| a | b |\n"
            "| - | - |\n"
            "| 1 | 2 |\n"
            "| 3 | 4 |\n"
        ),
        metadata={"content_format": "markdown"},
    )
    chunker = MarkdownPipelineChunker(build_markdown_pipeline(chunk_size=25))

    chunks = chunker.chunk(document)

    code_chunks = [chunk for chunk in chunks if chunk.metadata["type"] == "code"]
    table_chunks = [chunk for chunk in chunks if chunk.metadata["type"] == "table"]

    assert len(code_chunks) == 1
    assert len(table_chunks) == 1
    assert len(code_chunks[0].content) > 25
    assert len(table_chunks[0].content) > 25


def test_markdown_pipeline_chunker_accepts_non_markdown_pipeline() -> None:
    """MarkdownPipelineChunker should also normalize plain pipeline output."""

    document = Document(
        id="doc-1",
        title="Markdown-ish",
        content="# Heading\n\n```python\nprint(1)\n```\n",
        metadata={"content_format": "markdown"},
    )
    chunker = MarkdownPipelineChunker(
        Pipeline()
        .process_with("text")
        .chunk_with("token", tokenizer="character", chunk_size=12)
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert all("type" not in chunk.metadata for chunk in chunks)
    assert [int(chunk.metadata["start_index"]) for chunk in chunks] == sorted(
        int(chunk.metadata["start_index"]) for chunk in chunks
    )


def test_graph_rag_ingest_text_with_text_format_uses_plain_chunker() -> None:
    """ingest_text(format='text') should stay on the generic text chunker."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
        markdown_pipeline=build_markdown_pipeline(),
    )

    asyncio.run(
        rag.ingest_text(
            "# Markdown-ish text\n\n```python\nprint(3)\n```\n",
            title="Plain text",
            format="text",
        )
    )

    assert all("type" not in node.metadata for node in graph_db.saved_chunk_nodes)
    assert graph_db.saved_doc_nodes[0].metadata["content_format"] == "text"


def test_graph_rag_chunk_document_falls_back_to_plain_chunker_for_markdown() -> None:
    """Markdown should use the generic chunker unless a pipeline is configured."""

    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=FakeGraphDB(),
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )
    document = Document(
        id="doc-1",
        title="Markdown fallback",
        content="# Heading\n\nText.\n\n```python\nprint(3)\n```\n",
        metadata={"content_format": "markdown"},
    )

    chunks = rag.chunk_document(document)

    assert chunks
    assert all("type" not in chunk.metadata for chunk in chunks)
    assert any("```python" in chunk.content for chunk in chunks)


def test_graph_rag_ingest_text_with_markdown_format_uses_markdown_pipeline() -> None:
    """ingest_text(format='markdown') should use the markdown pipeline."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
        markdown_pipeline=build_markdown_pipeline(),
    )

    asyncio.run(
        rag.ingest_text(
            "# In-memory markdown\n\n"
            "Text.\n\n"
            "```python\n"
            "print(3)\n"
            "```\n\n"
            "| k | v |\n"
            "| - | - |\n"
            "| a | b |\n",
            title="Markdown text",
            format="markdown",
        )
    )

    assert [node.metadata["type"] for node in graph_db.saved_chunk_nodes] == [
        "text",
        "code",
        "table",
    ]
    assert graph_db.saved_doc_nodes[0].embedding == []
    assert graph_db.saved_doc_nodes[0].metadata["content_format"] == "markdown"
    assert graph_db.setup_calls[1] == {"__chunk__": 3}


def test_graph_rag_ingest_and_ingest_text_share_markdown_ingestion_behavior(
    tmp_path: Path,
) -> None:
    """File and in-memory markdown ingestion should share the same core flow."""

    markdown = (
        "# Shared markdown\n\n"
        "Text.\n\n"
        "```python\n"
        "print(4)\n"
        "```\n\n"
        "| k | v |\n"
        "| - | - |\n"
        "| a | b |\n"
    )
    input_path = tmp_path / "shared.md"
    input_path.write_text(markdown)

    file_db = FakeGraphDB()
    file_rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=file_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
        markdown_pipeline=build_markdown_pipeline(),
    )
    text_db = FakeGraphDB()
    text_rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=text_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
        markdown_pipeline=build_markdown_pipeline(),
    )

    asyncio.run(file_rag.ingest(input_path))
    asyncio.run(text_rag.ingest_text(markdown, title="shared", format="markdown"))

    assert file_db.setup_calls == text_db.setup_calls
    assert [node.content for node in file_db.saved_chunk_nodes] == [
        node.content for node in text_db.saved_chunk_nodes
    ]
    assert [node.metadata["type"] for node in file_db.saved_chunk_nodes] == [
        node.metadata["type"] for node in text_db.saved_chunk_nodes
    ]
    assert len(file_db.saved_entities) == len(text_db.saved_entities) == 1


def test_graph_rag_ingest_routes_pdf_markdown_through_markdown_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Markdown converted from PDF should flow through markdown chunking."""

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4 placeholder")

    monkeypatch.setattr(
        document_processing.pymupdf4llm,
        "to_markdown",
        lambda path: (
            "# PDF markdown\n\n"
            "Text.\n\n"
            "```python\n"
            "print(5)\n"
            "```\n\n"
            "| a | b |\n"
            "| - | - |\n"
            "| 1 | 2 |\n"
        ),
    )

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
        markdown_pipeline=build_markdown_pipeline(),
    )

    asyncio.run(rag.ingest(input_path))

    assert graph_db.saved_doc_nodes[0].metadata["source_format"] == "pdf"
    assert graph_db.saved_doc_nodes[0].metadata["content_format"] == "markdown"
    assert [node.metadata["type"] for node in graph_db.saved_chunk_nodes] == [
        "text",
        "code",
        "table",
    ]


def test_graph_rag_search_uses_configured_retrievers() -> None:
    """GraphRAG.search should delegate through the configured retrievers."""

    graph_db = FakeGraphDB()
    labels = ["__chunk__", "__entity__"]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="fulltext",
                search_labels=labels,
            ),
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=labels,
            ),
        ),
    )

    asyncio.run(rag.search("machine intelligence", limit=5))

    assert graph_db.fulltext_calls[0] == (labels, "machine intelligence", 5)
    assert graph_db.vector_calls[0] == (
        labels,
        [42.0, float(len("machine intelligence")), 7.0],
        5,
    )


def test_default_graph_rag_retrievers_do_not_vector_search_documents() -> None:
    """Default GraphRAG search should not request document vector search."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
    )
    rag._retrievers[1].keyword_extractor = StaticKeywordExtractor(["Turing"])

    asyncio.run(rag.search("machine intelligence", limit=5))

    assert graph_db.vector_calls
    assert all("__document__" not in labels for labels, _, _ in graph_db.vector_calls)
    assert [labels for labels, _, _ in graph_db.vector_calls] == [
        ["__chunk__", "__memory__"],
        ["__entity__"],
    ]


def test_graph_rag_search_keeps_highest_score_and_distinct_labels() -> None:
    """GraphRAG should merge retriever results by label-plus-id and max score."""

    graph_db = FakeGraphDB()
    shared_chunk_low = NodeHit(
        node=Node(
            id="shared",
            label="__chunk__",
            semantic_key="chunk_shared",
            name="Chunk shared",
        ),
        score=0.4,
        matched_on="fulltext",
    )
    shared_chunk_high = NodeHit(
        node=Node(
            id="shared",
            label="__chunk__",
            semantic_key="chunk_shared",
            name="Chunk shared",
        ),
        score=0.9,
        matched_on="vector",
    )
    same_id_entity = NodeHit(
        node=Node(
            id="shared",
            label="Person",
            semantic_key="person_shared",
            name="Shared Person",
        ),
        score=0.7,
        matched_on="keyword_path",
    )
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            StaticRetriever([shared_chunk_low]),
            StaticRetriever([shared_chunk_high, same_id_entity]),
        ),
    )

    hits = asyncio.run(rag.search("machine intelligence", limit=5))

    assert [
        (tuple(sorted(hit.node.labels)), hit.node.id, hit.score) for hit in hits
    ] == [
        (("__chunk__",), "shared", 0.9),
    ]


def test_graph_rag_search_falls_back_to_working_retrievers() -> None:
    """GraphRAG should return partial results when one retriever fails."""

    graph_db = FakeGraphDB()
    surviving_hit = NodeHit(
        node=Node(
            id="entity_1",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan Turing",
        ),
        score=0.8,
        matched_on="keyword_path",
    )
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(FailingRetriever(), StaticRetriever([surviving_hit])),
    )

    hits = asyncio.run(rag.search("machine intelligence", limit=5))

    assert hits == [surviving_hit]


def test_graph_rag_search_raises_when_all_retrievers_fail() -> None:
    """GraphRAG should fail when every configured retriever errors."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(FailingRetriever(),),
    )

    with pytest.raises(RuntimeError, match="All retrievers failed"):
        asyncio.run(rag.search("machine intelligence", limit=5))


def test_graph_rag_search_can_return_memory_hits() -> None:
    """GraphRAG.search should include memory hits when retrievers return them."""

    graph_db = FakeGraphDB()
    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Tooling preference",
        content="Use uv for package management.",
        metadata={"user_id": "user-1"},
    )
    graph_db.vector_results = [NodeHit(node=memory, score=0.85, matched_on="vector")]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(
            TextRetriever(
                db=graph_db,
                embedding=FakeEmbedder(),
                search_method="vector",
                search_labels=["__chunk__", "__memory__"],
            ),
        ),
    )

    hits = asyncio.run(rag.search("tooling", limit=5))

    assert len(hits) == 1
    assert hits[0].node.id == "memory_1"
    assert graph_db.vector_calls[0][0] == ["__chunk__", "__memory__"]


def test_graph_rag_remember_persists_memory_without_chunking() -> None:
    """remember() should persist a memory node and owner-linked entity graph."""

    graph_db = FakeGraphDB()
    memory_graph = KnowledgeGraph(
        nodes=[
            Node(
                id="entity_1",
                label="Concept",
                semantic_key="concept_uv",
                name="uv",
                embedding=[1.0, 0.0, 0.0],
            )
        ],
        relationships=[],
    )
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(memory_graph),
        retrievers=(StaticRetriever([]),),
    )
    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Tooling preference",
        content="Use uv for package management.",
        metadata={"user_id": "user-1"},
    )

    asyncio.run(rag.remember(memory))

    assert graph_db.deleted_memory_ids == []
    assert len(graph_db.upserted_nodes) == 1
    persisted_memory = graph_db.upserted_nodes[0]
    assert isinstance(persisted_memory, MemoryNode)
    assert persisted_memory.id == "memory_1"
    assert persisted_memory.embedding
    assert graph_db.saved_entities[-1][0] == ["memory_1"]
    assert list(graph_db.saved_entities[-1][1]) == ["memory_1"]


def test_graph_rag_remember_replaces_existing_memory_by_id() -> None:
    """remember(memory_id=...) should replace the existing memory id."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )
    memory = MemoryNode(
        id="fresh-memory",
        semantic_key="fresh-memory",
        name="Updated memory",
        content="Replaced content.",
        metadata={"user_id": "user-1"},
    )

    asyncio.run(rag.remember(memory, memory_id="memory_123"))

    assert graph_db.deleted_memory_ids == ["memory_123"]
    assert graph_db.upserted_nodes[0].id == "memory_123"
    assert graph_db.upserted_nodes[0].semantic_key == "memory_123"
    assert graph_db.saved_entities[-1][0] == ["memory_123"]


def test_graph_rag_remember_accepts_raw_text_and_generates_memory_defaults() -> None:
    """remember() should accept raw text and synthesize the memory payload."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )

    persisted_memory = asyncio.run(
        rag.remember(
            "User prefers uv for Python package management.",
            metadata={"user_id": "user-1"},
        )
    )

    assert isinstance(persisted_memory, MemoryNode)
    assert persisted_memory.id
    assert persisted_memory.semantic_key == persisted_memory.id
    assert persisted_memory.name == "User prefers uv for Python package management."
    assert persisted_memory.metadata == {"user_id": "user-1"}
    assert graph_db.upserted_nodes[0].id == persisted_memory.id


def test_graph_rag_remember_raw_text_reuses_update_id() -> None:
    """Raw-text remember should use memory_id as the canonical persisted id."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )

    persisted_memory = asyncio.run(
        rag.remember(
            "Updated memory text.",
            memory_id="memory_123",
            metadata={"user_id": "user-1"},
        )
    )

    assert graph_db.deleted_memory_ids == ["memory_123"]
    assert persisted_memory.id == "memory_123"
    assert persisted_memory.semantic_key == "memory_123"
    assert graph_db.upserted_nodes[0].id == "memory_123"


def test_graph_rag_remember_persists_explicit_links_to_related_node_ids() -> None:
    """remember() should add explicit memory links to any existing node ids."""

    graph_db = FakeGraphDB()
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(KnowledgeGraph()),
        retrievers=(StaticRetriever([]),),
    )
    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Agent memory",
        content="Remember links.",
        metadata={"user_id": "user-1"},
    )

    asyncio.run(
        rag.remember(
            memory,
            related_node_ids=["entity_1", "chunk_1", "entity_1", "memory_1"],
        )
    )

    assert [
        (relationship.source, relationship.target, relationship.label)
        for relationship in graph_db.upserted_relationships
    ] == [
        ("memory_1", "entity_1", "__related__"),
        ("memory_1", "chunk_1", "__related__"),
    ]


def test_graph_rag_recall_filters_memories_and_adds_k_hop_context() -> None:
    """recall() should search only memories, post-filter by user_id, and expand context."""

    graph_db = FakeGraphDB()
    memory_user_1 = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Agent memory",
        content="ReAct and worker agents.",
        metadata={"user_id": "user-1"},
    )
    memory_user_2 = MemoryNode(
        id="memory_2",
        semantic_key="memory_2",
        name="Other memory",
        content="Different user memory.",
        metadata={"user_id": "user-2"},
    )
    entity = Node(
        id="entity_1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
    )
    worker = Node(
        id="entity_2",
        label="Concept",
        semantic_key="concept_worker",
        name="Worker",
    )
    graph_db.vector_results = [
        NodeHit(node=memory_user_1, score=0.9, matched_on="vector"),
        NodeHit(node=memory_user_2, score=0.8, matched_on="vector"),
    ]
    graph_db.fulltext_results = [
        NodeHit(node=memory_user_2, matched_on="fulltext"),
    ]
    graph_db.recall_results = {
        "memory_1": [
            NeighborRelationship(
                source_id="memory_1",
                source_name="Agent memory",
                relationship_label="__mentions__",
                target=entity,
            ),
            NeighborRelationship(
                source_id="entity_1",
                source_name="ReAct",
                relationship_label="RELATED_TO",
                target=worker,
            ),
        ],
    }
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
    )

    hits = asyncio.run(
        rag.recall("react", user_id="user-1", limit=5, hops=2, limit_per_hop=5)
    )

    assert graph_db.vector_calls[0][0] == ["__memory__"]
    assert graph_db.fulltext_calls[0][0] == ["__memory__"]
    assert graph_db.recall_calls == [(["memory_1"], 2, 5)]
    assert [hit.node.id for hit in hits] == ["memory_1"]
    assert hits[0].node.properties["recall_context"] == (
        "Agent memory -[__mentions__]-> ReAct\nReAct -[RELATED_TO]-> Worker"
    )


def test_graph_rag_similarity_helpers_use_injected_similarity_finder() -> None:
    """GraphRAG should expose entity similarity helpers via the injected finder."""

    graph_db = FakeGraphDB()
    graph_db.entities = [
        Node(
            id="entity_1",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan Turing",
            embedding=[1.0, 0.0, 0.0],
        ),
        Node(
            id="entity_2",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan M. Turing",
            embedding=[0.9, 0.1, 0.0],
        ),
    ]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
        similarity_finder=EntitySimilarityFinder(
            graph_db,
            matcher=RapidFuzzEntitySimilarityMatcher(db=graph_db),
        ),
    )

    hits = asyncio.run(
        rag.find_similar_entities(
            graph_db.entities[0],
            threshold=85.0,
        )
    )
    collisions = asyncio.run(rag.find_entity_collision_candidates(threshold=85.0))

    assert [hit.node.id for hit in hits] == ["entity_2"]
    assert [group.semantic_key for group in collisions] == ["person_alan-turing"]


class StaticExtractor:
    """Extractor stub that always returns a pre-built KnowledgeGraph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    async def extract(self, text: str) -> KnowledgeGraph:
        return self._graph


class _CallCountingFinder:
    """Test-only proxy that records every search() call made against an EntitySimilarityFinder."""

    def __init__(self, inner: EntitySimilarityFinder) -> None:
        self._inner = inner
        self.search_calls: list[Node] = []

    async def search(
        self,
        entity: Node,
        *,
        limit: int = 10,
        threshold: float | None = None,
        candidates: list[Node] | None = None,
    ) -> list[NodeHit]:
        self.search_calls.append(entity)
        return await self._inner.search(
            entity,
            limit=limit,
            threshold=threshold,
            candidates=candidates,
        )

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def test_graph_rag_resolves_entities_on_ingest_when_enabled() -> None:
    """When resolve_entities_on_ingest=True, extracted nodes matching a persisted
    node above the threshold should be swapped for the persisted node before
    persistence, and relationship endpoints should be rewritten accordingly.
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    extracted_node = Node(
        id="extracted-1",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )
    # Cosine([0.1,0.2,0.31], [0.1,0.2,0.3]) ≈ 0.99998, well above 0.9 threshold.

    extracted_kg = KnowledgeGraph(
        nodes=[extracted_node],
        relationships=[
            Relationship(
                id="rel-1",
                source="extracted-1",
                target="extracted-1",
                label="RELATED_TO",
                properties={},
            )
        ],
    )

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(extracted_kg),
        retrievers=(StaticRetriever([]),),
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    asyncio.run(rag.ingest_text("ReAct agents are a kind of ReAct.", title="t"))

    # Inspect the chunk_graphs that were passed to persist_entities_and_relationships.
    persisted_chunk_graphs = fake_db.saved_entities[-1][1]
    all_node_ids = {
        node.id for graph in persisted_chunk_graphs.values() for node in graph.nodes
    }
    all_rel_sources = {
        rel.source
        for graph in persisted_chunk_graphs.values()
        for rel in graph.relationships
    }
    all_rel_targets = {
        rel.target
        for graph in persisted_chunk_graphs.values()
        for rel in graph.relationships
    }

    assert "extracted-1" not in all_node_ids, (
        "extracted-1 should have been replaced by persisted-1"
    )
    assert "persisted-1" in all_node_ids, (
        "persisted-1 should appear in the persisted chunk graphs"
    )
    assert all_rel_sources == {"persisted-1"}, (
        "relationship source should have been rewritten to exactly persisted-1"
    )
    assert all_rel_targets == {"persisted-1"}, (
        "relationship target should have been rewritten to exactly persisted-1"
    )


def test_graph_rag_rejects_out_of_range_entity_resolution_threshold() -> None:
    """GraphRAG should reject thresholds outside the cosine [-1, 1] range."""

    graph_db = FakeGraphDB()
    with pytest.raises(ValueError, match="entity_resolution_threshold"):
        GraphRAG(
            model="test-model",
            embedding_model="test-embedding",
            db=graph_db,
            embedding=FakeEmbedder(),
            kg_extractor=ConcurrencyTrackingExtractor(),
            retrievers=(StaticRetriever([]),),
            entity_resolution_threshold=1.5,
        )


def test_graph_rag_does_not_resolve_entities_when_disabled() -> None:
    """When resolve_entities_on_ingest=False (the default), extracted nodes are
    persisted as-is regardless of similarity to any persisted node.
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    extracted_node = Node(
        id="extracted-1",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )

    extracted_kg = KnowledgeGraph(nodes=[extracted_node], relationships=[])

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(extracted_kg),
        retrievers=(StaticRetriever([]),),
        # resolve_entities_on_ingest defaults to False
    )

    asyncio.run(rag.ingest_text("ReAct agents are a kind of ReAct.", title="t"))

    persisted_chunk_graphs = fake_db.saved_entities[-1][1]
    all_node_ids = {
        node.id for graph in persisted_chunk_graphs.values() for node in graph.nodes
    }

    assert "extracted-1" in all_node_ids, (
        "extracted-1 should be persisted as-is when resolution is disabled"
    )


def test_graph_rag_exposes_combined_duplicate_candidate_report() -> None:
    """GraphRAG should expose the two-step duplicate-finding heuristic."""

    graph_db = FakeGraphDB()
    graph_db.entities = [
        Node(
            id="entity_1",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan Turing",
            embedding=[1.0, 0.0, 0.0],
        ),
        Node(
            id="entity_2",
            label="Person",
            semantic_key="person_alan-turing",
            name="Alan M. Turing",
            embedding=[0.95, 0.05, 0.0],
        ),
        Node(
            id="entity_3",
            label="Person",
            semantic_key="person_grace-hopper",
            name="Grace Hopper",
            embedding=[0.0, 1.0, 0.0],
        ),
        Node(
            id="entity_4",
            label="Person",
            semantic_key="person_grace-m-hopper",
            name="Grace M. Hopper",
            embedding=[0.0, 0.95, 0.05],
        ),
    ]
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
    )

    report = asyncio.run(
        rag.find_entity_duplicate_candidates(
            threshold=0.9,
            skip_semantic_key_collisions_in_similarity_scan=True,
        )
    )

    assert list(report.semantic_key_collisions) == ["person_alan-turing"]
    assert [
        group.semantic_key for group in report.semantic_key_collision_candidates
    ] == ["person_alan-turing"]
    assert [result.source.id for result in report.similarity_candidates] == ["entity_3"]
    assert [hit.node.id for hit in report.similarity_candidates[0].hits] == ["entity_4"]


def test_graph_rag_dedupe_entities_builds_merge_reports_and_calls_db() -> None:
    """dedupe_entities() should choose a master, merge labels, and call the DB."""

    graph_db = FakeGraphDB()
    graph_db.entities = [
        Node(
            id="entity_master",
            labels=frozenset({"Concept"}),
            semantic_key="concept_react-agent",
            name="ReAct agent",
            properties={"kind": "concept"},
            embedding=[1.0, 0.0, 0.0],
        ),
        Node(
            id="entity_dup",
            labels=frozenset({"AgentType"}),
            semantic_key="agent_react",
            name="ReAct agent",
            properties={"kind": "agent"},
            embedding=[1.0, 0.0, 0.0],
        ),
        Node(
            id="entity_other",
            labels=frozenset({"Tool"}),
            semantic_key="tool_worker",
            name="Worker",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]
    graph_db.relationship_counts = {"entity_master": 3, "entity_dup": 1}
    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=graph_db,
        embedding=FakeEmbedder(),
        kg_extractor=ConcurrencyTrackingExtractor(),
        retrievers=(StaticRetriever([]),),
    )

    reports = asyncio.run(
        rag.dedupe_entities(
            threshold=0.9,
            min_merge_score=0.9,
        )
    )

    assert reports == [
        MergeReport(
            master_id="entity_master",
            duplicate_ids=("entity_dup",),
            source="similarity",
            merged_labels=("AgentType", "Concept"),
            property_conflicts=("kind",),
        )
    ]
    assert len(graph_db.merges) == 1
    merged_master, duplicate_ids = graph_db.merges[0]
    assert duplicate_ids == ("entity_dup",)
    assert merged_master.id == "entity_master"
    assert merged_master.semantic_key == "concept_react-agent"
    assert merged_master.labels == frozenset({"AgentType", "Concept"})
    assert merged_master.properties == {"kind": "concept"}


def test_resolve_extracted_entities_searches_each_unique_id_once() -> None:
    """When the same extracted entity appears in two chunks, the similarity
    search should be issued exactly once (de-duplication by entity id).
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    # Extracted node is the same object in both chunks — same id, nearly
    # identical embedding so cosine similarity ≈ 0.99998 > 0.9 threshold.
    extracted = Node(
        id="extracted-1",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    inner_finder = EntitySimilarityFinder(db=fake_db)
    counting_finder = _CallCountingFinder(inner_finder)

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(
            KnowledgeGraph(nodes=[extracted], relationships=[])
        ),
        retrievers=(StaticRetriever([]),),
        similarity_finder=counting_finder,
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    # Directly call the private method with two chunks that share the same
    # extracted entity.  This sidesteps the need to produce exactly two chunks
    # from ingest_text and lets the test focus on the de-duplication logic.
    chunk_graphs = {
        "chunk-a": KnowledgeGraph(nodes=[extracted], relationships=[]),
        "chunk-b": KnowledgeGraph(nodes=[extracted], relationships=[]),
    }
    rewritten = asyncio.run(rag._resolve_extracted_entities(chunk_graphs))

    # One search call, not two — the implementation deduplicates by entity id.
    assert len(counting_finder.search_calls) == 1, (
        f"Expected exactly 1 similarity search, got {len(counting_finder.search_calls)}"
    )

    # Both chunks should have the extracted id replaced by the persisted id.
    for chunk_id, graph in rewritten.items():
        node_ids = {n.id for n in graph.nodes}
        assert "extracted-1" not in node_ids, (
            f"chunk '{chunk_id}': extracted-1 should have been replaced"
        )
        assert "persisted-1" in node_ids, (
            f"chunk '{chunk_id}': persisted-1 should appear after rewrite"
        )


def test_resolve_extracted_entities_rewrites_relationship_endpoints() -> None:
    """Relationship source/target pointing at a resolved entity must be rewritten
    to the persisted id.  Endpoints pointing at an unresolved entity stay as-is.
    """

    persisted_node = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    # This extracted node is close enough to be resolved (cosine ≈ 0.99998).
    resolved_extracted = Node(
        id="extracted-resolved",
        label="Concept",
        semantic_key="concept_react-agents",
        name="ReAct agents",
        embedding=[0.1, 0.2, 0.31],
    )
    # This extracted node has an orthogonal embedding — it will not match any
    # persisted node so it should be left untouched.
    unresolved_extracted = Node(
        id="extracted-unresolved",
        label="Concept",
        semantic_key="concept_other",
        name="Something else entirely",
        embedding=[0.0, 0.0, 1.0],
    )

    fake_db = FakeGraphDB()
    fake_db.entities = [persisted_node]

    rag = GraphRAG(
        model="test-model",
        embedding_model="test-embedding",
        db=fake_db,
        embedding=FakeEmbedder(),
        kg_extractor=StaticExtractor(
            KnowledgeGraph(nodes=[resolved_extracted], relationships=[])
        ),
        retrievers=(StaticRetriever([]),),
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    # chunk-a: has a relationship whose source is the resolved entity (self-loop)
    # and a mixed relationship whose source is resolved but target is unresolved.
    # chunk-b: contains only the unresolved entity, no relationships.
    chunk_graphs = {
        "chunk-a": KnowledgeGraph(
            nodes=[resolved_extracted],
            relationships=[
                Relationship(
                    id="rel-1",
                    source="extracted-resolved",
                    target="extracted-resolved",
                    label="RELATED_TO",
                    properties={},
                ),
                Relationship(
                    id="rel-3",
                    source="extracted-resolved",
                    target="extracted-unresolved",
                    label="INFLUENCES",
                    properties={},
                ),
            ],
        ),
        "chunk-b": KnowledgeGraph(
            nodes=[unresolved_extracted],
            relationships=[
                Relationship(
                    id="rel-2",
                    source="extracted-unresolved",
                    target="extracted-unresolved",
                    label="RELATED_TO",
                    properties={},
                )
            ],
        ),
    }
    rewritten = asyncio.run(rag._resolve_extracted_entities(chunk_graphs))

    # chunk-a: resolved entity's id must appear in nodes and in both endpoints of rel-1.
    chunk_a = rewritten["chunk-a"]
    chunk_a_node_ids = {n.id for n in chunk_a.nodes}
    assert "persisted-1" in chunk_a_node_ids, (
        "chunk-a: resolved node should carry persisted-1"
    )
    assert "extracted-resolved" not in chunk_a_node_ids, (
        "chunk-a: extracted-resolved should be gone after rewrite"
    )
    rel_1 = next(r for r in chunk_a.relationships if r.id == "rel-1")
    assert rel_1.source == "persisted-1", (
        "chunk-a rel-1 source should be rewritten to persisted-1"
    )
    assert rel_1.target == "persisted-1", (
        "chunk-a rel-1 target should be rewritten to persisted-1"
    )

    # rel-3 (mixed): source is resolved → rewritten; target is unresolved → stays.
    rel_3 = next(r for r in chunk_a.relationships if r.id == "rel-3")
    assert rel_3.source == "persisted-1", (
        "chunk-a rel-3 source should be rewritten to persisted-1"
    )
    assert rel_3.target == "extracted-unresolved", (
        "chunk-a rel-3 target should remain extracted-unresolved"
    )

    # chunk-b: unresolved entity and its relationship endpoints must be untouched.
    chunk_b = rewritten["chunk-b"]
    chunk_b_node_ids = {n.id for n in chunk_b.nodes}
    assert "extracted-unresolved" in chunk_b_node_ids, (
        "chunk-b: unresolved node should be preserved as-is"
    )
    assert chunk_b.relationships[0].source == "extracted-unresolved", (
        "chunk-b relationship source should remain extracted-unresolved"
    )
    assert chunk_b.relationships[0].target == "extracted-unresolved", (
        "chunk-b relationship target should remain extracted-unresolved"
    )
