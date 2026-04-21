"""Ad-hoc FalkorDB pipeline verification script.

This script uses deterministic stub components so it can verify indexing,
vector persistence, and the modular pipeline flow without external model
credentials.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import autoroot  # noqa: F401

from src.grawiki.core.commons import Chunk
from src.grawiki.core.pipeline import GrawikiPipeline
from src.grawiki.db.falkordb import FalkorGraphDB
from src.grawiki.graph.models import KnowledgeGraph, Node, Relationship


class StubEmbeddingResult:
    """Minimal embedding result compatible with the pipeline."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class StubEmbedder:
    """Deterministic embedder for local verification."""

    async def embed_documents(self, documents: str | list[str]) -> StubEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        return StubEmbeddingResult([self._embed(text) for text in texts])

    @staticmethod
    def _embed(text: str) -> list[float]:
        values = [0.0, 0.0, 0.0, 0.0]
        for idx, char in enumerate(text):
            values[idx % 4] += (ord(char) % 31) / 31.0
        return [round(value, 6) for value in values]


class StubKnowledgeGraphExtractor:
    """Deterministic chunk-level extractor for local verification."""

    async def extract(self, chunk: Chunk) -> KnowledgeGraph:
        person_name = "Alan Turing"
        concept_name = "Machine intelligence"
        return KnowledgeGraph(
            nodes=[
                Node(
                    id=f"{chunk.id}_entity_1",
                    label="Person",
                    semantic_key="person_alan-turing",
                    name=person_name,
                    embedding=StubEmbedder._embed(person_name),
                ),
                Node(
                    id=f"{chunk.id}_entity_2",
                    label="Concept",
                    semantic_key="concept_machine-intelligence",
                    name=concept_name,
                    embedding=StubEmbedder._embed(concept_name),
                ),
            ],
            relationships=[
                Relationship(
                    id=f"{chunk.id}_rel_1",
                    source=f"{chunk.id}_entity_1",
                    target=f"{chunk.id}_entity_2",
                    label="inspired",
                )
            ],
        )


async def run_modular_flow(db_path: Path, graph_name: str) -> FalkorGraphDB:
    """Run the modular pipeline step by step."""

    adapter = FalkorGraphDB(db_path, graph_name)
    pipeline = GrawikiPipeline(
        model="stub-model",
        embedding_model="stub-embedding",
        graph_db=adapter,
        max_workers=2,
        embedder=StubEmbedder(),
        kg_extractor=StubKnowledgeGraphExtractor(),
    )

    input_path = db_path.parent / "debug_input.txt"
    input_path.write_text(
        "Alan Turing inspired later work on machine intelligence.\n"
        "Graph memory can connect documents, chunks, and entities.\n"
    )

    await pipeline.setup_db()
    document = pipeline.read_document(input_path)
    chunks = pipeline.chunk_document(document)
    document_embedding = await pipeline.embed_document(document)
    chunk_embeddings = await pipeline.embed_chunks(chunks)
    document_node = pipeline.build_document_node(document, document_embedding)
    chunk_nodes = pipeline.build_chunk_nodes(chunks, chunk_embeddings)
    await pipeline.persist_documents_and_chunks(document_node, chunk_nodes)
    chunk_graphs = await pipeline.extract_chunk_graphs(chunks)
    await pipeline.persist_entities_and_relationships(chunks, chunk_graphs)

    return adapter


def print_debug_queries(adapter: FalkorGraphDB) -> None:
    """Print persisted graph state and index metadata."""

    print("Indexes:", adapter.ro_query("CALL db.indexes()").result_set)
    print(
        "Documents:",
        adapter.ro_query(
            "MATCH (d:__document__) RETURN d.id, d.name ORDER BY d.name"
        ).result_set,
    )
    print(
        "Chunks:",
        adapter.ro_query(
            "MATCH (c:__chunk__) RETURN c.id, c.name ORDER BY c.id"
        ).result_set,
    )
    print(
        "Entities:",
        adapter.ro_query(
            "MATCH (e:__entity__) RETURN e.name, e.label, e.semantic_key ORDER BY e.name"
        ).result_set,
    )
    print(
        "Fulltext search:",
        adapter.query_fulltext_nodes(
            "__entity__",
            "Turing",
            return_expression="node.name ORDER BY node.name",
        ).result_set,
    )
    print(
        "Vector search:",
        adapter.query_similar_nodes(
            "__entity__",
            StubEmbedder._embed("Alan Turing"),
            2,
            return_expression="node.name, score",
        ).result_set,
    )
    print(
        "Vector explain:",
        adapter.explain_vector_query(
            "__entity__",
            StubEmbedder._embed("Alan Turing"),
            2,
        ),
    )


async def main() -> None:
    """Run the modular debug flow inside an isolated FalkorDB database."""

    with TemporaryDirectory() as tmp_dir:
        adapter = await run_modular_flow(Path(tmp_dir) / "debug.db", "debug_graph")
        print_debug_queries(adapter)


if __name__ == "__main__":
    asyncio.run(main())
