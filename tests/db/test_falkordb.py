"""Integration tests for the FalkorDB adapter."""

import asyncio

import pytest

from grawiki.core.commons import Chunk
from grawiki.db.base import NeighborRelationship, NodeHit
from grawiki.db.falkordb import FalkorGraphDB
from grawiki.graph.models import (
    ChunkNode,
    DocumentNode,
    KnowledgeGraph,
    MemoryNode,
    Node,
    Relationship,
)


@pytest.fixture
def graph_db(tmp_path):
    """Create an isolated FalkorDBLite-backed graph for each test."""

    graph = FalkorGraphDB(
        "test_graph",
        db_path=tmp_path / "graph.db",
        vector_index_m=32,
        vector_index_ef_construction=200,
        vector_index_ef_runtime=10,
    )
    try:
        yield graph
    finally:
        graph.close()


@pytest.fixture(autouse=True)
def _setup_indexes(graph_db: FalkorGraphDB) -> FalkorGraphDB:
    """Pre-create all indexes once after graph creation."""

    asyncio.run(graph_db.setup())
    return graph_db


def _index_rows_by_key(rows: list[list[object]]) -> dict[tuple[str, str], list[object]]:
    """Map FalkorDB index rows by ``(label, property)``."""

    index_map: dict[tuple[str, str], list[object]] = {}
    for row in rows:
        label = row[0]
        for property_name in row[2]:
            index_map[(label, property_name)] = row
    return index_map


def test_save_documents_chunks_entities_and_relationships_with_indexes(
    graph_db: FalkorGraphDB,
) -> None:
    """The adapter should persist nodes, embeddings, and indexes."""

    document_node = DocumentNode(
        id="doc_1",
        label="__document__",
        semantic_key="document_doc_1",
        name="Graph Memory",
        content="Graph databases support memory.",
        metadata={"source": "unit-test"},
        embedding=[1.0, 0.0, 0.0],
    )
    chunk_nodes = [
        ChunkNode(
            id="chunk_1",
            label="__chunk__",
            semantic_key="chunk_chunk_1",
            name="Chunk chunk_1",
            document_id="doc_1",
            content="Alan Turing studied computability.",
            metadata={"order": "1"},
            embedding=[1.0, 0.0, 0.0],
        ),
        ChunkNode(
            id="chunk_2",
            label="__chunk__",
            semantic_key="chunk_chunk_2",
            name="Chunk chunk_2",
            document_id="doc_1",
            content="Alan Turing inspired graph memory work.",
            metadata={"order": "2"},
            embedding=[0.8, 0.2, 0.0],
        ),
    ]
    chunks = [
        Chunk(id="chunk_1", document_id="doc_1", content=chunk_nodes[0].content),
        Chunk(id="chunk_2", document_id="doc_1", content=chunk_nodes[1].content),
    ]
    chunk_graphs = {
        "chunk_1": KnowledgeGraph(
            nodes=[
                Node(
                    id="entity_1",
                    label="Person",
                    semantic_key="person_alan-turing",
                    name="Alan Turing",
                    embedding=[1.0, 0.0, 0.0],
                ),
                Node(
                    id="entity_2",
                    label="Concept",
                    semantic_key="concept_computability",
                    name="Computability",
                    embedding=[0.0, 1.0, 0.0],
                ),
            ],
            relationships=[
                Relationship(
                    id="rel_1",
                    source="entity_1",
                    target="entity_2",
                    label="studied",
                )
            ],
        ),
        "chunk_2": KnowledgeGraph(
            nodes=[
                Node(
                    id="entity_3",
                    label="Person",
                    semantic_key="person_alan-turing",
                    name="Alan Turing",
                    properties={"field": "AI"},
                    embedding=[1.0, 0.0, 0.0],
                )
            ]
        ),
    }

    asyncio.run(graph_db.save_docs_and_chunks_to_db([document_node], chunk_nodes))
    asyncio.run(
        graph_db.save_entities_and_rels([chunk.id for chunk in chunks], chunk_graphs)
    )

    assert graph_db.ro_query("MATCH (d:__document__) RETURN count(d)").result_set == [
        [1]
    ]
    assert graph_db.ro_query("MATCH (c:__chunk__) RETURN count(c)").result_set == [[2]]
    assert graph_db.ro_query(
        "MATCH (:__document__)-[r:__has_chunk__]->(:__chunk__) RETURN count(r)"
    ).result_set == [[2]]
    assert graph_db.ro_query(
        "MATCH (:__document__)-[r:__has_chunk__]->(:__chunk__) "
        "RETURN DISTINCT r.label, r.properties"
    ).result_set == [["__has_chunk__", "{}"]]
    assert graph_db.ro_query("MATCH (e:__entity__) RETURN count(e)").result_set == [[2]]
    assert graph_db.ro_query(
        "MATCH (:__chunk__)-[r:__mentions__]->(:__entity__) RETURN count(r)"
    ).result_set == [[3]]
    assert graph_db.ro_query(
        "MATCH (:__chunk__)-[r:__mentions__]->(:__entity__) "
        "RETURN DISTINCT r.label, r.properties"
    ).result_set == [["__mentions__", "{}"]]
    assert graph_db.ro_query(
        "MATCH (:__entity__ {semantic_key: 'person_alan-turing'})-[r:studied]->"
        "(:__entity__ {semantic_key: 'concept_computability'}) RETURN count(r)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (e:__entity__ {semantic_key: 'person_alan-turing'}) RETURN e.properties"
    ).result_set == [['{"field": "AI"}']]

    indexes = _index_rows_by_key(graph_db.ro_query("CALL db.indexes()").result_set)
    assert "FULLTEXT" in indexes[("__document__", "name")][2]["name"]
    assert "FULLTEXT" in indexes[("__document__", "content")][2]["content"]
    assert "FULLTEXT" in indexes[("__chunk__", "name")][2]["name"]
    assert "FULLTEXT" in indexes[("__chunk__", "content")][2]["content"]
    assert "FULLTEXT" in indexes[("__entity__", "name")][2]["name"]
    assert "VECTOR" in indexes[("__document__", "embedding")][2]["embedding"]
    assert indexes[("__document__", "embedding")][3]["embedding"]["dimension"] == 3
    assert indexes[("__document__", "embedding")][3]["embedding"]["M"] == 32
    assert (
        indexes[("__document__", "embedding")][3]["embedding"]["similarityFunction"]
        == "cosine"
    )
    assert "VECTOR" in indexes[("__chunk__", "embedding")][2]["embedding"]
    assert "VECTOR" in indexes[("__entity__", "embedding")][2]["embedding"]

    assert graph_db.query_fulltext_nodes(
        "__entity__",
        "Turing",
        return_expression="node.name",
    ).result_set == [["Alan Turing"]]
    vector_results = graph_db.query_similar_nodes(
        "__entity__",
        [1.0, 0.0, 0.0],
        2,
        return_expression="node.name, score",
    ).result_set
    assert vector_results[0][0] == "Alan Turing"
    assert vector_results[0][1] == pytest.approx(1.0, abs=1e-6)
    explain_result = graph_db.explain_vector_query("__entity__", [1.0, 0.0, 0.0], 2)
    assert "ProcedureCall" in str(explain_result)

    fulltext_graph_results = asyncio.run(
        graph_db.search("Graph", method="fulltext", limit=5)
    )
    assert fulltext_graph_results["__document__"][0]["name"] == "Graph Memory"
    assert any(
        hit["name"] == "Chunk chunk_2" for hit in fulltext_graph_results["__chunk__"]
    )

    fulltext_turing_results = asyncio.run(
        graph_db.search("Turing", method="fulltext", limit=5)
    )
    assert any(
        hit["name"] == "Chunk chunk_1" for hit in fulltext_turing_results["__chunk__"]
    )
    assert fulltext_turing_results["__entity__"][0]["name"] == "Alan Turing"

    vector_search_results = asyncio.run(
        graph_db.search(
            "ignored raw query",
            method="vector",
            limit=2,
            query_embedding=[1.0, 0.0, 0.0],
        )
    )
    assert vector_search_results["__document__"][0]["name"] == "Graph Memory"
    assert vector_search_results["__chunk__"][0]["name"] == "Chunk chunk_1"
    assert vector_search_results["__entity__"][0]["name"] == "Alan Turing"


def test_vector_search_requires_query_embedding(graph_db: FalkorGraphDB) -> None:
    """Vector search should fail fast when no query embedding is supplied."""

    with pytest.raises(ValueError, match="requires a query embedding"):
        asyncio.run(graph_db.search("hello", method="vector"))


def test_save_entities_and_rels_rejects_unknown_owner(graph_db: FalkorGraphDB) -> None:
    """Owner graphs should fail fast when their owning node id is missing."""

    owner_ids = ["chunk_1"]
    owner_graphs = {"missing_chunk": KnowledgeGraph()}

    with pytest.raises(ValueError, match="Unknown owner ids"):
        asyncio.run(graph_db.save_entities_and_rels(owner_ids, owner_graphs))


def test_save_entities_and_rels_rejects_relationships_with_unknown_nodes(
    graph_db: FalkorGraphDB,
) -> None:
    """Relationships should reference nodes present in the same chunk graph."""

    owner_graphs = {
        "chunk_1": KnowledgeGraph(
            nodes=[],
            relationships=[
                Relationship(
                    id="rel_1",
                    source="missing_source",
                    target="missing_target",
                    label="relates_to",
                )
            ],
        )
    }

    with pytest.raises(ValueError, match="Relationship references a node missing"):
        asyncio.run(graph_db.save_entities_and_rels(["chunk_1"], owner_graphs))


@pytest.fixture
def populated_graph_db(graph_db: FalkorGraphDB) -> FalkorGraphDB:
    """Populate the graph with one document, two chunks, and two entities."""

    document_node = DocumentNode(
        id="doc_1",
        semantic_key="document_doc_1",
        name="Graph Memory",
        content="Graph databases support memory.",
        metadata={"source": "unit-test"},
        embedding=[1.0, 0.0, 0.0],
    )
    chunk_nodes = [
        ChunkNode(
            id="chunk_1",
            semantic_key="chunk_chunk_1",
            name="Chunk chunk_1",
            document_id="doc_1",
            content="Alan Turing studied computability.",
            metadata={"order": "1"},
            embedding=[1.0, 0.0, 0.0],
        ),
        ChunkNode(
            id="chunk_2",
            semantic_key="chunk_chunk_2",
            name="Chunk chunk_2",
            document_id="doc_1",
            content="Alan Turing inspired graph memory work.",
            metadata={"order": "2"},
            embedding=[0.8, 0.2, 0.0],
        ),
    ]
    chunks = [
        Chunk(id="chunk_1", document_id="doc_1", content=chunk_nodes[0].content),
        Chunk(id="chunk_2", document_id="doc_1", content=chunk_nodes[1].content),
    ]
    chunk_graphs = {
        "chunk_1": KnowledgeGraph(
            nodes=[
                Node(
                    id="entity_turing",
                    label="Person",
                    semantic_key="person_alan-turing",
                    name="Alan Turing",
                    embedding=[1.0, 0.0, 0.0],
                ),
                Node(
                    id="entity_comp",
                    label="Concept",
                    semantic_key="concept_computability",
                    name="Computability",
                    embedding=[0.0, 1.0, 0.0],
                ),
            ],
            relationships=[
                Relationship(
                    id="rel_1",
                    source="entity_turing",
                    target="entity_comp",
                    label="studied",
                )
            ],
        ),
    }

    asyncio.run(graph_db.save_docs_and_chunks_to_db([document_node], chunk_nodes))
    asyncio.run(
        graph_db.save_entities_and_rels([chunk.id for chunk in chunks], chunk_graphs)
    )
    return graph_db


def test_ensure_indexes_creates_indexes_for_memory_label(
    graph_db: FalkorGraphDB,
) -> None:
    """``ensure_indexes`` should register new labels (e.g. ``__memory__``)."""

    asyncio.run(
        graph_db.ensure_indexes(
            labels=["__memory__"],
            vector_dims={"__memory__": 3},
        )
    )

    indexes = _index_rows_by_key(graph_db.ro_query("CALL db.indexes()").result_set)
    assert "FULLTEXT" in indexes[("__memory__", "name")][2]["name"]
    assert "FULLTEXT" in indexes[("__memory__", "content")][2]["content"]
    assert "VECTOR" in indexes[("__memory__", "embedding")][2]["embedding"]
    assert indexes[("__memory__", "embedding")][3]["embedding"]["dimension"] == 3


def test_save_entities_and_rels_supports_memory_owners(graph_db: FalkorGraphDB) -> None:
    """Owner-based persistence should link memories to extracted entities."""

    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Agent memory",
        content="Remember ReAct.",
        metadata={"user_id": "user-1"},
    )
    asyncio.run(graph_db.upsert_nodes([memory]))
    asyncio.run(
        graph_db.save_entities_and_rels(
            [memory.id],
            {
                memory.id: KnowledgeGraph(
                    nodes=[
                        Node(
                            id="entity_1",
                            label="Concept",
                            semantic_key="concept_react",
                            name="ReAct",
                            embedding=[1.0, 0.0, 0.0],
                        )
                    ],
                    relationships=[],
                )
            },
        )
    )

    assert graph_db.ro_query(
        "MATCH (:__memory__ {id: 'memory_1'})-[:__mentions__]->"
        "(:__entity__ {id: 'entity_1'}) RETURN count(*)"
    ).result_set == [[1]]


def test_upsert_relationships_supports_memory_links_to_any_node_ids(
    graph_db: FalkorGraphDB,
) -> None:
    """System id-based links should allow memory relationships to any node family."""

    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Agent memory",
        content="Remember chunk and entity.",
        metadata={"user_id": "user-1"},
    )
    chunk = ChunkNode(
        id="chunk_1",
        semantic_key="chunk_chunk_1",
        name="Chunk chunk_1",
        document_id="doc_1",
        content="Chunk content.",
    )
    entity = Node(
        id="entity_1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
    )
    asyncio.run(graph_db.upsert_nodes([memory, chunk, entity]))
    asyncio.run(
        graph_db.upsert_relationships(
            [
                Relationship(
                    id="rel_chunk",
                    source="memory_1",
                    target="chunk_1",
                    label="__related__",
                ),
                Relationship(
                    id="rel_entity",
                    source="memory_1",
                    target="entity_1",
                    label="__related__",
                ),
            ]
        )
    )

    assert graph_db.ro_query(
        "MATCH (:__memory__ {id: 'memory_1'})-[:__related__]->(:__chunk__ {id: 'chunk_1'}) "
        "RETURN count(*)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (:__memory__ {id: 'memory_1'})-[:__related__]->(:__entity__ {id: 'entity_1'}) "
        "RETURN count(*)"
    ).result_set == [[1]]


def test_delete_memory_detaches_links_and_prunes_zero_relation_entities(
    graph_db: FalkorGraphDB,
) -> None:
    """Deleting a memory should remove it and prune directly-mentioned orphan entities."""

    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Agent memory",
        content="Remember ReAct.",
        metadata={"user_id": "user-1"},
    )
    entity = Node(
        id="entity_1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
    )
    asyncio.run(graph_db.upsert_nodes([memory, entity]))
    asyncio.run(
        graph_db.upsert_relationships(
            [
                Relationship(
                    id="rel_mentions",
                    source="memory_1",
                    target="entity_1",
                    label="__mentions__",
                )
            ]
        )
    )

    asyncio.run(graph_db.delete_memory("memory_1"))

    assert graph_db.ro_query(
        "MATCH (m:__memory__ {id: 'memory_1'}) RETURN count(m)"
    ).result_set == [[0]]
    assert graph_db.ro_query(
        "MATCH (e:__entity__ {id: 'entity_1'}) RETURN count(e)"
    ).result_set == [[0]]


def test_delete_memory_keeps_entities_with_remaining_relationships(
    graph_db: FalkorGraphDB,
) -> None:
    """Deleting a memory should keep directly-mentioned entities that still have edges."""

    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Agent memory",
        content="Remember ReAct.",
        metadata={"user_id": "user-1"},
    )
    chunk = ChunkNode(
        id="chunk_1",
        semantic_key="chunk_chunk_1",
        name="Chunk chunk_1",
        document_id="doc_1",
        content="ReAct is important.",
    )
    entity = Node(
        id="entity_1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
    )
    asyncio.run(graph_db.upsert_nodes([memory, chunk, entity]))
    asyncio.run(
        graph_db.upsert_relationships(
            [
                Relationship(
                    id="rel_memory_mentions",
                    source="memory_1",
                    target="entity_1",
                    label="__mentions__",
                ),
                Relationship(
                    id="rel_chunk_mentions",
                    source="chunk_1",
                    target="entity_1",
                    label="__mentions__",
                ),
            ]
        )
    )

    asyncio.run(graph_db.delete_memory("memory_1"))

    assert graph_db.ro_query(
        "MATCH (m:__memory__ {id: 'memory_1'}) RETURN count(m)"
    ).result_set == [[0]]
    assert graph_db.ro_query(
        "MATCH (e:__entity__ {id: 'entity_1'}) RETURN count(e)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (:__chunk__ {id: 'chunk_1'})-[:__mentions__]->(:__entity__ {id: 'entity_1'}) "
        "RETURN count(*)"
    ).result_set == [[1]]


def test_fulltext_search_returns_node_hits_with_subclasses(
    populated_graph_db: FalkorGraphDB,
) -> None:
    """Full-text primitive should return flat ``NodeHit`` list with proper subclasses."""

    hits = asyncio.run(
        populated_graph_db.fulltext_search(
            labels=["__document__", "__chunk__", "__entity__"],
            query_text="Turing",
            limit=5,
        )
    )

    assert all(isinstance(hit, NodeHit) for hit in hits)
    assert all(hit.matched_on == "fulltext" for hit in hits)

    chunk_hits = [
        hit for hit in hits if hit.node.labels == frozenset({ChunkNode.system_label})
    ]
    entity_hits = [hit for hit in hits if hit.node.labels == frozenset({"Person"})]
    assert chunk_hits, "expected at least one chunk hit"
    assert isinstance(chunk_hits[0].node, ChunkNode)
    assert chunk_hits[0].node.content
    assert chunk_hits[0].node.document_id == "doc_1"
    assert entity_hits, "expected at least one entity hit"
    assert entity_hits[0].node.name == "Alan Turing"
    assert entity_hits[0].node.semantic_key == "person_alan-turing"


def test_vector_search_returns_scored_node_hits(
    populated_graph_db: FalkorGraphDB,
) -> None:
    """Vector primitive should populate ``score`` and reconstruct Node subclasses."""

    hits = asyncio.run(
        populated_graph_db.vector_search(
            labels=["__document__", "__chunk__", "__entity__"],
            query_embedding=[1.0, 0.0, 0.0],
            limit=2,
        )
    )

    assert all(isinstance(hit, NodeHit) for hit in hits)
    assert all(hit.matched_on == "vector" for hit in hits)
    assert all(isinstance(hit.score, float) for hit in hits)

    top_document = next(
        hit for hit in hits if hit.node.labels == frozenset({DocumentNode.system_label})
    )
    assert isinstance(top_document.node, DocumentNode)
    assert top_document.node.name == "Graph Memory"
    assert top_document.score == pytest.approx(1.0, abs=1e-6)

    chunk_hits = [
        hit for hit in hits if hit.node.labels == frozenset({ChunkNode.system_label})
    ]
    assert [hit.node.name for hit in chunk_hits] == ["Chunk chunk_1", "Chunk chunk_2"]
    assert chunk_hits[0].score > chunk_hits[1].score


def test_vector_search_skips_missing_document_index_and_returns_chunk_hits(
    graph_db: FalkorGraphDB,
) -> None:
    """Vector search should keep searching labels that have vector indexes."""

    document_node = DocumentNode(
        id="doc_no_embedding",
        semantic_key="document_doc_no_embedding",
        name="Document without embedding",
        content="Chunk-level retrieval only.",
    )
    chunk_node = ChunkNode(
        id="chunk_with_embedding",
        semantic_key="chunk_chunk_with_embedding",
        name="Chunk with embedding",
        document_id="doc_no_embedding",
        content="Chunk vectors remain searchable.",
        doc_position=0,
        embedding=[1.0, 0.0, 0.0],
    )
    asyncio.run(graph_db.save_docs_and_chunks_to_db([document_node], [chunk_node]))

    hits = asyncio.run(
        graph_db.vector_search(
            labels=["__document__", "__chunk__"],
            query_embedding=[1.0, 0.0, 0.0],
            limit=5,
        )
    )

    assert [hit.node.id for hit in hits] == ["chunk_with_embedding"]
    assert isinstance(hits[0].node, ChunkNode)


def test_list_entities_returns_persisted_entities(
    populated_graph_db: FalkorGraphDB,
) -> None:
    """Entity listing should preserve ontology labels and optionally embeddings."""

    entities_without_embeddings = asyncio.run(populated_graph_db.list_entities())
    entities_with_embeddings = asyncio.run(
        populated_graph_db.list_entities(include_embeddings=True)
    )

    assert [
        (entity.id, entity.labels, entity.name)
        for entity in entities_without_embeddings
    ] == [
        ("entity_comp", frozenset({"Concept"}), "Computability"),
        ("entity_turing", frozenset({"Person"}), "Alan Turing"),
    ]
    assert all(not entity.embedding for entity in entities_without_embeddings)
    assert entities_with_embeddings[0].embedding == [0.0, 1.0, 0.0]
    assert entities_with_embeddings[1].embedding == [1.0, 0.0, 0.0]


def test_neighbor_relationships_returns_one_hop_context(
    populated_graph_db: FalkorGraphDB,
) -> None:
    """``neighbor_relationships`` should return typed one-hop context rows."""

    contexts = asyncio.run(
        populated_graph_db.neighbor_relationships(
            node_ids=["entity_turing", "entity_comp"],
            limit_per_node=2,
        )
    )

    assert set(contexts) == {"entity_turing", "entity_comp"}
    assert all(
        isinstance(relationship, NeighborRelationship)
        for relationship in contexts["entity_turing"]
    )
    assert any(
        relationship.relationship_label == "studied"
        and relationship.target.name == "Computability"
        for relationship in contexts["entity_turing"]
    )
    assert any(
        relationship.relationship_label == "studied"
        and relationship.target.name == "Alan Turing"
        for relationship in contexts["entity_comp"]
    )


def test_recall_subgraph_uses_k_hop_paths_without_crossing_other_memories(
    graph_db: FalkorGraphDB,
) -> None:
    """Recall should expand through entities but not traverse into other memories."""

    memory_1 = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="Seed memory",
        content="Remember ReAct.",
        metadata={"user_id": "user-1"},
    )
    memory_2 = MemoryNode(
        id="memory_2",
        semantic_key="memory_2",
        name="Other memory",
        content="Also mentions ReAct.",
        metadata={"user_id": "user-2"},
    )
    react = Node(
        id="entity_react",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
    )
    worker = Node(
        id="entity_worker",
        label="Concept",
        semantic_key="concept_worker",
        name="Worker",
    )

    asyncio.run(graph_db.upsert_nodes([memory_1, memory_2, react, worker]))
    asyncio.run(
        graph_db.upsert_relationships(
            [
                Relationship(
                    id="rel_memory_1",
                    source="memory_1",
                    target="entity_react",
                    label="__mentions__",
                ),
                Relationship(
                    id="rel_memory_2",
                    source="memory_2",
                    target="entity_react",
                    label="__mentions__",
                ),
                Relationship(
                    id="rel_entity",
                    source="entity_react",
                    target="entity_worker",
                    label="RELATED_TO",
                ),
            ]
        )
    )

    contexts = asyncio.run(
        graph_db.recall_subgraph(memory_ids=["memory_1"], hops=2, limit_per_memory=10)
    )

    assert list(contexts) == ["memory_1"]
    assert [
        (
            relationship.source_id,
            relationship.relationship_label,
            relationship.target.id,
        )
        for relationship in contexts["memory_1"]
    ] == [
        ("memory_1", "__mentions__", "entity_react"),
        ("entity_react", "RELATED_TO", "entity_worker"),
    ]


def test_merge_entity_nodes_redirects_and_deduplicates_relationships(
    graph_db: FalkorGraphDB,
) -> None:
    """Merging should redirect edges, drop self-loops, and collapse duplicates.

    Before
    ------
    K -__mentions__-> B
    A -knows-> C
    B -knows-> C
    B -knows-> E
    D -uses-> B
    B -related_to-> A

    After merging B into A
    ----------------------
    K -__mentions__-> A
    A -knows-> C        # only one remains
    A -knows-> E        # redirected from B
    D -uses-> A

                        # B -related_to-> A becomes A -related_to-> A, so dropped
    B is deleted
    """

    chunk = ChunkNode(
        id="chunk_1",
        semantic_key="chunk_chunk_1",
        name="Chunk chunk_1",
        document_id="doc_1",
        content="ReAct agent example.",
    )
    master = Node(
        id="entity_a",
        labels=frozenset({"Concept"}),
        semantic_key="concept_react-agent",
        name="ReAct agent",
    )
    duplicate = Node(
        id="entity_b",
        labels=frozenset({"AgentType"}),
        semantic_key="agent_react",
        name="ReAct agent",
    )
    node_c = Node(
        id="entity_c",
        labels=frozenset({"Tool"}),
        semantic_key="tool_target-c",
        name="Target C",
    )
    node_d = Node(
        id="entity_d",
        labels=frozenset({"Tool"}),
        semantic_key="tool_source-d",
        name="Source D",
    )
    node_e = Node(
        id="entity_e",
        labels=frozenset({"Tool"}),
        semantic_key="tool_target-e",
        name="Target E",
    )

    asyncio.run(
        graph_db.upsert_nodes([chunk, master, duplicate, node_c, node_d, node_e])
    )
    asyncio.run(
        graph_db.upsert_relationships(
            [
                Relationship(
                    id="rel_master_out",
                    source="entity_a",
                    target="entity_c",
                    label="knows",
                ),
                Relationship(
                    id="rel_dup_out",
                    source="entity_b",
                    target="entity_c",
                    label="knows",
                ),
                Relationship(
                    id="rel_incoming",
                    source="entity_d",
                    target="entity_b",
                    label="uses",
                ),
                Relationship(
                    id="rel_dup_out_unique",
                    source="entity_b",
                    target="entity_e",
                    label="knows",
                ),
                Relationship(
                    id="rel_self_loop_source",
                    source="entity_b",
                    target="entity_a",
                    label="related_to",
                ),
                Relationship(
                    id="rel_mentions",
                    source="chunk_1",
                    target="entity_b",
                    label="__mentions__",
                ),
            ]
        )
    )

    asyncio.run(
        graph_db.merge_entity_nodes(
            master=Node(
                id="entity_a",
                labels=frozenset({"AgentType", "Concept"}),
                semantic_key="concept_react-agent",
                name="ReAct agent",
            ),
            duplicate_ids=["entity_b"],
        )
    )

    assert graph_db.ro_query(
        "MATCH (n:__entity__ {id: 'entity_b'}) RETURN count(n)"
    ).result_set == [[0]]
    assert graph_db.ro_query(
        "MATCH (:__entity__ {id: 'entity_d'})-[:uses]->(:__entity__ {id: 'entity_a'}) "
        "RETURN count(*)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (:__entity__ {id: 'entity_a'})-[:knows]->(:__entity__ {id: 'entity_c'}) "
        "RETURN count(*)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (:__entity__ {id: 'entity_a'})-[:knows]->(:__entity__ {id: 'entity_e'}) "
        "RETURN count(*)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (:__chunk__ {id: 'chunk_1'})-[:__mentions__]->(:__entity__ {id: 'entity_a'}) "
        "RETURN count(*)"
    ).result_set == [[1]]
    assert graph_db.ro_query(
        "MATCH (n:__entity__ {id: 'entity_b'})-[r]-() RETURN count(r)"
    ).result_set == [[0]]
    assert graph_db.ro_query(
        "MATCH (:__entity__ {id: 'entity_a'})-[:related_to]->(:__entity__ {id: 'entity_a'}) "
        "RETURN count(*)"
    ).result_set == [[0]]
    assert graph_db.ro_query(
        "MATCH (n:__entity__ {id: 'entity_a'}) RETURN n.labels"
    ).result_set == [[["AgentType", "Concept"]]]
