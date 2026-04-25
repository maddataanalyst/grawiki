import autoroot
from dotenv import load_dotenv

from grawiki import rag
from grawiki.db.falkordb import FalkorGraphDB
from grawiki.rag.graph_rag import GraphRAG
from pathlib import Path
from tests import rag
from grawiki.graph.models import Node


load_dotenv(override=True)


async def main():
    database = FalkorGraphDB(db_path="local_falcor.db", graph_name="grawiki")
    rag = GraphRAG(
        db=database,
        model="openrouter:openai/gpt-5-mini",
        embedding_model="openrouter:openai/text-embedding-3-small",
        max_workers=10,
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    from datetime import datetime
    from grawiki.graph.models import MemoryNode, Relationship
    from grawiki.core.commons import Chunk

    # This might be a result of the agent operation - e.g. previous query
    nodes = await rag.search("Reflexion agents")
    memory_node_ids = [hit.node.id for hit in nodes[:3]]

    content = """User wanted a complete analysis of the reflexion agents, reflexion and orchestrator-worker patterns"""

    memory = MemoryNode(
        id="memory_1",
        semantic_key="memory_1",
        name="User memory1",
        properties={"user_id": "user_1223"},
        creation_date=datetime.now().isoformat(),
        content=content,
    )

    # TODO: unneccessary step - turning Memory to Chunk.
    # It seems that extract_kg_per_chunk is designed to work with Chunks,
    # but we could also have an extract_kg_per_node that works directly with MemoryNodes.
    # This would avoid the redundant step of converting a MemoryNode to a Chunk just for KG extraction.
    chunk_memory = Chunk(
        id=memory.id,
        document_id=memory.id,
        content=content,
    )

    concepts_from_memory = await rag.extract_kg_per_chunk([chunk_memory])

    memory_rels = []
    for idx, node_id in enumerate(memory_node_ids):
        rel = Relationship(
            id=f"rel_{idx}",
            source=memory.id,
            target=node_id,
            label="MENTIONS",
        )
        memory_rels.append(rel)

    existing_memory_entities = await rag._resolve_extracted_entities(
        concepts_from_memory
    )

    raw_extracted_ids = [entity.id for entity in concepts_from_memory["memory_1"].nodes]
    found_existing_nodes = []
    new_nodes = []
    for node in existing_memory_entities["memory_1"].nodes:
        if node.id not in raw_extracted_ids:
            found_existing_nodes.append((node.id, node.name))
        else:
            new_nodes.append((node.id, node.name))

    print("Found existing nodes that match extracted concepts:")
    for node_id, node_name in found_existing_nodes:
        print(f"- {node_id}: {node_name}")

    print("\nNewly extracted nodes that do not match existing entities:")
    for node_id, node_name in new_nodes:
        print(f"- {node_id}: {node_name}")

    # database.upsert_relationships([memory_rels])
    # database.upsert_nodes([memory])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
