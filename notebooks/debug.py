import autoroot
from dotenv import load_dotenv

from grawiki.db.falkordb import FalkorGraphDB
from grawiki.rag.graph_rag import GraphRAG
from pathlib import Path
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

    sim_finder = rag._entity_similarity
    duplicates = await sim_finder.find_duplicate_candidates(limit=10, threshold=0.9)
    duplicate_candidate_1 = duplicates.similarity_candidates[0]
    dupl_source_1 = duplicate_candidate_1.source
    print("Duplicate candidate source node:")
    dupl_dict = dict(dupl_source_1)
    dupl_dict.pop("embedding", None)  # Remove embedding from printout for readability
    print(dupl_dict)

    merged_properties = dict(dupl_source_1.properties)
    for hit in duplicate_candidate_1.hits:
        for prop, value in hit.node.properties.items():
            if prop not in merged_properties:
                merged_properties[prop] = value

    merged_labels = set([dupl_source_1.label])
    for hit in duplicate_candidate_1.hits:
        merged_labels.add(hit.node.label)

    new_node = Node(
        name=dupl_source_1.name,
        semantic_key=dupl_source_1.semantic_key,
        id=dupl_source_1.id,
        label=list(merged_labels)[0],
        properties=merged_properties,
        embedding=dupl_source_1.embedding,
    )

    collected_relationships = []
    relationship_update_queries = []
    for hit in duplicate_candidate_1.hits:
        hit_rels = database.query(
            f"MATCH (n {{id: '{hit.node.id}'}})-[r]->(m) RETURN r.id"
        ).result_set
        for rel in hit_rels:
            collected_relationships.append(rel[0])
            rel_update_query = f"""MATCH (n_new: {new_node.label} {{id: '{new_node.id}'}})
            MERGE (n_new)-[r {{id: '{rel[0]}'}}]->(m)"""
            relationship_update_queries.append(rel_update_query)

    print("Collected relationships to update:", collected_relationships)
    print("Generated relationship update queries:")
    for query in relationship_update_queries:
        print(query)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
