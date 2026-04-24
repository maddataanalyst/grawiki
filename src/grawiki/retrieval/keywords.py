from pydantic import Field, BaseModel
from pydantic_ai import Agent
from typing import Sequence
from grawiki.core.embedding import Embedding
from grawiki.db.base import GraphDB, NodeHit
from grawiki.graph.models import Node
from grawiki.retrieval.base import Retriever


class Keywords(BaseModel):
    keywords: Sequence[str] = Field(
        ..., description="List of distinct key phrases extracted from the query"
    )


# TODO: add number of max keywords to return
KEYWORDS_EXTRACTION_PROMPT = """Extract distinct key phrases from the following query. Return only the key phrases as a list."""


class KeywordsPathRetriever(Retriever):
    def __init__(self, model: str, db: GraphDB, embedding: Embedding):
        self.model = model
        self.agent = Agent(
            model=model, output_type=Keywords, system_prompt=KEYWORDS_EXTRACTION_PROMPT
        )
        self.db = db
        self.embedding = embedding

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        keywords = await self.agent.run(query)
        similar_node_ids = {}
        if keywords and keywords.output:
            for kw in keywords.output.keywords:
                embed_res = await self.embedding.embed_query(kw)
                keyword_embeds = embed_res.embeddings[0]
                sim_node_hits = await self.db.vector_search(
                    labels=["__entity__"], query_embedding=keyword_embeds, limit=limit
                )
                for node_hit in sim_node_hits:
                    sim_score = node_hit.score
                    id = node_hit.node.id
                    name = node_hit.node.name
                    key = (id, name)
                    if key in similar_node_ids:
                        similar_node_ids[key] = max(similar_node_ids[key], sim_score)
                    else:
                        similar_node_ids[key] = sim_score
        top_k_words = sorted(
            similar_node_ids.items(), key=lambda x: x[1], reverse=True
        )[:limit]
        similar_node_ids = {k: score for k, score in top_k_words}
        node_results = []
        for (id, name), similarity in similar_node_ids.items():
            path_text = f"Source Node: {name} (id: {id}), similarity: {similarity}\n"
            q_res = self.db.query(
                f"MATCH (n {{id: '{id}'}})-[r]-(n2) RETURN n.name, r.label, n2 LIMIT 5"
            )
            for record in q_res.result_set:
                relation_name = record[1]
                target_name = record[2].properties["name"]
                target_content = record[2].properties.get("content", "")
                path_text += f"\n  -[{relation_name}]-> NAME: {target_name}: CONTENT: {target_content}\n"
            node_results.append((id, name, similarity, path_text))
        hits = [
            NodeHit(
                Node(
                    id=id,
                    label="__entity__",
                    semantic_key=f"__entity__:{id}-{name}-path-expansion",
                    name=name,
                    properties={"content": text},
                ),
                score=similarity,
                matched_on="path_expansion",
            )
            for id, name, similarity, text in node_results
        ]
        return hits
