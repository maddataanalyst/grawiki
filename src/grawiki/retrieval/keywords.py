"""Keyword-driven retriever with graph-context expansion."""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from grawiki.core.embedding import Embedding
from grawiki.db.base import GraphDB, NeighborRelationship, NodeHit
from grawiki.graph.models import Node
from grawiki.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class Keywords(BaseModel):
    """Structured output returned by the keyword extraction model.

    Parameters
    ----------
    keywords : Sequence[str]
        Distinct key phrases extracted from the query.
    """

    keywords: Sequence[str] = Field(
        ..., description="List of distinct key phrases extracted from the query."
    )


class KeywordExtractor(Protocol):
    """Protocol for extracting key phrases from a raw query."""

    async def extract(self, query: str) -> list[str]:
        """Extract keyword phrases from ``query``."""


KEYWORDS_EXTRACTION_PROMPT = (
    "Extract a short list of distinct key phrases from the user query. "
    "Return only the key phrases."
)


class _AgentKeywordExtractor:
    """Keyword extractor backed by a PydanticAI structured-output agent."""

    def __init__(self, model: str) -> None:
        self._agent = Agent(
            model=model,
            output_type=Keywords,
            system_prompt=KEYWORDS_EXTRACTION_PROMPT,
        )

    async def extract(self, query: str) -> list[str]:
        """Extract keyword phrases from ``query`` using the configured agent."""

        result = await self._agent.run(query)
        if result is None or result.output is None:
            return []
        return list(result.output.keywords)


class KeywordsPathRetriever(Retriever):
    """Retrieve entity seeds from extracted keywords and attach graph context.

    Parameters
    ----------
    model : str
        Chat model used by the default keyword extractor.
    db : GraphDB
        Graph database adapter used for vector search and neighbor expansion.
    embedding : Embedding
        Shared embedder used to embed extracted keywords.
    keyword_extractor : KeywordExtractor | None, optional
        Override for tests or alternative extraction strategies.
    path_limit : int, optional
        Maximum number of one-hop relationships included per seed entity.
    """

    def __init__(
        self,
        *,
        model: str,
        db: GraphDB,
        embedding: Embedding,
        keyword_extractor: KeywordExtractor | None = None,
        path_limit: int = 5,
    ) -> None:
        if path_limit < 1:
            raise ValueError("path_limit must be at least 1.")
        self.model = model
        self.db = db
        self.embedding = embedding
        self.keyword_extractor = keyword_extractor or _AgentKeywordExtractor(model)
        self.path_limit = path_limit

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        """Retrieve entity hits enriched with one-hop graph context.

        Parameters
        ----------
        query : str
            Raw user query text.
        limit : int, optional
            Maximum number of enriched entity hits to return.

        Returns
        -------
        list[NodeHit]
            Scored entity hits with synthesized context text stored in
            ``node.properties["content"]``.
        """

        if limit < 1:
            return []

        raw_keywords = await self.keyword_extractor.extract(query)
        keywords = _normalize_keywords(raw_keywords)
        if not keywords:
            return []

        best_hits: dict[tuple[str, str], NodeHit] = {}
        for keyword in keywords:
            embedding_result = await self.embedding.embed_query(keyword)
            if not embedding_result.embeddings:
                logger.warning(
                    "Skipping keyword %r because the embedder returned no vectors.",
                    keyword,
                )
                continue
            keyword_embedding = list(embedding_result.embeddings[0])
            seed_hits = await self.db.vector_search(
                labels=["__entity__"],
                query_embedding=keyword_embedding,
                limit=limit,
            )
            for hit in seed_hits:
                key = (hit.node.label, hit.node.id)
                current = best_hits.get(key)
                if current is None or hit.score > current.score:
                    best_hits[key] = hit

        if not best_hits:
            return []

        top_hits = sorted(best_hits.values(), key=lambda hit: hit.score, reverse=True)[
            :limit
        ]
        contexts = await self.db.neighbor_relationships(
            node_ids=[hit.node.id for hit in top_hits],
            limit_per_node=self.path_limit,
        )
        return [
            NodeHit(
                node=_path_node_from_hit(hit, contexts.get(hit.node.id, [])),
                score=hit.score,
                matched_on="keyword_path",
            )
            for hit in top_hits
        ]


def _normalize_keywords(keywords: Sequence[str]) -> list[str]:
    """Return distinct, non-empty keywords preserving first occurrence."""

    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        cleaned = keyword.strip()
        if not cleaned:
            continue
        dedupe_key = cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(cleaned)
    return normalized


def _path_node_from_hit(
    hit: NodeHit, relationships: Sequence[NeighborRelationship]
) -> Node:
    """Return an entity node annotated with synthesized graph context text."""

    properties = dict(hit.node.properties)
    properties["content"] = _build_path_text(hit, relationships)
    return Node(
        id=hit.node.id,
        label=hit.node.label,
        semantic_key=hit.node.semantic_key,
        name=hit.node.name,
        properties=properties,
        embedding=list(hit.node.embedding),
    )


def _build_path_text(
    hit: NodeHit, relationships: Sequence[NeighborRelationship]
) -> str:
    """Render one entity hit plus one-hop graph context as readable text."""

    lines = [
        f"Source Node: {hit.node.name} (id: {hit.node.id}), similarity: {hit.score:.4f}"
    ]
    if not relationships:
        lines.append("No connected graph context found.")
        return "\n".join(lines)

    for relationship in relationships:
        target = relationship.target
        content = _node_content(target)
        line = (
            f"-[{relationship.relationship_label}]-> "
            f"NAME: {target.name}; LABEL: {target.label}"
        )
        if content:
            line += f"; CONTENT: {content}"
        lines.append(line)
    return "\n".join(lines)


def _node_content(node: Node) -> str:
    """Return the best available text content for a graph node."""

    explicit_content = getattr(node, "content", "")
    if isinstance(explicit_content, str) and explicit_content:
        return explicit_content
    content_property = node.properties.get("content", "")
    return content_property if isinstance(content_property, str) else ""
