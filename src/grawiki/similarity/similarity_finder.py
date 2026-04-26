"""High-level entity similarity orchestration."""

from __future__ import annotations

from collections import defaultdict

from grawiki.db.base import GraphDB, NodeHit
from grawiki.graph.models import Node
from grawiki.similarity.base import EntitySimilarityMatcher
from grawiki.similarity.models import (
    EntityDuplicateCandidates,
    EntitySimilarityResult,
    SemanticKeyCollisionCandidates,
)
from grawiki.similarity.vector import VectorEntitySimilarityMatcher


class EntitySimilarityFinder:
    """Inspect entity collisions and search for merge candidates.

    Parameters
    ----------
    db : GraphDB
        Graph database adapter used to load persisted entity nodes.
    matcher : EntitySimilarityMatcher | None, optional
        Similarity matcher implementation used to produce candidate matches.
        Defaults to :class:`~grawiki.similarity.vector.VectorEntitySimilarityMatcher`.
    """

    def __init__(
        self,
        db: GraphDB,
        *,
        matcher: EntitySimilarityMatcher | None = None,
    ) -> None:
        """Initialize the entity similarity finder.

        Parameters
        ----------
        db : GraphDB
            Graph database adapter used to load persisted entity nodes.
        matcher : EntitySimilarityMatcher | None, optional
            Similarity matcher used for candidate selection. When omitted, the
            finder uses :class:`~grawiki.similarity.vector.VectorEntitySimilarityMatcher`.
        """

        self.db = db
        self.matcher = matcher or VectorEntitySimilarityMatcher(
            db=db,
        )

    async def find_semantic_key_collisions(
        self,
        *,
        include_embeddings: bool = False,
    ) -> dict[str, list[Node]]:
        """Return entity groups whose semantic key occurs more than once.

        Parameters
        ----------
        include_embeddings : bool, optional
            Whether the returned entities should include embeddings.

        Returns
        -------
        dict[str, list[Node]]
            Entity groups keyed by semantic key, including only keys with more
            than one entity.

        Notes
        -----
        This method is intended as a lightweight integrity check before running
        more expensive similarity matching.
        """

        entities = await self.db.list_entities(include_embeddings=include_embeddings)
        grouped: dict[str, list[Node]] = defaultdict(list)
        for entity in entities:
            grouped[entity.semantic_key].append(entity)
        return {
            semantic_key: nodes
            for semantic_key, nodes in grouped.items()
            if len(nodes) > 1
        }

    async def search(
        self,
        entity: Node,
        *,
        limit: int = 10,
        threshold: float | None = None,
        candidates: list[Node] | None = None,
    ) -> list[NodeHit]:
        """Return similarity candidates for one entity.

        Parameters
        ----------
        entity : Node
            Source entity used as the similarity query.
        limit : int, optional
            Maximum number of candidate hits to return.
        threshold : float | None, optional
            Optional strategy-specific minimum score.
        candidates : list[Node] | None, optional
            Optional pre-filtered candidate pool.

        Returns
        -------
        list[NodeHit]
            Ranked candidate hits.

        """

        return await self.matcher.search(
            entity=entity,
            limit=limit,
            threshold=threshold,
            candidates=candidates,
        )

    async def find_collision_candidates(
        self,
        *,
        collisions: dict[str, list[Node]] | None = None,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[SemanticKeyCollisionCandidates]:
        """Run similarity search inside semantic-key collision groups.

        Parameters
        ----------
        collisions : dict[str, list[Node]] | None, optional
            Precomputed semantic-key collision groups. When omitted, collisions
            are loaded from the database.
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
        Similarity matching is restricted to members of each collision group so
        that the output is safe to use as a merge-candidate inspection aid.
        """

        collision_groups = collisions or await self.find_semantic_key_collisions(
            include_embeddings=True
        )
        groups: list[SemanticKeyCollisionCandidates] = []
        for semantic_key, entities in collision_groups.items():
            results: list[EntitySimilarityResult] = []
            for entity in entities:
                hits = await self.search(
                    entity,
                    limit=limit,
                    threshold=threshold,
                    candidates=entities,
                )
                results.append(EntitySimilarityResult(source=entity, hits=hits))
            groups.append(
                SemanticKeyCollisionCandidates(
                    semantic_key=semantic_key,
                    results=results,
                )
            )
        groups.sort(key=lambda group: group.semantic_key)
        return groups

    async def find_similarity_candidates(
        self,
        *,
        limit: int = 10,
        threshold: float | None = None,
        entities: list[Node] | None = None,
        skip_semantic_key_collisions: bool = False,
        semantic_key_collisions: dict[str, list[Node]] | None = None,
    ) -> list[EntitySimilarityResult]:
        """Run a broader matcher-based duplicate scan across entities.

        Parameters
        ----------
        limit : int, optional
            Maximum number of candidate hits returned per source entity.
        threshold : float | None, optional
            Optional matcher-specific minimum score.
        entities : list[Node] | None, optional
            Optional explicit entity pool. When omitted, persisted entities are
            loaded from the database with embeddings included.
        skip_semantic_key_collisions : bool, optional
            Whether entities already participating in an exact semantic-key
            collision should be excluded from this broader similarity scan.
        semantic_key_collisions : dict[str, list[Node]] | None, optional
            Precomputed semantic-key collision groups used when
            ``skip_semantic_key_collisions`` is enabled. When omitted, the
            groups are loaded from the database.

        Returns
        -------
        list[EntitySimilarityResult]
            Ranked similarity candidates grouped by source entity.

        Notes
        -----
        Each entity pair is considered at most once by searching only against
        entities that appear later in the candidate order.
        """

        available = (
            list(entities)
            if entities is not None
            else await self.db.list_entities(include_embeddings=True)
        )
        skipped_ids: set[str] = set()
        if skip_semantic_key_collisions:
            collision_groups = (
                semantic_key_collisions
                or await self.find_semantic_key_collisions(include_embeddings=True)
            )
            skipped_ids = _entity_ids_in_collisions(collision_groups)

        filtered = [entity for entity in available if entity.id not in skipped_ids]
        results: list[EntitySimilarityResult] = []
        for index, entity in enumerate(filtered):
            hits = await self.search(
                entity,
                limit=limit,
                threshold=threshold,
                candidates=filtered[index + 1 :],
            )
            if not hits:
                continue
            results.append(EntitySimilarityResult(source=entity, hits=hits))
        return results

    async def find_duplicate_candidates(
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
            Combined report containing exact collisions, verified collision-group
            candidates, and broader matcher-based similarity candidates.
        """

        semantic_key_collisions = await self.find_semantic_key_collisions(
            include_embeddings=True
        )
        semantic_key_collision_candidates = await self.find_collision_candidates(
            collisions=semantic_key_collisions,
            limit=limit,
            threshold=threshold,
        )
        similarity_candidates = await self.find_similarity_candidates(
            limit=limit,
            threshold=threshold,
            skip_semantic_key_collisions=skip_semantic_key_collisions_in_similarity_scan,
            semantic_key_collisions=semantic_key_collisions,
        )
        return EntityDuplicateCandidates(
            semantic_key_collisions=semantic_key_collisions,
            semantic_key_collision_candidates=semantic_key_collision_candidates,
            similarity_candidates=similarity_candidates,
        )


def _entity_ids_in_collisions(collisions: dict[str, list[Node]]) -> set[str]:
    """Return the set of entity ids participating in semantic-key collisions.

    Parameters
    ----------
    collisions : dict[str, list[Node]]
        Collision groups keyed by semantic key.

    Returns
    -------
    set[str]
        Distinct entity identifiers present across all collision groups.
    """

    return {entity.id for group in collisions.values() for entity in group}
