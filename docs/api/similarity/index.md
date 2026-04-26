# Similarity and Deduplication

GraWiki exposes entity deduplication as a user-facing workflow rather than only as low-level helpers.

The recommended progression is:

1. Inspect exact semantic-key collisions with [`GraphRAG.find_entity_collision_candidates`][grawiki.rag.graph_rag.GraphRAG.find_entity_collision_candidates] or [`EntitySimilarityFinder`][grawiki.similarity.similarity_finder.EntitySimilarityFinder].
2. Run the broader duplicate scan with [`GraphRAG.find_entity_duplicate_candidates`][grawiki.rag.graph_rag.GraphRAG.find_entity_duplicate_candidates].
3. Execute merges with [`GraphRAG.dedupe_entities`][grawiki.rag.graph_rag.GraphRAG.dedupe_entities] when the candidates are acceptable.

The duplicate workflow has two stages:

- Exact collision detection by `semantic_key`.
- Broader matcher-based scanning using vector or fuzzy similarity.

Supporting APIs you will usually need:

- [`EntitySimilarityFinder`][grawiki.similarity.similarity_finder.EntitySimilarityFinder]
- [`EntitySimilarityMatcher`][grawiki.similarity.base.EntitySimilarityMatcher]
- [`VectorEntitySimilarityMatcher`][grawiki.similarity.vector.VectorEntitySimilarityMatcher]
- [`RapidFuzzEntitySimilarityMatcher`][grawiki.similarity.fuzzy.RapidFuzzEntitySimilarityMatcher]
- [`EntityDuplicateCandidates`][grawiki.similarity.models.EntityDuplicateCandidates]
- [`SemanticKeyCollisionCandidates`][grawiki.similarity.models.SemanticKeyCollisionCandidates]
- [`MergeReport`][grawiki.similarity.deduplication.MergeReport]

The same similarity stack also powers ingest-time entity resolution through [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG] when `resolve_entities_on_ingest` is enabled.

For a task-oriented walkthrough, see [How to deduplicate entities](../../how-to/deduplicate-entities.md).

::: grawiki.similarity
