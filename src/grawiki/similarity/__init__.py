"""Entity similarity search and collision inspection."""

from grawiki.similarity.base import EntitySimilarityMatcher
from grawiki.similarity.fuzzy import RapidFuzzEntitySimilarityMatcher
from grawiki.similarity.models import (
    EntityDuplicateCandidates,
    EntitySimilarityResult,
    SemanticKeyCollisionCandidates,
)
from grawiki.similarity.similarity_finder import EntitySimilarityFinder
from grawiki.similarity.vector import VectorEntitySimilarityMatcher

__all__ = [
    "EntityDuplicateCandidates",
    "EntitySimilarityResult",
    "EntitySimilarityMatcher",
    "EntitySimilarityFinder",
    "RapidFuzzEntitySimilarityMatcher",
    "SemanticKeyCollisionCandidates",
    "VectorEntitySimilarityMatcher",
]
