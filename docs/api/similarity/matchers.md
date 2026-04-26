# Matchers

Matchers provide the scoring strategy behind duplicate-candidate discovery.

- [`EntitySimilarityMatcher`][grawiki.similarity.base.EntitySimilarityMatcher] defines the protocol.
- [`VectorEntitySimilarityMatcher`][grawiki.similarity.vector.VectorEntitySimilarityMatcher] compares entity embeddings with cosine similarity.
- [`RapidFuzzEntitySimilarityMatcher`][grawiki.similarity.fuzzy.RapidFuzzEntitySimilarityMatcher] compares entity names with string similarity.

::: grawiki.similarity.base

::: grawiki.similarity.vector

::: grawiki.similarity.fuzzy
