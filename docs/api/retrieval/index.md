# Retrieval Overview

GraWiki keeps query-time embedding and ranking in the retrieval layer rather than in the database adapter. The common contract is [`Retriever`][grawiki.retrieval.base.Retriever].

The default `GraphRAG` setup combines two complementary retrievers:

- [`TextRetriever`][grawiki.retrieval.text.TextRetriever] for vector or full-text lookup over stored nodes.
- [`KeywordsPathRetriever`][grawiki.retrieval.keywords.KeywordsPathRetriever] for extracting keyword seeds, finding related entities, and attaching one-hop graph context.

::: grawiki.retrieval.base
