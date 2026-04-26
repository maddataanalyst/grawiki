# Text Retriever

[`TextRetriever`][grawiki.retrieval.text.TextRetriever] is the default text-oriented retrieval implementation. It embeds queries on the application side, calls the raw `GraphDB` search primitives, and deduplicates overlapping hits.

::: grawiki.retrieval.text.TextRetriever
