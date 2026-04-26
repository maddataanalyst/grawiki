"""Public GraWiki package surface.

The top-level package intentionally re-exports :class:`grawiki.GraphRAG` as the
main entry point for users who want document ingestion, retrieval, memory, and
entity-deduplication workflows through one facade.
"""

from grawiki.rag.graph_rag import GraphRAG

__all__ = ["GraphRAG"]
