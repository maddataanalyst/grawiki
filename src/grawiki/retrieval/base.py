from typing import Protocol

from grawiki.db.base import NodeHit


class Retriever(Protocol):
    """Protocol for retriever implementations."""

    async def retrieve(
        self, query: str, limit: int = 5, *args, **kwargs
    ) -> list[NodeHit]:
        """Run a retrieval query and return a list of hits."""
