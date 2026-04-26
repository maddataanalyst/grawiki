# Graph Models

These models define the persisted graph schema used across ingestion, retrieval, memory, and deduplication. [`Node`][grawiki.graph.models.Node], [`Relationship`][grawiki.graph.models.Relationship], and [`KnowledgeGraph`][grawiki.graph.models.KnowledgeGraph] are the core durable shapes; `DocumentNode`, `ChunkNode`, and `MemoryNode` are specialized system node types layered on top.

::: grawiki.graph.models
