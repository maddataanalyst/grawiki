# Store and recall memory

This guide shows how to persist a memory with [`GraphRAG.remember`][grawiki.rag.graph_rag.GraphRAG.remember] and retrieve it later with [`GraphRAG.recall`][grawiki.rag.graph_rag.GraphRAG.recall].

## Persist a memory

```python
memory = await rag.remember(
    "Filip is a researcher working on graph-backed agent memory.",
    metadata={"user_id": "user-123"},
)
```

`remember(...)` stores a `__memory__` node, embeds the memory text, extracts entities from the memory body, and persists those links back into the graph.

You can also pass `related_node_ids` when the memory should be attached to existing graph nodes.

## Recall matching memories

```python
hits = await rag.recall(
    "What do we know about Filip's work?",
    user_id="user-123",
    hops=2,
    limit=5,
)
```

`recall(...)` searches memory nodes first and then expands connected graph context around the selected memory hits.

## Minimal end-to-end example

```python
await rag.remember(
    "The user prefers graph-based retrieval experiments.",
    metadata={"user_id": "user-123"},
)

hits = await rag.recall(
    "What retrieval approach does the user prefer?",
    user_id="user-123",
    hops=1,
    limit=3,
)
```

The second maintained notebook also wraps these calls as `pydantic_ai` tools. The public GraWiki API remains the primary interface, and it is the better place to start when building application code.
