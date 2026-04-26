# GraWiki Implementation Plan

This plan reorganizes GraWiki around four modes of operation (RAG
ingestion, RAG querying, memory remember, memory recall) with two
top-level public facades (`GraphRAG`, `GraphMemory`) built on cleanly
split primitive layers.

It is an incremental migration: each step is shippable on its own and
keeps the test suite green.

---

## Target architecture

```
src/grawiki/
├── __init__.py                 # exports GraphRAG, GraphMemory
├── core/
│   ├── commons.py              # Document, Chunk
│   └── embedding.py            # Embedder protocol + DefaultEmbedder wrapper  ✅ done
├── doc_processing/             # unchanged
│   ├── document_processing.py
│   └── chunkers.py
├── graph/
│   ├── models.py               # Node / Relationship / KnowledgeGraph + Document/Chunk/MemoryNode  ✅ (Extracted* moved out)
│   ├── extraction.py           # KnowledgeGraphExtractor + Extracted* transient types             ✅ done
│   ├── prompts.py              # KG_EXTRACTION_PROMPT                                             ✅ done (renamed)
│   └── dedup.py                # (future) entity deduplication
├── db/                         # STORAGE ENGINE ONLY — no embedding, no policy
│   ├── base.py                 # GraphDB — persistence + raw search primitives
│   ├── cypher.py               # label-parameterized Cypher builders (currently cypher_queries.py)
│   └── falkordb.py             # FalkorGraphDB
├── retrieval/                  # RETRIEVAL STRATEGY — owns embedder on query path
│   ├── retriever.py            # Retriever: fulltext / vector / expand
│   └── types.py                # NodeHit (currently in db/base.py), RecallHit, etc.
├── rag/
│   ├── __init__.py             # exports GraphRAG
│   ├── graph_rag.py            # GraphRAG facade — ingest + search
│   └── synthesis.py            # (future) LLM answer synthesis
└── memory/
    ├── __init__.py             # exports GraphMemory
    └── graph_memory.py         # GraphMemory facade — remember + recall
```

### Key architectural invariants (do not break)

- **DB = storage engine, Retriever = strategy.** The DB never embeds
  queries and never ranks or filters by recency/strategy; it exposes
  raw primitives that take explicit inputs (`vector_search` takes a
  pre-computed embedding). The retrieval layer owns the embedder on
  the query side and composes DB primitives.
- **One embedder instance per pipeline.** Both `KnowledgeGraphExtractor`
  and the retrieval layer take an injected `Embedder` — never
  construct one internally.
- **Two-phase node lifecycle.** `ExtractedNode` (pre-id, name-as-key)
  lives in `graph/extraction.py`; `Node` (post-id, UUID-as-key) lives
  in `graph/models.py`. Do not merge them. Making `Node.id` optional
  is explicitly rejected — trades type safety for apparent simplicity.
- **Flat `list[NodeHit]` on the new primitives.** The old
  grouped-by-family dict shape is legacy; callers that want grouping
  compute it themselves from `hit.node.label`.
- **Class naming: prefer noun forms.** `Retriever` not `Querier`;
  `MemoryRecall` over `MemoryRecaller`. Methods over agent-class names
  where possible (recall is a method on `GraphMemory`, not its own
  class).

### Public API shape (what the user sees)

```python
rag = grawiki.GraphRAG(model=..., embedding_model=..., db=db)
await rag.ingest(path)
hits = await rag.search("question")

memory = grawiki.GraphMemory(model=..., embedding_model=..., db=db)
await memory.remember("Met with Alice about Q3 plan")
recalled = await memory.recall("Alice", since=last_week)

# As agent tools
@agent.tool
async def search(q: str): return await rag.search(q)
@agent.tool
async def recall(q: str): return await memory.recall(q)
```

Both facades can share a `GraphDB`, which enables memories to link to
entities that RAG ingestion already wrote.

---

## Status — what is already done

### ✅ Step 1 — Shared embedder

- Added `src/grawiki/core/embedding.py` with `Embedder` Protocol +
  `DefaultEmbedder` wrapper around `pydantic_ai.Embedder`.
- `KnowledgeGraphExtractor.__init__` now takes `embedder: Embedder`
  instead of `embedding: str`.
- `GrawikiPipeline` constructs one `DefaultEmbedder` and shares it
  with the extractor. No more duplicate model warmups.

### ✅ Step 2 — Flatten graph submodule names

- `src/grawiki/graph/graph_extraction.py` → `graph/extraction.py`.
- `src/grawiki/graph/graph_prompts.py` → `graph/prompts.py`.
- Test files renamed to match (`test_extraction.py`, `test_prompts.py`).
- `agent_tools/CODEMAP.md` updated.

### ✅ Step 3a — Read-path DB primitives + `NodeHit`

- Added `NodeHit` dataclass in `db/base.py`: `(node: Node, score: float,
  matched_on: str)`. Used by the new primitives.
- Added four abstract methods on `GraphDB`:
  - `ensure_indexes(*, labels, vector_dims=None)` — create fulltext +
    vector indexes for any label; supports `__memory__`.
  - `fulltext_search(*, labels, query_text, limit) -> list[NodeHit]`
  - `vector_search(*, labels, query_embedding, limit) -> list[NodeHit]`
    (takes pre-computed embedding — strategy concern)
  - `neighbors(*, node_ids, rel_types, depth) -> list[Node]` — walks
    outgoing edges; `rel_types` are *validated* (regex), not sanitized,
    so reserved types like `__mentions__` pass through unmangled.
- Implemented all four in `FalkorGraphDB`. Search primitives
  reconstruct the correct `Node` subclass (`DocumentNode` / `ChunkNode`
  / `MemoryNode` / base `Node`) from Cypher rows with full fields.
- Four new tests in `tests/db/test_falkordb.py` cover each primitive.
- Legacy `search`, `save_docs_and_chunks_to_db`,
  `save_entities_and_rels` are **unchanged** — still abstract, still
  dict-shaped return.

### ✅ Bonus — Move `Extracted*` types out of `graph/models.py`

- `ExtractedNode`, `ExtractedRelationship`, `ExtractedKnowledgeGraph`
  moved into `graph/extraction.py`.
- `Node.from_extracted_node` classmethod removed; replaced by
  module-level `_node_from_extracted(extracted_node) -> Node` helper
  in `extraction.py` (the extractor is the only caller).
- Test moved from `test_models.py` to `test_extraction.py`.
- `graph/models.py` now exposes pure durable domain schema.

**Test status**: 24 tests pass. `uv run pre-commit run --all-files` clean.

---

## Status — what is still to do

### ⬜ Step 3b — Write-path DB primitives + legacy wrappers

Add `upsert_nodes` and `upsert_relationships` to `GraphDB` and
implement them in `FalkorGraphDB`. Then convert the three legacy
methods into concrete wrappers over the primitives.

**Signatures**:

```python
@abstractmethod
async def upsert_nodes(self, nodes: Sequence[Node]) -> None:
    """Upsert nodes. Dispatches on label for persistence semantics."""

@abstractmethod
async def upsert_relationships(self, rels: Sequence[Relationship]) -> None:
    """Upsert relationships between existing nodes (matched by id)."""
```

**Design notes**:

- `upsert_nodes` handles *node properties only*. Parent/child linkage
  (doc → chunk via `__has_chunk__`, chunk → entity via `__mentions__`)
  becomes explicit relationships created via `upsert_relationships`.
  This is a semantic change from today, where linkage is baked into
  the chunk/entity upsert Cypher.
- `upsert_relationships` matches endpoints by `id` (not
  `semantic_key`). The existing entity-to-entity MERGE by
  `semantic_key` pattern should be preserved via a
  type-check-and-dispatch inside the method, OR via a second primitive
  — decide during implementation. Start with id-based matching; the
  entity MERGE-by-semantic-key pattern is specific to
  LLM-extracted entity deduplication and may move into the retrieval
  layer as an upsert policy later.
- Once primitives work, rewrite the legacy methods as wrappers:

```python
# Concrete in base class (not abstract anymore)
async def save_docs_and_chunks_to_db(self, doc_nodes, chunk_nodes):
    await self.upsert_nodes([*doc_nodes, *chunk_nodes])
    has_chunk = [
        Relationship(
            id=str(uuid.uuid4()),
            source=chunk.document_id,
            target=chunk.id,
            label="__has_chunk__",
        )
        for chunk in chunk_nodes
    ]
    await self.upsert_relationships(has_chunk)

async def save_entities_and_rels(self, chunks, chunk_graphs):
    # similar decomposition
    ...

async def search(self, query, method, *, limit=10, query_embedding=None):
    if method == "fulltext":
        hits = await self.fulltext_search(
            labels=["__document__", "__chunk__", "__entity__"],
            query_text=query, limit=limit,
        )
    else:
        if query_embedding is None:
            raise ValueError("Vector search requires a query embedding.")
        hits = await self.vector_search(
            labels=["__document__", "__chunk__", "__entity__"],
            query_embedding=query_embedding, limit=limit,
        )
    return _group_hits_by_label(hits, limit=limit)  # preserves dict shape
```

**Parity tests** must confirm identical graph state between the legacy
calls and their new wrapper implementations. The existing
`test_save_documents_chunks_entities_and_relationships_with_indexes`
test already asserts the end-state; it should continue to pass
unchanged once the wrappers are in place.

### ⬜ Step 4 — Parametrize Cypher

- Rename `src/grawiki/db/cypher_queries.py` → `db/cypher.py`.
- Replace per-family builders (`build_document_upsert_query`,
  `build_chunk_upsert_query`, `build_entity_upsert_query`,
  `build_relationship_upsert_query`) with label-parameterized ones:
  - `upsert_node_cypher(labels, properties)` — one implementation for
    any label.
  - `upsert_rel_cypher(rel_type)` — works for any rel type between any
    nodes matched by id.
- Keep `sanitize_cypher_identifier` for entity-ontology labels where
  the LLM may produce weird characters. Do not use it on reserved
  system rel types like `__mentions__` (strips underscores; use regex
  validation instead, same pattern as in `FalkorGraphDB.neighbors`).
- Existing tests in `tests/db/test_cypher_queries.py` will need to be
  renamed/updated; the underlying Cypher output can change shape as
  long as the end-to-end integration tests in `test_falkordb.py` stay
  green.

### ⬜ Step 5 — Introduce `retrieval/` layer

Create `src/grawiki/retrieval/retriever.py`:

```python
class Retriever:
    """Embed queries, call DB primitives, compose results."""

    def __init__(self, db: GraphDB, embedder: Embedder): ...

    async def fulltext(self, query: str, *, labels, limit=10) -> list[NodeHit]:
        return await self.db.fulltext_search(
            labels=labels, query_text=query, limit=limit,
        )

    async def vector(self, query: str, *, labels, limit=10) -> list[NodeHit]:
        result = await self.embedder.embed_query(query)
        embedding = list(result.embeddings[0])
        return await self.db.vector_search(
            labels=labels, query_embedding=embedding, limit=limit,
        )

    async def expand(self, seeds: list[NodeHit], *, rel_types=None, depth=1) -> list[Node]:
        return await self.db.neighbors(
            node_ids=[h.node.id for h in seeds],
            rel_types=rel_types, depth=depth,
        )
```

- Move the `_deduplicate_hits` helper out of `FalkorGraphDB` into a
  private utility on `Retriever` — it is a strategy concern.
- Keep the class small on the first pass; hybrid / rerankers / recency
  decay land later.
- Unit-test with a fake `GraphDB` (tests/retrieval/test_retriever.py).

### ⬜ Step 6 — Build `GraphRAG` facade

Create `src/grawiki/rag/graph_rag.py`:

```python
class GraphRAG:
    def __init__(
        self,
        *,
        model: str,
        embedding_model: str,
        db: GraphDB,
        chunking_strategy: str = "sentence",
        max_workers: int = 4,
    ):
        self._embedder = DefaultEmbedder(embedding_model)
        self._chunker = Chunker(strategy=chunking_strategy)
        self._extractor = KnowledgeGraphExtractor(model=model, embedder=self._embedder)
        self._db = db
        self._retriever = Retriever(db=db, embedder=self._embedder)
        self._max_workers = max_workers

    async def ingest(self, path: Path) -> None: ...        # ← ex-GrawikiPipeline.ingest_file
    async def ingest_text(self, text: str, title: str) -> None: ...

    async def search(
        self, query: str, *,
        method: Literal["fulltext", "vector"] = "vector",
        limit: int = 10,
    ) -> list[NodeHit]: ...                                 # ← ex-GrawikiPipeline.search
```

- Port the body of `GrawikiPipeline.ingest_file` wholesale into
  `GraphRAG.ingest`. Port `GrawikiPipeline.search` into `GraphRAG.search`,
  now using `Retriever` instead of `GraphDB.search` directly.
- Delete `src/grawiki/core/pipeline.py`.
- Add `from grawiki.rag import GraphRAG` to top-level
  `src/grawiki/__init__.py`.
- Move `tests/core/test_pipeline.py` → `tests/rag/test_graph_rag.py`
  with import updates. Behavior-preservation: the existing test assertions
  should pass unchanged against the new facade.

### ⬜ Step 7 — Wire `__memory__` into index setup used by facades

When `GraphRAG` / `GraphMemory` call `ensure_indexes` for setup, include
`__memory__` in the label set so memory indexes are created alongside
the others. Smoke test: constructing a `GraphMemory` with a fresh DB
results in fulltext + vector indexes for `__memory__` after the first
operation.

### ⬜ Step 8 — Build `GraphMemory.remember`

Create `src/grawiki/memory/graph_memory.py`:

```python
class GraphMemory:
    def __init__(
        self,
        *,
        model: str,                # accepted for API symmetry; unused in MVP
        embedding_model: str,
        db: GraphDB,
        chunking_strategy: str = "sentence",
    ):
        self._embedder = DefaultEmbedder(embedding_model)
        self._chunker = Chunker(strategy=chunking_strategy)
        self._db = db
        self._retriever = Retriever(db=db, embedder=self._embedder)

    async def remember(
        self, text: str, *, metadata: dict[str, str] | None = None,
    ) -> MemoryNode: ...
```

**MVP algorithm** (no KG extraction on remember):

1. Chunk+embed the input text.
2. Create a `MemoryNode` (use `creation_date` default = now ISO).
3. `upsert_nodes([memory_node])` (plus any chunk-style memory chunks
   if chunked — decide if memory ingestion produces one node or one
   per chunk; MVP: one node per remember call, content = full text).
4. Use `self._retriever.vector(text, labels=["__entity__"], limit=5)`
   to find top matching entities.
5. `upsert_relationships([...])` to write `__mentions__` edges from
   memory to each matched entity.

**Test**: populate entities via `GraphRAG.ingest(fixture_doc)`, then
`GraphMemory.remember("text about an entity")` on the same DB, then
assert the memory and `__mentions__` edges exist.

### ⬜ Step 9 — Build `GraphMemory.recall`

```python
async def recall(
    self, query: str, *,
    since: datetime | None = None,
    limit: int = 10,
    hop_entities: bool = True,
) -> list[RecallHit]: ...
```

Implementation:

1. `Retriever.vector(query, labels=["__memory__"], limit=limit)`.
2. If `since` is set, post-filter hits by `node.creation_date >= since`
   in Python (do not push to DB for MVP — keeps `search_nodes.filters`
   out of scope).
3. If `hop_entities`, `Retriever.expand(seeds, rel_types=["__mentions__"], depth=1)`.
4. Zip each memory hit with its mentioned entities into a `RecallHit`
   shape (define in `retrieval/types.py` or `memory/types.py`).

**Test**: seed memories with fixed `created_at` timestamps, verify
recency filter and hop expansion.

### ⬜ Step 10 — Delete legacy wrappers

Once `GraphRAG` and `GraphMemory` fully cover the use cases and their
tests are green, delete:

- `save_documents_and_chunks`, `save_docs_and_chunks_to_db`,
  `save_entities_and_rels` from `GraphDB`.
- The dict-shaped `search()` from `GraphDB`.
- Any remaining test that exercised them directly.

---

## Explicitly out of plan (future work)

- **Entity deduplication** (`src/grawiki/graph/dedup.py`) — invoked as
  a post-ingest hook by `GraphRAG.ingest`. Strategy choice
  (semantic_key merge vs embedding threshold vs LLM arbiter) is a
  separate design conversation.
- **LLM answer synthesis** (`src/grawiki/rag/synthesis.py`) — called
  inside `GraphRAG.search` to turn retrieved context into an answer.
- **Hybrid search** in `Retriever` — RRF or weighted merge of fulltext
  and vector. Add only when a real use case asks for it.
- **KG extraction on remember** — opt-in flag on `GraphMemory.remember`
  that runs the extractor over memory text. MVP defers this because
  it doubles cost/latency per memory.

---

## Verification commands

After every step:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Ship each step independently.
