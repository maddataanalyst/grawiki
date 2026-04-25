# Similar-Entity Usage Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use `EntitySimilarityFinder` in two workflows — (A) resolve newly-extracted entities against persisted duplicates *before* persisting, reusing existing ids; and (B) merge already-persisted duplicates into a single master node.

**Architecture:** Neither scenario needs a new orchestrator class. `EntitySimilarityFinder.search(...)` already produces the candidate matches we need. Scenario A becomes a private method on `GraphRAG` that rewrites `chunk_graphs` between extraction and persistence. Scenario B adds one genuinely new piece — a `GraphDB.merge_entity_nodes` storage primitive — plus a small master-selection helper and a `GraphRAG.dedupe_entities` method. Strategy stays out of the DB adapter; the adapter only owns edge redirection and node deletion.

**Tech Stack:** Python 3.12, `pydantic`, `pydantic_ai`, FalkorDBLite via `FalkorGraphDB`, `pytest` with async support, `uv` for dependency management.

---

## Progress

| Scenario | Task | Status | Notes |
| --- | --- | --- | --- |
| A | A.1 Add resolution hook to `GraphRAG` | ✅ Shipped | commits `138a67d` (feat) + `afe98d8` (validation/test fix). Threshold validation (`[-1.0, 1.0]`) added during code review — not in original plan. |
| A | A.2 Cross-chunk de-duplication coverage | ✅ Shipped | commits `dc51442` (tests) + `6020f59` (type-annotation + mixed-endpoint rel coverage). |
| A | A.3 Notebook sanity check | ⏸ Deferred | Manual step against `notebooks/local_falkor.db`; run it when you next open the notebook. Not tracked by tests. |
| B | B.0 Fix `__mentions__` target bug | ✅ Shipped | `__mentions__` now targets entity ids; batch persistence rewrites duplicate semantic keys to a canonical id before relationship persistence. |
| B | B.1 Declare `merge_entity_nodes` primitive | ✅ Shipped | Added `entity_relationship_counts(...)` and `merge_entity_nodes(...)` to `GraphDB`. |
| B | B.2 Implement `merge_entity_nodes` on `FalkorGraphDB` | ✅ Shipped | Falkor merge rewrites touched edges in Python, drops rewrite-created self-loops, deduplicates canonical edges, then deletes duplicates. |
| B | B.3 `pick_master` helper + `MergeReport` | ✅ Shipped | Implemented in `src/grawiki/similarity/deduplication.py` with tests. |
| B | B.4 `GraphRAG.dedupe_entities` facade | ✅ Shipped | Added merge orchestration and dry-run reporting. |
| B | B.5 Notebook end-to-end check | ⏸ Deferred | Manual smoke test against `notebooks/local_falcor.db` still recommended before destructive runs. |

**Current repository state:** Scenario A and Scenario B are implemented. The test suite is green at 62 / 62.

---

## Ground-Truth Findings (from inspecting `notebooks/local_falkor.db`)

These shaped the design and should drive defaults:

1. **Zero exact `semantic_key` collisions exist in the real data.** The LLM-generated semantic keys are unique enough that the "exact collision" path never fires — the value of duplicate detection is entirely in the fuzzy similarity path.
2. **Cross-label duplicates dominate**, e.g. `'State'` as `Type`, `Concept`, and `Class`; `'StateGraph'` as `Workflow`, `Concept`, and `Class`; `'ReAct agents'` vs `'ReAct agent'` vs `'ReAct'` under different labels. The shipped implementation therefore performs duplicate inspection across all entities without label gating, and merged nodes preserve the union of ontology labels.
3. **Latent bug in `save_entities_and_rels`.** `src/grawiki/db/base.py:320` writes `target=node.semantic_key` when building `__mentions__` relationships, but every other relationship uses `target=node.id`. Task B.0 verifies and fixes this before Scenario B runs — otherwise edge redirection during merges will miss `__mentions__`.

---

## File Structure

### Scenario A — pre-persistence resolution ✅ Shipped

- ✅ `src/grawiki/rag/graph_rag.py` — added two constructor args (`resolve_entities_on_ingest`, `entity_resolution_threshold`), threshold validation, private `_resolve_extracted_entities` method, and call sites in `ingest` + `ingest_text`. +113 lines.
- ✅ `tests/rag/test_graph_rag.py` — added `StaticExtractor` and `_CallCountingFinder` helpers plus 5 tests (happy path, default-off, threshold validation, cross-chunk idempotency, rel-endpoint rewriting with mixed-endpoint coverage). +376 lines.
- ✅ `docs/CODEMAP.md` — DI bullet, new public-methods bullet, and Data Flow step 8 for the resolution hook.

No new module, no new classes. The rewrite was ~40 production-code lines.

### Scenario B — post-persistence deduplication ✅ Shipped

- **Modified** `src/grawiki/db/base.py` — `__mentions__` now uses entity ids; added `entity_relationship_counts(...)` and `merge_entity_nodes(...)`.
- **Modified** `src/grawiki/db/falkordb.py` — implemented canonical id rewriting for entity persistence plus `merge_entity_nodes(...)`.
- **Modified** `src/grawiki/db/cypher.py` — added id-based relationship upsert support used during merge recreation.
- **Created** `src/grawiki/similarity/deduplication.py` — `MergeReport`, `pick_master(...)`, and merged-master construction helpers.
- **Modified** `src/grawiki/rag/graph_rag.py` — added `dedupe_entities(...)` and merge-group construction.
- **Created** `tests/similarity/test_deduplication.py` — tests for the deduplication helpers.
- **Modified** `tests/db/test_falkordb.py` — added a real FalkorDBLite merge integration test.
- **Modified** `tests/rag/test_graph_rag.py` — added integration coverage for `dedupe_entities(...)`.
- **Modified** `docs/CODEMAP.md` — documented the new primitive, helper module, and facade method.

---

## Scenario A — Pre-Persistence Entity Resolution ✅ SHIPPED

> Scenario A is complete and merged. The per-task steps below are preserved as a record of what was done and why. If you are resuming work, skip to **Scenario B**.

### Task A.1: Add resolution hook to `GraphRAG` — ✅ DONE (commits `138a67d`, `afe98d8`)

Shipped `GraphRAG.resolve_entities_on_ingest` / `entity_resolution_threshold` constructor args, threshold validation in `__init__`, and the `_resolve_extracted_entities` method. Additional threshold validation guard (`ValueError` when outside `[-1.0, 1.0]`) was added during code review — not in the original plan but worth keeping.

**Files:**
- Modify: `src/grawiki/rag/graph_rag.py`
- Modify: `tests/rag/test_graph_rag.py`

- [ ] **Step 1: Write a failing test for the happy path**

In `tests/rag/test_graph_rag.py`, following the conventions of the existing tests (inspect the file first for the fake-DB / fake-extractor pattern), add:

```python
@pytest.mark.asyncio
async def test_ingest_resolves_to_persisted_entity_when_similar(
    fake_db, fake_extractor, fake_embedding
):
    # Preload the fake DB with one persisted entity.
    persisted = Node(
        id="persisted-1",
        label="Concept",
        semantic_key="concept_react",
        name="ReAct",
        embedding=[0.1, 0.2, 0.3],
    )
    fake_db.preload_entities([persisted])

    # fake_extractor yields one entity with a near-duplicate name and a close embedding.
    fake_extractor.set_output(
        extracted_nodes=[
            Node(
                id="extracted-1",
                label="Concept",
                semantic_key="concept_react-agents",
                name="ReAct agents",
                embedding=[0.1, 0.2, 0.31],  # cosine > 0.92 vs persisted
            ),
        ]
    )

    rag = GraphRAG(
        model="stub-model",
        embedding_model="stub-embedder",
        db=fake_db,
        embedding=fake_embedding,
        kg_extractor=fake_extractor,
        resolve_entities_on_ingest=True,
        entity_resolution_threshold=0.9,
    )

    await rag.ingest_text("ReAct agents are a kind of ReAct.", title="t")

    # The extraction's node should have been swapped for the persisted one
    # before persistence, so the only entity upserted was persisted-1.
    upserted_entity_ids = {
        n.id for n in fake_db.upserted_nodes if n.label not in {"__document__", "__chunk__"}
    }
    assert upserted_entity_ids == {"persisted-1"}
```

Also add a second test verifying the flag defaults to `False` and that without it enabled the extracted node is persisted as-is.

- [ ] **Step 2: Run to see both fail**

Run: `uv run pytest tests/rag/test_graph_rag.py -k resolves_to_persisted -v`
Expected: FAIL.

- [ ] **Step 3: Implement the hook in `GraphRAG`**

In `src/grawiki/rag/graph_rag.py`:

a. Add two keyword-only constructor args, documented in the NumPy docstring:
```python
resolve_entities_on_ingest: bool = False,
entity_resolution_threshold: float = 0.92,
```

b. Store them on `self`.

c. Add the private method:

```python
async def _resolve_extracted_entities(
    self, chunk_graphs: dict[str, KnowledgeGraph]
) -> dict[str, KnowledgeGraph]:
    """Replace extracted entities with persisted duplicates when similar.

    Parameters
    ----------
    chunk_graphs : dict[str, KnowledgeGraph]
        Freshly-extracted chunk graphs. Not mutated.

    Returns
    -------
    dict[str, KnowledgeGraph]
        New dict with matched nodes replaced by the persisted node and all
        relationships rewritten accordingly. When no matches are found the
        input is returned unchanged.

    Notes
    -----
    Uses ``self._entity_similarity`` so the same similarity strategy that
    powers the duplicate-candidate helpers is applied during ingestion.
    ``same_label_only`` is ``False`` because real data shows duplicates
    routinely cross ontology labels (``'State'`` appears as both
    ``Type`` and ``Class``, for example).
    """

    # Collect unique extracted entities across all chunks.
    unique: dict[str, Node] = {}
    for graph in chunk_graphs.values():
        for node in graph.nodes:
            unique.setdefault(node.id, node)

    # One similarity search per unique extracted entity.
    resolved: dict[str, Node] = {}
    for ext_id, ext_node in unique.items():
        hits = await self._entity_similarity.search(
            ext_node,
            limit=1,
            threshold=self.entity_resolution_threshold,
            same_label_only=False,
        )
        if hits:
            resolved[ext_id] = hits[0].node

    if not resolved:
        return chunk_graphs

    rewritten: dict[str, KnowledgeGraph] = {}
    for cid, graph in chunk_graphs.items():
        new_nodes = [resolved.get(n.id, n) for n in graph.nodes]
        new_rels = [
            Relationship(
                id=r.id,
                source=resolved[r.source].id if r.source in resolved else r.source,
                target=resolved[r.target].id if r.target in resolved else r.target,
                label=r.label,
                properties=dict(r.properties),
            )
            for r in graph.relationships
        ]
        rewritten[cid] = KnowledgeGraph(nodes=new_nodes, relationships=new_rels)
    return rewritten
```

d. In both `ingest` and `ingest_text`, after `chunk_graphs = await self.extract_kg_per_chunk(chunks)` and before `persist_entities_and_relationships(...)`:

```python
if self.resolve_entities_on_ingest:
    chunk_graphs = await self._resolve_extracted_entities(chunk_graphs)
```

e. Add the needed import at the top of the module: `Relationship` (if not already imported).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/rag/test_graph_rag.py -v`
Expected: PASS.

- [ ] **Step 5: Update CODEMAP**

In `docs/CODEMAP.md`, under `src/grawiki/rag/graph_rag.py`:
- Add `resolve_entities_on_ingest` and `entity_resolution_threshold` to the "Supports dependency injection through..." bullet.
- Add a bullet under `Important public methods` clarifying that when `resolve_entities_on_ingest=True`, extracted entities are matched against persisted ones before persistence and existing ids are reused.

- [ ] **Step 6: Run full checks and commit**

```bash
uv run pre-commit run --all-files
uv run pytest
git add src/grawiki/rag/graph_rag.py tests/rag/test_graph_rag.py docs/CODEMAP.md
git commit -m "feat(rag): optional entity resolution during ingest"
```

---

### Task A.2: Cross-chunk de-duplication coverage — ✅ DONE (commits `dc51442`, `6020f59`)

Shipped two tests using a test-local `_CallCountingFinder` proxy: one asserting exactly-one similarity search per unique extracted id across multiple chunks, the other asserting rewritten relationship endpoints (including a mixed-endpoint case where only `source` is resolved) — the follow-up commit tightened the proxy's return-type annotation and added the mixed-endpoint coverage after code review.

**Files:**
- Modify: `tests/rag/test_graph_rag.py`

- [ ] **Step 1: Add a test where two chunks reference the same extracted entity**

Verify that `_resolve_extracted_entities` only calls `finder.search` once for it (use a counter on the fake similarity finder or the existing `fake_db.upserted_nodes` shape to observe idempotence).

- [ ] **Step 2: Add a test where one chunk has a relationship touching a resolved node**

Verify that the rewritten relationship's source/target use the persisted id, not the extracted id. This catches regressions where only `graph.nodes` gets rewritten but `graph.relationships` is forgotten.

- [ ] **Step 3: Run tests and commit**

```bash
uv run pytest tests/rag/test_graph_rag.py -v
git add tests/rag/test_graph_rag.py
git commit -m "test(rag): cover cross-chunk entity resolution edge cases"
```

---

### Task A.3: Notebook sanity check — ⏸ DEFERRED

Manual end-to-end verification step that has not been run yet. Do this the next time you open `notebooks/debug.ipynb` against `local_falkor.db`. Not blocking for any other work.

**Files:**
- Modify: `notebooks/debug.ipynb` (or a new cell)

- [ ] Add a cell that ingests a snippet containing a known near-duplicate of something already in `local_falkor.db` (e.g. "ReAct agents orchestrate tasks.") with `resolve_entities_on_ingest=True, entity_resolution_threshold=0.9, same_label_only=False`. After ingestion, query `db.list_entities()` and confirm no new node was created for "ReAct agents" — the existing persisted id should have been reused in the `__mentions__` edge. Document the observed behavior in a markdown cell.

No commit unless the notebook is tracked.

---

## Scenario B — Post-Persistence Entity Deduplication ⏳ NEXT PHASE

> **Resume here.** Start with Task B.0 — it is a prerequisite fix that unblocks every later task in this scenario.
>
> Scenario B is destructive (it deletes nodes). Back up `notebooks/local_falkor.db` before the first real run and prefer `dry_run=True` until a `MergeReport` sample looks correct.

### Task B.0: Fix the `__mentions__` target bug ⏳ START HERE

**Files:**
- Modify: `src/grawiki/db/base.py`
- Modify: `tests/db/test_falkordb.py` (or wherever `save_entities_and_rels` is covered)

- [ ] **Step 1: Verify the bug**

Run a quick query against the existing `notebooks/local_falkor.db`:

```python
rows = db.ro_query(
    "MATCH (c:__chunk__)-[:__mentions__]->(x) RETURN x.id LIMIT 5"
)
# Confirm x.id looks like a UUID, not a semantic_key string.
```

Then read the current code at `src/grawiki/db/base.py:320` — it uses `target=node.semantic_key`. Check `src/grawiki/db/falkordb.py` `upsert_relationships` to understand why, despite that, stored targets end up as UUIDs (the adapter likely re-resolves through a different matching path, or the data is pre-dating the current line). Record the root cause in the commit message.

- [ ] **Step 2: Fix `save_entities_and_rels`**

Change `target=node.semantic_key` to `target=node.id` at `src/grawiki/db/base.py:320`. Semantic keys are a property, not an identifier.

- [ ] **Step 3: Add a regression test**

In `tests/db/test_falkordb.py` (or the matching integration test file), add a test that ingests a small graph and asserts that all `__mentions__` edges have their `target` equal to a node id present in the graph.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest
git add src/grawiki/db/base.py tests/db/test_falkordb.py
git commit -m "fix(db): __mentions__ target must be node id, not semantic_key"
```

---

### Task B.1: Add the `merge_entity_nodes` DB primitive (abstract)

**Files:**
- Modify: `src/grawiki/db/base.py`

- [ ] **Step 1: Declare the abstract method**

In `GraphDB`, near `upsert_relationships`:

```python
@abstractmethod
async def merge_entity_nodes(
    self,
    *,
    master_id: str,
    duplicate_ids: Sequence[str],
) -> None:
    """Redirect all relationships from duplicates to master, then delete duplicates.

    Parameters
    ----------
    master_id : str
        Identifier of the node that absorbs all incoming and outgoing edges.
    duplicate_ids : Sequence[str]
        Identifiers of nodes to merge into the master and then delete.
        Must not contain ``master_id``.

    Raises
    ------
    ValueError
        If ``master_id`` appears in ``duplicate_ids``.

    Notes
    -----
    Relationship types and directions are preserved. Self-loops created by
    redirection (e.g. original ``dup -knows-> master`` would become
    ``master -knows-> master``) are dropped. Node properties on duplicates
    are not merged onto the master — that is a policy decision and lives
    in the orchestration layer.
    """
```

- [ ] **Step 2: Update CODEMAP** and commit.

```bash
git add src/grawiki/db/base.py docs/CODEMAP.md
git commit -m "feat(db): declare merge_entity_nodes primitive"
```

---

### Task B.2: Implement `merge_entity_nodes` in `FalkorGraphDB`

**Files:**
- Modify: `src/grawiki/db/falkordb.py`
- Modify: `src/grawiki/db/cypher.py`
- Modify: `tests/db/test_falkordb.py`

- [ ] **Step 1: Failing test**

Create a test that:
1. Sets up `FalkorGraphDB`, creates entity nodes `A`, `B`, `C` with a chunk `K`.
2. Creates edges: `A -[knows]-> B`, `C -[knows]-> B`, `B -[likes]-> C`, `K -[__mentions__]-> B`.
3. Calls `await db.merge_entity_nodes(master_id="A", duplicate_ids=["B"])`.
4. Asserts:
   - Node `B` no longer exists.
   - `C -[knows]-> A` exists.
   - `A -[likes]-> C` exists.
   - `K -[__mentions__]-> A` exists.
   - No `A -[knows]-> A` self-loop was created.

- [ ] **Step 2: Run to see it fail.**

Run: `uv run pytest tests/db/test_falkordb.py -k merge_entity_nodes -v`

- [ ] **Step 3: Add Cypher builders in `src/grawiki/db/cypher.py`**

Because FalkorDB Cypher cannot parameterize relationship types, the adapter must first enumerate the distinct relationship types touching the duplicates, then issue per-type queries. Add three small builders:

```python
def build_distinct_rel_types_for_duplicates_cypher() -> str: ...
def build_redirect_outgoing_cypher(rel_type: str) -> str: ...
def build_redirect_incoming_cypher(rel_type: str) -> str: ...
def build_delete_nodes_by_id_cypher() -> str: ...
```

Each returns a parameterized Cypher string using `$master_id` and `$dup_ids`. Make sure `rel_type` is sanitized through the existing label/relationship sanitizer.

- [ ] **Step 4: Implement the method in `FalkorGraphDB`**

```python
async def merge_entity_nodes(
    self,
    *,
    master_id: str,
    duplicate_ids: Sequence[str],
) -> None:
    dup_list = list(duplicate_ids)
    if master_id in dup_list:
        raise ValueError("master_id must not be in duplicate_ids")
    if not dup_list:
        return

    params = {"master_id": master_id, "dup_ids": dup_list}

    # 1. Enumerate distinct relationship types touching any duplicate.
    result = self.ro_query(
        build_distinct_rel_types_for_duplicates_cypher(), params=params
    )
    rel_types = [row[0] for row in result.result_set]

    # 2. Redirect edges per type, both directions.
    for rel_type in rel_types:
        self.query(build_redirect_outgoing_cypher(rel_type), params=params)
        self.query(build_redirect_incoming_cypher(rel_type), params=params)

    # 3. Delete duplicate nodes.
    self.query(build_delete_nodes_by_id_cypher(), params=params)
```

(Wrap sync FalkorDB calls so this method is `async` consistent with the rest of the interface — follow the same pattern the adapter already uses for `upsert_nodes` etc.)

- [ ] **Step 5: Run test and commit.**

```bash
uv run pytest tests/db/test_falkordb.py -v
git add src/grawiki/db/falkordb.py src/grawiki/db/cypher.py tests/db/test_falkordb.py
git commit -m "feat(db): implement merge_entity_nodes on FalkorGraphDB"
```

---

### Task B.3: Add `pick_master` helper and `MergeReport`

**Files:**
- Create: `src/grawiki/similarity/deduplication.py`
- Create: `tests/similarity/test_deduplication.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for pick_master and MergeReport."""

from __future__ import annotations

from grawiki.graph.models import Node
from grawiki.similarity.deduplication import MergeReport, pick_master


def _node(id_: str, name: str) -> Node:
    return Node(id=id_, label="Person", semantic_key=f"p_{name.lower()}", name=name)


def test_pick_master_prefers_longest_name():
    assert pick_master([_node("1", "A"), _node("2", "Alan Turing"), _node("3", "AT")]).id == "2"


def test_pick_master_tiebreaks_by_smallest_id():
    a = _node("2", "Alan")
    b = _node("1", "Alan")
    assert pick_master([a, b]).id == "1"


def test_merge_report_fields():
    r = MergeReport(master_id="m", duplicate_ids=("a", "b"), source="similarity")
    assert r.master_id == "m"
    assert r.duplicate_ids == ("a", "b")
    assert r.source == "similarity"
```

- [ ] **Step 2: Run to see it fail.**

- [ ] **Step 3: Implement the module**

```python
"""Post-persistence entity deduplication helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from grawiki.graph.models import Node


@dataclass
class MergeReport:
    """Record of one merge decision.

    Parameters
    ----------
    master_id : str
        Identifier of the node that absorbed the group.
    duplicate_ids : tuple[str, ...]
        Identifiers of the nodes that were merged and deleted.
    source : str
        Where the group came from: ``"collision"`` or ``"similarity"``.
    """

    master_id: str
    duplicate_ids: tuple[str, ...]
    source: str


def pick_master(nodes: Sequence[Node]) -> Node:
    """Return the best-fit master from a duplicate group.

    Order: longest ``name``, then lexicographically smallest ``id``.

    Parameters
    ----------
    nodes : Sequence[Node]
        Candidate nodes. Must be non-empty.

    Returns
    -------
    Node
        The chosen master.
    """

    return sorted(nodes, key=lambda n: (-len(n.name), n.id))[0]
```

- [ ] **Step 4: Run tests and commit.**

```bash
uv run pytest tests/similarity/test_deduplication.py -v
git add src/grawiki/similarity/deduplication.py tests/similarity/test_deduplication.py
git commit -m "feat(similarity): add pick_master helper and MergeReport"
```

---

### Task B.4: Add `dedupe_entities` on `GraphRAG`

**Files:**
- Modify: `src/grawiki/rag/graph_rag.py`
- Modify: `tests/rag/test_graph_rag.py`

- [ ] **Step 1: Failing integration test**

Preload the fake DB with three persisted entities that would form a duplicate group. Stub the similarity finder to return them as candidates above the threshold. Call `await rag.dedupe_entities(min_merge_score=0.9)` and assert:
- `fake_db.merges == [("master-id", ("dup1-id", "dup2-id"))]`
- the returned reports match.

- [ ] **Step 2: Run to see it fail.**

- [ ] **Step 3: Implement**

In `src/grawiki/rag/graph_rag.py`:

```python
from grawiki.similarity.deduplication import MergeReport, pick_master

async def dedupe_entities(
    self,
    *,
    limit: int = 10,
    threshold: float | None = None,
    same_label_only: bool = False,
    min_merge_score: float = 0.95,
    dry_run: bool = False,
) -> list[MergeReport]:
    """Find duplicate candidates and merge them.

    Parameters
    ----------
    limit, threshold, same_label_only
        Forwarded to
        :meth:`EntitySimilarityFinder.find_duplicate_candidates`.
        ``same_label_only`` defaults to ``False`` because real extractions
        produce cross-label duplicates (``'State'`` as both ``Class`` and
        ``Concept``).
    min_merge_score : float, optional
        Minimum matcher score required to include a candidate in a merge
        group. Default 0.95 is conservative; raise for stricter matching.
    dry_run : bool, optional
        If True, return reports without calling the DB.

    Returns
    -------
    list[MergeReport]
        One report per merged group.
    """

    candidates = await self._entity_similarity.find_duplicate_candidates(
        limit=limit,
        threshold=threshold,
        same_label_only=same_label_only,
    )

    reports: list[MergeReport] = []
    seen: set[str] = set()

    # Stage 1: exact semantic-key collision groups.
    for skg in candidates.semantic_key_collision_candidates:
        group = _flatten_collision_group(skg, min_score=min_merge_score)
        group = [n for n in group if n.id not in seen]
        if len(group) < 2:
            continue
        reports.append(_make_report(group, source="collision"))
        seen.update(n.id for n in group)

    # Stage 2: broader similarity-based groups.
    for res in candidates.similarity_candidates:
        group = [res.source] + [h.node for h in res.hits if h.score >= min_merge_score]
        group = [n for n in group if n.id not in seen]
        if len(group) < 2:
            continue
        reports.append(_make_report(group, source="similarity"))
        seen.update(n.id for n in group)

    if not dry_run:
        for r in reports:
            await self._db.merge_entity_nodes(
                master_id=r.master_id, duplicate_ids=list(r.duplicate_ids)
            )
    return reports


def _make_report(group: list[Node], *, source: str) -> MergeReport:
    master = pick_master(group)
    dup_ids = tuple(n.id for n in group if n.id != master.id)
    return MergeReport(master_id=master.id, duplicate_ids=dup_ids, source=source)


def _flatten_collision_group(
    skg: SemanticKeyCollisionCandidates, *, min_score: float
) -> list[Node]:
    seen: dict[str, Node] = {}
    for res in skg.results:
        seen.setdefault(res.source.id, res.source)
        for hit in res.hits:
            if hit.score >= min_score:
                seen.setdefault(hit.node.id, hit.node)
    return list(seen.values())
```

The `_make_report` and `_flatten_collision_group` helpers can live at module level at the bottom of `graph_rag.py` — they are specific to this orchestration and not worth a separate module.

- [ ] **Step 4: Run tests and commit.**

```bash
uv run pre-commit run --all-files
uv run pytest
git add src/grawiki/rag/graph_rag.py tests/rag/test_graph_rag.py docs/CODEMAP.md
git commit -m "feat(rag): add dedupe_entities facade"
```

---

### Task B.5: Notebook end-to-end check

- [ ] In `notebooks/debug.ipynb`, add cells that:
  1. `candidates = await rag.find_entity_duplicate_candidates(threshold=0.85, same_label_only=False)` — print the sizes.
  2. `reports = await rag.dedupe_entities(min_merge_score=0.9, same_label_only=False, dry_run=True)` — inspect before merging for real.
  3. Promote to `dry_run=False` once the reports look right.
  4. Re-run (1) to confirm the numbers dropped.

Document what you saw in a markdown cell. This is the most useful smoke test because the real embeddings are already in the DB.

---

## Risks and Rollback Notes

- **Scenario A is live.** The flag defaults to `False` so the new behavior is dormant for existing callers. Threshold validation (`[-1.0, 1.0]`) rejects out-of-range cosine values at construction. Log-level auditing of each resolution was discussed but is not yet implemented — worth revisiting if you start using this in production.
- **Scenario A can silently merge** if the threshold is too low or the embedding model drifts. Keep it opt-in (`resolve_entities_on_ingest=False` by default) and log at `INFO` each resolution so users can audit what was merged.
- **Scenario B is destructive.** `merge_entity_nodes` deletes nodes. Back up the FalkorDB directory before the first real run. `dry_run=True` plus a manual inspection of the returned `MergeReport`s is the recommended workflow.
- **Property merging is out of scope.** When duplicates have different `properties`, only the master's properties survive. If a duplicate has richer properties than the master, this loses data. Add a per-group property-merge step later if it becomes a real problem.
- **Parallel duplicate edges.** If `A -knows-> X` and `dup -knows-> X` both exist, after merge there will be two `A -knows-> X` edges. FalkorDB does not deduplicate relationships automatically. Revisit only if downstream queries care.
- **Threshold tuning.** With only 113 entities to work with, pick defaults that err toward false-negatives (under-merge) rather than false-positives (over-merge). `0.92` for Scenario A and `0.95` for Scenario B match that stance.
