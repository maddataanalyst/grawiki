# Knowledge-Graph Extraction Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reduce wall-clock time for `GraphRAG.ingest()` on typical documents (currently ~60 seconds on gpt-5-mini via the OpenAI API) by eliminating redundant embedding round-trips, raising concurrency where it safely improves throughput, and giving the developer a repeatable benchmark harness to justify every tuning decision. Provider-side latency is treated as a fixed cost — every win must come from client-side work.

**Architecture:** The extractor currently does both the LLM call and the embedding call inside `KnowledgeGraphExtractor.extract(chunk)`, which means the embedding API is hit once per chunk. Most entity names repeat across chunks in real documents, so this work is largely redundant. The plan moves embedding out of the extractor: `KnowledgeGraphExtractor` becomes LLM-only, and `GraphRAG.extract_kg_per_chunk` follows extraction with a single batched `embed_documents` call covering all unique entity names across all chunks. The concurrency semaphore in `GraphRAG.extract_kg_per_chunk` is retained; its default is tuned after benchmarking.

**Tech Stack:** Python 3.12, `asyncio`, `pydantic_ai.Agent` for LLM structured output, `pydantic_ai.Embedder` for embeddings, `pytest` for tests, `uv` for dependency management. No new runtime dependencies.

---

## Cost Model (why the current code is slow)

Per chunk, `KnowledgeGraphExtractor.extract(chunk)`:

1. **LLM call** — `await self.agent.run(chunk.content)`. Dominant cost: usually 3–15 s on gpt-5-mini depending on chunk length and number of triplets produced.
2. **Missing-node repair** — CPU only, microseconds.
3. **Entity embedding call** — `await self.embedding.embed_documents(node_names)`. ~100–500 ms.

For `N` chunks with concurrency `W`:

- Wall-clock ≈ `ceil(N / W) × (T_llm_worst_in_batch + T_embed)`.
- For `N = 12`, `W = 4`, `T_llm = 8 s`, `T_embed = 0.4 s`: ≈ 3 × 8.4 ≈ 25 s under perfect conditions, but reality lands at 45–60 s because some chunks take 15+ seconds, embedding adds up, and connection setup per request costs ~100 ms.

Where the client-side slack is:

| Cost | Current behavior | Savings opportunity |
| --- | --- | --- |
| Per-chunk embedding API call | 1 call/chunk, names duplicated across chunks | Batch across chunks; dedup by name → 1 call total |
| `max_workers = 4` | Hard-coded | Tune per-account rate limits (likely 8–16) |
| Per-chunk HTTP overhead | Each call is its own request | Batch amortizes this |
| Persistent cache | None | Future: cache entity-name embeddings across runs |

The structural win (batched cross-chunk embedding) is the biggest lever. Concurrency tuning is the easiest win. Caching comes last.

---

## Progress

| Phase | Task | Status | Notes |
| --- | --- | --- | --- |
| 1 | 1.1 Add benchmark harness | ⏳ Not started | Prereq for everything else — measure before tuning |
| 1 | 1.2 Baseline measurement | ⏳ Not started | Record timings on a fixed corpus |
| 2 | 2.1 Bump & expose `max_workers` default | ⏳ Not started | One-line default + benchmark sweep |
| 3 | 3.1 Make extractor embedding-optional | ⏳ Not started | Backwards-compatible deprecation |
| 3 | 3.2 Batch embedding across chunks | ⏳ Not started | Main structural win |
| 3 | 3.3 Update notebook caller | ⏳ Not started | Remove the `embedding=` arg from `KnowledgeGraphExtractor` in `notebooks/debug.ipynb` |
| 4 | 4.1 Persistent embedding cache (optional) | 🟡 Optional | Only if re-ingestion is common |
| 4 | 4.2 HTTP client reuse audit (optional) | 🟡 Optional | Low confidence this is an issue |

**Start with Phase 1.** Do not skip the benchmark — without numbers, every later decision is guesswork.

---

## File Structure

- **Create** `scripts/benchmark_ingest.py` — small async script that ingests a fixed document with configurable `max_workers`, prints total time and per-stage breakdown, writes results to `benchmarks/ingest_results.csv` for comparison across runs.
- **Create** `benchmarks/` directory with a tracked results CSV (git-tracked) and a fixed `benchmarks/input.txt` corpus (one representative doc, committed) that every run consumes.
- **Modify** `src/grawiki/graph/extraction.py` — make `embedding` optional on `KnowledgeGraphExtractor.__init__`; when `None`, `extract()` returns nodes with empty embeddings. Emit `DeprecationWarning` when `embedding` is passed (Phase 3.1).
- **Modify** `src/grawiki/rag/graph_rag.py` — after `extract_kg_per_chunk` completes, run a new private method `_embed_entity_nodes_across_chunks(chunk_graphs)` that:
  1. Collects every unique `(name, label)` pair across all chunks where the node's `embedding` is empty.
  2. Issues a single `embed_documents` call.
  3. Writes the resulting vectors back onto the matching nodes in all chunk graphs.
- **Modify** `notebooks/debug.ipynb` — drop the `embedding=embedding` argument from the `KnowledgeGraphExtractor` instantiation cell.
- **Modify** `tests/graph/test_extraction.py` — update tests to reflect the new "embeddings off by default" behavior; add a test that verifies nodes come out with empty embeddings when `embedding=None`.
- **Modify** `tests/rag/test_graph_rag.py` — add a test verifying the cross-chunk batched embedding step (one `embed_documents` call covering all unique names across all chunks).
- **Modify** `docs/CODEMAP.md` — reflect the responsibility shift (extractor becomes pure LLM; orchestrator owns embedding attachment).

---

## Phase 1: Measurement

### Task 1.1: Add benchmark harness

**Files:**
- Create: `scripts/benchmark_ingest.py`
- Create: `benchmarks/input.txt`
- Create: `benchmarks/ingest_results.csv`

- [ ] **Step 1: Pick a representative document**

Copy one existing file from `notebooks/experimental_data/` to `benchmarks/input.txt`. Pick something typical — not the shortest, not the longest. Commit it so benchmarks are reproducible across machines.

- [ ] **Step 2: Write `scripts/benchmark_ingest.py`**

```python
"""Timing harness for GraphRAG.ingest.

Usage
-----
uv run python scripts/benchmark_ingest.py --max-workers 4 --runs 3 --label baseline
uv run python scripts/benchmark_ingest.py --max-workers 8 --runs 3 --label workers-8

Each run appends one row to benchmarks/ingest_results.csv with columns:
timestamp, label, max_workers, runs, mean_total_s, mean_llm_s, mean_embed_s, notes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import time
from pathlib import Path
from statistics import mean

from pydantic_ai import Embedder

from grawiki.db.falkordb import FalkorGraphDB
from grawiki.rag.graph_rag import GraphRAG


async def run_once(max_workers: int) -> dict[str, float]:
    # Use a disposable on-disk FalkorDB per run to avoid cross-run state.
    tmp_db = Path(f"/tmp/bench_falkor_{os.getpid()}_{time.time_ns()}.db")
    db = FalkorGraphDB(str(tmp_db), graph_name="bench")
    embedding = Embedder("openai:text-embedding-3-small")
    rag = GraphRAG(
        model="openai:gpt-5-mini",
        embedding_model="text-embedding-3-small",
        db=db,
        max_workers=max_workers,
        embedding=embedding,
    )
    start = time.monotonic()
    await rag.ingest(Path("benchmarks/input.txt"))
    total = time.monotonic() - start
    db.close()
    # Clean up the temp DB so reruns don't grow disk.
    for p in tmp_db.parent.glob(tmp_db.name + "*"):
        p.unlink(missing_ok=True)
    return {"total_s": total}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    results = []
    for i in range(args.runs):
        print(f"  run {i + 1}/{args.runs}...")
        r = await run_once(args.max_workers)
        print(f"    total: {r['total_s']:.2f} s")
        results.append(r)

    mean_total = mean(r["total_s"] for r in results)
    csv_path = Path("benchmarks/ingest_results.csv")
    csv_path.parent.mkdir(exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "label", "max_workers", "runs", "mean_total_s", "notes"])
        w.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            args.label,
            args.max_workers,
            args.runs,
            f"{mean_total:.3f}",
            args.notes,
        ])
    print(f"label={args.label} max_workers={args.max_workers} runs={args.runs} mean_total={mean_total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

Note: the script only measures total time for now. Per-stage breakdown (LLM vs embed) comes with Phase 3 once we have clean hooks. Adding instrumentation earlier would require touching production code just for measurement — premature.

- [ ] **Step 3: Run 3 times to establish a baseline**

```bash
uv run python scripts/benchmark_ingest.py --max-workers 4 --runs 3 --label baseline-mw4
```

Record the mean. Commit the harness + input:

```bash
git add scripts/benchmark_ingest.py benchmarks/input.txt benchmarks/ingest_results.csv
git commit -m "chore(bench): add ingest benchmark harness and baseline input"
```

### Task 1.2: Baseline measurement

- [ ] **Step 1: Run the benchmark with the current defaults**

```bash
uv run python scripts/benchmark_ingest.py --max-workers 4 --runs 3 --label baseline-mw4 --notes "main at 6020f59"
```

Record the mean total time. This is your **baseline**. Every subsequent change must be measured against it.

- [ ] **Step 2: Commit the results row**

```bash
git add benchmarks/ingest_results.csv
git commit -m "chore(bench): record baseline ingest timing"
```

No code changes in this task; the goal is to lock in a number.

---

## Phase 2: Concurrency tuning (the one-line win)

### Task 2.1: Bump and benchmark `max_workers`

**Files:**
- Modify: `src/grawiki/rag/graph_rag.py` (default value only)

- [ ] **Step 1: Benchmark at several worker counts**

Before changing anything, run:

```bash
uv run python scripts/benchmark_ingest.py --max-workers 8  --runs 3 --label sweep-mw8
uv run python scripts/benchmark_ingest.py --max-workers 12 --runs 3 --label sweep-mw12
uv run python scripts/benchmark_ingest.py --max-workers 16 --runs 3 --label sweep-mw16
```

If your OpenAI tier has tight rate limits, 12+ may actually *slow down* due to retries. Watch the console output for rate-limit errors.

- [ ] **Step 2: Pick the best number that isn't flaky**

"Best" means: fastest mean, no rate-limit errors, stable across runs. If 8 and 12 are within 10% of each other, prefer 8 (more headroom for other API usage).

- [ ] **Step 3: Update the default**

In `src/grawiki/rag/graph_rag.py`, change `max_workers: int = 4` to the chosen value. Update the class docstring and `__init__` docstring to reflect the new default.

- [ ] **Step 4: Run the full test suite and commit**

```bash
uv run pytest
uv run pre-commit run --all-files
git add src/grawiki/rag/graph_rag.py benchmarks/ingest_results.csv
git commit -m "perf(rag): raise default max_workers to N based on benchmark sweep"
```

(Replace `N` with the actual chosen value.)

- [ ] **Step 5: Record the "after" row**

```bash
uv run python scripts/benchmark_ingest.py --max-workers N --runs 3 --label after-phase2
```

Commit the updated CSV.

---

## Phase 3: Structural change — batch embedding across chunks

This is where the real win lives. Changes to the extractor interface are deliberately kept narrow: `KnowledgeGraphExtractor` becomes pure LLM extraction; `GraphRAG` owns the embedding attachment step.

### Task 3.1: Make extractor embedding-optional (backwards-compatible)

**Files:**
- Modify: `src/grawiki/graph/extraction.py`
- Modify: `tests/graph/test_extraction.py`

- [ ] **Step 1: Failing test — extractor returns nodes with empty embeddings when `embedding=None`**

In `tests/graph/test_extraction.py`, add a test that constructs `KnowledgeGraphExtractor(model="...", embedding=None)` and calls `extract()` on a stubbed agent; asserts every returned node has `embedding == []`.

Because the test will need to stub out the LLM agent, either reuse any existing stub pattern in that file or inject a pre-built `Agent` via the `*args`/`**kwargs` the constructor already forwards. Inspect the existing tests before writing; they will tell you the idiomatic stubbing style.

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest tests/graph/test_extraction.py -v
```

- [ ] **Step 3: Implement**

In `src/grawiki/graph/extraction.py`:

a. Make the `embedding` parameter `Embedding | None = None`, documented in the NumPy docstring.

b. In `_build_knowledge_graph`, guard the embedding step:

```python
if node_names and self.embedding is not None:
    embedding_result = await self.embedding.embed_documents(node_names)
    for node_name, embedding in zip(node_names, embedding_result.embeddings, strict=True):
        nodes_by_name[node_name].embedding = list(embedding)
```

c. When `embedding` is passed, emit a one-time `DeprecationWarning` from `__init__`:

```python
import warnings

if embedding is not None:
    warnings.warn(
        "Passing `embedding` to KnowledgeGraphExtractor is deprecated. "
        "GraphRAG now owns batched entity embedding; the extractor will "
        "become pure LLM extraction in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
```

This preserves backward compatibility for one release while signalling the intent.

- [ ] **Step 4: Tests pass**

```bash
uv run pytest tests/graph/test_extraction.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/grawiki/graph/extraction.py tests/graph/test_extraction.py
git commit -m "refactor(graph): make KnowledgeGraphExtractor embedding optional"
```

### Task 3.2: Batch embedding across chunks in `GraphRAG`

**Files:**
- Modify: `src/grawiki/rag/graph_rag.py`
- Modify: `tests/rag/test_graph_rag.py`

- [ ] **Step 1: Failing test**

In `tests/rag/test_graph_rag.py`, add a test `test_graph_rag_batches_entity_embeddings_across_chunks`:

- Uses `FakeGraphDB` + a `FakeEmbedder` wrapper that counts `embed_documents` calls.
- Uses a custom extractor that returns two different chunk graphs, each with two entities (`Alice`, `Bob` in chunk-a; `Bob`, `Charlie` in chunk-b — note `Bob` is shared).
- The extractor returns nodes with `embedding=[]` (new behavior).
- After `rag.extract_kg_per_chunk(chunks)` + the new batch-embedding step, asserts:
  - `FakeEmbedder.embed_documents` was called exactly **once** across the whole pipeline (ignoring any doc/chunk-level embedding calls; the test should focus on the entity-names call, which is the new batched one).
  - The embedded name list was deduped: exactly 3 unique names (`{"Alice", "Bob", "Charlie"}`), not 4.
  - Every node in both chunk graphs ends up with a non-empty `embedding` attribute.

The shape is similar to the existing `_CallCountingFinder` pattern — a thin wrapper around `FakeEmbedder` that records each call and forwards.

- [ ] **Step 2: Run to see it fail**

- [ ] **Step 3: Implement the batched embedding step**

In `src/grawiki/rag/graph_rag.py`, add a private method near `_resolve_extracted_entities`:

```python
async def _embed_entity_nodes_across_chunks(
    self, chunk_graphs: dict[str, KnowledgeGraph]
) -> None:
    """Attach embeddings to every entity node across chunks in one API call.

    Mutates ``chunk_graphs`` in place. Nodes that already have a non-empty
    embedding (e.g. from a test stub or an injected extractor that chose to
    embed itself) are left untouched.

    Parameters
    ----------
    chunk_graphs : dict[str, KnowledgeGraph]
        Extracted chunk graphs produced by the extractor.

    Notes
    -----
    Names are deduplicated before the embedding call; one vector per unique
    ``name`` is written to every node that shares that name. This is
    correct because the extractor uses ``name`` as its within-graph
    identifier, and the embedding of the name string is independent of the
    chunk it was extracted from.
    """

    # Collect unique names that still need embedding.
    names_to_embed: list[str] = []
    seen: set[str] = set()
    for graph in chunk_graphs.values():
        for node in graph.nodes:
            if node.embedding:
                continue
            if node.name in seen:
                continue
            seen.add(node.name)
            names_to_embed.append(node.name)

    if not names_to_embed:
        return

    result = await self._embedding.embed_documents(names_to_embed)
    by_name = {
        name: list(vec)
        for name, vec in zip(names_to_embed, result.embeddings, strict=True)
    }

    for graph in chunk_graphs.values():
        for node in graph.nodes:
            if node.embedding:
                continue
            vec = by_name.get(node.name)
            if vec is not None:
                node.embedding = vec
```

Call it in `extract_kg_per_chunk`, after `asyncio.gather` returns and before the method returns:

```python
chunk_graphs = dict(results)
await self._embed_entity_nodes_across_chunks(chunk_graphs)
return chunk_graphs
```

Important: `_resolve_extracted_entities` (from Scenario A) runs *after* this step in `ingest` / `ingest_text`, so the entity resolution hook still compares a freshly-embedded node against the persisted-entity pool. Good.

- [ ] **Step 4: Test passes and full suite is green**

```bash
uv run pytest
uv run pre-commit run --all-files
```

- [ ] **Step 5: Benchmark**

Run the benchmark again:

```bash
uv run python scripts/benchmark_ingest.py --max-workers N --runs 3 --label after-phase3
```

Expect a measurable drop (typical savings: 2–5 s on a 12-chunk doc depending on name overlap).

- [ ] **Step 6: Commit**

```bash
git add src/grawiki/rag/graph_rag.py tests/rag/test_graph_rag.py benchmarks/ingest_results.csv
git commit -m "perf(rag): batch entity embeddings across chunks"
```

### Task 3.3: Update the notebook and CODEMAP

**Files:**
- Modify: `notebooks/debug.ipynb`
- Modify: `docs/CODEMAP.md`

- [ ] **Step 1: Update the notebook**

In `notebooks/debug.ipynb`, the cell that constructs `KnowledgeGraphExtractor` currently passes `embedding=embedding`. Remove that argument. A `DeprecationWarning` is still emitted if the old form is used, but since the embedding is no longer consumed by the extractor, it is noise.

- [ ] **Step 2: Update CODEMAP**

In `docs/CODEMAP.md`, under `src/grawiki/graph/extraction.py`, clarify that the extractor is now pure LLM extraction and emits nodes with empty embeddings — `GraphRAG.extract_kg_per_chunk` attaches embeddings in one batched call post-extraction. Add a sentence under `src/grawiki/rag/graph_rag.py` documenting the new `_embed_entity_nodes_across_chunks` helper in the data-flow overview.

Also update the Data Flow Overview section: between the current step 7 (extract) and step 8 (resolve), insert a new step "7.5 `GraphRAG._embed_entity_nodes_across_chunks` attaches entity embeddings in one batched API call across all chunks". Renumber downstream steps.

- [ ] **Step 3: Commit**

```bash
git add notebooks/debug.ipynb docs/CODEMAP.md
git commit -m "docs: update CODEMAP and notebook for extractor/embedding split"
```

---

## Phase 4: Optional follow-ups

These are **not required** to hit the speedup goal. They are worth considering if benchmark numbers after Phase 3 are still disappointing, or if the workload shifts to heavy re-ingestion.

### Task 4.1: Persistent embedding cache

**When worth it:** You re-ingest the same or overlapping documents multiple times per day (common during notebook exploration). First run pays the full embedding cost; subsequent runs get cache hits.

**Shape:**
- Small disk-backed cache (e.g. `~/.cache/grawiki/entity_embeddings/<model_id>.sqlite`).
- Key = `(normalized_name, model_id)`. Normalize case-fold + strip.
- Value = embedding vector.
- `_embed_entity_nodes_across_chunks` checks cache before issuing API call; only sends uncached names.
- LRU eviction or size cap is unnecessary for a small personal project — let it grow.

**Risks:** stale cache if embedding model changes. Including `model_id` in the key handles that. If you change embedding model and forget to invalidate, delete the SQLite file.

Defer until Phase 3 numbers are known.

### Task 4.2: HTTP client reuse audit

**When worth it:** If per-request TCP/TLS setup is visibly adding latency (benchmarks plateau above the purely-LLM-bound limit).

**Shape:**
- Inspect `pydantic_ai.Embedder` and `pydantic_ai.Agent` internals: do they share an `httpx.AsyncClient` across calls?
- If they instantiate a new client per call, configure one shared client and pass it through.
- Verify keep-alive is enabled.

This is usually a ~50–100 ms per-call saving at best. Low confidence it matters at current volumes. Park it.

---

## Risks and Rollback Notes

- **Concurrency hides rate-limit errors.** When bumping `max_workers`, watch for HTTP 429. Gracefully retrying 429 is out of scope — for now, if you hit it, drop the worker count.
- **Benchmarks are noisy.** Provider latency varies by minute and time of day. Always average 3+ runs and be skeptical of sub-10% improvements.
- **`DeprecationWarning` on the old extractor signature** is explicit but silent by default. A careful notebook user might miss it. Mention the deprecation in `docs/CODEMAP.md` under the extractor section so future agents are aware.
- **Order of operations.** The pipeline becomes: extract (parallel LLM) → batch-embed entities (single call) → resolve (Scenario A) → persist. All three post-extraction steps depend on the preceding one's output. Keep them sequential; do not parallelize across them.
- **Tests with stub embeddings.** `ConcurrencyTrackingExtractor` and `StaticExtractor` in `tests/rag/test_graph_rag.py` return nodes with preset embeddings. The `_embed_entity_nodes_across_chunks` helper skips nodes that already have a non-empty `embedding`, so these stubs continue to work as-is. Document this contract on the helper's docstring so future test-stub authors don't get surprised.
- **Benchmark corpus is public?** `benchmarks/input.txt` will be committed. Do not use a document containing secrets or copyrighted material.
