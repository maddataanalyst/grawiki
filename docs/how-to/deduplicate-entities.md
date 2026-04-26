# Deduplicate entities

This guide shows the recommended order for reviewing and merging duplicate entities. Start with inspection, review the proposed matches, then apply merges only after you are satisfied with the candidate groups.

## Inspect semantic-key collisions first

Exact semantic-key collisions are the safest place to start.

```python
collision_groups = await rag.find_entity_collision_candidates(
    limit=5,
    threshold=0.9,
)
```

These groups show entities that already share the same `semantic_key`, together with similarity annotations for review.

## Inspect broader duplicate candidates

Then run the broader matcher-based scan.

```python
duplicate_candidates = await rag.find_entity_duplicate_candidates(
    limit=5,
    threshold=0.9,
)
```

This report combines semantic-key collisions with additional similarity-based candidates.

## Run a dry merge first

Use `dry_run=True` before applying destructive changes.

```python
dry_run_reports = await rag.dedupe_entities(
    limit=5,
    threshold=0.9,
    min_merge_score=0.95,
    dry_run=True,
)
```

Review the returned `MergeReport` objects before proceeding. In particular, check the chosen master node, duplicate ids, merged labels, and property conflicts.

## Apply merges

Apply merges only after reviewing the dry-run output.

```python
applied_reports = await rag.dedupe_entities(
    limit=5,
    threshold=0.9,
    min_merge_score=0.95,
    dry_run=False,
)
```

This step updates the graph and removes duplicate entity nodes.

## Safety notes

- Prefer reviewing collision groups before broader similarity groups.
- Use conservative thresholds until you understand the shape of your data.
- Treat `dry_run=False` as a destructive operation and keep it out of exploratory notebooks until the candidate sets look correct.

For the lower-level types involved in this workflow, see [Similarity and Deduplication](../api/similarity/index.md).
