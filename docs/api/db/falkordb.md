# FalkorDB adapter

[`FalkorGraphDB`][grawiki.db.falkordb.FalkorGraphDB] is the project's current `GraphDB` implementation. Most applications should start with [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG], which uses the adapter through the backend-agnostic database interface. Use `FalkorGraphDB` directly when you need to configure the backend, inspect Cypher queries, or work with FalkorDB-specific helpers.

## Connection modes

`FalkorGraphDB` supports two deployment modes.

- FalkorDBLite uses `db_path` and runs an embedded, file-backed database. This mode does not require Docker.
- Full FalkorDB uses `host` and `port` to connect to an external server process.

Provide exactly one of `db_path` or `host`. Passing both, or neither, raises `ValueError`.

## Installation

Install the dependency set that matches the deployment mode.

### FalkorDBLite

```bash
uv sync --group falkordblite
```

### Full FalkorDB server

```bash
uv sync --group falkordb
```

If you want a local Docker-backed server for examples or notebook work, start the compose file from the repository root:

```bash
docker compose -f notebooks/docker-compose.yml up -d
```

## Minimal examples

### FalkorDBLite

```python
from grawiki.db import FalkorGraphDB

database = FalkorGraphDB(
    "my_graph",
    db_path="/tmp/my_graph.db",
)
```

### Full FalkorDB server

```python
from grawiki.db import FalkorGraphDB

database = FalkorGraphDB(
    "my_graph",
    host="localhost",
    port=6379,
)
```

## Operational notes

- `close()` should be called explicitly in tests and short-lived scripts. This is especially important for FalkorDBLite, which manages an embedded Redis-backed process.
- The FalkorDB Browser UI at `http://localhost:3000` applies to server mode, including the Docker setup above. FalkorDBLite does not expose that browser.
- `GraphRAG.ingest(...)`, `GraphRAG.ingest_text(...)`, and `GraphRAG.remember(...)` usually call `setup()` for you. Direct adapter usage may require an explicit `await database.setup(...)` before indexing or search operations.

## Advanced adapter helpers

`FalkorGraphDB` also exposes backend-specific methods that are useful when you are debugging queries or building custom tooling:

- `query(...)` executes write-capable Cypher.
- `ro_query(...)` executes read-only Cypher and is the simplest option for ad hoc inspection.
- `explain(...)` returns FalkorDB's execution plan for a Cypher query.

Advanced users can also tune vector index creation through the constructor arguments `vector_similarity_function`, `vector_index_m`, `vector_index_ef_construction`, and `vector_index_ef_runtime`.

::: grawiki.db.falkordb.FalkorGraphDB
