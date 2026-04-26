# GraWiki Notebooks

This directory contains tutorial notebooks for getting started with GraWiki.

## Setup

You can run the notebooks with either a file-based database (FalkorDBLite) or a full FalkorDB server via Docker.

### Option A — FalkorDBLite (file-based, no Docker)

Install the notebook stack from a repository checkout:

```bash
uv sync --extra falkordblite --extra notebooks --extra viz
```

If you only need the published package dependencies, install:

```bash
pip install 'grawiki[falkordblite,notebooks,viz]'
```

Use it in Python:

```python
from grawiki.db import FalkorGraphDB

graph = FalkorGraphDB(
    "my_graph",
    db_path="/tmp/my_graph.db",
)
```

### Option B — Full FalkorDB with Docker

Start the database server:

```bash
docker compose -f notebooks/docker-compose.yml up -d
```

Install the Docker-backed notebook stack from a repository checkout:

```bash
uv sync --extra falkordb --extra notebooks --extra viz
```

If you only need the published package dependencies, install:

```bash
pip install 'grawiki[falkordb,notebooks,viz]'
```

Use it in Python:

```python
from grawiki.db import FalkorGraphDB

graph = FalkorGraphDB(
    "my_graph",
    host="localhost",
    port=6379,
)
```

When using Docker mode, the FalkorDB Browser web UI is available at `http://localhost:3000`.

## Running the Notebooks

Run the notebooks in order:

1. `01_ingest_and_deduplicate.ipynb` — builds the graph from source documents
2. `02_agent_memory_and_recall.ipynb` — demonstrates agent memory workflows
3. `03_visualize_graph.ipynb` — visualizes the resulting graph structure
