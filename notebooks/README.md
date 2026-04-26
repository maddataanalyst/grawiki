# GraWiki Notebooks

This directory contains tutorial notebooks for getting started with GraWiki.

## Setup

You can run the notebooks with either a file-based database (FalkorDBLite) or a full FalkorDB server via Docker.

### Option A — FalkorDBLite (file-based, no Docker)

Install the lightweight dependency:

```bash
uv sync --group falkordblite
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
cd notebooks && docker-compose up -d
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
