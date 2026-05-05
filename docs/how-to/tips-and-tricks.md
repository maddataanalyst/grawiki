# Tips & Tricks

## Speed up KG extraction

The knowledge graph extraction process can be significantly sped up by lowering the reasoning effort of the underlying LLM. This is useful during ingestion when you prioritize throughput over deep reasoning.

Pass `reasoning_effort: "minimal"` through `kg_extractor_kwargs` when initializing `GraphRAG`:

```python
from grawiki.db import FalkorGraphDB
from grawiki.rag import GraphRAG

database = FalkorGraphDB(db_path="/tmp/my_graph.db", graph_name="grawiki")
rag = GraphRAG(
    model="openai:gpt-4.1-mini",
    embedding_model="openai:text-embedding-3-small",
    db=database,
    kg_extractor_kwargs={"reasoning_effort": "minimal"},
)
```

**Note:** This option is supported by certain providers (e.g., OpenAI). The `reasoning_effort` parameter is forwarded to the model's API and can reduce token usage and latency during extraction. Use `"minimal"` for faster ingestion, and consider higher reasoning levels when extraction quality is critical.

## Change KG extraction language

`GraphRAG` defaults to English when extracting entity names, relationship labels, and textual properties. Override that per facade instance with `kg_output_language`:

```python
from grawiki.db import FalkorGraphDB
from grawiki.rag import GraphRAG

database = FalkorGraphDB(db_path="/tmp/my_graph.db", graph_name="grawiki")
rag = GraphRAG(
    model="openai:gpt-4.1-mini",
    embedding_model="openai:text-embedding-3-small",
    db=database,
    kg_output_language="Polish",
)
```
