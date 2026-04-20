# GraWiki

This repositoy contains a Graph-Wiki (GraWiki) -- a lightweight implementation of fusion of two concepts:

1. GraphRAG
2. and Andrej Karpathy's "LLM Wiki" agentic memory.

The main idea is to use a graph database as the "memory" of an LLM agent, and to use the LLM to extract structured knowledge from text and store it in the graph. The graph can then be queried by the LLM to answer questions, generate summaries, or perform other tasks.

Two sets of functionalities are available here:

1. Knowledge graph extraction from text and storing it in the graph database. Later -- it is possible to query the graph database to retrieve relevant information for a given query, and use it to answer questions or generate summaries.
2. Storage and retrieval of "agentic memories" -- a result of previous queries + RAG results + agent reasoning and answer.

## Development

Install development tooling:

```bash
uv sync --group dev
```

Install git hooks:

```bash
uv run pre-commit install
```

Run all configured checks manually:

```bash
uv run pre-commit run --all-files
```
