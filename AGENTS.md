# AGENTS.md

## Project Overview

GraWiki is an early-stage open source Python project that combines GraphRAG-style knowledge extraction with Andrej Karpathy's "LLM Wiki" concept for agentic memory.

The repository currently focuses on two core capabilities:

1. Extracting structured knowledge from text and storing it in a graph representation.
2. Storing and retrieving agent memories built from prior queries, retrieved context, reasoning, and answers.

## Repository Layout

- `src/grawiki/`: main application package.
- `src/grawiki/doc_processing/`: document loading and chunking.
- `src/grawiki/graph/`: graph models, prompts, and extraction logic.
- `main.py`: minimal entrypoint.
- `notebooks/`: exploratory research and debugging notebooks.

## Coding Rules

1. Always use `uv` for package management.
   - Add and update dependencies with `uv add`, `uv remove`, and `uv sync`.
   - Do not use `pip install`, `poetry`, or other package managers for this repository.

2. Write docstrings in NumPy format everywhere.
   - Apply NumPy-style docstrings to modules, classes, functions, and methods.
   - Keep docstrings accurate and update them when behavior or signatures change.

3. Use `pytest` for testing.
   - Add tests for new behavior and regressions.
   - Prefer deterministic unit tests over notebook-only validation.

4. Run linting before commits.
   - Use the configured `pre-commit` hooks before committing changes.
   - Ensure `ruff` checks pass before opening or updating a pull request.

## Standard Developer Commands

Install development dependencies:

```bash
uv sync --group dev
```

Install git hooks:

```bash
uv run pre-commit install
```

Run all lint checks:

```bash
uv run pre-commit run --all-files
```

Run tests:

```bash
uv run pytest
```
