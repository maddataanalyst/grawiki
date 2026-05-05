# Ingest a document

This guide shows the stepwise ingestion workflow exposed by [`GraphRAG`][grawiki.rag.graph_rag.GraphRAG]. Use it when you want to inspect intermediate artifacts such as chunks, embeddings, document nodes, or per-chunk extraction results.

If you only need the full pipeline, use `await rag.ingest(path)` instead.

Format detection happens once at read or entry time:

- `.md` and `.markdown` files are loaded as markdown content.
- `.pdf` files are converted to markdown in memory with `pymupdf4llm` and then follow the same markdown chunking path.
- `ingest_text(...)` defaults to plain text, but you can explicitly set `format="markdown"` for in-memory markdown.

## Prepare the database and facade

```python
from pathlib import Path

from grawiki.db import FalkorGraphDB
from grawiki.doc_processing.chunkers import build_default_markdown_pipeline
from grawiki.doc_processing.chunk_processors import HypotheticalQuestionsChunkProcessor
from grawiki.rag import GraphRAG

database = FalkorGraphDB(
    "my_graph",
    db_path="/tmp/my_graph.db",
)
rag = GraphRAG(
    model="openai:gpt-4.1-mini",
    embedding_model="openai:text-embedding-3-small",
    db=database,
    markdown_pipeline=build_default_markdown_pipeline(),
    chunk_processors=[
        HypotheticalQuestionsChunkProcessor(
            model="openai:gpt-4.1-mini",
            num_question=3,
        )
    ],
)

await database.setup()
source_path = Path("path/to/document.md")
```

`markdown_pipeline=` is optional. When you omit it, markdown and PDF-derived markdown fall back to the generic text `Chunker`. Pass `build_default_markdown_pipeline()` or any other `Pipeline` only when you want markdown-aware text/code/table chunk preservation.

By default, KG extraction emits English entity names and relationship labels. Set `kg_output_language="Polish"` or another language on `GraphRAG(...)` when you want the extracted graph strings localized.

## Run the ingestion steps explicitly

```python
document = rag.read_document(source_path)
chunks = rag.chunk_document(document)
chunks = await rag.process_chunks(chunks)
document_embedding = await rag.embed_document(document)
chunk_embeddings = await rag.embed_chunks(chunks)

document_node = rag.build_document_node(document, document_embedding)
chunk_nodes = rag.build_chunk_nodes(chunks, chunk_embeddings)
await rag.persist_document_and_chunks(document_node, chunk_nodes)

chunk_graphs = await rag.extract_kg_per_chunk(chunks)
await rag.persist_entities_and_relationships(
    [chunk.id for chunk in chunks],
    chunk_graphs,
)
```

## What each step does

1. `read_document(...)` loads the source file into a `Document` model and records `filepath`, `source_format`, and `content_format` metadata.
2. `chunk_document(...)` splits the document into `Chunk` models. Plain text always uses the generic text chunker. Markdown content uses that same generic chunker unless you explicitly pass `markdown_pipeline=` on `GraphRAG(...)`; with a configured markdown pipeline, ordered text, code, and table chunks are preserved.
3. `process_chunks(...)` applies any configured `chunk_processors=` in order. Use this step to keep the explicit workflow aligned with one-shot ingestion.
4. `embed_document(...)` returns an empty list during normal ingestion. `embed_chunks(...)` creates the retrieval vectors used for chunk search and indexing.
5. `build_document_node(...)` and `build_chunk_nodes(...)` convert the transient models into persisted graph-node shapes. Document nodes are written without vector embeddings by default.
6. `persist_document_and_chunks(...)` writes the document and chunk nodes before extraction.
7. `extract_kg_per_chunk(...)` produces one extracted graph per processed chunk.
8. `persist_entities_and_relationships(...)` writes the extracted entities and relationships back to the graph.

## Ingest markdown already in memory

When you already have markdown text in memory, `ingest_text(..., format="markdown")` follows the same content-format routing. With `markdown_pipeline=` configured it uses markdown-aware chunking; otherwise it falls back to the generic text chunker:

```python
await rag.ingest_text(
    "# Notes\n\nThis paragraph becomes a text chunk.\n\n```python\nprint(1)\n```",
    title="In-memory markdown",
    format="markdown",
)
```

## Ingest a PDF through the markdown path

PDF input is converted to markdown in memory first, then chunked according to the same markdown-content rules:

```python
pdf_document = rag.read_document(Path("path/to/paper.pdf"))
assert pdf_document.metadata["source_format"] == "pdf"
assert pdf_document.metadata["content_format"] == "markdown"

pdf_chunks = rag.chunk_document(pdf_document)
```

## One-shot alternative

When you do not need intermediate artifacts, use the facade method:

```python
await rag.ingest(source_path)
```

`GraphRAG.ingest(...)` runs the same sequence, including `process_chunks(...)`, and calls `setup()` internally.
