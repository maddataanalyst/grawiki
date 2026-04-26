# Concepts

GraWiki combines several related ideas: knowledge graphs, heterogeneous graphs, graph-based retrieval, and LLM-maintained memory.

## What a knowledge graph is

A knowledge graph is a graph-structured representation of entities and the
relationships between them. In practice, that usually means:

- nodes represent entities, concepts, people, organizations, documents, or memories,
- edges represent typed relations between those nodes,
- both nodes and edges can carry properties,
- the graph is meant to preserve semantics, not only connectivity.

Text alone hides structure inside sequences of tokens. Knowledge graphs make that structure explicit by representing entities, relationships, and local neighborhoods of facts [@hogan2021knowledgegraphs].

In GraWiki, that translates into a property-graph style representation where:

- source files become document nodes,
- chunks become chunk nodes,
- extracted entities become reusable graph nodes,
- typed relationships connect documents, chunks, entities, and memories.

## Knowledge graphs and heterogeneous graphs

Knowledge graphs are closely related to heterogeneous graphs. A heterogeneous
graph is the more general graph-learning term for a graph with multiple node
types and edge types. Following the formulation summarized in your source
material, such a graph can be written as `G = (V, E, A, R, phi, psi)`, where
`phi` maps vertices to node types and `psi` maps edges to relation types
[@shi2022heterogeneous].

The distinction is useful:

- `knowledge graph` emphasizes meaning, entities, and semantic relations,
- `heterogeneous graph` emphasizes typed structure suitable for machine learning,
- in practice, the same underlying graph can be viewed through both lenses.

GraWiki primarily builds a knowledge graph, but the same structure can also be treated as a heterogeneous graph when discussing embeddings, similarity, and downstream graph learning.

## Why graph structure matters

Many relationships are not local to one sentence or one chunk. Documents repeat concepts, connect them indirectly, and distribute context across paragraphs or across multiple sources. A graph representation preserves those links more naturally than a flat chunk store.

This is one reason graph-enhanced retrieval has become useful in RAG systems. Graph structure supports neighborhood expansion, multi-hop context, and entity-centric retrieval patterns that are awkward to express with vector search alone [@jiang2024graphrag; @pan2024kgrag].

At a practical level, that means a graph can help with:

- entity-centric search instead of only chunk-centric search,
- tracing relationships between concepts and sources,
- expanding context through neighboring nodes,
- reusing accumulated structure over time instead of rediscovering it on every query.

## LLMs for knowledge graph extraction

Turning text into a knowledge graph is not new, but LLMs changed the practical workflow. Traditional NLP pipelines can derive dependency graphs and other local syntactic structures, while LLMs are often better at extracting higher-level semantic entities and relations from longer passages.

The general extraction pattern is:

1. Define a target schema for nodes and relations.
2. Chunk or otherwise prepare the source text.
3. Ask the model for structured graph output.
4. Normalize entity names and relation labels.
5. Validate and persist the resulting graph.

Recent work has explored this pattern directly, including benchmark and survey
work on LLM-based graph construction from text [@gillani2024kgextraction;
@mihindukulasooriya2023text2kgbench; @zhu2024llmkg].

GraWiki follows that family of approaches but keeps the persistence and retrieval layers simple: extract typed entities and relations, persist them, and retrieve from the resulting graph later.

## From knowledge graph to graph learning

Once a graph has typed nodes and typed edges, it can also be treated as input to graph learning methods rather than only as a database or retrieval structure.

Graph neural networks operate by propagating information across neighborhoods.
In message-passing terms, nodes repeatedly collect messages from neighbors,
aggregate them, and update their representations [@gilmer2017mpnn;
@hamilton2020grl].

This allows a system to learn structure-aware embeddings based on text features, topology, relation types, and local neighborhoods. That is especially relevant for tasks such as:

- link prediction,
- entity similarity,
- node classification,
- graph-level reasoning.

GraWiki does not currently expose a full GNN training pipeline, but the project is compatible with that direction: extract semantic graph structure first, then make it available for retrieval, memory, and later graph reasoning workflows.

## LLM wiki style memory

The other half of GraWiki is persistent memory. The graph does not only represent facts extracted from documents; it also stores durable records of prior agent work. GraWiki persists memories as explicit graph nodes and can recall them together with linked graph context.

This is adjacent to Andrej Karpathy's April 2026 `LLM Wiki` note and related
tweet, which describe an LLM-maintained, persistent markdown knowledge base
rather than query-time rediscovery from raw documents alone [@karpathy2026llmwiki;
@karpathy2026llmwiki_tweet].

> "Instead of just retrieving from raw documents at query time, the LLM
> incrementally builds and maintains a persistent wiki - a structured,
> interlinked collection of markdown files that sits between you and the raw
> sources." [@karpathy2026llmwiki]

The tweet version makes the same point in more operational language: raw sources
are collected, compiled by an LLM into a markdown wiki, then queried and updated
over time, with outputs filed back into the knowledge base [@karpathy2026llmwiki_tweet].

GraWiki is not an Obsidian wiki tool, but it shares the same concern: knowledge should accumulate and become reusable instead of being rebuilt from scratch for every prompt.

## Why combine these ideas

Combining knowledge-graph extraction with persistent memory gives GraWiki a wider scope than classic RAG:

- document ingestion creates reusable graph structure,
- agent interactions can add memory to the same graph,
- retrieval can draw from both extracted knowledge and prior experience,
- the graph becomes a shared substrate for context assembly.

This combination matches a broader direction in LLM systems work: external knowledge stores, explicit retrieval, structured representations, and engineered workflows matter as much as the base model in practical applications [@ng2024aie; @galkin2024foundationkg; @huang2025kgfm].
