# Concepts

GraWiki sits at the intersection of several ideas that are often discussed
separately: knowledge graphs, heterogeneous graphs, graph-based retrieval, and
LLM-maintained memory.

## What a knowledge graph is

A knowledge graph is a graph-structured representation of entities and the
relationships between them. In practice, that usually means:

- nodes represent entities, concepts, people, organizations, documents, or memories,
- edges represent typed relations between those nodes,
- both nodes and edges can carry properties,
- the graph is meant to preserve semantics, not only connectivity.

This matters because text alone hides structure inside sequences of tokens.
Knowledge graphs make that structure explicit: what the entities are, how they
relate, and which neighborhoods of facts belong together [@hogan2021knowledgegraphs].

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

That distinction is useful:

- `knowledge graph` emphasizes meaning, entities, and semantic relations,
- `heterogeneous graph` emphasizes typed structure suitable for machine learning,
- in practice, the same underlying graph can be viewed through both lenses.

GraWiki mainly builds a knowledge graph first, but it is helpful to think of the
same structure as a heterogeneous graph when discussing embeddings, similarity,
and downstream graph learning.

## Why graph structure matters

Graph structure is valuable because many important relationships are not local to
one sentence or one chunk. Documents often mention concepts repeatedly, connect
them indirectly, and rely on context spread across paragraphs or across multiple
sources. A graph representation can preserve those cross-document and
cross-section links more naturally than flat chunk stores alone.

This is one of the reasons graph-enhanced retrieval has become attractive in RAG
systems: graph structure supports neighborhood expansion, multi-hop context, and
entity-centric retrieval patterns that are awkward to model with pure vector
search [@jiang2024graphrag; @pan2024kgrag].

At a practical level, that means a graph can help with:

- entity-centric search instead of only chunk-centric search,
- tracing relationships between concepts and sources,
- expanding context through neighboring nodes,
- reusing accumulated structure over time instead of rediscovering it on every query.

## LLMs for knowledge graph extraction

Turning text into a knowledge graph is not new, but LLMs changed the operating
point. Traditional NLP pipelines can derive dependency graphs and other local
syntactic structures, while LLMs are often better at extracting higher-level
semantic entities and relations from longer passages of text.

The general extraction pattern is:

1. Define a target schema for nodes and relations.
2. Chunk or otherwise prepare the source text.
3. Ask the model for structured graph output.
4. Normalize entity names and relation labels.
5. Validate and persist the resulting graph.

Recent work has explored this pattern directly, including benchmark and survey
work on LLM-based graph construction from text [@gillani2024kgextraction;
@mihindukulasooriya2023text2kgbench; @zhu2024llmkg].

GraWiki follows that family of approaches, but keeps the persistence layer and
retrieval layer intentionally simple: extract typed entities and relations,
persist them, then retrieve from the resulting graph later.

## From knowledge graph to graph learning

Your article materials also emphasize an important second step: once a graph has
typed nodes and typed edges, it can be viewed as an input to graph learning
methods rather than only as a database or retrieval structure.

Graph neural networks operate by propagating information across neighborhoods.
In message-passing terms, nodes repeatedly collect messages from neighbors,
aggregate them, and update their representations [@gilmer2017mpnn;
@hamilton2020grl].

That matters for knowledge graphs because it lets a system learn structure-aware
embeddings based not only on raw text features, but also on topology, relation
types, and local neighborhoods. This is especially relevant for tasks such as:

- link prediction,
- entity similarity,
- node classification,
- graph-level reasoning.

GraWiki does not currently expose a full GNN training pipeline, but the project
is aligned with that direction conceptually: extract semantic graph structure
first, then make it available for retrieval, memory, and eventually richer graph
reasoning workflows.

## LLM wiki style memory

The other half of GraWiki is the memory side: a graph should not only represent
facts extracted from documents, but also durable records of prior agent work.
GraWiki stores memories as explicit graph nodes and can recall them together with
linked graph context.

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

GraWiki is not an Obsidian wiki tool, but it shares the same high-level concern:
knowledge should accumulate and become reusable, instead of being rebuilt from
scratch for every prompt.

## Why combine these ideas

Combining knowledge-graph extraction with persistent memory gives GraWiki a
broader target than classic RAG:

- document ingestion creates reusable graph structure,
- agent interactions can add memory to the same graph,
- retrieval can draw from both extracted knowledge and prior experience,
- the graph becomes a shared substrate for context assembly.

That combination reflects a wider trend in LLM systems work: external knowledge
stores, explicit retrieval, structured representations, and engineered workflows
matter as much as the base model for useful applications [@ng2024aie;
@galkin2024foundationkg; @huang2025kgfm].
