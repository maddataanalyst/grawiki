# Database Abstractions

The database layer is centered on [`GraphDB`][grawiki.db.base.GraphDB], which defines the backend-agnostic persistence, indexing, search, neighbor-expansion, and merge primitives used elsewhere in the project.

This is the main reference page to use when implementing a new backend or trying to understand which responsibilities belong to the adapter versus the retrieval layer.

::: grawiki.db.base
