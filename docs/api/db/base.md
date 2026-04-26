# Database Abstractions

The database layer is centered on [`GraphDB`][grawiki.db.base.GraphDB], which defines the backend-agnostic persistence, indexing, search, neighbor-expansion, and merge primitives used elsewhere in the project.

Use this page when implementing a new backend or when clarifying which responsibilities belong to the database adapter rather than the retrieval layer. For the current concrete implementation, see [FalkorDB adapter](falkordb.md).

::: grawiki.db.base
