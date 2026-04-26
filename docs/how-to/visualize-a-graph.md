# Visualize a graph

This guide shows a lightweight visualization workflow built on `FalkorGraphDB.ro_query(...)` and `networkx`. It is an advanced, FalkorDB-specific workflow because it uses direct Cypher queries rather than only facade methods.

## Query a small subgraph

Start by selecting a bounded set of entity nodes and the relationships between them.

```python
seed_rows = database.ro_query(
    "MATCH (e:__entity__) RETURN e.id, e.name, labels(e) ORDER BY e.semantic_key, e.id LIMIT 12"
).result_set
seed_ids = [str(row[0]) for row in seed_rows]

edge_rows = database.ro_query(
    "MATCH (source)-[rel]->(target) "
    "WHERE source.id IN $ids AND target.id IN $ids "
    "RETURN source.id, source.name, labels(source), "
    "target.id, target.name, labels(target), type(rel) "
    "ORDER BY source.name, type(rel), target.name",
    {"ids": seed_ids},
).result_set
```

If the directed query returns no rows, broaden it to an undirected neighborhood query.

## Convert the rows into a `networkx` graph

```python
import networkx as nx


def node_family(labels) -> str:
    labels = set(labels)
    for system_label in ("__memory__", "__chunk__", "__document__", "__entity__"):
        if system_label in labels:
            return system_label
    return "other"


def build_graph(rows):
    graph = nx.Graph()
    for (
        source_id,
        source_name,
        source_labels,
        target_id,
        target_name,
        target_labels,
        rel_label,
    ) in rows:
        graph.add_node(
            source_id,
            name=source_name,
            labels=list(source_labels),
            family=node_family(source_labels),
        )
        graph.add_node(
            target_id,
            name=target_name,
            labels=list(target_labels),
            family=node_family(target_labels),
        )
        graph.add_edge(source_id, target_id, label=rel_label)
    return graph


graph = build_graph(edge_rows)
```

## Draw the graph

```python
import math
import matplotlib.pyplot as plt


def draw_graph(graph, title: str, *, draw_edge_labels: bool = True):
    color_map = {
        "__entity__": "#5B8FF9",
        "__memory__": "#F6BD16",
        "__chunk__": "#61DDAA",
        "__document__": "#65789B",
        "other": "#BFBFBF",
    }
    if graph.number_of_nodes() == 0:
        print(f"{title}: graph is empty.")
        return

    plt.figure(figsize=(12, 8))
    k = 1.4 / math.sqrt(max(graph.number_of_nodes(), 1))
    pos = nx.spring_layout(graph, seed=44, k=k)
    node_colors = [
        color_map.get(graph.nodes[node]["family"], color_map["other"])
        for node in graph.nodes
    ]
    labels = {node: graph.nodes[node]["name"][:50] for node in graph.nodes}

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1200, alpha=0.95)
    nx.draw_networkx_edges(graph, pos, width=1.8, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9)

    if draw_edge_labels and graph.number_of_edges() <= 20:
        edge_labels = {(u, v): data["label"] for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.show()


draw_graph(graph, "Entity-focused graph view")
```

## Focus on one entity or memory

For smaller inspection views, query one entity ego graph or one memory-centered subgraph with `ro_query(...)`, then reuse the same `build_graph(...)` and `draw_graph(...)` helpers.

Call `database.close()` when the session is finished, especially when you are using FalkorDBLite.
