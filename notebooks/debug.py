# %% # Imports
import autoroot  # noqa
from pathlib import Path
from importlib import reload

from dotenv import load_dotenv
from rich import print as pprint
from tqdm.auto import tqdm
import src.grawiki.doc_processing.chunkers as chunkers
import src.grawiki.doc_processing.document_processing as doc_proc
import src.grawiki.graph.graph_extraction as kg
import src.grawiki.graph.models as graph_models
from falkordb import FalkorDB

doc_proc = reload(doc_proc)
chunkers = reload(chunkers)

# %% # Load environment variables
load_dotenv(override=True)

# %% # Doc experiments
doc_path = Path("experimental_data/agent_architectures.txt")
doc = doc_proc.read_document(doc_path)
pprint(doc.content[:10])

# %% # Chunking docs
chunker = chunkers.Chunker(strategy="sentence")
chunks = chunker.chunk(doc)
pprint(chunks[0].content[:10])


# %% # Graph db prep
db = FalkorDB(host="localhost", port=6379)
g = db.select_graph("experiment")


# %% # Node creation
document_node = graph_models.DocumentNode.from_document(doc)
chunks = [graph_models.ChunkNode.from_chunk(chunk) for chunk in chunks]


doc_create_cyper = """CREATE (d:__document__ {id: $id, name: $name, content: $content, metadata: $metadata})"""
chunk_upsert_cypher = """
MATCH (d:__document__ {id: $document_id})
MERGE (c:__chunk__ {id: $id})
ON CREATE SET
    c.name = $name,
    c.content = $content,
    c.metadata = $metadata,
    c.document_id = $document_id
MERGE (d)-[:__has_chunk__]->(c)
RETURN d, c
"""

node_upsert_cypher = """
MATCH (c:__chunk__ {id: $chunk_id})
MERGE (n:__entity__:{label:$label} {id: $id})
ON CREATE SET
    n.name = $name,
    n.properties = $properties

MERGE (c)-[:__mentions__]->(n)
RETURN c, n
"""

node_rels_upsert_cypher = """
MATCH (source:__entity__ {id: $source_id})
MATCH (target:__entity__ {id: $target_id})
MERGE (source)-[r:{label}]->(target)
"""

# %% # Insert document
document_dict = document_node.model_dump()
document_dict["metadata"] = [f"{k}:{v}" for k, v in document_dict["metadata"].items()]
inset_dict = g.query(doc_create_cyper, document_dict)


# %% # Insert chunks


for chunk in chunks:
    chunk_dict = chunk.model_dump()
    chunk_dict["metadata"] = [f"{k}:{v}" for k, v in chunk_dict["metadata"].items()]
    result = g.query(chunk_upsert_cypher, chunk_dict)


# %% # Chunk entity extraction
async def inject():
    kg_extractor = kg.KnowledgeGraphExtractor(model="openai:gpt-5-mini")
    chunk_graphs = {}
    for chunk in tqdm(chunks, desc="Extracting entities from chunks"):
        result = await kg_extractor.extract(chunk)
        chunk_graphs[chunk.id] = result


chunk_graphs = {}

# %% # Insert nodes from chunk graphs
for chunk_id, graph in chunk_graphs.items():
    for node in graph.nodes:
        node_dict = node.model_dump()
        node_dict["metadata"] = [f"{k}:{v}" for k, v in node_dict["metadata"].items()]
        result = g.query(chunk_upsert_cypher, node_dict)
