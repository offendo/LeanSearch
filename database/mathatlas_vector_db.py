import logging
import os

import chromadb
from more_itertools import chunked
from neo4j import GraphDatabase
from tqdm import tqdm

from .embedding import MistralEmbedding

logger = logging.getLogger(__name__)


def create_vector_db(path: str, batch_size: int):
    with open("prompt/mathatlas_embedding_instruction.txt") as fp:
        instruction = fp.read()
    embedding = MistralEmbedding(os.environ["EMBEDDING_URL"], instruction)

    client = chromadb.PersistentClient(path)
    collection = client.create_collection(
        name="mathatlassearch",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    neo4j_driver = GraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "password"),
    )
    neo4j_driver.verify_connectivity()
    neo4j_driver.verify_authentication()

    query = """MATCH (n:Name)-[:NAMES]->(d:Definition) RETURN n,d"""
    records, _, _ = neo4j_driver.execute_query(query)

    pbar = tqdm(records, desc="Creating embeddings")

    for batch in chunked(pbar, batch_size):
        batch_doc = []
        batch_id = []
        for record in batch:
            name_node, definition_node = record
            name_element_id = name_node.element_id
            informal_name = name_node.get("text")
            informal_description = definition_node.get("text")
            element_id = definition_node.element_id
            node_id = definition_node.get("id")

            # Format the ID and document
            batch_id.append(f"nameelementid=`{name_element_id}`;elementid=`{element_id}`;nodeid=`{node_id}`;")
            batch_doc.append(f"{informal_name}\n{informal_description}")
            if os.environ["DRY_RUN"] == "true":
                logger.info("DRY_RUN:skipped embedding: %s", batch_id[-1])
        if os.environ["DRY_RUN"] == "true":
            return
        batch_embedding = embedding.embed(batch_doc)
        collection.add(embeddings=batch_embedding, ids=batch_id, documents=batch_doc)
