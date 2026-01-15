import os
from collections.abc import Iterable

import chromadb
import re
from jixia.structs import DeclarationKind, LeanName, parse_name
from psycopg import Connection
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict

from database.embedding import MistralEmbedding


class Record(BaseModel):
    module_name: LeanName
    kind: DeclarationKind
    name: LeanName
    start: int | None
    stop: int | None
    signature: str
    type: str
    value: str | None
    docstring: str | None
    informal_name: str
    informal_description: str

    model_config = ConfigDict(extra="allow")

class QueryResult(BaseModel):
    result: Record
    distance: float

class MathAtlasRecord(BaseModel):
    name: str
    text: str
    node_id: str
    element_id: str
    name_element_id: str

class MathAtlasQueryResult(BaseModel):
    result: MathAtlasRecord
    distance: float

class Retriever:
    def __init__(self, path: str, conn: Connection):
        self.conn = conn
        self.client = chromadb.PersistentClient(path)
        self.collection = self.client.get_collection(name="leansearch", embedding_function=None)
        with open("prompt/retrieve_instruction.txt") as fp:
            instruction = fp.read()
        self.embedding = MistralEmbedding(os.environ["EMBEDDING_URL"], instruction)

        # Enable mathatlas searching
        with open("prompt/mathatlas_retrieve_instruction.txt") as fp:
            instruction = fp.read()
        self.mathatlas_embedding = MistralEmbedding(os.environ["EMBEDDING_URL"], instruction)
        self.mathatlas_collection = self.client.get_collection(name="mathatlassearch", embedding_function=None)

    def batch_fetch(self, name: Iterable[LeanName]) -> list[Record]:
        ret = []
        with self.conn.cursor(row_factory=class_row(Record)) as cursor:
            for n in name:
                cursor.execute(
                    """
                    SELECT * FROM record
                    WHERE name = %s
                    """,
                    (Jsonb(n),),
                )
                ret.append(cursor.fetchone())
        return ret

    def batch_search(self, query: list[str], num_results: int) -> list[list[QueryResult]]:
        query_embedding = self.embedding.embed(query)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=num_results,
            include=["distances"],
        )
        ret = []
        with self.conn.cursor(row_factory=class_row(Record)) as cursor:
            for ids, distances in zip(results["ids"], results["distances"]):
                current_results = []
                for doc_id, distance in zip(ids, distances):
                    # NILAY - index is not the right column here, it was inserted with name
                    # Also, module name doesn't even exist in the doc ID
                    # module_name, _, index = doc_id.partition(":")
                    # module_name = parse_name(module_name)
                    name = doc_id.split(" ")
                    cursor.execute(
                        """
                        SELECT * FROM record
                        WHERE name = %s
                        """,
                        (Jsonb(name),),
                    )
                    result = cursor.fetchone()
                    if result is not None:
                        current_results.append(QueryResult(result=result, distance=distance))
                ret.append(current_results)
        return ret

    def mathatlas_batch_search(self, query: list[str], num_results: int) -> list[list[MathAtlasQueryResult]]:
        query_embedding = self.mathatlas_embedding.embed(query)
        results = self.mathatlas_collection.query(
            query_embeddings=query_embedding,
            n_results=num_results,
            include=["distances", "documents"],
        )
        ret = []
        for ids, distances, docs in zip(results["ids"], results["distances"], results["documents"]):
            current_results = []
            for doc_id, dist, doc in zip(ids, distances, docs):
                # doc_id format = "nameelementid=`{name_element_id}`;elementid=`{element_id}`;nodeid=`{node_id}`;" 
                name_element_id, element_id, node_id = re.findall(r"=`(.*?)`", doc_id)
                name, text = doc.split('\n', 1)
                result = MathAtlasRecord(
                    name=name,
                    text=text,
                    name_element_id=name_element_id,
                    element_id=element_id,
                    node_id=node_id
                )
                current_results.append(MathAtlasQueryResult(result=result, distance=dist))

            ret.append(current_results)
        return ret
