import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from tqdm import tqdm
from typing import List, Union, Dict
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client import IndexParams

from utils.logger import logger
from utils.bge_retriver import BGEM3EmbeddingFunction


class VectorDataBase(object):

    embedding_model_dim = 1024
    batch_size = 16
    top_k = 5
    metric_type = "IP"

    def __init__(self, db_name: str = "database/text.db") -> None:
        self.client = MilvusClient(db_name)
        self.embedding_fn = BGEM3EmbeddingFunction(
            model_name='./model/ChatLaw-Text2Vec',
            device='cpu',           # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=False,         # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            batch_size=self.batch_size
        )
        self._index_params = None
        self._search_params = {
            "params": {
                "ef": 50,           # search param of HNSW
                "radius": 0.25,     # Radius of the search circle
                "range_filter": 1.0 # Range filter to filter out vectors that are not within the search circle
            }
        }

    def create_collection(self, renew: bool = False, collection_name: str = "text_collection") -> bool:
        if self.client.has_collection(collection_name=collection_name):
            if renew:
                self.drop_collection(collection_name)
                self._create_collection(collection_name)
                return True
            else:
                logger.info(f"Collection: {collection_name} existed")
                return False
        else:
            self._create_collection(collection_name)
            return True

    def _create_collection(self, collection_name):
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # The vectors we will use in this demo has 1024 dimensions
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_model_dim, mmap_enabled=True)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)

        if not self.index_params:
            self.index_params = self.client.prepare_index_params()
            self.index_params.add_index(
                field_name="id"
            )

            self.index_params.add_index(
                field_name="vector", 
                index_type="HNSW",
                metric_type=self.metric_type,
                params={
                    "M": 16,                # number of neighbors
                    "efConstruction": 200,  # search depth
                },
            )

            self.index_params.add_index(
                field_name="doc_id"
            )

        self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=self.index_params
            )
        logger.info(f"Collection: {collection_name} created")

    @property
    def index_params(self) -> IndexParams:
        return self._index_params
    
    @index_params.setter
    def index_params(self, index_params: IndexParams) -> None:
        self._index_params = index_params

    def insert(self, data: List[Dict], collection_name: str = "text_collection") -> Dict:
        interval = 2048
        for i in tqdm(range(0, len(data), interval)):
            res = self.client.insert(collection_name=collection_name, data=data[i:i+interval])
        logger.info(f"Client inserted {res['insert_count']} items.")
        return res
    
    def delete(self, id: int) -> None:
        raise NotImplementedError("Todo")
    
    def update(self, id: int) -> None:
        raise NotImplementedError("Todo")

    def search(self, queries: List[str], top_k: int = None, collection_name: str = "text_collection") -> Dict[str, List[str]]:
        result = {}
        query_embeddings = self.embedding_fn.encode_queries(queries)['dense']
        for i, query_embedding in enumerate(query_embeddings):
            res = self.client.search(
                collection_name=collection_name,        # target collection
                data=[query_embedding],                  # query vectors
                limit=top_k if top_k else self.top_k,   # number of returned entities
                search_params=self._search_params,      # Search parameters
                group_by_field="doc_id",                # Group results by document ID
                group_size=2,                           # returned at most 2 passages per document, the default value is 1
                output_fields=["text", "doc_id"],  # specifies fields to be returned
            )

            candidates = []
            for j, item in enumerate(res[0]):
                candidates.append(item['entity']['text'])
                logger.info(f"Question: {queries[i]}, No.{j+1} result: text_id: {item['id']}, text: {item['entity']['text']}, doc_id: {item['entity']['doc_id']}, score: {item['distance']:0.3f}")
            result[queries[i]] = candidates
        return result
    
    def hybrid_searcch(self):
        raise NotImplementedError("Todo")
    
    def describe_collection(self, collection_name: str = "text_collection") -> None:
        logger.info(f"Collection info: {self.client.describe_collection(collection_name=collection_name)}")
        # logger.info(f"Collection partitions info: {self.client.list_partitions(collection_name=collection_name)}")
        logger.info(f"Collection indices info: {self.client.list_indexes(collection_name=collection_name)}")
    
    def list_collections(self) -> None:
        logger.info(f"Collection list: {self.client.list_collections()}")
    
    def load_collection(self, collection_name: str = "text_collection") -> None:
        self.client.load_collection(collection_name=collection_name)
        res = self.client.get_load_state(collection_name=collection_name)
        logger.info(f"Collection: {collection_name}, state: {res['state']}")
    
    def release_collection(self, collection_name: str = "text_collection") -> None:
        self.client.release_collection(collection_name=collection_name)
        res = self.client.get_load_state(collection_name=collection_name)
        logger.info(f"Collection: {collection_name}, state: {res['state']}")

    def drop_collection(self, collection_name: str = "text_collection") -> None:
        self.client.drop_collection(collection_name=collection_name)
        if not self.client.has_collection(collection_name):
            logger.info(f"Client droped collection: {collection_name}")
        else:
            logger.error(f"Client failed to drop collection: {collection_name}")

    def get_data_embbedding(self, docs: List) -> List:
        texts = [d['content'] for d in docs]
        docs_embeddings = self.embedding_fn.encode_documents(texts)['dense']
        data = [
            {"id": i, "vector": docs_embeddings[i], "text": texts[i], "doc_id": docs[i]['segment_id']}
            for i in range(len(docs))
        ]

        logger.info(f"Client converted {len(data)} entities, dim: {len(data[0]['vector'])}, each with fields: {data[0].keys()}")
        return data
    
    @staticmethod
    def flush_collection(collection_name: str = "text_collection") -> None:
        raise NotImplementedError("Todo")


def  get_vdb(db_name: str = "database/text.db"):
    vdb = VectorDataBase(db_name)
    return vdb

def get_docs(data_file):
    with open(data_file, 'r', encoding="utf-8") as f:
        results = [json.loads(i) for i in f]
    return results

def init_vdb(db_name, collection_name, data_file):
    docs = get_docs(data_file)
    vdb = get_vdb(db_name)
    created = vdb.create_collection(collection_name=collection_name)

    if created:
        data = vdb.get_data_embbedding(docs)
        vdb.insert(data, collection_name)

    vdb.describe_collection(collection_name)
    vdb.list_collections()
    vdb.load_collection(collection_name)
    # vdb.release_collection(collection_name)
    return vdb

data_file = "./data/laws_data_chunk.jsonl"
db_name = "database/text.db"
collection_name = "text_collection"
vdb = init_vdb(db_name, collection_name, data_file)
