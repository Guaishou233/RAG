import os
import sys
import jieba
import requests
from typing import List, Any, Set
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_from_disk
from datasets import Dataset
from huggingface_hub import HfApi

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(__name__))

from utils.logger import logger
from utils.bge_reranker import BGERerankFunction


class Retriever:
    def __init__(self, url: str = None) -> None:
        self.url = url if url else "http://127.0.0.1:5050/search"
        self.headers = {
            "Content-Type": "application/json; charset=utf-8"
        }

    def __call__(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        data = {
        "query": queries,
        "top_k": top_k
        }
        response = requests.post(self.url, json=data, headers=self.headers)

        if response.status_code == 200:
            try:
                result_dict = response.json()
                result = list(result_dict.values())
            except ValueError:
                logger.warning("It is better to check the format of your query.")
                result = []
            return result
        else:
            logger.info(f"Request failed, error code: {response.status_code}, msg: {response.text}")
            return []


class Reranker:
    def __init__(self, model_path: str = None, stopwords_path: str = None) -> None:
        model_path = model_path if model_path else "./model/bge-reranker-v2-m3"
        self.rerank_fn = BGERerankFunction(
            model_name=model_path,  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
            device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )

        # load stopwords
        stopwords_path = stopwords_path if stopwords_path else "./pipeline/stopwords.txt"
        self.stopwords = self.load_stopwords(stopwords_path)
    
    def __call__(self, queries: List[str], documents: List[List[str]], top_k: int = 5) -> List[dict]:
        results = []
        for i, query in enumerate(queries):
            candidates = self.compute_similarities(query, documents[i])
            result = self.rerank_fn(query=query, documents=candidates, top_k=top_k)
            results.append(result)
        return results
    
    @staticmethod
    def load_stopwords(filepath: str) -> Set[str]:
        with open(filepath, 'r', encoding='utf-8') as file:
            stopwords = set([line.strip() for line in file])
        return stopwords

    @staticmethod
    def tokenize_and_filter_stopwords(text: str, stopwords: Set[str]) -> List[str]:
        tokens = jieba.lcut(text)
        filtered_tokens = [token for token in tokens if token not in stopwords and token.strip()]
        return filtered_tokens

    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def compute_similarities(self, query: str, candidates: List[str], threshold: float = 0.90) -> List[str]:
        """
        If `query` is very similar with a certain `candidate`, we concat `query` and `candidate` as new `candidate` to import text-wise features
        so that when send `[query, candidate]` into reranker model, it could achieve better order
        """
        # drop stopwords
        query_keywords = self.tokenize_and_filter_stopwords(query, self.stopwords)
        query_tokens = set(query_keywords)
        
        # jaccard similarity
        similar_candidates = []
        for candidate in candidates:
            keywords = self.tokenize_and_filter_stopwords(candidate, self.stopwords)
            candidate_tokens = set(keywords)
            similarity = self.jaccard_similarity(query_tokens, candidate_tokens)
            if similarity > threshold:
                candidate = " ".join(keywords) + "\n" + candidate
            similar_candidates.append(candidate)
        
        return similar_candidates




class Pipeline:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.reranker = Reranker()
    
    def __call__(self, queries: List[str], top_k: int = 5) -> List[dict]:
        candidates = self.retriever(queries)
        results = self.reranker(queries, candidates, top_k)
        return results
        
def add_column(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    ref = pipeline([instruction + input_text],top_k = 5)
    example["document"] = ref
    return example

    
if __name__ == "__main__":

    pipeline = Pipeline()

    try:
        # 从本地磁盘加载数据集
        dataset = load_from_disk("/root/autodl-tmp/data/china_law_dataset")
        print(f"已从 /root/autodl-tmp/data/china_law_dataset 加载数据集")
        # 获取测试集
        test_dataset = dataset['test']
        split_test_dataset = test_dataset.select(range(100))
        # 打印测试集的第一条数据
        print(split_test_dataset)
        del dataset
    except Exception as e:
        print(f"从磁盘加载数据集时出错: {e}")

    document = []
    document_with_eva = []
    for data in split_test_dataset:
        instruction = data["instruction"]
        input_text = data.get("input", "")
        ref = pipeline([instruction + input_text],top_k = 5)
        texts = [item["text"] for item in ref[0]]  
    
        document.append(texts)
        document_with_eva.append(ref)
    
    new_dataset = split_test_dataset.add_column("document", document)
    new_dataset = new_dataset.add_column("document_with_eva", document_with_eva)

    print(new_dataset)
    from huggingface_hub import login
    login(token="input_access_token")
    new_dataset.push_to_hub("gauishou233/data_with_rag")

    