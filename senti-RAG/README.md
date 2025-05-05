# 1. prepration
+ Run `data/get_data.ipynb` to get `data/sentiment.jsonl`
+ `python utils/text2chunk.py` to get `data/sentiment_chunk.jsonl`
+ Downloads [bge-m3](https://huggingface.co/BAAI/bge-m3) and [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) from huggingface. Then store them in `model` folder.

# 2. start RAG server
`python server.py`

Once the server has started, you could access it by
```python
import time
import requests

def get_pipeline_candiates(query):
    url = "http://127.0.0.1:5050/pipeline"
    data = {
        "query": ["杭州"],
        "top_k": 5
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }
    s = time.time()
    response = requests.post(url, json=data, headers=headers)
    # print(f"=========== running time: {time.time()-s}")

    if response.status_code == 200:
        try:
            result = response.json()
            # pprint(list(zip(query, result)))
        except ValueError:
            # print(response.text)
            result = []
    else:
        print(f"Request failed, error code: {response.status_code}")
        print(response.text)
        result = []
    
    print(result)
```
The code above will return a list of related candidates about your queries. You can use the candidates as references and send them to LLMs to have a simple Encyclopedia RAG system.
