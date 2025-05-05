import time
import requests

def get_pipeline_candiates(query):
    url = "http://127.0.0.1:5050/pipeline"
    data = {
        "query": ["杭州天气怎么样?"],
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

if __name__ == "__main__":
    get_pipeline_candiates("杭州天气怎么样?")