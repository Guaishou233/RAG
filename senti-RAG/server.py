from flask import Flask, request, jsonify

from database.vector_database import vdb
from pipeline.main_pipeline import Pipeline as ppline


app = Flask(__name__)
collection_name = "text_collection"

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', [])
    top_k = data.get('top_k', 10)

    if len(query) < 1:
        return jsonify({"status": "error", "message": "Request must have at least 1 question."}), 400
    
    results = vdb.search(query, top_k, collection_name)
    return jsonify(results), 200


@app.route('/pipeline', methods=['POST'])
def pipeline():
    data = request.json
    query = data.get('query', [])
    top_k = data.get('top_k', 5)

    if len(query) < 1:
        return jsonify({"status": "error", "message": "Request must have at least 1 question."}), 400
    
    results = ppline(query, top_k)
    return jsonify(results), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)


"""
curl -X POST "http://127.0.0.1:5050/search" -H "Content-Type: application/json; charset=utf-8" -d '{
  "query": ["杭州", "飞机"],
  "top_k": 5
}'
"""
