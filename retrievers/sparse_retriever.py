import json
from rank_bm25 import BM25Okapi

class SparseRetriever:
    """
    Sparse keyword-based retriever using BM25
    """

    def __init__(self, h6_path: str):
        with open(h6_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.data = {}
        for item in raw["results"]:
            if "id" in item and "text" in item:
                self.data[item["id"]] = {
                    "description": item["text"],
                    "isLeaf": item.get("isLeaf", "0")
                }

        self.keys = list(self.data.keys())
        self.docs = [self.data[k]["description"].lower().split() for k in self.keys]
        self.bm25 = BM25Okapi(self.docs)

    def search(self, query: str, top_k: int = 5):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.keys, scores), key=lambda x: x[1], reverse=True)
        results = []
        for code, score in ranked[:top_k]:
            results.append({
                "doc_id": code,
                "score": float(score),
                "source": "sparse"
            })
        return results
