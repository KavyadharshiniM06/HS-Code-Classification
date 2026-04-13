import json
import re
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
            if (
                "id" in item
                and "text" in item
                and item.get("isLeaf") == "1"
                and re.fullmatch(r"\d{6}", str(item["id"]))
            ):
                self.data[item["id"]] = {
                    "description": item["text"],
                    "isLeaf": item.get("isLeaf", "0")
                }

        self.keys = list(self.data.keys())
        self.docs = [self._tokenize(self.data[k]["description"]) for k in self.keys]
        self.bm25 = BM25Okapi(self.docs)

    def search(self, query: str, top_k: int = 5):
        tokens = self._tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        max_score = max(scores) if scores.any() else 1
        scores = [s / max_score for s in scores]

        ranked = sorted(zip(self.keys, scores), key=lambda x: x[1], reverse=True)

        results = []
        for code, score in ranked[:top_k]:
            results.append({
                "doc_id": code,
                "score": float(score),
                "text": self.data[code]["description"],
                "rank": len(results) + 1,
                "source": "sparse"
            })

        return results

    def _tokenize(self, text: str):
        normalized = str(text or "").lower()
        normalized = normalized.replace("&", " ")
        normalized = re.sub(r"[^a-z0-9\s\-/\.]", " ", normalized)
        normalized = normalized.replace("/", " ").replace("-", " ")

        tokens = []
        for token in re.findall(r"[a-z0-9\.]+", normalized):
            token = token.strip(".")
            if len(token) <= 1:
                continue
            if token.isdigit():
                continue
            tokens.append(token)

        return tokens