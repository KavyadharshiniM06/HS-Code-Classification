import json
import re
import math
from collections import Counter


class KeywordRetriever:
    """
    Rule-based keyword matching retriever with TF-IDF-style scoring.

    Improvements over original:
    - IDF weighting: rare terms score higher than common ones
    - Length normalization: short queries aren't penalized unfairly
    - Partial stemming retained from original
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
                    "isLeaf": item.get("isLeaf", "0"),
                }

        # Pre-tokenize all descriptions
        self.tokenized_data = {}
        for code, info in self.data.items():
            tokens = self._tokenize(info["description"])
            self.tokenized_data[code] = tokens

        # Build IDF weights
        self._idf = self._build_idf()

    def _build_idf(self) -> dict:
        """
        Compute inverse document frequency for each term across the corpus.
        IDF = log((N + 1) / (df + 1)) + 1  (smoothed)
        """
        N = len(self.tokenized_data)
        df: Counter = Counter()
        for tokens in self.tokenized_data.values():
            for t in set(tokens):
                df[t] += 1

        idf = {}
        for term, count in df.items():
            idf[term] = math.log((N + 1) / (count + 1)) + 1.0
        return idf

    def search(self, query: str, top_k: int = 5):
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        query_set = set(query_tokens)
        scores = {}

        for code, doc_tokens in self.tokenized_data.items():
            doc_set = set(doc_tokens)
            matched = query_set.intersection(doc_set)

            if not matched:
                continue

            # IDF-weighted intersection score, normalized by query length
            idf_score = sum(self._idf.get(t, 1.0) for t in matched)
            max_idf = sum(self._idf.get(t, 1.0) for t in query_set)

            if max_idf > 0:
                scores[code] = idf_score / max_idf
            else:
                scores[code] = 0.0

        # Normalize to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (code, score) in enumerate(ranked[:top_k], start=1):
            results.append({
                "doc_id": code,
                "score": float(score),
                "text": self.data[code]["description"],
                "rank": rank,
                "source": "keyword",
            })

        return results

    def _tokenize(self, text: str):
        normalized = str(text or "").lower()
        normalized = normalized.replace("&", " ")
        normalized = normalized.replace("/", " ")
        normalized = normalized.replace("-", " ")

        tokens = []
        for token in re.findall(r"[a-z0-9]+", normalized):
            if len(token) <= 1:
                continue
            if token.isdigit():
                continue
            token = self._normalize(token)
            tokens.append(token)

        return tokens

    def _normalize(self, token):
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token