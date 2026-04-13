import json
import re


class KeywordRetriever:
    """
    Rule-based keyword matching retriever
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

        self.tokenized_data = {}
        for code, info in self.data.items():
            tokens = self._tokenize(info["description"])
            self.tokenized_data[code] = tokens

    def search(self, query: str, top_k: int = 5):
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = {}
        for code, doc_tokens in self.tokenized_data.items():
            score = self._calculate_score(query_tokens, doc_tokens)
            if score > 0:
                scores[code] = score

        if scores:
            max_score = max(scores.values())
            scores = {k: v / max_score for k, v in scores.items()}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (code, score) in enumerate(ranked[:top_k], start=1):
            results.append({
                "doc_id": code,
                "score": float(score),
                "text": self.data[code]["description"],
                "rank": rank,
                "source": "keyword"
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

    def _calculate_score(self, query_tokens, doc_tokens):
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        matches = len(query_set.intersection(doc_set))
        if query_set:
            return matches / len(query_set)
        return 0

    def _normalize(self, token):
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token