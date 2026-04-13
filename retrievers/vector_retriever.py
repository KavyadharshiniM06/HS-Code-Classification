import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import re


class VectorRetriever:
    def __init__(self, faiss_index_path, meta_path, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if len(self.meta) != self.index.ntotal:
            raise ValueError("FAISS index and meta file lengths do not match!")

    def retrieve(self, query, top_k=5):

        candidate_k = max(top_k * 20, 100)

        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_emb, candidate_k)

        # convert distance → similarity
        scores = self._normalize_dense_scores(-scores.astype("float32", copy=False))

        query_tokens = self._tokenize(query)
        head_token = self._head_token(query_tokens)
        query_bigrams = self._bigrams(query_tokens)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0 or idx >= len(self.meta):
                continue

            description = self.meta[idx]["description"]

            doc_tokens = self._tokenize(description)

            overlap = self._overlap_ratio(query_tokens, doc_tokens)

            bigram_overlap = self._bigram_overlap(
                query_bigrams,
                self._bigrams(doc_tokens)
            )

            head_match = 1.0 if head_token and head_token in doc_tokens else 0.0

            head_penalty = (
                0.12 if head_token and len(query_tokens) > 1 and not head_match else 0.0
            )

            adjusted_score = (
                float(score)
                + (0.30 * overlap)
                + (0.20 * bigram_overlap)
                + (0.18 * head_match)
                - head_penalty
            )

            results.append({
                "hs_code": self.meta[idx]["hs_code"],
                "description": description,
                "score": adjusted_score,
                "dense_score": float(score),
                "overlap_score": round(overlap, 4),
                "bigram_score": round(bigram_overlap, 4),
                "head_match": int(head_match),
                "source": "vector"
            })

        results.sort(key=lambda item: item["score"], reverse=True)

        return results[:top_k]

    def search(self, query, top_k=5):
        results = self.retrieve(query, top_k=top_k)

        return [
            {
                "doc_id": item["hs_code"],
                "text": item["description"],
                "score": item["score"],
                "source": item["source"],
            }
            for item in results
        ]

    def _normalize_dense_scores(self, scores):
        if scores.size == 0:
            return scores

        normalized = np.zeros_like(scores, dtype=np.float32)
        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            return normalized

        finite_scores = scores[finite_mask]
        max_abs_score = float(np.max(np.abs(finite_scores)))

        if max_abs_score <= 1e-8:
            normalized[finite_mask] = finite_scores
            return normalized

        normalized[finite_mask] = finite_scores / max_abs_score
        return normalized

    def _tokenize(self, text: str):
        tokens = []

        for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):

            if len(token) <= 1 or token.isdigit():
                continue

            token = self._normalize_token(token)

            if token and token not in self._query_stopwords():
                tokens.append(token)

        return tokens

    def _overlap_ratio(self, query_tokens, doc_tokens):
        if not query_tokens or not doc_tokens:
            return 0.0

        return len(set(query_tokens) & set(doc_tokens)) / len(set(query_tokens))

    def _bigrams(self, tokens):
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    def _bigram_overlap(self, query_bigrams, doc_bigrams):
        if not query_bigrams or not doc_bigrams:
            return 0.0

        return len(query_bigrams & doc_bigrams) / len(query_bigrams)

    def _head_token(self, tokens):
        return tokens[-1] if tokens else ""

    def _normalize_token(self, token: str):

        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"

        if token.endswith("es") and len(token) > 4:
            return token[:-2]

        if token.endswith("s") and len(token) > 3:
            return token[:-1]

        return token

    def _query_stopwords(self):
        return {
            "for",
            "with",
            "and",
            "the",
            "other",
            "not",
            "than",
            "type",
            "set",
            "piece",
            "pc",
            "pcs",
            "unit",
        }
