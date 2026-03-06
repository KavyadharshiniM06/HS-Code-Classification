from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(self, sparse_retriever, vector_retriever, alpha=0.4):
        self.sparse = sparse_retriever
        self.vector = vector_retriever
        self.alpha = alpha

    def search(self, query, top_k=5):

        # -------------------------
        # 1. Sparse Retrieval
        # -------------------------
        sparse_hits = self.sparse.search(query, top_k=top_k)
        sparse_res = {d["doc_id"]: d["score"] for d in sparse_hits}

        if sparse_res:
            max_sparse = max(sparse_res.values())
            if max_sparse > 0:
                sparse_res = {
                    k: v / max_sparse for k, v in sparse_res.items()
                }

        # -------------------------
        # 2. Dense Retrieval
        # -------------------------
        vector_hits = self.vector.retrieve(query, top_k=top_k)

        # Convert cosine safely (if already 0–1, this still works fine)
        vector_res = {}
        for hit in vector_hits:
            raw_score = float(hit["score"])
            normalized = (raw_score + 1) / 2   # handles [-1,1]
            vector_res[hit["hs_code"]] = normalized

        # -------------------------
        # 3. Combine Scores
        # -------------------------
        combined = {}
        all_keys = set(sparse_res) | set(vector_res)

        for k in all_keys:
            s = sparse_res.get(k, 0.0)
            v = vector_res.get(k, 0.0)
            combined[k] = self.alpha * s + (1 - self.alpha) * v

        # -------------------------
        # 4. Rank and Format
        # -------------------------
        ranked = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for rank, (code, score) in enumerate(ranked, start=1):
            results.append({
                "doc_id": code,
                "score": round(score, 4),
                "rank": rank,
                "source": "hybrid"
            })

        return results