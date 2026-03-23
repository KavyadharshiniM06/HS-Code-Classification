from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(self, sparse_retriever, vector_retriever, alpha=0.5):
        self.sparse = sparse_retriever
        self.vector = vector_retriever
        self.alpha = alpha

    def search(self, query, top_k=5):

        candidate_k = max(top_k * 10,50)

        # 1. Sparse Retrieval
        sparse_hits = self.sparse.search(query, top_k=candidate_k)
        sparse_res = {d["doc_id"]: d["score"] for d in sparse_hits}

        
        # 2. Dense Retrieval
        vector_hits = self.vector.retrieve(query, top_k=candidate_k)

        vector_res = {}
        for hit in vector_hits:
            score = float(hit["score"])
            vector_res[hit["hs_code"]] = score

        
        # 3. Combine
        combined = {}
        all_keys = set(sparse_res) | set(vector_res)

        for k in all_keys:
            s = sparse_res.get(k, 0)
            v = vector_res.get(k, 0)
            combined[k] = self.alpha * s + (1 - self.alpha) * v

        # 4. Rank
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (code, score) in enumerate(ranked, start=1):
            results.append({
                "doc_id": code,
                "score": round(score, 4),
                "rank": rank,
                "source": "hybrid"
            })

        return results