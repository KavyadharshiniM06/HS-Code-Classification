import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class VectorRetriever:
    def __init__(self, faiss_index_path, meta_path, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)

        # Load metadata for mapping index -> HS code
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if len(self.meta) != self.index.ntotal:
            raise ValueError("FAISS index and meta file lengths do not match!")

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append({
                "hs_code": self.meta[idx]["hs_code"],
                "description": self.meta[idx]["description"],
                "score": float(score),
                "source": "vector"
            })
        return results
