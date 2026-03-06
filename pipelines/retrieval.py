class RetrievalPipeline:
    def __init__(self, retriever):
        """
        retriever: HybridRetriever | SparseRetriever | VectorRetriever
        """
        self.retriever = retriever

    def retrieve(self, query, top_k=5):
        """
        Runs retrieval and standardizes output for RAG augmentation
        """
        raw_results = self.retriever.search(query, top_k=top_k)

        formatted_results = []
        for rank, r in enumerate(raw_results, start=1):
            formatted_results.append({
                "rank": rank,
                "doc_id": r["doc_id"],
                "text": r.get("text", ""),
                "score": round(r["score"], 4),
                "retrieval_type": r["source"]
            })

        return formatted_results
