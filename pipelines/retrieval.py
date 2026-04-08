class RetrievalPipeline:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, query, top_k=5):

        raw_results = self.retriever.search(query, top_k=top_k)

        scores = [r["score"] for r in raw_results]
        max_score = max(scores) if scores else 1.0

        formatted_results = []

        for rank, r in enumerate(raw_results, start=1):

            normalized_score = r["score"] / max_score

            confidence = (
                "high" if normalized_score > 0.7
                else "medium" if normalized_score > 0.4
                else "low"
            )

            formatted_results.append({
                "rank": rank,
                "doc_id": r["doc_id"],
                "text": r.get("text", ""),
                "score": round(normalized_score, 4),
                "confidence": confidence,
                "retrieval_type": r.get("source", "unknown")
            })

        return formatted_results