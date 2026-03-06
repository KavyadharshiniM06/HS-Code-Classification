class ContextAugmenter:
    def __init__(
        self,
        max_docs=5,
        min_score=0.0,
        include_metadata=True
    ):
        self.max_docs = max_docs
        self.min_score = min_score
        self.include_metadata = include_metadata

    def build_context(self, retrieved_docs):
        """
        retrieved_docs: output from retrieval.py
        returns: augmented context string
        """

        filtered = [
            d for d in retrieved_docs
            if d["score"] >= self.min_score
        ][:self.max_docs]

        if not filtered:
            return ""

        context_blocks = []
        for doc in filtered:
            block = self._format_block(doc)
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def _format_block(self, doc):
        """
        Internal formatter for each retrieved document
        """

        if self.include_metadata:
            return (
                f"[HS CODE: {doc['doc_id']} | "
                f"RANK: {doc['rank']} | "
                f"SCORE: {doc['score']}]\n"
                f"{doc['text']}"
            )

        return doc["text"]
