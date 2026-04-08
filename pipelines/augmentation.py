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

    def _remove_duplicates(self, docs):
        seen = set()
        unique = []

        for doc in docs:
            text = doc["text"].lower()

            # simple duplicate check
            key = " ".join(text.split()[:6])

            if key not in seen:
                unique.append(doc)
                seen.add(key)

        return unique

    def build_context(self, retrieved_docs):
        """
        retrieved_docs: output from retrieval.py
        returns: augmented context string
        """

        filtered = [
                d for d in retrieved_docs
                if d["score"] >= self.min_score
            ]

        filtered = self._remove_duplicates(filtered)

        filtered = filtered[:self.max_docs]

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
        source = doc.get("source", "hybrid")
        if self.include_metadata:
            confidence = round(doc["score"] * 100, 2)

            return (
                f"[HS CODE: {doc['doc_id']} | "
                f"SOURCE: {source} | "
                f"CONFIDENCE: {confidence}% | "
                f"RANK: {doc['rank']}]\n"
                f"Description: {doc['text']}"
            )

        return doc["text"]
