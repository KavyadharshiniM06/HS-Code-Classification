from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.keyword_retriever import KeywordRetriever
import re


class HybridRetriever:
    def __init__(self, sparse_retriever, vector_retriever, alpha=0.5,keyword=None):
        self.sparse = sparse_retriever
        self.vector = vector_retriever
        self.keyword = keyword
        self.alpha = alpha

    def search(self, query, top_k=5):

        candidate_k = max(top_k * 15, 75)
        query_variants = self._build_query_variants(query)

        sparse_runs = []
        vector_runs = []
        keyword_runs = []
        doc_text = {}
        doc_scores = {}

        for variant in query_variants:

            # Sparse Retrieval
            sparse_hits = self.sparse.search(variant, top_k=candidate_k)
            sparse_runs.append(sparse_hits)

            for hit in sparse_hits:
                doc_text[hit["doc_id"]] = hit.get("text", "")
                doc_scores[hit["doc_id"]] = max(
                    doc_scores.get(hit["doc_id"], 0),
                    hit.get("score", 0)
                )

            # Vector Retrieval
            vector_hits = self.vector.retrieve(variant, top_k=candidate_k)
            vector_runs.append(vector_hits)

            for hit in vector_hits:
                doc_text[hit["hs_code"]] = hit.get("description", "")
                doc_scores[hit["hs_code"]] = max(
                    doc_scores.get(hit["hs_code"], 0),
                    hit.get("score", 0)
                )
            
            # Keyword Retrieval
            if self.keyword:
                keyword_hits = self.keyword.search(variant, top_k=candidate_k)
                keyword_runs.append(keyword_hits)

                for hit in keyword_hits:
                    doc_text[hit["doc_id"]] = hit.get("text", "")
                    doc_scores[hit["doc_id"]] = max(
                        doc_scores.get(hit["doc_id"], 0),
                        hit.get("score", 0)
                    )

        # RRF Fusion
        combined = {}
        self._accumulate_rrf(combined, sparse_runs, "doc_id", weight=self.alpha)
        self._accumulate_rrf(combined, vector_runs, "hs_code", weight=1 - self.alpha)
        if self.keyword:
            self._accumulate_rrf(combined, keyword_runs, "doc_id", weight=0.3)

        # Token Overlap Boost
        query_tokens = self._tokenize(query)
        for code in list(combined.keys()):

            if not re.fullmatch(r"\d{6}", str(code)):
                combined.pop(code, None)
                continue

            overlap = self._overlap_ratio(
                query_tokens,
                self._tokenize(doc_text.get(code, ""))
            )

            # improved boosting
            combined[code] += 0.15 * overlap

            # confidence boost using retrieval score
            combined[code] += 0.05 * doc_scores.get(code, 0)

        # Sort results
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (code, score) in enumerate(ranked, start=1):
            results.append({
                "doc_id": code,
                "text": doc_text.get(code, ""),
                "score": round(score, 4),
                "rank": rank,
                "source": "hybrid"
            })

        return results

    def _accumulate_rrf(self, combined, runs, key_name, weight, k=60):
        for run in runs:
            for rank, item in enumerate(run, start=1):
                code = item[key_name]
                combined[code] = combined.get(code, 0.0) + (weight / (k + rank))

    def _build_query_variants(self, query: str):
        base = " ".join(re.findall(r"[a-z0-9]+", str(query or "").lower()))
        if not base:
            return []

        tokens = base.split()
        variants = [base]

        # first two words
        if len(tokens) >= 2:
            variants.append(" ".join(tokens[:2]))
            variants.append(" ".join(tokens[-2:]))

        # remove short tokens
        if len(tokens) >= 3:
            variants.append(" ".join(token for token in tokens if len(token) > 2))

        # single important tokens
        for token in tokens:
            if len(token) > 4:
                variants.append(token)

        unique = []
        seen = set()

        for variant in variants:
            variant = variant.strip()
            if variant and variant not in seen:
                unique.append(variant)
                seen.add(variant)

        return unique

    def _tokenize(self, text: str):
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(token) > 1 and not token.isdigit()
        }

    def _overlap_ratio(self, query_tokens, doc_tokens):
        if not query_tokens or not doc_tokens:
            return 0.0
        return len(query_tokens & doc_tokens) / len(query_tokens)