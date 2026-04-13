"""
Adaptive Query Reformulator
===========================
Wraps any retriever with a confidence-feedback loop.
If top-1 score < threshold, reformulates the query and re-retrieves,
keeping the best result across iterations.
"""

import re
from typing import List, Dict, Any


STOPWORDS = {
    "and", "or", "of", "the", "in", "for", "with", "not",
    "other", "than", "their", "such", "as", "to", "a", "an",
}


class AdaptiveQueryReformulator:
    """
    Confidence-driven iterative query reformulation.

    Strategies applied in order when confidence is low:
      1. Expand: append top-result tokens not in query
      2. Contract: keep only content-bearing tokens (drop stopwords / short)
      3. Head: use only the most specific (last) content token
    """

    def __init__(
        self,
        retriever,
        confidence_threshold: float = 0.55,
        max_iters: int = 3,
    ):
        self.retriever = retriever
        self.threshold = confidence_threshold
        self.max_iters = max_iters

    def retrieve_with_feedback(
        self, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        strategies = [
            self._expand,
            self._contract,
            self._head_only,
        ]

        best_results: List[Dict] = []
        best_score: float = -1.0
        trace = []

        current_query = query
        for iteration in range(self.max_iters + 1):
            results = self.retriever.retrieve(current_query, top_k=top_k)
            top_score = results[0]["score"] if results else 0.0

            trace.append({
                "iteration": iteration,
                "query": current_query,
                "top_score": top_score,
            })

            if top_score > best_score:
                best_score = top_score
                best_results = results

            if top_score >= self.threshold or iteration >= len(strategies):
                break

            # Apply next strategy
            top_desc = results[0].get("text", "") if results else ""
            current_query = strategies[iteration](current_query, top_desc)
            if not current_query:
                break

        return {
            "results": best_results,
            "reformulation_trace": trace,
            "final_confidence": best_score,
            "reformulated": len(trace) > 1,
        }

    def _expand(self, query: str, top_desc: str) -> str:
        """Add discriminative tokens from top result."""
        q_tokens = set(re.findall(r"[a-z]+", query.lower()))
        desc_tokens = re.findall(r"[a-z]+", top_desc.lower())
        new = [
            t for t in desc_tokens
            if t not in q_tokens and t not in STOPWORDS and len(t) > 3
        ]
        expansion = " ".join(new[:2])
        return f"{query} {expansion}".strip() if expansion else query

    def _contract(self, query: str, _top_desc: str) -> str:
        """Keep only content-bearing tokens."""
        tokens = re.findall(r"[a-z]+", query.lower())
        content = [t for t in tokens if t not in STOPWORDS and len(t) > 3]
        return " ".join(content[:5]) if content else query

    def _head_only(self, query: str, _top_desc: str) -> str:
        """Use only the most specific token."""
        tokens = re.findall(r"[a-z]+", query.lower())
        content = [t for t in tokens if t not in STOPWORDS and len(t) > 3]
        return content[-1] if content else query