"""
Cross-Encoder Reranker — Novel Contribution #3
===============================================
IEEE novelty claim: "Taxonomy-Aware Cross-Encoder Reranking for
Harmonized System Code Retrieval under Noisy Invoice Conditions"

Key idea: bi-encoder retrieval (FAISS) finds candidates fast but
compares query and document in SEPARATE embedding spaces. A cross-
encoder reads them JOINTLY — full attention between query and each
candidate — producing far more accurate relevance scores.

Expected gain: +5 to +8pp Acc@1 over bi-encoder alone (consistent
with MS-MARCO and BEIR benchmarks for cross-encoder post-processing).

Model choices (in order of speed/accuracy tradeoff):
  1. BAAI/bge-reranker-base     — 278M params, fast, good
  2. BAAI/bge-reranker-large    — 560M params, best accuracy
  3. cross-encoder/ms-marco-MiniLM-L-6-v2 — 22M params, fastest

Usage:
    reranker = CrossEncoderReranker()
    reranked  = reranker.rerank(query, candidates, top_k=5)
"""

import re
import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved HS code candidates using a cross-encoder model.

    The cross-encoder scores (query, candidate_description) pairs
    jointly, giving dramatically better ranking than cosine similarity
    alone. This is the standard two-stage retrieval paradigm used in
    production IR systems (Google, Bing, MS Azure Cognitive Search).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. BAAI/bge-reranker-base is recommended
        for the accuracy/latency tradeoff in production.
    batch_size : int
        Number of (query, doc) pairs to score per forward pass.
    max_length : int
        Max token length. HS descriptions are short; 256 is sufficient.
    taxonomy_weight : float
        Weight for taxonomy-distance bonus (0.0 = disabled).
        When > 0, codes in the predicted chapter get a small bonus,
        breaking ties in a semantically meaningful direction.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
        max_length: int = 256,
        taxonomy_weight: float = 0.05,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.taxonomy_weight = taxonomy_weight
        self._model = None  # lazy load

    def _load_model(self):
        """Lazy-load the cross-encoder to avoid startup overhead."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            logger.info(f"CrossEncoder loaded: {self.model_name}")
        except Exception as e:
            logger.warning(
                f"CrossEncoder load failed ({e}). "
                "Falling back to retrieval scores only."
            )
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        predicted_chapter: Optional[str] = None,
        predicted_heading: Optional[str] = None,
    ) -> List[Dict]:
        """
        Rerank candidates using cross-encoder scores.

        Parameters
        ----------
        query : str
            Cleaned product description.
        candidates : list[dict]
            Each dict must have 'doc_id', 'text', 'score'.
        top_k : int
            Number of results to return.
        predicted_chapter : str, optional
            2-digit chapter inferred from initial retrieval (for
            taxonomy-aware tie-breaking).
        predicted_heading : str, optional
            4-digit heading inferred from initial retrieval.

        Returns
        -------
        list[dict]
            Re-ordered candidates with 'rerank_score' added.
        """
        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            # Graceful degradation: return original order
            for i, d in enumerate(candidates[:top_k], 1):
                d["rerank_score"] = d.get("score", 0.0)
                d["rank"] = i
            return candidates[:top_k]

        # Build (query, description) pairs
        pairs = []
        for cand in candidates:
            text = cand.get("text") or cand.get("description", "")
            pairs.append([query, text])

        # Score in batches (avoids OOM on large candidate sets)
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i: i + self.batch_size]
            batch_scores = self._model.predict(batch, show_progress_bar=False)
            if hasattr(batch_scores, "tolist"):
                batch_scores = batch_scores.tolist()
            scores.extend(batch_scores)

        # Normalize cross-encoder scores to [0, 1]
        arr = np.array(scores, dtype=np.float32)
        # Cross-encoders output logits; sigmoid converts to probability
        arr = 1.0 / (1.0 + np.exp(-arr))
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())

        # Apply taxonomy-aware bonus
        for i, cand in enumerate(candidates):
            code = str(cand.get("doc_id", ""))
            bonus = 0.0
            if predicted_chapter and code[:2] == predicted_chapter:
                bonus += self.taxonomy_weight * 0.5
            if predicted_heading and code[:4] == predicted_heading:
                bonus += self.taxonomy_weight * 0.5
            arr[i] += bonus

        # Merge rerank score with original retrieval score (weighted avg)
        rerank_weight = 0.75  # cross-encoder dominates
        retrieval_weight = 0.25

        results = []
        for i, cand in enumerate(candidates):
            orig_score = float(cand.get("score", 0.0))
            rerank_score = float(arr[i])
            final_score = (
                rerank_weight * rerank_score
                + retrieval_weight * orig_score
            )
            new_cand = dict(cand)
            new_cand["rerank_score"] = round(rerank_score, 4)
            new_cand["retrieval_score"] = round(orig_score, 4)
            new_cand["score"] = round(final_score, 4)
            results.append(new_cand)

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, d in enumerate(results, 1):
            d["rank"] = i

        return results[:top_k]

    def infer_hierarchy(self, candidates: List[Dict]) -> tuple:
        """
        Vote on the most likely chapter and heading from top-N candidates.
        Used to compute taxonomy bonus before reranking.
        """
        from collections import Counter
        chapter_votes: Counter = Counter()
        heading_votes: Counter = Counter()
        for i, cand in enumerate(candidates[:10]):
            weight = 1.0 / (i + 1)  # rank-weighted voting
            code = str(cand.get("doc_id", ""))
            if len(code) >= 2:
                chapter_votes[code[:2]] += weight
            if len(code) >= 4:
                heading_votes[code[:4]] += weight
        top_chapter = chapter_votes.most_common(1)[0][0] if chapter_votes else None
        top_heading = heading_votes.most_common(1)[0][0] if heading_votes else None
        return top_chapter, top_heading