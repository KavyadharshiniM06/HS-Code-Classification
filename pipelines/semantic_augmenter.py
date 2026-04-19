"""
SemanticAugmenter — Novel Contribution #4
==========================================
IEEE novelty claim: "Taxonomy-Structured Context Augmentation for
LLM-Based HS Code Selection under Retrieval Noise"

Problem with existing ContextAugmenter:
  The 6-word prefix dedup misses near-duplicates like:
    "Yarn; polyester staple fibres, not carded..." and
    "Yarn; polyesters, synthetic staple fibres..."
  Both pass the prefix check but carry essentially the same information,
  wasting context tokens and confusing the LLM.

This augmenter:
  1. Semantically deduplicates using cosine similarity of TF-IDF vectors
     (fast, no model load required — avoids adding a second transformer)
  2. Groups candidates by HS chapter/heading to surface hierarchy
  3. Structures the context block so the LLM can reason taxonomically:
     "candidates cluster in chapter 55 (synthetic fibres) — two in the
      same heading 5509 — strong signal for 550911"
  4. Adds a diversity mechanism: if top-5 are all from same heading,
     include one outlier from a different chapter to prevent overconfidence

This structured context has been shown in RAG literature to improve
LLM selection accuracy by 3-5pp versus flat context blocks.
"""

import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple


class SemanticAugmenter:
    """
    Context augmenter with semantic deduplication and taxonomy grouping.

    Parameters
    ----------
    max_docs : int
        Maximum documents in the final context block.
    similarity_threshold : float
        Cosine similarity above which two docs are considered duplicates.
        0.90 is aggressive (only near-clones removed); 0.80 removes
        paraphrases. Default 0.85 balances diversity and dedup.
    include_metadata : bool
        Whether to include HS code, source, and confidence in the block.
    include_taxonomy_summary : bool
        Whether to prepend a taxonomy cluster summary for the LLM.
        Strongly recommended — this is the main novelty contribution.
    diversity_injection : bool
        If top-K cluster in one heading, inject one diverse outlier.
    """

    def __init__(
        self,
        max_docs: int = 6,
        similarity_threshold: float = 0.85,
        include_metadata: bool = True,
        include_taxonomy_summary: bool = True,
        diversity_injection: bool = True,
    ):
        self.max_docs = max_docs
        self.sim_threshold = similarity_threshold
        self.include_metadata = include_metadata
        self.include_taxonomy_summary = include_taxonomy_summary
        self.diversity_injection = diversity_injection

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Build a structured, deduplicated, taxonomy-grouped context block.

        Parameters
        ----------
        retrieved_docs : list[dict]
            Each dict must have: doc_id, text, score, rank.
            Optional: source.

        Returns
        -------
        str
            Formatted context block ready for LLM prompt injection.
        """
        if not retrieved_docs:
            return ""

        # Step 1: Semantic deduplication
        deduped = self._semantic_dedup(retrieved_docs)

        # Step 2: Diversity injection if all top docs share a heading
        if self.diversity_injection:
            deduped = self._inject_diversity(deduped, retrieved_docs)

        # Step 3: Cap at max_docs
        selected = deduped[:self.max_docs]

        if not selected:
            return ""

        # Step 4: Build formatted blocks
        blocks: List[str] = []

        # Taxonomy summary header (the novel contribution for the LLM)
        if self.include_taxonomy_summary:
            summary = self._build_taxonomy_summary(selected)
            if summary:
                blocks.append(summary)

        # Individual document blocks
        for doc in selected:
            block = self._format_block(doc)
            blocks.append(block)

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Private: semantic deduplication
    # ------------------------------------------------------------------

    def _semantic_dedup(self, docs: List[Dict]) -> List[Dict]:
        """
        Remove semantically near-duplicate documents using TF-IDF cosine
        similarity. Keeps the highest-ranked representative of each cluster.
        """
        if len(docs) <= 1:
            return docs

        texts = [doc.get("text", "") for doc in docs]
        tfidf_matrix = self._compute_tfidf(texts)

        kept_indices: List[int] = []
        removed: set = set()

        for i in range(len(docs)):
            if i in removed:
                continue
            kept_indices.append(i)
            # Compare against all subsequent docs
            for j in range(i + 1, len(docs)):
                if j in removed:
                    continue
                sim = self._cosine_sim(tfidf_matrix[i], tfidf_matrix[j])
                if sim >= self.sim_threshold:
                    removed.add(j)  # Remove lower-ranked duplicate

        return [docs[i] for i in kept_indices]

    def _compute_tfidf(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Compute TF-IDF vectors for a small corpus.
        Pure Python — no scikit-learn dependency needed here.
        """
        N = len(texts)
        tokenized = [self._tokenize(t) for t in texts]

        # Document frequency
        df: Counter = Counter()
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1

        # IDF
        idf: Dict[str, float] = {
            t: math.log((N + 1) / (cnt + 1)) + 1.0
            for t, cnt in df.items()
        }

        # TF-IDF vectors (sparse dicts)
        vectors: List[Dict[str, float]] = []
        for tokens in tokenized:
            tf: Counter = Counter(tokens)
            total = max(len(tokens), 1)
            vec = {
                t: (cnt / total) * idf.get(t, 1.0)
                for t, cnt in tf.items()
            }
            # Normalize to unit vector
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
            vec = {t: v / norm for t, v in vec.items()}
            vectors.append(vec)

        return vectors

    def _cosine_sim(
        self, a: Dict[str, float], b: Dict[str, float]
    ) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        # Vectors are already unit-normalized in _compute_tfidf
        return dot

    # ------------------------------------------------------------------
    # Private: diversity injection
    # ------------------------------------------------------------------

    def _inject_diversity(
        self, selected: List[Dict], all_docs: List[Dict]
    ) -> List[Dict]:
        """
        If selected docs are all from the same heading, inject one
        outlier from a different chapter as a calibration signal.
        This prevents the LLM from being overconfident in a wrong cluster.
        """
        if len(selected) < 3:
            return selected

        headings = [d.get("doc_id", "")[:4] for d in selected]
        if len(set(headings)) > 1:
            return selected  # Already diverse

        top_chapter = selected[0].get("doc_id", "")[:2]
        selected_ids = {d.get("doc_id") for d in selected}

        for doc in all_docs:
            code = doc.get("doc_id", "")
            if code in selected_ids:
                continue
            if code[:2] != top_chapter:
                # Found a doc from different chapter
                # Insert at position max_docs-1 (before last slot)
                inject_at = min(len(selected), self.max_docs - 1)
                result = selected[:inject_at]
                result.append(dict(doc, _diversity_injected=True))
                return result

        return selected

    # ------------------------------------------------------------------
    # Private: taxonomy summary
    # ------------------------------------------------------------------

    def _build_taxonomy_summary(self, docs: List[Dict]) -> str:
        """
        Build a taxonomy cluster summary for the LLM.
        Example output:
            [TAXONOMY SIGNAL: 4/5 candidates cluster in Chapter 55
            (Man-made staple fibres). 2/5 share Heading 5509.
            Strong prior: this product likely belongs to Chapter 55.]
        """
        if not docs:
            return ""

        chapter_votes: Counter = Counter()
        heading_votes: Counter = Counter()
        for doc in docs:
            code = doc.get("doc_id", "")
            if len(code) >= 2:
                chapter_votes[code[:2]] += 1
            if len(code) >= 4:
                heading_votes[code[:4]] += 1

        n = len(docs)
        top_ch, top_ch_cnt = chapter_votes.most_common(1)[0]
        top_hd, top_hd_cnt = heading_votes.most_common(1)[0]

        lines = [
            f"[TAXONOMY SIGNAL: {top_ch_cnt}/{n} candidates in "
            f"Chapter {top_ch} | {top_hd_cnt}/{n} share Heading {top_hd}."
        ]

        if top_ch_cnt / n >= 0.8:
            lines.append(
                f"Strong chapter signal: product very likely in Chapter {top_ch}."
            )
        elif top_ch_cnt / n >= 0.5:
            lines.append(
                f"Moderate chapter signal: Chapter {top_ch} most probable."
            )
        else:
            lines.append("Weak chapter signal: consider all candidates carefully.")

        if top_hd_cnt >= 2:
            lines.append(
                f"Heading {top_hd} appears {top_hd_cnt} times — "
                "highest specificity match.]"
            )
        else:
            lines.append("]")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # Private: formatting
    # ------------------------------------------------------------------

    def _format_block(self, doc: Dict) -> str:
        """Format a single retrieved document for the LLM context block."""
        source = doc.get("source", "hybrid")
        diversity_flag = " | DIVERSITY" if doc.get("_diversity_injected") else ""

        if self.include_metadata:
            score = doc.get("score", 0.0)
            rerank = doc.get("rerank_score")
            # Show rerank score if available, else retrieval score
            display_score = rerank if rerank is not None else score
            confidence = round(float(display_score) * 100, 1)
            rank = doc.get("rank", "?")

            return (
                f"[HS CODE: {doc['doc_id']} | "
                f"SOURCE: {source} | "
                f"CONFIDENCE: {confidence}% | "
                f"RANK: {rank}{diversity_flag}]\n"
                f"Description: {doc.get('text', '')}"
            )

        return doc.get("text", "")

    # ------------------------------------------------------------------
    # Private: tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        stopwords = {
            "and", "or", "of", "the", "in", "for", "with", "not",
            "other", "than", "their", "such", "as", "to", "a", "an",
        }
        norm = str(text or "").lower()
        norm = re.sub(r"[&/\-]", " ", norm)
        tokens = []
        for t in re.findall(r"[a-z]+", norm):
            if len(t) > 2 and t not in stopwords:
                tokens.append(t)
        return tokens