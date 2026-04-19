"""
DualPathRetriever — Novel Contribution #2 (revised)
====================================================
IEEE novelty claim: "Confidence-Gated Dual-Path Retrieval with
Keyword Signal Injection for Noisy HS Code Classification"

Problem with existing EnhancedHybridRetriever:
  The ablation shows pure Vector (0.794 Acc@1) beats all hybrid
  configurations. Classic BM25+vector RRF *hurts* because BM25 drags
  in unrelated codes that match noise words in OCR'd descriptions.

This design fixes that with three principles:
  1. VECTOR-PRIMARY: BGE-M3 FAISS is the sole ranking authority.
  2. KEYWORD SIGNAL INJECTION (not RRF fusion): KeywordRetriever score
     is used as a SOFT BOOST on candidates already in the vector set,
     not as a separate retrieval path that adds noise candidates.
  3. BM25 RESCUE: activated ONLY when vector top-1 score < threshold,
     as a fallback — not as a primary signal.

This matches the design used in industrial IR (Elasticsearch, Azure
Cognitive Search) where dense retrieval is primary and lexical signals
post-process rather than pollute the candidate set.
"""

import re
import json
import faiss
import numpy as np
import math
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

import logging
logger = logging.getLogger(__name__)


class DualPathRetriever:
    """
    Vector-primary retriever with keyword signal injection.

    Architecture (in order of execution):
      1. Multi-variant vector retrieval → top-K*8 candidates
      2. Keyword score injection (soft boost on existing candidates)
      3. Ontology hierarchy bonus (chapter/heading voting)
      4. BM25 rescue pass (only if top score < confidence_floor)
      5. Final sort and output

    Parameters
    ----------
    faiss_index_path : str
        Path to base FAISS index (h6.faiss).
    meta_path : str
        Path to metadata JSON (h6_meta.json).
    h6_path : str
        Path to H6.json (for BM25 and keyword retriever).
    enriched_index_path : str, optional
        Path to ontology-enriched FAISS index.
    model_name : str
        Sentence transformer for dense encoding.
    confidence_floor : float
        Vector top-1 score below which BM25 rescue activates.
    keyword_boost : float
        Max boost from keyword signal (fraction of score range).
    enriched_weight : float
        Blend weight for enriched index scores.
    """

    def __init__(
        self,
        faiss_index_path: str,
        meta_path: str,
        h6_path: str,
        enriched_index_path: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        confidence_floor: float = 0.30,
        keyword_boost: float = 0.12,
        enriched_weight: float = 0.35,
        hierarchy_bonus: float = 0.08,
    ):
        self.confidence_floor = confidence_floor
        self.keyword_boost = keyword_boost
        self.enriched_weight = enriched_weight
        self.hierarchy_bonus = hierarchy_bonus

        # Dense model and primary index
        self.model = SentenceTransformer(model_name)
        self.base_index = faiss.read_index(faiss_index_path)

        # Enriched index (optional)
        self.enriched_index: Optional[faiss.Index] = None
        if enriched_index_path:
            try:
                self.enriched_index = faiss.read_index(enriched_index_path)
                logger.info("Enriched index loaded.")
            except Exception as e:
                logger.warning(f"Enriched index not found: {e}")

        # Metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if len(self.meta) != self.base_index.ntotal:
            raise ValueError(
                f"FAISS index size ({self.base_index.ntotal}) != "
                f"meta size ({len(self.meta)})"
            )

        # Build supporting structures
        self._build_bm25(h6_path)
        self._build_keyword_idf(h6_path)
        self._build_token_table()
        self._build_hierarchy_index()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Main retrieval method. Returns list of ranked dicts with
        doc_id, text, score, rank, source keys.
        """
        candidate_k = max(top_k * 25, 200)
        query_variants = self._build_query_variants(query)

        # ── Step 1: Dense vector retrieval (multi-variant) ────────────
        candidate_scores: Dict[int, float] = {}
        for weight, variant in query_variants:
            scores, indices = self._faiss_search(self.base_index, variant, candidate_k)
            self._merge_scores(candidate_scores, scores, indices, weight=weight)

        # ── Step 2: Enriched index fusion ─────────────────────────────
        if self.enriched_index is not None:
            for weight, variant in query_variants[:3]:
                scores, indices = self._faiss_search(
                    self.enriched_index, variant, candidate_k
                )
                self._merge_scores(
                    candidate_scores, scores, indices,
                    weight=weight * self.enriched_weight
                )

        # Filter to valid 6-digit codes
        candidate_scores = {
            idx: score for idx, score in candidate_scores.items()
            if 0 <= idx < len(self.meta)
            and re.fullmatch(r"\d{6}", str(self.meta[idx].get("hs_code", "")))
        }

        if not candidate_scores:
            return []

        # Normalize scores to [0, 1]
        max_s = max(candidate_scores.values())
        if max_s > 0:
            candidate_scores = {k: v / max_s for k, v in candidate_scores.items()}

        top_score = max(candidate_scores.values())

        # ── Step 3: Keyword signal injection ──────────────────────────
        # Only boost candidates already in the vector set — don't add new ones
        query_tokens = self._tokenize_set(query)
        query_bigrams = self._bigrams(list(query_tokens))

        for idx in candidate_scores:
            code = self.meta[idx].get("hs_code", "")
            doc_tokens = self.token_table.get(code, set())
            overlap = self._overlap(query_tokens, doc_tokens)
            bigram_ov = self._bigram_overlap(
                query_bigrams, self._bigrams(list(doc_tokens))
            )
            idf_score = self._idf_overlap_score(query_tokens, doc_tokens)
            # Combined keyword signal: weighted average of overlap signals
            kw_signal = (
                0.40 * overlap
                + 0.30 * bigram_ov
                + 0.30 * idf_score
            )
            candidate_scores[idx] += self.keyword_boost * kw_signal

        # ── Step 4: Hierarchy-aware bonus ─────────────────────────────
        top_chapter, top_heading = self._infer_hierarchy(
            candidate_scores, top_n=15
        )
        for idx in candidate_scores:
            code = self.meta[idx].get("hs_code", "")
            bonus = 0.0
            if top_chapter and code[:2] == top_chapter:
                bonus += self.hierarchy_bonus * 0.5
            if top_heading and code[:4] == top_heading:
                bonus += self.hierarchy_bonus * 0.5
            candidate_scores[idx] += bonus

        # ── Step 5: BM25 rescue (only when vector confidence is low) ──
        if top_score < self.confidence_floor:
            rescue = self._bm25_rescue(query, top_k=100)
            for code, bm25_score in rescue.items():
                # Only inject if code is NOT already in candidate set
                # to avoid polluting the vector ranking
                already_in = any(
                    self.meta[i].get("hs_code") == code
                    for i in candidate_scores
                )
                if not already_in:
                    # Find meta index for this code
                    for idx, item in enumerate(self.meta):
                        if item.get("hs_code") == code:
                            # Inject with reduced weight so rescue
                            # candidates rank below existing ones
                            candidate_scores[idx] = 0.25 * bm25_score
                            break

        # ── Step 6: Sort and format ───────────────────────────────────
        ranked = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        results = []
        seen_codes: set = set()
        for idx, score in ranked:
            code = self.meta[idx].get("hs_code", "")
            if code in seen_codes:
                continue
            seen_codes.add(code)
            results.append({
                "doc_id": code,
                "hs_code": code,
                "text": self.meta[idx].get("description", ""),
                "description": self.meta[idx].get("description", ""),
                "score": round(float(score), 4),
                "rank": len(results) + 1,
                "source": "dual_path",
                "chapter": code[:2],
                "heading": code[:4],
            })
            if len(results) >= top_k:
                break

        return results

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Alias for compatibility with existing pipeline interfaces."""
        return self.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_bm25(self, h6_path: str):
        """Build BM25 index for rescue pass."""
        from rank_bm25 import BM25Okapi
        with open(h6_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.bm25_codes: List[str] = []
        docs: List[List[str]] = []
        for item in raw["results"]:
            if (
                "id" in item and "text" in item
                and item.get("isLeaf") == "1"
                and re.fullmatch(r"\d{6}", str(item["id"]))
            ):
                self.bm25_codes.append(item["id"])
                docs.append(self._tokenize(item["text"]))
        self.bm25 = BM25Okapi(docs)

    def _build_keyword_idf(self, h6_path: str):
        """Build IDF weights for keyword signal injection."""
        with open(h6_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        N = 0
        df: Counter = Counter()
        for item in raw["results"]:
            if item.get("isLeaf") == "1" and re.fullmatch(r"\d{6}", str(item.get("id", ""))):
                tokens = set(self._tokenize(item.get("text", "")))
                for t in tokens:
                    df[t] += 1
                N += 1
        self._idf: Dict[str, float] = {
            t: math.log((N + 1) / (cnt + 1)) + 1.0
            for t, cnt in df.items()
        }

    def _idf_overlap_score(self, query_tokens: set, doc_tokens: set) -> float:
        """IDF-weighted token overlap (normalised by query IDF mass)."""
        if not query_tokens or not doc_tokens:
            return 0.0
        matched = query_tokens & doc_tokens
        idf_matched = sum(self._idf.get(t, 1.0) for t in matched)
        idf_query = sum(self._idf.get(t, 1.0) for t in query_tokens)
        return idf_matched / idf_query if idf_query > 0 else 0.0

    def _build_token_table(self):
        """Pre-compute token sets for fast keyword signal calculation."""
        self.token_table: Dict[str, set] = {}
        for item in self.meta:
            code = item.get("hs_code", "")
            self.token_table[code] = self._tokenize_set(
                item.get("description", "")
            )

    def _build_hierarchy_index(self):
        """Chapter → [meta indices] and heading → [meta indices] maps."""
        self.chapter_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.heading_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, item in enumerate(self.meta):
            code = item.get("hs_code", "")
            if len(code) >= 2:
                self.chapter_to_indices[code[:2]].append(i)
            if len(code) >= 4:
                self.heading_to_indices[code[:4]].append(i)

    def _faiss_search(
        self, index: faiss.Index, query: str, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode and search a FAISS index, returning normalised scores."""
        emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        k = min(k, index.ntotal)
        distances, indices = index.search(emb, k)
        # HNSW: distances are L2; convert to cosine similarity ≈ 1 - d/2
        scores = np.clip(1.0 - distances[0] / 2.0, 0.0, 1.0)
        mx = scores.max() if scores.max() > 0 else 1.0
        return scores / mx, indices[0]

    def _merge_scores(
        self,
        combined: Dict[int, float],
        scores: np.ndarray,
        indices: np.ndarray,
        weight: float,
        rrf_k: int = 60,
    ):
        """RRF-style score accumulation (rank-weighted, not raw-score merge)."""
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            idx = int(idx)
            if idx < 0:
                continue
            rrf = weight / (rrf_k + rank)
            combined[idx] = combined.get(idx, 0.0) + rrf

    def _infer_hierarchy(
        self, candidate_scores: Dict[int, float], top_n: int = 15
    ) -> Tuple[Optional[str], Optional[str]]:
        """Vote on most likely chapter/heading from top-N candidates."""
        top = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        chapter_v: Counter = Counter()
        heading_v: Counter = Counter()
        for idx, score in top:
            code = self.meta[idx].get("hs_code", "")
            if len(code) >= 2:
                chapter_v[code[:2]] += score
            if len(code) >= 4:
                heading_v[code[:4]] += score
        ch = chapter_v.most_common(1)[0][0] if chapter_v else None
        hd = heading_v.most_common(1)[0][0] if heading_v else None
        return ch, hd

    def _bm25_rescue(self, query: str, top_k: int = 50) -> Dict[str, float]:
        """BM25 rescue pass — only called when vector confidence is low."""
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        mx = float(scores.max()) if scores.any() else 1.0
        if mx <= 0:
            return {}
        normed = scores / mx
        ranked = sorted(
            zip(self.bm25_codes, normed), key=lambda x: x[1], reverse=True
        )[:top_k]
        return {code: float(s) for code, s in ranked if s > 0}

    def _build_query_variants(
        self, query: str
    ) -> List[Tuple[float, str]]:
        """
        Generate query variants with weights.
        Full query gets highest weight; partial queries get lower weight.
        Weighting rather than flat RRF ensures full query dominates.
        """
        base = " ".join(re.findall(r"[a-z0-9]+", str(query or "").lower()))
        if not base:
            return []
        tokens = base.split()
        stopwords = {
            "and", "or", "of", "the", "in", "for", "with", "not",
            "other", "than", "their", "such", "as", "to", "a", "an",
        }
        content = [t for t in tokens if t not in stopwords and len(t) > 2]
        variants: List[Tuple[float, str]] = [(1.0, base)]

        if content and " ".join(content) != base:
            variants.append((0.70, " ".join(content)))
        if len(tokens) >= 2:
            variants.append((0.50, " ".join(tokens[-2:])))
        if content:
            variants.append((0.40, content[-1]))  # head token
        for t in tokens:
            if len(t) > 5 and t not in (v for _, v in variants):
                variants.append((0.35, t))
                break

        # Deduplicate
        seen: set = set()
        unique: List[Tuple[float, str]] = []
        for w, v in variants:
            v = v.strip()
            if v and v not in seen:
                unique.append((w, v))
                seen.add(v)
        return unique

    def _tokenize(self, text: str) -> List[str]:
        norm = re.sub(r"[&/\-]", " ", str(text or "").lower())
        stopwords = {
            "and", "or", "of", "the", "in", "for", "with", "not",
            "other", "than", "their", "as", "to", "a", "an",
        }
        tokens = []
        for t in re.findall(r"[a-z0-9]+", norm):
            if len(t) <= 1 or t.isdigit() or t in stopwords:
                continue
            tokens.append(self._stem(t))
        return tokens

    def _tokenize_set(self, text: str) -> set:
        return set(self._tokenize(text))

    def _stem(self, t: str) -> str:
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 4:
            return t[:-2]
        if t.endswith("s") and len(t) > 3:
            return t[:-1]
        return t

    def _bigrams(self, tokens: list) -> set:
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    def _overlap(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a)

    def _bigram_overlap(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a)