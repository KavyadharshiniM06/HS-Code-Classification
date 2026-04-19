"""
Enhanced Hybrid Retriever
=========================
Key insight from ablation results:
  - Vector (α=1.0): Recall@1=0.728, MRR=0.804  ← best overall
  - BM25 (α=0.0):   Recall@1=0.710, MRR=0.756
  - Any hybrid:     Recall@1=0.720-0.724, MRR=0.730-0.732  ← underperforms

Strategy: Use vector as primary, BM25 as a re-ranking signal only when
vector confidence is low. Also fuse enriched index when available.

FIX (OOM): Accept a pre-loaded SentenceTransformer via the `model` parameter
so all retriever configurations in an ablation study share a single model
instance (~1 GB) instead of each loading their own copy.
"""

import re
import json
import faiss
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional, Tuple


class EnhancedHybridRetriever:
    """
    Vector-first hybrid retriever.

    Architecture:
      1. Multi-variant query expansion (base, bigrams, head, shortened)
      2. Parallel search: primary vector + enriched vector (if available)
      3. RRF fusion across all query variants and indexes
      4. Token-overlap re-ranking signal
      5. BM25 rescue pass when top vector score < confidence_floor

    Parameters
    ----------
    model : SentenceTransformer, optional
        A pre-loaded model to reuse across multiple retriever instances.
        Pass this to avoid loading ~1 GB into RAM multiple times during
        ablation studies. If None, a new model is loaded from `model_name`.
    """

    def __init__(
        self,
        faiss_index_path: str,
        meta_path: str,
        h6_path: str,
        enriched_index_path: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        model: Optional[SentenceTransformer] = None,   # ← shared model
        confidence_floor: float = 0.35,
        bm25_rescue_weight: float = 0.15,
        enriched_weight: float = 0.30,
        overlap_boost: float = 0.20,
        bigram_boost: float = 0.12,
    ):
        self.confidence_floor = confidence_floor
        self.bm25_rescue_weight = bm25_rescue_weight
        self.enriched_weight = enriched_weight
        self.overlap_boost = overlap_boost
        self.bigram_boost = bigram_boost

        # Reuse shared model if provided, otherwise load once from disk
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name)

        # Load primary FAISS index
        self.base_index = faiss.read_index(faiss_index_path)

        # Load enriched index (optional)
        self.enriched_index = None
        if enriched_index_path:
            try:
                self.enriched_index = faiss.read_index(enriched_index_path)
                print(f"✅ Enriched index loaded: {enriched_index_path}")
            except Exception as e:
                print(f"⚠️  Enriched index not found, using base only: {e}")

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if len(self.meta) != self.base_index.ntotal:
            raise ValueError(
                f"FAISS index ({self.base_index.ntotal}) and meta ({len(self.meta)}) mismatch"
            )

        # Build BM25 index for rescue pass
        self._build_bm25(h6_path)

        # Pre-build token lookup table for fast overlap scoring
        self._build_token_table()

    def _build_bm25(self, h6_path: str):
        with open(h6_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.bm25_keys = []
        bm25_docs = []

        for item in raw["results"]:
            if (
                "id" in item
                and "text" in item
                and item.get("isLeaf") == "1"
                and re.fullmatch(r"\d{6}", str(item["id"]))
            ):
                self.bm25_keys.append(item["id"])
                bm25_docs.append(self._tokenize(item["text"]))

        self.bm25 = BM25Okapi(bm25_docs)

        # Map hs_code → bm25 index for cross-lookup
        self.bm25_code_to_idx = {code: i for i, code in enumerate(self.bm25_keys)}
        self.bm25_idx_to_code = {i: code for i, code in enumerate(self.bm25_keys)}

        # Store descriptions for token overlap
        self.bm25_descriptions = {}
        for item in raw["results"]:
            if "id" in item and "text" in item:
                self.bm25_descriptions[item["id"]] = item["text"]

    def _build_token_table(self):
        """Pre-compute token sets for fast overlap scoring."""
        self.token_table: Dict[str, set] = {}
        for i, item in enumerate(self.meta):
            code = item.get("hs_code", "")
            desc = item.get("description", "")
            self.token_table[code] = self._tokenize_set(desc)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        candidate_k = max(top_k * 25, 150)
        query_variants = self._build_query_variants(query)

        if not query_variants:
            return []

        # Step 1: Vector retrieval across all variants
        combined_scores: Dict[int, float] = {}
        doc_text: Dict[str, str] = {}

        for variant in query_variants:
            scores, idxs = self._faiss_search(self.base_index, variant, candidate_k)
            self._accumulate_rrf(combined_scores, scores, idxs, weight=1.0)

        # Step 2: Enriched index fusion (if available)
        if self.enriched_index is not None:
            for variant in query_variants[:3]:  # top 3 variants only for speed
                scores, idxs = self._faiss_search(
                    self.enriched_index, variant, candidate_k
                )
                self._accumulate_rrf(
                    combined_scores, scores, idxs, weight=self.enriched_weight
                )

        # Collect doc metadata
        for idx in combined_scores:
            if 0 <= idx < len(self.meta):
                code = self.meta[idx].get("hs_code", "")
                doc_text[idx] = self.meta[idx].get("description", "")
                _ = code  # ensure mapping exists

        # Step 3: Token overlap re-ranking boost
        query_tokens = self._tokenize_set(query)
        query_bigrams = self._bigrams(list(query_tokens))

        for idx in list(combined_scores.keys()):
            if idx < 0 or idx >= len(self.meta):
                combined_scores.pop(idx, None)
                continue
            code = self.meta[idx].get("hs_code", "")
            if not re.fullmatch(r"\d{6}", str(code)):
                combined_scores.pop(idx, None)
                continue
            doc_tokens = self.token_table.get(code, set())
            overlap = self._overlap(query_tokens, doc_tokens)
            bigram_ov = self._bigram_overlap(
                query_bigrams, self._bigrams(list(doc_tokens))
            )
            combined_scores[idx] += self.overlap_boost * overlap
            combined_scores[idx] += self.bigram_boost * bigram_ov

        # Sort initial ranking
        ranked = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k * 3]

        # Step 4: BM25 rescue pass if top confidence is low
        top_score = ranked[0][1] if ranked else 0.0
        if top_score < self.confidence_floor:
            bm25_codes = self._bm25_rescue(query, top_k=candidate_k)
            for code, score in bm25_codes.items():
                for idx, item in enumerate(self.meta):
                    if item.get("hs_code") == code:
                        if idx in combined_scores:
                            combined_scores[idx] += self.bm25_rescue_weight * score
                        else:
                            combined_scores[idx] = self.bm25_rescue_weight * score
                        break

            ranked = sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_k * 3]

        # Step 5: Format results
        results = []
        seen_codes = set()
        for idx, score in ranked:
            if idx < 0 or idx >= len(self.meta):
                continue
            code = self.meta[idx].get("hs_code", "")
            if not re.fullmatch(r"\d{6}", str(code)):
                continue
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
                "source": "enhanced_hybrid",
            })
            if len(results) >= top_k:
                break

        return results

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Alias compatible with existing pipeline interfaces."""
        return self.retrieve(query, top_k=top_k)

    def _faiss_search(
        self, index, query: str, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode query and search a FAISS index."""
        emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        k = min(k, index.ntotal)
        distances, indices = index.search(emb, k)
        scores = 1.0 - (distances[0] / 2.0)
        mx = scores.max() if scores.max() > 0 else 1.0
        return scores / mx, indices[0]

    def _accumulate_rrf(
        self,
        combined: Dict[int, float],
        scores: np.ndarray,
        indices: np.ndarray,
        weight: float,
        rrf_k: int = 60,
    ):
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            idx = int(idx)
            if idx < 0:
                continue
            rrf_score = weight / (rrf_k + rank)
            combined[idx] = combined.get(idx, 0.0) + rrf_score

    def _bm25_rescue(self, query: str, top_k: int = 50) -> Dict[str, float]:
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        mx = max(scores) if scores.any() else 1.0
        if mx <= 0:
            return {}
        normalized = scores / mx
        ranked = sorted(
            zip(self.bm25_keys, normalized), key=lambda x: x[1], reverse=True
        )[:top_k]
        return {code: float(score) for code, score in ranked if score > 0}

    def _build_query_variants(self, query: str) -> List[str]:
        base = " ".join(re.findall(r"[a-z0-9]+", str(query or "").lower()))
        if not base:
            return []
        tokens = base.split()
        variants = [base]

        if len(tokens) >= 2:
            variants.append(" ".join(tokens[:2]))
            variants.append(" ".join(tokens[-2:]))
        stopwords = {"and", "or", "of", "the", "in", "for", "with", "not", "than"}
        content = [t for t in tokens if t not in stopwords and len(t) > 2]
        if content and " ".join(content) != base:
            variants.append(" ".join(content))
        if content:
            variants.append(content[-1])
        for t in tokens:
            if len(t) > 5:
                variants.append(t)
                break

        seen = set()
        unique = []
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                unique.append(v)
                seen.add(v)
        return unique

    def _tokenize(self, text: str) -> List[str]:
        normalized = str(text or "").lower()
        normalized = re.sub(r"[&/\-]", " ", normalized)
        tokens = []
        stopwords = {
            "and", "or", "of", "the", "in", "for", "with", "not",
            "other", "than", "their", "such", "as", "to", "a", "an",
            "its", "from", "into", "whether", "except",
        }
        for token in re.findall(r"[a-z0-9]+", normalized):
            if len(token) <= 1 or token.isdigit() or token in stopwords:
                continue
            token = self._stem(token)
            tokens.append(token)
        return tokens

    def _tokenize_set(self, text: str) -> set:
        return set(self._tokenize(text))

    def _stem(self, token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

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