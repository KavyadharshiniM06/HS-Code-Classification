"""
Enhanced ICCA-RAG Pipeline
===========================
Replaces ICCARAGPipeline with a vector-first strategy.

Key improvements over the original:
  - Uses EnhancedHybridRetriever (vector-first, BM25 rescue)
  - AdaptiveQueryReformulator with 3-strategy loop
  - Better candidate filtering (token overlap + score gap)
  - Richer context augmentation block
  - Groq LLM decision with hallucination guard
"""

import json
import re
from typing import List, Dict, Optional

from utils.parsing import ReceiptParser
from utils.llm_cleaner import GroqCleaner
from retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
from retrievers.adaptive_reformulator import AdaptiveQueryReformulator
from pipelines.augmentation import ContextAugmenter
from pipelines.generation import HSCodeGenerator
from pipelines.llm_wrapper import LocalLLM


class EnhancedICCARAGPipeline:
    """
    Drop-in replacement for ICCARAGPipeline.

    Parameters
    ----------
    h6_path : str
        Path to H6.json HS code taxonomy.
    faiss_index_path : str
        Path to base FAISS index (h6.faiss).
    meta_path : str
        Path to metadata JSON (h6_meta.json).
    enriched_index_path : str, optional
        Path to ontology-enriched FAISS index (h6_enriched.faiss).
        If not provided or not found, falls back to base index only.
    confidence_threshold : float
        Minimum top-1 retrieval score to skip reformulation loop.
    max_reformulation_iters : int
        Maximum adaptive reformulation iterations.
    """

    def __init__(
        self,
        h6_path: str,
        faiss_index_path: str,
        meta_path: str,
        enriched_index_path: Optional[str] = None,
        confidence_threshold: float = 0.55,
        max_reformulation_iters: int = 3,
    ):
        # Preprocessing
        self.parser = ReceiptParser()
        self.cleaner = GroqCleaner()

        # Core retriever
        self.retriever = EnhancedHybridRetriever(
            faiss_index_path=faiss_index_path,
            meta_path=meta_path,
            h6_path=h6_path,
            enriched_index_path=enriched_index_path,
            confidence_floor=0.35,
            bm25_rescue_weight=0.15,
            enriched_weight=0.30,
            overlap_boost=0.20,
            bigram_boost=0.12,
        )

        # Adaptive reformulation wrapper
        self.adaptive = AdaptiveQueryReformulator(
            retriever=self.retriever,
            confidence_threshold=confidence_threshold,
            max_iters=max_reformulation_iters,
        )

        # RAG components
        self.augmenter = ContextAugmenter(
            max_docs=6,
            min_score=0.0,
            include_metadata=True,
        )
        self.llm = LocalLLM()
        self.generator = HSCodeGenerator(self.llm)

        # HS code description lookup
        with open(h6_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.h6_index: Dict[str, str] = {
            item["id"]: item["text"]
            for item in raw["results"]
            if "id" in item and "text" in item
        }

    # ------------------------------------------------------------------
    def predict(self, raw_text: str, top_k: int = 5) -> List[Dict]:
        """
        Predict HS codes for all product lines in raw_text.

        Returns a list of dicts, one per detected product line:
          raw_line, cleaned_line, prediction, confidence,
          reasoning, retrieved_candidates
        """
        parsed_lines = self.parser.parse(raw_text)
        results = []

        for line in parsed_lines:
            if self.parser.is_non_product(line):
                continue
            if len(line.split()) < 2:
                continue

            result = self._process_line(line, top_k=top_k)
            if result is not None:
                results.append(result)

        return results

    # ------------------------------------------------------------------
    def _process_line(self, line: str, top_k: int) -> Optional[Dict]:
        # 1. LLM normalization
        cleaned = self.cleaner.clean(line)
        if not cleaned:
            return None

        # 2. Adaptive retrieval
        output = self.adaptive.retrieve_with_feedback(cleaned, top_k=top_k)
        retrieved_docs = output["results"]

        # 3. Quality filter
        filtered = self._quality_filter(cleaned, retrieved_docs)
        if not filtered:
            filtered = retrieved_docs  # fall back to unfiltered

        # 4. Attach description text
        for doc in filtered:
            doc["text"] = self.h6_index.get(doc["doc_id"], doc.get("text", ""))

        # 5. Augment context
        context = self.augmenter.build_context(filtered)

        # 6. LLM generation
        gen = self.generator.generate(query=cleaned, augmented_context=context)

        # 7. Hallucination guard
        valid_codes = {d["doc_id"] for d in filtered}
        if gen["prediction"] and gen["prediction"] not in valid_codes:
            # Fall back to top retrieved code
            gen["prediction"] = filtered[0]["doc_id"] if filtered else None
            gen["confidence"] = filtered[0]["score"] if filtered else 0.0
            gen["reasoning"] += (
                "\n[NOTE] LLM prediction not in candidates; "
                "substituted top retrieval result."
            )

        return {
            "raw_line": line,
            "cleaned_line": cleaned,
            "prediction": gen["prediction"],
            "confidence": gen["confidence"],
            "reasoning": gen["reasoning"],
            "retrieved_candidates": filtered,
            "reformulated": output.get("reformulated", False),
            "reformulation_trace": output.get("reformulation_trace", []),
        }

    # ------------------------------------------------------------------
    def _quality_filter(
        self, query: str, docs: List[Dict], max_docs: int = 5
    ) -> List[Dict]:
        """
        Remove low-quality candidates:
          - Score gap > 0.25 below top-1
          - Zero token overlap with query AND below gap threshold
        """
        if not docs:
            return []

        query_tokens = {
            t for t in re.findall(r"[a-z0-9]+", query.lower())
            if len(t) > 2 and not t.isdigit()
        }
        top_score = docs[0]["score"]
        gap_threshold = max(top_score - 0.25, 0.0)

        filtered = []
        for doc in docs:
            doc_tokens = {
                t for t in re.findall(
                    r"[a-z0-9]+", doc.get("text", "").lower()
                )
                if len(t) > 2 and not t.isdigit()
            }
            has_overlap = bool(query_tokens & doc_tokens)
            above_gap = doc["score"] >= gap_threshold

            if has_overlap or above_gap:
                filtered.append(doc)

        return filtered[:max_docs] if filtered else docs[:max_docs]

    # ------------------------------------------------------------------
    def predict_single(self, query: str, top_k: int = 5) -> Dict:
        """
        Predict HS code for a single cleaned product description.
        Bypasses receipt parsing — useful for direct API calls.
        """
        output = self.adaptive.retrieve_with_feedback(query, top_k=top_k)
        retrieved_docs = output["results"]
        filtered = self._quality_filter(query, retrieved_docs)
        if not filtered:
            filtered = retrieved_docs

        for doc in filtered:
            doc["text"] = self.h6_index.get(doc["doc_id"], doc.get("text", ""))

        context = self.augmenter.build_context(filtered)
        gen = self.generator.generate(query=query, augmented_context=context)

        valid_codes = {d["doc_id"] for d in filtered}
        if gen["prediction"] and gen["prediction"] not in valid_codes:
            gen["prediction"] = filtered[0]["doc_id"] if filtered else None
            gen["confidence"] = filtered[0]["score"] if filtered else 0.0

        return {
            "query": query,
            "prediction": gen["prediction"],
            "confidence": gen["confidence"],
            "reasoning": gen["reasoning"],
            "retrieved_candidates": filtered,
        }