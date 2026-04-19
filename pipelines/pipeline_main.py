import json
import re

from utils.parsing import ReceiptParser
from utils.llm_cleaner import GroqCleaner

from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever

from pipelines.retrieval import RetrievalPipeline
from pipelines.augmentation import ContextAugmenter
from pipelines.generation import HSCodeGenerator
from pipelines.llm_wrapper import LocalLLM

from novelty.retrievers.hierarchical_retriever import (
    HierarchicalRetriever, AdaptiveQueryReformulator
)


class ICCARAGPipeline:

    def __init__(self, h6_path, faiss_index_path, meta_path, alpha=0.25):

        # -----------------------
        # Preprocessing
        # -----------------------
        self.parser = ReceiptParser()
        self.cleaner = GroqCleaner()

        # -----------------------
        # Retrieval Components
        # -----------------------
        # FIX: instantiate sparse + vector so self.hybrid exists
        self.sparse = SparseRetriever(h6_path)
        self.vector = VectorRetriever(faiss_index_path, meta_path)
        self.hybrid = HybridRetriever(self.sparse, self.vector, alpha=alpha)

        self.hier_retriever = HierarchicalRetriever(
            faiss_index_path=faiss_index_path,
            meta_path=meta_path,
            enriched_index_path="indexing/vector_store/h6_enriched.faiss",
            alpha_h=0.25,
            alpha_e=0.40,
        )
        self.adaptive = AdaptiveQueryReformulator(
            self.hier_retriever,
            confidence_threshold=0.55,   # was 0.65 — too aggressive, caused over-reformulation
            max_iters=2,
        )

        self.retrieval_pipeline = RetrievalPipeline(self.hybrid)

        # -----------------------
        # RAG Components
        # -----------------------
        self.augmenter = ContextAugmenter(max_docs=6)
        self.llm = LocalLLM()
        self.generator = HSCodeGenerator(self.llm)

        # -----------------------
        # Load H6 descriptions
        # -----------------------
        with open(h6_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.h6_index = {
            item["id"]: item["text"]
            for item in raw["results"]
            if "id" in item and "text" in item
        }

    # --------------------------------------------------
    # Basic deterministic cleaning
    # --------------------------------------------------
    def basic_clean(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r"[^\w\s\-\.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # --------------------------------------------------
    # Main prediction method
    # --------------------------------------------------
    def predict(self, raw_text, top_k=5):

        parsed_lines = self.parser.parse(raw_text)
        results_per_product = []

        for line in parsed_lines:

            if self.parser.is_non_product(line):
                continue

            if len(line.split()) < 2:
                continue

            # 1. Basic Clean
            cleaned_basic = self.basic_clean(line)

            # 2. LLM Cleaning (semantic normalization)
            cleaned_line = self.cleaner.clean(cleaned_basic)

            # FIX: fall back to basic clean if LLM cleaner returns garbage
            # (single char, empty, or a prefix artifact like "v -")
            if not cleaned_line or len(cleaned_line.strip()) <= 2:
                cleaned_line = self._fallback_clean(cleaned_basic)

            if not cleaned_line:
                continue

            # 3. Hierarchical Retrieval with adaptive reformulation
            output = self.adaptive.retrieve_with_feedback(cleaned_line, top_k=top_k)
            retrieved_docs = output["results"]

            # 4. Hybrid retrieval as a parallel signal and fallback
            hybrid_docs = self.hybrid.search(cleaned_line, top_k=top_k)
            # Normalize hybrid doc keys to match hierarchical format
            for d in hybrid_docs:
                d.setdefault("hs_code", d.get("doc_id", ""))

            # 5. Merge: combine hierarchical + hybrid, deduplicate by doc_id
            merged = self._merge_results(retrieved_docs, hybrid_docs, top_k=top_k)

            # 6. Quality filter (FIX: keep top_k not just 3)
            filtered_docs = self._filter_low_quality_candidates(cleaned_line, merged, top_k=top_k)

            # 7. Fallback to sparse if retrieval quality is still low
            if not filtered_docs or filtered_docs[0].get("score", 0) < 0.03:
                sparse_docs = RetrievalPipeline(self.sparse).retrieve(cleaned_line, top_k=top_k)
                if sparse_docs:
                    retrieved_docs = [
                        {
                            "doc_id": d["doc_id"],
                            "hs_code": d["doc_id"],
                            "text": d.get("text", ""),
                            "score": d["score"],
                            "rank": i + 1,
                            "source": "sparse",
                        }
                        for i, d in enumerate(sparse_docs)
                    ]
                else:
                    retrieved_docs = filtered_docs or merged
            else:
                retrieved_docs = filtered_docs

            if not retrieved_docs:
                results_per_product.append({
                    "raw_line": line,
                    "cleaned_line": cleaned_line,
                    "prediction": None,
                    "confidence": 0.0,
                    "reasoning": "No retrieval candidates found.",
                    "retrieved_candidates": [],
                })
                continue

            # 8. Attach description text
            for doc in retrieved_docs:
                code = doc.get("doc_id") or doc.get("hs_code", "")
                doc["text"] = self.h6_index.get(code, doc.get("text", ""))

            # 9. Context Augmentation
            context = self.augmenter.build_context(retrieved_docs)

            # 10. Final LLM Decision
            generation_result = self.generator.generate(
                query=cleaned_line,
                augmented_context=context,
            )

            # Reject hallucinated HS codes not present in retrieved candidates
            candidate_codes = {
                doc.get("doc_id") or doc.get("hs_code", "")
                for doc in retrieved_docs
            }
            if (
                generation_result["prediction"]
                and generation_result["prediction"] not in candidate_codes
            ):
                generation_result["prediction"] = None
                generation_result["confidence"] = 0.0
                generation_result["reasoning"] = (
                    generation_result["reasoning"]
                    + "\n[NOTE] Predicted code not supported by retrieved candidates."
                )

            results_per_product.append({
                "raw_line": line,
                "cleaned_line": cleaned_line,
                "prediction": generation_result["prediction"],
                "confidence": generation_result["confidence"],
                "reasoning": generation_result["reasoning"],
                "retrieved_candidates": retrieved_docs,
            })

        return results_per_product

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _fallback_clean(self, text: str) -> str:
        """
        Lightweight rule-based fallback when LLM cleaner fails.
        Strips codes, noise tokens, keeps substantive words.
        """
        noise = {
            "gstin", "invoice", "total", "subtotal", "mrp", "price", "rate",
            "amount", "date", "cash", "card", "gst", "cgst", "sgst", "igst",
            "tax", "discount", "batch", "lot", "qty", "quantity",
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        kept = [
            t for t in tokens
            if len(t) > 2
            and not t.isdigit()
            and t not in noise
            and re.search(r"[a-z]", t)
        ]
        return " ".join(kept[:8])  # cap at 8 tokens for retrieval quality

    def _merge_results(self, primary: list, secondary: list, top_k: int = 5) -> list:
        """
        Merge two ranked lists, deduplicating by doc_id.
        Primary scores take precedence; secondary adds coverage.
        """
        seen = {}
        for doc in primary:
            code = doc.get("doc_id") or doc.get("hs_code", "")
            if code and code not in seen:
                seen[code] = dict(doc)
                seen[code]["doc_id"] = code

        for doc in secondary:
            code = doc.get("doc_id") or doc.get("hs_code", "")
            if not code or code in seen:
                continue
            seen[code] = dict(doc)
            seen[code]["doc_id"] = code

        merged = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        # Re-rank
        for i, d in enumerate(merged[:top_k], start=1):
            d["rank"] = i
        return merged[:top_k]

    def _filter_low_quality_candidates(
        self, query: str, retrieved_docs: list, top_k: int = 5
    ) -> list:
        """
        FIX: original version cut to 3 docs and was too aggressive.
        Now keeps up to top_k docs, only dropping zero-overlap candidates
        when the top score is meaningfully higher.
        """
        if not retrieved_docs:
            return []

        top_score = retrieved_docs[0].get("score", 0)

        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 2 and not token.isdigit()
        }

        filtered = []
        for doc in retrieved_docs:
            doc_tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", doc.get("text", "").lower())
                if len(token) > 2 and not token.isdigit()
            }
            has_overlap = bool(query_tokens & doc_tokens)
            score_gap = top_score - doc.get("score", 0)

            # Only drop a doc if it has zero overlap AND lags the leader by >0.05
            if query_tokens and not has_overlap and score_gap > 0.05:
                continue

            filtered.append(doc)

        return filtered[:top_k]