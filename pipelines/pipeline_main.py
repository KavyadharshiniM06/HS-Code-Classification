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
        self.sparse = SparseRetriever(h6_path)
        self.vector = VectorRetriever(faiss_index_path, meta_path)
        self.hybrid = HybridRetriever(self.sparse, self.vector, alpha)
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

        # Remove special characters except useful separators
        text = re.sub(r"[^\w\s\-\.]", " ", text)

        # Collapse spaces
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

            # 1️⃣ Basic Clean
            cleaned_basic = self.basic_clean(line)

            # 2️⃣ LLM Cleaning (semantic normalization)
            cleaned_line = self.cleaner.clean(cleaned_basic)
            if not cleaned_line:
                continue

            # 3️⃣ Hybrid Retrieval
            retrieved_docs = self.retrieval_pipeline.retrieve(cleaned_line, top_k=top_k)
            filtered_docs = self._filter_low_quality_candidates(cleaned_line, retrieved_docs)
            # fallback to sparse if retrieval quality is low
            if not filtered_docs or filtered_docs[0]["score"] < 0.03:
                
                sparse_docs = RetrievalPipeline(self.sparse).retrieve(
                    cleaned_line,
                    top_k=top_k
                )

                if sparse_docs:
                    retrieved_docs = sparse_docs
                else:
                    retrieved_docs = filtered_docs or retrieved_docs
            if filtered_docs:
                retrieved_docs = filtered_docs
            elif retrieved_docs:
                # Keep raw retrieval results when filtering removes everything,
                # since a high-scoring match may still be useful for the LLM.
                pass
            else:
                results_per_product.append({
                    "raw_line": line,
                    "cleaned_line": cleaned_line,
                    "prediction": None,
                    "confidence": 0.0,
                    "reasoning": "No retrieval candidates found; showing raw line.",
                    "retrieved_candidates": []
                })
                continue
            # 4️⃣ Attach description text to retrieved docs
            for doc in retrieved_docs:
                code = doc["doc_id"]
                doc["text"] = self.h6_index.get(code, "")

            # 5️⃣ Context Augmentation
            context = self.augmenter.build_context(retrieved_docs)

            # 6️⃣ Final LLM Decision
            generation_result = self.generator.generate(
                query=cleaned_line,
                augmented_context=context
            )

            # Reject hallucinated HS codes that are not present in the retrieved candidates.
            candidate_codes = {doc["doc_id"] for doc in retrieved_docs}
            if generation_result["prediction"] and generation_result["prediction"] not in candidate_codes:
                generation_result["prediction"] = None
                generation_result["confidence"] = 0.0
                generation_result["reasoning"] = (
                    generation_result["reasoning"] +
                    "\n[NOTE] Predicted code was not supported by retrieved candidates."
                )

            results_per_product.append({
                "raw_line": line,
                "cleaned_line": cleaned_line,
                "prediction": generation_result["prediction"],
                "confidence": generation_result["confidence"],
                "reasoning": generation_result["reasoning"],
                "retrieved_candidates": retrieved_docs
            })

        return results_per_product

    def _filter_low_quality_candidates(self, query: str, retrieved_docs: list):
        if not retrieved_docs:
            return []

        top_score = retrieved_docs[0]["score"]
        

        query_tokens = {
            token for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 2 and not token.isdigit()
        }

        filtered = []
        for doc in retrieved_docs:
            doc_tokens = {
                token for token in re.findall(r"[a-z0-9]+", doc.get("text", "").lower())
                if len(token) > 2 and not token.isdigit()
            }
            if query_tokens and not (query_tokens & doc_tokens) and doc["score"] < top_score - 0.03:
                continue
            filtered.append(doc)

        return filtered[:3]
