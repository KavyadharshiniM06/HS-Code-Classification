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

    def __init__(self, h6_path, faiss_index_path, meta_path, alpha=0.6):

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
        self.augmenter = ContextAugmenter(max_docs=5)
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
            retrieved_docs = self.retrieval_pipeline.retrieve(
                cleaned_line,
                top_k=top_k
            )
            print(f"DEBUG: Retrieved keys are: {retrieved_docs[0].keys()}")
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

            results_per_product.append({
                "raw_line": line,
                "cleaned_line": cleaned_line,
                "prediction": generation_result["prediction"],
                "confidence": generation_result["confidence"],
                "reasoning": generation_result["reasoning"],
                "retrieved_candidates": retrieved_docs
            })

        return results_per_product
