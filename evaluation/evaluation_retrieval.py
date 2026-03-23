from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever

import sys
import os
import re
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

H6_PATH = "data/H6.json"
FAISS_INDEX_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"
DATASET_PATH = "evaluation/evaluation_dataset.csv"

def extract_hs_code(text):
    match = re.search(r"\d{6}", str(text))
    return match.group(0) if match else str(text).strip()

def load_dataset(csv_path):
    dataset = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset.append({
                "query": row["query"],
                "true_hs_code": row["true_hs_code"]
            })

    return dataset

def evaluate(retriever, dataset, top_k=5):

    recall_at_1 = 0
    recall_at_k = 0
    mrr_total = 0

    for sample in dataset:

        query = sample["query"]

        true_code = extract_hs_code(sample["true_hs_code"])

        if hasattr(retriever, "search"):

            results = retriever.search(query, top_k=top_k)

            ranked_codes = []

            for item in results:
                code = item[0] if isinstance(item, (list, tuple)) else item
                ranked_codes.append(extract_hs_code(code))
        elif hasattr(retriever, "retrieve"):

            results = retriever.retrieve(query, top_k=top_k)
            ranked_codes = [extract_hs_code(r["hs_code"]) for r in results]

        else:
            raise ValueError("Retriever does not have search or retrieve method")

        if ranked_codes and ranked_codes[0] == true_code:
            recall_at_1 += 1

        if true_code in ranked_codes:
            recall_at_k += 1
            rank = ranked_codes.index(true_code) + 1
            mrr_total += 1.0 / rank

    n = len(dataset)

    return {
        "Recall@1": recall_at_1 / n,
        f"Recall@{top_k}": recall_at_k / n,
        "MRR": mrr_total / n
    }


if __name__ == "__main__":

    dataset=load_dataset(DATASET_PATH)

    sparse = SparseRetriever(H6_PATH)
    vector = VectorRetriever(FAISS_INDEX_PATH, META_PATH)
    hybrid = HybridRetriever(sparse, vector, alpha=0.6)

    print("Evaluating BM25...")
    sparse_results = evaluate(sparse, dataset)

    print("Evaluating Vector...")
    vector_results = evaluate(vector, dataset)

    print("Evaluating Hybrid...")
    hybrid_results = evaluate(hybrid, dataset)

    print("\nRESULTS:")
    print("BM25:", sparse_results)
    print("Vector:", vector_results)
    print("Hybrid:", hybrid_results)