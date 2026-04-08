import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrievers.sparse_retriever import SparseRetriever
# from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.keyword_retriever import KeywordRetriever
from evaluation.evaluation_retrieval import evaluate # Reuse your eval logic

# Paths
H6_PATH = "data/H6.json"
FAISS_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"

def run_ablation():
    # 1. Load data
    # Assuming you have a fixed evaluation set to keep results consistent
    eval_df = pd.read_csv("evaluation/evaluation_dataset.csv")
    dataset = eval_df.to_dict('records')
    
    sparse = SparseRetriever(H6_PATH)
    # vector = VectorRetriever(FAISS_PATH, META_PATH)
    print("Initializing KeywordRetriever...")
    keyword = KeywordRetriever(H6_PATH)
    print("KeywordRetriever initialized successfully.")
    
    ablation_results = []

    # Test 1: Pure BM25 (Alpha = 0)
    print("Running Ablation: Pure BM25...")
    res = evaluate(sparse, dataset)
    ablation_results.append({"Configuration": "Pure BM25 (α=0.0)", **res})

    # Test 2: Pure Vector (Alpha = 1)
    # print("Running Ablation: Pure Vector...")
    # res = evaluate(vector, dataset)
    # ablation_results.append({"Configuration": "Pure Vector (α=1.0)", **res})

    # Test 3: Keyword Matching Baseline
    print("Running Ablation: Keyword Matching...")
    res = evaluate(keyword, dataset)
    print(f"Keyword results: {res}")
    ablation_results.append({"Configuration": "Keyword Matching", **res})

    # Test 4: Hybrid Sweep (0.2, 0.5, 0.8)
    for a in [0.2, 0.5, 0.8]:
        print(f"Running Ablation: Hybrid (α={a})...")
        hybrid = HybridRetriever(sparse, vector, alpha=a)
        res = evaluate(hybrid, dataset)
        ablation_results.append({"Configuration": f"Hybrid (α={a})", **res})

    # 2. Save Results
    output_df = pd.DataFrame(ablation_results)
    output_df.to_csv("evaluation/ablation_results.csv", index=False)
    print("\n--- ABLATION RESULTS SAVED ---")
    print(output_df)

if __name__ == "__main__":
    run_ablation()