"""
Enhanced ablation evaluation
=============================
Compares the new EnhancedHybridRetriever against the original baselines.
Outputs a CSV compatible with your existing evaluation/ablation_results.csv.

Run:
  python evaluation/run_enhanced_ablation.py
"""

import sys
import os
import csv
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
from evaluation.evaluation_retrieval import evaluate, load_dataset

H6_PATH = "data/H6.json"
FAISS_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"
ENRICHED_FAISS_PATH = "indexing/vector_store/h6_enriched.faiss"
EVAL_CSV = "evaluation/evaluation_dataset.csv"
OUTPUT_CSV = "evaluation/enhanced_ablation_results.csv"


def run():
    print("📂 Loading evaluation dataset...")
    dataset = load_dataset(EVAL_CSV)
    print(f"   → {len(dataset)} samples")

    results = []

    print("\n⚙️  Initializing retrievers...")
    sparse = SparseRetriever(H6_PATH)
    vector = VectorRetriever(FAISS_PATH, META_PATH)

    enriched_path = ENRICHED_FAISS_PATH if os.path.exists(ENRICHED_FAISS_PATH) else None

    enhanced = EnhancedHybridRetriever(
        faiss_index_path=FAISS_PATH,
        meta_path=META_PATH,
        h6_path=H6_PATH,
        enriched_index_path=enriched_path,
    )

    configs = [
        ("BM25 (baseline)", sparse),
        ("Vector (baseline)", vector),
        ("Enhanced Hybrid (new)", enhanced),
    ]

    for name, retriever in configs:
        print(f"\n{'='*55}")
        print(f"  Evaluating: {name}")
        t0 = time.time()
        metrics = evaluate(retriever, dataset, top_k=5)
        elapsed = round(time.time() - t0, 1)
        metrics["Configuration"] = name
        metrics["Elapsed_s"] = elapsed
        results.append(metrics)

        print(f"  Recall@1  : {metrics['Recall@1']:.3f}")
        print(f"  Recall@5  : {metrics['Recall@5']:.3f}")
        print(f"  MRR       : {metrics['MRR']:.3f}")
        print(f"  Time      : {elapsed}s")

    # Save CSV
    os.makedirs("evaluation", exist_ok=True)
    fieldnames = ["Configuration", "Recall@1", "Recall@5", "MRR",
                  "Precision@1", "Precision@5", "Elapsed_s"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved → {OUTPUT_CSV}")
    print("\n  Summary:")
    print(f"  {'Config':<30} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    print("  " + "-" * 50)
    for r in results:
        print(
            f"  {r['Configuration']:<30} "
            f"{r['Recall@1']:>6.3f} "
            f"{r['Recall@5']:>6.3f} "
            f"{r['MRR']:>6.3f}"
        )


if __name__ == "__main__":
    run()