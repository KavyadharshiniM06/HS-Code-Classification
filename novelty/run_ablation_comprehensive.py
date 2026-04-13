"""
Master Ablation Runner
======================
Runs the full ablation study comparing:
  - Baseline BM25
  - Baseline Vector
  - Baseline Hybrid (existing)
  - + Ontology Enrichment (novel)
  - + Hierarchical Retrieval (novel)
  - + Adaptive Query Reformulation (novel)
  - Full System (all contributions combined)

Generates:
  1. evaluation/ablation_comprehensive.csv  ← main paper table
  2. evaluation/ablation_comprehensive.json ← full report
  3. evaluation/tde_analysis.json           ← taxonomy error analysis
  4. Console output ready to copy into LaTeX table

Run:
  python run_ablation_comprehensive.py
"""

import sys
import os
import json
import csv
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever
from novelty.evaluation.comprehensive_evaluator import (
    ComprehensiveEvaluator,
    analyze_tde_distribution,
)
from retrievers.hierarchical_retriever import (
    HierarchicalRetriever,
    AdaptiveQueryReformulator,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
H6_PATH = "data/H6.json"
FAISS_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"
ENRICHED_FAISS_PATH = "indexing/vector_store/h6_enriched.faiss"
EVAL_CSV = "evaluation/evaluation_dataset.csv"
OUTPUT_DIR = "evaluation"

TOP_K = 5


def load_dataset(csv_path: str):
    dataset = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append({
                "query": row["query"],
                "true_hs_code": row["true_hs_code"],
            })
    return dataset


class WrappedHybrid:
    """Wrapper to make HybridRetriever compatible with ComprehensiveEvaluator."""
    def __init__(self, sparse, vector, alpha):
        self.hybrid = HybridRetriever(sparse, vector, alpha=alpha)

    def retrieve(self, query, top_k=5):
        results = self.hybrid.search(query, top_k=top_k)
        return [{"doc_id": r["doc_id"], "score": r["score"]} for r in results]


class WrappedAdaptive:
    """Wrapper for AdaptiveQueryReformulator."""
    def __init__(self, base_retriever):
        self.reformulator = AdaptiveQueryReformulator(
            base_retriever, confidence_threshold=0.65, max_iters=2
        )

    def retrieve(self, query, top_k=5):
        output = self.reformulator.retrieve_with_feedback(query, top_k=top_k)
        return output["results"]


def main():
    print("📂 Loading evaluation dataset...")
    dataset = load_dataset(EVAL_CSV)
    print(f"   → {len(dataset)} samples")

    print("\n⚙️  Initializing retrievers...")
    sparse = SparseRetriever(H6_PATH)
    vector = VectorRetriever(FAISS_PATH, META_PATH)

    # Check if enriched index exists
    enriched_available = os.path.exists(ENRICHED_FAISS_PATH)
    if not enriched_available:
        print("   ⚠️  Enriched FAISS index not found.")
        print("       Run: python novelty/indexing/build_enriched_index.py first")
        print("       Falling back to base index for enriched retriever.")

    # Build retriever configurations
    retrievers = {}

    # Baseline 1: Pure BM25
    retrievers["BM25 (baseline)"] = sparse

    # Baseline 2: Pure Vector
    retrievers["Vector (baseline)"] = vector

    # Baseline 3: Hybrid α=0.6 (existing best)
    retrievers["Hybrid α=0.6 (existing)"] = WrappedHybrid(sparse, vector, alpha=0.6)

    # Novel 1: Hierarchical Retrieval (no enrichment)
    hier_base = HierarchicalRetriever(
        faiss_index_path=FAISS_PATH,
        meta_path=META_PATH,
        enriched_index_path=None,
        alpha_h=0.25,
        alpha_e=0.0,
    )
    retrievers["Hierarchical (no enrichment)"] = hier_base

    # Novel 2: Ontology-Enriched Hierarchical
    hier_enriched = HierarchicalRetriever(
        faiss_index_path=FAISS_PATH,
        meta_path=META_PATH,
        enriched_index_path=ENRICHED_FAISS_PATH if enriched_available else None,
        alpha_h=0.25,
        alpha_e=0.40,
    )
    retrievers["Hierarchical + Ontology Enrichment"] = hier_enriched

    # Novel 3: Full system (Hierarchical + Enrichment + Adaptive Reformulation)
    full_system = WrappedAdaptive(hier_enriched)
    retrievers["Full System (all contributions)"] = full_system

    # ── Run ablation ──
    evaluator = ComprehensiveEvaluator()
    all_results = evaluator.run_ablation_study(retrievers, dataset, top_k=TOP_K)

    # ── Save results ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV (for paper table)
    csv_path = os.path.join(OUTPUT_DIR, "ablation_comprehensive.csv")
    evaluator.save_csv_report(all_results, csv_path)

    # JSON (full report)
    json_path = os.path.join(OUTPUT_DIR, "ablation_comprehensive.json")
    clean = [{k: v for k, v in r.items() if k != "_per_sample"} for r in all_results]
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"✅ Full JSON report → {json_path}")

    # TDE analysis for best system
    best = all_results[-1]
    if "_per_sample" in best:
        tde_dist = analyze_tde_distribution(best["_per_sample"])
        tde_path = os.path.join(OUTPUT_DIR, "tde_analysis.json")
        with open(tde_path, "w") as f:
            json.dump(tde_dist, f, indent=2)
        print(f"\n📊 TDE Distribution (Full System):")
        for label, pct in tde_dist.items():
            bar = "█" * int(pct * 40)
            print(f"  {label:35s} {pct:.3f}  {bar}")

    # ── LaTeX table output ──
    print("\n" + "=" * 65)
    print("  LaTeX Table (copy into your IEEE paper):")
    print("=" * 65)
    _print_latex_table(all_results, TOP_K)


def _print_latex_table(all_results, k):
    """Print a LaTeX-ready table for the IEEE paper."""
    def latex_bold(value, enabled=False):
        text = str(value)
        return rf"\textbf{{{text}}}" if enabled else text

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation Study: Retrieval Performance on HS Code Dataset}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(f"Configuration & Acc@1 & Acc@{k} & MRR & NDCG@{k} & WHAcc & MTD \\\\")
    print(r"\hline")
    for r in all_results:
        name = r["configuration"]
        # Bold the best system
        bold = r == all_results[-1]
        accuracy_top1 = f"{r['accuracy_top1']:.3f}"
        row = (
            f"{latex_bold(name, bold)} & "
            f"{latex_bold(accuracy_top1, bold)} & "
            f"{r['accuracy_topk']:.3f} & "
            f"{r['mrr']:.3f} & "
            f"{r[f'ndcg_at_{k}']:.3f} & "
            f"{r['weighted_hs_accuracy']:.3f} & "
            f"{r['mean_taxonomy_distance']:.3f} \\\\"
        )
        print(row)
    print(r"\hline")
    print(r"\multicolumn{7}{l}{\footnotesize WHAcc = Weighted HS Accuracy; MTD = Mean Taxonomy Distance (lower is better)}")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
