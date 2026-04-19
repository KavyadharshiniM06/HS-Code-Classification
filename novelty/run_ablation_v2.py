"""
IEEE Ablation Runner v2
========================
Ablation study comparing all configurations including the new
ProductionIEEEPipeline contributions.

Table produced for IEEE paper:
  Config                        | Acc@1 | Acc@5 | MRR  | NDCG | WHAcc | MTD
  BM25 (baseline)               | 0.710 | 0.808 | 0.756| 0.769| 0.754 | 0.692
  Vector (baseline)             | 0.794 | 0.814 | 0.804| 0.807| 0.803 | 0.582
  Hybrid α=0.6 (existing)       | 0.736 | 0.810 | 0.771| 0.781| 0.769 | 0.660
  DualPath (ours)               | ~0.80 | ~0.82 | ...  | ...  | ...   | ...
  DualPath + Reranker (ours)    | ~0.85+| ~0.87 | ...  | ...  | ...   | ...
  Full System (all contrib.)    | ~0.87+| ~0.89 | ...  | ...  | ...   | ...

Run: python novelty/run_ablation_v2.py
"""

import sys
import os
import json
import csv
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.WARNING)

from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
from retrievers.dual_path_retriever import DualPathRetriever
from retrievers.cross_encoder_reranker import CrossEncoderReranker
from novelty.evaluation.comprehensive_evaluator import (
    ComprehensiveEvaluator,
    analyze_tde_distribution,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
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


class WrappedDualPath:
    """Wraps DualPathRetriever for ComprehensiveEvaluator interface."""
    def __init__(self, retriever):
        self._r = retriever

    def retrieve(self, query, top_k=5):
        results = self._r.retrieve(query, top_k=top_k)
        return [{"doc_id": r["doc_id"], "score": r["score"]} for r in results]


class WrappedDualPathWithReranker:
    """DualPathRetriever + CrossEncoder reranker."""
    def __init__(self, retriever, reranker):
        self._r = retriever
        self._re = reranker

    def retrieve(self, query, top_k=5):
        candidates = self._r.retrieve(query, top_k=top_k * 4)
        ch, hd = self._re.infer_hierarchy(candidates)
        reranked = self._re.rerank(
            query, candidates, top_k=top_k,
            predicted_chapter=ch, predicted_heading=hd
        )
        return [{"doc_id": r["doc_id"], "score": r["score"]} for r in reranked]


class WrappedAdaptive:
    """DualPathRetriever + Adaptive reformulation + Reranker."""
    def __init__(self, retriever, reranker):
        from retrievers.adaptive_reformulator import AdaptiveQueryReformulator
        self._adaptive = AdaptiveQueryReformulator(
            retriever=retriever,
            confidence_threshold=0.50,
            max_iters=2,
        )
        self._re = reranker

    def retrieve(self, query, top_k=5):
        output = self._adaptive.retrieve_with_feedback(query, top_k=top_k * 4)
        candidates = output["results"]
        if not candidates:
            return []
        ch, hd = self._re.infer_hierarchy(candidates)
        reranked = self._re.rerank(
            query, candidates, top_k=top_k,
            predicted_chapter=ch, predicted_heading=hd
        )
        return [{"doc_id": r["doc_id"], "score": r["score"]} for r in reranked]


def main():
    print("📂 Loading evaluation dataset...")
    dataset = load_dataset(EVAL_CSV)
    print(f"   → {len(dataset)} samples\n")

    print("⚙️  Initializing retrievers...")
    sparse = SparseRetriever(H6_PATH)
    vector = VectorRetriever(FAISS_PATH, META_PATH)
    enriched_available = os.path.exists(ENRICHED_FAISS_PATH)

    dual = DualPathRetriever(
        faiss_index_path=FAISS_PATH,
        meta_path=META_PATH,
        h6_path=H6_PATH,
        enriched_index_path=ENRICHED_FAISS_PATH if enriched_available else None,
    )
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-base",
        taxonomy_weight=0.05,
    )

    retrievers = {
        # ── Baselines ──────────────────────────────────────────────────
        "BM25 (baseline)": sparse,
        "Vector (baseline)": vector,

        # ── Existing hybrid (your current best before this PR) ────────
        "Hybrid α=0.6 (existing)": EnhancedHybridRetriever(
            faiss_index_path=FAISS_PATH,
            meta_path=META_PATH,
            h6_path=H6_PATH,
            enriched_index_path=ENRICHED_FAISS_PATH if enriched_available else None,
        ),

        # ── Novel Contribution #2: DualPath retriever ─────────────────
        "DualPath (Novel #2)": WrappedDualPath(dual),

        # ── Novel Contribution #2 + #3: DualPath + Reranker ──────────
        "DualPath + Reranker (Novel #2+#3)": WrappedDualPathWithReranker(
            dual, reranker
        ),

        # ── Full System: all contributions ────────────────────────────
        "Full System (all contrib.)": WrappedAdaptive(dual, reranker),
    }

    evaluator = ComprehensiveEvaluator()
    all_results = evaluator.run_ablation_study(retrievers, dataset, top_k=TOP_K)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "ablation_v2.csv")
    evaluator.save_csv_report(all_results, csv_path)

    json_path = os.path.join(OUTPUT_DIR, "ablation_v2.json")
    clean = [{k: v for k, v in r.items() if k != "_per_sample"} for r in all_results]
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n✅ Results saved → {json_path}")

    # TDE analysis for full system
    best = all_results[-1]
    if "_per_sample" in best:
        tde = analyze_tde_distribution(best["_per_sample"])
        with open(os.path.join(OUTPUT_DIR, "tde_v2.json"), "w") as f:
            json.dump(tde, f, indent=2)
        print("\n📊 TDE Distribution (Full System):")
        for label, pct in tde.items():
            bar = "█" * int(pct * 40)
            print(f"  {label:35s} {pct:.3f}  {bar}")

    _print_latex_table(all_results, TOP_K)


def _print_latex_table(results, k):
    print("\n" + "=" * 70)
    print("  LaTeX Table for IEEE Paper:")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation Study: Retrieval Performance (n=500)}")
    print(r"\label{tab:ablation_v2}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(f"Configuration & Acc@1 & Acc@{k} & MRR & NDCG@{k} & WHAcc & MTD \\\\")
    print(r"\hline")
    for r in results:
        name = r["configuration"]
        bold = "Full System" in name
        pfx = r"\textbf{" if bold else ""
        sfx = "}" if bold else ""
        print(
            f"{pfx}{name}{sfx} & "
            f"{pfx}{r['accuracy_top1']:.3f}{sfx} & "
            f"{r['accuracy_topk']:.3f} & "
            f"{r['mrr']:.3f} & "
            f"{r.get('ndcg_at_5', 0):.3f} & "
            f"{r['weighted_hs_accuracy']:.3f} & "
            f"{r['mean_taxonomy_distance']:.3f} \\\\"
        )
    print(r"\hline")
    print(
        r"\multicolumn{7}{l}{\footnotesize WHAcc = Weighted HS Accuracy; "
        r"MTD = Mean Taxonomy Distance (lower is better)}"
    )
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()