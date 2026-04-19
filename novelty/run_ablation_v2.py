"""
Master Ablation Runner (memory-safe)
=====================================
Loads BAAI/bge-m3 ONCE and passes the shared instance to every retriever
that needs it. This prevents the OOM crash caused by each retriever loading
its own ~1 GB model copy.

Run:
  python novelty/run_ablation_v2.py
"""

import sys
import os
import json
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer          # ← load once here

from retrievers.sparse_retriever import SparseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
from novelty.evaluation.comprehensive_evaluator import (
    ComprehensiveEvaluator,
    analyze_tde_distribution,
)
from retrievers.hierarchical_retriever import (
    HierarchicalRetriever,
    AdaptiveQueryReformulator,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
H6_PATH               = "data/H6.json"
FAISS_PATH            = "indexing/vector_store/h6.faiss"
META_PATH             = "indexing/vector_store/h6_meta.json"
ENRICHED_FAISS_PATH   = "indexing/vector_store/h6_enriched.faiss"
EVAL_CSV              = "evaluation/evaluation_dataset.csv"
OUTPUT_DIR            = "evaluation"
MODEL_NAME            = "BAAI/bge-m3"
TOP_K                 = 5


# ── Dataset loader ─────────────────────────────────────────────────────────────
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


# ── Thin wrappers so every retriever exposes .retrieve() ──────────────────────
class WrappedVector:
    """VectorRetriever already has retrieve(); this normalises the key names."""
    def __init__(self, vector: VectorRetriever):
        self.vector = vector

    def retrieve(self, query, top_k=5):
        results = self.vector.retrieve(query, top_k=top_k)
        return [{"doc_id": r["hs_code"], "score": r["score"]} for r in results]


class WrappedSparse:
    def __init__(self, sparse: SparseRetriever):
        self.sparse = sparse

    def retrieve(self, query, top_k=5):
        results = self.sparse.search(query, top_k=top_k)
        return [{"doc_id": r["doc_id"], "score": r["score"]} for r in results]


class WrappedEnhanced:
    def __init__(self, retriever: EnhancedHybridRetriever):
        self.retriever = retriever

    def retrieve(self, query, top_k=5):
        results = self.retriever.retrieve(query, top_k=top_k)
        return [{"doc_id": r["doc_id"], "score": r["score"]} for r in results]


class WrappedAdaptive:
    def __init__(self, base_retriever):
        self.reformulator = AdaptiveQueryReformulator(
            base_retriever, confidence_threshold=0.65, max_iters=2
        )

    def retrieve(self, query, top_k=5):
        output = self.reformulator.retrieve_with_feedback(query, top_k=top_k)
        return output["results"]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("📂 Loading evaluation dataset...")
    dataset = load_dataset(EVAL_CSV)
    print(f"   → {len(dataset)} samples")

    # ── Load the sentence transformer ONCE ────────────────────────────────────
    print(f"\n🧠 Loading embedding model ({MODEL_NAME}) — this happens only once...")
    shared_model = SentenceTransformer(MODEL_NAME)
    print("   ✅ Model loaded and will be shared across all retrievers.")

    enriched_available = os.path.exists(ENRICHED_FAISS_PATH)
    if not enriched_available:
        print(f"\n   ⚠️  Enriched index not found at {ENRICHED_FAISS_PATH}")
        print("       Run: python novelty/indexing/build_enriched_index.py")
        print("       Falling back to base index for enriched configurations.")

    print("\n⚙️  Initializing retrievers (model shared, no extra RAM)...")

    # Baselines
    sparse = SparseRetriever(H6_PATH)

    # VectorRetriever loads its own model internally; pass shared_model via monkey-patch
    # to avoid a second load. VectorRetriever.__init__ accepts model_name as a string,
    # so we subclass it minimally.
    vector = _make_vector_retriever(shared_model)

    # Enhanced hybrid configs — all share `shared_model`
    enhanced_base = EnhancedHybridRetriever(
        faiss_index_path=FAISS_PATH,
        meta_path=META_PATH,
        h6_path=H6_PATH,
        enriched_index_path=None,
        model=shared_model,          # ← shared
    )

    enriched_path = ENRICHED_FAISS_PATH if enriched_available else None
    enhanced_enriched = EnhancedHybridRetriever(
        faiss_index_path=FAISS_PATH,
        meta_path=META_PATH,
        h6_path=H6_PATH,
        enriched_index_path=enriched_path,
        model=shared_model,          # ← shared
    )

    # Hierarchical retriever also loads a model; patch it the same way
    hier_base = _make_hierarchical(
        shared_model, FAISS_PATH, META_PATH, enriched_index_path=None
    )
    hier_enriched = _make_hierarchical(
        shared_model, FAISS_PATH, META_PATH,
        enriched_index_path=enriched_path,
        alpha_e=0.40,
    )

    # ── Build retriever map ────────────────────────────────────────────────────
    retrievers = {
        "BM25 (baseline)":                  WrappedSparse(sparse),
        "Vector (baseline)":                WrappedVector(vector),
        "Enhanced Hybrid (no enrichment)":  WrappedEnhanced(enhanced_base),
        "Enhanced Hybrid + Enrichment":     WrappedEnhanced(enhanced_enriched),
        "Hierarchical (no enrichment)":     hier_base,
        "Hierarchical + Ontology Enrich.":  hier_enriched,
        "Full System (Hier+Enrich+Adapt)":  WrappedAdaptive(hier_enriched),
    }

    # ── Run ablation ───────────────────────────────────────────────────────────
    evaluator = ComprehensiveEvaluator()
    all_results = evaluator.run_ablation_study(retrievers, dataset, top_k=TOP_K)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path  = os.path.join(OUTPUT_DIR, "ablation_comprehensive.csv")
    json_path = os.path.join(OUTPUT_DIR, "ablation_comprehensive.json")

    evaluator.save_csv_report(all_results, csv_path)

    clean = [{k: v for k, v in r.items() if k != "_per_sample"} for r in all_results]
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"✅ Full JSON report → {json_path}")

    # TDE analysis for best (last) system
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

    _print_latex_table(all_results, TOP_K)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_vector_retriever(shared_model: SentenceTransformer) -> "VectorRetriever":
    """
    Build a VectorRetriever but replace its internal model with the shared one
    so no second copy of bge-m3 is loaded.
    """
    vr = VectorRetriever.__new__(VectorRetriever)
    vr.model = shared_model

    import faiss, json as _json
    vr.index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        vr.meta = _json.load(f)
    if len(vr.meta) != vr.index.ntotal:
        raise ValueError("VectorRetriever: FAISS / meta length mismatch")
    return vr


def _make_hierarchical(
    shared_model: SentenceTransformer,
    faiss_index_path: str,
    meta_path: str,
    enriched_index_path=None,
    alpha_h: float = 0.25,
    alpha_e: float = 0.0,
) -> "HierarchicalRetriever":
    """
    Build a HierarchicalRetriever sharing the pre-loaded model.
    """
    hr = HierarchicalRetriever.__new__(HierarchicalRetriever)
    hr.model     = shared_model
    hr.alpha_h   = alpha_h
    hr.alpha_e   = alpha_e

    import faiss, json as _json
    from collections import defaultdict

    hr.base_index = faiss.read_index(faiss_index_path)

    hr.enriched_index = None
    if enriched_index_path:
        try:
            hr.enriched_index = faiss.read_index(enriched_index_path)
        except Exception:
            pass

    with open(meta_path, "r", encoding="utf-8") as f:
        hr.meta = _json.load(f)

    # Rebuild hierarchy tables (copied from HierarchicalRetriever._build_hierarchy_tables)
    hr.chapter_to_indices = defaultdict(list)
    hr.heading_to_indices = defaultdict(list)
    for i, item in enumerate(hr.meta):
        code = item.get("hs_code", "")
        if len(code) >= 2:
            hr.chapter_to_indices[code[:2]].append(i)
        if len(code) >= 4:
            hr.heading_to_indices[code[:4]].append(i)

    return hr


def _print_latex_table(all_results, k):
    def bold(v, flag):
        return rf"\textbf{{{v}}}" if flag else str(v)

    print("\n" + "=" * 65)
    print("  LaTeX Table:")
    print("=" * 65)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation Study: Retrieval Performance on HS Code Dataset}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(f"Configuration & Acc@1 & Acc@{k} & MRR & NDCG@{k} & WHAcc & MTD \\\\")
    print(r"\hline")
    for r in all_results:
        is_best = r == all_results[-1]
        print(
    f"{bold(r['configuration'], is_best)} & "
    f"{bold(format(r['accuracy_top1'], '.3f'), is_best)} & "
    f"{r['accuracy_topk']:.3f} & "
    f"{r['mrr']:.3f} & "
    f"{r[f'ndcg_at_{k}']:.3f} & "
    f"{r['weighted_hs_accuracy']:.3f} & "
    f"{r['mean_taxonomy_distance']:.3f} \\\\"
)
    print(r"\hline")
    print(r"\multicolumn{7}{l}{\footnotesize WHAcc = Weighted HS Accuracy; MTD = Mean Taxonomy Distance}")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()