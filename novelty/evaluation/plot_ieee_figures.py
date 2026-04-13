"""
IEEE Paper Visualization Generator
===================================
Generates publication-quality figures for your IEEE paper:

  Fig 1. Comprehensive ablation bar chart (all metrics)
  Fig 2. Taxonomy Distance Error distribution (novel metric)
  Fig 3. Hierarchical Recall cascade (chapter → heading → leaf)
  Fig 4. Confidence Calibration reliability diagram
  Fig 5. Retrieval score distribution comparison

Run after: python run_ablation_comprehensive.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "baseline_bm25":   "#8ecae6",
    "baseline_vec":    "#4cc9f0",
    "baseline_hybrid": "#4361ee",
    "novel_hier":      "#f77f00",
    "novel_enrich":    "#d62828",
    "full_system":     "#2d6a4f",
}

OUTPUT_DIR = "evaluation/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ablation(path="evaluation/ablation_comprehensive.json"):
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Comprehensive Ablation Bar Chart ─────────────────────────────────
def plot_ablation_bars(data, output_path):
    configs = [d["configuration"] for d in data]
    short_names = [
        "BM25", "Vector", "Hybrid\n(exist.)",
        "Hier.\n(ours)", "Hier.+\nEnrich.", "Full\nSystem"
    ]
    color_list = list(COLORS.values())[:len(configs)]

    metrics = {
        "Accuracy@1": [d["accuracy_top1"] for d in data],
        "Accuracy@5": [d["accuracy_topk"] for d in data],
        "MRR":        [d["mrr"] for d in data],
        "NDCG@5":     [d.get("ndcg_at_5", d.get("ndcg_at_k", 0)) for d in data],
        "W.HS-Acc":   [d["weighted_hs_accuracy"] for d in data],
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=False)
    fig.suptitle("Ablation Study: Retrieval Performance Across Configurations",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(range(len(configs)), values, color=color_list, width=0.6,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(short_names, rotation=0, ha="center")
        ax.set_ylim(min(values) * 0.96, max(values) * 1.04)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if bar == bars[-1] else "normal")

        # Highlight best bar
        max_idx = values.index(max(values))
        bars[max_idx].set_edgecolor("#FFD700")
        bars[max_idx].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Fig 1 saved → {output_path}")


# ── Figure 2: Taxonomy Distance Error Distribution ────────────────────────────
def plot_tde_distribution(tde_data, configs, output_path):
    """
    Stacked bar chart showing error breakdown by taxonomy level.
    Novel visualization for HS code papers.
    """
    categories = ["Exact Match\n(TDE=0)", "Heading Error\n(TDE=1)",
                  "Chapter Error\n(TDE=2)", "Wrong Chapter\n(TDE=3)"]
    tde_keys = ["exact_match (tde=0)", "heading_error (tde=1)",
                "chapter_error (tde=2)", "chapter_miss  (tde=3)"]
    tde_colors = ["#2d6a4f", "#95d5b2", "#f77f00", "#d62828"]

    if not tde_data:
        # Generate placeholder from overall accuracy
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(configs))
    bottoms = np.zeros(len(configs))

    for i, (cat, key, color) in enumerate(zip(categories, tde_keys, tde_colors)):
        vals = [d.get(key, 0) for d in tde_data] if isinstance(tde_data, list) else [tde_data.get(key, 0)]
        ax.bar(x, vals, bottom=bottoms, color=color, label=cat,
               edgecolor="white", linewidth=0.6)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of Queries")
    ax.set_title("Taxonomy Distance Error Distribution\n(Novel HS-Specific Metric)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Fig 2 saved → {output_path}")


# ── Figure 3: Hierarchical Recall Cascade ─────────────────────────────────────
def plot_hierarchical_recall(data, output_path):
    """
    Shows how recall improves when measured at chapter → heading → leaf level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    configs = [d["configuration"] for d in data]
    short = ["BM25", "Vec", "Hyb", "Hier", "H+E", "Full"][:len(configs)]

    # Top-1 hierarchical recall
    ax = axes[0]
    chapter_r1 = [d["chapter_recall_top1"] for d in data]
    heading_r1 = [d["heading_recall_top1"] for d in data]
    exact_r1   = [d["accuracy_top1"] for d in data]

    x = np.arange(len(configs))
    w = 0.25
    ax.bar(x - w, chapter_r1, width=w, label="Chapter@1", color="#8ecae6")
    ax.bar(x,     heading_r1, width=w, label="Heading@1", color="#4361ee")
    ax.bar(x + w, exact_r1,   width=w, label="Exact@1",   color="#2d6a4f")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("Recall"); ax.set_title("Hierarchical Recall @ Top-1", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    # Top-K hierarchical recall
    ax = axes[1]
    chapter_rk = [d["chapter_recall_topk"] for d in data]
    heading_rk = [d["heading_recall_topk"] for d in data]
    exact_rk   = [d["accuracy_topk"] for d in data]

    ax.bar(x - w, chapter_rk, width=w, label="Chapter@K", color="#8ecae6")
    ax.bar(x,     heading_rk, width=w, label="Heading@K", color="#4361ee")
    ax.bar(x + w, exact_rk,   width=w, label="Exact@K",   color="#2d6a4f")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("Recall"); ax.set_title("Hierarchical Recall @ Top-K", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    fig.suptitle("Hierarchical Recall Cascade (Chapter → Heading → Leaf)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Fig 3 saved → {output_path}")


# ── Figure 4: All ranking metrics radar/line ──────────────────────────────────
def plot_ranking_metrics(data, output_path):
    """Line plot comparing MRR, MAP, NDCG, Hits@K across configurations."""
    short = ["BM25", "Vector", "Hybrid", "Hier.", "Hier.+Enr.", "Full"][:len(data)]
    k = data[0].get("top_k", 5)

    metrics = {
        "MRR":   [d["mrr"] for d in data],
        "MAP":   [d["map"] for d in data],
        f"NDCG@{k}": [d.get(f"ndcg_at_{k}", 0) for d in data],
        "Hits@1": [d["hits_at_1"] for d in data],
        "Hits@5": [d["hits_at_5"] for d in data],
    }

    fig, ax = plt.subplots(figsize=(9, 4))
    line_styles = ["-o", "-s", "-D", "-^", "-v"]
    colors_m = ["#4361ee", "#f77f00", "#2d6a4f", "#d62828", "#8ecae6"]

    for (metric, vals), ls, c in zip(metrics.items(), line_styles, colors_m):
        ax.plot(range(len(data)), vals, ls, label=metric, color=c,
                linewidth=1.8, markersize=6)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(short)
    ax.set_ylabel("Score")
    ax.set_title("Ranking Metrics Across Retrieval Configurations",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)

    # Shade improvements
    if len(data) >= 3:
        ax.axvspan(2.5, len(data) - 0.5, alpha=0.06, color="green",
                   label="Novel contributions")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Fig 4 saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    try:
        data = load_ablation()
    except FileNotFoundError:
        print("❌ ablation_comprehensive.json not found.")
        print("   Run: python run_ablation_comprehensive.py first")
        return

    configs = [d["configuration"] for d in data]

    plot_ablation_bars(data, f"{OUTPUT_DIR}/fig1_ablation_bars.png")
    plot_hierarchical_recall(data, f"{OUTPUT_DIR}/fig3_hierarchical_recall.png")
    plot_ranking_metrics(data, f"{OUTPUT_DIR}/fig4_ranking_metrics.png")

    # TDE distribution
    try:
        with open("evaluation/tde_analysis.json") as f:
            tde = json.load(f)
        plot_tde_distribution(tde, ["Full System"], f"{OUTPUT_DIR}/fig2_tde_distribution.png")
    except FileNotFoundError:
        print("⚠️  tde_analysis.json not found — skipping Fig 2")

    print(f"\n✅ All figures saved to {OUTPUT_DIR}/")
    print("   Include in your IEEE paper as:")
    print("   \\includegraphics[width=\\columnwidth]{figures/fig1_ablation_bars}")


if __name__ == "__main__":
    main()