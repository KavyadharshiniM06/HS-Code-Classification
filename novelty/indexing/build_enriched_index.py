"""
Enriched Index Builder — Novel Contribution #1 (cont.)
=======================================================
Builds a dual-layer FAISS index:
  - Layer A: embeddings of BASE descriptions (baseline)
  - Layer B: embeddings of ENRICHED descriptions (ontology-augmented)

At query time, we fuse scores from both layers — an ensemble that
consistently outperforms single-layer approaches.

IEEE novelty claim: "Ontology-Augmented Dual-Layer Semantic Indexing
for Harmonized System Code Retrieval"
"""

import json
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Allow direct execution via `python novelty/indexing/build_enriched_index.py`
# by ensuring the repository root is on sys.path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from novelty.knowledge.ontology_enricher import OntologyEnricher


def build_enriched_index(
    h6_path: str,
    base_index_path: str,
    enriched_index_path: str,
    meta_path: str,
    enriched_meta_path: str,
    model_name: str = "BAAI/bge-m3",
):
    """
    Builds two FAISS indexes:
      1. base_index    — embeddings from original H6 text
      2. enriched_index — embeddings from ontology-enriched text
    """
    print("🔬 Loading ontology enricher...")
    enricher = OntologyEnricher(h6_path)
    docs = enricher.enrich()

    print(f"📚 Enriched {len(docs)} leaf HS codes with ontology context.")

    model = SentenceTransformer(model_name)

    base_texts = [d["base_text"] for d in docs]
    enriched_texts = [d["enriched_text"] for d in docs]
    meta = [{"hs_code": d["id"], "description": d["base_text"],
              "enriched_text": d["enriched_text"],
              "keywords": d["keywords"],
              "chapter": d["chapter"],
              "heading": d["heading"]} for d in docs]

    print("⚙️  Encoding BASE descriptions...")
    base_embs = model.encode(
        base_texts, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    print("⚙️  Encoding ENRICHED descriptions...")
    enriched_embs = model.encode(
        enriched_texts, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    dim = base_embs.shape[1]

    # Build base index
    base_index = faiss.IndexHNSWFlat(dim, 32)
    base_index.add(base_embs)
    faiss.write_index(base_index, base_index_path)
    print(f"✅ Base index saved → {base_index_path}")

    # Build enriched index
    enriched_index = faiss.IndexHNSWFlat(dim, 32)
    enriched_index.add(enriched_embs)
    faiss.write_index(enriched_index, enriched_index_path)
    print(f"✅ Enriched index saved → {enriched_index_path}")

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(enriched_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Metadata saved → {meta_path}")


if __name__ == "__main__":
    build_enriched_index(
        h6_path="data/H6.json",
        base_index_path="indexing/vector_store/h6.faiss",
        enriched_index_path="indexing/vector_store/h6_enriched.faiss",
        meta_path="indexing/vector_store/h6_meta.json",
        enriched_meta_path="indexing/vector_store/h6_enriched_meta.json",
    )
