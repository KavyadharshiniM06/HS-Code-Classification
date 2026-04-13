"""
ICCA-RAG main entry point (enhanced pipeline)
=============================================
Runs the full enhanced pipeline on a sample receipt.
"""

from pipelines.enhanced_pipeline import EnhancedICCARAGPipeline

H6_PATH = "data/H6.json"
FAISS_INDEX_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"
ENRICHED_FAISS_PATH = "indexing/vector_store/h6_enriched.faiss"  # optional

RECEIPT = """
ENGINE OIL 4L
MRP 2450
GSTIN 29ABCDE1234Z
TOTAL 2450
MOBILE PHONE CHARGER FAST 20W
TRACTOR DIESEL ENGINE PART
LED BULB 9W COOL WHITE
AIR FILTER FOR DIESEL ENGINE
COTTON T-SHIRT MEN XL
STAINLESS STEEL COOKWARE SET
"""


def main():
    import os
    enriched = ENRICHED_FAISS_PATH if os.path.exists(ENRICHED_FAISS_PATH) else None

    print("⏳ Initializing enhanced ICCA-RAG pipeline...")
    pipeline = EnhancedICCARAGPipeline(
        h6_path=H6_PATH,
        faiss_index_path=FAISS_INDEX_PATH,
        meta_path=META_PATH,
        enriched_index_path=enriched,
        confidence_threshold=0.55,
        max_reformulation_iters=3,
    )

    print("⏳ Running multi-line prediction...")
    results = pipeline.predict(RECEIPT, top_k=5)

    for item in results:
        print(f"\n🧾 Raw:     {item['raw_line']}")
        print(f"   Cleaned: {item['cleaned_line']}")
        if item.get("reformulated"):
            print(f"   ↻ Query was reformulated")
        if item["prediction"]:
            print(f"   ✅ Prediction: {item['prediction']}  (conf: {item['confidence']:.2f})")
        else:
            print(f"   ❌ No prediction")
        if item["retrieved_candidates"]:
            print("   Top candidates:")
            for r in item["retrieved_candidates"][:3]:
                print(
                    f"     {r['doc_id']} | {r['score']:.3f} | "
                    f"{r['text'][:60]}..."
                )
        print("-" * 70)


if __name__ == "__main__":
    main()