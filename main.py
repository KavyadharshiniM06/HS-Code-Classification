from pipelines.pipeline_main import ICCARAGPipeline

H6_PATH = "data/H6.json"
FAISS_INDEX_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"

RECEIPT = """
ENGINE OIL 4L
MRP 2450
GSTIN 29ABCDE1234Z
TOTAL 2450
MOBILE PHONE CHARGER FAST 20W
TRACTOR DIESEL ENGINE PART
LED BULB 9W COOL WHITE
AIR FILTER FOR DIESEL ENGINE
"""

def main():
    print("⏳ Initializing ICCA-RAG pipeline...")
    pipeline = ICCARAGPipeline(H6_PATH, FAISS_INDEX_PATH, META_PATH)

    print("⏳ Running multi-line prediction...")
    results_per_product = pipeline.predict(RECEIPT, top_k=5)

    for item in results_per_product:
        print(f"\n🧾 Cleaned Product: {item['product']}")
        if item['predictions']:
            print("Top HS Code Predictions:")
            for r in item['predictions']:
                print(f"  {r['hs_code']} | score: {r['score']} | {r['description']}")
        else:
            print("  No HS codes found.")
        print("-" * 60)

if __name__ == "__main__":
    main()
