import os
import csv
import pytesseract
from PIL import Image
from pipelines.enhanced_pipeline import ICCARAGPipeline

IMAGE_DIR = r"data\SROIE2019\train\img"
OUTPUT_CSV = "results/sroie_icca_rag_results_new.csv"

H6_PATH = "data/H6.json"
FAISS_INDEX_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"

NUM_IMAGES = 10

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Pipeline
pipeline = ICCARAGPipeline(H6_PATH, FAISS_INDEX_PATH, META_PATH)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))][:NUM_IMAGES]
os.makedirs("results", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 1. Added "retrieved_description" to the header
    writer.writerow([
        "image_name",
        "raw_line",
        "cleaned_line",
        "final_prediction",
        "confidence",
        "reasoning",
        "retrieved_code",
        "retrieved_description", 
        "retrieved_score"
    ])

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)

        try:
            image = Image.open(img_path)
            raw_text = pytesseract.image_to_string(image)

            # Predict returns a list of dictionaries per product line
            results = pipeline.predict(raw_text, top_k=5)

            for item in results:
                retrieved_docs = item.get("retrieved_candidates", [])

                if not retrieved_docs:
                    writer.writerow([
                        img_name, item["raw_line"], item["cleaned_line"],
                        item["prediction"], item["confidence"], item["reasoning"],
                        "", "", ""
                    ])
                else:
                    for doc in retrieved_docs:
                        # 2. Extract description from the 'text' key populated in pipeline_main.py
                        writer.writerow([
                            img_name,
                            item["raw_line"],
                            item["cleaned_line"],
                            item["prediction"],
                            item["confidence"],
                            item["reasoning"],
                            doc["doc_id"],
                            doc.get("text", "No Description Found"), 
                            doc["score"]
                        ])

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

print(f"\n✅ Results with descriptions saved to: {OUTPUT_CSV}")
