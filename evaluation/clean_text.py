import pandas as pd
from tqdm import tqdm
from retrievers.vector_retriever import VectorRetriever
import os
from utils.llm_cleaner import GroqCleaner

# --- PATHS & CONFIG ---
FAISS_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"
INPUT_CSV = "evaluation/evaluation_dataset.csv"
OUTPUT_CSV = "evaluation/final_results.csv"

def generate_comparative_csv(input_csv, output_path):
    # 1. Initialize Retriever and Cleaner
    vector = VectorRetriever(FAISS_PATH, META_PATH)
    cleaner = GroqCleaner() # Uses your class from above
    
    # 2. Load 500 samples
    df = pd.read_csv(input_csv).head(500)
    final_data = []

    print(f"🚀 Starting Denoising + Comparative Generation (500 samples)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ground_truth = str(row['true_hs_code']).strip().zfill(6)
        raw_text = str(row['query'])
        
        # 3. GENERATE THE CLEANED TEXT (Option B)
        # This is where the magic happens for your evaluation
        clean_text = cleaner.clean(raw_text)

        # 4. Retrieval on UNCLEANED
        raw_results = vector.retrieve(raw_text, top_k=5)
        raw_codes = [str(res['hs_code']).zfill(6) for res in raw_results]
        raw_top_score = raw_results[0]['score'] if raw_results else 0

        # 5. Retrieval on CLEANED
        if clean_text:
            clean_results = vector.retrieve(clean_text, top_k=5)
            clean_codes = [str(res['hs_code']).zfill(6) for res in clean_results]
            clean_top_score = clean_results[0]['score'] if clean_results else 0
        else:
            clean_codes = []
            clean_top_score = 0

        # 6. Append data
        final_data.append({
            "ground_truth": ground_truth,
            "raw_line": raw_text,
            "cleaned_line": clean_text,
            "raw_retrieved_top5": ",".join(raw_codes),
            "raw_top_score": round(raw_top_score, 4),
            "clean_retrieved_top5": ",".join(clean_codes),
            "clean_top_score": round(clean_top_score, 4)
        })

    # 7. Save to CSV
    results_df = pd.DataFrame(final_data)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Comparison file with LIVE cleaning saved to {output_path}")

if __name__ == "__main__":
    generate_comparative_csv(INPUT_CSV, OUTPUT_CSV)
