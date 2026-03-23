import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load your results
df = pd.read_csv("evaluation/final_results.csv")

# Load a lightweight model for semantic checking
model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_denoising(df):
    raw_texts = df['raw_line'].astype(str).tolist()
    cleaned_texts = df['cleaned_line'].astype(str).tolist()
    
    # 1. Calculate Semantic Retention (Are they still the same thing?)
    print("Computing Semantic Retention...")
    emb_raw = model.encode(raw_texts, convert_to_tensor=True)
    emb_clean = model.encode(cleaned_texts, convert_to_tensor=True)
    
    # Calculate cosine similarity for each pair
    cos_sims = util.cos_sim(emb_raw, emb_clean).diagonal()
    avg_retention = cos_sims.mean().item()
    
    # 2. Calculate Noise Reduction (Length Difference)
    # Junk characters and extra spaces usually make raw lines longer or "messier"
    raw_lens = [len(x) for x in raw_texts]
    clean_lens = [len(x) for x in cleaned_texts]
    compression_ratio = np.mean(clean_lens) / np.mean(raw_lens)

    return {
        "Avg Semantic Retention": avg_retention,
        "Compression Ratio": compression_ratio,
        "Min Retention": cos_sims.min().item()
    }

results = evaluate_denoising(df)
print(f"\n--- GLOBAL DENOISING REPORT ---")
print(f"Meaning Preserved: {results['Avg Semantic Retention']:.4f}")
print(f"Data Compression: {results['Compression Ratio']:.4f} (Lower = More junk removed)")