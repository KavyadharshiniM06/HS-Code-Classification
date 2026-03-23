import pandas as pd

def calculate_denoising_metrics(file_path):
    df = pd.read_csv(file_path)
    
    # 1. CHARACTER NOISE REDUCTION (Lexical)
    # Measures how much 'junk' (codes, typos, symbols) was stripped away
    raw_len = df['raw_line'].str.len().mean()
    clean_len = df['cleaned_line'].str.len().mean()
    noise_reduction_pct = ((raw_len - clean_len) / raw_len) * 100
    
    # 2. SEMANTIC CONFIDENCE BOOST (Signal)
    # Measures how much clearer the product became to the Vector Engine
    avg_raw_score = df['raw_top_score'].mean()
    avg_clean_score = df['clean_top_score'].mean()
    confidence_gain_pct = ((avg_clean_score - avg_raw_score) / avg_raw_score) * 100
    
    # 3. RETRIEVAL ACCURACY DELTA
    # Did cleaning actually help find the right code?
    def is_correct(row, col_name):
        top_result = str(row[col_name]).split(',')[0]
        return str(row['ground_truth']).strip() == top_result.strip()

    raw_acc = df.apply(lambda r: is_correct(r, 'raw_retrieved_top5'), axis=1).mean() * 100
    clean_acc = df.apply(lambda r: is_correct(r, 'clean_retrieved_top5'), axis=1).mean() * 100

    print("--- OVERALL SYSTEM PERFORMANCE ---")
    print(f"1. Noise Reduction Ratio:    {noise_reduction_pct:.2f}%")
    print(f"2. Semantic Confidence Gain: {confidence_gain_pct:.2f}%")
    print(f"3. Accuracy Shift (Top-1):   {raw_acc:.2f}% -> {clean_acc:.2f}%")
    
    return noise_reduction_pct, confidence_gain_pct

calculate_denoising_metrics("evaluation/final_results.csv")