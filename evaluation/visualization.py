import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Config': ['BM25', 'H(0.2)', 'H(0.5)', 'H(0.8)', 'Vector'],
    'Recall@1': [0.710, 0.720, 0.722, 0.724, 0.728],
    'Recall@5': [0.744, 0.744, 0.744, 0.744, 0.748],
    'MRR': [0.726, 0.730, 0.731, 0.732, 0.737]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df['Config'], df['Recall@1'], marker='o', label='Recall@1 (Accuracy)')
plt.plot(df['Config'], df['Recall@5'], marker='D', label='Recall@5 (Coverage)')
plt.plot(df['Config'], df['MRR'], marker='s', label='MRR (Rank Quality)')

plt.title('Complete Ablation Study: Retrieval Performance Metrics', fontsize=14)
plt.ylabel('Score (0.0 - 1.0)')
plt.xlabel('Retriever Configuration')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0.70, 0.76) 

plt.savefig('full_ablation_chart.png')
plt.show()