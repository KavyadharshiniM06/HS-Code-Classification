import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

H6_PATH = "data/H6.json"
INDEX_PATH = "indexing/vector_store/h6.faiss"
META_PATH = "indexing/vector_store/h6_meta.json"

model = SentenceTransformer("BAAI/bge-m3")

with open(H6_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

texts, meta = [], []

for item in raw["results"]:
    if "id" in item and "text" in item and item.get("isLeaf") == "1":
        texts.append(item["text"])
        meta.append({
            "hs_code": item["id"],
            "description": item["text"]
        })

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

dim = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 32)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("✅ Vector index built and saved")
