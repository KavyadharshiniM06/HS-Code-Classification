import json
import random
import re

def introduce_noise(text, noise_level="medium"):
    words = text.split()
    noisy_words = []

    for w in words:
        if len(w) <= 3:
            noisy_words.append(w)
            continue

        if noise_level == "low":
            if random.random() < 0.1:
                w = w[:-1]  # remove last character

        elif noise_level == "medium":
            if random.random() < 0.2:
                w = w[:-1]
            if random.random() < 0.2:
                w = w.replace("e", "", 1)

        elif noise_level == "high":
            if random.random() < 0.3:
                w = w[:-1]
            if random.random() < 0.3:
                w = re.sub(r"[aeiou]", "", w)

        noisy_words.append(w)

    return " ".join(noisy_words)


def create_synthetic_dataset(h6_path, samples=1000, noise_level="medium"):
    with open(h6_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = []

    leaf_nodes = [
        item for item in raw["results"]
        if item.get("isLeaf") == "1"
    ]

    selected = random.sample(leaf_nodes, min(samples, len(leaf_nodes)))

    for item in selected:
        original_text = item["text"]
        noisy_query = introduce_noise(original_text, noise_level)

        dataset.append({
            "query": noisy_query,
            "true_hs_code": item["id"]
        })

    return dataset