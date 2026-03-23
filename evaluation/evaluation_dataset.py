import json
import csv
import random

H6_PATH = "data/H6.json"
OUTPUT_CSV = "evaluation/evaluation_dataset.csv"
NUM_SAMPLES = 500


def introduce_noise(text):

    words = text.split()

    for i in range(len(words)):

        if random.random() < 0.3:  # 30% chance to corrupt a word

            word = list(words[i])

            if len(word) > 3:

                op = random.choice(["delete", "swap", "replace"])

                if op == "delete":
                    pos = random.randint(0, len(word)-1)
                    del word[pos]

                elif op == "swap" and len(word) > 2:
                    pos = random.randint(0, len(word)-2)
                    word[pos], word[pos+1] = word[pos+1], word[pos]

                elif op == "replace":
                    pos = random.randint(0, len(word)-1)
                    word[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")

                words[i] = "".join(word)

    if random.random() < 0.2:
        random.shuffle(words)

    return " ".join(words)


def generate_dataset():

    with open(H6_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["results"]

    # keep only real HS codes
    items = [x for x in items if x["id"].isdigit()]

    rows = []

    for _ in range(NUM_SAMPLES):

        item = random.choice(items)

        hs_code = item["id"]
        description = item["text"]

        noisy_query = introduce_noise(description)

        rows.append({
            "query": noisy_query,
            "true_hs_code": hs_code
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=["query", "true_hs_code"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate_dataset()
    print("Noisy evaluation dataset generated successfully.")