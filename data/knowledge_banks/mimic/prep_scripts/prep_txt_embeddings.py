import json
import numpy as np
import requests
import os

SERVER_URL = "http://localhost:8099/"
INPUT_JSON = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_knowledge_bank.json"
OUTPUT_NPY = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_knowledge_bank_embeddings.npy"

BATCH_SIZE = 512

os.makedirs(os.path.dirname(OUTPUT_NPY), exist_ok=True)


with open(INPUT_JSON, "r", encoding="utf-8") as f:
    texts = json.load(f)

assert isinstance(texts, list), "JSON must contain a list of text strings."

print(f"[INFO] Total texts: {len(texts)}")
print(f"[INFO] Batch size: {BATCH_SIZE}")
print(f"[INFO] Total batches: {(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE}")

all_embs = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]

    response = requests.post(
        SERVER_URL,
        data={"text": json.dumps(batch)},
        timeout=3600,
    )
    response.raise_for_status()

    data = response.json()
    embs = np.array(data["embeddings"], dtype=np.float32)
    all_embs.append(embs)

    print(f"done {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

embeddings = np.concatenate(all_embs, axis=0)

print("final shape:", embeddings.shape)
np.save(OUTPUT_NPY, embeddings)
print(f"Saved embeddings to {OUTPUT_NPY}")


# python3 data/knowledge_banks/mimic/prep_scripts/prep_txt_embeddings.py