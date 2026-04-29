import json
import numpy as np
import requests
import os

SERVER_URL = "http://localhost:8099/"
INPUT_CSV = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/datasets/mimic_cxr_imagesets/MimicCxrImageSets.csv"  # contains path column
OUTPUT_NPY = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_image_embeddings.npy"
OUTPUT_INDEX = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_image_path_to_idx.json"

BATCH_SIZE = 512

os.makedirs(os.path.dirname(OUTPUT_NPY), exist_ok=True)

# ---- load image paths ----
import pandas as pd
df = pd.read_csv(INPUT_CSV)

image_paths = df["path"].tolist()

print(f"[INFO] Total images: {len(image_paths)}")
print(f"[INFO] Batch size: {BATCH_SIZE}")
print(f"[INFO] Total batches: {(len(image_paths) + BATCH_SIZE - 1)//BATCH_SIZE}")

all_embs = []
path_to_idx = {}

counter = 0

for i in range(0, len(image_paths), BATCH_SIZE):
    batch = image_paths[i:i + BATCH_SIZE]

    response = requests.post(
        SERVER_URL,
        data={"image": json.dumps(batch)},
        timeout=3600,
    )
    response.raise_for_status()

    data = response.json()
    embs = np.array(data["embeddings"], dtype=np.float32)
    all_embs.append(embs)

    # build mapping
    for j, path in enumerate(batch):
        path_to_idx[path] = counter + j

    counter += len(batch)

    print(f"done {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)}")

# ---- final save ----
embeddings = np.concatenate(all_embs, axis=0)

print("final shape:", embeddings.shape)
np.save(OUTPUT_NPY, embeddings)

with open(OUTPUT_INDEX, "w") as f:
    json.dump(path_to_idx, f)

print(f"[DONE] Saved embeddings to {OUTPUT_NPY}")
print(f"[DONE] Saved index mapping to {OUTPUT_INDEX}")


# python3 data/knowledge_banks/mimic/prep_scripts/prep_img_embeddings.py