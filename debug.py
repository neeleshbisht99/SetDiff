#%%#
import json

#%%#
filename = 'hypothesis_corpus.json'
hypo_data = {}
with open(filename, 'r') as file:
    hypo_data = json.load(file)

root = "data/VisDiffBench"
easy = [json.loads(line) for line in open(f"{root}/easy.jsonl")]
medium = [json.loads(line) for line in open(f"{root}/medium.jsonl")]
hard = [json.loads(line) for line in open(f"{root}/hard.jsonl")]
data = easy + medium + hard

for idx in range(0, 150):
    item = data[idx]
    a_b = f"{item['set1']}_{item['set2']}"
    b_a = f"{item['set2']}_{item['set1']}"
    if a_b not in hypo_data:
        print(f"a_b: {a_b} not present, {idx}")
    if b_a not in hypo_data:
        print(f"b_a: {b_a} not present, {idx}")

#%%#