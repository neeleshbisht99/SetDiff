#%%#
import json
import random

#%%#
filename = 'hypothesis_knowledge_bank.json'
hypo_data = {}
with open(filename, 'r') as file:
    hypo_data = json.load(file)

knowledge_bank = []
random.seed(0)
for k, v in hypo_data.items():
    knowledge_bank.extend(v)

random.shuffle(knowledge_bank)

filename = 'knowledge_bank.json'
with open(filename, 'w') as file:
    json.dump(knowledge_bank, file, indent=4)

#%%#