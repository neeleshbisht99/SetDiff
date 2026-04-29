import json
from rdflib import Graph, RDFS, OWL, URIRef, Namespace

# Load OWL file
g = Graph()
g.parse("/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/prep_scripts/RadLex.owl", format="xml")

RADLEX = Namespace("http://radlex.org/RID/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# Extract preferred labels
terms = set()
for s, p, o in g.triples((None, SKOS.prefLabel, None)):
    terms.add(str(o).lower().strip())

# Also try RDFS label
for s, p, o in g.triples((None, RDFS.label, None)):
    terms.add(str(o).lower().strip())

print(f"Total terms: {len(terms)}")

# Filter chest-relevant
chest_keywords = [
    "lung", "pulmonary", "pleural", "cardiac", "chest",
    "thorax", "thoracic", "mediastin", "bronch", "trachea",
    "diaphragm", "rib", "atelectasis", "pneumo", "effusion",
    "opacity", "consolidation", "nodule", "mass", "fibrosis",
    "pneumonia", "edema", "emphysema", "silhouette", "hilar"
]

filtered = [t for t in terms if any(k in t for k in chest_keywords)]
print(f"Chest-relevant: {len(filtered)}")

with open("radlex_chest_kb.json", "w") as f:
    json.dump(sorted(filtered), f, indent=2)


#python3 data/knowledge_banks/mimic/prep_scripts/prep_v3.py