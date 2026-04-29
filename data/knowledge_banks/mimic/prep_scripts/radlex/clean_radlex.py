import json

with open("/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/prep_scripts/radlex_chest_kb.json") as f:
    terms = json.load(f)

# Keep only findings/pathology relevant terms
KEEP_KEYWORDS = [
    "opacity", "consolidation", "effusion", "pneumothorax", "atelectasis",
    "edema", "cardiomegaly", "nodule", "mass", "fibrosis", "emphysema",
    "pneumonia", "infiltrate", "pleural", "mediastinal", "pulmonary",
    "interstitial", "ground glass", "reticular", "silhouette", "hyperinflation",
    "collapse", "scarring", "thickening", "lucency", "cavitary", "blunting",
    "widening", "shift", "elevation", "diaphragm", "cardiac", "hilar"
]

# Reject anatomical/non-finding terms
REJECT_KEYWORDS = [
    "nerve", "root", "rootlet", "vein", "artery", "ganglion", "branch",
    "segment", "bronchus", "lobe of", "part of", "surface of", "layer",
    "wall of", "tract", "duct", "node group", "lymph node group"
]

# Reject non-English
def is_english(term):
    try:
        term.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

filtered = []
for term in terms:
    if not is_english(term):
        continue
    if any(r in term for r in REJECT_KEYWORDS):
        continue
    if not any(k in term for k in KEEP_KEYWORDS):
        continue
    if len(term.split()) < 2 or len(term.split()) > 8:
        continue
    filtered.append(term)

filtered = sorted(set(filtered))
print(f"Original: {len(terms)}, Cleaned: {len(filtered)}")

with open("/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/prep_scripts/radlex_chest_kb_clean.json", "w") as f:
    json.dump(filtered, f, indent=2)


# python3 data/knowledge_banks/mimic/prep_scripts/clean_radlex.py