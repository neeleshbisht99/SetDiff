#!/usr/bin/env python3
"""
Build an improved MIMIC-CXR knowledge bank for CCDiff.

Strategy:
1. Start with the 26 clinical set descriptions (ground truth concepts)
2. Add curated clinical expansion phrases per set
3. Filter the extracted report phrases (remove negations, too-short, too-generic)
4. Deduplicate and cap at a reasonable size
"""

import json
import re
from pathlib import Path

# ── 1. The 26 clinical set descriptions (your paired set names) ─────────────
SET_DESCRIPTIONS = [
    "clear lungs with no acute abnormality",
    "diffuse bilateral lung opacities",
    "localized lung opacity",
    "enlarged cardiac silhouette and large heart",
    "pleural fluid at the lung bases",
    "air in the pleural space with absent lung markings",
    "medical tubes or implanted devices present",
    "post-surgical changes of the chest",
    "reduced lung volume due to atelectasis",
    "hyperinflated lungs with increased lung volume",
    "chest wall or rib abnormality",
    "abnormal widening or contour of the mediastinum",
    "elevation of hemidiaphragm",
    "upper-lobe-predominant lung abnormality",
    "lower-lobe-predominant lung abnormality",
    "interstitial lung disease with reticular pattern",
    "ground-glass opacity pattern",
    "pulmonary nodules or lung mass",
    "unilateral lung abnormality",
    "asymmetric lung volumes",
    "apical pleural thickening or scarring",
    "costophrenic angle abnormality without large effusion",
    "mediastinal or tracheal shift",
    "focal lucency or cavitary lung lesion",
]

# ── 2. Curated clinical expansion phrases ───────────────────────────────────
CLINICAL_EXPANSIONS = [
    # Effusion / pleural
    "pleural effusion",
    "bilateral pleural effusions",
    "left pleural effusion",
    "right pleural effusion",
    "small pleural effusion",
    "moderate pleural effusion",
    "large pleural effusion",
    "blunting of costophrenic angle",
    "pleural thickening",
    "loculated pleural effusion",

    # Pneumothorax
    "pneumothorax",
    "left pneumothorax",
    "right pneumothorax",
    "tension pneumothorax",
    "small pneumothorax",

    # Consolidation / opacity
    "consolidation",
    "airspace consolidation",
    "focal consolidation",
    "patchy consolidation",
    "bilateral consolidation",
    "lobar consolidation",
    "ground glass opacity",
    "diffuse ground glass opacities",
    "airspace opacity",
    "patchy opacity",
    "bilateral opacities",

    # Atelectasis
    "atelectasis",
    "basilar atelectasis",
    "subsegmental atelectasis",
    "left lower lobe atelectasis",
    "right lower lobe atelectasis",
    "bibasilar atelectasis",
    "lobar collapse",
    "volume loss",

    # Cardiac
    "cardiomegaly",
    "mild cardiomegaly",
    "moderate cardiomegaly",
    "enlarged cardiac silhouette",
    "enlarged heart",
    "normal heart size",
    "cardiomediastinal silhouette",

    # Edema / congestion
    "pulmonary edema",
    "mild pulmonary edema",
    "moderate pulmonary edema",
    "severe pulmonary edema",
    "pulmonary vascular congestion",
    "vascular congestion",
    "interstitial edema",

    # Mediastinum
    "mediastinal widening",
    "widened mediastinum",
    "mediastinal shift",
    "tracheal deviation",
    "tracheal shift",
    "mediastinal contour abnormality",

    # Nodules / mass
    "pulmonary nodule",
    "lung nodule",
    "lung mass",
    "pulmonary mass",
    "cavitary lesion",
    "cavitary nodule",
    "focal lucency",

    # Interstitial / fibrosis
    "interstitial markings",
    "reticular pattern",
    "reticular opacities",
    "interstitial lung disease",
    "pulmonary fibrosis",
    "honeycombing",
    "coarse interstitial pattern",

    # Hyperinflation / emphysema
    "hyperinflation",
    "hyperinflated lungs",
    "emphysema",
    "increased lung volumes",
    "flattened diaphragm",
    "barrel chest",

    # Diaphragm
    "elevated hemidiaphragm",
    "right hemidiaphragm elevation",
    "left hemidiaphragm elevation",
    "diaphragmatic elevation",

    # Ribs / chest wall
    "rib fracture",
    "rib fractures",
    "chest wall abnormality",
    "osseous abnormality",

    # Post-surgical
    "sternotomy wires",
    "post sternotomy",
    "surgical clips",
    "post-surgical changes",
    "median sternotomy",

    # Apical
    "apical scarring",
    "apical pleural thickening",
    "apical fibrosis",
    "upper lobe fibrosis",

    # Unilateral
    "unilateral opacity",
    "unilateral lung abnormality",
    "asymmetric lung volumes",

    # Normal
    "clear lungs",
    "lungs are clear",
    "no acute cardiopulmonary process",
    "no acute findings",
    "normal chest radiograph",
]

# ── 3. Phrases to filter from report-extracted bank ─────────────────────────
NEGATION_PREFIXES = ("no ", "without ", "absent ", "negative ", "not ")
SKIP_SUBSTRINGS = [
    "unchanged", "improved", "worsened", "prior", "compared",
    "recommend", "follow", "history", "clinical", "evaluation",
    "please", "note", "redemonstrat", "again", "persistent",
]
TOO_GENERIC = {
    "lung", "lungs", "heart", "right", "left", "upper", "lower",
    "chest", "small", "mild", "large", "bilateral", "normal",
    "low lung", "right lung", "left lung", "lower lung", "upper lung",
}


def is_good_report_phrase(phrase: str) -> bool:
    """Filter report-extracted phrases."""
    p = phrase.lower().strip()

    # Remove negations
    if any(p.startswith(neg) for neg in NEGATION_PREFIXES):
        return False

    # Remove phrases with bad substrings
    if any(bad in p for bad in SKIP_SUBSTRINGS):
        return False

    # Remove too-generic single/double word phrases
    if p in TOO_GENERIC:
        return False

    # Must be at least 2 words
    if len(p.split()) < 2:
        return False

    return True


def load_report_phrases(txt_path: str, max_phrases: int = 1000) -> list:
    """Load and filter report-extracted phrases, cap at max_phrases."""
    path = Path(txt_path)
    if not path.exists():
        print(f"Warning: {txt_path} not found, skipping report phrases.")
        return []

    phrases = []
    with open(path, "r") as f:
        for line in f:
            phrase = line.strip()
            if phrase and is_good_report_phrase(phrase):
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                break

    return phrases


def build_knowledge_bank(
    report_phrases_txt: str = "mimic_report_phrases_cleaned.txt",
    output_json: str = "mimic_kb.json",
    max_report_phrases: int = 1000,
):
    # Start with set descriptions + expansions
    kb = list(SET_DESCRIPTIONS) + list(CLINICAL_EXPANSIONS)

    # Add filtered report phrases
    report_phrases = load_report_phrases(report_phrases_txt, max_report_phrases)
    kb.extend(report_phrases)

    # Deduplicate preserving order
    seen = set()
    final_kb = []
    for phrase in kb:
        p = phrase.lower().strip()
        if p not in seen:
            seen.add(p)
            final_kb.append(phrase.lower().strip())

    # Save
    with open(output_json, "w") as f:
        json.dump(final_kb, f, indent=2)

    print(f"Set descriptions:   {len(SET_DESCRIPTIONS)}")
    print(f"Clinical expansions: {len(CLINICAL_EXPANSIONS)}")
    print(f"Report phrases:     {len(report_phrases)}")
    print(f"Total (deduped):    {len(final_kb)}")
    print(f"Saved to: {output_json}")

    return final_kb


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.report_phrases_txt = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/prep_scripts/raw/mimic_report_phrases_cleaned.txt"
    args.output_json = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_knowledge_bank.json"
    args.max_report_phrases = 1000
    build_knowledge_bank(
        report_phrases_txt=args.report_phrases_txt,
        output_json=args.output_json,
        max_report_phrases=args.max_report_phrases,
    )

# python3 data/knowledge_banks/mimic/prep_scripts/prep_v2.py