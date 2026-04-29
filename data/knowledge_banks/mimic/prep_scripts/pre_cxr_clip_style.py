#!/usr/bin/env python3
"""
Expand knowledge bank phrases into report-style sentences for CXR-CLIP.

CXR-CLIP is trained on radiology reports, so report-style sentences
embed much better than isolated terms.
"""

import json
from pathlib import Path

# ── Templates that mimic radiology report style ─────────────────────────────
# TEMPLATES = [
#     "there is {}",
#     "{} is present",
#     "the lungs show {}",
#     "findings consistent with {}",
#     "evidence of {}",
#     "chest radiograph demonstrates {}",
#     "the chest x-ray shows {}",
# ]

TEMPLATES = [
    # Original ones
    "there is {}",
    "{} is present",
    "the lungs show {}",
    "findings consistent with {}",
    "evidence of {}",
    "chest radiograph demonstrates {}",
    "the chest x-ray shows {}",

    # More radiology-specific
    # "{}",                                          # keep original
    # "bilateral {}",
    # "left {}",
    # "right {}",
    # "mild {}",
    # "moderate {}",
    # "severe {}",
    # "small {}",
    # "large {}",
    # "acute {}",
    # "chronic {}",
    # "new {}",
    # "the patient has {}",
    # "impression: {}",
    # "there is a {}",
    # "there are {}",
    # "no {}",                                       # negative findings
    # "no evidence of {}",
    # "without {}",
    # "lung fields show {}",
    # "radiograph shows {}",
    # "x-ray demonstrates {}",
    # "noted {}",
    # "visualized {}",
]

# ── Phrases that already sound like sentences (don't template these) ─────────
ALREADY_SENTENCE_PREFIXES = (
    "there is", "there are", "the lungs", "findings",
    "evidence of", "chest", "no ", "normal ", "clear lungs",
)


def is_already_sentence(phrase: str) -> bool:
    p = phrase.lower().strip()
    return any(p.startswith(prefix) for prefix in ALREADY_SENTENCE_PREFIXES)


def expand_kb(
    input_json: str,
    output_json: str,
    templates: list = TEMPLATES,
):
    with open(input_json, "r") as f:
        phrases = json.load(f)

    print(f"Input phrases: {len(phrases)}")

    expanded = []
    seen = set()

    for phrase in phrases:
        phrase = phrase.lower().strip()

        # Always keep original phrase
        if phrase not in seen:
            seen.add(phrase)
            expanded.append(phrase)

        # Skip templating if already sentence-like
        if is_already_sentence(phrase):
            continue

        # Apply templates
        for template in templates:
            sentence = template.format(phrase)
            if sentence not in seen:
                seen.add(sentence)
                expanded.append(sentence)

    print(f"Expanded phrases: {len(expanded)}")

    with open(output_json, "w") as f:
        json.dump(expanded, f, indent=2)

    print(f"Saved to: {output_json}")
    return expanded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_json = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_raw_knowledge_bank.json"
    args.output_json = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/mimic_knowledge_bank.json"

    expand_kb(
        input_json=args.input_json,
        output_json=args.output_json
    )

# python3 data/knowledge_banks/mimic/prep_scripts/pre_cxr_clip_style.py