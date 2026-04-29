#!/usr/bin/env python3
"""Clean a raw flat list of extracted MIMIC report phrases.

Input:
    JSON array of strings

Output:
    1. cleaned JSON array
    2. cleaned TXT file
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STOP_EDGE_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "within",
}

BAD_ANYWHERE = {
    "again",
    "appears",
    "clinical correlation",
    "compared",
    "comparison",
    "concerning for",
    "could represent",
    "follow-up",
    "followup",
    "history of",
    "if clinically indicated",
    "improved",
    "improvement",
    "increased from prior",
    "is again",
    "may represent",
    "new",
    "no significant change",
    "not significantly changed",
    "persistent",
    "persists",
    "possible",
    "possibly",
    "prior",
    "questionable",
    "recommend",
    "recommended",
    "re-demonstrated",
    "redemonstrated",
    "slightly improved",
    "suggesting",
    "suggestive of",
    "unchanged",
    "versus",
    "worsening",
}

DEVICE_WORDS = {
    "catheter",
    "device",
    "devices",
    "lead",
    "leads",
    "line",
    "lines",
    "pacemaker",
    "picc",
    "port",
    "tube",
    "tubes",
    "wire",
    "wires",
}

KEEP_DEVICE_PHRASES = {
    "sternotomy wires",
}

GOOD_KEYWORDS = {
    "airspace",
    "apical",
    "aspiration",
    "atelectasis",
    "basal",
    "basilar",
    "bilateral",
    "blunting",
    "calcified",
    "calcification",
    "cardiac",
    "cardiomegaly",
    "cardiomediastinal",
    "cavitary",
    "clear",
    "collapse",
    "congestion",
    "consolidation",
    "costophrenic",
    "density",
    "densities",
    "diaphragm",
    "diaphragmatic",
    "diffuse",
    "edema",
    "effusion",
    "effusions",
    "elevated",
    "elevation",
    "emphysema",
    "enlarged",
    "fibrosis",
    "focal",
    "glass",
    "ground",
    "heart",
    "hemidiaphragm",
    "hilar",
    "hyperinflation",
    "hyperinflated",
    "infiltrate",
    "interstitial",
    "lesion",
    "low",
    "lower",
    "lucency",
    "lung",
    "lungs",
    "mass",
    "mediastinal",
    "mediastinum",
    "nodule",
    "nodules",
    "opacity",
    "opacities",
    "patchy",
    "pleural",
    "pneumonia",
    "pneumothorax",
    "process",
    "prominence",
    "pulmonary",
    "reticular",
    "rib",
    "right",
    "scarring",
    "shift",
    "silhouette",
    "small",
    "space",
    "thickening",
    "tracheal",
    "trachea",
    "unilateral",
    "upper",
    "vascular",
    "volume",
    "volumes",
    "widening",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input-json", required=True)
    # parser.add_argument("--output-json", default="mimic_report_phrases.cleaned.json")
    # parser.add_argument("--output-txt", default="mimic_report_phrases.cleaned.txt")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=7)
    parser.add_argument(
        "--keep-devices",
        action="store_true",
        help="Keep device/tube phrases instead of filtering them out",
    )
    return parser.parse_args()


def normalize(phrase: str) -> str:
    phrase = phrase.lower().strip()
    phrase = re.sub(r"[^a-z0-9+\-\s]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def trim_edge_stopwords(words: list[str]) -> list[str]:
    while words and words[0] in STOP_EDGE_WORDS:
        words = words[1:]
    while words and words[-1] in STOP_EDGE_WORDS:
        words = words[:-1]
    return words


def is_bad_phrase(phrase: str, min_words: int, max_words: int, keep_devices: bool) -> bool:
    words = phrase.split()
    if len(words) < min_words or len(words) > max_words:
        return True

    if any(token.isdigit() for token in words):
        return True

    if not any(word in GOOD_KEYWORDS for word in words):
        return True

    if any(bad in phrase for bad in BAD_ANYWHERE):
        return True

    if words[0] in STOP_EDGE_WORDS or words[-1] in STOP_EDGE_WORDS:
        return True

    if not keep_devices and phrase not in KEEP_DEVICE_PHRASES:
        if any(word in DEVICE_WORDS for word in words):
            return True

    if len(words) == 2 and all(word in {"left", "right", "upper", "lower", "mild", "small", "the", "no"} for word in words):
        return True

    if phrase.endswith("is") or phrase.endswith("are"):
        return True

    return False


def clean_phrase(phrase: str) -> str | None:
    phrase = normalize(phrase)
    words = trim_edge_stopwords(phrase.split())
    if not words:
        return None
    phrase = " ".join(words)
    phrase = re.sub(r"\bno evidence of\b", "no", phrase)
    phrase = re.sub(r"\bwithout evidence of\b", "without", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase or None


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    args = parse_args()

    args.input_json = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/raw/mimic_report_phrases.json"
    args.output_json = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/raw/mimic_report_phrases_cleaned.json"
    args.output_txt = "/shared/scratch/0/home/v_neelesh_bisht/projects/InvDiff/data/knowledge_banks/mimic/raw/mimic_report_phrases_cleaned.txt"

    input_path = Path(args.input_json)
    phrases = json.loads(input_path.read_text(encoding="utf-8"))

    cleaned = []
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        phrase = clean_phrase(phrase)
        if not phrase:
            continue
        if is_bad_phrase(phrase, args.min_words, args.max_words, args.keep_devices):
            continue
        cleaned.append(phrase)

    cleaned = dedupe_preserve_order(cleaned)

    Path(args.output_json).write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    Path(args.output_txt).write_text("\n".join(cleaned) + "\n", encoding="utf-8")

    print(f"Input phrases: {len(phrases)}")
    print(f"Cleaned phrases: {len(cleaned)}")
    print(f"JSON output: {args.output_json}")
    print(f"TXT output: {args.output_txt}")


if __name__ == "__main__":
    main()
