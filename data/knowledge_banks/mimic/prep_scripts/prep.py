#!/usr/bin/env python3
"""Extract short radiology phrases from Hugging Face MIMIC-style report datasets.

This script reads report text from `findings` and `impression` columns, applies
lightweight rule-based cleaning, and writes a flat knowledge bank as:

1. a JSON array of strings
2. a text file with one phrase per line

Example:
    python extract_mimic_report_phrases.py \
      --dataset-name TNMayo/mimic-cxr \
      --split train \
      --output-json mimic_phrases.json \
      --output-txt mimic_phrases.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Iterable, List, Sequence


RADIOLOGY_KEYWORDS = {
    "abdomen",
    "airspace",
    "apical",
    "aspiration",
    "atelectasis",
    "basal",
    "basilar",
    "bilateral",
    "blunting",
    "bronchovascular",
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
    "densities",
    "density",
    "device",
    "diaphragm",
    "diaphragmatic",
    "diffuse",
    "effusion",
    "effusions",
    "edema",
    "elevation",
    "emphysema",
    "enlarged",
    "fibrosis",
    "fissural",
    "focal",
    "fracture",
    "glass",
    "granuloma",
    "heart",
    "hemidiaphragm",
    "hilum",
    "hilar",
    "hyperinflation",
    "hyperinflated",
    "infiltrate",
    "infiltrates",
    "interstitial",
    "lesion",
    "loculated",
    "low",
    "lower",
    "lucency",
    "lung",
    "lungs",
    "mass",
    "mediastinal",
    "mediastinum",
    "metastases",
    "nodule",
    "nodules",
    "opacity",
    "opacities",
    "patchy",
    "pleura",
    "pleural",
    "pneumonia",
    "pneumothorax",
    "process",
    "prominence",
    "pulmonary",
    "reticular",
    "rib",
    "right",
    "round",
    "scarring",
    "shift",
    "silhouette",
    "small",
    "space",
    "sternotomy",
    "streaky",
    "subsegmental",
    "thickening",
    "trachea",
    "tracheal",
    "tube",
    "tubes",
    "unilateral",
    "upper",
    "vascular",
    "volume",
    "volumes",
    "widening",
    "wire",
    "wires",
}

BAD_SUBSTRINGS = {
    "clinical correlation",
    "recommend",
    "recommended",
    "follow-up",
    "followup",
    "history of",
    "evaluation of",
    "comparison to",
    "compared with",
    "comparison with",
    "prior study",
    "previous study",
    "study performed",
    "unchanged",
    "improved",
    "worsened",
    "increased from prior",
    "decreased from prior",
    "please note",
}

UNCERTAINTY_WORDS = {
    "may",
    "might",
    "possibly",
    "possible",
    "suggesting",
    "suggestive",
    "questionable",
    "probable",
    "likely",
    "cannot",
    "could",
}

LEADING_FILLER_PATTERNS = [
    r"^there is\s+",
    r"^there are\s+",
    r"^the lungs are\s+",
    r"^lungs are\s+",
    r"^the heart is\s+",
    r"^heart is\s+",
    r"^findings (are )?consistent with\s+",
    r"^evidence of\s+",
    r"^no evidence of\s+",
    r"^again seen\s+",
    r"^redemonstration of\s+",
    r"^redemonstrated\s+",
    r"^noted is\s+",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset-name", required=True, help="Hugging Face dataset name")
    # parser.add_argument("--config", default=None, help="Optional dataset config/subset")
    # parser.add_argument("--split", default="train", help="Dataset split to read")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["findings", "impression"],
        help="Text columns to mine",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum frequency required to keep a phrase",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Minimum number of words in a kept phrase",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=7,
        help="Maximum number of words in a kept phrase",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows for quick tests",
    )
    # parser.add_argument(
    #     "--output-json",
    #     default="mimic_report_phrases.json",
    #     help="Output JSON array path",
    # )
    # parser.add_argument(
    #     "--output-txt",
    #     default="mimic_report_phrases.txt",
    #     help="Output text file path",
    # )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[_/]", " ", text)
    text = re.sub(r"[^a-z0-9,;:.+\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_clauses(text: str) -> List[str]:
    chunks = re.split(r"[.;:\n]+", text)
    clauses: List[str] = []
    for chunk in chunks:
        parts = re.split(r"\s*,\s*|\s+and\s+|\s+with\s+", chunk)
        clauses.extend(parts)
    return [c.strip(" ,-") for c in clauses if c.strip(" ,-")]


def clean_clause(clause: str) -> str:
    clause = re.sub(r"\bcompared (to|with)\b.*$", "", clause).strip()
    clause = re.sub(r"\bin comparison to\b.*$", "", clause).strip()
    for pattern in LEADING_FILLER_PATTERNS:
        clause = re.sub(pattern, "", clause).strip()
    clause = re.sub(r"\bno evidence of\b", "no", clause)
    clause = re.sub(r"\bwithout evidence of\b", "without", clause)
    clause = re.sub(r"\s+", " ", clause).strip(" ,-")
    return clause


def is_good_phrase(phrase: str, min_words: int, max_words: int) -> bool:
    if not phrase:
        return False
    if any(bad in phrase for bad in BAD_SUBSTRINGS):
        return False
    words = phrase.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if any(word in UNCERTAINTY_WORDS for word in words):
        return False
    if not any(word in RADIOLOGY_KEYWORDS for word in words):
        return False
    if re.fullmatch(r"[0-9 .+-]+", phrase):
        return False
    return True


def shrink_phrase(phrase: str, max_words: int) -> str:
    words = phrase.split()
    if len(words) <= max_words:
        return phrase

    keyword_positions = [i for i, word in enumerate(words) if word in RADIOLOGY_KEYWORDS]
    if not keyword_positions:
        return " ".join(words[:max_words])

    center = keyword_positions[-1]
    start = max(0, center - (max_words - 1))
    end = min(len(words), start + max_words)
    start = max(0, end - max_words)
    return " ".join(words[start:end])


def extract_phrases(text: str, min_words: int, max_words: int) -> Iterable[str]:
    normalized = normalize_text(text)
    for clause in split_into_clauses(normalized):
        clause = clean_clause(clause)
        if not clause:
            continue

        phrase = shrink_phrase(clause, max_words)
        if is_good_phrase(phrase, min_words, max_words):
            yield phrase

        words = phrase.split()
        for size in range(min(max_words, len(words)), min_words - 1, -1):
            for start in range(0, len(words) - size + 1):
                ngram = " ".join(words[start : start + size])
                if is_good_phrase(ngram, min_words, max_words):
                    yield ngram


def unique_in_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output




def main() -> None:
    args = parse_args()

    args.dataset_name = "itsanmolgupta/mimic-cxr-dataset"
    args.split = "train"
    args.output_json = "mimic_report_phrases.json"
    args.output_txt = "mimic_report_phrases.txt"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "This script requires the `datasets` package. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    phrase_counts: Counter[str] = Counter()

    for row in dataset:
        for column in args.columns:
            value = row.get(column)
            if not isinstance(value, str) or not value.strip():
                continue
            phrase_counts.update(extract_phrases(value, args.min_words, args.max_words))

    phrases = [
        phrase
        for phrase, count in sorted(
            phrase_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= args.min_count
    ]
    phrases = unique_in_order(phrases)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(phrases, f, indent=2)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for phrase in phrases:
            f.write(phrase + "\n")

    print(f"Rows processed: {len(dataset)}")
    print(f"Phrases kept: {len(phrases)}")
    print(f"JSON output: {args.output_json}")
    print(f"TXT output: {args.output_txt}")


if __name__ == "__main__":
    main()


# python3 data/knowledge_banks/mimic/prep.py 