import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, REVIEWS_PATH
from src.preprocessing import clean_text


TIME_PATTERNS = [
    "reorder",
    "re-order",
    "restock",
    "repurchase",
    "buy it again",
    "buy again",
    "order it again",
    "order again",
    "third bottle",
    "fourth bottle",
    "second bottle",
    "every couple of months",
    "every few months",
    "every other month",
    "every month",
    "monthly",
    "weekly",
    "bi-monthly",
    "part of my routine",
    "my routine",
    "when it runs out",
    "when i run out",
    "keep coming back",
    "keep buying",
    "regularly",
    "refill",
    "refilled",
]


def _matches_time_patterns(text: str) -> bool:
    lower = text.lower()
    if any(pat in lower for pat in TIME_PATTERNS):
        return True
    return False


def _iter_timebased_reviews(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if not _matches_time_patterns(text):
                continue

            user_id = obj.get("user_id")
            parent_asin = obj.get("parent_asin")
            if not user_id or not parent_asin:
                continue

            cleaned = clean_text(text)
            if not cleaned:
                continue

            yield {
                "user_id": user_id,
                "parent_asin": parent_asin,
                "text": text,
                "clean_text": cleaned,
                "label_intent": "",
            }


def _reservoir_sample(iterator, sample_size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    total = 0
    for idx, item in enumerate(iterator):
        if idx < sample_size:
            sample.append(item)
        else:
            j = rng.randint(0, idx)
            if j < sample_size:
                sample[j] = item
        total = idx + 1
    return sample, total


def main():
    parser = argparse.ArgumentParser(
        description="Sample time-based reorder/routine reviews for labeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(REVIEWS_PATH),
        help="Path to the All_Beauty.jsonl reviews file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DATA_DIR) / "unlabeled_timebased_1000.csv",
        help="Output CSV path for the unlabeled sample.",
    )
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of reviews to sample.")
    parser.add_argument(
        "--random-state",
        dest="random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input file: {args.input}")

    records, eligible = _reservoir_sample(
        _iter_timebased_reviews(args.input),
        args.sample_size,
        int(args.random_state),
    )
    if len(records) < args.sample_size:
        raise SystemExit(
            f"Only found {len(records)} eligible reviews (needed {args.sample_size})."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} rows to {args.output}")
    print(f"Eligible reviews scanned: {eligible}")


if __name__ == "__main__":
    main()
