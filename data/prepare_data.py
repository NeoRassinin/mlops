"""Generate a small, balanced sentiment-classification dataset (CSV).

This is an MLOps lab, not a research task, so the data is synthesised from
templates + sentiment vocab. It is fully offline and deterministic (fixed seed),
which keeps the dataset version stable and reproducible across runs.

Output: data/sentiment.csv  with columns: text,label   (label in {positive, negative})
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

POSITIVE_WORDS = [
    "amazing", "excellent", "fantastic", "wonderful", "great", "brilliant",
    "superb", "delightful", "outstanding", "perfect", "enjoyable", "awesome",
    "pleasant", "impressive", "lovely", "satisfying", "happy", "remarkable",
]
NEGATIVE_WORDS = [
    "terrible", "awful", "horrible", "disappointing", "boring", "useless",
    "broken", "poor", "frustrating", "annoying", "mediocre", "unpleasant",
    "sluggish", "buggy", "overpriced", "dreadful", "clumsy", "underwhelming",
]

POSITIVE_TEMPLATES = [
    "This product is absolutely {w1}, I would buy it again.",
    "What a {w1} experience, the service was {w2} too.",
    "I loved this movie, the acting was {w1} and {w2}.",
    "The food was {w1} and the staff were {w2}.",
    "Truly {w1}! Everything about it felt {w2}.",
    "An {w1} purchase, totally {w2} and worth every cent.",
    "I am so happy with the results, simply {w1}.",
    "Five stars, a {w1} and {w2} place to visit.",
]
NEGATIVE_TEMPLATES = [
    "This product is absolutely {w1}, I want a refund.",
    "What a {w1} experience, the service was {w2} too.",
    "I disliked this movie, the plot was {w1} and {w2}.",
    "The food was {w1} and the staff were {w2}.",
    "Truly {w1}. Everything about it felt {w2}.",
    "An {w1} purchase, totally {w2} and a waste of money.",
    "I am so let down, it was {w1} and {w2}.",
    "One star, a {w1} and {w2} place to avoid.",
]


def make_rows(n_per_class: int, rng: random.Random,
              noise: float = 0.0) -> list[tuple[str, str]]:
    """Build balanced rows. ``noise`` fraction of labels are randomly flipped so the
    task is not perfectly separable - this makes different hyperparameters produce
    different metrics (required for the lab's 'differences in metrics')."""
    rows: list[tuple[str, str]] = []
    for _ in range(n_per_class):
        t = rng.choice(POSITIVE_TEMPLATES)
        rows.append((t.format(w1=rng.choice(POSITIVE_WORDS),
                              w2=rng.choice(POSITIVE_WORDS)), "positive"))
        t = rng.choice(NEGATIVE_TEMPLATES)
        rows.append((t.format(w1=rng.choice(NEGATIVE_WORDS),
                              w2=rng.choice(NEGATIVE_WORDS)), "negative"))
    if noise > 0:
        for i, (text, label) in enumerate(rows):
            if rng.random() < noise:
                rows[i] = (text, "negative" if label == "positive" else "positive")
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=1000,
                    help="rows per class (total = 2x this)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.12,
                    help="fraction of labels to randomly flip (keeps task non-trivial)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "sentiment.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = make_rows(args.n_per_class, rng, noise=args.noise)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
