"""Stage 1 - create and version a ClearML Dataset.

Creates a ClearML Dataset, uploads the CSV, and finalizes a version. Prints the
dataset_id, which train.py consumes so that training pulls data from ClearML
(not from a local path).

Usage:
    python src/upload_dataset.py --version 1.0.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Dataset

PROJECT = "sentiment-mlops"
DATASET_NAME = "sentiment-reviews"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sentiment.csv")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--name", default=DATASET_NAME)
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"{args.csv} not found - run `python data/prepare_data.py` first.")

    dataset = Dataset.create(
        dataset_name=args.name,
        dataset_project=args.project,
        dataset_version=args.version,
        description="Synthetic balanced sentiment reviews (positive/negative).",
    )
    dataset.add_files(path=str(args.csv))
    dataset.add_tags(["sentiment", "text-classification", f"v{args.version}"])
    dataset.upload()
    dataset.finalize()

    print("=" * 60)
    print(f"Dataset created: {args.project}/{args.name}  v{args.version}")
    print(f"DATASET_ID = {dataset.id}")
    print("=" * 60)
    print("Use it in training, e.g.:")
    print(f"  python src/train.py --dataset-id {dataset.id} --queue students")


if __name__ == "__main__":
    main()
