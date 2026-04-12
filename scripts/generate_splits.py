"""
Generate stratified train / val / test split CSVs from the processed dataset.

Scans data/processed/{real,fake}/ and writes three CSVs to data/splits/:
    train.csv, val.csv, test.csv

Each CSV has columns: filepath (absolute), label (0=spoof, 1=live), split.

Usage:
    python scripts/generate_splits.py \
        --processed-root data/processed \
        --output-dir data/splits \
        --train 0.70 --val 0.15 --test 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASS_IDX

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_samples(processed_root: Path) -> pd.DataFrame:
    """Walk processed_root/{real,fake}/ and collect (filepath, label) pairs."""
    rows = []
    for cls_name, label in [("real", CLASS_IDX["live"]), ("fake", CLASS_IDX["spoof"])]:
        cls_dir = processed_root / cls_name
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} does not exist — skipping.")
            continue
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        for p in imgs:
            rows.append({"filepath": str(p.resolve()), "label": label})
    return pd.DataFrame(rows)


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split: train / val / test."""
    test_frac = 1.0 - train_frac - val_frac

    # First split: train vs rest
    train_df, rest_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: val vs test from 'rest'
    relative_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=(1.0 - relative_val),
        stratify=rest_df["label"],
        random_state=seed,
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def print_distribution(name: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index()
    label_map = {v: k for k, v in CLASS_IDX.items()}
    parts = [f"{label_map.get(lbl, lbl)}={cnt}" for lbl, cnt in counts.items()]
    print(f"  {name:8s}: {len(df):6d} samples  ({', '.join(parts)})")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test split CSVs")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir",     type=Path, default=Path("data/splits"))
    parser.add_argument("--train", type=float, default=0.70, help="Train fraction")
    parser.add_argument("--val",   type=float, default=0.15, help="Val fraction")
    parser.add_argument("--test",  type=float, default=0.15, help="Test fraction (derived)")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "train + val + test must sum to 1.0"

    # Resolve relative paths from repo root
    repo_root = Path(__file__).resolve().parent.parent
    processed_root = args.processed_root if args.processed_root.is_absolute() \
        else repo_root / args.processed_root
    output_dir = args.output_dir if args.output_dir.is_absolute() \
        else repo_root / args.output_dir

    print(f"Scanning {processed_root} ...")
    df = collect_samples(processed_root)

    if df.empty:
        print("No images found. Run preprocess_dataset.py first.")
        sys.exit(1)

    print(f"\nTotal samples: {len(df)}")
    print_distribution("all", df)

    train_df, val_df, test_df = split_dataframe(
        df, args.train, args.val, args.seed
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df["split"] = split_name

    print("\nSplit distribution:")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print_distribution(name, sdf)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(  output_dir / "val.csv",   index=False)
    test_df.to_csv( output_dir / "test.csv",  index=False)

    print(f"\nSplit CSVs written to: {output_dir}")


if __name__ == "__main__":
    main()
