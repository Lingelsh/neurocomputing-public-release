"""Generic dataset preparation script.

Usage:
    python scripts/prepare_dataset.py --input <path_to_csv> --output <output_dir>

Expected CSV format:
    'idx', 'func', 'target'
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    
    # Required columns validation
    required = {"idx", "func", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    # Simple split: 80% train, 10% valid, 10% test
    # In a real scenario, use more robust splitting (e.g., by project or commit)
    df["split"] = np.random.choice(["train", "valid", "test"], size=len(df), p=[0.8, 0.1, 0.1])

    # Save splits
    for split in ["train", "valid", "test"]:
        part = df[df["split"] == split].drop(columns=["split"])
        part.to_json(out_dir / f"{split}.jsonl", orient="records", lines=True)
        print(f"Saved {split} set to {out_dir / f'{split}.jsonl'} ({len(part)} samples)")

if __name__ == "__main__":
    main()
