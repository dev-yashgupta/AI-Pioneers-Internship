from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import download_and_extract, load_bank_marketing
from src.preprocessing import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 1 Bank Marketing preprocessing workflow.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional local semicolon-delimited CSV.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "full_dataset")
    args = parser.parse_args()
    dataset_path = args.data_path or download_and_extract(ROOT / "data" / "raw")
    frame = load_bank_marketing(dataset_path)
    summary = run_pipeline(frame, args.output_dir)
    print(json.dumps({
        "dataset": str(dataset_path), "output_dir": str(args.output_dir),
        "rows_after_cleaning": summary["cleaning"]["rows_after_duplicate_removal"],
        "duplicates_removed": summary["cleaning"]["duplicate_rows_removed"],
        "train_rows": summary["feature_preparation"]["train_rows"],
        "test_rows": summary["feature_preparation"]["test_rows"],
        "encoded_features": summary["feature_preparation"]["encoded_feature_count"]
    }, indent=2))


if __name__ == "__main__":
    main()
