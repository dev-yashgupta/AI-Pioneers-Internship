from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_data, prepare_model_features


def test_cleaning_removes_duplicate_and_handles_missing():
    frame = pd.read_csv(ROOT / "data/sample/bank_marketing_demo_synthetic.csv", sep=";")
    cleaned, summary = clean_data(frame)
    assert summary["duplicate_rows_removed"] == 1
    assert len(cleaned) == len(frame) - 1
    assert int(cleaned.isna().sum().sum()) == 0


def test_feature_preparation_excludes_duration_and_splits(tmp_path):
    frame = pd.read_csv(ROOT / "data/sample/bank_marketing_demo_synthetic.csv", sep=";")
    cleaned, _ = clean_data(frame)
    manifest = prepare_model_features(cleaned, tmp_path)
    assert "duration" in manifest["excluded_from_features"]
    assert manifest["train_rows"] + manifest["test_rows"] == len(cleaned)
    assert (tmp_path / "X_train_processed.csv").exists()
    assert (tmp_path / "X_test_processed.csv").exists()
