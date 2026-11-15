"""Reusable cleaning, EDA, feature preparation, and export functions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "y"
LEAKAGE_COLUMNS = ["duration"]


def normalize_text(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().replace("", pd.NA)
    return cleaned


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if TARGET not in frame.columns:
        raise ValueError(f"Target column '{TARGET}' is required")
    original_rows = len(frame)
    data = normalize_text(frame)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    missing_before = data.isna().sum().astype(int)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in data.columns if c not in numeric_columns]
    for column in numeric_columns:
        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())
    for column in categorical_columns:
        if data[column].isna().any():
            data[column] = data[column].fillna("unknown")
    data[TARGET] = data[TARGET].astype("string").str.lower().str.strip()
    unexpected = sorted(set(data[TARGET].dropna().unique()) - {"yes", "no"})
    if unexpected:
        raise ValueError(f"Unexpected target labels found: {unexpected}")
    summary = {
        "rows_before": int(original_rows),
        "rows_after_duplicate_removal": int(len(data)),
        "duplicate_rows_removed": duplicate_count,
        "columns": int(data.shape[1]),
        "missing_values_before_imputation": int(missing_before.sum()),
        "missing_values_after_imputation": int(data.isna().sum().sum()),
        "missing_by_column_before_imputation": missing_before.to_dict(),
    }
    return data, summary


def outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number])
    rows = []
    for column in numeric.columns:
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((numeric[column] < lower) | (numeric[column] > upper)).sum())
        rows.append({
            "feature": column, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_bound": float(lower), "upper_bound": float(upper),
            "outlier_count": count, "outlier_percent": round(100 * count / max(len(numeric), 1), 3)
        })
    return pd.DataFrame(rows)


def build_eda(frame: pd.DataFrame, figures_dir: str | Path) -> dict[str, Any]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame.describe(include="all").transpose().to_csv(figures_dir.parent / "descriptive_statistics.csv")
    if "age" in frame:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        frame["age"].dropna().plot(kind="hist", bins=25, ax=ax, edgecolor="white")
        ax.set(title="Customer age distribution", xlabel="Age", ylabel="Number of records")
        fig.tight_layout()
        fig.savefig(figures_dir / "age_distribution.png", dpi=160)
        plt.close(fig)
    if TARGET in frame:
        fig, ax = plt.subplots(figsize=(6, 4))
        frame[TARGET].value_counts().plot(kind="bar", ax=ax)
        ax.set(title="Term-deposit subscription response", xlabel="Subscription (y)", ylabel="Number of records")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(figures_dir / "target_distribution.png", dpi=160)
        plt.close(fig)
    numeric = frame.select_dtypes(include=[np.number])
    if len(numeric.columns) >= 2:
        corr = numeric.corr(numeric_only=True)
        corr.to_csv(figures_dir.parent / "numeric_correlation.csv")
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(corr, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        ax.set_title("Numeric feature correlation matrix")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / "numeric_correlation_heatmap.png", dpi=160)
        plt.close(fig)
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "target_counts": frame[TARGET].value_counts().to_dict() if TARGET in frame else {},
        "numeric_features": numeric.columns.tolist(),
        "categorical_features": frame.select_dtypes(exclude=[np.number]).columns.tolist(),
    }


def prepare_model_features(frame: pd.DataFrame, output_dir: str | Path, random_state: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if TARGET not in frame:
        raise ValueError("Target column 'y' is required")
    excluded = [c for c in LEAKAGE_COLUMNS if c in frame]
    features = frame.drop(columns=excluded + [TARGET], errors="ignore")
    target = frame[TARGET].map({"no": 0, "yes": 1})
    if target.isna().any():
        raise ValueError("Target contains values other than yes/no")
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in features.columns if c not in numeric_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target.astype(int), test_size=0.20, random_state=random_state,
        stratify=target.astype(int)
    )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ], remainder="drop", verbose_feature_names_out=False)
    train_values = transformer.fit_transform(X_train)
    test_values = transformer.transform(X_test)
    names = transformer.get_feature_names_out().tolist()
    pd.DataFrame(train_values, columns=names, index=X_train.index).to_csv(output_dir / "X_train_processed.csv", index=False)
    pd.DataFrame(test_values, columns=names, index=X_test.index).to_csv(output_dir / "X_test_processed.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=[TARGET])
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=[TARGET])
    joblib.dump(transformer, output_dir / "preprocessor.joblib")
    manifest = {
        "target": TARGET, "target_mapping": {"no": 0, "yes": 1},
        "excluded_from_features": excluded,
        "numeric_input_features": numeric_columns,
        "categorical_input_features": categorical_columns,
        "encoded_feature_count": len(names), "encoded_feature_names": names,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "test_size": 0.20, "random_state": random_state,
        "fit_policy": "Imputers, one-hot encoder and scaler are fitted on training data only."
    }
    (output_dir / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_pipeline(data: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned, cleaning_summary = clean_data(data)
    cleaned.to_csv(output_dir / "cleaned_bank_marketing.csv", index=False)
    quality = pd.DataFrame({
        "column": list(cleaning_summary["missing_by_column_before_imputation"]),
        "missing_before_imputation": list(cleaning_summary["missing_by_column_before_imputation"].values()),
        "missing_after_imputation": [int(cleaned[c].isna().sum()) for c in cleaned.columns],
        "dtype_after_cleaning": [str(cleaned[c].dtype) for c in cleaned.columns]
    })
    quality.to_csv(output_dir / "data_quality_report.csv", index=False)
    outlier_summary(cleaned).to_csv(output_dir / "outlier_summary.csv", index=False)
    eda = build_eda(cleaned, output_dir / "figures")
    manifest = prepare_model_features(cleaned, output_dir / "model_ready")
    summary = {
        "cleaning": cleaning_summary, "eda": eda, "feature_preparation": manifest,
        "note": "Outliers are reported, not removed automatically. duration is excluded from features for the pre-call prediction use case."
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
