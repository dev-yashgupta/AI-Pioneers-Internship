# Week 01 — Python for Machine Learning & Data Preprocessing

This folder contains the Week 01 practical implementation, notebook, tests, synthetic demo outputs, and report materials.

## Dataset

The workflow is designed for the UCI **Bank Marketing** dataset (dataset ID 222), using `bank-additional-full.csv` (41,188 records, 20 inputs and target `y`). The data describes telephone marketing campaigns of a Portuguese banking institution and whether a client subscribed to a term deposit.

- Official page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Download archive: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- DOI: https://doi.org/10.24432/C5K306
- License: CC BY 4.0. Cite Moro, Rita, and Cortez (2014).

The `data/sample/` fixture is **synthetic demonstration data** for offline testing. It is not UCI data or real customer data. The official CSV is downloaded on first run and is not committed.

## Setup and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/run_pipeline.py
```

Run offline using the synthetic fixture:

```bash
python scripts/run_pipeline.py --data-path data/sample/bank_marketing_demo_synthetic.csv --output-dir outputs/demo
pytest -q tests
```

Open the notebook with `jupyter notebook notebooks/Week_01_Python_Data_Preprocessing.ipynb`.

## Preprocessing decisions

1. Inspect shape, data types, nulls, duplicate rows, and descriptive statistics.
2. Trim text; convert blank strings into missing values.
3. Impute missing numerical values with the median and missing categorical values with `unknown`; retain existing `unknown` tokens as category values.
4. Remove exact duplicate rows and record the count.
5. Report possible outliers using IQR bounds; do not automatically delete them.
6. Keep target `y` separate. Exclude `duration` from pre-call feature inputs because it is only known after the call and could leak target information.
7. One-hot encode categorical features and standardize numeric features.
8. Create a stratified 80/20 train/test split. Fit preprocessing transformations on training data only.

The task prepares data only; it does not train or evaluate a classifier.

## Outputs

- `cleaned_bank_marketing.csv` — cleaned source-scale data
- `data_quality_report.csv` — missing values and data types
- `outlier_summary.csv` — IQR outlier counts and limits
- `descriptive_statistics.csv` and `numeric_correlation.csv`
- `figures/` — age, target, and correlation plots
- `model_ready/X_train_processed.csv`, `X_test_processed.csv`
- `model_ready/y_train.csv`, `y_test.csv`
- `model_ready/preprocessor.joblib` and `preprocessing_manifest.json`
- `run_summary.json`

## Dataset citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
