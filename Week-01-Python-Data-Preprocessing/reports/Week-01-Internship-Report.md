# Week 01 Internship Report
## Python for Machine Learning & Data Preprocessing

**Internship:** AI Pioneers — Machine Learning Internship  
**Date:** 15 November 2026

## Executive summary

This practical develops a reproducible Python workflow for preparing banking campaign data for analysis and later machine learning work. It covers data acquisition, inspection, cleaning, missing-value treatment, duplicate removal, exploratory data analysis, feature selection, categorical encoding, numerical scaling, and a stratified training/test split. The workflow is implemented as reusable Python modules, command-line scripts, and a Jupyter notebook.

## 1. Objectives

- Use Python, NumPy, and Pandas to work with structured data.
- Inspect dataset dimensions, types, descriptive statistics, and missing values.
- Clean blank values, handle missing observations, and remove exact duplicates.
- Explore distributions and report possible outliers.
- Select suitable features, encode categorical attributes, and scale numerical values.
- Export a cleaned dataset and model-ready feature matrices.

## 2. Dataset

The workflow is designed around the UCI Machine Learning Repository Bank Marketing dataset (ID 222), using the full additional CSV file. The dataset describes direct telephone marketing campaigns of a Portuguese banking institution and whether a customer subscribed to a term deposit. The official full additional dataset contains 41,188 records, 20 input attributes, and the target y.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing  
DOI: https://doi.org/10.24432/C5K306  
License: CC BY 4.0. Cite Moro, Rita, and Cortez (2014).

A small synthetic fixture is included for offline smoke testing. It is not UCI source data or actual customer information. Raw official data is downloaded at runtime and not committed.

## 3. Tools

Python 3, NumPy, Pandas, Matplotlib, scikit-learn, Jupyter Notebook, and pytest.

## 4. Methodology

1. Load the official semicolon-delimited CSV and inspect records, columns, types, missing values, duplicates, and descriptive statistics.
2. Trim whitespace in text fields and convert blank strings to missing values.
3. Impute numeric nulls using the column median. Fill genuinely missing categorical cells with the explicit value unknown; retain the dataset's existing unknown tokens as category values.
4. Count and remove exact duplicate rows.
5. Calculate IQR outlier bounds and report possible outliers. Do not automatically delete them because extreme values may be legitimate observations.
6. Separate target y from inputs. Exclude duration from model-ready input features for a pre-call use case because call duration is only known after the call and may cause target leakage.
7. One-hot encode categorical fields using an encoder that tolerates unseen test categories.
8. Standardize numerical inputs.
9. Create a stratified 80/20 train/test split. Fit imputers, encoder, and scaler on training data only, then apply them to test data.

## 5. Outputs

- Cleaned source-scale data.
- Per-column data-quality and missing-value report.
- IQR outlier summary and descriptive statistics.
- Numeric correlation matrix and EDA charts.
- One-hot encoded and scaled training/test feature matrices.
- Training/test target labels and fitted preprocessing object.
- JSON manifest and complete run summary.

## 6. Validation

The code and notebook were smoke-tested using the synthetic fixture. That fixture contains 120 illustrative rows; one exact duplicate was removed, leaving 119 rows. The demonstration run generated a stratified 95-row training set and 24-row test set with 52 encoded features. These values validate the pipeline only. They are not results from the UCI full dataset. Run the documented download and processing commands to create the UCI-derived output totals.

## 7. Learning outcomes

- Practical use of NumPy arrays and vectorized calculations.
- Pandas DataFrame inspection, filtering, cleaning, and export.
- Understanding blanks, null values, and explicit category tokens.
- Duplicate detection and IQR-based outlier review.
- Feature selection and target leakage awareness.
- Categorical encoding, numerical scaling, and stratified split design.
- Understanding why preprocessing must be fitted using training data only.

## 8. Conclusion

Week 1 establishes the data-preparation foundation for subsequent machine learning tasks. The project converts a source banking dataset into repeatable cleaned data, quality reports, EDA outputs, and encoded/scaled train and test matrices. Separating reusable code, notebook explanations, raw inputs, generated outputs, and documentation makes the work reproducible and reviewable. This task does not train or evaluate a classifier.

## References

1. Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
2. UCI Machine Learning Repository. Bank Marketing dataset, ID 222. https://archive.ics.uci.edu/dataset/222/bank+marketing
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
