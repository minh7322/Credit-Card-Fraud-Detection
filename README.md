# Credit Card Fraud Detection

Machine Learning project for credit card fraud detection using XGBoost, time-sorted chunk evaluation, and drift analysis.

## Project Overview

This project analyzes a large-scale credit card transaction dataset and builds a fraud detection pipeline focused on class imbalance, feature selection, XGBoost modeling, threshold tuning, and cross-chunk drift evaluation.

Key points:

- The dataset is highly imbalanced, with fraud around 1% of transactions.
- Chunk 01 is used for detailed EDA, feature engineering, and model tuning.
- Chunks 02-16 are used for future-period validation and drift analysis.
- XGBoost with class weighting is selected as the main operational baseline.
- Low F1 is diagnosed separately as either ranking/concept drift or threshold calibration drift.

## Main Files

- `Data_fraud.ipynb`: main notebook containing EDA, feature engineering, modeling, and analysis.
- `run_xgb_chunks_16_02_16.py`: script for evaluating tuned XGBoost models on chunks 02-16.
- `chunk_model_results_16/`: saved cross-chunk metrics and plots.
- `latex_report/`: LaTeX report source and extracted figures.
- `report_template_fraud_clean/`: clean report template package.
- `GIẢI_THÍCH_CODE_CHI_TIẾT.md`: Vietnamese explanation of the code.

## Data Note

Raw and processed data files are intentionally excluded from Git because they are too large for normal GitHub storage:

- `data.csv`
- `data_chunks_*/`
- `database/`
- `*.parquet`

To reproduce the notebook fully, place the dataset files back into the same local paths before running.

## Environment

Recommended Python packages are listed in `requirements.txt`.

## Results Summary

On chunk 01, tuned XGBoost models reach PR-AUC above 0.96 and F1 around 0.956. Cross-chunk evaluation shows that performance changes over time, with chunks 5-6 behaving like concept or label drift candidates and chunks 10-11 mainly showing threshold calibration drift.

The recommended production-style approach is:

1. Use `xgb_class_weight` as the baseline model.
2. Use dynamic thresholding from the most recent validation window.
3. Monitor PR-AUC, F1, fraud rate, and feature drift.
4. Retrain only when ranking quality drops, not merely when F1 drops.
