# Short-Horizon Return Prediction Challenge

## Competition Overview
Financial markets reward forecasting under uncertainty, not hindsight. This challenge asks participants to build a robust regression model that predicts the short-horizon percentage return of `Price` from an anonymized multivariate tabular time-series dataset.

Each row is a snapshot of system state at time `t`, including:
- Current-state feature values
- Engineered lag-difference signals from prior windows

The task is intentionally realistic: row-level samples are shuffled and raw temporal identifiers are removed, so the model must learn predictive structure directly from the feature space.

## Problem Statement
Predict:

$$
TARGET = 100 \times \frac{Price[t+H] - Price[t]}{Price[t]}
$$

Where:
- `H` is the unknown forward horizon used by the competition organizers
- `TARGET` is available only in the training split

This is a regression task.

## Evaluation Metric
Submissions are scored using $R^2$ (coefficient of determination):

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Higher is better.

## Dataset Summary
- Training rows: `661,574`
- Test rows: `410,139`
- Predictive features: `445`

### Column Layout
- `ID`: unique row identifier
- `TARGET`: regression label (train only)
- `445` predictive input features

### Feature Composition
- `112` base/current-state features
- `111` features with `_LagT1`
- `111` features with `_LagT2`
- `111` features with `_LagT3`

Special notes:
- `Price` is a feature and also defines the future-return target.
- `SO3_T` is a standalone covariate and does not have lag variants.

### Lag Feature Meaning
For any base feature `<feature>`:
- `<feature>_LagT1 = feature[t] - feature[t-T1]`
- `<feature>_LagT2 = feature[t] - feature[t-T2]`
- `<feature>_LagT3 = feature[t] - feature[t-T3]`

With `T3 > T2 > T1`.

## How To Interpret Feature Names
Feature names are anonymized lineage tags, for example:
- `S01_F01_U01`
- `S04_V19_V12`
- `SO3_T`
- `Price`
- `S03_D02_A09_A02_B07_E07_E08`

Interpretation guidance:
- Shared prefixes usually indicate related source families.
- Multiple code blocks often indicate derived or mixed lineage.
- Suffixes (`_LagT1`, `_LagT2`, `_LagT3`) indicate lag-difference transforms.

Do not assume business semantics from naming alone.

## Submission Rules
- File name must be `submission.csv`
- File must contain exactly 2 columns: `ID`, `TARGET`
- Typical format:

```csv
ID,TARGET
672374,0.0
672375,0.0
672376,0.0
```

Leaderboard behavior:
- Public leaderboard is visible during competition.
- Final ranking is based on private leaderboard.

## Workspace File Reference
This repository currently contains the following relevant files:

- `train.parquet`
	- Labeled training data (`ID`, features, `TARGET`).
- `test.parquet`
	- Unlabeled test data (`ID`, features).
- `sample_submission.csv`
	- Template submission with required column format.
- `code_2.py`
	- End-to-end experimental pipeline using LightGBM, XGBoost, CatBoost, Optuna tuning, feature selection, stacking/blending, diagnostics, and final `submission.csv` generation.
- `code_3.py`
	- Hybrid script that currently contains two parts:
	  1. A baseline placeholder pipeline with CLI arguments (`--data-dir`, `--output`, `--folds`, `--seed`, `--max-train-rows`) that trains `HistGradientBoostingRegressor` with K-Fold CV and writes a submission file.
	  2. A full advanced pipeline section (LightGBM + Optuna + feature engineering + null importance + XGBoost + CatBoost + stacking/blending + diagnostics) that writes `submission.csv`.

## Code_3 Implementation Notes
`code_3.py` is currently a combined development script rather than a single clean entrypoint.

Important behavior based on current content:
- The file starts with a lightweight baseline script and an `if __name__ == "__main__":` block.
- After that block, a second full training pipeline is defined and executed in the same file.
- Running the file will therefore execute both sections sequentially unless you manually trim one of them.

High-level flow in the advanced section:
1. Load parquet data and profile target/features.
2. Preprocess (near-constant drop, winsorization, target transform).
3. Engineer additional interaction and lag-derived features.
4. Train and tune LightGBM with cross-validation.
5. Perform feature selection using null importances.
6. Train XGBoost and CatBoost.
7. Build ridge stack and optimized blend.
8. Clip final predictions and write `submission.csv`.

Dependencies used by the advanced section include:
- `lightgbm`
- `xgboost`
- `catboost`
- `optuna`
- `scipy`
- `matplotlib`
- `tqdm`

## Practical Modeling Notes
- Because rows are shuffled and sequence IDs are hidden, rely on robust tabular regression rather than sequence reconstruction.
- Lag-difference features are central; interactions between base values and lag signals are often useful.
- Validate with strong cross-validation, track out-of-fold $R^2$, and avoid leakage.

## Quick Start Checklist
1. Load `train.parquet` and `test.parquet`.
2. Use `ID` only for submission mapping, not as a predictive signal.
3. Train regression model(s) on `TARGET`.
4. Predict on test rows.
5. Save predictions in `submission.csv` with columns `ID,TARGET`.
6. Verify row count and column order before submitting.

