# Bike Sharing Demand Forecasting

## 1. Problem
Predict the total number of bikes rented from calendar, season and weather attributes. This is a supervised regression problem.

## 2. Dataset
**Official source:** UCI Bike Sharing, dataset ID 275.

Official page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Exact archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

UCI reports 17,389 instances and 13 features. The target `cnt` is the total rental count. `casual` and `registered` are removed because they directly compose the target and would create leakage. citeturn0search1

## 3. Pipeline

1. Download the official UCI dataset.
2. Combine features and target.
3. Remove leakage fields and raw date representation.
4. Split into train/test sets.
5. Standardize numeric features.
6. Train Linear Regression as an interpretable baseline.
7. Measure MAE, RMSE and R².
8. Serialize the complete preprocessing/model pipeline.
9. Load the artifact and run a real inference example.

## 4. Why Linear Regression?

It is intentionally a baseline. It provides a simple reference before trying Ridge, Random Forest or Gradient Boosting.

## 5. Run

`python -m venv .venv`

` .venv\\Scripts\\activate` on Windows.

`pip install -r requirements.txt`

`python download_data.py`

`python train.py`

`python predict.py`

## 6. Outputs

`data/dataset.csv` is generated locally. `models/model.joblib` contains the fitted pipeline. Both are ignored by Git.

## 7. Recruiter review points

This project demonstrates leakage awareness, reproducible preprocessing, regression metrics, model persistence and inference—not just `LinearRegression().fit()`.

## 8. Next experiments

Compare Ridge, Random Forest and Gradient Boosting. Add cross-validation, residual plots, time-aware validation and a FastAPI inference endpoint.