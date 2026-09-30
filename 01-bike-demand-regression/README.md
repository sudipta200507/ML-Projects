# Bike Sharing Demand Forecasting

## Goal
Predict daily/hourly bike rental demand from weather, season and calendar variables. This is a regression problem with a practical capacity-planning use case.

## Dataset
UCI Bike Sharing, dataset ID 275. The repository describes 17,389 records with weather and seasonal information and a regression task.

## Algorithm
Linear Regression is used as the baseline. The project deliberately starts with an interpretable model so you can later compare it against Ridge, Random Forest and Gradient Boosting.

## Pipeline
1. Download the public dataset with `ucimlrepo`.
2. Remove target leakage columns such as `casual` and `registered`.
3. Remove the raw date field.
4. Split into train/test sets.
5. Fit Linear Regression.
6. Report MAE, RMSE and R2.
7. Save the fitted pipeline to `models/model.joblib`.

## Run in VS Code
```bash
git clone https://github.com/sudipta200507/ML-Projects.git
cd ML-Projects/01-bike-demand-regression
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python download_data.py
python train.py
python predict.py
```

## What to learn
Study the difference between MAE, RMSE and R2. Then replace Linear Regression with Random Forest and Gradient Boosting and compare errors.
