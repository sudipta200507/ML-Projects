# Adult Income Classification

## Problem
Predict whether annual income is above the dataset threshold from census-derived attributes.

## Dataset
**Official source:** UCI Adult / Census Income, ID 2.

Official page: https://archive.ics.uci.edu/dataset/2/adult

Exact archive: https://archive.ics.uci.edu/static/public/2/adult.zip

UCI reports 48,842 instances and 14 features. The target asks whether income exceeds $50K/year. citeturn0search10

## Pipeline

1. Retrieve the public dataset.
2. Separate numeric and categorical features.
3. Impute missing values.
4. One-hot encode categorical attributes.
5. Train a class-balanced Random Forest.
6. Evaluate precision, recall, F1 and ROC-AUC.
7. Persist the full preprocessing/model pipeline.
8. Run inference on a held-out example.

## Why Random Forest?

The model can learn non-linear interactions without requiring a linear decision boundary. It also gives a useful feature-importance baseline for later explainability work.

## Responsible-use note

This is a research benchmark. It must not be treated as an employment, lending, insurance or eligibility decision system. Any real deployment would require fairness analysis, legal review, governance and representative current data.

## Run

`pip install -r requirements.txt`

`python download_data.py`

`python train.py`

`python predict.py`

## Next experiments

Compare a single tree, Logistic Regression and Gradient Boosting. Add cross-validation, calibration and subgroup error analysis.