# Adult Income Classification

## Goal
Classify whether a person's annual income exceeds the dataset threshold. This is a classic tabular classification problem involving mixed numeric and categorical data.

## Algorithm
Random Forest captures non-linear feature interactions and provides a useful ensemble baseline.

## Pipeline
Dataset retrieval -> missing-value handling -> one-hot encoding -> Random Forest -> precision/recall/F1/ROC-AUC -> serialized pipeline.

## VS Code
```bash
git clone https://github.com/sudipta200507/ML-Projects.git
cd ML-Projects/03-income-classification
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python download_data.py
python train.py
python predict.py
```

## Important
This is an educational benchmark. Avoid using the model for real employment, lending or eligibility decisions without domain validation, fairness analysis and legal review.
