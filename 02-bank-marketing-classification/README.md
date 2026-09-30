# Bank Marketing Response Prediction

## Goal
Predict whether a customer will subscribe to a term deposit after a bank marketing campaign.

## Dataset
UCI Bank Marketing. UCI describes this as a classification problem from Portuguese bank direct-marketing campaigns.

## Algorithm
Logistic Regression provides a strong interpretable classification baseline for binary outcomes.

## Pipeline
Download -> categorical/numeric preprocessing -> one-hot encoding -> scaling -> Logistic Regression -> classification report -> saved pipeline.

## VS Code
```bash
git clone https://github.com/sudipta200507/ML-Projects.git
cd ML-Projects/02-bank-marketing-classification
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python download_data.py
python train.py
python predict.py
```

## Experiments
Compare class weighting, regularization strength C, threshold selection and precision/recall. Do not judge a marketing system from accuracy alone.
