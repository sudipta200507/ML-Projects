# Breast Cancer Diagnostic Classification

## Problem
Classify observations in the Wisconsin Diagnostic benchmark using measurements computed from digitized fine-needle-aspirate images.

## Dataset
**Official source:** UCI Breast Cancer Wisconsin (Diagnostic), ID 17.

Official page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Exact archive: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip

UCI reports 569 instances and 30 real-valued features. citeturn0search7

## Pipeline

Dataset retrieval → stratified split → standardization → RBF SVM → classification report → ROC-AUC → serialized pipeline → inference.

## Why SVM?

The RBF kernel demonstrates a non-linear decision boundary and makes feature scaling, regularization (`C`) and kernel width (`gamma`) meaningful engineering choices.

## Evaluation

The script reports precision, recall, F1 and ROC-AUC. For a medical classification benchmark, sensitivity/specificity and threshold behaviour are important additional analyses.

## Run

`pip install -r requirements.txt`

`python download_data.py`

`python train.py`

`python predict.py`

## Safety

This is an educational benchmark, not a medical diagnostic system. The output must never be used for clinical decisions.

## Next experiments

Compare SVM with Logistic Regression and Random Forest; add nested cross-validation, calibration and sensitivity/specificity analysis.