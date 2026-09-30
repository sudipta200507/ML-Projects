# Phishing Website Classification

## Problem
Classify websites from engineered features associated with phishing behaviour. This is a defensive security-ML benchmark.

## Dataset
**Official source:** UCI Phishing Websites, ID 327.

Official page: https://archive.ics.uci.edu/dataset/327/phishing

Exact archive: https://archive.ics.uci.edu/static/public/327/phishing+websites.zip

UCI reports 11,055 instances and 30 integer features, collected mainly from PhishTank, MillerSmiles and search operators. citeturn0search3

## Pipeline

Official data → train/test split → feature standardization → k-NN → classification report/confusion matrix → model persistence → inference.

## Why k-NN?

It demonstrates instance-based learning and makes feature scaling essential. The project is intentionally simple enough to inspect before comparing against tree ensembles and neural models.

## Experiments

Change `k` from 3 to 5, 11 and 21. Compare precision, recall and F1. Then build the same pipeline with Random Forest and compare error patterns.

## Run

`pip install -r requirements.txt`

`python download_data.py`

`python train.py`

`python predict.py`

## Security limitation

A benchmark prediction is not proof that a live URL is safe. Real phishing detection needs current data, URL/domain intelligence, adversarial testing and analyst review.

## Portfolio value

This project connects classical ML concepts—distance, scaling, classification metrics—to a real defensive cybersecurity problem.