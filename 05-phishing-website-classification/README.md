# Phishing Website Classification

## Problem
Classify websites using engineered phishing indicators from a public benchmark.

## Dataset
UCI Phishing Websites, dataset ID 327. UCI documents 11,055 instances and 30 integer features derived from public phishing-related sources.

## Model
k-Nearest Neighbors with standardization. This demonstrates distance-based classification and the importance of feature scaling.

## Run
```bash
git clone https://github.com/sudipta200507/ML-Projects.git
cd ML-Projects/05-phishing-website-classification
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python download_data.py
python train.py
python predict.py
```

## Experiments
Try k=3, 5, 11 and 21. Compare precision, recall and F1, then implement the same task with Random Forest.

Security note: a benchmark prediction is not proof that a live URL is safe.