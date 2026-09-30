# Breast Cancer Diagnostic Classification

## Goal
Classify diagnostic measurements into the two labels in the public Wisconsin Diagnostic dataset.

## Algorithm
Support Vector Machine with an RBF kernel is used to study margin-based classification and feature scaling.

## Pipeline
UCI retrieval -> train/test split -> standardization -> SVM -> classification metrics -> model persistence.

## VS Code
```bash
git clone https://github.com/sudipta200507/ML-Projects.git
cd ML-Projects/04-breast-cancer-diagnosis
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python download_data.py
python train.py
python predict.py
```

## Safety
This is a machine-learning study, not a medical diagnostic system. Never use the output to make clinical decisions.
