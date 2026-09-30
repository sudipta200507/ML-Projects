# ML Projects

A reproducible machine-learning laboratory covering classical supervised learning on official public datasets.

## Project map

| Project | Problem | Model | Dataset |
|---|---|---|---|
| 01 | Bike demand regression | Linear Regression | UCI Bike Sharing |
| 02 | Campaign response | Logistic Regression | UCI Bank Marketing |
| 03 | Income classification | Random Forest | UCI Adult |
| 04 | Diagnostic classification | SVM | UCI Breast Cancer Wisconsin Diagnostic |
| 05 | Phishing classification | k-NN | UCI Phishing Websites |

## Exact dataset sources

- Bike Sharing — UCI ID 275 — https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset — direct archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- Bank Marketing — UCI ID 222 — https://archive.ics.uci.edu/dataset/222/bank+marketing — direct archive: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- Adult — UCI ID 2 — https://archive.ics.uci.edu/dataset/2/adult — direct archive: https://archive.ics.uci.edu/static/public/2/adult.zip
- Breast Cancer Wisconsin Diagnostic — UCI ID 17 — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic — direct archive: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip
- Phishing Websites — UCI ID 327 — https://archive.ics.uci.edu/dataset/327/phishing — direct archive: https://archive.ics.uci.edu/static/public/327/phishing+websites.zip

UCI documents Bike Sharing as a regression dataset with 17,389 instances and Phishing Websites as a classification dataset with 11,055 instances and 30 features. citeturn0search1turn0search3

## Engineering workflow

Each project follows:

**source data → validation → preprocessing → split → training → evaluation → model persistence → inference**

Datasets and trained binaries are generated locally and ignored by Git. This keeps the repository reviewable and reproducible.

## Run a project

1. Enter a numbered project directory.
2. Create a virtual environment.
3. Install its requirements.
4. Run `python download_data.py`.
5. Run `python train.py`.
6. Run `python predict.py`.
7. Read that project's README for the exact features, target, metrics and experiments.

## Why these projects exist

The objective is to show the full ML workflow—not merely that a model can be instantiated in ten lines. Each project is a baseline that can be extended with cross-validation, hyperparameter search, calibration, explainability, experiment tracking and API deployment.

## Dataset integrity

No fabricated datasets or fabricated performance numbers are used. Dataset provenance is documented at project level, and benchmark performance must not be interpreted as production performance.
