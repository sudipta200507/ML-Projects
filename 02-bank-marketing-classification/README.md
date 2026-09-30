# Bank Marketing Response Prediction

## Problem
Predict whether a client subscribes to a term deposit after a direct-marketing campaign.

## Dataset
**Official source:** UCI Bank Marketing, ID 222.

Official page: https://archive.ics.uci.edu/dataset/222/bank+marketing

Exact archive: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

The dataset contains campaign/customer attributes and a binary target `y`. UCI documents multiple versions, including the full `bank-additional-full.csv` with 41,188 examples and 20 inputs. citeturn0search9

## Pipeline

Download → identify numeric/categorical columns → impute missing values → one-hot encode categoricals → Logistic Regression → classification report → ROC-AUC → saved pipeline → inference.

## Why Logistic Regression?

It is an interpretable binary-classification baseline and exposes the importance of probability thresholds, class imbalance and feature coefficients.

## Evaluation

Accuracy alone is insufficient. The training script reports precision, recall, F1 and ROC-AUC. For campaign systems, changing the decision threshold changes the precision/recall trade-off.

## Run

`python -m venv .venv`

` .venv\\Scripts\\activate`

`pip install -r requirements.txt`

`python download_data.py`

`python train.py`

`python predict.py`

## Reproducibility

The dataset is downloaded from UCI at runtime. No copied dataset is committed. The preprocessing transformer is saved with the classifier so inference receives the same transformations as training.

## Next experiments

Add stratified cross-validation, class-weight experiments, threshold optimization, calibration curves, coefficient analysis and a REST inference API.