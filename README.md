# SWEMLS Coursework 1 – Acute Kidney Injury Detection

## Overview
This project implements a machine learning system to predict acute kidney injury (AKI) from patient demographics and historical creatinine blood test results.

The system:
- extracts clinically motivated features from creatinine time series,
- trains a logistic regression classifier on historical patient data,
- prioritizes recall using the F3 score, and
- outputs AKI predictions in the required CSV format for automated evaluation.

The implementation is designed to run inside the supplied Docker environment.

---

## Model Design

### Features
For each patient, the model extracts:
- age
- sex (binary encoding)
- number of prior creatinine measurements
- latest creatinine value
- change relative to historical baseline (minimum and median)
- most recent creatinine delta
- time since previous measurement
- variability of previous measurements (standard deviation)

Missing or insufficient history results in NaN values, which are handled during preprocessing.

---

### Model
- Classifier: Logistic Regression (scikit-learn)
- Class weighting: balanced (to prioritize recall)
- Imputation: Median imputation per feature column
- Decision threshold: Default = 0.2


The threshold was selected empirically by maximizing the F3 score on a held-out
validation split of the training data, reflecting the higher clinical cost of
false negatives.

---

## Running the Model

### Local execution (without Docker)
Run:
python3 model.py --training training.csv --input test.csv --output aki.csv

Optional flags:
- --print-metrics : compute and print F3 score if labels are present
- --tune-threshold : tune the decision threshold on a validation split (for analysis only)

---

### Docker execution (matches marking environment)
Run:
docker build -t coursework1 .
docker run --rm -v "$PWD":/data coursework1

This produces:
 /data/aki.csv

containing a single aki column with values y or n.

---

## Automated Verification

The code includes an automated validation pathway to ensure correctness after changes.

When run with:
python3 model.py --print-metrics

the system:
1. trains the model,
2. evaluates predictions on labelled data,
3. computes the F3 score, and
4. fails automatically if performance drops below an expected threshold (F3 = 0.95).

---

## Dependencies

Listed in requirements.txt:
- numpy
- pandas
- scikit-learn

These libraries are mature, actively maintained, widely adopted in production ML systems, 
and are commonly used in clinical and healthcare data pipelines.

---

## Notes
- The model is retrained deterministically on each run.
- Threshold tuning is disabled by default for reproducibility in the marking environment.