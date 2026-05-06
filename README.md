# Acute Kidney Injury Prediction

ML system for early detection of Acute Kidney Injury (AKI) from patient creatinine time series.

## Overview

Acute Kidney Injury affects 10-15% of hospitalized patients. This system analyzes routine blood test results to flag at-risk patients before symptoms appear, enabling earlier intervention.

**Approach:**
- Extracts clinical features from creatinine time series (trends, baselines, variability)
- Logistic regression optimized for high recall (F3 score: 0.95+)
- Handles missing data and irregular measurement intervals
- Production-ready with Docker deployment

---

## Usage

**Local:**
```bash
python3 model.py --training training.csv --input test.csv --output aki.csv
```

**Docker:**
```bash
docker build -t aki-prediction .
docker run --rm -v "$PWD":/data aki-prediction
```

Output: `aki.csv` with predictions (`y` = AKI detected, `n` = no AKI)

---

## Technical Details

**Features extracted per patient:**
- Demographics (age, sex)
- Latest creatinine value and change from baseline
- Measurement patterns (frequency, recency, variability)

**Model:**
- Classifier: Logistic Regression (scikit-learn)
- Class weighting: Balanced (prioritizes recall over precision)
- Decision threshold: Tuned to maximize F3 score

**Why F3?** Clinical context: missing an AKI case (false negative) is far more costly than a false alarm. F3 weighs recall 3x higher than precision.

---

## Validation

```bash
python3 model.py --print-metrics
```

Trains model, evaluates on test data, and verifies F3 ≥ 0.95.

---

## Key Files

- `model.py` - Training and inference pipeline
- `Dockerfile` - Production container
- `requirements.txt` - Dependencies (numpy, pandas, scikit-learn)

---

## Context

Coursework project for Software Engineering for Machine Learning Systems (Imperial College London, MSc AI). Demonstrates production ML engineering for healthcare applications.

**Note:** Educational implementation - not intended for clinical use without validation and regulatory approval.
