#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
import math
import os

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score


def extract_creatinine_history(patient_record: dict):
    """
    Extract and order a patient's creatinine result history from a CSV row.

    Collects all valid (date, value) pairs from columns named
    `creatinine_date_i` and `creatinine_result_i`, converts dates to `datetime`
    objects and results to floats, then sorts chronologically (oldest -> newest).

    Parameters
    ----------
    patient_record : dict
        Maps column names to string patient values.

    Returns
    -------
    list[tuple[datetime, float]]
        A list of (timestamp, creatinine_value) tuples sorted in ascending
        time order. If no valid measurements are present, the list is empty.
    """
    result_history = []
    i = 0

    while True:
        date_key = f"creatinine_date_{i}"
        value_key = f"creatinine_result_{i}"
        if date_key not in patient_record:
            break

        date = patient_record.get(date_key)
        value = patient_record.get(value_key)

        # robust empties
        date_s = str(date).strip() if date is not None else ""
        value_s = str(value).strip() if value is not None else ""

        if date_s and date_s.lower() != "nan" and value_s and value_s.lower() != "nan":
            result_history.append((datetime.fromisoformat(date_s), float(value_s)))

        i += 1

    result_history.sort(key=lambda x: x[0])
    return result_history


def extract_patient_features(patient_record: dict) -> dict:
    """
    Convert a single patient record into a fixed set of numeric features.

    Features summarize the latest creatinine and its change relative to a
    baseline computed from prior measurements (min/median), recent trend,
    time gap, and the amount/variability of history.

    Parameters
    ----------
    patient_record : dict
        Maps column names to patient values.

    Returns
    -------
    dict
        Maps feature names to floats. Some values may be NaN if there is
        insufficient creatinine history.
    """
    age = float(patient_record["age"]) if patient_record.get("age") not in (None, "") else math.nan
    sex_m = 1.0 if str(patient_record.get("sex", "")).strip().lower() == "m" else 0.0

    history = extract_creatinine_history(patient_record)

    # Defaults
    num_results = float(len(history))
    latest = math.nan
    min_prev = math.nan
    median_prev = math.nan
    ratio_to_min = math.nan
    ratio_to_median = math.nan
    delta_to_min = math.nan
    delta_to_median = math.nan
    latest_delta = math.nan
    hours_since_previous = math.nan
    std_prev = math.nan

    if len(history) >= 1:
        dates = [d for d, _ in history]
        results = [v for _, v in history]
        latest = float(results[-1])

    if len(history) >= 2:
        prev_vals = results[:-1]
        min_prev = float(min(prev_vals))
        median_prev = float(np.median(prev_vals))

        latest_delta = float(results[-1] - results[-2])

        dt_hours = (dates[-1] - dates[-2]).total_seconds() / 3600.0
        hours_since_previous = float(dt_hours) if dt_hours >= 0 else math.nan

        delta_to_min = float(latest - min_prev)
        delta_to_median = float(latest - median_prev)

        ratio_to_min = float(latest / min_prev) if min_prev != 0 else math.inf
        ratio_to_median = float(latest / median_prev) if median_prev != 0 else math.inf

        std_prev = float(np.std(prev_vals))

    return {
        "age": age,
        "sex_m": sex_m,
        "num_results": num_results,
        "latest": latest,
        "min_prev": min_prev,
        "median_prev": median_prev,
        "ratio_to_min": ratio_to_min,
        "ratio_to_median": ratio_to_median,
        "delta_to_min": delta_to_min,
        "delta_to_median": delta_to_median,
        "latest_delta": latest_delta,
        "hours_since_previous": hours_since_previous,
        "std_prev": std_prev,
    }


def load_features(records_filepath, has_label):
    """
    Load patient records from a CSV file and construct a feature matrix.

    Reads a CSV file containing patient demographics and creatinine history,
    extracts features for each patient using `extract_patient_features`, 
    and returns the resulting feature matrix.
    Infinite values are converted to NaN and missing values are imputed
    using the median of each feature column.
    If labels are present (training data), the function also extracts the
    acute kidney injury (AKI) outcome as a binary target variable.

    Parameters
    ----------
    records_filepath : str
        Path to the CSV file containing patient records.
    has_label : bool
        Whether the CSV file contains an `aki` column. If True, the function
        returns both features and labels; otherwise, only features are returned.

    Returns
    -------
    pandas.DataFrame or tuple(pandas.DataFrame, pandas.Series)
        If `has_label` is False, returns a DataFrame `X` containing the
        engineered features.
        If `has_label` is True, returns a tuple `(X, y)` where `X` is the
        feature DataFrame and `y` is a Series of binary AKI labels
        (1 = AKI present, 0 = no AKI).
    """
    X_rows = []
    y = []

    with open(records_filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_rows.append(extract_patient_features(row))
            if has_label:
                y.append(1 if row["aki"] == "y" else 0)

    X = pd.DataFrame(X_rows)
    return (X, pd.Series(y)) if has_label else X


class AkiModel:
    """
    Simple AKI classifier using logistic regression.

    Uses median imputation and a fixed probability threshold for converting
    probabilities into binary predictions.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.imputer = SimpleImputer(strategy="median")
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # helps recall (important for F3)
            solver="lbfgs",
        )
        self.feature_columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AkiModel":
        # Keep a stable column order between train/test
        self.feature_columns = list(X.columns)

        X = X.reindex(columns=self.feature_columns)
        X = X.replace([math.inf, -math.inf], np.nan)
        X_imp = self.imputer.fit_transform(X)

        self.clf.fit(X_imp, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("Model has not been fitted yet.")

        # Ensure same columns/order as training (fill missing with NaN -> imputer handles it)
        X = X.reindex(columns=self.feature_columns, fill_value=np.nan)
        X = X.replace([math.inf, -math.inf], np.nan)
        X_imp = self.imputer.transform(X)

        probs = self.clf.predict_proba(X_imp)[:, 1]
        return (probs >= self.threshold).astype(int)


def csv_has_column(filepath: str, column: str) -> bool:
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return column in header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv", help="Path to input CSV (no aki column in marking).")
    parser.add_argument("--output", default="aki.csv", help="Path to output predictions CSV.")
    parser.add_argument(
        "--training",
        default=os.environ.get("TRAINING_PATH", "training.csv"),
        help="Path to training CSV. In marking this is /data/training.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predicting AKI (y).",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="If input has labels, print F3 score.",
    )
    args = parser.parse_args()

    # If we're running under the marking harness, /data/training.csv will exist.
    training_path = args.training
    if training_path == "training.csv" and os.path.exists("/data/training.csv"):
        training_path = "/data/training.csv"

    # Train
    X_train, y_train = load_features(training_path, has_label=True)
    model = AkiModel(threshold=args.threshold).fit(X_train, y_train)

    # Predict (input may or may not have labels)
    input_has_aki = csv_has_column(args.input, "aki")
    if input_has_aki:
        X_in, y_true = load_features(args.input, has_label=True)
    else:
        X_in = load_features(args.input, has_label=False)
        y_true = None

    pred_binary = model.predict(X_in)

    # Optional metrics (only if labels exist)
    if args.print_metrics and y_true is not None:
        f3 = fbeta_score(y_true, pred_binary, beta=3, zero_division=0)
        print(f"F3 score: {f3:.4f}")

    # Write predictions in required format
    y_pred = np.where(pred_binary == 1, "y", "n")
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aki"])
        for p in y_pred:
            w.writerow([p])


if __name__ == "__main__":
    main()