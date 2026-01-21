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
from sklearn.model_selection import train_test_split


def _clean_cell(value) -> str:
    """Convert a CSV cell to a stripped string, treating None/NaN as empty."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _parse_float(cell) -> float:
    """Parse a float from a CSV cell; return NaN if missing/unparseable."""
    s = _clean_cell(cell)
    if not s:
        return math.nan
    try:
        x = float(s)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _sex_is_male(cell) -> float:
    """
    Encode sex as 1.0 for male, 0.0 otherwise.

    Accepts common encodings: 'm', 'male'. Unknown/missing -> 0.0.
    """
    s = _clean_cell(cell).lower()
    return 1.0 if s in {"m", "male"} else 0.0


def _parse_label_aki(cell) -> int:
    """Parse an 'aki' label cell into 0/1. Unknown/missing defaults to 0."""
    s = _clean_cell(cell).lower()
    return 1 if s == "y" else 0


def extract_creatinine_history(patient_record: dict):
    """
    Extract and order a patient's creatinine result history from a CSV row.

    Reads (creatinine_date_i, creatinine_result_i) pairs from a patient record, 
    parses valid pairs, and returns them chronologically sorted (oldest -> newest).
    Invalid or missing entries are ignored.

    Parameters
    ----------
    patient_record:
        Mapping of column names to CSV cell values.

    Returns
    -------
    list[tuple[datetime, float]]
        Chronologically ordered creatinine measurements.
        Returns an empty list if no valid measurements are present.
    """
    result_history = []
    i = 0

    while True:
        date_key = f"creatinine_date_{i}"
        result_key = f"creatinine_result_{i}"

        # Stop when date column does not exist
        if date_key not in patient_record:
            break

        date_s = _clean_cell(patient_record.get(date_key))
        result_s = _clean_cell(patient_record.get(result_key))

        if date_s and result_s:
            try:
                timestamp = datetime.fromisoformat(date_s)
            # Skip invalid timestamps
            except (TypeError, ValueError):
                i += 1
                continue

            try:
                result = float(result_s)
            # Skip invalid measurements
            except (TypeError, ValueError):
                i += 1
                continue

            # Skip NaN/inf values.
            if math.isfinite(result):
                result_history.append((timestamp, result))

        i += 1

    result_history.sort(key=lambda pair: pair[0])
    return result_history


def extract_patient_features(patient_record: dict) -> dict:
    """
    Convert a single patient record into a fixed set of numeric features.

    Features summarize creatinine history: latest value, change relative to
    earlier measurements (min/median), the most recent delta, time gap, 
    and variability/count of measurements. 
    Missing or insufficient history yields NaN for the affected features.

    Parameters
    ----------
    patient_record:
        Mapping of column names to CSV cell values.

    Returns
    -------
    dict[str, float]
        Mapping of feature names to floats (may contain NaNs).
    """
    # Parse demographics
    age = _parse_float(patient_record.get("age"))
    sex_m = _sex_is_male(patient_record.get("sex"))

    result_history = extract_creatinine_history(patient_record)
    num_results = float(len(result_history))

    # Initialize all outputs to NaN to fill in when data exists.
    nan = math.nan
    features = {
        "latest": nan,
        "min_prev": nan,
        "median_prev": nan,
        "ratio_to_min": nan,
        "ratio_to_median": nan,
        "delta_to_min": nan,
        "delta_to_median": nan,
        "latest_delta": nan,
        "hours_since_previous": nan,
        "std_prev": nan,
    }

    if not result_history:
        return {
            "age": age, 
            "sex_m": sex_m, 
            "num_results": num_results, 
            **features
        }

    dates = [dt for dt, _ in result_history]
    results = [v for _, v in result_history]
    latest = float(results[-1])
    features["latest"] = latest

    if len(result_history) >= 2:
        prev_results = results[:-1]

        min_prev = float(min(prev_results))
        median_prev = float(np.median(prev_results))
        features["min_prev"] = min_prev
        features["median_prev"] = median_prev
        features["std_prev"] = float(np.std(prev_results))

        features["latest_delta"] = float(results[-1] - results[-2])

        dt_hours = (dates[-1] - dates[-2]).total_seconds() / 3600.0
        features["hours_since_previous"] = float(dt_hours) if dt_hours >= 0 else math.nan

        features["delta_to_min"] = float(latest - min_prev)
        features["delta_to_median"] = float(latest - median_prev)

        features["ratio_to_min"] = float(latest / min_prev) if min_prev != 0 else math.inf
        features["ratio_to_median"] = float(latest / median_prev) if median_prev != 0 else math.inf

    return {
        "age": age, 
        "sex_m": sex_m, 
        "num_results": num_results, 
        **features
    }


def load_features(records_filepath: str, has_label: bool):
    """
    Load patient records from a CSV file and construct a feature matrix.

    Reads patient rows and extracts engineered features using
    `extract_patient_features`. If `has_label` is True, also extracts the
    `aki` column as a binary target (1 for 'y', 0 otherwise).

    Parameters
    ----------
    records_filepath:
        Path to the CSV file containing patient records.
    has_label:
        Whether to extract labels from an `aki` column.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.Series)
        If `has_label` is False, returns a DataFrame X of features.
        If `has_label` is True, returns (X, y) where y is a 0/1 Series.
    """
    X_rows = []
    y = []

    with open(records_filepath, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {records_filepath}")

        if has_label and "aki" not in reader.fieldnames:
            raise ValueError(f"Expected 'aki' column in {records_filepath}, but it was not found.")

        for row in reader:
            X_rows.append(extract_patient_features(row))
            if has_label:
                y.append(_parse_label_aki(row.get("aki")))

    X = pd.DataFrame(X_rows)

    if has_label:
        return X, pd.Series(y, name="aki")
    return X


class AkiModel:
    """
    AKI classifier using logistic regression with median imputation.

    Preprocessing:
    - replace ±inf with NaN
    - median-impute per feature column

    Prediction:
    - predict probabilities with logistic regression
    - apply a probability threshold to produce 0/1 predictions
    """

    def __init__(self, threshold=0.5, tune_threshold=False, tune_grid=None, random_state=0):
        self.random_state = random_state
        self.threshold = float(threshold)
        self.tune_threshold = bool(tune_threshold)
        
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1.")
        
        self.tune_grid = (
            [float(t) for t in tune_grid]
            if tune_grid is not None
            else [t / 10 for t in range(1, 10)]  # 0.1 ... 0.9
        )

        self.imputer = SimpleImputer(strategy="median")
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced", # helps recall (important for F3)
            solver="lbfgs",
            random_state=random_state,
        )
        self.feature_columns = None

    
    def _validate_fit_inputs(self, X, y):
        if len(X) == 0:
            raise ValueError("Training feature matrix X is empty.")
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: len(X)={len(X)} len(y)={len(y)}")


    def _align_and_clean(self, X, fit=False):
        """
        Align feature columns and replace ±inf with NaN.

        If fit=True, establish the feature column order from X (training data).
        """
        if fit:
            self.feature_columns = list(X.columns)
        elif self.feature_columns is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_aligned = X.reindex(columns=self.feature_columns, fill_value=np.nan)
        return X_aligned.replace([math.inf, -math.inf], np.nan)
    

    def _fit_preprocess(self, X):
        X_clean = self._align_and_clean(X, fit=True)
        return self.imputer.fit_transform(X_clean)
    

    def _predict_preprocess(self, X):
        X_clean = self._align_and_clean(X, fit=False)
        return self.imputer.transform(X_clean)


    def _select_best_threshold(self, y_true, probabilities):
        """
        Select the decision threshold that maximizes the F3 score
        for the given true labels and predicted probabilities.
        """
        best_threshold = self.threshold
        best_f3 = -1.0

        for t in self.tune_grid:
            preds = (probabilities >= t).astype(int)
            f3 = fbeta_score(y_true, preds, beta=3, zero_division=0)
            if f3 > best_f3:
                best_f3 = f3
                best_threshold = float(t)

        return best_threshold


    def fit(self, X, y):
        self._validate_fit_inputs(X, y)

        if self.tune_threshold:
            # Validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=self.random_state,
            )

            # Fit on training split
            X_train_imp = self._fit_preprocess(X_train)
            self.clf.fit(X_train_imp, y_train)

            # Predict probabilities on validation split
            X_val_imp = self._predict_preprocess(X_val)
            probabilities = self.clf.predict_proba(X_val_imp)[:, 1]

            # Select threshold on validation data
            self.threshold = self._select_best_threshold(y_val, probabilities)

            # Refit on full dataset
            X_full_imp = self._fit_preprocess(X)
            self.clf.fit(X_full_imp, y)

        else:
            X_imp = self._fit_preprocess(X)
            self.clf.fit(X_imp, y)

        return self


    def predict(self, X):
        if len(X) == 0:
            return np.array([], dtype=int)

        X_imp = self._predict_preprocess(X)
        probs = self.clf.predict_proba(X_imp)[:, 1]
        return (probs >= self.threshold).astype(int)


def csv_has_column(filepath: str, column: str) -> bool:
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return column in header


def resolve_training_path(arg_path):
    if arg_path == "training.csv" and os.path.exists("/data/training.csv"):
        return "/data/training.csv"
    return arg_path


def write_predictions(output_path, pred_binary):
    y_pred = np.where(pred_binary == 1, "y", "n")
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aki"])
        w.writerows([[p] for p in y_pred])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv", help="Path to input CSV.")
    parser.add_argument("--output", default="aki.csv", help="Path to output predictions CSV.")
    parser.add_argument(
        "--training",
        default=os.environ.get("TRAINING_PATH", "training.csv"),
        help="Path to training CSV. In marking this is /data/training.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Probability threshold for predicting AKI (y).",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune the decision threshold on a validation split to maximize F3, then refit on full training data.",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="If input has labels, print F3 score.",
    )
    args = parser.parse_args()

    training_path = resolve_training_path(args.training)

    # Train model
    X_train, y_train = load_features(training_path, has_label=True)
    model = AkiModel(
        threshold=args.threshold,
        tune_threshold=args.tune_threshold,
        random_state=0,
    ).fit(X_train, y_train)

    # Load input (may have labels)
    input_has_aki = csv_has_column(args.input, "aki")
    if input_has_aki:
        X_in, y_true = load_features(args.input, has_label=True)
    else:
        X_in = load_features(args.input, has_label=False)
        y_true = None

    # Predict + write output
    pred_binary = model.predict(X_in)
    write_predictions(args.output, pred_binary)

    # Optional metrics + automated quality check
    if args.print_metrics and y_true is not None:
        f3 = fbeta_score(y_true, pred_binary, beta=3, zero_division=0)
        print(f"F3 score: {f3:.4f}")

        if args.tune_threshold:
            print(f"Chosen threshold: {model.threshold:.3f}")

        EXPECTED_MIN_F3 = 0.95
        if f3 < EXPECTED_MIN_F3:
            raise RuntimeError(f"Model validation failed: F3={f3:.4f} < expected {EXPECTED_MIN_F3}")


if __name__ == "__main__":
    main()