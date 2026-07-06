"""
Leave-One-Subject-Out (LOSO) cross-validation for subject-independent
model evaluation.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut


def evaluate_loso(
    df,
    feature_cols,
    model,
    label_col="label",
    subject_col="subject",
):
    """
    Evaluate a classifier with Leave-One-Subject-Out cross-validation.

    In each fold, one subject's rows are held out entirely for testing
    while the model is trained on all remaining subjects. This is repeated
    once per subject, which directly measures how well a model generalizes
    to a person it has never seen.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `feature_cols`, `label_col`, and `subject_col`.
    feature_cols : list of str
        Feature columns to use as model input.
    model : estimator
        An unfitted scikit-learn-compatible classifier. A fresh clone of
        this model is trained in each fold via `copy.deepcopy`; the
        object you pass in is never itself mutated.
    label_col : str, default="label"
        Column containing the binary classification target.
    subject_col : str, default="subject"
        Column identifying which subject each row belongs to. Defines the
        LOSO fold groups.

    Returns
    -------
    dict
        {
            "per_subject": pandas.DataFrame with columns
                ["subject", "accuracy", "f1"],
            "mean_accuracy": float,
            "std_accuracy": float,
            "mean_f1": float,
            "std_f1": float,
        }
    """
    X = df[feature_cols]
    y = df[label_col]
    groups = df[subject_col]

    logo = LeaveOneGroupOut()
    records = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_subject = groups.iloc[test_idx].values[0]

        fold_model = deepcopy(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        records.append({
            "subject": test_subject,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })

    per_subject = pd.DataFrame(records)

    return {
        "per_subject": per_subject,
        "mean_accuracy": per_subject["accuracy"].mean(),
        "std_accuracy": per_subject["accuracy"].std(),
        "mean_f1": per_subject["f1"].mean(),
        "std_f1": per_subject["f1"].std(),
    }
