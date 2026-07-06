import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from pactstress.evaluation import evaluate_loso


def make_synthetic_classification_df(n_subjects=5, n_per_subject=20, seed=0):
    """
    Synthetic dataset where a single feature perfectly separates the two
    classes, so a correctly-implemented LOSO loop should recover ~100%
    accuracy. This isolates bugs in the evaluation harness itself from
    modeling difficulty.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        subject_id = f"S{s}"
        for i in range(n_per_subject):
            label = i % 2
            feature = 10.0 + rng.normal(0, 0.1) if label == 1 else 0.0 + rng.normal(0, 0.1)
            rows.append({"subject": subject_id, "label": label, "feature": feature})
    return pd.DataFrame(rows)


def test_evaluate_loso_returns_expected_keys():
    df = make_synthetic_classification_df()
    result = evaluate_loso(
        df, feature_cols=["feature"], model=DecisionTreeClassifier(random_state=0)
    )
    assert set(result.keys()) == {
        "per_subject", "mean_accuracy", "std_accuracy", "mean_f1", "std_f1"
    }


def test_evaluate_loso_one_fold_per_subject():
    df = make_synthetic_classification_df(n_subjects=5)
    result = evaluate_loso(
        df, feature_cols=["feature"], model=DecisionTreeClassifier(random_state=0)
    )
    assert len(result["per_subject"]) == 5


def test_evaluate_loso_high_accuracy_on_separable_data():
    df = make_synthetic_classification_df()
    result = evaluate_loso(
        df, feature_cols=["feature"], model=DecisionTreeClassifier(random_state=0)
    )
    # Feature perfectly separates classes, so LOSO accuracy should be high
    # regardless of which subject is held out.
    assert result["mean_accuracy"] > 0.95


def test_evaluate_loso_does_not_mutate_input_model():
    df = make_synthetic_classification_df(n_subjects=3)
    model = DecisionTreeClassifier(random_state=0)
    evaluate_loso(df, feature_cols=["feature"], model=model)
    # the original model object passed in should remain unfitted
    assert not hasattr(model, "tree_")
