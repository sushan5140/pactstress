"""
Split conformal prediction for selective stress classification.

Standard binary classifiers force a prediction on every input, even when the
input doesn't resemble anything the model was trained on. PACT's own case
study (subject S2, see the accompanying paper) showed this concretely: a
physiologically atypical subject was confidently and consistently
misclassified, rather than flagged as uncertain.

This module wraps a fitted classifier's probability outputs in a split
conformal procedure (Vovk et al., 2005; see Angelopoulos & Bates, 2023 for an
accessible tutorial), which produces PREDICTION SETS rather than single
labels, with a distribution-free statistical guarantee: over repeated
calibration/test splits, the true label falls inside the predicted set at
least (1 - alpha) of the time, regardless of how well- or poorly-suited the
underlying classifier is to a given input.

For binary classification this yields one of three outcomes per prediction:
    {0}     - confidently non-stress
    {1}     - confidently stress
    {0, 1}  - abstain: the model cannot confidently distinguish the two
              classes for this input (this is the interesting case for
              atypical responders like S2)
"""

import numpy as np


def compute_qhat(calib_probs, calib_labels, alpha=0.1):
    """
    Compute the split conformal quantile threshold from a calibration set.

    Parameters
    ----------
    calib_probs : array-like, shape (n_calib, 2)
        Predicted class probabilities [P(class=0), P(class=1)] for each
        calibration example, from a classifier already fit on separate
        training data (never on the calibration set itself).
    calib_labels : array-like, shape (n_calib,)
        True binary labels for the calibration set.
    alpha : float, default=0.1
        Desired miscoverage rate. alpha=0.1 targets 90% coverage: the true
        label will fall inside the predicted set at least 90% of the time
        over repeated calibration/test splits.

    Returns
    -------
    float
        The conformal quantile threshold (qhat), used by `predict_sets`.
    """
    calib_probs = np.asarray(calib_probs)
    calib_labels = np.asarray(calib_labels)
    n = len(calib_labels)

    if n < 1:
        raise ValueError("Calibration set must contain at least one example.")

    # Nonconformity score: 1 - predicted probability assigned to the TRUE class.
    # A low score means the model was confident and correct; a high score
    # means the true class was assigned low probability.
    true_class_probs = calib_probs[np.arange(n), calib_labels]
    scores = 1 - true_class_probs

    # The (n+1)(1-alpha)/n quantile, clipped to 1.0, is the standard finite-
    # sample-correct split conformal threshold (Angelopoulos & Bates, 2023).
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, q_level, method="higher")
    return float(qhat)


def predict_sets(test_probs, qhat):
    """
    Construct a prediction set for each test example given a calibrated qhat.

    Parameters
    ----------
    test_probs : array-like, shape (n_test, 2)
        Predicted class probabilities [P(class=0), P(class=1)] for each
        test example.
    qhat : float
        Conformal threshold from `compute_qhat`.

    Returns
    -------
    list of frozenset
        One prediction set per test example: frozenset({0}), frozenset({1}),
        or frozenset({0, 1}) (abstain). An empty frozenset is possible in
        principle but not expected in the well-calibrated binary case.
    """
    test_probs = np.asarray(test_probs)
    sets = []
    for row in test_probs:
        included = {c for c in (0, 1) if (1 - row[c]) <= qhat}
        sets.append(frozenset(included))
    return sets


def summarize_predictions(pred_sets, true_labels):
    """
    Summarize conformal prediction set outcomes against true labels.

    Parameters
    ----------
    pred_sets : list of frozenset
        Output of `predict_sets`.
    true_labels : array-like
        True binary labels, same length as `pred_sets`.

    Returns
    -------
    dict
        {
            "coverage": fraction of examples where the true label is inside
                the predicted set (this should be >= 1-alpha on average, by
                the conformal guarantee),
            "singleton_rate": fraction of examples with a confident,
                single-label prediction,
            "abstain_rate": fraction of examples where the model abstained
                (predicted set = {0, 1}),
            "singleton_accuracy": accuracy restricted to singleton
                (non-abstained) predictions only — this is the accuracy of
                the model's *confident* predictions specifically.
        }
    """
    true_labels = np.asarray(true_labels)
    n = len(true_labels)
    if n == 0:
        raise ValueError("true_labels must not be empty.")

    covered = sum(1 for s, y in zip(pred_sets, true_labels) if y in s)
    singleton_mask = [len(s) == 1 for s in pred_sets]
    n_singleton = sum(singleton_mask)
    n_abstain = sum(1 for s in pred_sets if len(s) == 2)

    if n_singleton > 0:
        singleton_correct = sum(
            1 for s, y, is_single in zip(pred_sets, true_labels, singleton_mask)
            if is_single and y in s
        )
        singleton_accuracy = singleton_correct / n_singleton
    else:
        singleton_accuracy = float("nan")

    return {
        "coverage": covered / n,
        "singleton_rate": n_singleton / n,
        "abstain_rate": n_abstain / n,
        "singleton_accuracy": singleton_accuracy,
    }
