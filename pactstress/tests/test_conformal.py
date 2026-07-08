import numpy as np

from pactstress.conformal import compute_qhat, predict_sets, summarize_predictions


def make_calibrated_probs(n, seed=0, accuracy=0.85):
    """
    Synthetic well-behaved probability outputs: mostly confident and
    correct, occasionally confident and wrong, to exercise the conformal
    machinery realistically.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    probs = np.zeros((n, 2))
    for i, y in enumerate(labels):
        correct = rng.random() < accuracy
        p_true = rng.uniform(0.7, 0.99) if correct else rng.uniform(0.01, 0.4)
        probs[i, y] = p_true
        probs[i, 1 - y] = 1 - p_true
    return probs, labels


def test_qhat_is_between_zero_and_one():
    probs, labels = make_calibrated_probs(200)
    qhat = compute_qhat(probs, labels, alpha=0.1)
    assert 0.0 <= qhat <= 1.0


def test_qhat_raises_on_empty_calibration_set():
    try:
        compute_qhat(np.zeros((0, 2)), np.array([]), alpha=0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_predict_sets_output_shape():
    probs, labels = make_calibrated_probs(200)
    qhat = compute_qhat(probs, labels, alpha=0.1)
    test_probs, _ = make_calibrated_probs(50, seed=1)
    sets = predict_sets(test_probs, qhat)
    assert len(sets) == 50
    assert all(s <= {0, 1} for s in sets)


def test_empirical_coverage_meets_target_on_iid_data():
    """
    The core conformal guarantee: if calibration and test data are drawn
    from the same distribution, empirical coverage should be close to (and
    typically at least) 1 - alpha, on average over many repeated trials.
    A single trial can undershoot by chance, so we average over repeats.
    """
    alpha = 0.1
    coverages = []
    for seed in range(30):
        calib_probs, calib_labels = make_calibrated_probs(300, seed=seed)
        test_probs, test_labels = make_calibrated_probs(300, seed=seed + 1000)

        qhat = compute_qhat(calib_probs, calib_labels, alpha=alpha)
        pred_sets = predict_sets(test_probs, qhat)
        summary = summarize_predictions(pred_sets, test_labels)
        coverages.append(summary["coverage"])

    mean_coverage = np.mean(coverages)
    # allow small slack below the nominal target since this is an average
    # over finite trials, not an exact per-trial guarantee
    assert mean_coverage >= (1 - alpha) - 0.03, (
        f"mean empirical coverage {mean_coverage:.3f} fell meaningfully "
        f"below target {1 - alpha}"
    )


def test_summarize_predictions_flags_abstain_correctly():
    pred_sets = [frozenset({0}), frozenset({1}), frozenset({0, 1}), frozenset({0})]
    true_labels = [0, 1, 0, 1]  # last one is a confident WRONG prediction
    summary = summarize_predictions(pred_sets, true_labels)

    assert summary["abstain_rate"] == 0.25
    assert summary["singleton_rate"] == 0.75
    # 2 of 3 singleton predictions were correct
    assert abs(summary["singleton_accuracy"] - (2 / 3)) < 1e-9


def test_summarize_predictions_rejects_empty_input():
    try:
        summarize_predictions([], [])
        assert False, "expected ValueError"
    except ValueError:
        pass
