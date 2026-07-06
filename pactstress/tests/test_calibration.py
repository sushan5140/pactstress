import numpy as np
import pandas as pd
import pytest

from pactstress.calibration import PACTCalibrator


def make_synthetic_df(seed=0):
    """
    Two synthetic subjects with clearly different baselines, so we can
    verify calibration actually removes the baseline offset per subject.
    Baseline values are noisy around a per-subject mean (not a monotonic
    ramp), so held-out baseline windows are representative of the
    calibration reference rather than systematically higher or lower.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Subject A: baseline MeanNN centered at 800, stress centered at 700
    for _ in range(6):
        rows.append({"subject": "A", "label": 0, "MeanNN": 800 + rng.normal(0, 1)})
    for _ in range(4):
        rows.append({"subject": "A", "label": 1, "MeanNN": 700 + rng.normal(0, 1)})

    # Subject B: baseline MeanNN centered at 600 (different physiology), stress at 500
    for _ in range(6):
        rows.append({"subject": "B", "label": 0, "MeanNN": 600 + rng.normal(0, 1)})
    for _ in range(4):
        rows.append({"subject": "B", "label": 1, "MeanNN": 500 + rng.normal(0, 1)})

    return pd.DataFrame(rows)


def test_calibration_removes_reference_windows():
    df = make_synthetic_df()
    calibrator = PACTCalibrator(feature_cols=["MeanNN"], n_calib_windows=4)
    result = calibrator.fit_transform(df)

    # 4 calibration windows removed per subject, 2 subjects -> 8 rows removed
    assert len(result) == len(df) - 8


def test_calibration_centers_baseline_near_zero():
    df = make_synthetic_df()
    calibrator = PACTCalibrator(feature_cols=["MeanNN"], n_calib_windows=4)
    result = calibrator.fit_transform(df)

    # Remaining baseline windows for each subject should be roughly
    # centered near zero, since they were normalized against a reference
    # drawn from the same subject's own baseline distribution.
    remaining_baseline = result[result["label"] == 0]
    assert abs(remaining_baseline["MeanNN"].mean()) < 2.0


def test_calibration_raises_on_insufficient_baseline_windows():
    df = make_synthetic_df()
    # ask for more calibration windows than subject B's baseline windows minus reserve
    calibrator = PACTCalibrator(feature_cols=["MeanNN"], n_calib_windows=10)
    with pytest.raises(ValueError):
        calibrator.fit_transform(df)


def test_get_calibration_stats_before_fit_raises():
    calibrator = PACTCalibrator(feature_cols=["MeanNN"])
    with pytest.raises(KeyError):
        calibrator.get_calibration_stats("A")


def test_get_calibration_stats_after_fit():
    df = make_synthetic_df()
    calibrator = PACTCalibrator(feature_cols=["MeanNN"], n_calib_windows=4)
    calibrator.fit_transform(df)

    stats_a = calibrator.get_calibration_stats("A")
    assert "mean" in stats_a and "std" in stats_a
    # Subject A's calibration mean should be near 800 (its baseline region)
    assert 799 <= stats_a["mean"]["MeanNN"] <= 803
