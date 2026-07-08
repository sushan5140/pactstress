"""
pactstress: Personalized Adaptive Calibration for sTress detection.

A small toolkit for subject-independent wearable stress detection,
built around one core idea: calibrate physiological features against
each person's own resting baseline, not population-wide statistics.

Typical usage
-------------
>>> from pactstress import extract_windowed_features, PACTCalibrator, evaluate_loso
>>> from sklearn.ensemble import RandomForestClassifier
>>>
>>> df = extract_windowed_features(ecg, eda, resp, labels, subject_id="S2")
>>> calibrator = PACTCalibrator(feature_cols=["MeanNN", "SDNN", "RMSSD"])
>>> calibrated_df = calibrator.fit_transform(df)
>>> results = evaluate_loso(calibrated_df, feature_cols=[...], model=RandomForestClassifier())
"""

from .calibration import PACTCalibrator
from .evaluation import evaluate_loso, evaluate_loso_conformal
from .windowing import segment_windows, majority_label
from .conformal import compute_qhat, predict_sets, summarize_predictions

__all__ = [
    "PACTCalibrator",
    "evaluate_loso",
    "evaluate_loso_conformal",
    "segment_windows",
    "majority_label",
    "compute_qhat",
    "predict_sets",
    "summarize_predictions",
]

__version__ = "0.2.0"

# extract_windowed_features requires neurokit2, which is an optional
# dependency (heavier install, only needed for raw signal processing).
# Import it lazily so the rest of the package works without it.
try:
    from .features import extract_windowed_features
    __all__.append("extract_windowed_features")
except ImportError:
    pass
