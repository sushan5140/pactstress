"""
Example: applying pactstress to WESAD-style data.

This assumes you've already loaded WESAD subject .pkl files and have
access to each subject's chest ECG/EDA/Resp signals and per-sample labels.
See https://doi.org/10.1145/3242969.3242985 for the dataset itself.

Run with: pip install pactstress[signals] first (for extract_windowed_features).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pactstress import extract_windowed_features, PACTCalibrator, evaluate_loso

FEATURE_COLS = [
    "MeanNN", "SDNN", "RMSSD",
    "EDA_Mean", "EDA_SCR_Count", "EDA_SCR_Amplitude_Mean",
    "Resp_Rate_Mean", "Resp_Amplitude_Mean",
]


def build_dataset(subject_data):
    """
    subject_data: dict mapping subject_id -> {'ecg':.., 'eda':.., 'resp':.., 'labels':.., 'fs':..}
    """
    all_features = []
    for subject_id, signals in subject_data.items():
        df = extract_windowed_features(
            ecg_signal=signals["ecg"],
            eda_signal=signals["eda"],
            resp_signal=signals["resp"],
            labels=signals["labels"],
            subject_id=subject_id,
            fs=signals.get("fs", 700),
            window_sec=60,
            step_sec=30,
            valid_labels=(1, 2),
            stress_label=2,
        )
        all_features.append(df)
    return pd.concat(all_features, ignore_index=True)


def main(subject_data):
    df = build_dataset(subject_data)
    print(f"Extracted {len(df)} windows across {df['subject'].nunique()} subjects")

    # Uncalibrated (population-level) baseline
    uncalibrated_results = evaluate_loso(
        df, feature_cols=FEATURE_COLS,
        model=RandomForestClassifier(n_estimators=100, random_state=42),
    )
    print(f"\nPopulation-level model:")
    print(f"  Mean accuracy: {uncalibrated_results['mean_accuracy']:.3f} "
          f"(SD={uncalibrated_results['std_accuracy']:.3f})")
    print(f"  Mean F1:       {uncalibrated_results['mean_f1']:.3f} "
          f"(SD={uncalibrated_results['std_f1']:.3f})")

    # PACT-calibrated
    calibrator = PACTCalibrator(feature_cols=FEATURE_COLS, n_calib_windows=4)
    calibrated_df = calibrator.fit_transform(df)

    calibrated_results = evaluate_loso(
        calibrated_df, feature_cols=FEATURE_COLS,
        model=RandomForestClassifier(n_estimators=100, random_state=42),
    )
    print(f"\nPACT-calibrated model:")
    print(f"  Mean accuracy: {calibrated_results['mean_accuracy']:.3f} "
          f"(SD={calibrated_results['std_accuracy']:.3f})")
    print(f"  Mean F1:       {calibrated_results['mean_f1']:.3f} "
          f"(SD={calibrated_results['std_f1']:.3f})")

    print("\nPer-subject breakdown (calibrated):")
    print(calibrated_results["per_subject"].sort_values("accuracy"))


if __name__ == "__main__":
    # Replace with real WESAD subject data loaded from .pkl files.
    raise SystemExit(
        "This is a template — populate `subject_data` with real WESAD "
        "signals before running. See README.md for the expected format."
    )
