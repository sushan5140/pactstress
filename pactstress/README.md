# pactstress

**PACT: Personalized Adaptive Calibration for sTress detection**

A small, focused toolkit for subject-independent wearable stress detection.
Built around one core idea: physiological stress models trained on
population-level data generalize poorly to individuals whose resting
physiology differs from the group average. `pactstress` addresses this by
calibrating each subject's features against their own resting-state
statistics, computed from a small, fixed set of that subject's baseline
windows — not the full session, and never using any window to normalize
itself.

This package was built alongside a study on the [WESAD dataset](https://doi.org/10.1145/3242969.3242985)
(Schmidt et al., 2018), where PACT improved mean subject-independent
accuracy from 87.5% to 90.7% under Leave-One-Subject-Out cross-validation.

## Installation

```bash
pip install pactstress

# to also use the raw-signal feature extraction utilities (ECG/EDA/Resp):
pip install pactstress[signals]
```

## Why calibration matters

A resting heart rate of 100 bpm means something different for someone
whose baseline is normally 100 than for someone whose baseline is normally
65. A model trained on population averages has no way to know which kind
of person it's looking at unless it's told. PACT gives it that
information directly, using a short calibration period — similar to how
a real wearable device might ask a user to sit calmly for a couple of
minutes before making predictions.

## Quick start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pactstress import PACTCalibrator, evaluate_loso

# df: one row per (subject, window), with feature columns + a binary
# 'label' column (0 = baseline, 1 = stress) + a 'subject' column.
df = pd.read_csv("your_features.csv")

feature_cols = ["MeanNN", "SDNN", "RMSSD", "EDA_Mean", "EDA_SCR_Count"]

# Step 1: calibrate each subject's features against their own baseline
calibrator = PACTCalibrator(feature_cols=feature_cols, n_calib_windows=4)
calibrated_df = calibrator.fit_transform(df, subject_col="subject", label_col="label")

# Step 2: evaluate with subject-independent cross-validation
results = evaluate_loso(
    calibrated_df,
    feature_cols=feature_cols,
    model=RandomForestClassifier(n_estimators=100, random_state=42),
)

print(f"Mean accuracy: {results['mean_accuracy']:.3f}")
print(f"Mean F1: {results['mean_f1']:.3f}")
print(results["per_subject"])
```

## Extracting features from raw signals

If you're starting from raw ECG/EDA/respiration signals rather than
pre-extracted features, `pactstress[signals]` (requires `neurokit2`)
provides a windowing + feature-extraction pipeline:

```python
from pactstress import extract_windowed_features

df = extract_windowed_features(
    ecg_signal=ecg,       # 1D array, sampled at `fs`
    eda_signal=eda,
    resp_signal=resp,
    labels=per_sample_labels,   # same length as signals
    subject_id="S2",
    fs=700,
    window_sec=60,
    step_sec=30,
    valid_labels=(1, 2),   # e.g. WESAD: 1=baseline, 2=stress
    stress_label=2,
)
```

This segments the signals into overlapping windows, assigns each window a
label by majority vote, and extracts HRV, EDA, and respiration features
per window using NeuroKit2.

## API overview

| Function / Class | Purpose |
|---|---|
| `segment_windows(signal_length, fs, window_sec, step_sec)` | Compute sliding-window index boundaries |
| `majority_label(label_segment, valid_labels)` | Assign a window's label by majority vote |
| `extract_windowed_features(...)` | Full raw-signal → feature-table pipeline (requires `neurokit2`) |
| `PACTCalibrator(feature_cols, n_calib_windows, baseline_label)` | Personal baseline calibration |
| `evaluate_loso(df, feature_cols, model, label_col, subject_col)` | Leave-One-Subject-Out cross-validation |

## A note on methodological correctness

An earlier version of this calibration approach (during development)
computed each subject's reference statistics from *all* of their baseline
windows, including ones used for evaluation. This produced misleadingly
high accuracy (95–97%) because evaluation windows contributed to their own
normalization reference. `PACTCalibrator` avoids this by using a small,
fixed subset of a subject's earliest baseline windows as the calibration
reference, and removing those specific windows from the returned dataset
entirely — so no window is ever normalized using its own value, and no
evaluation data leaks into calibration statistics. If you're extending
this package, preserve that separation; it's the difference between a
real result and an inflated one.

## Known limitations

- Calibration requires a minimum number of baseline windows per subject
  (`n_calib_windows`, default 4). Subjects with fewer baseline windows
  than this will raise a `ValueError`.
- `PACTCalibrator` assumes a fixed calibration reference collected once;
  it does not currently support adaptive/rolling recalibration over time.
- Developed and validated on WESAD (15 subjects, chest-worn sensors,
  laboratory-induced stress). Performance on wrist-worn consumer sensors
  or real-world (non-laboratory) stress has not been evaluated.

## Running tests

```bash
pip install pactstress[dev]
pytest tests/
```

## License

MIT
