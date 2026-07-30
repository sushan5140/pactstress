# pactstress

**PACT: Personalized Adaptive Calibration for sTress detection.**
A small toolkit that calibrates wearable physiological features against
each subject's own resting baseline — because a population model has no
way to know whose body it's looking at.

> The package itself (source, tests, docs, examples) lives in the
> [`pactstress/`](pactstress/) subdirectory — see
> [`pactstress/README.md`](pactstress/README.md) for the full package docs.
> This page is the repo landing page.

```bash
pip install pactstress

# to also use the raw-signal feature extraction utilities (ECG/EDA/Resp):
pip install pactstress[signals]
```

---

## The problem

A resting heart rate of 100 bpm means something different for someone
whose baseline is normally 100 than for someone whose baseline is normally
65. A model trained on population-level averages has no way to distinguish
these two people unless it's told — and most wearable stress-detection
pipelines never tell it.

`pactstress` calibrates each subject's features against their own
resting-state statistics, computed from a small, fixed set of that
subject's baseline windows — never the full session, and never a window
normalized against itself. It's the same idea as a real wearable device
asking you to sit calmly for a couple of minutes before it starts making
predictions.

We validated this on the [WESAD dataset](https://doi.org/10.1145/3242969.3242985)
(Schmidt et al., 2018): under Leave-One-Subject-Out cross-validation,
personal-baseline calibration raised mean subject-independent accuracy
from **87.5% to 90.7%**.

## The problem underneath the problem

Calibration doesn't fix every subject. The original case study found
subject S2: clean, artifact-free signals, but a physiologically atypical
stress response (stable heart rate, slow deep breathing) that the model
got confidently, consistently wrong — F1 = 0.000 for that subject, even
after calibration. A model that's *wrong and confident* is a worse failure
mode than one that's *uncertain and says so*.

v0.2 adds a split-conformal selective-prediction layer on top: instead of
forcing "stress" or "not-stress," the model can output `{stress,
not-stress}` (abstain) when it can't confidently tell, backed by a
distribution-free coverage guarantee rather than an ad hoc confidence
threshold. We validated it on WESAD in two configurations, and the two
don't simply agree:

| Metric | Conformal on raw features | Conformal on **PACT-calibrated** features |
|---|---|---|
| Mean coverage (target 0.90) | 0.899 | 0.865 |
| Mean abstain rate | 0.153 | 0.136 |
| Mean singleton accuracy | 0.930 | 0.926 |
| **S10** (worst population-level subject, 42.9% acc.) | abstains **100%** of windows | abstains 81.4% of windows |
| **S2** (the atypical case above) | abstains 11.9%, singleton acc. 0.673 | abstains **0%**, singleton acc. 0.618 |

On raw features, conformal prediction does exactly what it's supposed to:
S10 abstains on every window rather than being confidently wrong, and S2
is partially caught. But stacked on top of PACT-calibrated features,
coverage drops meaningfully below target and **S2's abstain rate falls to
zero** — the personalization that helps accuracy also removes the
distributional signal the conformal layer needs to flag S2 as atypical.
Reporting both configurations, rather than only the better-looking one, is
itself the point: see
[`pactstress/docs/conformal_extension.md`](pactstress/docs/conformal_extension.md)
for the full method and interpretation.

## Quickstart

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

Selective prediction (v0.2) wraps the same evaluation in a conformal layer:

```python
from pactstress import evaluate_loso_conformal

result = evaluate_loso_conformal(
    df, feature_cols=feature_cols,
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    alpha=0.1,
)
print(result["mean_coverage"], result["mean_abstain_rate"])
```

For a lower-level API — inspecting individual prediction sets rather than
aggregate summaries — use `compute_qhat`, `predict_sets`, and
`summarize_predictions` directly; see their docstrings in
[`pactstress/src/pactstress/conformal.py`](pactstress/src/pactstress/conformal.py).

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
per window using NeuroKit2. See
[`pactstress/examples/wesad_example.py`](pactstress/examples/wesad_example.py)
for a full end-to-end script (population-level baseline vs.
PACT-calibrated, per-subject breakdown).

## API

| Function / Class | Purpose |
|---|---|
| `segment_windows(signal_length, fs, window_sec, step_sec)` | Compute sliding-window index boundaries |
| `majority_label(label_segment, valid_labels)` | Assign a window's label by majority vote |
| `extract_windowed_features(...)` | Full raw-signal → feature-table pipeline (requires `neurokit2`) |
| `PACTCalibrator(feature_cols, n_calib_windows, baseline_label)` | Personal baseline calibration |
| `evaluate_loso(df, feature_cols, model, label_col, subject_col)` | Leave-One-Subject-Out cross-validation |
| `evaluate_loso_conformal(df, feature_cols, model, alpha, n_calib_subjects, ...)` | LOSO wrapped in split conformal prediction |
| `compute_qhat`, `predict_sets`, `summarize_predictions` | Lower-level conformal building blocks |

## Things this package will tell you that others won't

**An earlier version of this calibration leaked evaluation data into
itself, and it looked *better* for it.** During development, subject
reference statistics were computed from *all* of a subject's baseline
windows, including ones later used for evaluation. That produced
misleadingly high accuracy (95–97%) because evaluation windows contributed
to their own normalization reference. `PACTCalibrator` avoids this with a
small, fixed subset of a subject's earliest baseline windows as the
calibration reference, removing those specific windows from the returned
dataset entirely — no window is ever normalized using its own value, and
no evaluation data leaks into calibration statistics. The conformal layer
(`evaluate_loso_conformal`) follows the same discipline: calibration
subjects are split out subject-wise, distinct from both the proper
training group and the held-out test subject.

**Personalization and uncertainty-detection can work against each other.**
See "The problem underneath the problem" above — PACT calibration improves
mean accuracy but can simultaneously erase the exact signal a conformal
layer needs to flag an atypical subject. This package reports that tension
rather than only the configuration that looks best.

## Honest limitations

- Calibration requires a minimum number of baseline windows per subject
  (`n_calib_windows`, default 4). Subjects with fewer baseline windows
  than this will raise a `ValueError`.
- `PACTCalibrator` assumes a fixed calibration reference collected once;
  it does not currently support adaptive/rolling recalibration over time.
- Developed and validated on WESAD (15 subjects, chest-worn sensors,
  laboratory-induced stress). Performance on wrist-worn consumer sensors
  or real-world (non-laboratory) stress has not been evaluated.
- Conformal prediction guarantees *marginal* coverage (averaged across all
  test examples), not per-subject coverage — a subject with unusual
  features can still be individually over- or under-covered even when the
  overall guarantee holds. This is a known property of the method, not a
  bug in this implementation.
- `n_calib_subjects` in `evaluate_loso_conformal` is a small, fixed number
  of subjects per fold; with only 15 subjects total in WESAD, the
  calibration set itself is small, which makes empirical coverage noisier
  than it would be with a larger dataset. Treat per-fold coverage numbers
  as approximate, and look at the mean across folds rather than any single
  fold in isolation.
- The conformal layer ensures the model reports honestly when it doesn't
  know; it does not fix the underlying reason a subject like S2 is hard to
  classify in the first place.

## Running tests

```bash
cd pactstress
pip install .[dev]
pytest tests/
```

## Citation

If you use this package, please cite the accompanying study:

> PACT: Personalized Adaptive Calibration for Stress Detection from
> Multimodal Wearable Sensor Data. 2026.
> Repository: https://github.com/sushan5140/pactstress

## License

MIT
