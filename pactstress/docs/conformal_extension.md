# Extension: Selective Prediction for PACT via Conformal Prediction

*This document extends the PACT paper and package (v0.1) with a second
mechanism, released as v0.2. It assumes familiarity with the original PACT
methodology — see the main [README](../README.md) and paper for background
on the dataset, features, and calibration approach.*

## Motivation

PACT's own case study found a real limit to personal baseline calibration.
Subject S2's stress response didn't match the population pattern the model
learned — clean, artifact-free signals, but a physiologically atypical
reaction (stable heart rate, slow deep breathing) during the labeled stress
task. PACT still forced a binary prediction on every window, and got S2
confidently, consistently wrong (F1 = 0.000 for that subject even after
calibration).

That's the wrong failure mode for a health-adjacent system. A model that's
*wrong and confident* is worse than one that's *uncertain and says so* — the
second case, at least, tells the user (or a downstream system) not to trust
the output. This extension addresses that directly: instead of asking
"stress or not," we ask "stress, not-stress, or *I can't confidently tell*,"
and back that third option with a real statistical guarantee rather than an
ad hoc confidence heuristic.

## Method: split conformal prediction

We use split conformal prediction (Vovk et al., 2005), a distribution-free
method that wraps any classifier's probability outputs in a procedure with a
finite-sample coverage guarantee — it does not assume the classifier is
well-calibrated, well-specified, or even good; the guarantee holds
regardless.

For each LOSO fold:

1. **Split training subjects further**, subject-wise, into a *proper
   training* group and a small *calibration* group (distinct from the held-
   out test subject, and distinct from each other). This mirrors the same
   leakage discipline as `PACTCalibrator`: calibration data must be fixed
   and independent of what's being evaluated.
2. **Fit the classifier** on the proper training group only.
3. **Compute nonconformity scores** on the calibration group: for each
   calibration example, `1 - P(true class)` under the fitted model.
4. **Set the threshold** `qhat` as the `⌈(n+1)(1-α)⌉/n` quantile of those
   scores — the standard finite-sample-correct split conformal threshold.
5. **Construct prediction sets** on the test subject: include class `c` in
   the set if `P(c) ≥ 1 - qhat`. This yields `{0}`, `{1}`, or `{0, 1}`
   (abstain) per window.

With `α = 0.1`, the guarantee is: averaged over repeated calibration/test
splits, the true label falls inside the predicted set at least 90% of the
time — regardless of whether the underlying classifier is any good. What
conformal prediction adds is not better accuracy, but an honest signal of
*when the model doesn't know*.

## What to expect from this extension

Applying this to the same WESAD features and Random Forest classifier used
in PACT, the outcome we're specifically checking for is whether S2's
windows shift from confident-and-wrong toward abstain (`{0, 1}`) — i.e.,
whether the model's uncertainty machinery actually recognizes S2 as a hard
case, rather than continuing to confidently misclassify.

Run `evaluate_loso_conformal` (see `examples/`) to reproduce this on real
WESAD data. Report, per subject:
- **coverage** — should track the target (1-α) on average across subjects
- **abstain_rate** — expected to be higher for atypical subjects (S2, and
  possibly S8/S15, which showed calibration instability in the original
  PACT results) than for well-behaved subjects
- **singleton_accuracy** — accuracy restricted to confident predictions
  only; this should be noticeably higher than PACT's raw per-subject
  accuracy, since it excludes the cases the model flags as uncertain

## Usage

```python
from sklearn.ensemble import RandomForestClassifier
from pactstress import evaluate_loso_conformal

result = evaluate_loso_conformal(
    df,                      # same feature DataFrame used with evaluate_loso
    feature_cols=feature_cols,
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    alpha=0.1,               # target 90% coverage
    n_calib_subjects=2,      # subjects reserved for conformal calibration per fold
)

print(result["mean_coverage"])          # should track ~0.90
print(result["mean_abstain_rate"])
print(result["per_subject"])            # per-subject coverage / abstain / singleton accuracy
```

For a lower-level API — e.g., to inspect individual prediction sets rather
than aggregate summaries — use `compute_qhat`, `predict_sets`, and
`summarize_predictions` directly; see their docstrings in
`src/pactstress/conformal.py`.

## Honest limitations

- Conformal prediction guarantees *marginal* coverage (averaged across all
  test examples), not per-subject coverage. A subject with unusual features
  can still be individually over- or under-covered even when the overall
  guarantee holds — this is a known property of the method, not a bug in
  this implementation.
- `n_calib_subjects` is a small, fixed number of subjects per fold; with
  only 15 subjects total in WESAD, the calibration set itself is small,
  which makes the empirical coverage noisier than it would be with a larger
  dataset. Treat per-fold coverage numbers as approximate, and look at the
  mean across folds rather than any single fold in isolation.
- This does not fix the underlying reason a subject is hard to classify
  (see the S2 case study in the main paper) — it only ensures the model
  reports honestly when it doesn't know, rather than papering over the
  difficulty with a confident wrong answer.

## Reference

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a
Random World*. Springer. (Foundational split conformal prediction method.)

Angelopoulos, A. N., & Bates, S. (2023). A gentle introduction to conformal
prediction and distribution-free uncertainty quantification. *Foundations
and Trends in Machine Learning*, 16(4), 494-591.
