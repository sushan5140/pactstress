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

Validated on real WESAD data (see "Validated results" below), this
mechanism does successfully catch the model's worst population-level
failure case, and partially catches S2 — but combining it naively with
PACT's own calibration turns out to weaken rather than strengthen this
effect. That tension is itself one of this extension's findings.

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

## Validated results on WESAD

We ran `evaluate_loso_conformal` (α = 0.1, targeting 90% coverage) in two
configurations: directly on raw population-level features, and on top of
PACT-calibrated features. The results reveal a real tension between the two
techniques, not a clean win from stacking them.

### Configuration 1: Conformal prediction on raw (uncalibrated) features

| Metric | Value |
|---|---|
| Mean coverage | 0.899 (target: 0.90) |
| Mean abstain rate | 0.153 |
| Mean singleton accuracy | 0.930 |

Coverage lands almost exactly on target, confirming the conformal guarantee
holds on real physiological data, not just synthetic test cases. Singleton
accuracy (0.930) is meaningfully higher than PACT's raw population-level
accuracy (0.875) — excluding the ~15% of windows the model flags as
uncertain, its confident predictions are noticeably more trustworthy.

Critically, **S10 — the worst-performing subject in the original
population-level model (42.9% accuracy, confidently wrong on most
windows) — abstains on 100% of its windows** under conformal prediction,
rather than being confidently misclassified. This is the core behavior this
extension was built to produce. **S2 is partially caught**: an 11.9%
abstain rate (the 6th-highest of 15 subjects) and the lowest singleton
accuracy in the dataset (0.673) — the model doesn't abstain on S2 nearly as
often as on S10, but it is noticeably less confident and less accurate on
S2 even when it does commit to an answer.

### Configuration 2: Conformal prediction on top of PACT-calibrated features

| Metric | Value |
|---|---|
| Mean coverage | 0.865 (target: 0.90) |
| Mean abstain rate | 0.136 |
| Mean singleton accuracy | 0.926 |

This is the more complete version of the intended pipeline (PACT, then
conformal on top), and it does not simply improve on Configuration 1.
Coverage drops meaningfully below the 0.90 target — a real shortfall, not
noise. More strikingly, **S2's abstain rate drops to 0%**: the conformal
layer, which partially caught S2 when working on raw features, misses S2
entirely once PACT calibration has been applied first. S2's singleton
accuracy under this configuration (0.618) matches PACT's original
calibrated accuracy for that subject exactly, since the model is now never
abstaining and always committing to a (frequently wrong) answer for S2.

Meanwhile S10 is still partially caught (81.4% abstain rate) — reduced from
100% in Configuration 1, but still substantial.

### Interpretation

The most likely explanation: PACT's per-subject normalization is, by
design, effective precisely because it removes between-person baseline
differences — that is its entire purpose. But that same normalization may
also remove the distributional signal a conformal/uncertainty layer needs
to recognize that a subject's data is atypical relative to the population.
By centering S2's features around S2's own baseline, PACT makes S2's
calibrated data look statistically unremarkable next to everyone else's
calibrated data, even though the underlying physiological response is
genuinely unusual (see the S2 case study in the main paper).

This is a genuine, non-obvious tension between two individually reasonable
design goals — personalization (PACT) and outlier-awareness (conformal
prediction) — not a bug in either component. It suggests naively stacking
personalization on top of uncertainty quantification can undermine the
latter, and that combining them well is a real open problem rather than a
solved one. Practical implications and possible directions:

- **Conformal prediction on raw features may be preferable** to stacking
  it after PACT, specifically when the goal is catching atypical
  individuals rather than maximizing average accuracy.
- **A joint calibration procedure** — one that computes the conformal
  threshold using un-normalized, or partially normalized, distributional
  information — might recover S2-detection while retaining PACT's accuracy
  gains, but this was not implemented or tested here.
- **Reporting both configurations**, rather than only the better-looking
  one, is itself the point: a system that quietly loses its ability to
  flag hard cases when combined with an accuracy-improving technique is a
  realistic failure mode worth knowing about before deployment, not a
  result to average away.

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
