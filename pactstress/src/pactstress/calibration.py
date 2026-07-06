"""
PACT: Personalized Adaptive Calibration for sTress detection.

Normalizes each subject's physiological features against their own
resting-state statistics, computed from a small, fixed set of that
subject's baseline windows, rather than population-wide statistics.

This avoids two leakage pitfalls that are easy to introduce by accident:

1. Computing a subject's reference statistics from *all* of their windows
   (including the ones being evaluated) lets each window influence its own
   normalization, inflating apparent performance.
2. Even restricting the reference to baseline windows still leaks if those
   same baseline windows are also present in the evaluation set.

PACT avoids both by using a small, fixed subset of each subject's earliest
baseline windows purely as a calibration reference, then removing those
specific windows from the dataset entirely before training or evaluation.
"""

import pandas as pd


class PACTCalibrator:
    """
    Personal baseline calibration for subject-independent physiological
    stress detection.

    Parameters
    ----------
    feature_cols : list of str
        Names of the feature columns to calibrate.
    n_calib_windows : int, default=4
        Number of each subject's earliest baseline windows to use as the
        calibration reference. These windows are removed from the returned
        dataset so that no window is ever normalized using its own value.
    baseline_label : int, default=0
        The value in `label_col` that identifies non-stress/baseline
        windows, from which calibration windows are drawn.

    Examples
    --------
    >>> calibrator = PACTCalibrator(feature_cols=["MeanNN", "RMSSD"])
    >>> calibrated_df = calibrator.fit_transform(
    ...     df, subject_col="subject", label_col="label"
    ... )
    """

    def __init__(self, feature_cols, n_calib_windows=4, baseline_label=0):
        if n_calib_windows < 1:
            raise ValueError("n_calib_windows must be at least 1.")
        self.feature_cols = list(feature_cols)
        self.n_calib_windows = n_calib_windows
        self.baseline_label = baseline_label
        self.calibration_stats_ = {}

    def fit_transform(self, df, subject_col="subject", label_col="label"):
        """
        Compute per-subject calibration statistics and apply them.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain `feature_cols`, `subject_col`, and `label_col`.
        subject_col : str, default="subject"
            Column identifying which subject each row belongs to.
        label_col : str, default="label"
            Column identifying the stress/baseline condition of each row.

        Returns
        -------
        pandas.DataFrame
            A copy of `df`, with feature columns standardized per-subject
            and calibration-reference rows removed. Index is reset.

        Raises
        ------
        ValueError
            If any subject has fewer baseline windows than
            `n_calib_windows`.
        """
        df_out = df.copy()
        for col in self.feature_cols:
            df_out[col] = df_out[col].astype(float)

        rows_to_drop = []
        self.calibration_stats_ = {}

        for subject in df[subject_col].unique():
            subject_mask = df[subject_col] == subject
            baseline_mask = subject_mask & (df[label_col] == self.baseline_label)
            baseline_indices = df[baseline_mask].index.tolist()

            if len(baseline_indices) < self.n_calib_windows:
                raise ValueError(
                    f"Subject '{subject}' has only {len(baseline_indices)} "
                    f"baseline window(s), fewer than n_calib_windows="
                    f"{self.n_calib_windows}. Reduce n_calib_windows or "
                    f"exclude this subject."
                )

            calib_indices = baseline_indices[: self.n_calib_windows]
            calib_data = df.loc[calib_indices, self.feature_cols]

            mean = calib_data.mean()
            std = calib_data.std()
            self.calibration_stats_[subject] = {"mean": mean, "std": std}

            remaining_indices = [
                i for i in df[subject_mask].index.tolist() if i not in calib_indices
            ]
            for col in self.feature_cols:
                df_out.loc[remaining_indices, col] = (
                    df.loc[remaining_indices, col] - mean[col]
                ) / (std[col] + 1e-8)

            rows_to_drop.extend(calib_indices)

        df_out = df_out.drop(index=rows_to_drop).reset_index(drop=True)
        return df_out

    def get_calibration_stats(self, subject):
        """Return the {mean, std} calibration reference computed for a subject."""
        if subject not in self.calibration_stats_:
            raise KeyError(
                f"No calibration statistics for subject '{subject}'. "
                f"Call fit_transform first."
            )
        return self.calibration_stats_[subject]
