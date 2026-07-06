"""
Sliding-window segmentation for physiological time-series signals.
"""

import numpy as np


def segment_windows(signal_length, fs, window_sec=60, step_sec=30):
    """
    Compute start/end sample indices for a sliding window scheme.

    Parameters
    ----------
    signal_length : int
        Total number of samples in the signal.
    fs : int or float
        Sampling rate in Hz.
    window_sec : float, default=60
        Window length in seconds.
    step_sec : float, default=30
        Step size between consecutive window starts, in seconds.
        step_sec < window_sec produces overlapping windows.

    Returns
    -------
    list of tuple(int, int)
        A list of (start_index, end_index) pairs, one per window.
    """
    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive.")

    window_size = int(fs * window_sec)
    step_size = int(fs * step_sec)

    if window_size > signal_length:
        return []

    starts = range(0, signal_length - window_size + 1, step_size)
    return [(s, s + window_size) for s in starts]


def majority_label(label_segment, valid_labels=None):
    """
    Determine the majority label within a window of per-sample labels.

    Parameters
    ----------
    label_segment : array-like
        Per-sample labels within a single window.
    valid_labels : set or list, optional
        If provided, windows whose majority label is not in this set
        return None instead of the majority label. Useful for excluding
        transient/undefined condition codes.

    Returns
    -------
    int or None
        The majority label, or None if it is not in `valid_labels`.
    """
    label_segment = np.asarray(label_segment)
    values, counts = np.unique(label_segment, return_counts=True)
    majority = values[np.argmax(counts)]

    if valid_labels is not None and majority not in valid_labels:
        return None
    return int(majority)
