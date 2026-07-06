"""
Physiological feature extraction for ECG, EDA, and respiration signals.

Requires neurokit2 (https://github.com/neuropsychology/NeuroKit).
"""

import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
except ImportError as e:
    raise ImportError(
        "pactstress.features requires neurokit2. Install it with "
        "`pip install neurokit2`."
    ) from e

from .windowing import segment_windows, majority_label


def extract_ecg_features(ecg_segment, fs):
    """Extract heart-rate variability features from one ECG window."""
    cleaned = nk.ecg_clean(ecg_segment, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    hrv = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
    return {
        "MeanNN": hrv["HRV_MeanNN"].values[0],
        "SDNN": hrv["HRV_SDNN"].values[0],
        "RMSSD": hrv["HRV_RMSSD"].values[0],
    }


def extract_eda_features(eda_segment, fs):
    """Extract skin-conductance features from one EDA window."""
    signals, info = nk.eda_process(eda_segment, sampling_rate=fs)
    scr_peaks = info["SCR_Peaks"]
    amplitude_mean = (
        float(np.mean(signals["SCR_Amplitude"])) if len(scr_peaks) > 0 else 0.0
    )
    return {
        "EDA_Mean": float(np.mean(eda_segment)),
        "EDA_SCR_Count": len(scr_peaks),
        "EDA_SCR_Amplitude_Mean": amplitude_mean,
    }


def extract_resp_features(resp_segment, fs):
    """Extract respiration rate/amplitude features from one respiration window."""
    signals, info = nk.rsp_process(resp_segment, sampling_rate=fs)
    if "RSP_Rate_Mean" in info:
        rate_mean = info["RSP_Rate_Mean"]
    else:
        rate_mean = float(np.mean(signals["RSP_Rate"]))
    return {
        "Resp_Rate_Mean": rate_mean,
        "Resp_Amplitude_Mean": float(np.mean(signals["RSP_Amplitude"])),
    }


def extract_windowed_features(
    ecg_signal,
    eda_signal,
    resp_signal,
    labels,
    subject_id,
    fs=700,
    window_sec=60,
    step_sec=30,
    valid_labels=(1, 2),
    stress_label=2,
):
    """
    Segment aligned ECG/EDA/respiration signals into windows and extract
    physiological features + a binary stress label for each window.

    Parameters
    ----------
    ecg_signal, eda_signal, resp_signal : array-like
        Raw signals, all the same length, all sampled at `fs`.
    labels : array-like
        Per-sample condition labels, same length as the signals.
    subject_id : str
        Identifier for the subject these signals belong to. Stored in the
        output DataFrame's 'subject' column for later subject-wise splitting.
    fs : int, default=700
        Sampling rate in Hz.
    window_sec, step_sec : float
        Window length and step size in seconds. step_sec < window_sec
        gives overlapping windows.
    valid_labels : tuple of int, default=(1, 2)
        Per-sample label values to keep (others are dropped as transient/
        undefined). Follows WESAD's convention: 1=baseline, 2=stress.
    stress_label : int, default=2
        Which value in `valid_labels` indicates the stress condition; used
        to construct the binary 'label' output column (1=stress, 0=other).

    Returns
    -------
    pandas.DataFrame
        One row per window, with feature columns, a binary 'label' column,
        and a 'subject' column.
    """
    windows = segment_windows(len(ecg_signal), fs, window_sec, step_sec)
    rows = []

    for start, end in windows:
        label_segment = labels[start:end]
        maj_label = majority_label(label_segment, valid_labels=set(valid_labels))
        if maj_label is None:
            continue

        try:
            row = {}
            row.update(extract_ecg_features(ecg_signal[start:end], fs))
            row.update(extract_eda_features(eda_signal[start:end], fs))
            row.update(extract_resp_features(resp_signal[start:end], fs))
            row["label"] = 1 if maj_label == stress_label else 0
            row["subject"] = subject_id
            rows.append(row)
        except Exception:
            # Skip windows where signal processing fails (e.g. corrupted segment).
            continue

    return pd.DataFrame(rows)
