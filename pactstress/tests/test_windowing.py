import numpy as np
import pytest

from pactstress.windowing import segment_windows, majority_label


def test_segment_windows_basic():
    # 10 seconds of signal at fs=10 -> 100 samples, 4-second windows, 2-second step
    windows = segment_windows(signal_length=100, fs=10, window_sec=4, step_sec=2)
    assert windows[0] == (0, 40)
    assert windows[1] == (20, 60)
    # every window should be exactly window_size long
    assert all((end - start) == 40 for start, end in windows)


def test_segment_windows_signal_shorter_than_window():
    windows = segment_windows(signal_length=10, fs=10, window_sec=4, step_sec=2)
    assert windows == []


def test_segment_windows_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        segment_windows(signal_length=100, fs=10, window_sec=0, step_sec=2)
    with pytest.raises(ValueError):
        segment_windows(signal_length=100, fs=10, window_sec=4, step_sec=-1)


def test_majority_label_simple_majority():
    labels = np.array([1, 1, 1, 2, 2])
    assert majority_label(labels) == 1


def test_majority_label_filters_invalid():
    labels = np.array([3, 3, 3, 3])  # e.g. transient/undefined condition
    result = majority_label(labels, valid_labels={1, 2})
    assert result is None


def test_majority_label_passes_valid():
    labels = np.array([2, 2, 2, 1])
    result = majority_label(labels, valid_labels={1, 2})
    assert result == 2
