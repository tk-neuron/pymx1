import numpy as np
from scipy.signal import find_peaks
from scipy import stats


def detect_peaks(sig, thr, distance=100):
    peaks, _ = find_peaks(-1 * sig, height=thr, distance=distance)
    return peaks


def thr_mad(sig, n_mad=6):
    return stats.median_abs_deviation(sig) * n_mad


def thr_std(sig, n_std=7):
    return np.std(sig) * n_std
