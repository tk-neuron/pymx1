import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass


def extract_waveforms(sig, peaks, n_pre=20, n_post=40):
    sig = np.atleast_2d(sig)
    n_channels, n_frames = sig.shape

    peaks = np.sort(peaks)
    peaks_ = peaks[(n_pre <= peaks) & (peaks <= n_frames - n_post)]  # waveforms must be within data range
    waveforms = np.array([sig[:, peak - n_pre: peak + n_post] for peak in peaks_])
    return peaks_, waveforms


def concat_waveforms(waveforms):
    waveforms = np.atleast_3d(waveforms)
    n_peaks, n_channels, n_frames = waveforms.shape
    return waveforms.reshape(-1, n_channels * n_frames)


def normalize_waveform(waveform):
    return waveform / (np.max(waveform) - np.min(waveform))


def interpolate_waveform(wv, n=5, t_pre=-1, t_post=2):
    @dataclass
    class result:
        t: np.ndarray
        wv: np.ndarray
        t_: np.ndarray
        wv_: np.ndarray
        
    ns = wv.shape[0]
    t = np.linspace(t_pre, t_post, num=ns, endpoint=True)
    t_ = np.linspace(t_pre, t_post, num=ns * n, endpoint=True)
    f = interp1d(t, wv, kind='cubic')
    return result(t, wv, t_, f(t_))


def extract_template(waveforms):
    return np.mean(waveforms, axis=0)
