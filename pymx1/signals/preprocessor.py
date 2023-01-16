import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y


def offset_signal(data):
    data = np.atleast_2d(data)
    return data - np.mean(data, axis=1).reshape(-1, 1)


def batch_filter(data, lowcut=500, highcut=3000, fs=20000, order=3):
    data_av = offset_signal(data)
    data_filt = bandpass_filter(data=data_av, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    return data_filt
