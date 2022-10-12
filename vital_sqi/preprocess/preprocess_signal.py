
import numpy as np
import pandas as pd
from scipy import signal
from vital_sqi.common.generate_template import squeeze_template


def taper_signal(s, window=None, shift_min_to_zero=True):
    """Pinning the leftmost and rightmost signal to the zero baseline
    and amplifying the remainder according to the window shape.

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.
    window :
        sequence, array of floats indicates the windows types
        as described in scipy.windows.
        (Default value = None)
    shift_min_to_zero : bool
        (Default value = True)

    Returns
    -------
    processed_s : pandas DataFrame
        Processed signal.
    """
    if shift_min_to_zero:
        s = s-np.min(s)
    if window is None:

        window = signal.windows.tukey(len(s), 0.9)
    s = np.array(window) * s
    return np.array(s)


def smooth_signal(s, window_len=5, window='flat'):
    """ Smoothing signal
    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.
    window_len : int
        (Default value = 5)
    window : str
         (Default value = 'flat')
         Options are: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.

    Returns
    -------
    processed_s : pandas DataFrame
        Processed signal.
    """
    assert isinstance(window_len, int), 'Expected an integer value.'
    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'], \
        'Options are "flat", "hanning", "hamming", "bartlett", "blackman"'

    s = np.array(s)
    if s.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if s.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return s

    s = np.r_[s[window_len - 1:0:-1], s, s[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    # y = np.convolve(w / w.sum(), s, mode='valid')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y


def scale_pattern(s, window_size):
    """
    This method is ONLY used for small segment to compare with the template.
    Please change to use scipy.signal.resample function for the purpose of
    resampling.

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.
    window_size : int

    Returns
    -------
    processed_s : pandas DataFrame
        Processed signal.
    
    """
    scale_res = []
    if len(s) == window_size:
        return np.array(s)
    if len(s) < window_size:
        # spanning the signal
        span_ratio = (window_size/len(s))
        for idx in range(0, int(window_size)):
            if idx-span_ratio < 0:
                scale_res.append(s[0])
            else:
                scale_res.append(np.mean(s[int(idx/span_ratio)]))
    else:
        scale_res = squeeze_template(s, window_size)

    # scale_res = smooth_window(scale_res, span_size=5)
    # scale_res = smooth(scale_res, span_size=5)
    smoothed_scale_res = smooth_signal(scale_res)
    processed_s = pd.DataFrame(smoothed_scale_res)
    return processed_s

