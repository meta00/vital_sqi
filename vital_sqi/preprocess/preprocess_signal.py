
import numpy as np
from scipy import signal


def taper_signal(s, window=None, shift_min_to_zero=True):
    """Pin the leftmost and rightmost signal to the zero baseline
    and amplify the remainder according to the window shape.

    Parameters
    ----------
    s :
        array-like signal
    window :
        sequence, array of floats indicates the windows types
        as described in scipy.windows (Default value = None)
    shift_min_to_zero :
        (Default value = True)

    Returns
    -------

    
    """
    if shift_min_to_zero:
        s = s-np.min(s)
    if window is None:
        window = signal.windows.tukey(len(s), 0.9)
    s = np.array(window) * s
    return np.array(s)


def smooth_signal(s, window_len=5, window='flat'):
    """

    Parameters
    ----------
    s :
        
    window_len :
        int
        (Default value = 5)
    window :
         (Default value = 'flat')
         Options are: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'

    Returns
    -------

    """
    s = np.array(s)
    if s.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if s.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return s

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', "
                          "'hamming', 'bartlett', 'blackman'")

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
    """expose
    This method is ONLY used for small segment to compare with the template.
    Please change to use scipy.signal.resample function for the purpose of
    resampling.

    Parameters
    ----------
    s :
        param window_size:
    window_size :
        

    Returns
    -------

    
    """
    scale_res = []
    if len(s) == window_size:
        return np.array(s)
    if len(s)<window_size:
        #spanning the signal
        span_ratio = (window_size/len(s))
        for idx in range(0,int(window_size)):
            if idx-span_ratio<0:
                scale_res.append(s[0])
            else:
                scale_res.append(np.mean(s[int(idx/span_ratio)]))
    else:
        scale_res = squeeze_template(s, window_size)

    # scale_res = smooth_window(scale_res, span_size=5)
    # scale_res = smooth(scale_res, span_size=5)
    smoothed_scale_res = smooth_signal(scale_res)
    return np.array(smoothed_scale_res)


def squeeze_template(s, width):
    """handy

    Parameters
    ----------
    s :
        param width:
    width :
        

    Returns
    -------

    
    """
    s = np.array(s)
    total_len = len(s)
    span_unit = 2
    out_res = []
    for i in range(int(width)):
        if i == 0:
            centroid = (total_len/width)*i
        else:
            centroid = (total_len/width)*i
        left_point = int(centroid)-span_unit
        right_point = int(centroid+span_unit)
        if left_point <0:
            left_point=0
        if right_point >len(s):
            left_point=len(s)
        out_res.append(np.mean(s[left_point:right_point]))
    return np.array(out_res)