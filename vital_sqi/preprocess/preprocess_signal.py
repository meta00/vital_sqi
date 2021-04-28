import numpy as np
from scipy import signal

def tapering(signal_data,window=None,shift_min_to_zero=True):
    """
    expose
    Pin the leftmost and rightmost signal to the zero baseline
    and amplify the remainder according to the window shape
    :param signal_data: list,
    :param window:sequence, array of floats indicates the windows types
    as described in scipy.windows
    :return: the tapered signal
    """
    if shift_min_to_zero:
        signal_data = signal_data-np.min(signal_data)
    if window == None:
        window = signal.windows.tukey(len(signal_data),0.9)
    signal_data_tapered = np.array(window) * (signal_data)
    return np.array(signal_data_tapered)

def smooth(x,window_len=5,window='flat'):
    """
    expose
    :param x:
    :param window_len:
    :param window:
    :return:
    """
    x = np.array(x)
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    # y = np.convolve(w / w.sum(), s, mode='valid')
    y = np.convolve(w / w.sum(), x, mode='same')
    return y

def scale_pattern(s,window_size):
    """
    expose
    This method is ONLY used for small segment to compare with the template.
    Please change to use scipy.signal.resample function for the purpose of
    resampling.

    :param s:
    :param window_size:
    :return:
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
    smmoothed_scale_res = smooth(scale_res)
    return np.array(smmoothed_scale_res)

def squeeze_template(s,width):
    """
    handy
    :param s:
    :param width:
    :return:
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