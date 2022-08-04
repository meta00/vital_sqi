"""Signal quality indexes based on dynamic template matching
"""
import numpy as np
from vital_sqi.common.generate_template import (
        ppg_absolute_dual_skewness_template,
        ppg_dual_double_frequency_template,
        ppg_nonlinear_dynamic_system_template,
        ecg_dynamic_template
    )
from vital_sqi.common.utils import check_valid_signal
from scipy.spatial.distance import euclidean
from scipy.signal import resample
from librosa.sequence import dtw
from sklearn.preprocessing import MinMaxScaler

def dtw_sqi(s, template_type, template_size = 100,simple_mode=False):
    """
    Euclidean distance between signal and its template

    Parameters
    ----------
    s :
        array_like, signal containing int or float values.
        
    template_sequence :
        array_like, signal containing int or float values.

    Returns
    -------

    """
    check_valid_signal(s)
    s = resample(s, template_size).reshape(-1)
    if template_type > 3 or type(template_type) != int:
        raise ValueError("Invalid template type")
    if template_type == 0:
        reference = ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    elif template_type == 1:
        reference = ppg_dual_double_frequency_template(template_size)
    if template_type == 2:
        reference = ppg_absolute_dual_skewness_template(template_size)
    if template_type == 3:
        reference = np.array(ecg_dynamic_template(template_size)).reshape(-1)

    if simple_mode:
        cost = 0
        for i in range(template_size):
            cost = cost + euclidean([s[i]], [reference[i]])
        dtw_cost = cost / template_size
    else:
        beat = resample(s, template_size)
        scaler = MinMaxScaler(feature_range=(0, 1))
        beat = scaler.fit_transform(beat.reshape(-1, 1)).reshape(-1)

        reference = resample(reference, template_size)
        scaler = MinMaxScaler(feature_range=(0, 1))
        reference = scaler.fit_transform(reference.reshape(-1, 1)).reshape(-1)

        D, wp = dtw(beat,reference)
        dtw_cost = np.mean([D[i][j] for i, j in zip(wp[:, 1], wp[:, 0])]).item()

    return dtw_cost