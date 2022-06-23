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

def dtw_sqi(s, template_type, template_size = 100):
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
    s = resample(s, template_size)
    if template_type > 3 or type(template_type) != int:
        raise ValueError("Invalid template type")
    if template_type == 0:
        reference = ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    elif template_type == 1:
        reference = ppg_dual_double_frequency_template(template_size)
    if template_type == 2:
        reference = ppg_absolute_dual_skewness_template(template_size)
    if template_type == 3:
        reference = ecg_dynamic_template(template_size)

    dtw_distances = np.ones((template_size,template_size)) * \
                    np.inf
    #first matching sample is set to zero
    dtw_distances[0, 0] = 0
    cost=0
    for i in range(template_size):
        cost = cost + euclidean(s[i], reference[i])
    return cost/template_size
