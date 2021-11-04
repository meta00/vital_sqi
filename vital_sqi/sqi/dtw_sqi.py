import numpy as np
import sys
import os
if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    from dtw import dtw
    sys.stdout = old_stdout
else:
    from dtw import dtw

from vital_sqi.common.generate_template import (
        ppg_absolute_dual_skewness_template,
        ppg_dual_double_frequency_template,
        ppg_nonlinear_dynamic_system_template,
        ecg_dynamic_template
    )
from vital_sqi.common.utils import check_valid_signal
from scipy.spatial.distance import euclidean

def compute_dtw_distance(input_sequence, template_sequence):
    dtw_distances = np.ones((len(input_sequence),len(template_sequence))) * np.inf
    #first matching sample is set to zero
    dtw_distances[0,0] = 0
    for i in range(len(input_sequence)):
        for j in range(len(template_sequence)):
            cost = euclidean(input_sequence[i],template_sequence[j])
            # dtw_distances



def dtw_sqi(x, template_type=0):
    """Using DTW to get the mapping point distance between a signal and its
    template. The DTW SQI is the ratio of the distance sum to
    the trace of cost matrix. The closer to 1 the better SQI.

    Parameters
    ----------
    x :
        array_like, signal containing int or float values.
    template_type :
        int,
        0: ppg_absolute_dual_skewness_template,
        1: ppg_dual_double_frequency_template,
        2: ppg_nonlinear_dynamic_system_template,
        3: ecg_dynamic_template
        default = 0

    Returns
    -------

    """
    check_valid_signal(x)
    if template_type > 3 or type(template_type) != int:
        raise ValueError("Invalid template type")
    if template_type == 0:
        reference = ppg_nonlinear_dynamic_system_template(len(x)).reshape(-1)
    elif template_type == 1:
        reference = ppg_dual_double_frequency_template(len(x))
    if template_type == 2:
        reference = ppg_absolute_dual_skewness_template(len(x))
    if template_type == 3:
        reference = ecg_dynamic_template(len(x))
    alignmentOBE = dtw(x, reference, keep_internals=True,
                       step_pattern='asymmetric', open_end=True,
                       open_begin=True)
    match_distance = []
    for i in range(len(alignmentOBE.index2)):
        match_distance.append(
                alignmentOBE.costMatrix[i][alignmentOBE.index2[i]]
                )
    trace = alignmentOBE.costMatrix.trace()
    if trace == 0:
        ratio = float(np.log(1))
    else:
        ratio = float(np.log(np.sum(match_distance)/trace))
    return ratio
