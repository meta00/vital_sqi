import numpy as np
import sys, os
if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    sys.stdout = open(os.devnull, 'w')
    from dtw import dtw
    sys.stdout = sys.__stdout__
else:
    from dtw import dtw

import logging

from vital_sqi.common.generate_template import (
        ppg_absolute_dual_skewness_template,
        ppg_dual_double_frequency_template,
        ppg_nonlinear_dynamic_system_template,
        ecg_dynamic_template
    )
from vital_sqi.common.utils import check_valid_signal


def dtw_sqi(x, template_type=0):
    """
    Using DTW to get the mapping point distance between a signal and its
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
    type
        float, the matching score with the chosen template

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
        ratio = float(1)
    else:
        ratio = float(np.sum(match_distance)/trace)

    return ratio
