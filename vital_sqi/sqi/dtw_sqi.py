import numpy as np
from dtw import dtw
from vital_sqi.common.generate_template import ppg_absolute_dual_skewness_template, \
        ppg_dual_double_frequency_template,ppg_nonlinear_dynamic_system_template

def dtw_sqi(x,template_type=0):
    """
    Expose
    Using DTW to get the mapping point distance.
    The dwt sqi output is the ratio between the sum of mapping distance and the trace of cost matrix.
    The closer to 1 the better sqi
    :param x: sequence, array of signal represent the template
    :param template_type: int, default = 0
    :return: float, the ratio score of matching with the chosen template
    """
    if template_type == 0:
        reference = ppg_nonlinear_dynamic_system_template(len(x)).reshape(-1)
    elif template_type == 1:
        reference = ppg_dual_double_frequency_template(len(x))
    elif template_type == 2:
        reference = ppg_absolute_dual_skewness_template(len(x))
    alignmentOBE = dtw(x, reference,keep_internals=True,step_pattern='asymmetric',
        open_end=True, open_begin=True)

    match_distance = []
    for i in range(len(alignmentOBE.index2)):
        match_distance.append(alignmentOBE.costMatrix[i][alignmentOBE.index2[i]])
    trace_ratio = np.sum(match_distance)/alignmentOBE.costMatrix.trace()

    return trace_ratio