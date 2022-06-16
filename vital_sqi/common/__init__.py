"""
vital_sqi.common
================
A subpackage for shared operations across package vital_sqi
"""

from vital_sqi.common.generate_template import (
	ppg_dual_double_frequency_template,
	ppg_absolute_dual_skewness_template,
	ppg_nonlinear_dynamic_system_template,
	ecg_dynamic_template,
	squeeze_template
	)
from vital_sqi.common.rpeak_detection import (
	PeakDetector
	)
from vital_sqi.common.utils import *

