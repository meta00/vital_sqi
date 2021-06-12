"""
vital_sqi.sqi
==============
A subpackage for signal quality index calculation, including:
	- Standard: Statistical domain and xx
	- Dynamic template maching based
	- Peak detection based
	- Heart rate variability based
"""

from vital_sqi.sqi.dtw_sqi import (
	dtw_sqi
	)
from vital_sqi.sqi.standard_sqi import (
	perfusion_sqi,
	kurtosis_sqi,
	skewness_sqi,
	entropy_sqi,
	signal_to_noise_sqi,
	zero_crossings_rate_sqi,
	mean_crossing_rate_sqi
	)
from vital_sqi.sqi.rpeaks_sqi import (
	correlogram_sqi
	)