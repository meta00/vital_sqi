"""
vital_sqi.sqi
========
A subpackage for signal quality index calculation, including:
	- Standard: Statistical domain and xx
	- Dynamic template maching based
	- Peak detection based
	- Heart rate variability based
"""

from vital_sqi.sqi import (
	dtw_sqi,
	rpeaks_sqi,
	standard_sqi
)