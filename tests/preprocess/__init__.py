"""
vital_sqi.preprocess
========

A subpackage for all waveform preprocessing such as filtering, detrend etc.
edit, resample.
"""

from vital_sqi.preprocess.band_filter import (
	BandpassFilter
	)
from vital_sqi.preprocess.preprocess_signal import (
	tapering,
	smooth,
	scale_pattern,
	squeeze_template
	)