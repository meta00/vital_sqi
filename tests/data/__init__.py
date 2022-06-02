"""
vital_sqi.data
========
A subpackage for all raw waveform data manipulation such as read/write,
edit, resample.
"""

from vital_sqi.preprocess.segment_split import (
	split_to_segments,
	generate_segment_idx
	)