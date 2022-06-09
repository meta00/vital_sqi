"""
vital_sqi
=========
A package to extract signal quality indexes from physiological signals,
including ECG and PPG.
"""

from vital_sqi import (
	common,
	data,
	preprocess,
	sqi,
	rule,
	pipeline
)

from vital_sqi.rule.rule_class import Rule
from vital_sqi.rule.ruleset_class import RuleSet
from vital_sqi.data.signal_sqi_class import SignalSQI
from vital_sqi.data.signal_io import *
from vital_sqi.pipeline import *

__version__ = '0.1.0'