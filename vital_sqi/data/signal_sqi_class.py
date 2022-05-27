import numpy as np
from numpy.lib.function_base import trim_zeros
from vital_sqi.preprocess.band_filter import BandpassFilter
from vital_sqi.common.rpeak_detection import PeakDetector
import vital_sqi.sqi as sq

"""
Class containing signal, header and sqi
"""
import numpy as np
import pandas as pd
import datetime as dt


class SignalSQI:
    """ """
    def __init__(self, wave_type=None,
                 signals=None,
                 sampling_rate=None,
                 # start_datetime=None,
                 info=None,
                 # segments=None,
                 sqis=None,
                 rules=None,
                 ruleset=None):
        self.signals = signals
        self.sampling_rate = sampling_rate
        # self.start_datetime = start_datetime
        self.wave_type = wave_type
        self.info = info
        self.sqis = sqis
        # self.segments = segments
        self.rules = rules
        self.ruleset = ruleset
        self.isSplit = False
        self.isFiltered = False
        self.SQIComputed = False
        # self.unfilteredsignal = None

    def __setattr__(self, name, value):
        if name == 'wave_type':
            assert value == 'ecg' or value == 'ppg', \
                'Expected either ecg or ppg.'
        if name == 'signals':
            assert isinstance(value, pd.DataFrame), 'Expected signals as a' \
                                                    'dataframe with ' \
                                                    'one channel per column.'
        if name == 'sampling_rate':
            assert np.isreal(value), 'Expected a numeric value. Sampling ' \
                                     'rate is round up to the nearest integer.'
        # if name == 'start_datetime':
        #     assert isinstance(value, str) or \
        #            isinstance(value, dt.datetime) or \
        #            isinstance(value, dt.date) or value is None, \
        #         'Expected str or datetime object, or None'
        if name == 'sqis':
            assert isinstance(value, pd.DataFrame) or value is None, \
                'Expected SQI table as a dataframe or None'
        if name == 'rules':
            assert isinstance(value, list) or \
                   value is None, 'Expected rules as a list of Rule objects.'
        if name == 'ruleset':
            assert isinstance(value, dict) or \
                   value is None, 'Expected ruleset as a RuleSet object'
        super().__setattr__(name, value)
        return

