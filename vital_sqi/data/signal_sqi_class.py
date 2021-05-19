"""
Class containing signal, header and sqi
"""
import numpy as np
import pandas as pd
import datetime as dt


class SignalSQI:
    """ """
    def __init__(self, wave_type=None, signals=None, sampling_rate=None,
                 start_datetime=None, info=None, sqis=None, segments=None):
        self.signals = signals
        self.sampling_rate = sampling_rate
        self.start_datetime = start_datetime
        self.wave_type = wave_type
        self.info = info
        self.sqis = sqis
        self.segments = segments

    def __setattr__(self, name, value):
        if name == 'wave_type':
            assert value == 'ecg' or value == 'ppg', \
                'Expected either ecg or ppg.'
        if name == 'signals':
            assert isinstance(value, np.ndarray), 'Expected signals to be ' \
                                                  'numpy array, with one ' \
                                                  'channel per column.'
        if name == 'sampling_rate':
            assert isinstance(value, int), 'Expected int'
        if name == 'start_datetime':
            assert isinstance(value, str) or \
                   isinstance(value, dt.datetime) or \
                   isinstance(value, dt.date), 'Expected str or datetime ' \
                                               'object'
        if name == 'sqis':
            assert isinstance(value, np.ndarray) or \
                   isinstance(value,pd.DataFrame), 'Expected SQI table as ' \
                                                   'array or data frame'
        if name == 'segments':
            assert isinstance(value, list), 'Expected a list of signal ' \
                                            'segments'

        super().__setattr__(name, value)
