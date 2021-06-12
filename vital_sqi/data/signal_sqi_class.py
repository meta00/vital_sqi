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
    def __init__(self, wave_type=None, signals=None, sampling_rate=None,
                 start_datetime=None, info=None, segments=None, sqis=None,
                 rules=None, ruleset=None):
        self.signals = signals
        self.sampling_rate = sampling_rate
        self.start_datetime = start_datetime
        self.wave_type = wave_type
        self.info = info
        self.sqis = sqis
        self.segments = segments
        self.rules = rules
        self.ruleset = ruleset
        self.isSplit = False
        self.isFiltered = False
        self.SQIComputed = False
        self.unfilteredsignal = None

    def __setattr__(self, name, value):
        if name == 'wave_type':
            assert value == 'ecg' or value == 'ppg', \
                'Expected either ecg or ppg.'
        if name == 'signals':
            assert isinstance(value, np.ndarray), 'Expected signals to be ' \
                                                  'numpy array, with one ' \
                                                  'channel per column.'
        if name == 'sampling_rate':
            assert np.isreal(value), 'Expected a numeric value. Sampling ' \
                                     'rate is round up to the nearest integer.'
        if name == 'start_datetime':
            assert isinstance(value, str) or \
                   isinstance(value, dt.datetime) or \
                   isinstance(value, dt.date) or value is None, \
                'Expected str or datetime object, or None'
        if name == 'sqis':
            assert isinstance(value, np.ndarray) or \
                   isinstance(value,pd.DataFrame) or value is None, \
                'Expected SQI table as array or data frame or None'
        if name == 'segments':
            assert isinstance(value, list) or \
                   isinstance(value, np.ndarray) or value is None,  \
                'Expected a list or ndarray of signal segments or None'
        if name == 'rules':
            assert isinstance(value, list) or \
                   value is None, 'Expected a list of Rule objects.'
        if name == 'ruleset':
            assert isinstance(value, dict) or \
                   value is None, 'Expected an object of RuleSet'
        super().__setattr__(name, value)
        return

    
    def update_info(self, info):
        """

        Parameters
        ----------
        info :
            

        Returns
        -------
        object of SignalSQI class
        
        """
        self.info = info
        return self

    def update_signal(self, signals):
        """

        Parameters
        ----------
        signals : numpy.ndarray of shape (m, n)
        m is the number of rows and n is the number of channels of the signal.
            

        Returns
        -------
        object of class SignalSQI
        
        """
        self.signals = signals
        return self

    def update_sqi_indexes(self, sqi_indexes):
        """

        Parameters
        ----------
        sqi_indexes : numpy.ndarray of shape (m, n)
        m is the number of signal segments, n is the number of SQIs.
            

        Returns
        -------
        object of class SignalSQI
        
        """
        self.sqi_indexes = sqi_indexes
        return self

    def update_sampling_rate(self, sampling_rate):
        """

        Parameters
        ----------
        sampling_rate : float
        Note: sampling_rate must be correct to reliably infer RR intervals,
        etc.

            

        Returns
        -------
        object of class SignalSQI
        """
        self.sampling_rate = sampling_rate
        return self

    def update_start_datetime(self, start_datetime):
        """

        Parameters
        ----------
        start_datetime : datetime
        start date and

        Returns
        -------
        object of si
        """
        self.start_datetime = start_datetime
        return self
    
    def update_segment_indices(self, segments):
        """

        Parameters
        ----------
        segments : ndarray of indices

        Returns
        -------
        object of si
        """
        self.segments = segments
        return self
