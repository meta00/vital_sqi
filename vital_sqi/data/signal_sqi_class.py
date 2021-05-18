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
                   value is None, 'Expected a list of signal segments or None'
        if name == 'rules':
            assert isinstance(value, list) or \
                   value is None, 'Expected a list of Rule objects.'
        if name == 'ruleset':
            assert isinstance(value, dict) or \
                   value is None, 'Expected an object of RuleSet'
        super().__setattr__(name, value)
        return

        self.isSplit = False
        self.isFiltered = False
        self.SQIComputed = False
        self.unfilteredsignal = None

    def split_to_segments(self, minute_remove=0.5, segment_duration=30):
        """

        Parameters
        ----------
        minute_remove : float
        number of minutes to be removed from start and end of the signal

        segment_duration : int
        duration of segment in seconds            

        Returns
        -------
        object of SignalSQI class
        
        """
        if self.signals.ndim == 1 and not self.isSplit:
            #Compute size of to be removed slice and each segment
            removal = round(minute_remove*60*self.sampling_rate)
            segment_size = segment_duration*self.sampling_rate
            #Remove the excess parts at the start and end according to calculate size of removal
            tmp_signal = self.signals[removal:-removal]
            #Calculate how many segments can fit into remaining data and reshape the array
            excess = len(tmp_signal) % segment_size
            tmp_signal = np.reshape(tmp_signal[:-excess], (segment_size, -1), 'F').T
            self.isSplit = True
            self.unfilteredsignal = tmp_signal
            self.update_signal(tmp_signal)
        elif not self.isSplit:
            raise Exception("Not supported for multidimensional signals")
        else:
            return self


    def filter_preprocess(self, hp_cutoff_order=(1, 1), lp_cutoff_order=(20, 4), filter_type='butter'):
        """

        Parameters
        ----------
        hp_cutoff_order : touple (int, int)
        high pass cutoff frequency in Hz and filter order

        hp_cutoff_order : touple (int, int)
        low pass cutoff frequency in Hz and filter order

        filter_type : string
        type of digital filter used        

        Returns
        -------
        object of SignalSQI class
        
        """
        if self.isSplit and not self.isFiltered:
            filter = BandpassFilter(band_type=filter_type, fs=self.sampling_rate)
            tmp_signals = self.signals
            for idx, _ in enumerate(tmp_signals):
                tmp_signals[idx] = filter.signal_highpass_filter(tmp_signals[idx], cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
                tmp_signals[idx] = filter.signal_lowpass_filter(tmp_signals[idx], cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
            self.isFiltered = True
            self.update_signal(tmp_signals)
        elif not self.isSplit:
            raise Exception("Dataset is not split, please run split_to_segments before filtering")
        else:
            return self
    
    def compute_all_SQI(self, primary_peakdet=7, secondary_peakdet=2, template_type=0):
        """

        Parameters
        ----------
        primary_peakdet : int
        Selects one of the peakdetectors from the PeakDetector class. The primary one is used to segment the waveform

        secondary_peakdet : int
        Selects one of the peakdetectors from the PeakDetector class. The secondary peakdetector is used to compute MSQ SQI

        template_type : int
        Selects which template from the dtw SQI should be used       

        Returns
        -------
        object of SignalSQI class
        
        """
        number_of_sqis = 6
        if self.isSplit and self.isFiltered and not self.SQIComputed:
            #Start by running the primary peak detector and finding the peaks and troughs
            sqi_vector = np.zeros((len(self.signals), number_of_sqis))
            for idx, signal_seg in enumerate(self.signals):
                detector = PeakDetector()
                peak_list, trough_list = detector.ppg_detector(signal_seg, primary_peakdet)
                perfusion_sqi = sq.standard_sqi.perfusion_sqi(y=signal_seg, x=self.unfilteredsignal[idx])
                skewness_sqi = sq.standard_sqi.per_beat_sqi(sqi_func=sq.standard_sqi.skewness_sqi, troughs=trough_list, signal=signal_seg)
                kurtosis_sqi = sq.standard_sqi.per_beat_sqi(sqi_func=sq.standard_sqi.kurtosis_sqi, troughs=trough_list, signal=signal_seg)
                entropy_sqi = sq.standard_sqi.per_beat_sqi(sqi_func=sq.standard_sqi.entropy_sqi, troughs=trough_list, signal=signal_seg)
                snr_sqi = sq.standard_sqi.signal_to_noise_sqi(signal_seg)
                msq_sqi = sq.standard_sqi.msq_sqi(y=signal_seg, peaks_1=peak_list, peak_detect2=6)
                sqi_vector[idx] = np.array([perfusion_sqi[0], np.mean(skewness_sqi), np.mean(kurtosis_sqi), np.mean(entropy_sqi), snr_sqi, msq_sqi])
            self.update_sqi_indexes(sqi_vector)
        elif not self.isSplit or not self.isFiltered:
            raise Exception("The dataset needs to be split and filtered first")
        else:
            return self
    
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
