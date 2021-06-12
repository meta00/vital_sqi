import numpy as np
from vital_sqi.preprocess.band_filter import BandpassFilter
from vital_sqi.common.rpeak_detection import PeakDetector
import vital_sqi.sqi as sq

def signal_preprocess(signal_channel=None, hp_cutoff_order=(1, 1), lp_cutoff_order=(20, 4), trim_amount=20, filter_type='butter', sampling_rate=100):
    """
    This function takes input signal, conducts standard preprocessing steps
    for ppg, and outputs modified signals.
    and ecg
    signals.
    """
    if signal_channel is None:
        raise Exception("No signal provided")
    else:   
        filter = BandpassFilter(band_type=filter_type, fs=sampling_rate)
        #Trim the amount of seconds from start and end of the signal
        signal_channel = signal_channel.T[trim_amount*sampling_rate:-trim_amount*sampling_rate].T
        for idx, channel in enumerate(signal_channel):
            signal_channel[idx] = filter.signal_highpass_filter(channel, cutoff=hp_cutoff_order[0], order=hp_cutoff_order[1])
            signal_channel[idx] = filter.signal_lowpass_filter(channel, cutoff=lp_cutoff_order[0], order=lp_cutoff_order[1])
    
    return signal_channel

def compute_all_SQI(signal=None, segments=None, raw_signal=None, primary_peakdet=7, secondary_peakdet=6, template_type=0):
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
        SQI_list = []
        if (signal is None) or (segments is None) or (raw_signal is None):
            raise Exception("Signal or segments not provided, can't extract all SQI's")
        else:
            for idx, segment_boundary in enumerate(segments[:-1]):
                #Prepare signal segments
                detector = PeakDetector()
                signal_segment = signal[segment_boundary:segments[idx+1]]
                raw_segment = raw_signal[segment_boundary:segments[idx+1]]
                peak_list, trough_list = detector.ppg_detector(signal_segment, primary_peakdet)
                SQI_dict = {'peaks':peak_list, 'troughs':trough_list}
                SQI_dict['perfusion'] = sq.standard_sqi.perfusion_sqi(y=signal_segment, x=raw_segment)
                SQI_dict['skewness'] = sq.standard_sqi.per_beat_sqi(sqi_func=sq.standard_sqi.skewness_sqi, troughs=trough_list, signal=signal_segment, taper=True)
                SQI_dict['kurtosis'] = sq.standard_sqi.per_beat_sqi(sqi_func=sq.standard_sqi.kurtosis_sqi, troughs=trough_list, signal=signal_segment, taper=True)
                SQI_dict['entropy'] = sq.standard_sqi.per_beat_sqi(sqi_func=sq.standard_sqi.entropy_sqi, troughs=trough_list, signal=signal_segment, taper=False)
                SQI_dict['snr'] = sq.standard_sqi.signal_to_noise_sqi(signal_segment)
                SQI_dict['zero_cross'] = sq.standard_sqi.zero_crossings_rate_sqi(signal_segment)
                SQI_dict['msq'] = sq.standard_sqi.msq_sqi(y=signal_segment, peaks_1=peak_list, peak_detect2=secondary_peakdet)
                SQI_dict['correlogram'] = sq.rpeaks_sqi.correlogram_sqi(signal_segment)
                SQI_dict['dtw'] = sq.standard_sqi.per_beat_sqi(sqi_func=sq.dtw_sqi, troughs=trough_list, signal=signal_segment, taper=True, template_type=template_type)  
                SQI_list.append(SQI_dict)
        
        return SQI_list
            
        

def compute_multiple_SQIs():
    """
    This function takes input signal, computes a
    number of SQIs, and outputs a table (row - signal segments, column SQI
    values°
    """
    return


def make_rule_set():
    """
    This function take a rule dictionary file and order of SQIs as input and
    generate a RuleSet object.
    """
    return


def get_cutpoints():
    """
    This function takes a list of segments and their called quality
    decisions (both from SignalSQI object) and calculates cut-points for
    accepted signal chunks.
    """
    return


def get_clean_signals():
    """
    This function takes raw signal file and cut-points to extract clean
    signal files.
    """
    return


def basic_ppg_pipeline():
    """ """
    return


def basic_ecg_pipeline():
    """ """
    return
