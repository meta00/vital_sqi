import numpy as np
import pandas as pd
from scipy.sparse import coo
from vital_sqi.preprocess.band_filter import BandpassFilter
from vital_sqi.common.rpeak_detection import PeakDetector
import vital_sqi.sqi as sq


def compute_SQI(signal, segment_length='30s', primary_peakdet=7,
				secondary_peakdet=6, wave_type='ppg', sampling_rate=100,
				template_type=1):

	"""
    Run the segment SQI computation using pandas groupby().apply() on the whole dataset.

    Parameters
    ----------
    signal : dataframe
    Whole raw signal in dataframe format as constructed from the source file after the reader function 

    segment_length : string
    Defines the length of each segment based on the "timedelta" column of the dataframe


    primary_peakdet : int
    Selects one of the peakdetectors from the PeakDetector class. The primary one is used to segment the waveform

    secondary_peakdet : int
    Selects one of the peakdetectors from the PeakDetector class. The secondary peakdetector is used to compute MSQ SQI

    wave_type : string
    Defines the type of signal, only 'ppg' or 'ecg' allowed
    
    sampling_rate : int
    Sampling rate of the signal
    
    template_type : int
    Selects which template from the dtw SQI should be used       

    Returns
    -------
    Dataframe with column correspoding to each SQI

    """
	if wave_type == 'ppg':
		sqis = signal.groupby(pd.Grouper(freq=segment_length)).apply(
			segment_PPG_SQI_extraction, sampling_rate, primary_peakdet,
			secondary_peakdet, (1, 1), (20, 4), template_type)
	elif wave_type == 'ecg':
		sqis = signal.groupby(pd.Grouper(freq=segment_length)).apply(
			segment_ECG_SQI_extraction, sampling_rate, primary_peakdet,
			secondary_peakdet, (1, 1), (20, 4), template_type)
	else:
		raise Exception(
			"Wrong type of waveform supplied. Only accepts 'ppg' or 'ecg'.")
	return sqis


def segment_PPG_SQI_extraction(signal_segment, sampling_rate=100,
							   primary_peakdet=7, secondary_peakdet=6,
							   hp_cutoff_order=(1, 1), lp_cutoff_order=(20, 4),
							   template_type=1):
	"""
    Extract all package available SQIs from a single segment of PPG waveform. Return a dataframe with all SQIs and cut points for each segment.

    Parameters
    ----------
    signal_segment : array-like
    A segment of raw signal. The length is user defined in compute_SQI() function

    sampling_rate : int
    Sampling rate of the signal

    primary_peakdet : int
    Selects one of the peakdetectors from the PeakDetector class. The primary one is used to segment the waveform

    secondary_peakdet : int
    Selects one of the peakdetectors from the PeakDetector class. The secondary peakdetector is used to compute MSQ SQI

    hp_cutoff_order : touple (int, int)
    A high pass filter parameters, cutoff frequency and order

    Lp_cutoff_order : touple (int, int)
    A low pass filter parameters, cutoff frequency and order
    
    template_type : int
    Selects which template from the dtw SQI should be used       

    Returns
    -------
    Pandas series object with all SQIs for the given segment

    """
	raw_segment = signal_segment[signal_segment.columns[1]].to_numpy()
	# Prepare final dictonary that will be converted to dataFrame at the end
	SQI_dict = {'first': signal_segment['idx'][0],
				'last' : signal_segment['idx'][-1]}
	# Prepare filter and filter signal
	filt = BandpassFilter(band_type='butter', fs=sampling_rate)
	filtered_segment = filt.signal_highpass_filter(raw_segment,
												   cutoff=hp_cutoff_order[0],
												   order=hp_cutoff_order[1])
	filtered_segment = filt.signal_lowpass_filter(filtered_segment,
												  cutoff=lp_cutoff_order[0],
												  order=lp_cutoff_order[1])
	# Prepare primary peak detector and perform peak detection
	detector = PeakDetector()
	peak_list, trough_list = detector.ppg_detector(filtered_segment,
												   primary_peakdet)
	# Helpful lists for iteration
	variations_stats = ['', '_mean', '_median', '_std']
	variations_acf = ['_peak1', '_peak2', '_peak3', '_value1', '_value2',
					  '_value3']
	stats_functions = [('skewness', sq.standard_sqi.skewness_sqi),
					   ('kurtosis', sq.standard_sqi.kurtosis_sqi),
					   ('entropy', sq.standard_sqi.entropy_sqi)]
	# Raw signal SQI computation
	SQI_dict['snr'] = np.mean(sq.standard_sqi.signal_to_noise_sqi(raw_segment))
	SQI_dict['perfusion'] = sq.standard_sqi.perfusion_sqi(y=filtered_segment,
														  x=raw_segment)
	SQI_dict['mean_cross'] = sq.standard_sqi.mean_crossing_rate_sqi(raw_segment)
	# Filtered signal SQI computation
	SQI_dict['zero_cross'] = sq.standard_sqi.zero_crossings_rate_sqi(
		filtered_segment)
	SQI_dict['msq'] = sq.standard_sqi.msq_sqi(y=filtered_segment,
											  peaks_1=peak_list,
											  peak_detect2=secondary_peakdet)
	correlogram_list = sq.rpeaks_sqi.correlogram_sqi(filtered_segment)
	for idx, variations in enumerate(variations_acf):
		SQI_dict['correlogram' + variations] = correlogram_list[idx]
	# Per beat SQI calculation
	dtw_list = sq.standard_sqi.per_beat_sqi(sqi_func=sq.dtw_sqi,
											troughs=trough_list,
											signal=filtered_segment, taper=True,
											template_type=template_type)
	SQI_dict['dtw_mean'] = np.mean(dtw_list)
	SQI_dict['dtw_std'] = np.std(dtw_list)
	for funcion in stats_functions:
		SQI_dict[funcion[0] + variations_stats[0]] = funcion[1](
			filtered_segment)
		statSQI_list = sq.standard_sqi.per_beat_sqi(sqi_func=funcion[1],
													troughs=trough_list,
													signal=filtered_segment,
													taper=True)
		SQI_dict[funcion[0] + variations_stats[1]] = np.mean(statSQI_list)
		SQI_dict[funcion[0] + variations_stats[2]] = np.median(statSQI_list)
		SQI_dict[funcion[0] + variations_stats[3]] = np.std(statSQI_list)
	#
	return pd.Series(SQI_dict)


def segment_ECG_SQI_extraction(signal_segment, sampling_rate=100,
							   primary_peakdet=7, secondary_peakdet=6,
							   hp_cutoff_order=(1, 1), lp_cutoff_order=(20, 4),
							   template_type=1):
	raw_segment = signal_segment[signal_segment.columns[1]].to_numpy()
	SQI_dict = {'first': signal_segment['idx'][0],
				'last' : signal_segment['idx'][-1]}
	# TODO ECG highlevel SQI extraction
	return pd.Series(SQI_dict)


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
