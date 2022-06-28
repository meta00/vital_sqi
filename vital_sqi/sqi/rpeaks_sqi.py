"""Signal quality indexes based on R peak detection
This module capitalises on different R peak detectors for ECG and PGG,
using the resulted NN series to evaluate the raw signal quality.
- Ratio of ectopic beats
- Correlogram
- MSQ: Evaluate the consistency of two NN intervals by two peak detectors.
- Interpolation: Comparison of interpolated and non-interpolated NN intervals
    evaluate gap in the non-interpolated NN interval.
"""
import numpy as np
import scipy.interpolate
from scipy import signal
import heartpy as hp
from heartpy.analysis import clean_rr_intervals,calc_rr
from heartpy.peakdetection import check_peaks
from hrvanalysis.preprocessing import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from statsmodels.tsa.stattools import acf
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.common.utils import get_nn


def ectopic_sqi(s, rule_index=1, sample_rate=100, rpeak_detector=0,
                wave_type='ppg', low_rri=300,
                high_rri=2000, ):
    """
    Evaluate the invalid peaks (which exceeds normal range)
    base on HRV rules: Malik, Karlsson, Kamath, Acar
    Output the ratio of invalid
    Parameters
    ----------
    s :

    rule_index:
        0: Default Outlier Peak
        1: Malik
        2: Karlsson
        3: Kamath
        4: Acar
        (Default rule is Malik)

    sample_rate :
        (Default value = 100)
    rpeak_detector :
        (Default value = 0)
        To explain other detector options
    low_rri :
        (Default value = 300)
    high_rri :
        (Default value = 2000)

    Returns
    -------

    
    """
    rules = ["malik", "karlsson", "kamath", "acar"]
    # try:
    #     wd, m = hp.process(s, sample_rate, calc_freq=True)
    # except:
    #     try:
    #         wd, m = hp.process(s, sample_rate)
    #     except:
    #         return np.nan
    #
    # # if rpeak_detector in [1, 2, 3, 4]:
    # if wave_type=='ecg':
    #     detector = PeakDetector(wave_type='ecg')
    #     peak_list = detector.ecg_detector(s, rpeak_detector)[0]
    # else:
    #     detector = PeakDetector(wave_type='ppg')
    #     peak_list = detector.ppg_detector(s, rpeak_detector,
    #                                       preprocess=False)[0]
    # wd["peaklist"] = peak_list
    # wd = calc_rr(peak_list, sample_rate, working_data=wd)
    #
    # rr_intervals = wd["RR_list"]
    rr_intervals = get_nn(s,wave_type=wave_type,sample_rate=sample_rate,
                          rpeak_method=rpeak_detector,remove_ectopic_beat=False)
    rr_intervals_cleaned = remove_outliers(rr_intervals, low_rri=low_rri,
                                           high_rri=high_rri)
    number_outliers = len(np.where(np.isnan(rr_intervals_cleaned))[0])
    outlier_ratio = number_outliers/(len(rr_intervals_cleaned)-number_outliers)
    if rule_index == 0:
        return outlier_ratio

    interpolated_rr_intervals = interpolate_nan_values(rr_intervals_cleaned)

    rule = rules[rule_index]
    nn_intervals = remove_ectopic_beats(interpolated_rr_intervals,
                                            method=rule)
    number_ectopics = len(np.where(np.isnan(nn_intervals))[0])
    ectopic_ratio = number_ectopics/(len(nn_intervals)-number_ectopics)

    return ectopic_ratio


def correlogram_sqi(s, sample_rate=100, time_lag=3, n_selection=3):
    """The method is based on the paper 'Classification of the Quality of
    Wristband-based Photoplethysmography Signals'

    Parameters
    ----------
    s :
        Raw data
    sample_rate :
        (Default value = 100)
    time_lag :
        (Default value = 3)
    n_selection :
        (Default value = 3)

    Returns
    -------

    
    """
    nlags = time_lag*sample_rate
    corr = acf(s, nlags=nlags)
    corr_peaks_idx = signal.find_peaks(corr)[0]
    corr_peaks_value = corr[corr_peaks_idx]
    if n_selection > len(corr_peaks_value):
        n_selection = len(corr_peaks_value)

    corr_sqi = [i for i in corr_peaks_idx[:n_selection]]+\
          [i for i in corr_peaks_value[:n_selection]]
    return corr_sqi


def interpolation_sqi(s):
    """

    Parameters
    ----------
    s :
        
    To be developed
    Returns
    -------

    """

    return 0


def msq_sqi(s, peak_detector_1=7, peak_detector_2=6,wave_type='ppg'):
    """
    MSQ SQI as defined in Elgendi et al
    "Optimal Signal Quality Index for Photoplethysmogram Signals"
    with modification of the second algorithm used.
    Instead of Bing's, a SciPy built-in implementation is used.
    The SQI tracks the agreement between two peak detectors
    to evaluate quality of the signal.

    Parameters
    ----------
    s : sequence
        A signal with peaks.

    peak_detector_1 : array of int
        Type of the primary peak detection algorithm, default = Billauer

    peak_detect2 : int
        Type of the second peak detection algorithm, default = Scipy

    Returns
    -------
    msq_sqi : number
        MSQ SQI value for the given signal

    """
    if wave_type=='ppg':
        detector = PeakDetector(wave_type='ppg')
        peaks_1, trough_list_1 = detector.ppg_detector(s, detector_type=peak_detector_1)
        peaks_2, trough_list_2 = detector.ppg_detector(s, detector_type=peak_detector_2, preprocess=False)
    else:
        detector = PeakDetector(wave_type='ecg')
        peaks_1, trough_list = detector.ecg_detector(s, detector_type=peak_detector_1)
        peaks_2 = detector.ecg_detector(s, detector_type=peak_detector_2, preprocess=False)
    if len(peaks_1)==0 or len(peaks_2)==0:
        return 0.0
    elif len(peaks_1) != len(peaks_2):
        return np.abs(len(peaks_1)-len(peaks_2))/np.mean([len(peaks_1),len(peaks_2)])
    peak1_dom = len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)
    peak2_dom = len(np.intersect1d(peaks_2,peaks_1))/len(peaks_2)
    return min(peak1_dom, peak2_dom)